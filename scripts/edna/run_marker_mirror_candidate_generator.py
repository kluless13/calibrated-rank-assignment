#!/usr/bin/env python3
"""Run MarkerMirror candidate generation from FASTA/CSV marker fragments.

This is the research handoff from MarkerMirror into the production pipeline:

    12S/16S fragment -> shared MarkerMirror vector space -> target-marker
    candidate species list

It writes candidates, not final assignments.  Downstream evidence checks and
rank/no-call calibration are still required before claiming species
identification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.export_marker_mirror_candidate_rankings import build_prototypes, load_projection_head, project, tax_value
from scripts.edna.train_marker_mirror_bridge import Logger, clean_sequence, embed_sequences, flatten, load_hf, load_species_json, load_taxonomy

RANKS = ("species", "genus", "family", "order")
MARKER_INPUT_DIRS = {
    "12S": ROOT / "data" / "edna" / "stalder_inputs" / "multisource",
    "16S": ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource",
}
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "remote_runs"
    / "2026-06-03"
    / "rtx_pro_6000"
    / "marker_mirror_bridge"
    / "nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best_seed1903"
    / "marker_mirror_shared_projection_head.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="FASTA or CSV with marker sequences.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-marker", choices=("12S", "16S"), required=True)
    parser.add_argument("--target-marker", choices=("12S", "16S"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--query-id-column")
    parser.add_argument("--sequence-column")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--target-max-per-species", type=int, default=4)
    parser.add_argument("--target-species-limit", type=int)
    parser.add_argument(
        "--target-embedding-cache",
        type=Path,
        help="Optional .npz cache for target-marker foundation embeddings. Loaded if valid, otherwise written.",
    )
    parser.add_argument(
        "--refresh-target-embedding-cache",
        action="store_true",
        help="Recompute and overwrite --target-embedding-cache even if a valid cache exists.",
    )
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def unique_ids(raw_ids: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for idx, raw in enumerate(raw_ids, start=1):
        base = clean(raw) or f"query_{idx:06d}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def parse_fasta(path: Path, limit: int | None) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    rows.append({"query_id": current_id, "nucleotides": clean_sequence("".join(chunks))})
                    if limit is not None and len(rows) >= limit:
                        return pd.DataFrame(rows)
                current_id = line[1:].strip().split()[0] or f"query_{len(rows) + 1:06d}"
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None and (limit is None or len(rows) < limit):
            rows.append({"query_id": current_id, "nucleotides": clean_sequence("".join(chunks))})
    if not rows:
        raise RuntimeError(f"No FASTA records found in {path}")
    frame = pd.DataFrame(rows)
    frame["query_id"] = unique_ids(frame["query_id"].astype(str).tolist())
    return frame


def parse_csv_input(path: Path, id_column: str | None, sequence_column: str | None, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit).copy()
    columns = set(frame.columns)
    seq_col = sequence_column or first_existing(columns, ["nucleotides", "sequence", "seq", "barcode"])
    if seq_col is None or seq_col not in frame.columns:
        raise RuntimeError("Could not find a sequence column. Use --sequence-column.")
    id_col = id_column or first_existing(columns, ["query_id", "processid", "sample_id", "id"])
    out = pd.DataFrame()
    raw_ids = frame[id_col].astype(str).tolist() if id_col and id_col in frame.columns else ["" for _ in range(len(frame))]
    out["query_id"] = unique_ids(raw_ids)
    out["nucleotides"] = frame[seq_col].map(clean_sequence)
    for output_col, candidates in [
        ("tree_label", ["tree_label", "species_tree_label"]),
        ("species", ["species_name", "species"]),
        ("genus", ["genus_name", "genus"]),
        ("family", ["family_name", "family"]),
        ("order", ["order_name", "order"]),
    ]:
        source = first_existing(columns, candidates)
        out[output_col] = frame[source].map(clean) if source and source in frame.columns else ""
    return out


def load_queries(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if path.suffix.lower() in {".fa", ".fasta", ".fna"}:
        frame = parse_fasta(path, args.limit)
    else:
        frame = parse_csv_input(path, args.query_id_column, args.sequence_column, args.limit)
    frame["nucleotides"] = frame["nucleotides"].map(clean_sequence)
    frame = frame[frame["nucleotides"].map(bool)].copy()
    if frame.empty:
        raise RuntimeError("All input sequences were empty after A/C/G/T cleaning.")
    for col in ["tree_label", "species", "genus", "family", "order"]:
        if col not in frame.columns:
            frame[col] = ""
    return frame.reset_index(drop=True)


def load_target_reference(marker: str, max_per_species: int, species_limit: int | None) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    input_dir = MARKER_INPUT_DIRS[marker]
    species = load_species_json(input_dir / "train_species_sequences.json", max_per_species)
    labels = sorted(species)
    if species_limit is not None:
        labels = labels[:species_limit]
    tax = load_taxonomy(input_dir / "candidate_species.csv")
    rows = flatten(species, labels, marker)
    return rows, tax


def sequence_hashes(sequences: list[str]) -> np.ndarray:
    hashes = [hashlib.sha1(clean_sequence(seq).encode("ascii")).hexdigest() for seq in sequences]
    return np.asarray(hashes, dtype="U40")


def target_cache_metadata(args: argparse.Namespace, target_rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "marker": args.target_marker,
        "model": args.model,
        "max_length": int(args.max_length),
        "target_max_per_species": int(args.target_max_per_species),
        "target_species_limit": args.target_species_limit,
        "target_rows": int(len(target_rows)),
        "target_reference_species": int(target_rows["tree_label"].nunique()),
    }


def load_target_embedding_cache(
    cache_path: Path | None,
    args: argparse.Namespace,
    target_rows: pd.DataFrame,
    logger: Logger,
) -> np.ndarray | None:
    if cache_path is None or args.refresh_target_embedding_cache or not cache_path.exists():
        return None
    expected_meta = target_cache_metadata(args, target_rows)
    expected_labels = target_rows["tree_label"].astype(str).to_numpy(dtype="U")
    expected_hashes = sequence_hashes(target_rows["nucleotides"].astype(str).tolist())
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata_json"].item()))
            labels = cache["tree_labels"].astype(str)
            hashes = cache["sequence_hashes"].astype(str)
            embeddings = cache["embeddings"].astype(np.float32)
    except Exception as exc:
        logger.log(f"Target embedding cache could not be read ({exc}); recomputing")
        return None
    mismatches = [
        key
        for key, expected in expected_meta.items()
        if metadata.get(key) != expected
    ]
    if mismatches:
        logger.log(f"Target embedding cache metadata mismatch fields={mismatches}; recomputing")
        return None
    if len(labels) != len(expected_labels) or not np.array_equal(labels, expected_labels):
        logger.log("Target embedding cache label order mismatch; recomputing")
        return None
    if len(hashes) != len(expected_hashes) or not np.array_equal(hashes, expected_hashes):
        logger.log("Target embedding cache sequence hash mismatch; recomputing")
        return None
    logger.log(f"Loaded target embedding cache {cache_path} rows={len(embeddings)}")
    return embeddings


def write_target_embedding_cache(
    cache_path: Path | None,
    args: argparse.Namespace,
    target_rows: pd.DataFrame,
    target_emb: np.ndarray,
    logger: Logger,
) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = target_cache_metadata(args, target_rows)
    np.savez_compressed(
        cache_path,
        embeddings=target_emb.astype(np.float32),
        tree_labels=target_rows["tree_label"].astype(str).to_numpy(dtype="U"),
        sequence_hashes=sequence_hashes(target_rows["nucleotides"].astype(str).tolist()),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    logger.log(f"Wrote target embedding cache {cache_path} rows={len(target_emb)}")


def write_candidates(
    queries: pd.DataFrame,
    query_vecs: np.ndarray,
    target_rows: pd.DataFrame,
    target_vecs: np.ndarray,
    target_tax: dict[str, dict[str, str]],
    top_k: int,
    query_marker: str,
    target_marker: str,
    output_path: Path,
) -> pd.DataFrame:
    proto, proto_arr = build_prototypes(target_rows, target_vecs, target_tax)
    sims = query_vecs @ proto_arr.T
    k = min(top_k, sims.shape[1])
    top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(top_idx, np.argsort(-np.take_along_axis(sims, top_idx, axis=1), axis=1), axis=1)
    rows: list[dict[str, Any]] = []
    for query_i, indices in enumerate(ordered):
        query = queries.iloc[query_i]
        query_label = clean(query.get("tree_label", "")).replace(" ", "_")
        for rank_i, proto_i in enumerate(indices, start=1):
            candidate = proto.iloc[int(proto_i)]
            candidate_label = str(candidate["tree_label"])
            item: dict[str, Any] = {
                "query_id": str(query["query_id"]),
                "query_marker": query_marker,
                "target_marker": target_marker,
                "candidate_rank": rank_i,
                "candidate_tree_label": candidate_label,
                "score": float(sims[query_i, int(proto_i)]),
            }
            for rank in RANKS:
                cv = tax_value(target_tax, candidate_label, rank)
                item[f"candidate_{rank}"] = cv
                qv = clean(query.get(rank, ""))
                item[f"query_{rank}"] = qv
                item[f"match_{rank}"] = bool(qv and cv and qv == cv)
            if query_label:
                item["query_tree_label"] = query_label
                item["match_species_tree_label"] = bool(query_label == candidate_label)
            rows.append(item)
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


def summarize(candidates: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "query_count": int(queries["query_id"].nunique()),
            "candidate_rows": int(len(candidates)),
            "top_k": int(candidates["candidate_rank"].max()) if len(candidates) else 0,
        }
    ]
    if "match_species_tree_label" in candidates.columns and candidates["match_species_tree_label"].any():
        labelled = candidates[candidates["query_tree_label"].map(clean).astype(bool)]
        for k in [1, 5, 10, 50]:
            top = labelled[labelled["candidate_rank"] <= k]
            if top.empty:
                continue
            per_query = top.groupby("query_id")
            rows[0][f"known_species_top{k}_pct"] = 100.0 * float(per_query["match_species_tree_label"].max().mean())
            for rank in ["genus", "family", "order"]:
                rows[0][f"known_{rank}_top{k}_pct"] = 100.0 * float(per_query[f"match_{rank}"].max().mean())
    return pd.DataFrame(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    if args.query_marker == args.target_marker:
        raise RuntimeError("--query-marker and --target-marker must differ for MarkerMirror 12S/16S mode.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_candidate_generator.log")
    device = choose_device(args.device)
    logger.log(f"Arguments: {vars(args)} device={device}")
    queries = load_queries(args.input, args)
    logger.log(f"Loaded queries rows={len(queries)}")
    target_rows, target_tax = load_target_reference(args.target_marker, args.target_max_per_species, args.target_species_limit)
    logger.log(f"Loaded target references marker={args.target_marker} rows={len(target_rows)} species={target_rows['tree_label'].nunique()}")
    head, checkpoint_args = load_projection_head(args.checkpoint, device)
    logger.log(f"Loaded MarkerMirror checkpoint {args.checkpoint}")
    tokenizer, model, masked_lm = load_hf(args.model, device, logger)
    query_emb = embed_sequences(
        queries["nucleotides"].map(clean_sequence).tolist(),
        tokenizer,
        model,
        masked_lm,
        device,
        args.embed_batch_size,
        args.max_length,
        logger,
        args.query_marker,
    )
    target_emb = load_target_embedding_cache(args.target_embedding_cache, args, target_rows, logger)
    cache_status = "loaded" if target_emb is not None else "not_used"
    if target_emb is None:
        target_emb = embed_sequences(
            target_rows["nucleotides"].map(clean_sequence).tolist(),
            tokenizer,
            model,
            masked_lm,
            device,
            args.embed_batch_size,
            args.max_length,
            logger,
            args.target_marker,
        )
        write_target_embedding_cache(args.target_embedding_cache, args, target_rows, target_emb, logger)
        cache_status = "written" if args.target_embedding_cache is not None else "not_requested"
    query_projected = project(head, query_emb, device)
    target_projected = project(head, target_emb, device)
    candidates = write_candidates(
        queries,
        query_projected,
        target_rows,
        target_projected,
        target_tax,
        args.top_k,
        args.query_marker,
        args.target_marker,
        args.output_dir / "marker_mirror_candidate_generator_candidates.csv",
    )
    summary = summarize(candidates, queries)
    summary.to_csv(args.output_dir / "marker_mirror_candidate_generator_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/run_marker_mirror_candidate_generator.py",
        "input": str(args.input),
        "query_marker": args.query_marker,
        "target_marker": args.target_marker,
        "checkpoint": str(args.checkpoint),
        "target_reference_species": int(target_rows["tree_label"].nunique()),
        "query_count": int(len(queries)),
        "candidate_rows": int(len(candidates)),
        "device": device,
        "target_embedding_cache": str(args.target_embedding_cache) if args.target_embedding_cache else None,
        "target_embedding_cache_status": cache_status,
        "claim_boundary": "Research candidate-generator mode only. Downstream evidence checks and calibrated rank/no-call are required for final assignment.",
        "checkpoint_args": json_safe(checkpoint_args),
    }
    (args.output_dir / "marker_mirror_candidate_generator_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote candidate rows={len(candidates)}")


if __name__ == "__main__":
    main()
