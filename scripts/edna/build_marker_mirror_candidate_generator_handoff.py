#!/usr/bin/env python3
"""Build evidence-ready rows from MarkerMirror candidate-generator output.

This is the production-handoff bridge:

    candidate-generator candidates
      -> same-marker sequence/reference checks
      -> tree/resolvability/candidate-list evidence
      -> rank/no-call-ready source table

The output is not a final assignment.  It deliberately separates production
features from optional evaluation labels so hidden labels are not required for
specimen-style inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.build_marker_mirror_evidence_join import (
    RANKS,
    TOP_KS,
    best_same_marker_evidence,
    load_marker_sequences,
    load_resolvability_table,
    load_tree_cache,
    resolvability_features,
    tree_distance,
)
from scripts.edna.run_marker_mirror_candidate_generator import load_queries
from scripts.edna.train_marker_mirror_bridge import Logger, clean_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Candidate CSV from run_marker_mirror_candidate_generator.py.")
    parser.add_argument(
        "--query-input",
        type=Path,
        help="Original FASTA/CSV query input. If omitted, read from candidate-generator manifest in the same directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-id-column")
    parser.add_argument("--sequence-column")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-name", default="marker_mirror_candidate_generator")
    parser.add_argument("--split", default="production_handoff")
    parser.add_argument("--marker-a-name", default="12S")
    parser.add_argument("--marker-a-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource")
    parser.add_argument("--marker-b-name", default="16S")
    parser.add_argument("--marker-b-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource")
    parser.add_argument("--max-a-per-species", type=int, default=8)
    parser.add_argument("--max-b-per-species", type=int, default=8)
    parser.add_argument("--tree-file", type=Path, default=ROOT / "data" / "phylo" / "actinopt_12k_treePL.tre")
    parser.add_argument("--disable-tree", action="store_true")
    parser.add_argument(
        "--marker-resolvability-table",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_marker_resolvability_by_species.csv",
    )
    parser.add_argument("--marker-resolvability-identity", type=float, default=0.99)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def sha1_sequence(seq: str) -> str:
    return hashlib.sha1(clean_sequence(seq).encode("ascii")).hexdigest()


def infer_query_input(candidates_path: Path, query_input: Path | None) -> Path:
    if query_input is not None:
        return query_input
    manifest = candidates_path.parent / "marker_mirror_candidate_generator_manifest.json"
    if not manifest.exists():
        raise RuntimeError("--query-input is required when candidate-generator manifest is absent.")
    data = json.loads(manifest.read_text())
    path = Path(str(data.get("input", "")))
    if not path:
        raise RuntimeError("Candidate-generator manifest does not contain an input path.")
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_query_table(path: Path, args: argparse.Namespace, logger: Logger) -> pd.DataFrame:
    query_args = argparse.Namespace(
        query_id_column=args.query_id_column,
        sequence_column=args.sequence_column,
        limit=args.limit,
    )
    queries = load_queries(path, query_args)
    queries["query_sequence_hash"] = queries["nucleotides"].map(sha1_sequence)
    queries["query_sequence_length"] = queries["nucleotides"].map(len).astype(int)
    logger.log(f"Loaded query input rows={len(queries)} from {path}")
    return queries


def add_query_columns(candidates: pd.DataFrame, queries: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    cols = ["query_id", "nucleotides", "query_sequence_hash", "query_sequence_length"]
    for rank in ["tree_label", *RANKS]:
        if rank in queries.columns:
            cols.append(rank)
    query_meta = queries[cols].copy()
    rename = {"tree_label": "input_query_tree_label"}
    for rank in RANKS:
        rename[rank] = f"input_query_{rank}"
    query_meta = query_meta.rename(columns=rename)
    out = candidates.merge(query_meta, on="query_id", how="left", validate="many_to_one")
    missing = int(out["nucleotides"].isna().sum())
    if missing:
        logger.log(f"WARNING: candidate rows without query sequence={missing}")
    out["query_sequence_available"] = out["nucleotides"].notna().astype(int)
    out["nucleotides"] = out["nucleotides"].fillna("").map(clean_sequence)
    return out


def add_query_group_features(frame: pd.DataFrame, markers: dict[str, dict[str, Any]], logger: Logger) -> pd.DataFrame:
    out_frames = []
    group_cols = ["model", "direction", "split", "query_id"]
    for idx, (_, group) in enumerate(frame.groupby(group_cols, sort=False), start=1):
        group = group.sort_values("candidate_rank").copy()
        scores = group["score"].to_numpy(dtype=float)
        top_label = str(group.iloc[0]["candidate_tree_label"])
        query_marker = str(group.iloc[0]["query_marker"])
        query_seq = str(group.iloc[0]["nucleotides"])
        exact_species = markers[query_marker]["exact_index"].get(query_seq, set()) if query_seq else set()
        group["top1_score"] = float(scores[0]) if len(scores) else math.nan
        group["score_margin_to_top1"] = float(scores[0]) - group["score"].astype(float) if len(scores) else math.nan
        group["score_margin_1_2"] = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
        group["score_margin_1_5"] = float(scores[0] - scores[min(4, len(scores) - 1)]) if len(scores) else math.nan
        group["score_margin_1_10"] = float(scores[0] - scores[min(9, len(scores) - 1)]) if len(scores) else math.nan
        group["query_exact_sequence_species_count"] = int(len(exact_species))
        group["query_exact_sequence_ambiguous"] = int(len(exact_species) > 1)
        group["candidate_is_top1"] = (group["candidate_tree_label"].astype(str) == top_label).astype(int)
        for k in TOP_KS:
            subset = group[group["candidate_rank"] <= k]
            group[f"top{k}_score_mean"] = float(subset["score"].mean()) if len(subset) else math.nan
            group[f"top{k}_score_std"] = float(subset["score"].std(ddof=0)) if len(subset) else math.nan
            for rank in RANKS:
                values = subset[f"candidate_{rank}"].astype(str).replace({"nan": "", "None": ""})
                counts = values[values != ""].value_counts()
                group[f"top{k}_{rank}_unique"] = int(len(counts))
                group[f"top{k}_{rank}_mode_count"] = int(counts.iloc[0]) if len(counts) else 0
                group[f"top{k}_{rank}_mode_fraction"] = float(counts.iloc[0] / max(len(subset), 1)) if len(counts) else 0.0
                group[f"candidate_top{k}_{rank}_support"] = group[f"candidate_{rank}"].astype(str).map(counts).fillna(0).astype(int)
                group[f"candidate_top{k}_{rank}_support_fraction"] = group[f"candidate_top{k}_{rank}_support"] / max(len(subset), 1)
        out_frames.append(group)
        if idx == 1 or idx % 500 == 0:
            logger.log(f"Added candidate-generator query features groups={idx}")
    return pd.concat(out_frames, ignore_index=True)


def add_candidate_evidence(
    frame: pd.DataFrame,
    markers: dict[str, dict[str, Any]],
    tree_cache: dict[str, Any] | None,
    resolvability: dict[tuple[str, str], dict[str, Any]],
    logger: Logger,
) -> pd.DataFrame:
    seq_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    top1_by_query = frame[frame["candidate_rank"] == 1].set_index(["model", "direction", "split", "query_id"])["candidate_tree_label"].astype(str).to_dict()
    rows = []
    for idx, row in frame.iterrows():
        query_marker = str(row["query_marker"])
        target_marker = str(row["target_marker"])
        candidate_label = str(row["candidate_tree_label"])
        query_seq = str(row.get("nucleotides", ""))
        same_marker = best_same_marker_evidence(query_seq, candidate_label, query_marker, markers, seq_cache)
        top_key = (row["model"], row["direction"], row["split"], row["query_id"])
        top_label = top1_by_query.get(top_key, "")
        item = row.drop(labels=["nucleotides"], errors="ignore").to_dict()
        item.update(same_marker)
        item["candidate_has_query_marker_reference"] = int(candidate_label in markers[query_marker]["sequences"])
        item["candidate_query_marker_reference_count"] = int(len(markers[query_marker]["sequences"].get(candidate_label, [])))
        item["candidate_has_target_marker_reference"] = int(candidate_label in markers[target_marker]["sequences"])
        item["candidate_target_marker_reference_count"] = int(len(markers[target_marker]["sequences"].get(candidate_label, [])))
        item["candidate_has_both_marker_references"] = int(
            item["candidate_has_query_marker_reference"] and item["candidate_has_target_marker_reference"]
        )
        item["tree_distance_to_top1_candidate"] = tree_distance(candidate_label, top_label, tree_cache)
        query_label = clean(row.get("query_tree_label", "")) or clean(row.get("input_query_tree_label", ""))
        item.update(resolvability_features("query_marker", query_marker, query_label, resolvability))
        item.update(resolvability_features("candidate_query_marker", query_marker, candidate_label, resolvability))
        item.update(resolvability_features("candidate_target_marker", target_marker, candidate_label, resolvability))
        item["production_label_available"] = int(bool(query_label))
        rows.append(item)
        if idx == 0 or (idx + 1) % 50000 == 0:
            logger.log(f"Added candidate-generator evidence rows={idx + 1}/{len(frame)}")
    return pd.DataFrame(rows)


def feature_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    blocked_prefixes = ("match_", "query_", "input_query_")
    blocked = {
        "model",
        "split",
        "direction",
        "query_id",
        "query_marker",
        "target_marker",
        "candidate_tree_label",
        "candidate_species",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
        "query_tree_label",
        "input_query_tree_label",
        "query_sequence_hash",
    }
    rows = []
    for col in frame.columns:
        if col in blocked or any(col.startswith(prefix) for prefix in blocked_prefixes):
            role = "label_or_identifier"
        elif pd.api.types.is_numeric_dtype(frame[col]):
            role = "production_numeric_feature"
        else:
            role = "metadata_or_categorical"
        rows.append({"column": col, "role": role, "non_null": int(frame[col].notna().sum())})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(["model", "direction", "split"], dropna=False):
        model, direction, split = key
        top = group[group["candidate_rank"] == 1]
        row = {
            "model": model,
            "direction": direction,
            "split": split,
            "n_candidate_rows": int(len(group)),
            "n_query": int(group["query_id"].nunique()),
            "top1_same_marker_sequence_available_rate_pct": 100.0 * float(top["same_marker_sequence_available"].mean()) if len(top) else math.nan,
            "top1_same_marker_exact_match_rate_pct": 100.0 * float(top["same_marker_exact_match"].mean()) if len(top) else math.nan,
            "top1_same_marker_identity_mean": float(top["same_marker_best_identity"].mean()) if len(top) else math.nan,
            "top1_candidate_has_both_marker_references_rate_pct": 100.0 * float(top["candidate_has_both_marker_references"].mean()) if len(top) else math.nan,
            "query_exact_sequence_ambiguous_rate_pct": 100.0 * float(top["query_exact_sequence_ambiguous"].mean()) if len(top) else math.nan,
            "label_available_rate_pct": 100.0 * float(top["production_label_available"].mean()) if len(top) else math.nan,
        }
        for k in [1, 5, 10, 50]:
            subset = group[group["candidate_rank"] <= k]
            if subset.empty:
                continue
            per_query = subset.groupby("query_id")
            for rank in RANKS:
                col = "match_species_tree_label" if rank == "species" and "match_species_tree_label" in subset.columns else f"match_{rank}"
                if col in subset.columns:
                    row[f"known_{rank}_top{k}_pct"] = 100.0 * float(per_query[col].max().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_candidate_generator_handoff.log")
    logger.log(f"Arguments: {vars(args)}")
    query_input = infer_query_input(args.candidates, args.query_input)
    queries = load_query_table(query_input, args, logger)
    candidates = pd.read_csv(args.candidates)
    logger.log(f"Loaded candidate-generator candidates rows={len(candidates)} queries={candidates['query_id'].nunique()}")
    if candidates.empty:
        raise RuntimeError("Candidate table is empty.")
    candidates["model"] = args.model_name
    candidates["direction"] = candidates["query_marker"].astype(str) + "->" + candidates["target_marker"].astype(str)
    candidates["split"] = args.split
    candidates = add_query_columns(candidates, queries, logger)
    markers = load_marker_sequences(args)
    tree_cache = load_tree_cache(args.tree_file, not args.disable_tree, logger)
    resolvability = load_resolvability_table(args.marker_resolvability_table, args.marker_resolvability_identity, logger)
    joined = add_query_group_features(candidates, markers, logger)
    joined = add_candidate_evidence(joined, markers, tree_cache, resolvability, logger)
    summary = summarize(joined)
    inventory = feature_inventory(joined)

    evidence_path = args.output_dir / "marker_mirror_candidate_generator_evidence_handoff.csv.gz"
    joined.to_csv(evidence_path, index=False)
    summary.to_csv(args.output_dir / "marker_mirror_candidate_generator_evidence_handoff_summary.csv", index=False)
    inventory.to_csv(args.output_dir / "marker_mirror_candidate_generator_evidence_handoff_feature_inventory.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_marker_mirror_candidate_generator_handoff.py",
        "candidates": str(args.candidates),
        "query_input": str(query_input),
        "rows": int(len(joined)),
        "queries": int(joined["query_id"].nunique()),
        "tree_evidence_enabled": tree_cache is not None,
        "marker_resolvability_table": str(args.marker_resolvability_table),
        "marker_resolvability_identity": args.marker_resolvability_identity,
        "marker_resolvability_rows": len(resolvability),
        "production_numeric_feature_count": int((inventory["role"] == "production_numeric_feature").sum()),
        "claim_boundary": "Candidate-generator evidence handoff only. Labels, if present, are evaluation diagnostics and not production features.",
    }
    (args.output_dir / "marker_mirror_candidate_generator_evidence_handoff_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote {evidence_path} rows={len(joined)}")


if __name__ == "__main__":
    main()
