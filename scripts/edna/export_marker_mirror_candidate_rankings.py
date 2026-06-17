#!/usr/bin/env python3
"""Export MarkerMirror per-query candidate rankings.

The MarkerMirror training scripts write aggregate top-k metrics.  The pipeline
needs the actual candidate list per query:

    query marker fragment -> ranked candidate species from another marker

This exporter reloads a saved projection head, rebuilds the same held-out
species splits, recomputes frozen foundation embeddings, and writes auditable
top-k candidate tables for downstream reranking, rank/no-call calibration, and
reason-code analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import (
    Logger,
    ProjectionHead,
    clean_sequence,
    embed_sequences,
    flatten,
    load_hf,
    load_species_json,
    load_taxonomy,
    split_species,
)

RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shared_12s_16s", "triad_12s_16s_coi"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--marker-a-name", default="12S")
    parser.add_argument("--marker-a-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource")
    parser.add_argument("--marker-b-name", default="16S")
    parser.add_argument("--marker-b-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource")
    parser.add_argument("--marker-c-name", default="COI")
    parser.add_argument("--marker-c-input-dir", type=Path, default=ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs" / "eval_c")
    parser.add_argument("--max-a-per-species", type=int, default=4)
    parser.add_argument("--max-b-per-species", type=int, default=4)
    parser.add_argument("--max-c-per-species", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--include-frozen", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def marker_specs(args: argparse.Namespace) -> list[tuple[str, Path, int]]:
    specs = [
        (args.marker_a_name, args.marker_a_input_dir, args.max_a_per_species),
        (args.marker_b_name, args.marker_b_input_dir, args.max_b_per_species),
    ]
    if args.mode == "triad_12s_16s_coi":
        specs.append((args.marker_c_name, args.marker_c_input_dir, args.max_c_per_species))
    return specs


def infer_projection_head(state: dict[str, torch.Tensor], device: str) -> ProjectionHead:
    input_dim = int(state["net.0.weight"].shape[1])
    hidden_dim = int(state["net.0.weight"].shape[0])
    output_dim = int(state["net.3.weight"].shape[0])
    head = ProjectionHead(input_dim, hidden_dim, output_dim).to(device)
    head.load_state_dict(state)
    head.eval()
    return head


def load_projection_head(path: Path, device: str) -> tuple[ProjectionHead, dict[str, Any]]:
    # These checkpoints are produced by our local training scripts and include
    # the original argparse payload.  PyTorch 2.6+ defaults to weights-only
    # loading, which rejects pathlib values in that payload.
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("shared_head", payload)
    if not isinstance(state, dict):
        raise ValueError(f"Could not find shared_head state dict in {path}")
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    return infer_projection_head(state, device), args


def project(head: ProjectionHead | None, arr: np.ndarray, device: str) -> np.ndarray:
    if head is None:
        z = torch.tensor(arr, device=device)
        return nn.functional.normalize(z, dim=-1).cpu().numpy()
    outs: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(arr), 512):
            z = torch.tensor(arr[start : start + 512], device=device)
            outs.append(head(z).cpu().numpy())
    return np.vstack(outs).astype("float32")


def build_prototypes(rows: pd.DataFrame, emb: np.ndarray, tax: dict[str, dict[str, str]]) -> tuple[pd.DataFrame, np.ndarray]:
    proto_rows = []
    proto_vecs = []
    for label, idxs in rows.groupby("tree_label").groups.items():
        label = str(label)
        proto_rows.append({"tree_label": label, **tax.get(label, {})})
        proto_vecs.append(emb[list(idxs)].mean(axis=0))
    proto = pd.DataFrame(proto_rows)
    proto_arr = np.vstack(proto_vecs).astype("float32")
    proto_arr /= np.linalg.norm(proto_arr, axis=1, keepdims=True).clip(min=1e-8)
    return proto, proto_arr


def tax_value(tax: dict[str, dict[str, str]], label: str, rank: str) -> str:
    if rank == "species":
        return str(tax.get(label, {}).get("species") or label.replace("_", " "))
    if rank == "genus":
        return str(tax.get(label, {}).get("genus") or label.split("_")[0])
    return str(tax.get(label, {}).get(rank) or "")


def pair_species(markers: dict[str, dict[str, Any]], left: str, right: str, labels: list[str]) -> list[str]:
    left_species = markers[left]["species"]
    right_species = markers[right]["species"]
    return [label for label in labels if label in left_species and label in right_species and left_species[label] and right_species[label]]


def export_direction(
    *,
    run_name: str,
    model_name: str,
    split_name: str,
    query_marker: str,
    target_marker: str,
    labels: list[str],
    markers: dict[str, dict[str, Any]],
    tax: dict[str, dict[str, str]],
    head: ProjectionHead | None,
    top_k: int,
    device: str,
) -> pd.DataFrame:
    query_rows = markers[query_marker]["rows"]
    query_emb = markers[query_marker]["emb"]
    target_rows = markers[target_marker]["rows"]
    target_emb = markers[target_marker]["emb"]
    mask = query_rows["tree_label"].isin(labels).to_numpy()
    query_rows = query_rows[mask].reset_index(drop=True)
    query_emb = query_emb[mask]
    if query_rows.empty:
        return pd.DataFrame()

    q = project(head, query_emb, device)
    c = project(head, target_emb, device)
    proto, proto_arr = build_prototypes(target_rows, c, tax)
    sims = q @ proto_arr.T
    k = min(top_k, sims.shape[1])
    top_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(top_idx, np.argsort(-np.take_along_axis(sims, top_idx, axis=1), axis=1), axis=1)

    out = []
    for row_i, indices in enumerate(ordered):
        query = query_rows.iloc[row_i]
        true_label = str(query["tree_label"])
        for rank_i, proto_i in enumerate(indices, start=1):
            candidate = proto.iloc[int(proto_i)]
            candidate_label = str(candidate["tree_label"])
            row: dict[str, Any] = {
                "run": run_name,
                "model": model_name,
                "split": split_name,
                "direction": f"{query_marker}_to_{target_marker}",
                "query_marker": query_marker,
                "target_marker": target_marker,
                "query_id": f"{query_marker}:{true_label}:{int(query['seq_index'])}",
                "query_tree_label": true_label,
                "query_seq_index": int(query["seq_index"]),
                "candidate_rank": rank_i,
                "candidate_tree_label": candidate_label,
                "score": float(sims[row_i, int(proto_i)]),
            }
            for rank in RANKS:
                qv = tax_value(tax, true_label, rank)
                cv = tax_value(tax, candidate_label, rank)
                row[f"query_{rank}"] = qv
                row[f"candidate_{rank}"] = cv
                row[f"match_{rank}"] = bool(qv and qv == cv)
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_candidate_rankings.log")
    logger.log(f"Arguments: {vars(args)}")

    head, checkpoint_args = load_projection_head(args.checkpoint, args.device)
    seed = int(args.seed if args.seed is not None else checkpoint_args.get("seed", 1901))
    logger.log(f"Loaded checkpoint {args.checkpoint}; seed={seed}")

    specs = marker_specs(args)
    markers: dict[str, dict[str, Any]] = {}
    tax: dict[str, dict[str, str]] = {}
    for name, input_dir, max_per_species in specs:
        species = load_species_json(input_dir / "train_species_sequences.json", max_per_species)
        markers[name] = {"input_dir": input_dir, "species": species}
        tax.update(load_taxonomy(input_dir / "candidate_species.csv"))

    if args.mode == "shared_12s_16s":
        split_base = sorted(set(markers[args.marker_a_name]["species"]) & set(markers[args.marker_b_name]["species"]))
    else:
        split_base = sorted(set().union(*(set(marker["species"]) for marker in markers.values())))
    splits = split_species(split_base, seed)
    logger.log(f"Split base species={len(split_base)} train/val/test={[len(splits[k]) for k in ['train','val','test']]}")

    tokenizer, model, masked_lm = load_hf(args.model, args.device, logger)
    for name, marker in markers.items():
        rows = flatten(marker["species"], sorted(marker["species"]), name)
        marker["rows"] = rows
        marker["emb"] = embed_sequences(
            rows["nucleotides"].map(clean_sequence).tolist(),
            tokenizer,
            model,
            masked_lm,
            args.device,
            args.embed_batch_size,
            args.max_length,
            logger,
            name,
        )

    directions: list[tuple[str, str]] = []
    names = [name for name, _, _ in specs]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            directions.append((left, right))
            directions.append((right, left))

    frames = []
    run_name = args.checkpoint.parent.name
    for split_name, split_labels in splits.items():
        for query_marker, target_marker in directions:
            labels = pair_species(markers, query_marker, target_marker, split_labels)
            if not labels:
                continue
            if args.include_frozen:
                logger.log(f"Exporting frozen {split_name} {query_marker}->{target_marker} labels={len(labels)}")
                frames.append(
                    export_direction(
                        run_name=run_name,
                        model_name="frozen_nt_marker_mirror",
                        split_name=split_name,
                        query_marker=query_marker,
                        target_marker=target_marker,
                        labels=labels,
                        markers=markers,
                        tax=tax,
                        head=None,
                        top_k=args.top_k,
                        device=args.device,
                    )
                )
            logger.log(f"Exporting learned {split_name} {query_marker}->{target_marker} labels={len(labels)}")
            frames.append(
                export_direction(
                    run_name=run_name,
                    model_name="marker_mirror_projection",
                    split_name=split_name,
                    query_marker=query_marker,
                    target_marker=target_marker,
                    labels=labels,
                    markers=markers,
                    tax=tax,
                    head=head,
                    top_k=args.top_k,
                    device=args.device,
                )
            )

    out = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    out_path = args.output_dir / "marker_mirror_candidate_rankings.csv.gz"
    out.to_csv(out_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "run": run_name,
        "model": args.model,
        "seed": seed,
        "top_k": args.top_k,
        "include_frozen": bool(args.include_frozen),
        "rows": int(len(out)),
        "directions": [f"{left}->{right}" for left, right in directions],
        "splits": {key: len(value) for key, value in splits.items()},
        "claim_boundary": "Candidate-generation export only; downstream rank/no-call calibration is required for final assignments.",
    }
    (args.output_dir / "marker_mirror_candidate_rankings_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.log(f"Wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()
