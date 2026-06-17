#!/usr/bin/env python3
"""Evaluate an APPLES-like local distance-placement baseline.

This is not the official APPLES implementation. It is an explicitly labelled
distance-placement comparator that uses the same prepared placement inputs as
EPA-ng, computes aligned query-to-reference p-distances inside a candidate
neighborhood, and scores the nearest reference placement through the same
tree-distance/rank-backoff layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio import SeqIO

from eval_zero_shot_candidate_predictions import split_ranked_labels
from progress_logging import ProgressLogger, default_log_path
from score_fish_tree_placement_outputs import (
    RANKS,
    canonical_label,
    clean,
    load_candidate_taxonomy,
    load_query_taxonomy,
    load_tree_distance_index,
    min_tree_distance_to_labels,
    most_specific_rank,
    tree_distance,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLACEMENT_ROOT = Path(
    "results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/phylo_placement"
)
DEFAULT_BASELINE_ROOT = Path(
    "results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment"
)
DEFAULT_TREE_FILE = Path("data/phylo/actinopt_12k_treePL.tre")


def encode_sequence(text: str) -> np.ndarray:
    arr = np.full(len(text), 5, dtype=np.int8)
    mapping = {
        ord("A"): 0,
        ord("C"): 1,
        ord("G"): 2,
        ord("T"): 3,
        ord("-"): 4,
    }
    raw = np.frombuffer(text.upper().encode("ascii", errors="ignore"), dtype=np.uint8)
    arr = np.full(raw.shape[0], 5, dtype=np.int8)
    for code, value in mapping.items():
        arr[raw == code] = value
    return arr


def load_alignment(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for record in SeqIO.parse(path, "fasta"):
        out[canonical_label(record.id)] = encode_sequence(str(record.seq))
    return out


def query_alignment_key(record_id: str) -> str:
    parts = str(record_id).split("|")
    if len(parts) >= 2:
        return parts[1]
    return record_id


def load_query_alignment(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for record in SeqIO.parse(path, "fasta"):
        out[query_alignment_key(record.id)] = encode_sequence(str(record.seq))
        out[canonical_label(record.id)] = out[query_alignment_key(record.id)]
    return out


def p_distance(query: np.ndarray, reference: np.ndarray) -> tuple[float, int, int]:
    valid = (query < 4) & (reference < 4)
    comparable = int(valid.sum())
    if comparable == 0:
        return float("nan"), 0, 0
    mismatches = int((query[valid] != reference[valid]).sum())
    return mismatches / comparable, comparable, mismatches


def finite_median(values: Iterable[object]) -> float | None:
    finite = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    return float(pd.Series(finite).median()) if finite else None


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for (split, method), sub in df.groupby(["split", "method"], dropna=False):
        row: dict[str, object] = {
            "split": split,
            "method": method,
            "n_queries_scored": int(len(sub)),
            "missing_query_metadata": int(pd.to_numeric(sub["missing_query_metadata"], errors="coerce").fillna(0).sum()),
            "candidate_neighborhood_size_median": finite_median(sub["candidate_neighborhood_size"]),
            "aligned_p_distance_median": finite_median(sub["aligned_p_distance"]),
            "placement_min_tree_distance_median": finite_median(sub["placement_min_tree_distance_to_placed_clade"]),
            "placement_min_tree_distance_mean": float(
                pd.to_numeric(sub["placement_min_tree_distance_to_placed_clade"], errors="coerce").dropna().mean()
            ),
            "placement_excess_tree_distance_median": finite_median(sub["placement_excess_tree_distance_vs_nearest_reference"]),
            "placement_excess_tree_distance_mean": float(
                pd.to_numeric(sub["placement_excess_tree_distance_vs_nearest_reference"], errors="coerce").dropna().mean()
            ),
            "nearest_reference_match_rate": float(sub["nearest_reference_match"].astype(bool).mean()),
            "source_file": str(sub["source_file"].iloc[0]),
        }
        for rank in RANKS:
            row[f"{rank}_in_placed_clade_rate"] = float(sub[f"{rank}_in_placed_clade"].astype(bool).mean())
        for rank in list(RANKS) + ["none"]:
            row[f"most_specific_rank_{rank}_rate"] = float((sub["most_specific_placed_rank"] == rank).mean())
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_split(
    split: str,
    args: argparse.Namespace,
    taxon_nodes: dict[str, object],
    depths: dict[object, float],
    ancestor_lists: dict[str, list[object]],
    ancestor_sets: dict[str, set[object]],
    logger: ProgressLogger,
) -> list[dict[str, object]]:
    split_root = args.placement_root / split
    prediction_path = args.baseline_root / f"baselines_{split}" / args.candidate_source / "zero_shot_candidate_predictions.csv"
    logger.log(f"Loading {split} APPLES-like inputs from {split_root}")
    reference_alignment = load_alignment(split_root / "reference_msa.fasta")
    query_alignment = load_query_alignment(split_root / "query_msa.fasta")
    candidate_tax = load_candidate_taxonomy(split_root)
    query_tax = load_query_taxonomy(split_root)
    reference_labels = sorted(reference_alignment)
    predictions = pd.read_csv(prediction_path)
    logger.log(
        f"Scoring {len(predictions)} {split} queries against {len(reference_labels)} references "
        f"using {args.candidate_source} top-{args.candidate_top_k} neighborhoods"
    )

    nearest_reference_cache: dict[str, tuple[float, str]] = {}
    rows: list[dict[str, object]] = []
    method = f"apples_like_{args.candidate_source}_distance_top{args.candidate_top_k}"
    for index, pred in predictions.iterrows():
        processid = clean(pred.get("processid", ""))
        truth = query_tax.get(processid, {})
        true_species = canonical_label(truth.get("species", pred.get("true_tree_label", "")))
        q = query_alignment.get(processid)
        top_labels = [canonical_label(label) for label in split_ranked_labels(pred.get("top_tree_labels", ""))]
        top_labels = [label for label in top_labels if label in reference_alignment][: args.candidate_top_k]

        best_label = ""
        best_distance = float("nan")
        best_comparable = 0
        best_mismatches = 0
        if q is not None:
            for label in top_labels:
                distance, comparable, mismatches = p_distance(q, reference_alignment[label])
                if not math.isfinite(distance):
                    continue
                if not math.isfinite(best_distance) or distance < best_distance:
                    best_label = label
                    best_distance = distance
                    best_comparable = comparable
                    best_mismatches = mismatches

        nearest_reference_distance = float("nan")
        nearest_reference_label = ""
        placed_distance = float("nan")
        excess_distance = float("nan")
        if true_species:
            if true_species not in nearest_reference_cache:
                nearest_reference_cache[true_species] = min_tree_distance_to_labels(
                    true_species,
                    reference_labels,
                    taxon_nodes,
                    depths,
                    ancestor_lists,
                    ancestor_sets,
                )
            nearest_reference_distance, nearest_reference_label = nearest_reference_cache[true_species]
            if best_label:
                placed_distance = tree_distance(
                    true_species,
                    best_label,
                    taxon_nodes,
                    depths,
                    ancestor_lists,
                    ancestor_sets,
                )
                if math.isfinite(placed_distance) and math.isfinite(nearest_reference_distance):
                    excess_distance = placed_distance - nearest_reference_distance

        placed_tax = candidate_tax.get(best_label, {})
        row: dict[str, object] = {
            "split": split,
            "method": method,
            "processid": processid,
            "candidate_source": args.candidate_source,
            "candidate_neighborhood_size": len(top_labels),
            "placed_tree_label": best_label,
            "aligned_p_distance": best_distance,
            "aligned_comparable_sites": best_comparable,
            "aligned_mismatches": best_mismatches,
            "nearest_reference_tree_label": nearest_reference_label,
            "nearest_reference_tree_distance": nearest_reference_distance,
            "nearest_placed_clade_tree_label": best_label,
            "placement_min_tree_distance_to_placed_clade": placed_distance,
            "placement_excess_tree_distance_vs_nearest_reference": excess_distance,
            "nearest_reference_match": bool(math.isfinite(excess_distance) and abs(excess_distance) <= args.tree_distance_tolerance),
            "missing_query_metadata": int(processid not in query_tax),
            "source_file": str(prediction_path),
        }
        for rank in RANKS:
            target = true_species if rank == "species" else clean(truth.get(rank, pred.get(f"{rank}_name", "")))
            placed_value = best_label if rank == "species" else clean(placed_tax.get(rank, ""))
            row[f"{rank}_in_placed_clade"] = bool(target and placed_value and target == placed_value)
        row["most_specific_placed_rank"] = most_specific_rank(row)
        rows.append(row)
        if args.progress_every and (index + 1) % args.progress_every == 0:
            logger.log(f"{split}: scored {index + 1}/{len(predictions)} queries")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement-root", type=Path, default=DEFAULT_PLACEMENT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
    )
    parser.add_argument("--tree-file", type=Path, default=DEFAULT_TREE_FILE)
    parser.add_argument("--splits", nargs="+", default=["eval_c", "seen_test", "unseen_genera"])
    parser.add_argument("--candidate-source", choices=["blast", "vsearch", "kmer"], default="vsearch")
    parser.add_argument("--candidate-top-k", type=int, default=25)
    parser.add_argument("--tree-distance-tolerance", type=float, default=1e-9)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    logger.log(f"Loading full tree from {args.tree_file}")
    taxon_nodes, depths, ancestor_lists, ancestor_sets = load_tree_distance_index(args.tree_file)

    per_query_rows: list[dict[str, object]] = []
    for split in args.splits:
        per_query_rows.extend(
            evaluate_split(
                split,
                args,
                taxon_nodes,
                depths,
                ancestor_lists,
                ancestor_sets,
                logger,
            )
        )

    summary_rows = summarize(per_query_rows)
    per_query_path = args.output_dir / "apples_like_distance_placement_per_query.csv"
    summary_path = args.output_dir / "apples_like_distance_placement_summary.csv"
    logger.log(f"Writing {per_query_path}")
    write_csv(per_query_path, per_query_rows)
    logger.log(f"Writing {summary_path}")
    write_csv(summary_path, summary_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/eval_apples_like_distance_placement.py",
        "placement_root": str(args.placement_root),
        "baseline_root": str(args.baseline_root),
        "candidate_source": args.candidate_source,
        "candidate_top_k": args.candidate_top_k,
        "splits": args.splits,
        "per_query_rows": len(per_query_rows),
        "summary_rows": len(summary_rows),
        "outputs": {
            "per_query": str(per_query_path),
            "summary": str(summary_path),
        },
        "claim_boundary": [
            "This is APPLES-like local distance placement, not official APPLES.",
            "It uses aligned p-distance within a top-k candidate neighborhood seeded by a classical candidate generator.",
            "It should be used as a distance-placement comparator until official APPLES is installed or reproduced.",
        ],
    }
    manifest_path = args.output_dir / "apples_like_distance_placement_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
