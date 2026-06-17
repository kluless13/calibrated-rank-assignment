#!/usr/bin/env python3
"""Build manuscript-facing 12S/eDNA evidence-decomposition tables.

This script consumes the merged 12S/eDNA source tables and writes compact
tables that answer three questions:

1. Which evidence arm performs best at each taxonomic rank?
2. How much does each arm improve over sequence-only and BLAST baselines?
3. At what score thresholds can a method make rank-specific calls instead of
   forcing species labels?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
RANKS = ("species", "genus", "family", "order")
TOP_KS = (1, 5, 10)
TARGET_ACCURACIES = (50.0, 70.0, 80.0, 90.0, 95.0)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def evidence_arm(row: pd.Series) -> str:
    context = clean(row.get("context"))
    encoder = clean(row.get("encoder"))
    prior = clean(row.get("prior_source"))

    if encoder == "blast" and context == "sequence_only":
        return "classical_sequence_only_blast"
    if context == "sequence_only":
        return "neural_sequence_tree_only"
    if context == "sequence_plus_geography":
        return "sequence_tree_plus_geography"
    if context == "sequence_plus_same_sample_cooccurrence":
        return "sequence_tree_plus_same_sample_cooccurrence"
    if context == "learned_cooccurrence" and prior == "rls_obis":
        return "sequence_tree_plus_learned_rls_obis_cooccurrence"
    if context == "learned_cooccurrence" and prior == "fishglob_public_50k":
        return "sequence_tree_plus_learned_fishglob_cooccurrence"
    if context == "geography_prior_only":
        return "geography_prior_only"
    if context == "cooccurrence_prior_only":
        return "same_sample_cooccurrence_prior_only"
    if encoder == "blast":
        return "classical_blast_plus_context"
    return context or "unknown"


def evidence_family(arm: str) -> str:
    if arm.startswith("classical"):
        return "classical_sequence"
    if arm.endswith("prior_only"):
        return "context_without_query_sequence"
    if "plus_geography" in arm:
        return "sequence_plus_geography"
    if "cooccurrence" in arm:
        return "sequence_plus_cooccurrence"
    if arm == "neural_sequence_tree_only":
        return "sequence_tree_only"
    return "other"


def run_label(row: pd.Series) -> str:
    parts = [clean(row.get("run_name"))]
    prior = clean(row.get("prior_source"))
    weight = clean(row.get("prior_weight"))
    if prior:
        parts.append(prior)
    if weight:
        parts.append(f"w={weight}")
    return " | ".join(part for part in parts if part)


def build_asv_wide(asv: pd.DataFrame) -> list[dict[str, Any]]:
    key_cols = [
        "run_name",
        "dataset",
        "encoder",
        "seed",
        "context",
        "prior_source",
        "prior_weight",
        "prediction_rows",
        "sample_query_rows",
        "sample_count",
        "source",
    ]
    rows: list[dict[str, Any]] = []
    for _, group in asv.groupby(key_cols, dropna=False):
        first = group.iloc[0]
        arm = evidence_arm(first)
        out: dict[str, Any] = {
            "run_name": clean(first.get("run_name")),
            "run_label": run_label(first),
            "dataset": clean(first.get("dataset")),
            "encoder": clean(first.get("encoder")),
            "seed": clean(first.get("seed")),
            "context": clean(first.get("context")),
            "prior_source": clean(first.get("prior_source")),
            "prior_weight": clean(first.get("prior_weight")),
            "evidence_arm": arm,
            "evidence_family": evidence_family(arm),
            "prediction_rows": clean(first.get("prediction_rows")),
            "sample_query_rows": clean(first.get("sample_query_rows")),
            "sample_count": clean(first.get("sample_count")),
            "source": clean(first.get("source")),
        }
        for _, metric_row in group.iterrows():
            rank = clean(metric_row.get("rank"))
            if rank not in RANKS:
                continue
            for top_k in TOP_KS:
                out[f"asv_{rank}_top{top_k}_pct"] = as_float(metric_row.get(f"top{top_k}_pct"))
        rows.append(out)
    return sorted(rows, key=lambda row: (row["evidence_family"], row["evidence_arm"], row["run_name"]))


def sample_metric_lookup(sample: pd.DataFrame) -> dict[tuple[str, str, int, str], float]:
    lookup: dict[tuple[str, str, int, str], float] = {}
    for _, row in sample.iterrows():
        key = (
            clean(row.get("run_name")),
            clean(row.get("rank")),
            int(as_float(row.get("top_k")) or 0),
            clean(row.get("metric")),
        )
        value = as_float(row.get("mean"))
        if value is not None:
            lookup[key] = value
    return lookup


def add_sample_metrics(rows: list[dict[str, Any]], sample: pd.DataFrame) -> None:
    lookup = sample_metric_lookup(sample)
    for row in rows:
        run = row["run_name"]
        for rank in RANKS:
            for top_k in (1, 10):
                for metric in ("precision", "recall", "jaccard"):
                    row[f"sample_{rank}_top{top_k}_{metric}_mean"] = lookup.get((run, rank, top_k, metric))


def best_sequence_baselines(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    sequence_best: dict[str, float] = {}
    blast_best: dict[str, float] = {}
    for rank in RANKS:
        column = f"asv_{rank}_top10_pct"
        seq_values = [
            value
            for row in rows
            if row["context"] == "sequence_only"
            for value in [as_float(row.get(column))]
            if value is not None
        ]
        blast_values = [
            value
            for row in rows
            if row["encoder"] == "blast"
            for value in [as_float(row.get(column))]
            if value is not None
        ]
        sequence_best[rank] = max(seq_values) if seq_values else float("nan")
        blast_best[rank] = max(blast_values) if blast_values else float("nan")
    return sequence_best, blast_best


def build_best_by_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequence_best, blast_best = best_sequence_baselines(rows)
    out_rows: list[dict[str, Any]] = []
    for arm in sorted({row["evidence_arm"] for row in rows}):
        arm_rows = [row for row in rows if row["evidence_arm"] == arm]
        for rank in RANKS:
            top10_col = f"asv_{rank}_top10_pct"
            valid = [row for row in arm_rows if as_float(row.get(top10_col)) is not None]
            if not valid:
                continue
            best = max(valid, key=lambda row: as_float(row.get(top10_col)) or -1.0)
            top10 = as_float(best.get(top10_col))
            seq_base = sequence_best.get(rank)
            blast_base = blast_best.get(rank)
            out_rows.append(
                {
                    "evidence_arm": arm,
                    "evidence_family": best["evidence_family"],
                    "rank": rank,
                    "best_run_name": best["run_name"],
                    "encoder": best["encoder"],
                    "prior_source": best["prior_source"],
                    "prior_weight": best["prior_weight"],
                    "asv_top1_pct": best.get(f"asv_{rank}_top1_pct"),
                    "asv_top5_pct": best.get(f"asv_{rank}_top5_pct"),
                    "asv_top10_pct": top10,
                    "sample_top10_jaccard_mean": best.get(f"sample_{rank}_top10_jaccard_mean"),
                    "delta_vs_best_sequence_only_top10_pct": None
                    if top10 is None or pd.isna(seq_base)
                    else top10 - seq_base,
                    "delta_vs_best_blast_top10_pct": None
                    if top10 is None or pd.isna(blast_base)
                    else top10 - blast_base,
                }
            )
    return out_rows


def build_rank_no_call_operating_points(calibration: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = ["method", "dataset", "encoder", "context", "prior_source", "prior_weight", "rank"]
    for key, group in calibration.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        group = group.copy()
        group["assignment_rate_pct_float"] = group["assignment_rate_pct"].map(as_float)
        group["rank_accuracy_pct_float"] = group["rank_accuracy_pct"].map(as_float)
        for target in TARGET_ACCURACIES:
            eligible = group[group["rank_accuracy_pct_float"] >= target].copy()
            if eligible.empty:
                rows.append(
                    {
                        **{col: clean(value) for col, value in meta.items()},
                        "target_accuracy_pct": target,
                        "status": "no_operating_point",
                        "threshold": "",
                        "assignment_rate_pct": 0.0,
                        "rank_accuracy_pct": "",
                        "n_assigned": 0,
                        "interpretation": "No score threshold reached this target accuracy on the current validation table.",
                    }
                )
                continue
            best = eligible.sort_values(
                ["assignment_rate_pct_float", "rank_accuracy_pct_float"],
                ascending=[False, False],
            ).iloc[0]
            rows.append(
                {
                    **{col: clean(value) for col, value in meta.items()},
                    "target_accuracy_pct": target,
                    "status": "available",
                    "threshold": best.get("threshold", ""),
                    "assignment_rate_pct": best.get("assignment_rate_pct", ""),
                    "rank_accuracy_pct": best.get("rank_accuracy_pct", ""),
                    "n_assigned": best.get("n_assigned", ""),
                    "interpretation": "Use this row as a diagnostic rank/no-call operating point; independent calibration is still required before manuscript claims.",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    args = parser.parse_args()

    source_dir = args.source_dir
    logger = ProgressLogger(default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    asv_path = source_dir / "merged_global_edna_asv_metrics.csv"
    sample_path = source_dir / "merged_global_edna_sample_metrics.csv"
    calibration_path = source_dir / "merged_global_edna_calibration_curves.csv"

    logger.log(f"Reading {rel(asv_path)}")
    asv = pd.read_csv(asv_path)
    logger.log(f"Reading {rel(sample_path)}")
    sample = pd.read_csv(sample_path)
    logger.log(f"Reading {rel(calibration_path)}")
    calibration = pd.read_csv(calibration_path)

    logger.log("Building ASV/sample-wide evidence matrix")
    evidence_rows = build_asv_wide(asv)
    add_sample_metrics(evidence_rows, sample)

    logger.log("Building best-by-rank decomposition rows")
    best_rows = build_best_by_rank(evidence_rows)

    logger.log("Building diagnostic rank/no-call operating points")
    no_call_rows = build_rank_no_call_operating_points(calibration)

    evidence_fields = [
        "run_name",
        "run_label",
        "dataset",
        "encoder",
        "seed",
        "context",
        "prior_source",
        "prior_weight",
        "evidence_arm",
        "evidence_family",
        "prediction_rows",
        "sample_query_rows",
        "sample_count",
    ]
    for rank in RANKS:
        for top_k in TOP_KS:
            evidence_fields.append(f"asv_{rank}_top{top_k}_pct")
    for rank in RANKS:
        for top_k in (1, 10):
            for metric in ("precision", "recall", "jaccard"):
                evidence_fields.append(f"sample_{rank}_top{top_k}_{metric}_mean")
    evidence_fields.append("source")

    logger.log("Writing edna_evidence_decomposition_matrix.csv")
    write_csv(source_dir / "edna_evidence_decomposition_matrix.csv", evidence_rows, evidence_fields)

    logger.log("Writing edna_evidence_best_by_rank.csv")
    write_csv(
        source_dir / "edna_evidence_best_by_rank.csv",
        best_rows,
        [
            "evidence_arm",
            "evidence_family",
            "rank",
            "best_run_name",
            "encoder",
            "prior_source",
            "prior_weight",
            "asv_top1_pct",
            "asv_top5_pct",
            "asv_top10_pct",
            "sample_top10_jaccard_mean",
            "delta_vs_best_sequence_only_top10_pct",
            "delta_vs_best_blast_top10_pct",
        ],
    )

    logger.log("Writing edna_rank_no_call_operating_points.csv")
    write_csv(
        source_dir / "edna_rank_no_call_operating_points.csv",
        no_call_rows,
        [
            "method",
            "dataset",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "rank",
            "target_accuracy_pct",
            "status",
            "threshold",
            "assignment_rate_pct",
            "rank_accuracy_pct",
            "n_assigned",
            "interpretation",
        ],
    )

    manifest = {
        "inputs": {
            "asv_metrics": rel(asv_path),
            "sample_metrics": rel(sample_path),
            "calibration_curves": rel(calibration_path),
        },
        "outputs": {
            "evidence_decomposition_matrix": rel(source_dir / "edna_evidence_decomposition_matrix.csv"),
            "evidence_best_by_rank": rel(source_dir / "edna_evidence_best_by_rank.csv"),
            "rank_no_call_operating_points": rel(source_dir / "edna_rank_no_call_operating_points.csv"),
        },
        "row_counts": {
            "evidence_decomposition_matrix": len(evidence_rows),
            "evidence_best_by_rank": len(best_rows),
            "rank_no_call_operating_points": len(no_call_rows),
        },
        "notes": [
            "Evidence-decomposition rows summarize ASV-level and sample-level Global_eDNA validation arms.",
            "Rank/no-call operating points are diagnostic because they are derived from the current validation table; independent calibration remains a manuscript requirement.",
            "Deltas compare ASV top-10 against the best available sequence-only and BLAST-family rows for the same taxonomic rank.",
        ],
    }
    manifest_path = source_dir / "edna_evidence_decomposition_manifest.json"
    logger.log("Writing edna_evidence_decomposition_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
