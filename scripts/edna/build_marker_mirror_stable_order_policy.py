#!/usr/bin/env python3
"""Apply the stable MarkerMirror + BLASTN + VSEARCH order policy.

Exp 111 found one conservative policy that transfers cleanly across repeated
query-species splits: only emit an order call when MarkerMirror, BLASTN, and
VSEARCH all agree on the top-1 order. This script turns that diagnostic into an
auditable production-style assignment table with reason codes.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = Path(__file__).stem
SOURCE_PREFIXES = ("mm", "blast", "vsearch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_blast_vsearch_calibration_repair_features.csv",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_blast_vsearch_calibration_repair_thresholds.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def finite_threshold_stats(thresholds: pd.DataFrame, target: float) -> dict[str, float]:
    subset = thresholds[
        (thresholds["strategy"] == "global_precision")
        & (thresholds["policy"] == "mm_blast_vsearch_agree_top1_mode")
        & (thresholds["rank"] == "order")
        & (thresholds["source"] == "__global__")
        & (np.isclose(pd.to_numeric(thresholds["target"], errors="coerce"), target))
        & np.isfinite(pd.to_numeric(thresholds["threshold"], errors="coerce"))
    ].copy()
    values = pd.to_numeric(subset["threshold"], errors="coerce").dropna()
    if values.empty:
        return {
            "threshold_repeat_count": 0,
            "threshold_min": math.inf,
            "threshold_median": math.inf,
            "threshold_max": math.inf,
        }
    return {
        "threshold_repeat_count": int(len(values)),
        "threshold_min": float(values.min()),
        "threshold_median": float(values.median()),
        "threshold_max": float(values.max()),
    }


def agreement_reason(row: pd.Series, agreement: bool, confidence: float, locked_threshold: float) -> str:
    mm_order = clean(row.get("mm_top1_order_mode"))
    blast_order = clean(row.get("blast_top1_order_mode"))
    vsearch_order = clean(row.get("vsearch_top1_order_mode"))
    if agreement and confidence >= locked_threshold:
        return "stable_all_source_top1_order_agreement"
    if agreement:
        return "all_source_order_agreement_below_locked_threshold"
    if not mm_order or not blast_order or not vsearch_order:
        return "missing_one_or_more_order_sources"
    if blast_order == vsearch_order and blast_order != mm_order:
        return "classical_sources_agree_marker_mirror_conflicts"
    if mm_order == blast_order and mm_order != vsearch_order:
        return "marker_mirror_blast_agree_vsearch_conflicts"
    if mm_order == vsearch_order and mm_order != blast_order:
        return "marker_mirror_vsearch_agree_blast_conflicts"
    return "all_source_order_conflict"


def compute_confidence(row: pd.Series) -> float:
    values: list[float] = []
    for prefix in SOURCE_PREFIXES:
        score = pd.to_numeric(pd.Series([row.get(f"{prefix}_top1_score")]), errors="coerce").iloc[0]
        fraction = pd.to_numeric(pd.Series([row.get(f"{prefix}_top1_order_mode_fraction")]), errors="coerce").iloc[0]
        if pd.isna(score) or pd.isna(fraction):
            return math.nan
        values.append(float(score) * float(fraction))
    return min(values) if values else math.nan


def build_assignments(features: pd.DataFrame, policy_name: str, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        mm_order = clean(row.get("mm_top1_order_mode"))
        blast_order = clean(row.get("blast_top1_order_mode"))
        vsearch_order = clean(row.get("vsearch_top1_order_mode"))
        truth_order = clean(row.get("query_order"))
        confidence = compute_confidence(row)
        agreement = bool(mm_order and mm_order == blast_order == vsearch_order)
        assigned = bool(agreement and np.isfinite(confidence) and confidence >= threshold)
        assigned_taxon = mm_order if assigned else ""
        known_truth = bool(truth_order)
        rows.append(
            {
                "query_id": clean(row.get("query_id")),
                "source": clean(row.get("source")),
                "query_species": clean(row.get("query_species")),
                "query_genus": clean(row.get("query_genus")),
                "query_family": clean(row.get("query_family")),
                "query_order": truth_order,
                "policy": policy_name,
                "assigned_rank": "order" if assigned else "no_call",
                "assigned_taxon": assigned_taxon,
                "confidence": confidence,
                "correct": bool(assigned and assigned_taxon == truth_order) if known_truth else np.nan,
                "false_species_call": False,
                "reason_code": agreement_reason(row, agreement, confidence, threshold),
                "locked_threshold": threshold,
                "mm_top1_order": mm_order,
                "mm_top1_score": row.get("mm_top1_score"),
                "blast_top1_order": blast_order,
                "blast_top1_score": row.get("blast_top1_score"),
                "vsearch_top1_order": vsearch_order,
                "vsearch_top1_score": row.get("vsearch_top1_score"),
            }
        )
    return pd.DataFrame(rows)


def summarize(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, group in assignments.groupby("policy", sort=False):
        assigned = group[group["assigned_rank"] != "no_call"]
        assigned_known = assigned[assigned["correct"].notna()]
        n_correct = int(assigned_known["correct"].astype(bool).sum()) if len(assigned_known) else math.nan
        n_incorrect = int(len(assigned_known) - n_correct) if len(assigned_known) else math.nan
        rows.append(
            {
                "policy": policy,
                "n_queries": int(len(group)),
                "n_assigned": int(len(assigned)),
                "coverage_pct": pct(len(assigned), len(group)),
                "assigned_precision_pct": pct(n_correct, len(assigned_known)) if len(assigned_known) else math.nan,
                "false_species_call_rate_pct": 0.0,
                "n_no_call": int(len(group) - len(assigned)),
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "n_assigned_known_truth": int(len(assigned_known)),
                "locked_threshold": float(group["locked_threshold"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_source(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (policy, source), group in assignments.groupby(["policy", "source"], dropna=False, sort=False):
        assigned = group[group["assigned_rank"] != "no_call"]
        assigned_known = assigned[assigned["correct"].notna()]
        n_correct = int(assigned_known["correct"].astype(bool).sum()) if len(assigned_known) else math.nan
        n_incorrect = int(len(assigned_known) - n_correct) if len(assigned_known) else math.nan
        rows.append(
            {
                "policy": policy,
                "source": source,
                "n_queries": int(len(group)),
                "n_assigned": int(len(assigned)),
                "coverage_pct": pct(len(assigned), len(group)),
                "assigned_precision_pct": pct(n_correct, len(assigned_known)) if len(assigned_known) else math.nan,
                "n_no_call": int(len(group) - len(assigned)),
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
                "n_assigned_known_truth": int(len(assigned_known)),
            }
        )
    return pd.DataFrame(rows)


def production_assignments(assignments: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    frame = assignments[assignments["policy"] == policy_name].copy()
    return pd.DataFrame(
        {
            "query_id": frame["query_id"],
            "source": frame["source"],
            "decision_mode": "marker_mirror_blastn_vsearch_stable_order_v1",
            "policy": frame["policy"],
            "assigned_rank": frame["assigned_rank"],
            "assigned_label": frame["assigned_taxon"],
            "confidence": frame["confidence"],
            "assignment_reason": frame["reason_code"],
            "locked_threshold": frame["locked_threshold"],
            "mm_top1_order": frame["mm_top1_order"],
            "mm_top1_score": frame["mm_top1_score"],
            "blast_top1_order": frame["blast_top1_order"],
            "blast_top1_score": frame["blast_top1_score"],
            "vsearch_top1_order": frame["vsearch_top1_order"],
            "vsearch_top1_score": frame["vsearch_top1_score"],
        }
    )


def main() -> None:
    args = parse_args()
    log_file = args.log_file or (
        ROOT / "results" / "paper1_phylo_calibrated_assignment" / "logs" / f"{SCRIPT_NAME}.log"
    )
    logger = ProgressLogger(log_file)
    logger.start(SCRIPT_NAME)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Loading features: {rel(args.features)}")
    features = pd.read_csv(args.features)
    logger.log(f"Loaded {len(features):,} query rows")

    logger.log(f"Loading thresholds: {rel(args.thresholds)}")
    thresholds = pd.read_csv(args.thresholds)
    threshold_stats = finite_threshold_stats(thresholds, args.target)
    locked_threshold = threshold_stats["threshold_max"]
    logger.log(
        "Stable order threshold stats: "
        f"n={threshold_stats['threshold_repeat_count']}, "
        f"min={threshold_stats['threshold_min']}, "
        f"median={threshold_stats['threshold_median']}, "
        f"max={threshold_stats['threshold_max']}"
    )

    unthresholded = build_assignments(
        features=features,
        policy_name="stable_all_source_top1_order_agreement_unthresholded",
        threshold=-math.inf,
    )
    locked = build_assignments(
        features=features,
        policy_name="stable_all_source_top1_order_agreement_max_repeat_threshold",
        threshold=locked_threshold,
    )
    assignments = pd.concat([unthresholded, locked], ignore_index=True)

    summary = summarize(assignments)
    by_source = summarize_by_source(assignments)
    production = production_assignments(
        assignments,
        policy_name="stable_all_source_top1_order_agreement_max_repeat_threshold",
    )
    reason_counts = (
        assignments.groupby(["policy", "reason_code"], dropna=False)
        .size()
        .reset_index(name="n_queries")
        .sort_values(["policy", "n_queries"], ascending=[True, False])
    )

    assignments_path = args.output_dir / "marker_mirror_stable_order_policy_assignments.csv"
    production_path = args.output_dir / "marker_mirror_stable_order_policy_production_assignments.csv"
    summary_path = args.output_dir / "marker_mirror_stable_order_policy_summary.csv"
    by_source_path = args.output_dir / "marker_mirror_stable_order_policy_by_source.csv"
    reasons_path = args.output_dir / "marker_mirror_stable_order_policy_reason_counts.csv"
    manifest_path = args.output_dir / "marker_mirror_stable_order_policy_manifest.json"

    logger.log(f"Writing assignments: {rel(assignments_path)}")
    assignments.to_csv(assignments_path, index=False)
    production.to_csv(production_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_source.to_csv(by_source_path, index=False)
    reason_counts.to_csv(reasons_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "features": rel(args.features),
            "thresholds": rel(args.thresholds),
        },
        "outputs": {
            "assignments": rel(assignments_path),
            "production_assignments": rel(production_path),
            "summary": rel(summary_path),
            "by_source": rel(by_source_path),
            "reason_counts": rel(reasons_path),
        },
        "target": args.target,
        "threshold_stats": threshold_stats,
        "notes": [
            "Labels are used only for diagnostics in the output tables.",
            "production_assignments strips truth/correctness columns and is the current CLI handoff format.",
            "The production-visible policy emits order only when MarkerMirror, BLASTN, and VSEARCH agree on top-1 order.",
            "The max-repeat threshold is the most conservative global precision threshold observed across Exp 111 repeats.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.log(f"Summary:\n{summary.to_string(index=False)}")
    logger.done(SCRIPT_NAME)


if __name__ == "__main__":
    main()
