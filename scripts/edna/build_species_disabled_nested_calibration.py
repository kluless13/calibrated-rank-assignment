#!/usr/bin/env python3
"""Stress-test species-disabled eDNA rank backoff with nested calibration.

The full sequence+tree Eco-Phylo posterior is already split into calibration
and evaluation site groups. The deployed species-disabled policy learns
genus/family/order thresholds on all calibration groups, then evaluates on held
out groups. This script asks whether those thresholds are stable by repeatedly
splitting the calibration groups again:

calibration groups -> threshold-fit groups + calibration-holdout groups

Thresholds are learned only on the threshold-fit groups, then applied to both
the calibration-holdout groups and the original held-out evaluation groups.
This does not retrain the posterior model; it is a threshold-stability test.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "eco_phylo_posterior"
    / "candidate_level_sequence_tree_evidence_full"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR
RANKS = ("genus", "family", "order")
TARGET_ACCURACIES = (80.0, 90.0, 95.0)
GROUP_COLS = ("sample_id", "query_processid")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def stable_fraction(value: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def learn_threshold(rows: pd.DataFrame, rank: str, target_accuracy: float) -> dict[str, Any]:
    correct_col = f"{rank}_candidate_correct"
    work = rows[
        (rows["rank"] == rank)
        & np.isfinite(pd.to_numeric(rows["posterior_probability"], errors="coerce"))
    ].copy()
    if work.empty:
        return {
            "rank": rank,
            "target_accuracy_pct": target_accuracy,
            "status": "no_fit_rows",
            "threshold": np.nan,
            "n_fit_rows": 0,
            "n_fit_accepted": 0,
            "fit_assignment_rate_pct": 0.0,
            "fit_accuracy_pct": np.nan,
        }

    work[correct_col] = work[correct_col].astype(bool)
    work = work.sort_values("posterior_probability", ascending=False)
    cumulative_correct = work[correct_col].to_numpy(dtype=float).cumsum()
    counts = np.arange(1, len(work) + 1, dtype=float)
    cumulative_accuracy = 100.0 * cumulative_correct / counts
    eligible = np.where(cumulative_accuracy >= target_accuracy)[0]
    if len(eligible) == 0:
        return {
            "rank": rank,
            "target_accuracy_pct": target_accuracy,
            "status": "no_threshold_at_target",
            "threshold": np.nan,
            "n_fit_rows": int(len(work)),
            "n_fit_accepted": 0,
            "fit_assignment_rate_pct": 0.0,
            "fit_accuracy_pct": np.nan,
        }

    idx = int(eligible[-1])
    return {
        "rank": rank,
        "target_accuracy_pct": target_accuracy,
        "status": "available",
        "threshold": float(work.iloc[idx]["posterior_probability"]),
        "n_fit_rows": int(len(work)),
        "n_fit_accepted": int(idx + 1),
        "fit_assignment_rate_pct": float(100.0 * (idx + 1) / len(work)),
        "fit_accuracy_pct": float(cumulative_accuracy[idx]),
    }


def selected_for_rank(rows: pd.DataFrame, rank: str, threshold: float) -> pd.DataFrame:
    work = rows[
        (rows["rank"] == rank)
        & (pd.to_numeric(rows["posterior_probability"], errors="coerce") >= threshold)
    ].copy()
    if work.empty:
        return work
    work["assigned_rank"] = rank
    work["assigned_value"] = work[f"candidate_{rank}"]
    work["assigned_correct"] = work[f"{rank}_candidate_correct"].astype(bool)
    return work


def build_backoff_assignments(rows: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rank in RANKS:
        threshold = thresholds.get(rank)
        if threshold is not None and np.isfinite(threshold):
            frames.append(selected_for_rank(rows, rank, threshold))
    if not frames:
        return pd.DataFrame()
    assigned = pd.concat(frames, ignore_index=True)
    if assigned.empty:
        return assigned
    priority = {"genus": 0, "family": 1, "order": 2}
    assigned["rank_priority"] = assigned["assigned_rank"].map(priority)
    assigned = assigned.sort_values(
        ["sample_id", "query_processid", "rank_priority", "posterior_probability"],
        ascending=[True, True, True, False],
    )
    return assigned.groupby(list(GROUP_COLS), sort=False, as_index=False).head(1).copy()


def summarize_assignments(
    assignments: pd.DataFrame,
    n_query: int,
    repeat: int,
    target_accuracy: float,
    evaluation_split: str,
    threshold_status: str,
) -> dict[str, Any]:
    counts = assignments["assigned_rank"].value_counts().to_dict() if not assignments.empty else {}
    n_assigned = int(len(assignments))
    accuracy = (
        float(100.0 * assignments["assigned_correct"].sum() / n_assigned)
        if n_assigned
        else np.nan
    )
    return {
        "repeat": repeat,
        "target_accuracy_pct": target_accuracy,
        "evaluation_split": evaluation_split,
        "threshold_status": threshold_status,
        "n_query": int(n_query),
        "n_assigned": n_assigned,
        "assignment_rate_pct": float(100.0 * n_assigned / n_query) if n_query else 0.0,
        "assigned_accuracy_pct": accuracy,
        "genus_assignments": int(counts.get("genus", 0)),
        "family_assignments": int(counts.get("family", 0)),
        "order_assignments": int(counts.get("order", 0)),
        "no_call_rate_pct": float(100.0 * (n_query - n_assigned) / n_query) if n_query else 0.0,
    }


def aggregate_summary(per_repeat: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "assignment_rate_pct",
        "assigned_accuracy_pct",
        "n_assigned",
        "genus_assignments",
        "family_assignments",
        "order_assignments",
    ]
    for (target, split), group in per_repeat.groupby(["target_accuracy_pct", "evaluation_split"], sort=True):
        n_query_values = pd.to_numeric(group["n_query"], errors="coerce").dropna()
        row: dict[str, Any] = {
            "target_accuracy_pct": float(target),
            "evaluation_split": split,
            "n_repeats": int(len(group)),
            "n_query_mean": float(n_query_values.mean()) if len(n_query_values) else 0.0,
            "n_query_min": float(n_query_values.min()) if len(n_query_values) else 0.0,
            "n_query_max": float(n_query_values.max()) if len(n_query_values) else 0.0,
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_min"] = np.nan
                row[f"{metric}_p05"] = np.nan
                row[f"{metric}_p95"] = np.nan
                row[f"{metric}_max"] = np.nan
            else:
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_p05"] = float(values.quantile(0.05))
                row[f"{metric}_p95"] = float(values.quantile(0.95))
                row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return rows


def split_calibration_groups(selected: pd.DataFrame, repeat: int, fit_fraction: float) -> tuple[set[str], set[str]]:
    groups = (
        selected[selected["calibration_split"] == "calibration"]["sample_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    fit_groups: set[str] = set()
    holdout_groups: set[str] = set()
    for group in groups:
        if stable_fraction(group, f"species-disabled-nested-{repeat}") < fit_fraction:
            fit_groups.add(group)
        else:
            holdout_groups.add(group)
    return fit_groups, holdout_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = args.input_dir / "eco_phylo_candidate_posterior_selected_predictions.csv.gz"
    logger.log(f"Reading selected posterior predictions from {rel(selected_path)}")
    selected = pd.read_csv(selected_path, low_memory=False)
    selected = selected[selected["rank"].isin(RANKS)].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    logger.log(f"Loaded {len(selected):,} selected genus/family/order rows")

    per_repeat_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []

    evaluation = selected[selected["calibration_split"] == "evaluation"].copy()
    n_evaluation_query = len(evaluation[list(GROUP_COLS)].drop_duplicates())
    logger.log(f"Evaluation groups rows={len(evaluation):,} n_query={n_evaluation_query:,}")

    for repeat in range(args.repeats):
        fit_groups, holdout_groups = split_calibration_groups(selected, repeat, args.fit_fraction)
        fit = selected[
            (selected["calibration_split"] == "calibration") & (selected["sample_id"].isin(fit_groups))
        ].copy()
        holdout = selected[
            (selected["calibration_split"] == "calibration") & (selected["sample_id"].isin(holdout_groups))
        ].copy()
        n_holdout_query = len(holdout[list(GROUP_COLS)].drop_duplicates())
        logger.log(
            "Repeat "
            f"{repeat + 1}/{args.repeats}: fit_groups={len(fit_groups):,} "
            f"holdout_groups={len(holdout_groups):,} fit_rows={len(fit):,} "
            f"holdout_queries={n_holdout_query:,}"
        )

        for target in TARGET_ACCURACIES:
            learned = [learn_threshold(fit, rank, target) for rank in RANKS]
            thresholds = {
                row["rank"]: float(row["threshold"])
                for row in learned
                if row["status"] == "available" and np.isfinite(row["threshold"])
            }
            threshold_status = "available" if thresholds else "no_threshold"
            for row in learned:
                row.update(
                    {
                        "repeat": repeat,
                        "fit_fraction": args.fit_fraction,
                        "n_fit_groups": len(fit_groups),
                        "n_holdout_groups": len(holdout_groups),
                    }
                )
                threshold_rows.append(row)

            for split_name, split_rows, n_query in (
                ("calibration_holdout", holdout, n_holdout_query),
                ("evaluation", evaluation, n_evaluation_query),
            ):
                assignments = build_backoff_assignments(split_rows, thresholds)
                if not assignments.empty:
                    assignments = assignments.copy()
                    assignments["repeat"] = repeat
                    assignments["target_accuracy_pct"] = target
                    assignments["evaluation_split"] = split_name
                    assignment_frames.append(assignments)
                per_repeat_rows.append(
                    summarize_assignments(
                        assignments,
                        n_query,
                        repeat,
                        target,
                        split_name,
                        threshold_status,
                    )
                )

    per_repeat = pd.DataFrame(per_repeat_rows)
    aggregate_rows = aggregate_summary(per_repeat)
    thresholds = pd.DataFrame(threshold_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()

    per_repeat_path = args.output_dir / "eco_phylo_species_disabled_nested_calibration_per_repeat.csv"
    aggregate_path = args.output_dir / "eco_phylo_species_disabled_nested_calibration_summary.csv"
    thresholds_path = args.output_dir / "eco_phylo_species_disabled_nested_calibration_thresholds.csv"
    assignments_path = args.output_dir / "eco_phylo_species_disabled_nested_calibration_assignments.csv.gz"

    logger.log(f"Writing per-repeat nested calibration summary to {rel(per_repeat_path)}")
    per_repeat.to_csv(per_repeat_path, index=False)
    logger.log(f"Writing aggregate nested calibration summary to {rel(aggregate_path)}")
    write_csv(
        aggregate_path,
        aggregate_rows,
        [
            "target_accuracy_pct",
            "evaluation_split",
            "n_repeats",
            "n_query_mean",
            "n_query_min",
            "n_query_max",
            "assignment_rate_pct_mean",
            "assignment_rate_pct_std",
            "assignment_rate_pct_min",
            "assignment_rate_pct_p05",
            "assignment_rate_pct_p95",
            "assignment_rate_pct_max",
            "assigned_accuracy_pct_mean",
            "assigned_accuracy_pct_std",
            "assigned_accuracy_pct_min",
            "assigned_accuracy_pct_p05",
            "assigned_accuracy_pct_p95",
            "assigned_accuracy_pct_max",
            "n_assigned_mean",
            "n_assigned_std",
            "n_assigned_min",
            "n_assigned_p05",
            "n_assigned_p95",
            "n_assigned_max",
            "genus_assignments_mean",
            "genus_assignments_std",
            "genus_assignments_min",
            "genus_assignments_p05",
            "genus_assignments_p95",
            "genus_assignments_max",
            "family_assignments_mean",
            "family_assignments_std",
            "family_assignments_min",
            "family_assignments_p05",
            "family_assignments_p95",
            "family_assignments_max",
            "order_assignments_mean",
            "order_assignments_std",
            "order_assignments_min",
            "order_assignments_p05",
            "order_assignments_p95",
            "order_assignments_max",
        ],
    )
    logger.log(f"Writing learned nested thresholds to {rel(thresholds_path)}")
    thresholds.to_csv(thresholds_path, index=False)
    logger.log(f"Writing nested assignments to {rel(assignments_path)}")
    with gzip.open(assignments_path, "wt", newline="") as handle:
        assignments.to_csv(handle, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": rel(Path(__file__)),
        "input_dir": rel(args.input_dir),
        "output_dir": rel(args.output_dir),
        "selected_predictions": rel(selected_path),
        "summary_csv": rel(aggregate_path),
        "per_repeat_csv": rel(per_repeat_path),
        "thresholds_csv": rel(thresholds_path),
        "assignments_csv": rel(assignments_path),
        "rank_order": RANKS,
        "target_accuracies_pct": TARGET_ACCURACIES,
        "repeats": args.repeats,
        "fit_fraction": args.fit_fraction,
        "notes": [
            "Species rank is intentionally disabled.",
            "Posterior model weights are not retrained; this is a threshold-stability test.",
            "Thresholds are learned on calibration subgroups and evaluated on calibration-holdout plus original held-out evaluation groups.",
        ],
    }
    manifest_path = args.output_dir / "eco_phylo_species_disabled_nested_calibration_manifest.json"
    logger.log(f"Writing manifest to {rel(manifest_path)}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
