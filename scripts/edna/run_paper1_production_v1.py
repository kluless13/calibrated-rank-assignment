#!/usr/bin/env python3
"""Package locked Paper 1 COI pipeline runs into production-v1 outputs.

This script does not retrain models and does not recalculate retrieval. It
starts from an existing `run_paper1_coi_pipeline.py` run, applies the locked
mode-specific calibration thresholds, and writes final rank/no-call
assignments plus a manifest.

The intended default operating point is the conservative CNN seed1206
p-distance-reranked mode calibrated on seen-test rows at target precision 0.99.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")
POLICY_FEATURES = {
    "species": "confidence_relative_margin",
    "genus": "genus_top10_consensus",
    "family": "family_top10_consensus",
    "order": "order_top10_consensus",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_mode(manifest: dict[str, Any]) -> str:
    return "__".join(
        [
            clean(manifest.get("prediction_set")) or "unknown",
            clean(manifest.get("retrieval_mode")) or "unknown_retrieval",
            clean(manifest.get("rerank_mode")) or "none",
            clean(manifest.get("assignment_source")) or "vector",
        ]
    )


def split_from_manifest(manifest: dict[str, Any]) -> str:
    input_dir = clean(manifest.get("input_dir"))
    return Path(input_dir).name if input_dir else "unknown_split"


def load_thresholds(path: Path, pipeline_mode: str, target_precision: float) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    target = pd.to_numeric(frame["target_precision"], errors="coerce")
    sub = frame[
        (frame["pipeline_mode"].astype(str) == pipeline_mode)
        & (target == float(target_precision))
    ].copy()
    thresholds: dict[str, dict[str, Any]] = {}
    for _, row in sub.iterrows():
        rank = clean(row.get("rank"))
        threshold = pd.to_numeric(pd.Series([row.get("threshold")]), errors="coerce").iloc[0]
        if rank in RANKS and np.isfinite(threshold):
            thresholds[rank] = {
                "feature": clean(row.get("feature")) or POLICY_FEATURES[rank],
                "threshold": float(threshold),
                "calibration_precision": float(row.get("calibration_precision", np.nan)),
                "calibration_coverage": float(row.get("calibration_coverage", np.nan)),
            }
    missing = [rank for rank in RANKS if rank not in thresholds]
    if missing:
        raise RuntimeError(
            f"Missing production thresholds for mode={pipeline_mode} target={target_precision}: {missing}"
        )
    return thresholds


def truth_for_rank(frame: pd.DataFrame, rank: str) -> pd.Series:
    if rank == "species":
        return frame["true_tree_label"].map(lambda value: clean(value).replace(" ", "_"))
    return frame[f"true_{rank}"].map(clean)


def add_correctness_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for rank in RANKS:
        truth = truth_for_rank(out, rank)
        pred = out[f"pred_{rank}"].map(lambda value: clean(value).replace(" ", "_"))
        known = truth != ""
        out[f"production_{rank}_correct"] = np.where(known, pred == truth, np.nan)
    return out


def apply_policy(frame: pd.DataFrame, thresholds: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = add_correctness_columns(frame)
    assigned_ranks: list[str] = []
    assigned_labels: list[str] = []
    reasons: list[str] = []
    correct: list[bool] = []
    for _, row in out.iterrows():
        chosen_rank = "no_call"
        chosen_label = ""
        reason = "no_rank_threshold_met"
        is_correct: bool | float = np.nan
        for rank in RANKS:
            item = thresholds[rank]
            feature = item["feature"]
            threshold = item["threshold"]
            try:
                value = float(row.get(feature))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= threshold:
                chosen_rank = rank
                chosen_label = clean(row.get(f"pred_{rank}"))
                reason = f"{feature}>={threshold:g}"
                correctness = row.get(f"production_{rank}_correct", np.nan)
                is_correct = np.nan if pd.isna(correctness) else bool(correctness)
                break
        assigned_ranks.append(chosen_rank)
        assigned_labels.append(chosen_label)
        reasons.append(reason)
        correct.append(is_correct)
    out["production_assigned_rank"] = assigned_ranks
    out["production_assigned_label"] = assigned_labels
    out["production_assignment_reason"] = reasons
    out["production_assigned_correct"] = correct
    return out


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    assigned = frame[frame["production_assigned_rank"] != "no_call"].copy()
    species = frame[frame["production_assigned_rank"] == "species"].copy()
    assigned_known = assigned[assigned["production_assigned_correct"].notna()].copy()
    species_known = species[species["production_assigned_correct"].notna()].copy()
    false_species = (
        int((~species_known["production_assigned_correct"].astype(bool)).sum())
        if len(species_known)
        else 0
    )
    summary: dict[str, Any] = {
        "n_queries": int(len(frame)),
        "n_assigned": int(len(assigned)),
        "coverage": float(len(assigned) / len(frame)) if len(frame) else np.nan,
        "assigned_precision": (
            float(assigned_known["production_assigned_correct"].astype(bool).mean())
            if len(assigned_known)
            else np.nan
        ),
        "assigned_correct": (
            int(assigned_known["production_assigned_correct"].astype(bool).sum())
            if len(assigned_known)
            else np.nan
        ),
        "known_truth_assigned_count": int(len(assigned_known)),
        "false_species_call_rate_all_queries": (
            float(false_species / len(frame))
            if len(frame) and len(species_known)
            else np.nan
        ),
    }
    for rank in list(RANKS) + ["no_call"]:
        summary[f"assigned_{rank}_count"] = int((frame["production_assigned_rank"] == rank).sum())
    return summary


def process_run(
    run_dir: Path,
    output_root: Path,
    threshold_path: Path,
    target_precision: float,
    logger: ProgressLogger,
) -> dict[str, Any]:
    manifest_path = run_dir / "pipeline_manifest.json"
    assignments_path = run_dir / "pipeline_rank_assignments.csv"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing {manifest_path}")
    if not assignments_path.exists():
        raise RuntimeError(f"Missing {assignments_path}")
    manifest = json.loads(manifest_path.read_text())
    mode = run_mode(manifest)
    split = split_from_manifest(manifest)
    out_dir = output_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Processing {run_dir.name}: split={split} mode={mode}")

    thresholds = load_thresholds(threshold_path, mode, target_precision)
    frame = pd.read_csv(assignments_path)
    production = apply_policy(frame, thresholds)
    summary = summarize(production)
    summary.update(
        {
            "split": split,
            "pipeline_mode": mode,
            "source_run_dir": str(run_dir),
            "target_precision": float(target_precision),
        }
    )

    assignments_out = out_dir / "production_v1_assignments.csv"
    summary_out = out_dir / "production_v1_summary.csv"
    manifest_out = out_dir / "production_v1_manifest.json"
    logger.log(f"Writing {assignments_out}")
    production.to_csv(assignments_out, index=False)
    write_csv(summary_out, [summary], list(summary.keys()))
    final_manifest = {
        "generated_by": "scripts/edna/run_paper1_production_v1.py",
        "source_manifest": str(manifest_path),
        "source_assignments": str(assignments_path),
        "source_run_dir": str(run_dir),
        "split": split,
        "pipeline_mode": mode,
        "target_precision": float(target_precision),
        "thresholds": str(threshold_path),
        "rank_thresholds": thresholds,
        "outputs": {
            "assignments": str(assignments_out),
            "summary": str(summary_out),
        },
        "claim_boundary": (
            "Production v1 packages an existing saved-embedding COI pipeline run. "
            "It is not raw FASTA-to-model inference yet. The default p-distance "
            "mode uses train-reference p-distance reranking and locked "
            "seen-test-derived mode thresholds."
        ),
    }
    manifest_out.write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n")
    summary["manifest"] = str(manifest_out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/pipeline_calibration/pipeline_mode_thresholds.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/production_v1"),
    )
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    summaries = [
        process_run(
            run_dir=run_dir,
            output_root=args.output_root,
            threshold_path=args.thresholds,
            target_precision=args.target_precision,
            logger=logger,
        )
        for run_dir in args.input_run_dir
    ]
    aggregate_path = args.output_root / "production_v1_summary_all.csv"
    logger.log(f"Writing {aggregate_path}")
    if summaries:
        write_csv(aggregate_path, summaries, list(summaries[0].keys()))
    manifest_path = args.output_root / "production_v1_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_by": "scripts/edna/run_paper1_production_v1.py",
                "target_precision": float(args.target_precision),
                "thresholds": str(args.thresholds),
                "input_run_dirs": [str(path) for path in args.input_run_dir],
                "aggregate_summary": str(aggregate_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
