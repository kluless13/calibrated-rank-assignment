#!/usr/bin/env python3
"""Calibrate executable Paper 1 pipeline modes from seen-test rows.

This is a narrow calibration layer for `run_paper1_coi_pipeline.py` outputs. It
learns rank/no-call thresholds on the seen-test pipeline rows for each
retrieval/rerank mode, then evaluates those locked thresholds on Eval C and
unseen-genera rows. It is intended to keep p-distance reranking honest: reranked
candidate order should not inherit vector-order thresholds without a check.
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
TARGET_PRECISIONS = (0.80, 0.90, 0.95, 0.99)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


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
    return Path(input_dir).name if input_dir else ""


def load_run(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame] | None:
    manifest_path = run_dir / "pipeline_manifest.json"
    assignments_path = run_dir / "pipeline_rank_assignments.csv"
    if not manifest_path.exists() or not assignments_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_csv(assignments_path)
    manifest["run_name"] = run_dir.name
    manifest["manifest_path"] = str(manifest_path)
    manifest["assignments_path"] = str(assignments_path)
    frame["split"] = split_from_manifest(manifest)
    frame["pipeline_mode"] = run_mode(manifest)
    frame["run_name"] = run_dir.name
    for rank in RANKS:
        if rank == "species":
            truth = frame["true_tree_label"].map(lambda value: clean(value).replace(" ", "_"))
        else:
            truth = frame[f"true_{rank}"].map(clean)
        pred = frame[f"pred_{rank}"].map(lambda value: clean(value).replace(" ", "_"))
        frame[f"{rank}_correct"] = (truth != "") & (pred == truth)
    return manifest, frame


def learn_thresholds(calibration: pd.DataFrame, target_precision: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, feature in POLICY_FEATURES.items():
        work = calibration[np.isfinite(pd.to_numeric(calibration[feature], errors="coerce"))].copy()
        if work.empty:
            rows.append(
                {
                    "rank": rank,
                    "feature": feature,
                    "target_precision": target_precision,
                    "n_calibration": 0,
                    "n_calibration_accepted": 0,
                    "calibration_coverage": 0.0,
                    "calibration_precision": np.nan,
                    "threshold": np.nan,
                }
            )
            continue
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        work = work.sort_values(feature, ascending=False).reset_index(drop=True)
        correct = work[f"{rank}_correct"].astype(float).to_numpy()
        cumulative_precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        eligible = np.where(cumulative_precision >= target_precision)[0]
        if len(eligible) == 0:
            rows.append(
                {
                    "rank": rank,
                    "feature": feature,
                    "target_precision": target_precision,
                    "n_calibration": int(len(work)),
                    "n_calibration_accepted": 0,
                    "calibration_coverage": 0.0,
                    "calibration_precision": np.nan,
                    "threshold": np.nan,
                }
            )
            continue
        idx = int(eligible[-1])
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "target_precision": target_precision,
                "n_calibration": int(len(work)),
                "n_calibration_accepted": idx + 1,
                "calibration_coverage": float((idx + 1) / len(work)),
                "calibration_precision": float(cumulative_precision[idx]),
                "threshold": float(work[feature].iloc[idx]),
            }
        )
    return rows


def evaluate_policy(evaluation: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, Any]:
    assigned_rank: list[str] = []
    assigned_correct: list[bool] = []
    for _, row in evaluation.iterrows():
        chosen = "no_call"
        correct = False
        for rank in RANKS:
            threshold = thresholds.get(rank)
            if threshold is None or not math.isfinite(threshold):
                continue
            value = row.get(POLICY_FEATURES[rank])
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric) and numeric >= threshold:
                chosen = rank
                correct = bool(row.get(f"{rank}_correct", False))
                break
        assigned_rank.append(chosen)
        assigned_correct.append(correct)
    policy = pd.DataFrame({"assigned_rank": assigned_rank, "correct": assigned_correct})
    assigned = policy[policy["assigned_rank"] != "no_call"]
    species_calls = policy[policy["assigned_rank"] == "species"]
    false_species = int((~species_calls["correct"]).sum()) if len(species_calls) else 0
    out: dict[str, Any] = {
        "n_evaluation": int(len(policy)),
        "n_assigned": int(len(assigned)),
        "coverage": float(len(assigned) / len(policy)) if len(policy) else np.nan,
        "assigned_precision": float(assigned["correct"].mean()) if len(assigned) else np.nan,
        "assigned_correct": int(assigned["correct"].sum()) if len(assigned) else 0,
        "false_species_call_rate_all_queries": float(false_species / len(policy)) if len(policy) else np.nan,
    }
    for rank in list(RANKS) + ["no_call"]:
        out[f"assigned_{rank}_count"] = int((policy["assigned_rank"] == rank).sum())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/pipeline_runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/pipeline_calibration"),
    )
    parser.add_argument("--calibration-split", default="seen_test")
    parser.add_argument("--evaluation-splits", nargs="+", default=["eval_c", "unseen_genera"])
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    manifests: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for run_dir in sorted(args.pipeline_root.iterdir()):
        if not run_dir.is_dir():
            continue
        loaded = load_run(run_dir)
        if not loaded:
            continue
        manifest, frame = loaded
        manifests.append(manifest)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No executable pipeline runs found under {args.pipeline_root}")
    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    logger.log(f"Loaded {len(manifests)} pipeline runs and {len(all_rows)} query rows")

    threshold_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for mode, mode_df in all_rows.groupby("pipeline_mode", dropna=False):
        calibration = mode_df[mode_df["split"] == args.calibration_split].copy()
        if calibration.empty:
            continue
        for target in TARGET_PRECISIONS:
            learned = learn_thresholds(calibration, target)
            thresholds = {
                row["rank"]: float(row["threshold"])
                for row in learned
                if np.isfinite(pd.to_numeric(pd.Series([row["threshold"]]), errors="coerce").iloc[0])
            }
            for row in learned:
                row.update(
                    {
                        "pipeline_mode": mode,
                        "calibration_split": args.calibration_split,
                    }
                )
                threshold_rows.append(row)
            for split in args.evaluation_splits:
                evaluation = mode_df[mode_df["split"] == split].copy()
                if evaluation.empty:
                    continue
                summary = evaluate_policy(evaluation, thresholds)
                summary.update(
                    {
                        "pipeline_mode": mode,
                        "calibration_split": args.calibration_split,
                        "evaluation_split": split,
                        "target_precision": target,
                    }
                )
                policy_rows.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = args.output_dir / "pipeline_mode_thresholds.csv"
    policy_path = args.output_dir / "pipeline_mode_policy_summary.csv"
    logger.log(f"Writing {threshold_path}")
    write_csv(
        threshold_path,
        threshold_rows,
        [
            "pipeline_mode",
            "calibration_split",
            "rank",
            "feature",
            "target_precision",
            "n_calibration",
            "n_calibration_accepted",
            "calibration_coverage",
            "calibration_precision",
            "threshold",
        ],
    )
    logger.log(f"Writing {policy_path}")
    write_csv(
        policy_path,
        policy_rows,
        [
            "pipeline_mode",
            "calibration_split",
            "evaluation_split",
            "target_precision",
            "n_evaluation",
            "n_assigned",
            "coverage",
            "assigned_precision",
            "assigned_correct",
            "false_species_call_rate_all_queries",
            "assigned_species_count",
            "assigned_genus_count",
            "assigned_family_count",
            "assigned_order_count",
            "assigned_no_call_count",
        ],
    )
    manifest = {
        "generated_by": "scripts/edna/calibrate_paper1_pipeline_modes.py",
        "pipeline_root": str(args.pipeline_root),
        "output_dir": str(args.output_dir),
        "calibration_split": args.calibration_split,
        "evaluation_splits": args.evaluation_splits,
        "n_pipeline_runs": len(manifests),
        "threshold_rows": len(threshold_rows),
        "policy_rows": len(policy_rows),
        "claim_boundary": "Prospective calibration over executable pipeline rows. p-distance rows are more honest than inherited vector thresholds, but still need strict tree-pruned validation for final missing-reference claims.",
    }
    manifest_path = args.output_dir / "pipeline_mode_calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
