#!/usr/bin/env python3
"""Prototype hierarchical selective rank assignment from existing COI outputs.

This imports selective-classification/conformal-style thinking into the Paper 1
pipeline without training a new model. It calibrates per-rank score thresholds
on the seen-test split, applies the deepest supported rank to held-out splits,
and optionally guards assignments with reference-gap warnings.

The output is a source-table prototype, not a final production policy.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("seen_test", "eval_c", "unseen_genera")
RANKS = ("species", "genus", "family", "order")
TARGETS = (0.90, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "production_v1",
    )
    parser.add_argument(
        "--reason-code-table",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "production_reason_code_assignments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_assignments(production_root: Path, reason_code_table: Path) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = production_root / split / "production_v1_assignments.csv"
        frame = pd.read_csv(path)
        frame["split"] = split
        frames.append(frame)
    assignments = pd.concat(frames, ignore_index=True)
    gap = pd.read_csv(
        reason_code_table,
        usecols=[
            "split",
            "processid",
            "species_gap_warning",
            "genus_gap_warning",
            "family_gap_warning",
            "reason_codes",
        ],
    )
    gap = gap.drop_duplicates(["split", "processid"], keep="first")
    return assignments.merge(gap, on=["split", "processid"], how="left")


def rank_score(frame: pd.DataFrame, rank: str) -> pd.Series:
    if rank == "species":
        return frame["species_top10_consensus"].fillna(0.0).astype(float) + frame[
            "confidence_relative_margin"
        ].fillna(0.0).astype(float).clip(lower=0.0)
    return frame[f"{rank}_top10_consensus"].fillna(0.0).astype(float)


def rank_correct(frame: pd.DataFrame, rank: str) -> pd.Series:
    return frame[f"production_{rank}_correct"].fillna(False).astype(bool)


def find_threshold(cal: pd.DataFrame, rank: str, target: float) -> tuple[float, float, int]:
    scores = rank_score(cal, rank)
    correct = rank_correct(cal, rank)
    table = pd.DataFrame({"score": scores, "correct": correct})
    rows = []
    for threshold in sorted(table["score"].unique(), reverse=True):
        selected = table[table["score"] >= threshold]
        if selected.empty:
            continue
        precision = float(selected["correct"].mean())
        rows.append((float(threshold), precision, int(len(selected))))
    eligible = [row for row in rows if row[1] >= target]
    if not eligible:
        return float("inf"), 0.0, 0
    # Lowest threshold with target precision maximizes coverage under the target.
    return eligible[-1]


def gap_blocks(row: pd.Series, rank: str) -> bool:
    if rank == "species":
        return bool(row.get("species_gap_warning", False))
    if rank == "genus":
        return bool(row.get("genus_gap_warning", False))
    if rank == "family":
        return bool(row.get("family_gap_warning", False))
    return False


def assign_row(row: pd.Series, thresholds: dict[str, float], policy: str) -> tuple[str, str, str]:
    for rank in RANKS:
        if policy == "gap_guarded" and gap_blocks(row, rank):
            continue
        score = (
            (float(row.get("species_top10_consensus", 0.0) or 0.0)
             + max(float(row.get("confidence_relative_margin", 0.0) or 0.0), 0.0))
            if rank == "species"
            else float(row.get(f"{rank}_top10_consensus", 0.0) or 0.0)
        )
        if score >= thresholds[rank]:
            label = str(row.get(f"pred_{rank}", ""))
            return rank, label, f"{policy}:{rank}_score>={thresholds[rank]:.6g}"
    return "no_call", "", f"{policy}:no_rank_met"


def summarize(frame: pd.DataFrame, policy: str, target: float, split: str) -> dict[str, Any]:
    assigned = frame[frame["selective_assigned_rank"] != "no_call"].copy()
    correct = assigned["selective_assigned_correct"].fillna(False).astype(bool)
    false_species = frame[
        (frame["selective_assigned_rank"] == "species")
        & (~frame["selective_assigned_correct"].fillna(False).astype(bool))
    ]
    row: dict[str, Any] = {
        "policy": policy,
        "target_precision": target,
        "split": split,
        "n_queries": int(len(frame)),
        "n_assigned": int(len(assigned)),
        "coverage_pct": 100.0 * len(assigned) / max(len(frame), 1),
        "assigned_precision_pct": 100.0 * float(correct.mean()) if len(assigned) else None,
        "false_species_call_rate_all_queries_pct": 100.0 * len(false_species) / max(len(frame), 1),
        "assigned_species_count": int((frame["selective_assigned_rank"] == "species").sum()),
        "assigned_genus_count": int((frame["selective_assigned_rank"] == "genus").sum()),
        "assigned_family_count": int((frame["selective_assigned_rank"] == "family").sum()),
        "assigned_order_count": int((frame["selective_assigned_rank"] == "order").sum()),
        "assigned_no_call_count": int((frame["selective_assigned_rank"] == "no_call").sum()),
    }
    return row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or args.output_dir / "hierarchical_selective_rank_prototype.log")
    logger.log("Loading production and reason-code assignments")
    assignments = load_assignments(args.production_root, args.reason_code_table)
    logger.log(f"Loaded {len(assignments):,} assignment rows")

    cal = assignments[assignments["split"] == "seen_test"].copy()
    threshold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        thresholds = {}
        for rank in RANKS:
            threshold, cal_precision, cal_n = find_threshold(cal, rank, target)
            thresholds[rank] = threshold
            threshold_rows.append(
                {
                    "target_precision": target,
                    "rank": rank,
                    "threshold": threshold,
                    "calibration_precision_pct": 100.0 * cal_precision,
                    "calibration_selected_count": cal_n,
                }
            )
        logger.log(f"Target {target:.2f} thresholds: {thresholds}")
        for policy in ("consensus_only", "gap_guarded"):
            for split in SPLITS:
                frame = assignments[assignments["split"] == split].copy()
                assigned = frame.apply(lambda row: assign_row(row, thresholds, policy), axis=1, result_type="expand")
                assigned.columns = [
                    "selective_assigned_rank",
                    "selective_assigned_label",
                    "selective_assignment_reason",
                ]
                frame = pd.concat([frame.reset_index(drop=True), assigned.reset_index(drop=True)], axis=1)
                frame["selective_assigned_correct"] = False
                for rank in RANKS:
                    mask = frame["selective_assigned_rank"] == rank
                    frame.loc[mask, "selective_assigned_correct"] = frame.loc[
                        mask, f"production_{rank}_correct"
                    ].fillna(False).astype(bool)
                summary_rows.append(summarize(frame, policy, target, split))
                if split != "seen_test" and target == 0.99:
                    examples = frame[
                        [
                            "split",
                            "processid",
                            "true_tree_label",
                            "selective_assigned_rank",
                            "selective_assigned_label",
                            "selective_assigned_correct",
                            "selective_assignment_reason",
                            "reason_codes",
                        ]
                    ].head(50)
                    examples["policy"] = policy
                    examples["target_precision"] = target
                    example_rows.extend(examples.to_dict(orient="records"))

    thresholds_out = args.output_dir / "hierarchical_selective_rank_thresholds.csv"
    summary_out = args.output_dir / "hierarchical_selective_rank_summary.csv"
    examples_out = args.output_dir / "hierarchical_selective_rank_examples.csv"
    manifest_out = args.output_dir / "hierarchical_selective_rank_manifest.json"
    pd.DataFrame(threshold_rows).to_csv(thresholds_out, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    pd.DataFrame(example_rows).to_csv(examples_out, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__)),
        "production_root": rel(args.production_root),
        "reason_code_table": rel(args.reason_code_table),
        "outputs": {
            "thresholds": rel(thresholds_out),
            "summary": rel(summary_out),
            "examples": rel(examples_out),
        },
        "claim_boundary": (
            "Prototype only. This imports selective prediction ideas but uses "
            "existing production-v1 scores and seen-test calibration; it is not "
            "a full conformal set-valued classifier yet."
        ),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote {rel(summary_out)}")


if __name__ == "__main__":
    main()
