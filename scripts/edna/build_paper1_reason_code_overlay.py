#!/usr/bin/env python3
"""Build user-facing reason codes for Paper 1 production-v1 assignments.

This is an evidence-accounting overlay, not a new classifier. It joins the
current COI production-v1 rank/no-call outputs with existing v2 reference-gap
probabilities and emits concise reason codes explaining why the pipeline made a
species/genus/family/order/no-call decision.
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
GAP_RANKS = ("species", "genus", "family")
DECISION_RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "production_v1",
    )
    parser.add_argument(
        "--reference-gap-run",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "reference_gap_detector"
        / "coi_mlp_seed1301_v2_candidate_evidence_target095",
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


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).lower() in {"true", "1", "yes"}


def load_thresholds(path: Path) -> dict[str, float]:
    thresholds = pd.read_csv(path)
    return {clean(row["rank"]): float(row["threshold"]) for _, row in thresholds.iterrows()}


def load_gap_predictions(path: Path) -> pd.DataFrame:
    gap = pd.read_csv(path)
    keep = ["split", "processid", *[f"gap_p_{rank}" for rank in GAP_RANKS]]
    missing = [column for column in keep if column not in gap.columns]
    if missing:
        raise ValueError(f"Gap predictions missing columns: {missing}")
    return gap[keep].copy()


def call_warning(row: pd.Series, rank: str, thresholds: dict[str, float]) -> bool:
    probability = row.get(f"gap_p_{rank}")
    if pd.isna(probability):
        return False
    return float(probability) >= thresholds[rank]


def consensus_value(row: pd.Series, rank: str) -> float:
    column = f"{rank}_top10_consensus"
    value = row.get(column)
    if pd.isna(value):
        return 0.0
    return float(value)


def margin_value(row: pd.Series) -> float:
    value = row.get("confidence_relative_margin")
    if pd.isna(value):
        return 0.0
    return float(value)


def reason_codes(row: pd.Series, thresholds: dict[str, float]) -> tuple[str, str]:
    assigned_rank = clean(row.get("production_assigned_rank"))
    assignment_reason = clean(row.get("production_assignment_reason"))
    codes: list[str] = []

    warnings = {
        rank: call_warning(row, rank, thresholds)
        for rank in GAP_RANKS
    }

    if assigned_rank in {"", "no_call"}:
        codes.append("no_call_no_rank_met")
        if margin_value(row) < 0.02:
            codes.append("weak_species_margin")
        if consensus_value(row, "order") < 0.8:
            codes.append("low_order_consensus")
        for rank, flagged in warnings.items():
            if flagged:
                codes.append(f"possible_missing_{rank}_reference")
        return codes[0], ";".join(dict.fromkeys(codes))

    if assigned_rank == "species":
        codes.append("species_supported_by_margin")
        if warnings["species"]:
            codes.append("possible_missing_species_reference")
        return codes[0], ";".join(dict.fromkeys(codes))

    if assigned_rank in DECISION_RANKS:
        if assignment_reason:
            codes.append(f"{assigned_rank}_supported_by_policy_threshold")
        else:
            codes.append(f"{assigned_rank}_supported")

    if assigned_rank in {"genus", "family", "order"}:
        codes.append("species_not_supported_at_operating_point")
        if warnings["species"]:
            codes.append("possible_missing_species_reference")

    if assigned_rank in {"family", "order"}:
        if warnings["genus"]:
            codes.append("possible_missing_genus_reference")
        if consensus_value(row, "genus") < 1.0:
            codes.append("genus_ambiguous_in_top10")

    if assigned_rank == "order":
        if warnings["family"]:
            codes.append("possible_missing_family_reference")
        if consensus_value(row, "family") < 1.0:
            codes.append("family_ambiguous_in_top10")

    if assigned_rank in {"genus", "family", "order"} and not any(warnings.values()):
        codes.append("broader_rank_chosen_by_calibration")

    return codes[0], ";".join(dict.fromkeys(codes))


def split_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, assigned_rank, primary_reason), group in frame.groupby(
        ["split", "production_assigned_rank", "primary_reason_code"],
        dropna=False,
    ):
        assigned = group["production_assigned_rank"].map(clean).ne("no_call")
        precision = (
            float(group.loc[assigned, "production_assigned_correct_bool"].mean() * 100.0)
            if assigned.any()
            else float("nan")
        )
        rows.append(
            {
                "split": split,
                "assigned_rank": assigned_rank,
                "primary_reason_code": primary_reason,
                "n_queries": int(len(group)),
                "assigned_count": int(assigned.sum()),
                "assigned_precision_pct": precision,
                "species_gap_warning_pct": float(group["species_gap_warning"].mean() * 100.0),
                "genus_gap_warning_pct": float(group["genus_gap_warning"].mean() * 100.0),
                "family_gap_warning_pct": float(group["family_gap_warning"].mean() * 100.0),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    script_name = Path(__file__).name
    logger.start(script_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.reference_gap_run / "reference_gap_thresholds.csv")
    gap = load_gap_predictions(args.reference_gap_run / "reference_gap_predictions.csv")
    gap = gap[gap["split"].isin(SPLITS)].copy()
    logger.log(f"Loaded gap predictions rows={len(gap):,} from {rel(args.reference_gap_run)}")

    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        path = args.production_root / split / "production_v1_assignments.csv"
        logger.log(f"Loading production assignments {rel(path)}")
        prod = pd.read_csv(path)
        prod["split"] = split
        merged = prod.merge(
            gap[gap["split"] == split],
            on=["split", "processid"],
            how="left",
        )
        for rank in GAP_RANKS:
            merged[f"gap_p_{rank}"] = merged[f"gap_p_{rank}"].fillna(0.0).astype(float)
            merged[f"{rank}_gap_warning"] = merged.apply(
                lambda row, r=rank: call_warning(row, r, thresholds),
                axis=1,
            )
        codes = merged.apply(lambda row: reason_codes(row, thresholds), axis=1)
        merged["primary_reason_code"] = [item[0] for item in codes]
        merged["reason_codes"] = [item[1] for item in codes]
        merged["production_assigned_correct_bool"] = merged["production_assigned_correct"].map(truthy)
        frames.append(merged)

    all_rows = pd.concat(frames, ignore_index=True)
    output_cols = [
        "split",
        "processid",
        "true_tree_label",
        "true_genus",
        "true_family",
        "true_order",
        "pred_species",
        "pred_genus",
        "pred_family",
        "pred_order",
        "production_assigned_rank",
        "production_assigned_label",
        "production_assigned_correct",
        "production_assignment_reason",
        "gap_p_species",
        "gap_p_genus",
        "gap_p_family",
        "species_gap_warning",
        "genus_gap_warning",
        "family_gap_warning",
        "primary_reason_code",
        "reason_codes",
    ]
    assignments_path = args.output_dir / "production_reason_code_assignments.csv"
    logger.log(f"Writing reason-coded assignments {rel(assignments_path)}")
    all_rows[output_cols].to_csv(assignments_path, index=False)

    summary_path = args.output_dir / "production_reason_code_summary.csv"
    logger.log(f"Writing reason-code summary {rel(summary_path)}")
    pd.DataFrame(split_summary(all_rows)).to_csv(summary_path, index=False)

    examples = (
        all_rows[
            all_rows["reason_codes"].str.contains("possible_missing|ambiguous|no_call", regex=True)
        ]
        .sort_values(["split", "production_assigned_correct_bool", "gap_p_species"], ascending=[True, True, False])
        .head(200)
    )
    examples_path = args.output_dir / "production_reason_code_examples.csv"
    logger.log(f"Writing reason-code examples {rel(examples_path)}")
    examples[output_cols].to_csv(examples_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "production_root": rel(args.production_root),
        "reference_gap_run": rel(args.reference_gap_run),
        "thresholds": thresholds,
        "outputs": {
            "assignments": rel(assignments_path),
            "summary": rel(summary_path),
            "examples": rel(examples_path),
        },
        "claim_boundary": (
            "Reason codes are an explanatory overlay over existing production-v1 "
            "assignments and v2 reference-gap probabilities. They are not a new "
            "classifier and should not be treated as independent accuracy metrics."
        ),
    }
    manifest_path = args.output_dir / "production_reason_code_manifest.json"
    logger.log(f"Writing manifest {rel(manifest_path)}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.done(script_name)


if __name__ == "__main__":
    main()
