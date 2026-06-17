#!/usr/bin/env python3
"""Diagnose MarkerMirror calibration transfer from controlled splits to handoff.

The integrated rank/no-call calibrator works well on controlled held-out
MarkerMirror splits but transfers weakly to the full multisource handoff.  This
script turns that into auditable source tables by comparing retrieval support,
feature distributions, and assignment precision across cohorts and query
strata.
"""

from __future__ import annotations

import argparse
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

from scripts.edna.apply_marker_mirror_integrated_rank_calibrator import safe_feature_columns
from scripts.edna.train_marker_mirror_bridge import Logger

RANKS = ("species", "genus", "family", "order")
TOP_KS = (1, 5, 10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-evidence-table", type=Path, required=True)
    parser.add_argument("--handoff-evidence-table", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controlled-model", default="marker_mirror_projection")
    parser.add_argument("--controlled-direction", default="12S_to_16S")
    parser.add_argument("--handoff-direction", default="12S->16S")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def source_from_query_id(query_id: Any) -> str:
    text = str(query_id)
    if ":" in text:
        return text.split(":", 1)[0]
    return "unknown"


def normalized_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text


def load_filtered(args: argparse.Namespace, logger: Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    controlled = pd.read_csv(args.controlled_evidence_table)
    handoff = pd.read_csv(args.handoff_evidence_table)
    assignments = pd.read_csv(args.assignments)
    controlled = controlled[
        (controlled["model"].astype(str) == args.controlled_model)
        & (controlled["direction"].astype(str) == args.controlled_direction)
    ].copy()
    handoff = handoff[handoff["direction"].astype(str) == args.handoff_direction].copy()
    if controlled.empty:
        raise RuntimeError("No controlled rows after model/direction filtering.")
    if handoff.empty:
        raise RuntimeError("No handoff rows after direction filtering.")
    if assignments.empty:
        raise RuntimeError("Assignments table is empty.")
    for frame in (controlled, handoff, assignments):
        frame["query_id"] = frame["query_id"].astype(str)
    controlled["query_source"] = "controlled"
    handoff["query_source"] = handoff["query_id"].map(source_from_query_id)
    assignments["query_source"] = assignments["query_id"].map(source_from_query_id)
    logger.log(
        f"Loaded controlled_rows={len(controlled)} handoff_rows={len(handoff)} assignments={len(assignments)}"
    )
    return controlled, handoff, assignments


def query_level(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    target_labels = set(frame["candidate_tree_label"].dropna().astype(str).map(normalized_label))
    rows: list[dict[str, Any]] = []
    for query_id, group in frame.groupby("query_id", sort=False):
        group = group.sort_values("candidate_rank")
        first = group.iloc[0]
        query_label = normalized_label(first.get("query_tree_label", first.get("input_query_tree_label", "")))
        row: dict[str, Any] = {
            "cohort": cohort,
            "split": normalized_label(first.get("split", cohort)),
            "model": normalized_label(first.get("model", "")),
            "direction": normalized_label(first.get("direction", "")),
            "query_id": query_id,
            "query_source": normalized_label(first.get("query_source", source_from_query_id(query_id))),
            "query_tree_label": query_label,
            "query_has_target_marker_reference": int(query_label in target_labels) if query_label else 0,
            "top1_score": float(first.get("score", np.nan)),
            "top1_same_marker_sequence_available": int(first.get("same_marker_sequence_available", 0) == 1),
            "top1_same_marker_exact_match": int(first.get("same_marker_exact_match", 0) == 1),
            "top1_same_marker_identity": float(first.get("same_marker_best_identity", np.nan)),
            "top1_candidate_has_both_marker_references": int(
                first.get("candidate_has_both_marker_references", 0) == 1
            ),
            "query_sequence_length": float(first.get("query_sequence_length", np.nan)),
            "query_exact_sequence_species_count": float(first.get("query_exact_sequence_species_count", np.nan)),
            "query_exact_sequence_ambiguous": int(first.get("query_exact_sequence_ambiguous", 0) == 1),
        }
        for k in TOP_KS:
            subset = group[group["candidate_rank"] <= k]
            for rank in RANKS:
                match_col = f"match_{rank}"
                row[f"top{k}_{rank}_hit"] = (
                    bool(subset[match_col].astype(bool).any()) if match_col in subset.columns else False
                )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bool_rate(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return math.nan
    return 100.0 * float(frame[col].astype(bool).mean())


def cohort_summary(query_frames: list[pd.DataFrame]) -> pd.DataFrame:
    all_query = pd.concat(query_frames, ignore_index=True)
    rows = []
    for keys, group in all_query.groupby(["cohort", "model", "direction", "split"], dropna=False):
        cohort, model, direction, split = keys
        row: dict[str, Any] = {
            "cohort": cohort,
            "model": model,
            "direction": direction,
            "split": split,
            "n_query": int(group["query_id"].nunique()),
            "query_has_target_marker_reference_pct": summarize_bool_rate(group, "query_has_target_marker_reference"),
            "top1_same_marker_sequence_available_pct": summarize_bool_rate(
                group, "top1_same_marker_sequence_available"
            ),
            "top1_same_marker_exact_match_pct": summarize_bool_rate(group, "top1_same_marker_exact_match"),
            "top1_candidate_has_both_marker_references_pct": summarize_bool_rate(
                group, "top1_candidate_has_both_marker_references"
            ),
            "top1_score_mean": float(group["top1_score"].mean()),
            "top1_same_marker_identity_mean": float(group["top1_same_marker_identity"].mean()),
            "query_sequence_length_mean": float(group["query_sequence_length"].mean()),
            "query_exact_sequence_ambiguous_pct": summarize_bool_rate(group, "query_exact_sequence_ambiguous"),
        }
        for k in TOP_KS:
            for rank in RANKS:
                row[f"top{k}_{rank}_hit_pct"] = summarize_bool_rate(group, f"top{k}_{rank}_hit")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cohort", "model", "direction", "split"]).reset_index(drop=True)


def handoff_strata_summary(handoff_query: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    assign_cols = [
        "query_id",
        "assigned_rank",
        "assigned_correct",
        "selected_candidate_probability",
        "selected_candidate_score",
    ]
    merged = handoff_query.merge(assignments[assign_cols], on="query_id", how="left", validate="one_to_one")
    merged["assigned"] = merged["assigned_rank"].fillna("no_call") != "no_call"
    merged["assigned_correct_bool"] = merged["assigned_correct"].fillna(False).astype(bool)
    merged["species_call"] = merged["assigned_rank"].fillna("") == "species"
    merged["false_species_call"] = merged["species_call"] & (~merged["assigned_correct_bool"])
    merged["top50_depth"] = np.select(
        [
            merged["top50_species_hit"],
            merged["top50_genus_hit"],
            merged["top50_family_hit"],
            merged["top50_order_hit"],
        ],
        ["species", "genus", "family", "order"],
        default="none",
    )
    strata = [
        ("all", pd.Series("all", index=merged.index)),
        ("query_source", merged["query_source"].astype(str)),
        ("target_marker_reference", merged["query_has_target_marker_reference"].map({1: "present", 0: "absent"})),
        ("top50_depth", merged["top50_depth"]),
        (
            "top1_same_marker_available",
            merged["top1_same_marker_sequence_available"].map({1: "available", 0: "missing"}),
        ),
    ]
    rows = []
    for stratum, values in strata:
        temp = merged.assign(stratum_value=values)
        for value, group in temp.groupby("stratum_value", dropna=False):
            assigned = group[group["assigned"]]
            labelled = assigned[assigned["assigned_correct"].notna()]
            row = {
                "stratum": stratum,
                "value": value,
                "n_query": int(group["query_id"].nunique()),
                "assigned_n": int(len(assigned)),
                "coverage_pct": 100.0 * len(assigned) / max(len(group), 1),
                "assigned_precision_pct": (
                    100.0 * float(labelled["assigned_correct_bool"].mean()) if len(labelled) else math.nan
                ),
                "false_species_call_rate_pct": 100.0 * int(group["false_species_call"].sum()) / max(len(group), 1),
                "top50_species_hit_pct": summarize_bool_rate(group, "top50_species_hit"),
                "top50_genus_hit_pct": summarize_bool_rate(group, "top50_genus_hit"),
                "top50_family_hit_pct": summarize_bool_rate(group, "top50_family_hit"),
                "top50_order_hit_pct": summarize_bool_rate(group, "top50_order_hit"),
                "top1_same_marker_sequence_available_pct": summarize_bool_rate(
                    group, "top1_same_marker_sequence_available"
                ),
                "query_has_target_marker_reference_pct": summarize_bool_rate(
                    group, "query_has_target_marker_reference"
                ),
            }
            for rank in ("species", "genus", "family", "order", "no_call"):
                row[f"{rank}_assignment_n"] = int((group["assigned_rank"].fillna("no_call") == rank).sum())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["stratum", "value"]).reset_index(drop=True)


def feature_drift(
    controlled: pd.DataFrame,
    handoff: pd.DataFrame,
    features: list[str],
    logger: Logger,
) -> pd.DataFrame:
    frames = []
    controlled = controlled.copy()
    handoff = handoff.copy()
    controlled["cohort"] = "controlled_" + controlled["split"].astype(str)
    handoff["cohort"] = "production_handoff"
    for level, selector in (("candidate_rows", None), ("top1_rows", lambda df: df["candidate_rank"] == 1)):
        parts = []
        for frame in (controlled, handoff):
            selected = frame if selector is None else frame[selector(frame)].copy()
            parts.append(selected)
        combined = pd.concat(parts, ignore_index=True)
        for feature in features:
            if feature not in combined.columns:
                continue
            baseline = combined[combined["cohort"] == "controlled_val"][feature].dropna().astype(float)
            baseline_mean = float(baseline.mean()) if len(baseline) else math.nan
            baseline_std = float(baseline.std(ddof=0)) if len(baseline) else math.nan
            for cohort, group in combined.groupby("cohort", dropna=False):
                values = group[feature].dropna().astype(float)
                mean = float(values.mean()) if len(values) else math.nan
                std = float(values.std(ddof=0)) if len(values) else math.nan
                standardized_delta = (
                    (mean - baseline_mean) / baseline_std
                    if pd.notna(mean) and pd.notna(baseline_mean) and baseline_std not in (0.0, math.nan)
                    else math.nan
                )
                frames.append(
                    {
                        "level": level,
                        "feature": feature,
                        "cohort": cohort,
                        "n_non_null": int(len(values)),
                        "missing_pct": 100.0 * (1.0 - len(values) / max(len(group), 1)),
                        "mean": mean,
                        "std": std,
                        "median": float(values.median()) if len(values) else math.nan,
                        "p05": float(values.quantile(0.05)) if len(values) else math.nan,
                        "p95": float(values.quantile(0.95)) if len(values) else math.nan,
                        "standardized_delta_vs_controlled_val": standardized_delta,
                    }
                )
    out = pd.DataFrame(frames)
    if out.empty:
        logger.log("WARNING: no feature drift rows produced")
        return out
    return out.sort_values(["level", "feature", "cohort"]).reset_index(drop=True)


def top_drift_features(drift: pd.DataFrame) -> pd.DataFrame:
    if drift.empty:
        return drift
    prod = drift[drift["cohort"] == "production_handoff"].copy()
    prod["abs_standardized_delta_vs_controlled_val"] = prod["standardized_delta_vs_controlled_val"].abs()
    return (
        prod.sort_values(["level", "abs_standardized_delta_vs_controlled_val"], ascending=[True, False])
        .groupby("level", as_index=False, group_keys=False)
        .head(30)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_calibration_transfer_diagnostics.log")
    logger.log(f"Arguments: {vars(args)}")
    controlled, handoff, assignments = load_filtered(args, logger)
    features = safe_feature_columns(controlled, handoff)
    logger.log(f"Shared production-safe feature count={len(features)}")

    controlled_query = query_level(controlled, "controlled")
    handoff_query = query_level(handoff, "production_handoff")
    cohorts = cohort_summary([controlled_query, handoff_query])
    strata = handoff_strata_summary(handoff_query, assignments)
    drift = feature_drift(controlled, handoff, features, logger)
    top_drift = top_drift_features(drift)

    cohorts.to_csv(args.output_dir / "marker_mirror_calibration_transfer_cohort_summary.csv", index=False)
    strata.to_csv(args.output_dir / "marker_mirror_calibration_transfer_handoff_strata.csv", index=False)
    drift.to_csv(args.output_dir / "marker_mirror_calibration_transfer_feature_drift.csv", index=False)
    top_drift.to_csv(args.output_dir / "marker_mirror_calibration_transfer_top_feature_drift.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_marker_mirror_calibration_transfer_diagnostics.py",
        "controlled_evidence_table": str(args.controlled_evidence_table),
        "handoff_evidence_table": str(args.handoff_evidence_table),
        "assignments": str(args.assignments),
        "controlled_model": args.controlled_model,
        "controlled_direction": args.controlled_direction,
        "handoff_direction": args.handoff_direction,
        "feature_count": len(features),
        "rows": {
            "controlled_candidate_rows": int(len(controlled)),
            "handoff_candidate_rows": int(len(handoff)),
            "controlled_query_rows": int(len(controlled_query)),
            "handoff_query_rows": int(len(handoff_query)),
            "cohort_summary": int(len(cohorts)),
            "handoff_strata": int(len(strata)),
            "feature_drift": int(len(drift)),
        },
        "claim_boundary": "Diagnostics use known labels only to explain transfer behavior; they are not production-time inputs.",
    }
    (args.output_dir / "marker_mirror_calibration_transfer_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.log(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
