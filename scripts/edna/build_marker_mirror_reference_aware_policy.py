#!/usr/bin/env python3
"""Evaluate reference-aware gates over MarkerMirror rank/no-call assignments.

This is a policy-repair diagnostic for the production-style MarkerMirror handoff.
It starts from an existing assignment table, then asks whether additional
reference/candidate-list gates improve assigned precision.  Gates marked
``production_safe`` use only features available at inference time.  Gates marked
``diagnostic_oracle`` use labels/reference truth and are upper-bound diagnostics
only.
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

from scripts.edna.train_marker_mirror_bridge import Logger

RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-evidence-table", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text


def source_from_query_id(query_id: Any) -> str:
    text = str(query_id)
    return text.split(":", 1)[0] if ":" in text else "unknown"


def query_level(evidence: pd.DataFrame) -> pd.DataFrame:
    target_labels = set(evidence["candidate_tree_label"].dropna().astype(str).map(norm))
    rows = []
    for query_id, group in evidence.groupby("query_id", sort=False):
        group = group.sort_values("candidate_rank")
        first = group.iloc[0]
        query_label = norm(first.get("query_tree_label", first.get("input_query_tree_label", "")))
        row: dict[str, Any] = {
            "query_id": query_id,
            "query_source": source_from_query_id(query_id),
            "query_tree_label": query_label,
            "query_has_target_marker_reference": int(query_label in target_labels) if query_label else 0,
            "top1_score": float(first.get("score", np.nan)),
            "score_margin_1_2": float(first.get("score_margin_1_2", np.nan)),
            "score_margin_1_5": float(first.get("score_margin_1_5", np.nan)),
            "score_margin_1_10": float(first.get("score_margin_1_10", np.nan)),
            "top1_same_marker_sequence_available": int(first.get("same_marker_sequence_available", 0) == 1),
            "top1_same_marker_identity": float(first.get("same_marker_best_identity", np.nan)),
            "top1_candidate_has_both_marker_references": int(
                first.get("candidate_has_both_marker_references", 0) == 1
            ),
            "top5_family_mode_fraction": float(first.get("top5_family_mode_fraction", np.nan)),
            "top10_family_mode_fraction": float(first.get("top10_family_mode_fraction", np.nan)),
            "top50_family_mode_fraction": float(first.get("top50_family_mode_fraction", np.nan)),
            "top5_order_mode_fraction": float(first.get("top5_order_mode_fraction", np.nan)),
            "top10_order_mode_fraction": float(first.get("top10_order_mode_fraction", np.nan)),
            "top50_order_mode_fraction": float(first.get("top50_order_mode_fraction", np.nan)),
        }
        for k in (1, 5, 10, 50):
            subset = group[group["candidate_rank"] <= k]
            for rank in RANKS:
                match_col = f"match_{rank}"
                row[f"top{k}_{rank}_hit"] = (
                    bool(subset[match_col].astype(bool).any()) if match_col in subset.columns else False
                )
        rows.append(row)
    return pd.DataFrame(rows)


def selected_candidate_level(evidence: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    assigned = assignments[assignments["assigned_rank"].fillna("no_call") != "no_call"].copy()
    if assigned.empty:
        return pd.DataFrame(columns=["query_id"])
    selected = assigned[["query_id", "selected_candidate_tree_label"]].copy()
    selected["selected_candidate_tree_label"] = selected["selected_candidate_tree_label"].astype(str)
    evidence = evidence.copy()
    evidence["candidate_tree_label"] = evidence["candidate_tree_label"].astype(str)
    merged = selected.merge(
        evidence,
        left_on=["query_id", "selected_candidate_tree_label"],
        right_on=["query_id", "candidate_tree_label"],
        how="left",
    )
    rows = []
    for query_id, group in merged.groupby("query_id", sort=False):
        row = group.iloc[0]
        rows.append(
            {
                "query_id": query_id,
                "selected_same_marker_sequence_available": int(row.get("same_marker_sequence_available", 0) == 1),
                "selected_same_marker_identity": float(row.get("same_marker_best_identity", np.nan)),
                "selected_candidate_has_both_marker_references": int(
                    row.get("candidate_has_both_marker_references", 0) == 1
                ),
                "selected_tree_distance_to_top1_candidate": float(row.get("tree_distance_to_top1_candidate", np.nan)),
                "selected_candidate_top50_family_support_fraction": float(
                    row.get("candidate_top50_family_support_fraction", np.nan)
                ),
                "selected_candidate_top50_order_support_fraction": float(
                    row.get("candidate_top50_order_support_fraction", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy(frame: pd.DataFrame, name: str, gate_type: str, mask: pd.Series) -> dict[str, Any]:
    kept = frame[mask].copy()
    assigned = kept[kept["assigned_rank"].fillna("no_call") != "no_call"]
    labelled = assigned[assigned["assigned_correct"].notna()]
    n_query = int(len(frame))
    row = {
        "policy": name,
        "gate_type": gate_type,
        "n_query": n_query,
        "assigned_n": int(len(assigned)),
        "coverage_pct": 100.0 * len(assigned) / max(n_query, 1),
        "assigned_precision_pct": (
            100.0 * float(labelled["assigned_correct"].astype(bool).mean()) if len(labelled) else math.nan
        ),
        "false_species_call_rate_pct": (
            100.0
            * int(((labelled["assigned_rank"] == "species") & (~labelled["assigned_correct"].astype(bool))).sum())
            / max(n_query, 1)
            if len(labelled)
            else math.nan
        ),
    }
    for rank in ("species", "genus", "family", "order", "no_call"):
        row[f"{rank}_assignment_n"] = int((kept["assigned_rank"].fillna("no_call") == rank).sum())
    return row


def build_threshold_masks(frame: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    masks: list[tuple[str, str, pd.Series]] = []
    assigned = frame[frame["assigned_rank"].fillna("no_call") != "no_call"]
    candidate_features = [
        ("selected_probability", "selected_candidate_probability", ">="),
        ("top1_score", "top1_score", ">="),
        ("score_margin_1_2", "score_margin_1_2", ">="),
        ("score_margin_1_5", "score_margin_1_5", ">="),
        ("top10_family_mode_fraction", "top10_family_mode_fraction", ">="),
        ("top50_family_mode_fraction", "top50_family_mode_fraction", ">="),
        ("top10_order_mode_fraction", "top10_order_mode_fraction", ">="),
        ("top50_order_mode_fraction", "top50_order_mode_fraction", ">="),
        ("selected_same_marker_identity", "selected_same_marker_identity", ">="),
        ("selected_tree_distance_to_top1_candidate", "selected_tree_distance_to_top1_candidate", "<="),
    ]
    for label, col, op in candidate_features:
        if col not in frame.columns or assigned[col].notna().sum() < 10:
            continue
        values = assigned[col].dropna().astype(float)
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            threshold = float(values.quantile(q))
            if op == ">=":
                mask = (frame["assigned_rank"].fillna("no_call") == "no_call") | (frame[col].astype(float) >= threshold)
            else:
                mask = (frame["assigned_rank"].fillna("no_call") == "no_call") | (frame[col].astype(float) <= threshold)
            masks.append((f"{label}_q{q:.2f}_{op}_{threshold:.6g}", "production_safe_threshold", mask.fillna(False)))
    binary_gates = [
        ("top1_same_marker_available", "top1_same_marker_sequence_available"),
        ("top1_candidate_has_both_refs", "top1_candidate_has_both_marker_references"),
        ("selected_same_marker_available", "selected_same_marker_sequence_available"),
        ("selected_candidate_has_both_refs", "selected_candidate_has_both_marker_references"),
    ]
    for label, col in binary_gates:
        if col in frame.columns:
            mask = (frame["assigned_rank"].fillna("no_call") == "no_call") | (frame[col].fillna(0).astype(int) == 1)
            masks.append((label, "production_safe_binary", mask))
    oracle_gates = [
        ("oracle_query_species_in_target_ref", "query_has_target_marker_reference"),
        ("oracle_top50_species_hit", "top50_species_hit"),
        ("oracle_top50_genus_hit", "top50_genus_hit"),
        ("oracle_top50_family_hit", "top50_family_hit"),
        ("oracle_top50_order_hit", "top50_order_hit"),
    ]
    for label, col in oracle_gates:
        if col in frame.columns:
            mask = (frame["assigned_rank"].fillna("no_call") == "no_call") | frame[col].fillna(False).astype(bool)
            masks.append((label, "diagnostic_oracle", mask))
    if {"top50_species_hit", "top50_genus_hit", "top50_family_hit"}.issubset(frame.columns):
        mask = (frame["assigned_rank"].fillna("no_call") == "no_call") | (
            frame["top50_species_hit"].astype(bool)
            | frame["top50_genus_hit"].astype(bool)
            | frame["top50_family_hit"].astype(bool)
        )
        masks.append(("oracle_top50_family_or_better", "diagnostic_oracle", mask))
    return masks


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_reference_aware_policy.log")
    logger.log(f"Arguments: {vars(args)}")
    evidence = pd.read_csv(args.handoff_evidence_table)
    assignments = pd.read_csv(args.assignments)
    evidence["query_id"] = evidence["query_id"].astype(str)
    assignments["query_id"] = assignments["query_id"].astype(str)
    logger.log(f"Loaded evidence_rows={len(evidence)} assignments={len(assignments)}")
    q = query_level(evidence)
    selected = selected_candidate_level(evidence, assignments)
    frame = q.merge(assignments, on="query_id", how="left", validate="one_to_one")
    if not selected.empty:
        frame = frame.merge(selected, on="query_id", how="left", validate="one_to_one")
    frame["assigned_rank"] = frame["assigned_rank"].fillna("no_call")
    rows = [summarize_policy(frame, "baseline", "baseline", pd.Series(True, index=frame.index))]
    for name, gate_type, mask in build_threshold_masks(frame):
        rows.append(summarize_policy(frame, name, gate_type, mask))
    summary = pd.DataFrame(rows).sort_values(
        ["assigned_precision_pct", "coverage_pct"], ascending=[False, False], na_position="last"
    )
    frame.to_csv(args.output_dir / "marker_mirror_reference_aware_policy_per_query.csv", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_reference_aware_policy_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_marker_mirror_reference_aware_policy.py",
        "handoff_evidence_table": str(args.handoff_evidence_table),
        "assignments": str(args.assignments),
        "rows": {
            "evidence": int(len(evidence)),
            "queries": int(len(frame)),
            "policies": int(len(summary)),
        },
        "claim_boundary": "Production-safe gates are candidate policy diagnostics; diagnostic_oracle gates use labels/reference truth and are not production policies.",
    }
    (args.output_dir / "marker_mirror_reference_aware_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.log(f"Wrote reference-aware policy diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
