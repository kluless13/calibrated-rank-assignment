#!/usr/bin/env python3
"""Independently validate reference-aware MarkerMirror abstention gates.

The previous reference-aware sweep chose gates and evaluated them on the same
labelled full-query handoff.  This script uses repeated species-level splits:
choose one production-safe gate on calibration species, then evaluate that
locked gate on held-out species.  Source-holdout checks are also reported when
enough assignments are available.
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

from scripts.edna.build_marker_mirror_reference_aware_policy import (
    query_level,
    selected_candidate_level,
    source_from_query_id,
)
from scripts.edna.train_marker_mirror_bridge import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-evidence-table", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="0.95,0.99")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def assemble_query_frame(evidence_path: Path, assignments_path: Path, logger: Logger) -> pd.DataFrame:
    evidence = pd.read_csv(evidence_path)
    assignments = pd.read_csv(assignments_path)
    evidence["query_id"] = evidence["query_id"].astype(str)
    assignments["query_id"] = assignments["query_id"].astype(str)
    logger.log(f"Loaded evidence_rows={len(evidence)} assignments={len(assignments)}")
    q = query_level(evidence)
    selected = selected_candidate_level(evidence, assignments)
    frame = q.merge(assignments, on="query_id", how="left", validate="one_to_one")
    if not selected.empty:
        frame = frame.merge(selected, on="query_id", how="left", validate="one_to_one")
    if "query_tree_label" not in frame.columns:
        label_cols = [col for col in ("query_tree_label_x", "query_tree_label_y") if col in frame.columns]
        if label_cols:
            frame["query_tree_label"] = frame[label_cols[0]]
            for col in label_cols[1:]:
                frame["query_tree_label"] = frame["query_tree_label"].where(
                    frame["query_tree_label"].notna() & (frame["query_tree_label"].astype(str) != ""),
                    frame[col],
                )
        else:
            frame["query_tree_label"] = ""
    frame["assigned_rank"] = frame["assigned_rank"].fillna("no_call")
    frame["assigned"] = frame["assigned_rank"] != "no_call"
    frame["assigned_correct_bool"] = frame["assigned_correct"].fillna(False).astype(bool)
    frame["query_source"] = frame["query_id"].map(source_from_query_id)
    frame["query_tree_label"] = frame["query_tree_label"].fillna("").astype(str)
    return frame


def candidate_gate_specs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    assigned = frame[frame["assigned"]]
    specs: list[dict[str, Any]] = [{"policy": "baseline", "feature": "", "op": "all", "threshold": math.nan}]
    numeric = [
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
    quantiles = np.linspace(0.05, 0.95, 19)
    for label, col, op in numeric:
        if col not in frame.columns:
            continue
        values = assigned[col].dropna().astype(float)
        if len(values) < 10:
            continue
        for q in quantiles:
            threshold = float(values.quantile(q))
            specs.append(
                {
                    "policy": f"{label}_{op}_{threshold:.6g}",
                    "feature": col,
                    "op": op,
                    "threshold": threshold,
                }
            )
    binary = [
        ("top1_same_marker_available", "top1_same_marker_sequence_available"),
        ("top1_candidate_has_both_refs", "top1_candidate_has_both_marker_references"),
        ("selected_same_marker_available", "selected_same_marker_sequence_available"),
        ("selected_candidate_has_both_refs", "selected_candidate_has_both_marker_references"),
    ]
    for label, col in binary:
        if col in frame.columns:
            specs.append({"policy": label, "feature": col, "op": "==1", "threshold": 1.0})
    return specs


def gate_mask(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    assigned = frame["assigned_rank"].fillna("no_call") != "no_call"
    if spec["op"] == "all":
        return pd.Series(True, index=frame.index)
    col = spec["feature"]
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    if spec["op"] == ">=":
        keep_assigned = frame[col].astype(float) >= float(spec["threshold"])
    elif spec["op"] == "<=":
        keep_assigned = frame[col].astype(float) <= float(spec["threshold"])
    elif spec["op"] == "==1":
        keep_assigned = frame[col].fillna(0).astype(int) == 1
    else:
        raise ValueError(f"Unknown op: {spec['op']}")
    return (~assigned) | keep_assigned.fillna(False)


def metrics(frame: pd.DataFrame, mask: pd.Series, prefix: str = "") -> dict[str, Any]:
    kept = frame[mask].copy()
    assigned = kept[kept["assigned"]]
    labelled = assigned[assigned["assigned_correct"].notna()]
    out: dict[str, Any] = {
        f"{prefix}n_query": int(len(frame)),
        f"{prefix}assigned_n": int(len(assigned)),
        f"{prefix}coverage_pct": 100.0 * len(assigned) / max(len(frame), 1),
        f"{prefix}assigned_precision_pct": (
            100.0 * float(labelled["assigned_correct_bool"].mean()) if len(labelled) else math.nan
        ),
        f"{prefix}false_species_call_rate_pct": (
            100.0
            * int(((labelled["assigned_rank"] == "species") & (~labelled["assigned_correct_bool"])).sum())
            / max(len(frame), 1)
            if len(labelled)
            else math.nan
        ),
    }
    for rank in ("species", "genus", "family", "order"):
        rank_rows = labelled[labelled["assigned_rank"] == rank]
        out[f"{prefix}{rank}_assignment_n"] = int((assigned["assigned_rank"] == rank).sum())
        out[f"{prefix}{rank}_precision_pct"] = (
            100.0 * float(rank_rows["assigned_correct_bool"].mean()) if len(rank_rows) else math.nan
        )
    return out


def choose_policy(cal: pd.DataFrame, specs: list[dict[str, Any]], target: float) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    for spec in specs:
        mask = gate_mask(cal, spec)
        row = {**spec, **metrics(cal, mask)}
        rows.append(row)
    table = pd.DataFrame(rows)
    viable = table[table["assigned_precision_pct"] >= 100.0 * target].copy()
    if viable.empty:
        chosen = table.sort_values(["assigned_precision_pct", "coverage_pct"], ascending=[False, False]).iloc[0].to_dict()
        chosen["fit_status"] = "target_not_met_best_available"
    else:
        chosen = viable.sort_values(["coverage_pct", "assigned_precision_pct"], ascending=[False, False]).iloc[0].to_dict()
        chosen["fit_status"] = "target_met"
    spec = {
        "policy": chosen["policy"],
        "feature": chosen["feature"],
        "op": chosen["op"],
        "threshold": chosen["threshold"],
    }
    return spec, chosen


def split_species(frame: pd.DataFrame, rng: np.random.Generator, calibration_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.array(sorted(frame["query_tree_label"].replace("", np.nan).dropna().unique()))
    rng.shuffle(labels)
    n_cal = max(1, int(round(len(labels) * calibration_fraction)))
    cal_labels = set(labels[:n_cal])
    cal = frame[frame["query_tree_label"].isin(cal_labels)].copy()
    eva = frame[~frame["query_tree_label"].isin(cal_labels)].copy()
    return cal, eva


def repeated_species_validation(
    frame: pd.DataFrame,
    specs: list[dict[str, Any]],
    targets: list[float],
    repeats: int,
    calibration_fraction: float,
    seed: int,
    logger: Logger,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(repeats):
        cal, eva = split_species(frame, rng, calibration_fraction)
        if cal.empty or eva.empty:
            continue
        for target in targets:
            spec, fit = choose_policy(cal, specs, target)
            mask = gate_mask(eva, spec)
            row = {
                "strategy": "species_split",
                "repeat": repeat,
                "target_precision": target,
                "policy": spec["policy"],
                "feature": spec["feature"],
                "op": spec["op"],
                "threshold": spec["threshold"],
                "fit_status": fit["fit_status"],
                **metrics(cal, gate_mask(cal, spec), prefix="cal_"),
                **metrics(eva, mask, prefix="eval_"),
            }
            rows.append(row)
        if repeat == 0 or (repeat + 1) % 10 == 0:
            logger.log(f"Completed species split repeat={repeat + 1}/{repeats}")
    return pd.DataFrame(rows)


def source_holdout_validation(frame: pd.DataFrame, specs: list[dict[str, Any]], targets: list[float]) -> pd.DataFrame:
    rows = []
    for source in sorted(frame["query_source"].dropna().unique()):
        eva = frame[frame["query_source"] == source].copy()
        cal = frame[frame["query_source"] != source].copy()
        if cal.empty or eva.empty or cal["assigned"].sum() < 20 or eva["assigned"].sum() < 5:
            continue
        for target in targets:
            spec, fit = choose_policy(cal, specs, target)
            rows.append(
                {
                    "strategy": "source_holdout",
                    "repeat": source,
                    "target_precision": target,
                    "policy": spec["policy"],
                    "feature": spec["feature"],
                    "op": spec["op"],
                    "threshold": spec["threshold"],
                    "fit_status": fit["fit_status"],
                    **metrics(cal, gate_mask(cal, spec), prefix="cal_"),
                    **metrics(eva, gate_mask(eva, spec), prefix="eval_"),
                }
            )
    return pd.DataFrame(rows)


def aggregate(per_split: pd.DataFrame) -> pd.DataFrame:
    if per_split.empty:
        return per_split
    metric_cols = [
        "eval_assigned_n",
        "eval_coverage_pct",
        "eval_assigned_precision_pct",
        "eval_false_species_call_rate_pct",
        "cal_coverage_pct",
        "cal_assigned_precision_pct",
    ]
    rows = []
    for keys, group in per_split.groupby(["strategy", "target_precision"], dropna=False):
        row = {"strategy": keys[0], "target_precision": keys[1], "n_splits": int(len(group))}
        for col in metric_cols:
            values = group[col].dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{col}_median"] = float(values.median()) if len(values) else math.nan
            row[f"{col}_p05"] = float(values.quantile(0.05)) if len(values) else math.nan
            row[f"{col}_p95"] = float(values.quantile(0.95)) if len(values) else math.nan
        row["target_met_rate_pct"] = 100.0 * float((group["fit_status"] == "target_met").mean())
        row["eval_target_met_rate_pct"] = 100.0 * float(group["eval_assigned_precision_pct"].ge(100.0 * keys[1]).mean())
        row["most_common_policy"] = group["policy"].value_counts().index[0] if len(group) else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["strategy", "target_precision"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_reference_aware_policy_validation.log")
    logger.log(f"Arguments: {vars(args)}")
    targets = [float(item) for item in args.targets.split(",") if item.strip()]
    frame = assemble_query_frame(args.handoff_evidence_table, args.assignments, logger)
    specs = candidate_gate_specs(frame)
    logger.log(f"Candidate production-safe gates={len(specs)}")
    per_split = repeated_species_validation(
        frame,
        specs,
        targets,
        args.repeats,
        args.calibration_fraction,
        args.seed,
        logger,
    )
    source = source_holdout_validation(frame, specs, targets)
    if not source.empty:
        per_split = pd.concat([per_split, source], ignore_index=True)
    summary = aggregate(per_split)
    per_split.to_csv(args.output_dir / "marker_mirror_reference_aware_policy_validation_per_split.csv", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_reference_aware_policy_validation_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_marker_mirror_reference_aware_policy_validation.py",
        "handoff_evidence_table": str(args.handoff_evidence_table),
        "assignments": str(args.assignments),
        "targets": targets,
        "repeats": args.repeats,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
        "candidate_gate_count": len(specs),
        "rows": {
            "queries": int(len(frame)),
            "assigned_queries": int(frame["assigned"].sum()),
            "per_split": int(len(per_split)),
            "summary": int(len(summary)),
        },
        "claim_boundary": "Repeated split diagnostic. Gates are selected on calibration cohorts and evaluated on held-out cohorts, but final production thresholds still need external validation.",
    }
    (args.output_dir / "marker_mirror_reference_aware_policy_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.log(f"Wrote independent validation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
