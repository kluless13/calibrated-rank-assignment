#!/usr/bin/env python3
"""Build and validate a union MarkerMirror/same-marker rank policy.

This is the next step after the Exp 103 candidate-support audit.  It keeps
production candidates separate from evaluation labels:

    production table:
        query_id, candidate source, candidate rank, candidate taxon, score

    diagnostic table:
        optional labels used only to score rank/no-call policies

The current policies are deliberately simple. They test whether the union
candidate path can make conservative family/order or agreement-based calls
before we train another model.
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
    parser.add_argument(
        "--marker-mirror-evidence",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "production_handoff_fullref_all_queries_12s_to_16s"
        / "evidence_handoff"
        / "marker_mirror_candidate_generator_evidence_handoff.csv.gz",
    )
    parser.add_argument(
        "--same-marker-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_same_marker_kmer_candidates_top50.csv.gz",
    )
    parser.add_argument(
        "--query-table",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "zero_shot_queries.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "union_candidate_rank_policy",
    )
    parser.add_argument(
        "--source-table-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--targets", default="0.95,0.99")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def load_queries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.rename(
        columns={
            "processid": "query_id",
            "tree_label": "query_tree_label",
            "species_name": "query_species",
            "genus_name": "query_genus",
            "family_name": "query_family",
            "order_name": "query_order",
        }
    )[
        [
            "query_id",
            "source",
            "query_tree_label",
            "query_species",
            "query_genus",
            "query_family",
            "query_order",
        ]
    ].copy()


def production_union_candidates(mm: pd.DataFrame, same: pd.DataFrame) -> pd.DataFrame:
    mm_cols = [
        "query_id",
        "candidate_rank",
        "candidate_tree_label",
        "score",
        "candidate_species",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
    ]
    mm_prod = mm[mm_cols].copy()
    mm_prod["candidate_source"] = "marker_mirror_12s_to_16s"
    mm_prod["query_marker"] = "12S"
    mm_prod["candidate_marker"] = "16S"
    same_prod = same[mm_cols + ["candidate_source"]].copy()
    same_prod["query_marker"] = "12S"
    same_prod["candidate_marker"] = "12S"
    out = pd.concat([mm_prod, same_prod], ignore_index=True)
    out = out[
        [
            "query_id",
            "query_marker",
            "candidate_marker",
            "candidate_source",
            "candidate_rank",
            "candidate_tree_label",
            "candidate_species",
            "candidate_genus",
            "candidate_family",
            "candidate_order",
            "score",
        ]
    ].sort_values(["query_id", "candidate_source", "candidate_rank"])
    return out


def top1_frame(queries: pd.DataFrame, mm: pd.DataFrame, same: pd.DataFrame) -> pd.DataFrame:
    mm1 = mm[mm["candidate_rank"] == 1].copy().add_prefix("mm_").rename(columns={"mm_query_id": "query_id"})
    sm1 = same[same["candidate_rank"] == 1].copy().add_prefix("sm_").rename(columns={"sm_query_id": "query_id"})
    frame = queries.merge(mm1, on="query_id", how="left").merge(sm1, on="query_id", how="left")
    for source in ("mm", "sm"):
        for rank in RANKS:
            frame[f"{source}_{rank}_correct"] = (
                frame[f"{source}_candidate_{rank if rank != 'species' else 'tree_label'}"].astype(str)
                == frame[f"query_{rank if rank != 'species' else 'tree_label'}"].astype(str)
            )
    for rank in ("genus", "family", "order"):
        frame[f"top1_sources_agree_{rank}"] = (
            frame[f"mm_candidate_{rank}"].map(clean) == frame[f"sm_candidate_{rank}"].map(clean)
        ) & frame[f"sm_candidate_{rank}"].map(clean).astype(bool)
    return frame


def apply_static_policies(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        # Policy 1: deepest agreement between top-1 MarkerMirror and top-1 same-marker candidates.
        assigned = ("no_call", "", None)
        for rank in ("genus", "family", "order"):
            if bool(row.get(f"top1_sources_agree_{rank}", False)):
                assigned = (rank, clean(row.get(f"sm_candidate_{rank}", "")), bool(row.get(f"sm_{rank}_correct", False)))
                break
        rows.append(
            {
                "policy": "top1_source_agreement_deepest",
                "target_precision": math.nan,
                "query_id": row["query_id"],
                "assigned_rank": assigned[0],
                "assigned_taxon": assigned[1],
                "assigned_correct": assigned[2],
            }
        )
        # Policy 2: source agreement, but do not emit genus because current genus
        # precision is not high enough for the conservative pipeline.
        assigned = ("no_call", "", None)
        for rank in ("family", "order"):
            if bool(row.get(f"top1_sources_agree_{rank}", False)):
                assigned = (rank, clean(row.get(f"sm_candidate_{rank}", "")), bool(row.get(f"sm_{rank}_correct", False)))
                break
        rows.append(
            {
                "policy": "top1_source_agreement_family_order",
                "target_precision": math.nan,
                "query_id": row["query_id"],
                "assigned_rank": assigned[0],
                "assigned_taxon": assigned[1],
                "assigned_correct": assigned[2],
            }
        )
    return pd.DataFrame(rows)


def score_gate_assignments(frame: pd.DataFrame, rank: str, threshold: float, target: float, policy: str) -> pd.DataFrame:
    assigned = frame["sm_score"].astype(float) >= float(threshold)
    return pd.DataFrame(
        {
            "policy": policy,
            "target_precision": target,
            "query_id": frame["query_id"],
            "assigned_rank": np.where(assigned, rank, "no_call"),
            "assigned_taxon": np.where(assigned, frame[f"sm_candidate_{rank}"].map(clean), ""),
            "assigned_correct": np.where(assigned, frame[f"sm_{rank}_correct"].astype(bool), pd.NA),
            "threshold": threshold,
        }
    )


def assignment_metrics(assignments: pd.DataFrame, n_query: int, label: dict[str, Any]) -> dict[str, Any]:
    assigned = assignments[assignments["assigned_rank"] != "no_call"].copy()
    labelled = assigned.dropna(subset=["assigned_correct"])
    out = {
        **label,
        "n_query": int(n_query),
        "assigned_n": int(len(assigned)),
        "coverage_pct": pct(len(assigned), n_query),
        "assigned_precision_pct": 100.0 * float(labelled["assigned_correct"].astype(bool).mean()) if len(labelled) else math.nan,
        "false_species_call_rate_pct": 0.0,
    }
    for rank in RANKS:
        rank_rows = labelled[labelled["assigned_rank"] == rank]
        out[f"{rank}_assignment_n"] = int((assigned["assigned_rank"] == rank).sum())
        out[f"{rank}_precision_pct"] = (
            100.0 * float(rank_rows["assigned_correct"].astype(bool).mean()) if len(rank_rows) else math.nan
        )
    return out


def fit_threshold(frame: pd.DataFrame, rank: str, target: float) -> tuple[float | None, str]:
    values = sorted(set(float(x) for x in frame["sm_score"].dropna()))
    if not values:
        return None, "no_scores"
    candidates = np.unique(np.quantile(values, np.linspace(0.0, 0.99, 100)))
    rows = []
    for threshold in candidates:
        assignments = score_gate_assignments(frame, rank, float(threshold), target, f"same_marker_top1_{rank}_score_gate")
        metrics = assignment_metrics(assignments, len(frame), {})
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows)
    viable = table[table["assigned_precision_pct"] >= 100.0 * target].copy()
    if viable.empty:
        best = table.sort_values(["assigned_precision_pct", "coverage_pct"], ascending=[False, False]).iloc[0]
        return float(best["threshold"]), "target_not_met_best_available"
    best = viable.sort_values(["coverage_pct", "assigned_precision_pct"], ascending=[False, False]).iloc[0]
    return float(best["threshold"]), "target_met"


def repeated_validation(
    frame: pd.DataFrame,
    targets: list[float],
    repeats: int,
    calibration_fraction: float,
    seed: int,
    logger: Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    species = np.array(sorted(frame["query_tree_label"].dropna().astype(str).unique()))
    per_split: list[dict[str, Any]] = []
    assignment_rows: list[pd.DataFrame] = []
    for repeat in range(repeats):
        rng.shuffle(species)
        n_cal = max(1, int(round(len(species) * calibration_fraction)))
        cal_species = set(species[:n_cal])
        cal = frame[frame["query_tree_label"].isin(cal_species)].copy()
        eva = frame[~frame["query_tree_label"].isin(cal_species)].copy()
        if cal.empty or eva.empty:
            continue
        for rank in ("family", "order"):
            for target in targets:
                threshold, fit_status = fit_threshold(cal, rank, target)
                if threshold is None:
                    continue
                assignments = score_gate_assignments(
                    eva, rank, threshold, target, f"same_marker_top1_{rank}_score_gate"
                )
                assignments["repeat"] = repeat
                assignments["fit_status"] = fit_status
                assignment_rows.append(assignments)
                row = assignment_metrics(
                    assignments,
                    len(eva),
                    {
                        "validation_type": "species_split",
                        "repeat": repeat,
                        "rank": rank,
                        "target_precision": target,
                        "threshold": threshold,
                        "fit_status": fit_status,
                    },
                )
                per_split.append(row)
        if repeat == 0 or (repeat + 1) % 10 == 0:
            logger.log(f"Completed union rank-policy validation repeat={repeat + 1}/{repeats}")
    return pd.DataFrame(per_split), pd.concat(assignment_rows, ignore_index=True) if assignment_rows else pd.DataFrame()


def summarize_repeats(per_split: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if per_split.empty:
        return pd.DataFrame()
    for key, group in per_split.groupby(["validation_type", "rank", "target_precision"], dropna=False):
        validation_type, rank, target = key
        rows.append(
            {
                "validation_type": validation_type,
                "rank": rank,
                "target_precision": target,
                "repeats": int(len(group)),
                "mean_coverage_pct": float(group["coverage_pct"].mean()),
                "mean_assigned_precision_pct": float(group["assigned_precision_pct"].mean()),
                "precision_p05_pct": float(group["assigned_precision_pct"].quantile(0.05)),
                "precision_p95_pct": float(group["assigned_precision_pct"].quantile(0.95)),
                "target_met_rate_pct": 100.0 * float((group["assigned_precision_pct"] >= 100.0 * float(target)).mean()),
                "mean_threshold": float(group["threshold"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_table_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_union_rank_policy.log")
    targets = [float(x) for x in args.targets.split(",") if x.strip()]

    logger.log("Loading union rank-policy inputs")
    queries = load_queries(args.query_table)
    mm = pd.read_csv(args.marker_mirror_evidence)
    same = pd.read_csv(args.same_marker_candidates)
    prod = production_union_candidates(mm, same)
    top1 = top1_frame(queries, mm, same)
    logger.log(f"Loaded queries={len(queries)} union_candidate_rows={len(prod)}")

    static_assignments = apply_static_policies(top1)
    static_summary = pd.DataFrame(
        [
            assignment_metrics(group, len(top1), {"policy": policy, "target_precision": math.nan})
            for policy, group in static_assignments.groupby("policy")
        ]
    )
    per_split, split_assignments = repeated_validation(
        top1, targets, args.repeats, args.calibration_fraction, args.seed, logger
    )
    split_summary = summarize_repeats(per_split)

    prod_path = args.output_dir / "marker_mirror_union_production_candidates.csv.gz"
    top1_path = args.output_dir / "marker_mirror_union_top1_diagnostic_features.csv"
    static_assign_path = args.output_dir / "marker_mirror_union_static_policy_assignments.csv"
    static_summary_path = args.output_dir / "marker_mirror_union_static_policy_summary.csv"
    split_path = args.output_dir / "marker_mirror_union_score_gate_validation_per_split.csv"
    split_summary_path = args.output_dir / "marker_mirror_union_score_gate_validation_summary.csv"
    prod.to_csv(prod_path, index=False)
    top1.to_csv(top1_path, index=False)
    static_assignments.to_csv(static_assign_path, index=False)
    static_summary.to_csv(static_summary_path, index=False)
    per_split.to_csv(split_path, index=False)
    split_summary.to_csv(split_summary_path, index=False)
    if not split_assignments.empty:
        split_assignments.to_csv(args.output_dir / "marker_mirror_union_score_gate_validation_assignments.csv.gz", index=False)

    # Source-table snapshots.
    static_summary.to_csv(args.source_table_dir / "marker_mirror_union_static_policy_summary.csv", index=False)
    split_summary.to_csv(args.source_table_dir / "marker_mirror_union_score_gate_validation_summary.csv", index=False)
    per_split.to_csv(args.source_table_dir / "marker_mirror_union_score_gate_validation_per_split.csv", index=False)
    manifest = {
        "generated_by": "scripts/edna/build_marker_mirror_union_rank_policy.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "marker_mirror_evidence": str(args.marker_mirror_evidence),
        "same_marker_candidates": str(args.same_marker_candidates),
        "query_table": str(args.query_table),
        "n_query": int(len(top1)),
        "n_union_candidate_rows": int(len(prod)),
        "targets": targets,
        "repeats": args.repeats,
        "calibration_fraction": args.calibration_fraction,
        "claim_boundary": "Union rank/no-call diagnostic. Same-marker arm is k-mer evidence; score gates are independently species-split validated but not field-eDNA production thresholds.",
        "outputs": {
            "production_candidates": str(prod_path),
            "top1_diagnostic_features": str(top1_path),
            "static_policy_summary": str(static_summary_path),
            "score_gate_validation_summary": str(split_summary_path),
        },
    }
    manifest_path = args.output_dir / "marker_mirror_union_rank_policy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.source_table_dir / "marker_mirror_union_rank_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.log(f"Wrote {prod_path}")
    logger.log(f"Wrote {static_summary_path}")
    logger.log(f"Wrote {split_summary_path}")


if __name__ == "__main__":
    main()
