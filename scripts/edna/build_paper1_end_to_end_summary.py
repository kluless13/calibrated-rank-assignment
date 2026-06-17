#!/usr/bin/env python3
"""Build compact end-to-end Paper 1 pipeline summary tables."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
RANK_CAL = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "rank_adaptive_calibration"
PIPELINE_CAL = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "pipeline_calibration"
PRODUCTION_V1 = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "production_v1"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_value(df: pd.DataFrame, metric: str, where: dict[str, object] | None = None, maximize: bool = True) -> dict[str, object]:
    work = df.copy()
    if where:
        for key, value in where.items():
            work = work[work[key].astype(str) == str(value)]
    if work.empty or metric not in work:
        return {}
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return {}
    return work.sort_values(metric, ascending=not maximize).iloc[0].to_dict()


def build_stage_rows(source: Path, rank_cal: Path) -> list[dict[str, object]]:
    coi = read_csv(source / "pipeline_coi_method_benchmark.csv")
    placement = read_csv(source / "pipeline_placement_benchmark.csv")
    vector = read_csv(source / "pipeline_vector_index_benchmark.csv")
    edna = read_csv(source / "pipeline_edna_method_benchmark.csv")
    edna_best_by_rank = read_csv(source / "edna_evidence_best_by_rank.csv")
    edna_no_call = read_csv(source / "edna_rank_no_call_operating_points.csv")
    edna_independent_no_call = read_csv(
        ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "global_edna_independent_rank_calibration"
        / "global_edna_independent_rank_calibration_summary.csv"
    )
    strict = read_csv(source / "strict_missing_reference_summary.csv")
    bootstrap = read_csv(rank_cal / "missing_reference_aware_policy_bootstrap.csv")
    pipeline_cal = read_csv(PIPELINE_CAL / "pipeline_mode_policy_summary.csv")
    production_v1 = read_csv(PRODUCTION_V1 / "production_v1_summary_all.csv")
    raw_production_v1 = read_csv(source / "raw_sequence_production_v1_summary.csv")
    rows: list[dict[str, object]] = []

    for split in ["eval_c", "unseen_genera"]:
        best_family = best_value(coi, "family_top10_pct", {"split": split})
        if best_family:
            rows.append(
                {
                    "pipeline_stage": "candidate_retrieval",
                    "split_or_dataset": split,
                    "selected_method": best_family.get("method", ""),
                    "metric": "family_top10_pct",
                    "value": best_family.get("family_top10_pct", ""),
                    "unit": "pct",
                    "source_table": "pipeline_coi_method_benchmark.csv",
                    "claim_boundary": "retrieval accuracy, not calibrated final assignment",
                }
            )

    for split in ["eval_c", "unseen_genera"]:
        if vector.empty or "split" not in vector or "retrieval_mode" not in vector:
            speed_pool = pd.DataFrame()
        else:
            speed_pool = vector[vector["split"].astype(str) == split].copy()
            controlled = speed_pool[speed_pool["retrieval_mode"].astype(str).str.contains("controlled", na=False)]
            if not controlled.empty:
                speed_pool = controlled
            else:
                speed_pool = speed_pool[~speed_pool["retrieval_mode"].astype(str).str.contains("_x", na=False)]
        best_speed = best_value(speed_pool, "milliseconds_per_query", maximize=False)
        if best_speed:
            rows.append(
                {
                    "pipeline_stage": "fast_vector_retrieval",
                    "split_or_dataset": split,
                    "selected_method": best_speed.get("method", ""),
                    "metric": f"{best_speed.get('retrieval_mode', '')}_ms_per_query",
                    "value": best_speed.get("milliseconds_per_query", ""),
                    "unit": "ms/query",
                    "source_table": "pipeline_vector_index_benchmark.csv",
                    "claim_boundary": "local repeat-based vector timing when available; target-hardware replication still needed",
                }
            )

    for split in ["eval_c", "unseen_genera"]:
        apples = placement[
            (placement["split"].astype(str) == split)
            & (placement["method"].astype(str).str.startswith("apples_like"))
        ]
        if not apples.empty:
            row = apples.iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "distance_placement",
                    "split_or_dataset": split,
                    "selected_method": row.get("method", ""),
                    "metric": "nearest_reference_match_pct",
                    "value": row.get("nearest_reference_match_pct", ""),
                    "unit": "pct",
                    "source_table": "pipeline_placement_benchmark.csv",
                    "claim_boundary": "APPLES-like local distance placement, not official APPLES",
                }
            )

    for split in ["eval_c", "unseen_genera"]:
        cnn = coi[(coi["method"].astype(str) == "cnn_seed1206") & (coi["split"].astype(str) == split)]
        if not cnn.empty:
            row = cnn.iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "rank_no_call_assignment",
                    "split_or_dataset": split,
                    "selected_method": "cnn_seed1206_missing_ref_policy_target_0.99",
                    "metric": "assigned_precision_pct",
                    "value": row.get("missing_ref_policy_observed_precision_pct", ""),
                    "unit": "pct",
                    "source_table": "pipeline_coi_method_benchmark.csv",
                    "claim_boundary": "seen-test-derived policy; bootstrap/stress tested over query rows",
                }
            )
            rows.append(
                {
                    "pipeline_stage": "false_species_call_control",
                    "split_or_dataset": split,
                    "selected_method": "cnn_seed1206_missing_ref_policy_target_0.99",
                    "metric": "false_species_call_rate_pct",
                    "value": row.get("missing_ref_policy_false_species_call_rate_pct", ""),
                    "unit": "pct",
                    "source_table": "pipeline_coi_method_benchmark.csv",
                    "claim_boundary": "false species calls among all queries under locked rank/no-call policy",
                }
            )

    if not bootstrap.empty:
        cnn_eval = bootstrap[
            (bootstrap["prediction_set"].astype(str) == "cnn_seed1206")
            & (bootstrap["evaluation_split"].astype(str) == "eval_c")
        ]
        if not cnn_eval.empty:
            row = cnn_eval.iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "rank_policy_uncertainty",
                    "split_or_dataset": "eval_c",
                    "selected_method": "cnn_seed1206_missing_ref_policy_target_0.99",
                    "metric": "assigned_precision_bootstrap_ci95",
                    "value": f"{100 * row.get('assigned_precision_ci95_low', 0):.2f}-{100 * row.get('assigned_precision_ci95_high', 0):.2f}",
                    "unit": "pct",
                    "source_table": "missing_reference_aware_policy_bootstrap.csv",
                    "claim_boundary": "bootstrap over query rows, not independent dataset replication",
                }
            )

    if not pipeline_cal.empty:
        pdistance = pipeline_cal[
            (pipeline_cal["pipeline_mode"].astype(str).str.contains("p_distance", na=False))
            & (pd.to_numeric(pipeline_cal["target_precision"], errors="coerce") == 0.99)
        ].copy()
        for split in ["eval_c", "unseen_genera"]:
            sub = pdistance[pdistance["evaluation_split"].astype(str) == split]
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "reranked_rank_no_call_assignment",
                    "split_or_dataset": split,
                    "selected_method": "cnn_seed1206_pdistance_rerank_target_0.99",
                    "metric": "assigned_precision_pct",
                    "value": 100.0 * float(row.get("assigned_precision", 0.0)),
                    "unit": "pct",
                    "source_table": "pipeline_mode_policy_summary.csv",
                    "claim_boundary": "seen-test-derived rerank-specific thresholds; interpret with strict pruned rank-backoff validation",
                }
            )
            rows.append(
                {
                    "pipeline_stage": "reranked_false_species_call_control",
                    "split_or_dataset": split,
                    "selected_method": "cnn_seed1206_pdistance_rerank_target_0.99",
                    "metric": "false_species_call_rate_pct",
                    "value": 100.0 * float(row.get("false_species_call_rate_all_queries", 0.0)),
                    "unit": "pct",
                    "source_table": "pipeline_mode_policy_summary.csv",
                    "claim_boundary": "zero false species calls here occur because target-0.99 p-distance calibration backs off from species calls",
                }
            )

    if not production_v1.empty:
        for split in ["eval_c", "unseen_genera"]:
            sub = production_v1[production_v1["split"].astype(str) == split].copy()
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "production_v1_rank_no_call",
                    "split_or_dataset": split,
                    "selected_method": row.get("pipeline_mode", "production_v1"),
                    "metric": "assigned_precision_pct",
                    "value": 100.0 * float(row.get("assigned_precision", 0.0)),
                    "unit": "pct",
                    "source_table": "production_v1/production_v1_summary_all.csv",
                    "claim_boundary": "packaged saved-embedding COI production-v1 output; not raw FASTA-to-final-call inference yet",
                }
            )
            rows.append(
                {
                    "pipeline_stage": "production_v1_false_species_call_control",
                    "split_or_dataset": split,
                    "selected_method": row.get("pipeline_mode", "production_v1"),
                    "metric": "false_species_call_rate_pct",
                    "value": 100.0 * float(row.get("false_species_call_rate_all_queries", 0.0)),
                    "unit": "pct",
                    "source_table": "production_v1/production_v1_summary_all.csv",
                    "claim_boundary": "packaged saved-embedding COI production-v1 output; conservative p-distance calibration makes no species calls on held-out missing-reference splits",
                }
            )

    if not raw_production_v1.empty:
        for split in ["eval_c", "unseen_genera"]:
            sub = raw_production_v1[raw_production_v1["split"].astype(str) == split].copy()
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "raw_sequence_production_v1_latency",
                    "split_or_dataset": split,
                    "selected_method": "cnn_seed1206_exact_pdistance_rank_backoff",
                    "metric": "total_ms_per_query",
                    "value": row.get("total_ms_per_query", ""),
                    "unit": "ms/query",
                    "source_table": "raw_sequence_production_v1_summary.csv",
                    "claim_boundary": "raw split sequences through embedding export, exact vector retrieval, top-k p-distance rerank, and locked rank/no-call packaging; research pipeline, not packaged CLI",
                }
            )

    if not strict.empty:
        complete = int((strict["status"].astype(str) == "complete").sum()) if "status" in strict else 0
        rows.append(
            {
                "pipeline_stage": "strict_missing_reference_validation",
                "split_or_dataset": "eval_c_unseen_genera",
                "selected_method": "cnn_seed1206_strict_pruned_inputs",
                "metric": "complete_strict_runs",
                "value": f"{complete}/{len(strict)}",
                "unit": "runs",
                "source_table": "strict_missing_reference_summary.csv",
                "claim_boundary": "strict retrained CNN metrics complete; hidden ranks collapse to zero and broader-rank support is summarized separately",
            }
        )

    for context in ["geography_prior_only", "cooccurrence_prior_only"]:
        sub = edna[(edna["benchmark_level"].astype(str) == "global_edna_asv") & (edna["context"].astype(str) == context)]
        if sub.empty:
            continue
        best = best_value(sub, "family_top10_pct")
        if best:
            rows.append(
                {
                    "pipeline_stage": "edna_evidence_decomposition",
                    "split_or_dataset": "Global_eDNA",
                    "selected_method": best.get("run_name", ""),
                    "metric": f"{context}_family_top10_pct",
                    "value": best.get("family_top10_pct", ""),
                    "unit": "pct",
                    "source_table": "pipeline_edna_method_benchmark.csv",
                    "claim_boundary": "pure evidence arm, not final assignment system",
                }
            )

    if not edna_best_by_rank.empty:
        contextual = edna_best_by_rank[
            edna_best_by_rank["evidence_family"].astype(str).isin(
                ["sequence_plus_geography", "sequence_plus_cooccurrence"]
            )
        ].copy()
        for rank in ["genus", "family", "order"]:
            sub = contextual[contextual["rank"].astype(str) == rank].copy()
            if sub.empty:
                continue
            sub["asv_top10_pct"] = pd.to_numeric(sub["asv_top10_pct"], errors="coerce")
            sub = sub.dropna(subset=["asv_top10_pct"])
            if sub.empty:
                continue
            row = sub.sort_values("asv_top10_pct", ascending=False).iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "edna_integrated_evidence",
                    "split_or_dataset": "Global_eDNA",
                    "selected_method": row.get("best_run_name", ""),
                    "metric": f"{rank}_asv_top10_pct",
                    "value": row.get("asv_top10_pct", ""),
                    "unit": "pct",
                    "source_table": "edna_evidence_best_by_rank.csv",
                    "claim_boundary": "best current context-integrated arm; not a finalized posterior or independently calibrated rank/no-call policy",
                }
            )

    if not edna_no_call.empty:
        available = edna_no_call[edna_no_call["status"].astype(str) == "available"].copy()
        available["target_accuracy_pct"] = pd.to_numeric(available["target_accuracy_pct"], errors="coerce")
        available["assignment_rate_pct"] = pd.to_numeric(available["assignment_rate_pct"], errors="coerce")
        diagnostic = available[available["target_accuracy_pct"] == 50.0].copy()
        if not diagnostic.empty:
            row = diagnostic.sort_values("assignment_rate_pct", ascending=False).iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "edna_diagnostic_rank_no_call",
                    "split_or_dataset": "Global_eDNA",
                    "selected_method": row.get("method", ""),
                    "metric": f"{row.get('rank', '')}_assignment_rate_at_50pct_accuracy",
                    "value": row.get("assignment_rate_pct", ""),
                    "unit": "pct_assigned",
                    "source_table": "edna_rank_no_call_operating_points.csv",
                    "claim_boundary": "diagnostic same-validation operating point only; independent calibration is still required",
                }
            )

    if not edna_independent_no_call.empty:
        independent = edna_independent_no_call[
            edna_independent_no_call["status"].astype(str) == "available"
        ].copy()
        independent["target_accuracy_pct"] = pd.to_numeric(independent["target_accuracy_pct"], errors="coerce")
        independent["eval_assignment_rate_pct"] = pd.to_numeric(independent["eval_assignment_rate_pct"], errors="coerce")
        diagnostic = independent[independent["target_accuracy_pct"] == 50.0].copy()
        if not diagnostic.empty:
            row = diagnostic.sort_values("eval_assignment_rate_pct", ascending=False).iloc[0].to_dict()
            rows.append(
                {
                    "pipeline_stage": "edna_site_heldout_rank_no_call",
                    "split_or_dataset": "Global_eDNA_site20_holdout",
                    "selected_method": row.get("method", ""),
                    "metric": f"{row.get('rank', '')}_assignment_rate_at_50pct_calibrated_accuracy",
                    "value": row.get("eval_assignment_rate_pct", ""),
                    "unit": "pct_assigned",
                    "source_table": "global_edna_independent_rank_calibration_summary.csv",
                    "claim_boundary": "site-heldout threshold transfer; current operating points are modest and do not yet support high-accuracy eDNA rank/no-call claims",
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--rank-calibration-dir", type=Path, default=RANK_CAL)
    parser.add_argument("--output-dir", type=Path, default=SOURCE)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    rows = build_stage_rows(args.source_dir, args.rank_calibration_dir)
    output_path = args.output_dir / "pipeline_end_to_end_summary.csv"
    logger.log(f"Writing {len(rows)} end-to-end stage rows to {output_path}")
    write_csv(
        output_path,
        rows,
        [
            "pipeline_stage",
            "split_or_dataset",
            "selected_method",
            "metric",
            "value",
            "unit",
            "source_table",
            "claim_boundary",
        ],
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_paper1_end_to_end_summary.py",
        "rows": len(rows),
        "output_csv": str(output_path),
    }
    manifest_path = args.output_dir / "pipeline_end_to_end_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
