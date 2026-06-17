#!/usr/bin/env python3
"""Build manuscript-facing asset plans from Paper 1 source tables.

This script does not create new metrics. It packages existing source tables
into writing-oriented inventories: figure plans, table plans, claim/evidence
links, operating points, and remaining-result checklists.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
SOURCE = PAPER1 / "source_tables"
PIPELINE_CAL = PAPER1 / "pipeline_calibration"
EDNA_FULL_POSTERIOR = PAPER1 / "eco_phylo_posterior" / "candidate_level_sequence_tree_evidence_full"
EDNA_NESTED_POSTERIOR = (
    PAPER1 / "eco_phylo_posterior" / "candidate_level_sequence_tree_evidence_nested_fit70_rep0"
)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def fmt_pct_value(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def figure_plan() -> list[dict[str, Any]]:
    return [
        {
            "figure": "Fig 1",
            "panel": "A-D",
            "title": "Uncertainty-aware barcode/eDNA inference pipeline",
            "source_tables": "pipeline_component_status.csv; pipeline_next_actions.csv",
            "status": "ready_for_schematic",
            "depends_on": "",
            "purpose": "Show vector/classical candidate retrieval, tree/rerank diagnostics, calibration, rank/no-call, and eDNA context arms.",
        },
        {
            "figure": "Fig 2",
            "panel": "A-C",
            "title": "COI candidate retrieval and tree-space geometry",
            "source_tables": "pipeline_coi_method_benchmark.csv; tree_recovery_metrics.csv; neighborhood_preservation.csv",
            "status": "ready",
            "depends_on": "",
            "purpose": "Compare neural and classical candidate retrieval, tree-distance recovery, and neighborhood preservation.",
        },
        {
            "figure": "Fig 3",
            "panel": "A-D",
            "title": "Rank-adaptive assignment under missing references",
            "source_tables": "pipeline_run_summary.csv; pipeline_mode_policy_summary.csv; missing_reference_aware_policy_bootstrap.csv; strict_missing_reference_summary.csv; strict_rank_backoff_summary.csv",
            "status": "ready",
            "depends_on": "",
            "purpose": "Show coverage, assigned precision, false species-call control, and p-distance conservative backoff.",
        },
        {
            "figure": "Fig 4",
            "panel": "A-C",
            "title": "Vector-first retrieval speed and recall",
            "source_tables": "pipeline_vector_index_benchmark.csv; controlled_vector_speed_benchmark.csv",
            "status": "ready_with_larger_scale_stress_optional",
            "depends_on": "larger_reference_stress_if_making_deployment_scale_claims",
            "purpose": "Separate speed feasibility from final deployment claims.",
        },
        {
            "figure": "Fig 5",
            "panel": "A-C",
            "title": "Phylogenetic placement comparators and Fernando-aligned diagnostics",
            "source_tables": "pipeline_placement_benchmark.csv; placement_simulated_tree_pcp_summary.csv",
            "status": "diagnostic_ready_exact_fernando_pending",
            "depends_on": "fernando_completeness_sweeps_if_claiming_direct_comparability",
            "purpose": "Position EPA-ng and APPLES-like results without overclaiming exact Fernando PCP.",
        },
        {
            "figure": "Fig 6",
            "panel": "A-D",
            "title": "12S/eDNA marker ambiguity and Eco-Phylo posterior",
            "source_tables": "merged_12s_resolvability_summary.csv; pipeline_edna_method_benchmark.csv; eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv; eco_phylo_species_disabled_nested_calibration_summary.csv",
            "status": "ready_with_species_disabled_policy",
            "depends_on": "final_decision_on_paper_scope",
            "purpose": "Show that eDNA species assignment is marker-limited and that evidence fusion can support conservative genus/family/order/no-call decisions.",
        },
    ]


def table_plan() -> list[dict[str, Any]]:
    return [
        {
            "table": "Table 1",
            "title": "Datasets, markers, splits, and missing-reference regimes",
            "source_tables": "source_table_manifest.json; strict_missing_reference_summary.csv; strict_rank_backoff_summary.csv; merged_12s_resolvability_summary.csv",
            "status": "ready",
        },
        {
            "table": "Table 2",
            "title": "COI retrieval, tree recovery, and classical comparator results",
            "source_tables": "pipeline_coi_method_benchmark.csv; pipeline_placement_benchmark.csv",
            "status": "ready",
        },
        {
            "table": "Table 3",
            "title": "Executable pipeline operating points",
            "source_tables": "pipeline_run_summary.csv; pipeline_mode_policy_summary.csv",
            "status": "ready",
        },
        {
            "table": "Table 4",
            "title": "eDNA evidence decomposition, 12S resolvability, and species-disabled posterior",
            "source_tables": "pipeline_edna_method_benchmark.csv; merged_12s_resolvability_summary.csv; eco_phylo_species_disabled_nested_calibration_summary.csv",
            "status": "ready_if_merged_scope_kept",
        },
        {
            "table": "Supplementary Table S1",
            "title": "Claim boundaries and incomplete comparators",
            "source_tables": "pipeline_next_actions.csv; CLAIM_BOUNDARIES.md",
            "status": "ready",
        },
    ]


def claim_evidence_map() -> list[dict[str, Any]]:
    return [
        {
            "claim": "The system should output the deepest defensible rank, not force species.",
            "status": "supported",
            "primary_evidence": "pipeline_mode_policy_summary.csv; missing_reference_aware_policy_bootstrap.csv; strict_missing_reference_summary.csv; strict_rank_backoff_summary.csv",
            "boundary": "Strict validation is a CNN seed1206 stress test; final deployment still needs chosen operating points and broader replication.",
        },
        {
            "claim": "Classical sequence tools remain strong when close references exist.",
            "status": "supported",
            "primary_evidence": "pipeline_coi_method_benchmark.csv",
            "boundary": "Do not frame the paper as neural methods replacing BLAST/VSEARCH/k-mer.",
        },
        {
            "claim": "Learned vector retrieval is a fast candidate-generation layer.",
            "status": "supported",
            "primary_evidence": "pipeline_vector_index_benchmark.csv; controlled_vector_speed_benchmark.csv",
            "boundary": "Controlled Vast timing covers CNN seed1206 Eval C; deployment-scale claims still need larger reference stress tests.",
        },
        {
            "claim": "Train-reference p-distance reranking can reduce false species calls by backing off rank.",
            "status": "supported",
            "primary_evidence": "pipeline_mode_policy_summary.csv; strict_rank_backoff_summary.csv",
            "boundary": "At target 0.99 it makes no species calls; claim conservative rank-backoff, not species-level superiority.",
        },
        {
            "claim": "EPA-ng and APPLES-like placement are included as comparators.",
            "status": "partial",
            "primary_evidence": "pipeline_placement_benchmark.csv",
            "boundary": "APPLES-like is not official APPLES; current Fernando-like PCP is not exact Fernando PCP.",
        },
        {
            "claim": "12S species-level eDNA assignment is often marker-limited and needs evidence integration.",
            "status": "supported_as_scope_extension",
            "primary_evidence": "merged_12s_resolvability_summary.csv; pipeline_edna_method_benchmark.csv",
            "boundary": "Do not claim 12S sequence-only solves species assignment.",
        },
        {
            "claim": "A species-disabled Eco-Phylo posterior can make conservative higher-rank eDNA calls.",
            "status": "supported_with_boundary",
            "primary_evidence": "eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv; eco_phylo_species_disabled_nested_calibration_summary.csv; candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv",
            "boundary": "Species remains disabled. True nested posterior fit confirms useful higher-rank transfer but the mixed target-95 policy is below 95% held-out accuracy.",
        },
    ]


def operating_points() -> list[dict[str, Any]]:
    raw = read_csv(SOURCE / "pipeline_run_summary.csv")
    calibrated = read_csv(PIPELINE_CAL / "pipeline_mode_policy_summary.csv")
    rows: list[dict[str, Any]] = []
    if not raw.empty:
        for _, row in raw.iterrows():
            if str(row.get("split", "")) not in {"eval_c", "unseen_genera"}:
                continue
            rows.append(
                {
                    "source": "raw_pipeline_run",
                    "mode": row.get("run_name", ""),
                    "split": row.get("split", ""),
                    "target_precision": row.get("target_precision", ""),
                    "coverage_pct": fmt_pct(row.get("coverage")),
                    "assigned_precision_pct": fmt_pct(row.get("assigned_precision")),
                    "false_species_call_rate_pct": fmt_pct(row.get("false_species_call_rate_all_queries")),
                    "candidate_stage_ms_per_query": row.get("candidate_stage_ms_per_query", ""),
                    "species_calls": row.get("assigned_species_count", ""),
                    "genus_calls": row.get("assigned_genus_count", ""),
                    "family_calls": row.get("assigned_family_count", ""),
                    "order_calls": row.get("assigned_order_count", ""),
                    "no_calls": row.get("assigned_no_call_count", ""),
                    "claim_boundary": "Raw executable row; p-distance raw rows are less safe than rerank-specific calibration.",
                }
            )
    if not calibrated.empty:
        work = calibrated[pd.to_numeric(calibrated["target_precision"], errors="coerce") == 0.99]
        for _, row in work.iterrows():
            rows.append(
                {
                    "source": "seen_test_calibrated_pipeline_mode",
                    "mode": row.get("pipeline_mode", ""),
                    "split": row.get("evaluation_split", ""),
                    "target_precision": row.get("target_precision", ""),
                    "coverage_pct": fmt_pct(row.get("coverage")),
                    "assigned_precision_pct": fmt_pct(row.get("assigned_precision")),
                    "false_species_call_rate_pct": fmt_pct(row.get("false_species_call_rate_all_queries")),
                    "candidate_stage_ms_per_query": "",
                    "species_calls": row.get("assigned_species_count", ""),
                    "genus_calls": row.get("assigned_genus_count", ""),
                    "family_calls": row.get("assigned_family_count", ""),
                    "order_calls": row.get("assigned_order_count", ""),
                    "no_calls": row.get("assigned_no_call_count", ""),
                    "claim_boundary": "Seen-test-derived operating point; interpret with strict tree-pruned rank-backoff validation.",
                }
            )
    edna_nested = read_csv(EDNA_FULL_POSTERIOR / "eco_phylo_species_disabled_nested_calibration_summary.csv")
    if not edna_nested.empty:
        work = edna_nested[
            (edna_nested["evaluation_split"].astype(str) == "evaluation")
            & (pd.to_numeric(edna_nested["target_accuracy_pct"], errors="coerce") >= 90.0)
        ]
        for _, row in work.iterrows():
            rows.append(
                {
                    "source": "edna_species_disabled_nested_threshold_stability",
                    "mode": "eco_phylo_sequence_tree_species_disabled_rank_backoff",
                    "split": row.get("evaluation_split", ""),
                    "target_precision": row.get("target_accuracy_pct", ""),
                    "coverage_pct": fmt_pct_value(row.get("assignment_rate_pct_mean")),
                    "assigned_precision_pct": fmt_pct_value(row.get("assigned_accuracy_pct_mean")),
                    "false_species_call_rate_pct": "0.00",
                    "candidate_stage_ms_per_query": "",
                    "species_calls": 0,
                    "genus_calls": fmt_pct_value(row.get("genus_assignments_mean")),
                    "family_calls": fmt_pct_value(row.get("family_assignments_mean")),
                    "order_calls": fmt_pct_value(row.get("order_assignments_mean")),
                    "no_calls": "",
                    "claim_boundary": "Species is disabled; this is threshold-stability over a previously trained posterior, not a full nested posterior retrain.",
                }
            )
    true_nested = read_csv(EDNA_NESTED_POSTERIOR / "eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv")
    if not true_nested.empty:
        work = true_nested[
            (true_nested["split"].astype(str) == "evaluation")
            & (pd.to_numeric(true_nested["target_accuracy_pct"], errors="coerce") >= 90.0)
        ]
        for _, row in work.iterrows():
            rows.append(
                {
                    "source": "edna_true_nested_model_species_disabled",
                    "mode": "eco_phylo_sequence_tree_nested_fit70_species_disabled_rank_backoff",
                    "split": row.get("split", ""),
                    "target_precision": row.get("target_accuracy_pct", ""),
                    "coverage_pct": fmt_pct_value(row.get("assignment_rate_pct")),
                    "assigned_precision_pct": fmt_pct_value(row.get("assigned_accuracy_pct")),
                    "false_species_call_rate_pct": "0.00",
                    "candidate_stage_ms_per_query": "",
                    "species_calls": 0,
                    "genus_calls": row.get("genus_assignments", ""),
                    "family_calls": row.get("family_assignments", ""),
                    "order_calls": row.get("order_assignments", ""),
                    "no_calls": "",
                    "claim_boundary": "Species is disabled; this is the true nested fit70 posterior run, but target labels are calibration targets rather than guaranteed held-out accuracy.",
                }
            )
    return rows


def missing_results_checklist() -> list[dict[str, Any]]:
    actions = read_csv(SOURCE / "pipeline_next_actions.csv")
    strict = read_csv(SOURCE / "strict_missing_reference_summary.csv")
    rows: list[dict[str, Any]] = []
    if not actions.empty:
        for _, row in actions.iterrows():
            if str(row.get("status", "")) == "complete":
                continue
            rows.append(
                {
                    "item": row.get("task", ""),
                    "status": row.get("status", ""),
                    "needed_for": row.get("why", ""),
                    "blocked_by_or_next_step": row.get("blocked_by", ""),
                    "priority": row.get("priority", ""),
                }
            )
    if not strict.empty:
        pending = strict[strict["status"].astype(str) != "complete"] if "status" in strict else strict
        if len(pending):
            rows.append(
                {
                    "item": "strict_missing_reference_validation",
                    "status": f"{len(strict) - len(pending)}/{len(strict)} complete",
                    "needed_for": "Final missing-reference and rank-backoff claim.",
                    "blocked_by_or_next_step": "Vast queue currently running; copy and summarize when complete.",
                    "priority": 1,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER1 / "manuscript_assets",
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    outputs = [
        (
            args.output_dir / "figure_plan.csv",
            figure_plan(),
            ["figure", "panel", "title", "source_tables", "status", "depends_on", "purpose"],
        ),
        (
            args.output_dir / "table_plan.csv",
            table_plan(),
            ["table", "title", "source_tables", "status"],
        ),
        (
            args.output_dir / "claim_evidence_map.csv",
            claim_evidence_map(),
            ["claim", "status", "primary_evidence", "boundary"],
        ),
        (
            args.output_dir / "pipeline_operating_points.csv",
            operating_points(),
            [
                "source",
                "mode",
                "split",
                "target_precision",
                "coverage_pct",
                "assigned_precision_pct",
                "false_species_call_rate_pct",
                "candidate_stage_ms_per_query",
                "species_calls",
                "genus_calls",
                "family_calls",
                "order_calls",
                "no_calls",
                "claim_boundary",
            ],
        ),
        (
            args.output_dir / "missing_results_checklist.csv",
            missing_results_checklist(),
            ["item", "status", "needed_for", "blocked_by_or_next_step", "priority"],
        ),
    ]
    manifest_files: dict[str, int] = {}
    for path, rows, fieldnames in outputs:
        logger.log(f"Writing {len(rows)} rows to {path}")
        write_csv(path, rows, fieldnames)
        manifest_files[path.name] = len(rows)

    manifest = {
        "generated_by": "scripts/edna/build_paper1_manuscript_assets.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "files": manifest_files,
        "claim_boundary": "Planning/source inventory only. Manuscript figures still require visual rendering and final operating-point selection.",
    }
    manifest_path = args.output_dir / "manuscript_asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
