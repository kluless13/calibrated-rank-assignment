#!/usr/bin/env python3
"""Build Paper 1 pipeline-level benchmark tables.

The lower-level source tables already measure retrieval, tree geometry,
reference diagnostics, placement, vector speed, 12S resolvability, Global_eDNA,
and calibration. This script joins those pieces into a small set of pipeline
tables that answer: what works now, where does it work, and what is still
missing before the pipeline can support manuscript claims.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
SOURCE = PAPER1 / "source_tables"
OUT_DIR = SOURCE

RANKS = ("species", "genus", "family", "order")
SPLITS = ("eval_c", "seen_test", "unseen_genera")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def read_csv(path: Path) -> pd.DataFrame:
    if not exists_nonempty(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float | str:
    if value in ("", None):
        return ""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(out) else out


def pct(value: Any) -> float | str:
    out = finite(value)
    return "" if out == "" else 100.0 * float(out)


def first_value(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df.columns:
        return ""
    value = df.iloc[0][column]
    return "" if pd.isna(value) else value


def clean_group_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return value


def status_rows() -> list[dict[str, Any]]:
    checks = [
        (
            "coi_candidate_retrieval",
            "available",
            "retrieval_metrics.csv",
            "COI species/genus/family/order top-k across neural, classical, and controls.",
        ),
        (
            "coi_tree_geometry",
            "available",
            "tree_recovery_metrics.csv; tree_distance_*; neighborhood_preservation.csv",
            "Sequence-to-tree embedding geometry and neighborhood preservation.",
        ),
        (
            "coi_missing_reference_rank_backoff",
            "available",
            "candidate_ablation_rank_backoff.csv; full_candidate_embedding_ablation*.csv; strict_missing_reference_summary.csv",
            "Post-hoc hidden species/genus/family diagnostics and strict tree-pruned retraining summaries are available.",
        ),
        (
            "coi_reference_diagnostics",
            "available",
            "reference_diagnostics_summary.csv",
            "Nearest-reference tree-distance bins fixed locally.",
        ),
        (
            "coi_vector_first_retrieval",
            "available",
            "vector_first_retrieval_metrics.csv; vector_first_runtime_comparison.csv; ann_vector_*.csv; controlled_vector_speed_benchmark.csv",
            "Exact cosine, HNSW ANN, stress, and repeat-based controlled vector benchmarks; final production speed still needs target-hardware replication.",
        ),
        (
            "coi_phylogenetic_placement",
            "partial",
            "placement_rank_diagnostics_summary.csv; placement_lwr_rank_summary.csv; placement_rank_backoff_summary.csv; apples_like_distance_placement_summary.csv; placement_simulated_tree_pcp_summary.csv",
            "EPA-ng and a labelled APPLES-like local distance-placement diagnostic are scored for all clean splits; simulated-placement-tree PCP exists, but matched Fernando completeness sweeps and official APPLES remain.",
        ),
        (
            "coi_rank_adaptive_no_call",
            "partial",
            "../rank_adaptive_calibration/*.csv; pipeline_run_summary.csv",
            "Same-split curves, seen-test-to-heldout transfer, missing-reference-aware policy, bootstrap intervals, exact/HNSW executable pipeline rows, p-distance rerank rows, and strict pruned validation exist; final manuscript operating-point table remains.",
        ),
        (
            "12s_marker_resolvability",
            "available",
            "merged_12s_resolvability_summary.csv",
            "Exact and near-exact marker information ceiling.",
        ),
        (
            "global_edna_evidence_integration",
            "available",
            "merged_global_edna_*; merged_edna_evidence_arm_status.csv; edna_evidence_decomposition_matrix.csv; edna_evidence_best_by_rank.csv; edna_rank_no_call_operating_points.csv; ../global_edna_independent_rank_calibration/*.csv",
            "Sequence/tree, learned co-occurrence, geography-only, same-sample co-occurrence-only, diagnostic rank/no-call, and site-heldout eDNA calibration arms are separated.",
        ),
        (
            "global_edna_eco_phylo_posterior",
            "partial",
            "../eco_phylo_posterior/*.csv; ../eco_phylo_posterior/*.json; ../eco_phylo_posterior/candidate_level/*.csv; ../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/*.csv; ../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/*.csv",
            "Top-1 method-arm posterior, full-table candidate-level posterior, full sequence+tree posterior, species-disabled threshold-stability, and true nested fit70 posterior outputs exist. Direct train-reference 12S p-distance evidence and inference-safe candidate tree-neighborhood evidence are included. Species thresholds still fail transfer. Family/order target-95 transfer well individually; mixed species-disabled target-95 reaches 38.9% held-out assignment at 93.4% accuracy under the true nested fit.",
        ),
    ]
    rows = []
    for component, status, files, notes in checks:
        rows.append(
            {
                "pipeline_component": component,
                "status": status,
                "source_tables": files,
                "notes": notes,
            }
        )
    return rows


def build_coi_method_rows() -> list[dict[str, Any]]:
    retrieval = read_csv(SOURCE / "retrieval_metrics.csv")
    tree = read_csv(SOURCE / "tree_recovery_metrics.csv")
    ablation = read_csv(SOURCE / "candidate_ablation_rank_backoff.csv")
    policy = read_csv(PAPER1 / "rank_adaptive_calibration" / "rank_adaptive_policy_summary.csv")
    prospective_policy = read_csv(PAPER1 / "rank_adaptive_calibration" / "prospective_rank_adaptive_policy_summary.csv")
    missing_ref_policy = read_csv(PAPER1 / "rank_adaptive_calibration" / "missing_reference_aware_policy_summary.csv")
    missing_ref_bootstrap = read_csv(PAPER1 / "rank_adaptive_calibration" / "missing_reference_aware_policy_bootstrap.csv")
    runtime = read_csv(SOURCE / "vector_first_runtime_comparison.csv")
    rows: list[dict[str, Any]] = []

    if retrieval.empty:
        return rows

    grouped = retrieval.groupby(["method", "method_family", "split"], dropna=False)
    for (method, family, split), group in grouped:
        row: dict[str, Any] = {
            "method": method,
            "method_family": family,
            "split": split,
        }
        for rank in RANKS:
            rank_row = group[group["rank"] == rank]
            row[f"{rank}_top1_pct"] = finite(first_value(rank_row, "top1_pct"))
            row[f"{rank}_top5_pct"] = finite(first_value(rank_row, "top5_pct"))
            row[f"{rank}_top10_pct"] = finite(first_value(rank_row, "top10_pct"))

        if not tree.empty:
            tree_row = tree[
                (tree["method"] == method)
                & (tree["split"] == split)
                & (tree["comparison"] == "zero_shot_reference")
            ]
            row["tree_zero_shot_reference_pearson"] = finite(first_value(tree_row, "pearson_r"))
            row["tree_zero_shot_reference_spearman"] = finite(first_value(tree_row, "spearman_r"))

        if not ablation.empty:
            for target in ("genus", "family", "order"):
                hidden = ablation[
                    (ablation["method"] == method)
                    & (ablation["split"] == split)
                    & (ablation["ablation"] == "hide_true_species")
                    & (ablation["target_rank"] == target)
                ]
                row[f"hide_true_species_{target}_top10_pct"] = finite(first_value(hidden, "top10_pct"))

        policy_name_candidates = policy_names_for_method(method)
        if not policy.empty:
            best_policy = policy[
                (policy["split"] == split)
                & (policy["prediction_set"].isin(policy_name_candidates))
                & (policy["confidence_feature"] == "confidence_margin")
                & (policy["target_precision"] == 0.9)
            ]
            if not best_policy.empty:
                row["rank_policy_target_precision"] = 0.9
                row["rank_policy_coverage_pct"] = 100.0 * float(first_value(best_policy, "coverage"))
                row["rank_policy_observed_precision_pct"] = 100.0 * float(first_value(best_policy, "assigned_precision"))
                row["rank_policy_species_calls"] = first_value(best_policy, "assigned_species_count")
                row["rank_policy_genus_calls"] = first_value(best_policy, "assigned_genus_count")
                row["rank_policy_family_calls"] = first_value(best_policy, "assigned_family_count")
                row["rank_policy_order_calls"] = first_value(best_policy, "assigned_order_count")
                row["rank_policy_no_calls"] = first_value(best_policy, "assigned_no_call_count")

        if not prospective_policy.empty:
            prospective = prospective_policy[
                (prospective_policy["evaluation_split"] == split)
                & (prospective_policy["prediction_set"].isin(policy_name_candidates))
                & (prospective_policy["confidence_feature"] == "confidence_margin")
                & (prospective_policy["target_precision"] == 0.9)
            ]
            if not prospective.empty:
                row["prospective_rank_policy_calibration_split"] = first_value(prospective, "calibration_split")
                row["prospective_rank_policy_target_precision"] = 0.9
                row["prospective_rank_policy_coverage_pct"] = pct(first_value(prospective, "coverage"))
                row["prospective_rank_policy_observed_precision_pct"] = pct(first_value(prospective, "assigned_precision"))
                row["prospective_rank_policy_species_calls"] = first_value(prospective, "assigned_species_count")
                row["prospective_rank_policy_genus_calls"] = first_value(prospective, "assigned_genus_count")
                row["prospective_rank_policy_family_calls"] = first_value(prospective, "assigned_family_count")
                row["prospective_rank_policy_order_calls"] = first_value(prospective, "assigned_order_count")
                row["prospective_rank_policy_no_calls"] = first_value(prospective, "assigned_no_call_count")

        if not missing_ref_policy.empty:
            locked = missing_ref_policy[
                (missing_ref_policy["evaluation_split"] == split)
                & (missing_ref_policy["prediction_set"].isin(policy_name_candidates))
                & (missing_ref_policy["target_precision"] == 0.99)
            ]
            if not locked.empty:
                row["missing_ref_policy_target_precision"] = 0.99
                row["missing_ref_policy_coverage_pct"] = pct(first_value(locked, "coverage"))
                row["missing_ref_policy_observed_precision_pct"] = pct(first_value(locked, "assigned_precision"))
                row["missing_ref_policy_precision_ci95_low_pct"] = pct(first_value(locked, "assigned_precision_ci95_low"))
                row["missing_ref_policy_precision_ci95_high_pct"] = pct(first_value(locked, "assigned_precision_ci95_high"))
                row["missing_ref_policy_species_calls"] = first_value(locked, "assigned_species_count")
                row["missing_ref_policy_genus_calls"] = first_value(locked, "assigned_genus_count")
                row["missing_ref_policy_family_calls"] = first_value(locked, "assigned_family_count")
                row["missing_ref_policy_order_calls"] = first_value(locked, "assigned_order_count")
                row["missing_ref_policy_no_calls"] = first_value(locked, "assigned_no_call_count")
                if not missing_ref_bootstrap.empty:
                    boot = missing_ref_bootstrap[
                        (missing_ref_bootstrap["evaluation_split"] == split)
                        & (missing_ref_bootstrap["prediction_set"].isin(policy_name_candidates))
                        & (missing_ref_bootstrap["target_precision"] == 0.99)
                    ]
                    if not boot.empty:
                        row["missing_ref_policy_bootstrap_precision_ci95_low_pct"] = pct(
                            first_value(boot, "assigned_precision_ci95_low")
                        )
                        row["missing_ref_policy_bootstrap_precision_ci95_high_pct"] = pct(
                            first_value(boot, "assigned_precision_ci95_high")
                        )
                        row["missing_ref_policy_false_species_call_rate_pct"] = pct(
                            first_value(boot, "false_species_call_rate_all_queries")
                        )
                        row["missing_ref_policy_false_species_call_rate_ci95_low_pct"] = pct(
                            first_value(boot, "false_species_call_rate_all_queries_ci95_low")
                        )
                        row["missing_ref_policy_false_species_call_rate_ci95_high_pct"] = pct(
                            first_value(boot, "false_species_call_rate_all_queries_ci95_high")
                        )

        if not runtime.empty:
            runtime_method = method
            if method.startswith("cnn_seed"):
                runtime_method = "cnn"
            runtime_row = runtime[
                (runtime["method"].astype(str) == runtime_method)
                & (runtime["split"].astype(str) == split)
            ]
            if runtime_row.empty and method in {"blast", "vsearch", "kmer"}:
                runtime_row = runtime[
                    (runtime["method"].astype(str) == method)
                    & (runtime["split"].astype(str) == split)
                ]
            if not runtime_row.empty:
                row["runtime_mode"] = first_value(runtime_row, "retrieval_mode")
                row["runtime_ms_per_query"] = finite(first_value(runtime_row, "milliseconds_per_query"))

        rows.append(row)

    return rows


def policy_names_for_method(method: str) -> list[str]:
    names = [method]
    if method.startswith("mamba_"):
        mapped = method.replace("mamba_", "neural_")
        names.append(mapped)
        names.append(mapped.replace("_seqval", ""))
    return list(dict.fromkeys(names))


def build_placement_rows() -> list[dict[str, Any]]:
    placement = read_csv(SOURCE / "placement_rank_diagnostics_summary.csv")
    apples_like = read_csv(SOURCE / "apples_like_distance_placement_summary.csv")
    pcp_like = read_csv(SOURCE / "placement_pcp_like_summary.csv")
    simulated_pcp = read_csv(SOURCE / "placement_simulated_tree_pcp_summary.csv")
    if not apples_like.empty:
        placement = pd.concat([placement, apples_like], ignore_index=True, sort=False)
    rows: list[dict[str, Any]] = []
    if placement.empty:
        return rows
    pcp_index: dict[tuple[str, str], dict[str, Any]] = {}
    if not pcp_like.empty:
        for _, pcp_row in pcp_like.iterrows():
            pcp_index[(str(pcp_row.get("split", "")), str(pcp_row.get("method", "")))] = dict(pcp_row)
    simulated_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not simulated_pcp.empty:
        for _, simulated_row in simulated_pcp.iterrows():
            key = (
                str(simulated_row.get("split", "")),
                str(simulated_row.get("method", "")),
                str(simulated_row.get("pcp_level", "")),
            )
            simulated_index[key] = dict(simulated_row)
    for _, item in placement.iterrows():
        method = str(item.get("method", ""))
        caution = (
            "APPLES-like local p-distance placement; not official APPLES"
            if method.startswith("apples_like")
            else "placed-clade containment, tree-distance-to-placed-clade, and simulated-tree PCP diagnostics; not matched Fernando completeness PCP"
        )
        row = {
            "split": item.get("split", ""),
            "method": method,
            "species_in_placed_clade_pct": 100.0 * float(item.get("species_in_placed_clade_rate", 0.0)),
            "genus_in_placed_clade_pct": 100.0 * float(item.get("genus_in_placed_clade_rate", 0.0)),
            "family_in_placed_clade_pct": 100.0 * float(item.get("family_in_placed_clade_rate", 0.0)),
            "order_in_placed_clade_pct": 100.0 * float(item.get("order_in_placed_clade_rate", 0.0)),
            "placement_min_tree_distance_median": finite(item.get("placement_min_tree_distance_median")),
            "placement_min_tree_distance_mean": finite(item.get("placement_min_tree_distance_mean")),
            "placement_excess_tree_distance_median": finite(item.get("placement_excess_tree_distance_median")),
            "placement_excess_tree_distance_mean": finite(item.get("placement_excess_tree_distance_mean")),
            "most_specific_rank_species_pct": 100.0 * finite(item.get("most_specific_rank_species_rate")),
            "most_specific_rank_genus_pct": 100.0 * finite(item.get("most_specific_rank_genus_rate")),
            "most_specific_rank_family_pct": 100.0 * finite(item.get("most_specific_rank_family_rate")),
            "most_specific_rank_order_pct": 100.0 * finite(item.get("most_specific_rank_order_rate")),
            "most_specific_rank_none_pct": 100.0 * finite(item.get("most_specific_rank_none_rate")),
            "n_queries_scored": item.get("n_queries_scored", ""),
            "missing_query_metadata": item.get("missing_query_metadata", ""),
            "nearest_reference_match_pct": pct(item.get("nearest_reference_match_rate", "")),
            "caution": caution,
            "source_file": item.get("jplace_file", item.get("source_file", "")),
        }
        pcp_row = pcp_index.get((str(item.get("split", "")), method))
        if pcp_row:
            row["pcp_like_exact_sister_match_pct"] = pct(pcp_row.get("supported_exact_sister_match_rate"))
            row["pcp_like_any_sister_overlap_pct"] = pct(pcp_row.get("supported_any_sister_overlap_rate"))
            row["pcp_like_best_jaccard_median"] = finite(pcp_row.get("supported_best_jaccard_median"))
            row["pcp_like_lwr_ge_0p9_exact_sister_match_pct"] = pct(
                pcp_row.get("lwr_ge_0p9_exact_sister_match_rate")
            )
            row["pcp_like_lwr_ge_0p9_any_sister_overlap_pct"] = pct(
                pcp_row.get("lwr_ge_0p9_any_sister_overlap_rate")
            )
        for level, prefix in (
            ("species_representative", "simulated_species"),
            ("sequence_level", "simulated_sequence"),
        ):
            simulated_row = simulated_index.get((str(item.get("split", "")), method, level))
            if simulated_row:
                row[f"{prefix}_pcp_exact_pct"] = pct(simulated_row.get("represented_pcp_represented_exact_rate"))
                row[f"{prefix}_pcp_overlap_pct"] = pct(
                    simulated_row.get("represented_pcp_represented_any_overlap_rate")
                )
                row[f"{prefix}_pcp_jaccard_median"] = finite(
                    simulated_row.get("represented_pcp_represented_jaccard_median")
                )
                row[f"{prefix}_pcp_lwr_ge_0p9_exact_pct"] = pct(
                    simulated_row.get("lwr_ge_0p9_pcp_represented_exact_rate")
                )
                row[f"{prefix}_pcp_lwr_ge_0p9_overlap_pct"] = pct(
                    simulated_row.get("lwr_ge_0p9_pcp_represented_any_overlap_rate")
                )
                row[f"{prefix}_pcp_represented_sister_rate_pct"] = pct(
                    simulated_row.get("represented_full_sister_rate")
                )
        rows.append(row)
    return rows


def build_vector_index_rows() -> list[dict[str, Any]]:
    runtime = read_csv(SOURCE / "ann_vector_runtime_comparison.csv")
    recall = read_csv(SOURCE / "ann_vector_recall_against_exact.csv")
    stress = read_csv(SOURCE / "ann_vector_stress_runtime.csv")
    controlled = read_csv(SOURCE / "controlled_vector_speed_benchmark.csv")
    rows: list[dict[str, Any]] = []
    if runtime.empty and stress.empty and controlled.empty:
        return rows
    vector_runtime = pd.DataFrame()
    if not runtime.empty:
        vector_runtime = runtime[
            runtime["retrieval_mode"].astype(str).str.startswith("hnsw")
            | (runtime["retrieval_mode"].astype(str) == "exact_vector_cosine")
        ].copy()
    if not stress.empty:
        stress = stress.copy()
        stress["retrieval_mode"] = stress["retrieval_mode"].astype(str) + "_x" + stress["stress_multiplier"].astype(str)
        vector_runtime = pd.concat([vector_runtime, stress], ignore_index=True, sort=False)
    if not controlled.empty:
        controlled = controlled.copy()
        controlled["index_mb"] = controlled.get("index_mb_median", "")
        controlled["index_build_seconds"] = controlled.get("index_build_seconds_median", "")
        controlled["search_seconds"] = controlled.get("search_seconds_median", "")
        controlled["queries_per_second"] = controlled.get("queries_per_second_median", "")
        controlled["milliseconds_per_query"] = controlled.get("milliseconds_per_query_median", "")
        controlled["source_file"] = controlled.get("source_file", "")
        vector_runtime = pd.concat([vector_runtime, controlled], ignore_index=True, sort=False)
    if vector_runtime.empty:
        return rows

    recall_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if not recall.empty:
        for _, item in recall.iterrows():
            key = (
                str(item.get("method", "")),
                str(item.get("seed", "")),
                str(item.get("split", "")),
                str(item.get("retrieval_mode", "")),
            )
            recall_index.setdefault(key, {})
            top_k = item.get("top_k", "")
            recall_index[key][f"recall_at_{top_k}"] = finite(item.get("mean_recall_vs_exact", ""))
            if int(top_k) == 1:
                recall_index[key]["top1_match_rate"] = finite(item.get("top1_match_rate", ""))

    for _, item in vector_runtime.iterrows():
        key = (
            str(item.get("method", "")),
            str(item.get("seed", "")),
            str(item.get("split", "")),
            str(item.get("retrieval_mode", "")),
        )
        rec = recall_index.get(key, {})
        rows.append(
            {
                "method": item.get("method", ""),
                "seed": item.get("seed", ""),
                "split": item.get("split", ""),
                "retrieval_mode": item.get("retrieval_mode", ""),
                "n_queries": item.get("n_queries", ""),
                "n_candidates": item.get("n_candidates", ""),
                "embedding_dim": item.get("embedding_dim", ""),
                "index_mb": finite(item.get("index_mb", "")),
                "index_build_seconds": finite(item.get("index_build_seconds", "")),
                "search_seconds": finite(item.get("search_seconds", "")),
                "queries_per_second": finite(item.get("queries_per_second", "")),
                "milliseconds_per_query": finite(item.get("milliseconds_per_query", "")),
                "recall_at_1": rec.get("recall_at_1", ""),
                "recall_at_5": rec.get("recall_at_5", ""),
                "recall_at_10": rec.get("recall_at_10", ""),
                "recall_at_50": rec.get("recall_at_50", ""),
                "top1_match_rate": rec.get("top1_match_rate", ""),
                "source_file": item.get("source_file", ""),
            }
        )
    return rows


def build_edna_rows() -> list[dict[str, Any]]:
    zero = read_csv(SOURCE / "merged_12s_zero_shot_model_metrics.csv")
    asv = read_csv(SOURCE / "merged_global_edna_asv_metrics.csv")
    sample = read_csv(SOURCE / "merged_global_edna_sample_metrics.csv")
    rows: list[dict[str, Any]] = []

    if not zero.empty:
        for (run_name, dataset, encoder, context, prior_source, prior_weight), group in zero.groupby(
            ["run_name", "dataset", "encoder", "context", "prior_source", "prior_weight"],
            dropna=False,
        ):
            row: dict[str, Any] = {
                "run_name": clean_group_value(run_name),
                "dataset": clean_group_value(dataset),
                "encoder": clean_group_value(encoder),
                "context": clean_group_value(context),
                "prior_source": clean_group_value(prior_source),
                "prior_weight": clean_group_value(prior_weight),
                "benchmark_level": "12s_zero_shot_candidate",
            }
            for rank in RANKS:
                rank_row = group[group["rank"] == rank]
                row[f"{rank}_top10_pct"] = finite(first_value(rank_row, "top10_pct"))
            rows.append(row)

    if not asv.empty:
        for (run_name, dataset, encoder, context, prior_source, prior_weight), group in asv.groupby(
            ["run_name", "dataset", "encoder", "context", "prior_source", "prior_weight"],
            dropna=False,
        ):
            row = {
                "run_name": clean_group_value(run_name),
                "dataset": clean_group_value(dataset),
                "encoder": clean_group_value(encoder),
                "context": clean_group_value(context),
                "prior_source": clean_group_value(prior_source),
                "prior_weight": clean_group_value(prior_weight),
                "benchmark_level": "global_edna_asv",
            }
            for rank in RANKS:
                rank_row = group[group["rank"] == rank]
                row[f"{rank}_top10_pct"] = finite(first_value(rank_row, "top10_pct"))
            rows.append(row)

    if not sample.empty:
        precision = sample[(sample["metric"] == "precision") & (sample["top_k"] == 10)]
        for (run_name, dataset, encoder, context, prior_source, prior_weight), group in precision.groupby(
            ["run_name", "dataset", "encoder", "context", "prior_source", "prior_weight"],
            dropna=False,
        ):
            row = {
                "run_name": clean_group_value(run_name),
                "dataset": clean_group_value(dataset),
                "encoder": clean_group_value(encoder),
                "context": clean_group_value(context),
                "prior_source": clean_group_value(prior_source),
                "prior_weight": clean_group_value(prior_weight),
                "benchmark_level": "global_edna_sample_top10_precision",
            }
            for rank in RANKS:
                rank_row = group[group["rank"] == rank]
                row[f"{rank}_top10_pct"] = 100.0 * float(first_value(rank_row, "mean")) if not rank_row.empty else ""
            rows.append(row)

    return rows


def build_best_task_rows(
    coi_rows: list[dict[str, Any]],
    edna_rows: list[dict[str, Any]],
    placement_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coi = pd.DataFrame(coi_rows)
    if not coi.empty:
        for split in SPLITS:
            sub = coi[coi["split"] == split].copy()
            if sub.empty:
                continue
            for rank in RANKS:
                metric = f"{rank}_top10_pct"
                if metric not in sub:
                    continue
                work = sub.copy()
                work[metric] = pd.to_numeric(work[metric], errors="coerce")
                work = work.dropna(subset=[metric])
                if work.empty:
                    continue
                best = work.sort_values(metric, ascending=False).iloc[0]
                rows.append(
                    {
                        "domain": "coi",
                        "task": f"{split}_{rank}_top10_retrieval",
                        "best_method": best["method"],
                        "value": best[metric],
                        "unit": "pct",
                        "interpretation": "forced top-k retrieval; not calibrated rank/no-call",
                    }
                )
            if "tree_zero_shot_reference_pearson" in sub:
                work = sub.copy()
                work["tree_zero_shot_reference_pearson"] = pd.to_numeric(
                    work["tree_zero_shot_reference_pearson"], errors="coerce"
                )
                work = work.dropna(subset=["tree_zero_shot_reference_pearson"])
                if not work.empty:
                    best = work.sort_values("tree_zero_shot_reference_pearson", ascending=False).iloc[0]
                    rows.append(
                        {
                            "domain": "coi",
                            "task": f"{split}_tree_recovery_zero_shot_reference_pearson",
                            "best_method": best["method"],
                            "value": best["tree_zero_shot_reference_pearson"],
                            "unit": "pearson_r",
                            "interpretation": "sequence-to-tree geometry recovery",
                        }
                    )
            if "rank_policy_coverage_pct" in sub:
                work = sub.copy()
                work["rank_policy_coverage_pct"] = pd.to_numeric(work["rank_policy_coverage_pct"], errors="coerce")
                work = work.dropna(subset=["rank_policy_coverage_pct"])
                if not work.empty:
                    best = work.sort_values("rank_policy_coverage_pct", ascending=False).iloc[0]
                    rows.append(
                        {
                            "domain": "coi",
                            "task": f"{split}_rank_policy_coverage_at_target_0p9",
                            "best_method": best["method"],
                            "value": best["rank_policy_coverage_pct"],
                            "unit": "pct_coverage",
                            "interpretation": "same-split diagnostic policy, not final calibration",
                        }
                    )
            if "prospective_rank_policy_observed_precision_pct" in sub:
                work = sub.copy()
                work["prospective_rank_policy_observed_precision_pct"] = pd.to_numeric(
                    work["prospective_rank_policy_observed_precision_pct"], errors="coerce"
                )
                work["prospective_rank_policy_coverage_pct"] = pd.to_numeric(
                    work.get("prospective_rank_policy_coverage_pct"), errors="coerce"
                )
                work = work.dropna(subset=["prospective_rank_policy_observed_precision_pct"])
                if not work.empty:
                    best = work.sort_values(
                        ["prospective_rank_policy_observed_precision_pct", "prospective_rank_policy_coverage_pct"],
                        ascending=False,
                    ).iloc[0]
                    rows.append(
                        {
                            "domain": "coi",
                            "task": f"{split}_prospective_rank_policy_precision_at_target_0p9",
                            "best_method": best["method"],
                            "value": best["prospective_rank_policy_observed_precision_pct"],
                            "unit": "pct_precision",
                            "interpretation": "thresholds learned on seen-test and evaluated on this split",
                        }
                    )
            if "missing_ref_policy_observed_precision_pct" in sub:
                work = sub.copy()
                work["missing_ref_policy_observed_precision_pct"] = pd.to_numeric(
                    work["missing_ref_policy_observed_precision_pct"], errors="coerce"
                )
                work["missing_ref_policy_coverage_pct"] = pd.to_numeric(
                    work.get("missing_ref_policy_coverage_pct"), errors="coerce"
                )
                work = work.dropna(subset=["missing_ref_policy_observed_precision_pct"])
                if not work.empty:
                    best = work.sort_values(
                        ["missing_ref_policy_observed_precision_pct", "missing_ref_policy_coverage_pct"],
                        ascending=False,
                    ).iloc[0]
                    rows.append(
                        {
                            "domain": "coi",
                            "task": f"{split}_missing_reference_policy_precision_at_target_0p99",
                            "best_method": best["method"],
                            "value": best["missing_ref_policy_observed_precision_pct"],
                            "unit": "pct_precision",
                            "interpretation": "candidate locked rank-backoff policy using top-k taxonomy consensus",
                        }
                    )

    edna = pd.DataFrame(edna_rows)
    if not edna.empty:
        for level in sorted(edna["benchmark_level"].dropna().unique()):
            sub = edna[edna["benchmark_level"] == level].copy()
            for rank in RANKS:
                metric = f"{rank}_top10_pct"
                if metric not in sub:
                    continue
                work = sub.copy()
                work[metric] = pd.to_numeric(work[metric], errors="coerce")
                work = work.dropna(subset=[metric])
                if work.empty:
                    continue
                best = work.sort_values(metric, ascending=False).iloc[0]
                rows.append(
                    {
                        "domain": "12s_edna",
                        "task": f"{level}_{rank}_top10",
                        "best_method": best["run_name"],
                        "value": best[metric],
                        "unit": "pct",
                        "interpretation": "current 12S/eDNA evidence arm benchmark",
                    }
                )

    placement = pd.DataFrame(placement_rows)
    if not placement.empty:
        for rank in RANKS:
            metric = f"{rank}_in_placed_clade_pct"
            work = placement.copy()
            work[metric] = pd.to_numeric(work[metric], errors="coerce")
            work = work.dropna(subset=[metric])
            if work.empty:
                continue
            best = work.sort_values(metric, ascending=False).iloc[0]
            rows.append(
                {
                    "domain": "coi_placement",
                    "task": f"{best['split']}_{rank}_placed_clade_containment",
                    "best_method": best["method"],
                    "value": best[metric],
                    "unit": "pct",
                    "interpretation": "Fernando-adjacent containment diagnostic, not full PCP",
                }
            )

    vector = pd.DataFrame(vector_rows)
    if not vector.empty:
        hnsw = vector[vector["retrieval_mode"].astype(str).str.startswith("hnsw")].copy()
        for split in SPLITS:
            sub = hnsw[hnsw["split"].astype(str) == split].copy()
            if sub.empty:
                continue
            sub["milliseconds_per_query"] = pd.to_numeric(sub["milliseconds_per_query"], errors="coerce")
            fastest = sub.dropna(subset=["milliseconds_per_query"]).sort_values("milliseconds_per_query").head(1)
            if not fastest.empty:
                best = fastest.iloc[0]
                rows.append(
                    {
                        "domain": "coi_vector_index",
                        "task": f"{split}_fastest_hnsw_ms_per_query",
                        "best_method": f"{best['method']}_{best['seed']}_{best['retrieval_mode']}",
                        "value": best["milliseconds_per_query"],
                        "unit": "ms_per_query",
                        "interpretation": "local HNSW vector-index speed; rerun on controlled hardware before final speed claims",
                    }
                )
            if "recall_at_10" in sub:
                sub["recall_at_10"] = pd.to_numeric(sub["recall_at_10"], errors="coerce")
                best_recall = sub.dropna(subset=["recall_at_10"]).sort_values("recall_at_10", ascending=False).head(1)
                if not best_recall.empty:
                    best = best_recall.iloc[0]
                    rows.append(
                        {
                            "domain": "coi_vector_index",
                            "task": f"{split}_best_hnsw_recall_at_10_vs_exact",
                            "best_method": f"{best['method']}_{best['seed']}_{best['retrieval_mode']}",
                            "value": best["recall_at_10"],
                            "unit": "fraction",
                            "interpretation": "HNSW recall against exact vector top-10, not direct biological accuracy",
                        }
                    )
    return rows


def build_next_actions() -> list[dict[str, Any]]:
    placement = read_csv(SOURCE / "placement_rank_diagnostics_summary.csv")
    apples_like = read_csv(SOURCE / "apples_like_distance_placement_summary.csv")
    evidence = read_csv(SOURCE / "merged_edna_evidence_arm_status.csv")
    ann_manifest = SOURCE / "ann_vector_retrieval_manifest.json"
    controlled_vector = SOURCE / "controlled_vector_speed_benchmark.csv"
    bootstrap = PAPER1 / "rank_adaptive_calibration" / "missing_reference_aware_policy_bootstrap.csv"
    pdistance_calibration = PAPER1 / "pipeline_calibration" / "pipeline_mode_policy_summary.csv"
    strict = read_csv(SOURCE / "strict_missing_reference_summary.csv")
    edna_decomposition = SOURCE / "edna_evidence_decomposition_matrix.csv"
    edna_best_by_rank = SOURCE / "edna_evidence_best_by_rank.csv"
    edna_no_call = SOURCE / "edna_rank_no_call_operating_points.csv"
    edna_independent_no_call = PAPER1 / "global_edna_independent_rank_calibration" / "global_edna_independent_rank_calibration_summary.csv"
    strict_complete = 0
    strict_total = 0
    if not strict.empty and "status" in strict:
        strict_total = len(strict)
        strict_complete = int((strict["status"].astype(str) == "complete").sum())
    actions = [
        {
            "priority": 1,
            "task": "finish_remaining_epa_ng_splits",
            "why": "EPA-ng placement comparators are now available for Eval C, seen-test, and unseen-genera.",
            "blocked_by": "",
            "status": "in_progress" if len(placement) < 3 else "complete",
        },
        {
            "priority": 2,
            "task": "official_apples_or_documented_apples_equivalent",
            "why": "A labelled local APPLES-like distance-placement diagnostic now exists; official APPLES is only needed for an official APPLES reproduction claim.",
            "blocked_by": "official APPLES install/run, or a methods appendix documenting the local p-distance objective as APPLES-like only",
            "status": "partial" if not apples_like.empty else "pending",
        },
        {
            "priority": 3,
            "task": "implement_fernando_style_pcp_or_tree_error",
            "why": "Placed-clade containment, LWR/rank-backoff, tree-distance, sister-clade, and simulated-placement-tree PCP diagnostics now exist; exact Fernando comparability still needs matched backbone-completeness sweeps.",
            "blocked_by": "need to run generated 99/80/60/40/20% random and family-stratified completeness sweeps on Linux/Vast",
            "status": "partial",
        },
        {
            "priority": 4,
            "task": "controlled_vector_index_benchmark",
            "why": "ANN HNSW benchmark now exists locally and a controlled Vast HNSW timing table exists for CNN seed1206 Eval C.",
            "blocked_by": "larger-scale reference stress test if making deployment-scale speed claims",
            "status": "complete" if controlled_vector.exists() else ("partial" if ann_manifest.exists() else "pending"),
        },
        {
            "priority": 5,
            "task": "calibrate_pdistance_reranked_pipeline",
            "why": "Train-reference p-distance reranking now has seen-test-derived thresholds evaluated on Eval C and unseen-genera.",
            "blocked_by": "" if pdistance_calibration.exists() else "need reranked score features and thresholds learned on seen-test then evaluated on Eval C/unseen-genera",
            "status": "complete" if pdistance_calibration.exists() else "pending",
        },
        {
            "priority": 6,
            "task": "independent_rank_no_call_calibration",
            "why": "Seen-test-to-heldout transfer, candidate consensus thresholds, and bootstrap intervals now exist.",
            "blocked_by": (
                "strict tree-pruned input packs are ready; run experiments/paper1_phylo_calibrated_assignment/runs/08_vast_strict_missing_reference_cnn.sh"
                if strict_total and strict_complete == 0
                else "need final operating-point choice and manuscript-facing operating-point table"
            ),
            "status": "complete" if strict_total and strict_complete == strict_total and bootstrap.exists() else ("partial" if bootstrap.exists() or strict_total else "pending"),
        },
        {
            "priority": 7,
            "task": "complete_edna_evidence_decomposition",
            "why": "Stalder-adjacent claim needs sequence/tree/ecology arms separated cleanly.",
            "blocked_by": "run scripts/edna/build_edna_evidence_decomposition.py" if not edna_decomposition.exists() else "",
            "status": "complete" if edna_decomposition.exists() and edna_best_by_rank.exists() else "partial",
        },
        {
            "priority": 8,
            "task": "independent_edna_rank_no_call_calibration",
            "why": "Site-heldout Global_eDNA rank/no-call calibration now exists; current results show only modest family/order operating points and no high-accuracy species/genus thresholds.",
            "blocked_by": "need a stronger pre-registered Eco-Phylo posterior before manuscript-level eDNA rank/no-call claims" if edna_independent_no_call.exists() else "need a held-out calibration/evaluation split or a pre-registered threshold policy before manuscript-level rank/no-call claims",
            "status": "complete" if edna_independent_no_call.exists() else ("partial" if edna_no_call.exists() else "pending"),
        },
    ]
    if not evidence.empty:
        required = evidence[evidence["evidence_arm"].isin(["geography_or_range_only", "cooccurrence_only"])]
        if not required.empty and set(required["status"].astype(str)) == {"available"}:
            for action in actions:
                if action["task"] == "complete_edna_evidence_decomposition":
                    action["status"] = "complete" if edna_decomposition.exists() and edna_best_by_rank.exists() else "partial"
                    action["blocked_by"] = "" if action["status"] == "complete" else action["blocked_by"]
                    action["why"] = "Sequence/tree, geography-only, co-occurrence-only, and learned co-occurrence arms are now separated in manuscript-facing eDNA source tables."
    return actions


def main() -> None:
    logger = ProgressLogger(default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.log("Loading source tables and building pipeline status rows")
    component_rows = status_rows()
    logger.log("Building COI method benchmark rows")
    coi_rows = build_coi_method_rows()
    logger.log("Building placement benchmark rows")
    placement_rows = build_placement_rows()
    logger.log("Building vector-index benchmark rows")
    vector_rows = build_vector_index_rows()
    logger.log("Building 12S/eDNA method benchmark rows")
    edna_rows = build_edna_rows()
    logger.log("Building best-by-task summary rows")
    best_rows = build_best_task_rows(coi_rows, edna_rows, placement_rows, vector_rows)
    logger.log("Building next-action rows")
    action_rows = build_next_actions()

    logger.log("Writing pipeline_component_status.csv")
    write_csv(
        OUT_DIR / "pipeline_component_status.csv",
        component_rows,
        ["pipeline_component", "status", "source_tables", "notes"],
    )
    logger.log(f"Writing pipeline_coi_method_benchmark.csv with {len(coi_rows)} rows")
    write_csv(
        OUT_DIR / "pipeline_coi_method_benchmark.csv",
        coi_rows,
        [
            "method",
            "method_family",
            "split",
            "species_top1_pct",
            "species_top5_pct",
            "species_top10_pct",
            "genus_top1_pct",
            "genus_top5_pct",
            "genus_top10_pct",
            "family_top1_pct",
            "family_top5_pct",
            "family_top10_pct",
            "order_top1_pct",
            "order_top5_pct",
            "order_top10_pct",
            "tree_zero_shot_reference_pearson",
            "tree_zero_shot_reference_spearman",
            "hide_true_species_genus_top10_pct",
            "hide_true_species_family_top10_pct",
            "hide_true_species_order_top10_pct",
            "rank_policy_target_precision",
            "rank_policy_coverage_pct",
            "rank_policy_observed_precision_pct",
            "rank_policy_species_calls",
            "rank_policy_genus_calls",
            "rank_policy_family_calls",
            "rank_policy_order_calls",
            "rank_policy_no_calls",
            "prospective_rank_policy_calibration_split",
            "prospective_rank_policy_target_precision",
            "prospective_rank_policy_coverage_pct",
            "prospective_rank_policy_observed_precision_pct",
            "prospective_rank_policy_species_calls",
            "prospective_rank_policy_genus_calls",
            "prospective_rank_policy_family_calls",
            "prospective_rank_policy_order_calls",
            "prospective_rank_policy_no_calls",
            "missing_ref_policy_target_precision",
            "missing_ref_policy_coverage_pct",
            "missing_ref_policy_observed_precision_pct",
            "missing_ref_policy_precision_ci95_low_pct",
            "missing_ref_policy_precision_ci95_high_pct",
            "missing_ref_policy_bootstrap_precision_ci95_low_pct",
            "missing_ref_policy_bootstrap_precision_ci95_high_pct",
            "missing_ref_policy_false_species_call_rate_pct",
            "missing_ref_policy_false_species_call_rate_ci95_low_pct",
            "missing_ref_policy_false_species_call_rate_ci95_high_pct",
            "missing_ref_policy_species_calls",
            "missing_ref_policy_genus_calls",
            "missing_ref_policy_family_calls",
            "missing_ref_policy_order_calls",
            "missing_ref_policy_no_calls",
            "runtime_mode",
            "runtime_ms_per_query",
        ],
    )
    logger.log(f"Writing pipeline_placement_benchmark.csv with {len(placement_rows)} rows")
    write_csv(
        OUT_DIR / "pipeline_placement_benchmark.csv",
        placement_rows,
        [
            "split",
            "method",
            "species_in_placed_clade_pct",
            "genus_in_placed_clade_pct",
            "family_in_placed_clade_pct",
            "order_in_placed_clade_pct",
            "placement_min_tree_distance_median",
            "placement_min_tree_distance_mean",
            "placement_excess_tree_distance_median",
            "placement_excess_tree_distance_mean",
            "nearest_reference_match_pct",
            "pcp_like_exact_sister_match_pct",
            "pcp_like_any_sister_overlap_pct",
            "pcp_like_best_jaccard_median",
            "pcp_like_lwr_ge_0p9_exact_sister_match_pct",
            "pcp_like_lwr_ge_0p9_any_sister_overlap_pct",
            "simulated_species_pcp_exact_pct",
            "simulated_species_pcp_overlap_pct",
            "simulated_species_pcp_jaccard_median",
            "simulated_species_pcp_lwr_ge_0p9_exact_pct",
            "simulated_species_pcp_lwr_ge_0p9_overlap_pct",
            "simulated_species_pcp_represented_sister_rate_pct",
            "simulated_sequence_pcp_exact_pct",
            "simulated_sequence_pcp_overlap_pct",
            "simulated_sequence_pcp_jaccard_median",
            "simulated_sequence_pcp_lwr_ge_0p9_exact_pct",
            "simulated_sequence_pcp_lwr_ge_0p9_overlap_pct",
            "simulated_sequence_pcp_represented_sister_rate_pct",
            "most_specific_rank_species_pct",
            "most_specific_rank_genus_pct",
            "most_specific_rank_family_pct",
            "most_specific_rank_order_pct",
            "most_specific_rank_none_pct",
            "n_queries_scored",
            "missing_query_metadata",
            "caution",
            "source_file",
        ],
    )
    logger.log(f"Writing pipeline_vector_index_benchmark.csv with {len(vector_rows)} rows")
    write_csv(
        OUT_DIR / "pipeline_vector_index_benchmark.csv",
        vector_rows,
        [
            "method",
            "seed",
            "split",
            "retrieval_mode",
            "n_queries",
            "n_candidates",
            "embedding_dim",
            "index_mb",
            "index_build_seconds",
            "search_seconds",
            "queries_per_second",
            "milliseconds_per_query",
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "recall_at_50",
            "top1_match_rate",
            "source_file",
        ],
    )
    logger.log(f"Writing pipeline_edna_method_benchmark.csv with {len(edna_rows)} rows")
    write_csv(
        OUT_DIR / "pipeline_edna_method_benchmark.csv",
        edna_rows,
        [
            "run_name",
            "dataset",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "benchmark_level",
            "species_top10_pct",
            "genus_top10_pct",
            "family_top10_pct",
            "order_top10_pct",
        ],
    )
    logger.log(f"Writing pipeline_best_by_task.csv with {len(best_rows)} rows")
    write_csv(
        OUT_DIR / "pipeline_best_by_task.csv",
        best_rows,
        ["domain", "task", "best_method", "value", "unit", "interpretation"],
    )
    logger.log(f"Writing pipeline_next_actions.csv with {len(action_rows)} rows")
    write_csv(
        OUT_DIR / "pipeline_next_actions.csv",
        action_rows,
        ["priority", "task", "why", "blocked_by", "status"],
    )

    manifest = {
        "generated_by": rel(Path(__file__)),
        "output_dir": rel(OUT_DIR),
        "files": {
            "pipeline_component_status.csv": len(component_rows),
            "pipeline_coi_method_benchmark.csv": len(coi_rows),
            "pipeline_placement_benchmark.csv": len(placement_rows),
            "pipeline_vector_index_benchmark.csv": len(vector_rows),
            "pipeline_edna_method_benchmark.csv": len(edna_rows),
            "pipeline_best_by_task.csv": len(best_rows),
            "pipeline_next_actions.csv": len(action_rows),
        },
        "notes": [
            "This is a benchmark ledger over existing source tables, not a new model result.",
            "Placement rows currently use placed-clade containment, tree-distance-to-placed-clade, sister-clade, and simulated-placement-tree PCP diagnostics. They should not be reported as exact Fernando PCP until matched backbone-completeness sweeps are run.",
            "Rank/no-call policy columns include same-split diagnostics, seen-test-to-heldout transfer, missing-reference consensus thresholds, and bootstrap intervals where available.",
        ],
    }
    with (OUT_DIR / "pipeline_benchmark_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.log("Wrote pipeline_benchmark_manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
