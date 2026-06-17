#!/usr/bin/env python3
"""Build merged Paper 1 source tables for the 12S/eDNA work package.

These tables make the former Paper 2 evidence auditable inside the merged
Paper 1 manuscript. They summarize marker resolvability, 12S zero-shot model
metrics, and Global_eDNA validation without making any single architecture the
center of the claim.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
EXACT_RESOLVABILITY = ROOT / "results" / "edna" / "resolvability"
REMOTE_2026_05_30 = ROOT / "results" / "remote_runs" / "2026-05-30" / "rtx_pro_6000"
REMOTE_2026_06_02 = ROOT / "results" / "remote_runs" / "2026-06-02" / "rtx_pro_6000"
NEAR_EXACT_RESOLVABILITY = REMOTE_2026_05_30 / "resolvability_near_exact"
TAXDNA_SSM = REMOTE_2026_05_30 / "taxdna_ssm"
TAXDNA_SSM_ROOTS = [
    TAXDNA_SSM,
    REMOTE_2026_06_02 / "taxdna_ssm",
]
GLOBAL_TROPICAL_VALIDATION = ROOT / "results" / "edna" / "global_tropical_validation"
GLOBAL_EDNA_CALIBRATION = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "global_edna_rank_calibration"

RANKS = ("species", "genus", "family", "order")
TOP_KS = (1, 5, 10)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> float | str:
    if value in ("", None):
        return ""
    return 100.0 * float(value)


def dataset_from_resolvability_dir(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        return name
    return name[: -len(suffix)]


def classify_taxdna_run(run_name: str) -> dict[str, Any]:
    dataset = "multisource_teleo" if "multisource_teleo" in run_name else "multisource"
    if "_cnn" in run_name:
        encoder = "cnn"
    elif "_ssm" in run_name:
        encoder = "ssm"
    else:
        encoder = "unknown"

    seed_match = re.search(r"_seed(\d+)", run_name)
    seed = seed_match.group(1) if seed_match else ""
    context = "zero_shot"
    prior_source = ""
    prior_weight = ""
    if "sequence_only_validation" in run_name:
        context = "sequence_only"
    elif "learned_cooccurrence" in run_name:
        context = "learned_cooccurrence"
        prior_source = "fishglob_public_50k" if "fishglob" in run_name else "rls_obis"
        weight_match = re.search(r"_w(\d{3})", run_name)
        if weight_match:
            prior_weight = int(weight_match.group(1)) / 100.0

    return {
        "run_name": run_name,
        "dataset": dataset,
        "encoder": encoder,
        "seed": seed,
        "context": context,
        "prior_source": prior_source,
        "prior_weight": prior_weight,
    }


def classify_global_calibration_method(method_name: str) -> dict[str, Any]:
    encoder = "ssm" if method_name.startswith("ssm_") else "cnn" if method_name.startswith("cnn_") else "unknown"
    context = "sequence_only" if "sequence_only" in method_name else "learned_cooccurrence"
    prior_source = ""
    if "rls_obis" in method_name:
        prior_source = "rls_obis"
    elif "fishglob" in method_name:
        prior_source = "fishglob_public_50k"
    weight_match = re.search(r"_w(\d{3})_", method_name)
    prior_weight = int(weight_match.group(1)) / 100.0 if weight_match else ""
    return {
        "method": method_name,
        "dataset": "multisource_teleo",
        "encoder": encoder,
        "context": context,
        "prior_source": prior_source,
        "prior_weight": prior_weight,
    }


def parse_weight(run_name: str) -> str | float:
    weight_match = re.search(r"_w(\d{3})", run_name)
    if weight_match:
        return int(weight_match.group(1)) / 100.0
    weight_match = re.search(r"_w(\d+)", run_name)
    if weight_match:
        return float(weight_match.group(1))
    return ""


def classify_global_tropical_run(run_name: str) -> dict[str, Any]:
    dataset = "multisource_teleo"
    seed_match = re.search(r"_seed(\d+)", run_name)
    seed = seed_match.group(1) if seed_match else ""

    if run_name.startswith("blast"):
        encoder = "blast"
    elif run_name.startswith("rls_prior_only") or run_name.startswith("obis_prior_only"):
        encoder = "none"
    elif "sample_cooccurrence_prior_only" in run_name:
        encoder = "none"
    else:
        encoder = "ssm"

    context = "sequence_only"
    prior_source = ""
    prior_weight: str | float = ""
    if run_name.startswith("rls_prior_only"):
        context = "geography_prior_only"
        prior_source = "rls"
    elif run_name.startswith("obis_prior_only"):
        context = "geography_prior_only"
        prior_source = "obis_occurrence"
    elif "sample_cooccurrence_prior_only" in run_name:
        context = "cooccurrence_prior_only"
        prior_source = "same_sample_other_asvs"
    elif "_cooccurrence_w" in run_name:
        context = "sequence_plus_same_sample_cooccurrence"
        prior_source = "same_sample_other_asvs"
        prior_weight = parse_weight(run_name)
    elif "_rls_obis" in run_name:
        context = "sequence_plus_geography"
        prior_source = "rls_obis"
        prior_weight = parse_weight(run_name)
    elif "_obis" in run_name:
        context = "sequence_plus_geography"
        prior_source = "obis_occurrence"
        prior_weight = parse_weight(run_name)
    elif "_rls_geo" in run_name:
        context = "sequence_plus_geography"
        prior_source = "rls"
        prior_weight = parse_weight(run_name)

    return {
        "run_name": run_name,
        "dataset": dataset,
        "encoder": encoder,
        "seed": seed,
        "context": context,
        "prior_source": prior_source,
        "prior_weight": prior_weight,
    }


def build_resolvability_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    overview = EXACT_RESOLVABILITY / "resolvability_overview.csv"
    if overview.exists():
        for row in read_csv_dicts(overview):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "mode": "exact_identity",
                    "identity": 1.0,
                    "species_with_sequences": row.get("species_with_sequences", ""),
                    "sequence_records": row.get("sequence_records_used", ""),
                    "cluster_count": row.get("exact_cluster_count", ""),
                    "query_count": row.get("query_count", ""),
                    "query_cluster_has_reference_rate_pct": pct(row.get("query_reference_exact_cluster_found_rate")),
                    "species_oracle_supported_rate_pct": pct(row.get("query_species_oracle_supported_rate")),
                    "genus_oracle_supported_rate_pct": pct(row.get("query_genus_oracle_supported_rate")),
                    "family_oracle_supported_rate_pct": pct(row.get("query_family_oracle_supported_rate")),
                    "order_oracle_supported_rate_pct": pct(row.get("query_order_oracle_supported_rate")),
                    "source": rel(overview),
                }
            )

    if NEAR_EXACT_RESOLVABILITY.exists():
        for path in sorted(NEAR_EXACT_RESOLVABILITY.glob("*_near_exact_acgt/near_exact_resolvability_summary.csv")):
            dataset = dataset_from_resolvability_dir(path.parent, "_near_exact_acgt")
            for row in read_csv_dicts(path):
                rows.append(
                    {
                        "dataset": dataset,
                        "mode": "near_exact_identity",
                        "identity": row.get("identity", ""),
                        "species_with_sequences": "",
                        "sequence_records": row.get("record_count", ""),
                        "cluster_count": row.get("cluster_count", ""),
                        "query_count": row.get("query_count", ""),
                        "query_cluster_has_reference_rate_pct": pct(row.get("query_cluster_has_reference_rate")),
                        "species_oracle_supported_rate_pct": pct(row.get("query_species_oracle_supported_rate")),
                        "genus_oracle_supported_rate_pct": pct(row.get("query_genus_oracle_supported_rate")),
                        "family_oracle_supported_rate_pct": pct(row.get("query_family_oracle_supported_rate")),
                        "order_oracle_supported_rate_pct": pct(row.get("query_order_oracle_supported_rate")),
                        "source": rel(path),
                    }
                )

    return rows


def build_zero_shot_model_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in TAXDNA_SSM_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.glob("multisource*/zero_shot_metrics/zero_shot_candidate_metrics.json")):
            run = classify_taxdna_run(path.parents[1].name)
            payload = read_json(path)
            for rank in RANKS:
                metrics = payload.get("metrics", {}).get(rank, {})
                rows.append(
                    {
                        **run,
                        "rank": rank,
                        "query_count": payload.get("query_count", ""),
                        "candidate_count": payload.get("candidate_count", ""),
                        "eligible_queries": metrics.get("eligible_queries", ""),
                        "top1_pct": pct(metrics.get("top1")),
                        "top5_pct": pct(metrics.get("top5")),
                        "top10_pct": pct(metrics.get("top10")),
                        "mean_first_hit_rank": metrics.get("mean_first_hit_rank", ""),
                        "source": rel(path),
                    }
                )
    return rows


def build_global_edna_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    asv_rows: list[dict[str, Any]] = []

    taxdna_paths: list[tuple[Path, dict[str, Any]]] = []
    for root in TAXDNA_SSM_ROOTS:
        if root.exists():
            paths = sorted(root.glob("global_edna*/**/global_edna_validation_metrics.json"))
            paths.extend(sorted(root.glob("global_edna*/global_edna_validation_metrics.json")))
            for path in sorted(set(paths)):
                run_name = path.parents[1].name if path.parent.name == "global_edna_validation" else path.parent.name
                taxdna_paths.append((path, classify_taxdna_run(run_name)))

    global_paths: list[tuple[Path, dict[str, Any]]] = []
    if GLOBAL_TROPICAL_VALIDATION.exists():
        paths = sorted(GLOBAL_TROPICAL_VALIDATION.glob("*/global_edna_validation/global_edna_validation_metrics.json"))
        paths.extend(sorted(GLOBAL_TROPICAL_VALIDATION.glob("*/global_edna_validation_metrics.json")))
        for path in sorted(set(paths)):
            run_name = path.parents[1].name if path.parent.name == "global_edna_validation" else path.parent.name
            global_paths.append((path, classify_global_tropical_run(run_name)))

    for path, run in sorted(taxdna_paths + global_paths, key=lambda item: str(item[0])):
        payload = read_json(path)

        asv_metrics = payload.get("asv_metrics", {})
        for rank in RANKS:
            metrics = asv_metrics.get(rank, {})
            asv_rows.append(
                {
                    **run,
                    "rank": rank,
                    "assigned_rows": metrics.get("assigned_rows", ""),
                    "top1_pct": pct(metrics.get("top1")),
                    "top5_pct": pct(metrics.get("top5")),
                    "top10_pct": pct(metrics.get("top10")),
                    "prediction_rows": payload.get("prediction_rows", ""),
                    "sample_query_rows": payload.get("sample_query_rows", ""),
                    "sample_count": payload.get("sample_count", ""),
                    "source": rel(path),
                }
            )

        summary = payload.get("sample_metric_summary", {})
        for rank in RANKS:
            for top_k in TOP_KS:
                for metric in ("precision", "recall", "jaccard", "pred_richness"):
                    key = f"{rank}_top{top_k}_{metric}"
                    stats = summary.get(key)
                    if not isinstance(stats, dict):
                        continue
                    sample_rows.append(
                        {
                            **run,
                            "rank": rank,
                            "top_k": top_k,
                            "metric": metric,
                            "mean": stats.get("mean", ""),
                            "median": stats.get("median", ""),
                            "min": stats.get("min", ""),
                            "max": stats.get("max", ""),
                            "sample_count": payload.get("sample_count", ""),
                            "prediction_rows": payload.get("prediction_rows", ""),
                            "sample_query_rows": payload.get("sample_query_rows", ""),
                            "source": rel(path),
                        }
                    )

    return sample_rows, asv_rows


def build_evidence_status_rows(
    resolvability_rows: list[dict[str, Any]],
    zero_shot_rows: list[dict[str, Any]],
    global_sample_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    contexts = {str(row.get("context", "")) for row in global_sample_rows}
    prior_sources = {str(row.get("prior_source", "")) for row in global_sample_rows}
    return [
        {
            "evidence_arm": "12s_marker_resolvability",
            "status": "available" if resolvability_rows else "missing",
            "current_table": "merged_12s_resolvability_summary.csv",
            "scope": "exact and near-exact sequence identity oracle support",
            "remaining_gap": "add marker-window/primer-region sensitivity if final dataset supports it",
        },
        {
            "evidence_arm": "12s_sequence_tree_encoder_zero_shot",
            "status": "available" if zero_shot_rows else "missing",
            "current_table": "merged_12s_zero_shot_model_metrics.csv",
            "scope": "SSM/CNN candidate retrieval against 12S candidate sets",
            "remaining_gap": "add more encoder families only if they answer a specific comparison question",
        },
        {
            "evidence_arm": "global_edna_sequence_tree_encoder_only",
            "status": "available" if "sequence_only" in contexts else "missing",
            "current_table": "merged_global_edna_asv_metrics.csv; merged_global_edna_sample_metrics.csv",
            "scope": "Global_eDNA validation from sequence/tree encoder predictions only",
            "remaining_gap": "separate ASV and sample/site figures with calibrated rank/no-call",
        },
        {
            "evidence_arm": "global_edna_sequence_tree_plus_rls_obis_cooccurrence",
            "status": "available" if "rls_obis" in prior_sources else "missing",
            "current_table": "merged_global_edna_asv_metrics.csv; merged_global_edna_sample_metrics.csv",
            "scope": "learned co-occurrence reranking from RLS/OBIS-derived priors",
            "remaining_gap": "ablate co-occurrence weight and check where it helps versus degrades",
        },
        {
            "evidence_arm": "global_edna_sequence_tree_plus_fishglob_cooccurrence",
            "status": "available" if "fishglob_public_50k" in prior_sources else "missing",
            "current_table": "merged_global_edna_asv_metrics.csv; merged_global_edna_sample_metrics.csv",
            "scope": "learned co-occurrence reranking from public FISHGLOB reconstruction",
            "remaining_gap": "document reconstruction limits and compare to RLS/OBIS priors",
        },
        {
            "evidence_arm": "geography_or_range_only",
            "status": "available" if "geography_prior_only" in contexts else "missing",
            "current_table": "merged_global_edna_asv_metrics.csv; merged_global_edna_sample_metrics.csv",
            "scope": "site/location prior without sequence evidence",
            "remaining_gap": "compare RLS and OBIS prior-only rows against sequence-only and sequence+prior arms",
        },
        {
            "evidence_arm": "cooccurrence_only",
            "status": "available" if "cooccurrence_prior_only" in contexts else "missing",
            "current_table": "merged_global_edna_asv_metrics.csv; merged_global_edna_sample_metrics.csv",
            "scope": "same-sample community prior for the current query without its own sequence score",
            "remaining_gap": "label clearly: other ASVs' sequence predictions define the sample context",
        },
        {
            "evidence_arm": "sequence_tree_ecology_rank_no_call",
            "status": "partial",
            "current_table": "COI rank_adaptive_calibration tables; merged_global_edna_calibration_curves.csv",
            "scope": "calibrated species/genus/family/order/no-call under marker ambiguity",
            "remaining_gap": "replace diagnostic same-data eDNA curves with independent calibration and rank/no-call policy",
        },
    ]


def build_global_edna_calibration_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not GLOBAL_EDNA_CALIBRATION.exists():
        return rows
    for path in sorted(GLOBAL_EDNA_CALIBRATION.glob("*/calibration_curve.csv")):
        method = classify_global_calibration_method(path.parent.name)
        for curve_row in read_csv_dicts(path):
            for rank in RANKS:
                rows.append(
                    {
                        **method,
                        "rank": rank,
                        "threshold": curve_row.get("threshold", ""),
                        "n_query": curve_row.get("n_query", ""),
                        "n_assigned": curve_row.get("n_assigned", ""),
                        "assignment_rate_pct": pct(curve_row.get("assignment_rate")),
                        "rank_n": curve_row.get(f"{rank}_n", ""),
                        "rank_accuracy_pct": pct(curve_row.get(f"{rank}_accuracy")),
                        "source": rel(path),
                    }
                )
    return rows


def main() -> None:
    logger = ProgressLogger(default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.log("Building 12S exact and near-exact resolvability rows")
    resolvability_rows = build_resolvability_rows()
    logger.log(f"Built {len(resolvability_rows)} resolvability rows")
    logger.log("Building 12S zero-shot model rows")
    zero_shot_rows = build_zero_shot_model_rows()
    logger.log(f"Built {len(zero_shot_rows)} zero-shot model rows")
    logger.log("Building Global_eDNA sample and ASV rows")
    global_sample_rows, global_asv_rows = build_global_edna_rows()
    logger.log(f"Built {len(global_sample_rows)} sample rows and {len(global_asv_rows)} ASV rows")
    logger.log("Building evidence-arm status rows")
    evidence_status_rows = build_evidence_status_rows(
        resolvability_rows,
        zero_shot_rows,
        global_sample_rows,
    )
    logger.log("Building Global_eDNA calibration curve rows")
    global_calibration_rows = build_global_edna_calibration_rows()
    logger.log(f"Built {len(global_calibration_rows)} calibration rows")

    logger.log("Writing merged_12s_resolvability_summary.csv")
    write_csv(
        OUT_DIR / "merged_12s_resolvability_summary.csv",
        resolvability_rows,
        [
            "dataset",
            "mode",
            "identity",
            "species_with_sequences",
            "sequence_records",
            "cluster_count",
            "query_count",
            "query_cluster_has_reference_rate_pct",
            "species_oracle_supported_rate_pct",
            "genus_oracle_supported_rate_pct",
            "family_oracle_supported_rate_pct",
            "order_oracle_supported_rate_pct",
            "source",
        ],
    )

    logger.log("Writing merged_12s_zero_shot_model_metrics.csv")
    write_csv(
        OUT_DIR / "merged_12s_zero_shot_model_metrics.csv",
        zero_shot_rows,
        [
            "run_name",
            "dataset",
            "encoder",
            "seed",
            "context",
            "prior_source",
            "prior_weight",
            "rank",
            "query_count",
            "candidate_count",
            "eligible_queries",
            "top1_pct",
            "top5_pct",
            "top10_pct",
            "mean_first_hit_rank",
            "source",
        ],
    )

    logger.log("Writing merged_global_edna_sample_metrics.csv")
    write_csv(
        OUT_DIR / "merged_global_edna_sample_metrics.csv",
        global_sample_rows,
        [
            "run_name",
            "dataset",
            "encoder",
            "seed",
            "context",
            "prior_source",
            "prior_weight",
            "rank",
            "top_k",
            "metric",
            "mean",
            "median",
            "min",
            "max",
            "sample_count",
            "prediction_rows",
            "sample_query_rows",
            "source",
        ],
    )

    logger.log("Writing merged_global_edna_asv_metrics.csv")
    write_csv(
        OUT_DIR / "merged_global_edna_asv_metrics.csv",
        global_asv_rows,
        [
            "run_name",
            "dataset",
            "encoder",
            "seed",
            "context",
            "prior_source",
            "prior_weight",
            "rank",
            "assigned_rows",
            "top1_pct",
            "top5_pct",
            "top10_pct",
            "prediction_rows",
            "sample_query_rows",
            "sample_count",
            "source",
        ],
    )

    logger.log("Writing merged_edna_evidence_arm_status.csv")
    write_csv(
        OUT_DIR / "merged_edna_evidence_arm_status.csv",
        evidence_status_rows,
        [
            "evidence_arm",
            "status",
            "current_table",
            "scope",
            "remaining_gap",
        ],
    )

    logger.log("Writing merged_global_edna_calibration_curves.csv")
    write_csv(
        OUT_DIR / "merged_global_edna_calibration_curves.csv",
        global_calibration_rows,
        [
            "method",
            "dataset",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "rank",
            "threshold",
            "n_query",
            "n_assigned",
            "assignment_rate_pct",
            "rank_n",
            "rank_accuracy_pct",
            "source",
        ],
    )

    manifest = {
        "generated_source_tables": {
            "merged_12s_resolvability_summary.csv": len(resolvability_rows),
            "merged_12s_zero_shot_model_metrics.csv": len(zero_shot_rows),
            "merged_global_edna_sample_metrics.csv": len(global_sample_rows),
            "merged_global_edna_asv_metrics.csv": len(global_asv_rows),
            "merged_edna_evidence_arm_status.csv": len(evidence_status_rows),
            "merged_global_edna_calibration_curves.csv": len(global_calibration_rows),
        },
        "inputs": {
            "exact_resolvability": rel(EXACT_RESOLVABILITY),
            "near_exact_resolvability": rel(NEAR_EXACT_RESOLVABILITY),
            "taxdna_ssm_roots": [rel(path) for path in TAXDNA_SSM_ROOTS],
            "global_tropical_validation": rel(GLOBAL_TROPICAL_VALIDATION),
            "global_edna_calibration": rel(GLOBAL_EDNA_CALIBRATION),
        },
        "notes": [
            "12S resolvability rows are marker-information diagnostics, not model results.",
            "Global_eDNA ground truth is the published Global_eDNA table assignment, not independent visual census.",
            "Rows are intended for the merged Paper 1 evidence ledger.",
        ],
    }
    with (OUT_DIR / "merged_edna_source_table_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.log("Wrote merged_edna_source_table_manifest.json")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
