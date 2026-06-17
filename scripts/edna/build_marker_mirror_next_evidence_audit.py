#!/usr/bin/env python3
"""Audit genuinely new evidence sources for MarkerMirror family/genus work.

This script does not train a model or rerun threshold repair. It summarizes
which non-threshold evidence sources already exist locally, which join keys they
support, and which gaps must be filled before another family/genus attempt is
scientifically different from Exp 119/121.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


SCRIPT_NAME = "build_marker_mirror_next_evidence_audit"
ROOT = Path(".")
DEFAULT_SOURCE_TABLE_DIR = (
    ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
)
DEFAULT_TEXT_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "manuscript_assets"
    / "marker_mirror"
    / "text"
)


def read_csv(path: Path, logger: ProgressLogger, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        logger.log(f"missing optional input: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, **kwargs)
    logger.log(f"loaded {path} rows={len(df)} cols={len(df.columns)}")
    return df


def weighted_mean_pct(df: pd.DataFrame, value_col: str, weight_col: str = "n_queries") -> float:
    if df.empty or value_col not in df or weight_col not in df:
        return float("nan")
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    values = pd.to_numeric(df[value_col], errors="coerce")
    denom = weights.sum()
    if denom <= 0:
        return float("nan")
    return float((values.fillna(0.0) * weights).sum() / denom)


def bool_weighted_pct(df: pd.DataFrame, value_col: str, weight_col: str = "n_queries") -> float:
    if df.empty or value_col not in df or weight_col not in df:
        return float("nan")
    work = df.copy()
    work[value_col] = work[value_col].astype(bool).astype(float) * 100.0
    return weighted_mean_pct(work, value_col, weight_col)


def pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}%"


def top_weighted_label(
    df: pd.DataFrame, label_col: str, weight_col: str = "n_queries"
) -> str:
    if df.empty or label_col not in df or weight_col not in df:
        return "NA"
    grouped = (
        df.groupby(label_col, dropna=False)[weight_col]
        .sum()
        .sort_values(ascending=False)
    )
    if grouped.empty:
        return "NA"
    return str(grouped.index[0])


def top_species(df: pd.DataFrame, n: int = 5) -> str:
    if df.empty or "query_tree_label" not in df or "curation_priority_score" not in df:
        return "NA"
    sub = df.sort_values("curation_priority_score", ascending=False).head(n)
    return "; ".join(
        f"{row.query_tree_label} ({row.curation_priority_score:g})"
        for row in sub.itertuples(index=False)
    )


def summarize_resolvability(resolvability: pd.DataFrame) -> str:
    if resolvability.empty:
        return "No local marker-resolvability summary found."
    parts: list[str] = []
    for row in resolvability.itertuples(index=False):
        marker = getattr(row, "marker", "marker")
        identity = getattr(row, "identity", None)
        backend = getattr(row, "ambiguity_backend", "unknown_backend")
        species = getattr(row, "species_oracle_supported_rate_pct", float("nan"))
        genus = getattr(row, "genus_oracle_supported_rate_pct", float("nan"))
        family = getattr(row, "family_oracle_supported_rate_pct", float("nan"))
        parts.append(
            f"{marker} identity={identity:g} backend={backend}: "
            f"species/genus/family support {pct(species)}/{pct(genus)}/{pct(family)}"
        )
    return " | ".join(parts)


def best_context_metric(metrics: pd.DataFrame, keyword: str) -> str:
    if metrics.empty:
        return "No merged eDNA metric table found."
    text_cols = ["run_name", "encoder", "context", "prior_source"]
    haystack = metrics[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
    sub = metrics[
        haystack.str.contains(keyword.lower())
        & metrics["metric"].astype(str).isin(["jaccard", "precision", "recall"])
        & metrics["top_k"].isin([1, 5, 10])
    ].copy()
    if sub.empty:
        return f"No local {keyword} rows found in merged eDNA metrics."
    priority = {"jaccard": 0, "precision": 1, "recall": 2}
    sub["metric_priority"] = sub["metric"].map(priority).fillna(9)
    sub = sub.sort_values(["metric_priority", "mean"], ascending=[True, False])
    row = sub.iloc[0]
    return (
        f"best {row.metric}={float(row['mean']):.3f} for "
        f"{row.encoder}/{row.context}/{row.prior_source}, rank={row['rank']}, "
        f"top_k={int(row.top_k)}"
    )


def build_lineage_rows(curation: pd.DataFrame, lineage_rank: str) -> pd.DataFrame:
    lineage_col = f"query_{lineage_rank}"
    if curation.empty or lineage_col not in curation:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for lineage, group in curation.groupby(lineage_col, dropna=False):
        n_queries = int(group["n_queries"].sum())
        rows.append(
            {
                "lineage_rank": lineage_rank,
                "lineage": lineage,
                "species_group_count": int(group["query_tree_label"].nunique()),
                "query_count": n_queries,
                "query_pct": 100.0
                * n_queries
                / max(float(curation["n_queries"].sum()), 1.0),
                "species_present_12s_query_weighted_pct": bool_weighted_pct(
                    group, "query_species_present_in_12s_reference"
                ),
                "species_present_16s_query_weighted_pct": bool_weighted_pct(
                    group, "query_species_present_in_16s_reference"
                ),
                "marker_mirror_genus_hit_weighted_pct": weighted_mean_pct(
                    group, "marker_mirror_genus_hit_pct"
                ),
                "same_marker_genus_hit_weighted_pct": weighted_mean_pct(
                    group, "same_marker_genus_hit_pct"
                ),
                "union_genus_hit_weighted_pct": weighted_mean_pct(
                    group, "union_genus_hit_pct"
                ),
                "union_family_hit_weighted_pct": weighted_mean_pct(
                    group, "union_family_hit_pct"
                ),
                "union_order_hit_weighted_pct": weighted_mean_pct(
                    group, "union_order_hit_pct"
                ),
                "static_family_order_assigned_weighted_pct": weighted_mean_pct(
                    group, "static_family_order_assigned_pct"
                ),
                "dominant_reason": top_weighted_label(group, "primary_reason_mode"),
                "dominant_next_action": top_weighted_label(
                    group, "recommended_next_action_mode"
                ),
                "top_curation_species": top_species(group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["query_count", "union_family_hit_weighted_pct"], ascending=[False, False]
    )


def build_audit_rows(
    curation: pd.DataFrame,
    reason_summary: pd.DataFrame,
    resolvability: pd.DataFrame,
    rank_repair: pd.DataFrame,
    blast_support: pd.DataFrame,
    vsearch_support: pd.DataFrame,
    edna_metrics: pd.DataFrame,
) -> list[dict[str, str]]:
    total_queries = int(curation["n_queries"].sum()) if not curation.empty else 0
    present_16s = bool_weighted_pct(curation, "query_species_present_in_16s_reference")
    union_genus = weighted_mean_pct(curation, "union_genus_hit_pct")
    union_family = weighted_mean_pct(curation, "union_family_hit_pct")
    species_absent_note = (
        f"{pct(present_16s)} query-weighted 16S species coverage; "
        f"union genus/family support {pct(union_genus)}/{pct(union_family)} "
        f"over {total_queries} full-query 12S rows"
    )

    high_rank_reason = "NA"
    if not reason_summary.empty and "primary_reason_code" in reason_summary:
        sub = reason_summary[
            reason_summary["primary_reason_code"].eq("high_rank_union_support_genus")
        ]
        if not sub.empty:
            row = sub.iloc[0]
            high_rank_reason = (
                f"{int(row.n_queries)} rows ({float(row.query_pct):.1f}%) have "
                "genus-level union support without enabled family/genus calls"
            )

    rank_boundary = "NA"
    if not rank_repair.empty:
        pieces = []
        for row in rank_repair.itertuples(index=False):
            pieces.append(
                f"{row.rank}: stable={bool(row.stable_target99_available)}, "
                f"best target-met={pct(row.best_target_met_rate_pct, 0)}"
            )
        rank_boundary = "; ".join(pieces)

    blast_vsearch_signal = "NA"
    if not blast_support.empty and not vsearch_support.empty:
        b = blast_support[blast_support["candidate_source"].str.contains("union", na=False)]
        v = vsearch_support[vsearch_support["candidate_source"].str.contains("union", na=False)]
        if not b.empty and not v.empty:
            rb = b.iloc[0]
            rv = v.iloc[0]
            blast_vsearch_signal = (
                f"BLAST union genus/family/order "
                f"{pct(rb.genus_hit_pct)}/{pct(rb.family_hit_pct)}/{pct(rb.order_hit_pct)}; "
                f"VSEARCH union {pct(rv.genus_hit_pct)}/{pct(rv.family_hit_pct)}/{pct(rv.order_hit_pct)}"
            )

    return [
        {
            "evidence_source": "lineage_specific_reference_coverage",
            "current_status": "ready_as_diagnostic_feature",
            "existing_artifacts": "marker_mirror_union_reference_curation_priorities.csv; marker_mirror_union_reason_code_per_query.csv",
            "production_join_key": "query taxon/source plus candidate lineage; no hidden correctness needed for deployment-side coverage flags",
            "best_current_signal": species_absent_note,
            "family_genus_relevance": "High: separates absent-reference cases from retrieval/calibration failures before attempting family/genus.",
            "implementation_next_step": "Add lineage coverage and curation-priority features to a new candidate/list compiler; validate by species-split cohorts.",
            "priority": "P0",
            "claim_caveat": "Current labels describe this benchmark/reference table, not true global species absence.",
        },
        {
            "evidence_source": "alignment_backed_marker_resolvability",
            "current_status": "partial_proxy_available",
            "existing_artifacts": "marker_mirror_marker_resolvability_summary.csv; marker_mirror_marker_resolvability_by_species.csv",
            "production_join_key": "query/candidate species or taxon label; marker identity/resolvability group",
            "best_current_signal": summarize_resolvability(resolvability),
            "family_genus_relevance": "High: tells the compiler whether the marker can biologically separate species/genus/family.",
            "implementation_next_step": "Replace rare-kmer 0.99 proxy with VSEARCH/edlib clustering if the rows become claim-facing.",
            "priority": "P0",
            "claim_caveat": "The 0.99 rows are currently a rare-kmer prefix-identity proxy, not full alignment clustering.",
        },
        {
            "evidence_source": "source_specific_calibration_and_lineage_mismatch",
            "current_status": "ready_as_diagnostic_feature",
            "existing_artifacts": "marker_mirror_union_candidate_support_summary.csv; marker_mirror_high_coverage_rank_repair_comparison.csv",
            "production_join_key": "query source, lineage, candidate-source agreement, BLAST/VSEARCH/MarkerMirror top-k modes",
            "best_current_signal": f"{rank_boundary}; {high_rank_reason}",
            "family_genus_relevance": "Medium-high: family/genus may fail only in specific sources/lineages; source-aware abstention can prevent global overfitting.",
            "implementation_next_step": "Train/evaluate source-and-lineage stratified abstention only after adding reference coverage and resolvability features.",
            "priority": "P1",
            "claim_caveat": "Source-specific validation can become small-sample fragile; require minimum-repeat stability.",
        },
        {
            "evidence_source": "same_marker_alignment_evidence",
            "current_status": "already_integrated_for_order",
            "existing_artifacts": "marker_mirror_union_blast_candidate_support_summary.csv; marker_mirror_union_vsearch_candidate_support_summary.csv",
            "production_join_key": "query id and candidate species from BLASTN/VSEARCH candidate tables",
            "best_current_signal": blast_vsearch_signal,
            "family_genus_relevance": "Medium: strong high-rank support exists, but it was insufficient for stable target-0.99 family/genus alone.",
            "implementation_next_step": "Use as a base evidence source; do not rerun threshold-only repair unless combined with new coverage/resolvability/context evidence.",
            "priority": "P1",
            "claim_caveat": "Species is blocked by held-out same-marker reference absence by split design.",
        },
        {
            "evidence_source": "geography_range_prior",
            "current_status": "available_for_eDNA_context_not_marker_mirror_full_query",
            "existing_artifacts": "global_tropical_validation/*geo*; merged_global_edna_sample_metrics.csv; OBIS/RLS range-prior inputs",
            "production_join_key": "sample/site coordinates or region plus candidate species/range",
            "best_current_signal": best_context_metric(edna_metrics, "geo"),
            "family_genus_relevance": "Potentially high for field eDNA, low for unlabeled reference-only MarkerMirror queries without site metadata.",
            "implementation_next_step": "Define a field-eDNA validation cohort with site metadata, then join candidate taxa to RLS/OBIS range priors.",
            "priority": "P2",
            "claim_caveat": "Not applicable to arbitrary FASTA input unless location metadata is supplied.",
        },
        {
            "evidence_source": "cooccurrence_prior",
            "current_status": "available_for_eDNA_context_not_marker_mirror_full_query",
            "existing_artifacts": "cooccurrence_inputs/*; learned/sample cooccurrence validation outputs; merged_global_edna_sample_metrics.csv",
            "production_join_key": "sample id plus other detected ASVs/candidate taxa",
            "best_current_signal": best_context_metric(edna_metrics, "cooccurrence"),
            "family_genus_relevance": "Potentially high for multi-ASV field samples, not available for single isolated 12S query records.",
            "implementation_next_step": "Add candidate-list co-occurrence features only in a sample-aware eDNA mode; keep separate from single-query MarkerMirror.",
            "priority": "P2",
            "claim_caveat": "Can leak sample evidence if the current query is not excluded; use leave-query-out co-occurrence only.",
        },
        {
            "evidence_source": "active_reference_curation_value",
            "current_status": "ready_as_planning_artifact",
            "existing_artifacts": "marker_mirror_union_reference_curation_priorities.csv",
            "production_join_key": "query species/lineage, missing-marker flags, curation priority score",
            "best_current_signal": f"top curation priorities: {top_species(curation)}",
            "family_genus_relevance": "High for improving future family/genus/species coverage by targeting missing lineages instead of only training harder.",
            "implementation_next_step": "Simulate reference additions for top-priority species/lineages, then rerun candidate support before training another compiler.",
            "priority": "P0",
            "claim_caveat": "A curation recommendation is not a model result; it is a value-of-information layer.",
        },
    ]


def write_plan(
    path: Path,
    audit: pd.DataFrame,
    lineage: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    p0 = audit[audit["priority"].eq("P0")]
    top_lineages = lineage.sort_values("query_count", ascending=False).head(8)
    with path.open("w") as handle:
        handle.write("# MarkerMirror Family/Genus Next Evidence Plan\n\n")
        handle.write("## Purpose\n\n")
        handle.write(
            "Exp 119 and Exp 121 showed that threshold-only and set-valued "
            "family/genus repairs do not transfer cleanly. This audit records "
            "which genuinely new evidence sources are available before another "
            "family/genus attempt.\n\n"
        )
        handle.write("## Highest-Priority Evidence Sources\n\n")
        for row in p0.itertuples(index=False):
            handle.write(f"### {row.evidence_source}\n\n")
            handle.write(f"- Status: {row.current_status}\n")
            handle.write(f"- Current signal: {row.best_current_signal}\n")
            handle.write(f"- Join key: {row.production_join_key}\n")
            handle.write(f"- Next step: {row.implementation_next_step}\n")
            handle.write(f"- Caveat: {row.claim_caveat}\n\n")
        handle.write("## Largest Lineage-Level Coverage Rows\n\n")
        if top_lineages.empty:
            handle.write("No lineage table was generated.\n\n")
        else:
            cols = [
                "lineage_rank",
                "lineage",
                "query_count",
                "species_present_16s_query_weighted_pct",
                "union_genus_hit_weighted_pct",
                "union_family_hit_weighted_pct",
                "dominant_next_action",
            ]
            handle.write(top_lineages[cols].to_markdown(index=False))
            handle.write("\n\n")
        handle.write("## Recommended Next Experiment\n\n")
        handle.write(
            "Do not rerun a family/genus threshold repair by itself. The next "
            "scientifically different experiment should add lineage-specific "
            "reference coverage and alignment-backed marker-resolvability "
            "features to the candidate/list compiler, then validate with the "
            "same species-split target-0.99 transfer rule. Geography and "
            "co-occurrence should be reserved for sample-aware field eDNA mode "
            "because arbitrary FASTA queries do not carry site/sample context.\n\n"
        )
        handle.write("## Generated Files\n\n")
        for key, value in manifest["outputs"].items():
            handle.write(f"- `{key}`: `{value}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-table-dir", type=Path, default=DEFAULT_SOURCE_TABLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SOURCE_TABLE_DIR)
    parser.add_argument("--text-output-dir", type=Path, default=DEFAULT_TEXT_DIR)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=default_log_path(ROOT, SCRIPT_NAME),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    logger.start(SCRIPT_NAME)
    logger.log(f"source_table_dir={args.source_table_dir}")
    logger.log(f"output_dir={args.output_dir}")
    logger.log(f"text_output_dir={args.text_output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.text_output_dir.mkdir(parents=True, exist_ok=True)

    curation = read_csv(
        args.source_table_dir / "marker_mirror_union_reference_curation_priorities.csv",
        logger,
    )
    reason_summary = read_csv(
        args.source_table_dir / "marker_mirror_union_reason_code_summary.csv",
        logger,
    )
    resolvability = read_csv(
        args.source_table_dir / "marker_mirror_marker_resolvability_summary.csv",
        logger,
    )
    rank_repair = read_csv(
        args.source_table_dir / "marker_mirror_high_coverage_rank_repair_comparison.csv",
        logger,
    )
    blast_support = read_csv(
        args.source_table_dir / "marker_mirror_union_blast_candidate_support_summary.csv",
        logger,
    )
    vsearch_support = read_csv(
        args.source_table_dir / "marker_mirror_union_vsearch_candidate_support_summary.csv",
        logger,
    )
    edna_metrics = read_csv(
        args.source_table_dir / "merged_global_edna_sample_metrics.csv",
        logger,
    )

    audit = pd.DataFrame(
        build_audit_rows(
            curation=curation,
            reason_summary=reason_summary,
            resolvability=resolvability,
            rank_repair=rank_repair,
            blast_support=blast_support,
            vsearch_support=vsearch_support,
            edna_metrics=edna_metrics,
        )
    )
    audit_path = args.output_dir / "marker_mirror_next_evidence_source_audit.csv"
    audit.to_csv(audit_path, index=False)
    logger.log(f"wrote {audit_path} rows={len(audit)}")

    lineage_frames = [
        build_lineage_rows(curation, "order"),
        build_lineage_rows(curation, "family"),
    ]
    lineage = pd.concat([df for df in lineage_frames if not df.empty], ignore_index=True)
    lineage_path = args.output_dir / "marker_mirror_reference_coverage_by_lineage.csv"
    lineage.to_csv(lineage_path, index=False)
    logger.log(f"wrote {lineage_path} rows={len(lineage)}")

    manifest_path = args.output_dir / "marker_mirror_next_evidence_source_manifest.json"
    plan_path = args.text_output_dir / "marker_mirror_family_genus_next_evidence_plan.md"
    manifest = {
        "script": f"scripts/edna/{SCRIPT_NAME}.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Audit non-threshold evidence sources for future MarkerMirror family/genus work.",
        "claim_boundary": (
            "Planning and data-availability artifact only; no family/genus "
            "rank is enabled by this script."
        ),
        "inputs": {
            "source_table_dir": str(args.source_table_dir),
            "curation": str(
                args.source_table_dir / "marker_mirror_union_reference_curation_priorities.csv"
            ),
            "reason_summary": str(
                args.source_table_dir / "marker_mirror_union_reason_code_summary.csv"
            ),
            "resolvability": str(
                args.source_table_dir / "marker_mirror_marker_resolvability_summary.csv"
            ),
            "rank_repair": str(
                args.source_table_dir / "marker_mirror_high_coverage_rank_repair_comparison.csv"
            ),
            "edna_metrics": str(args.source_table_dir / "merged_global_edna_sample_metrics.csv"),
        },
        "outputs": {
            "audit": str(audit_path),
            "lineage_coverage": str(lineage_path),
            "plan": str(plan_path),
            "manifest": str(manifest_path),
        },
        "recommended_next_experiment": (
            "Add lineage-specific reference coverage plus alignment-backed "
            "marker-resolvability to a new candidate/list compiler; reserve "
            "geography/co-occurrence for sample-aware eDNA mode."
        ),
    }
    write_plan(plan_path, audit, lineage, manifest)
    logger.log(f"wrote {plan_path}")

    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    logger.log(f"wrote {manifest_path}")
    logger.done(SCRIPT_NAME)


if __name__ == "__main__":
    main()
