#!/usr/bin/env python3
"""Build active reference-curation value-of-information tables for MarkerMirror.

This is an evidence-accounting layer, not a rank/no-call policy. It combines
the existing MarkerMirror union reason codes, BLAST/VSEARCH-backed order policy
outputs, and VSEARCH marker-resolvability diagnostics to rank which reference
curation actions would most plausibly change future evidence.

The labelled VSEARCH oracle columns are included only for benchmark curation
triage. They must not be used as production inference features.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = "build_marker_mirror_active_reference_value"
IDENTITIES = ("0p99", "0p98", "0p97", "0p95")
RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-table-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "active_reference_value",
    )
    parser.add_argument(
        "--vsearch-resolvability-dir",
        type=Path,
        default=ROOT
        / "results"
        / "remote_runs"
        / "2026-06-04"
        / "rtx_pro_6000"
        / "marker_mirror_vsearch_resolvability_20260604"
        / "12s",
    )
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean(value).lower()
    return text in {"true", "1", "yes", "y"}


def pct(numer: float, denom: float) -> float:
    return 100.0 * float(numer) / float(denom) if denom else float("nan")


def pct_text(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}%"


def read_csv(path: Path, logger: ProgressLogger, *, required: bool = True, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        message = f"missing {'required' if required else 'optional'} input: {rel(path)}"
        if required:
            raise FileNotFoundError(message)
        logger.log(message)
        return pd.DataFrame()
    df = pd.read_csv(path, **kwargs)
    logger.log(f"loaded {rel(path)} rows={len(df)} cols={len(df.columns)}")
    return df


def mode_value(values: pd.Series) -> tuple[str, int]:
    cleaned = [clean(value) for value in values if clean(value)]
    if not cleaned:
        return "", 0
    value, count = Counter(cleaned).most_common(1)[0]
    return value, int(count)


def bool_pct(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.map(truthy).mean() * 100.0)


def mean_pct(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").mean() * 100.0)


def top_species(group: pd.DataFrame, n: int = 5) -> str:
    if group.empty:
        return ""
    sub = group.sort_values("active_reference_value_score", ascending=False).head(n)
    return "; ".join(
        f"{row.query_tree_label} ({row.active_reference_value_score:.1f})"
        for row in sub.itertuples(index=False)
    )


def aggregate_reason_queries(reason: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tree_label, group in reason.groupby("query_tree_label", dropna=False):
        primary, primary_count = mode_value(group["primary_reason_code"])
        action, action_count = mode_value(group["recommended_next_action"])
        ref_gap, ref_gap_count = mode_value(group["reference_gap_status"])
        rows.append(
            {
                "query_tree_label": tree_label,
                "reason_query_rows": len(group),
                "primary_reason_mode_from_queries": primary,
                "primary_reason_mode_query_count": primary_count,
                "recommended_next_action_mode_from_queries": action,
                "recommended_next_action_mode_query_count": action_count,
                "reference_gap_status_mode": ref_gap,
                "reference_gap_status_mode_query_count": ref_gap_count,
                "static_family_order_assigned_query_pct": bool_pct(
                    group["static_family_order_assigned"]
                ),
                "top1_sources_agree_genus_pct": bool_pct(
                    group.get("top1_sources_agree_genus", pd.Series(dtype=bool))
                ),
                "top1_sources_agree_family_pct": bool_pct(
                    group.get("top1_sources_agree_family", pd.Series(dtype=bool))
                ),
                "top1_sources_agree_order_pct": bool_pct(
                    group.get("top1_sources_agree_order", pd.Series(dtype=bool))
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_assignments(path: Path, logger: ProgressLogger, reason: pd.DataFrame, prefix: str) -> pd.DataFrame:
    assignments = read_csv(path, logger, required=False)
    if assignments.empty:
        return pd.DataFrame(columns=["query_tree_label"])
    if "query_id" not in assignments or "assigned_rank" not in assignments:
        logger.log(f"assignment table missing expected columns: {rel(path)}")
        return pd.DataFrame(columns=["query_tree_label"])

    query_lookup = reason[["query_id", "query_tree_label"]].drop_duplicates()
    joined = assignments.merge(query_lookup, on="query_id", how="left")
    rows: list[dict[str, Any]] = []
    for tree_label, group in joined.groupby("query_tree_label", dropna=False):
        rows.append(
            {
                "query_tree_label": tree_label,
                f"{prefix}_assigned_order_query_pct": pct(
                    (group["assigned_rank"].astype(str) == "order").sum(), len(group)
                ),
                f"{prefix}_no_call_query_pct": pct(
                    (group["assigned_rank"].astype(str) == "no_call").sum(), len(group)
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_query_features(reason: pd.DataFrame, query_features: pd.DataFrame) -> pd.DataFrame:
    if query_features.empty:
        return pd.DataFrame(columns=["query_tree_label"])
    query_lookup = reason[["query_id", "query_tree_label"]].drop_duplicates()
    joined = query_features.merge(query_lookup, on="query_id", how="left")
    rows: list[dict[str, Any]] = []
    for tree_label, group in joined.groupby("query_tree_label", dropna=False):
        row: dict[str, Any] = {"query_tree_label": tree_label}
        for ident in IDENTITIES:
            col = f"vsearch_id{ident}_cluster_has_reference"
            if col in group:
                row[f"production_vsearch_id{ident}_cluster_has_reference_pct"] = bool_pct(group[col])
            ref_frac = f"vsearch_id{ident}_cluster_reference_fraction"
            if ref_frac in group:
                row[f"production_vsearch_id{ident}_cluster_reference_fraction_mean_pct"] = mean_pct(group[ref_frac])
        if "vsearch_resolvability_reference_identity_count" in group:
            row["production_vsearch_reference_identity_count_mean"] = float(
                pd.to_numeric(
                    group["vsearch_resolvability_reference_identity_count"], errors="coerce"
                ).mean()
            )
        if "vsearch_resolvability_max_identity_with_reference" in group:
            row["production_vsearch_max_identity_with_reference_mean"] = float(
                pd.to_numeric(
                    group["vsearch_resolvability_max_identity_with_reference"], errors="coerce"
                ).mean()
            )
            row["production_vsearch_max_identity_with_reference_max"] = float(
                pd.to_numeric(
                    group["vsearch_resolvability_max_identity_with_reference"], errors="coerce"
                ).max()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_oracles(vsearch_dir: Path, logger: ProgressLogger) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for ident in IDENTITIES:
        identity_label = ident.replace("p", ".")
        path = vsearch_dir / f"query_oracle_id{ident}.csv"
        oracle = read_csv(path, logger, required=False)
        if oracle.empty:
            continue
        if "true_tree_label" not in oracle:
            logger.log(f"query oracle missing true_tree_label: {rel(path)}")
            continue
        rows: list[dict[str, Any]] = []
        for tree_label, group in oracle.groupby("true_tree_label", dropna=False):
            deepest, deepest_count = mode_value(group["deepest_supported_rank"])
            row: dict[str, Any] = {
                "query_tree_label": tree_label,
                f"diagnostic_vsearch_id{ident}_identity": float(identity_label),
                f"diagnostic_vsearch_id{ident}_query_count": int(len(group)),
                f"diagnostic_vsearch_id{ident}_cluster_has_reference_pct": bool_pct(
                    group["cluster_has_reference"]
                ),
                f"diagnostic_vsearch_id{ident}_true_species_in_cluster_pct": bool_pct(
                    group["true_species_in_cluster"]
                ),
                f"diagnostic_vsearch_id{ident}_cluster_species_count_mean": float(
                    pd.to_numeric(group["cluster_species_count"], errors="coerce").mean()
                ),
                f"diagnostic_vsearch_id{ident}_deepest_supported_rank_mode": deepest,
                f"diagnostic_vsearch_id{ident}_deepest_supported_rank_mode_count": deepest_count,
            }
            for rank in RANKS:
                row[f"diagnostic_vsearch_id{ident}_{rank}_oracle_supported_pct"] = bool_pct(
                    group[f"{rank}_oracle_supported"]
                )
            rows.append(row)
        agg = pd.DataFrame(rows)
        merged = agg if merged is None else merged.merge(agg, on="query_tree_label", how="outer")
    return merged if merged is not None else pd.DataFrame(columns=["query_tree_label"])


def choose_action(row: pd.Series) -> tuple[str, str, float]:
    n_queries = float(row.get("n_queries", 0.0) or 0.0)
    present_12s = truthy(row.get("query_species_present_in_12s_reference"))
    present_16s = truthy(row.get("query_species_present_in_16s_reference"))
    mm_species = float(row.get("marker_mirror_species_hit_pct", 0.0) or 0.0)
    mm_genus = float(row.get("marker_mirror_genus_hit_pct", 0.0) or 0.0)
    same_genus = float(row.get("same_marker_genus_hit_pct", 0.0) or 0.0)
    union_species = float(row.get("union_species_hit_pct", 0.0) or 0.0)
    union_genus = float(row.get("union_genus_hit_pct", 0.0) or 0.0)
    union_family = float(row.get("union_family_hit_pct", 0.0) or 0.0)
    union_order = float(row.get("union_order_hit_pct", 0.0) or 0.0)
    high_order = float(row.get("high_coverage_order_assigned_order_query_pct", 0.0) or 0.0)
    v99_ref = float(row.get("diagnostic_vsearch_id0p99_cluster_has_reference_pct", 0.0) or 0.0)
    v99_species = float(row.get("diagnostic_vsearch_id0p99_species_oracle_supported_pct", 0.0) or 0.0)
    v99_genus = float(row.get("diagnostic_vsearch_id0p99_genus_oracle_supported_pct", 0.0) or 0.0)
    prod_identity = float(
        row.get("production_vsearch_max_identity_with_reference_mean", 0.0) or 0.0
    )

    reference_gap_weight = (0.0 if present_12s else 0.65) + (0.0 if present_16s else 0.45)
    support_weight = 0.35 * union_genus / 100.0 + 0.20 * union_family / 100.0 + 0.10 * union_order / 100.0
    oracle_weight = 0.35 * v99_species / 100.0 + 0.15 * v99_genus / 100.0
    no_call_weight = 0.30 * (1.0 - min(max(high_order, 0.0), 100.0) / 100.0)
    retrieval_failure_weight = 0.75 if present_16s and mm_genus < 20.0 and union_genus < 50.0 else 0.0
    score = n_queries * (1.0 + reference_gap_weight + support_weight + oracle_weight + no_call_weight + retrieval_failure_weight)

    if not present_12s and not present_16s:
        if v99_species >= 80.0 or (v99_species >= 60.0 and v99_ref < 50.0):
            action = "add_12s_and_16s_species_reference_high_expected_value"
            rationale = "species absent from both current marker references, but labelled VSEARCH clusters suggest species-level separation is often possible"
        elif v99_genus >= 80.0 or union_genus >= 90.0:
            action = "add_reference_then_validate_genus_family_not_species"
            rationale = "reference gap is dominant and high-rank evidence is strong, but current 12S evidence may not justify species"
        else:
            action = "collect_multi_marker_reference_or_sample_context"
            rationale = "both marker references are missing and current candidate evidence is not strong enough for a direct rank unlock"
    elif present_16s and not present_12s and mm_species < 5.0 and mm_genus < 20.0:
        action = "improve_cross_marker_retrieval_or_curate_16s_target"
        rationale = "16S target species exists in the current reference, but MarkerMirror retrieval is weak"
    elif not present_12s and same_genus >= 90.0:
        action = "add_12s_same_marker_reference_then_revalidate"
        rationale = "same-marker BLAST/VSEARCH-like evidence recovers the correct genus, but species is absent by design"
    elif union_species > 0.0:
        action = "species_candidate_available_needs_independent_calibration"
        rationale = "a species candidate appears in the union set, but species output remains disabled until calibration transfers cleanly"
    elif prod_identity >= 0.99 and v99_species < 50.0 and union_genus >= 80.0:
        action = "marker_ambiguity_collect_second_marker_or_ecology"
        rationale = "near-exact references exist, but 12S clusters remain ambiguous for species; add independent marker or sample-aware evidence"
    elif high_order >= 70.0:
        action = "order_safe_lower_curation_urgency"
        rationale = "current high-coverage order mode already emits an order call for most rows in this species group"
    elif union_genus >= 80.0:
        action = "calibration_or_context_needed_for_rank_lift"
        rationale = "candidate evidence reaches genus/family, but previous calibration attempts did not transfer cleanly"
    else:
        action = "manual_review_low_candidate_support"
        rationale = "current candidate evidence is weak or conflicting"

    return action, rationale, round(float(score), 3)


def build_species_table(
    curation: pd.DataFrame,
    reason_agg: pd.DataFrame,
    stable_agg: pd.DataFrame,
    high_agg: pd.DataFrame,
    feature_agg: pd.DataFrame,
    oracle_agg: pd.DataFrame,
) -> pd.DataFrame:
    out = curation.copy()
    for table in (reason_agg, stable_agg, high_agg, feature_agg, oracle_agg):
        if not table.empty:
            out = out.merge(table, on="query_tree_label", how="left")

    actions = [choose_action(row) for _, row in out.iterrows()]
    out["active_reference_action"] = [action for action, _, _ in actions]
    out["active_reference_rationale"] = [rationale for _, rationale, _ in actions]
    out["active_reference_value_score"] = [score for _, _, score in actions]

    preferred_cols = [
        "query_tree_label",
        "query_species",
        "query_genus",
        "query_family",
        "query_order",
        "n_queries",
        "n_sources",
        "dominant_source",
        "query_species_present_in_12s_reference",
        "query_species_present_in_16s_reference",
        "reference_gap_status_mode",
        "primary_reason_mode",
        "recommended_next_action_mode",
        "marker_mirror_species_hit_pct",
        "marker_mirror_genus_hit_pct",
        "same_marker_genus_hit_pct",
        "union_species_hit_pct",
        "union_genus_hit_pct",
        "union_family_hit_pct",
        "union_order_hit_pct",
        "static_family_order_assigned_pct",
        "stable_order_assigned_order_query_pct",
        "high_coverage_order_assigned_order_query_pct",
        "production_vsearch_id0p99_cluster_has_reference_pct",
        "production_vsearch_max_identity_with_reference_mean",
        "diagnostic_vsearch_id0p99_cluster_has_reference_pct",
        "diagnostic_vsearch_id0p99_species_oracle_supported_pct",
        "diagnostic_vsearch_id0p99_genus_oracle_supported_pct",
        "diagnostic_vsearch_id0p99_family_oracle_supported_pct",
        "diagnostic_vsearch_id0p99_order_oracle_supported_pct",
        "diagnostic_vsearch_id0p99_deepest_supported_rank_mode",
        "active_reference_action",
        "active_reference_rationale",
        "curation_priority_score",
        "active_reference_value_score",
        "curation_recommendation",
    ]
    existing_cols = [col for col in preferred_cols if col in out.columns]
    remaining_cols = [col for col in out.columns if col not in existing_cols]
    out = out[existing_cols + remaining_cols].sort_values(
        ["active_reference_value_score", "n_queries"], ascending=[False, False]
    )
    return out


def build_lineage_table(species: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, col in [
        ("order", "query_order"),
        ("family", "query_family"),
        ("genus", "query_genus"),
    ]:
        if col not in species:
            continue
        for lineage, group in species.groupby(col, dropna=False):
            total_queries = float(group["n_queries"].sum())
            if total_queries <= 0:
                continue
            action, action_count = mode_value(group["active_reference_action"])
            rows.append(
                {
                    "lineage_rank": rank,
                    "lineage": lineage,
                    "species_group_count": int(group["query_tree_label"].nunique()),
                    "query_count": int(total_queries),
                    "active_reference_value_score_sum": round(
                        float(group["active_reference_value_score"].sum()), 3
                    ),
                    "active_reference_value_score_mean": round(
                        float(group["active_reference_value_score"].mean()), 3
                    ),
                    "species_present_12s_weighted_pct": round(
                        float(
                            (group["query_species_present_in_12s_reference"].astype(bool) * group["n_queries"]).sum()
                            / total_queries
                            * 100.0
                        ),
                        3,
                    ),
                    "species_present_16s_weighted_pct": round(
                        float(
                            (group["query_species_present_in_16s_reference"].astype(bool) * group["n_queries"]).sum()
                            / total_queries
                            * 100.0
                        ),
                        3,
                    ),
                    "union_genus_hit_weighted_pct": round(
                        float((group["union_genus_hit_pct"] * group["n_queries"]).sum() / total_queries),
                        3,
                    ),
                    "union_family_hit_weighted_pct": round(
                        float((group["union_family_hit_pct"] * group["n_queries"]).sum() / total_queries),
                        3,
                    ),
                    "union_order_hit_weighted_pct": round(
                        float((group["union_order_hit_pct"] * group["n_queries"]).sum() / total_queries),
                        3,
                    ),
                    "dominant_active_reference_action": action,
                    "dominant_active_reference_action_species_count": action_count,
                    "top_active_reference_species": top_species(group),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["active_reference_value_score_sum", "query_count"], ascending=[False, False]
    )


def build_action_table(species: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for action, group in species.groupby("active_reference_action", dropna=False):
        rows.append(
            {
                "active_reference_action": action,
                "species_group_count": int(group["query_tree_label"].nunique()),
                "query_count": int(group["n_queries"].sum()),
                "active_reference_value_score_sum": round(
                    float(group["active_reference_value_score"].sum()), 3
                ),
                "mean_union_genus_hit_pct": round(float(group["union_genus_hit_pct"].mean()), 3),
                "mean_union_family_hit_pct": round(float(group["union_family_hit_pct"].mean()), 3),
                "mean_union_order_hit_pct": round(float(group["union_order_hit_pct"].mean()), 3),
                "mean_vsearch_0p99_species_oracle_supported_pct": round(
                    float(
                        pd.to_numeric(
                            group.get(
                                "diagnostic_vsearch_id0p99_species_oracle_supported_pct",
                                pd.Series(dtype=float),
                            ),
                            errors="coerce",
                        ).mean()
                    ),
                    3,
                ),
                "top_species": top_species(group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["active_reference_value_score_sum", "query_count"], ascending=[False, False]
    )


def write_outputs(
    species: pd.DataFrame,
    lineage: pd.DataFrame,
    actions: pd.DataFrame,
    args: argparse.Namespace,
    logger: ProgressLogger,
) -> dict[str, str]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_table_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for name, table in [
        ("species", species),
        ("lineage", lineage),
        ("actions", actions),
    ]:
        bridge_path = args.output_dir / f"marker_mirror_active_reference_value_{name}.csv"
        source_path = args.source_table_dir / f"marker_mirror_active_reference_value_{name}.csv"
        table.to_csv(bridge_path, index=False)
        table.to_csv(source_path, index=False)
        outputs[f"{name}_bridge"] = rel(bridge_path)
        outputs[f"{name}_source_table"] = rel(source_path)
        logger.log(f"wrote {name} rows={len(table)} to {rel(source_path)}")

    outputs["manifest_bridge"] = rel(
        args.output_dir / "marker_mirror_active_reference_value_manifest.json"
    )
    outputs["manifest_source_table"] = rel(
        args.source_table_dir / "marker_mirror_active_reference_value_manifest.json"
    )

    manifest = {
        "script": SCRIPT_NAME,
        "created_at_utc": utc_now(),
        "inputs": {
            "source_table_dir": rel(args.source_table_dir),
            "vsearch_resolvability_dir": rel(args.vsearch_resolvability_dir),
        },
        "outputs": outputs,
        "caveats": [
            "This is an active reference-curation/value-of-information diagnostic, not a production rank/no-call policy.",
            "Diagnostic VSEARCH oracle columns use benchmark labels to prioritize curation; they must not be used as production inference features.",
            "Reference-gap columns describe the current benchmark/reference tables, not global species absence.",
            "Family/genus/species remain disabled unless a later independently validated policy transfers cleanly.",
        ],
    }
    for path in [
        args.output_dir / "marker_mirror_active_reference_value_manifest.json",
        args.source_table_dir / "marker_mirror_active_reference_value_manifest.json",
    ]:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        logger.log(f"wrote manifest {rel(path)}")
    return outputs


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, SCRIPT_NAME))
    logger.start(SCRIPT_NAME)

    source = args.source_table_dir
    curation = read_csv(source / "marker_mirror_union_reference_curation_priorities.csv", logger)
    reason = read_csv(source / "marker_mirror_union_reason_code_per_query.csv", logger)
    stable_agg = aggregate_assignments(
        source / "marker_mirror_stable_order_policy_production_assignments.csv",
        logger,
        reason,
        "stable_order",
    )
    high_agg = aggregate_assignments(
        source / "marker_mirror_high_coverage_order_repair_assignments.csv",
        logger,
        reason,
        "high_coverage_order",
    )
    query_features = read_csv(
        source / "marker_mirror_vsearch_resolvability_policy_diagnostic_query_features.csv",
        logger,
        required=False,
    )

    reason_agg = aggregate_reason_queries(reason)
    feature_agg = aggregate_query_features(reason, query_features)
    oracle_agg = aggregate_oracles(args.vsearch_resolvability_dir, logger)
    logger.log(
        "aggregated species inputs "
        f"reason={len(reason_agg)} stable={len(stable_agg)} high={len(high_agg)} "
        f"features={len(feature_agg)} oracle={len(oracle_agg)}"
    )

    species = build_species_table(curation, reason_agg, stable_agg, high_agg, feature_agg, oracle_agg)
    lineage = build_lineage_table(species)
    actions = build_action_table(species)
    outputs = write_outputs(species, lineage, actions, args, logger)

    top = species.head(8)[
        [
            "query_tree_label",
            "n_queries",
            "active_reference_action",
            "active_reference_value_score",
            "union_genus_hit_pct",
            "diagnostic_vsearch_id0p99_species_oracle_supported_pct",
        ]
    ]
    logger.log("top active reference targets:\n" + top.to_string(index=False))
    logger.log(f"outputs={json.dumps(outputs, sort_keys=True)}")
    logger.done(SCRIPT_NAME)


if __name__ == "__main__":
    main()
