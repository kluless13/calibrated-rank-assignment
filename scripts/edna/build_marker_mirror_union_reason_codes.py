#!/usr/bin/env python3
"""Build reason-code and curation-priority tables for MarkerMirror union calls.

This is an evidence-accounting layer for Exp 106.  It does not train a new
classifier.  It takes the current MarkerMirror + same-marker union candidate
support and emits:

* query-level reason codes explaining why species/genus/family/order/no-call is
  currently supported;
* summary counts by reason code and data source;
* reference-curation priorities showing which missing marker references block
  more specific calls.

The same-marker arm is still labelled as a k-mer audit until an alignment-backed
VSEARCH/edlib/BLAST replacement is run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")
SUPPORT_RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-support",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_union_candidate_support_per_query.csv",
    )
    parser.add_argument(
        "--top1-features",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "union_candidate_rank_policy"
        / "marker_mirror_union_top1_diagnostic_features.csv",
    )
    parser.add_argument(
        "--static-assignments",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "union_candidate_rank_policy"
        / "marker_mirror_union_static_policy_assignments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


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


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def deepest_rank(prefix: str, row: pd.Series) -> str:
    for rank in SUPPORT_RANKS:
        if truthy(row.get(f"{prefix}_top50_{rank}_hit")):
            return rank
    return "no_support"


def rank_depth(rank: str) -> int:
    order = {"species": 4, "genus": 3, "family": 2, "order": 1, "no_support": 0, "no_call": 0}
    return order.get(clean(rank), 0)


def mode_value(values: pd.Series) -> tuple[str, int]:
    cleaned = [clean(v) for v in values if clean(v)]
    if not cleaned:
        return "", 0
    value, count = Counter(cleaned).most_common(1)[0]
    return value, int(count)


def load_family_order_static_assignments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["policy"] == "top1_source_agreement_family_order"].copy()
    keep = ["query_id", "assigned_rank", "assigned_taxon", "assigned_correct"]
    missing = [column for column in keep if column not in frame.columns]
    if missing:
        raise ValueError(f"Static assignment table missing columns: {missing}")
    return frame[keep].rename(
        columns={
            "assigned_rank": "static_family_order_assigned_rank",
            "assigned_taxon": "static_family_order_assigned_taxon",
            "assigned_correct": "static_family_order_assigned_correct",
        }
    )


def load_top1_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keep = [
        "query_id",
        "mm_score",
        "sm_score",
        "mm_candidate_tree_label",
        "mm_candidate_species",
        "mm_candidate_genus",
        "mm_candidate_family",
        "mm_candidate_order",
        "sm_candidate_tree_label",
        "sm_candidate_species",
        "sm_candidate_genus",
        "sm_candidate_family",
        "sm_candidate_order",
        "top1_sources_agree_genus",
        "top1_sources_agree_family",
        "top1_sources_agree_order",
        "mm_same_marker_best_identity",
        "mm_same_marker_best_pdistance",
        "mm_query_marker_resolvability_rank",
    ]
    existing = [column for column in keep if column in frame.columns]
    return frame[existing].copy()


def normalize_support_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "query_tree_label_x": "query_tree_label",
        "query_species_x": "query_species",
        "query_genus_x": "query_genus",
        "query_family_x": "query_family",
        "query_order_x": "query_order",
    }
    out = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns}).copy()
    for fallback, target in [
        ("query_tree_label_y", "query_tree_label"),
        ("query_species_y", "query_species"),
        ("query_genus_y", "query_genus"),
        ("query_family_y", "query_family"),
        ("query_order_y", "query_order"),
        ("query_source", "source"),
    ]:
        if target not in out.columns and fallback in out.columns:
            out[target] = out[fallback]
        elif fallback in out.columns:
            out[target] = out[target].where(out[target].map(clean).astype(bool), out[fallback])
    return out


def reason_for_row(row: pd.Series) -> tuple[str, str, str]:
    present_12s = truthy(row.get("query_species_present_in_12s_reference"))
    present_16s = truthy(row.get("query_species_present_in_16s_reference"))
    union_rank = clean(row.get("union_deepest_supported_rank"))
    same_rank = clean(row.get("same_marker_deepest_supported_rank"))
    marker_rank = clean(row.get("marker_mirror_deepest_supported_rank"))
    assigned_rank = clean(row.get("static_family_order_assigned_rank")) or "no_call"

    codes: list[str] = []
    if assigned_rank in {"family", "order"}:
        codes.append(f"conservative_{assigned_rank}_source_agreement")

    if not present_12s and not present_16s:
        codes.append("species_reference_gap_both_markers")
    elif not present_16s:
        codes.append("species_reference_gap_target_16s")
    elif present_16s and not truthy(row.get("marker_mirror_top50_species_hit")):
        codes.append("species_present_in_16s_but_not_retrieved")

    if rank_depth(union_rank) >= rank_depth("genus") and union_rank != "species":
        codes.append(f"high_rank_union_support_{union_rank}")
    elif union_rank == "species":
        codes.append("species_in_union_candidate_set")

    if rank_depth(same_rank) > rank_depth(marker_rank):
        codes.append("same_marker_source_improves_rank_support")
    elif rank_depth(marker_rank) > rank_depth(same_rank):
        codes.append("cross_marker_source_improves_rank_support")

    if truthy(row.get("top1_sources_agree_order")) and not truthy(row.get("top1_sources_agree_family")):
        codes.append("top1_sources_agree_order_only")
    if truthy(row.get("top1_sources_agree_family")):
        codes.append("top1_sources_agree_family")

    if not codes:
        codes.append("low_or_conflicting_candidate_support")

    if assigned_rank in {"family", "order"}:
        primary = codes[0]
        action = f"emit_{assigned_rank}_with_species_disabled"
    elif "species_in_union_candidate_set" in codes:
        primary = "species_candidate_available_needs_calibration"
        action = "candidate_for_species_level_calibration"
    elif any(code.startswith("high_rank_union_support") for code in codes):
        primary = next(code for code in codes if code.startswith("high_rank_union_support"))
        action = "candidate_for_rank_adaptive_assignment"
    elif any("reference_gap" in code for code in codes):
        primary = next(code for code in codes if "reference_gap" in code)
        action = "prioritize_reference_curation"
    else:
        primary = codes[0]
        action = "no_call_or_collect_more_evidence"

    return primary, ";".join(dict.fromkeys(codes)), action


def add_reason_codes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for prefix in ("marker_mirror", "same_marker", "union"):
        out[f"{prefix}_deepest_supported_rank"] = out.apply(lambda row, p=prefix: deepest_rank(p, row), axis=1)

    reason_rows = [reason_for_row(row) for _, row in out.iterrows()]
    out["primary_reason_code"] = [row[0] for row in reason_rows]
    out["all_reason_codes"] = [row[1] for row in reason_rows]
    out["recommended_next_action"] = [row[2] for row in reason_rows]
    out["static_family_order_assigned"] = out["static_family_order_assigned_rank"].map(clean).isin({"family", "order"})
    out["static_family_order_assigned_correct_bool"] = out["static_family_order_assigned_correct"].map(truthy)
    out["reference_gap_status"] = out.apply(
        lambda row: (
            "species_not_in_current_12s_or_16s_reference"
            if not truthy(row.get("query_species_present_in_12s_reference"))
            and not truthy(row.get("query_species_present_in_16s_reference"))
            else "species_not_in_current_16s_reference"
            if not truthy(row.get("query_species_present_in_16s_reference"))
            else "species_not_in_current_12s_reference"
            if not truthy(row.get("query_species_present_in_12s_reference"))
            else "species_present_in_both_marker_references"
        ),
        axis=1,
    )
    return out


def reason_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(frame)
    for (reason, action), group in frame.groupby(["primary_reason_code", "recommended_next_action"], dropna=False):
        assigned = group["static_family_order_assigned"]
        rows.append(
            {
                "primary_reason_code": reason,
                "recommended_next_action": action,
                "n_queries": int(len(group)),
                "query_pct": pct(len(group), total),
                "assigned_family_order_count": int(assigned.sum()),
                "assigned_family_order_precision_pct": (
                    float(group.loc[assigned, "static_family_order_assigned_correct_bool"].mean() * 100.0)
                    if assigned.any()
                    else math.nan
                ),
                "species_present_12s_pct": float(group["query_species_present_in_12s_reference"].map(truthy).mean() * 100.0),
                "species_present_16s_pct": float(group["query_species_present_in_16s_reference"].map(truthy).mean() * 100.0),
                "union_species_hit_pct": float(group["union_top50_species_hit"].map(truthy).mean() * 100.0),
                "union_genus_hit_pct": float(group["union_top50_genus_hit"].map(truthy).mean() * 100.0),
                "union_family_hit_pct": float(group["union_top50_family_hit"].map(truthy).mean() * 100.0),
                "union_order_hit_pct": float(group["union_top50_order_hit"].map(truthy).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["n_queries", "primary_reason_code"], ascending=[False, True])


def source_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source, reason), group in frame.groupby(["source", "primary_reason_code"], dropna=False):
        assigned = group["static_family_order_assigned"]
        rows.append(
            {
                "source": source,
                "primary_reason_code": reason,
                "n_queries": int(len(group)),
                "query_pct_within_source": pct(len(group), frame[frame["source"] == source].shape[0]),
                "assigned_family_order_count": int(assigned.sum()),
                "assigned_family_order_precision_pct": (
                    float(group.loc[assigned, "static_family_order_assigned_correct_bool"].mean() * 100.0)
                    if assigned.any()
                    else math.nan
                ),
                "species_present_12s_pct": float(group["query_species_present_in_12s_reference"].map(truthy).mean() * 100.0),
                "species_present_16s_pct": float(group["query_species_present_in_16s_reference"].map(truthy).mean() * 100.0),
                "union_deepest_mode": mode_value(group["union_deepest_supported_rank"])[0],
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "n_queries"], ascending=[True, False])


def curation_action(row: pd.Series) -> str:
    present_12s = truthy(row.get("query_species_present_in_12s_reference"))
    present_16s = truthy(row.get("query_species_present_in_16s_reference"))
    species_hit_pct = float(row.get("union_species_hit_pct", 0.0))
    genus_hit = float(row.get("union_genus_hit_pct", 0.0)) >= 50.0
    family_hit = float(row.get("union_family_hit_pct", 0.0)) >= 50.0

    if not present_12s and not present_16s:
        return "add_12s_and_16s_reference_for_species"
    if not present_16s and (genus_hit or family_hit):
        return "add_16s_reference_for_cross_marker_species_resolution"
    if not present_16s:
        return "add_16s_reference_then_retest_marker_mirror"
    if present_16s and species_hit_pct < 50.0 and genus_hit:
        return "improve_cross_marker_retrieval_or_check_marker_ambiguity"
    if present_16s and species_hit_pct < 50.0:
        return "improve_cross_marker_retrieval_or_collect_more_evidence"
    if species_hit_pct >= 50.0:
        return "candidate_species_present_calibrate_species_call"
    return "collect_more_marker_or_taxonomy_evidence"


def curation_priority(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["query_tree_label", "query_species", "query_genus", "query_family", "query_order"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        reason, reason_count = mode_value(group["primary_reason_code"])
        action, _ = mode_value(group["recommended_next_action"])
        source_mode, source_count = mode_value(group["source"])
        row = {
            "query_tree_label": keys[0],
            "query_species": keys[1],
            "query_genus": keys[2],
            "query_family": keys[3],
            "query_order": keys[4],
            "n_queries": int(len(group)),
            "n_sources": int(group["source"].nunique(dropna=True)),
            "dominant_source": source_mode,
            "dominant_source_count": source_count,
            "query_species_present_in_12s_reference": bool(group["query_species_present_in_12s_reference"].map(truthy).any()),
            "query_species_present_in_16s_reference": bool(group["query_species_present_in_16s_reference"].map(truthy).any()),
            "marker_mirror_species_hit_pct": float(group["marker_mirror_top50_species_hit"].map(truthy).mean() * 100.0),
            "marker_mirror_genus_hit_pct": float(group["marker_mirror_top50_genus_hit"].map(truthy).mean() * 100.0),
            "same_marker_genus_hit_pct": float(group["same_marker_top50_genus_hit"].map(truthy).mean() * 100.0),
            "union_species_hit_pct": float(group["union_top50_species_hit"].map(truthy).mean() * 100.0),
            "union_genus_hit_pct": float(group["union_top50_genus_hit"].map(truthy).mean() * 100.0),
            "union_family_hit_pct": float(group["union_top50_family_hit"].map(truthy).mean() * 100.0),
            "union_order_hit_pct": float(group["union_top50_order_hit"].map(truthy).mean() * 100.0),
            "primary_reason_mode": reason,
            "primary_reason_mode_count": reason_count,
            "recommended_next_action_mode": action,
            "static_family_order_assigned_pct": float(group["static_family_order_assigned"].mean() * 100.0),
        }
        score = float(row["n_queries"])
        if not row["query_species_present_in_16s_reference"]:
            score += 50.0
        if not row["query_species_present_in_12s_reference"]:
            score += 25.0
        if row["union_genus_hit_pct"] >= 50.0 and row["union_species_hit_pct"] == 0.0:
            score += 15.0
        if row["n_sources"] > 1:
            score += 5.0
        row["curation_priority_score"] = score
        rows.append(row)

    out = pd.DataFrame(rows)
    out["curation_recommendation"] = out.apply(curation_action, axis=1)
    return out.sort_values(
        ["curation_priority_score", "n_queries", "query_species"],
        ascending=[False, False, True],
    )


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    script_name = Path(__file__).name
    logger.start(script_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Loading candidate support {rel(args.candidate_support)}")
    support = normalize_support_columns(pd.read_csv(args.candidate_support))
    logger.log(f"Candidate-support rows={len(support):,}")

    logger.log(f"Loading top1 diagnostic features {rel(args.top1_features)}")
    top1 = load_top1_features(args.top1_features)
    logger.log(f"Top1 feature rows={len(top1):,}")

    logger.log(f"Loading static family/order assignments {rel(args.static_assignments)}")
    static = load_family_order_static_assignments(args.static_assignments)
    logger.log(f"Static assignment rows={len(static):,}")

    frame = support.merge(top1, on="query_id", how="left").merge(static, on="query_id", how="left")
    frame["static_family_order_assigned_rank"] = frame["static_family_order_assigned_rank"].fillna("no_call")
    frame["static_family_order_assigned_taxon"] = frame["static_family_order_assigned_taxon"].fillna("")
    frame["static_family_order_assigned_correct"] = frame["static_family_order_assigned_correct"].map(truthy)
    frame = add_reason_codes(frame)
    logger.log(f"Built reason-code frame rows={len(frame):,}")

    per_query_path = args.output_dir / "marker_mirror_union_reason_code_per_query.csv"
    summary_path = args.output_dir / "marker_mirror_union_reason_code_summary.csv"
    source_summary_path = args.output_dir / "marker_mirror_union_reason_code_by_source.csv"
    curation_path = args.output_dir / "marker_mirror_union_reference_curation_priorities.csv"
    manifest_path = args.output_dir / "marker_mirror_union_reason_code_manifest.json"

    frame.to_csv(per_query_path, index=False)
    reason_summary(frame).to_csv(summary_path, index=False)
    source_summary(frame).to_csv(source_summary_path, index=False)
    curation_priority(frame).to_csv(curation_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "candidate_support": rel(args.candidate_support),
            "top1_features": rel(args.top1_features),
            "static_assignments": rel(args.static_assignments),
        },
        "outputs": {
            "per_query": rel(per_query_path),
            "summary": rel(summary_path),
            "by_source": rel(source_summary_path),
            "reference_curation_priorities": rel(curation_path),
        },
        "n_queries": int(len(frame)),
        "n_primary_reason_codes": int(frame["primary_reason_code"].nunique()),
        "same_marker_caveat": "same-marker candidate source is k-mer/TF-IDF audit, not final BLAST/VSEARCH/edlib evidence",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote {rel(per_query_path)}")
    logger.log(f"Wrote {rel(summary_path)}")
    logger.log(f"Wrote {rel(source_summary_path)}")
    logger.log(f"Wrote {rel(curation_path)}")
    logger.log(f"Wrote {rel(manifest_path)}")
    logger.done(script_name)


if __name__ == "__main__":
    main()
