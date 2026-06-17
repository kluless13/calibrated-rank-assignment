#!/usr/bin/env python3
"""Build auditable Eco-Phylo posterior prototype inputs from existing outputs.

This is not a new model and it does not invent a posterior score. It
consolidates the evidence already produced by the Global_eDNA runs into a
single table that can be used to design the next Eco-Phylo posterior:

- sequence/model candidate score;
- learned co-occurrence/range-prior arm labels;
- marker resolvability ceilings;
- calibration/evaluation site split;
- top-1 correctness by species/genus/family/order;
- candidate/reference evidence when available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "edna" / "real_edna_queries" / "global_tropical_multisource_teleo"
DEFAULT_METHODS = ROOT / "configs" / "runs" / "2026-05-31-paper1-merged-global-edna-calibration-methods.json"
DEFAULT_SOURCE_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
DEFAULT_CALIBRATION_DIR = (
    ROOT / "results" / "paper1_phylo_calibrated_assignment" / "global_edna_independent_rank_calibration"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "eco_phylo_posterior"
RANKS = ("species", "genus", "family", "order")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def clean(value: object) -> str:
    if not nonempty(value):
        return ""
    return str(value).strip()


def as_float(value: object) -> float | None:
    if not nonempty(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_list(value: object) -> list[Any]:
    if not nonempty(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    for sep in ("|", ",", ";"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def normalize_label(value: object) -> str:
    return clean(value).replace(" ", "_")


def split_label(value: object) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return "calibration" if int(digest[:8], 16) % 2 == 0 else "evaluation"


def rank_value(label: str | None, rank: str, taxonomy: dict[str, dict[str, object]]) -> str:
    if not label:
        return ""
    label = label.replace(" ", "_")
    if rank == "species":
        return label
    if rank == "genus":
        value = taxonomy.get(label, {}).get("genus_name")
        return clean(value) or label.split("_", 1)[0]
    return clean(taxonomy.get(label, {}).get(f"{rank}_name"))


def true_rank_value(row: pd.Series, rank: str, taxonomy: dict[str, dict[str, object]]) -> str:
    if rank == "species":
        return normalize_label(row.get("true_tree_label"))
    value = clean(row.get(f"true_{rank}_name"))
    if value:
        return value
    return rank_value(normalize_label(row.get("true_tree_label")), rank, taxonomy)


def method_metadata(method_name: str) -> dict[str, Any]:
    if method_name.startswith("ssm_"):
        encoder = "ssm"
    elif method_name.startswith("cnn_"):
        encoder = "cnn"
    elif method_name.startswith("blast_"):
        encoder = "blast"
    else:
        encoder = "unknown"

    if "sequence_only" in method_name:
        context = "sequence_only"
    elif "cooccurrence" in method_name:
        context = "learned_cooccurrence"
    else:
        context = "unknown"

    if "rls_obis" in method_name:
        prior_source = "rls_obis"
    elif "fishglob" in method_name:
        prior_source = "fishglob_public_50k"
    else:
        prior_source = ""

    prior_weight = ""
    for token, weight in (("w025", 0.25), ("w050", 0.50), ("w100", 1.0), ("w200", 2.0)):
        if token in method_name:
            prior_weight = weight
            break

    if encoder in {"ssm", "cnn"} and context == "sequence_only":
        evidence_arm = "sequence_tree_only"
        evidence_family = "sequence_tree"
    elif context == "learned_cooccurrence":
        evidence_arm = f"sequence_tree_plus_{prior_source}_learned_cooccurrence"
        evidence_family = "sequence_tree_plus_ecology"
    else:
        evidence_arm = context
        evidence_family = "other"

    return {
        "encoder": encoder,
        "context": context,
        "prior_source": prior_source,
        "prior_weight": prior_weight,
        "evidence_arm": evidence_arm,
        "evidence_family": evidence_family,
        "sequence_score_available": encoder in {"ssm", "cnn", "blast"},
        "tree_candidate_evidence_available": encoder in {"ssm", "cnn"},
        "geography_prior_available": prior_source == "rls_obis",
        "cooccurrence_prior_available": context == "learned_cooccurrence",
    }


def load_marker_ceiling(source_dir: Path, dataset: str, identity: float) -> dict[str, Any]:
    path = source_dir / "merged_12s_resolvability_summary.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    subset = table[(table["dataset"] == dataset) & (table["identity"].astype(float).round(4) == round(identity, 4))]
    if subset.empty:
        subset = table[table["dataset"] == dataset].sort_values("identity", ascending=False).head(1)
    if subset.empty:
        return {}
    row = subset.iloc[0]
    return {
        "marker_resolvability_dataset": clean(row.get("dataset")),
        "marker_resolvability_mode": clean(row.get("mode")),
        "marker_identity_threshold": as_float(row.get("identity")),
        "marker_query_cluster_has_reference_rate_pct": as_float(row.get("query_cluster_has_reference_rate_pct")),
        "marker_species_oracle_supported_rate_pct": as_float(row.get("species_oracle_supported_rate_pct")),
        "marker_genus_oracle_supported_rate_pct": as_float(row.get("genus_oracle_supported_rate_pct")),
        "marker_family_oracle_supported_rate_pct": as_float(row.get("family_oracle_supported_rate_pct")),
        "marker_order_oracle_supported_rate_pct": as_float(row.get("order_oracle_supported_rate_pct")),
    }


def merge_predictions(predictions_path: Path, sample_map: pd.DataFrame, split_by: str) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path)
    if "query_processid" not in pred.columns:
        pred["query_processid"] = pred.get("processid")

    sample_cols = [
        "sample_id",
        "query_processid",
        "true_tree_label",
        "true_species_name",
        "true_genus_name",
        "true_family_name",
        "true_order_name",
        "read_count",
        "asv_count",
        split_by,
    ]
    sample_cols = [column for column in sample_cols if column in sample_map.columns]

    if "sample_id" in pred.columns:
        merged = pred.merge(
            sample_map[sample_cols],
            on=["sample_id", "query_processid"],
            how="left",
            suffixes=("", "_sample"),
        )
        for column in ("true_tree_label", "true_species_name", "true_genus_name", "true_family_name", "true_order_name"):
            sample_column = f"{column}_sample"
            if sample_column in merged.columns:
                merged[column] = merged[column].where(merged[column].map(nonempty), merged[sample_column])
    else:
        pred_small = pred.drop_duplicates("query_processid")
        merged = sample_map[sample_cols].merge(pred_small, on="query_processid", how="left", suffixes=("_sample", ""))
        for column in ("true_tree_label", "true_species_name", "true_genus_name", "true_family_name", "true_order_name"):
            sample_column = f"{column}_sample"
            if sample_column in merged.columns:
                merged[column] = merged[sample_column]
    return merged


def build_feature_rows(
    method_name: str,
    predictions_path: Path,
    sample_map: pd.DataFrame,
    taxonomy: dict[str, dict[str, object]],
    split_by: str,
    marker_ceiling: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = method_metadata(method_name)
    merged = merge_predictions(predictions_path, sample_map, split_by)
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        labels = [normalize_label(item) for item in parse_json_list(row.get("top_tree_labels")) if nonempty(item)]
        scores = [as_float(item) for item in parse_json_list(row.get("top_scores"))]
        scores = [score for score in scores if score is not None]
        pred_label = normalize_label(row.get("pred_tree_label")) or (labels[0] if labels else "")
        top1_label = labels[0] if labels else pred_label
        top2_label = labels[1] if len(labels) > 1 else ""
        top1_score = as_float(row.get("pred_score"))
        if top1_score is None and scores:
            top1_score = scores[0]
        top2_score = scores[1] if len(scores) > 1 else None
        score_margin = top1_score - top2_score if top1_score is not None and top2_score is not None else None
        true_label = normalize_label(row.get("true_tree_label"))

        feature: dict[str, Any] = {
            "method": method_name,
            **meta,
            "predictions": rel(predictions_path),
            "sample_id": clean(row.get("sample_id")),
            "query_processid": clean(row.get("query_processid")),
            "split_by": split_by,
            "split_key": clean(row.get(split_by)),
            "calibration_split": split_label(row.get(split_by)),
            "read_count": row.get("read_count", ""),
            "asv_count": row.get("asv_count", ""),
            "true_tree_label": true_label,
            "top1_tree_label": top1_label,
            "top2_tree_label": top2_label,
            "pred_tree_label": pred_label,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "top1_top2_score_margin": score_margin,
            "score_direction": "higher_is_better_by_output_order",
            "top_candidate_count": len(labels),
            "top1_genus": rank_value(top1_label, "genus", taxonomy),
            "top1_family": rank_value(top1_label, "family", taxonomy),
            "top1_order": rank_value(top1_label, "order", taxonomy),
            "top1_has_reference_sequence": taxonomy.get(top1_label, {}).get("has_reference_sequence", ""),
            "top1_reference_sequence_count": taxonomy.get(top1_label, {}).get("reference_sequence_count", ""),
            "true_has_reference_sequence": taxonomy.get(true_label, {}).get("has_reference_sequence", ""),
            "true_reference_sequence_count": taxonomy.get(true_label, {}).get("reference_sequence_count", ""),
            **marker_ceiling,
        }
        for rank in RANKS:
            true_value = true_rank_value(row, rank, taxonomy)
            pred_value = rank_value(top1_label, rank, taxonomy)
            feature[f"true_{rank}"] = true_value
            feature[f"top1_{rank}"] = pred_value
            feature[f"{rank}_eligible"] = bool(true_value and pred_value)
            feature[f"{rank}_top1_correct"] = bool(true_value and pred_value and true_value == pred_value)
        rows.append(feature)

    summary = {
        "method": method_name,
        **meta,
        "rows": int(len(rows)),
        "calibration_rows": int(sum(row["calibration_split"] == "calibration" for row in rows)),
        "evaluation_rows": int(sum(row["calibration_split"] == "evaluation" for row in rows)),
        "predictions": rel(predictions_path),
    }
    return rows, summary


def summarize_rank_correctness(features: pd.DataFrame, calibration: pd.DataFrame) -> list[dict[str, Any]]:
    calibration_lookup: dict[tuple[str, str, float], pd.Series] = {}
    if not calibration.empty:
        for _, row in calibration.iterrows():
            key = (clean(row.get("method")), clean(row.get("rank")), float(row.get("target_accuracy_pct")))
            calibration_lookup[key] = row

    rows: list[dict[str, Any]] = []
    group_cols = ["method", "encoder", "context", "prior_source", "prior_weight", "evidence_arm", "calibration_split"]
    for key, group in features.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        for rank in RANKS:
            eligible = group[group[f"{rank}_eligible"]]
            out: dict[str, Any] = {
                **meta,
                "rank": rank,
                "n_query": int(len(group)),
                "n_eligible": int(len(eligible)),
                "top1_accuracy_pct": 100.0 * eligible[f"{rank}_top1_correct"].sum() / len(eligible) if len(eligible) else np.nan,
                "mean_top1_score": group["top1_score"].mean(),
                "median_top1_score": group["top1_score"].median(),
                "mean_top1_top2_score_margin": group["top1_top2_score_margin"].mean(),
            }
            for target in (50.0, 70.0, 80.0):
                cal = calibration_lookup.get((clean(meta["method"]), rank, target))
                out[f"target{int(target)}_status"] = clean(cal.get("status")) if cal is not None else ""
                out[f"target{int(target)}_eval_assignment_rate_pct"] = (
                    cal.get("eval_assignment_rate_pct") if cal is not None else ""
                )
                out[f"target{int(target)}_eval_rank_accuracy_pct"] = (
                    cal.get("eval_rank_accuracy_pct") if cal is not None else ""
                )
            rows.append(out)
    return rows


def build_method_design(
    method_rows: list[dict[str, Any]],
    evidence: pd.DataFrame,
    calibration: pd.DataFrame,
    marker_ceiling: dict[str, Any],
) -> list[dict[str, Any]]:
    best_lookup: dict[tuple[str, str], pd.Series] = {}
    if not evidence.empty:
        for _, row in evidence.iterrows():
            best_lookup[(clean(row.get("best_run_name")), clean(row.get("rank")))] = row

    calibration_available = set()
    if not calibration.empty:
        available = calibration[calibration["status"].map(clean) == "available"]
        calibration_available = set(zip(available["method"].map(clean), available["rank"].map(clean)))

    rows: list[dict[str, Any]] = []
    for method in method_rows:
        for rank in RANKS:
            best = best_lookup.get((clean(method["method"]), rank))
            rows.append(
                {
                    **method,
                    "rank": rank,
                    "existing_asv_top1_pct": best.get("asv_top1_pct") if best is not None else "",
                    "existing_asv_top5_pct": best.get("asv_top5_pct") if best is not None else "",
                    "existing_asv_top10_pct": best.get("asv_top10_pct") if best is not None else "",
                    "has_site_heldout_calibration_threshold": (clean(method["method"]), rank) in calibration_available,
                    **marker_ceiling,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--methods-json", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-by", default="site20")
    parser.add_argument("--resolvability-dataset", default="multisource_teleo")
    parser.add_argument("--resolvability-identity", type=float, default=0.99)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Reading sample map from {rel(args.input_dir / 'sample_query_map.csv')}")
    sample_map = pd.read_csv(args.input_dir / "sample_query_map.csv")
    if args.split_by not in sample_map.columns:
        raise SystemExit(f"--split-by column not found in sample map: {args.split_by}")
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")
    methods = json.loads(args.methods_json.read_text())
    marker_ceiling = load_marker_ceiling(args.source_dir, args.resolvability_dataset, args.resolvability_identity)
    logger.log(f"Using marker ceiling: {marker_ceiling}")

    feature_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for method in methods:
        method_name = method["method"]
        predictions_path = ROOT / method["predictions"]
        if not predictions_path.exists():
            logger.log(f"Skipping missing predictions for {method_name}: {rel(predictions_path)}")
            continue
        logger.log(f"Building posterior features for {method_name}")
        rows, summary = build_feature_rows(
            method_name=method_name,
            predictions_path=predictions_path,
            sample_map=sample_map,
            taxonomy=taxonomy,
            split_by=args.split_by,
            marker_ceiling=marker_ceiling,
        )
        feature_rows.extend(rows)
        method_rows.append(summary)
        logger.log(f"Added {len(rows)} rows for {method_name}")

    features = pd.DataFrame(feature_rows)
    feature_path = args.output_dir / "eco_phylo_posterior_query_features.csv.gz"
    logger.log(f"Writing query feature table to {rel(feature_path)}")
    with gzip.open(feature_path, "wt", newline="") as handle:
        features.to_csv(handle, index=False)

    feature_sample_path = args.output_dir / "eco_phylo_posterior_query_features_sample.csv"
    logger.log(f"Writing query feature sample to {rel(feature_sample_path)}")
    features.head(1000).to_csv(feature_sample_path, index=False)

    calibration_path = args.calibration_dir / "global_edna_independent_rank_calibration_summary.csv"
    calibration = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    evidence_path = args.source_dir / "edna_evidence_best_by_rank.csv"
    evidence = pd.read_csv(evidence_path) if evidence_path.exists() else pd.DataFrame()

    rank_rows = summarize_rank_correctness(features, calibration)
    rank_path = args.output_dir / "eco_phylo_posterior_rank_correctness_summary.csv"
    logger.log(f"Writing rank correctness summary to {rel(rank_path)}")
    write_csv(
        rank_path,
        rank_rows,
        [
            "method",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "evidence_arm",
            "calibration_split",
            "rank",
            "n_query",
            "n_eligible",
            "top1_accuracy_pct",
            "mean_top1_score",
            "median_top1_score",
            "mean_top1_top2_score_margin",
            "target50_status",
            "target50_eval_assignment_rate_pct",
            "target50_eval_rank_accuracy_pct",
            "target70_status",
            "target70_eval_assignment_rate_pct",
            "target70_eval_rank_accuracy_pct",
            "target80_status",
            "target80_eval_assignment_rate_pct",
            "target80_eval_rank_accuracy_pct",
        ],
    )

    design_rows = build_method_design(method_rows, evidence, calibration, marker_ceiling)
    design_path = args.output_dir / "eco_phylo_posterior_method_design.csv"
    logger.log(f"Writing method design table to {rel(design_path)}")
    write_csv(
        design_path,
        design_rows,
        [
            "method",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "evidence_arm",
            "evidence_family",
            "sequence_score_available",
            "tree_candidate_evidence_available",
            "geography_prior_available",
            "cooccurrence_prior_available",
            "rows",
            "calibration_rows",
            "evaluation_rows",
            "predictions",
            "rank",
            "existing_asv_top1_pct",
            "existing_asv_top5_pct",
            "existing_asv_top10_pct",
            "has_site_heldout_calibration_threshold",
            "marker_resolvability_dataset",
            "marker_resolvability_mode",
            "marker_identity_threshold",
            "marker_query_cluster_has_reference_rate_pct",
            "marker_species_oracle_supported_rate_pct",
            "marker_genus_oracle_supported_rate_pct",
            "marker_family_oracle_supported_rate_pct",
            "marker_order_oracle_supported_rate_pct",
        ],
    )

    manifest = {
        "generated_by": rel(Path(__file__)),
        "input_dir": rel(args.input_dir),
        "methods_json": rel(args.methods_json),
        "source_dir": rel(args.source_dir),
        "calibration_dir": rel(args.calibration_dir),
        "output_dir": rel(args.output_dir),
        "split_by": args.split_by,
        "methods_seen": len(method_rows),
        "query_feature_rows": int(len(features)),
        "rank_summary_rows": len(rank_rows),
        "method_design_rows": len(design_rows),
        "marker_ceiling": marker_ceiling,
        "outputs": {
            "query_features": rel(feature_path),
            "query_features_sample": rel(feature_sample_path),
            "rank_correctness_summary": rel(rank_path),
            "method_design": rel(design_path),
        },
        "notes": [
            "This is a posterior input/design artifact, not a trained Eco-Phylo posterior.",
            "Scores and rankings are copied from existing prediction files; no new metric is invented here.",
            "Calibration split follows the existing deterministic site20 split used by the independent rank calibration.",
            "Marker ceilings are dataset-level 12S resolvability diagnostics, not per-query oracle labels.",
        ],
    }
    manifest_path = args.output_dir / "eco_phylo_posterior_manifest.json"
    logger.log(f"Writing manifest to {rel(manifest_path)}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
