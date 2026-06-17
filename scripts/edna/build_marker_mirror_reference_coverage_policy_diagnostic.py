#!/usr/bin/env python3
"""Diagnostic family/genus policy using lineage-reference coverage features.

Exp 119/121 showed that threshold-only and set-valued family/genus repairs do
not transfer cleanly. Exp 127 identified lineage-specific reference coverage as
a genuinely new evidence source. This script tests that idea directly:

* build candidate-policy rows from existing BLASTN/VSEARCH/MarkerMirror policy
  evidence;
* join candidate-lineage coverage features derived from the current benchmark
  reference/curation tables;
* train a rank-specific policy-row correctness model;
* calibrate thresholds on species-disjoint calibration rows;
* evaluate on species-disjoint held-out rows.

This is a diagnostic. The lineage coverage rows are benchmark/reference-table
features and should not be treated as global absence statements.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from progress_logging import ProgressLogger, default_log_path


SCRIPT_NAME = "build_marker_mirror_reference_coverage_policy_diagnostic"
ROOT = Path(".")
SOURCE_TABLE_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
OUTPUT_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "marker_mirror_bridge"
    / "reference_coverage_policy_diagnostic"
)
RANKS = ("genus", "family", "order")
TARGETS = (0.95, 0.99)


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


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def read_csv(path: Path, logger: ProgressLogger, **kwargs: Any) -> pd.DataFrame:
    logger.log(f"loading {path}")
    df = pd.read_csv(path, **kwargs)
    logger.log(f"loaded rows={len(df):,} cols={len(df.columns):,}")
    return df


def load_query_meta(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "query_id": frame["processid"].astype(str),
            "source": frame["source"].astype(str),
            "query_tree_label": frame["tree_label"].astype(str),
            "query_species": frame["species_name"].astype(str),
            "query_genus": frame["genus_name"].map(clean),
            "query_family": frame["family_name"].map(clean),
            "query_order": frame["order_name"].map(clean),
        }
    )


def weighted_mean_pct(df: pd.DataFrame, value_col: str, weight_col: str = "n_queries") -> float:
    if df.empty or value_col not in df or weight_col not in df:
        return math.nan
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    denom = weights.sum()
    return float((values * weights).sum() / denom) if denom else math.nan


def bool_weighted_pct(df: pd.DataFrame, value_col: str, weight_col: str = "n_queries") -> float:
    if df.empty or value_col not in df or weight_col not in df:
        return math.nan
    work = df.copy()
    work[value_col] = work[value_col].astype(bool).astype(float) * 100.0
    return weighted_mean_pct(work, value_col, weight_col)


def build_lineage_feature_table(curation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lineage_specs = {
        "genus": "query_genus",
        "family": "query_family",
        "order": "query_order",
    }
    total_queries = max(float(curation["n_queries"].sum()), 1.0)
    for rank, column in lineage_specs.items():
        for lineage, group in curation.groupby(column, dropna=False):
            prefix = f"{rank}_lineage"
            n_queries = int(group["n_queries"].sum())
            rows.append(
                {
                    "assigned_rank": rank,
                    "assigned_taxon": clean(lineage),
                    f"{prefix}_species_group_count": int(group["query_tree_label"].nunique()),
                    f"{prefix}_query_count": n_queries,
                    f"{prefix}_query_pct": 100.0 * n_queries / total_queries,
                    f"{prefix}_species_present_12s_pct": bool_weighted_pct(
                        group, "query_species_present_in_12s_reference"
                    ),
                    f"{prefix}_species_present_16s_pct": bool_weighted_pct(
                        group, "query_species_present_in_16s_reference"
                    ),
                    f"{prefix}_marker_mirror_genus_hit_pct": weighted_mean_pct(
                        group, "marker_mirror_genus_hit_pct"
                    ),
                    f"{prefix}_same_marker_genus_hit_pct": weighted_mean_pct(
                        group, "same_marker_genus_hit_pct"
                    ),
                    f"{prefix}_union_genus_hit_pct": weighted_mean_pct(
                        group, "union_genus_hit_pct"
                    ),
                    f"{prefix}_union_family_hit_pct": weighted_mean_pct(
                        group, "union_family_hit_pct"
                    ),
                    f"{prefix}_union_order_hit_pct": weighted_mean_pct(
                        group, "union_order_hit_pct"
                    ),
                    f"{prefix}_static_assigned_pct": weighted_mean_pct(
                        group, "static_family_order_assigned_pct"
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_policy_features(policy_rows: pd.DataFrame, meta: pd.DataFrame, lineage: pd.DataFrame) -> pd.DataFrame:
    rows = policy_rows[policy_rows["rank"].isin(RANKS)].copy()
    rows["assigned_rank"] = rows["rank"].map(clean)
    rows["assigned_taxon"] = rows["assigned_taxon"].map(clean)
    rows = rows[rows["assigned_taxon"].astype(bool)].copy()
    rows = rows.merge(
        meta[
            [
                "query_id",
                "query_tree_label",
                "query_species",
                "query_genus",
                "query_family",
                "query_order",
            ]
        ],
        on="query_id",
        how="left",
    )
    rows = rows.merge(lineage, on=["assigned_rank", "assigned_taxon"], how="left")

    rows["confidence"] = pd.to_numeric(rows["confidence"], errors="coerce")
    rows["policy_uses_blast"] = rows["policy"].str.contains("blast", case=False, na=False)
    rows["policy_uses_vsearch"] = rows["policy"].str.contains("vsearch", case=False, na=False)
    rows["policy_uses_marker_mirror"] = rows["policy"].str.startswith("mm_", na=False)
    rows["policy_requires_agreement"] = rows["policy"].str.contains("agree", case=False, na=False)
    rows["policy_top1"] = rows["policy"].str.contains("top1", case=False, na=False)
    rows["policy_top10"] = rows["policy"].str.contains("top10", case=False, na=False)
    rows["confidence_logit"] = np.log(
        np.clip(rows["confidence"].fillna(0.0), 1e-6, 1 - 1e-6)
        / np.clip(1 - rows["confidence"].fillna(0.0), 1e-6, 1)
    )
    rows["correct"] = rows["correct"].astype(bool)
    return rows


def split_species(
    species: list[str] | np.ndarray,
    rng: np.random.Generator,
    train_fraction: float,
    calibration_fraction: float,
) -> tuple[set[str], set[str], set[str]]:
    values = np.array(sorted(clean(value) for value in species if clean(value)))
    rng.shuffle(values)
    n_train = max(1, int(round(len(values) * train_fraction)))
    n_cal = max(1, int(round(len(values) * calibration_fraction)))
    if n_train + n_cal >= len(values):
        n_cal = max(1, len(values) - n_train - 1)
    train = set(values[:n_train])
    cal = set(values[n_train : n_train + n_cal])
    eval_ = set(values[n_train + n_cal :])
    return train, cal, eval_


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical = ["source", "policy"]
    blocked = {
        "query_id",
        "rank",
        "assigned_rank",
        "assigned_taxon",
        "correct",
        "query_tree_label",
        "query_species",
        "query_genus",
        "query_family",
        "query_order",
    }
    for column in frame.columns:
        if column in blocked or column in categorical:
            continue
        if pd.api.types.is_bool_dtype(frame[column]) or pd.api.types.is_numeric_dtype(frame[column]):
            if frame[column].notna().sum() > 0:
                numeric.append(column)
    return sorted(numeric), categorical


def active_numeric_columns(frames: list[pd.DataFrame], numeric: list[str]) -> list[str]:
    active: list[str] = []
    for column in numeric:
        if any(frame[column].notna().sum() > 0 for frame in frames):
            active.append(column)
    return active


def encode(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    eval_: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    frames = []
    for frame in (train, cal, eval_):
        x_num = frame[numeric].copy()
        for column in x_num.columns:
            if pd.api.types.is_bool_dtype(x_num[column]):
                x_num[column] = x_num[column].astype(int)
        x_cat = pd.get_dummies(frame[categorical].astype(str), columns=categorical, dummy_na=False)
        frames.append(pd.concat([x_num.reset_index(drop=True), x_cat.reset_index(drop=True)], axis=1))
    columns = sorted(set().union(*(set(frame.columns) for frame in frames)))
    aligned = [frame.reindex(columns=columns, fill_value=0) for frame in frames]
    return aligned[0], aligned[1], aligned[2], columns


def make_model(model_type: str, seed: int):
    if model_type == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.04,
                l2_regularization=0.08,
                random_state=seed,
            ),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
    )


def fit_threshold(cal: pd.DataFrame, target: float, min_assignments: int) -> dict[str, Any]:
    if cal.empty or "prob_correct" not in cal:
        return {"threshold": math.inf, "fit_status": "no_calibration_rows"}
    probs = cal["prob_correct"].to_numpy(dtype=float)
    labels = cal["correct"].astype(bool).to_numpy()
    order = np.argsort(probs)[::-1]
    probs = probs[order]
    labels = labels[order]
    best = None
    fallback = None
    for threshold in np.unique(probs)[::-1]:
        keep = probs >= threshold
        if keep.sum() < min_assignments:
            continue
        precision = float(labels[keep].mean())
        coverage = pct(int(keep.sum()), len(labels))
        row = (float(threshold), precision, coverage, int(keep.sum()))
        if fallback is None or (precision, coverage) > (fallback[1], fallback[2]):
            fallback = row
        if precision >= target:
            best = row
    threshold, precision, coverage, n = best or fallback or (math.inf, math.nan, math.nan, 0)
    return {
        "threshold": threshold,
        "fit_precision": precision,
        "fit_coverage_pct": coverage,
        "fit_n": n,
        "fit_status": "target_met" if best else "target_not_met_best_available",
    }


def evaluate(eval_rows: pd.DataFrame, threshold: float, n_eval_queries: int, target: float) -> dict[str, Any]:
    if math.isinf(threshold):
        assigned = eval_rows.iloc[0:0].copy()
    else:
        candidates = eval_rows[eval_rows["prob_correct"] >= threshold].copy()
        assigned = (
            candidates.sort_values(["query_id", "prob_correct", "confidence"], ascending=[True, False, False])
            .drop_duplicates("query_id")
            .copy()
        )
    n_assigned = len(assigned)
    n_correct = int(assigned["correct"].astype(bool).sum()) if n_assigned else 0
    precision = n_correct / n_assigned if n_assigned else math.nan
    return {
        "n_eval_queries": int(n_eval_queries),
        "n_assigned": int(n_assigned),
        "n_correct": int(n_correct),
        "coverage_pct": pct(n_assigned, n_eval_queries),
        "assigned_precision": precision,
        "target_met": bool(n_assigned > 0 and precision >= target),
    }


def run_nested(
    rows: pd.DataFrame,
    ranks: list[str],
    targets: list[float],
    model_type: str,
    repeats: int,
    train_fraction: float,
    calibration_fraction: float,
    seed: int,
    min_assignments: int,
    logger: ProgressLogger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_species = sorted(rows["query_tree_label"].dropna().astype(str).unique())
    numeric, categorical = feature_columns(rows)
    logger.log(f"feature columns numeric={len(numeric)} categorical={len(categorical)}")
    per_split_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    assignment_rows: list[pd.DataFrame] = []

    for repeat in range(repeats):
        train_species, cal_species, eval_species = split_species(
            all_species, rng, train_fraction, calibration_fraction
        )
        for rank in ranks:
            rank_rows = rows[rows["assigned_rank"].eq(rank)].copy()
            train = rank_rows[rank_rows["query_tree_label"].isin(train_species)].copy()
            cal = rank_rows[rank_rows["query_tree_label"].isin(cal_species)].copy()
            eval_ = rank_rows[rank_rows["query_tree_label"].isin(eval_species)].copy()
            n_eval_queries = int(
                rows[rows["query_tree_label"].isin(eval_species)]["query_id"].nunique()
            )
            if train.empty or cal.empty or eval_["correct"].nunique() < 1 or train["correct"].nunique() < 2:
                continue
            rank_numeric = active_numeric_columns([train, cal, eval_], numeric)
            x_train, x_cal, x_eval, encoded_cols = encode(
                train, cal, eval_, rank_numeric, categorical
            )
            model = make_model(model_type, seed + repeat)
            model.fit(x_train, train["correct"].astype(int))
            cal = cal.copy()
            eval_ = eval_.copy()
            cal["prob_correct"] = model.predict_proba(x_cal)[:, 1]
            eval_["prob_correct"] = model.predict_proba(x_eval)[:, 1]
            for target in targets:
                threshold = fit_threshold(cal, target, min_assignments)
                result = evaluate(eval_, threshold["threshold"], n_eval_queries, target)
                per_split_rows.append(
                    {
                        "repeat": repeat,
                        "rank": rank,
                        "target": target,
                        "model_type": model_type,
                        "train_rows": int(len(train)),
                        "calibration_rows": int(len(cal)),
                        "eval_rows": int(len(eval_)),
                        "encoded_feature_count": int(len(encoded_cols)),
                        **threshold,
                        **result,
                    }
                )
                threshold_rows.append(
                    {
                        "repeat": repeat,
                        "rank": rank,
                        "target": target,
                        "model_type": model_type,
                        **threshold,
                    }
                )
                assigned = eval_[eval_["prob_correct"] >= threshold["threshold"]].copy()
                if not assigned.empty:
                    assigned = (
                        assigned.sort_values(
                            ["query_id", "prob_correct", "confidence"],
                            ascending=[True, False, False],
                        )
                        .drop_duplicates("query_id")
                        .copy()
                    )
                    assigned["repeat"] = repeat
                    assigned["target"] = target
                    assignment_rows.append(
                        assigned[
                            [
                                "repeat",
                                "target",
                                "query_id",
                                "source",
                                "query_tree_label",
                                "assigned_rank",
                                "assigned_taxon",
                                "policy",
                                "confidence",
                                "prob_correct",
                                "correct",
                            ]
                        ]
                    )
        if repeat == 0 or (repeat + 1) % 10 == 0:
            logger.log(f"completed repeat {repeat + 1}/{repeats}")

    per_split = pd.DataFrame(per_split_rows)
    thresholds = pd.DataFrame(threshold_rows)
    assignments = pd.concat(assignment_rows, ignore_index=True) if assignment_rows else pd.DataFrame()
    feature_inventory = pd.DataFrame(
        [{"feature": feature, "feature_type": "numeric"} for feature in numeric]
        + [{"feature": feature, "feature_type": "categorical"} for feature in categorical]
    )
    return per_split, thresholds, assignments, feature_inventory


def summarize(per_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in per_split.groupby(["rank", "target", "model_type"], sort=False):
        rank, target, model_type = keys
        rows.append(
            {
                "rank": rank,
                "target": float(target),
                "model_type": model_type,
                "outer_repeats": int(len(group)),
                "mean_coverage_pct": float(group["coverage_pct"].mean()),
                "median_coverage_pct": float(group["coverage_pct"].median()),
                "mean_assigned_precision": float(group["assigned_precision"].mean()),
                "median_assigned_precision": float(group["assigned_precision"].median()),
                "precision_p05": float(group["assigned_precision"].quantile(0.05)),
                "min_assigned_precision": float(group["assigned_precision"].min(skipna=True)),
                "target_met_rate_pct": 100.0 * float(group["target_met"].mean()),
                "mean_n_assigned": float(group["n_assigned"].mean()),
                "recommendation": "",
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["recommendation"] = np.where(
            (out["target"] >= 0.99) & (out["target_met_rate_pct"] >= 100.0),
            "candidate_for_followup_validation",
            "diagnostic_only_do_not_enable",
        )
        out = out.sort_values(
            ["target_met_rate_pct", "mean_coverage_pct", "mean_assigned_precision"],
            ascending=[False, False, False],
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-rows",
        type=Path,
        default=SOURCE_TABLE_DIR / "marker_mirror_blast_vsearch_calibration_repair_policy_rows.csv.gz",
    )
    parser.add_argument(
        "--query-table",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "zero_shot_queries.csv",
    )
    parser.add_argument(
        "--curation-table",
        type=Path,
        default=SOURCE_TABLE_DIR / "marker_mirror_union_reference_curation_priorities.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--source-table-dir", type=Path, default=SOURCE_TABLE_DIR)
    parser.add_argument("--ranks", default="genus,family,order")
    parser.add_argument("--targets", default="0.95,0.99")
    parser.add_argument("--model-type", choices=("hgb", "logistic"), default="hgb")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--min-calibration-assignments", type=int, default=25)
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--log-file", type=Path, default=default_log_path(ROOT, SCRIPT_NAME))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_table_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file)
    logger.start(SCRIPT_NAME)

    ranks = [rank.strip() for rank in args.ranks.split(",") if rank.strip()]
    targets = [float(value.strip()) for value in args.targets.split(",") if value.strip()]
    logger.log(f"ranks={ranks} targets={targets} repeats={args.repeats} model={args.model_type}")

    meta = load_query_meta(args.query_table)
    logger.log(f"loaded query meta rows={len(meta):,}")
    curation = read_csv(args.curation_table, logger)
    lineage = build_lineage_feature_table(curation)
    lineage_path = args.output_dir / "marker_mirror_policy_lineage_reference_features.csv"
    lineage.to_csv(lineage_path, index=False)
    logger.log(f"wrote {lineage_path} rows={len(lineage):,}")

    policies = read_csv(args.policy_rows, logger)
    rows = add_policy_features(policies, meta, lineage)
    feature_rows_path = args.output_dir / "marker_mirror_reference_coverage_policy_rows.csv.gz"
    rows.to_csv(feature_rows_path, index=False)
    logger.log(f"wrote {feature_rows_path} rows={len(rows):,}")

    per_split, thresholds, assignments, feature_inventory = run_nested(
        rows=rows,
        ranks=ranks,
        targets=targets,
        model_type=args.model_type,
        repeats=args.repeats,
        train_fraction=args.train_fraction,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        min_assignments=args.min_calibration_assignments,
        logger=logger,
    )
    summary = summarize(per_split)

    prefix = "marker_mirror_reference_coverage_policy_diagnostic"
    outputs = {
        "summary": args.source_table_dir / f"{prefix}_summary.csv",
        "per_split": args.source_table_dir / f"{prefix}_per_split.csv",
        "thresholds": args.source_table_dir / f"{prefix}_thresholds.csv",
        "features": args.source_table_dir / f"{prefix}_features.csv",
        "lineage_features": args.source_table_dir / f"{prefix}_lineage_features.csv",
        "manifest": args.source_table_dir / f"{prefix}_manifest.json",
    }
    local_outputs = {
        "summary": args.output_dir / f"{prefix}_summary.csv",
        "per_split": args.output_dir / f"{prefix}_per_split.csv",
        "thresholds": args.output_dir / f"{prefix}_thresholds.csv",
        "features": args.output_dir / f"{prefix}_features.csv",
        "assignments": args.output_dir / f"{prefix}_assignments.csv.gz",
        "lineage_features": lineage_path,
        "feature_rows": feature_rows_path,
        "manifest": args.output_dir / f"{prefix}_manifest.json",
    }

    summary.to_csv(outputs["summary"], index=False)
    per_split.to_csv(outputs["per_split"], index=False)
    thresholds.to_csv(outputs["thresholds"], index=False)
    feature_inventory.to_csv(outputs["features"], index=False)
    lineage.to_csv(outputs["lineage_features"], index=False)
    summary.to_csv(local_outputs["summary"], index=False)
    per_split.to_csv(local_outputs["per_split"], index=False)
    thresholds.to_csv(local_outputs["thresholds"], index=False)
    feature_inventory.to_csv(local_outputs["features"], index=False)
    if not assignments.empty:
        assignments.to_csv(local_outputs["assignments"], index=False)

    manifest = {
        "script": f"scripts/edna/{SCRIPT_NAME}.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "policy_rows": str(args.policy_rows),
            "query_table": str(args.query_table),
            "curation_table": str(args.curation_table),
        },
        "outputs": {key: str(value) for key, value in {**outputs, **local_outputs}.items()},
        "model_type": args.model_type,
        "ranks": ranks,
        "targets": targets,
        "repeats": args.repeats,
        "claim_boundary": (
            "Diagnostic only. Uses benchmark/reference-table lineage coverage "
            "features to test whether new evidence stabilizes family/genus; "
            "does not enable family/genus calls."
        ),
    }
    for path in [outputs["manifest"], local_outputs["manifest"]]:
        with path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    logger.log(f"wrote summary {outputs['summary']}")
    if not summary.empty:
        logger.log("summary top rows:\n" + summary.head(8).to_string(index=False))
    logger.done(SCRIPT_NAME)


if __name__ == "__main__":
    main()
