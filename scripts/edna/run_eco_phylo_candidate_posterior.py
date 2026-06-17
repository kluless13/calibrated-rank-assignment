#!/usr/bin/env python3
"""Fit a candidate-level Eco-Phylo posterior over Global_eDNA evidence.

This consumes `build_eco_phylo_candidate_posterior_inputs.py` output. Unlike
the first posterior prototype, the unit of inference is a candidate species row
from a method/query ranking, not the already-compressed top-1 method output.

The model is deliberately conservative:

- train on deterministic calibration sites and evaluate on held-out sites;
- do not use true taxon labels as features;
- use only inference-time evidence columns;
- select one candidate per query/rank before applying rank thresholds.
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSTERIOR_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "eco_phylo_posterior"
    / "candidate_level"
)
DEFAULT_FEATURES = DEFAULT_POSTERIOR_DIR / "eco_phylo_candidate_features_top5.csv.gz"
RANKS = ("species", "genus", "family", "order")
TARGET_ACCURACIES = (50.0, 60.0, 70.0, 80.0, 90.0, 95.0)
GROUP_COLS = ("sample_id", "query_processid")
BASE_NUMERIC_FEATURES = (
    "prior_weight_num",
    "candidate_rank_num",
    "candidate_rank_inverse",
    "candidate_score",
    "candidate_score_method_z",
    "candidate_score_delta_from_top1",
    "candidate_score_ratio_to_top1",
    "sequence_only_score",
    "sequence_only_score_method_z",
    "sequence_only_rank_num",
    "sequence_only_rank_inverse",
    "sequence_only_available_num",
    "blast_candidate_pident",
    "blast_candidate_rank_num",
    "blast_candidate_rank_inverse",
    "blast_candidate_in_top50_num",
    "pdistance_available_num",
    "candidate_pdistance",
    "candidate_sequence_identity",
    "sequence_evidence_available_num",
    "sequence_evidence_reference_count",
    "sequence_evidence_overlap_bases",
    "sequence_evidence_mismatches",
    "tree_evidence_available_num",
    "candidate_is_top1_num",
    "tree_distance_to_top1_candidate",
    "tree_distance_to_candidate_set_min",
    "tree_distance_to_candidate_set_mean",
    "tree_distance_to_candidate_set_max",
    "topk_pairwise_tree_distance_min",
    "topk_pairwise_tree_distance_mean",
    "topk_pairwise_tree_distance_max",
    "topk_unique_genus_count",
    "topk_unique_family_count",
    "topk_unique_order_count",
    "same_genus_as_top1_num",
    "same_genus_topk_fraction",
    "same_family_as_top1_num",
    "same_family_topk_fraction",
    "same_order_as_top1_num",
    "same_order_topk_fraction",
    "rls_prior_log1p",
    "rls_prior_supported_num",
    "obis_prior_log1p",
    "obis_prior_supported_num",
    "log_read_count",
    "log_asv_count",
    "candidate_has_reference_sequence_num",
    "log_candidate_reference_sequence_count",
    "marker_normalized_sequence_length",
    "marker_observed_exact_cluster_species_count",
    "marker_reference_exact_cluster_found_num",
    "marker_observed_exact_cluster_found_num",
    "marker_true_species_in_observed_exact_cluster_num",
)
BASE_CATEGORICAL_FEATURES = (
    "method",
    "encoder",
    "context",
    "prior_source",
    "evidence_family",
    "rls_prior_source",
    "obis_prior_source",
    "marker_deepest_supported_rank",
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None


def group_key(frame: pd.DataFrame) -> pd.Series:
    return frame["sample_id"].astype(str) + "\t" + frame["query_processid"].astype(str)


def stable_fraction(value: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def read_features(
    path: Path,
    logger: ProgressLogger,
    max_query_groups: int,
    random_state: int,
    chunksize: int,
    sequence_evidence: Path | None,
    tree_evidence: Path | None,
) -> pd.DataFrame:
    usecols = [
        "method",
        "encoder",
        "context",
        "prior_source",
        "prior_weight",
        "evidence_family",
        "sample_id",
        "query_processid",
        "calibration_split",
        "read_count",
        "asv_count",
        "true_tree_label",
        "candidate_tree_label",
        "candidate_rank",
        "candidate_score",
        "candidate_score_method_z",
        "candidate_score_delta_from_top1",
        "candidate_score_ratio_to_top1",
        "sequence_only_score",
        "sequence_only_score_method_z",
        "sequence_only_rank",
        "sequence_only_available",
        "blast_candidate_pident",
        "blast_candidate_rank",
        "blast_candidate_in_top50",
        "pdistance_available",
        "candidate_pdistance",
        "rls_prior_log1p",
        "rls_prior_supported",
        "rls_prior_source",
        "obis_prior_log1p",
        "obis_prior_supported",
        "obis_prior_source",
        "candidate_has_reference_sequence",
        "candidate_reference_sequence_count",
        "marker_normalized_sequence_length",
        "marker_observed_exact_cluster_species_count",
        "marker_reference_exact_cluster_found",
        "marker_observed_exact_cluster_found",
        "marker_true_species_in_observed_exact_cluster",
        "marker_deepest_supported_rank",
    ]
    for rank in RANKS:
        usecols.extend([f"true_{rank}", f"candidate_{rank}", f"{rank}_eligible", f"{rank}_candidate_correct"])
    logger.log(f"Reading candidate posterior features from {rel(path)}")
    if max_query_groups <= 0:
        features = pd.read_csv(path, usecols=usecols, low_memory=False)
        features = merge_sequence_evidence(features, sequence_evidence, logger)
        return merge_tree_evidence(features, tree_evidence, logger)

    logger.log(f"Sampling complete query groups for local run: max_query_groups={max_query_groups:,}")
    group_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=list(GROUP_COLS), chunksize=chunksize):
        group_frames.append(chunk.drop_duplicates())
    groups = pd.concat(group_frames, ignore_index=True).drop_duplicates()
    if len(groups) > max_query_groups:
        groups = groups.sample(n=max_query_groups, random_state=random_state)
    selected = set(group_key(groups))
    logger.log(f"Selected {len(selected):,} complete query groups")

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        keep = group_key(chunk).isin(selected)
        if keep.any():
            frames.append(chunk.loc[keep].copy())
    sampled = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)
    logger.log(f"Read {len(sampled):,} candidate rows after complete-query sampling")
    sampled = merge_sequence_evidence(sampled, sequence_evidence, logger)
    return merge_tree_evidence(sampled, tree_evidence, logger)


def merge_sequence_evidence(features: pd.DataFrame, sequence_evidence: Path | None, logger: ProgressLogger) -> pd.DataFrame:
    if sequence_evidence is None:
        features["candidate_sequence_identity"] = np.nan
        features["sequence_evidence_available"] = 0
        features["sequence_evidence_reference_count"] = np.nan
        features["sequence_evidence_overlap_bases"] = np.nan
        features["sequence_evidence_mismatches"] = np.nan
        return features
    if not sequence_evidence.exists():
        raise SystemExit(f"Sequence-evidence file does not exist: {sequence_evidence}")
    logger.log(f"Merging candidate sequence evidence from {rel(sequence_evidence)}")
    evidence = pd.read_csv(
        sequence_evidence,
        usecols=[
            "query_processid",
            "candidate_tree_label",
            "sequence_evidence_available",
            "sequence_pdistance",
            "sequence_identity",
            "sequence_overlap_bases",
            "sequence_mismatches",
            "sequence_reference_count",
        ],
        low_memory=False,
    )
    evidence = evidence.drop_duplicates(["query_processid", "candidate_tree_label"])
    merged = features.merge(evidence, on=["query_processid", "candidate_tree_label"], how="left")
    available = pd.to_numeric(merged["sequence_evidence_available"], errors="coerce").fillna(0.0)
    sequence_pdistance = pd.to_numeric(merged["sequence_pdistance"], errors="coerce")
    sequence_identity = pd.to_numeric(merged["sequence_identity"], errors="coerce")
    merged["pdistance_available"] = np.maximum(
        pd.to_numeric(merged["pdistance_available"], errors="coerce").fillna(0.0),
        available,
    )
    merged["candidate_pdistance"] = sequence_pdistance.combine_first(
        pd.to_numeric(merged["candidate_pdistance"], errors="coerce")
    )
    merged["candidate_sequence_identity"] = sequence_identity
    merged["sequence_evidence_available"] = available
    merged["sequence_evidence_reference_count"] = pd.to_numeric(
        merged["sequence_reference_count"], errors="coerce"
    )
    merged["sequence_evidence_overlap_bases"] = pd.to_numeric(
        merged["sequence_overlap_bases"], errors="coerce"
    )
    merged["sequence_evidence_mismatches"] = pd.to_numeric(
        merged["sequence_mismatches"], errors="coerce"
    )
    logger.log(
        "Merged sequence evidence into "
        f"{int(available.sum()):,}/{len(merged):,} candidate rows"
    )
    return merged


TREE_EVIDENCE_NUMERIC = (
    "tree_evidence_available",
    "candidate_is_top1",
    "tree_distance_to_top1_candidate",
    "tree_distance_to_candidate_set_min",
    "tree_distance_to_candidate_set_mean",
    "tree_distance_to_candidate_set_max",
    "topk_pairwise_tree_distance_min",
    "topk_pairwise_tree_distance_mean",
    "topk_pairwise_tree_distance_max",
    "topk_unique_genus_count",
    "topk_unique_family_count",
    "topk_unique_order_count",
    "same_genus_as_top1",
    "same_genus_topk_fraction",
    "same_family_as_top1",
    "same_family_topk_fraction",
    "same_order_as_top1",
    "same_order_topk_fraction",
)


def add_empty_tree_evidence(features: pd.DataFrame) -> pd.DataFrame:
    for column in TREE_EVIDENCE_NUMERIC:
        if column not in features.columns:
            features[column] = np.nan
    features["tree_evidence_available"] = features["tree_evidence_available"].fillna(0.0)
    features["candidate_is_top1"] = features["candidate_is_top1"].fillna(0.0)
    features["same_genus_as_top1"] = features["same_genus_as_top1"].fillna(0.0)
    features["same_family_as_top1"] = features["same_family_as_top1"].fillna(0.0)
    features["same_order_as_top1"] = features["same_order_as_top1"].fillna(0.0)
    return features


def merge_tree_evidence(features: pd.DataFrame, tree_evidence: Path | None, logger: ProgressLogger) -> pd.DataFrame:
    if tree_evidence is None:
        return add_empty_tree_evidence(features)
    if not tree_evidence.exists():
        raise SystemExit(f"Tree-evidence file does not exist: {tree_evidence}")
    logger.log(f"Merging candidate tree-neighborhood evidence from {rel(tree_evidence)}")
    usecols = [
        "method",
        "sample_id",
        "query_processid",
        "candidate_tree_label",
        *TREE_EVIDENCE_NUMERIC,
    ]
    evidence = pd.read_csv(tree_evidence, usecols=usecols, low_memory=False)
    evidence = evidence.drop_duplicates(["method", "sample_id", "query_processid", "candidate_tree_label"])
    merged = features.merge(
        evidence,
        on=["method", "sample_id", "query_processid", "candidate_tree_label"],
        how="left",
    )
    merged = add_empty_tree_evidence(merged)
    available = pd.to_numeric(merged["tree_evidence_available"], errors="coerce").fillna(0.0)
    logger.log(f"Merged tree-neighborhood evidence into {int(available.sum()):,}/{len(merged):,} candidate rows")
    return merged


def add_inference_features(features: pd.DataFrame, logger: ProgressLogger) -> pd.DataFrame:
    frame = features.copy()
    for column in BASE_CATEGORICAL_FEATURES:
        frame[column] = frame[column].map(clean).replace("", "none")

    numeric_cols = [
        "prior_weight",
        "candidate_rank",
        "candidate_score",
        "candidate_score_method_z",
        "candidate_score_delta_from_top1",
        "candidate_score_ratio_to_top1",
        "sequence_only_score",
        "sequence_only_score_method_z",
        "sequence_only_rank",
        "sequence_only_available",
        "blast_candidate_pident",
        "blast_candidate_rank",
        "blast_candidate_in_top50",
        "pdistance_available",
        "candidate_pdistance",
        "candidate_sequence_identity",
        "sequence_evidence_available",
        "sequence_evidence_reference_count",
        "sequence_evidence_overlap_bases",
        "sequence_evidence_mismatches",
        *TREE_EVIDENCE_NUMERIC,
        "rls_prior_log1p",
        "rls_prior_supported",
        "obis_prior_log1p",
        "obis_prior_supported",
        "read_count",
        "asv_count",
        "candidate_has_reference_sequence",
        "candidate_reference_sequence_count",
        "marker_normalized_sequence_length",
        "marker_observed_exact_cluster_species_count",
        "marker_reference_exact_cluster_found",
        "marker_observed_exact_cluster_found",
        "marker_true_species_in_observed_exact_cluster",
    ]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["prior_weight_num"] = frame["prior_weight"].fillna(0.0)
    frame["candidate_rank_num"] = frame["candidate_rank"].fillna(999.0)
    frame["candidate_rank_inverse"] = 1.0 / frame["candidate_rank_num"].clip(lower=1.0)
    frame["sequence_only_rank_num"] = frame["sequence_only_rank"].fillna(999.0)
    frame["sequence_only_rank_inverse"] = 1.0 / frame["sequence_only_rank_num"].clip(lower=1.0)
    frame["sequence_only_available_num"] = frame["sequence_only_available"].fillna(0.0)
    frame["blast_candidate_rank_num"] = frame["blast_candidate_rank"].fillna(999.0)
    frame["blast_candidate_rank_inverse"] = 1.0 / frame["blast_candidate_rank_num"].clip(lower=1.0)
    frame["blast_candidate_in_top50_num"] = frame["blast_candidate_in_top50"].fillna(0.0)
    frame["pdistance_available_num"] = frame["pdistance_available"].fillna(0.0)
    frame["sequence_evidence_available_num"] = frame["sequence_evidence_available"].fillna(0.0)
    frame["tree_evidence_available_num"] = frame["tree_evidence_available"].fillna(0.0)
    frame["candidate_is_top1_num"] = frame["candidate_is_top1"].fillna(0.0)
    frame["same_genus_as_top1_num"] = frame["same_genus_as_top1"].fillna(0.0)
    frame["same_family_as_top1_num"] = frame["same_family_as_top1"].fillna(0.0)
    frame["same_order_as_top1_num"] = frame["same_order_as_top1"].fillna(0.0)
    frame["rls_prior_supported_num"] = frame["rls_prior_supported"].fillna(0.0)
    frame["obis_prior_supported_num"] = frame["obis_prior_supported"].fillna(0.0)
    frame["log_read_count"] = np.log1p(frame["read_count"].fillna(0.0).clip(lower=0.0))
    frame["log_asv_count"] = np.log1p(frame["asv_count"].fillna(0.0).clip(lower=0.0))
    frame["candidate_has_reference_sequence_num"] = frame["candidate_has_reference_sequence"].fillna(0.0)
    frame["log_candidate_reference_sequence_count"] = np.log1p(
        frame["candidate_reference_sequence_count"].fillna(0.0).clip(lower=0.0)
    )
    frame["marker_reference_exact_cluster_found_num"] = frame["marker_reference_exact_cluster_found"].fillna(0.0)
    frame["marker_observed_exact_cluster_found_num"] = frame["marker_observed_exact_cluster_found"].fillna(0.0)
    frame["marker_true_species_in_observed_exact_cluster_num"] = (
        frame["marker_true_species_in_observed_exact_cluster"].fillna(0.0)
    )

    for rank in RANKS:
        frame[f"{rank}_eligible"] = frame[f"{rank}_eligible"].astype(bool)
        frame[f"{rank}_candidate_correct"] = frame[f"{rank}_candidate_correct"].astype(bool)
    logger.log("Added candidate-level inference features")
    return frame


def add_nested_split(
    features: pd.DataFrame,
    fit_fraction: float,
    repeat: int,
    logger: ProgressLogger,
) -> pd.DataFrame:
    frame = features.copy()
    frame["posterior_split"] = frame["calibration_split"].astype(str)
    if fit_fraction <= 0.0:
        return frame
    if not 0.0 < fit_fraction < 1.0:
        raise SystemExit("--nested-calibration-fit-fraction must be between 0 and 1")

    calibration_mask = frame["calibration_split"].astype(str) == "calibration"
    calibration_groups = frame.loc[calibration_mask, "sample_id"].astype(str).drop_duplicates()
    fit_groups = {
        group
        for group in calibration_groups
        if stable_fraction(group, f"candidate-posterior-nested-{repeat}") < fit_fraction
    }
    fit_mask = calibration_mask & frame["sample_id"].astype(str).isin(fit_groups)
    threshold_mask = calibration_mask & ~frame["sample_id"].astype(str).isin(fit_groups)
    frame.loc[fit_mask, "posterior_split"] = "calibration_fit"
    frame.loc[threshold_mask, "posterior_split"] = "calibration_threshold"
    logger.log(
        "Assigned nested posterior splits: "
        f"fit_groups={len(fit_groups):,} "
        f"threshold_groups={len(calibration_groups) - len(fit_groups):,} "
        f"fit_rows={int(fit_mask.sum()):,} "
        f"threshold_rows={int(threshold_mask.sum()):,}"
    )
    return frame


def rank_feature_columns(_rank: str) -> tuple[list[str], list[str]]:
    return list(BASE_NUMERIC_FEATURES), list(BASE_CATEGORICAL_FEATURES)


def balanced_train_sample(
    train: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
    logger: ProgressLogger,
    rank: str,
) -> pd.DataFrame:
    if not max_rows or len(train) <= max_rows:
        return train
    positives = train[train["target"] == 1]
    negatives = train[train["target"] == 0]
    if positives.empty or negatives.empty:
        return train.sample(n=max_rows, random_state=random_state)
    n_pos = min(len(positives), max_rows // 2)
    n_neg = max_rows - n_pos
    pos_sample = positives.sample(n=n_pos, random_state=random_state) if len(positives) > n_pos else positives
    neg_sample = negatives.sample(n=min(n_neg, len(negatives)), random_state=random_state)
    sampled = pd.concat([pos_sample, neg_sample], ignore_index=True).sample(frac=1.0, random_state=random_state)
    logger.log(
        f"Training sample for rank={rank}: {len(sampled):,}/{len(train):,} rows "
        f"({len(pos_sample):,} positives, {len(neg_sample):,} negatives)"
    )
    return sampled


def fit_rank_model(train: pd.DataFrame, rank: str) -> Pipeline:
    numeric, categorical = rank_feature_columns(rank)
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", make_encoder(), categorical),
        ]
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=1206,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)]).fit(train[numeric + categorical], train["target"])


def choose_best(scored: pd.DataFrame, rank: str) -> pd.DataFrame:
    scored = scored[scored[f"{rank}_eligible"]].copy()
    if scored.empty:
        return scored
    ordered = scored.sort_values(["sample_id", "query_processid", "posterior_probability"], ascending=[True, True, False])
    return ordered.groupby(list(GROUP_COLS), sort=False, as_index=False).head(1).copy()


def threshold_rows(calibration_best: pd.DataFrame, evaluation_best: pd.DataFrame, rank: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    calibration_ordered = calibration_best.sort_values("posterior_probability", ascending=False).copy()
    probabilities = calibration_ordered["posterior_probability"].to_numpy(dtype=float)
    correct = calibration_ordered[f"{rank}_candidate_correct"].to_numpy(dtype=float)
    assigned_counts = np.arange(1, len(calibration_ordered) + 1, dtype=float)
    cumulative_accuracy = 100.0 * np.cumsum(correct) / assigned_counts if len(correct) else np.array([])
    n_cal = len(calibration_best)
    n_eval = len(evaluation_best)
    for target in TARGET_ACCURACIES:
        eligible_idx = np.flatnonzero(cumulative_accuracy >= target)
        if len(eligible_idx) == 0:
            rows.append(
                {
                    "rank": rank,
                    "target_accuracy_pct": target,
                    "status": "no_threshold",
                    "threshold": "",
                    "calibration_assignment_rate_pct": 0.0,
                    "calibration_rank_accuracy_pct": "",
                    "calibration_n_assigned": 0,
                    "evaluation_assignment_rate_pct": 0.0,
                    "evaluation_rank_accuracy_pct": "",
                    "evaluation_n_assigned": 0,
                }
            )
            continue
        best_idx = int(eligible_idx[-1])
        threshold = float(probabilities[best_idx])
        calibration_n_assigned = int(best_idx + 1)
        calibration_accuracy = float(cumulative_accuracy[best_idx])
        eval_assigned = evaluation_best[evaluation_best["posterior_probability"] >= threshold]
        eval_accuracy = (
            100.0 * eval_assigned[f"{rank}_candidate_correct"].sum() / len(eval_assigned)
            if len(eval_assigned)
            else np.nan
        )
        rows.append(
            {
                "rank": rank,
                "target_accuracy_pct": target,
                "status": "available",
                "threshold": threshold,
                "calibration_assignment_rate_pct": 100.0 * calibration_n_assigned / n_cal if n_cal else 0.0,
                "calibration_rank_accuracy_pct": calibration_accuracy,
                "calibration_n_assigned": calibration_n_assigned,
                "evaluation_assignment_rate_pct": 100.0 * len(eval_assigned) / n_eval if n_eval else 0.0,
                "evaluation_rank_accuracy_pct": eval_accuracy,
                "evaluation_n_assigned": int(len(eval_assigned)),
            }
        )
    return rows


def summarize_method_selection(best_rows: pd.DataFrame, rank: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_cols = ["calibration_split", "method", "encoder", "context", "prior_source", "prior_weight"]
    for key, group in best_rows.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        rows.append(
            {
                **meta,
                "rank": rank,
                "selected_rows": int(len(group)),
                "selected_rate_pct": 100.0 * len(group) / len(best_rows) if len(best_rows) else 0.0,
                "rank_accuracy_pct": 100.0 * group[f"{rank}_candidate_correct"].sum() / len(group) if len(group) else np.nan,
                "mean_posterior_probability": group["posterior_probability"].mean(),
            }
        )
    return rows


def build_rank_backoff(
    selected_by_rank: dict[str, pd.DataFrame],
    operating_points: pd.DataFrame,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_keys = selected_by_rank["order"][
        selected_by_rank["order"]["calibration_split"] == split
    ][list(GROUP_COLS)].drop_duplicates()
    n_query = len(base_keys)
    for target in TARGET_ACCURACIES:
        thresholds: dict[str, float] = {}
        for rank in RANKS:
            match = operating_points[
                (operating_points["rank"] == rank)
                & (operating_points["target_accuracy_pct"] == target)
                & (operating_points["status"] == "available")
            ]
            if not match.empty:
                thresholds[rank] = float(match.iloc[0]["threshold"])

        assigned_rows: list[dict[str, Any]] = []
        for rank in RANKS:
            if rank not in thresholds:
                continue
            best = selected_by_rank[rank]
            best = best[best["calibration_split"] == split].copy()
            best = best[best["posterior_probability"] >= thresholds[rank]]
            best["assigned_rank"] = rank
            best["assigned_correct"] = best[f"{rank}_candidate_correct"]
            best["assigned_value"] = best[f"candidate_{rank}"]
            assigned_rows.extend(best.to_dict(orient="records"))

        if assigned_rows:
            assigned = pd.DataFrame(assigned_rows)
            rank_priority = {"species": 0, "genus": 1, "family": 2, "order": 3}
            assigned["rank_priority"] = assigned["assigned_rank"].map(rank_priority)
            assigned = assigned.sort_values(
                ["sample_id", "query_processid", "rank_priority", "posterior_probability"],
                ascending=[True, True, True, False],
            )
            final = assigned.groupby(list(GROUP_COLS), sort=False, as_index=False).head(1).copy()
        else:
            final = pd.DataFrame()

        if final.empty:
            rows.append(
                {
                    "split": split,
                    "target_accuracy_pct": target,
                    "status": "no_threshold",
                    "n_query": int(n_query),
                    "n_assigned": 0,
                    "assignment_rate_pct": 0.0,
                    "assigned_accuracy_pct": "",
                    "species_assignments": 0,
                    "genus_assignments": 0,
                    "family_assignments": 0,
                    "order_assignments": 0,
                    "no_call_rate_pct": 100.0,
                }
            )
            continue
        counts = final["assigned_rank"].value_counts().to_dict()
        rows.append(
            {
                "split": split,
                "target_accuracy_pct": target,
                "status": "available",
                "n_query": int(n_query),
                "n_assigned": int(len(final)),
                "assignment_rate_pct": 100.0 * len(final) / n_query if n_query else 0.0,
                "assigned_accuracy_pct": 100.0 * final["assigned_correct"].sum() / len(final) if len(final) else np.nan,
                "species_assignments": int(counts.get("species", 0)),
                "genus_assignments": int(counts.get("genus", 0)),
                "family_assignments": int(counts.get("family", 0)),
                "order_assignments": int(counts.get("order", 0)),
                "no_call_rate_pct": 100.0 * (n_query - len(final)) / n_query if n_query else 0.0,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_POSTERIOR_DIR)
    parser.add_argument(
        "--max-train-rows-per-rank",
        type=int,
        default=1_250_000,
        help="Balanced calibration-row cap per rank; set 0 to train on all rows.",
    )
    parser.add_argument(
        "--max-query-groups",
        type=int,
        default=0,
        help="If positive, sample this many complete sample/query groups before scoring.",
    )
    parser.add_argument("--random-state", type=int, default=1206)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--sequence-evidence",
        type=Path,
        help="Optional candidate sequence-evidence table built by build_eco_phylo_candidate_sequence_evidence.py.",
    )
    parser.add_argument(
        "--tree-evidence",
        type=Path,
        help="Optional candidate tree-neighborhood table built by build_eco_phylo_candidate_tree_evidence.py.",
    )
    parser.add_argument(
        "--nested-calibration-fit-fraction",
        type=float,
        default=0.0,
        help=(
            "If positive, split original calibration groups into model-fit and "
            "threshold-calibration subsets. This enables a true nested "
            "fit/threshold/evaluation posterior run."
        ),
    )
    parser.add_argument("--nested-repeat", type=int, default=0)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = add_inference_features(
        read_features(
            args.features,
            logger,
            max_query_groups=args.max_query_groups,
            random_state=args.random_state,
            chunksize=args.chunksize,
            sequence_evidence=args.sequence_evidence,
            tree_evidence=args.tree_evidence,
        ),
        logger,
    )
    features = add_nested_split(
        features,
        fit_fraction=args.nested_calibration_fit_fraction,
        repeat=args.nested_repeat,
        logger=logger,
    )
    nested_mode = args.nested_calibration_fit_fraction > 0.0
    max_rows = args.max_train_rows_per_rank if args.max_train_rows_per_rank > 0 else None

    model_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    selected_by_rank: dict[str, pd.DataFrame] = {}

    for rank in RANKS:
        logger.log(f"Fitting candidate posterior for rank={rank}")
        rank_frame = features[features[f"{rank}_eligible"]].copy()
        rank_frame["target"] = rank_frame[f"{rank}_candidate_correct"].astype(int)
        split_col = "posterior_split" if nested_mode else "calibration_split"
        train_label = "calibration_fit" if nested_mode else "calibration"
        threshold_label = "calibration_threshold" if nested_mode else "calibration"
        train = rank_frame[rank_frame[split_col] == train_label].copy()
        threshold = rank_frame[rank_frame[split_col] == threshold_label].copy()
        evaluation = rank_frame[rank_frame[split_col] == "evaluation"].copy()
        if train["target"].nunique() < 2:
            logger.log(f"Skipping rank={rank}: calibration target has one class")
            continue

        train_for_model = balanced_train_sample(train, max_rows, 1206, logger, rank)
        model = fit_rank_model(train_for_model, rank)
        numeric, categorical = rank_feature_columns(rank)
        threshold_best = pd.DataFrame()
        evaluation_best = pd.DataFrame()
        for name, subset in (("calibration", threshold), ("evaluation", evaluation)):
            logger.log(f"Scoring rank={rank} split={name} rows={len(subset):,}")
            probs = model.predict_proba(subset[numeric + categorical])[:, 1]
            out = subset.copy()
            out["calibration_split"] = name
            out["rank"] = rank
            out["posterior_probability"] = probs
            selected = choose_best(out, rank)
            selected_frames.append(
                selected[
                    [
                        "rank",
                        "calibration_split",
                        "sample_id",
                        "query_processid",
                        "method",
                        "encoder",
                        "context",
                        "prior_source",
                        "prior_weight",
                        "true_tree_label",
                        "candidate_tree_label",
                        f"true_{rank}",
                        f"candidate_{rank}",
                        f"{rank}_candidate_correct",
                        "posterior_probability",
                        "candidate_rank",
                        "candidate_score",
                        "candidate_score_method_z",
                        "sequence_only_score_method_z",
                        "blast_candidate_pident",
                        "candidate_pdistance",
                        "candidate_sequence_identity",
                        "sequence_evidence_available",
                        "tree_evidence_available",
                        "tree_distance_to_top1_candidate",
                        "tree_distance_to_candidate_set_mean",
                        "topk_pairwise_tree_distance_mean",
                        "same_genus_topk_fraction",
                        "same_family_topk_fraction",
                        "same_order_topk_fraction",
                        "rls_prior_log1p",
                        "obis_prior_log1p",
                    ]
                ].copy()
            )
            if name == "calibration":
                threshold_best = selected
            else:
                evaluation_best = selected

            y_true = subset["target"].to_numpy(dtype=int)
            model_rows.append(
                {
                    "rank": rank,
                    "split": name,
                    "row_count": int(len(subset)),
                    "positive_rate_pct": 100.0 * float(np.mean(y_true)) if len(y_true) else np.nan,
                    "roc_auc": safe_metric(roc_auc_score, y_true, probs),
                    "average_precision": safe_metric(average_precision_score, y_true, probs),
                    "brier_score": float(brier_score_loss(y_true, probs)) if len(y_true) else np.nan,
                    "chooser_query_count": int(len(selected)),
                    "chooser_forced_accuracy_pct": 100.0 * selected[f"{rank}_candidate_correct"].sum() / len(selected)
                    if len(selected)
                    else np.nan,
                }
            )

        selected_by_rank[rank] = pd.concat([threshold_best, evaluation_best], ignore_index=True)
        operating_rows.extend(threshold_rows(threshold_best, evaluation_best, rank))
        method_rows.extend(summarize_method_selection(selected_by_rank[rank], rank))

    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    selected_path = args.output_dir / "eco_phylo_candidate_posterior_selected_predictions.csv.gz"
    logger.log(f"Writing selected candidate posterior predictions to {rel(selected_path)}")
    with gzip.open(selected_path, "wt", newline="") as handle:
        selected_all.to_csv(handle, index=False)

    model_path = args.output_dir / "eco_phylo_candidate_posterior_model_summary.csv"
    logger.log(f"Writing candidate model summary to {rel(model_path)}")
    write_csv(
        model_path,
        model_rows,
        [
            "rank",
            "split",
            "row_count",
            "positive_rate_pct",
            "roc_auc",
            "average_precision",
            "brier_score",
            "chooser_query_count",
            "chooser_forced_accuracy_pct",
        ],
    )

    operating_path = args.output_dir / "eco_phylo_candidate_posterior_operating_points.csv"
    logger.log(f"Writing candidate operating points to {rel(operating_path)}")
    write_csv(
        operating_path,
        operating_rows,
        [
            "rank",
            "target_accuracy_pct",
            "status",
            "threshold",
            "calibration_assignment_rate_pct",
            "calibration_rank_accuracy_pct",
            "calibration_n_assigned",
            "evaluation_assignment_rate_pct",
            "evaluation_rank_accuracy_pct",
            "evaluation_n_assigned",
        ],
    )

    method_path = args.output_dir / "eco_phylo_candidate_posterior_method_selection_summary.csv"
    logger.log(f"Writing candidate method selection summary to {rel(method_path)}")
    write_csv(
        method_path,
        method_rows,
        [
            "calibration_split",
            "method",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "rank",
            "selected_rows",
            "selected_rate_pct",
            "rank_accuracy_pct",
            "mean_posterior_probability",
        ],
    )

    op_table = pd.DataFrame(operating_rows)
    backoff_rows: list[dict[str, Any]] = []
    if all(rank in selected_by_rank for rank in RANKS):
        for split in ("calibration", "evaluation"):
            backoff_rows.extend(build_rank_backoff(selected_by_rank, op_table, split))

    backoff_path = args.output_dir / "eco_phylo_candidate_posterior_rank_backoff_summary.csv"
    logger.log(f"Writing candidate rank-backoff summary to {rel(backoff_path)}")
    write_csv(
        backoff_path,
        backoff_rows,
        [
            "split",
            "target_accuracy_pct",
            "status",
            "n_query",
            "n_assigned",
            "assignment_rate_pct",
            "assigned_accuracy_pct",
            "species_assignments",
            "genus_assignments",
            "family_assignments",
            "order_assignments",
            "no_call_rate_pct",
        ],
    )

    manifest = {
        "generated_by": rel(Path(__file__)),
        "features": rel(args.features),
        "output_dir": rel(args.output_dir),
        "ranks": RANKS,
        "target_accuracies_pct": TARGET_ACCURACIES,
        "max_train_rows_per_rank": args.max_train_rows_per_rank,
        "max_query_groups": args.max_query_groups,
        "random_state": args.random_state,
        "sequence_evidence": rel(args.sequence_evidence) if args.sequence_evidence else "",
        "tree_evidence": rel(args.tree_evidence) if args.tree_evidence else "",
        "nested_calibration_fit_fraction": args.nested_calibration_fit_fraction,
        "nested_repeat": args.nested_repeat,
        "outputs": {
            "selected_predictions": rel(selected_path),
            "model_summary": rel(model_path),
            "operating_points": rel(operating_path),
            "method_selection_summary": rel(method_path),
            "rank_backoff_summary": rel(backoff_path),
        },
        "notes": [
            "Models are fit on calibration site20 groups and evaluated on held-out site20 groups.",
            "Rows are candidate-level evidence rows from existing method top-k outputs.",
            "Optional sequence-evidence features are joined by query_processid and candidate_tree_label.",
            "Optional tree-neighborhood features are joined by method, sample_id, query_processid, and candidate_tree_label.",
            "Features exclude true taxon labels and use only inference-time evidence.",
            "Training may use a balanced calibration-row cap; scoring still uses all eligible candidate rows.",
            "If max_query_groups is positive, this is a complete-query sampled prototype rather than a full-table result.",
            "Rank-backoff applies species, then genus, then family, then order thresholds learned on calibration sites.",
            "If nested_calibration_fit_fraction is positive, models are fit on calibration_fit groups and thresholds are learned on separate calibration_threshold groups.",
        ],
    }
    manifest_path = args.output_dir / "eco_phylo_candidate_posterior_model_manifest.json"
    logger.log(f"Writing manifest to {rel(manifest_path)}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
