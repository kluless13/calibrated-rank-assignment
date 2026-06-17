#!/usr/bin/env python3
"""Fit and evaluate a held-out Eco-Phylo posterior over Global_eDNA rows.

The input table is produced by `build_eco_phylo_posterior_ablation.py`. This
script trains rank-specific probability models on deterministic calibration
sites and evaluates them on held-out sites. It uses only features available at
inference time: method/context labels, sequence/tree candidate score, score
margin, read metadata, candidate reference evidence, and ecological arm labels.

The posterior is used in two ways:

1. choose the best evidence arm per query/rank;
2. apply rank-specific thresholds to decide species/genus/family/order/no-call.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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
DEFAULT_POSTERIOR_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "eco_phylo_posterior"
DEFAULT_FEATURES = DEFAULT_POSTERIOR_DIR / "eco_phylo_posterior_query_features.csv.gz"
RANKS = ("species", "genus", "family", "order")
TARGET_ACCURACIES = (50.0, 60.0, 70.0, 80.0, 90.0, 95.0)
GROUP_COLS = ("sample_id", "query_processid")
BASE_NUMERIC_FEATURES = (
    "top1_score",
    "top1_top2_score_margin",
    "top1_score_method_z",
    "top1_top2_score_margin_method_z",
    "top1_score_method_percentile",
    "top1_top2_score_margin_method_percentile",
    "log_read_count",
    "log_asv_count",
    "prior_weight_num",
    "top_candidate_count",
    "top1_has_reference_sequence_num",
    "log_top1_reference_sequence_count",
)
BASE_CATEGORICAL_FEATURES = (
    "method",
    "encoder",
    "context",
    "prior_source",
    "evidence_arm",
    "evidence_family",
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


def read_features(path: Path, logger: ProgressLogger) -> pd.DataFrame:
    usecols = [
        "method",
        "encoder",
        "context",
        "prior_source",
        "prior_weight",
        "evidence_arm",
        "evidence_family",
        "sample_id",
        "query_processid",
        "calibration_split",
        "read_count",
        "asv_count",
        "true_tree_label",
        "top1_tree_label",
        "top1_score",
        "top1_top2_score_margin",
        "top_candidate_count",
        "top1_has_reference_sequence",
        "top1_reference_sequence_count",
        "marker_species_oracle_supported_rate_pct",
        "marker_genus_oracle_supported_rate_pct",
        "marker_family_oracle_supported_rate_pct",
        "marker_order_oracle_supported_rate_pct",
    ]
    for rank in RANKS:
        usecols.extend([f"true_{rank}", f"top1_{rank}", f"{rank}_eligible", f"{rank}_top1_correct"])
    logger.log(f"Reading posterior features from {rel(path)}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def add_inference_features(features: pd.DataFrame, logger: ProgressLogger) -> pd.DataFrame:
    frame = features.copy()
    for column in BASE_CATEGORICAL_FEATURES:
        frame[column] = frame[column].map(clean).replace("", "none")
    for column in ("top1_score", "top1_top2_score_margin", "read_count", "asv_count", "prior_weight"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["prior_weight_num"] = frame["prior_weight"].fillna(0.0)
    frame["log_read_count"] = np.log1p(frame["read_count"].fillna(0.0).clip(lower=0.0))
    frame["log_asv_count"] = np.log1p(frame["asv_count"].fillna(0.0).clip(lower=0.0))
    frame["top1_has_reference_sequence_num"] = (
        pd.to_numeric(frame["top1_has_reference_sequence"], errors="coerce").fillna(0.0)
    )
    frame["log_top1_reference_sequence_count"] = np.log1p(
        pd.to_numeric(frame["top1_reference_sequence_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    calibration = frame["calibration_split"] == "calibration"
    for column in ("top1_score", "top1_top2_score_margin"):
        z_col = f"{column}_method_z"
        pct_col = f"{column}_method_percentile"
        frame[z_col] = 0.0
        frame[pct_col] = 0.5
        for method, group in frame.groupby("method", sort=False):
            train_values = group.loc[calibration.loc[group.index], column].dropna().to_numpy(dtype=float)
            if train_values.size == 0:
                continue
            mean = float(np.mean(train_values))
            std = float(np.std(train_values)) or 1.0
            values = group[column].fillna(mean).to_numpy(dtype=float)
            frame.loc[group.index, z_col] = (values - mean) / std
            ordered = np.sort(train_values)
            frame.loc[group.index, pct_col] = np.searchsorted(ordered, values, side="right") / len(ordered)
    logger.log("Added inference-time score normalization features")
    return frame


def rank_feature_columns(rank: str) -> tuple[list[str], list[str]]:
    numeric = list(BASE_NUMERIC_FEATURES) + [f"marker_{rank}_oracle_supported_rate_pct"]
    categorical = list(BASE_CATEGORICAL_FEATURES)
    return numeric, categorical


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


def safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None


def choose_best(scored: pd.DataFrame, rank: str) -> pd.DataFrame:
    scored = scored.copy()
    scored = scored[scored[f"{rank}_eligible"]].copy()
    if scored.empty:
        return scored
    ordered = scored.sort_values(["sample_id", "query_processid", "posterior_probability"], ascending=[True, True, False])
    return ordered.groupby(list(GROUP_COLS), sort=False, as_index=False).head(1).copy()


def threshold_rows(calibration_best: pd.DataFrame, evaluation_best: pd.DataFrame, rank: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    calibration_ordered = calibration_best.sort_values("posterior_probability", ascending=False).copy()
    probabilities = calibration_ordered["posterior_probability"].to_numpy(dtype=float)
    correct = calibration_ordered[f"{rank}_top1_correct"].to_numpy(dtype=float)
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
            100.0 * eval_assigned[f"{rank}_top1_correct"].sum() / len(eval_assigned) if len(eval_assigned) else np.nan
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
                "rank_accuracy_pct": 100.0 * group[f"{rank}_top1_correct"].sum() / len(group) if len(group) else np.nan,
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
            best["assigned_correct"] = best[f"{rank}_top1_correct"]
            best["assigned_value"] = best[f"top1_{rank}"]
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
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = add_inference_features(read_features(args.features, logger), logger)

    model_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    selected_by_rank: dict[str, pd.DataFrame] = {}

    for rank in RANKS:
        logger.log(f"Fitting posterior for rank={rank}")
        rank_frame = features[features[f"{rank}_eligible"]].copy()
        rank_frame["target"] = rank_frame[f"{rank}_top1_correct"].astype(int)
        train = rank_frame[rank_frame["calibration_split"] == "calibration"].copy()
        evaluation = rank_frame[rank_frame["calibration_split"] == "evaluation"].copy()
        if train["target"].nunique() < 2:
            logger.log(f"Skipping rank={rank}: calibration target has one class")
            continue

        model = fit_rank_model(train, rank)
        numeric, categorical = rank_feature_columns(rank)
        for name, subset in (("calibration", train), ("evaluation", evaluation)):
            probs = model.predict_proba(subset[numeric + categorical])[:, 1]
            out = subset.copy()
            out["rank"] = rank
            out["posterior_probability"] = probs
            out["split_eval"] = name
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
                        "top1_tree_label",
                        f"true_{rank}",
                        f"top1_{rank}",
                        f"{rank}_top1_correct",
                        "posterior_probability",
                        "top1_score",
                        "top1_top2_score_margin",
                    ]
                ].copy()
            )
            if name == "calibration":
                calibration_best = selected
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
                    "chooser_forced_accuracy_pct": 100.0 * selected[f"{rank}_top1_correct"].sum() / len(selected)
                    if len(selected)
                    else np.nan,
                }
            )

        selected_by_rank[rank] = pd.concat([calibration_best, evaluation_best], ignore_index=True)
        operating_rows.extend(threshold_rows(calibration_best, evaluation_best, rank))
        method_rows.extend(summarize_method_selection(selected_by_rank[rank], rank))

    selected_all = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    selected_path = args.output_dir / "eco_phylo_posterior_selected_predictions.csv.gz"
    logger.log(f"Writing selected posterior predictions to {rel(selected_path)}")
    with gzip.open(selected_path, "wt", newline="") as handle:
        selected_all.to_csv(handle, index=False)

    model_path = args.output_dir / "eco_phylo_posterior_model_summary.csv"
    logger.log(f"Writing model summary to {rel(model_path)}")
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

    operating_path = args.output_dir / "eco_phylo_posterior_operating_points.csv"
    logger.log(f"Writing operating points to {rel(operating_path)}")
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

    method_path = args.output_dir / "eco_phylo_posterior_method_selection_summary.csv"
    logger.log(f"Writing method selection summary to {rel(method_path)}")
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
    for split in ("calibration", "evaluation"):
        backoff_rows.extend(build_rank_backoff(selected_by_rank, op_table, split))

    backoff_path = args.output_dir / "eco_phylo_posterior_rank_backoff_summary.csv"
    logger.log(f"Writing rank-backoff summary to {rel(backoff_path)}")
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
        "outputs": {
            "selected_predictions": rel(selected_path),
            "model_summary": rel(model_path),
            "operating_points": rel(operating_path),
            "method_selection_summary": rel(method_path),
            "rank_backoff_summary": rel(backoff_path),
        },
        "notes": [
            "Models are fit on calibration site20 groups and evaluated on held-out site20 groups.",
            "This posterior selects among existing evidence-arm predictions; it does not generate new candidates.",
            "Features exclude true taxon labels and use only inference-time evidence.",
            "Rank-backoff applies species, then genus, then family, then order thresholds learned on calibration sites.",
        ],
    }
    manifest_path = args.output_dir / "eco_phylo_posterior_model_manifest.json"
    logger.log(f"Writing manifest to {rel(manifest_path)}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
