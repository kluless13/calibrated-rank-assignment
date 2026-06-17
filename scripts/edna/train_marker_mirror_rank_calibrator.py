#!/usr/bin/env python3
"""Train a feature-based rank/no-call calibrator for MarkerMirror candidates.

Raw top-1 MarkerMirror score is not enough for calibrated rank/no-call.  This
script turns a per-query candidate list into features such as score margins,
top-k consensus, and taxonomic ambiguity, trains per-rank classifiers on the
train split, selects thresholds on validation, and evaluates locked thresholds
on held-out rows.

This is still a diagnostic model-development layer.  The final pipeline should
join these marker-bridge features with sequence similarity, tree evidence,
ecology, and reference-gap diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import Logger

RANK_ORDER = ("species", "genus", "family", "order")
K_VALUES = (5, 10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="0.90,0.95")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--threshold-split", default="val")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def entropy(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    probs = weights / max(float(weights.sum()), 1e-12)
    return float(-(probs * np.log(probs.clip(min=1e-12))).sum())


def make_query_features(group: pd.DataFrame) -> dict[str, Any]:
    group = group.sort_values("candidate_rank")
    top = group.iloc[0]
    scores = group["score"].to_numpy(dtype=float)
    row: dict[str, Any] = {
        "run": top["run"],
        "model": top["model"],
        "split": top["split"],
        "direction": top["direction"],
        "query_id": top["query_id"],
        "query_marker": top["query_marker"],
        "target_marker": top["target_marker"],
        "query_tree_label": top["query_tree_label"],
        "candidate_tree_label": top["candidate_tree_label"],
        "top1_score": float(scores[0]),
        "score_margin_1_2": float(scores[0] - scores[1]) if len(scores) > 1 else 0.0,
        "score_margin_1_5": float(scores[0] - scores[min(4, len(scores) - 1)]),
        "score_margin_1_10": float(scores[0] - scores[min(9, len(scores) - 1)]),
    }
    for rank in RANK_ORDER:
        row[f"query_{rank}"] = top[f"query_{rank}"]
        row[f"candidate_{rank}"] = top[f"candidate_{rank}"]
        row[f"match_{rank}"] = bool(top[f"match_{rank}"])
    for k in K_VALUES:
        subset = group[group["candidate_rank"] <= k]
        subset_scores = subset["score"].to_numpy(dtype=float)
        row[f"top{k}_score_mean"] = float(np.mean(subset_scores)) if len(subset_scores) else 0.0
        row[f"top{k}_score_std"] = float(np.std(subset_scores)) if len(subset_scores) else 0.0
        row[f"top{k}_score_min"] = float(np.min(subset_scores)) if len(subset_scores) else 0.0
        row[f"top{k}_score_entropy"] = entropy(subset_scores)
        for rank in RANK_ORDER:
            top_value = str(top[f"candidate_{rank}"])
            values = subset[f"candidate_{rank}"].astype(str)
            support = int((values == top_value).sum()) if top_value else 0
            row[f"top{k}_{rank}_support"] = support
            row[f"top{k}_{rank}_support_frac"] = support / max(len(subset), 1)
            row[f"top{k}_{rank}_unique"] = int(values.replace({"": np.nan}).dropna().nunique())
    return row


def build_features(candidate_path: Path, logger: Logger) -> pd.DataFrame:
    frame = pd.read_csv(candidate_path)
    logger.log(f"Loaded candidate rows={len(frame)} queries={frame['query_id'].nunique()}")
    rows = []
    group_cols = ["model", "direction", "split", "query_id"]
    for idx, (_, group) in enumerate(frame.groupby(group_cols, sort=False), start=1):
        rows.append(make_query_features(group))
        if idx == 1 or idx % 1000 == 0:
            logger.log(f"Featurized query group {idx}")
    features = pd.DataFrame(rows)
    logger.log(f"Built feature rows={len(features)}")
    return features


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("match_", "query_", "candidate_")
    blocked = {
        "run",
        "model",
        "split",
        "direction",
        "query_id",
        "query_marker",
        "target_marker",
        "query_tree_label",
        "candidate_tree_label",
    }
    cols = []
    for col in frame.columns:
        if col in blocked or any(col.startswith(prefix) for prefix in blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def fit_rank_model(train: pd.DataFrame, features: list[str], rank: str):
    y = train[f"match_{rank}"].astype(int)
    if y.nunique() < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train[features], y)
    return model


def fit_threshold(scores: np.ndarray, labels: np.ndarray, target: float) -> dict[str, Any]:
    thresholds = np.sort(np.unique(scores))[::-1]
    best = None
    for threshold in thresholds:
        mask = scores >= threshold
        n = int(mask.sum())
        if n == 0:
            continue
        precision = float(labels[mask].mean())
        if precision >= target:
            best = (float(threshold), precision, n)
    if best is None:
        return {"threshold": None, "fit_precision": None, "fit_coverage": 0.0, "fit_n": 0}
    threshold, precision, n = best
    return {
        "threshold": threshold,
        "fit_precision": precision,
        "fit_coverage": 100.0 * n / max(len(labels), 1),
        "fit_n": n,
    }


def add_probabilities(frame: pd.DataFrame, models: dict[str, Any], features: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for rank in RANK_ORDER:
        model = models.get(rank)
        col = f"prob_match_{rank}"
        if model is None:
            out[col] = np.nan
        else:
            out[col] = model.predict_proba(out[features])[:, 1]
    return out


def apply_policy(frame: pd.DataFrame, thresholds: dict[str, float | None], target: float) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        assigned_rank = "no_call"
        assigned_taxon = ""
        assigned_correct = False
        for rank in RANK_ORDER:
            threshold = thresholds.get(rank)
            prob = row.get(f"prob_match_{rank}")
            if threshold is not None and pd.notna(prob) and float(prob) >= float(threshold):
                assigned_rank = rank
                assigned_taxon = str(row[f"candidate_{rank}"])
                assigned_correct = bool(row[f"match_{rank}"])
                break
        record = row.to_dict()
        record.update(
            {
                "target_precision": target,
                "assigned_rank": assigned_rank,
                "assigned_taxon": assigned_taxon,
                "assigned_correct": assigned_correct,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def summarize(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in assignments.groupby(["target_precision", "model", "direction", "split"], dropna=False):
        target, model, direction, split = key
        assigned = group[group["assigned_rank"] != "no_call"]
        n_query = int(group["query_id"].nunique())
        assigned_n = int(len(assigned))
        rank_counts = assigned["assigned_rank"].value_counts().to_dict()
        false_species = assigned[(assigned["assigned_rank"] == "species") & (~assigned["match_species"].astype(bool))]
        rows.append(
            {
                "target_precision": target,
                "model": model,
                "direction": direction,
                "split": split,
                "n_query": n_query,
                "assigned_n": assigned_n,
                "coverage_pct": 100.0 * assigned_n / max(n_query, 1),
                "assigned_precision_pct": 100.0 * float(assigned["assigned_correct"].mean()) if assigned_n else float("nan"),
                "false_species_call_rate_pct": 100.0 * len(false_species) / max(n_query, 1),
                "species_calls": int(rank_counts.get("species", 0)),
                "genus_calls": int(rank_counts.get("genus", 0)),
                "family_calls": int(rank_counts.get("family", 0)),
                "order_calls": int(rank_counts.get("order", 0)),
                "no_calls": int(n_query - assigned_n),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_rank_calibrator.log")
    logger.log(f"Arguments: {vars(args)}")
    targets = [float(item) for item in args.targets.split(",") if item.strip()]
    features = build_features(args.candidate_rankings, logger)
    feature_cols = feature_columns(features)
    logger.log(f"Feature columns={len(feature_cols)}")

    threshold_rows = []
    assignment_frames = []
    for (model_name, direction), group in features.groupby(["model", "direction"], dropna=False):
        train = group[group["split"] == args.train_split]
        threshold_frame = group[group["split"] == args.threshold_split]
        rank_models = {rank: fit_rank_model(train, feature_cols, rank) for rank in RANK_ORDER}
        scored = add_probabilities(group, rank_models, feature_cols)
        for target in targets:
            thresholds: dict[str, float | None] = {}
            for rank in RANK_ORDER:
                model = rank_models.get(rank)
                if model is None or threshold_frame.empty:
                    fit = {"threshold": None, "fit_precision": None, "fit_coverage": 0.0, "fit_n": 0}
                else:
                    threshold_scored = add_probabilities(threshold_frame, {rank: model}, feature_cols)
                    fit = fit_threshold(
                        threshold_scored[f"prob_match_{rank}"].to_numpy(dtype=float),
                        threshold_scored[f"match_{rank}"].astype(int).to_numpy(),
                        target,
                    )
                thresholds[rank] = fit["threshold"]
                threshold_rows.append(
                    {
                        "target_precision": target,
                        "model": model_name,
                        "direction": direction,
                        "train_split": args.train_split,
                        "threshold_split": args.threshold_split,
                        "rank": rank,
                        **fit,
                    }
                )
            logger.log(f"target={target:.2f} model={model_name} direction={direction} thresholds={thresholds}")
            assignment_frames.append(apply_policy(scored, thresholds, target))

    thresholds = pd.DataFrame(threshold_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    summary = summarize(assignments) if not assignments.empty else pd.DataFrame()

    features.to_csv(args.output_dir / "marker_mirror_rank_calibrator_features.csv.gz", index=False)
    thresholds.to_csv(args.output_dir / "marker_mirror_rank_calibrator_thresholds.csv", index=False)
    assignments.to_csv(args.output_dir / "marker_mirror_rank_calibrator_assignments.csv.gz", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_rank_calibrator_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rankings": str(args.candidate_rankings),
        "targets": targets,
        "train_split": args.train_split,
        "threshold_split": args.threshold_split,
        "feature_columns": feature_cols,
        "rows": {
            "features": int(len(features)),
            "thresholds": int(len(thresholds)),
            "assignments": int(len(assignments)),
            "summary": int(len(summary)),
        },
        "claim_boundary": "MarkerMirror candidate-feature calibrator; diagnostic until joined with sequence/tree/ecology evidence.",
    }
    (args.output_dir / "marker_mirror_rank_calibrator_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.log(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
