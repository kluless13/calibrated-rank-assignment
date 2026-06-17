#!/usr/bin/env python3
"""Build rank-adaptive calibration/no-call tables for fish-tree predictions.

The inputs are zero-shot candidate prediction CSVs plus their matching
zero_shot_candidate_per_query.csv files. The script is intentionally encoder
agnostic: neural models, BLAST, VSEARCH, k-mer baselines, and negative controls
all produce the same source tables.

This is an empirical diagnostic, not a deployment calibration claim. Use an
independent calibration split before reporting final no-call thresholds as
prospective operating points.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
RANKS = ["species", "genus", "family", "order"]
TOP_KS = [1, 5, 10]
COVERAGE_GRID = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
TARGET_PRECISIONS = [0.80, 0.90, 0.95, 0.99]


@dataclass(frozen=True)
class PredictionSet:
    split: str
    name: str
    group: str
    predictions_csv: Path
    per_query_csv: Path


def clean(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def parse_score_list(value: object) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            raw = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def parse_label_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            raw = parser(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        if isinstance(raw, list):
            return [clean(item).replace(" ", "_") for item in raw if clean(item)]
    return [item.strip().replace(" ", "_") for item in re_split_labels(text)]


def re_split_labels(text: str) -> list[str]:
    for sep in ("|", ";", ","):
        if sep in text:
            return [part for part in text.split(sep) if part.strip()]
    return [text]


def load_candidate_taxonomy(split: str) -> dict[str, dict[str, str]]:
    candidate_path = ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs" / split / "candidate_species.csv"
    if not candidate_path.exists():
        return {}
    candidates = pd.read_csv(candidate_path)
    out = {}
    for _, row in candidates.iterrows():
        label = clean(row.get("tree_label", "")).replace(" ", "_")
        if not label:
            continue
        out[label] = {
            "species": label,
            "genus": clean(row.get("genus_name", "")) or clean(row.get("genus_from_label", "")),
            "family": clean(row.get("family_name", "")),
            "order": clean(row.get("order_name", "")),
        }
    return out


def add_candidate_consensus_features(merged: pd.DataFrame, split: str) -> pd.DataFrame:
    candidate_tax = load_candidate_taxonomy(split)
    if "top_tree_labels" not in merged.columns:
        for rank in RANKS:
            merged[f"{rank}_top10_consensus"] = np.nan
            merged[f"{rank}_top10_unique"] = np.nan
        return merged
    consensus_rows = []
    unique_rows = []
    top_label_values = merged["top_tree_labels"].map(parse_label_list)
    for labels in top_label_values:
        labels = labels[:10]
        top = labels[0] if labels else ""
        top_tax = candidate_tax.get(top, {})
        consensus_record = {}
        unique_record = {}
        for rank in RANKS:
            if rank == "species":
                values = labels
                target = top
            else:
                values = [candidate_tax.get(label, {}).get(rank, "") for label in labels]
                target = top_tax.get(rank, "")
            values = [value for value in values if value]
            if not values or not target:
                consensus_record[rank] = np.nan
                unique_record[rank] = np.nan
            else:
                consensus_record[rank] = float(sum(value == target for value in values) / len(values))
                unique_record[rank] = int(len(set(values)))
        consensus_rows.append(consensus_record)
        unique_rows.append(unique_record)
    for rank in RANKS:
        merged[f"{rank}_top10_consensus"] = [row[rank] for row in consensus_rows]
        merged[f"{rank}_top10_unique"] = [row[rank] for row in unique_rows]
    return merged


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_prediction_set(item: PredictionSet) -> pd.DataFrame:
    pred = pd.read_csv(item.predictions_csv)
    per_query = pd.read_csv(item.per_query_csv)
    pred_columns = ["processid", "pred_score"]
    if "top_scores" in pred.columns:
        pred_columns.append("top_scores")
    if "top_tree_labels" in pred.columns:
        pred_columns.append("top_tree_labels")
    merged = per_query.merge(
        pred[pred_columns],
        on="processid",
        how="left",
    )
    top_scores = merged["top_scores"].map(parse_score_list) if "top_scores" in merged.columns else pd.Series([[]] * len(merged))
    merged["confidence_score"] = pd.to_numeric(merged.get("pred_score"), errors="coerce")
    merged["confidence_margin"] = [
        float(scores[0] - scores[1]) if len(scores) >= 2 else np.nan for scores in top_scores
    ]
    merged["confidence_relative_margin"] = [
        float((scores[0] - scores[1]) / abs(scores[0])) if len(scores) >= 2 and scores[0] else np.nan
        for scores in top_scores
    ]
    merged["split"] = item.split
    merged["prediction_set"] = item.name
    merged["prediction_group"] = item.group
    merged = add_candidate_consensus_features(merged, item.split)
    for rank in RANKS:
        for k in TOP_KS:
            col = f"{rank}_top{k}"
            if col in merged.columns:
                merged[col] = merged[col].map(parse_bool)
    return merged


def discover_prediction_sets(root: Path) -> list[PredictionSet]:
    sets: list[PredictionSet] = []
    paper1 = root / "paper1_phylo_calibrated_assignment"

    for split in ["eval_c", "seen_test", "unseen_genera"]:
        for method in ["kmer", "blast", "vsearch"]:
            base = paper1 / f"baselines_{split}" / method
            pred = base / "zero_shot_candidate_predictions.csv"
            per_query = base / "zero_shot_metrics" / "zero_shot_candidate_per_query.csv"
            if pred.exists() and per_query.exists():
                sets.append(PredictionSet(split, method, "sequence_baseline", pred, per_query))

    neural_patterns = {
        "eval_c": [
            ("neural_cosine512_seed1206", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval"),
            ("neural_cosine512_seed1207", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1207"),
            ("neural_cosine512_seed1208", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1208"),
        ],
        "seen_test": [
            ("neural_cosine512_seed1206", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seen_test"),
            ("neural_cosine512_seed1207", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1207_seen_test"),
            ("neural_cosine512_seed1208", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1208_seen_test"),
        ],
        "unseen_genera": [
            ("neural_cosine512_seed1206", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_unseen_genera"),
            ("neural_cosine512_seed1207", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1207_unseen_genera"),
            ("neural_cosine512_seed1208", root / "coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1208_unseen_genera"),
        ],
    }
    for split, paths in neural_patterns.items():
        for name, base in paths:
            pred = base / "zero_shot_candidate_predictions.csv"
            per_query = base / "zero_shot_metrics" / "zero_shot_candidate_per_query.csv"
            if pred.exists() and per_query.exists():
                sets.append(PredictionSet(split, name, "neural", pred, per_query))

    encoder_root = root / "paper1_encoder_benchmarks"
    split_suffixes = {
        "eval_c": "",
        "seen_test": "_seen_test",
        "unseen_genera": "_unseen_genera",
    }
    for split, suffix in split_suffixes.items():
        for encoder in ["cnn", "bilstm", "transformer"]:
            name = f"{encoder}_seed1206"
            base = encoder_root / f"coi_{encoder}_seed1206{suffix}"
            pred = base / "zero_shot_candidate_predictions.csv"
            per_query = base / "zero_shot_metrics" / "zero_shot_candidate_per_query.csv"
            if pred.exists() and per_query.exists():
                sets.append(PredictionSet(split, name, "encoder_benchmark", pred, per_query))

    query_embedding_root = paper1 / "query_embeddings"
    for split in split_suffixes:
        for encoder in ["cnn", "bilstm", "transformer"]:
            name = f"{encoder}_seed1206"
            base = query_embedding_root / f"coi_{encoder}_seed1206_{split}"
            pred = base / "zero_shot_candidate_predictions.csv"
            per_query = base / "zero_shot_metrics" / "zero_shot_candidate_per_query.csv"
            if pred.exists() and per_query.exists():
                sets.append(PredictionSet(split, name, "encoder_benchmark", pred, per_query))

    cnn_repeat_root = paper1 / "cnn_seed_repeats"
    for split, suffix in split_suffixes.items():
        for seed in ["1207", "1208"]:
            name = f"cnn_seed{seed}"
            base = cnn_repeat_root / f"coi_cnn_seed{seed}{suffix}"
            pred = base / "zero_shot_candidate_predictions.csv"
            per_query = base / "zero_shot_metrics" / "zero_shot_candidate_per_query.csv"
            if pred.exists() and per_query.exists():
                sets.append(PredictionSet(split, name, "encoder_benchmark_repeat", pred, per_query))

    for split in ["eval_c", "seen_test", "unseen_genera"]:
        for seed in ["seed1206", "seed1207", "seed1208"]:
            for control in ["shuffled_labels", "random_ranked"]:
                base = paper1 / f"negative_controls_{seed}_{split}" / control
                pred = base / "zero_shot_candidate_predictions.csv"
                per_query = base / "zero_shot_metrics" / "zero_shot_candidate_per_query.csv"
                if pred.exists() and per_query.exists():
                    sets.append(PredictionSet(split, f"{control}_{seed}", "negative_control", pred, per_query))

    return sets


def coverage_curves(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["confidence_score", "confidence_margin", "confidence_relative_margin"]
    for (split, group, name), sub in df.groupby(["split", "prediction_group", "prediction_set"], dropna=False):
        for feature in features:
            work = sub[np.isfinite(pd.to_numeric(sub[feature], errors="coerce"))].copy()
            if work.empty:
                continue
            work = work.sort_values(feature, ascending=False).reset_index(drop=True)
            for coverage in COVERAGE_GRID:
                n_accept = max(1, int(round(len(work) * coverage)))
                accepted = work.iloc[:n_accept]
                row = {
                    "split": split,
                    "prediction_group": group,
                    "prediction_set": name,
                    "confidence_feature": feature,
                    "coverage": float(n_accept / len(work)),
                    "n_total": int(len(work)),
                    "n_accepted": int(n_accept),
                    "threshold": float(accepted[feature].iloc[-1]),
                }
                for rank in RANKS:
                    for k in TOP_KS:
                        col = f"{rank}_top{k}"
                        if col in accepted:
                            row[col] = float(accepted[col].mean())
                rows.append(row)
    return pd.DataFrame(rows)


def threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = ["confidence_score", "confidence_margin", "confidence_relative_margin"]
    for (split, group, name), sub in df.groupby(["split", "prediction_group", "prediction_set"], dropna=False):
        for feature in features:
            work = sub[np.isfinite(pd.to_numeric(sub[feature], errors="coerce"))].copy()
            if work.empty:
                continue
            work = work.sort_values(feature, ascending=False).reset_index(drop=True)
            for rank in RANKS:
                col = f"{rank}_top1"
                if col not in work:
                    continue
                correct = work[col].astype(float).to_numpy()
                cumulative_precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
                for target in TARGET_PRECISIONS:
                    eligible = np.where(cumulative_precision >= target)[0]
                    if len(eligible) == 0:
                        rows.append(
                            {
                                "split": split,
                                "prediction_group": group,
                                "prediction_set": name,
                                "confidence_feature": feature,
                                "rank": rank,
                                "target_precision": target,
                                "n_total": int(len(work)),
                                "n_accepted": 0,
                                "coverage": 0.0,
                                "observed_precision": np.nan,
                                "threshold": np.nan,
                            }
                        )
                        continue
                    idx = int(eligible[-1])
                    rows.append(
                        {
                            "split": split,
                            "prediction_group": group,
                            "prediction_set": name,
                            "confidence_feature": feature,
                            "rank": rank,
                            "target_precision": target,
                            "n_total": int(len(work)),
                            "n_accepted": idx + 1,
                            "coverage": float((idx + 1) / len(work)),
                            "observed_precision": float(cumulative_precision[idx]),
                            "threshold": float(work[feature].iloc[idx]),
                        }
                    )
    return pd.DataFrame(rows)


def rank_adaptive_policy(df: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature = "confidence_margin"
    threshold_rows = thresholds[
        (thresholds["confidence_feature"] == feature)
        & np.isfinite(pd.to_numeric(thresholds["threshold"], errors="coerce"))
    ]
    for (split, group, name, target), thresh_sub in threshold_rows.groupby(
        ["split", "prediction_group", "prediction_set", "target_precision"],
        dropna=False,
    ):
        sub = df[
            (df["split"] == split)
            & (df["prediction_group"] == group)
            & (df["prediction_set"] == name)
            & np.isfinite(pd.to_numeric(df[feature], errors="coerce"))
        ].copy()
        if sub.empty:
            continue
        thresh_by_rank = {
            row["rank"]: float(row["threshold"])
            for _, row in thresh_sub.iterrows()
            if np.isfinite(row["threshold"])
        }
        assigned_rank = []
        assigned_correct = []
        for _, query in sub.iterrows():
            chosen = "no_call"
            correct = False
            for rank in RANKS:
                threshold = thresh_by_rank.get(rank)
                if threshold is not None and float(query[feature]) >= threshold:
                    chosen = rank
                    correct = bool(query.get(f"{rank}_top1", False))
                    break
            assigned_rank.append(chosen)
            assigned_correct.append(correct)
        policy = pd.DataFrame({"assigned_rank": assigned_rank, "correct": assigned_correct})
        assigned = policy[policy["assigned_rank"] != "no_call"]
        row = {
            "split": split,
            "prediction_group": group,
            "prediction_set": name,
            "confidence_feature": feature,
            "target_precision": float(target),
            "n_total": int(len(policy)),
            "n_assigned": int(len(assigned)),
            "coverage": float(len(assigned) / len(policy)),
            "assigned_precision": float(assigned["correct"].mean()) if len(assigned) else np.nan,
        }
        successes = int(assigned["correct"].sum()) if len(assigned) else 0
        ci_low, ci_high = wilson_interval(successes, len(assigned))
        row["assigned_correct"] = successes
        row["assigned_precision_ci95_low"] = ci_low
        row["assigned_precision_ci95_high"] = ci_high
        for rank in RANKS + ["no_call"]:
            row[f"assigned_{rank}_count"] = int((policy["assigned_rank"] == rank).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def thresholds_from_calibration_split(
    calibration: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    rows = []
    work = calibration[np.isfinite(pd.to_numeric(calibration[feature], errors="coerce"))].copy()
    if work.empty:
        return pd.DataFrame(rows)
    work = work.sort_values(feature, ascending=False).reset_index(drop=True)
    for rank in RANKS:
        col = f"{rank}_top1"
        if col not in work:
            continue
        correct = work[col].astype(float).to_numpy()
        cumulative_precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        for target in TARGET_PRECISIONS:
            eligible = np.where(cumulative_precision >= target)[0]
            if len(eligible) == 0:
                rows.append(
                    {
                        "rank": rank,
                        "target_precision": target,
                        "n_calibration": int(len(work)),
                        "n_calibration_accepted": 0,
                        "calibration_coverage": 0.0,
                        "calibration_precision": np.nan,
                        "threshold": np.nan,
                    }
                )
                continue
            idx = int(eligible[-1])
            rows.append(
                {
                    "rank": rank,
                    "target_precision": target,
                    "n_calibration": int(len(work)),
                    "n_calibration_accepted": idx + 1,
                    "calibration_coverage": float((idx + 1) / len(work)),
                    "calibration_precision": float(cumulative_precision[idx]),
                    "threshold": float(work[feature].iloc[idx]),
                }
            )
    return pd.DataFrame(rows)


def prospective_rank_adaptive_policy(
    df: pd.DataFrame,
    calibration_split: str,
    evaluation_splits: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows = []
    policy_rows = []
    features = ["confidence_score", "confidence_margin", "confidence_relative_margin"]
    for (group, name), method_df in df.groupby(["prediction_group", "prediction_set"], dropna=False):
        calibration = method_df[method_df["split"] == calibration_split].copy()
        if calibration.empty:
            continue
        for feature in features:
            learned = thresholds_from_calibration_split(calibration, feature)
            if learned.empty:
                continue
            for _, row in learned.iterrows():
                threshold_rows.append(
                    {
                        "calibration_split": calibration_split,
                        "prediction_group": group,
                        "prediction_set": name,
                        "confidence_feature": feature,
                        **row.to_dict(),
                    }
                )
            learned = learned[np.isfinite(pd.to_numeric(learned["threshold"], errors="coerce"))].copy()
            for evaluation_split in evaluation_splits:
                evaluation = method_df[
                    (method_df["split"] == evaluation_split)
                    & np.isfinite(pd.to_numeric(method_df[feature], errors="coerce"))
                ].copy()
                if evaluation.empty:
                    continue
                for target in TARGET_PRECISIONS:
                    target_thresholds = learned[learned["target_precision"] == target]
                    thresh_by_rank = {
                        item["rank"]: float(item["threshold"])
                        for _, item in target_thresholds.iterrows()
                        if np.isfinite(item["threshold"])
                    }
                    assigned_rank = []
                    assigned_correct = []
                    for _, query in evaluation.iterrows():
                        chosen = "no_call"
                        correct = False
                        for rank in RANKS:
                            threshold = thresh_by_rank.get(rank)
                            if threshold is not None and float(query[feature]) >= threshold:
                                chosen = rank
                                correct = bool(query.get(f"{rank}_top1", False))
                                break
                        assigned_rank.append(chosen)
                        assigned_correct.append(correct)
                    policy = pd.DataFrame({"assigned_rank": assigned_rank, "correct": assigned_correct})
                    assigned = policy[policy["assigned_rank"] != "no_call"]
                    out = {
                        "calibration_split": calibration_split,
                        "evaluation_split": evaluation_split,
                        "prediction_group": group,
                        "prediction_set": name,
                        "confidence_feature": feature,
                        "target_precision": float(target),
                        "n_calibration": int(len(calibration)),
                        "n_evaluation": int(len(policy)),
                        "n_assigned": int(len(assigned)),
                        "coverage": float(len(assigned) / len(policy)),
                        "assigned_precision": float(assigned["correct"].mean()) if len(assigned) else np.nan,
                    }
                    successes = int(assigned["correct"].sum()) if len(assigned) else 0
                    ci_low, ci_high = wilson_interval(successes, len(assigned))
                    out["assigned_correct"] = successes
                    out["assigned_precision_ci95_low"] = ci_low
                    out["assigned_precision_ci95_high"] = ci_high
                    for rank in RANKS + ["no_call"]:
                        out[f"assigned_{rank}_count"] = int((policy["assigned_rank"] == rank).sum())
                    policy_rows.append(out)
    return pd.DataFrame(threshold_rows), pd.DataFrame(policy_rows)


CONSENSUS_POLICY_FEATURES = {
    "species": "confidence_relative_margin",
    "genus": "genus_top10_consensus",
    "family": "family_top10_consensus",
    "order": "order_top10_consensus",
}


def missing_reference_aware_policy(
    df: pd.DataFrame,
    calibration_split: str,
    evaluation_splits: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows = []
    policy_rows = []
    for (group, name), method_df in df.groupby(["prediction_group", "prediction_set"], dropna=False):
        calibration = method_df[method_df["split"] == calibration_split].copy()
        if calibration.empty:
            continue
        learned_by_target: dict[float, dict[str, float]] = {target: {} for target in TARGET_PRECISIONS}
        for rank, feature in CONSENSUS_POLICY_FEATURES.items():
            work = calibration[np.isfinite(pd.to_numeric(calibration[feature], errors="coerce"))].copy()
            if work.empty or f"{rank}_top1" not in work:
                continue
            work = work.sort_values(feature, ascending=False).reset_index(drop=True)
            correct = work[f"{rank}_top1"].astype(float).to_numpy()
            cumulative_precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
            for target in TARGET_PRECISIONS:
                eligible = np.where(cumulative_precision >= target)[0]
                if len(eligible) == 0:
                    threshold_rows.append(
                        {
                            "calibration_split": calibration_split,
                            "prediction_group": group,
                            "prediction_set": name,
                            "rank": rank,
                            "feature": feature,
                            "target_precision": target,
                            "n_calibration": int(len(work)),
                            "n_calibration_accepted": 0,
                            "calibration_coverage": 0.0,
                            "calibration_precision": np.nan,
                            "threshold": np.nan,
                        }
                    )
                    continue
                idx = int(eligible[-1])
                threshold = float(work[feature].iloc[idx])
                learned_by_target[target][rank] = threshold
                threshold_rows.append(
                    {
                        "calibration_split": calibration_split,
                        "prediction_group": group,
                        "prediction_set": name,
                        "rank": rank,
                        "feature": feature,
                        "target_precision": target,
                        "n_calibration": int(len(work)),
                        "n_calibration_accepted": idx + 1,
                        "calibration_coverage": float((idx + 1) / len(work)),
                        "calibration_precision": float(cumulative_precision[idx]),
                        "threshold": threshold,
                    }
                )
        for evaluation_split in evaluation_splits:
            evaluation = method_df[method_df["split"] == evaluation_split].copy()
            if evaluation.empty:
                continue
            for target in TARGET_PRECISIONS:
                thresholds = learned_by_target.get(target, {})
                assigned_rank = []
                assigned_correct = []
                for _, query in evaluation.iterrows():
                    chosen = "no_call"
                    correct = False
                    for rank in RANKS:
                        feature = CONSENSUS_POLICY_FEATURES[rank]
                        threshold = thresholds.get(rank)
                        value = query.get(feature)
                        if threshold is None or not np.isfinite(value):
                            continue
                        if float(value) >= threshold:
                            chosen = rank
                            correct = bool(query.get(f"{rank}_top1", False))
                            break
                    assigned_rank.append(chosen)
                    assigned_correct.append(correct)
                policy = pd.DataFrame({"assigned_rank": assigned_rank, "correct": assigned_correct})
                assigned = policy[policy["assigned_rank"] != "no_call"]
                successes = int(assigned["correct"].sum()) if len(assigned) else 0
                ci_low, ci_high = wilson_interval(successes, len(assigned))
                row = {
                    "calibration_split": calibration_split,
                    "evaluation_split": evaluation_split,
                    "prediction_group": group,
                    "prediction_set": name,
                    "target_precision": float(target),
                    "n_calibration": int(len(calibration)),
                    "n_evaluation": int(len(policy)),
                    "n_assigned": int(len(assigned)),
                    "coverage": float(len(assigned) / len(policy)),
                    "assigned_precision": float(assigned["correct"].mean()) if len(assigned) else np.nan,
                    "assigned_correct": successes,
                    "assigned_precision_ci95_low": ci_low,
                    "assigned_precision_ci95_high": ci_high,
                }
                for rank in RANKS + ["no_call"]:
                    row[f"assigned_{rank}_count"] = int((policy["assigned_rank"] == rank).sum())
                policy_rows.append(row)
    return pd.DataFrame(threshold_rows), pd.DataFrame(policy_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/remote_runs/2026-05-30/rtx_pro_6000"),
        help="Local copied result root containing Paper 1 outputs and neural runs.",
    )
    parser.add_argument(
        "--extra-root",
        action="append",
        type=Path,
        default=[],
        help="Additional copied result root. Later roots override duplicate split/group/name prediction sets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration"),
    )
    parser.add_argument("--calibration-split", default="seen_test")
    parser.add_argument(
        "--evaluation-split",
        action="append",
        default=["eval_c", "unseen_genera"],
        help="Split to evaluate with thresholds learned on --calibration-split. May be repeated.",
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_set_index: dict[tuple[str, str, str], PredictionSet] = {}
    for root in [args.root] + args.extra_root:
        logger.log(f"Discovering prediction sets under {root}")
        for item in discover_prediction_sets(root):
            prediction_set_index[(item.split, item.group, item.name)] = item
    prediction_sets = list(prediction_set_index.values())
    if not prediction_sets:
        raise SystemExit(f"No prediction sets found under {args.root}")
    logger.log(f"Discovered {len(prediction_sets)} prediction sets")

    frames = []
    for item in prediction_sets:
        logger.log(f"Loading {item.group}/{item.name}/{item.split}")
        frames.append(load_prediction_set(item))
    per_query = pd.concat(frames, ignore_index=True)
    logger.log(f"Loaded {len(per_query)} per-query rows")
    logger.log("Building coverage curves")
    curves = coverage_curves(per_query)
    logger.log("Building target-precision threshold summary")
    thresholds = threshold_summary(per_query)
    logger.log("Building rank-adaptive policy summary")
    policies = rank_adaptive_policy(per_query, thresholds)
    logger.log(
        f"Building prospective split-transfer policies using calibration split {args.calibration_split}"
    )
    prospective_thresholds, prospective_policies = prospective_rank_adaptive_policy(
        per_query,
        args.calibration_split,
        args.evaluation_split,
    )
    logger.log("Building missing-reference-aware consensus policy")
    missing_ref_thresholds, missing_ref_policies = missing_reference_aware_policy(
        per_query,
        args.calibration_split,
        args.evaluation_split,
    )

    per_query_path = args.output_dir / "calibration_per_query.csv"
    curves_path = args.output_dir / "coverage_calibration_curves.csv"
    thresholds_path = args.output_dir / "target_precision_thresholds.csv"
    policies_path = args.output_dir / "rank_adaptive_policy_summary.csv"
    prospective_thresholds_path = args.output_dir / "prospective_rank_thresholds.csv"
    prospective_policies_path = args.output_dir / "prospective_rank_adaptive_policy_summary.csv"
    missing_ref_thresholds_path = args.output_dir / "missing_reference_aware_thresholds.csv"
    missing_ref_policies_path = args.output_dir / "missing_reference_aware_policy_summary.csv"
    logger.log(f"Writing per-query calibration rows to {per_query_path}")
    per_query.to_csv(per_query_path, index=False)
    logger.log(f"Writing coverage curves to {curves_path}")
    curves.to_csv(curves_path, index=False)
    logger.log(f"Writing target-precision thresholds to {thresholds_path}")
    thresholds.to_csv(thresholds_path, index=False)
    logger.log(f"Writing rank-adaptive policies to {policies_path}")
    policies.to_csv(policies_path, index=False)
    logger.log(f"Writing prospective thresholds to {prospective_thresholds_path}")
    prospective_thresholds.to_csv(prospective_thresholds_path, index=False)
    logger.log(f"Writing prospective policies to {prospective_policies_path}")
    prospective_policies.to_csv(prospective_policies_path, index=False)
    logger.log(f"Writing missing-reference-aware thresholds to {missing_ref_thresholds_path}")
    missing_ref_thresholds.to_csv(missing_ref_thresholds_path, index=False)
    logger.log(f"Writing missing-reference-aware policies to {missing_ref_policies_path}")
    missing_ref_policies.to_csv(missing_ref_policies_path, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "extra_roots": [str(path) for path in args.extra_root],
        "output_dir": str(args.output_dir),
        "prediction_set_count": len(prediction_sets),
        "query_rows": int(len(per_query)),
        "calibration_split": args.calibration_split,
        "evaluation_splits": args.evaluation_split,
        "coverage_grid": COVERAGE_GRID,
        "target_precisions": TARGET_PRECISIONS,
        "confidence_features": ["confidence_score", "confidence_margin", "confidence_relative_margin"],
        "caveat": "Same-split diagnostics are empirical. Prospective tables transfer thresholds from the calibration split to held-out evaluation splits.",
        "outputs": {
            "per_query": str(per_query_path),
            "coverage_curves": str(curves_path),
            "target_precision_thresholds": str(thresholds_path),
            "rank_adaptive_policy_summary": str(policies_path),
            "prospective_rank_thresholds": str(prospective_thresholds_path),
            "prospective_rank_adaptive_policy_summary": str(prospective_policies_path),
            "missing_reference_aware_thresholds": str(missing_ref_thresholds_path),
            "missing_reference_aware_policy_summary": str(missing_ref_policies_path),
        },
    }
    manifest_path = args.output_dir / "rank_adaptive_calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
