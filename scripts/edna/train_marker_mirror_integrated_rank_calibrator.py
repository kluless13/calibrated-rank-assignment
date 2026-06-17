#!/usr/bin/env python3
"""Train a candidate-level integrated MarkerMirror rank/no-call diagnostic.

This is the first prototype of:

    MarkerMirror candidates
      -> sequence/reference/tree evidence
      -> candidate-level rank probabilities
      -> calibrated species/genus/family/order/no-call

For each model/direction and each taxonomic rank, the script trains a
candidate-level classifier on train rows.  On validation rows it selects the
best candidate per query for that rank, fits a precision threshold, and then
applies locked thresholds to held-out test rows.

The output is a diagnostic source table, not the final production policy.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import Logger

RANK_ORDER = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="0.90,0.95")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--threshold-split", default="val")
    parser.add_argument("--model-type", choices=("logistic", "hgb"), default="logistic")
    parser.add_argument("--max-train-rows-per-rank", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("match_", "query_", "candidate_species", "candidate_genus", "candidate_family", "candidate_order")
    blocked = {
        "run",
        "model",
        "split",
        "direction",
        "query_marker",
        "target_marker",
        "query_id",
        "query_tree_label",
        "query_seq_index",
        "candidate_tree_label",
        "candidate_species",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
    }
    cols = []
    for col in frame.columns:
        if col in blocked or any(col.startswith(prefix) for prefix in blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            if frame[col].notna().sum() == 0:
                continue
            cols.append(col)
    return cols


def make_model(model_type: str, seed: int):
    if model_type == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed),
    )


def balanced_sample(frame: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    positives = frame[frame[label_col].astype(bool)]
    negatives = frame[~frame[label_col].astype(bool)]
    if positives.empty or negatives.empty:
        return frame.sample(n=max_rows, random_state=seed)
    half = max_rows // 2
    pos = positives.sample(n=min(len(positives), half), random_state=seed)
    neg_n = max_rows - len(pos)
    neg = negatives.sample(n=min(len(negatives), neg_n), random_state=seed)
    sampled = pd.concat([pos, neg], ignore_index=True)
    if len(sampled) < max_rows and len(frame) > len(sampled):
        rest = frame.drop(sampled.index, errors="ignore")
        if len(rest):
            sampled = pd.concat(
                [sampled, rest.sample(n=min(len(rest), max_rows - len(sampled)), random_state=seed)],
                ignore_index=True,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_rank_model(train: pd.DataFrame, features: list[str], rank: str, args: argparse.Namespace):
    label_col = f"match_{rank}"
    y = train[label_col].astype(int)
    if y.nunique() < 2:
        return None
    sampled = balanced_sample(train, label_col, args.max_train_rows_per_rank, args.seed)
    model = make_model(args.model_type, args.seed)
    model.fit(sampled[features], sampled[label_col].astype(int))
    return model


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


def select_best_by_rank(frame: pd.DataFrame, rank: str) -> pd.DataFrame:
    prob_col = f"prob_match_{rank}"
    rows = frame.dropna(subset=[prob_col]).copy()
    if rows.empty:
        return rows
    idx = rows.groupby(["model", "direction", "split", "query_id"], dropna=False)[prob_col].idxmax()
    return rows.loc[idx].reset_index(drop=True)


def fit_threshold(selected: pd.DataFrame, rank: str, target: float) -> dict[str, Any]:
    prob_col = f"prob_match_{rank}"
    if selected.empty or selected[prob_col].notna().sum() == 0:
        return {"threshold": None, "fit_precision": None, "fit_coverage": 0.0, "fit_n": 0}
    scores = selected[prob_col].to_numpy(dtype=float)
    labels = selected[f"match_{rank}"].astype(int).to_numpy()
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
        "fit_coverage": 100.0 * n / max(len(selected), 1),
        "fit_n": n,
    }


def apply_policy(scored: pd.DataFrame, thresholds: dict[str, float | None], target: float) -> pd.DataFrame:
    selected_by_rank = {rank: select_best_by_rank(scored, rank).set_index("query_id") for rank in RANK_ORDER}
    query_ids = sorted(scored["query_id"].astype(str).unique())
    rows = []
    meta = scored.groupby("query_id", sort=False).first()
    for query_id in query_ids:
        assigned_rank = "no_call"
        assigned_taxon = ""
        assigned_correct = False
        selected_candidate = ""
        selected_score = np.nan
        selected_probability = np.nan
        for rank in RANK_ORDER:
            table = selected_by_rank[rank]
            threshold = thresholds.get(rank)
            if query_id not in table.index or threshold is None:
                continue
            row = table.loc[query_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            prob = row.get(f"prob_match_{rank}")
            if pd.notna(prob) and float(prob) >= float(threshold):
                assigned_rank = rank
                assigned_taxon = str(row[f"candidate_{rank}"])
                assigned_correct = bool(row[f"match_{rank}"])
                selected_candidate = str(row["candidate_tree_label"])
                selected_score = float(row["score"])
                selected_probability = float(prob)
                break
        first = meta.loc[query_id]
        rows.append(
            {
                "target_precision": target,
                "model": first["model"],
                "direction": first["direction"],
                "split": first["split"],
                "query_id": query_id,
                "query_marker": first["query_marker"],
                "target_marker": first["target_marker"],
                "query_tree_label": first["query_tree_label"],
                "assigned_rank": assigned_rank,
                "assigned_taxon": assigned_taxon,
                "assigned_correct": assigned_correct,
                "selected_candidate_tree_label": selected_candidate,
                "selected_candidate_score": selected_score,
                "selected_candidate_probability": selected_probability,
            }
        )
    return pd.DataFrame(rows)


def summarize(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in assignments.groupby(["target_precision", "model", "direction", "split"], dropna=False):
        target, model, direction, split = key
        assigned = group[group["assigned_rank"] != "no_call"]
        n_query = int(group["query_id"].nunique())
        assigned_n = int(len(assigned))
        rank_counts = assigned["assigned_rank"].value_counts().to_dict()
        rank_precisions = {
            f"{rank}_precision_pct": (
                100.0
                * float(assigned[assigned["assigned_rank"] == rank]["assigned_correct"].mean())
                if len(assigned[assigned["assigned_rank"] == rank])
                else float("nan")
            )
            for rank in RANK_ORDER
        }
        false_species = assigned[(assigned["assigned_rank"] == "species") & (~assigned["assigned_correct"].astype(bool))]
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
                **rank_precisions,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_integrated_rank_calibrator.log")
    logger.log(f"Arguments: {vars(args)}")
    targets = [float(item) for item in args.targets.split(",") if item.strip()]
    frame = pd.read_csv(args.evidence_table)
    features = feature_columns(frame)
    logger.log(f"Loaded evidence rows={len(frame)} queries={frame['query_id'].nunique()} features={len(features)}")

    threshold_rows = []
    assignment_frames = []
    model_inventory = []
    for (model_name, direction), group in frame.groupby(["model", "direction"], dropna=False):
        train = group[group["split"] == args.train_split]
        val = group[group["split"] == args.threshold_split]
        if train.empty or val.empty:
            logger.log(f"Skipping model={model_name} direction={direction}: train/val missing")
            continue
        rank_models = {}
        for rank in RANK_ORDER:
            rank_models[rank] = train_rank_model(train, features, rank, args)
            model_inventory.append(
                {
                    "model": model_name,
                    "direction": direction,
                    "rank": rank,
                    "trained": rank_models[rank] is not None,
                    "train_positive_rate_pct": 100.0 * float(train[f"match_{rank}"].astype(bool).mean()),
                }
            )
        scored = add_probabilities(group, rank_models, features)
        val_scored = scored[scored["split"] == args.threshold_split]
        for target in targets:
            thresholds: dict[str, float | None] = {}
            for rank in RANK_ORDER:
                selected_val = select_best_by_rank(val_scored, rank)
                fit = fit_threshold(selected_val, rank, target)
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
    inventory = pd.DataFrame(model_inventory)

    thresholds.to_csv(args.output_dir / "marker_mirror_integrated_rank_thresholds.csv", index=False)
    assignments.to_csv(args.output_dir / "marker_mirror_integrated_rank_assignments.csv.gz", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_integrated_rank_summary.csv", index=False)
    inventory.to_csv(args.output_dir / "marker_mirror_integrated_rank_model_inventory.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_table": str(args.evidence_table),
        "targets": targets,
        "train_split": args.train_split,
        "threshold_split": args.threshold_split,
        "model_type": args.model_type,
        "features": features,
        "rows": {
            "thresholds": int(len(thresholds)),
            "assignments": int(len(assignments)),
            "summary": int(len(summary)),
            "inventory": int(len(inventory)),
        },
        "claim_boundary": "Integrated MarkerMirror rank/no-call diagnostic; final pipeline still needs external ecology/reference-gap integration.",
    }
    (args.output_dir / "marker_mirror_integrated_rank_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
