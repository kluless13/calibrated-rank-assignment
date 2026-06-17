#!/usr/bin/env python3
"""Apply an integrated MarkerMirror rank/no-call calibrator to handoff rows.

This trains rank-specific candidate classifiers on an offline MarkerMirror
evidence table, fits thresholds on its validation split, and applies those
locked thresholds to a production-style candidate-generator evidence table.

Labels in the apply table are used only for optional evaluation summaries.  They
are excluded from model features.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_integrated_rank_calibrator import (
    RANK_ORDER,
    add_probabilities,
    fit_threshold,
    make_model,
    select_best_by_rank,
)
from scripts.edna.train_marker_mirror_bridge import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-evidence-table", type=Path, required=True)
    parser.add_argument("--apply-evidence-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="0.99")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--threshold-split", default="val")
    parser.add_argument("--train-direction", default="12S_to_16S")
    parser.add_argument("--apply-direction", default="12S->16S")
    parser.add_argument("--train-model", default="marker_mirror_projection")
    parser.add_argument("--model-type", choices=("logistic", "hgb"), default="logistic")
    parser.add_argument(
        "--disable-ranks",
        default="",
        help="Comma-separated ranks to disable at assignment time, e.g. species.",
    )
    parser.add_argument("--max-train-rows-per-rank", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def safe_feature_columns(train: pd.DataFrame, apply: pd.DataFrame) -> list[str]:
    blocked = {
        "run",
        "model",
        "split",
        "direction",
        "query_marker",
        "target_marker",
        "query_id",
        "query_tree_label",
        "input_query_tree_label",
        "query_seq_index",
        "candidate_tree_label",
        "candidate_species",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
        "query_sequence_hash",
        "production_label_available",
    }
    blocked_prefixes = (
        "match_",
        "query_",
        "input_query_",
        "candidate_species",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
    )
    train_numeric = {
        col
        for col in train.columns
        if pd.api.types.is_numeric_dtype(train[col]) and train[col].notna().sum() > 0
    }
    apply_numeric = {
        col
        for col in apply.columns
        if pd.api.types.is_numeric_dtype(apply[col]) and apply[col].notna().sum() > 0
    }
    features = []
    for col in sorted(train_numeric & apply_numeric):
        if col in blocked or any(col.startswith(prefix) for prefix in blocked_prefixes):
            continue
        features.append(col)
    return features


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
    if len(sampled) < max_rows:
        remaining = frame.drop(sampled.index, errors="ignore")
        if len(remaining):
            sampled = pd.concat(
                [sampled, remaining.sample(n=min(len(remaining), max_rows - len(sampled)), random_state=seed)],
                ignore_index=True,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_models(train: pd.DataFrame, features: list[str], args: argparse.Namespace, logger: Logger) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for rank in RANK_ORDER:
        label_col = f"match_{rank}"
        if label_col not in train.columns:
            models[rank] = None
            logger.log(f"rank={rank} missing label column; skipped")
            continue
        y = train[label_col].astype(int)
        if y.nunique() < 2:
            models[rank] = None
            logger.log(f"rank={rank} has one class only; skipped")
            continue
        sampled = balanced_sample(train, label_col, args.max_train_rows_per_rank, args.seed)
        model = make_model(args.model_type, args.seed)
        model.fit(sampled[features], sampled[label_col].astype(int))
        models[rank] = model
        logger.log(
            f"trained rank={rank} rows={len(sampled)} positive_rate={100.0 * float(sampled[label_col].astype(bool).mean()):.2f}%"
        )
    return models


def fit_thresholds(val_scored: pd.DataFrame, targets: list[float]) -> tuple[pd.DataFrame, dict[float, dict[str, float | None]]]:
    rows = []
    by_target: dict[float, dict[str, float | None]] = {}
    for target in targets:
        thresholds: dict[str, float | None] = {}
        for rank in RANK_ORDER:
            selected = select_best_by_rank(val_scored, rank)
            fit = fit_threshold(selected, rank, target)
            thresholds[rank] = fit["threshold"]
            rows.append({"target_precision": target, "rank": rank, **fit})
        by_target[target] = thresholds
    return pd.DataFrame(rows), by_target


def apply_policy(scored: pd.DataFrame, thresholds: dict[str, float | None], target: float, disabled_ranks: set[str]) -> pd.DataFrame:
    selected_by_rank = {rank: select_best_by_rank(scored, rank).set_index("query_id") for rank in RANK_ORDER}
    meta = scored.groupby("query_id", sort=False).first()
    rows = []
    for query_id in sorted(scored["query_id"].astype(str).unique()):
        assigned_rank = "no_call"
        assigned_taxon = ""
        selected_candidate = ""
        selected_score = np.nan
        selected_probability = np.nan
        selected_correct: bool | None = None
        for rank in RANK_ORDER:
            if rank in disabled_ranks:
                continue
            table = selected_by_rank[rank]
            threshold = thresholds.get(rank)
            if threshold is None or query_id not in table.index:
                continue
            row = table.loc[query_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            prob = row.get(f"prob_match_{rank}")
            if pd.notna(prob) and float(prob) >= float(threshold):
                assigned_rank = rank
                assigned_taxon = str(row.get(f"candidate_{rank}", ""))
                selected_candidate = str(row.get("candidate_tree_label", ""))
                selected_score = float(row.get("score", np.nan))
                selected_probability = float(prob)
                match_col = f"match_{rank}"
                selected_correct = bool(row[match_col]) if match_col in row and pd.notna(row[match_col]) else None
                break
        first = meta.loc[query_id]
        rows.append(
            {
                "target_precision": target,
                "model": first.get("model", "marker_mirror_candidate_generator"),
                "direction": first.get("direction", ""),
                "split": first.get("split", ""),
                "query_id": query_id,
                "query_marker": first.get("query_marker", ""),
                "target_marker": first.get("target_marker", ""),
                "query_tree_label": first.get("query_tree_label", first.get("input_query_tree_label", "")),
                "assigned_rank": assigned_rank,
                "assigned_taxon": assigned_taxon,
                "selected_candidate_tree_label": selected_candidate,
                "selected_candidate_score": selected_score,
                "selected_candidate_probability": selected_probability,
                "assigned_correct": selected_correct,
            }
        )
    return pd.DataFrame(rows)


def summarize(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in assignments.groupby(["target_precision", "model", "direction", "split"], dropna=False):
        target, model, direction, split = key
        assigned = group[group["assigned_rank"] != "no_call"]
        assigned_n = int(len(assigned))
        n_query = int(group["query_id"].nunique())
        row = {
            "target_precision": target,
            "model": model,
            "direction": direction,
            "split": split,
            "n_query": n_query,
            "assigned_n": assigned_n,
            "coverage_pct": 100.0 * assigned_n / max(n_query, 1),
            "species_calls": int((assigned["assigned_rank"] == "species").sum()),
            "genus_calls": int((assigned["assigned_rank"] == "genus").sum()),
            "family_calls": int((assigned["assigned_rank"] == "family").sum()),
            "order_calls": int((assigned["assigned_rank"] == "order").sum()),
            "no_calls": int(n_query - assigned_n),
            "label_available": bool(group["assigned_correct"].notna().any()),
        }
        labelled = assigned.dropna(subset=["assigned_correct"])
        if len(labelled):
            row["assigned_precision_pct"] = 100.0 * float(labelled["assigned_correct"].astype(bool).mean())
            false_species = labelled[(labelled["assigned_rank"] == "species") & (~labelled["assigned_correct"].astype(bool))]
            row["false_species_call_rate_pct"] = 100.0 * len(false_species) / max(n_query, 1)
            for rank in RANK_ORDER:
                rank_rows = labelled[labelled["assigned_rank"] == rank]
                row[f"{rank}_precision_pct"] = 100.0 * float(rank_rows["assigned_correct"].astype(bool).mean()) if len(rank_rows) else math.nan
        else:
            row["assigned_precision_pct"] = math.nan
            row["false_species_call_rate_pct"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_integrated_rank_apply.log")
    logger.log(f"Arguments: {vars(args)}")
    targets = [float(item) for item in args.targets.split(",") if item.strip()]
    disabled_ranks = {item.strip() for item in args.disable_ranks.split(",") if item.strip()}
    train_frame = pd.read_csv(args.train_evidence_table)
    apply_frame = pd.read_csv(args.apply_evidence_table)
    if args.train_direction:
        train_frame = train_frame[train_frame["direction"].astype(str) == args.train_direction].copy()
    if args.apply_direction:
        apply_frame = apply_frame[apply_frame["direction"].astype(str) == args.apply_direction].copy()
    if args.train_model:
        train_frame = train_frame[train_frame["model"].astype(str) == args.train_model].copy()
    if train_frame.empty:
        raise RuntimeError("No training rows after direction/split filtering.")
    if apply_frame.empty:
        raise RuntimeError("No apply rows after direction filtering.")
    features = safe_feature_columns(train_frame, apply_frame)
    if not features:
        raise RuntimeError("No safe shared numeric features found.")
    train = train_frame[train_frame["split"].astype(str) == args.train_split].copy()
    val = train_frame[train_frame["split"].astype(str) == args.threshold_split].copy()
    if train.empty or val.empty:
        raise RuntimeError("Training or threshold split is empty.")
    logger.log(
        f"Loaded train rows={len(train_frame)} apply rows={len(apply_frame)} train_split_rows={len(train)} val_rows={len(val)} features={len(features)}"
    )
    models = train_models(train, features, args, logger)
    val_scored = add_probabilities(val, models, features)
    apply_scored = add_probabilities(apply_frame, models, features)
    thresholds, by_target = fit_thresholds(val_scored, targets)
    assignment_frames = [
        apply_policy(apply_scored, thresholds_for_target, target, disabled_ranks)
        for target, thresholds_for_target in by_target.items()
    ]
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    summary = summarize(assignments)
    feature_inventory = pd.DataFrame(
        {
            "feature": features,
            "train_non_null": [int(train_frame[col].notna().sum()) for col in features],
            "apply_non_null": [int(apply_frame[col].notna().sum()) for col in features],
        }
    )

    thresholds.to_csv(args.output_dir / "marker_mirror_candidate_generator_rank_apply_thresholds.csv", index=False)
    assignments.to_csv(args.output_dir / "marker_mirror_candidate_generator_rank_apply_assignments.csv", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_candidate_generator_rank_apply_summary.csv", index=False)
    feature_inventory.to_csv(args.output_dir / "marker_mirror_candidate_generator_rank_apply_feature_inventory.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/apply_marker_mirror_integrated_rank_calibrator.py",
        "train_evidence_table": str(args.train_evidence_table),
        "apply_evidence_table": str(args.apply_evidence_table),
        "targets": targets,
        "train_split": args.train_split,
        "threshold_split": args.threshold_split,
        "train_direction": args.train_direction,
        "apply_direction": args.apply_direction,
        "train_model": args.train_model,
        "model_type": args.model_type,
        "disabled_ranks": sorted(disabled_ranks),
        "feature_count": len(features),
        "rows": {
            "train_evidence": int(len(train_frame)),
            "apply_evidence": int(len(apply_frame)),
            "assignments": int(len(assignments)),
            "summary": int(len(summary)),
        },
        "claim_boundary": "Production-style apply diagnostic. Labels, if present, are used only for evaluation summaries; final production validation still needs larger external/field tests.",
    }
    (args.output_dir / "marker_mirror_candidate_generator_rank_apply_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
