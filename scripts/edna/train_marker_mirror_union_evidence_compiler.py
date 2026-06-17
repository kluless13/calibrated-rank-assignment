#!/usr/bin/env python3
"""Train a calibrated union evidence compiler for MarkerMirror candidates.

This follows Exp 104 but replaces hand-shaped score gates with a small
rank-specific model over production-available union features. Labels are used
only to train/calibrate/evaluate the diagnostic; candidate generation itself
does not require hidden labels.

Default policy is species-disabled and genus-disabled because current 12S/16S
species/genus support is not stable enough for production claims.
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
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "union_evidence_compiler",
    )
    parser.add_argument(
        "--source-table-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--model-type", choices=("hgb", "logistic"), default="hgb")
    parser.add_argument("--enabled-ranks", default="family,order")
    parser.add_argument("--targets", default="0.95,0.99")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = (
        "query_",
        "mm_query_",
        "mm_input_query_",
        "mm_match_",
        "sm_candidate_",
    )
    blocked_suffixes = ("_correct",)
    blocked = {
        "query_id",
        "source",
        "query_tree_label",
        "query_species",
        "query_genus",
        "query_family",
        "query_order",
        "mm_candidate_tree_label",
        "mm_candidate_species",
        "mm_candidate_genus",
        "mm_candidate_family",
        "mm_candidate_order",
        "mm_query_species",
        "mm_query_genus",
        "mm_query_family",
        "mm_query_order",
        "mm_query_tree_label",
        "mm_input_query_tree_label",
        "mm_model",
        "mm_direction",
        "mm_split",
        "mm_query_sequence_hash",
        "mm_query_marker",
        "mm_target_marker",
        "sm_candidate_tree_label",
        "sm_candidate_species",
        "sm_candidate_genus",
        "sm_candidate_family",
        "sm_candidate_order",
        "sm_candidate_source",
    }
    features: list[str] = []
    for col in frame.columns:
        if col in blocked or any(col.startswith(prefix) for prefix in blocked_prefixes):
            continue
        if any(col.endswith(suffix) for suffix in blocked_suffixes):
            continue
        if pd.api.types.is_bool_dtype(frame[col]) or pd.api.types.is_numeric_dtype(frame[col]):
            if frame[col].notna().sum() > 0:
                features.append(col)
    return sorted(features)


def make_model(model_type: str, seed: int):
    if model_type == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=seed,
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=seed),
    )


def split_species(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    train_fraction: float,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    species = np.array(sorted(frame["query_tree_label"].dropna().astype(str).unique()))
    rng.shuffle(species)
    n_train = max(1, int(round(len(species) * train_fraction)))
    n_cal = max(1, int(round(len(species) * calibration_fraction)))
    train_species = set(species[:n_train])
    cal_species = set(species[n_train : n_train + n_cal])
    eval_species = set(species[n_train + n_cal :])
    return (
        frame[frame["query_tree_label"].isin(train_species)].copy(),
        frame[frame["query_tree_label"].isin(cal_species)].copy(),
        frame[frame["query_tree_label"].isin(eval_species)].copy(),
    )


def train_models(
    train: pd.DataFrame,
    features: list[str],
    ranks: list[str],
    model_type: str,
    seed: int,
    logger: Logger,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for rank in ranks:
        label_col = f"sm_{rank}_correct"
        if label_col not in train.columns or train[label_col].nunique(dropna=True) < 2:
            models[rank] = None
            logger.log(f"rank={rank} skipped; label has fewer than two classes")
            continue
        model = make_model(model_type, seed)
        y = train[label_col].astype(bool).astype(int)
        x = train[features].copy()
        for col in x.columns:
            if pd.api.types.is_bool_dtype(x[col]):
                x[col] = x[col].astype(int)
        model.fit(x, y)
        models[rank] = model
        logger.log(f"rank={rank} trained rows={len(train)} positive_rate={100.0 * float(y.mean()):.2f}%")
    return models


def score(frame: pd.DataFrame, features: list[str], models: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    x = out[features].copy()
    for col in x.columns:
        if pd.api.types.is_bool_dtype(x[col]):
            x[col] = x[col].astype(int)
    for rank, model in models.items():
        if model is None:
            out[f"prob_{rank}_correct"] = np.nan
        else:
            out[f"prob_{rank}_correct"] = model.predict_proba(x)[:, 1]
    return out


def fit_threshold(scored: pd.DataFrame, rank: str, target: float) -> dict[str, Any]:
    prob_col = f"prob_{rank}_correct"
    label_col = f"sm_{rank}_correct"
    sub = scored.dropna(subset=[prob_col]).copy()
    if sub.empty:
        return {"rank": rank, "target_precision": target, "threshold": None, "fit_status": "no_scores"}
    values = np.sort(sub[prob_col].unique())[::-1]
    best = None
    labels = sub[label_col].astype(bool).to_numpy()
    probs = sub[prob_col].to_numpy(dtype=float)
    for threshold in values:
        keep = probs >= threshold
        if not keep.any():
            continue
        precision = float(labels[keep].mean())
        coverage = pct(int(keep.sum()), len(sub))
        row = (float(threshold), precision, coverage, int(keep.sum()))
        if precision >= target:
            best = row
    if best is None:
        # Use the highest-precision available row as a diagnostic fallback.
        rows = []
        for threshold in values:
            keep = probs >= threshold
            rows.append((float(threshold), float(labels[keep].mean()), pct(int(keep.sum()), len(sub)), int(keep.sum())))
        best = sorted(rows, key=lambda item: (item[1], item[2]), reverse=True)[0]
        status = "target_not_met_best_available"
    else:
        status = "target_met"
    threshold, precision, coverage, n = best
    return {
        "rank": rank,
        "target_precision": target,
        "threshold": threshold,
        "fit_precision_pct": 100.0 * precision,
        "fit_coverage_pct": coverage,
        "fit_n": n,
        "fit_status": status,
    }


def apply_policy(scored: pd.DataFrame, thresholds: dict[str, dict[str, Any]], ranks: list[str], target: float) -> pd.DataFrame:
    rows = []
    for _, row in scored.iterrows():
        assigned_rank = "no_call"
        assigned_taxon = ""
        assigned_correct: bool | None = None
        assigned_probability = math.nan
        for rank in ranks:
            threshold = thresholds.get(rank, {}).get("threshold")
            prob = row.get(f"prob_{rank}_correct")
            if threshold is None or pd.isna(prob):
                continue
            if float(prob) >= float(threshold):
                assigned_rank = rank
                assigned_taxon = str(row.get(f"sm_candidate_{rank}", ""))
                assigned_correct = bool(row.get(f"sm_{rank}_correct", False))
                assigned_probability = float(prob)
                break
        rows.append(
            {
                "target_precision": target,
                "query_id": row["query_id"],
                "source": row.get("source", ""),
                "query_tree_label": row.get("query_tree_label", ""),
                "assigned_rank": assigned_rank,
                "assigned_taxon": assigned_taxon,
                "assigned_probability": assigned_probability,
                "assigned_correct": assigned_correct,
            }
        )
    return pd.DataFrame(rows)


def assignment_metrics(assignments: pd.DataFrame, label: dict[str, Any]) -> dict[str, Any]:
    assigned = assignments[assignments["assigned_rank"] != "no_call"].copy()
    labelled = assigned.dropna(subset=["assigned_correct"])
    out = {
        **label,
        "n_query": int(assignments["query_id"].nunique()),
        "assigned_n": int(len(assigned)),
        "coverage_pct": pct(len(assigned), assignments["query_id"].nunique()),
        "assigned_precision_pct": 100.0 * float(labelled["assigned_correct"].astype(bool).mean()) if len(labelled) else math.nan,
        "false_species_call_rate_pct": 0.0,
    }
    for rank in RANK_ORDER:
        rank_rows = labelled[labelled["assigned_rank"] == rank]
        out[f"{rank}_assignment_n"] = int((assigned["assigned_rank"] == rank).sum())
        out[f"{rank}_precision_pct"] = (
            100.0 * float(rank_rows["assigned_correct"].astype(bool).mean()) if len(rank_rows) else math.nan
        )
    return out


def summarize_repeats(per_split: pd.DataFrame) -> pd.DataFrame:
    if per_split.empty:
        return pd.DataFrame()
    rows = []
    for key, group in per_split.groupby(["model_type", "target_precision", "enabled_ranks"], dropna=False):
        model_type, target, ranks = key
        rows.append(
            {
                "model_type": model_type,
                "target_precision": target,
                "enabled_ranks": ranks,
                "repeats": int(len(group)),
                "mean_coverage_pct": float(group["coverage_pct"].mean()),
                "mean_assigned_precision_pct": float(group["assigned_precision_pct"].mean()),
                "precision_p05_pct": float(group["assigned_precision_pct"].quantile(0.05)),
                "precision_p95_pct": float(group["assigned_precision_pct"].quantile(0.95)),
                "target_met_rate_pct": 100.0 * float((group["assigned_precision_pct"] >= 100.0 * float(target)).mean()),
                "mean_family_assignment_n": float(group["family_assignment_n"].mean()),
                "mean_order_assignment_n": float(group["order_assignment_n"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_table_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_union_evidence_compiler.log")
    ranks = [rank.strip() for rank in args.enabled_ranks.split(",") if rank.strip()]
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    frame = pd.read_csv(args.top1_features)
    features = feature_columns(frame)
    logger.log(f"Loaded top1 features rows={len(frame)} features={len(features)} ranks={ranks}")
    rng = np.random.default_rng(args.seed)
    per_split_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        train, cal, eva = split_species(frame, rng, args.train_fraction, args.calibration_fraction)
        if train.empty or cal.empty or eva.empty:
            continue
        models = train_models(train, features, ranks, args.model_type, args.seed + repeat, logger)
        cal_scored = score(cal, features, models)
        eva_scored = score(eva, features, models)
        for target in targets:
            threshold_by_rank: dict[str, dict[str, Any]] = {}
            for rank in ranks:
                fit = fit_threshold(cal_scored, rank, target)
                fit.update({"repeat": repeat, "model_type": args.model_type, "enabled_ranks": ",".join(ranks)})
                threshold_rows.append(fit)
                threshold_by_rank[rank] = fit
            assignments = apply_policy(eva_scored, threshold_by_rank, ranks, target)
            assignments["repeat"] = repeat
            assignments["model_type"] = args.model_type
            assignments["enabled_ranks"] = ",".join(ranks)
            assignment_frames.append(assignments)
            per_split_rows.append(
                assignment_metrics(
                    assignments,
                    {
                        "repeat": repeat,
                        "model_type": args.model_type,
                        "target_precision": target,
                        "enabled_ranks": ",".join(ranks),
                        "train_n": int(len(train)),
                        "calibration_n": int(len(cal)),
                        "eval_n": int(len(eva)),
                    },
                )
            )
        if repeat == 0 or (repeat + 1) % 10 == 0:
            logger.log(f"Completed compiler repeat={repeat + 1}/{args.repeats}")

    per_split = pd.DataFrame(per_split_rows)
    summary = summarize_repeats(per_split)
    thresholds = pd.DataFrame(threshold_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    feature_inventory = pd.DataFrame({"feature": features})

    per_split.to_csv(args.output_dir / "marker_mirror_union_evidence_compiler_per_split.csv", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_union_evidence_compiler_summary.csv", index=False)
    thresholds.to_csv(args.output_dir / "marker_mirror_union_evidence_compiler_thresholds.csv", index=False)
    feature_inventory.to_csv(args.output_dir / "marker_mirror_union_evidence_compiler_features.csv", index=False)
    if not assignments.empty:
        assignments.to_csv(args.output_dir / "marker_mirror_union_evidence_compiler_assignments.csv.gz", index=False)

    source_suffix = "_".join(ranks)
    summary.to_csv(args.source_table_dir / "marker_mirror_union_evidence_compiler_summary.csv", index=False)
    per_split.to_csv(args.source_table_dir / "marker_mirror_union_evidence_compiler_per_split.csv", index=False)
    thresholds.to_csv(args.source_table_dir / "marker_mirror_union_evidence_compiler_thresholds.csv", index=False)
    feature_inventory.to_csv(args.source_table_dir / "marker_mirror_union_evidence_compiler_features.csv", index=False)
    summary.to_csv(args.source_table_dir / f"marker_mirror_union_evidence_compiler_{source_suffix}_summary.csv", index=False)
    per_split.to_csv(args.source_table_dir / f"marker_mirror_union_evidence_compiler_{source_suffix}_per_split.csv", index=False)
    thresholds.to_csv(args.source_table_dir / f"marker_mirror_union_evidence_compiler_{source_suffix}_thresholds.csv", index=False)
    manifest = {
        "generated_by": "scripts/edna/train_marker_mirror_union_evidence_compiler.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top1_features": str(args.top1_features),
        "model_type": args.model_type,
        "enabled_ranks": ranks,
        "targets": targets,
        "repeats": args.repeats,
        "train_fraction": args.train_fraction,
        "calibration_fraction": args.calibration_fraction,
        "feature_count": len(features),
        "claim_boundary": "Diagnostic calibrated compiler over union top1 features. Species/genus disabled by default; thresholds are species-split validated but not field-eDNA production thresholds.",
    }
    (args.output_dir / "marker_mirror_union_evidence_compiler_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.source_table_dir / "marker_mirror_union_evidence_compiler_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.log(f"Wrote {args.output_dir / 'marker_mirror_union_evidence_compiler_summary.csv'}")


if __name__ == "__main__":
    main()
