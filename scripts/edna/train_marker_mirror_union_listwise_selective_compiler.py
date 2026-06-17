#!/usr/bin/env python3
"""Train a list-level selective compiler for MarkerMirror union candidates.

Exp 105 showed that a top-1 HGB compiler was not enough.  This script uses
list-level evidence instead:

* MarkerMirror 12S->16S candidate-list concentration;
* same-marker 12S k-mer candidate-list concentration;
* edlib-reranked same-marker candidate-list concentration;
* agreement between candidate sources at genus/family/order;
* production-available score/margin/identity features.

Labels are used only for train/calibration/evaluation.  Candidate generation
and features themselves do not require hidden truth labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
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

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
RANK_ORDER = ("species", "genus", "family", "order")
DEFAULT_ENABLED_RANKS = ("family", "order")
TOP_KS = (1, 5, 10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--union-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "union_candidate_rank_policy"
        / "marker_mirror_union_production_candidates.csv.gz",
    )
    parser.add_argument(
        "--edlib-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_same_marker_edlib_candidates_top50.csv.gz",
    )
    parser.add_argument(
        "--query-table",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "zero_shot_queries.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "union_listwise_selective_compiler",
    )
    parser.add_argument(
        "--source-table-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--enabled-ranks", default="family,order")
    parser.add_argument("--targets", default="0.95,0.99")
    parser.add_argument("--model-type", choices=("hgb", "logistic"), default="hgb")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_queries(path: Path) -> pd.DataFrame:
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
            "query_sequence_length": frame["nucleotides"].astype(str).map(len),
        }
    )


def mode_stats(values: pd.Series) -> tuple[str, int, float, int]:
    cleaned = [clean(value) for value in values if clean(value)]
    if not cleaned:
        return "", 0, math.nan, 0
    mode, count = Counter(cleaned).most_common(1)[0]
    return mode, int(count), float(count) / float(len(cleaned)), int(len(set(cleaned)))


def margin(values: pd.Series, descending: bool = True) -> float:
    cleaned = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(cleaned) < 2:
        return math.nan
    sorted_values = np.sort(cleaned.to_numpy())
    if descending:
        sorted_values = sorted_values[::-1]
    return float(sorted_values[0] - sorted_values[1])


def list_features_for_source(group: pd.DataFrame, prefix: str, rank_col: str, score_col: str) -> dict[str, Any]:
    group = group.sort_values([rank_col, "candidate_rank"], ascending=[True, True]).copy()
    out: dict[str, Any] = {f"{prefix}_candidate_count": int(len(group))}
    if group.empty:
        return out

    score = pd.to_numeric(group[score_col], errors="coerce") if score_col in group.columns else pd.Series(dtype=float)
    out[f"{prefix}_top1_score"] = float(score.iloc[0]) if len(score) and not pd.isna(score.iloc[0]) else math.nan
    out[f"{prefix}_score_margin_1_2"] = margin(score)
    out[f"{prefix}_score_mean_top10"] = float(score.head(10).mean()) if len(score) else math.nan
    out[f"{prefix}_score_std_top10"] = float(score.head(10).std()) if len(score) > 1 else math.nan
    if "edlib_best_identity" in group.columns:
        identity = pd.to_numeric(group["edlib_best_identity"], errors="coerce")
        out[f"{prefix}_top1_edlib_identity"] = float(identity.iloc[0]) if len(identity) else math.nan
        out[f"{prefix}_edlib_identity_margin_1_2"] = margin(identity)
        out[f"{prefix}_edlib_identity_mean_top10"] = float(identity.head(10).mean())
    if "score" in group.columns and score_col != "score":
        kmer_score = pd.to_numeric(group["score"], errors="coerce")
        out[f"{prefix}_top1_kmer_score"] = float(kmer_score.iloc[0]) if len(kmer_score) else math.nan

    for top_k in TOP_KS:
        top = group.head(top_k)
        out[f"{prefix}_top{top_k}_n"] = int(len(top))
        for rank in ("species", "genus", "family", "order"):
            column = "candidate_tree_label" if rank == "species" else f"candidate_{rank}"
            mode, count, fraction, unique = mode_stats(top[column]) if column in top.columns else ("", 0, math.nan, 0)
            out[f"{prefix}_top{top_k}_{rank}_mode"] = mode
            out[f"{prefix}_top{top_k}_{rank}_mode_count"] = count
            out[f"{prefix}_top{top_k}_{rank}_mode_fraction"] = fraction
            out[f"{prefix}_top{top_k}_{rank}_unique"] = unique
    return out


def build_feature_frame(queries: pd.DataFrame, union: pd.DataFrame, edlib: pd.DataFrame) -> pd.DataFrame:
    mm = union[union["candidate_source"] == "marker_mirror_12s_to_16s"].copy()
    kmer = union[union["candidate_source"] == "same_marker_12s_kmer"].copy()
    edlib = edlib.copy()
    source_groups = {
        "mm": (mm.groupby("query_id", sort=False), "candidate_rank", "score"),
        "kmer": (kmer.groupby("query_id", sort=False), "candidate_rank", "score"),
        "edlib": (edlib.groupby("query_id", sort=False), "edlib_candidate_rank", "edlib_best_identity"),
    }

    rows: list[dict[str, Any]] = []
    for query in queries.itertuples(index=False):
        row: dict[str, Any] = query._asdict()
        query_id = row["query_id"]
        for prefix, (groups, rank_col, score_col) in source_groups.items():
            group = groups.get_group(query_id) if query_id in groups.groups else pd.DataFrame()
            row.update(list_features_for_source(group, prefix, rank_col, score_col))

        for top_k in TOP_KS:
            for rank in ("genus", "family", "order"):
                modes = {
                    prefix: clean(row.get(f"{prefix}_top{top_k}_{rank}_mode"))
                    for prefix in ("mm", "kmer", "edlib")
                }
                row[f"agree_mm_kmer_top{top_k}_{rank}"] = bool(modes["mm"] and modes["mm"] == modes["kmer"])
                row[f"agree_mm_edlib_top{top_k}_{rank}"] = bool(modes["mm"] and modes["mm"] == modes["edlib"])
                row[f"agree_kmer_edlib_top{top_k}_{rank}"] = bool(modes["kmer"] and modes["kmer"] == modes["edlib"])
                row[f"agree_all_sources_top{top_k}_{rank}"] = bool(
                    modes["mm"] and modes["mm"] == modes["kmer"] == modes["edlib"]
                )

        # Proposed production-safe taxon: use edlib top10 mode when available,
        # otherwise same-marker k-mer top10, otherwise MarkerMirror top10.
        for rank in ("genus", "family", "order"):
            proposed = (
                clean(row.get(f"edlib_top10_{rank}_mode"))
                or clean(row.get(f"kmer_top10_{rank}_mode"))
                or clean(row.get(f"mm_top10_{rank}_mode"))
            )
            row[f"proposed_{rank}_taxon"] = proposed
            row[f"proposed_{rank}_correct"] = bool(proposed and proposed == clean(row.get(f"query_{rank}")))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("query_", "proposed_")
    blocked = {"query_id", "source"}
    features: list[str] = []
    for column in frame.columns:
        if column in blocked or any(column.startswith(prefix) for prefix in blocked_prefixes):
            continue
        if pd.api.types.is_bool_dtype(frame[column]) or pd.api.types.is_numeric_dtype(frame[column]):
            if frame[column].notna().sum() > 0:
                features.append(column)
    return sorted(features)


def make_model(model_type: str, seed: int):
    if model_type == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            l2_regularization=0.08,
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


def numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame[features].copy()
    for column in out.columns:
        if pd.api.types.is_bool_dtype(out[column]):
            out[column] = out[column].astype(int)
    return out


def train_models(
    train: pd.DataFrame,
    features: list[str],
    ranks: list[str],
    model_type: str,
    seed: int,
    logger: ProgressLogger,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for rank in ranks:
        label = train[f"proposed_{rank}_correct"].astype(bool).astype(int)
        if label.nunique() < 2:
            logger.log(f"rank={rank} skipped; fewer than two label classes")
            models[rank] = None
            continue
        model = make_model(model_type, seed)
        model.fit(numeric_matrix(train, features), label)
        models[rank] = model
        logger.log(f"rank={rank} trained rows={len(train):,} positive_rate={100.0 * float(label.mean()):.2f}%")
    return models


def score(frame: pd.DataFrame, features: list[str], models: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    x = numeric_matrix(out, features)
    for rank, model in models.items():
        out[f"prob_{rank}_correct"] = np.nan if model is None else model.predict_proba(x)[:, 1]
    return out


def fit_threshold(scored: pd.DataFrame, rank: str, target: float) -> dict[str, Any]:
    prob_col = f"prob_{rank}_correct"
    label_col = f"proposed_{rank}_correct"
    sub = scored.dropna(subset=[prob_col]).copy()
    if sub.empty:
        return {"rank": rank, "target_precision": target, "threshold": None, "fit_status": "no_scores"}
    probs = sub[prob_col].to_numpy(dtype=float)
    labels = sub[label_col].astype(bool).to_numpy()
    best = None
    fallback = None
    for threshold in np.sort(np.unique(probs))[::-1]:
        keep = probs >= threshold
        if not keep.any():
            continue
        precision = float(labels[keep].mean())
        coverage = pct(int(keep.sum()), len(sub))
        row = (float(threshold), precision, coverage, int(keep.sum()))
        if fallback is None or (precision, coverage) > (fallback[1], fallback[2]):
            fallback = row
        if precision >= target:
            best = row
    status = "target_met" if best else "target_not_met_best_available"
    threshold, precision, coverage, n = best or fallback or (math.nan, math.nan, math.nan, 0)
    return {
        "rank": rank,
        "target_precision": target,
        "threshold": threshold,
        "fit_precision_pct": 100.0 * precision,
        "fit_coverage_pct": coverage,
        "fit_n": int(n),
        "fit_status": status,
    }


def apply_policy(scored: pd.DataFrame, ranks: list[str], thresholds: dict[str, dict[str, Any]], target: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in scored.iterrows():
        assigned_rank = "no_call"
        assigned_taxon = ""
        assigned_probability = math.nan
        assigned_correct: bool | None = None
        for rank in ranks:
            threshold = thresholds.get(rank, {}).get("threshold")
            probability = row.get(f"prob_{rank}_correct")
            if threshold is None or pd.isna(probability):
                continue
            if float(probability) >= float(threshold):
                assigned_rank = rank
                assigned_taxon = clean(row.get(f"proposed_{rank}_taxon"))
                assigned_probability = float(probability)
                assigned_correct = bool(row.get(f"proposed_{rank}_correct"))
                break
        rows.append(
            {
                "target_precision": target,
                "query_id": row["query_id"],
                "query_tree_label": row["query_tree_label"],
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


def summarize(per_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in per_split.groupby(["model_type", "target_precision", "enabled_ranks"], dropna=False):
        model_type, target, ranks = key
        rows.append(
            {
                "model_type": model_type,
                "target_precision": float(target),
                "enabled_ranks": ranks,
                "repeats": int(len(group)),
                "mean_coverage_pct": float(group["coverage_pct"].mean()),
                "mean_assigned_precision_pct": float(group["assigned_precision_pct"].mean()),
                "precision_p05_pct": float(group["assigned_precision_pct"].quantile(0.05)),
                "precision_p95_pct": float(group["assigned_precision_pct"].quantile(0.95)),
                "target_met_rate_pct": 100.0 * float((group["assigned_precision_pct"] >= 100.0 * float(target)).mean()),
                "mean_genus_assignment_n": float(group["genus_assignment_n"].mean()),
                "mean_family_assignment_n": float(group["family_assignment_n"].mean()),
                "mean_order_assignment_n": float(group["order_assignment_n"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.source_table_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or args.output_dir / "marker_mirror_union_listwise_selective_compiler.log")
    script_name = Path(__file__).name
    logger.start(script_name)

    ranks = [rank.strip() for rank in args.enabled_ranks.split(",") if rank.strip()]
    targets = [float(value.strip()) for value in args.targets.split(",") if value.strip()]
    logger.log(f"Loading queries {rel(args.query_table)}")
    queries = load_queries(args.query_table)
    logger.log(f"Loading union candidates {rel(args.union_candidates)}")
    union = pd.read_csv(args.union_candidates)
    logger.log(f"Loading edlib candidates {rel(args.edlib_candidates)}")
    edlib = pd.read_csv(args.edlib_candidates)
    logger.log(f"Building listwise feature frame queries={len(queries):,}")
    frame = build_feature_frame(queries, union, edlib)
    features = feature_columns(frame)
    logger.log(f"Feature frame rows={len(frame):,} features={len(features):,} ranks={ranks}")

    feature_frame_path = args.output_dir / "marker_mirror_union_listwise_feature_frame.csv.gz"
    frame.to_csv(feature_frame_path, index=False)

    rng = np.random.default_rng(args.seed)
    per_split_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    for repeat in range(args.repeats):
        train, cal, eva = split_species(frame, rng, args.train_fraction, args.calibration_fraction)
        if train.empty or cal.empty or eva.empty:
            continue
        models = train_models(train, features, ranks, args.model_type, args.seed + repeat, logger)
        cal_scored = score(cal, features, models)
        eval_scored = score(eva, features, models)
        for target in targets:
            threshold_by_rank: dict[str, dict[str, Any]] = {}
            for rank in ranks:
                fit = fit_threshold(cal_scored, rank, target)
                fit.update({"repeat": repeat, "model_type": args.model_type, "enabled_ranks": ",".join(ranks)})
                threshold_rows.append(fit)
                threshold_by_rank[rank] = fit
            assignments = apply_policy(eval_scored, ranks, threshold_by_rank, target)
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
            logger.log(f"Completed repeat {repeat + 1}/{args.repeats}")

    per_split = pd.DataFrame(per_split_rows)
    summary = summarize(per_split)
    thresholds = pd.DataFrame(threshold_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    feature_inventory = pd.DataFrame({"feature": features})

    prefix = "marker_mirror_union_listwise_selective"
    per_split.to_csv(args.output_dir / f"{prefix}_per_split.csv", index=False)
    summary.to_csv(args.output_dir / f"{prefix}_summary.csv", index=False)
    thresholds.to_csv(args.output_dir / f"{prefix}_thresholds.csv", index=False)
    feature_inventory.to_csv(args.output_dir / f"{prefix}_features.csv", index=False)
    if not assignments.empty:
        assignments.to_csv(args.output_dir / f"{prefix}_assignments.csv.gz", index=False)

    per_split.to_csv(args.source_table_dir / f"{prefix}_per_split.csv", index=False)
    summary.to_csv(args.source_table_dir / f"{prefix}_summary.csv", index=False)
    thresholds.to_csv(args.source_table_dir / f"{prefix}_thresholds.csv", index=False)
    feature_inventory.to_csv(args.source_table_dir / f"{prefix}_features.csv", index=False)
    source_suffix = "_".join(ranks)
    per_split.to_csv(args.source_table_dir / f"{prefix}_{source_suffix}_per_split.csv", index=False)
    summary.to_csv(args.source_table_dir / f"{prefix}_{source_suffix}_summary.csv", index=False)
    thresholds.to_csv(args.source_table_dir / f"{prefix}_{source_suffix}_thresholds.csv", index=False)
    feature_inventory.to_csv(args.source_table_dir / f"{prefix}_{source_suffix}_features.csv", index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "union_candidates": rel(args.union_candidates),
            "edlib_candidates": rel(args.edlib_candidates),
            "query_table": rel(args.query_table),
        },
        "outputs": {
            "feature_frame": rel(feature_frame_path),
            "summary": rel(args.output_dir / f"{prefix}_summary.csv"),
            "per_split": rel(args.output_dir / f"{prefix}_per_split.csv"),
            "thresholds": rel(args.output_dir / f"{prefix}_thresholds.csv"),
            "features": rel(args.output_dir / f"{prefix}_features.csv"),
        },
        "model_type": args.model_type,
        "enabled_ranks": ranks,
        "targets": targets,
        "repeats": args.repeats,
        "feature_count": len(features),
        "claim_boundary": "Diagnostic list-level selective compiler over existing union candidate lists. Species disabled by default; not field-eDNA production locked.",
    }
    (args.output_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.source_table_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote {rel(args.output_dir / f'{prefix}_summary.csv')}")
    logger.done(script_name)


if __name__ == "__main__":
    main()
