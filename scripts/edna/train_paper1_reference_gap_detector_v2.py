#!/usr/bin/env python3
"""Train a candidate-evidence reference-gap detector for Paper 1.

This v2 detector keeps the no-leakage stance of the honest v1 detector, but
uses richer top-k candidate evidence:

- score and p-distance traces;
- candidate taxonomic diversity and consensus;
- nearest-candidate reference density;
- inference-safe tree-neighborhood distances among retrieved candidates.

It does not use global candidate-set sizes or split-specific flags.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402
from train_paper1_candidate_reranker import TreeDistanceCache, load_tree_distances  # noqa: E402
from train_paper1_coi_evidence_model import clean, parse_json_list, safe_float, train_calib_split  # noqa: E402
from train_paper1_reference_gap_detector import (  # noqa: E402
    GAP_RANKS,
    add_gap_targets,
    default_runs,
    predict_probabilities,
    resolve_input_dir,
    set_seed,
    standardize,
    summarize_split,
    train_model,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_OUTPUT_DIR = PAPER1 / "reference_gap_detector" / "coi_mlp_seed1301_v2_candidate_evidence_target095"
DEFAULT_TREE_FILE = ROOT / "data" / "phylo" / "actinopt_12k_treePL.tre"


def normalize_species(value: Any) -> str:
    return clean(value).replace(" ", "_")


def finite(values: list[Any]) -> np.ndarray:
    numeric = np.array([safe_float(value) for value in values], dtype=float)
    return numeric[np.isfinite(numeric)]


def stats(values: list[Any], prefix: str, lower_is_better: bool = False) -> dict[str, float]:
    numeric = finite(values)
    out = {
        f"{prefix}_finite_count": float(len(numeric)),
        f"{prefix}_first": float(numeric[0]) if len(numeric) else np.nan,
        f"{prefix}_second": float(numeric[1]) if len(numeric) > 1 else np.nan,
        f"{prefix}_mean": float(numeric.mean()) if len(numeric) else np.nan,
        f"{prefix}_std": float(numeric.std()) if len(numeric) else np.nan,
        f"{prefix}_min": float(numeric.min()) if len(numeric) else np.nan,
        f"{prefix}_max": float(numeric.max()) if len(numeric) else np.nan,
    }
    if len(numeric) > 1:
        out[f"{prefix}_gap_1_2"] = (
            float(numeric[1] - numeric[0]) if lower_is_better else float(numeric[0] - numeric[1])
        )
    else:
        out[f"{prefix}_gap_1_2"] = np.nan
    return out


def finite_stats(values: list[float]) -> tuple[float, float, float]:
    numeric = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if not len(numeric):
        return np.nan, np.nan, np.nan
    return float(numeric.min()), float(numeric.mean()), float(numeric.max())


def taxonomy_maps(input_dir: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(input_dir / "candidate_species.csv")
    maps: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        label = normalize_species(row.get("tree_label"))
        if not label:
            continue
        maps[label] = {
            "genus": clean(row.get("genus_name")),
            "family": clean(row.get("family_name")),
            "order": clean(row.get("order_name")),
            "reference_sequence_count": safe_float(row.get("reference_sequence_count"), 0.0),
            "has_reference_sequence": safe_float(row.get("has_reference_sequence"), 0.0),
        }
    return maps


def top_taxonomy_features(labels: list[str], taxa: dict[str, dict[str, Any]], top_k: int) -> dict[str, float]:
    labels = [normalize_species(label) for label in labels[:top_k] if clean(label)]
    top = labels[0] if labels else ""
    top_tax = taxa.get(top, {})
    out: dict[str, float] = {
        f"top{top_k}_candidate_count": float(len(labels)),
        f"top{top_k}_top1_reference_sequence_count": safe_float(top_tax.get("reference_sequence_count"), 0.0),
        f"top{top_k}_top1_has_reference_sequence": safe_float(top_tax.get("has_reference_sequence"), 0.0),
    }
    for rank in ("genus", "family", "order"):
        values = [clean(taxa.get(label, {}).get(rank)) for label in labels]
        values = [value for value in values if value]
        top_value = clean(top_tax.get(rank))
        out[f"top{top_k}_unique_{rank}_count"] = float(len(set(values)))
        out[f"top{top_k}_same_{rank}_as_top1_fraction"] = (
            float(sum(1 for value in values if value == top_value) / len(values))
            if values and top_value
            else np.nan
        )
    counts = finite([taxa.get(label, {}).get("reference_sequence_count", np.nan) for label in labels])
    out[f"top{top_k}_reference_sequence_count_mean"] = float(counts.mean()) if len(counts) else np.nan
    out[f"top{top_k}_reference_sequence_count_max"] = float(counts.max()) if len(counts) else np.nan
    return out


def pdistance_threshold_features(values: list[Any], prefix: str) -> dict[str, float]:
    numeric = finite(values)
    out: dict[str, float] = {}
    for threshold in (0.01, 0.02, 0.05, 0.10, 0.20):
        safe_name = str(threshold).replace(".", "p")
        out[f"{prefix}_count_le_{safe_name}"] = float((numeric <= threshold).sum()) if len(numeric) else 0.0
        out[f"{prefix}_frac_le_{safe_name}"] = float((numeric <= threshold).mean()) if len(numeric) else np.nan
    return out


def tree_features(labels: list[str], distances: TreeDistanceCache | None, top_k: int) -> dict[str, float]:
    if distances is None:
        return {}
    labels = [normalize_species(label) for label in labels[:top_k] if clean(label)]
    top = labels[0] if labels else ""
    pair_distances: list[float] = []
    for left_idx, left in enumerate(labels):
        for right in labels[left_idx + 1 :]:
            pair_distances.append(distances.distance(left, right))
    pair_min, pair_mean, pair_max = finite_stats(pair_distances)
    top_distances = [distances.distance(top, label) for label in labels[1:] if top and label]
    top_min, top_mean, top_max = finite_stats(top_distances)
    return {
        f"top{top_k}_tree_pairwise_min": pair_min,
        f"top{top_k}_tree_pairwise_mean": pair_mean,
        f"top{top_k}_tree_pairwise_max": pair_max,
        f"top{top_k}_tree_distance_from_top1_min": top_min,
        f"top{top_k}_tree_distance_from_top1_mean": top_mean,
        f"top{top_k}_tree_distance_from_top1_max": top_max,
    }


def load_v2_run(run: Any, distances: TreeDistanceCache | None, logger: ProgressLogger) -> pd.DataFrame:
    input_dir = resolve_input_dir(run.run_dir)
    taxa = taxonomy_maps(input_dir)
    pred_path = run.run_dir / "pipeline_candidate_predictions.csv"
    logger.log(f"Loading candidate evidence split={run.split} role={run.role} from {pred_path}")
    predictions = pd.read_csv(pred_path)
    rows: list[dict[str, Any]] = []
    for idx, row in predictions.iterrows():
        labels = [normalize_species(label) for label in parse_json_list(row.get("top_tree_labels"))]
        scores = parse_json_list(row.get("top_scores"))
        pdistances = parse_json_list(row.get("top_pdistances"))
        item: dict[str, Any] = {
            "processid": row.get("processid"),
            "split": run.split,
            "role": run.role,
            "gap_run_name": run.run_dir.name,
            "candidate_input_dir": str(input_dir),
            "true_tree_label": normalize_species(row.get("true_tree_label")),
            "true_genus": clean(row.get("genus_name")),
            "true_family": clean(row.get("family_name")),
            "true_order": clean(row.get("order_name")),
            "pred_score": safe_float(row.get("pred_score")),
            "pred_pdistance": safe_float(row.get("pred_pdistance")),
            "top_label_known_in_candidate_taxonomy": float(labels[0] in taxa) if labels else 0.0,
        }
        item.update(stats(scores[:10], "top10_score"))
        item.update(stats(scores[:50], "top50_score"))
        item.update(stats(pdistances[:10], "top10_pdistance", lower_is_better=True))
        item.update(stats(pdistances[:50], "top50_pdistance", lower_is_better=True))
        item.update(pdistance_threshold_features(pdistances[:10], "top10_pdistance"))
        item.update(pdistance_threshold_features(pdistances[:50], "top50_pdistance"))
        item.update(top_taxonomy_features(labels, taxa, 10))
        item.update(top_taxonomy_features(labels, taxa, 50))
        item.update(tree_features(labels, distances, 10))
        rows.append(item)
        if idx and idx % 10000 == 0:
            logger.log(f"split={run.split} role={run.role} processed_rows={idx}")
    frame = pd.DataFrame(rows)
    return add_gap_targets(frame, input_dir)


def feature_columns(frame: pd.DataFrame, include_tree: bool) -> list[str]:
    excluded = {
        "processid",
        "split",
        "role",
        "gap_run_name",
        "candidate_input_dir",
        "true_tree_label",
        "true_genus",
        "true_family",
        "true_order",
        "candidate_species_count",
        "candidate_genus_count",
        "candidate_family_count",
        "candidate_order_count",
        *[f"{rank}_supported" for rank in GAP_RANKS],
        *[f"{rank}_gap" for rank in GAP_RANKS],
    }
    cols = []
    for col in frame.columns:
        if col in excluded:
            continue
        if not include_tree and "_tree_" in col:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target_precision: float) -> dict[str, Any]:
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order].astype(bool)
    tp = np.cumsum(sorted_labels)
    n = np.arange(1, len(sorted_probs) + 1)
    precision = tp / n
    recall = tp / max(int(sorted_labels.sum()), 1)
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        return {
            "threshold": math.inf,
            "calibration_precision": np.nan,
            "calibration_recall": 0.0,
            "calibration_assigned": 0,
        }
    idx = int(valid[-1])
    return {
        "threshold": float(sorted_probs[idx]),
        "calibration_precision": float(precision[idx]),
        "calibration_recall": float(recall[idx]),
        "calibration_assigned": int(idx + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tree-file", type=Path, default=DEFAULT_TREE_FILE)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--target-gap-precision", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--no-tree-features", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    distances = None
    if not args.no_tree_features:
        if not args.tree_file.exists():
            raise RuntimeError(f"Missing tree file: {args.tree_file}")
        logger.log(f"Loading tree distances from {args.tree_file}")
        distances = TreeDistanceCache(*load_tree_distances(args.tree_file))
        logger.log(f"Loaded tree labels: {len(distances.taxon_nodes):,}")

    frames = [load_v2_run(run, distances, logger) for run in default_runs()]
    examples = pd.concat(frames, ignore_index=True, sort=False)
    examples_path = args.output_dir / "reference_gap_examples.csv"
    logger.log(f"Writing examples to {examples_path}")
    examples.to_csv(examples_path, index=False)

    feature_names = feature_columns(examples, include_tree=not args.no_tree_features)
    train_mask = examples["role"].isin(["train_supported", "train_gap"]).to_numpy()
    train_frame = examples[train_mask].reset_index(drop=True)
    train_idx_local, calib_idx_local = train_calib_split(len(train_frame), args.calib_fraction, args.seed)
    global_train_indices = np.where(train_mask)[0][train_idx_local]
    global_calib_indices = np.where(train_mask)[0][calib_idx_local]

    all_values = examples[feature_names].to_numpy(dtype=np.float32)
    train_values = examples.iloc[global_train_indices][feature_names].to_numpy(dtype=np.float32)
    x, feature_stats = standardize(train_values, all_values, feature_names)
    y = examples[[f"{rank}_gap" for rank in GAP_RANKS]].to_numpy(dtype=np.float32)
    logger.log(
        f"Training rows={len(global_train_indices)} calib_rows={len(global_calib_indices)} "
        f"features={len(feature_names)} target_gap_precision={args.target_gap_precision}"
    )
    model, history = train_model(x, y, global_train_indices, global_calib_indices, args, logger)
    probs = predict_probabilities(model, x, args.batch_size, args.cpu)
    scored = examples.copy()
    for idx, rank in enumerate(GAP_RANKS):
        scored[f"gap_p_{rank}"] = probs[:, idx]

    threshold_rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    for idx, rank in enumerate(GAP_RANKS):
        item = choose_threshold(probs[global_calib_indices, idx], y[global_calib_indices, idx], args.target_gap_precision)
        thresholds[rank] = float(item["threshold"])
        threshold_rows.append(
            {
                "target_gap_precision": float(args.target_gap_precision),
                "rank": rank,
                **item,
                "calibration_rows": int(len(global_calib_indices)),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for (_split, _role), split_frame in scored.groupby(["split", "role"], sort=True):
        summary_rows.extend(summarize_split(split_frame, thresholds, args.target_gap_precision))

    predictions_path = args.output_dir / "reference_gap_predictions.csv"
    thresholds_path = args.output_dir / "reference_gap_thresholds.csv"
    summary_path = args.output_dir / "reference_gap_summary.csv"
    history_path = args.output_dir / "reference_gap_train_history.csv"
    stats_path = args.output_dir / "reference_gap_feature_stats.json"
    manifest_path = args.output_dir / "reference_gap_manifest.json"
    logger.log(f"Writing predictions to {predictions_path}")
    scored.to_csv(predictions_path, index=False)
    write_csv(thresholds_path, threshold_rows)
    write_csv(summary_path, summary_rows)
    write_csv(history_path, history)
    stats_path.write_text(json.dumps({"features": feature_names, "stats": feature_stats}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "generated_by": "scripts/edna/train_paper1_reference_gap_detector_v2.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "target_gap_precision": float(args.target_gap_precision),
        "tree_features": not args.no_tree_features,
        "tree_file": str(args.tree_file) if not args.no_tree_features else None,
        "gap_ranks": list(GAP_RANKS),
        "features": feature_names,
        "outputs": {
            "examples": str(examples_path),
            "predictions": str(predictions_path),
            "thresholds": str(thresholds_path),
            "summary": str(summary_path),
            "history": str(history_path),
            "feature_stats": str(stats_path),
        },
        "claim_boundary": (
            "Candidate-evidence reference-gap detector v2. Uses candidate-list, "
            "p-distance, taxonomy-diversity, reference-density, and tree-neighborhood "
            "features without global candidate-count or split-specific leakage. "
            "Diagnostic until deployment-matched calibration is finalized."
        ),
    }
    logger.log(f"Writing manifest to {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
