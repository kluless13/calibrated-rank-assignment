#!/usr/bin/env python3
"""Apply the Paper 1 COI DL evidence model to an existing pipeline run.

This is the inference-time companion to
`train_paper1_coi_evidence_model.py`. It starts from a completed
`run_paper1_coi_pipeline.py` output directory, rebuilds the same evidence
features, applies the saved MLP, and writes rank/no-call assignments.

Correctness fields are reported as unavailable for unlabeled specimens rather
than treating them as incorrect.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402
from train_paper1_coi_evidence_model import (  # noqa: E402
    RANKS,
    EvidenceMLP,
    RunFiles,
    clean,
    load_run,
    predict_probabilities,
    safe_float,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "dl_evidence_rank_backoff"
    / "coi_mlp_seed1206_pdistance"
)


def split_from_pipeline_manifest(path: Path) -> str:
    manifest_path = path / "pipeline_manifest.json"
    if not manifest_path.exists():
        return path.name
    manifest = json.loads(manifest_path.read_text())
    input_dir = clean(manifest.get("input_dir"))
    return Path(input_dir).name if input_dir else path.name


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_thresholds(model_dir: Path, target_precision: float) -> dict[str, float]:
    path = model_dir / "coi_dl_evidence_thresholds.csv"
    frame = pd.read_csv(path)
    target = pd.to_numeric(frame["target_precision"], errors="coerce")
    sub = frame[np.isclose(target, float(target_precision))].copy()
    thresholds: dict[str, float] = {}
    for _, row in sub.iterrows():
        rank = clean(row.get("rank"))
        threshold = safe_float(row.get("threshold"))
        if rank in RANKS:
            thresholds[rank] = float(threshold)
    missing = [rank for rank in RANKS if rank not in thresholds]
    if missing:
        raise RuntimeError(f"Missing DL thresholds for target_precision={target_precision}: {missing}")
    return thresholds


def standardize_with_saved_stats(frame: pd.DataFrame, stats_path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    payload = json.loads(stats_path.read_text())
    features = [str(feature) for feature in payload["features"]]
    stats = payload["stats"]
    values = frame.reindex(columns=features).to_numpy(dtype=np.float32).copy()
    for idx, feature in enumerate(features):
        item = stats[feature]
        median = float(item.get("median", 0.0))
        mean = float(item.get("mean", 0.0))
        std = float(item.get("std", 1.0)) or 1.0
        column = values[:, idx]
        column[~np.isfinite(column)] = median
        values[:, idx] = (column - mean) / std
    return values.astype(np.float32), features, payload


def load_saved_model(model_dir: Path, n_features: int, cpu: bool) -> EvidenceMLP:
    model_path = model_dir / "coi_dl_evidence_model.pt"
    map_location = "cpu" if cpu or not torch.cuda.is_available() else None
    state = torch.load(model_path, map_location=map_location)
    first_weight = state.get("net.0.weight")
    if first_weight is None:
        raise RuntimeError(f"Cannot infer hidden_dim from {model_path}")
    hidden_dim = int(first_weight.shape[0])
    model = EvidenceMLP(n_features=n_features, hidden_dim=hidden_dim, dropout=0.0)
    model.load_state_dict(state)
    return model


def add_unknown_truth_masks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    truth_columns = {
        "species": "true_tree_label",
        "genus": "true_genus",
        "family": "true_family",
        "order": "true_order",
    }
    for rank, column in truth_columns.items():
        known = out[column].map(clean) != ""
        out[f"{rank}_truth_known"] = known
        out.loc[~known, f"{rank}_correct"] = np.nan
    out["any_truth_known"] = out[[f"{rank}_truth_known" for rank in RANKS]].any(axis=1)
    return out


def apply_inference_policy(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    enabled_ranks: tuple[str, ...],
) -> pd.DataFrame:
    out = frame.copy()
    assigned_ranks: list[str] = []
    assigned_labels: list[str] = []
    reasons: list[str] = []
    correct: list[bool | float] = []
    for _, row in out.iterrows():
        chosen_rank = "no_call"
        chosen_label = ""
        reason = "no_dl_probability_threshold_met"
        is_correct: bool | float = np.nan
        for rank in enabled_ranks:
            prob = safe_float(row.get(f"dl_p_{rank}"), default=-np.inf)
            threshold = thresholds[rank]
            if math.isfinite(threshold) and prob >= threshold:
                chosen_rank = rank
                chosen_label = clean(row.get(f"pred_{rank}"))
                reason = f"dl_p_{rank}>={threshold:.6g}"
                raw_correct = row.get(f"{rank}_correct", np.nan)
                is_correct = np.nan if pd.isna(raw_correct) else bool(raw_correct)
                break
        assigned_ranks.append(chosen_rank)
        assigned_labels.append(chosen_label)
        reasons.append(reason)
        correct.append(is_correct)
    out["dl_assigned_rank"] = assigned_ranks
    out["dl_assigned_label"] = assigned_labels
    out["dl_assignment_reason"] = reasons
    out["dl_assigned_correct"] = correct
    return out


def summarize_inference(frame: pd.DataFrame, target_precision: float, policy: str) -> dict[str, Any]:
    assigned = frame[frame["dl_assigned_rank"] != "no_call"].copy()
    assigned_known = assigned[assigned["dl_assigned_correct"].notna()].copy()
    species_known = frame[
        (frame["dl_assigned_rank"] == "species") & frame["dl_assigned_correct"].notna()
    ].copy()
    false_species = (
        int((~species_known["dl_assigned_correct"].astype(bool)).sum())
        if len(species_known)
        else 0
    )
    summary: dict[str, Any] = {
        "target_precision": float(target_precision),
        "policy": policy,
        "split": clean(frame["split"].iloc[0]) if len(frame) else "",
        "n_queries": int(len(frame)),
        "n_assigned": int(len(assigned)),
        "coverage": float(len(assigned) / len(frame)) if len(frame) else np.nan,
        "known_truth_queries": int(frame["any_truth_known"].sum()) if len(frame) else 0,
        "known_truth_assigned_count": int(len(assigned_known)),
        "assigned_precision_if_known": (
            float(assigned_known["dl_assigned_correct"].astype(bool).mean())
            if len(assigned_known)
            else np.nan
        ),
        "assigned_correct_if_known": (
            int(assigned_known["dl_assigned_correct"].astype(bool).sum())
            if len(assigned_known)
            else np.nan
        ),
        "false_species_call_rate_all_queries_if_known": (
            float(false_species / len(frame)) if len(frame) and len(species_known) else np.nan
        ),
    }
    for rank in list(RANKS) + ["no_call"]:
        summary[f"assigned_{rank}_count"] = int((frame["dl_assigned_rank"] == rank).sum())
    return summary


def process_run(args: argparse.Namespace, logger: ProgressLogger) -> dict[str, Any]:
    run_dir = args.pipeline_run_dir
    run_files = RunFiles(
        split=split_from_pipeline_manifest(run_dir),
        run_dir=run_dir,
        assignments=run_dir / "pipeline_rank_assignments.csv",
        predictions=run_dir / "pipeline_candidate_predictions.csv",
        manifest=run_dir / "pipeline_manifest.json",
    )
    for path in [run_files.assignments, run_files.predictions, run_files.manifest]:
        if not path.exists():
            raise RuntimeError(f"Missing required pipeline artifact: {path}")

    logger.log(f"Loading pipeline evidence from {run_dir}")
    frame = add_unknown_truth_masks(load_run(run_files))
    x, features, stats_payload = standardize_with_saved_stats(
        frame,
        args.model_dir / "coi_dl_evidence_feature_stats.json",
    )
    logger.log(f"Loaded {len(frame)} rows and {len(features)} DL evidence features")
    model = load_saved_model(args.model_dir, n_features=len(features), cpu=args.cpu)
    probs = predict_probabilities(model, x, batch_size=args.batch_size, cpu=args.cpu)
    scored = frame.copy()
    for idx, rank in enumerate(RANKS):
        scored[f"dl_p_{rank}"] = probs[:, idx]

    thresholds = load_thresholds(args.model_dir, args.target_precision)
    if args.policy == "species_enabled":
        enabled_ranks = RANKS
    elif args.policy == "species_disabled":
        enabled_ranks = ("genus", "family", "order")
    else:
        raise RuntimeError(f"Unsupported policy: {args.policy}")
    assigned = apply_inference_policy(scored, thresholds, enabled_ranks=enabled_ranks)
    summary = summarize_inference(assigned, args.target_precision, args.policy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "coi_dl_evidence_applied_predictions.csv"
    summary_path = args.output_dir / "coi_dl_evidence_applied_summary.csv"
    manifest_path = args.output_dir / "coi_dl_evidence_applied_manifest.json"
    logger.log(f"Writing predictions to {predictions_path}")
    assigned.to_csv(predictions_path, index=False)
    write_csv(summary_path, [summary], list(summary.keys()))
    manifest = {
        "generated_by": "scripts/edna/apply_paper1_coi_evidence_model.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_dir": str(run_dir),
        "model_dir": str(args.model_dir),
        "target_precision": float(args.target_precision),
        "policy": args.policy,
        "enabled_ranks": list(enabled_ranks),
        "thresholds": thresholds,
        "features": features,
        "feature_stats_source": str(args.model_dir / "coi_dl_evidence_feature_stats.json"),
        "feature_stats_hash_note": "Feature order and standardization are loaded from the saved training artifact.",
        "stats_feature_count": len(stats_payload.get("features", [])),
        "outputs": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
        },
        "claim_boundary": (
            "Inference-time DL evidence model over existing COI vector and "
            "p-distance evidence traces. It is a decision layer, not a new "
            "candidate retriever. Correctness/precision are unavailable for "
            "unlabeled input rows."
        ),
    }
    logger.log(f"Writing manifest to {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return {"summary": summary, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--policy", choices=["species_enabled", "species_disabled"], default="species_disabled")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    result = process_run(args, logger)
    logger.done(Path(__file__).name)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
