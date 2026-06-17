#!/usr/bin/env python3
"""Train a missing-reference-aware COI rank/no-call calibrator.

This is the next DL layer after the production-v1 p-distance policy, the
candidate rerankers, and the v2 reference-gap detector. It trains on a mixed
deployment-like pool:

- normal supported seen-test rows;
- strict Eval C hidden species/genus/family rows.

It evaluates on:

- normal held-out fish and unseen-genera rows;
- strict unseen-genera hidden species/genus/family rows.

The model uses pipeline evidence plus v2 reference-gap probabilities as soft
features. It does not use split labels, roles, global candidate-count leakage,
or synthetic strict-pack IDs as model features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
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
    apply_rank_policy,
    clean,
    feature_columns as base_feature_columns,
    load_run,
    predict_probabilities,
    safe_float,
    set_seed,
    standardize,
    summarize_policy,
    train_model,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_OUTPUT_DIR = PAPER1 / "dl_evidence_rank_backoff" / "coi_mlp_seed1401_missing_reference_aware_v2_gap"
DEFAULT_GAP_RUN = PAPER1 / "reference_gap_detector" / "coi_mlp_seed1301_v2_candidate_evidence_target095"


@dataclass(frozen=True)
class SourceRun:
    split: str
    role: str
    run_dir: Path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_source_runs() -> list[SourceRun]:
    normal = PAPER1 / "pipeline_runs"
    strict = PAPER1 / "dl_evidence_rank_backoff" / "strict_pipeline_runs"
    return [
        SourceRun("seen_test", "train_supported", normal / "coi_cnn_seed1206_seen_test_target099_pdistance_experimental"),
        SourceRun("eval_c", "eval_supported", normal / "coi_cnn_seed1206_eval_c_target099_pdistance_experimental"),
        SourceRun("unseen_genera", "eval_supported", normal / "coi_cnn_seed1206_unseen_genera_target099_pdistance_experimental"),
        SourceRun("eval_c_hide_species", "train_gap", strict / "eval_c_hide_species_pdistance"),
        SourceRun("eval_c_hide_genus", "train_gap", strict / "eval_c_hide_genus_pdistance"),
        SourceRun("eval_c_hide_family", "train_gap", strict / "eval_c_hide_family_pdistance"),
        SourceRun("unseen_genera_hide_species", "eval_gap", strict / "unseen_genera_hide_species_pdistance"),
        SourceRun("unseen_genera_hide_genus", "eval_gap", strict / "unseen_genera_hide_genus_pdistance"),
        SourceRun("unseen_genera_hide_family", "eval_gap", strict / "unseen_genera_hide_family_pdistance"),
    ]


def source_to_run_files(source: SourceRun) -> RunFiles:
    return RunFiles(
        split=source.split,
        run_dir=source.run_dir,
        assignments=source.run_dir / "pipeline_rank_assignments.csv",
        predictions=source.run_dir / "pipeline_candidate_predictions.csv",
        manifest=source.run_dir / "pipeline_manifest.json",
    )


def validate_sources(runs: list[SourceRun]) -> None:
    missing: list[str] = []
    for source in runs:
        files = source_to_run_files(source)
        for path in [files.assignments, files.predictions, files.manifest]:
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(missing))


def load_gap_probabilities(path: Path, logger: ProgressLogger) -> pd.DataFrame:
    predictions = path / "reference_gap_predictions.csv"
    if not predictions.exists():
        raise FileNotFoundError(predictions)
    logger.log(f"Loading v2 reference-gap probabilities from {predictions}")
    cols = ["processid", "split", "gap_p_species", "gap_p_genus", "gap_p_family"]
    return pd.read_csv(predictions, usecols=cols)


def load_source(source: SourceRun, gap: pd.DataFrame, logger: ProgressLogger) -> pd.DataFrame:
    run_files = source_to_run_files(source)
    logger.log(f"Loading split={source.split} role={source.role} from {source.run_dir}")
    frame = load_run(run_files)
    frame["split"] = source.split
    frame["role"] = source.role
    frame = frame.merge(gap[gap["split"] == source.split], on=["processid", "split"], how="left")
    for rank in ("species", "genus", "family"):
        frame[f"gap_p_{rank}"] = pd.to_numeric(frame[f"gap_p_{rank}"], errors="coerce").fillna(0.0)
    frame["gap_p_max"] = frame[["gap_p_species", "gap_p_genus", "gap_p_family"]].max(axis=1)
    frame["gap_p_more_specific_than_family"] = frame[["gap_p_species", "gap_p_genus"]].max(axis=1)
    return frame


def build_examples(sources: list[SourceRun], gap: pd.DataFrame, logger: ProgressLogger) -> pd.DataFrame:
    frames = [load_source(source, gap, logger) for source in sources]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    logger.log(f"Built {len(frame)} examples across {frame['split'].nunique()} splits")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols = base_feature_columns(frame)
    blocked = {"candidate_species_count", "candidate_genus_count", "candidate_family_count", "candidate_order_count"}
    return [col for col in cols if col not in blocked]


def train_calib_split(indices: np.ndarray, calib_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_calib = max(1, int(round(len(shuffled) * calib_fraction)))
    n_calib = min(n_calib, len(shuffled) - 1)
    return np.sort(shuffled[n_calib:]), np.sort(shuffled[:n_calib])


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target_precision: float) -> dict[str, Any]:
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order].astype(bool)
    tp = np.cumsum(sorted_labels)
    n = np.arange(1, len(sorted_probs) + 1)
    precision = tp / n
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        return {
            "threshold": math.inf,
            "calibration_precision": np.nan,
            "calibration_coverage": 0.0,
            "calibration_assigned": 0,
        }
    idx = int(valid[-1])
    return {
        "threshold": float(sorted_probs[idx]),
        "calibration_precision": float(precision[idx]),
        "calibration_coverage": float((idx + 1) / len(sorted_probs)),
        "calibration_assigned": int(idx + 1),
    }


def compact_predictions(frame: pd.DataFrame, max_target: float) -> pd.DataFrame:
    keep = [
        "processid",
        "split",
        "role",
        "true_tree_label",
        "true_genus",
        "true_family",
        "true_order",
        "pred_species",
        "pred_genus",
        "pred_family",
        "pred_order",
        "gap_p_species",
        "gap_p_genus",
        "gap_p_family",
        *[f"{rank}_correct" for rank in RANKS],
        *[f"dl_p_{rank}" for rank in RANKS],
    ]
    for policy in ("species_enabled", "species_disabled"):
        keep.extend(
            [
                f"target_{max_target:g}_{policy}_dl_assigned_rank",
                f"target_{max_target:g}_{policy}_dl_assigned_label",
                f"target_{max_target:g}_{policy}_dl_assignment_reason",
                f"target_{max_target:g}_{policy}_dl_assigned_correct",
            ]
        )
    return frame[[col for col in keep if col in frame.columns]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gap-run-dir", type=Path, default=DEFAULT_GAP_RUN)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--target-precision", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    sources = discover_source_runs()
    validate_sources(sources)
    gap = load_gap_probabilities(args.gap_run_dir, logger)
    examples = build_examples(sources, gap, logger)
    features = feature_columns(examples)
    logger.log(f"Using {len(features)} features: {', '.join(features)}")

    eligible = examples["role"].isin(["train_supported", "train_gap"]).to_numpy()
    eligible_idx = np.where(eligible)[0]
    train_idx, calib_idx = train_calib_split(eligible_idx, args.calib_fraction, args.seed)
    train_values = examples.iloc[train_idx][features].to_numpy(dtype=np.float32)
    all_values = examples[features].to_numpy(dtype=np.float32)
    standardized, feature_stats = standardize(train_values, all_values, features)
    y_all = examples[[f"{rank}_correct" for rank in RANKS]].to_numpy(dtype=np.float32)
    logger.log(
        f"Training mixed calibrator: train={len(train_idx)} calib={len(calib_idx)} "
        f"total={len(examples)} features={len(features)}"
    )

    model, history = train_model(standardized, y_all, train_idx, calib_idx, args, logger)
    probabilities = predict_probabilities(model, standardized, args.batch_size, args.cpu)
    scored = examples.copy()
    for idx, rank in enumerate(RANKS):
        scored[f"dl_p_{rank}"] = probabilities[:, idx]

    thresholds_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    max_target = max(args.target_precision)
    policy_rank_sets = {
        "species_enabled": RANKS,
        "species_disabled": ("genus", "family", "order"),
    }
    for target in args.target_precision:
        thresholds: dict[str, float] = {}
        for idx, rank in enumerate(RANKS):
            item = choose_threshold(probabilities[calib_idx, idx], y_all[calib_idx, idx], target)
            thresholds[rank] = float(item["threshold"])
            thresholds_rows.append(
                {
                    "target_precision": float(target),
                    "rank": rank,
                    **item,
                    "calibration_rows": int(len(calib_idx)),
                    "calibration_source": "mixed_supported_seen_test_plus_strict_eval_c",
                }
            )
        for policy, ranks in policy_rank_sets.items():
            assigned = apply_rank_policy(scored, thresholds, enabled_ranks=ranks)
            if target == max_target:
                for column in [
                    "dl_assigned_rank",
                    "dl_assigned_label",
                    "dl_assignment_reason",
                    "dl_assigned_correct",
                ]:
                    scored[f"target_{target:g}_{policy}_{column}"] = assigned[column]
            for split, split_frame in assigned.groupby("split"):
                row = summarize_policy(split_frame, target, policy)
                row["role"] = clean(split_frame["role"].iloc[0]) if len(split_frame) else ""
                row["calibration_source"] = "mixed_supported_seen_test_plus_strict_eval_c"
                row["claim_boundary"] = (
                    "Model-development result. The calibrator trains on normal supported rows "
                    "plus strict Eval C missing-reference packs and evaluates on unseen-genera "
                    "normal/strict packs. Do not promote unless false species calls remain zero "
                    "and unseen-genera family/order precision improves."
                )
                summary_rows.append(row)

    predictions_path = args.output_dir / "missing_ref_aware_predictions.csv"
    thresholds_path = args.output_dir / "missing_ref_aware_thresholds.csv"
    summary_path = args.output_dir / "missing_ref_aware_summary.csv"
    history_path = args.output_dir / "missing_ref_aware_train_history.csv"
    stats_path = args.output_dir / "missing_ref_aware_feature_stats.json"
    model_path = args.output_dir / "missing_ref_aware_model.pt"
    manifest_path = args.output_dir / "missing_ref_aware_manifest.json"

    logger.log(f"Writing compact predictions to {predictions_path}")
    compact_predictions(scored, max_target).to_csv(predictions_path, index=False)
    write_csv(thresholds_path, thresholds_rows)
    write_csv(summary_path, summary_rows)
    write_csv(history_path, history)
    torch.save(model.state_dict(), model_path)
    stats_path.write_text(json.dumps({"features": features, "stats": feature_stats}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "generated_by": "scripts/edna/train_paper1_missing_reference_aware_calibrator.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [{"split": run.split, "role": run.role, "run_dir": str(run.run_dir)} for run in sources],
        "gap_run_dir": str(args.gap_run_dir),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "features": features,
        "n_examples": int(len(examples)),
        "n_train": int(len(train_idx)),
        "n_calib": int(len(calib_idx)),
        "target_precision": [float(x) for x in args.target_precision],
        "claim_boundary": "Model-development missing-reference-aware calibrator; not production default.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
