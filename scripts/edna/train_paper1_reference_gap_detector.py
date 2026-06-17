#!/usr/bin/env python3
"""Train a first COI reference-gap detector for Paper 1.

The model predicts whether the current candidate/reference set does not support
species, genus, or family claims for a query. Targets are derived mechanically:
if the true rank value is absent from `candidate_species.csv`, that rank is a
gap for the current candidate universe.

This first version is a diagnostic model:

- train/calibrate on supported seen-test rows plus Eval C strict hidden packs;
- evaluate on normal held-out/unseen rows and strict unseen-genera packs;
- do not use it as final manuscript calibration until seen-test strict packs or
  another independent training design exists.
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
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402
from train_paper1_coi_evidence_model import (  # noqa: E402
    RunFiles,
    clean,
    feature_columns,
    load_run,
    safe_float,
    train_calib_split,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_OUTPUT_DIR = PAPER1 / "reference_gap_detector" / "coi_mlp_seed1206"
GAP_RANKS = ("species", "genus", "family")


@dataclass(frozen=True)
class GapRun:
    split: str
    role: str
    run_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "pipeline_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing {manifest_path}")
    return json.loads(manifest_path.read_text())


def resolve_input_dir(path: Path) -> Path:
    manifest = load_manifest(path)
    raw = clean(manifest.get("input_dir"))
    if not raw:
        raise RuntimeError(f"pipeline_manifest.json has no input_dir for {path}")
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    candidate = ROOT / raw
    if candidate.exists():
        return candidate
    raise RuntimeError(f"Cannot resolve input_dir={raw} for {path}")


def candidate_support_sets(input_dir: Path) -> dict[str, set[str]]:
    candidates = pd.read_csv(input_dir / "candidate_species.csv")
    return {
        "species": {clean(value).replace(" ", "_") for value in candidates["tree_label"] if clean(value)},
        "genus": {clean(value) for value in candidates["genus_name"] if clean(value)},
        "family": {clean(value) for value in candidates["family_name"] if clean(value)},
        "order": {clean(value) for value in candidates["order_name"] if clean(value)},
    }


def add_gap_targets(frame: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    support = candidate_support_sets(input_dir)
    true_values = {
        "species": out["true_tree_label"].map(lambda value: clean(value).replace(" ", "_")),
        "genus": out["true_genus"].map(clean),
        "family": out["true_family"].map(clean),
    }
    for rank in GAP_RANKS:
        supported = true_values[rank].map(lambda value: bool(value and value in support[rank]))
        out[f"{rank}_supported"] = supported.astype(int)
        out[f"{rank}_gap"] = (~supported).astype(int)
    out["candidate_species_count"] = float(len(support["species"]))
    out["candidate_genus_count"] = float(len(support["genus"]))
    out["candidate_family_count"] = float(len(support["family"]))
    out["candidate_order_count"] = float(len(support["order"]))
    return out


def load_gap_run(run: GapRun) -> pd.DataFrame:
    files = RunFiles(
        split=run.split,
        run_dir=run.run_dir,
        assignments=run.run_dir / "pipeline_rank_assignments.csv",
        predictions=run.run_dir / "pipeline_candidate_predictions.csv",
        manifest=run.run_dir / "pipeline_manifest.json",
    )
    frame = load_run(files)
    frame["role"] = run.role
    frame["gap_run_name"] = run.run_dir.name
    input_dir = resolve_input_dir(run.run_dir)
    frame["candidate_input_dir"] = str(input_dir)
    return add_gap_targets(frame, input_dir)


def default_runs() -> list[GapRun]:
    normal = PAPER1 / "pipeline_runs"
    strict = PAPER1 / "dl_evidence_rank_backoff" / "strict_pipeline_runs"
    return [
        GapRun("seen_test", "train_supported", normal / "coi_cnn_seed1206_seen_test_target099_pdistance_experimental"),
        GapRun("eval_c", "eval_supported", normal / "coi_cnn_seed1206_eval_c_target099_pdistance_experimental"),
        GapRun("unseen_genera", "eval_supported", normal / "coi_cnn_seed1206_unseen_genera_target099_pdistance_experimental"),
        GapRun("eval_c_hide_species", "train_gap", strict / "eval_c_hide_species_pdistance"),
        GapRun("eval_c_hide_genus", "train_gap", strict / "eval_c_hide_genus_pdistance"),
        GapRun("eval_c_hide_family", "train_gap", strict / "eval_c_hide_family_pdistance"),
        GapRun("unseen_genera_hide_species", "eval_gap", strict / "unseen_genera_hide_species_pdistance"),
        GapRun("unseen_genera_hide_genus", "eval_gap", strict / "unseen_genera_hide_genus_pdistance"),
        GapRun("unseen_genera_hide_family", "eval_gap", strict / "unseen_genera_hide_family_pdistance"),
    ]


def build_examples(runs: list[GapRun], logger: ProgressLogger) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        if not (run.run_dir / "pipeline_rank_assignments.csv").exists():
            logger.log(f"Skipping missing run {run.run_dir}")
            continue
        logger.log(f"Loading {run.split} role={run.role} from {run.run_dir}")
        frames.append(load_gap_run(run))
    if not frames:
        raise RuntimeError("No reference-gap runs were loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def standardize(
    train_values: np.ndarray,
    all_values: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    cleaned = all_values.copy()
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(feature_names):
        train_col = train_values[:, idx]
        finite = train_col[np.isfinite(train_col)]
        median = float(np.median(finite)) if len(finite) else 0.0
        mean = float(np.mean(finite)) if len(finite) else 0.0
        std = float(np.std(finite)) if len(finite) and np.std(finite) > 1e-8 else 1.0
        col = cleaned[:, idx]
        col[~np.isfinite(col)] = median
        cleaned[:, idx] = (col - mean) / std
        stats[name] = {"median": median, "mean": mean, "std": std}
    return cleaned.astype(np.float32), stats


class GapMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(GAP_RANKS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    args: argparse.Namespace,
    logger: ProgressLogger,
) -> tuple[GapMLP, list[dict[str, Any]]]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.log(f"Training reference-gap MLP on device={device}")
    model = GapMLP(x.shape[1], args.hidden_dim, args.dropout).to(device)
    train_x = torch.tensor(x[train_idx], dtype=torch.float32)
    train_y = torch.tensor(y[train_idx], dtype=torch.float32)
    calib_x = torch.tensor(x[calib_idx], dtype=torch.float32, device=device)
    calib_y = torch.tensor(y[calib_idx], dtype=torch.float32, device=device)
    positives = train_y.sum(dim=0)
    negatives = train_y.shape[0] - positives
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=args.max_pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience_left = args.patience
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.shape[0]
            seen += int(batch_x.shape[0])
        model.eval()
        with torch.no_grad():
            calib_loss = float(criterion(model(calib_x), calib_y).item())
        row = {"epoch": epoch, "train_loss": total_loss / max(seen, 1), "calib_loss": calib_loss}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            logger.log(f"epoch={epoch} train_loss={row['train_loss']:.5f} calib_loss={calib_loss:.5f}")
        if calib_loss < best_loss - 1e-5:
            best_loss = calib_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            logger.log(f"Early stopping at epoch={epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_probabilities(model: GapMLP, x: np.ndarray, batch_size: int, cpu: bool) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model = model.to(device)
    model.eval()
    out: list[np.ndarray] = []
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            out.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.vstack(out)


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


def metric_or_nan(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return np.nan


def summarize_split(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    target_precision: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in GAP_RANKS:
        y_true = frame[f"{rank}_gap"].to_numpy(dtype=int)
        y_score = frame[f"gap_p_{rank}"].to_numpy(dtype=float)
        threshold = thresholds[rank]
        pred = y_score >= threshold if math.isfinite(threshold) else np.zeros_like(y_true, dtype=bool)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            pred.astype(int),
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "split": clean(frame["split"].iloc[0]) if len(frame) else "",
                "role": clean(frame["role"].iloc[0]) if len(frame) else "",
                "target_precision": float(target_precision),
                "rank": rank,
                "n_rows": int(len(frame)),
                "gap_prevalence": float(y_true.mean()) if len(y_true) else np.nan,
                "threshold": float(threshold),
                "gap_flag_rate": float(pred.mean()) if len(pred) else np.nan,
                "gap_precision": float(precision),
                "gap_recall": float(recall),
                "gap_f1": float(f1),
                "roc_auc": metric_or_nan(roc_auc_score, y_true, y_score),
                "average_precision": metric_or_nan(average_precision_score, y_true, y_score),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--target-gap-precision", type=float, default=0.90)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument(
        "--include-candidate-counts",
        action="store_true",
        help=(
            "Include global candidate-set size features. Off by default because "
            "these can encode the synthetic strict-missing-reference pack rather "
            "than per-query uncertainty."
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    examples = build_examples(default_runs(), logger)
    examples_path = args.output_dir / "reference_gap_examples.csv"
    logger.log(f"Writing examples to {examples_path}")
    examples.to_csv(examples_path, index=False)

    target_columns = {f"{rank}_gap" for rank in GAP_RANKS} | {f"{rank}_supported" for rank in GAP_RANKS}
    candidate_count_columns = {
        "candidate_species_count",
        "candidate_genus_count",
        "candidate_family_count",
        "candidate_order_count",
    }
    feature_names = [
        name
        for name in feature_columns(examples)
        if name not in target_columns and (args.include_candidate_counts or name not in candidate_count_columns)
    ]
    if args.include_candidate_counts:
        for extra in sorted(candidate_count_columns):
            if extra in examples.columns and extra not in feature_names:
                feature_names.append(extra)
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
        f"features={len(feature_names)}"
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
    for (split, role), split_frame in scored.groupby(["split", "role"], sort=True):
        summary_rows.extend(summarize_split(split_frame, thresholds, args.target_gap_precision))

    predictions_path = args.output_dir / "reference_gap_predictions.csv"
    thresholds_path = args.output_dir / "reference_gap_thresholds.csv"
    summary_path = args.output_dir / "reference_gap_summary.csv"
    history_path = args.output_dir / "reference_gap_train_history.csv"
    model_path = args.output_dir / "reference_gap_model.pt"
    stats_path = args.output_dir / "reference_gap_feature_stats.json"
    manifest_path = args.output_dir / "reference_gap_manifest.json"

    logger.log(f"Writing predictions to {predictions_path}")
    scored.to_csv(predictions_path, index=False)
    write_csv(thresholds_path, threshold_rows)
    write_csv(summary_path, summary_rows)
    write_csv(history_path, history)
    torch.save(model.state_dict(), model_path)
    stats_path.write_text(json.dumps({"features": feature_names, "stats": feature_stats}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "generated_by": "scripts/edna/train_paper1_reference_gap_detector.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "target_gap_precision": float(args.target_gap_precision),
        "gap_ranks": list(GAP_RANKS),
        "features": feature_names,
        "outputs": {
            "examples": str(examples_path),
            "predictions": str(predictions_path),
            "thresholds": str(thresholds_path),
            "summary": str(summary_path),
            "history": str(history_path),
            "model": str(model_path),
            "feature_stats": str(stats_path),
        },
        "claim_boundary": (
            "First diagnostic COI reference-gap detector. It trains on supported "
            "seen-test rows plus Eval C strict hidden-reference rows and "
            "evaluates on normal held-out/unseen and strict unseen-genera rows. "
            "It should not be treated as final manuscript calibration until a "
            "fully independent strict training design is added."
        ),
    }
    logger.log(f"Writing manifest to {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
