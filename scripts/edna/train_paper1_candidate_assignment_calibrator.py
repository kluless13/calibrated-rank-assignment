#!/usr/bin/env python3
"""Train an independent assignment calibrator over selected candidate reranks.

The candidate reranker improves ordering, but its raw thresholds do not
transfer reliably across missing-reference splits. This script attacks that
specific problem: given the selected candidate for a rank, predict whether that
rank claim is supported.

It uses only inference-safe evidence columns from
candidate_reranker_selected_predictions.csv and fits thresholds on a held-out
seen-test query-group calibration split.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402
from train_paper1_candidate_reranker import choose_threshold, standardize  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_INPUT = (
    PAPER1
    / "candidate_reranker"
    / "coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10"
    / "candidate_reranker_selected_predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    PAPER1
    / "candidate_assignment_calibrator"
    / "coi_cnn_retrieval_hybrid_seed1301_tree10_selected_mlp"
)
RANKS = ("species", "genus", "family", "order")


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


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


def metric_or_nan(fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(fn(y_true, y_score))
    except ValueError:
        return np.nan


def split_seen_test_groups(frame: pd.DataFrame, calib_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seen = frame[frame["split"].eq("seen_test")][["processid"]].astype(str).drop_duplicates()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(seen))
    calib_n = max(1, int(round(len(seen) * calib_fraction)))
    calib_ids = set(seen.iloc[order[:calib_n]]["processid"].tolist())
    seen_mask = frame["split"].eq("seen_test")
    processids = frame["processid"].astype(str)
    calib_idx = frame.index[seen_mask & processids.isin(calib_ids)].to_numpy(dtype=int)
    train_idx = frame.index[seen_mask & ~processids.isin(calib_ids)].to_numpy(dtype=int)
    return train_idx, calib_idx


def load_examples(path: Path, logger: ProgressLogger) -> pd.DataFrame:
    logger.log(f"Loading selected reranker predictions from {path}")
    frame = pd.read_csv(path)
    required = {"split", "processid", "selection_rank", "selection_score", "selection_correct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")
    frame = frame.copy()
    for rank in RANKS:
        frame[f"rank_is_{rank}"] = frame["selection_rank"].eq(rank).astype(float)
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_exact = {
        "processid",
        "split",
        "query_index",
        "candidate_index",
        "candidate_tree_label",
        "true_species",
        "candidate_species",
        "true_genus",
        "candidate_genus",
        "true_family",
        "candidate_family",
        "true_order",
        "candidate_order",
        "species_correct",
        "genus_correct",
        "family_correct",
        "order_correct",
        "selection_correct",
        "selection_rank",
    }
    # These flags identify synthetic split membership rather than deployment evidence.
    excluded_exact.update(
        {
            "candidate_has_seen_test_query",
            "candidate_has_eval_c_query",
            "candidate_has_unseen_genera_query",
        }
    )
    out: list[str] = []
    for col in frame.columns:
        if col in excluded_exact:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            out.append(col)
    return out


class CalibrationMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class TrainResult:
    model: CalibrationMLP
    history: list[dict[str, Any]]


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    args: argparse.Namespace,
    logger: ProgressLogger,
) -> TrainResult:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.log(f"Training candidate assignment calibrator on device={device}")
    model = CalibrationMLP(x.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(
        torch.tensor(x[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    calib_x = torch.tensor(x[calib_idx], dtype=torch.float32, device=device)
    calib_y = torch.tensor(y[calib_idx], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = args.patience
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
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
    return TrainResult(model=model, history=history)


def predict(model: CalibrationMLP, x: np.ndarray, batch_size: int, cpu: bool) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model = model.to(device)
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    return np.concatenate(out)


def summarize(
    frame: pd.DataFrame,
    calib_idx: np.ndarray,
    target_precision: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calib = frame.iloc[calib_idx]
    thresholds: dict[str, dict[str, Any]] = {}
    threshold_rows: list[dict[str, Any]] = []
    for rank in RANKS:
        rank_frame = calib[calib["selection_rank"].eq(rank)]
        item = choose_threshold(
            rank_frame["p_calibrated"].to_numpy(dtype=float),
            rank_frame["selection_correct"].to_numpy(dtype=int),
            target_precision,
        )
        thresholds[rank] = item
        threshold_rows.append({"rank": rank, "target_precision": target_precision, **item})

    summary_rows: list[dict[str, Any]] = []
    for split in sorted(frame["split"].dropna().unique().tolist()):
        split_frame = frame[frame["split"].eq(split)]
        for rank in RANKS:
            rank_frame = split_frame[split_frame["selection_rank"].eq(rank)]
            if rank_frame.empty:
                continue
            threshold = float(thresholds[rank]["threshold"])
            selected = rank_frame[rank_frame["p_calibrated"] >= threshold]
            correct = selected["selection_correct"].astype(int).to_numpy()
            y_true = rank_frame["selection_correct"].astype(int).to_numpy()
            y_score = rank_frame["p_calibrated"].astype(float).to_numpy()
            assigned = int(len(selected))
            summary_rows.append(
                {
                    "split": split,
                    "rank": rank,
                    "target_precision": target_precision,
                    "threshold": threshold,
                    "top1_accuracy": float(y_true.mean()) if len(y_true) else np.nan,
                    "assigned_coverage": assigned / len(rank_frame) if len(rank_frame) else np.nan,
                    "assigned_precision": float(correct.mean()) if assigned else np.nan,
                    "assigned_count": assigned,
                    "correct_assigned_count": int(correct.sum()) if assigned else 0,
                    "n": int(len(rank_frame)),
                    "roc_auc": metric_or_nan(roc_auc_score, y_true, y_score),
                    "average_precision": metric_or_nan(average_precision_score, y_true, y_score),
                }
            )
    return summary_rows, threshold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-predictions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    frame = load_examples(args.selected_predictions, logger)
    features = feature_columns(frame)
    train_idx, calib_idx = split_seen_test_groups(frame, args.calib_fraction, args.seed)
    logger.log(
        f"rows={len(frame)} train_rows={len(train_idx)} calib_rows={len(calib_idx)} "
        f"features={len(features)} input={args.selected_predictions}"
    )
    y = frame["selection_correct"].astype(float).to_numpy(dtype=np.float32)
    all_values = frame[features].to_numpy(dtype=np.float32)
    train_values = frame.iloc[train_idx][features].to_numpy(dtype=np.float32)
    x, stats = standardize(train_values, all_values, features)
    trained = train_model(x, y, train_idx, calib_idx, args, logger)
    frame["p_calibrated"] = predict(trained.model, x, args.batch_size, args.cpu)
    summary_rows, threshold_rows = summarize(frame, calib_idx, args.target_precision)

    predictions_path = args.output_dir / "candidate_assignment_calibrator_predictions.csv"
    summary_path = args.output_dir / "candidate_assignment_calibrator_summary.csv"
    thresholds_path = args.output_dir / "candidate_assignment_calibrator_thresholds.csv"
    history_path = args.output_dir / "candidate_assignment_calibrator_train_history.csv"
    model_path = args.output_dir / "candidate_assignment_calibrator_model.pt"
    stats_path = args.output_dir / "candidate_assignment_calibrator_feature_stats.json"
    manifest_path = args.output_dir / "candidate_assignment_calibrator_manifest.json"

    logger.log(f"Writing predictions to {predictions_path}")
    frame.to_csv(predictions_path, index=False)
    write_csv(summary_path, summary_rows)
    write_csv(thresholds_path, threshold_rows)
    write_csv(history_path, trained.history)
    torch.save(trained.model.state_dict(), model_path)
    stats_path.write_text(json.dumps({"features": features, "stats": stats}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "generated_by": "scripts/edna/train_paper1_candidate_assignment_calibrator.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selected_predictions": str(args.selected_predictions),
        "seed": args.seed,
        "target_precision": args.target_precision,
        "calib_fraction": args.calib_fraction,
        "feature_count": len(features),
        "features": features,
        "outputs": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "thresholds": str(thresholds_path),
            "history": str(history_path),
            "model": str(model_path),
            "feature_stats": str(stats_path),
        },
        "claim_boundary": (
            "Independent assignment calibrator over selected candidate-reranker outputs. "
            "It audits whether a second DL layer can stabilize rank/no-call thresholds; "
            "it is not production unless held-out transfer meets precision targets."
        ),
    }
    logger.log(f"Writing manifest to {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
