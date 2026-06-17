#!/usr/bin/env python3
"""Train a small DL evidence model for COI rank/no-call assignment.

This is the first trainable layer above the current production-v1 pipeline.
It does not replace vector retrieval or p-distance reranking. Instead, it learns
from their evidence traces:

- vector similarity and score margins;
- top-k train-reference p-distance traces;
- top-10 taxonomic consensus features.

The model predicts whether the current candidate evidence supports each rank
species/genus/family/order. Thresholds are calibrated on held-out seen-reference
rows, then evaluated on held-out fish and unseen-genera rows.
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
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_RUN_ROOT = PAPER1 / "pipeline_runs"
DEFAULT_OUTPUT_DIR = PAPER1 / "dl_evidence_rank_backoff" / "coi_mlp_seed1206_pdistance"
RANKS = ("species", "genus", "family", "order")
RANK_PRIORITY = {rank: idx for idx, rank in enumerate(RANKS)}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def normalize_label(value: Any) -> str:
    return clean(value).replace(" ", "_")


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def parse_json_list(value: Any) -> list[Any]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def split_from_run_name(name: str) -> str:
    if "_seen_test_" in name:
        return "seen_test"
    if "_unseen_genera_" in name:
        return "unseen_genera"
    if "_eval_c_" in name:
        return "eval_c"
    return "unknown"


@dataclass(frozen=True)
class RunFiles:
    split: str
    run_dir: Path
    assignments: Path
    predictions: Path
    manifest: Path


def discover_runs(run_root: Path, pattern: str) -> list[RunFiles]:
    runs: list[RunFiles] = []
    for run_dir in sorted(run_root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        assignments = run_dir / "pipeline_rank_assignments.csv"
        predictions = run_dir / "pipeline_candidate_predictions.csv"
        manifest = run_dir / "pipeline_manifest.json"
        if assignments.exists() and predictions.exists() and manifest.exists():
            runs.append(
                RunFiles(
                    split=split_from_run_name(run_dir.name),
                    run_dir=run_dir,
                    assignments=assignments,
                    predictions=predictions,
                    manifest=manifest,
                )
            )
    return runs


def top_list_stats(values: list[Any], prefix: str, lower_is_better: bool = False) -> dict[str, float]:
    numeric = np.array([safe_float(value) for value in values], dtype=np.float32)
    numeric = numeric[np.isfinite(numeric)]
    out: dict[str, float] = {
        f"{prefix}_finite_count": float(len(numeric)),
        f"{prefix}_first": float(numeric[0]) if len(numeric) else np.nan,
        f"{prefix}_second": float(numeric[1]) if len(numeric) > 1 else np.nan,
        f"{prefix}_mean": float(numeric.mean()) if len(numeric) else np.nan,
        f"{prefix}_std": float(numeric.std()) if len(numeric) else np.nan,
        f"{prefix}_min": float(numeric.min()) if len(numeric) else np.nan,
        f"{prefix}_max": float(numeric.max()) if len(numeric) else np.nan,
    }
    if len(numeric) > 1:
        out[f"{prefix}_gap_1_2"] = float(numeric[1] - numeric[0]) if lower_is_better else float(numeric[0] - numeric[1])
    else:
        out[f"{prefix}_gap_1_2"] = np.nan
    return out


def load_run(run: RunFiles) -> pd.DataFrame:
    assignments = pd.read_csv(run.assignments)
    predictions = pd.read_csv(run.predictions)
    pred_cols = ["processid", "top_scores", "top_pdistances", "top_tree_labels", "pred_pdistance"]
    merged = assignments.merge(predictions[pred_cols], on="processid", how="left", suffixes=("", "_trace"))
    merged["split"] = run.split
    merged["source_run_dir"] = str(run.run_dir)
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        features: dict[str, Any] = {
            "processid": row["processid"],
            "split": run.split,
            "source_run_dir": str(run.run_dir),
            "true_tree_label": row.get("true_tree_label", ""),
            "true_genus": row.get("true_genus", ""),
            "true_family": row.get("true_family", ""),
            "true_order": row.get("true_order", ""),
        }
        for column in [
            "pred_score",
            "second_score",
            "confidence_margin",
            "confidence_relative_margin",
            "species_top10_consensus",
            "genus_top10_consensus",
            "family_top10_consensus",
            "order_top10_consensus",
            "pred_pdistance",
        ]:
            features[column] = safe_float(row.get(column))
        features.update(top_list_stats(parse_json_list(row.get("top_scores")), "top_score"))
        features.update(top_list_stats(parse_json_list(row.get("top_pdistances")), "top_pdistance", lower_is_better=True))

        true_values = {
            "species": normalize_label(row.get("true_tree_label")),
            "genus": clean(row.get("true_genus")),
            "family": clean(row.get("true_family")),
            "order": clean(row.get("true_order")),
        }
        pred_values = {
            "species": normalize_label(row.get("pred_species")),
            "genus": clean(row.get("pred_genus")),
            "family": clean(row.get("pred_family")),
            "order": clean(row.get("pred_order")),
        }
        for rank in RANKS:
            truth = true_values[rank]
            pred = pred_values[rank]
            features[f"{rank}_correct"] = int(bool(truth and pred and truth == pred))
            features[f"pred_{rank}"] = pred
        rows.append(features)
    return pd.DataFrame(rows)


def build_examples(runs: list[RunFiles]) -> pd.DataFrame:
    frames = [load_run(run) for run in runs]
    if not frames:
        raise RuntimeError("No runs were loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("pred_",)
    excluded = {
        "processid",
        "split",
        "source_run_dir",
        "true_tree_label",
        "true_genus",
        "true_family",
        "true_order",
        *[f"{rank}_correct" for rank in RANKS],
    }
    cols = []
    for col in frame.columns:
        if col in excluded:
            continue
        if any(col.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def train_calib_split(n: int, calib_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_calib = max(1, int(round(n * calib_fraction)))
    n_calib = min(n_calib, n - 1)
    calib = np.sort(indices[:n_calib])
    train = np.sort(indices[n_calib:])
    return train, calib


def standardize(
    train_values: np.ndarray,
    all_values: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, float]] = {}
    cleaned = all_values.copy()
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


class EvidenceMLP(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(RANKS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    args: argparse.Namespace,
    logger: ProgressLogger,
) -> tuple[EvidenceMLP, list[dict[str, Any]]]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.log(f"Training MLP on device={device}")
    model = EvidenceMLP(x.shape[1], args.hidden_dim, args.dropout).to(device)
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
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * batch_x.shape[0]
            seen += int(batch_x.shape[0])
        model.eval()
        with torch.no_grad():
            calib_loss = float(criterion(model(calib_x), calib_y).item())
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(seen, 1),
            "calib_loss": calib_loss,
        }
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


def predict_probabilities(model: EvidenceMLP, x: np.ndarray, batch_size: int, cpu: bool) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model = model.to(device)
    model.eval()
    out: list[np.ndarray] = []
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


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


def apply_rank_policy(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    enabled_ranks: tuple[str, ...] = RANKS,
) -> pd.DataFrame:
    out = frame.copy()
    ranks: list[str] = []
    labels: list[str] = []
    reasons: list[str] = []
    correct: list[bool] = []
    for _, row in out.iterrows():
        assigned_rank = "no_call"
        assigned_label = ""
        reason = "no_dl_probability_threshold_met"
        is_correct = False
        for rank in enabled_ranks:
            prob = safe_float(row.get(f"dl_p_{rank}"), default=-np.inf)
            threshold = thresholds[rank]
            if math.isfinite(threshold) and prob >= threshold:
                assigned_rank = rank
                assigned_label = clean(row.get(f"pred_{rank}"))
                reason = f"dl_p_{rank}>={threshold:.6g}"
                is_correct = bool(row.get(f"{rank}_correct"))
                break
        ranks.append(assigned_rank)
        labels.append(assigned_label)
        reasons.append(reason)
        correct.append(is_correct)
    out["dl_assigned_rank"] = ranks
    out["dl_assigned_label"] = labels
    out["dl_assignment_reason"] = reasons
    out["dl_assigned_correct"] = correct
    return out


def summarize_policy(frame: pd.DataFrame, target_precision: float, policy: str) -> dict[str, Any]:
    assigned = frame[frame["dl_assigned_rank"] != "no_call"].copy()
    species = frame[frame["dl_assigned_rank"] == "species"].copy()
    false_species = int((~species["dl_assigned_correct"].astype(bool)).sum()) if len(species) else 0
    row: dict[str, Any] = {
        "target_precision": float(target_precision),
        "policy": policy,
        "split": clean(frame["split"].iloc[0]) if len(frame) else "",
        "n_queries": int(len(frame)),
        "n_assigned": int(len(assigned)),
        "coverage": float(len(assigned) / len(frame)) if len(frame) else np.nan,
        "assigned_precision": float(assigned["dl_assigned_correct"].astype(bool).mean()) if len(assigned) else np.nan,
        "assigned_correct": int(assigned["dl_assigned_correct"].astype(bool).sum()) if len(assigned) else 0,
        "false_species_call_rate_all_queries": float(false_species / len(frame)) if len(frame) else np.nan,
    }
    for rank in list(RANKS) + ["no_call"]:
        row[f"assigned_{rank}_count"] = int((frame["dl_assigned_rank"] == rank).sum())
    return row


def bootstrap_policy(
    frame: pd.DataFrame,
    target_precision: float,
    policy: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n_bootstrap <= 0 or frame.empty:
        return []
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "coverage": [],
        "assigned_precision": [],
        "false_species_call_rate_all_queries": [],
    }
    n = len(frame)
    for _ in range(n_bootstrap):
        sample = frame.iloc[rng.integers(0, n, size=n)].copy()
        summary = summarize_policy(sample, target_precision, policy)
        for metric in values:
            value = safe_float(summary.get(metric))
            if math.isfinite(value):
                values[metric].append(value)
    rows: list[dict[str, Any]] = []
    for metric, metric_values in values.items():
        arr = np.array(metric_values, dtype=float)
        if len(arr) == 0:
            continue
        rows.append(
            {
                "target_precision": float(target_precision),
                "policy": policy,
                "split": clean(frame["split"].iloc[0]),
                "metric": metric,
                "mean": float(arr.mean()),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-pattern", default="coi_cnn_seed1206_*_target099_pdistance_experimental")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--calib-fraction", type=float, default=0.3)
    parser.add_argument("--target-precision", nargs="+", type=float, default=[0.90, 0.95, 0.99])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    runs = discover_runs(args.run_root, args.run_pattern)
    logger.log(f"Discovered {len(runs)} pipeline runs under {args.run_root}")
    if not runs:
        raise SystemExit("No pipeline runs found")
    examples = build_examples(runs)
    examples_path = args.output_dir / "coi_dl_evidence_training_examples.csv"
    logger.log(f"Writing examples to {examples_path}")
    examples.to_csv(examples_path, index=False)

    train_frame = examples[examples["split"] == "seen_test"].reset_index(drop=True)
    if train_frame.empty:
        raise RuntimeError("Training requires a seen_test split")
    feature_names = feature_columns(examples)
    train_idx, calib_idx = train_calib_split(len(train_frame), args.calib_fraction, args.seed)
    all_values = examples[feature_names].to_numpy(dtype=np.float32)
    train_values = train_frame.iloc[train_idx][feature_names].to_numpy(dtype=np.float32)
    standardized, feature_stats = standardize(train_values, all_values, feature_names)
    y_all = examples[[f"{rank}_correct" for rank in RANKS]].to_numpy(dtype=np.float32)
    seen_mask = examples["split"].to_numpy() == "seen_test"
    seen_indices = np.where(seen_mask)[0]
    train_global = seen_indices[train_idx]
    calib_global = seen_indices[calib_idx]
    logger.log(
        f"Training examples: train={len(train_global)} calib={len(calib_global)} "
        f"features={len(feature_names)}"
    )

    model, history = train_model(
        standardized,
        y_all,
        train_idx=train_global,
        calib_idx=calib_global,
        args=args,
        logger=logger,
    )
    probabilities = predict_probabilities(model, standardized, args.batch_size, args.cpu)
    scored = examples.copy()
    for idx, rank in enumerate(RANKS):
        scored[f"dl_p_{rank}"] = probabilities[:, idx]

    thresholds_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    predictions_out = scored.copy()
    policy_rank_sets = {
        "species_enabled": RANKS,
        "species_disabled": ("genus", "family", "order"),
    }
    for target in args.target_precision:
        thresholds: dict[str, float] = {}
        for idx, rank in enumerate(RANKS):
            item = choose_threshold(probabilities[calib_global, idx], y_all[calib_global, idx], target)
            thresholds[rank] = float(item["threshold"])
            thresholds_rows.append(
                {
                    "target_precision": float(target),
                    "rank": rank,
                    **item,
                    "calibration_rows": int(len(calib_global)),
                }
            )
        assigned_by_policy = {
            policy: apply_rank_policy(scored, thresholds, enabled_ranks=ranks)
            for policy, ranks in policy_rank_sets.items()
        }
        for policy, assigned in assigned_by_policy.items():
            for split, split_frame in assigned.groupby("split"):
                summary = summarize_policy(split_frame, target, policy)
                summary["threshold_source"] = "seen_test_calibration_holdout"
                summaries.append(summary)
                bootstrap_rows.extend(
                    bootstrap_policy(
                        split_frame,
                        target_precision=target,
                        policy=policy,
                        n_bootstrap=args.n_bootstrap,
                        seed=args.seed + int(target * 1000) + len(bootstrap_rows),
                    )
                )
        for rank in RANKS:
            predictions_out[f"target_{target:g}_threshold_{rank}"] = thresholds[rank]
        if target == max(args.target_precision):
            for policy, assigned in assigned_by_policy.items():
                for column in ["dl_assigned_rank", "dl_assigned_label", "dl_assignment_reason", "dl_assigned_correct"]:
                    predictions_out[f"target_{target:g}_{policy}_{column}"] = assigned[column]

    predictions_path = args.output_dir / "coi_dl_evidence_predictions.csv"
    threshold_path = args.output_dir / "coi_dl_evidence_thresholds.csv"
    summary_path = args.output_dir / "coi_dl_evidence_rank_backoff_summary.csv"
    bootstrap_path = args.output_dir / "coi_dl_evidence_rank_backoff_bootstrap.csv"
    history_path = args.output_dir / "coi_dl_evidence_train_history.csv"
    model_path = args.output_dir / "coi_dl_evidence_model.pt"
    stats_path = args.output_dir / "coi_dl_evidence_feature_stats.json"
    manifest_path = args.output_dir / "coi_dl_evidence_model_manifest.json"
    logger.log(f"Writing predictions to {predictions_path}")
    predictions_out.to_csv(predictions_path, index=False)
    write_csv(threshold_path, thresholds_rows)
    write_csv(summary_path, summaries)
    write_csv(bootstrap_path, bootstrap_rows)
    write_csv(history_path, history)
    torch.save(model.state_dict(), model_path)
    stats_path.write_text(json.dumps({"features": feature_names, "stats": feature_stats}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "generated_by": "scripts/edna/train_paper1_coi_evidence_model.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(args.run_root),
        "run_pattern": args.run_pattern,
        "runs": [str(run.run_dir) for run in runs],
        "seed": args.seed,
        "calib_fraction": args.calib_fraction,
        "target_precision": args.target_precision,
        "feature_count": len(feature_names),
        "features": feature_names,
        "outputs": {
            "examples": str(examples_path),
            "predictions": str(predictions_path),
            "thresholds": str(threshold_path),
            "summary": str(summary_path),
            "bootstrap": str(bootstrap_path),
            "history": str(history_path),
            "model": str(model_path),
            "feature_stats": str(stats_path),
        },
        "claim_boundary": (
            "First DL evidence-fusion/rank-backoff layer over existing COI "
            "pipeline features. It is trained on seen-test rows and evaluated "
            "on held-out fish and unseen-genera rows. It does not replace "
            "candidate retrieval or p-distance reranking."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
