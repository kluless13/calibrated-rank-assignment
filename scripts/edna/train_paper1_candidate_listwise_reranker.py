#!/usr/bin/env python3
"""Train a query-listwise or pairwise COI candidate reranker for Paper 1.

This is the next step after the pointwise candidate reranker. It keeps the same
candidate features, but trains over complete top-k candidate lists per query so
the loss directly rewards placing correct species/genus/family/order candidates
above the other candidates in the same retrieved neighborhood.
"""
from __future__ import annotations

import argparse
import json
import math
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
from train_paper1_candidate_reranker import (  # noqa: E402
    DEFAULT_BASELINE_ROOT,
    DEFAULT_OUTPUT_DIR,
    PAPER1,
    RANKS,
    CandidateMLP,
    TreeDistanceCache,
    build_candidate_rows,
    choose_threshold,
    configure_baseline_evidence,
    feature_columns,
    load_tree_distances,
    parse_run_spec,
    predict,
    resolve_runs,
    selected_candidates,
    set_seed,
    standardize,
    summarize,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GroupSplit:
    train_row_idx: np.ndarray
    calib_row_idx: np.ndarray
    train_keys: list[tuple[str, str]]
    calib_keys: list[tuple[str, str]]


def query_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["split", "processid"]].astype(str).drop_duplicates().reset_index(drop=True)


def split_seen_test_groups(frame: pd.DataFrame, calib_fraction: float, seed: int) -> GroupSplit:
    seen = frame[frame["split"].eq("seen_test")].copy()
    groups = query_key_frame(seen)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    calib_n = max(1, int(round(len(groups) * calib_fraction)))
    calib_group_idx = set(order[:calib_n].tolist())
    train_keys: list[tuple[str, str]] = []
    calib_keys: list[tuple[str, str]] = []
    for idx, row in groups.iterrows():
        key = (str(row["split"]), str(row["processid"]))
        if idx in calib_group_idx:
            calib_keys.append(key)
        else:
            train_keys.append(key)
    train_key_set = set(train_keys)
    calib_key_set = set(calib_keys)
    row_keys = list(zip(frame["split"].astype(str), frame["processid"].astype(str), strict=False))
    train_row_idx = np.array([idx for idx, key in enumerate(row_keys) if key in train_key_set], dtype=int)
    calib_row_idx = np.array([idx for idx, key in enumerate(row_keys) if key in calib_key_set], dtype=int)
    return GroupSplit(train_row_idx, calib_row_idx, train_keys, calib_keys)


def build_group_tensors(
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    keys: list[tuple[str, str]],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for key, group in frame.groupby(["split", "processid"], sort=False, dropna=False):
        split, processid = key
        group = group.sort_values("candidate_rank_1based")
        lookup[(str(split), str(processid))] = group.index.to_numpy(dtype=int)
    n_features = x.shape[1]
    out_x = np.zeros((len(keys), top_k, n_features), dtype=np.float32)
    out_y = np.zeros((len(keys), top_k, len(RANKS)), dtype=np.float32)
    out_mask = np.zeros((len(keys), top_k), dtype=bool)
    for group_idx, key in enumerate(keys):
        row_idx = lookup[key][:top_k]
        n = len(row_idx)
        if n == 0:
            continue
        out_x[group_idx, :n, :] = x[row_idx]
        out_y[group_idx, :n, :] = y[row_idx]
        out_mask[group_idx, :n] = True
    return out_x, out_y, out_mask


def listwise_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
    losses: list[torch.Tensor] = []
    for rank_idx in range(len(RANKS)):
        positives = (labels[:, :, rank_idx] > 0.5) & mask
        valid = positives.any(dim=1)
        if not bool(valid.any()):
            continue
        rank_logits = masked_logits[valid, :, rank_idx]
        log_probs = torch.log_softmax(rank_logits, dim=1)
        target = positives[valid].float()
        target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=1.0)
        losses.append(-(target * log_probs).sum(dim=1).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def pairwise_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
    losses: list[torch.Tensor] = []
    for rank_idx in range(len(RANKS)):
        rank_labels = labels[:, :, rank_idx]
        positives = (rank_labels > 0.5) & mask
        negatives = (rank_labels < 0.5) & mask
        valid = positives.any(dim=1) & negatives.any(dim=1)
        if not bool(valid.any()):
            continue
        rank_logits = masked_logits[valid, :, rank_idx]
        pos_mask = positives[valid]
        neg_mask = negatives[valid]
        diffs = rank_logits.unsqueeze(2) - rank_logits.unsqueeze(1)
        pair_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)
        if not bool(pair_mask.any()):
            continue
        losses.append(torch.nn.functional.softplus(-diffs[pair_mask]).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def ranking_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, loss_mode: str) -> torch.Tensor:
    if loss_mode == "pairwise":
        return pairwise_loss(logits, labels, mask)
    return listwise_loss(logits, labels, mask)


def train_listwise_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    calib_x: np.ndarray,
    calib_y: np.ndarray,
    calib_mask: np.ndarray,
    args: argparse.Namespace,
    logger: ProgressLogger,
) -> tuple[CandidateMLP, list[dict[str, Any]]]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.log(f"Training {args.loss_mode} candidate reranker on device={device}")
    model = CandidateMLP(train_x.shape[2], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
        torch.tensor(train_mask, dtype=torch.bool),
    )
    train_loader = DataLoader(train_ds, batch_size=args.group_batch_size, shuffle=True)
    calib_x_t = torch.tensor(calib_x, dtype=torch.float32, device=device)
    calib_y_t = torch.tensor(calib_y, dtype=torch.float32, device=device)
    calib_mask_t = torch.tensor(calib_mask, dtype=torch.bool, device=device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = args.patience
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = ranking_loss(logits, batch_y, batch_mask, args.loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.shape[0]
            seen += int(batch_x.shape[0])
        model.eval()
        with torch.no_grad():
            calib_loss = float(ranking_loss(model(calib_x_t), calib_y_t, calib_mask_t, args.loss_mode).item())
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        help="Pipeline run to use, formatted as split=path. Repeat for seen_test/eval_c/unseen_genera.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Root containing baselines_{split}/{method}/zero_shot_candidate_predictions.csv.",
    )
    parser.add_argument(
        "--baseline-methods",
        nargs="*",
        default=[],
        choices=["blast", "vsearch", "kmer"],
        help="Classical baseline prediction files to join as candidate evidence.",
    )
    parser.add_argument("--tree-neighborhood-features", action="store_true")
    parser.add_argument("--tree-file", type=Path, default=ROOT / "data" / "phylo" / "actinopt_12k_treePL.tre")
    parser.add_argument("--tree-neighborhood-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--loss-mode", choices=["listwise", "pairwise"], default="listwise")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--group-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--write-full-candidates", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    configure_baseline_evidence(args.baseline_root, args.baseline_methods)
    runs = resolve_runs(args)
    logger.log("Using reranker runs: " + ", ".join(f"{run.split}={run.run_dir}" for run in runs))
    if args.baseline_methods:
        logger.log(f"Using baseline evidence methods={args.baseline_methods} root={args.baseline_root}")
    tree_distances = None
    if args.tree_neighborhood_features:
        if not args.tree_file.exists():
            raise RuntimeError(f"Missing tree file for tree-neighborhood features: {args.tree_file}")
        logger.log(f"Loading tree-neighborhood distance index from {args.tree_file}")
        tree_distances = TreeDistanceCache(*load_tree_distances(args.tree_file))
        logger.log(f"Loaded tree labels for tree-neighborhood features: {len(tree_distances.taxon_nodes):,}")

    frames = [
        build_candidate_rows(
            run,
            args.top_k,
            logger,
            tree_distances,
            args.tree_neighborhood_size,
        )
        for run in runs
    ]
    candidates = pd.concat(frames, ignore_index=True, sort=False)
    candidates = candidates.sort_values(["split", "processid", "candidate_rank_1based"]).reset_index(drop=True)
    feature_names = feature_columns(candidates)
    y = candidates[[f"{rank}_correct" for rank in RANKS]].to_numpy(dtype=np.float32)
    group_split = split_seen_test_groups(candidates, args.calib_fraction, args.seed)
    all_values = candidates[feature_names].to_numpy(dtype=np.float32)
    train_values = candidates.iloc[group_split.train_row_idx][feature_names].to_numpy(dtype=np.float32)
    x, stats = standardize(train_values, all_values, feature_names)
    train_x, train_y, train_mask = build_group_tensors(candidates, x, y, group_split.train_keys, args.top_k)
    calib_x, calib_y, calib_mask = build_group_tensors(candidates, x, y, group_split.calib_keys, args.top_k)
    logger.log(
        f"candidate_rows={len(candidates)} train_groups={len(group_split.train_keys)} "
        f"calib_groups={len(group_split.calib_keys)} features={len(feature_names)}"
    )
    model, history = train_listwise_model(train_x, train_y, train_mask, calib_x, calib_y, calib_mask, args, logger)
    probs = predict(model, x, args.batch_size, args.cpu)
    scored = candidates.copy()
    for idx, rank in enumerate(RANKS):
        scored[f"p_{rank}"] = probs[:, idx]
    selected = selected_candidates(scored)
    thresholds: dict[str, float] = {}
    threshold_rows: list[dict[str, Any]] = []
    calib_keys = set(group_split.calib_keys)
    calib_selected = selected[
        selected.apply(lambda row: (str(row["split"]), str(row["processid"])) in calib_keys, axis=1)
    ]
    for rank in RANKS:
        rank_frame = calib_selected[calib_selected["selection_rank"].eq(rank)]
        item = choose_threshold(
            rank_frame["selection_score"].to_numpy(dtype=float),
            rank_frame["selection_correct"].to_numpy(dtype=int),
            args.target_precision,
        )
        thresholds[rank] = float(item["threshold"])
        threshold_rows.append({"rank": rank, "target_precision": float(args.target_precision), **item})
    summary_rows = summarize(scored, selected, thresholds, args.target_precision)

    selected_path = args.output_dir / "candidate_reranker_selected_predictions.csv"
    summary_path = args.output_dir / "candidate_reranker_summary.csv"
    thresholds_path = args.output_dir / "candidate_reranker_thresholds.csv"
    history_path = args.output_dir / "candidate_reranker_train_history.csv"
    sample_path = args.output_dir / "candidate_reranker_candidate_sample.csv"
    model_path = args.output_dir / "candidate_reranker_model.pt"
    stats_path = args.output_dir / "candidate_reranker_feature_stats.json"
    manifest_path = args.output_dir / "candidate_reranker_manifest.json"

    logger.log(f"Writing selected predictions to {selected_path}")
    selected.to_csv(selected_path, index=False)
    write_csv(summary_path, summary_rows)
    write_csv(thresholds_path, threshold_rows)
    write_csv(history_path, history)
    scored.head(min(len(scored), 20000)).to_csv(sample_path, index=False)
    if args.write_full_candidates:
        full_path = args.output_dir / "candidate_reranker_candidates.csv.gz"
        logger.log(f"Writing full scored candidates to {full_path}")
        scored.to_csv(full_path, index=False)
    torch.save(model.state_dict(), model_path)
    stats_path.write_text(json.dumps({"features": feature_names, "stats": stats}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "generated_by": "scripts/edna/train_paper1_candidate_listwise_reranker.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "top_k": args.top_k,
        "target_precision": float(args.target_precision),
        "loss_mode": (
            "query_pairwise_positive_vs_negative"
            if args.loss_mode == "pairwise"
            else "query_listwise_multi_positive_softmax"
        ),
        "runs": [{"split": run.split, "run_dir": str(run.run_dir)} for run in runs],
        "baseline_root": str(args.baseline_root) if args.baseline_methods else None,
        "baseline_methods": args.baseline_methods,
        "tree_neighborhood_features": bool(args.tree_neighborhood_features),
        "tree_file": str(args.tree_file) if args.tree_neighborhood_features else None,
        "tree_neighborhood_size": int(args.tree_neighborhood_size),
        "feature_count": len(feature_names),
        "features": feature_names,
        "outputs": {
            "selected_predictions": str(selected_path),
            "summary": str(summary_path),
            "thresholds": str(thresholds_path),
            "history": str(history_path),
            "candidate_sample": str(sample_path),
            "model": str(model_path),
            "feature_stats": str(stats_path),
        },
        "claim_boundary": (
            f"Query-{args.loss_mode} candidate reranker trained on seen-test query groups. "
            "It optimizes candidate ordering within each retrieved top-k list and "
            "uses inference-safe candidate evidence only. It is model development, "
            "not a final production rank/no-call policy."
        ),
    }
    logger.log(f"Writing manifest to {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
