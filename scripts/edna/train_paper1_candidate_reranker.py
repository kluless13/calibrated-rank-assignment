#!/usr/bin/env python3
"""Train a first candidate-level COI reranker for Paper 1.

This model scores each top-k candidate, instead of only scoring the final
query-level assignment. It is a development layer for the full pipeline:

vector retrieval -> candidate reranking -> rank/no-call -> reason diagnostics.

The first version trains on seen-test top-k candidate rows and evaluates on
held-out fish and unseen-genera. It should not be treated as final until BLAST
/ VSEARCH identity and tree-neighborhood features are added.
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
import dendropy
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402
from train_paper1_coi_evidence_model import (  # noqa: E402
    clean,
    parse_json_list,
    safe_float,
    train_calib_split,
)


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_OUTPUT_DIR = PAPER1 / "candidate_reranker" / "coi_mlp_seed1206_top50"
DEFAULT_BASELINE_ROOT = (
    ROOT
    / "results"
    / "remote_runs"
    / "2026-05-31"
    / "rtx_pro_6000"
    / "paper1_phylo_calibrated_assignment"
)
RANKS = ("species", "genus", "family", "order")
TREE_RANKS = ("genus", "family", "order")


@dataclass(frozen=True)
class RerankRun:
    split: str
    run_dir: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_species(value: Any) -> str:
    return clean(value).replace(" ", "_")


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


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "pipeline_manifest.json"
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    return json.loads(path.read_text())


def parse_run_spec(value: str) -> RerankRun:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run spec must be split=path")
    split, raw_path = value.split("=", 1)
    split = clean(split)
    if not split:
        raise argparse.ArgumentTypeError("run spec split cannot be empty")
    path = Path(raw_path)
    return RerankRun(split, path)


def resolve_input_dir(run_dir: Path) -> Path:
    raw = clean(load_manifest(run_dir).get("input_dir"))
    if not raw:
        raise RuntimeError(f"Missing input_dir in {run_dir / 'pipeline_manifest.json'}")
    path = Path(raw)
    if path.exists():
        return path
    path = ROOT / raw
    if path.exists():
        return path
    raise RuntimeError(f"Cannot resolve input_dir={raw}")


def taxonomy_maps(input_dir: Path) -> dict[str, dict[str, Any]]:
    candidates = pd.read_csv(input_dir / "candidate_species.csv")
    maps: dict[str, dict[str, Any]] = {}
    for _, row in candidates.iterrows():
        label = normalize_species(row.get("tree_label"))
        if not label:
            continue
        maps[label] = {
            "species": label,
            "genus": clean(row.get("genus_name")),
            "family": clean(row.get("family_name")),
            "order": clean(row.get("order_name")),
            "reference_sequence_count": safe_float(row.get("reference_sequence_count")),
            "has_seen_test_query": safe_float(row.get("has_seen_test_query")),
            "has_eval_c_query": safe_float(row.get("has_eval_c_query")),
            "has_unseen_genera_query": safe_float(row.get("has_unseen_genera_query")),
        }
    return maps


def load_tree_distances(tree_file: Path) -> tuple[
    dict[str, object],
    dict[object, float],
    dict[str, list[object]],
    dict[str, set[object]],
]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick")
    taxon_nodes: dict[str, object] = {}
    for node in tree.leaf_node_iter():
        if node.taxon is None or not node.taxon.label:
            continue
        label = normalize_species(node.taxon.label)
        if label:
            taxon_nodes[label] = node
    depths = {tree.seed_node: 0.0}
    stack = [tree.seed_node]
    while stack:
        node = stack.pop()
        for child in node.child_node_iter():
            depths[child] = depths[node] + float(child.edge_length or 0.0)
            stack.append(child)

    def ancestors(node: object) -> list[object]:
        out = []
        current = node
        while current is not None:
            out.append(current)
            current = current.parent_node
        return out

    ancestor_lists = {label: ancestors(node) for label, node in taxon_nodes.items()}
    ancestor_sets = {label: set(nodes) for label, nodes in ancestor_lists.items()}
    return taxon_nodes, depths, ancestor_lists, ancestor_sets


class TreeDistanceCache:
    def __init__(
        self,
        taxon_nodes: dict[str, object],
        depths: dict[object, float],
        ancestor_lists: dict[str, list[object]],
        ancestor_sets: dict[str, set[object]],
    ) -> None:
        self.taxon_nodes = taxon_nodes
        self.depths = depths
        self.ancestor_lists = ancestor_lists
        self.ancestor_sets = ancestor_sets
        self.cache: dict[tuple[str, str], float] = {}

    def distance(self, left: object, right: object) -> float:
        label_left = normalize_species(left)
        label_right = normalize_species(right)
        if (
            not label_left
            or not label_right
            or label_left not in self.taxon_nodes
            or label_right not in self.taxon_nodes
        ):
            return float("nan")
        if label_left == label_right:
            return 0.0
        key = tuple(sorted((label_left, label_right)))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        for node in self.ancestor_lists[label_left]:
            if node in self.ancestor_sets[label_right]:
                distance = float(
                    self.depths[self.taxon_nodes[label_left]]
                    + self.depths[self.taxon_nodes[label_right]]
                    - 2 * self.depths[node]
                )
                self.cache[key] = distance
                return distance
        self.cache[key] = float("nan")
        return float("nan")


def finite_or_nan(value: Any) -> float:
    out = safe_float(value)
    return out if np.isfinite(out) else np.nan


def finite_stats(values: list[float]) -> tuple[float, float, float]:
    finite = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(finite.min()), float(finite.mean()), float(finite.max())


def add_tree_neighborhood_features(
    out: dict[str, Any],
    candidate_label: str,
    labels: list[str],
    taxa: dict[str, dict[str, Any]],
    distances: TreeDistanceCache | None,
    neighborhood_size: int,
) -> None:
    if distances is None:
        return
    neighborhood = [label for label in labels[: max(1, neighborhood_size)] if label]
    top_label = labels[0] if labels else ""
    top_tax = taxa.get(top_label, {})
    candidate_tax = taxa.get(candidate_label, {})
    pair_distances: list[float] = []
    for left_idx, left in enumerate(neighborhood):
        for right in neighborhood[left_idx + 1 :]:
            pair_distances.append(distances.distance(left, right))
    pair_min, pair_mean, pair_max = finite_stats(pair_distances)
    candidate_distances = [
        distances.distance(candidate_label, other)
        for other in neighborhood
        if other and other != candidate_label
    ]
    candidate_min, candidate_mean, candidate_max = finite_stats(candidate_distances)
    top_distance = distances.distance(candidate_label, top_label)
    out.update(
        {
            "tree_evidence_available": int(
                np.isfinite(top_distance) or np.isfinite(candidate_mean) or np.isfinite(pair_mean)
            ),
            "candidate_is_top1": int(bool(candidate_label and candidate_label == top_label)),
            "tree_distance_to_top1_candidate": top_distance,
            "tree_distance_to_topk_min": candidate_min,
            "tree_distance_to_topk_mean": candidate_mean,
            "tree_distance_to_topk_max": candidate_max,
            "topk_pairwise_tree_distance_min": pair_min,
            "topk_pairwise_tree_distance_mean": pair_mean,
            "topk_pairwise_tree_distance_max": pair_max,
        }
    )
    for rank in TREE_RANKS:
        values = [clean(taxa.get(label, {}).get(rank, "")) for label in neighborhood]
        values = [value for value in values if value]
        candidate_value = clean(candidate_tax.get(rank))
        top_value = clean(top_tax.get(rank))
        out[f"topk_unique_{rank}_count"] = int(len(set(values)))
        out[f"candidate_same_{rank}_as_top1"] = int(bool(candidate_value and candidate_value == top_value))


def fraction_same(labels: list[str], taxa: dict[str, dict[str, Any]], candidate_taxon: str, rank: str, limit: int) -> float:
    if not candidate_taxon:
        return 0.0
    values = [taxa.get(normalize_species(label), {}).get(rank, "") for label in labels[:limit]]
    values = [clean(value) for value in values if clean(value)]
    if not values:
        return 0.0
    return float(sum(value == candidate_taxon for value in values) / len(values))


def build_candidate_rows(
    run: RerankRun,
    top_k: int,
    logger: ProgressLogger,
    tree_distances: TreeDistanceCache | None,
    tree_neighborhood_size: int,
) -> pd.DataFrame:
    pred_path = run.run_dir / "pipeline_candidate_predictions.csv"
    if not pred_path.exists():
        raise RuntimeError(f"Missing {pred_path}")
    input_dir = resolve_input_dir(run.run_dir)
    taxa = taxonomy_maps(input_dir)
    logger.log(f"Building candidate rows split={run.split} from {pred_path}")
    predictions = pd.read_csv(pred_path)
    rows: list[dict[str, Any]] = []
    baseline_lookup = load_baseline_lookups(run.split, logger)
    for query_idx, row in predictions.iterrows():
        labels = [normalize_species(value) for value in parse_json_list(row.get("top_tree_labels"))[:top_k]]
        scores = [finite_or_nan(value) for value in parse_json_list(row.get("top_scores"))[:top_k]]
        pdistances = [finite_or_nan(value) for value in parse_json_list(row.get("top_pdistances"))[:top_k]]
        if not labels:
            continue
        best_score = float(np.nanmax(scores)) if np.isfinite(scores).any() else np.nan
        best_pdistance = float(np.nanmin(pdistances)) if np.isfinite(pdistances).any() else np.nan
        true_values = {
            "species": normalize_species(row.get("true_tree_label")),
            "genus": clean(row.get("genus_name")),
            "family": clean(row.get("family_name")),
            "order": clean(row.get("order_name")),
        }
        for idx, label in enumerate(labels):
            tax = taxa.get(label, {})
            score = scores[idx] if idx < len(scores) else np.nan
            pdistance = pdistances[idx] if idx < len(pdistances) else np.nan
            candidate_values = {rank: clean(tax.get(rank)) for rank in RANKS}
            candidate_values["species"] = label
            out: dict[str, Any] = {
                "split": run.split,
                "processid": row.get("processid"),
                "query_index": int(query_idx),
                "candidate_index": int(idx),
                "candidate_rank_1based": int(idx + 1),
                "candidate_tree_label": label,
                "candidate_score": score,
                "candidate_pdistance": pdistance,
                "candidate_score_delta_best": score - best_score if np.isfinite(score) and np.isfinite(best_score) else np.nan,
                "candidate_pdistance_delta_best": pdistance - best_pdistance
                if np.isfinite(pdistance) and np.isfinite(best_pdistance)
                else np.nan,
                "reciprocal_rank": 1.0 / float(idx + 1),
                "reference_sequence_count": safe_float(tax.get("reference_sequence_count")),
                "candidate_has_seen_test_query": safe_float(tax.get("has_seen_test_query")),
                "candidate_has_eval_c_query": safe_float(tax.get("has_eval_c_query")),
                "candidate_has_unseen_genera_query": safe_float(tax.get("has_unseen_genera_query")),
            }
            add_baseline_features(out, row.get("processid"), label, baseline_lookup)
            add_tree_neighborhood_features(
                out,
                label,
                labels,
                taxa,
                tree_distances,
                tree_neighborhood_size,
            )
            for rank in ("genus", "family", "order"):
                out[f"candidate_{rank}_top10_fraction"] = fraction_same(labels, taxa, candidate_values[rank], rank, 10)
                out[f"candidate_{rank}_top50_fraction"] = fraction_same(labels, taxa, candidate_values[rank], rank, top_k)
            for rank in RANKS:
                out[f"true_{rank}"] = true_values[rank]
                out[f"candidate_{rank}"] = candidate_values[rank]
                out[f"{rank}_correct"] = int(bool(true_values[rank] and candidate_values[rank] == true_values[rank]))
            rows.append(out)
        if (query_idx + 1) % 5000 == 0:
            logger.log(f"split={run.split} processed_queries={query_idx + 1} candidate_rows={len(rows)}")
    return pd.DataFrame(rows)


_BASELINE_ROOT: Path | None = None
_BASELINE_METHODS: tuple[str, ...] = ()


def configure_baseline_evidence(root: Path | None, methods: list[str] | None) -> None:
    global _BASELINE_ROOT, _BASELINE_METHODS
    _BASELINE_ROOT = root
    _BASELINE_METHODS = tuple(methods or ())


def baseline_prediction_path(root: Path, split: str, method: str) -> Path:
    return root / f"baselines_{split}" / method / "zero_shot_candidate_predictions.csv"


def load_baseline_lookups(split: str, logger: ProgressLogger) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    lookups: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    if _BASELINE_ROOT is None or not _BASELINE_METHODS:
        return lookups
    for method in _BASELINE_METHODS:
        path = baseline_prediction_path(_BASELINE_ROOT, split, method)
        if not path.exists():
            logger.log(f"Baseline evidence missing split={split} method={method}: {path}")
            lookups[method] = {}
            continue
        logger.log(f"Loading baseline evidence split={split} method={method} from {path}")
        frame = pd.read_csv(path)
        method_lookup: dict[str, dict[str, dict[str, float]]] = {}
        for _, row in frame.iterrows():
            processid = clean(row.get("processid"))
            labels = [normalize_species(value) for value in parse_json_list(row.get("top_tree_labels"))]
            scores = [finite_or_nan(value) for value in parse_json_list(row.get("top_scores"))]
            best_score = float(np.nanmax(scores)) if np.isfinite(scores).any() else np.nan
            candidate_map: dict[str, dict[str, float]] = {}
            for idx, label in enumerate(labels):
                if not label:
                    continue
                score = scores[idx] if idx < len(scores) else np.nan
                if label not in candidate_map:
                    candidate_map[label] = {
                        f"{method}_rank_1based": float(idx + 1),
                        f"{method}_score": score,
                        f"{method}_score_delta_best": score - best_score
                        if np.isfinite(score) and np.isfinite(best_score)
                        else np.nan,
                        f"{method}_in_top50": 1.0,
                    }
            method_lookup[processid] = candidate_map
        lookups[method] = method_lookup
        logger.log(f"Loaded baseline evidence rows split={split} method={method}: {len(method_lookup)} queries")
    return lookups


def add_baseline_features(
    out: dict[str, Any],
    processid: Any,
    candidate_label: str,
    baseline_lookup: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> None:
    processid_clean = clean(processid)
    for method in _BASELINE_METHODS:
        evidence = baseline_lookup.get(method, {}).get(processid_clean, {}).get(candidate_label, {})
        out[f"{method}_rank_1based"] = evidence.get(f"{method}_rank_1based", np.nan)
        out[f"{method}_score"] = evidence.get(f"{method}_score", np.nan)
        out[f"{method}_score_delta_best"] = evidence.get(f"{method}_score_delta_best", np.nan)
        out[f"{method}_in_top50"] = evidence.get(f"{method}_in_top50", 0.0)


def default_runs() -> list[RerankRun]:
    root = PAPER1 / "pipeline_runs"
    return [
        RerankRun("seen_test", root / "coi_cnn_seed1206_seen_test_target099_pdistance_experimental"),
        RerankRun("eval_c", root / "coi_cnn_seed1206_eval_c_target099_pdistance_experimental"),
        RerankRun("unseen_genera", root / "coi_cnn_seed1206_unseen_genera_target099_pdistance_experimental"),
    ]


def resolve_runs(args: argparse.Namespace) -> list[RerankRun]:
    if args.run:
        return args.run
    return default_runs()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "split",
        "processid",
        "query_index",
        "candidate_tree_label",
        *[f"true_{rank}" for rank in RANKS],
        *[f"candidate_{rank}" for rank in RANKS],
        *[f"{rank}_correct" for rank in RANKS],
    }
    cols: list[str] = []
    for col in frame.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def standardize(train_values: np.ndarray, all_values: np.ndarray, names: list[str]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    cleaned = all_values.copy()
    stats: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
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


class CandidateMLP(nn.Module):
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


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    args: argparse.Namespace,
    logger: ProgressLogger,
) -> tuple[CandidateMLP, list[dict[str, Any]]]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    logger.log(f"Training candidate reranker on device={device}")
    model = CandidateMLP(x.shape[1], args.hidden_dim, args.dropout).to(device)
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
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
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


def predict(model: CandidateMLP, x: np.ndarray, batch_size: int, cpu: bool) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    model = model.to(device)
    model.eval()
    out: list[np.ndarray] = []
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            out.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.vstack(out)


def choose_threshold(scores: np.ndarray, labels: np.ndarray, target_precision: float) -> dict[str, Any]:
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(bool)
    tp = np.cumsum(sorted_labels)
    n = np.arange(1, len(sorted_scores) + 1)
    precision = tp / n
    recall = tp / max(int(sorted_labels.sum()), 1)
    valid = np.where(precision >= target_precision)[0]
    if len(valid) == 0:
        return {"threshold": math.inf, "calib_precision": np.nan, "calib_recall": 0.0, "calib_assigned": 0}
    idx = int(valid[-1])
    return {
        "threshold": float(sorted_scores[idx]),
        "calib_precision": float(precision[idx]),
        "calib_recall": float(recall[idx]),
        "calib_assigned": int(idx + 1),
    }


def selected_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    group_cols = ["split", "processid"]
    for rank in RANKS:
        idx = scored.groupby(group_cols, sort=False)[f"p_{rank}"].idxmax()
        part = scored.loc[idx].copy()
        part["selection_rank"] = rank
        part["selection_score"] = part[f"p_{rank}"]
        part["selection_correct"] = part[f"{rank}_correct"]
        selected.append(part)
    return pd.concat(selected, ignore_index=True, sort=False)


def summarize(scored: pd.DataFrame, selected: pd.DataFrame, thresholds: dict[str, float], target_precision: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, split_frame in selected.groupby("split", sort=True):
        for rank, rank_frame in split_frame.groupby("selection_rank", sort=False):
            labels = rank_frame["selection_correct"].to_numpy(dtype=int)
            scores = rank_frame["selection_score"].to_numpy(dtype=float)
            threshold = thresholds.get(rank, math.inf)
            assigned = scores >= threshold if math.isfinite(threshold) else np.zeros_like(labels, dtype=bool)
            assigned_n = int(assigned.sum())
            assigned_precision = float(labels[assigned].mean()) if assigned_n else np.nan
            rows.append(
                {
                    "split": split,
                    "rank": rank,
                    "target_precision": float(target_precision),
                    "n_queries": int(len(rank_frame)),
                    "reranker_top1_accuracy": float(labels.mean()) if len(labels) else np.nan,
                    "baseline_top1_accuracy": baseline_top1_accuracy(scored, split, rank),
                    "assigned_count": assigned_n,
                    "assigned_coverage": float(assigned.mean()) if len(assigned) else np.nan,
                    "assigned_precision": assigned_precision,
                    "threshold": float(threshold),
                    "candidate_level_roc_auc": metric_or_nan(
                        roc_auc_score,
                        scored.loc[scored["split"] == split, f"{rank}_correct"].to_numpy(dtype=int),
                        scored.loc[scored["split"] == split, f"p_{rank}"].to_numpy(dtype=float),
                    ),
                    "candidate_level_average_precision": metric_or_nan(
                        average_precision_score,
                        scored.loc[scored["split"] == split, f"{rank}_correct"].to_numpy(dtype=int),
                        scored.loc[scored["split"] == split, f"p_{rank}"].to_numpy(dtype=float),
                    ),
                }
            )
    return rows


def baseline_top1_accuracy(scored: pd.DataFrame, split: str, rank: str) -> float:
    first = scored[(scored["split"] == split) & (scored["candidate_index"] == 0)]
    if first.empty:
        return np.nan
    return float(first[f"{rank}_correct"].mean())


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
    parser.add_argument(
        "--tree-neighborhood-features",
        action="store_true",
        help="Add inference-safe candidate-neighborhood tree-distance and top-k diversity features.",
    )
    parser.add_argument(
        "--tree-file",
        type=Path,
        default=ROOT / "data" / "phylo" / "actinopt_12k_treePL.tre",
        help="Reference tree used when --tree-neighborhood-features is enabled.",
    )
    parser.add_argument(
        "--tree-neighborhood-size",
        type=int,
        default=10,
        help="Top-k neighborhood size used for tree-neighborhood features.",
    )
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-pos-weight", type=float, default=30.0)
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
    feature_names = feature_columns(candidates)
    train_mask = candidates["split"].eq("seen_test").to_numpy()
    train_frame = candidates[train_mask].reset_index(drop=True)
    train_idx_local, calib_idx_local = train_calib_split(len(train_frame), args.calib_fraction, args.seed)
    global_train_idx = np.where(train_mask)[0][train_idx_local]
    global_calib_idx = np.where(train_mask)[0][calib_idx_local]
    all_values = candidates[feature_names].to_numpy(dtype=np.float32)
    train_values = candidates.iloc[global_train_idx][feature_names].to_numpy(dtype=np.float32)
    x, stats = standardize(train_values, all_values, feature_names)
    y = candidates[[f"{rank}_correct" for rank in RANKS]].to_numpy(dtype=np.float32)
    logger.log(
        f"candidate_rows={len(candidates)} train_rows={len(global_train_idx)} "
        f"calib_rows={len(global_calib_idx)} features={len(feature_names)}"
    )
    model, history = train_model(x, y, global_train_idx, global_calib_idx, args, logger)
    probs = predict(model, x, args.batch_size, args.cpu)
    scored = candidates.copy()
    for idx, rank in enumerate(RANKS):
        scored[f"p_{rank}"] = probs[:, idx]
    selected = selected_candidates(scored)
    thresholds: dict[str, float] = {}
    threshold_rows: list[dict[str, Any]] = []
    calib_selected = selected[selected["split"].eq("seen_test")]
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
        "generated_by": "scripts/edna/train_paper1_candidate_reranker.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "top_k": args.top_k,
        "target_precision": float(args.target_precision),
        "runs": [{"split": run.split, "run_dir": str(run.run_dir)} for run in runs],
        "baseline_root": str(args.baseline_root) if args.baseline_methods else None,
        "baseline_methods": args.baseline_methods,
        "tree_neighborhood_features": bool(args.tree_neighborhood_features),
        "tree_file": str(args.tree_file) if args.tree_neighborhood_features else None,
        "tree_neighborhood_size": int(args.tree_neighborhood_size),
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
            "First candidate-level neural reranker trained on seen-test top-k "
            "candidate rows. It uses vector, p-distance, top-k taxonomic "
            "cluster, reference-count features, and any configured classical "
            "baseline top-k evidence. If tree_neighborhood_features is true, "
            "it also uses inference-safe distances and diversity within the "
            "retrieved candidate neighborhood, not true-query-to-candidate "
            "tree distances."
        ),
    }
    logger.log(f"Writing manifest to {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
