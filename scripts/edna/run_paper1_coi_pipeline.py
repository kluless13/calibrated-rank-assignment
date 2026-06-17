#!/usr/bin/env python3
"""Run the Paper 1 COI vector-first rank-adaptive inference pipeline.

This is the executable pipeline path:

1. load saved query embeddings and reference tree embeddings;
2. retrieve top-k candidates by vector cosine search;
3. compute top-candidate score margin and top-10 taxonomic consensus;
4. apply a seen-test-derived missing-reference-aware rank/no-call policy;
5. emit candidate predictions, rank-adaptive assignments, and summary metrics.

The script is intentionally model-agnostic as long as query embeddings use the
shared query_embeddings.npz format.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from build_vector_first_retrieval_benchmark import normalize, resolve_path, topk_indices  # noqa: E402
from phylo_zero_shot_common import load_query_embedding_npz, load_tree_embedding_npz  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")
POLICY_FEATURES = {
    "species": "confidence_relative_margin",
    "genus": "genus_top10_consensus",
    "family": "family_top10_consensus",
    "order": "order_top10_consensus",
}
NUC_TO_INT = np.full(256, 4, dtype=np.uint8)
for _idx, _base in enumerate(b"ACGT"):
    NUC_TO_INT[_base] = _idx
    NUC_TO_INT[_base + 32] = _idx


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_queries(input_dir: Path, processids: list[str]) -> pd.DataFrame:
    queries = pd.read_csv(input_dir / "zero_shot_queries.csv")
    queries["processid"] = queries["processid"].astype(str)
    indexed = queries.set_index("processid", drop=False)
    missing = [processid for processid in processids if processid not in indexed.index]
    if missing:
        raise RuntimeError(f"{len(missing)} processids were not found in {input_dir / 'zero_shot_queries.csv'}")
    return indexed.loc[processids].reset_index(drop=True)


def load_candidate_taxonomy(input_dir: Path, candidate_labels: list[str]) -> dict[str, dict[str, str]]:
    species_info_path = input_dir / "species_info.json"
    species_info = json.loads(species_info_path.read_text()) if species_info_path.exists() else {}
    candidate_path = input_dir / "candidate_species.csv"
    candidate_rows = {}
    if candidate_path.exists():
        candidate_rows = pd.read_csv(candidate_path).set_index("tree_label").to_dict(orient="index")
    taxonomy: dict[str, dict[str, str]] = {}
    for label in candidate_labels:
        info = species_info.get(label, {})
        row = candidate_rows.get(label, {})
        genus = clean(info.get("genus")) or clean(row.get("genus_name")) or clean(row.get("genus_from_label"))
        family = clean(info.get("family")) or clean(row.get("family_name"))
        order = clean(info.get("order")) or clean(row.get("order_name"))
        taxonomy[label] = {
            "species": label,
            "genus": genus,
            "family": family,
            "order": order,
        }
    return taxonomy


def load_policy_thresholds(path: Path, prediction_set: str, target_precision: float) -> dict[str, float]:
    rows = pd.read_csv(path)
    sub = rows[
        (rows["prediction_set"].astype(str) == prediction_set)
        & (pd.to_numeric(rows["target_precision"], errors="coerce") == float(target_precision))
    ].copy()
    thresholds: dict[str, float] = {}
    for _, row in sub.iterrows():
        rank = clean(row.get("rank"))
        threshold = pd.to_numeric(pd.Series([row.get("threshold")]), errors="coerce").iloc[0]
        if rank in RANKS and np.isfinite(threshold):
            thresholds[rank] = float(threshold)
    missing = [rank for rank in RANKS if rank not in thresholds]
    if missing:
        raise RuntimeError(
            f"Missing thresholds for {prediction_set} target={target_precision}: {missing} in {path}"
        )
    return thresholds


def candidate_search(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    q = normalize(query_embeddings.astype(np.float32))
    c = normalize(candidate_embeddings.astype(np.float32))
    idx_chunks = []
    score_chunks = []
    start = time.perf_counter()
    for offset in range(0, q.shape[0], block_size):
        scores = q[offset : offset + block_size] @ c.T
        idx = topk_indices(scores, top_k)
        row = np.arange(idx.shape[0])[:, None]
        idx_chunks.append(idx)
        score_chunks.append(scores[row, idx])
    elapsed = time.perf_counter() - start
    return np.vstack(idx_chunks), np.vstack(score_chunks), elapsed


def candidate_search_hnsw(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    ef_search: int,
    ef_construction: int,
    m: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        import hnswlib  # type: ignore
    except ImportError as exc:
        raise RuntimeError("hnswlib is required for --retrieval-mode hnsw") from exc
    q = normalize(query_embeddings.astype(np.float32))
    c = normalize(candidate_embeddings.astype(np.float32))
    index = hnswlib.Index(space="cosine", dim=int(c.shape[1]))
    index.init_index(max_elements=int(c.shape[0]), ef_construction=ef_construction, M=m)
    index.add_items(c, np.arange(c.shape[0]), num_threads=threads)
    index.set_ef(max(ef_search, top_k))
    start = time.perf_counter()
    labels, distances = index.knn_query(q, k=top_k, num_threads=threads)
    elapsed = time.perf_counter() - start
    return labels.astype(np.int64), (1.0 - distances).astype(np.float32), elapsed


def load_train_reference_sequences(input_dir: Path, candidate_labels: list[str]) -> dict[str, str]:
    path = input_dir / "train_species_sequences.json"
    raw = json.loads(path.read_text())
    references: dict[str, str] = {}
    for label in candidate_labels:
        seqs = raw.get(label, [])
        if not seqs:
            continue
        references[label] = max((str(seq) for seq in seqs), key=len)
    return references


def encode_sequence(sequence: str, max_len: int) -> np.ndarray:
    encoded = np.full(max_len, 4, dtype=np.uint8)
    raw = sequence.encode("ascii", errors="ignore")[:max_len]
    if raw:
        encoded[: len(raw)] = NUC_TO_INT[np.frombuffer(raw, dtype=np.uint8)]
    return encoded


def build_reference_matrix(
    candidate_labels: list[str],
    reference_sequences: dict[str, str],
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(candidate_labels), max_len), 4, dtype=np.uint8)
    has_reference = np.zeros(len(candidate_labels), dtype=bool)
    for idx, label in enumerate(candidate_labels):
        sequence = reference_sequences.get(label)
        if not sequence:
            continue
        matrix[idx] = encode_sequence(sequence, max_len)
        has_reference[idx] = True
    return matrix, has_reference


def pairwise_pdistance(query: np.ndarray, refs: np.ndarray) -> np.ndarray:
    valid = (refs < 4) & (query[None, :] < 4)
    denom = valid.sum(axis=1)
    mismatches = ((refs != query[None, :]) & valid).sum(axis=1)
    distances = np.full(refs.shape[0], np.inf, dtype=np.float32)
    ok = denom > 0
    distances[ok] = mismatches[ok] / denom[ok]
    return distances


def rerank_by_pdistance(
    queries: pd.DataFrame,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    candidate_labels: list[str],
    input_dir: Path,
    rerank_top_candidates: int,
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    reference_sequences = load_train_reference_sequences(input_dir, candidate_labels)
    ref_matrix, has_reference = build_reference_matrix(candidate_labels, reference_sequences, max_seq_len)
    limit = min(rerank_top_candidates, top_idx.shape[1])
    new_idx = top_idx.copy()
    new_scores = top_scores.copy()
    distance_out = np.full(top_idx.shape, np.nan, dtype=np.float32)
    start = time.perf_counter()
    missing_reference_rows = 0
    for row_idx, (_, query) in enumerate(queries.iterrows()):
        query_encoded = encode_sequence(str(query.get("nucleotides", "")), max_seq_len)
        candidate_subset = top_idx[row_idx, :limit]
        distances = pairwise_pdistance(query_encoded, ref_matrix[candidate_subset])
        distances[~has_reference[candidate_subset]] = np.inf
        if np.isinf(distances).all():
            missing_reference_rows += 1
            continue
        order = np.argsort(distances, kind="stable")
        reranked_subset = candidate_subset[order]
        new_idx[row_idx, :limit] = reranked_subset
        score_lookup = {int(idx): float(score) for idx, score in zip(top_idx[row_idx, :limit], top_scores[row_idx, :limit])}
        new_scores[row_idx, :limit] = np.array([score_lookup[int(idx)] for idx in reranked_subset], dtype=np.float32)
        distance_out[row_idx, :limit] = distances[order]
        if limit < top_idx.shape[1]:
            distance_out[row_idx, limit:] = np.nan
    elapsed = time.perf_counter() - start
    stats = {
        "reference_file": str(input_dir / "train_species_sequences.json"),
        "reference_species_with_train_sequence": int(has_reference.sum()),
        "reference_species_total": int(len(candidate_labels)),
        "rerank_top_candidates": int(limit),
        "rerank_max_seq_len": int(max_seq_len),
        "queries_with_no_reference_in_rerank_window": int(missing_reference_rows),
    }
    return new_idx, new_scores, distance_out, elapsed, stats


def consensus_features(labels: list[str], scores: list[float], taxonomy: dict[str, dict[str, str]]) -> dict[str, Any]:
    top = labels[0] if labels else ""
    top_tax = taxonomy.get(top, {})
    top10 = labels[:10]
    features: dict[str, Any] = {
        "pred_species": top_tax.get("species", top),
        "pred_genus": top_tax.get("genus", ""),
        "pred_family": top_tax.get("family", ""),
        "pred_order": top_tax.get("order", ""),
        "pred_score": scores[0] if scores else np.nan,
        "second_score": scores[1] if len(scores) > 1 else np.nan,
    }
    if len(scores) >= 2 and scores[0]:
        features["confidence_margin"] = float(scores[0] - scores[1])
        features["confidence_relative_margin"] = float((scores[0] - scores[1]) / abs(scores[0]))
    else:
        features["confidence_margin"] = np.nan
        features["confidence_relative_margin"] = np.nan
    for rank in RANKS:
        if rank == "species":
            values = top10
            target = top
        else:
            values = [taxonomy.get(label, {}).get(rank, "") for label in top10]
            target = top_tax.get(rank, "")
        values = [value for value in values if value]
        if not values or not target:
            features[f"{rank}_top10_consensus"] = np.nan
            features[f"{rank}_top10_unique"] = np.nan
        else:
            features[f"{rank}_top10_consensus"] = float(sum(value == target for value in values) / len(values))
            features[f"{rank}_top10_unique"] = int(len(set(values)))
    return features


def assign_rank(features: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, str, str]:
    for rank in RANKS:
        feature = POLICY_FEATURES[rank]
        value = features.get(feature)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        if numeric >= thresholds[rank]:
            return rank, str(features.get(f"pred_{rank}", "")), f"{feature}>={thresholds[rank]:.6g}"
    return "no_call", "", "no_policy_threshold_met"


def true_rank_values(query: pd.Series) -> dict[str, str]:
    return {
        "species": clean(query.get("tree_label")).replace(" ", "_"),
        "genus": clean(query.get("genus_name")),
        "family": clean(query.get("family_name")),
        "order": clean(query.get("order_name")),
    }


def is_assignment_correct(assigned_rank: str, assigned_label: str, query: pd.Series) -> bool | str:
    if assigned_rank == "no_call":
        return ""
    truth = true_rank_values(query).get(assigned_rank, "")
    return bool(truth and assigned_label == truth)


def build_rows(
    queries: pd.DataFrame,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    candidate_labels: list[str],
    taxonomy: dict[str, dict[str, str]],
    thresholds: dict[str, float],
    top_pdistances: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    labels_arr = np.array(candidate_labels, dtype=object)
    for row_idx, (_, query) in enumerate(queries.iterrows()):
        labels = [str(label) for label in labels_arr[top_idx[row_idx]].tolist()]
        scores = [float(score) for score in top_scores[row_idx].tolist()]
        pdistances = (
            [None if not math.isfinite(float(value)) else float(value) for value in top_pdistances[row_idx].tolist()]
            if top_pdistances is not None
            else []
        )
        features = consensus_features(labels, scores, taxonomy)
        assigned_rank, assigned_label, reason = assign_rank(features, thresholds)
        correct = is_assignment_correct(assigned_rank, assigned_label, query)
        prediction_rows.append(
            {
                "processid": query["processid"],
                "true_tree_label": query["tree_label"],
                "species_name": query.get("species_name", ""),
                "genus_name": query.get("genus_name", ""),
                "family_name": query.get("family_name", ""),
                "order_name": query.get("order_name", ""),
                "top_tree_labels": json.dumps(labels),
                "top_scores": json.dumps([round(score, 8) for score in scores]),
                "top_pdistances": json.dumps(
                    [None if value is None else round(value, 8) for value in pdistances]
                )
                if top_pdistances is not None
                else "",
                "pred_tree_label": labels[0] if labels else "",
                "pred_score": scores[0] if scores else "",
                "pred_pdistance": pdistances[0] if pdistances else "",
            }
        )
        assignment_rows.append(
            {
                "processid": query["processid"],
                "true_tree_label": query["tree_label"],
                "true_genus": query.get("genus_name", ""),
                "true_family": query.get("family_name", ""),
                "true_order": query.get("order_name", ""),
                "pred_species": features["pred_species"],
                "pred_genus": features["pred_genus"],
                "pred_family": features["pred_family"],
                "pred_order": features["pred_order"],
                "pred_score": features["pred_score"],
                "second_score": features["second_score"],
                "confidence_margin": features["confidence_margin"],
                "confidence_relative_margin": features["confidence_relative_margin"],
                "species_top10_consensus": features["species_top10_consensus"],
                "genus_top10_consensus": features["genus_top10_consensus"],
                "family_top10_consensus": features["family_top10_consensus"],
                "order_top10_consensus": features["order_top10_consensus"],
                "assigned_rank": assigned_rank,
                "assigned_label": assigned_label,
                "assignment_reason": reason,
                "assigned_correct": correct,
            }
        )
    return prediction_rows, assignment_rows


def summarize_assignments(
    assignments: list[dict[str, Any]],
    search_seconds: float,
    retrieval_mode: str,
    rerank_mode: str,
    rerank_seconds: float,
    assignment_source: str,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(assignments)
    rows: list[dict[str, Any]] = [
        {"metric": "retrieval_mode", "value": retrieval_mode, "unit": "mode"},
        {"metric": "rerank_mode", "value": rerank_mode, "unit": "mode"},
        {"metric": "assignment_source", "value": assignment_source, "unit": "source"},
        {"metric": "n_queries", "value": len(frame), "unit": "queries"},
        {
            "metric": "vector_search_seconds",
            "value": search_seconds,
            "unit": "seconds",
        },
        {
            "metric": "vector_ms_per_query",
            "value": 1000.0 * search_seconds / len(frame) if len(frame) else "",
            "unit": "ms/query",
        },
        {
            "metric": "rerank_seconds",
            "value": rerank_seconds,
            "unit": "seconds",
        },
        {
            "metric": "rerank_ms_per_query",
            "value": 1000.0 * rerank_seconds / len(frame) if len(frame) and rerank_seconds else "",
            "unit": "ms/query",
        },
        {
            "metric": "candidate_stage_seconds",
            "value": search_seconds + rerank_seconds,
            "unit": "seconds",
        },
        {
            "metric": "candidate_stage_ms_per_query",
            "value": 1000.0 * (search_seconds + rerank_seconds) / len(frame) if len(frame) else "",
            "unit": "ms/query",
        },
    ]
    assigned = frame[frame["assigned_rank"] != "no_call"].copy()
    rows.append({"metric": "coverage", "value": len(assigned) / len(frame) if len(frame) else "", "unit": "fraction"})
    for rank in list(RANKS) + ["no_call"]:
        rows.append(
            {
                "metric": f"assigned_{rank}_count",
                "value": int((frame["assigned_rank"] == rank).sum()),
                "unit": "queries",
            }
        )
    if len(assigned):
        correct = assigned["assigned_correct"].map(lambda value: bool(value) if value != "" else False)
        rows.append({"metric": "assigned_precision", "value": float(correct.mean()), "unit": "fraction"})
        species_calls = frame[frame["assigned_rank"] == "species"]
        if len(species_calls):
            false_species = species_calls["assigned_correct"].map(lambda value: not bool(value)).sum()
            rows.append(
                {
                    "metric": "false_species_call_rate_all_queries",
                    "value": float(false_species / len(frame)),
                    "unit": "fraction",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-embeddings",
        type=Path,
        default=Path(
            "results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings/coi_cnn_seed1206_eval_c/query_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/missing_reference_aware_thresholds.csv"),
    )
    parser.add_argument("--prediction-set", default="cnn_seed1206")
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/pipeline_runs/coi_cnn_seed1206_eval_c_target099"),
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[
            Path("."),
            Path("results/remote_runs/2026-05-30/rtx_pro_6000"),
            Path("results/remote_runs/2026-05-31/rtx_pro_6000"),
        ],
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--retrieval-mode", choices=["exact", "hnsw"], default="exact")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-search", type=int, default=50)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--rerank-mode", choices=["none", "p_distance"], default="none")
    parser.add_argument("--rerank-top-candidates", type=int, default=25)
    parser.add_argument("--rerank-max-seq-len", type=int, default=700)
    parser.add_argument("--assignment-source", choices=["vector", "reranked"], default="vector")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    if args.assignment_source == "reranked" and args.rerank_mode == "none":
        raise SystemExit("--assignment-source reranked requires --rerank-mode p_distance")
    if args.rerank_top_candidates > args.top_k:
        raise SystemExit("--rerank-top-candidates cannot exceed --top-k")

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Loading query embeddings from {args.query_embeddings}")
    processids, query_embeddings, metadata = load_query_embedding_npz(args.query_embeddings)
    input_dir = resolve_path(str(metadata["input_dir"]), args.search_root)
    tree_npz = resolve_path(str(metadata["tree_embedding_npz"]), args.search_root)
    logger.log(f"Resolved input_dir={input_dir}")
    logger.log(f"Resolved tree_npz={tree_npz}")
    candidate_labels, tree_embeddings, _ = load_tree_embedding_npz(tree_npz)
    queries = load_queries(input_dir, processids)
    taxonomy = load_candidate_taxonomy(input_dir, candidate_labels)
    thresholds = load_policy_thresholds(args.thresholds, args.prediction_set, args.target_precision)
    logger.log(f"Loaded thresholds: {thresholds}")

    logger.log(
        f"Running {args.retrieval_mode} vector retrieval: "
        f"{query_embeddings.shape[0]} queries x {tree_embeddings.shape[0]} candidates"
    )
    if args.retrieval_mode == "hnsw":
        top_idx, top_scores, search_seconds = candidate_search_hnsw(
            query_embeddings,
            tree_embeddings,
            top_k=args.top_k,
            ef_search=args.hnsw_ef_search,
            ef_construction=args.hnsw_ef_construction,
            m=args.hnsw_m,
            threads=args.threads,
        )
    else:
        top_idx, top_scores, search_seconds = candidate_search(
            query_embeddings,
            tree_embeddings,
            top_k=args.top_k,
            block_size=args.block_size,
        )
    logger.log(f"Vector retrieval complete in {search_seconds:.3f}s")
    assignment_top_idx = top_idx
    assignment_top_scores = top_scores
    assignment_top_pdistances = None
    rerank_seconds = 0.0
    rerank_stats: dict[str, Any] = {}
    if args.rerank_mode == "p_distance":
        logger.log(
            "Running train-reference p-distance rerank over "
            f"top {args.rerank_top_candidates} retrieved candidates"
        )
        reranked_idx, reranked_scores, rerank_distances, rerank_seconds, rerank_stats = rerank_by_pdistance(
            queries,
            top_idx,
            top_scores,
            candidate_labels,
            input_dir,
            rerank_top_candidates=args.rerank_top_candidates,
            max_seq_len=args.rerank_max_seq_len,
        )
        logger.log(f"p-distance rerank complete in {rerank_seconds:.3f}s")
        if args.assignment_source == "reranked":
            assignment_top_idx = reranked_idx
            assignment_top_scores = reranked_scores
            assignment_top_pdistances = rerank_distances

    prediction_rows, assignment_rows = build_rows(
        queries,
        assignment_top_idx,
        assignment_top_scores,
        candidate_labels,
        taxonomy,
        thresholds,
        top_pdistances=assignment_top_pdistances,
    )
    summary_rows = summarize_assignments(
        assignment_rows,
        search_seconds,
        retrieval_mode=args.retrieval_mode,
        rerank_mode=args.rerank_mode,
        rerank_seconds=rerank_seconds,
        assignment_source=args.assignment_source,
    )

    predictions_path = args.output_dir / "pipeline_candidate_predictions.csv"
    assignments_path = args.output_dir / "pipeline_rank_assignments.csv"
    summary_path = args.output_dir / "pipeline_summary.csv"
    logger.log(f"Writing {predictions_path}")
    write_csv(
        predictions_path,
        prediction_rows,
        [
            "processid",
            "true_tree_label",
            "species_name",
            "genus_name",
            "family_name",
            "order_name",
            "top_tree_labels",
            "top_scores",
            "top_pdistances",
            "pred_tree_label",
            "pred_score",
            "pred_pdistance",
        ],
    )
    logger.log(f"Writing {assignments_path}")
    write_csv(
        assignments_path,
        assignment_rows,
        [
            "processid",
            "true_tree_label",
            "true_genus",
            "true_family",
            "true_order",
            "pred_species",
            "pred_genus",
            "pred_family",
            "pred_order",
            "pred_score",
            "second_score",
            "confidence_margin",
            "confidence_relative_margin",
            "species_top10_consensus",
            "genus_top10_consensus",
            "family_top10_consensus",
            "order_top10_consensus",
            "assigned_rank",
            "assigned_label",
            "assignment_reason",
            "assigned_correct",
        ],
    )
    logger.log(f"Writing {summary_path}")
    write_csv(summary_path, summary_rows, ["metric", "value", "unit"])
    manifest = {
        "generated_by": "scripts/edna/run_paper1_coi_pipeline.py",
        "query_embeddings": str(args.query_embeddings),
        "input_dir": str(input_dir),
        "tree_embedding_npz": str(tree_npz),
        "thresholds": str(args.thresholds),
        "prediction_set": args.prediction_set,
        "target_precision": args.target_precision,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "hnsw": {
            "m": args.hnsw_m,
            "ef_search": args.hnsw_ef_search,
            "ef_construction": args.hnsw_ef_construction,
            "threads": args.threads,
        }
        if args.retrieval_mode == "hnsw"
        else None,
        "rerank_mode": args.rerank_mode,
        "assignment_source": args.assignment_source,
        "rerank_stats": rerank_stats,
        "outputs": {
            "candidate_predictions": str(predictions_path),
            "rank_assignments": str(assignments_path),
            "summary": str(summary_path),
        },
        "claim_boundary": (
            "Executable vector-first/rank-adaptive pipeline over saved embeddings. "
            "Exact vector mode with assignment_source=vector is the calibrated default. "
            "HNSW is an approximate speed mode. p-distance reranking uses train_species_sequences.json only; "
            "if assignment_source=reranked, rank/no-call thresholds should be recalibrated before making final claims. "
            "This does not include strict tree-pruned retraining unless run on pruned inputs."
        ),
    }
    manifest_path = args.output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
