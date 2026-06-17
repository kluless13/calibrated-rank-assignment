#!/usr/bin/env python3
"""Benchmark ANN vector-store retrieval from saved COI query embeddings.

This is the controlled speed layer for the "fast learned BLAST-like" track.
It compares exact vector cosine search against an HNSW vector index, then
records both biological top-k accuracy and ANN recall against exact search.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from build_vector_first_retrieval_benchmark import (  # noqa: E402
    TOP_KS,
    baseline_runtime_rows,
    load_candidate_taxonomy,
    load_queries,
    normalize,
    parse_run,
    query_rank_values,
    resolve_path,
    score_topk,
    topk_indices,
    write_csv,
)
from phylo_zero_shot_common import load_query_embedding_npz, load_tree_embedding_npz  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402

try:  # noqa: E402
    import hnswlib
except ModuleNotFoundError:  # pragma: no cover - availability depends on host
    hnswlib = None


ROOT = Path(__file__).resolve().parents[2]


def exact_search(q: np.ndarray, c: np.ndarray, block_size: int, top_k: int) -> tuple[np.ndarray, float]:
    chunks: list[np.ndarray] = []
    start = time.perf_counter()
    for offset in range(0, q.shape[0], block_size):
        scores = q[offset : offset + block_size] @ c.T
        chunks.append(topk_indices(scores, top_k))
    elapsed = time.perf_counter() - start
    return np.vstack(chunks), elapsed


def hnsw_search(
    q: np.ndarray,
    c: np.ndarray,
    top_k: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    threads: int,
) -> tuple[np.ndarray, float, float, int | None]:
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed")
    index = hnswlib.Index(space="cosine", dim=c.shape[1])
    start_build = time.perf_counter()
    index.init_index(max_elements=c.shape[0], ef_construction=ef_construction, M=m)
    index.set_num_threads(threads)
    index.add_items(c, np.arange(c.shape[0], dtype=np.int64), num_threads=threads)
    index.set_ef(ef_search)
    build_seconds = time.perf_counter() - start_build

    start_search = time.perf_counter()
    labels, _ = index.knn_query(q, k=top_k, num_threads=threads)
    search_seconds = time.perf_counter() - start_search

    index_bytes = None
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "hnsw.index"
        index.save_index(str(index_path))
        index_bytes = index_path.stat().st_size
    return labels.astype(np.int64), build_seconds, search_seconds, index_bytes


def recall_rows(
    exact_idx: np.ndarray,
    ann_idx: np.ndarray,
    method: str,
    seed: str,
    split: str,
    retrieval_mode: str,
) -> list[dict[str, object]]:
    rows = []
    for k in TOP_KS:
        capped = min(k, exact_idx.shape[1], ann_idx.shape[1])
        if capped == 0:
            continue
        recalls = []
        for exact_row, ann_row in zip(exact_idx[:, :capped], ann_idx[:, :capped], strict=True):
            recalls.append(len(set(exact_row.tolist()) & set(ann_row.tolist())) / capped)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "split": split,
                "retrieval_mode": retrieval_mode,
                "top_k": k,
                "n_queries": int(exact_idx.shape[0]),
                "mean_recall_vs_exact": float(np.mean(recalls)),
                "top1_match_rate": float(np.mean(exact_idx[:, 0] == ann_idx[:, 0])),
            }
        )
    return rows


def metric_rows_for_idx(
    top_idx: np.ndarray,
    query_tax,
    candidate_tax,
    method: str,
    seed: str,
    split: str,
    retrieval_mode: str,
    n_candidates: int,
    embedding_dim: int,
    source_file: Path,
) -> list[dict[str, object]]:
    rows = []
    for row in score_topk(top_idx, query_tax, candidate_tax):
        row.update(
            {
                "method": method,
                "seed": seed,
                "split": split,
                "retrieval_mode": retrieval_mode,
                "n_candidates": n_candidates,
                "embedding_dim": embedding_dim,
                "source_file": str(source_file),
            }
        )
        rows.append(row)
    return rows


def runtime_row(
    method: str,
    seed: str,
    split: str,
    retrieval_mode: str,
    q: np.ndarray,
    c: np.ndarray,
    build_seconds: float,
    search_seconds: float,
    index_bytes: int | None,
    source_file: Path,
) -> dict[str, object]:
    return {
        "method": method,
        "seed": seed,
        "split": split,
        "retrieval_mode": retrieval_mode,
        "n_queries": int(q.shape[0]),
        "n_candidates": int(c.shape[0]),
        "embedding_dim": int(c.shape[1]),
        "query_embedding_mb": float(q.nbytes / 1_000_000),
        "candidate_embedding_mb": float(c.nbytes / 1_000_000),
        "index_mb": float(index_bytes / 1_000_000) if index_bytes is not None else "",
        "index_build_seconds": float(build_seconds),
        "search_seconds": float(search_seconds),
        "queries_per_second": float(q.shape[0] / search_seconds) if search_seconds > 0 else None,
        "milliseconds_per_query": float(1000.0 * search_seconds / q.shape[0]) if q.shape[0] else None,
        "source_file": str(source_file),
    }


def benchmark_run(run, args, logger: ProgressLogger) -> tuple[list[dict], list[dict], list[dict]]:
    processids, query_embeddings, metadata = load_query_embedding_npz(run.path)
    input_dir = resolve_path(str(metadata["input_dir"]), args.search_root)
    tree_npz = resolve_path(str(metadata["tree_embedding_npz"]), args.search_root)
    candidate_labels, tree_embeddings, _ = load_tree_embedding_npz(tree_npz)

    queries = load_queries(input_dir, processids)
    query_tax = query_rank_values(queries)
    candidate_tax = load_candidate_taxonomy(input_dir, candidate_labels)
    q = normalize(query_embeddings.astype(np.float32))
    c = normalize(tree_embeddings.astype(np.float32))

    logger.log(
        f"Exact vector baseline for {run.method} seed={run.seed} split={run.split}: "
        f"{q.shape[0]} queries x {c.shape[0]} candidates"
    )
    exact_idx, exact_seconds = exact_search(q, c, args.block_size, args.top_k)
    metric_rows = metric_rows_for_idx(
        exact_idx,
        query_tax,
        candidate_tax,
        run.method,
        run.seed,
        run.split,
        "exact_vector_cosine",
        c.shape[0],
        c.shape[1],
        run.path,
    )
    runtime_rows = [
        runtime_row(
            run.method,
            run.seed,
            run.split,
            "exact_vector_cosine",
            q,
            c,
            0.0,
            exact_seconds,
            None,
            run.path,
        )
    ]
    recall = []

    if hnswlib is None:
        logger.log("Skipping HNSW: hnswlib is not installed")
        return metric_rows, runtime_rows, recall

    for m in args.hnsw_m:
        for ef_search in args.hnsw_ef_search:
            mode = f"hnsw_cosine_m{m}_ef{ef_search}"
            logger.log(f"Building/searching {mode} for {run.method} seed={run.seed} split={run.split}")
            ann_idx, build_seconds, search_seconds, index_bytes = hnsw_search(
                q,
                c,
                args.top_k,
                m=m,
                ef_construction=args.hnsw_ef_construction,
                ef_search=ef_search,
                threads=args.threads,
            )
            metric_rows.extend(
                metric_rows_for_idx(
                    ann_idx,
                    query_tax,
                    candidate_tax,
                    run.method,
                    run.seed,
                    run.split,
                    mode,
                    c.shape[0],
                    c.shape[1],
                    run.path,
                )
            )
            runtime_rows.append(
                runtime_row(
                    run.method,
                    run.seed,
                    run.split,
                    mode,
                    q,
                    c,
                    build_seconds,
                    search_seconds,
                    index_bytes,
                    run.path,
                )
            )
            recall.extend(recall_rows(exact_idx, ann_idx, run.method, run.seed, run.split, mode))
            logger.log(
                f"Finished {mode}: build={build_seconds:.3f}s search={search_seconds:.3f}s "
                f"qps={q.shape[0] / search_seconds if search_seconds > 0 else 0:.1f}"
            )

    return metric_rows, runtime_rows, recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-embedding-root",
        type=Path,
        action="append",
        default=[
            Path("results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings"),
            Path("results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/cnn_seed_repeats"),
        ],
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
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
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=max(TOP_KS))
    parser.add_argument("--hnsw-m", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--hnsw-ef-search", nargs="+", type=int, default=[20, 50, 100])
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    if hnswlib is None:
        logger.log("hnswlib is unavailable; output will contain exact-vector rows only")

    runs = []
    seen: set[Path] = set()
    for root in args.query_embedding_root:
        for path in sorted(root.glob("*/query_embeddings.npz")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run = parse_run(path)
            if run is not None:
                runs.append(run)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    logger.log(f"Benchmarking {len(runs)} embedding runs")

    metric_rows: list[dict] = []
    runtime_rows: list[dict] = []
    recall_rows_out: list[dict] = []
    for run in runs:
        rows, runtime, recall = benchmark_run(run, args, logger)
        metric_rows.extend(rows)
        runtime_rows.extend(runtime)
        recall_rows_out.extend(recall)

    for row in baseline_runtime_rows(args.baseline_root):
        row = dict(row)
        row.setdefault("index_mb", "")
        row.setdefault("index_build_seconds", "")
        runtime_rows.append(row)

    metric_path = args.output_dir / "ann_vector_retrieval_metrics.csv"
    runtime_path = args.output_dir / "ann_vector_runtime_comparison.csv"
    recall_path = args.output_dir / "ann_vector_recall_against_exact.csv"
    logger.log(f"Writing ANN vector metrics to {metric_path}")
    write_csv(
        metric_path,
        metric_rows,
        [
            "method",
            "seed",
            "split",
            "retrieval_mode",
            "target_rank",
            "top_k",
            "n_queries",
            "hit_pct",
            "n_candidates",
            "embedding_dim",
            "source_file",
        ],
    )
    logger.log(f"Writing ANN vector runtime comparison to {runtime_path}")
    write_csv(
        runtime_path,
        runtime_rows,
        [
            "method",
            "seed",
            "split",
            "retrieval_mode",
            "n_queries",
            "n_candidates",
            "embedding_dim",
            "query_embedding_mb",
            "candidate_embedding_mb",
            "index_mb",
            "index_build_seconds",
            "search_seconds",
            "queries_per_second",
            "milliseconds_per_query",
            "source_file",
        ],
    )
    logger.log(f"Writing ANN recall rows to {recall_path}")
    write_csv(
        recall_path,
        recall_rows_out,
        [
            "method",
            "seed",
            "split",
            "retrieval_mode",
            "top_k",
            "n_queries",
            "mean_recall_vs_exact",
            "top1_match_rate",
        ],
    )

    manifest = {
        "generated_by": "scripts/edna/build_ann_vector_retrieval_benchmark.py",
        "hnswlib_available": hnswlib is not None,
        "embedding_runs": len(runs),
        "metric_rows": len(metric_rows),
        "runtime_rows": len(runtime_rows),
        "recall_rows": len(recall_rows_out),
        "hnsw_m": args.hnsw_m,
        "hnsw_ef_search": args.hnsw_ef_search,
        "hnsw_ef_construction": args.hnsw_ef_construction,
        "metrics_csv": str(metric_path),
        "runtime_csv": str(runtime_path),
        "recall_csv": str(recall_path),
        "notes": [
            "Exact vector cosine is the no-approximation retrieval reference.",
            "HNSW recall is measured against exact vector top-k, not against biological truth directly.",
            "Classical BLAST/VSEARCH/k-mer runtime rows are included only where runtime.json exists.",
        ],
    }
    manifest_path = args.output_dir / "ann_vector_retrieval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
