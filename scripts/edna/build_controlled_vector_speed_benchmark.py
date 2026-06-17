#!/usr/bin/env python3
"""Repeat-based controlled vector retrieval speed benchmark.

This script measures exact vector search and HNSW search repeatedly on the same
saved COI query/reference embedding run. It is intended for manuscript-facing
speed claims, where a median and spread are more defensible than a single timing
row.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_ann_vector_retrieval_benchmark import hnswlib  # noqa: E402
from build_vector_first_retrieval_benchmark import normalize, parse_run, resolve_path, topk_indices  # noqa: E402
from phylo_zero_shot_common import load_query_embedding_npz, load_tree_embedding_npz  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float | str:
    if not values:
        return ""
    return float(np.percentile(np.array(values, dtype=float), q))


def median(values: list[float]) -> float | str:
    if not values:
        return ""
    return float(statistics.median(values))


def version_of(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "available_unknown" if package == "hnswlib" and hnswlib is not None else "not_installed"


def load_runs(args: argparse.Namespace):
    runs = []
    seen: set[Path] = set()
    for root in args.query_embedding_root:
        for path in sorted(root.glob("*/query_embeddings.npz")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run = parse_run(path)
            if run is None:
                continue
            if args.method and run.method != args.method:
                continue
            if args.seed and str(run.seed) != str(args.seed):
                continue
            if args.split and run.split != args.split:
                continue
            runs.append(run)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    return runs


def exact_once(q: np.ndarray, c: np.ndarray, block_size: int, top_k: int) -> tuple[float, int]:
    chunks: list[np.ndarray] = []
    start = time.perf_counter()
    for offset in range(0, q.shape[0], block_size):
        scores = q[offset : offset + block_size] @ c.T
        chunks.append(topk_indices(scores, top_k))
    elapsed = time.perf_counter() - start
    idx = np.vstack(chunks)
    return elapsed, int(idx[:, 0].sum()) if idx.size else 0


def hnsw_once(
    q: np.ndarray,
    c: np.ndarray,
    top_k: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    threads: int,
    warmup: int,
) -> tuple[float, float, int, int]:
    if hnswlib is None:
        raise RuntimeError("hnswlib is not installed")
    index = hnswlib.Index(space="cosine", dim=c.shape[1])
    start_build = time.perf_counter()
    index.init_index(max_elements=c.shape[0], ef_construction=ef_construction, M=m)
    index.set_num_threads(threads)
    index.add_items(c, np.arange(c.shape[0], dtype=np.int64), num_threads=threads)
    index.set_ef(ef_search)
    build_seconds = time.perf_counter() - start_build

    for _ in range(warmup):
        index.knn_query(q, k=top_k, num_threads=threads)

    start_search = time.perf_counter()
    labels, _ = index.knn_query(q, k=top_k, num_threads=threads)
    search_seconds = time.perf_counter() - start_search

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "hnsw.index"
        index.save_index(str(index_path))
        index_bytes = index_path.stat().st_size
    checksum = int(labels[:, 0].sum()) if labels.size else 0
    return build_seconds, search_seconds, index_bytes, checksum


def summary_row(
    detail_rows: list[dict[str, Any]],
    method: str,
    seed: str,
    split: str,
    retrieval_mode: str,
    q: np.ndarray,
    c: np.ndarray,
    top_k: int,
    source_file: Path,
) -> dict[str, Any]:
    search = [float(row["search_seconds"]) for row in detail_rows]
    build = [float(row["index_build_seconds"]) for row in detail_rows if row["index_build_seconds"] != ""]
    index_mb = [float(row["index_mb"]) for row in detail_rows if row["index_mb"] != ""]
    ms_per_query = [1000.0 * seconds / q.shape[0] for seconds in search]
    qps = [q.shape[0] / seconds for seconds in search if seconds > 0]
    return {
        "method": method,
        "seed": seed,
        "split": split,
        "retrieval_mode": retrieval_mode,
        "n_queries": int(q.shape[0]),
        "n_candidates": int(c.shape[0]),
        "embedding_dim": int(c.shape[1]),
        "top_k": int(top_k),
        "repeats": len(search),
        "index_mb_median": median(index_mb),
        "index_build_seconds_median": median(build),
        "index_build_seconds_p05": percentile(build, 5),
        "index_build_seconds_p95": percentile(build, 95),
        "search_seconds_median": median(search),
        "search_seconds_p05": percentile(search, 5),
        "search_seconds_p95": percentile(search, 95),
        "queries_per_second_median": median(qps),
        "milliseconds_per_query_median": median(ms_per_query),
        "milliseconds_per_query_p05": percentile(ms_per_query, 5),
        "milliseconds_per_query_p95": percentile(ms_per_query, 95),
        "source_file": str(source_file),
    }


def benchmark_run(run, args: argparse.Namespace, logger: ProgressLogger) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _processids, query_embeddings, metadata = load_query_embedding_npz(run.path)
    tree_npz = resolve_path(str(metadata["tree_embedding_npz"]), args.search_root)
    _candidate_labels, tree_embeddings, _ = load_tree_embedding_npz(tree_npz)

    q = normalize(query_embeddings.astype(np.float32))
    if args.max_queries is not None:
        q = q[: args.max_queries]
    c = normalize(tree_embeddings.astype(np.float32))

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    logger.log(f"Controlled exact vector benchmark: {run.method} seed={run.seed} split={run.split}")
    for _ in range(args.warmup):
        exact_once(q, c, args.block_size, args.top_k)
    exact_details = []
    for repeat in range(1, args.repeats + 1):
        search_seconds, checksum = exact_once(q, c, args.block_size, args.top_k)
        row = {
            "method": run.method,
            "seed": run.seed,
            "split": run.split,
            "retrieval_mode": "exact_vector_cosine_controlled",
            "repeat": repeat,
            "n_queries": int(q.shape[0]),
            "n_candidates": int(c.shape[0]),
            "embedding_dim": int(c.shape[1]),
            "top_k": args.top_k,
            "index_mb": "",
            "index_build_seconds": "",
            "search_seconds": float(search_seconds),
            "milliseconds_per_query": float(1000.0 * search_seconds / q.shape[0]),
            "top1_checksum": checksum,
            "source_file": str(run.path),
        }
        detail_rows.append(row)
        exact_details.append(row)
    summary_rows.append(
        summary_row(
            exact_details,
            run.method,
            run.seed,
            run.split,
            "exact_vector_cosine_controlled",
            q,
            c,
            args.top_k,
            run.path,
        )
    )

    if hnswlib is None:
        logger.log("Skipping controlled HNSW: hnswlib is not installed")
        return detail_rows, summary_rows

    for m in args.hnsw_m:
        for ef_search in args.hnsw_ef_search:
            mode = f"hnsw_cosine_m{m}_ef{ef_search}_controlled"
            logger.log(f"Controlled HNSW benchmark {mode}: {args.repeats} repeats")
            mode_details = []
            for repeat in range(1, args.repeats + 1):
                build_seconds, search_seconds, index_bytes, checksum = hnsw_once(
                    q,
                    c,
                    args.top_k,
                    m=m,
                    ef_construction=args.hnsw_ef_construction,
                    ef_search=ef_search,
                    threads=args.threads,
                    warmup=args.warmup,
                )
                row = {
                    "method": run.method,
                    "seed": run.seed,
                    "split": run.split,
                    "retrieval_mode": mode,
                    "repeat": repeat,
                    "n_queries": int(q.shape[0]),
                    "n_candidates": int(c.shape[0]),
                    "embedding_dim": int(c.shape[1]),
                    "top_k": args.top_k,
                    "index_mb": float(index_bytes / 1_000_000),
                    "index_build_seconds": float(build_seconds),
                    "search_seconds": float(search_seconds),
                    "milliseconds_per_query": float(1000.0 * search_seconds / q.shape[0]),
                    "top1_checksum": checksum,
                    "source_file": str(run.path),
                }
                detail_rows.append(row)
                mode_details.append(row)
            summary_rows.append(
                summary_row(mode_details, run.method, run.seed, run.split, mode, q, c, args.top_k, run.path)
            )

    return detail_rows, summary_rows


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
    parser.add_argument("--method", default="cnn")
    parser.add_argument("--seed", default="1206")
    parser.add_argument("--split", default="eval_c")
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--hnsw-m", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--hnsw-ef-search", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    runs = load_runs(args)
    logger.log(f"Controlled speed benchmarking {len(runs)} run(s)")
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for run in runs:
        detail, summary = benchmark_run(run, args, logger)
        detail_rows.extend(detail)
        summary_rows.extend(summary)

    detail_path = args.output_dir / "controlled_vector_speed_detail.csv"
    summary_path = args.output_dir / "controlled_vector_speed_benchmark.csv"
    logger.log(f"Writing controlled detail rows to {detail_path}")
    write_csv(
        detail_path,
        detail_rows,
        [
            "method",
            "seed",
            "split",
            "retrieval_mode",
            "repeat",
            "n_queries",
            "n_candidates",
            "embedding_dim",
            "top_k",
            "index_mb",
            "index_build_seconds",
            "search_seconds",
            "milliseconds_per_query",
            "top1_checksum",
            "source_file",
        ],
    )
    logger.log(f"Writing controlled summary rows to {summary_path}")
    write_csv(
        summary_path,
        summary_rows,
        [
            "method",
            "seed",
            "split",
            "retrieval_mode",
            "n_queries",
            "n_candidates",
            "embedding_dim",
            "top_k",
            "repeats",
            "index_mb_median",
            "index_build_seconds_median",
            "index_build_seconds_p05",
            "index_build_seconds_p95",
            "search_seconds_median",
            "search_seconds_p05",
            "search_seconds_p95",
            "queries_per_second_median",
            "milliseconds_per_query_median",
            "milliseconds_per_query_p05",
            "milliseconds_per_query_p95",
            "source_file",
        ],
    )
    manifest = {
        "generated_by": "scripts/edna/build_controlled_vector_speed_benchmark.py",
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_path),
        "runs": len(runs),
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "threads": args.threads,
        "hnswlib_available": hnswlib is not None,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor(),
            "numpy": np.__version__,
            "hnswlib": version_of("hnswlib"),
        },
        "notes": [
            "Use median milliseconds_per_query for speed claims.",
            "HNSW build time is reported separately from query search time.",
            "This benchmark measures vector retrieval only, not downstream reranking/calibration.",
        ],
    }
    manifest_path = args.output_dir / "controlled_vector_speed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
