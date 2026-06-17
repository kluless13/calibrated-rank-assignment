#!/usr/bin/env python3
"""Stress-test ANN vector retrieval speed with expanded reference catalogs.

The expansion is synthetic and is only a speed/memory stress test. It does not
create new biological candidates and must not be reported as retrieval accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from build_ann_vector_retrieval_benchmark import hnswlib, hnsw_search  # noqa: E402
from build_vector_first_retrieval_benchmark import normalize, parse_run, resolve_path, write_csv  # noqa: E402
from phylo_zero_shot_common import load_query_embedding_npz, load_tree_embedding_npz  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def expand_candidates(candidates: np.ndarray, multiplier: int, noise_std: float, seed: int) -> np.ndarray:
    if multiplier <= 1:
        return candidates.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    chunks = [candidates.astype(np.float32, copy=True)]
    for _ in range(multiplier - 1):
        noise = rng.normal(0.0, noise_std, size=candidates.shape).astype(np.float32)
        chunks.append(candidates.astype(np.float32, copy=True) + noise)
    return normalize(np.vstack(chunks))


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


def benchmark_run(run, args: argparse.Namespace, logger: ProgressLogger) -> list[dict[str, object]]:
    _processids, query_embeddings, metadata = load_query_embedding_npz(run.path)
    tree_npz = resolve_path(str(metadata["tree_embedding_npz"]), args.search_root)
    _candidate_labels, tree_embeddings, _ = load_tree_embedding_npz(tree_npz)

    q = normalize(query_embeddings.astype(np.float32))
    if args.max_queries is not None:
        q = q[: args.max_queries]
    base_c = normalize(tree_embeddings.astype(np.float32))

    rows: list[dict[str, object]] = []
    for multiplier in args.candidate_multipliers:
        c = expand_candidates(
            base_c,
            multiplier=multiplier,
            noise_std=args.synthetic_noise_std,
            seed=args.synthetic_seed + multiplier,
        )
        logger.log(
            f"Stress HNSW {run.method} seed={run.seed} split={run.split}: "
            f"{q.shape[0]} queries x {c.shape[0]} candidates "
            f"(multiplier={multiplier})"
        )
        for m in args.hnsw_m:
            for ef_search in args.hnsw_ef_search:
                ann_idx, build_seconds, search_seconds, index_bytes = hnsw_search(
                    q,
                    c,
                    top_k=args.top_k,
                    m=m,
                    ef_construction=args.hnsw_ef_construction,
                    ef_search=ef_search,
                    threads=args.threads,
                )
                rows.append(
                    {
                        "method": run.method,
                        "seed": run.seed,
                        "split": run.split,
                        "retrieval_mode": f"hnsw_stress_m{m}_ef{ef_search}",
                        "stress_multiplier": multiplier,
                        "synthetic_expansion": multiplier > 1,
                        "synthetic_noise_std": args.synthetic_noise_std if multiplier > 1 else 0.0,
                        "n_queries": int(q.shape[0]),
                        "n_base_candidates": int(base_c.shape[0]),
                        "n_candidates": int(c.shape[0]),
                        "embedding_dim": int(c.shape[1]),
                        "top_k": args.top_k,
                        "index_mb": float(index_bytes / 1_000_000) if index_bytes is not None else "",
                        "candidate_embedding_mb": float(c.nbytes / 1_000_000),
                        "index_build_seconds": float(build_seconds),
                        "search_seconds": float(search_seconds),
                        "queries_per_second": float(q.shape[0] / search_seconds) if search_seconds > 0 else "",
                        "milliseconds_per_query": float(1000.0 * search_seconds / q.shape[0]) if q.shape[0] else "",
                        "source_file": str(run.path),
                    }
                )
                logger.log(
                    f"Finished multiplier={multiplier} m={m} ef={ef_search}: "
                    f"build={build_seconds:.3f}s search={search_seconds:.3f}s "
                    f"ms/query={1000.0 * search_seconds / q.shape[0]:.4f}; "
                    f"first_label={int(ann_idx[0, 0]) if ann_idx.size else -1}"
                )
    return rows


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
    parser.add_argument("--candidate-multipliers", nargs="+", type=int, default=[1, 5, 10, 25])
    parser.add_argument("--synthetic-noise-std", type=float, default=1e-4)
    parser.add_argument("--synthetic-seed", type=int, default=1206)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--hnsw-m", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--hnsw-ef-search", nargs="+", type=int, default=[50])
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    if hnswlib is None:
        raise RuntimeError("hnswlib is required for ANN stress benchmarks")

    runs = load_runs(args)
    logger.log(f"Stress benchmarking {len(runs)} run(s)")
    rows: list[dict[str, object]] = []
    for run in runs:
        rows.extend(benchmark_run(run, args, logger))

    output_path = args.output_dir / "ann_vector_stress_runtime.csv"
    logger.log(f"Writing stress runtime table to {output_path}")
    write_csv(
        output_path,
        rows,
        [
            "method",
            "seed",
            "split",
            "retrieval_mode",
            "stress_multiplier",
            "synthetic_expansion",
            "synthetic_noise_std",
            "n_queries",
            "n_base_candidates",
            "n_candidates",
            "embedding_dim",
            "top_k",
            "index_mb",
            "candidate_embedding_mb",
            "index_build_seconds",
            "search_seconds",
            "queries_per_second",
            "milliseconds_per_query",
            "source_file",
        ],
    )
    manifest = {
        "generated_by": "scripts/edna/build_ann_vector_stress_benchmark.py",
        "output_csv": str(output_path),
        "rows": len(rows),
        "candidate_multipliers": args.candidate_multipliers,
        "synthetic_noise_std": args.synthetic_noise_std,
        "notes": [
            "Synthetic expansion is a speed/memory stress test only.",
            "Do not report expanded rows as biological retrieval accuracy.",
            "Use the non-stress ANN benchmark for recall against exact vector search.",
        ],
    }
    manifest_path = args.output_dir / "ann_vector_stress_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
