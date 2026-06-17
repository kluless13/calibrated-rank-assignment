#!/usr/bin/env python3
"""Benchmark exact vector-first retrieval from saved COI query embeddings.

This is the local, dependency-light version of the "learned BLAST-like" front
end. It uses exact cosine search over saved neural/tree embeddings instead of an
ANN library so that top-k recall is not confounded by index approximation.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from phylo_zero_shot_common import load_query_embedding_npz, load_tree_embedding_npz  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RANKS = ["species", "genus", "family", "order"]
TOP_KS = [1, 5, 10, 50]


@dataclass(frozen=True)
class EmbeddingRun:
    path: Path
    method: str
    seed: str
    split: str


def parse_run(path: Path) -> EmbeddingRun | None:
    name = path.parent.name
    parts = name.split("_")
    if len(parts) < 3 or parts[0] != "coi":
        return None
    method = parts[1]
    seed = parts[2].removeprefix("seed")
    split = "_".join(parts[3:]) if len(parts) > 3 else "eval_c"
    return EmbeddingRun(path=path, method=method, seed=seed, split=split)


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def resolve_path(raw_path: str, search_roots: Iterable[Path]) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    basename = path.parent.name
    filename = path.name
    for root in search_roots:
        matches = list(root.glob(f"**/{basename}/{filename}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve {raw_path!r}")


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def load_candidate_taxonomy(input_dir: Path, candidate_labels: list[str]) -> pd.DataFrame:
    species_info_path = input_dir / "species_info.json"
    species_info = json.loads(species_info_path.read_text()) if species_info_path.exists() else {}
    candidate_path = input_dir / "candidate_species.csv"
    candidate_rows = {}
    if candidate_path.exists():
        candidate_rows = pd.read_csv(candidate_path).set_index("tree_label").to_dict(orient="index")

    rows: list[dict[str, str]] = []
    for label in candidate_labels:
        info = species_info.get(label, {})
        row = candidate_rows.get(label, {})
        genus = clean(info.get("genus")) or clean(row.get("genus_name")) or clean(row.get("genus_from_label"))
        family = clean(info.get("family")) or clean(row.get("family_name"))
        order = clean(info.get("order")) or clean(row.get("order_name"))
        rows.append(
            {
                "tree_label": label,
                "species": label,
                "genus": genus,
                "family": family,
                "order": order,
            }
        )
    return pd.DataFrame(rows)


def load_queries(input_dir: Path, processids: list[str]) -> pd.DataFrame:
    queries = pd.read_csv(input_dir / "zero_shot_queries.csv")
    queries["processid"] = queries["processid"].astype(str)
    by_processid = queries.set_index("processid", drop=False)
    missing = [processid for processid in processids if processid not in by_processid.index]
    if missing:
        raise RuntimeError(f"{len(missing)} query processids were not found in {input_dir}/zero_shot_queries.csv")
    return by_processid.loc[processids].reset_index(drop=True)


def query_rank_values(queries: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["species"] = queries["tree_label"].astype(str)
    out["genus"] = queries.get("genus_name", "").map(clean)
    out["family"] = queries.get("family_name", "").map(clean)
    out["order"] = queries.get("order_name", "").map(clean)
    return out


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if scores.shape[1] <= top_k:
        idx = np.argsort(-scores, axis=1)
        return idx
    partial = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, partial], axis=1)
    return partial[row, order]


def score_topk(
    top_idx: np.ndarray,
    query_tax: pd.DataFrame,
    candidate_tax: pd.DataFrame,
) -> list[dict[str, object]]:
    candidate_values = {rank: candidate_tax[rank].astype(str).to_numpy() for rank in RANKS}
    rows: list[dict[str, object]] = []
    for rank in RANKS:
        values = candidate_values[rank]
        target_values = query_tax[rank].astype(str).to_numpy()
        eligible = np.array([bool(clean(value)) for value in target_values])
        n = int(eligible.sum())
        if n == 0:
            continue
        for k in TOP_KS:
            capped_k = min(k, top_idx.shape[1])
            top_values = values[top_idx[:, :capped_k]]
            hits = (top_values == target_values[:, None]).any(axis=1) & eligible
            rows.append(
                {
                    "target_rank": rank,
                    "top_k": k,
                    "n_queries": n,
                    "hit_pct": 100.0 * float(hits.sum()) / n,
                }
            )
    return rows


def benchmark_run(run: EmbeddingRun, search_roots: list[Path], block_size: int, top_k: int) -> tuple[list[dict], dict]:
    processids, query_embeddings, metadata = load_query_embedding_npz(run.path)
    input_dir = resolve_path(str(metadata["input_dir"]), search_roots)
    tree_npz = resolve_path(str(metadata["tree_embedding_npz"]), search_roots)
    candidate_labels, tree_embeddings, _ = load_tree_embedding_npz(tree_npz)

    queries = load_queries(input_dir, processids)
    query_tax = query_rank_values(queries)
    candidate_tax = load_candidate_taxonomy(input_dir, candidate_labels)

    q = normalize(query_embeddings.astype(np.float32))
    c = normalize(tree_embeddings.astype(np.float32))

    top_chunks: list[np.ndarray] = []
    start = time.perf_counter()
    for offset in range(0, q.shape[0], block_size):
        scores = q[offset : offset + block_size] @ c.T
        top_chunks.append(topk_indices(scores, top_k))
    elapsed = time.perf_counter() - start
    top_idx = np.vstack(top_chunks)

    rows = []
    for row in score_topk(top_idx, query_tax, candidate_tax):
        row.update(
            {
                "method": run.method,
                "seed": run.seed,
                "split": run.split,
                "retrieval_mode": "exact_vector_cosine",
                "n_candidates": int(c.shape[0]),
                "embedding_dim": int(c.shape[1]),
                "source_file": str(run.path),
            }
        )
        rows.append(row)

    runtime = {
        "method": run.method,
        "seed": run.seed,
        "split": run.split,
        "retrieval_mode": "exact_vector_cosine",
        "n_queries": int(q.shape[0]),
        "n_candidates": int(c.shape[0]),
        "embedding_dim": int(c.shape[1]),
        "query_embedding_mb": float(q.nbytes / 1_000_000),
        "candidate_embedding_mb": float(c.nbytes / 1_000_000),
        "search_seconds": float(elapsed),
        "queries_per_second": float(q.shape[0] / elapsed) if elapsed > 0 else None,
        "milliseconds_per_query": float(1000.0 * elapsed / q.shape[0]) if q.shape[0] else None,
        "source_file": str(run.path),
    }
    return rows, runtime


def baseline_runtime_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for runtime_path in sorted(root.glob("baselines_*/*/runtime.json")):
        method = runtime_path.parent.name
        split = runtime_path.parent.parent.name.removeprefix("baselines_")
        try:
            runtime = json.loads(runtime_path.read_text())
        except json.JSONDecodeError:
            continue
        pred_path = runtime_path.parent / "zero_shot_candidate_predictions.csv"
        n_queries = sum(1 for _ in pred_path.open()) - 1 if pred_path.exists() else None
        seconds = runtime.get("seconds")
        rows.append(
            {
                "method": method,
                "seed": "",
                "split": split,
                "retrieval_mode": "classical_sequence_baseline",
                "n_queries": n_queries,
                "n_candidates": "",
                "embedding_dim": "",
                "query_embedding_mb": "",
                "candidate_embedding_mb": "",
                "search_seconds": seconds,
                "queries_per_second": (n_queries / seconds) if n_queries and seconds else "",
                "milliseconds_per_query": (1000.0 * seconds / n_queries) if n_queries and seconds else "",
                "source_file": str(runtime_path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    metric_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    logger.log(f"Searching query embedding roots: {[str(path) for path in args.query_embedding_root]}")
    for root in args.query_embedding_root:
        for path in sorted(root.glob("*/query_embeddings.npz")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            run = parse_run(path)
            if run is None:
                logger.log(f"Skipping unrecognized embedding run: {path}")
                continue
            logger.log(f"Benchmarking {run.method} seed={run.seed} split={run.split}: {path}")
            rows, runtime = benchmark_run(run, args.search_root, args.block_size, args.top_k)
            metric_rows.extend(rows)
            runtime_rows.append(runtime)
            logger.log(
                "Finished "
                f"{run.method} seed={run.seed} split={run.split}: "
                f"{runtime['n_queries']} queries, {runtime['n_candidates']} candidates, "
                f"{runtime['search_seconds']:.3f}s"
            )

    logger.log(f"Loading baseline runtime rows from {args.baseline_root}")
    runtime_rows.extend(baseline_runtime_rows(args.baseline_root))

    metrics_path = args.output_dir / "vector_first_retrieval_metrics.csv"
    runtime_path = args.output_dir / "vector_first_runtime_comparison.csv"
    logger.log(f"Writing vector retrieval metrics to {metrics_path}")
    write_csv(
        metrics_path,
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
    logger.log(f"Writing runtime comparison to {runtime_path}")
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
            "search_seconds",
            "queries_per_second",
            "milliseconds_per_query",
            "source_file",
        ],
    )

    manifest = {
        "metrics_csv": str(metrics_path),
        "runtime_csv": str(runtime_path),
        "metric_rows": len(metric_rows),
        "runtime_rows": len(runtime_rows),
        "embedding_runs": len(seen),
        "retrieval_mode": "exact_vector_cosine",
        "note": "Exact cosine search is a dependency-light proxy for the vector-first retrieval layer; ANN indexing should be benchmarked separately.",
    }
    manifest_path = args.output_dir / "vector_first_retrieval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
