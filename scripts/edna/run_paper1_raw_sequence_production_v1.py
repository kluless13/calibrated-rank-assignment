#!/usr/bin/env python3
"""Run Paper 1 production v1 from raw split sequences.

This hardens production v1 beyond saved embedding packaging:

1. run the selected fish-tree encoder on raw `zero_shot_queries.csv` sequences;
2. run vector retrieval plus optional p-distance reranking from the exported
   query embeddings;
3. apply locked production-v1 rank/no-call thresholds;
4. write timing, manifest, and final assignment outputs.

The current intended use is CNN seed1206 on the clean COI fish-tree splits.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def run_step(name: str, command: list[str], logger: ProgressLogger) -> float:
    logger.log(f"START step={name}")
    logger.log(" ".join(command))
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start
    logger.log(f"DONE step={name} seconds={elapsed:.3f}")
    return elapsed


def read_pipeline_metric(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    sub = frame[frame["metric"].astype(str) == metric]
    if sub.empty:
        return None
    value = pd.to_numeric(sub["value"], errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--tree-embedding-npz", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/raw_sequence_production_v1"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/pipeline_calibration/pipeline_mode_thresholds.csv"),
    )
    parser.add_argument("--prediction-set", default="cnn_seed1206")
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rerank-top-candidates", type=int, default=25)
    parser.add_argument("--python", default=__import__("sys").executable)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    output_dir = args.output_root / args.split
    embedding_dir = output_dir / "embedding_export"
    pipeline_dir = output_dir / "pipeline_run"
    production_root = output_dir / "production_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    embedding_seconds = run_step(
        "raw_sequence_embedding_export",
        [
            args.python,
            "scripts/edna/train_fish_tree_encoder_benchmark.py",
            "predict",
            "--input-dir",
            str(args.input_dir),
            "--tree-file",
            "data/phylo/actinopt_12k_treePL.tre",
            "--output-dir",
            str(embedding_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--run-manifest",
            str(args.run_manifest),
            "--tree-embedding-npz",
            str(args.tree_embedding_npz),
            "--write-query-embeddings",
            "--predict-batch-size",
            str(args.predict_batch_size),
            "--num-workers",
            "4",
        ],
        logger,
    )

    pipeline_seconds = run_step(
        "vector_retrieval_pdistance_pipeline",
        [
            args.python,
            "scripts/edna/run_paper1_coi_pipeline.py",
            "--query-embeddings",
            str(embedding_dir / "query_embeddings.npz"),
            "--prediction-set",
            args.prediction_set,
            "--target-precision",
            str(args.target_precision),
            "--output-dir",
            str(pipeline_dir),
            "--top-k",
            str(args.top_k),
            "--retrieval-mode",
            "exact",
            "--rerank-mode",
            "p_distance",
            "--rerank-top-candidates",
            str(args.rerank_top_candidates),
            "--assignment-source",
            "reranked",
        ],
        logger,
    )

    production_seconds = run_step(
        "locked_production_v1_thresholds",
        [
            args.python,
            "scripts/edna/run_paper1_production_v1.py",
            "--target-precision",
            str(args.target_precision),
            "--thresholds",
            str(args.thresholds),
            "--output-root",
            str(production_root),
            "--input-run-dir",
            str(pipeline_dir),
        ],
        logger,
    )

    production_summary = production_root / args.split / "production_v1_summary.csv"
    if not production_summary.exists():
        candidates = sorted(production_root.glob("*/production_v1_summary.csv"))
        production_summary = candidates[0] if candidates else production_summary
    final = pd.read_csv(production_summary).iloc[0].to_dict() if production_summary.exists() else {}
    vector_seconds = read_pipeline_metric(pipeline_dir / "pipeline_summary.csv", "vector_search_seconds")
    rerank_seconds = read_pipeline_metric(pipeline_dir / "pipeline_summary.csv", "rerank_seconds")
    n_queries = int(final.get("n_queries", 0) or 0)
    total_seconds = embedding_seconds + pipeline_seconds + production_seconds
    summary: dict[str, Any] = {
        "split": args.split,
        "n_queries": n_queries,
        "embedding_export_seconds": embedding_seconds,
        "pipeline_seconds": pipeline_seconds,
        "production_packaging_seconds": production_seconds,
        "total_seconds": total_seconds,
        "total_ms_per_query": 1000.0 * total_seconds / n_queries if n_queries else None,
        "vector_retrieval_seconds": vector_seconds,
        "rerank_seconds": rerank_seconds,
        "assigned_precision": final.get("assigned_precision"),
        "coverage": final.get("coverage"),
        "false_species_call_rate_all_queries": final.get("false_species_call_rate_all_queries"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = output_dir / "raw_sequence_production_v1_timing_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    manifest = {
        "generated_by": "scripts/edna/run_paper1_raw_sequence_production_v1.py",
        "split": args.split,
        "input_dir": str(args.input_dir),
        "checkpoint": str(args.checkpoint),
        "run_manifest": str(args.run_manifest),
        "tree_embedding_npz": str(args.tree_embedding_npz),
        "outputs": {
            "embedding_dir": str(embedding_dir),
            "pipeline_dir": str(pipeline_dir),
            "production_root": str(production_root),
            "timing_summary": str(summary_path),
        },
        "summary": summary,
        "claim_boundary": (
            "Raw-sequence production-v1 timing for an existing clean split. "
            "This includes raw split sequence embedding export, exact-vector "
            "retrieval, top-k p-distance reranking, and locked rank/no-call "
            "packaging. It is still a research pipeline, not a packaged CLI."
        ),
    }
    manifest_path = output_dir / "raw_sequence_production_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    logger.log(f"Writing {summary_path}")
    logger.log(f"Writing {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
