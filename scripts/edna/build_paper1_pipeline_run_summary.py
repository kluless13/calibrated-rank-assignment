#!/usr/bin/env python3
"""Summarize executable Paper 1 pipeline runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_dict(path: Path) -> dict[str, str]:
    rows = pd.read_csv(path)
    return {str(row["metric"]): str(row["value"]) for _, row in rows.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/pipeline_runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(args.pipeline_root.glob("*/pipeline_manifest.json")):
        run_dir = manifest_path.parent
        summary_path = run_dir / "pipeline_summary.csv"
        if not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        metrics = summary_dict(summary_path)
        input_dir = str(manifest.get("input_dir", ""))
        split = Path(input_dir).name if input_dir else ""
        rows.append(
            {
                "run_name": run_dir.name,
                "split": split,
                "prediction_set": manifest.get("prediction_set", ""),
                "target_precision": manifest.get("target_precision", ""),
                "retrieval_mode": manifest.get("retrieval_mode", metrics.get("retrieval_mode", "")),
                "rerank_mode": manifest.get("rerank_mode", metrics.get("rerank_mode", "")),
                "assignment_source": manifest.get("assignment_source", metrics.get("assignment_source", "")),
                "n_queries": metrics.get("n_queries", ""),
                "vector_ms_per_query": metrics.get("vector_ms_per_query", ""),
                "rerank_ms_per_query": metrics.get("rerank_ms_per_query", ""),
                "candidate_stage_ms_per_query": metrics.get("candidate_stage_ms_per_query", ""),
                "coverage": metrics.get("coverage", ""),
                "assigned_precision": metrics.get("assigned_precision", ""),
                "false_species_call_rate_all_queries": metrics.get("false_species_call_rate_all_queries", ""),
                "assigned_species_count": metrics.get("assigned_species_count", ""),
                "assigned_genus_count": metrics.get("assigned_genus_count", ""),
                "assigned_family_count": metrics.get("assigned_family_count", ""),
                "assigned_order_count": metrics.get("assigned_order_count", ""),
                "assigned_no_call_count": metrics.get("assigned_no_call_count", ""),
                "summary_csv": str(summary_path),
                "assignments_csv": str(run_dir / "pipeline_rank_assignments.csv"),
                "manifest_json": str(manifest_path),
                "claim_boundary": manifest.get("claim_boundary", ""),
            }
        )

    out = args.output_dir / "pipeline_run_summary.csv"
    logger.log(f"Writing {len(rows)} executable pipeline run rows to {out}")
    write_csv(
        out,
        rows,
        [
            "run_name",
            "split",
            "prediction_set",
            "target_precision",
            "retrieval_mode",
            "rerank_mode",
            "assignment_source",
            "n_queries",
            "vector_ms_per_query",
            "rerank_ms_per_query",
            "candidate_stage_ms_per_query",
            "coverage",
            "assigned_precision",
            "false_species_call_rate_all_queries",
            "assigned_species_count",
            "assigned_genus_count",
            "assigned_family_count",
            "assigned_order_count",
            "assigned_no_call_count",
            "summary_csv",
            "assignments_csv",
            "manifest_json",
            "claim_boundary",
        ],
    )
    manifest = {
        "generated_by": "scripts/edna/build_paper1_pipeline_run_summary.py",
        "pipeline_root": str(args.pipeline_root),
        "output_csv": str(out),
        "rows": len(rows),
    }
    manifest_path = args.output_dir / "pipeline_run_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
