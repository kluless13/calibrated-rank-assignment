#!/usr/bin/env python3
"""Collect raw-sequence production-v1 timing and assignment summaries."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
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


def read_first(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if len(frame) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/raw_sequence_production_v1"),
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
    for split_dir in sorted(args.raw_root.iterdir() if args.raw_root.exists() else []):
        if not split_dir.is_dir():
            continue
        timing = read_first(split_dir / "raw_sequence_production_v1_timing_summary.csv")
        production_paths = sorted(split_dir.glob("production_v1/*/production_v1_summary.csv"))
        production = read_first(production_paths[0]) if production_paths else {}
        if not timing and not production:
            continue
        row = {
            "split": split_dir.name,
            "n_queries": timing.get("n_queries") or production.get("n_queries"),
            "total_seconds": timing.get("total_seconds"),
            "total_ms_per_query": timing.get("total_ms_per_query"),
            "embedding_export_seconds": timing.get("embedding_export_seconds"),
            "vector_retrieval_seconds": timing.get("vector_retrieval_seconds"),
            "rerank_seconds": timing.get("rerank_seconds"),
            "pipeline_seconds": timing.get("pipeline_seconds"),
            "production_packaging_seconds": timing.get("production_packaging_seconds"),
            "coverage": production.get("coverage") or timing.get("coverage"),
            "assigned_precision": production.get("assigned_precision") or timing.get("assigned_precision"),
            "false_species_call_rate_all_queries": production.get("false_species_call_rate_all_queries")
            if "false_species_call_rate_all_queries" in production
            else timing.get("false_species_call_rate_all_queries"),
            "assigned_species_count": production.get("assigned_species_count"),
            "assigned_genus_count": production.get("assigned_genus_count"),
            "assigned_family_count": production.get("assigned_family_count"),
            "assigned_order_count": production.get("assigned_order_count"),
            "assigned_no_call_count": production.get("assigned_no_call_count"),
            "timing_summary": str(split_dir / "raw_sequence_production_v1_timing_summary.csv"),
            "production_summary": str(production_paths[0]) if production_paths else "",
        }
        rows.append(row)

    output = args.output_dir / "raw_sequence_production_v1_summary.csv"
    fieldnames = [
        "split",
        "n_queries",
        "total_seconds",
        "total_ms_per_query",
        "embedding_export_seconds",
        "vector_retrieval_seconds",
        "rerank_seconds",
        "pipeline_seconds",
        "production_packaging_seconds",
        "coverage",
        "assigned_precision",
        "false_species_call_rate_all_queries",
        "assigned_species_count",
        "assigned_genus_count",
        "assigned_family_count",
        "assigned_order_count",
        "assigned_no_call_count",
        "timing_summary",
        "production_summary",
    ]
    logger.log(f"Writing {len(rows)} rows to {output}")
    write_csv(output, rows, fieldnames)
    manifest = {
        "generated_by": "scripts/edna/build_paper1_raw_sequence_production_summary.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(args.raw_root),
        "output_csv": str(output),
        "rows": len(rows),
    }
    manifest_path = args.output_dir / "raw_sequence_production_v1_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
