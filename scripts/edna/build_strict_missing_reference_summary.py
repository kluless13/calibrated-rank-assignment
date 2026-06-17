#!/usr/bin/env python3
"""Summarize strict missing-reference input packs and completed runs."""
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
RANKS = ("species", "genus", "family", "order")


def finite_metric(metrics: dict[str, Any], rank: str, key: str) -> float | str:
    value = metrics.get(rank, {}).get(key, "")
    if value in ("", None):
        return ""
    return 100.0 * float(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def result_dir_for(row: pd.Series, result_root: Path, model: str, seed: str) -> Path:
    return result_root / f"{row['name']}_{model}_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("data/phylo/paper1_strict_missing_reference_inputs/strict_missing_reference_manifest.csv"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/strict_missing_reference_cnn"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
    )
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--seed", default="1206")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    manifest = pd.read_csv(args.input_manifest)
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        run_dir = result_dir_for(item, args.result_root, args.model, args.seed)
        metrics_path = run_dir / "zero_shot_metrics" / "zero_shot_candidate_metrics.json"
        run_manifest = run_dir / "run_manifest.json"
        status = "complete" if metrics_path.exists() else "pending"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text()).get("metrics", {})
        row: dict[str, Any] = {
            "name": item["name"],
            "split": item["split"],
            "hide_rank": item["hide_rank"],
            "model": args.model,
            "seed": args.seed,
            "status": status,
            "hidden_candidate_species": item["hidden_candidate_species"],
            "kept_candidate_species": item["kept_candidate_species"],
            "kept_train_species": item["kept_train_species"],
            "query_rows": item["query_rows"],
            "query_species": item["query_species"],
            "run_dir": str(run_dir),
            "metrics_json": str(metrics_path) if metrics_path.exists() else "",
            "run_manifest": str(run_manifest) if run_manifest.exists() else "",
        }
        for rank in RANKS:
            row[f"{rank}_top1_pct"] = finite_metric(metrics, rank, "top1") if metrics else ""
            row[f"{rank}_top5_pct"] = finite_metric(metrics, rank, "top5") if metrics else ""
            row[f"{rank}_top10_pct"] = finite_metric(metrics, rank, "top10") if metrics else ""
        rows.append(row)

    out = args.output_dir / "strict_missing_reference_summary.csv"
    fields = [
        "name",
        "split",
        "hide_rank",
        "model",
        "seed",
        "status",
        "hidden_candidate_species",
        "kept_candidate_species",
        "kept_train_species",
        "query_rows",
        "query_species",
        "species_top1_pct",
        "species_top5_pct",
        "species_top10_pct",
        "genus_top1_pct",
        "genus_top5_pct",
        "genus_top10_pct",
        "family_top1_pct",
        "family_top5_pct",
        "family_top10_pct",
        "order_top1_pct",
        "order_top5_pct",
        "order_top10_pct",
        "run_dir",
        "metrics_json",
        "run_manifest",
    ]
    write_csv(out, rows, fields)
    manifest_json = {
        "generated_by": "scripts/edna/build_strict_missing_reference_summary.py",
        "input_manifest": str(args.input_manifest),
        "result_root": str(args.result_root),
        "output_csv": str(out),
        "rows": len(rows),
        "complete_rows": sum(1 for row in rows if row["status"] == "complete"),
        "notes": [
            "Pending rows mean the strict input pack exists but the retrained/pruned run has not completed.",
            "Metrics are forced top-k over the pruned candidate tree; use with rank/no-call tables before making claims.",
        ],
    }
    manifest_path = args.output_dir / "strict_missing_reference_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest_json, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {out} with {len(rows)} rows")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
