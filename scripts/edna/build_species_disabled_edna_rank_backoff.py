#!/usr/bin/env python3
"""Build species-disabled rank-backoff summaries for eDNA posterior outputs.

The full candidate posterior can learn high species probabilities, but the
current species thresholds do not transfer on held-out Global_eDNA groups. This
script keeps the useful higher-rank posterior evidence and recomputes backoff
with species disabled:

genus -> family -> order -> no-call

It consumes the selected candidate posterior predictions and operating-point
thresholds produced by `run_eco_phylo_candidate_posterior.py`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "eco_phylo_posterior"
    / "candidate_level_sequence_tree_evidence_full"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR
RANKS = ("genus", "family", "order")
TARGET_ACCURACIES = (50.0, 60.0, 70.0, 80.0, 90.0, 95.0)
GROUP_COLS = ("sample_id", "query_processid")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def threshold_map(operating: pd.DataFrame, target: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for rank in RANKS:
        match = operating[
            (operating["rank"] == rank)
            & (operating["target_accuracy_pct"] == target)
            & (operating["status"] == "available")
        ]
        if not match.empty:
            thresholds[rank] = float(match.iloc[0]["threshold"])
    return thresholds


def selected_for_rank(selected: pd.DataFrame, rank: str, split: str, threshold: float) -> pd.DataFrame:
    work = selected[
        (selected["rank"] == rank)
        & (selected["calibration_split"] == split)
        & (selected["posterior_probability"] >= threshold)
    ].copy()
    if work.empty:
        return work
    work["assigned_rank"] = rank
    work["assigned_value"] = work[f"candidate_{rank}"]
    work["assigned_correct"] = work[f"{rank}_candidate_correct"].astype(bool)
    return work


def build_assignments(selected: pd.DataFrame, operating: pd.DataFrame, split: str, target: float) -> pd.DataFrame:
    thresholds = threshold_map(operating, target)
    frames: list[pd.DataFrame] = []
    for rank in RANKS:
        if rank in thresholds:
            frames.append(selected_for_rank(selected, rank, split, thresholds[rank]))
    if not frames:
        return pd.DataFrame()
    assigned = pd.concat(frames, ignore_index=True)
    if assigned.empty:
        return assigned
    priority = {"genus": 0, "family": 1, "order": 2}
    assigned["rank_priority"] = assigned["assigned_rank"].map(priority)
    assigned = assigned.sort_values(
        ["sample_id", "query_processid", "rank_priority", "posterior_probability"],
        ascending=[True, True, True, False],
    )
    return assigned.groupby(list(GROUP_COLS), sort=False, as_index=False).head(1).copy()


def summarize(selected: pd.DataFrame, operating: pd.DataFrame, logger: ProgressLogger) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    assignment_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for split in ("calibration", "evaluation"):
        base = selected[
            (selected["rank"] == "order") & (selected["calibration_split"] == split)
        ][list(GROUP_COLS)].drop_duplicates()
        n_query = len(base)
        logger.log(f"Building species-disabled backoff split={split} n_query={n_query:,}")
        for target in TARGET_ACCURACIES:
            final = build_assignments(selected, operating, split, target)
            if not final.empty:
                final["target_accuracy_pct"] = target
                final["split"] = split
                assignment_frames.append(final)
            counts = final["assigned_rank"].value_counts().to_dict() if not final.empty else {}
            accuracy = (
                100.0 * final["assigned_correct"].sum() / len(final)
                if len(final)
                else np.nan
            )
            rows.append(
                {
                    "split": split,
                    "target_accuracy_pct": target,
                    "status": "available" if len(final) else "no_threshold",
                    "n_query": int(n_query),
                    "n_assigned": int(len(final)),
                    "assignment_rate_pct": 100.0 * len(final) / n_query if n_query else 0.0,
                    "assigned_accuracy_pct": accuracy,
                    "species_assignments": 0,
                    "genus_assignments": int(counts.get("genus", 0)),
                    "family_assignments": int(counts.get("family", 0)),
                    "order_assignments": int(counts.get("order", 0)),
                    "no_call_rate_pct": 100.0 * (n_query - len(final)) / n_query if n_query else 0.0,
                }
            )
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    return assignments, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = args.input_dir / "eco_phylo_candidate_posterior_selected_predictions.csv.gz"
    operating_path = args.input_dir / "eco_phylo_candidate_posterior_operating_points.csv"
    logger.log(f"Reading selected posterior predictions from {rel(selected_path)}")
    selected = pd.read_csv(selected_path)
    logger.log(f"Reading operating points from {rel(operating_path)}")
    operating = pd.read_csv(operating_path)

    assignments, rows = summarize(selected, operating, logger)
    summary_path = args.output_dir / "eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv"
    assignments_path = args.output_dir / "eco_phylo_candidate_posterior_species_disabled_rank_backoff_assignments.csv.gz"

    logger.log(f"Writing species-disabled summary to {rel(summary_path)}")
    write_csv(
        summary_path,
        rows,
        [
            "split",
            "target_accuracy_pct",
            "status",
            "n_query",
            "n_assigned",
            "assignment_rate_pct",
            "assigned_accuracy_pct",
            "species_assignments",
            "genus_assignments",
            "family_assignments",
            "order_assignments",
            "no_call_rate_pct",
        ],
    )
    logger.log(f"Writing species-disabled assignments to {rel(assignments_path)}")
    with gzip.open(assignments_path, "wt", newline="") as handle:
        assignments.to_csv(handle, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": rel(Path(__file__)),
        "input_dir": rel(args.input_dir),
        "output_dir": rel(args.output_dir),
        "selected_predictions": rel(selected_path),
        "operating_points": rel(operating_path),
        "summary_csv": rel(summary_path),
        "assignments_csv": rel(assignments_path),
        "rank_order": RANKS,
        "target_accuracies_pct": TARGET_ACCURACIES,
        "notes": [
            "Species rank is intentionally disabled because full sequence+tree posterior species thresholds do not transfer.",
            "Backoff applies genus, then family, then order thresholds learned on calibration groups.",
        ],
    }
    manifest_path = args.output_dir / "eco_phylo_candidate_posterior_species_disabled_rank_backoff_manifest.json"
    logger.log(f"Writing manifest to {rel(manifest_path)}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
