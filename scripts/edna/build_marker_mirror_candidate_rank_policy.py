#!/usr/bin/env python3
"""Build a simple rank/no-call diagnostic from MarkerMirror candidates.

This is not the final eDNA assignment policy.  It answers a narrower question:
if MarkerMirror is used as a candidate generator, can validation thresholds on
candidate score decide when to emit species/genus/family/order versus no-call?

The script fits score thresholds on validation rows, then applies those locked
thresholds to train/validation/test rows and writes auditable source tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import Logger

RANK_ORDER = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-rankings",
        type=Path,
        required=True,
        help="Per-query candidate table from export_marker_mirror_candidate_rankings.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="0.90,0.95")
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--top-rank", type=int, default=1, help="Use the rank-N candidate as the predicted taxon.")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def load_top_candidate(path: Path, top_rank: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    top = frame[frame["candidate_rank"] == top_rank].copy()
    if top.empty:
        raise ValueError(f"No candidate_rank={top_rank} rows in {path}")
    return top


def precision_at_threshold(rows: pd.DataFrame, rank: str, threshold: float) -> tuple[float, int]:
    selected = rows[rows["score"] >= threshold]
    n = len(selected)
    if n == 0:
        return float("nan"), 0
    return float(selected[f"match_{rank}"].mean()), n


def fit_threshold(rows: pd.DataFrame, rank: str, target: float) -> dict[str, Any]:
    if rows.empty:
        return {"threshold": None, "fit_precision": None, "fit_coverage": 0.0, "fit_n": 0}
    thresholds = np.unique(rows["score"].to_numpy(dtype=float))
    thresholds = np.sort(thresholds)[::-1]
    best = None
    for threshold in thresholds:
        precision, n = precision_at_threshold(rows, rank, float(threshold))
        if n > 0 and np.isfinite(precision) and precision >= target:
            best = (float(threshold), precision, n)
    if best is None:
        return {"threshold": None, "fit_precision": None, "fit_coverage": 0.0, "fit_n": 0}
    threshold, precision, n = best
    return {
        "threshold": threshold,
        "fit_precision": precision,
        "fit_coverage": 100.0 * n / max(len(rows), 1),
        "fit_n": int(n),
    }


def apply_policy(rows: pd.DataFrame, thresholds: dict[str, float | None], target: float) -> pd.DataFrame:
    out = []
    for _, row in rows.iterrows():
        assigned_rank = "no_call"
        correct = False
        assigned_taxon = ""
        for rank in RANK_ORDER:
            threshold = thresholds.get(rank)
            if threshold is not None and float(row["score"]) >= float(threshold):
                assigned_rank = rank
                correct = bool(row[f"match_{rank}"])
                assigned_taxon = str(row[f"candidate_{rank}"])
                break
        record = row.to_dict()
        record.update(
            {
                "target_precision": target,
                "assigned_rank": assigned_rank,
                "assigned_taxon": assigned_taxon,
                "assigned_correct": correct,
            }
        )
        out.append(record)
    return pd.DataFrame(out)


def summarize(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["target_precision", "model", "direction", "split"]
    for key, group in assignments.groupby(keys, dropna=False):
        target, model, direction, split = key
        assigned = group[group["assigned_rank"] != "no_call"]
        n_query = int(group["query_id"].nunique())
        assigned_n = int(len(assigned))
        rank_counts = assigned["assigned_rank"].value_counts().to_dict()
        false_species = assigned[(assigned["assigned_rank"] == "species") & (~assigned["match_species"].astype(bool))]
        rows.append(
            {
                "target_precision": target,
                "model": model,
                "direction": direction,
                "split": split,
                "n_query": n_query,
                "assigned_n": assigned_n,
                "coverage_pct": 100.0 * assigned_n / max(n_query, 1),
                "assigned_precision_pct": 100.0 * float(assigned["assigned_correct"].mean()) if assigned_n else float("nan"),
                "false_species_call_rate_pct": 100.0 * len(false_species) / max(n_query, 1),
                "species_calls": int(rank_counts.get("species", 0)),
                "genus_calls": int(rank_counts.get("genus", 0)),
                "family_calls": int(rank_counts.get("family", 0)),
                "order_calls": int(rank_counts.get("order", 0)),
                "no_calls": int(n_query - assigned_n),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_rank_policy.log")
    logger.log(f"Arguments: {vars(args)}")
    targets = [float(item) for item in args.targets.split(",") if item.strip()]
    top = load_top_candidate(args.candidate_rankings, args.top_rank)
    logger.log(f"Loaded top-candidate rows={len(top)} queries={top['query_id'].nunique()}")

    threshold_rows = []
    assignment_frames = []
    for target in targets:
        for (model, direction), group in top.groupby(["model", "direction"], dropna=False):
            fit_rows = group[group["split"] == args.fit_split]
            thresholds: dict[str, float | None] = {}
            for rank in RANK_ORDER:
                fit = fit_threshold(fit_rows, rank, target)
                thresholds[rank] = fit["threshold"]
                threshold_rows.append(
                    {
                        "target_precision": target,
                        "model": model,
                        "direction": direction,
                        "fit_split": args.fit_split,
                        "rank": rank,
                        **fit,
                    }
                )
            logger.log(f"target={target:.2f} model={model} direction={direction} thresholds={thresholds}")
            assignment_frames.append(apply_policy(group, thresholds, target))

    thresholds = pd.DataFrame(threshold_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    summary = summarize(assignments) if not assignments.empty else pd.DataFrame()

    thresholds.to_csv(args.output_dir / "marker_mirror_rank_policy_thresholds.csv", index=False)
    assignments.to_csv(args.output_dir / "marker_mirror_rank_policy_assignments.csv.gz", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_rank_policy_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rankings": str(args.candidate_rankings),
        "top_rank": args.top_rank,
        "fit_split": args.fit_split,
        "targets": targets,
        "rows": {"thresholds": len(thresholds), "assignments": len(assignments), "summary": len(summary)},
        "claim_boundary": "MarkerMirror-only rank/no-call diagnostic; not the final integrated pipeline policy.",
    }
    (args.output_dir / "marker_mirror_rank_policy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
