#!/usr/bin/env python3
"""Convert strict missing-reference metrics into rank-backoff source rows."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")
HIDDEN_RANKS = {
    "species": ("species",),
    "genus": ("species", "genus"),
    "family": ("species", "genus", "family"),
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def pct(row: pd.Series, rank: str, k: int) -> float:
    value = row.get(f"{rank}_top{k}_pct", 0.0)
    return float(value) if clean(value) else 0.0


def deepest_supported_rank(row: pd.Series, k: int, min_pct: float) -> str:
    for rank in RANKS:
        if pct(row, rank, k) >= min_pct:
            return rank
    return "no_call"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rows(frame: pd.DataFrame, top10_support_min_pct: float, top1_support_min_pct: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        hide_rank = clean(item.get("hide_rank"))
        hidden = HIDDEN_RANKS.get(hide_rank, ())
        hidden_top10_max = max((pct(item, rank, 10) for rank in hidden), default=0.0)
        hidden_top1_max = max((pct(item, rank, 1) for rank in hidden), default=0.0)
        top10_rank = deepest_supported_rank(item, 10, top10_support_min_pct)
        top1_rank = deepest_supported_rank(item, 1, top1_support_min_pct)
        rows.append(
            {
                "name": item.get("name", ""),
                "split": item.get("split", ""),
                "hide_rank": hide_rank,
                "model": item.get("model", ""),
                "seed": item.get("seed", ""),
                "query_rows": item.get("query_rows", ""),
                "query_species": item.get("query_species", ""),
                "hidden_candidate_species": item.get("hidden_candidate_species", ""),
                "kept_candidate_species": item.get("kept_candidate_species", ""),
                "kept_train_species": item.get("kept_train_species", ""),
                "hidden_rank_set": "|".join(hidden),
                "hidden_rank_top1_max_pct": hidden_top1_max,
                "hidden_rank_top10_max_pct": hidden_top10_max,
                "hidden_rank_zero_top10": bool(hidden_top10_max == 0.0),
                "deepest_supported_rank_top1": top1_rank,
                "deepest_supported_rank_top10": top10_rank,
                "top1_support_min_pct": top1_support_min_pct,
                "top10_support_min_pct": top10_support_min_pct,
                "species_top10_pct": pct(item, "species", 10),
                "genus_top10_pct": pct(item, "genus", 10),
                "family_top10_pct": pct(item, "family", 10),
                "order_top10_pct": pct(item, "order", 10),
                "rank_backoff_interpretation": (
                    f"Do not force {hide_rank}-or-deeper calls; "
                    f"top-10 evidence supports {top10_rank} at >= {top10_support_min_pct:g}%."
                    if top10_rank != "no_call"
                    else f"Do not force {hide_rank}-or-deeper calls; no tested rank clears top-10 support threshold."
                ),
                "claim_boundary": (
                    "Forced retrieval over a pruned candidate tree; use as stress evidence for rank/no-call policy, "
                    "not as prospective calibrated deployment performance."
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-summary",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables/strict_missing_reference_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
    )
    parser.add_argument("--top10-support-min-pct", type=float, default=10.0)
    parser.add_argument("--top1-support-min-pct", type=float, default=5.0)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    frame = pd.read_csv(args.strict_summary)
    frame = frame[frame["status"].astype(str) == "complete"].copy()
    rows = build_rows(frame, args.top10_support_min_pct, args.top1_support_min_pct)
    out = args.output_dir / "strict_rank_backoff_summary.csv"
    fields = [
        "name",
        "split",
        "hide_rank",
        "model",
        "seed",
        "query_rows",
        "query_species",
        "hidden_candidate_species",
        "kept_candidate_species",
        "kept_train_species",
        "hidden_rank_set",
        "hidden_rank_top1_max_pct",
        "hidden_rank_top10_max_pct",
        "hidden_rank_zero_top10",
        "deepest_supported_rank_top1",
        "deepest_supported_rank_top10",
        "top1_support_min_pct",
        "top10_support_min_pct",
        "species_top10_pct",
        "genus_top10_pct",
        "family_top10_pct",
        "order_top10_pct",
        "rank_backoff_interpretation",
        "claim_boundary",
    ]
    write_csv(out, rows, fields)
    manifest = {
        "generated_by": "scripts/edna/build_strict_rank_backoff_summary.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "strict_summary": str(args.strict_summary),
        "output_csv": str(out),
        "rows": len(rows),
        "top1_support_min_pct": args.top1_support_min_pct,
        "top10_support_min_pct": args.top10_support_min_pct,
        "notes": [
            "The deepest supported rank is a descriptive stress-test summary, not a calibrated deployment threshold.",
            "Hidden-rank zero values are expected when the true rank is absent from the candidate set.",
        ],
    }
    manifest_path = args.output_dir / "strict_rank_backoff_summary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {out} with {len(rows)} rows")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
