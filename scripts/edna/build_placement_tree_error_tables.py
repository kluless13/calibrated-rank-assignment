#!/usr/bin/env python3
"""Build matched placement tree-error tables for Paper 1.

These tables are Fernando-adjacent but not exact Fernando PCP. They ask whether
the placement reaches the nearest available reference location in the full tree
and how much excess tree distance remains otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
RANKS = ("species", "genus", "family", "order")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()


def load_rows(source_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ["placement_rank_diagnostics_per_query.csv", "apples_like_distance_placement_per_query.csv"]:
        path = source_dir / name
        df = read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_table"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def build_summary(df: pd.DataFrame, tolerance: float) -> list[dict[str, object]]:
    rows = []
    if df.empty:
        return rows
    for (split, method), sub in df.groupby(["split", "method"], dropna=False):
        excess = finite_series(sub["placement_excess_tree_distance_vs_nearest_reference"])
        placed = finite_series(sub["placement_min_tree_distance_to_placed_clade"])
        nearest = finite_series(sub["nearest_reference_tree_distance"])
        nearest_match = excess.le(tolerance) if not excess.empty else pd.Series(dtype=bool)
        row: dict[str, object] = {
            "split": split,
            "method": method,
            "n_queries": int(len(sub)),
            "n_with_finite_excess": int(len(excess)),
            "nearest_reference_match_pct": float(100.0 * nearest_match.mean()) if len(nearest_match) else "",
            "nearest_reference_tree_distance_median": float(nearest.median()) if len(nearest) else "",
            "placement_tree_distance_median": float(placed.median()) if len(placed) else "",
            "placement_excess_tree_distance_median": float(excess.median()) if len(excess) else "",
            "placement_excess_tree_distance_mean": float(excess.mean()) if len(excess) else "",
            "placement_excess_tree_distance_p90": float(excess.quantile(0.90)) if len(excess) else "",
            "source_tables": ";".join(sorted(set(sub["source_table"].astype(str)))),
            "claim_boundary": "nearest-reference tree-error diagnostic; not exact Fernando PCP",
        }
        for rank in RANKS:
            col = f"{rank}_in_placed_clade"
            row[f"{rank}_containment_pct"] = float(100.0 * sub[col].astype(bool).mean()) if col in sub else ""
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=SOURCE)
    parser.add_argument("--tree-distance-tolerance", type=float, default=1e-9)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    logger.log(f"Loading placement per-query rows from {args.source_dir}")
    df = load_rows(args.source_dir)
    logger.log(f"Loaded {len(df)} placement rows")
    summary_rows = build_summary(df, args.tree_distance_tolerance)

    summary_path = args.output_dir / "placement_tree_error_summary.csv"
    logger.log(f"Writing {summary_path}")
    write_csv(
        summary_path,
        summary_rows,
        [
            "split",
            "method",
            "n_queries",
            "n_with_finite_excess",
            "nearest_reference_match_pct",
            "nearest_reference_tree_distance_median",
            "placement_tree_distance_median",
            "placement_excess_tree_distance_median",
            "placement_excess_tree_distance_mean",
            "placement_excess_tree_distance_p90",
            "species_containment_pct",
            "genus_containment_pct",
            "family_containment_pct",
            "order_containment_pct",
            "source_tables",
            "claim_boundary",
        ],
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_placement_tree_error_tables.py",
        "input_rows": int(len(df)),
        "summary_rows": len(summary_rows),
        "tree_distance_tolerance": args.tree_distance_tolerance,
        "outputs": {"summary": str(summary_path)},
        "claim_boundary": "Nearest-reference tree-error diagnostic, not exact Fernando PCP.",
    }
    manifest_path = args.output_dir / "placement_tree_error_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
