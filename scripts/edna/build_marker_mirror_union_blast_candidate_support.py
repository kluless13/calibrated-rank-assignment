#!/usr/bin/env python3
"""Summarize MarkerMirror + BLASTN same-marker union candidate support."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--marker-mirror-support",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_union_candidate_support_per_query.csv",
    )
    parser.add_argument(
        "--blast-support",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_same_marker_blast_support_per_query.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def support_summary(frame: pd.DataFrame, source: str, prefix: str, top_k: int, total: int) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate_source": source, "top_k": top_k, "n_queries": total}
    for rank in RANKS:
        row[f"{rank}_hit_pct"] = float(frame[f"{prefix}_{rank}_hit"].map(truthy).mean() * 100.0)
    return row


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mm = pd.read_csv(args.marker_mirror_support)
    blast = pd.read_csv(args.blast_support)
    top_k = int(args.top_k)
    keep = ["query_id", *[f"top{top_k}_{rank}_hit" for rank in RANKS], "candidate_count"]
    blast = blast[[column for column in keep if column in blast.columns]].copy()
    merged = mm.merge(blast, on="query_id", how="left", suffixes=("", "_blast"))
    for rank in RANKS:
        merged[f"blast_top{top_k}_{rank}_hit"] = merged[f"top{top_k}_{rank}_hit"].map(truthy)
        merged[f"marker_mirror_top{top_k}_{rank}_hit"] = merged[f"marker_mirror_top50_{rank}_hit"].map(truthy)
        merged[f"union_blast_top{top_k}_{rank}_hit"] = (
            merged[f"marker_mirror_top{top_k}_{rank}_hit"] | merged[f"blast_top{top_k}_{rank}_hit"]
        )

    total = len(merged)
    summary = pd.DataFrame(
        [
            support_summary(merged, "marker_mirror_12s_to_16s", f"marker_mirror_top{top_k}", top_k, total),
            support_summary(merged, "same_marker_12s_blastn_local", f"blast_top{top_k}", top_k, total),
            support_summary(merged, "union_marker_mirror_plus_blastn", f"union_blast_top{top_k}", top_k, total),
        ]
    )
    per_query_path = args.output_dir / "marker_mirror_union_blast_candidate_support_per_query.csv"
    summary_path = args.output_dir / "marker_mirror_union_blast_candidate_support_summary.csv"
    manifest_path = args.output_dir / "marker_mirror_union_blast_candidate_support_manifest.json"
    merged.to_csv(per_query_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "marker_mirror_support": rel(args.marker_mirror_support),
            "blast_support": rel(args.blast_support),
        },
        "outputs": {
            "summary": rel(summary_path),
            "per_query": rel(per_query_path),
        },
        "top_k": top_k,
        "n_queries": int(total),
        "claim_boundary": "Union support summary over MarkerMirror 12S->16S and BLASTN local same-marker candidates.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote {rel(summary_path)}")
    logger.log(f"Wrote {rel(per_query_path)}")
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
