#!/usr/bin/env python3
"""Build slide-ready MarkerMirror tables and outline from manuscript assets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
ASSETS = PAPER1 / "manuscript_assets" / "marker_mirror"
FIGURES = ASSETS / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSETS)
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--output-dir", type=Path, default=ASSETS / "slide_tables")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pct(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}%"


def write_table_pair(df: pd.DataFrame, output_dir: Path, stem: str, logger: ProgressLogger) -> dict[str, str]:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    logger.log(f"wrote {rel(csv_path)} rows={len(df)}")
    logger.log(f"wrote {rel(md_path)} rows={len(df)}")
    return {"csv": rel(csv_path), "markdown": rel(md_path)}


def candidate_support(asset_dir: Path) -> pd.DataFrame:
    df = read_csv(asset_dir / "marker_mirror_candidate_support_table.csv")
    out = pd.DataFrame(
        {
            "Candidate source": df["candidate_source"],
            "Species": df["species_pct"].map(pct),
            "Genus": df["genus_pct"].map(pct),
            "Family": df["family_pct"].map(pct),
            "Order": df["order_pct"].map(pct),
            "Slide note": [
                "Learned cross-marker signal alone; useful but incomplete.",
                "Classical same-marker local alignment; cannot recover absent species.",
                "Best claim-facing union candidate support.",
                "Classical same-marker global alignment; independent check.",
                "Independent union confirmation.",
            ],
        }
    )
    return out


def order_policy(asset_dir: Path) -> pd.DataFrame:
    df = read_csv(asset_dir / "marker_mirror_order_policy_table.csv")
    mode_labels = {
        "stable_order": "Stable default",
        "high_coverage_order": "High-coverage diagnostic",
    }
    out = pd.DataFrame(
        {
            "Mode": df["mode"].map(mode_labels),
            "Calls": df.apply(lambda row: f"{int(row.n_assigned):,}/{int(row.n_queries):,}", axis=1),
            "Coverage": df["coverage_pct"].map(pct),
            "Precision": df["diagnostic_precision_pct"].map(pct),
            "Nested held-out coverage": df["nested_mean_heldout_coverage_pct"].map(pct),
            "Nested held-out precision": df["nested_mean_heldout_precision_pct"].map(pct),
            "Use": [
                "Default conservative order/no-call behavior.",
                "Explicit research mode; order/no-call only.",
            ],
        }
    )
    return out


def rank_boundary(asset_dir: Path) -> pd.DataFrame:
    df = read_csv(asset_dir / "marker_mirror_rank_boundary_table.csv")
    out = pd.DataFrame(
        {
            "Rank": df["rank"].str.title(),
            "Best single-label coverage": df["single_label_best_coverage_pct"].map(pct),
            "Best precision": df["single_label_best_precision_pct"].map(pct),
            "Target met rate": df["single_label_target_met_rate_pct"].map(pct),
            "Set-valued fallback": df.apply(
                lambda row: f"{row.best_set_full_query_truth_coverage_pct:.1f}% covered, mean set size {row.best_set_mean_size:.1f}",
                axis=1,
            ),
            "Decision": df["manuscript_interpretation"],
        }
    )
    return out


def runtime(asset_dir: Path) -> pd.DataFrame:
    df = read_csv(asset_dir / "marker_mirror_runtime_table.csv")
    out = pd.DataFrame(
        {
            "Stage": df["stage"],
            "Seconds": df["seconds"].map(lambda value: f"{float(value):.1f}"),
            "Runtime share": df["percent_runtime"].map(pct),
            "Slide note": [
                "Learned cross-marker stage is fast relative to alignment checks.",
                "Dominant runtime component.",
                "Faster independent same-marker check.",
                "Decision policy is negligible after evidence is built.",
            ],
        }
    )
    return out


def build_outline(
    output_dir: Path,
    figure_dir: Path,
    support: pd.DataFrame,
    policy: pd.DataFrame,
    ranks: pd.DataFrame,
    runtime_df: pd.DataFrame,
    logger: ProgressLogger,
) -> Path:
    blast_union = support[support["Candidate source"] == "MarkerMirror + BLASTN union"].iloc[0]
    stable = policy[policy["Mode"] == "Stable default"].iloc[0]
    high = policy[policy["Mode"] == "High-coverage diagnostic"].iloc[0]
    total_seconds = sum(float(value) for value in runtime_df["Seconds"])
    outline = f"""# MarkerMirror Slide Package

## Slide 1: What The Tool Does

**Claim:** MarkerMirror is an evidence compiler for short 12S fragments, not a forced species classifier.

Pipeline:

```text
12S FASTA/CSV
  -> learned 12S->16S candidates
  -> BLASTN 12S candidates
  -> VSEARCH 12S candidates
  -> shared evidence table
  -> order/no-call with reason code
```

Proof object: pipeline schematic or
`{rel(figure_dir / "marker_mirror_runtime_breakdown.png")}`.

Speaker note: species/genus/family remain disabled because current evidence does not transfer cleanly at target-99.

## Slide 2: Candidate Support

**Claim:** The union of learned cross-marker evidence and classical same-marker evidence recovers high-rank support.

Headline row: MarkerMirror + BLASTN union top-50 support is species {blast_union.Species}, genus {blast_union.Genus}, family {blast_union.Family}, order {blast_union.Order}.

Proof object: `{rel(figure_dir / "marker_mirror_candidate_support_bars.png")}`.

Speaker note: species is low by split design because held-out query species are absent from the same-marker 12S reference.

## Slide 3: Order/No-Call Modes

**Claim:** The pipeline can trade coverage for precision while staying honest about uncertainty.

Stable default: {stable.Calls} calls, {stable.Coverage} coverage, {stable.Precision} precision.

High-coverage diagnostic: {high.Calls} calls, {high.Coverage} full-table coverage, {high["Nested held-out coverage"]} mean held-out coverage, {high["Nested held-out precision"]} mean held-out precision.

Proof object: `{rel(figure_dir / "marker_mirror_order_policy_tradeoff.png")}`.

Speaker note: high-coverage mode is explicit order/no-call research mode, not default species identification.

## Slide 4: Why Family/Genus Are Disabled

**Claim:** Family/genus failure is evidence-level, not merely a thresholding problem.

Proof object: `{rel(figure_dir / "marker_mirror_rank_boundary.png")}`.

Speaker note: single-label repairs and set-valued outputs both failed to unlock useful target-99 family/genus output.

## Slide 5: Runtime And Implementation

**Claim:** The full wrapper is executable and the decision layer is not the bottleneck.

Runtime: {total_seconds:.1f} seconds for 3,566 queries on the Vast RTX host.

Proof object: `{rel(figure_dir / "marker_mirror_runtime_breakdown.png")}`.

Speaker note: BLASTN dominates runtime; MarkerMirror candidate generation took about 15 seconds in the full run.

## Claim Boundary

Safe phrasing:

> MarkerMirror combines learned cross-marker retrieval with BLASTN/VSEARCH same-marker evidence to make precise order/no-call decisions for short 12S fragments under missing-reference conditions.

Avoid:

- species identification is solved;
- family/genus calls are enabled;
- MarkerMirror replaces BLASTN/VSEARCH;
- current results are field-eDNA validation.
"""
    path = output_dir / "marker_mirror_slide_package_outline.md"
    path.write_text(outline, encoding="utf-8")
    logger.log(f"wrote {rel(path)}")
    return path


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).stem)
    logger.log(f"asset_dir={rel(args.asset_dir)}")
    logger.log(f"figure_dir={rel(args.figure_dir)}")
    logger.log(f"output_dir={rel(args.output_dir)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    support = candidate_support(args.asset_dir)
    policy = order_policy(args.asset_dir)
    ranks = rank_boundary(args.asset_dir)
    runtime_df = runtime(args.asset_dir)

    outputs = {
        "candidate_support": write_table_pair(support, args.output_dir, "marker_mirror_candidate_support_slide_table", logger),
        "order_policy": write_table_pair(policy, args.output_dir, "marker_mirror_order_policy_slide_table", logger),
        "rank_boundary": write_table_pair(ranks, args.output_dir, "marker_mirror_rank_boundary_slide_table", logger),
        "runtime": write_table_pair(runtime_df, args.output_dir, "marker_mirror_runtime_slide_table", logger),
    }
    outline_path = build_outline(args.output_dir, args.figure_dir, support, policy, ranks, runtime_df, logger)
    outputs["outline"] = rel(outline_path)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "asset_dir": rel(args.asset_dir),
            "figure_dir": rel(args.figure_dir),
        },
        "outputs": outputs,
        "claim_boundary": "Slide-ready tables and outline only. They repackage existing MarkerMirror metrics.",
    }
    manifest_path = args.output_dir / "marker_mirror_slide_tables_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"wrote {rel(manifest_path)}")
    logger.done(Path(__file__).stem)


if __name__ == "__main__":
    main()
