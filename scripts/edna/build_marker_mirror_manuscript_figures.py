#!/usr/bin/env python3
"""Render MarkerMirror manuscript figure drafts from manuscript asset tables."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
ASSETS = PAPER1 / "manuscript_assets" / "marker_mirror"
RANKS = ["species", "genus", "family", "order"]
RANK_LABELS = {"species": "Species", "genus": "Genus", "family": "Family", "order": "Order"}
COLORS = {
    "species": "#4C78A8",
    "genus": "#F58518",
    "family": "#54A24B",
    "order": "#B279A2",
    "precision": "#2F4B7C",
    "coverage": "#D95F02",
    "neutral": "#6B7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=ASSETS)
    parser.add_argument("--output-dir", type=Path, default=ASSETS / "figures")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def setup() -> None:
    (ROOT / "results" / ".matplotlib").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str, logger: ProgressLogger) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.tight_layout()
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
        logger.log(f"wrote {rel(path)}")
    plt.close(fig)
    return paths


def annotate_bars(ax: plt.Axes, values: Iterable[float], fmt: str = "{:.1f}") -> None:
    for patch, value in zip(ax.patches, values):
        if not np.isfinite(float(value)):
            continue
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height + 1.0,
            fmt.format(float(value)),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90 if len(ax.patches) > 12 else 0,
        )


def candidate_support(asset_dir: Path, output_dir: Path, logger: ProgressLogger) -> list[Path]:
    df = pd.read_csv(asset_dir / "marker_mirror_candidate_support_table.csv")
    short_labels = {
        "MarkerMirror 12S->16S only": "MarkerMirror\n12S->16S",
        "Same-marker 12S BLASTN": "BLASTN\n12S",
        "MarkerMirror + BLASTN union": "MarkerMirror +\nBLASTN union",
        "Same-marker 12S VSEARCH": "VSEARCH\n12S",
        "MarkerMirror + VSEARCH union": "MarkerMirror +\nVSEARCH union",
    }
    long = df.melt(
        id_vars=["candidate_source"],
        value_vars=[f"{rank}_pct" for rank in RANKS],
        var_name="rank",
        value_name="support_pct",
    )
    long["rank"] = long["rank"].str.replace("_pct", "", regex=False)
    source_order = df["candidate_source"].tolist()
    x = np.arange(len(source_order))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    values_for_labels: list[float] = []
    for idx, rank in enumerate(RANKS):
        values = (
            long[long["rank"] == rank]
            .set_index("candidate_source")
            .loc[source_order, "support_pct"]
            .to_numpy()
        )
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            label=RANK_LABELS[rank],
            color=COLORS[rank],
        )
        values_for_labels.extend(values)
    ax.set_title("MarkerMirror 12S candidate support by evidence source")
    ax.set_ylabel("True taxon present in top-50 candidates (%)")
    ax.set_ylim(0, 112)
    ax.set_xticks(x)
    ax.set_xticklabels([short_labels.get(label, label) for label in source_order])
    ax.legend(ncol=4, loc="upper left")
    annotate_bars(ax, values_for_labels)
    return save(fig, output_dir, "marker_mirror_candidate_support_bars", logger)


def order_policy(asset_dir: Path, output_dir: Path, logger: ProgressLogger) -> list[Path]:
    df = pd.read_csv(asset_dir / "marker_mirror_order_policy_table.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(
        df["coverage_pct"],
        df["diagnostic_precision_pct"],
        s=[120 if mode == "stable_order" else 160 for mode in df["mode"]],
        c=[COLORS["coverage"], COLORS["precision"]],
        edgecolors="black",
        linewidths=0.8,
    )
    for row in df.itertuples(index=False):
        label = row.mode.replace("_", " ")
        ax.annotate(
            f"{label}\n{row.coverage_pct:.1f}% cov, {row.diagnostic_precision_pct:.1f}% prec",
            (row.coverage_pct, row.diagnostic_precision_pct),
            xytext=(8, 8 if row.mode == "stable_order" else -30),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 0.8},
        )
    ax.axhline(99.0, color="#111827", linestyle="--", linewidth=1.0, label="99% precision target")
    ax.set_title("Order/no-call coverage versus precision")
    ax.set_xlabel("Queries assigned an order (%)")
    ax.set_ylabel("Diagnostic precision on assigned rows (%)")
    ax.set_xlim(0, 78)
    ax.set_ylim(98.7, 100.05)
    ax.legend(loc="lower right")
    return save(fig, output_dir, "marker_mirror_order_policy_tradeoff", logger)


def rank_boundary(asset_dir: Path, output_dir: Path, logger: ProgressLogger) -> list[Path]:
    df = pd.read_csv(asset_dir / "marker_mirror_rank_boundary_table.csv")
    df["rank_label"] = df["rank"].map({"order": "Order", "family": "Family", "genus": "Genus"})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    y = np.arange(len(df))
    ax.barh(y - 0.18, df["single_label_best_coverage_pct"], height=0.32, color=COLORS["coverage"], label="Best single-label coverage")
    ax.barh(y + 0.18, df["single_label_target_met_rate_pct"], height=0.32, color=COLORS["precision"], label="Target met rate")
    ax.set_yticks(y)
    ax.set_yticklabels(df["rank_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("%")
    ax.set_title("Single-label rank repair")
    ax.axvline(100, color="#111827", linestyle="--", linewidth=0.8)
    ax.legend(loc="lower right")
    for idx, row in enumerate(df.itertuples(index=False)):
        ax.text(row.single_label_best_coverage_pct + 1, idx - 0.18, f"{row.single_label_best_coverage_pct:.1f}", va="center", fontsize=8)
        ax.text(row.single_label_target_met_rate_pct + 1, idx + 0.18, f"{row.single_label_target_met_rate_pct:.0f}", va="center", fontsize=8)

    ax = axes[1]
    ax.barh(y, df["best_set_mean_size"], color="#9CA3AF")
    ax.set_yticks(y)
    ax.set_yticklabels(df["rank_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean set size")
    ax.set_title("Set-valued output becomes too broad")
    for idx, row in enumerate(df.itertuples(index=False)):
        ax.text(row.best_set_mean_size + 1, idx, f"{row.best_set_mean_size:.1f}", va="center", fontsize=8)
    ax.text(
        0.0,
        -0.26,
        "Family/genus remain disabled: current evidence cannot make useful target-99 calls.",
        transform=ax.transAxes,
        fontsize=8,
        color="#374151",
    )
    return save(fig, output_dir, "marker_mirror_rank_boundary", logger)


def runtime(asset_dir: Path, output_dir: Path, logger: ProgressLogger) -> list[Path]:
    df = pd.read_csv(asset_dir / "marker_mirror_runtime_table.csv")
    df = df.sort_values("seconds", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(df["stage"], df["seconds"], color=["#9CA3AF", "#9CA3AF", "#F58518", "#4C78A8"])
    ax.set_title("Full 12S wrapper runtime on Vast")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("")
    for patch, row in zip(ax.patches, df.itertuples(index=False)):
        ax.text(
            patch.get_width() + 4,
            patch.get_y() + patch.get_height() / 2,
            f"{row.seconds:.1f}s ({row.percent_runtime:.1f}%)",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(df["seconds"]) * 1.28)
    return save(fig, output_dir, "marker_mirror_runtime_breakdown", logger)


def write_slide_summary(asset_dir: Path, output_dir: Path, logger: ProgressLogger) -> Path:
    support = pd.read_csv(asset_dir / "marker_mirror_candidate_support_table.csv")
    policy = pd.read_csv(asset_dir / "marker_mirror_order_policy_table.csv")
    ranks = pd.read_csv(asset_dir / "marker_mirror_rank_boundary_table.csv")
    runtime_df = pd.read_csv(asset_dir / "marker_mirror_runtime_table.csv")
    blast_union = support[support["candidate_source"] == "MarkerMirror + BLASTN union"].iloc[0]
    stable = policy[policy["mode"] == "stable_order"].iloc[0]
    high = policy[policy["mode"] == "high_coverage_order"].iloc[0]
    rank_lines = []
    for row in ranks.itertuples(index=False):
        rank_lines.append(
            f"- {row.rank}: best single-label coverage {row.single_label_best_coverage_pct:.1f}% "
            f"at {row.single_label_best_precision_pct:.2f}% precision; target met "
            f"{row.single_label_target_met_rate_pct:.0f}% of repeats."
        )
    total_runtime = runtime_df["seconds"].sum()
    text = f"""# MarkerMirror Slide-Ready Result Summary

## Candidate Support

MarkerMirror + BLASTN union top-50 support on 3,566 12S queries:

- species: {blast_union.species_pct:.1f}%
- genus: {blast_union.genus_pct:.1f}%
- family: {blast_union.family_pct:.1f}%
- order: {blast_union.order_pct:.1f}%

Species stays low because the held-out query species are absent from the current same-marker 12S reference by split design.

## Order/No-Call Modes

- Stable default: {stable.n_assigned:.0f}/{stable.n_queries:.0f} order calls, {stable.coverage_pct:.1f}% coverage, {stable.diagnostic_precision_pct:.1f}% diagnostic precision.
- High-coverage diagnostic: {high.n_assigned:.0f}/{high.n_queries:.0f} order calls, {high.coverage_pct:.1f}% full-table coverage, {high.nested_mean_heldout_coverage_pct:.1f}% mean held-out coverage, {high.nested_mean_heldout_precision_pct:.1f}% mean held-out precision.

## Rank Boundary

{chr(10).join(rank_lines)}

Order is the only rank currently stable enough for an explicit high-coverage mode.

## Runtime

Full wrapper runtime on Vast was {total_runtime:.1f} seconds for 3,566 queries. BLASTN dominates the run time.
"""
    path = output_dir / "marker_mirror_slide_ready_summary.md"
    path.write_text(text, encoding="utf-8")
    logger.log(f"wrote {rel(path)}")
    return path


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).stem)
    logger.log(f"asset_dir={rel(args.asset_dir)}")
    logger.log(f"output_dir={rel(args.output_dir)}")
    setup()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.extend(candidate_support(args.asset_dir, args.output_dir, logger))
    written.extend(order_policy(args.asset_dir, args.output_dir, logger))
    written.extend(rank_boundary(args.asset_dir, args.output_dir, logger))
    written.extend(runtime(args.asset_dir, args.output_dir, logger))
    written.append(write_slide_summary(args.asset_dir, args.output_dir, logger))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {"asset_dir": rel(args.asset_dir)},
        "outputs": [rel(path) for path in written],
        "claim_boundary": "Figure drafts only. They render Exp 122 manuscript assets and do not add new metrics.",
    }
    manifest_path = args.output_dir / "marker_mirror_manuscript_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"wrote {rel(manifest_path)}")
    logger.done(Path(__file__).stem)


if __name__ == "__main__":
    main()
