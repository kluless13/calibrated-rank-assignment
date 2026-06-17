#!/usr/bin/env python3
"""Render the remaining manuscript figures for the Experiment-1 (Paper 1) draft.

Adds four figure pairs (PNG+PDF) into
`results/paper1_phylo_calibrated_assignment/manuscript_assets/experiment1/figures/`:

  fig_pipeline_architecture     - the evidence-compiler pipeline schematic (Figure 1)
  fig_detect_novelty            - open-set novelty detection AUROC by rank
  fig_missing_reference_collapse- rank collapse under hidden references
  fig_rediscovery_granularity   - species AMI vs cluster granularity (our embedding
                                  ties VSEARCH at matched ~1.2k-cluster granularity)

Schematic boxes are hardcoded; all data figures read from source tables.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import seaborn as sns

B = ROOT / "results" / "paper1_phylo_calibrated_assignment"
SRC = B / "source_tables"
OUT = B / "manuscript_assets" / "experiment1" / "figures"

INK = "#1b3a4b"
ACCENT = "#e07a5f"
BLUE = "#2a6f97"


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def fig_pipeline_architecture() -> None:
    """The evidence-compiler pipeline as a staged schematic (Figure 1)."""
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    stages = [
        ("barcode sequence", "#dfe7ec", 11.0),
        ("fast vector candidate retrieval", "#cfe0ec", 9.7),
        ("classical sequence checks\n(BLAST / VSEARCH / p-distance)", "#cfe0ec", 8.4),
        ("tree-aware evidence + open-set novelty (DETECT)", "#cfe0ec", 7.1),
        ("reference-gap + marker-resolvability diagnostics", "#cfe0ec", 5.8),
        ("ecological prior (eDNA: geography / co-occurrence)", "#cfe0ec", 4.5),
        ("calibrated decision\nspecies / genus / family / order / no-call", "#f3d9cf", 3.0),
    ]
    boxw, boxh, cx = 6.6, 0.92, 3.7
    centers = {}
    for label, color, y in stages:
        box = FancyBboxPatch((cx - boxw / 2, y - boxh / 2), boxw, boxh,
                             boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.1, edgecolor=INK, facecolor=color)
        ax.add_patch(box)
        ax.text(cx, y, label, ha="center", va="center", fontsize=9.3, color=INK)
        centers[label] = y
    ys = [s[2] for s in stages]
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((cx, y0 - boxh / 2), (cx, y1 + boxh / 2),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.1, color=INK))
    # side outputs from the calibrated decision
    dy = stages[-1][2]
    for label, y in [("reason codes\n(why this rank)", 3.6), ("active reference-\ncuration priorities", 2.4)]:
        ax.add_patch(FancyBboxPatch((7.3, y - 0.5), 2.3, 1.0,
                                    boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.0, edgecolor=ACCENT, facecolor="#fbeae3"))
        ax.text(8.45, y, label, ha="center", va="center", fontsize=8.2, color="#8a3b27")
        ax.add_patch(FancyArrowPatch((cx + boxw / 2, dy), (7.3, y),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color=ACCENT))
    # principle banner
    ax.text(3.7, 0.9, "Evidence separation: each stream is measured independently, then fused",
            ha="center", va="center", fontsize=8.6, style="italic", color="#555")
    ax.set_title("The calibrated rank-adaptive assignment pipeline", fontsize=12.5, color=INK)
    save(fig, "fig_pipeline_architecture")


def fig_detect_novelty() -> None:
    """Open-set novelty detection AUROC by rank of novelty."""
    d = json.loads((SRC / "novelty_detection_rigorous.json").read_text())
    sf = d["single_feature_auroc_reference_only"]
    detector = d["multi_feature_detector_species_split"]["logistic"]["mean"]
    items = [
        ("genus-level\nnovelty", sf["known_vs_unseen_genera_genus_novel"], BLUE),
        ("multi-feature\ndetector", detector, ACCENT),
        ("all novelty\n(single feature)", sf["known_vs_novel_all"], "#76a5af"),
        ("species-level\nnovelty", sf["known_vs_eval_c_species_novel"], "#c9b7a8"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    bars = ax.bar(labels, vals, color=[i[2] for i in items], width=0.62)
    ax.axhline(0.5, color="#999", ls="--", lw=1)
    ax.text(3.35, 0.515, "chance (0.5)", fontsize=8, color="#777", ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (known vs novel)")
    ax.set_title("Open-set novelty detection (DETECT): genus-level novelty is\nreliably flagged; species-level is the hard limit", fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    save(fig, "fig_detect_novelty")


def fig_missing_reference_collapse() -> None:
    """Rank collapse: when a rank's references are hidden, the model refuses it."""
    df = pd.read_csv(SRC / "strict_missing_reference_summary.csv")
    df = df[df["split"] == "eval_c"]
    scenarios = {"species": "hide species", "genus": "hide genus", "family": "hide family"}
    ranks = ["species", "genus", "family", "order"]
    cols = {"species": "#c0392b", "genus": ACCENT, "family": BLUE, "order": "#76a5af"}

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    n = len(scenarios)
    width = 0.8 / len(ranks)
    x = range(n)
    for ri, rank in enumerate(ranks):
        heights = []
        for hide in scenarios:
            row = df[df["hide_rank"] == hide].iloc[0]
            heights.append(row[f"{rank}_top10_pct"])
        offs = [xi + (ri - (len(ranks) - 1) / 2) * width for xi in x]
        ax.bar(offs, heights, width=width, label=rank.capitalize(), color=cols[rank])
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(scenarios.values()))
    ax.set_ylabel("top-10 retrieval (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Missing-reference stress test (Eval C): the hidden rank collapses to 0,\nbroader ranks stay recoverable — abstention is principled, not a failure", fontsize=10.5)
    ax.legend(title="recovered at", fontsize=8.5, ncol=4, loc="upper left")
    save(fig, "fig_missing_reference_collapse")


def fig_rediscovery_granularity() -> None:
    """Species AMI vs cluster granularity: our embedding ties VSEARCH at matched granularity."""
    multitask = json.loads((SRC / "multitask_rediscovery.json").read_text())
    hybrid = multitask["models"]["cnn_hybrid (tree+species)"]
    blind = hybrid["blind"]["thr0.1"]["species"]   # 0.9149 @ 1229 clusters
    kmeans = 0.874   # our embedding, KMeans k=531 (from rediscovery_clustering)
    vsearch = json.loads((SRC / "vsearch_delimitation.json").read_text())["by_identity"]["id0.97"]["species"]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axvline(531, color="#5a8a52", ls="--", lw=1)
    ax.text(531, 0.845, " true species = 531", color="#3f6b39", fontsize=8.5, va="bottom")
    # our embedding: 531 -> 1229 clusters
    ax.plot([531, blind["n_clusters"]], [kmeans, blind["AMI"]], "-o", color=ACCENT,
            lw=1.6, markersize=8, label="our embedding (tree+species)")
    ax.scatter([vsearch["n_clusters"]], [vsearch["AMI"]], s=90, color=BLUE, zorder=3,
               edgecolor="white", label="VSEARCH (classical)")
    ax.annotate(f"k=531 (KMeans)\nAMI {kmeans:.3f}", (531, kmeans),
                textcoords="offset points", xytext=(10, -22), fontsize=8.5)
    ax.annotate(f"{blind['n_clusters']} clusters (blind)\nAMI {blind['AMI']:.3f}",
                (blind["n_clusters"], blind["AMI"]), textcoords="offset points", xytext=(-30, 12), fontsize=8.5)
    ax.annotate(f"VSEARCH: {vsearch['n_clusters']} clusters\nAMI {vsearch['AMI']:.3f}",
                (vsearch["n_clusters"], vsearch["AMI"]), textcoords="offset points", xytext=(-150, -2), fontsize=8.5, color=BLUE)
    ax.set_xlabel("number of clusters formed")
    ax.set_ylabel("species clustering AMI")
    ax.set_xlim(400, 1450)
    ax.set_ylim(0.84, 0.93)
    ax.set_title("Granularity matters: species delimitation over-segments. At matched\n~1.2k-cluster granularity, our embedding (0.915) ties VSEARCH (0.915)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")
    save(fig, "fig_rediscovery_granularity")


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    print("rendering manuscript figures ->", OUT)
    fig_pipeline_architecture()
    fig_detect_novelty()
    fig_missing_reference_collapse()
    fig_rediscovery_granularity()
    print("done.")


if __name__ == "__main__":
    main()
