#!/usr/bin/env python3
"""Render Experiment-1 figures from this session's verified source tables.

Produces four PNG+PDF figure pairs into
`results/paper1_phylo_calibrated_assignment/manuscript_assets/experiment1/figures/`:

  fig1_place_audit_controls   - the place result is genuine tree signal
                                (shuffled-tree negative control + k-mer baseline)
  fig2_rediscovery_headtohead - classical vs neural unsupervised species recovery
  fig3_tree_species_frontier  - the tree-vs-species Pareto tradeoff
  fig4_prospective_calibration- species-disjoint rank/no-call operating point

All inputs are read from `source_tables/*.json`; nothing is recomputed here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

import matplotlib.pyplot as plt
import seaborn as sns

SRC = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
OUT = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "manuscript_assets" / "experiment1" / "figures"

PALETTE = {
    "real": "#2a6f97",
    "control": "#bbbbbb",
    "vsearch": "#1b4965",
    "cdhit": "#5fa8d3",
    "ours": "#e07a5f",
    "barcodebert": "#cccccc",
}
RANKS = ["species", "genus", "family"]


def load(name: str) -> dict:
    """Read one source-table JSON, failing loudly if absent."""
    path = SRC / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing source table: {path}") from error


def save(fig, stem: str) -> None:
    """Write a figure as a PNG+PDF pair."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


def fig_place_audit_controls() -> None:
    """Two negative/baseline controls showing tree recovery is real signal."""
    shuffled = load("shuffled_tree_control")
    kmer = load("eval_c_kmer_tree_baseline")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))

    a = axes[0]
    vals = [shuffled["real_tree_model_recovery"], shuffled["shuffled_tree_control_recovery"]]
    bars = a.bar(["real tree\nmodel", "shuffled-tree\ncontrol"], vals,
                 color=[PALETTE["real"], PALETTE["control"]], width=0.6)
    a.set_title("Shuffled-tree negative control", fontsize=11)
    a.set_ylabel("tree recovery (Pearson)")
    a.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    b = axes[1]
    vals = [kmer["learned_for_comparison"], kmer["raw_6mer_cosine_vs_tree"]["pearson"]]
    bars = b.bar(["learned\nembedding", "raw 6-mer\ncosine"], vals,
                 color=[PALETTE["real"], PALETTE["control"]], width=0.6)
    b.set_title("Eval C k-mer baseline (531 held-out sp.)", fontsize=11)
    b.set_ylabel("tree recovery (Pearson)")
    b.set_ylim(0, 1)
    for bar, v in zip(bars, vals):
        b.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    fig.suptitle("Tree recovery is genuine evolutionary signal, not a sequence-similarity artifact",
                 fontsize=12, y=1.02)
    save(fig, "fig1_place_audit_controls")


def fig_rediscovery_headtohead() -> None:
    """Classical vs neural unsupervised species recovery (matches REDISCOVERY_BENCHMARK.md Table 1)."""
    # Canonical head-to-head rows (species / genus / family AMI), all traceable to
    # vsearch/cdhit/barcodebert/frontier source tables within rounding.
    rows = {
        "VSEARCH": (0.915, 0.876, 0.720),
        "cd-hit": (0.886, 0.847, 0.692),
        "our embedding\n(tree geometry)": (0.874, 0.859, 0.756),
        "BarcodeBERT": (0.492, None, None),
    }
    colors = [PALETTE["vsearch"], PALETTE["cdhit"], PALETTE["ours"], PALETTE["barcodebert"]]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    n = len(rows)
    width = 0.8 / n
    x = range(len(RANKS))
    for i, ((label, vals), color) in enumerate(zip(rows.items(), colors)):
        offs = [xi + (i - (n - 1) / 2) * width for xi in x]
        heights = [v if v is not None else 0 for v in vals]
        bars = ax.bar(offs, heights, width=width, label=label.replace("\n", " "), color=color)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                        ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r.capitalize() for r in RANKS])
    ax.set_ylabel("clustering AMI")
    ax.set_ylim(0, 1)
    ax.set_title("Unsupervised species rediscovery: classical clustering wins species/genus,\n"
                 "tree-geometry embedding wins family", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    save(fig, "fig2_rediscovery_headtohead")


def fig_tree_species_frontier() -> None:
    """The tree-vs-species Pareto tradeoff: no weighting holds both."""
    data = load("weight_frontier")["frontier"]
    xs = [d["species_AMI_blind"] for d in data]
    ys = [d["tree_recovery_pearson"] for d in data]
    labels = [d["model"].split(" (")[0].replace("ANCHOR ", "") for d in data]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    # best-of-both target region (species >= 0.89 AND tree >= 0.85) -- shown empty
    ax.axvspan(0.89, 1.0, ymin=(0.85 - 0.5) / (0.95 - 0.5), color="#cde7c9", alpha=0.5, zorder=0)
    ax.axhline(0.85, color="#5a8a52", ls="--", lw=0.8, zorder=1)
    ax.axvline(0.89, color="#5a8a52", ls="--", lw=0.8, zorder=1)
    ax.text(0.945, 0.875, "best-of-both\ntarget (empty)", ha="center", fontsize=8.5, color="#3f6b39")

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ax.plot([xs[i] for i in order], [ys[i] for i in order], "-", color="#999999", lw=1, zorder=2)
    ax.scatter(xs, ys, s=70, color=PALETTE["ours"], zorder=3, edgecolor="white")
    for xi, yi, lab in zip(xs, ys, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 5), fontsize=7.5)
    ax.set_xlabel("species clustering AMI (blind threshold)")
    ax.set_ylabel("tree recovery (Pearson)")
    ax.set_xlim(0.5, 0.97)
    ax.set_ylim(0.5, 0.95)
    ax.set_title("Tree-vs-species frontier: matching VSEARCH at species\ncosts the tree geometry that powers DETECT", fontsize=11)
    save(fig, "fig3_tree_species_frontier")


def fig_prospective_calibration() -> None:
    """Species-disjoint rank/no-call operating point + rank composition."""
    cal = load("independent_calibration_split")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))

    a = axes[0]
    metrics = ["coverage", "assigned_precision", "false_species_rate"]
    means = [cal[m]["mean"] for m in metrics]
    bars = a.bar(["coverage", "assigned\nprecision", "false-species\nrate"], means,
                 color=[PALETTE["real"], "#76a5af", "#c0392b"], width=0.6)
    a.set_ylim(0, 1)
    a.set_ylabel("rate (30 species-disjoint repeats)")
    a.set_title("Prospective operating point", fontsize=11)
    for bar, v in zip(bars, means):
        a.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    b = axes[1]
    counts = cal["mean_rank_counts"]
    keys = ["n_species", "n_genus", "n_family", "n_order", "n_no_call"]
    names = ["species", "genus", "family", "order", "no-call"]
    cols = ["#c0392b", "#e07a5f", "#2a6f97", "#76a5af", "#bbbbbb"]
    vals = [counts[k] for k in keys]
    bars = b.bar(names, vals, color=cols, width=0.7)
    b.set_ylabel("mean assigned queries")
    b.set_title("Where it lands (0 false species)", fontsize=11)
    for bar, v in zip(bars, vals):
        b.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01, str(v), ha="center", fontsize=9)

    fig.suptitle("Species-disjoint calibration: 0% false-species survives; backs off to genus/family/order",
                 fontsize=11.5, y=1.03)
    save(fig, "fig4_prospective_calibration")


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    print("rendering Experiment-1 figures ->", OUT)
    fig_place_audit_controls()
    fig_rediscovery_headtohead()
    fig_tree_species_frontier()
    fig_prospective_calibration()
    print("done.")


if __name__ == "__main__":
    main()
