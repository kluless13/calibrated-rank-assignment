#!/usr/bin/env python3
"""Render local figure drafts from source-data CSVs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SOURCE_DIR = ROOT / "results" / "figures" / "source_data"
OUT_DIR = ROOT / "results" / "figures" / "plots"
RANK_ORDER = ["genus", "family", "order"]
EDNA_RANK_ORDER = ["species", "genus", "family", "order"]
EDNA_METHOD_LABELS = {
    "SSM sequence-only": "SSM sequence",
    "CNN sequence-only": "CNN sequence",
    "SSM RLS/OBIS learned co-occurrence w025": "SSM RLS/OBIS w0.25",
    "SSM RLS/OBIS learned co-occurrence w050": "SSM RLS/OBIS w0.50",
    "SSM RLS/OBIS learned co-occurrence w100": "SSM RLS/OBIS w1.00",
    "SSM RLS/OBIS learned co-occurrence w200": "SSM RLS/OBIS w2.00",
    "CNN RLS/OBIS learned co-occurrence w025": "CNN RLS/OBIS w0.25",
    "SSM FISHGLOB learned co-occurrence w025": "SSM FISHGLOB w0.25",
    "SSM FISHGLOB learned co-occurrence w100": "SSM FISHGLOB w1.00",
    "CNN FISHGLOB learned co-occurrence w025": "CNN FISHGLOB w0.25",
    "mamba_sequence_only": "Mamba sequence",
    "mamba_same_sample_cooccurrence_w005": "Mamba same-sample",
    "mamba_rls_geographic_site20_w005": "Mamba RLS prior",
    "mamba_obis_occurrence_site20_w005": "Mamba OBIS prior",
    "mamba_rls_obis_site20_w005": "Mamba RLS+OBIS",
    "rls_prior_only_site20": "RLS only",
    "obis_occurrence_prior_only_site20": "OBIS only",
    "blast_train_reference": "BLAST reference",
}
EDNA_METHOD_ORDER = [
    "SSM sequence-only",
    "CNN sequence-only",
    "SSM RLS/OBIS learned co-occurrence w025",
    "SSM RLS/OBIS learned co-occurrence w050",
    "SSM RLS/OBIS learned co-occurrence w100",
    "SSM RLS/OBIS learned co-occurrence w200",
    "CNN RLS/OBIS learned co-occurrence w025",
    "SSM FISHGLOB learned co-occurrence w025",
    "SSM FISHGLOB learned co-occurrence w100",
    "CNN FISHGLOB learned co-occurrence w025",
    "mamba_sequence_only",
    "mamba_rls_obis_site20_w005",
    "blast_train_reference",
]


def setup() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / ".matplotlib").mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    })


def savefig(name: str) -> None:
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def plot_coi_eval_c() -> None:
    path = SOURCE_DIR / "figure_coi_eval_c.csv"
    df = pd.read_csv(path)
    df = df[df["rank"].isin(RANK_ORDER)].copy()
    df["rank"] = pd.Categorical(df["rank"], categories=RANK_ORDER, ordered=True)
    method_order = ["6-mer 1NN", "BLAST", "VSEARCH", "6-mer curriculum", "char multihead curriculum"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    plt.figure(figsize=(7.5, 4.2))
    ax = sns.barplot(data=df.sort_values(["method", "rank"]), x="method", y="accuracy_pct", hue="rank")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("COI strict Eval C")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Rank", ncols=3, loc="lower right")
    savefig("coi_eval_c_accuracy")


def plot_12s_eval_c_heatmap() -> None:
    path = SOURCE_DIR / "figure_12s_eval_c_reference.csv"
    df = pd.read_csv(path)
    df = df[df["rank"].isin(RANK_ORDER)].copy()
    df["label"] = df["dataset"].str.replace("_", " ", regex=False) + " | " + df["method"]
    grouped = (
        df.groupby(["label", "rank"], as_index=False)["accuracy_pct"]
        .max()
        .pivot(index="label", columns="rank", values="accuracy_pct")
        .reindex(columns=RANK_ORDER)
    )
    order = grouped.mean(axis=1).sort_values(ascending=False).index
    grouped = grouped.loc[order]
    height = max(5.0, 0.27 * len(grouped) + 1.6)
    plt.figure(figsize=(6.8, height))
    ax = sns.heatmap(grouped, annot=True, fmt=".1f", cmap="viridis", vmin=0, vmax=100, cbar_kws={"label": "Accuracy (%)"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("12S Eval C reference benchmarks")
    savefig("12s_eval_c_reference_heatmap")


def plot_12s_abstention() -> None:
    path = SOURCE_DIR / "figure_12s_abstention.csv"
    df = pd.read_csv(path)
    df = df[df["rank"].isin(RANK_ORDER)].copy()
    df["rank"] = pd.Categorical(df["rank"], categories=RANK_ORDER, ordered=True)
    df["assignment_rate_pct"] = pd.to_numeric(df["assignment_rate_pct"], errors="coerce")
    df["assignment_rate_pct"] = df["assignment_rate_pct"].fillna(100.0 - pd.to_numeric(df["no_hit_rate_pct"], errors="coerce").fillna(0.0))
    df["plot_label"] = df["method"].str.replace("_", " ", regex=False)
    plt.figure(figsize=(7.0, 4.4))
    ax = sns.scatterplot(
        data=df,
        x="assignment_rate_pct",
        y="accuracy_pct",
        hue="rank",
        style="plot_label",
        s=70,
    )
    ax.set_xlabel("Assigned queries (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(70, 105)
    ax.set_title("12S BLAST/hybrid abstention")
    ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    savefig("12s_abstention_accuracy_assignment")


def plot_dataset_coverage() -> None:
    path = SOURCE_DIR / "figure_dataset_coverage.csv"
    df = pd.read_csv(path)
    df["label"] = df["marker"] + " " + df["dataset"].str.replace("_", " ", regex=False)
    long = df.melt(
        id_vars=["label"],
        value_vars=["total_sequences", "total_species", "total_genera"],
        var_name="measure",
        value_name="count",
    )
    measure_names = {
        "total_sequences": "Sequences",
        "total_species": "Species",
        "total_genera": "Genera",
    }
    long["measure"] = long["measure"].map(measure_names)
    plt.figure(figsize=(8.0, 4.8))
    ax = sns.barplot(data=long, x="label", y="count", hue="measure")
    ax.set_xlabel("")
    ax.set_ylabel("Count, log scale")
    ax.set_yscale("log")
    ax.set_title("Dataset coverage")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    savefig("dataset_coverage")


def plot_phylo_support() -> None:
    path = SOURCE_DIR / "figure_phylo_support.csv"
    df = pd.read_csv(path)
    tree = df[df["metric"].isin(["pearson_r", "spearman_r"])].copy()
    if "track" in tree.columns and (tree["track"] == "COI fish-tree").any():
        tree = tree[
            (tree["track"] == "COI fish-tree")
            & tree["method"].isin([
                "cosine dim384 seed1206",
                "hierarchical hybrid dim512 seed1206",
                "cosine dim512 seed1206",
            ])
            & tree["split"].isin([
                "eval_c_tree_recovery_zero_shot_reference",
                "unseen_genera_tree_recovery_zero_shot_reference",
            ])
        ].copy()
        split_names = {
            "eval_c_tree_recovery_zero_shot_reference": "Eval C",
            "unseen_genera_tree_recovery_zero_shot_reference": "Unseen genera",
        }
        method_names = {
            "cosine dim384 seed1206": "cosine-384",
            "hierarchical hybrid dim512 seed1206": "hier-512",
            "cosine dim512 seed1206": "cosine-512",
        }
        tree["series"] = tree["split"].map(split_names) + " | " + tree["method"].map(method_names)
    else:
        tree["series"] = tree["split"].str.replace("_", " ", regex=False) + " | " + tree["method"].str.replace("tree recovery unseen ", "", regex=False)
        tree = tree[tree["split"].isin(["tree_recovery_seen_species", "unseen_unseen", "unseen_train"])]
    plt.figure(figsize=(8.2, 4.8))
    ax = sns.barplot(data=tree, x="series", y="value", hue="metric")
    ax.set_xlabel("")
    ax.set_ylabel("Correlation")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fish Tree recovery")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    savefig("fish_tree_recovery")


def plot_global_edna_prior_matrix() -> None:
    path = SOURCE_DIR / "figure_global_edna_prior_matrix.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df[df["rank"].isin(EDNA_RANK_ORDER)].copy()
    df["method_label"] = df["method"].map(EDNA_METHOD_LABELS).fillna(df["method"].str.replace("_", " ", regex=False))
    df["method"] = pd.Categorical(df["method"], categories=EDNA_METHOD_ORDER, ordered=True)
    df["rank"] = pd.Categorical(df["rank"], categories=EDNA_RANK_ORDER, ordered=True)
    heat = (
        df.sort_values(["method", "rank"])
        .pivot(index="method_label", columns="rank", values="asv_top10_accuracy_pct")
        .reindex(columns=EDNA_RANK_ORDER)
    )
    ordered_labels = [EDNA_METHOD_LABELS[m] for m in EDNA_METHOD_ORDER if EDNA_METHOD_LABELS[m] in heat.index]
    heat = heat.loc[ordered_labels]
    plt.figure(figsize=(7.2, 4.8))
    ax = sns.heatmap(heat, annot=True, fmt=".1f", cmap="mako", vmin=0, vmax=70, cbar_kws={"label": "ASV top-10 accuracy (%)"})
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Global eDNA sequence and ecological-context validation")
    savefig("global_edna_prior_matrix")


def plot_global_edna_sample_jaccard() -> None:
    path = SOURCE_DIR / "figure_global_edna_prior_matrix.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df[df["rank"].isin(EDNA_RANK_ORDER)].copy()
    df["method_label"] = df["method"].map(EDNA_METHOD_LABELS).fillna(df["method"].str.replace("_", " ", regex=False))
    df["method"] = pd.Categorical(df["method"], categories=EDNA_METHOD_ORDER, ordered=True)
    df["rank"] = pd.Categorical(df["rank"], categories=EDNA_RANK_ORDER, ordered=True)
    plt.figure(figsize=(8.0, 4.5))
    ax = sns.barplot(
        data=df.sort_values(["method", "rank"]),
        x="method_label",
        y="sample_top1_jaccard_pct",
        hue="rank",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Sample top-1 Jaccard (%)")
    ax.set_title("Global eDNA sample-level overlap")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Rank", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    savefig("global_edna_sample_jaccard")


def plot_global_edna_calibration() -> None:
    path = SOURCE_DIR / "figure_global_edna_calibration.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    methods = ["mamba_sequence_only", "mamba_rls_obis_site20_w005", "blast_train_reference"]
    ranks = ["genus", "family", "order"]
    df = df[df["method"].isin(methods) & df["rank"].isin(ranks)].copy()
    if df.empty:
        return
    df["method_label"] = df["method"].map(EDNA_METHOD_LABELS)
    df["rank"] = pd.Categorical(df["rank"], categories=ranks, ordered=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), sharex=True, sharey=False)
    for ax, rank in zip(axes, ranks, strict=True):
        subset = df[df["rank"] == rank].sort_values("assignment_rate_pct")
        sns.lineplot(
            data=subset,
            x="assignment_rate_pct",
            y="accuracy_pct",
            hue="method_label",
            marker="o",
            ax=ax,
            legend=(rank == "order"),
        )
        ax.set_title(rank.title())
        ax.set_xlabel("Assigned ASVs (%)")
        ax.set_ylabel("Accuracy (%)" if rank == "genus" else "")
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        if rank == "order":
            ax.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    fig.suptitle("Global eDNA no-call calibration", y=1.03)
    savefig("global_edna_calibration")


def main() -> None:
    global SOURCE_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    SOURCE_DIR = args.source_dir
    OUT_DIR = args.output_dir
    setup()
    plot_coi_eval_c()
    plot_12s_eval_c_heatmap()
    plot_12s_abstention()
    plot_dataset_coverage()
    plot_phylo_support()
    plot_global_edna_prior_matrix()
    plot_global_edna_sample_jaccard()
    plot_global_edna_calibration()


if __name__ == "__main__":
    main()
