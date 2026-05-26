#!/usr/bin/env python3
"""Build a canonical results ledger from saved MarineMamba result JSONs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "summary"


RANKS = ("species", "genus", "family", "order")


def load(path: str) -> dict[str, Any] | None:
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text())


def accuracy(block: dict[str, Any] | None, rank: str) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get(rank)
    if isinstance(value, dict):
        return value.get("accuracy")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def n_total(block: dict[str, Any] | None, rank: str) -> int | None:
    if not isinstance(block, dict):
        return None
    value = block.get(rank)
    if isinstance(value, dict):
        return value.get("n_total")
    return None


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(100.0 * value, 4)


def add_rank_row(
    rows: list[dict[str, Any]],
    *,
    track: str,
    marker: str,
    dataset: str,
    method: str,
    split: str,
    source_file: str,
    metrics: dict[str, Any],
    provenance: str,
    notes: str = "",
    assignment_rate: float | None = None,
    no_hit_rate: float | None = None,
    n_query: int | None = None,
) -> None:
    row: dict[str, Any] = {
        "track": track,
        "marker": marker,
        "dataset": dataset,
        "method": method,
        "split": split,
        "provenance": provenance,
        "source_file": source_file,
        "n_query": n_query or metrics.get("n_query") if isinstance(metrics, dict) else n_query,
        "assignment_rate": pct(assignment_rate),
        "no_hit_rate": pct(no_hit_rate),
        "notes": notes,
    }
    for rank in RANKS:
        row[f"{rank}_accuracy_pct"] = pct(accuracy(metrics, rank))
        row[f"{rank}_n"] = n_total(metrics, rank)
    rows.append(row)


def add_scalar_row(
    rows: list[dict[str, Any]],
    *,
    track: str,
    marker: str,
    dataset: str,
    method: str,
    split: str,
    source_file: str,
    provenance: str,
    metric_name: str,
    metric_value: Any,
    notes: str = "",
) -> None:
    row = {
        "track": track,
        "marker": marker,
        "dataset": dataset,
        "method": method,
        "split": split,
        "provenance": provenance,
        "source_file": source_file,
        "n_query": None,
        "assignment_rate": None,
        "no_hit_rate": None,
        "notes": notes,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }
    for rank in RANKS:
        row[f"{rank}_accuracy_pct"] = None
        row[f"{rank}_n"] = None
    rows.append(row)


def add_kmer(rows: list[dict[str, Any]], path: str, *, track: str, marker: str, dataset: str, provenance: str) -> None:
    data = load(path)
    if not data:
        return
    if "test_seen_species" in data:
        add_rank_row(
            rows,
            track=track,
            marker=marker,
            dataset=dataset,
            method=f"{data.get('k', '')}-mer 1NN".strip(),
            split="seen_species_test",
            source_file=path,
            metrics=data["test_seen_species"],
            provenance=provenance,
            n_query=data.get("n_test"),
        )
    eval_c = data.get("eval_c_true_unseen_species_seen_genera")
    if eval_c:
        add_rank_row(
            rows,
            track=track,
            marker=marker,
            dataset=dataset,
            method=f"{data.get('k', '')}-mer 1NN".strip(),
            split="eval_c_unseen_species_seen_genera",
            source_file=path,
            metrics=eval_c,
            provenance=provenance,
            n_query=data.get("n_eval_c"),
        )


def add_curriculum(rows: list[dict[str, Any]], path: str, *, track: str, marker: str, dataset: str, method: str, provenance: str, notes: str = "") -> None:
    data = load(path)
    if not data:
        return
    if "test_accuracies" in data:
        metrics = {rank: {"accuracy": data["test_accuracies"].get(rank)} for rank in RANKS}
        add_rank_row(
            rows,
            track=track,
            marker=marker,
            dataset=dataset,
            method=method,
            split="seen_species_test",
            source_file=path,
            metrics=metrics,
            provenance=provenance,
            notes=notes,
        )
    eval_a = data.get("eval_a_unseen_genera")
    if eval_a:
        add_rank_row(
            rows,
            track=track,
            marker=marker,
            dataset=dataset,
            method=method,
            split="eval_a_unseen_genera",
            source_file=path,
            metrics=eval_a,
            provenance=provenance,
            notes=notes,
        )
    eval_c = data.get("eval_c_true_unseen_species_seen_genera")
    if eval_c:
        add_rank_row(
            rows,
            track=track,
            marker=marker,
            dataset=dataset,
            method=method,
            split="eval_c_unseen_species_seen_genera",
            source_file=path,
            metrics=eval_c,
            provenance=provenance,
            notes=notes,
            n_query=eval_c.get("n_query_sequences"),
        )


def add_similarity(rows: list[dict[str, Any]], path: str, *, track: str, marker: str, dataset: str, provenance: str) -> None:
    data = load(path)
    if not data:
        return
    for method_name, method_data in data.get("methods", {}).items():
        for split_key, split_name in [
            ("supervised_test", "seen_species_test"),
            ("eval_c_unseen_species", "eval_c_unseen_species_seen_genera"),
        ]:
            metrics = method_data.get(split_key)
            if not metrics:
                continue
            add_rank_row(
                rows,
                track=track,
                marker=marker,
                dataset=dataset,
                method=method_name.upper(),
                split=split_name,
                source_file=path,
                metrics=metrics,
                provenance=provenance,
                no_hit_rate=metrics.get("no_hit_rate"),
                n_query=metrics.get("n_query"),
            )


def add_checkpoint_predictions(rows: list[dict[str, Any]], path: str, *, track: str, marker: str, dataset: str, provenance: str) -> None:
    data = load(path)
    if not data:
        return
    for split_key, split_name in [
        ("supervised_test", "seen_species_test"),
        ("eval_c_unseen_species", "eval_c_unseen_species_seen_genera"),
    ]:
        split = data.get("splits", {}).get(split_key)
        if not split:
            continue
        for method_name in ("direct", "knn"):
            metrics = split.get(method_name)
            if metrics:
                add_rank_row(
                    rows,
                    track=track,
                    marker=marker,
                    dataset=dataset,
                    method=f"rank-focused 6-mer {method_name}",
                    split=split_name,
                    source_file=path,
                    metrics=metrics,
                    provenance=provenance,
                    notes="Prediction export from saved checkpoint.",
                )


def add_hybrid(rows: list[dict[str, Any]], path: str, *, track: str, marker: str, dataset: str, provenance: str) -> None:
    data = load(path)
    if not data:
        return
    for split_key, split_name in [
        ("supervised_test", "seen_species_test"),
        ("eval_c_unseen_species", "eval_c_unseen_species_seen_genera"),
    ]:
        split = data.get("splits", {}).get(split_key)
        if not split:
            continue
        forced = split.get("forced_blast")
        if forced:
            add_rank_row(
                rows,
                track=track,
                marker=marker,
                dataset=dataset,
                method="BLAST forced",
                split=split_name,
                source_file=path,
                metrics=forced,
                provenance=provenance,
                no_hit_rate=split.get("no_hit_rate"),
                n_query=split.get("n_query"),
            )
        for strategy in ("blast_assigned_only", "blast_or_knn", "blast_or_direct"):
            sweeps = [row for row in split.get("sweeps", []) if row.get("strategy") == strategy]
            for rank in ("genus", "family", "order"):
                valid = [row for row in sweeps if isinstance(row.get(rank), dict) and row[rank].get("accuracy") is not None]
                if not valid:
                    continue
                best = max(valid, key=lambda row: (row[rank]["accuracy"], row.get("assignment_rate", row.get("blast_assignment_rate", 0.0))))
                metrics = {rank: best[rank]}
                add_rank_row(
                    rows,
                    track=track,
                    marker=marker,
                    dataset=dataset,
                    method=f"{strategy} best_{rank}",
                    split=split_name,
                    source_file=path,
                    metrics=metrics,
                    provenance=provenance,
                    assignment_rate=best.get("assignment_rate", best.get("blast_assignment_rate")),
                    notes=f"Best {rank} threshold: pident>={best.get('pident_threshold')}, qcov>={best.get('qcov_threshold')}.",
                )


def add_phylo(rows: list[dict[str, Any]]) -> None:
    path = "results/phylo_fish_only_results.json"
    data = load(path)
    if data:
        add_rank_row(
            rows,
            track="COI phylo",
            marker="COI",
            dataset="fish_only",
            method="phylo fish-only embedding",
            split="seen_species_test",
            source_file=path,
            metrics={"species": {"accuracy": data.get("species_accuracy")}},
            provenance="legacy_phylo; likely valid unseen-species protocol but not yet rerun on v3 cleaned split",
            notes="Keep as supporting evidence; rerun on processed_clean before final reporting.",
        )
        eval_c = data.get("eval_c_stalder")
        if eval_c:
            metrics = {rank: {"accuracy": eval_c.get(rank)} for rank in ("genus", "family", "order")}
            add_rank_row(
                rows,
                track="COI phylo",
                marker="COI",
                dataset="fish_only",
                method="phylo fish-only embedding",
                split="eval_c_unseen_species_seen_genera",
                source_file=path,
                metrics=metrics,
                provenance="legacy_phylo; likely valid unseen-species protocol but not yet rerun on v3 cleaned split",
                notes="Comparable as phylogenetic generalization evidence, not as the cleaned COI v3 headline result.",
            )
        tree = data.get("tree_recovery", {})
        for metric in ("pearson_r", "spearman_r"):
            if metric in tree:
                add_scalar_row(
                    rows,
                    track="COI phylo",
                    marker="COI",
                    dataset="fish_only",
                    method="phylo fish-only embedding",
                    split="tree_recovery_seen_species",
                    source_file=path,
                    provenance="legacy_phylo; likely valid unseen-species protocol but not yet rerun on v3 cleaned split",
                    metric_name=metric,
                    metric_value=tree[metric],
                    notes="Tree-distance recovery using real Fish Tree of Life distances.",
                )
    for dim in (64, 128, 384):
        path = f"results/tree_recovery_unseen_dim{dim}.json"
        data = load(path)
        if not data:
            continue
        for block_name in ("unseen_unseen", "unseen_train"):
            block = data.get(block_name, {})
            for metric in ("pearson_r", "spearman_r"):
                if metric in block:
                    add_scalar_row(
                        rows,
                        track="COI phylo",
                        marker="COI",
                        dataset="fish_only",
                        method=f"tree recovery unseen dim{dim}",
                        split=block_name,
                        source_file=path,
                        provenance="legacy_phylo; likely valid unseen-species protocol but not yet rerun on v3 cleaned split",
                        metric_name=metric,
                        metric_value=block[metric],
                        notes=f"n_pairs={block.get('n_pairs')}; n_unseen_species={data.get('n_unseen_species')}.",
                    )


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    v3 = "clean_v3; copied from Vast remote_runs/2026-05-25"
    add_kmer(rows, "results/remote_runs/2026-05-25/h200/coi_v3_kmer/eval_c_kmer6_results.json", track="COI", marker="COI", dataset="processed_clean", provenance=v3)
    add_similarity(rows, "results/remote_runs/2026-05-25/h200/coi_v3_similarity/similarity_baselines_results.json", track="COI", marker="COI", dataset="processed_clean", provenance=v3)
    add_curriculum(rows, "results/remote_runs/2026-05-25/h200/coi_v3_6mer/curriculum_6mer_results.json", track="COI", marker="COI", dataset="processed_clean", method="6-mer curriculum", provenance=v3)
    add_curriculum(rows, "results/remote_runs/2026-05-25/rtx_pro/coi_v3_char/multihead_hierarchical_results.json", track="COI", marker="COI", dataset="processed_clean", method="char multihead curriculum", provenance=v3)

    add_phylo(rows)

    kmer_paths = {
        "mitohelper_full": "results/edna_12s_full_kmer/eval_c_kmer6_results.json",
        "mitohelper_mifish_exact": "results/edna_12s_mifish_kmer/eval_c_kmer6_results.json",
        "rcrux_cleaned": "results/edna_12s_rcrux_cleaned_kmer/eval_c_kmer6_results.json",
        "rcrux_blast_seed": "results/edna_12s_rcrux_blast_kmer/eval_c_kmer6_results.json",
        "mare_mage": "results/edna_12s_mare_mage_kmer/eval_c_kmer6_results.json",
        "multisource": "results/edna_12s_multisource_kmer/eval_c_kmer6_results.json",
    }
    for dataset, path in kmer_paths.items():
        add_kmer(rows, path, track="12S reference", marker="12S", dataset=dataset, provenance="local 12S split; audit-clean")

    add_curriculum(rows, "results/remote_runs/2026-05-25/h200/edna_12s_full_6mer/curriculum_6mer_results.json", track="12S reference", marker="12S", dataset="mitohelper_full", method="6-mer curriculum max_tokens_655", provenance=v3, notes="Known truncation failure mode on full-length 12S.")
    add_curriculum(rows, "results/remote_runs/2026-05-25/h200/edna_12s_full_6mer_maxtok1995/curriculum_6mer_results.json", track="12S reference", marker="12S", dataset="mitohelper_full", method="6-mer curriculum max_tokens_1995", provenance=v3)
    add_curriculum(rows, "results/remote_runs/2026-05-25/rtx_pro/edna_12s_rcrux_cleaned_6mer/curriculum_6mer_results.json", track="12S reference", marker="12S", dataset="rcrux_cleaned", method="6-mer curriculum", provenance=v3)
    add_curriculum(rows, "results/remote_runs/2026-05-25/rtx_pro/edna_12s_rcrux_blast_seed_6mer/curriculum_6mer_results.json", track="12S reference", marker="12S", dataset="rcrux_blast_seed", method="6-mer curriculum", provenance=v3)
    add_curriculum(rows, "results/remote_runs/2026-05-25/h200/edna_12s_mare_mage_6mer/curriculum_6mer_results.json", track="12S reference", marker="12S", dataset="mare_mage", method="6-mer curriculum", provenance=v3)
    add_curriculum(rows, "results/remote_runs/2026-05-25/rtx_pro/edna_12s_multisource_6mer/curriculum_6mer_results.json", track="12S reference", marker="12S", dataset="multisource", method="6-mer curriculum", provenance=v3)
    add_curriculum(rows, "results/remote_runs/2026-05-25/rtx_pro/edna_12s_rcrux_cleaned_char/multihead_hierarchical_results.json", track="12S reference", marker="12S", dataset="rcrux_cleaned", method="char multihead curriculum", provenance=v3)

    for dataset, path in {
        "mitohelper_full": "results/remote_runs/2026-05-25/rtx_pro/edna_12s_similarity/full/similarity_baselines_results.json",
        "rcrux_cleaned": "results/remote_runs/2026-05-25/rtx_pro/edna_12s_similarity/rcrux_cleaned/similarity_baselines_results.json",
        "rcrux_blast_seed": "results/remote_runs/2026-05-25/rtx_pro/edna_12s_similarity/rcrux_blast/similarity_baselines_results.json",
        "mare_mage": "results/remote_runs/2026-05-25/rtx_pro/edna_12s_similarity/mare_mage/similarity_baselines_results.json",
        "multisource": "results/remote_runs/2026-05-25/rtx_pro/edna_12s_multisource_similarity/similarity_baselines_results.json",
    }.items():
        add_similarity(rows, path, track="12S reference", marker="12S", dataset=dataset, provenance=v3)

    for dataset, train_path, pred_path, hybrid_path in [
        (
            "rcrux_cleaned",
            "results/remote_runs/2026-05-25/rtx_pro/edna_12s_rcrux_cleaned_6mer_rankonly/curriculum_6mer_results.json",
            "results/remote_runs/2026-05-25/rtx_pro/edna_12s_rcrux_cleaned_6mer_rankonly_predictions/neural_prediction_metrics.json",
            "results/remote_runs/2026-05-25/rtx_pro/edna_12s_rcrux_cleaned_rankonly_blast_hybrid/blast_threshold_hybrid_results.json",
        ),
        (
            "mare_mage",
            "results/remote_runs/2026-05-25/rtx_pro/edna_12s_mare_mage_6mer_rankonly/curriculum_6mer_results.json",
            "results/remote_runs/2026-05-25/rtx_pro/edna_12s_mare_mage_6mer_rankonly_predictions/neural_prediction_metrics.json",
            "results/remote_runs/2026-05-25/rtx_pro/edna_12s_mare_mage_rankonly_blast_hybrid/blast_threshold_hybrid_results.json",
        ),
        (
            "multisource",
            "results/remote_runs/2026-05-25/h200/edna_12s_multisource_6mer_rankonly/curriculum_6mer_results.json",
            "results/remote_runs/2026-05-25/h200/edna_12s_multisource_6mer_rankonly_predictions/neural_prediction_metrics.json",
            "results/remote_runs/2026-05-25/h200/edna_12s_multisource_rankonly_blast_hybrid/blast_threshold_hybrid_results.json",
        ),
    ]:
        add_curriculum(rows, train_path, track="12S reference", marker="12S", dataset=dataset, method="rank-focused 6-mer curriculum", provenance=v3, notes="Species loss disabled.")
        add_checkpoint_predictions(rows, pred_path, track="12S reference", marker="12S", dataset=dataset, provenance=v3)
        add_hybrid(rows, hybrid_path, track="12S reference", marker="12S", dataset=dataset, provenance=v3)

    # Local direct-head fallback hybrid for rCRUX was run after the remote copy.
    add_hybrid(
        rows,
        "results/edna_12s_rcrux_cleaned_rankonly_blast_hybrid_direct/blast_threshold_hybrid_results.json",
        track="12S reference",
        marker="12S",
        dataset="rcrux_cleaned",
        provenance="local follow-up; direct-head fallback variant",
    )

    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "results_ledger.json"
    csv_path = OUT_DIR / "results_ledger.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fieldnames = [
        "track",
        "marker",
        "dataset",
        "method",
        "split",
        "species_accuracy_pct",
        "genus_accuracy_pct",
        "family_accuracy_pct",
        "order_accuracy_pct",
        "species_n",
        "genus_n",
        "family_n",
        "order_n",
        "n_query",
        "assignment_rate",
        "no_hit_rate",
        "metric_name",
        "metric_value",
        "provenance",
        "source_file",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    write_outputs(rows)
    print(f"wrote {len(rows)} rows")
    print(f"json: {OUT_DIR / 'results_ledger.json'}")
    print(f"csv:  {OUT_DIR / 'results_ledger.csv'}")


if __name__ == "__main__":
    main()
