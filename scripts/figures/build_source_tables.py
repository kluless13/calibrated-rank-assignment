#!/usr/bin/env python3
"""Create figure-ready source tables from the canonical results ledger."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "results" / "summary" / "results_ledger.csv"
OUT_DIR = ROOT / "results" / "figures" / "source_data"
RANKS = ("genus", "family", "order")


def read_ledger() -> list[dict[str, str]]:
    with LEDGER.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def long_rank_rows(rows: list[dict[str, str]], predicate) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        if not predicate(row):
            continue
        for rank in RANKS:
            value = row.get(f"{rank}_accuracy_pct")
            if value == "":
                continue
            out.append(
                {
                    "track": row["track"],
                    "marker": row["marker"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "split": row["split"],
                    "rank": rank,
                    "accuracy_pct": value,
                    "assignment_rate_pct": row.get("assignment_rate", ""),
                    "no_hit_rate_pct": row.get("no_hit_rate", ""),
                    "provenance": row["provenance"],
                    "source_file": row["source_file"],
                    "notes": row["notes"],
                }
            )
    return out


def dataset_coverage_rows() -> list[dict[str, object]]:
    paths = [
        ("COI", "processed_clean", ROOT / "data" / "processed_clean" / "dataset_stats.json"),
        ("12S", "mitohelper_full", ROOT / "data" / "edna" / "processed_12s_full" / "dataset_stats.json"),
        ("12S", "rcrux_cleaned", ROOT / "data" / "edna" / "processed_12s_rcrux_cleaned" / "dataset_stats.json"),
        ("12S", "rcrux_blast_seed", ROOT / "data" / "edna" / "processed_12s_rcrux_blast_seeds" / "dataset_stats.json"),
        ("12S", "mare_mage", ROOT / "data" / "edna" / "processed_12s_mare_mage" / "dataset_stats.json"),
        ("12S", "multisource", ROOT / "data" / "edna" / "processed_12s_multisource" / "dataset_stats.json"),
    ]
    rows: list[dict[str, object]] = []
    for marker, dataset, path in paths:
        if not path.exists():
            continue
        stats = json.loads(path.read_text())
        rows.append(
            {
                "marker": marker,
                "dataset": dataset,
                "total_sequences": stats.get("total_sequences"),
                "total_species": stats.get("total_species"),
                "total_genera": stats.get("total_genera"),
                "train_size": stats.get("train_size"),
                "val_size": stats.get("val_size"),
                "test_size": stats.get("test_size"),
                "eval_c_size": stats.get("eval_c_size"),
                "eval_c_species_count": stats.get("eval_c_species_count"),
                "unseen_size": stats.get("unseen_size"),
                "source_file": str(path.relative_to(ROOT)),
            }
        )
    return rows


def phylo_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        if row["track"] != "COI phylo":
            continue
        if row.get("metric_name"):
            out.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "split": row["split"],
                    "metric": row["metric_name"],
                    "value": row["metric_value"],
                    "rank": "",
                    "accuracy_pct": "",
                    "provenance": row["provenance"],
                    "source_file": row["source_file"],
                    "notes": row["notes"],
                }
            )
            continue
        for rank in ("species", "genus", "family", "order"):
            value = row.get(f"{rank}_accuracy_pct")
            if value:
                out.append(
                    {
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "split": row["split"],
                        "metric": "accuracy_pct",
                        "value": value,
                        "rank": rank,
                        "accuracy_pct": value,
                        "provenance": row["provenance"],
                        "source_file": row["source_file"],
                        "notes": row["notes"],
                    }
                )
    return out


def main() -> None:
    rows = read_ledger()

    common_fields = [
        "track",
        "marker",
        "dataset",
        "method",
        "split",
        "rank",
        "accuracy_pct",
        "assignment_rate_pct",
        "no_hit_rate_pct",
        "provenance",
        "source_file",
        "notes",
    ]

    write_csv(
        OUT_DIR / "figure_coi_eval_c.csv",
        long_rank_rows(
            rows,
            lambda row: row["track"] == "COI" and row["split"] == "eval_c_unseen_species_seen_genera",
        ),
        common_fields,
    )

    write_csv(
        OUT_DIR / "figure_12s_eval_c_reference.csv",
        long_rank_rows(
            rows,
            lambda row: row["track"] == "12S reference"
            and row["split"] == "eval_c_unseen_species_seen_genera"
            and (
                row["method"] in {"6-mer 1NN", "BLAST", "VSEARCH", "6-mer curriculum", "rank-focused 6-mer knn", "rank-focused 6-mer direct"}
                or row["method"].endswith("max_tokens_1995")
            ),
        ),
        common_fields,
    )

    write_csv(
        OUT_DIR / "figure_12s_abstention.csv",
        long_rank_rows(
            rows,
            lambda row: row["track"] == "12S reference"
            and row["split"] == "eval_c_unseen_species_seen_genera"
            and ("blast_assigned_only" in row["method"] or row["method"] == "BLAST forced"),
        ),
        common_fields,
    )

    coverage_fields = [
        "marker",
        "dataset",
        "total_sequences",
        "total_species",
        "total_genera",
        "train_size",
        "val_size",
        "test_size",
        "eval_c_size",
        "eval_c_species_count",
        "unseen_size",
        "source_file",
    ]
    write_csv(OUT_DIR / "figure_dataset_coverage.csv", dataset_coverage_rows(), coverage_fields)

    write_csv(
        OUT_DIR / "figure_phylo_support.csv",
        phylo_rows(rows),
        [
            "dataset",
            "method",
            "split",
            "metric",
            "value",
            "rank",
            "accuracy_pct",
            "provenance",
            "source_file",
            "notes",
        ],
    )

    print(f"wrote source tables to {OUT_DIR}")


if __name__ == "__main__":
    main()
