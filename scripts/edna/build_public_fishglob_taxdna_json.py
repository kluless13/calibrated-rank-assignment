#!/usr/bin/env python3
"""Build a TAXDNA-style FISHGLOB co-occurrence JSON from public FISHGLOB exports."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def normalize_species_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def load_species_sequences(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text())
    return {
        str(label).strip(): [str(seq).strip().upper() for seq in sequences if str(seq).strip()]
        for label, sequences in raw.items()
    }


def load_candidate_map(candidate_species_csv: Path) -> dict[str, str]:
    candidates = pd.read_csv(candidate_species_csv)
    required = {"tree_label", "species_name"}
    missing = required - set(candidates.columns)
    if missing:
        raise SystemExit(f"Candidate species CSV missing columns: {sorted(missing)}")

    species_map: dict[str, str] = {}
    for _, row in candidates.iterrows():
        tree_label = row.get("tree_label")
        species_name = row.get("species_name")
        if nonempty(tree_label):
            label = str(tree_label).strip()
            species_map[normalize_species_name(label)] = label
        if nonempty(species_name) and nonempty(tree_label):
            species_map[normalize_species_name(species_name)] = str(tree_label).strip()
    return species_map


def has_positive_abundance(row: pd.Series, abundance_columns: list[str]) -> bool:
    for column in abundance_columns:
        if column not in row:
            continue
        try:
            if float(row[column]) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def build_groups(
    fishglob_csv: Path,
    species_map: dict[str, str],
    group_column: str,
    species_column: str,
    chunksize: int,
    encoding: str,
    require_positive_abundance: bool,
    abundance_columns: list[str],
) -> tuple[dict[str, set[str]], dict[str, object]]:
    groups: dict[str, set[str]] = defaultdict(set)
    unmatched_species = Counter()
    matched_species = Counter()
    rows = 0
    usable_rows = 0

    usecols = {group_column, species_column, *abundance_columns}
    for chunk in pd.read_csv(
        fishglob_csv,
        usecols=lambda col: col in usecols,
        chunksize=chunksize,
        encoding=encoding,
    ):
        missing = {group_column, species_column} - set(chunk.columns)
        if missing:
            raise SystemExit(f"FISHGLOB CSV missing columns: {sorted(missing)}")
        rows += int(len(chunk))
        for _, row in chunk.iterrows():
            group_value = row.get(group_column)
            species_value = row.get(species_column)
            if not nonempty(group_value) or not nonempty(species_value):
                continue
            if require_positive_abundance and not has_positive_abundance(row, abundance_columns):
                continue
            species_key = normalize_species_name(species_value)
            tree_label = species_map.get(species_key)
            if tree_label is None:
                unmatched_species[str(species_value).strip()] += 1
                continue
            group_id = f"FISHGLOB_{group_column}_{str(group_value).strip()}"
            groups[group_id].add(tree_label)
            matched_species[tree_label] += 1
            usable_rows += 1

    manifest = {
        "source": "FISHGLOB_public",
        "fishglob_csv": str(fishglob_csv),
        "encoding": encoding,
        "group_column": group_column,
        "species_column": species_column,
        "rows_read": rows,
        "matched_rows": usable_rows,
        "matched_unique_tree_labels": len(matched_species),
        "unmatched_unique_names": len(unmatched_species),
        "top_unmatched_names": unmatched_species.most_common(50),
        "top_matched_tree_labels": matched_species.most_common(50),
        "require_positive_abundance": require_positive_abundance,
        "abundance_columns": abundance_columns,
    }
    return groups, manifest


def attach_sequences(
    groups: dict[str, set[str]],
    train_sequences: dict[str, list[str]],
    min_species_per_group: int,
    max_groups: int | None,
) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    for group_id, labels in sorted(groups.items()):
        if len(labels) < min_species_per_group:
            continue
        output[group_id] = {
            label: train_sequences.get(label, [])
            for label in sorted(labels)
        }
        if max_groups is not None and len(output) >= max_groups:
            break
    return output


def summarize_groups(groups: dict[str, dict[str, list[str]]]) -> dict[str, object]:
    sizes = [len(group) for group in groups.values()]
    species = sorted({label for group in groups.values() for label in group})
    with_reference = sorted({
        label
        for group in groups.values()
        for label, sequences in group.items()
        if len(sequences) > 0
    })
    return {
        "groups": len(groups),
        "unique_species": len(species),
        "unique_species_with_reference_sequences": len(with_reference),
        "min_group_species": min(sizes) if sizes else 0,
        "median_group_species": float(pd.Series(sizes).median()) if sizes else 0.0,
        "max_group_species": max(sizes) if sizes else 0,
    }


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fishglob-csv", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/edna/real_edna_queries/global_tropical_multisource_teleo"),
        help="Directory containing candidate_species.csv and train_species_sequences.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/edna/cooccurrence_inputs/stalder_public"),
    )
    parser.add_argument("--output-name", default="fishglob_public_taxdna_cooccurrence.json")
    parser.add_argument("--group-column", default="haul_id")
    parser.add_argument("--species-column", default="accepted_name")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--encoding", default="latin1")
    parser.add_argument("--min-species-per-group", type=int, default=2)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--require-positive-abundance", action="store_true")
    parser.add_argument(
        "--abundance-columns",
        nargs="*",
        default=["num", "num_cpue", "num_cpua", "wgt", "wgt_cpue", "wgt_cpua"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_species_csv = args.input_dir / "candidate_species.csv"
    train_species_sequences = args.input_dir / "train_species_sequences.json"
    species_map = load_candidate_map(candidate_species_csv)
    train_sequences = load_species_sequences(train_species_sequences)

    raw_groups, source_manifest = build_groups(
        fishglob_csv=args.fishglob_csv,
        species_map=species_map,
        group_column=args.group_column,
        species_column=args.species_column,
        chunksize=args.chunksize,
        encoding=args.encoding,
        require_positive_abundance=args.require_positive_abundance,
        abundance_columns=args.abundance_columns,
    )
    groups = attach_sequences(
        raw_groups,
        train_sequences=train_sequences,
        min_species_per_group=args.min_species_per_group,
        max_groups=args.max_groups,
    )

    output_path = args.output_dir / args.output_name
    write_json(output_path, groups)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output_path),
        "input_dir": str(args.input_dir),
        "candidate_species_csv": str(candidate_species_csv),
        "train_species_sequences": str(train_species_sequences),
        "min_species_per_group": args.min_species_per_group,
        "max_groups": args.max_groups,
        "source": source_manifest,
        "summary": summarize_groups(groups),
        "notes": [
            "Output follows TAXDNA's co-occurrence JSON shape: group_id -> tree_label -> reference sequences or empty list.",
            "FISHGLOB rows are public scientific bottom-trawl survey species observations grouped by haul_id by default.",
            "This reconstructs a public FISHGLOB co-occurrence source; it is not the unreleased official TAXDNA FISHGLOB.json.",
        ],
    }
    manifest_path = args.output_dir / "fishglob_public_taxdna_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
