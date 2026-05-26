#!/usr/bin/env python3
"""Audit real eDNA occurrence labels against tree and 12S reference inputs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import dendropy
import pandas as pd


def read_tree_labels(tree_file: Path) -> set[str]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick", preserve_underscores=True)
    return {
        leaf.taxon.label.strip("'\"")
        for leaf in tree.leaf_node_iter()
        if leaf.taxon and leaf.taxon.label
    }


def clean_species_name(row: pd.Series) -> str | None:
    candidates = [
        row.get("species"),
        row.get("acceptedScientificName"),
        row.get("scientificName"),
    ]
    for value in candidates:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        text = text.replace("(", " ").replace(")", " ").replace(",", " ")
        parts = text.split()
        if len(parts) >= 2 and parts[0][0].isupper() and parts[1][0].islower():
            return f"{parts[0]} {parts[1]}"
    return None


def tree_label(species_name: str | None) -> str | None:
    if not species_name:
        return None
    return species_name.replace(" ", "_")


def load_reference_species(input_dir: Path) -> set[str]:
    path = input_dir / "species_sequences.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).keys())


def load_input_sets(input_dirs: list[Path]) -> dict[str, set[str]]:
    result = {}
    for input_dir in input_dirs:
        result[input_dir.name] = load_reference_species(input_dir)
    return result


def audit_source(csv_path: Path, tree_labels: set[str], input_sets: dict[str, set[str]]) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(csv_path)
    df["normalized_species_name"] = df.apply(clean_species_name, axis=1)
    df["tree_label"] = df["normalized_species_name"].map(tree_label)
    df["is_species_level"] = df["tree_label"].notna()
    df["is_in_tree"] = df["tree_label"].isin(tree_labels)

    source_id = csv_path.name.removesuffix(".records.csv")
    rows = []
    base = {
        "source_id": source_id,
        "records": int(len(df)),
        "events": int(df["eventID"].nunique(dropna=True)) if "eventID" in df.columns else 0,
        "species_level_records": int(df["is_species_level"].sum()),
        "tree_matched_records": int(df["is_in_tree"].sum()),
        "unique_species_level_names": int(df["normalized_species_name"].nunique(dropna=True)),
        "unique_tree_matched_species": int(df.loc[df["is_in_tree"], "tree_label"].nunique(dropna=True)),
    }
    for input_name, species in input_sets.items():
        in_reference = df["tree_label"].isin(species)
        zero_shot_candidate = df["is_in_tree"] & ~in_reference
        rows.append({
            **base,
            "input_set": input_name,
            "reference_species": len(species),
            "records_in_reference_species": int(in_reference.sum()),
            "records_tree_zero_shot_candidate": int(zero_shot_candidate.sum()),
            "unique_reference_species_seen": int(df.loc[in_reference, "tree_label"].nunique(dropna=True)),
            "unique_tree_zero_shot_candidate_species": int(
                df.loc[zero_shot_candidate, "tree_label"].nunique(dropna=True)
            ),
        })

    return df, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--occurrence-dir",
        type=Path,
        default=Path("data/edna/raw/real_edna/occurrences"),
        help="Directory with *.records.csv occurrence tables.",
    )
    parser.add_argument(
        "--tree-file",
        type=Path,
        default=Path("data/phylo/actinopt_12k_treePL.tre"),
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("data/edna/stalder_inputs/multisource"),
            Path("data/edna/stalder_inputs/multisource_teleo"),
            Path("data/edna/stalder_inputs/mitohelper_full_teleo"),
            Path("data/edna/stalder_inputs/rcrux_cleaned"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/edna/real_edna_overlap"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_labels = read_tree_labels(args.tree_file)
    input_sets = load_input_sets(args.input_dirs)

    all_rows = []
    for csv_path in sorted(args.occurrence_dir.glob("*.records.csv")):
        augmented, rows = audit_source(csv_path, tree_labels, input_sets)
        augmented.to_csv(args.output_dir / f"{csv_path.stem}.augmented.csv", index=False)
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    summary_path = args.output_dir / "real_edna_occurrence_overlap.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "occurrence_dir": str(args.occurrence_dir),
        "tree_file": str(args.tree_file),
        "input_dirs": [str(path) for path in args.input_dirs],
        "summary_csv": str(summary_path),
        "tree_tip_count": len(tree_labels),
        "rows": summary.to_dict(orient="records"),
    }
    manifest_path = args.output_dir / "real_edna_occurrence_overlap.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {summary_path} and {manifest_path}.")


if __name__ == "__main__":
    main()
