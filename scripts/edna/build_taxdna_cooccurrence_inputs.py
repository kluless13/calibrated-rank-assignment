#!/usr/bin/env python3
"""Build TAXDNA-style co-occurrence JSON inputs from local ecological tables."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RLS_META_COLUMNS = {
    "SurveyID",
    "Station",
    "SiteCode",
    "Site name",
    "SiteName",
    "SiteLat",
    "SiteLong",
    "Country",
    "State",
    "Location",
    "Ecoregion",
    "Province",
    "province",
    "Realm",
    "realm",
    "SurveyDate",
    "Depth",
    "site35",
    "site30",
    "site25",
    "site20",
    "site10",
    "site5",
    "coral_cover",
}


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def load_species_sequences(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text())
    return {
        str(label).strip(): [str(seq).strip().upper() for seq in sequences if str(seq).strip()]
        for label, sequences in raw.items()
    }


def load_species_map(candidate_species_csv: Path) -> dict[str, str]:
    candidates = pd.read_csv(candidate_species_csv)
    species_map: dict[str, str] = {}
    for _, row in candidates.iterrows():
        species_name = row.get("species_name")
        tree_label = row.get("tree_label")
        if nonempty(species_name) and nonempty(tree_label):
            species_map[str(species_name).strip()] = str(tree_label).strip()
    return species_map


def attach_sequences(labels: list[str], train_sequences: dict[str, list[str]]) -> dict[str, list[str]]:
    return {label: train_sequences.get(label, []) for label in sorted(set(labels))}


def filter_min_species(groups: dict[str, dict[str, list[str]]], min_species: int) -> dict[str, dict[str, list[str]]]:
    return {
        group_id: species
        for group_id, species in groups.items()
        if len(species) >= min_species
    }


def build_rls_groups(
    rls_species_csv: Path,
    species_map: dict[str, str],
    train_sequences: dict[str, list[str]],
    group_column: str,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, object]]:
    rls = pd.read_csv(rls_species_csv, sep=";", encoding="utf-8-sig")
    if group_column not in rls.columns:
        raise SystemExit(f"RLS table is missing group column {group_column!r}")
    species_columns = [col for col in rls.columns if col not in RLS_META_COLUMNS and col in species_map]
    if not species_columns:
        raise SystemExit(f"No RLS species columns match candidate species in {rls_species_csv}")

    grouped_counts: dict[str, Counter[str]] = {}
    survey_rows = 0
    for _, row in rls.iterrows():
        group_value = row.get(group_column)
        if not nonempty(group_value):
            continue
        labels = []
        for col in species_columns:
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                labels.append(species_map[col])
        if labels:
            survey_rows += 1
            grouped_counts.setdefault(str(group_value), Counter()).update(labels)

    groups = {
        f"RLS_{group_column}_{group_id}": attach_sequences(list(counts), train_sequences)
        for group_id, counts in grouped_counts.items()
    }
    manifest = {
        "source": "RLS",
        "rls_species_csv": str(rls_species_csv),
        "group_column": group_column,
        "matched_species_columns": len(species_columns),
        "survey_rows_with_candidates": survey_rows,
        "raw_group_count": len(groups),
    }
    return groups, manifest


def build_obis_groups(
    obis_prior_counts: Path,
    train_sequences: dict[str, list[str]],
    site_column: str,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, object]]:
    counts = pd.read_csv(obis_prior_counts)
    required = {"site_value", "tree_label", "occurrence_count"}
    missing = sorted(required - set(counts.columns))
    if missing:
        raise SystemExit(f"OBIS prior table missing columns: {missing}")

    grouped: dict[str, list[str]] = {}
    for _, row in counts.iterrows():
        try:
            occurrence_count = float(row.get("occurrence_count", 0.0))
        except (TypeError, ValueError):
            occurrence_count = 0.0
        if occurrence_count <= 0 or not nonempty(row.get("site_value")) or not nonempty(row.get("tree_label")):
            continue
        site_value = str(row["site_value"]).strip()
        grouped.setdefault(site_value, []).append(str(row["tree_label"]).strip())

    groups = {
        f"OBIS_{site_column}_{site_value}": attach_sequences(labels, train_sequences)
        for site_value, labels in grouped.items()
    }
    manifest = {
        "source": "OBIS",
        "obis_prior_counts": str(obis_prior_counts),
        "site_column": site_column,
        "raw_group_count": len(groups),
        "prior_rows": int(len(counts)),
    }
    return groups, manifest


def summarize_groups(groups: dict[str, dict[str, list[str]]]) -> dict[str, object]:
    sizes = [len(species) for species in groups.values()]
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
    parser.add_argument("--input-dir", type=Path, default=Path("data/edna/real_edna_queries/global_tropical_multisource_teleo"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/edna/cooccurrence_inputs/taxdna_ssm"))
    parser.add_argument(
        "--rls-species-csv",
        type=Path,
        default=Path("/Users/kluless/Downloads/Global_eDNA/data/RLS/RLS_species_NEW.csv"),
    )
    parser.add_argument(
        "--obis-prior-counts",
        type=Path,
        default=Path("data/edna/raw/real_edna/global_obis_range_prior_site20_pad05/obis_site_prior_counts.csv"),
    )
    parser.add_argument("--rls-group-column", default="SurveyID")
    parser.add_argument("--obis-site-column", default="site20")
    parser.add_argument("--min-species-per-group", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_sequences = load_species_sequences(args.input_dir / "train_species_sequences.json")
    species_map = load_species_map(args.input_dir / "candidate_species.csv")

    rls_groups, rls_manifest = build_rls_groups(
        rls_species_csv=args.rls_species_csv,
        species_map=species_map,
        train_sequences=train_sequences,
        group_column=args.rls_group_column,
    )
    obis_groups, obis_manifest = build_obis_groups(
        obis_prior_counts=args.obis_prior_counts,
        train_sequences=train_sequences,
        site_column=args.obis_site_column,
    )

    rls_groups = filter_min_species(rls_groups, args.min_species_per_group)
    obis_groups = filter_min_species(obis_groups, args.min_species_per_group)
    combined_groups = {**rls_groups, **obis_groups}

    outputs = {
        "rls": args.output_dir / "rls_taxdna_cooccurrence.json",
        "obis": args.output_dir / "obis_taxdna_cooccurrence.json",
        "combined": args.output_dir / "rls_obis_taxdna_cooccurrence.json",
    }
    write_json(outputs["rls"], rls_groups)
    write_json(outputs["obis"], obis_groups)
    write_json(outputs["combined"], combined_groups)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "candidate_species": str(args.input_dir / "candidate_species.csv"),
        "train_species_sequences": str(args.input_dir / "train_species_sequences.json"),
        "output_dir": str(args.output_dir),
        "min_species_per_group": args.min_species_per_group,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "rls_source": rls_manifest,
        "obis_source": obis_manifest,
        "summaries": {
            "rls": summarize_groups(rls_groups),
            "obis": summarize_groups(obis_groups),
            "combined": summarize_groups(combined_groups),
        },
        "notes": [
            "Files follow TAXDNA's co-occurrence JSON shape: location -> species tree label -> known reference sequences or empty list.",
            "RLS groups are empirical visual survey assemblages.",
            "OBIS groups are site-cell occurrence/range priors and are not exhaustive because some cells hit the fetch cap.",
            "Do not train co-occurrence on Global_eDNA validation labels when evaluating Global_eDNA; keep those labels for validation.",
        ],
    }
    manifest_path = args.output_dir / "taxdna_cooccurrence_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["summaries"], indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
