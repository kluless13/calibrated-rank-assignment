#!/usr/bin/env python3
"""Ingest the manually downloaded Dryad Global_eDNA directory.

The raw Dryad payload is large and contains duplicated paths. This script keeps
the raw data in place and writes compact, reproducible MarineMamba inputs:

- file inventory
- global sample metadata
- per-library sample mapping
- Teleo ASV taxonomy table
- sample x species read-count/presence table
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FIXED_MOTU_COLUMNS = [
    "OTU",
    "total",
    "cloud",
    "amplicon",
    "length",
    "abundance",
    "chimera",
    "spread",
    "quality",
    "sequence",
    "identity",
    "taxonomy",
    "references",
    "amplicon_id",
]


def clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE"}:
        return None
    return text


def clean_seq(value: object) -> str:
    return "".join(ch for ch in str(value).upper().strip() if ch in "ACGTN")


def tree_label(species_name: object) -> str | None:
    text = clean_text(species_name)
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    return f"{parts[0]}_{parts[1]}"


def file_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "suffix": path.suffix,
            "parent": path.parent.relative_to(root).as_posix(),
        })
    return pd.DataFrame(rows)


def read_global_metadata(root: Path) -> pd.DataFrame:
    path = root / "metadata" / "Metadata_eDNA_global_V6.csv"
    df = pd.read_csv(path, sep=";", dtype=str, encoding_errors="replace")
    df["source_file"] = path.relative_to(root).as_posix()
    return df


def read_library_samples(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "data" / "eDNA" / "swarm").glob("*/metadata/all_samples.csv")):
        df = pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["plate_well", "sequencing_run", "sample_id", "region_label", "marker"],
            dtype=str,
            encoding_errors="replace",
        )
        df["source_region_dir"] = path.parents[1].name
        df["source_file"] = path.relative_to(root).as_posix()
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def teleo_taxonomy_files(root: Path) -> list[Path]:
    return sorted((root / "data" / "eDNA" / "swarm").glob("*/*_teleo_ecotag_ncbi_motu.csv"))


def matching_table_file(taxonomy_path: Path) -> Path:
    return taxonomy_path.with_name(taxonomy_path.name.replace("_teleo_ecotag_ncbi_motu.csv", "_teleo_table_motu.csv"))


def read_teleo_taxonomy(root: Path) -> pd.DataFrame:
    frames = []
    for path in teleo_taxonomy_files(root):
        df = pd.read_csv(path, sep="\t", dtype=str, encoding_errors="replace", low_memory=False)
        prefix = path.name.replace("_teleo_ecotag_ncbi_motu.csv", "")
        df["source_region_dir"] = path.parent.name
        df["table_prefix"] = prefix
        df["source_file"] = path.relative_to(root).as_posix()
        df["amplicon_id"] = df["definition"].map(clean_text)
        df["sequence"] = df["sequence"].map(clean_seq)
        df["seq_len"] = df["sequence"].str.len()
        df["tree_label"] = df["species_name"].map(tree_label)
        df["is_blank_table"] = prefix.lower().startswith("blank")
        df["is_other_table"] = prefix.lower().startswith("other")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    keep = [
        "source_region_dir",
        "table_prefix",
        "source_file",
        "amplicon_id",
        "rank",
        "scientific_name",
        "species_name",
        "genus_name",
        "family_name",
        "order_name",
        "taxid",
        "best_identity:db_embl_std",
        "best_match:db_embl_std",
        "sequence",
        "seq_len",
        "tree_label",
        "is_blank_table",
        "is_other_table",
    ]
    return pd.concat(frames, ignore_index=True)[keep]


def read_reference_sets() -> dict[str, set[str]]:
    refs = {}
    for name in ["multisource", "multisource_teleo", "mitohelper_full_teleo", "rcrux_cleaned"]:
        path = Path("data/edna/stalder_inputs") / name / "species_sequences.json"
        if path.exists():
            refs[name] = set(json.loads(path.read_text()).keys())
    return refs


def read_tree_labels() -> set[str]:
    candidate_path = Path("data/edna/stalder_inputs/multisource/candidate_species.csv")
    if candidate_path.exists():
        return set(pd.read_csv(candidate_path, usecols=["tree_label"])["tree_label"].dropna().astype(str))
    return set()


def sample_columns(table: pd.DataFrame) -> list[str]:
    fixed = set(FIXED_MOTU_COLUMNS)
    return [col for col in table.columns if col not in fixed]


def build_sample_species_presence(root: Path, taxonomy: pd.DataFrame) -> pd.DataFrame:
    tax_cols = [
        "source_region_dir",
        "table_prefix",
        "amplicon_id",
        "rank",
        "species_name",
        "genus_name",
        "family_name",
        "order_name",
        "taxid",
        "sequence",
        "seq_len",
        "tree_label",
        "is_blank_table",
        "is_other_table",
    ]
    tax = taxonomy[tax_cols].copy()
    tax = tax[tax["rank"].astype(str).str.lower() == "species"].copy()
    tax = tax[tax["tree_label"].notna()].copy()
    tax = tax.drop_duplicates(["source_region_dir", "table_prefix", "amplicon_id"])

    aggregate: dict[tuple[str, str, str], dict[str, object]] = {}
    processed_tables = []

    for tax_path in teleo_taxonomy_files(root):
        table_path = matching_table_file(tax_path)
        if not table_path.exists():
            continue
        prefix = tax_path.name.replace("_teleo_ecotag_ncbi_motu.csv", "")
        region = tax_path.parent.name
        if prefix.lower().startswith("blank"):
            continue

        table = pd.read_csv(table_path, sep="\t", dtype=str, encoding_errors="replace", low_memory=False)
        table["amplicon_id"] = table["amplicon"].map(clean_text)
        merged = table.merge(
            tax,
            how="inner",
            on="amplicon_id",
            suffixes=("_table", ""),
        )
        merged = merged[
            (merged["source_region_dir"] == region)
            & (merged["table_prefix"] == prefix)
        ].copy()
        if merged.empty:
            continue

        samples = sample_columns(table)
        samples = [sample for sample in samples if sample in merged.columns]
        counts = merged[samples].apply(pd.to_numeric, errors="coerce").fillna(0)
        matrix = counts.to_numpy(dtype=np.float64, copy=False)
        nz_rows, nz_cols = np.nonzero(matrix > 0)

        for row_idx, col_idx in zip(nz_rows, nz_cols):
            row = merged.iloc[int(row_idx)]
            sample_id = samples[int(col_idx)]
            label = str(row["tree_label"])
            key = (sample_id, label, region)
            count = float(matrix[int(row_idx), int(col_idx)])
            current = aggregate.setdefault(
                key,
                {
                    "sample_id": sample_id,
                    "tree_label": label,
                    "source_region_dir": region,
                    "species_name": row["species_name"],
                    "genus_name": row["genus_name"],
                    "family_name": row["family_name"],
                    "order_name": row["order_name"],
                    "taxid": row["taxid"],
                    "read_count": 0.0,
                    "asv_count": 0,
                    "table_prefixes": set(),
                    "amplicon_ids": set(),
                    "is_other_table_observation": False,
                },
            )
            current["read_count"] = float(current["read_count"]) + count
            current["asv_count"] = int(current["asv_count"]) + 1
            current["table_prefixes"].add(prefix)
            current["amplicon_ids"].add(row["amplicon_id"])
            current["is_other_table_observation"] = bool(current["is_other_table_observation"]) or prefix.lower().startswith("other")

        processed_tables.append({
            "source_region_dir": region,
            "table_prefix": prefix,
            "table_file": table_path.relative_to(root).as_posix(),
            "taxonomy_file": tax_path.relative_to(root).as_posix(),
            "motu_rows": int(len(table)),
            "species_motu_rows_after_join": int(len(merged)),
            "sample_columns": int(len(samples)),
            "positive_sample_asv_cells": int(len(nz_rows)),
        })

    rows = []
    for value in aggregate.values():
        row = dict(value)
        row["read_count"] = int(row["read_count"]) if float(row["read_count"]).is_integer() else row["read_count"]
        row["table_prefixes"] = "|".join(sorted(row["table_prefixes"]))
        row["amplicon_ids"] = "|".join(sorted(row["amplicon_ids"]))
        rows.append(row)
    presence = pd.DataFrame(rows)
    presence.attrs["processed_tables"] = processed_tables
    return presence


def attach_metadata(presence: pd.DataFrame, global_meta: pd.DataFrame, library_samples: pd.DataFrame) -> pd.DataFrame:
    out = presence.copy()
    if not library_samples.empty:
        lib_cols = ["sample_id", "plate_well", "sequencing_run", "region_label", "marker"]
        lib = library_samples[lib_cols].drop_duplicates("sample_id")
        out = out.merge(lib, how="left", on="sample_id")

    if not global_meta.empty and "code_spygen" in global_meta.columns:
        # Library sample IDs often append replicate/well suffixes, e.g. SPY180761_01,
        # while the field metadata table stores the base SPY code.
        out["code_spygen_base"] = (
            out["sample_id"]
            .astype(str)
            .str.extract(r"^(SPY\d+)", expand=False)
            .fillna(out["sample_id"].astype(str))
        )
        keep = [
            "code_spygen",
            "code_explo",
            "rep",
            "date",
            "station",
            "site_name",
            "province",
            "country",
            "project",
            "sample_type",
            "sample_method",
            "latitude_start_clean",
            "longitude_start_clean",
            "dist_to_coast",
            "site35",
            "site30",
            "site25",
            "site20",
            "site10",
            "site5",
        ]
        keep = [col for col in keep if col in global_meta.columns]
        meta = global_meta[keep].drop_duplicates("code_spygen")
        out = out.merge(meta, how="left", left_on="code_spygen_base", right_on="code_spygen")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/Users/kluless/Downloads/Global_eDNA"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/edna/raw/real_edna/global_tropical"),
    )
    args = parser.parse_args()

    if not args.source_dir.exists():
        raise SystemExit(f"Missing source dir: {args.source_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory = file_inventory(args.source_dir)
    global_meta = read_global_metadata(args.source_dir)
    library_samples = read_library_samples(args.source_dir)
    taxonomy = read_teleo_taxonomy(args.source_dir)
    presence = build_sample_species_presence(args.source_dir, taxonomy)
    processed_tables = presence.attrs.get("processed_tables", [])
    presence = attach_metadata(presence, global_meta, library_samples)

    tree_labels = read_tree_labels()
    reference_sets = read_reference_sets()
    if "tree_label" in presence.columns:
        presence["is_in_tree"] = presence["tree_label"].isin(tree_labels)
        for name, labels in reference_sets.items():
            presence[f"is_in_reference_{name}"] = presence["tree_label"].isin(labels)
            presence[f"is_tree_zero_shot_candidate_{name}"] = presence["is_in_tree"] & ~presence[f"is_in_reference_{name}"]
    if "tree_label" in taxonomy.columns:
        taxonomy["is_in_tree"] = taxonomy["tree_label"].isin(tree_labels)

    inventory_path = args.output_dir / "global_edna_file_inventory.csv"
    metadata_path = args.output_dir / "global_edna_sample_metadata.csv"
    library_path = args.output_dir / "global_edna_library_samples.csv"
    taxonomy_path = args.output_dir / "global_edna_teleo_asv_taxonomy.csv"
    presence_path = args.output_dir / "global_edna_sample_species_presence.csv"
    processed_tables_path = args.output_dir / "global_edna_processed_tables.csv"

    inventory.to_csv(inventory_path, index=False)
    global_meta.to_csv(metadata_path, index=False)
    library_samples.to_csv(library_path, index=False)
    taxonomy.to_csv(taxonomy_path, index=False)
    presence.to_csv(presence_path, index=False)
    pd.DataFrame(processed_tables).to_csv(processed_tables_path, index=False)

    has_presence_rows = len(presence) > 0
    tree_mask = presence["is_in_tree"] if "is_in_tree" in presence.columns else pd.Series(False, index=presence.index)
    other_mask = (
        presence["is_other_table_observation"].astype(bool)
        if "is_other_table_observation" in presence.columns
        else pd.Series(False, index=presence.index)
    )
    clean_tree_mask = tree_mask & ~other_mask

    reference_overlap = {}
    for name in reference_sets:
        ref_col = f"is_in_reference_{name}"
        zero_col = f"is_tree_zero_shot_candidate_{name}"
        if ref_col not in presence.columns:
            continue
        ref_mask = presence[ref_col].astype(bool)
        zero_mask = presence[zero_col].astype(bool)
        reference_overlap[name] = {
            "presence_rows_in_reference": int(ref_mask.sum()),
            "presence_rows_tree_zero_shot_candidate": int(zero_mask.sum()),
            "unique_species_in_reference": int(presence.loc[ref_mask, "tree_label"].nunique()),
            "unique_species_tree_zero_shot_candidate": int(presence.loc[zero_mask, "tree_label"].nunique()),
            "non_other_presence_rows_in_reference": int((ref_mask & ~other_mask).sum()),
            "non_other_presence_rows_tree_zero_shot_candidate": int((zero_mask & ~other_mask).sum()),
            "non_other_unique_species_in_reference": int(presence.loc[ref_mask & ~other_mask, "tree_label"].nunique()),
            "non_other_unique_species_tree_zero_shot_candidate": int(presence.loc[zero_mask & ~other_mask, "tree_label"].nunique()),
        }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "files": {
            "inventory": str(inventory_path),
            "sample_metadata": str(metadata_path),
            "library_samples": str(library_path),
            "teleo_asv_taxonomy": str(taxonomy_path),
            "sample_species_presence": str(presence_path),
            "processed_tables": str(processed_tables_path),
        },
        "counts": {
            "source_files": int(len(inventory)),
            "source_size_bytes": int(inventory["size_bytes"].sum()),
            "metadata_samples": int(global_meta["code_spygen"].nunique()) if "code_spygen" in global_meta.columns else int(len(global_meta)),
            "library_sample_rows": int(len(library_samples)),
            "teleo_taxonomy_rows": int(len(taxonomy)),
            "teleo_species_taxonomy_rows": int((taxonomy["rank"].astype(str).str.lower() == "species").sum()) if len(taxonomy) else 0,
            "sample_species_presence_rows": int(len(presence)),
            "unique_samples_with_species": int(presence["sample_id"].nunique()) if has_presence_rows else 0,
            "unique_species_observed": int(presence["tree_label"].nunique()) if has_presence_rows else 0,
            "other_table_presence_rows": int(other_mask.sum()) if has_presence_rows else 0,
            "non_tree_presence_rows": int((~tree_mask).sum()) if has_presence_rows else 0,
            "tree_matched_presence_rows": int(tree_mask.sum()) if has_presence_rows else 0,
            "tree_matched_species": int(presence.loc[tree_mask, "tree_label"].nunique()) if has_presence_rows else 0,
            "tree_matched_non_other_presence_rows": int(clean_tree_mask.sum()) if has_presence_rows else 0,
            "tree_matched_non_other_samples": int(presence.loc[clean_tree_mask, "sample_id"].nunique()) if has_presence_rows else 0,
            "tree_matched_non_other_species": int(presence.loc[clean_tree_mask, "tree_label"].nunique()) if has_presence_rows else 0,
        },
        "reference_overlap": reference_overlap,
        "notes": [
            "Raw Global_eDNA files remain in the Downloads source directory; this output is a compact derived index.",
            "Presence rows are species-level Teleo assignments only; Blank tables are excluded.",
            "Other tables are retained but marked with is_other_table_observation.",
        ],
    }
    manifest_path = args.output_dir / "global_edna_ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
