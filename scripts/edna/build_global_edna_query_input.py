#!/usr/bin/env python3
"""Build open-candidate query inputs from the Global_eDNA Teleo ASV tables."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CONTRACT_FILES = [
    "species_sequences.json",
    "train_species_sequences.json",
    "species_info.json",
    "candidate_species.csv",
    "val_species.json",
    "test_species.json",
    "eval_c_species.json",
]


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def processid(region: object, table_prefix: object, amplicon_id: object) -> str:
    return f"global_edna:{region}:{table_prefix}:{amplicon_id}"


def split_pipe(value: object) -> list[str]:
    if not nonempty(value):
        return []
    return [part for part in str(value).split("|") if part]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input-dir", type=Path, default=Path("data/edna/stalder_inputs/multisource_teleo"))
    parser.add_argument("--global-edna-dir", type=Path, default=Path("data/edna/raw/real_edna/global_tropical"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/edna/real_edna_queries/global_tropical_multisource_teleo"))
    parser.add_argument("--include-other-table", action="store_true")
    parser.add_argument("--min-identity", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in CONTRACT_FILES:
        src = args.base_input_dir / filename
        if src.exists():
            shutil.copy2(src, args.output_dir / filename)

    candidates = pd.read_csv(args.output_dir / "candidate_species.csv")
    candidate_labels = set(candidates["tree_label"].dropna().astype(str))

    taxonomy_path = args.global_edna_dir / "global_edna_teleo_asv_taxonomy.csv"
    presence_path = args.global_edna_dir / "global_edna_sample_species_presence.csv"
    taxonomy = pd.read_csv(taxonomy_path)
    presence = pd.read_csv(presence_path)

    taxonomy = taxonomy[taxonomy["rank"].astype(str).str.lower() == "species"].copy()
    taxonomy = taxonomy[taxonomy["tree_label"].isin(candidate_labels)].copy()
    taxonomy = taxonomy[~taxonomy["is_blank_table"].astype(bool)].copy()
    if not args.include_other_table:
        taxonomy = taxonomy[~taxonomy["is_other_table"].astype(bool)].copy()
    if "best_identity:db_embl_std" in taxonomy.columns:
        identity = pd.to_numeric(taxonomy["best_identity:db_embl_std"], errors="coerce").fillna(0)
        taxonomy = taxonomy[identity >= args.min_identity].copy()

    taxonomy["processid"] = [
        processid(row.source_region_dir, row.table_prefix, row.amplicon_id)
        for row in taxonomy.itertuples(index=False)
    ]
    taxonomy["source"] = "global_tropical_dryad"
    taxonomy["source_id"] = "global_tropical_dryad"
    taxonomy["eventID"] = taxonomy["source_region_dir"].astype(str) + ":" + taxonomy["table_prefix"].astype(str)
    taxonomy["nucleotides"] = taxonomy["sequence"].astype(str)

    query_cols = [
        "processid",
        "source",
        "source_id",
        "eventID",
        "source_region_dir",
        "table_prefix",
        "amplicon_id",
        "tree_label",
        "species_name",
        "genus_name",
        "family_name",
        "order_name",
        "taxid",
        "seq_len",
        "nucleotides",
        "best_identity:db_embl_std",
        "best_match:db_embl_std",
    ]
    zero_shot = taxonomy[[col for col in query_cols if col in taxonomy.columns]].drop_duplicates("processid")
    zero_shot = zero_shot.sort_values(["source_region_dir", "table_prefix", "tree_label", "amplicon_id"])
    zero_shot.to_csv(args.output_dir / "zero_shot_queries.csv", index=False)

    processid_by_region_amplicon = {
        (str(row.source_region_dir), str(row.amplicon_id)): str(row.processid)
        for row in zero_shot.itertuples(index=False)
    }
    map_rows = []
    filtered_presence = presence[presence["tree_label"].isin(candidate_labels)].copy()
    filtered_presence = filtered_presence[filtered_presence["is_in_tree"].astype(bool)].copy()
    if not args.include_other_table and "is_other_table_observation" in filtered_presence.columns:
        filtered_presence = filtered_presence[~filtered_presence["is_other_table_observation"].astype(bool)].copy()
    for row in filtered_presence.itertuples(index=False):
        region = str(row.source_region_dir)
        amplicon_ids = split_pipe(getattr(row, "amplicon_ids", ""))
        matched = [
            processid_by_region_amplicon[(region, amplicon)]
            for amplicon in amplicon_ids
            if (region, amplicon) in processid_by_region_amplicon
        ]
        for query_processid in sorted(set(matched)):
            map_rows.append({
                "sample_id": row.sample_id,
                "source_region_dir": row.source_region_dir,
                "region_label": getattr(row, "region_label", None),
                "site_name": getattr(row, "site_name", None),
                "province": getattr(row, "province", None),
                "country": getattr(row, "country", None),
                "latitude_start_clean": getattr(row, "latitude_start_clean", None),
                "longitude_start_clean": getattr(row, "longitude_start_clean", None),
                "site35": getattr(row, "site35", None),
                "site30": getattr(row, "site30", None),
                "site25": getattr(row, "site25", None),
                "site20": getattr(row, "site20", None),
                "site10": getattr(row, "site10", None),
                "site5": getattr(row, "site5", None),
                "query_processid": query_processid,
                "true_tree_label": row.tree_label,
                "true_species_name": row.species_name,
                "true_genus_name": row.genus_name,
                "true_family_name": row.family_name,
                "true_order_name": row.order_name,
                "read_count": row.read_count,
                "asv_count": row.asv_count,
            })
    sample_map = pd.DataFrame(map_rows)
    sample_map.to_csv(args.output_dir / "sample_query_map.csv", index=False)

    base_manifest = json.loads((args.base_input_dir / "manifest.json").read_text())
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "base_input_dir": str(args.base_input_dir),
        "global_edna_dir": str(args.global_edna_dir),
        "taxonomy_csv": str(taxonomy_path),
        "presence_csv": str(presence_path),
        "output_dir": str(args.output_dir),
        "include_other_table": args.include_other_table,
        "min_identity": args.min_identity,
        "candidate_count": int(len(candidates)),
        "query_asvs": int(len(zero_shot)),
        "query_species": int(zero_shot["tree_label"].nunique()),
        "sample_query_rows": int(len(sample_map)),
        "samples": int(sample_map["sample_id"].nunique()) if len(sample_map) else 0,
        "base_manifest": {
            "data_dir": base_manifest.get("data_dir"),
            "amplicon": base_manifest.get("amplicon"),
            "reference_species": base_manifest.get("species_sequences_json_species"),
            "reference_sequences": base_manifest.get("species_sequences_json_sequences"),
        },
        "outputs": {
            "zero_shot_queries_csv": str(args.output_dir / "zero_shot_queries.csv"),
            "sample_query_map_csv": str(args.output_dir / "sample_query_map.csv"),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {args.output_dir}: {len(zero_shot)} ASV queries, "
        f"{zero_shot['tree_label'].nunique()} species, {sample_map['sample_id'].nunique() if len(sample_map) else 0} samples."
    )


if __name__ == "__main__":
    main()
