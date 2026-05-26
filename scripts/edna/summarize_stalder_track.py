#!/usr/bin/env python3
"""Build compact status tables for the Stalder/TAXDNA eDNA track."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def input_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        rows.append({
            "input_set": manifest_path.parent.name,
            "amplicon": manifest.get("amplicon"),
            "tree_tip_count": manifest.get("tree_tip_count"),
            "reference_species": manifest.get("species_sequences_json_species"),
            "reference_sequences": manifest.get("species_sequences_json_sequences"),
            "zero_shot_eval_c_rows": manifest.get("zero_shot_eval_c_rows"),
            "zero_shot_eval_c_species": manifest.get("zero_shot_eval_c_species"),
            "rows_matching_tree": manifest.get("rows_matching_tree"),
            "dropped_retained_species_not_in_tree": manifest.get("dropped_retained_species_not_in_tree"),
            "manifest": str(manifest_path),
        })
    return rows


def real_edna_rows(overlap_csv: Path) -> list[dict[str, object]]:
    if not overlap_csv.exists():
        return []
    return pd.read_csv(overlap_csv).to_dict(orient="records")


def prior_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for manifest_path in sorted(root.glob("*/regional_candidate_priors_manifest.json")):
        manifest = read_json(manifest_path)
        rows.append({
            "prior_set": manifest_path.parent.name,
            "radius_km": manifest.get("radius_km"),
            "same_source_only": manifest.get("same_source_only"),
            "occurrence_records_used": manifest.get("occurrence_records_used"),
            "event_count": manifest.get("event_count"),
            "events_with_any_regional_candidate": manifest.get("events_with_any_regional_candidate"),
            "median_regional_candidate_count": manifest.get("median_regional_candidate_count"),
            "manifest": str(manifest_path),
        })
    return rows


def template_rows(root: Path) -> list[dict[str, object]]:
    rows = []
    for manifest_path in sorted(root.glob("*/zero_shot_prediction_template_manifest.json")):
        manifest = read_json(manifest_path)
        rows.append({
            "template_set": manifest_path.parent.name,
            "query_count": manifest.get("query_count"),
            "candidate_count": manifest.get("candidate_count"),
            "template_csv": manifest.get("template_csv"),
            "manifest": str(manifest_path),
        })
    return rows


def global_edna_rows(manifest_path: Path) -> list[dict[str, object]]:
    manifest = read_json(manifest_path)
    if not manifest:
        return []

    rows: list[dict[str, object]] = []
    counts = manifest.get("counts", {})
    rows.append({
        "dataset": "global_tropical_dryad",
        "input_set": "all",
        "source_files": counts.get("source_files"),
        "source_size_bytes": counts.get("source_size_bytes"),
        "metadata_samples": counts.get("metadata_samples"),
        "library_sample_rows": counts.get("library_sample_rows"),
        "teleo_taxonomy_rows": counts.get("teleo_taxonomy_rows"),
        "teleo_species_taxonomy_rows": counts.get("teleo_species_taxonomy_rows"),
        "sample_species_presence_rows": counts.get("sample_species_presence_rows"),
        "unique_samples_with_species": counts.get("unique_samples_with_species"),
        "unique_species_observed": counts.get("unique_species_observed"),
        "other_table_presence_rows": counts.get("other_table_presence_rows"),
        "non_tree_presence_rows": counts.get("non_tree_presence_rows"),
        "tree_matched_presence_rows": counts.get("tree_matched_presence_rows"),
        "tree_matched_species": counts.get("tree_matched_species"),
        "tree_matched_non_other_presence_rows": counts.get("tree_matched_non_other_presence_rows"),
        "tree_matched_non_other_samples": counts.get("tree_matched_non_other_samples"),
        "tree_matched_non_other_species": counts.get("tree_matched_non_other_species"),
        "presence_rows_in_reference": None,
        "presence_rows_tree_zero_shot_candidate": None,
        "unique_species_in_reference": None,
        "unique_species_tree_zero_shot_candidate": None,
        "non_other_presence_rows_in_reference": None,
        "non_other_presence_rows_tree_zero_shot_candidate": None,
        "non_other_unique_species_in_reference": None,
        "non_other_unique_species_tree_zero_shot_candidate": None,
        "manifest": str(manifest_path),
    })
    for input_set, overlap in sorted(manifest.get("reference_overlap", {}).items()):
        rows.append({
            "dataset": "global_tropical_dryad",
            "input_set": input_set,
            "source_files": counts.get("source_files"),
            "source_size_bytes": counts.get("source_size_bytes"),
            "metadata_samples": counts.get("metadata_samples"),
            "library_sample_rows": counts.get("library_sample_rows"),
            "teleo_taxonomy_rows": counts.get("teleo_taxonomy_rows"),
            "teleo_species_taxonomy_rows": counts.get("teleo_species_taxonomy_rows"),
            "sample_species_presence_rows": counts.get("sample_species_presence_rows"),
            "unique_samples_with_species": counts.get("unique_samples_with_species"),
            "unique_species_observed": counts.get("unique_species_observed"),
            "other_table_presence_rows": counts.get("other_table_presence_rows"),
            "non_tree_presence_rows": counts.get("non_tree_presence_rows"),
            "tree_matched_presence_rows": counts.get("tree_matched_presence_rows"),
            "tree_matched_species": counts.get("tree_matched_species"),
            "tree_matched_non_other_presence_rows": counts.get("tree_matched_non_other_presence_rows"),
            "tree_matched_non_other_samples": counts.get("tree_matched_non_other_samples"),
            "tree_matched_non_other_species": counts.get("tree_matched_non_other_species"),
            "presence_rows_in_reference": overlap.get("presence_rows_in_reference"),
            "presence_rows_tree_zero_shot_candidate": overlap.get("presence_rows_tree_zero_shot_candidate"),
            "unique_species_in_reference": overlap.get("unique_species_in_reference"),
            "unique_species_tree_zero_shot_candidate": overlap.get("unique_species_tree_zero_shot_candidate"),
            "non_other_presence_rows_in_reference": overlap.get("non_other_presence_rows_in_reference"),
            "non_other_presence_rows_tree_zero_shot_candidate": overlap.get("non_other_presence_rows_tree_zero_shot_candidate"),
            "non_other_unique_species_in_reference": overlap.get("non_other_unique_species_in_reference"),
            "non_other_unique_species_tree_zero_shot_candidate": overlap.get("non_other_unique_species_tree_zero_shot_candidate"),
            "manifest": str(manifest_path),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/edna/stalder_track_status"))
    parser.add_argument("--input-root", type=Path, default=Path("data/edna/stalder_inputs"))
    parser.add_argument("--prior-root", type=Path, default=Path("data/edna/real_edna_priors"))
    parser.add_argument("--template-root", type=Path, default=Path("results/edna/zero_shot_templates"))
    parser.add_argument(
        "--overlap-csv",
        type=Path,
        default=Path("results/edna/real_edna_overlap/real_edna_occurrence_overlap.csv"),
    )
    parser.add_argument(
        "--stalder-asset-manifest",
        type=Path,
        default=Path("data/edna/raw/stalder_taxdna_manifest.json"),
    )
    parser.add_argument(
        "--global-edna-manifest",
        type=Path,
        default=Path("data/edna/raw/real_edna/global_tropical/global_edna_ingest_manifest.json"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "input_sets": input_rows(args.input_root),
        "real_edna_overlap": real_edna_rows(args.overlap_csv),
        "global_real_edna": global_edna_rows(args.global_edna_manifest),
        "regional_priors": prior_rows(args.prior_root),
        "prediction_templates": template_rows(args.template_root),
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(args.output_dir / f"{name}.csv", index=False)

    asset_manifest = read_json(args.stalder_asset_manifest)
    status = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tables": {name: str(args.output_dir / f"{name}.csv") for name in tables},
        "counts": {name: len(rows) for name, rows in tables.items()},
        "stalder_assets": {
            "manifest": str(args.stalder_asset_manifest),
            "lfs_pointer_files": asset_manifest.get("summary", {}).get("lfs_pointer_files"),
            "lfs_target_bytes": asset_manifest.get("summary", {}).get("lfs_target_bytes"),
            "bytes_on_disk": asset_manifest.get("summary", {}).get("bytes_on_disk"),
        },
        "next_gpu_jobs": [
            "Train 12S PhyloMamba on multisource_teleo and multisource input sets.",
            "Train TAXDNA-style CNN baseline on the same tree embeddings and input sets.",
            "Evaluate generated prediction CSVs with eval_zero_shot_candidate_predictions.py.",
            "Apply eval_regional_prior_rerank.py for sequence-plus-prior ablation on real-event predictions.",
        ],
    }
    status_path = args.output_dir / "stalder_track_status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {status_path}")


if __name__ == "__main__":
    main()
