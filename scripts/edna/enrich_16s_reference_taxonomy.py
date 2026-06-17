#!/usr/bin/env python3
"""Fill taxonomy columns in the 16S reference table from existing references."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_16S = ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource"
SOURCE_TABLES = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_lookup(paths: list[Path]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            label = str(row.get("tree_label") or "")
            if not label:
                continue
            current = lookup.setdefault(label, {})
            for key in ("species_name", "genus_name", "family_name", "order_name"):
                value = str(row.get(key) or "")
                if value and value.lower() != "nan" and not current.get(key):
                    current[key] = value
    return lookup


def main() -> None:
    target = DEFAULT_16S / "candidate_species.csv"
    rows = read_csv(target)
    taxonomy_sources = [
        ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "candidate_species.csv",
        ROOT / "data" / "edna" / "stalder_inputs" / "multisource_teleo" / "candidate_species.csv",
        ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs" / "eval_c" / "candidate_species.csv",
        ROOT / "data" / "coi" / "stalder_inputs" / "fish_tree" / "candidate_species.csv",
    ]
    lookup = load_lookup(taxonomy_sources)
    enriched = []
    filled_family = 0
    filled_order = 0
    for row in rows:
        label = str(row.get("tree_label") or "")
        source = lookup.get(label, {})
        out = dict(row)
        for key in ("species_name", "genus_name", "family_name", "order_name"):
            if not str(out.get(key) or "").strip() and source.get(key):
                out[key] = source[key]
        filled_family += int(bool(str(out.get("family_name") or "").strip()))
        filled_order += int(bool(str(out.get("order_name") or "").strip()))
        enriched.append(out)

    fieldnames = ["tree_label", "species_name", "genus_name", "family_name", "order_name"]
    write_csv(target, enriched, fieldnames)
    write_csv(SOURCE_TABLES / "marker_16s_candidate_taxonomy_enrichment.csv", enriched, fieldnames)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_species": str(target),
        "n_species": len(enriched),
        "with_family": filled_family,
        "with_order": filled_order,
        "taxonomy_sources": [str(path) for path in taxonomy_sources],
    }
    (DEFAULT_16S / "taxonomy_enrichment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (SOURCE_TABLES / "marker_16s_candidate_taxonomy_enrichment_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
