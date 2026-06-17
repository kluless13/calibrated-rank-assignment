#!/usr/bin/env python3
"""Audit local 16S material for Tri-Marker MarkerMirror.

This does not download new data.  It answers a narrower question: do we already
have a usable local 16S species-reference table comparable to our COI and 12S
``species_sequences.json`` inputs, or only occurrence/sample metadata?
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"


def has_sequence_text(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().upper()
        return len(text) >= 20 and sum(base in "ACGTN" for base in text) / max(len(text), 1) > 0.8
    if isinstance(value, list):
        return any(has_sequence_text(item) for item in value)
    if isinstance(value, dict):
        return any(has_sequence_text(item) for item in value.values())
    return False


def inspect_species_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "kind": "species_json", "read_error": str(exc)}
    n_species = len(data) if isinstance(data, dict) else 0
    n_sequences = 0
    if isinstance(data, dict):
        for seqs in data.values():
            if isinstance(seqs, list):
                n_sequences += len(seqs)
    return {
        "path": str(path),
        "kind": "species_json",
        "species": n_species,
        "sequences": n_sequences,
        "has_sequence_text": n_sequences > 0,
    }


def inspect_occurrence_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "kind": "occurrence_json", "read_error": str(exc)}
    records = data if isinstance(data, list) else []
    markers = Counter()
    statuses = Counter()
    sequence_like = 0
    species = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        statuses[str(rec.get("occurrenceStatus") or "").upper()] += 1
        if rec.get("species"):
            species.add(str(rec["species"]))
        for entry in rec.get("nucleotideSequence") or []:
            if isinstance(entry, dict):
                markers[str(entry.get("marker") or entry.get("targetGene") or "")] += 1
        for ext_rows in (rec.get("extensions") or {}).values():
            if not isinstance(ext_rows, list):
                continue
            for row in ext_rows:
                if not isinstance(row, dict):
                    continue
                marker = row.get("https://w3id.org/mixs/0000044") or row.get("https://w3id.org/mixs/0000045")
                if marker:
                    markers[str(marker)] += 1
        if has_sequence_text(rec.get("nucleotideSequence")) or has_sequence_text(rec.get("associatedSequences")):
            sequence_like += 1
    return {
        "path": str(path),
        "kind": "occurrence_json",
        "records": len(records),
        "species": len(species),
        "markers": json.dumps(dict(markers), sort_keys=True),
        "statuses": json.dumps(dict(statuses), sort_keys=True),
        "records_with_sequence_like_text": sequence_like,
        "has_sequence_text": sequence_like > 0,
    }


def inspect_occurrence_csv(path: Path) -> dict[str, Any]:
    records = 0
    species = set()
    sequence_like = 0
    statuses = Counter()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records += 1
                if row.get("species"):
                    species.add(row["species"])
                statuses[str(row.get("occurrenceStatus") or "").upper()] += 1
                if has_sequence_text(row.get("associatedSequences")):
                    sequence_like += 1
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "kind": "occurrence_csv", "read_error": str(exc)}
    return {
        "path": str(path),
        "kind": "occurrence_csv",
        "records": records,
        "species": len(species),
        "statuses": json.dumps(dict(statuses), sort_keys=True),
        "records_with_sequence_like_text": sequence_like,
        "has_sequence_text": sequence_like > 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for path in sorted((ROOT / "data").rglob("species_sequences.json")):
        text = str(path).lower()
        if "16s" in text or "rrnl" in text:
            rows.append(inspect_species_json(path))

    occurrence_root = ROOT / "data" / "edna" / "raw" / "real_edna" / "occurrences"
    for path in sorted(occurrence_root.glob("*16s*.json")):
        rows.append(inspect_occurrence_json(path))
    for path in sorted(occurrence_root.glob("*trawl_gbif.records.json")):
        # Scotian Shelf trawl file carries the local 16S GBIF records.
        rows.append(inspect_occurrence_json(path))
    for path in sorted(occurrence_root.glob("*16s*.csv")):
        rows.append(inspect_occurrence_csv(path))
    for path in sorted(occurrence_root.glob("*trawl_gbif.records.csv")):
        rows.append(inspect_occurrence_csv(path))

    out_csv = OUT / "marker_16s_local_source_audit.csv"
    keys = sorted({key for row in rows for key in row})
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_csv": str(out_csv),
        "n_rows": len(rows),
        "conclusion": (
            "Local 16S material is currently occurrence/sample metadata, not a "
            "curated species_sequences.json reference table, unless rows with "
            "kind=species_json appear in the CSV."
        ),
    }
    (OUT / "marker_16s_local_source_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
