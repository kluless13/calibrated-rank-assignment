#!/usr/bin/env python3
"""Download manageable real eDNA/trawl occurrence tables for validation."""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


GBIF_SOURCES = {
    "scotian_shelf_12s_gbif": {
        "dataset_key": "b4e78634-8cb2-4f16-9d86-febf2049f067",
        "landing_page": "https://www.gbif.org/dataset/b4e78634-8cb2-4f16-9d86-febf2049f067",
        "marker": "12S",
        "role": "eDNA occurrence labels",
    },
    "scotian_shelf_trawl_gbif": {
        "dataset_key": "1b84f19b-c060-432a-81b6-272034fa4fe9",
        "landing_page": "https://www.gbif.org/dataset/1b84f19b-c060-432a-81b6-272034fa4fe9",
        "marker": "traditional survey",
        "role": "paired/nearby biodiversity context",
    },
}

OBIS_SOURCES = {
    "new_jersey_12s_obis": {
        "dataset_id": "fe2ed263-2b21-47d7-a79f-f9b911132398",
        "landing_page": "https://obis.org/dataset/fe2ed263-2b21-47d7-a79f-f9b911132398",
        "marker": "12S V5",
        "role": "coastal eDNA occurrence labels",
    }
}

CSV_FIELDS = [
    "source_id",
    "source_kind",
    "dataset_id",
    "dataset_key",
    "record_id",
    "eventID",
    "occurrenceID",
    "basisOfRecord",
    "occurrenceStatus",
    "scientificName",
    "acceptedScientificName",
    "taxonRank",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "specificEpithet",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "country",
    "stateProvince",
    "locality",
    "eventDate",
    "year",
    "month",
    "day",
    "individualCount",
    "organismQuantity",
    "organismQuantityType",
    "sampleSizeValue",
    "sampleSizeUnit",
    "associatedSequences",
    "associatedReferences",
    "verbatimIdentification",
    "identificationRemarks",
]


def fetch_json(url: str, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "MarineMamba occurrence fetcher"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def gbif_records(dataset_key: str, timeout: int, page_limit: int, sleep: float) -> tuple[list[dict], dict]:
    records: list[dict] = []
    offset = 0
    total = None
    while True:
        query = urllib.parse.urlencode({
            "datasetKey": dataset_key,
            "limit": page_limit,
            "offset": offset,
        })
        payload = fetch_json(f"https://api.gbif.org/v1/occurrence/search?{query}", timeout)
        if total is None:
            total = int(payload.get("count", 0))
        page = payload.get("results", [])
        records.extend(page)
        offset += len(page)
        if payload.get("endOfRecords") or not page or offset >= total:
            break
        time.sleep(sleep)
    return records, {"reported_total": total, "downloaded": len(records)}


def obis_records(dataset_id: str, timeout: int) -> tuple[list[dict], dict]:
    query = urllib.parse.urlencode({"datasetid": dataset_id, "size": 10000})
    payload = fetch_json(f"https://api.obis.org/v3/occurrence?{query}", timeout)
    records = payload.get("results", [])
    return records, {"reported_total": payload.get("total"), "downloaded": len(records)}


def flatten_record(source_id: str, source_kind: str, source_meta: dict, record: dict) -> dict[str, object]:
    row = {field: None for field in CSV_FIELDS}
    row["source_id"] = source_id
    row["source_kind"] = source_kind
    row["dataset_id"] = source_meta.get("dataset_id") or record.get("dataset_id")
    row["dataset_key"] = source_meta.get("dataset_key") or record.get("datasetKey")
    row["record_id"] = record.get("key") or record.get("id") or record.get("occurrenceID")
    for field in CSV_FIELDS:
        if field in {"source_id", "source_kind", "dataset_id", "dataset_key", "record_id"}:
            continue
        if field in record:
            row[field] = record.get(field)
    return row


def write_outputs(output_dir: Path, source_id: str, source_kind: str, source_meta: dict, records: list[dict], stats: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{source_id}.records.json"
    csv_path = output_dir / f"{source_id}.records.csv"
    raw_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_record(source_id, source_kind, source_meta, record))
    species = {
        record.get("species") or record.get("acceptedScientificName") or record.get("scientificName")
        for record in records
    }
    species.discard(None)
    events = {record.get("eventID") for record in records}
    events.discard(None)
    return {
        **source_meta,
        "source_kind": source_kind,
        "records_json": str(raw_path),
        "records_csv": str(csv_path),
        "reported_total": stats.get("reported_total"),
        "downloaded_records": len(records),
        "unique_species_or_names": len(species),
        "unique_events": len(events),
        "species_or_name_sample": sorted(str(value) for value in species)[:25],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/edna/raw/real_edna/occurrences"),
        help="Output directory for raw JSON and flattened CSV tables.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--gbif-page-limit", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    manifest: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "sources": {},
    }

    for source_id, source_meta in GBIF_SOURCES.items():
        records, stats = gbif_records(
            source_meta["dataset_key"],
            timeout=args.timeout,
            page_limit=args.gbif_page_limit,
            sleep=args.sleep,
        )
        manifest["sources"][source_id] = write_outputs(
            args.output_dir, source_id, "gbif", source_meta, records, stats
        )

    for source_id, source_meta in OBIS_SOURCES.items():
        records, stats = obis_records(source_meta["dataset_id"], timeout=args.timeout)
        manifest["sources"][source_id] = write_outputs(
            args.output_dir, source_id, "obis", source_meta, records, stats
        )

    manifest_path = args.output_dir / "real_edna_occurrences_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(source["downloaded_records"] for source in manifest["sources"].values())
    print(f"Wrote {manifest_path}; downloaded {total} records from {len(manifest['sources'])} sources.")


if __name__ == "__main__":
    main()
