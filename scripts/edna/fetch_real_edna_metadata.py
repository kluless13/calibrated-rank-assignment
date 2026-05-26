#!/usr/bin/env python3
"""Fetch metadata manifests for real eDNA datasets targeted for validation."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SOURCES = {
    "scotian_shelf_12s_gbif": {
        "kind": "gbif_dataset",
        "dataset_key": "b4e78634-8cb2-4f16-9d86-febf2049f067",
        "url": "https://api.gbif.org/v1/dataset/b4e78634-8cb2-4f16-9d86-febf2049f067",
        "landing_page": "https://www.gbif.org/dataset/b4e78634-8cb2-4f16-9d86-febf2049f067",
        "why": "12S marker-specific paired eDNA/trawl dataset for Northwest Atlantic assemblage validation.",
    },
    "scotian_shelf_trawl_gbif": {
        "kind": "gbif_dataset",
        "dataset_key": "1b84f19b-c060-432a-81b6-272034fa4fe9",
        "url": "https://api.gbif.org/v1/dataset/1b84f19b-c060-432a-81b6-272034fa4fe9",
        "landing_page": "https://www.gbif.org/dataset/1b84f19b-c060-432a-81b6-272034fa4fe9",
        "why": "Companion traditional trawl/survey context for the Scotian Shelf eDNA dataset.",
    },
    "new_jersey_12s_obis": {
        "kind": "obis_occurrence_probe",
        "dataset_id": "fe2ed263-2b21-47d7-a79f-f9b911132398",
        "url": "https://api.obis.org/v3/occurrence?datasetid=fe2ed263-2b21-47d7-a79f-f9b911132398&size=1",
        "landing_page": "https://obis.org/dataset/fe2ed263-2b21-47d7-a79f-f9b911132398",
        "why": "Coastal New Jersey 12S V5 fish eDNA ASV occurrence table with coordinates and event IDs.",
    },
    "global_tropical_dryad": {
        "kind": "dryad_dataset",
        "doi": "10.5061/dryad.3xsj3txj2",
        "url": "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.3xsj3txj2",
        "extra_urls": {
            "version_files": "https://datadryad.org/api/v2/versions/172282/files"
        },
        "landing_page": "https://datadryad.org/dataset/doi:10.5061/dryad.3xsj3txj2",
        "why": "Potential real eDNA plus ecological survey validation set; file accessibility must be checked.",
    },
}


def fetch_json(url: str, timeout: int, retries: int) -> tuple[object | None, str | None]:
    headers = {"User-Agent": "MarineMamba metadata inventory"}
    request = urllib.request.Request(url, headers=headers)
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body), None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(1.0 + attempt)
    return None, last_error


def compact_metadata(
    source: dict[str, object],
    payload: object | None,
    extra_payloads: dict[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    extra_payloads = extra_payloads or {}
    kind = source["kind"]
    if kind == "gbif_dataset":
        contacts = payload.get("contacts") or []
        return {
            "title": payload.get("title"),
            "doi": payload.get("doi"),
            "type": payload.get("type"),
            "subtype": payload.get("subtype"),
            "publishing_organization_key": payload.get("publishingOrganizationKey"),
            "created": payload.get("created"),
            "modified": payload.get("modified"),
            "homepage": payload.get("homepage"),
            "contacts": [
                {
                    "firstName": contact.get("firstName"),
                    "lastName": contact.get("lastName"),
                    "organization": contact.get("organization"),
                    "type": contact.get("type"),
                }
                for contact in contacts[:5]
                if isinstance(contact, dict)
            ],
        }
    if kind == "obis_occurrence_probe":
        results = payload.get("results") or []
        first = results[0] if results and isinstance(results[0], dict) else {}
        return {
            "total_occurrences": payload.get("total"),
            "sample_record": {
                key: first.get(key)
                for key in [
                    "dataset_id",
                    "dataset_name",
                    "scientificName",
                    "eventID",
                    "decimalLatitude",
                    "decimalLongitude",
                    "associatedReferences",
                    "verbatimIdentification",
                    "kingdom",
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                ]
                if key in first
            },
        }
    if kind == "dryad_dataset":
        files_payload = extra_payloads.get("version_files")
        if isinstance(files_payload, dict):
            embedded = files_payload.get("_embedded")
            files = embedded.get("stash:files", []) if isinstance(embedded, dict) else []
        else:
            embedded = payload.get("_embedded")
            files = embedded.get("stash:files", []) if isinstance(embedded, dict) else []
        return {
            "title": payload.get("title"),
            "doi": payload.get("identifier"),
            "publication_date": payload.get("publicationDate"),
            "version": payload.get("versionNumber"),
            "storage_size": payload.get("storageSize"),
            "file_count": len(files) if isinstance(files, list) else None,
            "files": [
                {
                    "path": file_info.get("path"),
                    "size": file_info.get("size"),
                    "mimeType": file_info.get("mimeType"),
                    "digest": file_info.get("digest"),
                    "download_href": (
                        file_info.get("_links", {})
                        .get("stash:download", {})
                        .get("href")
                    ) if isinstance(file_info.get("_links"), dict) else None,
                }
                for file_info in files[:20]
                if isinstance(file_info, dict)
            ],
        }
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/edna/raw/real_edna/metadata"),
        help="Directory for fetched source metadata.",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "sources": {},
    }

    for name, source in SOURCES.items():
        payload, error = fetch_json(str(source["url"]), timeout=args.timeout, retries=args.retries)
        extra_payloads: dict[str, object] = {}
        extra_errors: dict[str, str | None] = {}
        for extra_name, extra_url in (source.get("extra_urls") or {}).items():
            extra_payload, extra_error = fetch_json(str(extra_url), timeout=args.timeout, retries=args.retries)
            extra_errors[str(extra_name)] = extra_error
            if extra_payload is not None:
                extra_payloads[str(extra_name)] = extra_payload
                (args.output_dir / f"{name}.{extra_name}.json").write_text(
                    json.dumps(extra_payload, indent=2, sort_keys=True) + "\n"
                )
        source_record = {
            **source,
            "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "success": error is None,
            "error": error,
            "extra_errors": extra_errors,
            "metadata_file": str(args.output_dir / f"{name}.json"),
            "compact": compact_metadata(source, payload, extra_payloads),
        }
        if payload is not None:
            (args.output_dir / f"{name}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        (args.output_dir / f"{name}.manifest.json").write_text(
            json.dumps(source_record, indent=2, sort_keys=True) + "\n"
        )
        manifest["sources"][name] = source_record

    manifest_path = args.output_dir / "real_edna_sources_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ok = sum(1 for record in manifest["sources"].values() if record["success"])
    print(f"Wrote {manifest_path}; fetched {ok}/{len(SOURCES)} sources.")


if __name__ == "__main__":
    main()
