#!/usr/bin/env python3
"""Fetch OBIS fish occurrences near Global_eDNA sites for an external range prior."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


OBIS_OCCURRENCE_URL = "https://api.obis.org/v3/occurrence"
DEFAULT_FIELDS = [
    "id",
    "occurrenceID",
    "basisOfRecord",
    "occurrenceStatus",
    "scientificName",
    "species",
    "genus",
    "family",
    "order",
    "class",
    "classid",
    "gigaclass",
    "gigaclassid",
    "superclass",
    "superclassid",
    "decimalLatitude",
    "decimalLongitude",
    "eventDate",
    "date_year",
    "dataset_id",
    "datasetID",
    "datasetName",
    "aphiaID",
    "speciesid",
]
RECORD_FIELDS = [
    "site_column",
    "site_value",
    "record_id",
    "occurrenceID",
    "basisOfRecord",
    "occurrenceStatus",
    "scientificName",
    "species",
    "genus",
    "family",
    "order",
    "class",
    "gigaclass",
    "superclass",
    "decimalLatitude",
    "decimalLongitude",
    "eventDate",
    "date_year",
    "dataset_id",
    "datasetID",
    "datasetName",
    "aphiaID",
    "speciesid",
    "matched_tree_label",
    "matched_species_name",
]


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def normalize_name(value: object) -> str:
    if not nonempty(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_name(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120] or "missing"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def bbox_wkt(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> str:
    return (
        "POLYGON (("
        f"{min_lon:.6f} {min_lat:.6f}, "
        f"{max_lon:.6f} {min_lat:.6f}, "
        f"{max_lon:.6f} {max_lat:.6f}, "
        f"{min_lon:.6f} {max_lat:.6f}, "
        f"{min_lon:.6f} {min_lat:.6f}"
        "))"
    )


def fetch_json(url: str, timeout: int, retries: int, sleep: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "MarineMamba Global_eDNA OBIS prior fetcher",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - exercised by live API instability.
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(sleep * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def load_candidate_maps(candidate_species: Path) -> tuple[dict[str, str], dict[str, str]]:
    candidates = pd.read_csv(candidate_species)
    name_to_label: dict[str, str] = {}
    label_to_name: dict[str, str] = {}
    for _, row in candidates.iterrows():
        if nonempty(row.get("species_name")) and nonempty(row.get("tree_label")):
            name = str(row["species_name"]).strip()
            label = str(row["tree_label"]).strip()
            name_to_label[normalize_name(name)] = label
            label_to_name[label] = name
    if not name_to_label:
        raise SystemExit(f"No species_name/tree_label rows found in {candidate_species}")
    return name_to_label, label_to_name


def site_groups(sample_map: pd.DataFrame, site_column: str, pad_degrees: float) -> list[dict[str, object]]:
    required = {site_column, "latitude_start_clean", "longitude_start_clean", "sample_id"}
    missing = sorted(required - set(sample_map.columns))
    if missing:
        raise SystemExit(f"Sample map is missing required columns: {missing}")

    cleaned = sample_map.dropna(subset=[site_column, "latitude_start_clean", "longitude_start_clean"])
    groups: list[dict[str, object]] = []
    for site_value, group in cleaned.groupby(site_column, sort=True):
        if not nonempty(site_value):
            continue
        lats = pd.to_numeric(group["latitude_start_clean"], errors="coerce").dropna()
        lons = pd.to_numeric(group["longitude_start_clean"], errors="coerce").dropna()
        if lats.empty or lons.empty:
            continue
        min_lat = clamp(float(lats.min()) - pad_degrees, -89.9, 89.9)
        max_lat = clamp(float(lats.max()) + pad_degrees, -89.9, 89.9)
        min_lon = clamp(float(lons.min()) - pad_degrees, -180.0, 180.0)
        max_lon = clamp(float(lons.max()) + pad_degrees, -180.0, 180.0)
        groups.append(
            {
                "site_value": str(site_value),
                "sample_count": int(group["sample_id"].nunique()),
                "query_row_count": int(len(group)),
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "geometry": bbox_wkt(min_lon, min_lat, max_lon, max_lat),
            }
        )
    return groups


def obis_records_for_site(
    *,
    geometry: str,
    taxonid: int,
    page_size: int,
    max_records: int,
    timeout: int,
    retries: int,
    sleep: float,
    fields: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int | bool | None]]:
    records: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    truncated = False
    while True:
        size = min(page_size, max_records - len(records))
        if size <= 0:
            truncated = True
            break
        params = {
            "taxonid": taxonid,
            "geometry": geometry,
            "size": size,
            "offset": offset,
            "fields": ",".join(fields),
        }
        url = f"{OBIS_OCCURRENCE_URL}?{urllib.parse.urlencode(params)}"
        payload = fetch_json(url, timeout=timeout, retries=retries, sleep=sleep)
        if total is None:
            raw_total = payload.get("total")
            total = int(raw_total) if raw_total is not None else None
        page = payload.get("results") or []
        if not isinstance(page, list) or not page:
            break
        records.extend(page)
        offset += len(page)
        if total is not None and offset >= total:
            break
        if len(records) >= max_records:
            truncated = True
            break
        time.sleep(sleep)
    return records, {"reported_total": total, "downloaded": len(records), "truncated": truncated}


def write_site_query_plan(output_dir: Path, site_column: str, taxonid: int, groups: list[dict[str, object]]) -> Path:
    path = output_dir / "obis_site_query_plan.csv"
    rows = []
    for site in groups:
        params = {
            "taxonid": taxonid,
            "geometry": site["geometry"],
            "size": "PAGE_SIZE",
            "offset": "OFFSET",
            "fields": ",".join(DEFAULT_FIELDS),
        }
        rows.append(
            {
                "site_column": site_column,
                "site_value": site["site_value"],
                "sample_count": site["sample_count"],
                "query_row_count": site["query_row_count"],
                "min_lat": site["min_lat"],
                "max_lat": site["max_lat"],
                "min_lon": site["min_lon"],
                "max_lon": site["max_lon"],
                "geometry": site["geometry"],
                "obis_url_template": f"{OBIS_OCCURRENCE_URL}?{urllib.parse.urlencode(params)}",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def match_record(record: dict[str, Any], name_to_label: dict[str, str], label_to_name: dict[str, str]) -> tuple[str | None, str | None]:
    for field in ["species", "acceptedScientificName", "scientificName"]:
        label = name_to_label.get(normalize_name(record.get(field)))
        if label:
            return label, label_to_name[label]
    return None, None


def flatten_record(
    site_column: str,
    site_value: str,
    record: dict[str, Any],
    name_to_label: dict[str, str],
    label_to_name: dict[str, str],
) -> dict[str, object]:
    label, species_name = match_record(record, name_to_label, label_to_name)
    row = {field: None for field in RECORD_FIELDS}
    row["site_column"] = site_column
    row["site_value"] = site_value
    row["record_id"] = record.get("id") or record.get("occurrenceID")
    row["matched_tree_label"] = label
    row["matched_species_name"] = species_name
    for field in RECORD_FIELDS:
        if field in row and row[field] is not None:
            continue
        if field in record:
            row[field] = record.get(field)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--candidate-species", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-column", default="site20")
    parser.add_argument("--taxonid", type=int, default=10194, help="OBIS/WoRMS taxon id; 10194 is Actinopterygii.")
    parser.add_argument("--bbox-pad-degrees", type=float, default=0.5)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--max-records-per-site", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit-sites", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the site query plan without contacting OBIS.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_site_records"
    raw_dir.mkdir(parents=True, exist_ok=True)

    name_to_label, label_to_name = load_candidate_maps(args.candidate_species)
    sample_map = pd.read_csv(args.sample_query_map)
    groups = site_groups(sample_map, args.site_column, args.bbox_pad_degrees)
    if args.limit_sites is not None:
        groups = groups[: args.limit_sites]
    query_plan_path = write_site_query_plan(args.output_dir, args.site_column, args.taxonid, groups)
    if args.dry_run:
        manifest = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "sample_query_map": str(args.sample_query_map),
            "candidate_species": str(args.candidate_species),
            "output_dir": str(args.output_dir),
            "site_column": args.site_column,
            "taxonid": args.taxonid,
            "bbox_pad_degrees": args.bbox_pad_degrees,
            "site_count": len(groups),
            "site_query_plan_csv": str(query_plan_path),
            "dry_run": True,
            "notes": [
                "No network requests were made.",
                "Running without --dry-run will send these site polygons to the OBIS occurrence API.",
            ],
        }
        (args.output_dir / "obis_range_prior_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(f"Wrote dry-run query plan {query_plan_path}")
        return

    all_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    site_stats: dict[str, dict[str, object]] = {}
    for site in groups:
        site_value = str(site["site_value"])
        site_path = raw_dir / f"{safe_name(site_value)}.records.json"
        if site_path.exists() and not args.force:
            records = json.loads(site_path.read_text())
            stats = {"reported_total": None, "downloaded": len(records), "truncated": False, "cache_hit": True}
        else:
            records, stats = obis_records_for_site(
                geometry=str(site["geometry"]),
                taxonid=args.taxonid,
                page_size=args.page_size,
                max_records=args.max_records_per_site,
                timeout=args.timeout,
                retries=args.retries,
                sleep=args.sleep,
                fields=DEFAULT_FIELDS,
            )
            site_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
            stats = {**stats, "cache_hit": False}

        counts: Counter[str] = Counter()
        datasets: dict[str, set[str]] = defaultdict(set)
        for record in records:
            row = flatten_record(args.site_column, site_value, record, name_to_label, label_to_name)
            all_rows.append(row)
            label = row.get("matched_tree_label")
            if nonempty(label):
                label_text = str(label)
                counts[label_text] += 1
                dataset = row.get("dataset_id") or row.get("datasetID")
                if nonempty(dataset):
                    datasets[label_text].add(str(dataset))

        for label, count in counts.most_common():
            count_rows.append(
                {
                    "site_column": args.site_column,
                    "site_value": site_value,
                    "tree_label": label,
                    "species_name": label_to_name.get(label),
                    "occurrence_count": int(count),
                    "unique_datasets": int(len(datasets[label])),
                }
            )
        site_stats[site_value] = {
            **site,
            **stats,
            "matched_candidate_species": int(len(counts)),
            "matched_candidate_occurrences": int(sum(counts.values())),
            "raw_records_path": str(site_path),
        }
        print(
            f"{site_value}: downloaded={stats['downloaded']} "
            f"matched_species={len(counts)} matched_occurrences={sum(counts.values())}",
            flush=True,
        )

    records_path = args.output_dir / "obis_site_records.csv"
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    prior_path = args.output_dir / "obis_site_prior_counts.csv"
    pd.DataFrame(count_rows).to_csv(prior_path, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sample_query_map": str(args.sample_query_map),
        "candidate_species": str(args.candidate_species),
        "output_dir": str(args.output_dir),
        "site_column": args.site_column,
        "taxonid": args.taxonid,
        "bbox_pad_degrees": args.bbox_pad_degrees,
        "page_size": args.page_size,
        "max_records_per_site": args.max_records_per_site,
        "site_count": len(groups),
        "site_query_plan_csv": str(query_plan_path),
        "records_csv": str(records_path),
        "prior_counts_csv": str(prior_path),
        "total_downloaded_records": int(len(all_rows)),
        "total_matched_candidate_occurrences": int(sum(row["occurrence_count"] for row in count_rows)),
        "unique_matched_candidate_species": int(pd.DataFrame(count_rows)["tree_label"].nunique()) if count_rows else 0,
        "site_stats": site_stats,
        "notes": [
            "OBIS occurrence search uses taxonid=10194, the Actinopterygii WoRMS/OBIS id observed in OBIS records.",
            "Records are independent occurrence evidence near each Global_eDNA site cell, not sequence reads from the validation samples.",
            "Species are matched exactly to the current open-candidate tree labels by scientific name.",
        ],
    }
    (args.output_dir / "obis_range_prior_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {prior_path} and {records_path}")


if __name__ == "__main__":
    main()
