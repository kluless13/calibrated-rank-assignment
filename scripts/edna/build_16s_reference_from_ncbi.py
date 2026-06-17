#!/usr/bin/env python3
"""Build a fish 16S reference table from NCBI nucleotide records.

The intended first use is a logged dry run:

    python3 scripts/edna/build_16s_reference_from_ncbi.py --dry-run

Then a bounded fetch:

    python3 scripts/edna/build_16s_reference_from_ncbi.py --max-records 5000

The output format mirrors the existing 12S/COI inputs enough for
MarkerMirror-style overlap audits:

    data/edna/stalder_inputs/16s_multisource/species_sequences.json
    data/edna/stalder_inputs/16s_multisource/candidate_species.csv
    data/edna/stalder_inputs/16s_multisource/species_info.json
    data/edna/stalder_inputs/16s_multisource/manifest.json

This script deliberately starts from explicit 16S/rrnL nucleotide records
rather than complete mitogenomes.  Complete-mitogenome feature extraction can
be added later if coverage is too low.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY = (
    '("16S ribosomal RNA"[Title] OR "16S rRNA"[Title] OR rrnL[Gene] OR 16S[Gene]) '
    "AND (mitochondrion[Filter] OR mitochondrial[All Fields]) "
    "AND Actinopterygii[Organism]"
)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource",
    )
    parser.add_argument(
        "--source-table-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-per-species", type=int, default=8)
    parser.add_argument("--min-length", type=int, default=80)
    parser.add_argument("--sleep-seconds", type=float, default=0.34)
    parser.add_argument("--email", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_sequence(value: object) -> str:
    return re.sub("[^ACGT]", "", str(value or "").upper())


def canonical_label(species_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", species_name.strip()).strip("_")


def binomial(organism: str) -> str:
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+", organism)
    if len(parts) < 2:
        return ""
    if parts[1].lower() in {"sp", "cf", "aff", "spp"}:
        return ""
    return f"{parts[0]} {parts[1]}"


def eutils_url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"


def read_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MarineMamba-16S-reference-builder"})
    with urllib.request.urlopen(req, timeout=120) as response:  # noqa: S310
        return response.read()


def esearch(args: argparse.Namespace) -> tuple[int, list[str], str, str]:
    params: dict[str, Any] = {
        "db": "nuccore",
        "term": args.query,
        "retmode": "json",
        "retmax": min(args.max_records, 100000),
        "usehistory": "y",
        "tool": "MarineMamba",
    }
    if args.email:
        params["email"] = args.email
    if args.api_key:
        params["api_key"] = args.api_key
    payload = json.loads(read_url(eutils_url("esearch.fcgi", params)).decode("utf-8"))
    result = payload.get("esearchresult", {})
    count = int(result.get("count", 0))
    ids = [str(item) for item in result.get("idlist", [])]
    return count, ids[: args.max_records], str(result.get("webenv", "")), str(result.get("querykey", ""))


def efetch_xml(ids: list[str], args: argparse.Namespace) -> ET.Element:
    params: dict[str, Any] = {
        "db": "nuccore",
        "id": ",".join(ids),
        "retmode": "xml",
        "tool": "MarineMamba",
    }
    if args.email:
        params["email"] = args.email
    if args.api_key:
        params["api_key"] = args.api_key
    return ET.fromstring(read_url(eutils_url("efetch.fcgi", params)))


def parse_gbseqs(root: ET.Element, min_length: int) -> list[dict[str, str]]:
    rows = []
    for gbseq in root.findall(".//GBSeq"):
        accession = gbseq.findtext("GBSeq_primary-accession") or gbseq.findtext("GBSeq_accession-version") or ""
        organism = gbseq.findtext("GBSeq_organism") or ""
        definition = gbseq.findtext("GBSeq_definition") or ""
        sequence = clean_sequence(gbseq.findtext("GBSeq_sequence") or "")
        species = binomial(organism)
        if not accession or not species or len(sequence) < min_length:
            continue
        rows.append(
            {
                "accession": accession,
                "organism": organism,
                "species_name": species,
                "tree_label": canonical_label(species),
                "definition": definition,
                "sequence": sequence,
                "length": str(len(sequence)),
            }
        )
    return rows


def load_existing_labels(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return {str(key) for key, value in data.items() if value}
    return set()


def write_outputs(rows: list[dict[str, str]], args: argparse.Namespace, count: int) -> None:
    by_species: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_species[row["tree_label"]].append(row)

    species_sequences: dict[str, list[str]] = {}
    species_info: dict[str, dict[str, Any]] = {}
    candidate_rows = []
    accession_rows = []
    for label, items in sorted(by_species.items()):
        chosen = sorted(items, key=lambda item: (-int(item["length"]), item["accession"]))[: args.max_per_species]
        species_sequences[label] = [item["sequence"] for item in chosen]
        species_name = chosen[0]["species_name"]
        genus = species_name.split()[0]
        species_info[label] = {
            "species_name": species_name,
            "genus_name": genus,
            "source": "NCBI nuccore explicit 16S/rrnL records",
            "n_sequences": len(chosen),
        }
        candidate_rows.append(
            {
                "tree_label": label,
                "species_name": species_name,
                "genus_name": genus,
                "family_name": "",
                "order_name": "",
            }
        )
        for item in chosen:
            accession_rows.append({key: item[key] for key in ("tree_label", "species_name", "accession", "length", "definition")})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "species_sequences.json").write_text(json.dumps(species_sequences, indent=2), encoding="utf-8")
    (args.output_dir / "train_species_sequences.json").write_text(json.dumps(species_sequences, indent=2), encoding="utf-8")
    (args.output_dir / "species_info.json").write_text(json.dumps(species_info, indent=2), encoding="utf-8")

    with (args.output_dir / "candidate_species.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tree_label", "species_name", "genus_name", "family_name", "order_name"])
        writer.writeheader()
        writer.writerows(candidate_rows)

    args.source_table_dir.mkdir(parents=True, exist_ok=True)
    with (args.source_table_dir / "marker_16s_reference_accessions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tree_label", "species_name", "accession", "length", "definition"])
        writer.writeheader()
        writer.writerows(accession_rows)

    existing_12s = load_existing_labels(ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "species_sequences.json")
    existing_coi = load_existing_labels(ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs" / "eval_c" / "species_sequences.json")
    labels_16s = set(species_sequences)
    overlap_rows = [
        {
            "set": "16s",
            "n_species": len(labels_16s),
        },
        {
            "set": "12s_intersection_16s",
            "n_species": len(existing_12s & labels_16s),
        },
        {
            "set": "coi_intersection_16s",
            "n_species": len(existing_coi & labels_16s),
        },
        {
            "set": "12s_intersection_16s_intersection_coi",
            "n_species": len(existing_12s & existing_coi & labels_16s),
        },
    ]
    with (args.source_table_dir / "marker_16s_overlap_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["set", "n_species"])
        writer.writeheader()
        writer.writerows(overlap_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "ncbi_count": count,
        "fetched_rows": len(rows),
        "species": len(species_sequences),
        "max_records": args.max_records,
        "max_per_species": args.max_per_species,
        "min_length": args.min_length,
        "output_dir": str(args.output_dir),
        "source": "NCBI nuccore explicit 16S/rrnL records",
        "claim_boundary": "Initial 16S reference construction; taxonomy family/order are not curated yet.",
        "overlap": overlap_rows,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.source_table_dir / "marker_16s_reference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    args.source_table_dir.mkdir(parents=True, exist_ok=True)
    count, ids, webenv, query_key = esearch(args)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "ncbi_count": count,
        "retmax": len(ids),
        "webenv_present": bool(webenv),
        "query_key_present": bool(query_key),
        "dry_run": args.dry_run,
        "output_dir": str(args.output_dir),
        "strategy": "Fetch explicit 16S/rrnL nuccore records, parse GBSeq sequence and organism, group by binomial species.",
    }
    (args.source_table_dir / "marker_16s_ncbi_query_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    rows: list[dict[str, str]] = []
    for start in range(0, len(ids), args.batch_size):
        batch = ids[start : start + args.batch_size]
        root = efetch_xml(batch, args)
        rows.extend(parse_gbseqs(root, args.min_length))
        print(
            json.dumps(
                {
                    "batch_start": start,
                    "batch_size": len(batch),
                    "rows_so_far": len(rows),
                    "time": datetime.now(timezone.utc).isoformat(),
                }
            ),
            flush=True,
        )
        time.sleep(args.sleep_seconds)
    write_outputs(rows, args, count)


if __name__ == "__main__":
    main()
