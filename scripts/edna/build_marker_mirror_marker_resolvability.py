#!/usr/bin/env python3
"""Build per-species marker resolvability features for MarkerMirror.

This script produces a reference-only ambiguity table for the 12S/16S
MarkerMirror handoff.  It asks, for each species and marker, whether the marker
evidence uniquely supports species or only a broader genus/family/order claim.

The near-exact backend is deliberately labelled as a proxy: it uses rare-kmer
blocking plus ungapped prefix identity so it can run without VSEARCH/edlib.  Use
it as an auditable pipeline feature and replace it with an alignment-backed
backend before making a final production claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import Logger, clean_sequence, load_species_json, load_taxonomy

RANKS = ("species", "genus", "family", "order")
RANK_ORDER = ("species", "genus", "family", "order", "unresolved")
RANK_SCORE = {rank: idx for idx, rank in enumerate(RANK_ORDER)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marker-a-name", default="12S")
    parser.add_argument("--marker-a-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource")
    parser.add_argument("--marker-b-name", default="16S")
    parser.add_argument("--marker-b-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource")
    parser.add_argument("--species-json-name", default="train_species_sequences.json")
    parser.add_argument("--max-per-species", type=int, default=8)
    parser.add_argument("--identities", default="1.0,0.99,0.98")
    parser.add_argument("--min-length", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--kmer", type=int, default=17)
    parser.add_argument("--signature-kmers", type=int, default=48)
    parser.add_argument("--max-block-size", type=int, default=800)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def rank_value(label: str, rank: str, taxonomy: dict[str, dict[str, str]]) -> str:
    if rank == "species":
        return label
    value = str(taxonomy.get(label, {}).get(rank, "") or "").strip()
    return "" if value.lower() in {"", "nan", "none"} else value


def deepest_supported_rank(labels: set[str], taxonomy: dict[str, dict[str, str]]) -> str:
    if not labels:
        return "unresolved"
    if len(labels) == 1:
        return "species"
    for rank in ("genus", "family", "order"):
        values = {rank_value(label, rank, taxonomy) for label in labels}
        values = {value for value in values if value}
        if len(values) == 1:
            return rank
    return "unresolved"


def normalize_sequence(seq: str, max_length: int) -> str:
    return clean_sequence(seq)[:max_length]


def load_marker(name: str, input_dir: Path, species_json_name: str, max_per_species: int, min_length: int, max_length: int) -> dict[str, Any]:
    raw = load_species_json(input_dir / species_json_name, max_per_species)
    taxonomy = load_taxonomy(input_dir / "candidate_species.csv")
    records = []
    for label, seqs in sorted(raw.items()):
        for seq_idx, seq in enumerate(seqs):
            normalized = normalize_sequence(seq, max_length)
            if len(normalized) >= min_length:
                records.append({"record_id": f"{name}_{len(records)}", "tree_label": label, "seq_index": seq_idx, "sequence": normalized})
    return {"name": name, "input_dir": input_dir, "taxonomy": taxonomy, "records": records}


def prefix_identity(left: str, right: str) -> tuple[float, int, int]:
    n = min(len(left), len(right))
    if n <= 0:
        return math.nan, 0, 0
    mismatches = sum(1 for a, b in zip(left[:n], right[:n]) if a != b)
    return 1.0 - (mismatches / n), n, mismatches


def kmer_set(seq: str, k: int) -> set[str]:
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[idx : idx + k] for idx in range(0, len(seq) - k + 1)}


def signatures(records: list[dict[str, Any]], k: int, signature_kmers: int, max_block_size: int, logger: Logger) -> tuple[list[set[str]], dict[str, list[int]]]:
    record_kmers = [kmer_set(str(record["sequence"]), k) for record in records]
    counts: Counter[str] = Counter()
    for kmers in record_kmers:
        counts.update(kmers)
    record_signatures: list[set[str]] = []
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, kmers in enumerate(record_kmers):
        usable = [kmer for kmer in kmers if counts[kmer] <= max_block_size]
        if not usable:
            usable = list(kmers)
        chosen = sorted(usable, key=lambda item: (counts[item], item))[:signature_kmers]
        sig = set(chosen)
        record_signatures.append(sig)
        for kmer in sig:
            inverted[kmer].append(idx)
    logger.log(f"Built rare-kmer signatures records={len(records)} kmers={len(counts)} indexed_kmers={len(inverted)}")
    return record_signatures, inverted


def exact_neighbors(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    sequence_to_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sequence_to_labels[str(record["sequence"])].add(str(record["tree_label"]))
    neighbors: dict[str, set[str]] = defaultdict(set)
    for record in records:
        label = str(record["tree_label"])
        neighbors[label].update(sequence_to_labels[str(record["sequence"])])
    return neighbors


def near_neighbors(
    records: list[dict[str, Any]],
    identity: float,
    k: int,
    signature_kmers: int,
    max_block_size: int,
    logger: Logger,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    neighbor_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        neighbor_labels[str(record["tree_label"])].add(str(record["tree_label"]))
    _, inverted = signatures(records, k, signature_kmers, max_block_size, logger)
    compared: set[tuple[int, int]] = set()
    candidate_pairs = 0
    accepted_pairs = 0
    for bucket in inverted.values():
        if len(bucket) < 2:
            continue
        for pos, left_idx in enumerate(bucket):
            left = records[left_idx]
            for right_idx in bucket[pos + 1 :]:
                key = (left_idx, right_idx) if left_idx < right_idx else (right_idx, left_idx)
                if key in compared:
                    continue
                compared.add(key)
                candidate_pairs += 1
                right = records[right_idx]
                ident, overlap, _ = prefix_identity(str(left["sequence"]), str(right["sequence"]))
                if overlap and ident >= identity:
                    left_label = str(left["tree_label"])
                    right_label = str(right["tree_label"])
                    neighbor_labels[left_label].add(right_label)
                    neighbor_labels[right_label].add(left_label)
                    accepted_pairs += 1
    stats = {
        "candidate_pairs_compared": candidate_pairs,
        "accepted_near_exact_pairs": accepted_pairs,
        "backend": "rare_kmer_prefix_identity_proxy",
    }
    logger.log(f"Near-exact proxy identity={identity} compared={candidate_pairs} accepted={accepted_pairs}")
    return neighbor_labels, stats


def rows_for_marker(marker: dict[str, Any], identities: list[float], args: argparse.Namespace, logger: Logger) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    records = marker["records"]
    labels = sorted({str(record["tree_label"]) for record in records})
    taxonomy = marker["taxonomy"]
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    backend_rows: list[dict[str, Any]] = []
    logger.log(f"Marker {marker['name']} records={len(records)} species={len(labels)}")
    for identity in identities:
        if identity >= 1.0:
            neighbors = exact_neighbors(records)
            stats = {"backend": "exact_sequence", "candidate_pairs_compared": 0, "accepted_near_exact_pairs": 0}
        else:
            neighbors, stats = near_neighbors(records, identity, args.kmer, args.signature_kmers, args.max_block_size, logger)
        backend_rows.append({"marker": marker["name"], "identity": identity, **stats})
        rank_counts: Counter[str] = Counter()
        for label in labels:
            neighbor_set = set(neighbors.get(label, {label}))
            deepest = deepest_supported_rank(neighbor_set, taxonomy)
            rank_counts[deepest] += 1
            row = {
                "marker": marker["name"],
                "identity": identity,
                "tree_label": label,
                "reference_record_count": sum(1 for record in records if str(record["tree_label"]) == label),
                "neighbor_species_count": len(neighbor_set),
                "deepest_supported_rank": deepest,
                "ambiguity_backend": stats["backend"],
                "neighbor_species_labels": json.dumps(sorted(neighbor_set)),
            }
            for rank in RANKS:
                row[f"{rank}_oracle_supported"] = int(RANK_SCORE[deepest] <= RANK_SCORE[rank])
            for rank in ("genus", "family", "order"):
                values = {rank_value(item, rank, taxonomy) for item in neighbor_set}
                values = {value for value in values if value}
                row[f"neighbor_{rank}_count"] = len(values)
            all_rows.append(row)
        summary = {
            "marker": marker["name"],
            "identity": identity,
            "species_count": len(labels),
            "sequence_record_count": len(records),
            "ambiguity_backend": stats["backend"],
        }
        for rank in RANK_ORDER:
            summary[f"{rank}_deepest_count"] = int(rank_counts.get(rank, 0))
        for rank in RANKS:
            supported = [row for row in all_rows if row["marker"] == marker["name"] and row["identity"] == identity and row[f"{rank}_oracle_supported"]]
            summary[f"{rank}_oracle_supported_rate_pct"] = 100.0 * len(supported) / max(len(labels), 1)
        summary_rows.append(summary)
    return pd.DataFrame(all_rows), pd.DataFrame(summary_rows), backend_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_marker_resolvability.log")
    logger.log(f"Arguments: {vars(args)}")
    identities = [float(item) for item in args.identities.split(",") if item.strip()]
    markers = [
        load_marker(args.marker_a_name, args.marker_a_input_dir, args.species_json_name, args.max_per_species, args.min_length, args.max_length),
        load_marker(args.marker_b_name, args.marker_b_input_dir, args.species_json_name, args.max_per_species, args.min_length, args.max_length),
    ]
    feature_frames = []
    summary_frames = []
    backend_rows = []
    for marker in markers:
        features, summary, backend = rows_for_marker(marker, identities, args, logger)
        feature_frames.append(features)
        summary_frames.append(summary)
        backend_rows.extend(backend)
    features = pd.concat(feature_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    backend = pd.DataFrame(backend_rows)
    features.to_csv(args.output_dir / "marker_mirror_marker_resolvability_by_species.csv", index=False)
    summary.to_csv(args.output_dir / "marker_mirror_marker_resolvability_summary.csv", index=False)
    backend.to_csv(args.output_dir / "marker_mirror_marker_resolvability_backend.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_marker_mirror_marker_resolvability.py",
        "marker_a_input_dir": str(args.marker_a_input_dir),
        "marker_b_input_dir": str(args.marker_b_input_dir),
        "species_json_name": args.species_json_name,
        "identities": identities,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "claim_boundary": "Reference-only MarkerMirror resolvability features. Near-exact rows use a rare-kmer prefix-identity proxy, not VSEARCH/edlib clustering.",
    }
    (args.output_dir / "marker_mirror_marker_resolvability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote features rows={len(features)} summary rows={len(summary)}")


if __name__ == "__main__":
    main()
