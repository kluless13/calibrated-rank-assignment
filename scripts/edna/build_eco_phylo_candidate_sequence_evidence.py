#!/usr/bin/env python3
"""Build candidate-level 12S sequence-distance evidence for Eco-Phylo posterior.

This script computes direct sequence evidence for query/candidate pairs already
present in the candidate posterior table. It uses only train-reference 12S
sequences for each candidate species, then reports the best ungapped sliding
p-distance/identity between the query sequence and any train-reference sequence.

The evidence is intentionally stored as a separate join table so the large
candidate feature table does not need to be rewritten.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "edna" / "real_edna_queries" / "global_tropical_multisource_teleo"
DEFAULT_CANDIDATE_FEATURES = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "eco_phylo_posterior"
    / "candidate_level"
    / "eco_phylo_candidate_features_top5.csv.gz"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "eco_phylo_posterior"
    / "candidate_level"
)

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def clean_sequence(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return "".join(base for base in str(value).upper() if base in {"A", "C", "G", "T", "N"})


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1].upper()


def best_ungapped_distance(
    query: str,
    reference: str,
    min_overlap_fraction: float,
    min_overlap_bases: int,
    include_reverse_complement: bool,
) -> dict[str, Any] | None:
    query = clean_sequence(query)
    reference = clean_sequence(reference)
    if not query or not reference:
        return None

    required_overlap = min(len(query), len(reference))
    required_overlap = max(min_overlap_bases, math.ceil(required_overlap * min_overlap_fraction))
    best: dict[str, Any] | None = None
    orientations = [("forward", reference)]
    if include_reverse_complement:
        orientations.append(("reverse_complement", reverse_complement(reference)))

    for orientation, ref in orientations:
        # offset is reference index minus query index. Negative offsets allow
        # the query to start before the reference; positive offsets shift the
        # query right along the reference.
        for offset in range(-len(query) + 1, len(ref)):
            q_start = max(0, -offset)
            r_start = max(0, offset)
            overlap = min(len(query) - q_start, len(ref) - r_start)
            if overlap < required_overlap:
                continue
            comparable = 0
            mismatches = 0
            for q_base, r_base in zip(query[q_start : q_start + overlap], ref[r_start : r_start + overlap]):
                if q_base == "N" or r_base == "N":
                    continue
                comparable += 1
                mismatches += int(q_base != r_base)
            if comparable < required_overlap:
                continue
            pdistance = mismatches / comparable if comparable else math.nan
            if best is None or (pdistance, -comparable, mismatches) < (
                best["sequence_pdistance"],
                -best["sequence_overlap_bases"],
                best["sequence_mismatches"],
            ):
                best = {
                    "sequence_pdistance": pdistance,
                    "sequence_identity": 1.0 - pdistance,
                    "sequence_overlap_bases": comparable,
                    "sequence_mismatches": mismatches,
                    "sequence_orientation": orientation,
                    "sequence_offset": offset,
                }
    return best


def load_unique_pairs(path: Path, chunksize: int, logger: ProgressLogger) -> pd.DataFrame:
    logger.log(f"Loading unique query/candidate pairs from {rel(path)}")
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=["query_processid", "candidate_tree_label"], chunksize=chunksize):
        frames.append(chunk.drop_duplicates())
    pairs = pd.concat(frames, ignore_index=True).drop_duplicates()
    logger.log(f"Found {len(pairs):,} unique query/candidate pairs")
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--candidate-features", type=Path, default=DEFAULT_CANDIDATE_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="eco_phylo_candidate_12s_sequence_evidence_top5.csv.gz")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--min-overlap-bases", type=int, default=35)
    parser.add_argument("--include-reverse-complement", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_unique_pairs(args.candidate_features, args.chunksize, logger)
    queries = pd.read_csv(args.input_dir / "zero_shot_queries.csv", usecols=["processid", "nucleotides"])
    query_sequences = {str(row.processid): clean_sequence(row.nucleotides) for row in queries.itertuples(index=False)}
    train_references: dict[str, list[str]] = json.loads((args.input_dir / "train_species_sequences.json").read_text())
    train_references = {
        label: [clean_sequence(seq) for seq in sequences if clean_sequence(seq)]
        for label, sequences in train_references.items()
    }
    train_references = {label: seqs for label, seqs in train_references.items() if seqs}
    logger.log(f"Loaded {len(query_sequences):,} query sequences and {len(train_references):,} train-reference species")

    fields = [
        "query_processid",
        "candidate_tree_label",
        "sequence_evidence_available",
        "sequence_evidence_method",
        "sequence_pdistance",
        "sequence_identity",
        "sequence_overlap_bases",
        "sequence_mismatches",
        "sequence_reference_count",
        "sequence_best_reference_index",
        "sequence_best_reference_length",
        "query_sequence_length",
        "sequence_orientation",
        "sequence_offset",
    ]
    rows: list[dict[str, Any]] = []
    available = 0
    missing_query = 0
    missing_reference = 0
    no_overlap = 0

    for idx, pair in enumerate(pairs.itertuples(index=False), start=1):
        query_id = str(pair.query_processid)
        candidate = str(pair.candidate_tree_label)
        query = query_sequences.get(query_id, "")
        references = train_references.get(candidate, [])
        row: dict[str, Any] = {
            "query_processid": query_id,
            "candidate_tree_label": candidate,
            "sequence_evidence_available": 0,
            "sequence_evidence_method": "best_ungapped_train_reference",
            "sequence_reference_count": len(references),
            "query_sequence_length": len(query),
        }
        if not query:
            missing_query += 1
            rows.append(row)
            continue
        if not references:
            missing_reference += 1
            rows.append(row)
            continue

        best: dict[str, Any] | None = None
        best_ref_idx = -1
        best_ref_len = 0
        for ref_idx, reference in enumerate(references):
            current = best_ungapped_distance(
                query=query,
                reference=reference,
                min_overlap_fraction=args.min_overlap_fraction,
                min_overlap_bases=args.min_overlap_bases,
                include_reverse_complement=args.include_reverse_complement,
            )
            if current is None:
                continue
            if best is None or (current["sequence_pdistance"], -current["sequence_overlap_bases"]) < (
                best["sequence_pdistance"],
                -best["sequence_overlap_bases"],
            ):
                best = current
                best_ref_idx = ref_idx
                best_ref_len = len(reference)
        if best is None:
            no_overlap += 1
            rows.append(row)
            continue

        available += 1
        row.update(best)
        row.update(
            {
                "sequence_evidence_available": 1,
                "sequence_best_reference_index": best_ref_idx,
                "sequence_best_reference_length": best_ref_len,
            }
        )
        rows.append(row)
        if idx % 100_000 == 0:
            logger.log(f"Processed {idx:,}/{len(pairs):,} pairs; evidence available for {available:,}")

    output_path = args.output_dir / args.output_name
    logger.log(f"Writing sequence evidence table to {rel(output_path)}")
    write_csv(output_path, rows, fields)

    summary_rows = [
        {
            "candidate_pairs": len(rows),
            "sequence_evidence_available_pairs": available,
            "missing_query_pairs": missing_query,
            "missing_reference_pairs": missing_reference,
            "no_overlap_pairs": no_overlap,
            "min_overlap_fraction": args.min_overlap_fraction,
            "min_overlap_bases": args.min_overlap_bases,
            "include_reverse_complement": args.include_reverse_complement,
        }
    ]
    summary_path = args.output_dir / "eco_phylo_candidate_12s_sequence_evidence_summary.csv"
    write_csv(summary_path, summary_rows, list(summary_rows[0]))
    manifest = {
        "generated_by": rel(Path(__file__)),
        "input_dir": rel(args.input_dir),
        "candidate_features": rel(args.candidate_features),
        "output": rel(output_path),
        "summary": rel(summary_path),
        "candidate_pairs": len(rows),
        "sequence_evidence_available_pairs": available,
        "notes": [
            "Uses train_species_sequences.json only, so candidates without train-reference 12S remain unavailable.",
            "Distance is best ungapped sliding p-distance, not BLAST and not a full affine-gap alignment.",
            "This evidence table is designed to be joined into the candidate posterior by query_processid and candidate_tree_label.",
        ],
    }
    manifest_path = args.output_dir / "eco_phylo_candidate_12s_sequence_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
