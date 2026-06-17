#!/usr/bin/env python3
"""Build same-marker 12S candidates with BLASTN local alignment.

This is the local-alignment companion to the VSEARCH same-marker candidate
run. It searches the current 12S query table against the current 12S reference
sequences with BLASTN, deduplicates sequence hits to species-level candidates,
and writes the same support tables used by the MarkerMirror union audits.

Important boundary: this is BLASTN local-alignment candidate generation over
the current reference table. It cannot recover query species absent from that
reference table, and it is not an end-to-end calibrated rank/no-call policy.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")
BLAST6_COLUMNS = [
    "query_id",
    "ref_sequence_id",
    "pident",
    "alignment_length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-table",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "zero_shot_queries.csv",
    )
    parser.add_argument(
        "--same-marker-reference-dir",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-ref-sequences-per-species", type=int, default=12)
    parser.add_argument("--max-target-seqs", type=int)
    parser.add_argument("--evalue", type=float, default=10.0)
    parser.add_argument("--task", default="blastn")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--blastn-bin", default="blastn")
    parser.add_argument("--makeblastdb-bin", default="makeblastdb")
    parser.add_argument("--report-top-ks", default="1,5,10,50")
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def normalize_sequence(value: Any) -> str:
    return "".join(ch for ch in str(value).upper() if ch in {"A", "C", "G", "T"})


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fasta_wrap(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def load_species_info(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for label, row in raw.items():
        out[str(label)] = {
            "candidate_tree_label": str(label),
            "candidate_species": clean(row.get("species_name")) or str(label).replace("_", " "),
            "candidate_genus": clean(row.get("genus")) or clean(row.get("genus_name")),
            "candidate_family": clean(row.get("family")) or clean(row.get("family_name")),
            "candidate_order": clean(row.get("order")) or clean(row.get("order_name")),
        }
    return out


def load_queries(path: Path, limit: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if limit:
        frame = frame.head(limit).copy()
    out = pd.DataFrame(
        {
            "query_id": frame["processid"].astype(str),
            "source": frame["source"].astype(str),
            "query_tree_label": frame["tree_label"].astype(str),
            "query_species": frame["species_name"].astype(str),
            "query_genus": frame["genus_name"].map(clean),
            "query_family": frame["family_name"].map(clean),
            "query_order": frame["order_name"].map(clean),
            "query_sequence": frame["nucleotides"].map(normalize_sequence),
        }
    )
    return out[out["query_sequence"].map(bool)].copy()


def load_reference(input_dir: Path, max_per_species: int) -> pd.DataFrame:
    species_sequences = json.loads((input_dir / "species_sequences.json").read_text(encoding="utf-8"))
    species_info = load_species_info(input_dir / "species_info.json")
    rows: list[dict[str, Any]] = []
    for species, sequences in species_sequences.items():
        info = species_info.get(str(species), {"candidate_tree_label": str(species)})
        used = 0
        seen: set[str] = set()
        for sequence in sequences:
            seq = normalize_sequence(sequence)
            if not seq or seq in seen:
                continue
            seen.add(seq)
            rows.append(
                {
                    "ref_sequence_id": f"{species}|{used}",
                    **info,
                    "sequence": seq,
                    "sequence_length": len(seq),
                }
            )
            used += 1
            if used >= max_per_species:
                break
    return pd.DataFrame(rows)


def write_fasta(frame: pd.DataFrame, id_col: str, seq_col: str, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            seq_id = clean(getattr(row, id_col))
            sequence = clean(getattr(row, seq_col))
            if seq_id and sequence:
                handle.write(f">{seq_id}\n{fasta_wrap(sequence)}\n")


def run_blast(args: argparse.Namespace, query_fasta: Path, ref_fasta: Path, db_prefix: Path, blast6_path: Path, log_path: Path) -> None:
    if blast6_path.exists() and not args.force:
        return
    blastn = shutil.which(args.blastn_bin) or args.blastn_bin
    makeblastdb = shutil.which(args.makeblastdb_bin) or args.makeblastdb_bin
    max_targets = args.max_target_seqs or max(args.top_k * args.max_ref_sequences_per_species * 2, args.top_k * 20)
    make_command = [
        makeblastdb,
        "-in",
        str(ref_fasta),
        "-dbtype",
        "nucl",
        "-out",
        str(db_prefix),
    ]
    blast_command = [
        blastn,
        "-query",
        str(query_fasta),
        "-db",
        str(db_prefix),
        "-task",
        str(args.task),
        "-evalue",
        str(args.evalue),
        "-strand",
        "both",
        "-num_threads",
        str(args.threads),
        "-max_target_seqs",
        str(max_targets),
        "-outfmt",
        "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
        "-out",
        str(blast6_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(make_command) + "\n")
        log.flush()
        subprocess.run(make_command, check=True, stdout=log, stderr=subprocess.STDOUT)
        log.write(" ".join(blast_command) + "\n")
        log.flush()
        subprocess.run(blast_command, check=True, stdout=log, stderr=subprocess.STDOUT)


def parse_hits(path: Path, reference: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    hits = pd.read_csv(path, sep="\t", names=BLAST6_COLUMNS)
    hits["blast_identity"] = pd.to_numeric(hits["pident"], errors="coerce") / 100.0
    hits["blast_alignment_length"] = pd.to_numeric(hits["alignment_length"], errors="coerce")
    hits["blast_bitscore"] = pd.to_numeric(hits["bitscore"], errors="coerce")
    hits["blast_evalue"] = pd.to_numeric(hits["evalue"], errors="coerce")
    ref_meta = reference.drop(columns=["sequence"], errors="ignore")
    hits = hits.merge(ref_meta, on="ref_sequence_id", how="left")
    hits = hits.sort_values(
        [
            "query_id",
            "candidate_tree_label",
            "blast_bitscore",
            "blast_identity",
            "blast_alignment_length",
            "blast_evalue",
        ],
        ascending=[True, True, False, False, False, True],
    )
    species_hits = hits.drop_duplicates(["query_id", "candidate_tree_label"], keep="first").copy()
    species_hits = species_hits.sort_values(
        [
            "query_id",
            "blast_bitscore",
            "blast_identity",
            "blast_alignment_length",
            "blast_evalue",
            "candidate_tree_label",
        ],
        ascending=[True, False, False, False, True, True],
    )
    species_hits["candidate_rank"] = species_hits.groupby("query_id").cumcount() + 1
    species_hits = species_hits[species_hits["candidate_rank"] <= top_k].copy()
    species_hits["candidate_source"] = "same_marker_12s_blastn_local"
    species_hits["score"] = species_hits["blast_bitscore"]
    keep = [
        "query_id",
        "candidate_rank",
        "candidate_source",
        "score",
        "candidate_tree_label",
        "candidate_species",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
        "ref_sequence_id",
        "blast_identity",
        "blast_alignment_length",
        "blast_bitscore",
        "blast_evalue",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "sequence_length",
    ]
    return species_hits[[column for column in keep if column in species_hits.columns]]


def hit_flags(group: pd.DataFrame, query: pd.Series, top_k: int) -> dict[str, bool]:
    subset = group.head(top_k)
    targets = {
        "species": clean(query["query_tree_label"]),
        "genus": clean(query["query_genus"]),
        "family": clean(query["query_family"]),
        "order": clean(query["query_order"]),
    }
    columns = {
        "species": "candidate_tree_label",
        "genus": "candidate_genus",
        "family": "candidate_family",
        "order": "candidate_order",
    }
    return {
        rank: bool(targets[rank] and not subset.empty and (subset[columns[rank]].astype(str) == targets[rank]).any())
        for rank in RANKS
    }


def support_tables(candidates: pd.DataFrame, queries: pd.DataFrame, top_ks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_by_id = queries.set_index("query_id")
    per_query_rows: list[dict[str, Any]] = []
    for query_id, query in query_by_id.iterrows():
        group = candidates[candidates["query_id"] == query_id].sort_values("candidate_rank")
        row: dict[str, Any] = {
            "query_id": query_id,
            "candidate_source_eval": "same_marker_blastn_local",
            "candidate_count": int(len(group)),
        }
        for top_k in top_ks:
            flags = hit_flags(group, query, top_k)
            for rank, value in flags.items():
                row[f"top{top_k}_{rank}_hit"] = bool(value)
        if len(group):
            first = group.iloc[0]
            row.update(
                {
                    "top1_candidate_tree_label": first.get("candidate_tree_label", ""),
                    "top1_candidate_genus": first.get("candidate_genus", ""),
                    "top1_candidate_family": first.get("candidate_family", ""),
                    "top1_candidate_order": first.get("candidate_order", ""),
                    "top1_blast_identity": float(first.get("blast_identity", math.nan)),
                    "top1_blast_alignment_length": float(first.get("blast_alignment_length", math.nan)),
                    "top1_blast_bitscore": float(first.get("blast_bitscore", math.nan)),
                }
            )
        per_query_rows.append(row)
    per_query = pd.DataFrame(per_query_rows)

    summary_rows: list[dict[str, Any]] = []
    for top_k in top_ks:
        row: dict[str, Any] = {
            "candidate_source_eval": "same_marker_blastn_local",
            "top_k": int(top_k),
            "n_queries": int(len(per_query)),
            "mean_candidate_count": float(per_query["candidate_count"].mean()),
        }
        for rank in RANKS:
            row[f"{rank}_hit_pct"] = float(per_query[f"top{top_k}_{rank}_hit"].mean() * 100.0)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), per_query


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    script_name = Path(__file__).name
    logger.start(script_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    top_ks = [int(value.strip()) for value in args.report_top_ks.split(",") if value.strip()]
    work_dir = args.output_dir / "marker_mirror_same_marker_blast_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    query_fasta = work_dir / "queries.fa"
    ref_fasta = work_dir / "reference.fa"
    db_prefix = work_dir / "reference_blastdb"
    blast6_path = work_dir / "blast_hits.blast6.tsv"
    blast_log = work_dir / "blastn.log"

    logger.log(f"Loading queries {rel(args.query_table)}")
    queries = load_queries(args.query_table, args.limit_queries)
    logger.log(f"Loaded queries rows={len(queries):,}")
    logger.log(f"Loading reference {rel(args.same_marker_reference_dir)}")
    reference = load_reference(args.same_marker_reference_dir, args.max_ref_sequences_per_species)
    logger.log(f"Loaded reference sequence rows={len(reference):,}")
    write_fasta(queries, "query_id", "query_sequence", query_fasta)
    write_fasta(reference, "ref_sequence_id", "sequence", ref_fasta)
    logger.log(f"Wrote FASTA query={rel(query_fasta)} reference={rel(ref_fasta)}")

    logger.log("Running BLASTN local alignment")
    run_blast(args, query_fasta, ref_fasta, db_prefix, blast6_path, blast_log)
    logger.log(f"Parsing BLAST hits {rel(blast6_path)}")
    candidates = parse_hits(blast6_path, reference, args.top_k)
    logger.log(f"Candidate rows={len(candidates):,}")
    summary, per_query = support_tables(candidates, queries, top_ks)

    candidate_path = args.output_dir / "marker_mirror_same_marker_blast_candidates_top50.csv.gz"
    summary_path = args.output_dir / "marker_mirror_same_marker_blast_support_summary.csv"
    per_query_path = args.output_dir / "marker_mirror_same_marker_blast_support_per_query.csv"
    manifest_path = args.output_dir / "marker_mirror_same_marker_blast_manifest.json"
    candidates.to_csv(candidate_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_query.to_csv(per_query_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "query_table": rel(args.query_table),
            "same_marker_reference_dir": rel(args.same_marker_reference_dir),
        },
        "outputs": {
            "candidates": rel(candidate_path),
            "summary": rel(summary_path),
            "per_query": rel(per_query_path),
            "work_dir": rel(work_dir),
        },
        "n_queries": int(len(queries)),
        "n_reference_sequences": int(len(reference)),
        "n_candidate_rows": int(len(candidates)),
        "top_k": int(args.top_k),
        "max_ref_sequences_per_species": int(args.max_ref_sequences_per_species),
        "blast": {
            "task": str(args.task),
            "evalue": float(args.evalue),
            "threads": int(args.threads),
            "max_target_seqs": int(args.max_target_seqs or max(args.top_k * args.max_ref_sequences_per_species * 2, args.top_k * 20)),
        },
        "claim_boundary": "BLASTN local-alignment same-marker candidate generation over current 12S reference; cannot recover species absent from the reference.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote {rel(candidate_path)}")
    logger.log(f"Wrote {rel(summary_path)}")
    logger.log(f"Wrote {rel(per_query_path)}")
    logger.log(f"Wrote {rel(manifest_path)}")
    logger.done(script_name)


if __name__ == "__main__":
    main()
