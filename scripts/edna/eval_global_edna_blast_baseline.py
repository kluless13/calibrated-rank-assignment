#!/usr/bin/env python3
"""Run a BLAST-style baseline for Global_eDNA open-candidate ASV validation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RANKS = ["species", "genus", "family", "order"]


def clean_seq(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper().strip() if ch in "ACGTN")


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def rank_value(label: str | None, rank: str, taxonomy: dict[str, dict[str, object]]) -> str:
    if not label:
        return ""
    if rank == "species":
        return label
    if rank == "genus":
        value = taxonomy.get(label, {}).get("genus_name")
        return str(value) if nonempty(value) else label.split("_", 1)[0]
    value = taxonomy.get(label, {}).get(f"{rank}_name")
    return str(value) if nonempty(value) else ""


def write_reference_fasta(train_species_sequences: dict[str, list[str]], path: Path) -> pd.DataFrame:
    rows = []
    with path.open("w") as handle:
        ref_idx = 0
        for label, seqs in sorted(train_species_sequences.items()):
            for seq_idx, seq in enumerate(seqs):
                cleaned = clean_seq(seq)
                if not cleaned:
                    continue
                sid = f"ref_{ref_idx}"
                handle.write(f">{sid}\n{cleaned}\n")
                rows.append({
                    "sid": sid,
                    "ref_index": ref_idx,
                    "tree_label": label,
                    "sequence_index": seq_idx,
                    "seq_len": len(cleaned),
                })
                ref_idx += 1
    return pd.DataFrame(rows)


def write_query_fasta(queries: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    with path.open("w") as handle:
        for qidx, row in queries.reset_index(drop=True).iterrows():
            cleaned = clean_seq(row["nucleotides"])
            qid = f"query_{qidx}"
            handle.write(f">{qid}\n{cleaned}\n")
            out = row.to_dict()
            out["qid"] = qid
            out["query_index"] = qidx
            out["seq_len_clean"] = len(cleaned)
            rows.append(out)
    return pd.DataFrame(rows)


def run_blast(
    ref_fasta: Path,
    query_fasta: Path,
    out_path: Path,
    threads: int,
    task: str,
    max_target_seqs: int,
    evalue: str,
) -> float:
    db_prefix = ref_fasta.parent / "blastdb"
    subprocess.run(
        ["makeblastdb", "-in", str(ref_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start = time.time()
    subprocess.run(
        [
            "blastn",
            "-task",
            task,
            "-query",
            str(query_fasta),
            "-db",
            str(db_prefix),
            "-outfmt",
            "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen",
            "-max_target_seqs",
            str(max_target_seqs),
            "-max_hsps",
            "1",
            "-evalue",
            evalue,
            "-dust",
            "no",
            "-num_threads",
            str(threads),
            "-out",
            str(out_path),
        ],
        check=True,
    )
    return time.time() - start


def parse_blast_hits(path: Path) -> pd.DataFrame:
    columns = [
        "qid",
        "sid",
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
        "qlen",
        "slen",
    ]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    hits = pd.read_csv(path, sep="\t", names=columns)
    for col in ["pident", "alignment_length", "bitscore", "qlen", "slen"]:
        hits[col] = pd.to_numeric(hits[col], errors="coerce")
    hits["qcov"] = hits["alignment_length"] / hits["qlen"].clip(lower=1)
    return hits


def collapse_hits_by_species(hits: pd.DataFrame, refs: pd.DataFrame, top_k: int) -> dict[str, list[dict[str, object]]]:
    if hits.empty:
        return {}
    merged = hits.merge(refs[["sid", "tree_label"]], on="sid", how="left")
    merged = merged[merged["tree_label"].notna()].copy()
    merged = merged.sort_values(
        ["qid", "bitscore", "pident", "qcov", "alignment_length"],
        ascending=[True, False, False, False, False],
    )
    best_by_query_species = merged.drop_duplicates(["qid", "tree_label"], keep="first")
    out: dict[str, list[dict[str, object]]] = {}
    for qid, sub in best_by_query_species.groupby("qid", sort=False):
        records = sub.head(top_k).to_dict(orient="records")
        out[str(qid)] = records
    return out


def build_prediction_tables(
    queries: pd.DataFrame,
    hits_by_query: dict[str, list[dict[str, object]]],
    taxonomy: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zero_rows = []
    rank_rows = []
    for _, row in queries.iterrows():
        hits = hits_by_query.get(str(row["qid"]), [])
        labels = [str(hit["tree_label"]) for hit in hits]
        scores = [float(hit["pident"]) for hit in hits]
        pred_label = labels[0] if labels else ""
        pred_score = scores[0] if scores else None
        true_label = str(row.get("tree_label", "")).replace(" ", "_")

        zero = {
            "processid": row["processid"],
            "true_tree_label": true_label,
            "species_name": row.get("species_name"),
            "top_tree_labels": json.dumps(labels),
            "top_scores": json.dumps([round(score, 6) for score in scores]),
            "pred_tree_label": pred_label,
            "pred_score": pred_score,
            "blast_pident": pred_score,
            "blast_qcov": float(hits[0]["qcov"]) if hits else None,
            "blast_bitscore": float(hits[0]["bitscore"]) if hits else None,
            "has_hit": bool(hits),
        }
        zero_rows.append(zero)

        rank = {
            "processid": row["processid"],
            "has_hit": bool(hits),
            "blast_pident": pred_score,
            "blast_qcov": float(hits[0]["qcov"]) if hits else None,
            "blast_bitscore": float(hits[0]["bitscore"]) if hits else None,
        }
        for level in RANKS:
            rank[f"true_{level}"] = rank_value(true_label, level, taxonomy)
            rank[f"blast_pred_{level}"] = rank_value(pred_label, level, taxonomy)
        rank_rows.append(rank)
    return pd.DataFrame(zero_rows), pd.DataFrame(rank_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--task", default="blastn-short")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-target-seqs", type=int, default=100)
    parser.add_argument("--evalue", default="1000")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_species_sequences = json.loads((args.input_dir / "train_species_sequences.json").read_text())
    queries = pd.read_csv(args.input_dir / "zero_shot_queries.csv")
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")

    with tempfile.TemporaryDirectory(prefix="global_edna_blast_") as tmp:
        tmpdir = Path(tmp)
        ref_fasta = tmpdir / "references.fasta"
        query_fasta = tmpdir / "queries.fasta"
        blast_out = tmpdir / "hits.tsv"
        refs = write_reference_fasta(train_species_sequences, ref_fasta)
        query_table = write_query_fasta(queries, query_fasta)
        elapsed = run_blast(
            ref_fasta,
            query_fasta,
            blast_out,
            threads=args.threads,
            task=args.task,
            max_target_seqs=args.max_target_seqs,
            evalue=args.evalue,
        )
        hits = parse_blast_hits(blast_out)

    hits_by_query = collapse_hits_by_species(hits, refs, args.top_k)
    zero_pred, rank_pred = build_prediction_tables(query_table, hits_by_query, taxonomy)

    zero_path = args.output_dir / "global_edna_blast_zero_shot_predictions.csv"
    rank_path = args.output_dir / "global_edna_blast_rank_predictions.csv"
    zero_pred.to_csv(zero_path, index=False)
    rank_pred.to_csv(rank_path, index=False)

    validation_dir = args.output_dir / "global_edna_validation"
    subprocess.run(
        [
            sys.executable,
            "scripts/edna/eval_global_edna_sample_validation.py",
            "--input-dir",
            str(args.input_dir),
            "--predictions",
            str(zero_path),
            "--sample-query-map",
            str(args.input_dir / "sample_query_map.csv"),
            "--output-dir",
            str(validation_dir),
        ],
        check=True,
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "zero_shot_predictions_csv": str(zero_path),
        "rank_predictions_csv": str(rank_path),
        "validation_dir": str(validation_dir),
        "blast_task": args.task,
        "threads": args.threads,
        "top_k": args.top_k,
        "max_target_seqs": args.max_target_seqs,
        "evalue": args.evalue,
        "elapsed_seconds": elapsed,
        "reference_sequences": int(len(refs)),
        "reference_species": int(refs["tree_label"].nunique()),
        "query_asvs": int(len(query_table)),
        "queries_with_hits": int(zero_pred["has_hit"].sum()),
        "notes": [
            "Reference database uses train_species_sequences.json only.",
            "Species absent from the reference cannot be exact species hits; higher-rank agreement remains meaningful.",
        ],
    }
    manifest_path = args.output_dir / "global_edna_blast_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {zero_path}, {rank_path}, and {validation_dir}")


if __name__ == "__main__":
    main()
