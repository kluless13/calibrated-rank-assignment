#!/usr/bin/env python3
"""Evaluate COI fish-tree sequence baselines on Stalder-style inputs.

This script is for the clean fish-tree inputs under
data/phylo/fish_tree_clean_phylo_inputs/{eval_c,seen_test,unseen_genera}.
It builds a reference database only from train_species_sequences.json, ranks
candidate labels by sequence similarity, then evaluates the ranked labels with
eval_zero_shot_candidate_predictions.py.

Important claim boundary: BLAST/VSEARCH/k-mer baselines can only rank species
that have reference sequences. The neural tree-space model can rank every
candidate species in the tree, including held-out species with no reference
sequence. That distinction is recorded in each manifest.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DNA = "ACGT"
RANK_COLUMNS = ["species_name", "genus_name", "family_name", "order_name"]


@dataclass
class ReferenceDB:
    labels: list[str]
    sequences: list[str]
    label_to_candidate_index: dict[str, int]
    candidate_labels: list[str]


def clean_seq(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in "ACGTN")


def fasta_seq(seq: object) -> str:
    seq_clean = clean_seq(seq)
    return seq_clean if seq_clean else "N"


def load_reference_db(input_dir: Path, max_ref_seqs_per_species: int | None) -> ReferenceDB:
    candidate_df = pd.read_csv(input_dir / "candidate_species.csv")
    candidate_labels = candidate_df["tree_label"].astype(str).tolist()
    label_to_candidate_index = {label: idx for idx, label in enumerate(candidate_labels)}
    raw = json.loads((input_dir / "train_species_sequences.json").read_text())
    labels: list[str] = []
    sequences: list[str] = []
    for label, seqs in raw.items():
        if label not in label_to_candidate_index:
            continue
        seq_list = [fasta_seq(seq) for seq in seqs if clean_seq(seq)]
        if max_ref_seqs_per_species is not None:
            seq_list = seq_list[:max_ref_seqs_per_species]
        for seq in seq_list:
            labels.append(label)
            sequences.append(seq)
    if not labels:
        raise RuntimeError(f"No usable reference sequences in {input_dir}")
    return ReferenceDB(
        labels=labels,
        sequences=sequences,
        label_to_candidate_index=label_to_candidate_index,
        candidate_labels=candidate_labels,
    )


def write_fasta(labels: list[str], sequences: list[str], path: Path, prefix: str) -> None:
    with path.open("w") as handle:
        for idx, (label, seq) in enumerate(zip(labels, sequences, strict=True)):
            handle.write(f">{prefix}_{idx}|{label}\n{seq}\n")


def write_query_fasta(queries: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for idx, row in queries.reset_index(drop=True).iterrows():
            processid = str(row.get("processid", f"query_{idx}"))
            handle.write(f">query_{idx}|{processid}\n{fasta_seq(row['nucleotides'])}\n")


def kmer_index(k: int) -> dict[str, int]:
    kmers = [""]
    for _ in range(k):
        kmers = [prefix + base for prefix in kmers for base in DNA]
    return {kmer: idx for idx, kmer in enumerate(kmers)}


def kmer_matrix(sequences: list[str], k: int, index: dict[str, int]) -> np.ndarray:
    mat = np.zeros((len(sequences), len(index)), dtype=np.float32)
    for row_idx, seq in enumerate(sequences):
        text = clean_seq(seq).replace("N", "")
        if len(text) < k:
            continue
        row = mat[row_idx]
        for pos in range(len(text) - k + 1):
            kmer = text[pos : pos + k]
            idx = index.get(kmer)
            if idx is not None:
                row[idx] += 1.0
        norm = float(np.linalg.norm(row))
        if norm > 0:
            row /= norm
    return mat


def top_species_from_scores(
    scores: np.ndarray,
    ref_candidate_indices: np.ndarray,
    candidate_labels: list[str],
    top_k: int,
) -> tuple[list[str], list[float]]:
    species_scores = np.full(len(candidate_labels), -np.inf, dtype=np.float32)
    np.maximum.at(species_scores, ref_candidate_indices, scores)
    valid = np.isfinite(species_scores)
    if not valid.any():
        return [], []
    top_n = min(top_k, int(valid.sum()))
    top_idx_unsorted = np.argpartition(-species_scores, np.arange(top_n))[:top_n]
    top_idx = top_idx_unsorted[np.argsort(-species_scores[top_idx_unsorted])]
    return [candidate_labels[int(idx)] for idx in top_idx], [float(species_scores[int(idx)]) for idx in top_idx]


def predict_kmer(input_dir: Path, ref: ReferenceDB, queries: pd.DataFrame, output_dir: Path, k: int, top_k: int, batch_size: int) -> Path:
    index = kmer_index(k)
    ref_mat = kmer_matrix(ref.sequences, k, index)
    ref_candidate_indices = np.array([ref.label_to_candidate_index[label] for label in ref.labels], dtype=np.int64)
    rows = []
    query_sequences = queries["nucleotides"].astype(str).tolist()
    start = time.time()
    for start_idx in range(0, len(query_sequences), batch_size):
        batch_sequences = query_sequences[start_idx : start_idx + batch_size]
        query_mat = kmer_matrix(batch_sequences, k, index)
        sim = query_mat @ ref_mat.T
        for local_idx in range(sim.shape[0]):
            labels, scores = top_species_from_scores(
                sim[local_idx],
                ref_candidate_indices,
                ref.candidate_labels,
                top_k,
            )
            query = queries.iloc[start_idx + local_idx]
            rows.append(prediction_row(query, labels, scores))
    elapsed = time.time() - start
    prediction_csv = output_dir / "zero_shot_candidate_predictions.csv"
    pd.DataFrame(rows).to_csv(prediction_csv, index=False)
    (output_dir / "runtime.json").write_text(json.dumps({"seconds": elapsed}, indent=2) + "\n")
    return prediction_csv


def parse_blast6(path: Path, top_k: int, score_field: str = "bitscore") -> dict[int, tuple[list[str], list[float]]]:
    per_query: dict[int, list[tuple[str, float]]] = {}
    if not path.exists():
        return {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            try:
                qidx = int(parts[0].split("|", 1)[0].rsplit("_", 1)[1])
                label = parts[1].split("|", 1)[1]
                score = float(parts[11] if score_field == "bitscore" else parts[2])
            except (IndexError, ValueError):
                continue
            per_query.setdefault(qidx, []).append((label, score))

    ranked = {}
    for qidx, hits in per_query.items():
        best_by_label: dict[str, float] = {}
        for label, score in hits:
            if label not in best_by_label or score > best_by_label[label]:
                best_by_label[label] = score
        ordered = sorted(best_by_label.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ranked[qidx] = ([label for label, _ in ordered], [float(score) for _, score in ordered])
    return ranked


def predict_blast(
    ref: ReferenceDB,
    queries: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    threads: int,
    task: str,
    max_target_seqs: int,
) -> Path:
    if not shutil.which("blastn") or not shutil.which("makeblastdb"):
        raise RuntimeError("blastn/makeblastdb not found")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_fasta = tmpdir / "reference.fasta"
        query_fasta = tmpdir / "queries.fasta"
        blast_out = output_dir / "blast_hits.tsv"
        db_prefix = tmpdir / "blastdb"
        write_fasta(ref.labels, ref.sequences, db_fasta, "ref")
        write_query_fasta(queries, query_fasta)
        subprocess.run(
            ["makeblastdb", "-in", str(db_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
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
                "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
                "-max_target_seqs",
                str(max_target_seqs),
                "-max_hsps",
                "1",
                "-evalue",
                "1e-5",
                "-num_threads",
                str(threads),
                "-out",
                str(blast_out),
            ],
            check=True,
        )
    elapsed = time.time() - start
    ranked = parse_blast6(blast_out, top_k=top_k, score_field="bitscore")
    prediction_csv = output_dir / "zero_shot_candidate_predictions.csv"
    rows = []
    for idx, query in queries.reset_index(drop=True).iterrows():
        labels, scores = ranked.get(idx, ([], []))
        rows.append(prediction_row(query, labels, scores))
    pd.DataFrame(rows).to_csv(prediction_csv, index=False)
    (output_dir / "runtime.json").write_text(json.dumps({"seconds": elapsed}, indent=2) + "\n")
    return prediction_csv


def predict_vsearch(
    ref: ReferenceDB,
    queries: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    threads: int,
    min_id: float,
    maxaccepts: int,
) -> Path:
    if not shutil.which("vsearch"):
        raise RuntimeError("vsearch not found")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_fasta = tmpdir / "reference.fasta"
        query_fasta = tmpdir / "queries.fasta"
        vsearch_out = output_dir / "vsearch_hits.tsv"
        write_fasta(ref.labels, ref.sequences, db_fasta, "ref")
        write_query_fasta(queries, query_fasta)
        start = time.time()
        subprocess.run(
            [
                "vsearch",
                "--usearch_global",
                str(query_fasta),
                "--db",
                str(db_fasta),
                "--id",
                str(min_id),
                "--maxaccepts",
                str(maxaccepts),
                "--maxrejects",
                "0",
                "--blast6out",
                str(vsearch_out),
                "--threads",
                str(threads),
                "--quiet",
            ],
            check=True,
        )
    elapsed = time.time() - start
    ranked = parse_blast6(vsearch_out, top_k=top_k, score_field="pident")
    prediction_csv = output_dir / "zero_shot_candidate_predictions.csv"
    rows = []
    for idx, query in queries.reset_index(drop=True).iterrows():
        labels, scores = ranked.get(idx, ([], []))
        rows.append(prediction_row(query, labels, scores))
    pd.DataFrame(rows).to_csv(prediction_csv, index=False)
    (output_dir / "runtime.json").write_text(json.dumps({"seconds": elapsed}, indent=2) + "\n")
    return prediction_csv


def prediction_row(query: pd.Series, labels: list[str], scores: list[float]) -> dict[str, object]:
    row = {
        "processid": query.get("processid", ""),
        "true_tree_label": query.get("tree_label", ""),
        "species_name": query.get("species_name", ""),
        "genus_name": query.get("genus_name", ""),
        "family_name": query.get("family_name", ""),
        "order_name": query.get("order_name", ""),
        "top_tree_labels": json.dumps(labels),
        "top_scores": json.dumps(scores),
        "pred_tree_label": labels[0] if labels else "",
        "pred_score": scores[0] if scores else None,
    }
    return row


def evaluate_predictions(input_dir: Path, prediction_csv: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/edna/eval_zero_shot_candidate_predictions.py",
            "--input-dir",
            str(input_dir),
            "--predictions",
            str(prediction_csv),
            "--output-dir",
            str(output_dir / "zero_shot_metrics"),
            "--top-k",
            "1",
            "5",
            "10",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["kmer", "blast", "vsearch"], default=["kmer", "blast", "vsearch"])
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--blast-task", default="megablast")
    parser.add_argument("--blast-max-target-seqs", type=int, default=500)
    parser.add_argument("--vsearch-min-id", type=float, default=0.5)
    parser.add_argument("--vsearch-maxaccepts", type=int, default=500)
    parser.add_argument("--max-ref-seqs-per-species", type=int, default=None)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading queries from {args.input_dir / 'zero_shot_queries.csv'}")
    queries = pd.read_csv(args.input_dir / "zero_shot_queries.csv").reset_index(drop=True)
    logger.log(f"Loading reference database from {args.input_dir}")
    ref = load_reference_db(args.input_dir, args.max_ref_seqs_per_species)
    logger.log(
        f"Loaded {len(queries)} queries, {len(ref.sequences)} reference sequences, "
        f"{len(set(ref.labels))} reference species, {len(ref.candidate_labels)} candidate species"
    )

    method_outputs = {}
    for method in args.methods:
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.log(f"Starting {method} baseline")
            if method == "kmer":
                prediction_csv = predict_kmer(args.input_dir, ref, queries, method_dir, args.k, args.top_k, args.batch_size)
            elif method == "blast":
                prediction_csv = predict_blast(
                    ref,
                    queries,
                    method_dir,
                    args.top_k,
                    args.threads,
                    args.blast_task,
                    args.blast_max_target_seqs,
                )
            else:
                prediction_csv = predict_vsearch(
                    ref,
                    queries,
                    method_dir,
                    args.top_k,
                    args.threads,
                    args.vsearch_min_id,
                    args.vsearch_maxaccepts,
                )
            logger.log(f"Evaluating {method} predictions from {prediction_csv}")
            evaluate_predictions(args.input_dir, prediction_csv, method_dir)
            method_outputs[method] = {
                "prediction_csv": str(prediction_csv),
                "metrics_json": str(method_dir / "zero_shot_metrics" / "zero_shot_candidate_metrics.json"),
            }
            logger.log(f"Finished {method} baseline")
        except Exception as exc:
            logger.log(f"{method} baseline failed: {exc}")
            method_outputs[method] = {"error": str(exc)}
            (method_dir / "error.txt").write_text(str(exc) + "\n")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "query_count": int(len(queries)),
        "reference_sequence_count": len(ref.sequences),
        "reference_species_count": len(set(ref.labels)),
        "candidate_species_count": len(ref.candidate_labels),
        "claim_boundary": (
            "Sequence baselines rank only species with reference sequences; "
            "held-out candidate species without sequence references cannot be recovered by species label."
        ),
        "args": vars(args),
        "methods": method_outputs,
    }
    manifest_path = args.output_dir / "fish_tree_candidate_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
