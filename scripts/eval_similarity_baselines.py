"""
Leakage-audited BLAST/VSEARCH-style nearest-hit baselines.

Builds a reference database only from supervised_train.csv, then evaluates
supervised_test.csv, eval_c_unseen_species.csv, and unseen.csv at species,
genus, family, and order where meaningful.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score


RANKS = [
    ("species", "species_name"),
    ("genus", "genus_name"),
    ("family", "family_name"),
    ("order", "order_name"),
]


def clean_seq(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in "ACGTN")


def write_fasta(df: pd.DataFrame, path: Path, prefix: str) -> dict[str, int]:
    id_to_pos = {}
    with path.open("w") as f:
        for pos, row in enumerate(df.itertuples(index=False)):
            seq_id = f"{prefix}_{pos}"
            id_to_pos[seq_id] = pos
            f.write(f">{seq_id}\n{clean_seq(getattr(row, 'nucleotides'))}\n")
    return id_to_pos


def parse_tabular(path: Path) -> dict[str, dict]:
    hits = {}
    if not path.exists():
        return hits
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            qid, sid = parts[0], parts[1]
            if qid in hits:
                continue
            try:
                pident = float(parts[2])
                aln_len = int(float(parts[3]))
                bitscore = float(parts[11])
            except ValueError:
                pident, aln_len, bitscore = None, None, None
            hits[qid] = {
                "sseqid": sid,
                "pident": pident,
                "alignment_length": aln_len,
                "bitscore": bitscore,
            }
    return hits


def evaluate_hits(
    train_df: pd.DataFrame,
    query_df: pd.DataFrame,
    hits: dict[str, dict],
    split_name: str,
) -> dict:
    q = query_df.reset_index(drop=True)
    no_hit = 0
    nearest_rows = []
    pidents = []
    qcovs = []

    for pos, row in q.iterrows():
        qid = f"query_{pos}"
        hit = hits.get(qid)
        if not hit:
            no_hit += 1
            nearest_rows.append(None)
            continue
        try:
            ref_pos = int(hit["sseqid"].rsplit("_", 1)[1])
        except (IndexError, ValueError):
            no_hit += 1
            nearest_rows.append(None)
            continue
        nearest_rows.append(train_df.iloc[ref_pos])
        if hit.get("pident") is not None:
            pidents.append(hit["pident"])
        if hit.get("alignment_length") is not None:
            qlen = max(1, len(clean_seq(row["nucleotides"])))
            qcovs.append(hit["alignment_length"] / qlen)

    result = {
        "n_query": int(len(q)),
        "n_hit": int(len(q) - no_hit),
        "no_hit_rate": float(no_hit / len(q)) if len(q) else None,
        "mean_pident": float(sum(pidents) / len(pidents)) if pidents else None,
        "mean_query_coverage": float(sum(qcovs) / len(qcovs)) if qcovs else None,
    }

    for rank, col in RANKS:
        if split_name == "eval_c_unseen_species" and rank == "species":
            result[rank] = {
                "accuracy": None,
                "note": "Not applicable: Eval C query species are absent from train reference labels.",
            }
            continue
        pairs = []
        for pos, nearest in enumerate(nearest_rows):
            if nearest is None:
                continue
            true = str(q.at[pos, col]) if col in q.columns else ""
            pred = str(nearest[col]) if col in nearest.index else ""
            if true and pred and true != "nan" and pred != "nan":
                pairs.append((true, pred))
        if pairs:
            y_true, y_pred = zip(*pairs)
            result[rank] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "n_correct": int(sum(a == b for a, b in pairs)),
                "n_total": int(len(pairs)),
            }
    return result


def run_blast(db_fasta: Path, query_fasta: Path, out_path: Path, threads: int, task: str) -> float:
    db_prefix = db_fasta.parent / "blastdb"
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
            "-task", task,
            "-query", str(query_fasta),
            "-db", str(db_prefix),
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-max_target_seqs", "1",
            "-max_hsps", "1",
            "-evalue", "1e-5",
            "-num_threads", str(threads),
            "-out", str(out_path),
        ],
        check=True,
    )
    return time.time() - start


def run_vsearch(db_fasta: Path, query_fasta: Path, out_path: Path, threads: int, min_id: float) -> float:
    start = time.time()
    subprocess.run(
        [
            "vsearch",
            "--usearch_global", str(query_fasta),
            "--db", str(db_fasta),
            "--id", str(min_id),
            "--maxaccepts", "1",
            "--maxrejects", "0",
            "--blast6out", str(out_path),
            "--threads", str(threads),
            "--quiet",
        ],
        check=True,
    )
    return time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=["blast", "vsearch"], default=["blast", "vsearch"])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--blast-task", default="megablast")
    parser.add_argument("--vsearch-min-id", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "supervised_train.csv").reset_index(drop=True)
    split_files = [
        ("supervised_test", data_dir / "supervised_test.csv"),
        ("eval_c_unseen_species", data_dir / "eval_c_unseen_species.csv"),
        ("unseen", data_dir / "unseen.csv"),
    ]
    splits = [(name, pd.read_csv(path).reset_index(drop=True)) for name, path in split_files if path.exists()]

    result = {
        "experiment": "similarity_baselines",
        "data_dir": str(data_dir),
        "n_train": int(len(train_df)),
        "methods": {},
        "audit": {},
    }

    if any(name == "eval_c_unseen_species" for name, _ in splits):
        eval_c_df = dict(splits)["eval_c_unseen_species"]
        result["audit"]["eval_c_species_leak_into_train"] = int(
            len(set(eval_c_df["species_name"].dropna()) & set(train_df["species_name"].dropna()))
        )
        result["audit"]["eval_c_sequence_leak_into_train"] = int(
            len(set(eval_c_df["nucleotides"]) & set(train_df["nucleotides"]))
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_fasta = tmpdir / "train.fasta"
        write_fasta(train_df, db_fasta, "ref")

        for method in args.methods:
            if method == "blast" and not shutil.which("blastn"):
                result["methods"][method] = {"error": "blastn not found"}
                continue
            if method == "vsearch" and not shutil.which("vsearch"):
                result["methods"][method] = {"error": "vsearch not found"}
                continue

            method_result = {}
            for split_name, query_df in splits:
                query_fasta = tmpdir / f"{split_name}.fasta"
                out_tab = tmpdir / f"{method}_{split_name}.tsv"
                write_fasta(query_df, query_fasta, "query")
                if method == "blast":
                    elapsed = run_blast(db_fasta, query_fasta, out_tab, args.threads, args.blast_task)
                else:
                    elapsed = run_vsearch(db_fasta, query_fasta, out_tab, args.threads, args.vsearch_min_id)
                hits = parse_tabular(out_tab)
                split_result = evaluate_hits(train_df, query_df, hits, split_name)
                split_result["seconds"] = float(elapsed)
                split_result["queries_per_second"] = float(len(query_df) / elapsed) if elapsed > 0 else None
                method_result[split_name] = split_result
            result["methods"][method] = method_result

    out = output_dir / "similarity_baselines_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
