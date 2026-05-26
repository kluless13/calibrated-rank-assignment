"""
Evaluate BLAST confidence/no-call curves and optional BLAST+neural hybrids.

The script builds a BLAST database from supervised_train.csv, records the top
hit for each query sequence, and sweeps identity/coverage thresholds. If a
neural predictions CSV from eval_6mer_checkpoint_predictions.py is provided,
it also evaluates a hybrid: use BLAST when it passes threshold, otherwise use
the neural embedding-kNN prediction.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd


RANKS = ["species", "genus", "family", "order"]
RANK_COLS = {
    "species": "species_name",
    "genus": "genus_name",
    "family": "family_name",
    "order": "order_name",
}


def clean_seq(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in "ACGTN")


def clean_text(value: object) -> str:
    text = str(value)
    return "" if text == "nan" else text


def write_fasta(df: pd.DataFrame, path: Path, prefix: str) -> None:
    with path.open("w") as f:
        for pos, row in df.reset_index(drop=True).iterrows():
            f.write(f">{prefix}_{pos}\n{clean_seq(row['nucleotides'])}\n")


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


def parse_hits(path: Path) -> dict[int, dict]:
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
            try:
                qpos = int(qid.rsplit("_", 1)[1])
                spos = int(sid.rsplit("_", 1)[1])
                pident = float(parts[2])
                aln_len = int(float(parts[3]))
                bitscore = float(parts[11])
            except (ValueError, IndexError):
                continue
            if qpos not in hits:
                hits[qpos] = {
                    "ref_index": spos,
                    "pident": pident,
                    "alignment_length": aln_len,
                    "bitscore": bitscore,
                }
    return hits


def prediction_table(train_df: pd.DataFrame, query_df: pd.DataFrame, hits: dict[int, dict], split_name: str) -> pd.DataFrame:
    rows = []
    train = train_df.reset_index(drop=True)
    query = query_df.reset_index(drop=True)
    for pos, row in query.iterrows():
        hit = hits.get(pos)
        out = {
            "split": split_name,
            "query_index": int(pos),
            "processid": clean_text(row.get("processid", "")),
            "sequence_length": int(len(clean_seq(row.get("nucleotides", "")))),
            "has_hit": bool(hit),
            "blast_ref_index": None,
            "blast_pident": None,
            "blast_qcov": None,
            "blast_bitscore": None,
        }
        for rank, col in RANK_COLS.items():
            out[f"true_{rank}"] = clean_text(row.get(col, ""))
            out[f"blast_pred_{rank}"] = ""
        if hit is not None:
            nearest = train.iloc[int(hit["ref_index"])]
            out["blast_ref_index"] = int(hit["ref_index"])
            out["blast_pident"] = float(hit["pident"])
            out["blast_qcov"] = float(hit["alignment_length"] / max(1, out["sequence_length"]))
            out["blast_bitscore"] = float(hit["bitscore"])
            for rank, col in RANK_COLS.items():
                out[f"blast_pred_{rank}"] = clean_text(nearest.get(col, ""))
        rows.append(out)
    return pd.DataFrame(rows)


def accuracy_for(df: pd.DataFrame, pred_prefix: str, rank: str, split_name: str) -> dict:
    if split_name == "eval_c_unseen_species" and rank == "species":
        return {
            "accuracy": None,
            "note": "Not applicable: Eval C query species are absent from training labels.",
        }
    true_col = f"true_{rank}"
    pred_col = f"{pred_prefix}_pred_{rank}"
    valid = df[df[true_col].astype(bool) & df[pred_col].astype(bool)]
    if len(valid) == 0:
        return {"accuracy": None, "n_total": 0}
    correct = valid[true_col] == valid[pred_col]
    return {
        "accuracy": float(correct.mean()),
        "n_correct": int(correct.sum()),
        "n_total": int(len(valid)),
    }


def assigned_sweep(pred_df: pd.DataFrame, split_name: str, pident_values: list[float], qcov_values: list[float]) -> list[dict]:
    rows = []
    n_query = len(pred_df)
    for pident in pident_values:
        for qcov in qcov_values:
            assigned = pred_df[
                pred_df["has_hit"]
                & (pred_df["blast_pident"] >= pident)
                & (pred_df["blast_qcov"] >= qcov)
            ].copy()
            row = {
                "strategy": "blast_assigned_only",
                "split": split_name,
                "pident_threshold": pident,
                "qcov_threshold": qcov,
                "assignment_rate": float(len(assigned) / n_query) if n_query else None,
                "n_assigned": int(len(assigned)),
                "n_query": int(n_query),
            }
            for rank in RANKS:
                row[rank] = accuracy_for(assigned, "blast", rank, split_name)
            rows.append(row)
    return rows


def hybrid_sweep(
    blast_df: pd.DataFrame,
    neural_df: pd.DataFrame,
    split_name: str,
    pident_values: list[float],
    qcov_values: list[float],
    neural_prefix: str,
) -> list[dict]:
    merged = blast_df.merge(
        neural_df,
        on=["split", "query_index"],
        how="left",
        suffixes=("", "_neural"),
    )
    rows = []
    for pident in pident_values:
        for qcov in qcov_values:
            use_blast = (
                merged["has_hit"]
                & (merged["blast_pident"] >= pident)
                & (merged["blast_qcov"] >= qcov)
            )
            eval_df = pd.DataFrame({
                "split": merged["split"],
                "query_index": merged["query_index"],
            })
            n_blast = int(use_blast.sum())
            for rank in RANKS:
                true_col = f"true_{rank}"
                neural_true_col = f"{true_col}_neural"
                eval_df[true_col] = merged[true_col].where(
                    merged[true_col].astype(bool),
                    merged.get(neural_true_col, ""),
                )
                blast_pred = merged[f"blast_pred_{rank}"].fillna("")
                neural_pred_col = f"{neural_prefix}_pred_{rank}"
                eval_df[f"hybrid_pred_{rank}"] = blast_pred.where(
                    use_blast,
                    merged.get(neural_pred_col, "").fillna(""),
                )
            row = {
                "strategy": f"blast_or_{neural_prefix}",
                "split": split_name,
                "pident_threshold": pident,
                "qcov_threshold": qcov,
                "blast_assignment_rate": float(n_blast / len(merged)) if len(merged) else None,
                "n_blast": n_blast,
                "n_query": int(len(merged)),
            }
            for rank in RANKS:
                row[rank] = accuracy_for(eval_df, "hybrid", rank, split_name)
            rows.append(row)
    return rows


def load_neural_predictions(path: Path | None) -> dict[str, pd.DataFrame]:
    if path is None:
        return {}
    neural_dir = Path(path)
    predictions = {}
    for csv_path in neural_dir.glob("*_neural_predictions.csv"):
        split_name = csv_path.name.replace("_neural_predictions.csv", "")
        predictions[split_name] = pd.read_csv(csv_path)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neural-predictions-dir", default=None)
    parser.add_argument("--neural-prefix", choices=["knn", "direct"], default="knn")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--blast-task", default="megablast")
    parser.add_argument("--pident-thresholds", default="80,85,90,92,94,95,96,97,98,99")
    parser.add_argument("--qcov-thresholds", default="0.8,0.9,0.95")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pident_values = [float(x) for x in args.pident_thresholds.split(",") if x]
    qcov_values = [float(x) for x in args.qcov_thresholds.split(",") if x]
    neural_predictions = load_neural_predictions(Path(args.neural_predictions_dir) if args.neural_predictions_dir else None)

    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    splits = [
        ("supervised_test", data_dir / "supervised_test.csv"),
        ("eval_c_unseen_species", data_dir / "eval_c_unseen_species.csv"),
        ("unseen", data_dir / "unseen.csv"),
    ]

    result = {
        "experiment": "blast_threshold_hybrid",
        "data_dir": str(data_dir),
        "neural_predictions_dir": args.neural_predictions_dir,
        "neural_prefix": args.neural_prefix,
        "splits": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        db_fasta = tmpdir / "train.fasta"
        write_fasta(train_df, db_fasta, "ref")
        for split_name, split_path in splits:
            if not split_path.exists():
                continue
            query_df = pd.read_csv(split_path)
            query_fasta = tmpdir / f"{split_name}.fasta"
            hit_path = tmpdir / f"{split_name}.blast6"
            write_fasta(query_df, query_fasta, "query")
            elapsed = run_blast(db_fasta, query_fasta, hit_path, args.threads, args.blast_task)
            hits = parse_hits(hit_path)
            pred_df = prediction_table(train_df, query_df, hits, split_name)
            pred_csv = output_dir / f"{split_name}_blast_predictions.csv"
            pred_df.to_csv(pred_csv, index=False)

            base_metrics = {rank: accuracy_for(pred_df, "blast", rank, split_name) for rank in RANKS}
            sweeps = assigned_sweep(pred_df, split_name, pident_values, qcov_values)
            if split_name in neural_predictions:
                sweeps.extend(
                    hybrid_sweep(
                        pred_df,
                        neural_predictions[split_name],
                        split_name,
                        pident_values,
                        qcov_values,
                        args.neural_prefix,
                    )
                )
            result["splits"][split_name] = {
                "prediction_csv": str(pred_csv),
                "seconds": float(elapsed),
                "n_query": int(len(query_df)),
                "n_hit": int(pred_df["has_hit"].sum()),
                "no_hit_rate": float((~pred_df["has_hit"]).mean()) if len(pred_df) else None,
                "forced_blast": base_metrics,
                "sweeps": sweeps,
            }

    result_path = output_dir / "blast_threshold_hybrid_results.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
