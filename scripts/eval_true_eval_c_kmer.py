"""
Leakage-audited k-mer nearest-neighbor baseline for true Eval C.

This is model-free and runs against the clean splits:
  supervised_train.csv
  supervised_test.csv
  eval_c_unseen_species.csv

It is useful as a defensible baseline before spending GPU time. Eval C species
accuracy is intentionally not reported because query species are absent from
the reference labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def as_str_list(series: pd.Series) -> list[str]:
    return series.fillna("").astype(str).tolist()


def evaluate_neighbors(train_df: pd.DataFrame, query_df: pd.DataFrame, train_matrix, query_matrix) -> dict:
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(train_matrix)
    indices = nn.kneighbors(query_matrix, return_distance=False).ravel()
    nearest = train_df.iloc[indices].reset_index(drop=True)
    query = query_df.reset_index(drop=True)

    result = {}
    for level, col in [
        ("species", "species_name"),
        ("genus", "genus_name"),
        ("family", "family_name"),
        ("order", "order_name"),
    ]:
        if col not in query.columns or col not in nearest.columns:
            continue
        true = as_str_list(query[col])
        pred = as_str_list(nearest[col])
        valid = [(t, p) for t, p in zip(true, pred) if t and p and t != "nan" and p != "nan"]
        if valid:
            t, p = zip(*valid)
            acc = float(accuracy_score(t, p))
            result[level] = {
                "accuracy": acc,
                "n_correct": int(sum(a == b for a, b in valid)),
                "n_total": int(len(valid)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed_clean")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-features", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    test_df = pd.read_csv(data_dir / "supervised_test.csv")
    eval_c_df = pd.read_csv(data_dir / "eval_c_unseen_species.csv")

    eval_c_species = set(eval_c_df["species_name"].dropna())
    seen_species = set(train_df["species_name"].dropna()) | set(test_df["species_name"].dropna())
    eval_c_genera = set(eval_c_df["genus_name"].dropna())
    train_genera = set(train_df["genus_name"].dropna())
    seq_leak = set(eval_c_df["nucleotides"]) & set(train_df["nucleotides"])

    if eval_c_species & seen_species:
        raise RuntimeError(f"Eval C species leak into seen train/test: {len(eval_c_species & seen_species)}")
    if eval_c_genera - train_genera:
        raise RuntimeError(f"Eval C genera missing from train: {len(eval_c_genera - train_genera)}")
    if seq_leak:
        raise RuntimeError(f"Eval C sequence leak into train: {len(seq_leak)}")

    print(f"train={len(train_df):,} test={len(test_df):,} eval_c={len(eval_c_df):,}")
    print(f"k={args.k}")
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(args.k, args.k),
        lowercase=False,
        binary=False,
        max_features=args.max_features,
    )
    x_train = vectorizer.fit_transform(as_str_list(train_df["nucleotides"]))
    x_test = vectorizer.transform(as_str_list(test_df["nucleotides"]))
    x_eval_c = vectorizer.transform(as_str_list(eval_c_df["nucleotides"]))
    x_train = normalize(x_train, norm="l2", copy=False)
    x_test = normalize(x_test, norm="l2", copy=False)
    x_eval_c = normalize(x_eval_c, norm="l2", copy=False)
    print(f"vocab={len(vectorizer.vocabulary_):,}")

    test_result = evaluate_neighbors(train_df, test_df, x_train, x_test)
    eval_c_result = evaluate_neighbors(train_df, eval_c_df, x_train, x_eval_c)
    eval_c_result["species"] = {
        "accuracy": None,
        "note": "Not applicable: Eval C query species are absent from reference labels.",
    }

    result = {
        "experiment": "true_eval_c_kmer_nearest_neighbor",
        "data_dir": str(data_dir),
        "k": args.k,
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_eval_c": int(len(eval_c_df)),
        "audit": {
            "eval_c_species_leak_into_seen_train_test": 0,
            "eval_c_genera_missing_from_train": 0,
            "eval_c_sequence_leak_into_train": 0,
        },
        "test_seen_species": test_result,
        "eval_c_true_unseen_species_seen_genera": eval_c_result,
    }

    out = output_dir / f"eval_c_kmer{args.k}_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
