"""
Build a leakage-audited multi-source 12S fish reference split.

This intentionally does not concatenate existing per-source splits. It loads the
raw Mitohelper, rCRUX, and Mare-MAGE sources, normalizes them into one table,
deduplicates exact sequences across sources, then chooses Eval A/Eval C holdouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from build_12s_splits import (
    SEED_EVAL_C_SPECIES,
    SEED_SUPERVISED_SPLIT,
    SEED_UNSEEN_GENERA,
    clean_seq,
    drop_nonmajority_hierarchy,
    load_mare_mage,
    load_mitohelper_tsv,
    load_rcrux_zip,
    make_eval_c_holdout,
    normalise_taxon,
    overlap_audit,
    sha256_of_file,
)


SOURCE_PRIORITY = {
    "mitohelper": 0,
    "mare_mage": 1,
    "rcrux_cleaned": 2,
    "rcrux_blast_seeds": 3,
}


def prepare_source(
    df: pd.DataFrame,
    source: str,
    class_filter: str,
    min_len: int,
    max_len: int,
    max_n_frac: float,
) -> pd.DataFrame:
    df = df.copy()
    for col in ["species_name", "genus_name", "family_name", "order_name", "class_name", "phylum_name"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].apply(normalise_taxon)

    df = df[[
        "processid", "nucleotides", "species_name", "genus_name",
        "family_name", "order_name", "class_name", "phylum_name", "taxid",
    ]].copy()
    if class_filter.lower() != "all":
        df = df[df["class_name"].astype(str) == class_filter].copy()

    df = df.dropna(subset=["processid", "nucleotides", "species_name", "genus_name"]).copy()
    df["species_name"] = df["species_name"].astype(str).str.strip()
    df = df[df["species_name"].str.contains(" ", na=False)].copy()
    df["nucleotides"] = df["nucleotides"].apply(clean_seq)
    df["seq_len"] = df["nucleotides"].str.len()
    df = df[(df["seq_len"] >= min_len) & (df["seq_len"] <= max_len)].copy()
    df["n_frac"] = df["nucleotides"].str.count("N") / df["seq_len"]
    df = df[df["n_frac"] <= max_n_frac].copy()

    df["source"] = source
    df["source_processid"] = df["processid"].astype(str)
    df["processid"] = source + ":" + df["source_processid"].astype(str)
    df["source_priority"] = SOURCE_PRIORITY[source]
    return df


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_multisource(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict]:
    counts: dict[str, int] = {"raw_merged": int(len(df))}
    hierarchy_stats: dict[str, int] = {}
    if args.enforce_hierarchy_consistency:
        before = len(df)
        df, hierarchy_stats = drop_nonmajority_hierarchy(df)
        counts["after_hierarchy_consistency"] = int(len(df))
        counts["hierarchy_rows_dropped"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset=["processid"], keep="first").copy()
    counts["after_processid_dedup"] = int(len(df))
    counts["processid_rows_dropped"] = int(before - len(df))

    seq_taxa = df.groupby("nucleotides")["species_name"].nunique(dropna=True)
    conflict_seqs = set(seq_taxa[seq_taxa > 1].index)
    counts["exact_sequence_conflict_groups_dropped"] = int(len(conflict_seqs))
    counts["exact_sequence_conflict_rows_dropped"] = int(df["nucleotides"].isin(conflict_seqs).sum())
    df = df[~df["nucleotides"].isin(conflict_seqs)].copy()

    before = len(df)
    df = df.sort_values(["source_priority", "source", "processid"])
    df = df.drop_duplicates(subset=["nucleotides"], keep="first").copy()
    counts["after_exact_sequence_dedup"] = int(len(df))
    counts["exact_sequence_duplicate_rows_dropped"] = int(before - len(df))

    genus_species_count = df.groupby("genus_name")["species_name"].nunique()
    genus_total = df.groupby("genus_name").size()
    candidates = genus_species_count[
        (genus_species_count >= args.min_genus_species)
        & (genus_species_count <= args.max_genus_species)
    ]
    top_n = genus_total.nlargest(args.top_genera_exclude).index
    candidates = candidates[~candidates.index.isin(top_n)]
    n_holdout = max(20, int(len(candidates) * args.unseen_genus_fraction)) if len(candidates) else 0
    rng_a = np.random.RandomState(SEED_UNSEEN_GENERA)
    holdout_genera = (
        sorted(rng_a.choice(candidates.index, size=min(n_holdout, len(candidates)), replace=False).tolist())
        if n_holdout else []
    )
    unseen = df[df["genus_name"].isin(holdout_genera)].copy()
    remaining = df[~df["genus_name"].isin(holdout_genera)].copy()

    eval_c_species, eval_c_meta = make_eval_c_holdout(
        remaining,
        min_species_per_genus=args.min_species_per_genus,
        min_supervised_species_count=args.min_supervised_species_count,
        eval_c_fraction=args.eval_c_fraction,
    )
    eval_c = remaining[remaining["species_name"].isin(eval_c_species)].copy()
    supervised_pool = remaining[~remaining["species_name"].isin(eval_c_species)].copy()

    sp_counts = supervised_pool["species_name"].value_counts()
    valid_species = sp_counts[sp_counts >= args.min_supervised_species_count].index
    supervised = supervised_pool[supervised_pool["species_name"].isin(valid_species)].copy()
    if supervised["species_name"].nunique() < 2:
        raise RuntimeError("Not enough supervised species after multi-source filtering.")

    n_supervised_species = supervised["species_name"].nunique()
    test_fraction = max(0.2, min(0.4, (n_supervised_species + 1) / len(supervised)))
    train_val, test = train_test_split(
        supervised,
        test_size=test_fraction,
        random_state=SEED_SUPERVISED_SPLIT,
        stratify=supervised["species_name"],
    )
    val_fraction = max(0.125, min(0.4, (n_supervised_species + 1) / len(train_val)))
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=SEED_SUPERVISED_SPLIT,
        stratify=train_val["species_name"],
    )
    pretrain = pd.concat(
        [train[["nucleotides", "processid"]], val[["nucleotides", "processid"]]],
        ignore_index=True,
    )

    splits = {
        "pre_training": pretrain,
        "supervised_train": train,
        "supervised_val": val,
        "supervised_test": test,
        "eval_c_unseen_species": eval_c,
        "unseen": unseen,
    }

    audit_splits = {
        "pre_training": pretrain.assign(
            species_name=pd.NA, genus_name=pd.NA, family_name=pd.NA, order_name=pd.NA,
        ),
        "supervised_train": train,
        "supervised_val": val,
        "supervised_test": test,
        "eval_c_unseen_species": eval_c,
        "unseen": unseen,
    }
    audit = overlap_audit(audit_splits)
    eval_c_species_set = set(eval_c["species_name"])
    seen_species_set = set(pd.concat([train, val, test])["species_name"])
    eval_c_genera_set = set(eval_c["genus_name"])
    train_genera_set = set(train["genus_name"])
    audit["eval_c_species_leak"] = int(len(eval_c_species_set & seen_species_set))
    audit["eval_c_seen_genus_check"] = {
        "query_genera": int(len(eval_c_genera_set)),
        "query_genera_in_train": int(len(eval_c_genera_set & train_genera_set)),
        "query_genera_missing_train": int(len(eval_c_genera_set - train_genera_set)),
        "missing_train_sample": sorted(eval_c_genera_set - train_genera_set)[:25],
    }
    audit["unseen_genus_leak"] = int(
        len(set(unseen["genus_name"]) & (set(train["genus_name"]) | set(val["genus_name"]) | set(test["genus_name"])))
    )

    expected_pretrain_pairs = {
        "pre_training__supervised_train",
        "pre_training__supervised_val",
    }

    def is_problem(pair: str, count: int) -> bool:
        return count != 0 and pair not in expected_pretrain_pairs

    problem = (
        any(is_problem(k, v) for k, v in audit["by_processid"].items())
        or any(is_problem(k, v) for k, v in audit["by_sequence"].items())
        or audit["eval_c_species_leak"] > 0
        or audit["eval_c_seen_genus_check"]["query_genera_missing_train"] > 0
        or audit["unseen_genus_leak"] > 0
    )
    if problem:
        raise RuntimeError(f"Multi-source 12S split audit failed: {audit}")

    meta = {
        "filter_counts": counts,
        "hierarchy_consistency": hierarchy_stats,
        "holdout_genera": holdout_genera,
        "n_holdout_genera": len(holdout_genera),
        "eval_c_species_sample": eval_c_species[:50],
        "n_eval_c_species": len(eval_c_species),
        "eval_c_meta": eval_c_meta,
        "overlap_audit": audit,
        "split_fractions": {
            "test_fraction": float(test_fraction),
            "val_fraction_of_train_val": float(val_fraction),
        },
        "totals": {
            "unique_post_dedup": int(len(df)),
            "unique_species": int(df["species_name"].nunique()),
            "unique_genera": int(df["genus_name"].nunique()),
            "supervised_species": int(supervised["species_name"].nunique()),
        },
        "source_counts_post_dedup": {
            str(k): int(v) for k, v in df["source"].value_counts().sort_index().items()
        },
    }
    return splits, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-tsv", default="data/edna/raw/mitofish.12S.Nov2025.tsv")
    parser.add_argument("--raw-zip", default="data/edna/raw/12S_efc.zip")
    parser.add_argument("--raw-mare-mage-fasta-zip", default="data/edna/raw/mare_mage/12sDBFasta.zip")
    parser.add_argument("--raw-mare-mage-taxonomy", default="data/edna/raw/mare_mage/12sDB_taxonomy.txt")
    parser.add_argument("--raw-mare-mage-reference", default="data/edna/raw/mare_mage/data_search-12S.csv")
    parser.add_argument("--out-dir", default="data/edna/processed_12s_multisource")
    parser.add_argument("--include-rcrux-blast-seeds", action="store_true")
    parser.add_argument("--class-filter", default="Actinopteri")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--max-n-frac", type=float, default=0.05)
    parser.add_argument("--enforce-hierarchy-consistency", action="store_true", default=True)
    parser.add_argument("--min-supervised-species-count", type=int, default=5)
    parser.add_argument("--min-species-per-genus", type=int, default=3)
    parser.add_argument("--eval-c-fraction", type=float, default=0.2)
    parser.add_argument("--min-genus-species", type=int, default=3)
    parser.add_argument("--max-genus-species", type=int, default=50)
    parser.add_argument("--top-genera-exclude", type=int, default=25)
    parser.add_argument("--unseen-genus-fraction", type=float, default=0.10)
    args = parser.parse_args()

    mito_df, mito_meta = load_mitohelper_tsv(Path(args.raw_tsv))
    mare_df, mare_meta = load_mare_mage(
        Path(args.raw_mare_mage_fasta_zip),
        Path(args.raw_mare_mage_taxonomy),
        Path(args.raw_mare_mage_reference),
    )
    rcrux_df, rcrux_meta = load_rcrux_zip(Path(args.raw_zip), "cleaned")

    frames = [
        prepare_source(mito_df, "mitohelper", args.class_filter, args.min_len, args.max_len, args.max_n_frac),
        prepare_source(mare_df, "mare_mage", args.class_filter, args.min_len, args.max_len, args.max_n_frac),
        prepare_source(rcrux_df, "rcrux_cleaned", args.class_filter, args.min_len, args.max_len, args.max_n_frac),
    ]
    source_meta = {
        "mitohelper": mito_meta,
        "mare_mage": mare_meta,
        "rcrux_cleaned": rcrux_meta,
    }
    if args.include_rcrux_blast_seeds:
        blast_df, blast_meta = load_rcrux_zip(Path(args.raw_zip), "blast_seeds")
        frames.append(
            prepare_source(blast_df, "rcrux_blast_seeds", args.class_filter, args.min_len, args.max_len, args.max_n_frac)
        )
        source_meta["rcrux_blast_seeds"] = blast_meta

    merged = pd.concat(frames, ignore_index=True)
    splits, meta = split_multisource(merged, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "nucleotides", "species_name", "genus_name", "family_name", "order_name",
        "processid", "taxid", "class_name", "source", "source_processid",
    ]
    file_sizes: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    for split_name, frame in splits.items():
        path = out_dir / f"{split_name}.csv"
        if split_name == "pre_training":
            frame.to_csv(path, index=False)
        else:
            frame[output_cols].to_csv(path, index=False)
        file_sizes[path.name] = int(len(frame))
        file_hashes[path.name] = sha256_of_file(path)
        print(f"wrote {path}: {len(frame):,}")

    manifest = {
        "schema_version": "12s-multisource-v1-2026-05-26",
        "sources": source_meta,
        "source_manifest_hash": sha256_text(json.dumps(source_meta, sort_keys=True, default=str)),
        "args": vars(args),
        "seeds": {
            "unseen_genera": SEED_UNSEEN_GENERA,
            "eval_c_species": SEED_EVAL_C_SPECIES,
            "supervised_split": SEED_SUPERVISED_SPLIT,
        },
        "file_sizes": file_sizes,
        "file_sha256": file_hashes,
        **meta,
    }
    with open(out_dir / "splits_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    stats = {
        "schema_version": "12s-multisource-v1-2026-05-26",
        "total_sequences": meta["totals"]["unique_post_dedup"],
        "total_species": meta["totals"]["unique_species"],
        "total_genera": meta["totals"]["unique_genera"],
        "pretrain_size": file_sizes["pre_training.csv"],
        "train_size": file_sizes["supervised_train.csv"],
        "val_size": file_sizes["supervised_val.csv"],
        "test_size": file_sizes["supervised_test.csv"],
        "eval_c_size": file_sizes["eval_c_unseen_species.csv"],
        "unseen_size": file_sizes["unseen.csv"],
        "eval_c_species_count": meta["n_eval_c_species"],
        "source_counts_post_dedup": meta["source_counts_post_dedup"],
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
