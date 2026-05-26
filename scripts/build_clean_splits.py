"""
build_clean_splits.py — frozen, reproducible split builder for MarineMamba v2.

Fixes the data-leakage and dedup issues identified in the May 2026 audit:

  1. Deduplicates by processid AND by exact nucleotide sequence before
     splitting. Previous pipeline produced 1,829 cross-split processid
     duplicates and 77,945 cross-split exact-sequence duplicates.

  2. Holds out species BEFORE pretraining/training so Eval C
     (Stalder-style unseen species, seen genera) is truly zero-shot.
     Previous pipeline did species holdout post-hoc on an already-trained
     model — that result was diagnostic only, not zero-shot.

  3. Excludes test sequences from the self-supervised pretrain set.
     Previous pipeline included all 'seen' data (incl. supervised test)
     in pretraining, so the backbone had unsupervised exposure to test
     sequences before evaluation.

  4. Writes a reproducibility manifest: SHA256 of every output split,
     RNG seeds, filter counts, overlap audit, holdout species/genera lists.

Output: data/processed_clean/

  pre_training.csv               self-sup pretrain (train + val only; excludes test, eval_c, unseen)
  supervised_train.csv           supervised training data
  supervised_val.csv             validation
  supervised_test.csv            in-distribution test (seen species)
  eval_c_unseen_species.csv      Stalder-style: unseen species, seen genera
  unseen.csv                     Eval A: unseen genera (zero-shot)
  splits_manifest.json           hashes, seeds, counts, overlap audit
  dataset_stats.json             same shape as v1 dataset_stats for compat

Dataset scope note:
  This builder keeps the original "Marine 869K" fetch scope (BOLD
  phylum/class queries). It does NOT apply WoRMS/FishBase marine
  validation. The dataset is best described as "broad animal COI with
  a marine-leaning fetch bias" — ~20%+ of sequences are in
  predominantly freshwater/terrestrial orders. See docs/DATA_QUALITY_AUDIT.md.

Usage:
    python3 scripts/build_clean_splits.py
    python3 scripts/build_clean_splits.py --min-species-per-genus 3 --eval-c-fraction 0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 660
MIN_LEN = 500
MAX_N_FRAC = 0.05
RAW_FILE = "merged_marine_barcodes.csv"

# RNG seeds. Fixed. If you change these, change the manifest version too.
SEED_UNSEEN_GENERA = 42
SEED_EVAL_C_SPECIES = 4242
SEED_SUPERVISED_SPLIT = 42


def pad_or_truncate(seq: str) -> str:
    if len(seq) >= MAX_LEN:
        return seq[:MAX_LEN]
    return "N" * (MAX_LEN - len(seq)) + seq


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def overlap_audit(splits: dict[str, pd.DataFrame]) -> dict:
    """Per-pair overlap counts on processid and exact sequence."""
    audit = {"by_processid": {}, "by_sequence": {}}
    pid_sets = {name: set(df["processid"]) for name, df in splits.items()}
    seq_sets = {name: set(df["nucleotides"]) for name, df in splits.items()}
    for a, b in combinations(splits, 2):
        audit["by_processid"][f"{a}__{b}"] = len(pid_sets[a] & pid_sets[b])
        audit["by_sequence"][f"{a}__{b}"] = len(seq_sets[a] & seq_sets[b])
    return audit


def duplicate_conflict_audit(df: pd.DataFrame, key: str) -> dict:
    """Count duplicate-key groups that disagree on taxonomy."""
    duplicate_mask = df.duplicated(subset=[key], keep=False)
    grouped = df.groupby(key, dropna=False)
    audit = {
        "duplicate_rows": int(duplicate_mask.sum()),
        "duplicate_groups": int(grouped.size().gt(1).sum()),
        "conflict_species_groups": int(grouped["species_name"].nunique(dropna=True).gt(1).sum()),
        "conflict_genus_groups": int(grouped["genus_name"].nunique(dropna=True).gt(1).sum()),
    }
    if "family_name" in df.columns:
        audit["conflict_family_groups"] = int(grouped["family_name"].nunique(dropna=True).gt(1).sum())
    if "order_name" in df.columns:
        audit["conflict_order_groups"] = int(grouped["order_name"].nunique(dropna=True).gt(1).sum())
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-file", default=RAW_FILE,
                    help="Filename under data/raw/ to load")
    ap.add_argument("--min-species-per-genus", type=int, default=3,
                    help="Min species per genus for Eval C holdout candidacy")
    ap.add_argument("--eval-c-fraction", type=float, default=0.2,
                    help="Fraction of eligible species to hold out for Eval C")
    ap.add_argument("--min-genus-species", type=int, default=3,
                    help="Min species per genus for Eval A unseen-genus candidacy")
    ap.add_argument("--max-genus-species", type=int, default=50,
                    help="Max species per genus for Eval A unseen-genus candidacy")
    ap.add_argument("--top-genera-exclude", type=int, default=50,
                    help="Exclude this many most-populous genera from Eval A pool")
    ap.add_argument("--unseen-genus-fraction", type=float, default=0.10,
                    help="Fraction of eligible genera to hold out for Eval A")
    ap.add_argument("--min-supervised-species-count", type=int, default=5,
                    help="Min sequence count per species to include in supervised splits")
    args = ap.parse_args()

    print("=" * 68)
    print("MARINEMAMBA v2 — CLEAN SPLIT BUILDER")
    print("=" * 68)

    counts: dict[str, int] = {}

    # ── Load ────────────────────────────────────────────────────────────
    raw_path = RAW_DIR / args.raw_file
    print(f"\nLoading {raw_path} ...")
    df = pd.read_csv(raw_path)
    counts["raw_loaded"] = len(df)
    print(f"  rows: {len(df):,}")

    # ── Clean sequences ─────────────────────────────────────────────────
    df["nucleotides"] = df["nucleotides"].astype(str).str.upper().str.strip()
    df["nucleotides"] = df["nucleotides"].str.replace(r"[^ACGTN]", "", regex=True)
    df["seq_len"] = df["nucleotides"].str.len()

    df = df[df["seq_len"] >= MIN_LEN].copy()
    counts["after_min_len"] = len(df)
    print(f"  after min_len {MIN_LEN}bp: {len(df):,}")

    df = df[df["seq_len"] <= 700].copy()
    counts["after_max_len_700"] = len(df)
    print(f"  after max_len 700bp: {len(df):,}")

    df["n_frac"] = df["nucleotides"].str.count("N") / df["seq_len"]
    df = df[df["n_frac"] <= MAX_N_FRAC].copy()
    counts["after_n_filter"] = len(df)
    print(f"  after N filter (<={MAX_N_FRAC:.0%}): {len(df):,}")

    df = df.dropna(subset=["species_name"])
    df = df[df["species_name"].astype(str).str.contains(" ", na=False)].copy()
    counts["after_binomial"] = len(df)
    print(f"  after requiring binomial species: {len(df):,}")

    if "genus_name" not in df.columns:
        df["genus_name"] = df["species_name"].str.split().str[0]
    else:
        df["genus_name"] = df["genus_name"].fillna(df["species_name"].str.split().str[0])

    # Pad/truncate to fixed length
    df["nucleotides"] = df["nucleotides"].apply(pad_or_truncate)

    dedup_audit: dict[str, dict] = {}

    # ── Dedup ───────────────────────────────────────────────────────────
    before = len(df)
    dedup_audit["processid_before_dedup"] = duplicate_conflict_audit(df, "processid")
    df = df.drop_duplicates(subset=["processid"], keep="first").copy()
    counts["after_processid_dedup"] = len(df)
    print(f"  after processid dedup: {len(df):,} (-{before - len(df):,})")

    before = len(df)
    dedup_audit["sequence_before_dedup_after_processid_dedup"] = duplicate_conflict_audit(df, "nucleotides")
    df = df.drop_duplicates(subset=["nucleotides"], keep="first").copy()
    counts["after_sequence_dedup"] = len(df)
    print(f"  after exact-sequence dedup: {len(df):,} (-{before - len(df):,})")

    df = df.reset_index(drop=True)
    print(f"\n  unique post-dedup rows: {len(df):,}")
    print(f"  unique species: {df['species_name'].nunique():,}")
    print(f"  unique genera:  {df['genus_name'].nunique():,}")

    # ── Holdout 1: unseen genera (Eval A) ───────────────────────────────
    print("\n" + "─" * 68)
    print("EVAL A holdout: unseen genera")
    print("─" * 68)
    genus_species_count = df.groupby("genus_name")["species_name"].nunique()
    genus_total = df.groupby("genus_name").size()
    candidates = genus_species_count[
        (genus_species_count >= args.min_genus_species)
        & (genus_species_count <= args.max_genus_species)
    ]
    top_n = genus_total.nlargest(args.top_genera_exclude).index
    candidates = candidates[~candidates.index.isin(top_n)]
    n_holdout = max(20, int(len(candidates) * args.unseen_genus_fraction))
    rng_a = np.random.RandomState(SEED_UNSEEN_GENERA)
    holdout_genera = rng_a.choice(candidates.index, size=min(n_holdout, len(candidates)), replace=False)
    holdout_genera = sorted(holdout_genera.tolist())
    print(f"  candidates: {len(candidates):,} genera")
    print(f"  holdout:    {len(holdout_genera):,} genera")

    unseen_mask = df["genus_name"].isin(holdout_genera)
    unseen = df[unseen_mask].copy()
    remaining = df[~unseen_mask].copy()
    print(f"  unseen rows: {len(unseen):,}")
    print(f"  remaining:   {len(remaining):,}")

    # ── Holdout 2: unseen species within seen genera (Eval C) ───────────
    print("\n" + "─" * 68)
    print("EVAL C holdout: unseen species within seen genera (true zero-shot)")
    print("─" * 68)
    species_counts_remaining = remaining["species_name"].value_counts()
    g_species = remaining.groupby("genus_name")["species_name"].unique()
    eligible_genera = g_species[g_species.map(len) >= args.min_species_per_genus]
    print(f"  eligible genera (>= {args.min_species_per_genus} species): {len(eligible_genera):,}")

    rng_c = np.random.RandomState(SEED_EVAL_C_SPECIES)
    eval_c_species: list[str] = []
    eval_c_reserved_reference_species: dict[str, str] = {}
    skipped_no_reference_survivor = 0
    skipped_no_holdout_candidate = 0
    for genus, species_list in eligible_genera.items():
        species_list = sorted(species_list.tolist())
        reference_survivors = [
            sp for sp in species_list
            if species_counts_remaining.get(sp, 0) >= args.min_supervised_species_count
        ]
        if not reference_survivors:
            skipped_no_reference_survivor += 1
            continue

        # Reserve one species that is guaranteed to survive the supervised
        # min-count filter, so every Eval C query genus is represented in train.
        reserved = str(rng_c.choice(reference_survivors))
        holdout_candidates = [sp for sp in species_list if sp != reserved]
        if not holdout_candidates:
            skipped_no_holdout_candidate += 1
            continue

        n_hold = max(1, int(round(len(species_list) * args.eval_c_fraction)))
        n_hold = min(n_hold, len(holdout_candidates))
        chosen = rng_c.choice(holdout_candidates, size=n_hold, replace=False)
        eval_c_species.extend(chosen.tolist())
        eval_c_reserved_reference_species[genus] = reserved
    eval_c_species = sorted(set(eval_c_species))
    print(f"  eval_c held-out species: {len(eval_c_species):,}")
    print(f"  skipped genera without supervised reference survivor: {skipped_no_reference_survivor:,}")
    print(f"  skipped genera without holdout candidate: {skipped_no_holdout_candidate:,}")

    eval_c_mask = remaining["species_name"].isin(eval_c_species)
    eval_c = remaining[eval_c_mask].copy()
    supervised_pool = remaining[~eval_c_mask].copy()
    print(f"  eval_c rows:            {len(eval_c):,}")
    print(f"  supervised_pool rows:   {len(supervised_pool):,}")

    # ── Supervised splits ───────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("SUPERVISED split (stratified by species, seen-species only)")
    print("─" * 68)
    sp_counts = supervised_pool["species_name"].value_counts()
    valid_species = sp_counts[sp_counts >= args.min_supervised_species_count].index
    supervised = supervised_pool[supervised_pool["species_name"].isin(valid_species)].copy()
    dropped_small = len(supervised_pool) - len(supervised)
    print(f"  species with >= {args.min_supervised_species_count} seqs kept: {len(valid_species):,}")
    print(f"  dropped (small species): {dropped_small:,} rows")

    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        supervised, test_size=0.2, random_state=SEED_SUPERVISED_SPLIT,
        stratify=supervised["species_name"],
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=SEED_SUPERVISED_SPLIT,
        stratify=train_val["species_name"],
    )
    print(f"  train: {len(train):,}   val: {len(val):,}   test: {len(test):,}")

    # Pretraining set: train + val ONLY (no test, no eval_c, no unseen).
    # Self-supervised pretraining should not see evaluation sequences.
    pretrain = pd.concat([train[["nucleotides", "processid"]],
                          val[["nucleotides", "processid"]]], ignore_index=True)
    print(f"  pretrain (train + val): {len(pretrain):,}")

    # ── Write outputs ───────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("WRITING SPLITS")
    print("─" * 68)
    output_cols = [c for c in
                   ["nucleotides", "species_name", "genus_name", "family_name", "order_name", "processid"]
                   if c in df.columns]

    splits_to_write = {
        "pre_training.csv": pretrain,
        "supervised_train.csv": train[output_cols],
        "supervised_val.csv": val[output_cols],
        "supervised_test.csv": test[output_cols],
        "eval_c_unseen_species.csv": eval_c[output_cols],
        "unseen.csv": unseen[output_cols],
    }

    file_hashes: dict[str, str] = {}
    file_sizes: dict[str, int] = {}
    for fname, frame in splits_to_write.items():
        path = OUT_DIR / fname
        frame.to_csv(path, index=False)
        file_hashes[fname] = sha256_of_file(path)
        file_sizes[fname] = len(frame)
        print(f"  {fname}: {len(frame):>10,} rows   sha256={file_hashes[fname][:12]}…")

    # ── Overlap audit ───────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("OVERLAP AUDIT (should all be zero)")
    print("─" * 68)
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

    # Eval C species disjoint from supervised species check (strict zero-shot)
    eval_c_species_set = set(eval_c["species_name"])
    train_species_set = set(train["species_name"])
    val_species_set = set(val["species_name"])
    test_species_set = set(test["species_name"])
    eval_c_in_train = eval_c_species_set & train_species_set
    eval_c_in_val = eval_c_species_set & val_species_set
    eval_c_in_test = eval_c_species_set & test_species_set
    audit["eval_c_species_leak"] = {
        "into_train": len(eval_c_in_train),
        "into_val": len(eval_c_in_val),
        "into_test": len(eval_c_in_test),
    }

    eval_c_genera_set = set(eval_c["genus_name"])
    train_genera_set = set(train["genus_name"])
    eval_c_genera_missing_train = sorted(eval_c_genera_set - train_genera_set)
    audit["eval_c_seen_genus_check"] = {
        "query_genera": len(eval_c_genera_set),
        "query_genera_in_train": len(eval_c_genera_set & train_genera_set),
        "query_genera_missing_train": len(eval_c_genera_missing_train),
        "missing_train_sample": eval_c_genera_missing_train[:25],
    }

    # Eval A species disjoint from training species check
    unseen_species_set = set(unseen["species_name"])
    unseen_in_train = unseen_species_set & train_species_set
    audit["unseen_species_in_train"] = len(unseen_in_train)

    # pretrain = train + val by construction. Skip those pairs in the leak check.
    expected_pretrain_pairs = {
        "pre_training__supervised_train",
        "pre_training__supervised_val",
    }

    def is_problem(pair_key: str, count: int) -> bool:
        if count == 0:
            return False
        if pair_key in expected_pretrain_pairs:
            return False
        return True

    problem_overlaps = (
        any(is_problem(k, v) for k, v in audit["by_processid"].items())
        or any(is_problem(k, v) for k, v in audit["by_sequence"].items())
        or sum(audit["eval_c_species_leak"].values()) > 0
        or audit["eval_c_seen_genus_check"]["query_genera_missing_train"] > 0
        or audit["unseen_species_in_train"] > 0
    )

    print("processid overlaps:")
    for k, v in audit["by_processid"].items():
        if k in expected_pretrain_pairs:
            marker = "i "  # expected (pretrain = train + val)
        else:
            marker = "  " if v == 0 else "!!"
        print(f"  {marker} {k}: {v}")
    print("exact-sequence overlaps:")
    for k, v in audit["by_sequence"].items():
        if k in expected_pretrain_pairs:
            marker = "i "  # expected (pretrain = train + val)
        else:
            marker = "  " if v == 0 else "!!"
        print(f"  {marker} {k}: {v}")
    print(f"eval_c species leak into supervised: {audit['eval_c_species_leak']}")
    print(f"eval_c seen-genus check: {audit['eval_c_seen_genus_check']}")
    print(f"unseen species in train (genus-leak check): {audit['unseen_species_in_train']}")
    print("(i marks expected overlaps: pretrain contains train + val by design.)")

    if problem_overlaps:
        raise RuntimeError("Split audit failed. Investigate before training.")
    else:
        print("\nOK: splits are clean. All evaluation sets are disjoint from training.")

    # ── Manifest ────────────────────────────────────────────────────────
    manifest = {
        "schema_version": "v3-2026-05-25",
        "raw_file": args.raw_file,
        "raw_file_sha256": sha256_of_file(RAW_DIR / args.raw_file),
        "seeds": {
            "unseen_genera": SEED_UNSEEN_GENERA,
            "eval_c_species": SEED_EVAL_C_SPECIES,
            "supervised_split": SEED_SUPERVISED_SPLIT,
        },
        "args": vars(args),
        "filter_counts": counts,
        "dedup_audit": dedup_audit,
        "file_sizes": file_sizes,
        "file_sha256": file_hashes,
        "holdout_genera": holdout_genera,
        "n_holdout_genera": len(holdout_genera),
        "eval_c_species_sample": eval_c_species[:50],
        "n_eval_c_species": len(eval_c_species),
        "eval_c_reserved_reference_species_sample": dict(
            list(eval_c_reserved_reference_species.items())[:50]
        ),
        "n_eval_c_reference_genera": len(eval_c_reserved_reference_species),
        "n_eval_c_skipped_no_reference_survivor": skipped_no_reference_survivor,
        "n_eval_c_skipped_no_holdout_candidate": skipped_no_holdout_candidate,
        "overlap_audit": audit,
        "totals": {
            "unique_post_dedup": int(len(df)),
            "unique_species": int(df["species_name"].nunique()),
            "unique_genera": int(df["genus_name"].nunique()),
            "supervised_species": int(supervised["species_name"].nunique()),
        },
    }
    with open(OUT_DIR / "splits_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Compat dataset_stats.json for downstream scripts
    stats = {
        "schema_version": "v3-2026-05-25",
        "total_sequences": int(len(df)),
        "total_species": int(df["species_name"].nunique()),
        "total_genera": int(df["genus_name"].nunique()),
        "pretrain_size": len(pretrain),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "eval_c_size": len(eval_c),
        "unseen_size": len(unseen),
        "unseen_genera": int(unseen["genus_name"].nunique()),
        "eval_c_species_count": len(eval_c_species),
        "n_classes": int(supervised["species_name"].nunique()),
        "holdout_genera": holdout_genera,
    }
    with open(OUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 68)
    print(f"DONE → {OUT_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
