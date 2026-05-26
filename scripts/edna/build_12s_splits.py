"""
Build leakage-audited 12S fish reference splits for the eDNA/Stalder track.

Default input is the Mitohelper Nov2025 MitoFish 12S TSV:
  data/edna/raw/mitofish.12S.Nov2025.tsv

The output schema mirrors the COI clean split:
  pre_training.csv
  supervised_train.csv
  supervised_val.csv
  supervised_test.csv
  eval_c_unseen_species.csv
  unseen.csv
  splits_manifest.json
  dataset_stats.json

For the closest Stalder-style setup, run with:
  python3 scripts/edna/build_12s_splits.py --region full_12s --class-filter Actinopteri

For the Mare-MAGE fish 12S reference database, run with:
  python3 scripts/edna/build_12s_splits.py --source mare_mage --enforce-hierarchy-consistency \
    --raw-mare-mage-fasta-zip data/edna/raw/mare_mage/12sDBFasta.zip \
    --raw-mare-mage-taxonomy data/edna/raw/mare_mage/12sDB_taxonomy.txt \
    --raw-mare-mage-reference data/edna/raw/mare_mage/data_search-12S.csv \
    --out-dir data/edna/processed_12s_mare_mage

The split builder enforces:
  - no accession or exact-sequence overlap across evaluation sets
  - Eval C species absent from train/val/test/pretrain
  - every Eval C query genus represented in supervised_train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


SEED_UNSEEN_GENERA = 1205
SEED_EVAL_C_SPECIES = 1206
SEED_SUPERVISED_SPLIT = 1207

MIFISH_U_F = "GTCGGTAAAACTCGTGCCAGC"
MIFISH_U_R = "CATAGTGGGGTATCTAATCCCAGTTTG"
MIFISH_U_R_RC = "CAAACTGGGATTAGATACCCCACTATG"


def clean_seq(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper().strip() if ch in "ACGTN")


def extract_mifish_amplicon(seq: str) -> str | None:
    """Exact primer extraction for MiFish-U oriented 12S sequences."""
    seq = clean_seq(seq)
    start = seq.find(MIFISH_U_F)
    if start < 0:
        return None
    search_start = start + len(MIFISH_U_F)
    end = seq.find(MIFISH_U_R_RC, search_start)
    if end < 0:
        end = seq.find(MIFISH_U_R, search_start)
    if end < 0:
        return None
    return seq[start:end + len(MIFISH_U_R_RC)]


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fasta(text: str) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current_id: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current_id = line[1:].split()[0]
            records[current_id] = []
        elif current_id is not None:
            records[current_id].append(line)
    return {key: clean_seq("".join(parts)) for key, parts in records.items()}


def normalise_taxon(value: object) -> object:
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE", "UNCLASSIFIED"}:
        return pd.NA
    return text


def load_mitohelper_tsv(raw_tsv: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    df = pd.read_csv(raw_tsv, sep="\t")
    rename = {
        "Accession": "processid",
        "Sequence": "nucleotides",
        "Species": "species_name",
        "Genus": "genus_name",
        "Family": "family_name",
        "Order": "order_name",
        "Class": "class_name",
        "Phylum": "phylum_name",
        "taxid": "taxid",
    }
    missing = [col for col in rename if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in Mitohelper TSV: {missing}")
    df = df.rename(columns=rename)
    df = df[list(rename.values())].copy()
    return df, {
        "source": "Mitohelper Nov2025 MitoFish 12S",
        "source_url": "https://zenodo.org/records/17602902",
        "raw_path_key": "raw_tsv",
        "raw_path": str(raw_tsv),
        "raw_sha256_key": "raw_tsv_sha256",
        "raw_sha256": sha256_of_file(raw_tsv),
    }


def load_rcrux_zip(raw_zip: Path, table: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if table == "cleaned":
        fasta_name = "12S_efc/derep_and_clean_db 2/12S_efc_derep_and_clean.fasta"
        taxonomy_name = "12S_efc/derep_and_clean_db 2/12S_efc_derep_and_clean_taxonomy.txt"
    elif table == "blast_seeds":
        fasta_name = "12S_efc/blast_seeds_output/12S_efc.fasta"
        taxonomy_name = "12S_efc/blast_seeds_output/12S_efc_taxonomy.txt"
    else:
        raise RuntimeError(f"Unsupported rCRUX table: {table}")

    with zipfile.ZipFile(raw_zip) as zf:
        fasta = parse_fasta(zf.read(fasta_name).decode("utf-8"))
        taxonomy_text = zf.read(taxonomy_name).decode("utf-8")

    rows = []
    for line in taxonomy_text.splitlines():
        if not line.strip():
            continue
        try:
            processid, taxonomy = line.split("\t", 1)
        except ValueError:
            continue
        levels = taxonomy.split(";")
        if len(levels) != 7:
            continue
        domain, phylum, class_name, order_name, family_name, genus_name, species_name = levels
        sequence = fasta.get(processid)
        if not sequence:
            continue
        rows.append({
            "processid": processid,
            "nucleotides": sequence,
            "species_name": species_name,
            "genus_name": genus_name,
            "family_name": family_name,
            "order_name": order_name,
            "class_name": class_name,
            "phylum_name": phylum,
            "taxid": pd.NA,
            "domain_name": domain,
        })

    return pd.DataFrame(rows), {
        "source": f"rCRUX MiFish Universal 12S + FishCARD ({table})",
        "source_url": "https://zenodo.org/records/8409239",
        "raw_path_key": "raw_zip",
        "raw_path": str(raw_zip),
        "raw_sha256_key": "raw_zip_sha256",
        "raw_sha256": sha256_of_file(raw_zip),
        "rcrux_table": table,
    }


def _strip_taxon_prefix(value: str) -> str:
    value = value.strip()
    if "_" in value:
        return value.split("_", 1)[1].strip()
    return value


def load_mare_mage(
    raw_fasta_zip: Path,
    raw_taxonomy: Path,
    raw_reference: Path | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    with zipfile.ZipFile(raw_fasta_zip) as zf:
        fasta = parse_fasta(zf.read("12sDB.fasta").decode("utf-8"))

    rows = []
    for line in raw_taxonomy.read_text().splitlines():
        if not line.strip():
            continue
        try:
            processid, taxonomy = line.split("\t", 1)
        except ValueError:
            continue
        levels = [_strip_taxon_prefix(part) for part in taxonomy.split(";")]
        if len(levels) < 9:
            continue
        sequence = fasta.get(processid)
        if not sequence:
            continue
        class_name = levels[4]
        if class_name == "Actinopterygii":
            class_name = "Actinopteri"
        rows.append({
            "processid": processid,
            "nucleotides": sequence,
            "species_name": levels[8],
            "genus_name": levels[7],
            "family_name": levels[6],
            "order_name": levels[5],
            "class_name": class_name,
            "phylum_name": levels[2],
            "taxid": pd.NA,
        })

    meta = {
        "source": "Mare-MAGE 12sDB all fish sequences",
        "source_url": "https://mare-mage.weebly.com/12sdb_all-sequences.html",
        "figshare_collection": "https://doi.org/10.6084/m9.figshare.c.5410161.v1",
        "raw_path_key": "raw_mare_mage_fasta_zip",
        "raw_path": str(raw_fasta_zip),
        "raw_sha256_key": "raw_mare_mage_fasta_zip_sha256",
        "raw_sha256": sha256_of_file(raw_fasta_zip),
        "raw_mare_mage_taxonomy": str(raw_taxonomy),
        "raw_mare_mage_taxonomy_sha256": sha256_of_file(raw_taxonomy),
    }
    if raw_reference and raw_reference.exists():
        meta["raw_mare_mage_reference"] = str(raw_reference)
        meta["raw_mare_mage_reference_sha256"] = sha256_of_file(raw_reference)
    return pd.DataFrame(rows), meta


def duplicate_conflict_audit(df: pd.DataFrame, key: str) -> dict:
    duplicate_mask = df.duplicated(subset=[key], keep=False)
    grouped = df.groupby(key, dropna=False)
    return {
        "duplicate_rows": int(duplicate_mask.sum()),
        "duplicate_groups": int(grouped.size().gt(1).sum()),
        "conflict_species_groups": int(grouped["species_name"].nunique(dropna=True).gt(1).sum()),
        "conflict_genus_groups": int(grouped["genus_name"].nunique(dropna=True).gt(1).sum()),
        "conflict_family_groups": int(grouped["family_name"].nunique(dropna=True).gt(1).sum()),
        "conflict_order_groups": int(grouped["order_name"].nunique(dropna=True).gt(1).sum()),
    }


def drop_nonmajority_hierarchy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {}
    cleaned = df.copy()
    for child, parent in [
        ("species_name", "genus_name"),
        ("genus_name", "family_name"),
        ("family_name", "order_name"),
    ]:
        before = len(cleaned)
        counts = cleaned.groupby([child, parent], dropna=False).size().reset_index(name="n")
        counts = counts.sort_values([child, "n", parent], ascending=[True, False, True])
        majority = counts.drop_duplicates(subset=[child], keep="first")[[child, parent]]
        majority = majority.rename(columns={parent: f"majority_{parent}"})
        cleaned = cleaned.merge(majority, on=child, how="left")
        keep = cleaned[parent].astype(str) == cleaned[f"majority_{parent}"].astype(str)
        dropped = int((~keep).sum())
        conflict_keys = int(
            cleaned.groupby(child)[parent].nunique(dropna=True).gt(1).sum()
        )
        cleaned = cleaned[keep].drop(columns=[f"majority_{parent}"]).copy()
        stats[f"{child}_to_{parent}_conflict_keys_before"] = conflict_keys
        stats[f"{child}_to_{parent}_rows_dropped"] = dropped
        stats[f"{child}_to_{parent}_rows_before"] = int(before)
        stats[f"{child}_to_{parent}_rows_after"] = int(len(cleaned))
    return cleaned, stats


def overlap_audit(splits: dict[str, pd.DataFrame]) -> dict:
    audit = {"by_processid": {}, "by_sequence": {}}
    pid_sets = {name: set(df["processid"]) for name, df in splits.items()}
    seq_sets = {name: set(df["nucleotides"]) for name, df in splits.items()}
    for a, b in combinations(splits, 2):
        audit["by_processid"][f"{a}__{b}"] = len(pid_sets[a] & pid_sets[b])
        audit["by_sequence"][f"{a}__{b}"] = len(seq_sets[a] & seq_sets[b])
    return audit


def make_eval_c_holdout(
    remaining: pd.DataFrame,
    min_species_per_genus: int,
    min_supervised_species_count: int,
    eval_c_fraction: float,
) -> tuple[list[str], dict]:
    rng = np.random.RandomState(SEED_EVAL_C_SPECIES)
    species_counts = remaining["species_name"].value_counts()
    genus_species = remaining.groupby("genus_name")["species_name"].unique()
    eligible = genus_species[genus_species.map(len) >= min_species_per_genus]

    holdout_species: list[str] = []
    reserved_reference_species: dict[str, str] = {}
    skipped_no_reference_survivor = 0
    skipped_no_holdout_candidate = 0

    for genus, species_array in eligible.items():
        species_list = sorted(species_array.tolist())
        reference_survivors = [
            sp for sp in species_list
            if species_counts.get(sp, 0) >= min_supervised_species_count
        ]
        if not reference_survivors:
            skipped_no_reference_survivor += 1
            continue
        reserved = str(rng.choice(reference_survivors))
        candidates = [sp for sp in species_list if sp != reserved]
        if not candidates:
            skipped_no_holdout_candidate += 1
            continue
        n_hold = max(1, int(round(len(species_list) * eval_c_fraction)))
        n_hold = min(n_hold, len(candidates))
        holdout_species.extend(rng.choice(candidates, size=n_hold, replace=False).tolist())
        reserved_reference_species[genus] = reserved

    meta = {
        "eligible_genera": int(len(eligible)),
        "reserved_reference_species_sample": dict(list(reserved_reference_species.items())[:50]),
        "n_reference_genera": int(len(reserved_reference_species)),
        "skipped_no_reference_survivor": int(skipped_no_reference_survivor),
        "skipped_no_holdout_candidate": int(skipped_no_holdout_candidate),
    }
    return sorted(set(holdout_species)), meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["mitohelper", "rcrux", "mare_mage"], default="mitohelper")
    parser.add_argument("--raw-tsv", default="data/edna/raw/mitofish.12S.Nov2025.tsv")
    parser.add_argument("--raw-zip", default="data/edna/raw/12S_efc.zip")
    parser.add_argument("--raw-mare-mage-fasta-zip", default="data/edna/raw/mare_mage/12sDBFasta.zip")
    parser.add_argument("--raw-mare-mage-taxonomy", default="data/edna/raw/mare_mage/12sDB_taxonomy.txt")
    parser.add_argument("--raw-mare-mage-reference", default="data/edna/raw/mare_mage/data_search-12S.csv")
    parser.add_argument("--rcrux-table", choices=["cleaned", "blast_seeds"], default="cleaned")
    parser.add_argument("--out-dir", default="data/edna/processed_12s")
    parser.add_argument("--region", choices=["full_12s", "mifish"], default="full_12s")
    parser.add_argument("--class-filter", default="Actinopteri",
                        help="Use 'all' to disable. Actinopteri is closest to Stalder-style ray-finned fish.")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=2000)
    parser.add_argument("--max-n-frac", type=float, default=0.05)
    parser.add_argument("--enforce-hierarchy-consistency", action="store_true",
                        help="Drop non-majority child->parent taxonomy mappings before holdout splits.")
    parser.add_argument("--min-supervised-species-count", type=int, default=5)
    parser.add_argument("--min-species-per-genus", type=int, default=3)
    parser.add_argument("--eval-c-fraction", type=float, default=0.2)
    parser.add_argument("--min-genus-species", type=int, default=3)
    parser.add_argument("--max-genus-species", type=int, default=50)
    parser.add_argument("--top-genera-exclude", type=int, default=25)
    parser.add_argument("--unseen-genus-fraction", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    print("=" * 70)
    print("12S eDNA CLEAN SPLIT BUILDER")
    print("=" * 70)
    if args.source == "mitohelper":
        raw_tsv = Path(args.raw_tsv)
        print(f"loading Mitohelper TSV: {raw_tsv}")
        df, source_meta = load_mitohelper_tsv(raw_tsv)
    elif args.source == "rcrux":
        raw_zip = Path(args.raw_zip)
        print(f"loading rCRUX ZIP: {raw_zip} [{args.rcrux_table}]")
        df, source_meta = load_rcrux_zip(raw_zip, args.rcrux_table)
    else:
        raw_fasta_zip = Path(args.raw_mare_mage_fasta_zip)
        raw_taxonomy = Path(args.raw_mare_mage_taxonomy)
        raw_reference = Path(args.raw_mare_mage_reference)
        print(f"loading Mare-MAGE FASTA ZIP: {raw_fasta_zip}")
        df, source_meta = load_mare_mage(raw_fasta_zip, raw_taxonomy, raw_reference)
    counts["raw_loaded"] = int(len(df))
    print(f"  raw rows: {len(df):,}")

    for col in ["species_name", "genus_name", "family_name", "order_name", "class_name", "phylum_name"]:
        df[col] = df[col].apply(normalise_taxon)
    df = df[[
        "processid", "nucleotides", "species_name", "genus_name",
        "family_name", "order_name", "class_name", "phylum_name", "taxid",
    ]].copy()

    if args.class_filter.lower() != "all":
        df = df[df["class_name"].astype(str) == args.class_filter].copy()
    counts["after_class_filter"] = int(len(df))
    print(f"  after class filter ({args.class_filter}): {len(df):,}")

    df = df.dropna(subset=["processid", "nucleotides", "species_name", "genus_name"]).copy()
    df["species_name"] = df["species_name"].astype(str).str.strip()
    df = df[df["species_name"].str.contains(" ", na=False)].copy()
    counts["after_binomial"] = int(len(df))

    if args.region == "mifish" and args.source == "mitohelper":
        df["nucleotides"] = df["nucleotides"].apply(extract_mifish_amplicon)
        df = df.dropna(subset=["nucleotides"]).copy()
    else:
        df["nucleotides"] = df["nucleotides"].apply(clean_seq)
    counts[f"after_region_{args.region}"] = int(len(df))
    print(f"  after region {args.region}: {len(df):,}")

    df["seq_len"] = df["nucleotides"].str.len()
    df = df[(df["seq_len"] >= args.min_len) & (df["seq_len"] <= args.max_len)].copy()
    counts["after_length_filter"] = int(len(df))
    print(f"  after length filter: {len(df):,}")

    df["n_frac"] = df["nucleotides"].str.count("N") / df["seq_len"]
    df = df[df["n_frac"] <= args.max_n_frac].copy()
    counts["after_n_filter"] = int(len(df))
    print(f"  after N filter: {len(df):,}")

    dedup_audit: dict[str, dict] = {}
    dedup_audit["processid_before_dedup"] = duplicate_conflict_audit(df, "processid")
    before = len(df)
    df = df.drop_duplicates(subset=["processid"], keep="first").copy()
    counts["after_processid_dedup"] = int(len(df))
    print(f"  after accession dedup: {len(df):,} (-{before - len(df):,})")

    dedup_audit["sequence_before_dedup_after_processid_dedup"] = duplicate_conflict_audit(df, "nucleotides")
    before = len(df)
    df = df.drop_duplicates(subset=["nucleotides"], keep="first").copy()
    counts["after_sequence_dedup"] = int(len(df))
    print(f"  after exact-sequence dedup: {len(df):,} (-{before - len(df):,})")

    hierarchy_stats: dict[str, int] = {}
    if args.enforce_hierarchy_consistency:
        before = len(df)
        df, hierarchy_stats = drop_nonmajority_hierarchy(df)
        counts["after_hierarchy_consistency"] = int(len(df))
        print(f"  after hierarchy consistency filter: {len(df):,} (-{before - len(df):,})")

    if len(df) == 0:
        raise RuntimeError("No records left after filtering.")

    print(f"  unique species: {df['species_name'].nunique():,}")
    print(f"  unique genera:  {df['genus_name'].nunique():,}")

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
    print(f"  Eval A holdout genera: {len(holdout_genera):,}; rows: {len(unseen):,}")

    eval_c_species, eval_c_meta = make_eval_c_holdout(
        remaining,
        min_species_per_genus=args.min_species_per_genus,
        min_supervised_species_count=args.min_supervised_species_count,
        eval_c_fraction=args.eval_c_fraction,
    )
    eval_c = remaining[remaining["species_name"].isin(eval_c_species)].copy()
    supervised_pool = remaining[~remaining["species_name"].isin(eval_c_species)].copy()
    print(f"  Eval C species: {len(eval_c_species):,}; rows: {len(eval_c):,}")

    sp_counts = supervised_pool["species_name"].value_counts()
    valid_species = sp_counts[sp_counts >= args.min_supervised_species_count].index
    supervised = supervised_pool[supervised_pool["species_name"].isin(valid_species)].copy()
    print(f"  supervised species: {len(valid_species):,}; rows: {len(supervised):,}")
    if supervised["species_name"].nunique() < 2:
        raise RuntimeError("Not enough supervised species after filtering.")

    from sklearn.model_selection import train_test_split

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
    print(f"  train/val/test/pretrain: {len(train):,}/{len(val):,}/{len(test):,}/{len(pretrain):,}")

    output_cols = [
        "nucleotides", "species_name", "genus_name", "family_name",
        "order_name", "processid", "taxid", "class_name",
    ]
    splits_to_write = {
        "pre_training.csv": pretrain,
        "supervised_train.csv": train[output_cols],
        "supervised_val.csv": val[output_cols],
        "supervised_test.csv": test[output_cols],
        "eval_c_unseen_species.csv": eval_c[output_cols],
        "unseen.csv": unseen[output_cols],
    }

    file_sizes: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    for fname, frame in splits_to_write.items():
        path = out_dir / fname
        frame.to_csv(path, index=False)
        file_sizes[fname] = int(len(frame))
        file_hashes[fname] = sha256_of_file(path)
        print(f"  wrote {fname}: {len(frame):,}")

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

    expected_pretrain_pairs = {
        "pre_training__supervised_train",
        "pre_training__supervised_val",
    }

    def is_problem(pair: str, count: int) -> bool:
        return count != 0 and pair not in expected_pretrain_pairs

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

    problem = (
        any(is_problem(k, v) for k, v in audit["by_processid"].items())
        or any(is_problem(k, v) for k, v in audit["by_sequence"].items())
        or audit["eval_c_species_leak"] > 0
        or audit["eval_c_seen_genus_check"]["query_genera_missing_train"] > 0
        or audit["unseen_genus_leak"] > 0
    )
    if problem:
        raise RuntimeError(f"12S split audit failed: {audit}")

    manifest = {
        "schema_version": "12s-v1-2026-05-25",
        **source_meta,
        "args": vars(args),
        "split_fractions": {
            "test_fraction": float(test_fraction),
            "val_fraction_of_train_val": float(val_fraction),
        },
        "seeds": {
            "unseen_genera": SEED_UNSEEN_GENERA,
            "eval_c_species": SEED_EVAL_C_SPECIES,
            "supervised_split": SEED_SUPERVISED_SPLIT,
        },
        "filter_counts": counts,
        "dedup_audit": dedup_audit,
        "hierarchy_consistency": hierarchy_stats,
        "file_sizes": file_sizes,
        "file_sha256": file_hashes,
        "holdout_genera": holdout_genera,
        "n_holdout_genera": len(holdout_genera),
        "eval_c_species_sample": eval_c_species[:50],
        "n_eval_c_species": len(eval_c_species),
        "eval_c_meta": eval_c_meta,
        "overlap_audit": audit,
        "totals": {
            "unique_post_dedup": int(len(df)),
            "unique_species": int(df["species_name"].nunique()),
            "unique_genera": int(df["genus_name"].nunique()),
            "supervised_species": int(supervised["species_name"].nunique()),
        },
    }
    with open(out_dir / "splits_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    stats = {
        "schema_version": "12s-v1-2026-05-25",
        "total_sequences": int(len(df)),
        "total_species": int(df["species_name"].nunique()),
        "total_genera": int(df["genus_name"].nunique()),
        "pretrain_size": int(len(pretrain)),
        "train_size": int(len(train)),
        "val_size": int(len(val)),
        "test_size": int(len(test)),
        "eval_c_size": int(len(eval_c)),
        "unseen_size": int(len(unseen)),
        "unseen_genera": int(unseen["genus_name"].nunique()),
        "eval_c_species_count": int(len(eval_c_species)),
        "n_classes": int(supervised["species_name"].nunique()),
        "holdout_genera": holdout_genera,
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("OK: 12S splits are audit-clean.")
    print(f"done: {out_dir}")


if __name__ == "__main__":
    main()
