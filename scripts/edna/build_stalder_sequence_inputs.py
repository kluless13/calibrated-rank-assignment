#!/usr/bin/env python3
"""Build Stalder/TAXDNA-style sequence inputs from MarineMamba 12S splits."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import dendropy
import pandas as pd


SPLIT_FILES = {
    "pre_training": "pre_training.csv",
    "supervised_train": "supervised_train.csv",
    "supervised_val": "supervised_val.csv",
    "supervised_test": "supervised_test.csv",
    "eval_c_unseen_species": "eval_c_unseen_species.csv",
    "unseen": "unseen.csv",
}

TELEO_F = "ACACCGCCCGTCACTCT"
TELEO_R = "CTTCCGGTACACTTACCATG"
TELEO_R_RC = "CATGGTAAGTGTACCGGAAG"

MIFISH_F = "GTCGGTAAAACTCGTGCCAGC"
MIFISH_R = "CATAGTGGGGTATCTAATCCCAGTTTG"
MIFISH_R_RC = "CAAACTGGGATTAGATACCCCACTATG"

TAXON_COLUMNS = ["genus_name", "family_name", "order_name", "class_name", "taxid"]


def clean_seq(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper().strip() if ch in "ACGTN")


def extract_primer_bounded(seq: object, forward: str, reverse: str, reverse_rc: str) -> str | None:
    seq = clean_seq(seq)
    start = seq.find(forward)
    if start < 0:
        return None
    search_start = start + len(forward)
    end = seq.find(reverse_rc, search_start)
    end_len = len(reverse_rc)
    if end < 0:
        end = seq.find(reverse, search_start)
        end_len = len(reverse)
    if end < 0:
        return None
    return seq[start:end + end_len]


def maybe_amplicon(seq: object, mode: str) -> str | None:
    if mode == "none":
        cleaned = clean_seq(seq)
        return cleaned or None
    if mode == "teleo":
        return extract_primer_bounded(seq, TELEO_F, TELEO_R, TELEO_R_RC)
    if mode == "mifish":
        return extract_primer_bounded(seq, MIFISH_F, MIFISH_R, MIFISH_R_RC)
    raise ValueError(f"Unsupported amplicon mode: {mode}")


def species_to_tree_label(species_name: object) -> str:
    return str(species_name).strip().replace(" ", "_")


def tree_label_to_species(label: str) -> str:
    return label.replace("_", " ")


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def clean_tax_value(value: object) -> object:
    return str(value).strip() if nonempty(value) else pd.NA


def first_nonempty(*values: object) -> object:
    for value in values:
        cleaned = clean_tax_value(value)
        if nonempty(cleaned):
            return cleaned
    return pd.NA


def read_tree_labels(tree_file: Path) -> list[str]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick", preserve_underscores=True)
    labels = []
    for leaf in tree.leaf_node_iter():
        if leaf.taxon and leaf.taxon.label:
            labels.append(leaf.taxon.label.strip("'\""))
    return labels


def load_external_taxonomy(taxonomy_csv: Path | None) -> dict[str, dict[str, object]]:
    if taxonomy_csv is None or not taxonomy_csv.exists():
        return {}
    df = pd.read_csv(taxonomy_csv)
    if "tree_label" not in df.columns:
        if "genus.species" in df.columns:
            df["tree_label"] = df["genus.species"].astype(str).str.strip().str.replace(" ", "_", regex=False)
        elif "species_name" in df.columns:
            df["tree_label"] = df["species_name"].astype(str).str.strip().str.replace(" ", "_", regex=False)
        else:
            raise RuntimeError(f"{taxonomy_csv} must contain tree_label, genus.species, or species_name")

    column_map = {
        "genus_name": ["genus_name", "genus"],
        "family_name": ["family_name", "family"],
        "order_name": ["order_name", "order"],
        "class_name": ["class_name", "class"],
        "taxid": ["taxid"],
    }
    rows: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        label = clean_tax_value(row.get("tree_label"))
        if not nonempty(label):
            continue
        tax: dict[str, object] = {}
        for out_col, source_cols in column_map.items():
            tax[out_col] = first_nonempty(*(row.get(col) for col in source_cols if col in row.index))
        rows[str(label)] = tax
    return rows


def read_split(data_dir: Path, split: str, amplicon: str) -> pd.DataFrame:
    path = data_dir / SPLIT_FILES[split]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "nucleotides" not in df.columns:
        raise RuntimeError(f"{path} must contain a nucleotides column")
    if "species_name" not in df.columns:
        return pd.DataFrame()
    for column in TAXON_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    if "processid" not in df.columns:
        df["processid"] = [f"{split}:{i}" for i in range(len(df))]
    if "source" not in df.columns:
        df["source"] = pd.NA

    df = df.copy()
    df["split"] = split
    df["original_nucleotides"] = df["nucleotides"].map(clean_seq)
    df["nucleotides"] = df["nucleotides"].map(lambda seq: maybe_amplicon(seq, amplicon))
    df["amplicon_retained"] = df["nucleotides"].notna()
    df = df[df["amplicon_retained"]].copy()
    df["seq_len"] = df["nucleotides"].str.len()
    df["tree_label"] = df["species_name"].map(species_to_tree_label)
    return df


def most_common_nonempty(values: pd.Series) -> object:
    cleaned = [
        str(value).strip()
        for value in values.dropna().tolist()
        if str(value).strip() and str(value).strip().lower() != "nan"
    ]
    if not cleaned:
        return pd.NA
    return Counter(cleaned).most_common(1)[0][0]


def sequence_dict(df: pd.DataFrame) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for label, sub in df.sort_values(["tree_label", "seq_len", "processid"]).groupby("tree_label"):
        grouped[str(label)] = sorted(set(sub["nucleotides"].astype(str)))
    return grouped


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_inputs(args: argparse.Namespace) -> dict[str, object]:
    tree_labels = read_tree_labels(args.tree_file)
    tree_label_set = set(tree_labels)
    tree_species = {label: tree_label_to_species(label) for label in tree_labels}
    external_taxonomy = load_external_taxonomy(args.taxonomy_csv)

    raw_split_counts: dict[str, int] = {}
    retained_split_counts: dict[str, int] = {}
    split_frames = []
    for split in SPLIT_FILES:
        path = args.data_dir / SPLIT_FILES[split]
        if not path.exists():
            continue
        raw_rows = sum(1 for _ in path.open()) - 1
        raw_split_counts[split] = max(raw_rows, 0)
        df = read_split(args.data_dir, split, args.amplicon)
        retained_split_counts[split] = int(len(df))
        split_frames.append(df)

    if not split_frames:
        raise RuntimeError(f"No split CSVs found in {args.data_dir}")
    all_rows = pd.concat(split_frames, ignore_index=True)
    all_rows["in_tree"] = all_rows["tree_label"].isin(tree_label_set)
    matched_rows = all_rows[all_rows["in_tree"]].copy()

    reference = matched_rows[matched_rows["split"].isin(args.reference_splits)].copy()
    train_reference = matched_rows[matched_rows["split"] == "supervised_train"].copy()
    eval_c = matched_rows[matched_rows["split"] == "eval_c_unseen_species"].copy()

    reference_species = set(reference["tree_label"])
    zero_shot_queries = eval_c[~eval_c["tree_label"].isin(reference_species)].copy()

    species_sequences = sequence_dict(reference)
    train_species_sequences = sequence_dict(train_reference)

    taxonomy = (
        matched_rows.groupby("tree_label", dropna=False)[TAXON_COLUMNS]
        .agg(most_common_nonempty)
        .reset_index()
    )
    taxonomy_by_label = taxonomy.set_index("tree_label").to_dict(orient="index")

    counts_by_label_split = (
        matched_rows.groupby(["tree_label", "split"]).size().unstack(fill_value=0)
    )

    candidates = []
    for label in tree_labels:
        split_counts = {
            f"{split}_rows": int(counts_by_label_split.loc[label, split])
            if label in counts_by_label_split.index and split in counts_by_label_split.columns
            else 0
            for split in SPLIT_FILES
        }
        source_tax = taxonomy_by_label.get(label, {})
        external_tax = external_taxonomy.get(label, {})
        reference_count = sum(
            split_counts.get(f"{split}_rows", 0) for split in args.reference_splits
        )
        genus_name = first_nonempty(source_tax.get("genus_name"), external_tax.get("genus_name"), tree_species[label].split(" ")[0])
        family_name = first_nonempty(source_tax.get("family_name"), external_tax.get("family_name"))
        order_name = first_nonempty(source_tax.get("order_name"), external_tax.get("order_name"))
        class_name = first_nonempty(source_tax.get("class_name"), external_tax.get("class_name"))
        taxid = first_nonempty(source_tax.get("taxid"), external_tax.get("taxid"))
        candidates.append({
            "tree_label": label,
            "species_name": tree_species[label],
            "genus_from_label": tree_species[label].split(" ")[0],
            "genus_name": genus_name,
            "family_name": family_name,
            "order_name": order_name,
            "class_name": class_name,
            "taxid": taxid,
            "has_any_sequence": int(label in set(matched_rows["tree_label"])),
            "has_reference_sequence": int(reference_count > 0),
            "reference_sequence_count": int(reference_count),
            "has_eval_c_query": int(split_counts.get("eval_c_unseen_species_rows", 0) > 0),
            **split_counts,
        })
    candidates_df = pd.DataFrame(candidates)

    species_info = {}
    for row in candidates:
        if pd.isna(row["genus_name"]) and pd.isna(row["family_name"]) and pd.isna(row["order_name"]):
            continue
        species_info[row["tree_label"]] = {
            "genus": None if pd.isna(row["genus_name"]) else row["genus_name"],
            "family": None if pd.isna(row["family_name"]) else row["family_name"],
            "order": None if pd.isna(row["order_name"]) else row["order_name"],
        }

    zero_cols = [
        "processid",
        "source",
        "tree_label",
        "species_name",
        "genus_name",
        "family_name",
        "order_name",
        "class_name",
        "taxid",
        "seq_len",
        "nucleotides",
    ]
    zero_shot_out = zero_shot_queries[zero_cols].sort_values(["tree_label", "processid"])

    val_species = sorted(set(matched_rows.loc[matched_rows["split"] == "supervised_val", "tree_label"]))
    test_species = sorted(set(matched_rows.loc[matched_rows["split"] == "supervised_test", "tree_label"]))
    eval_c_species = sorted(set(zero_shot_queries["tree_label"]))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "species_sequences.json", species_sequences)
    write_json(output_dir / "train_species_sequences.json", train_species_sequences)
    write_json(output_dir / "species_info.json", species_info)
    write_json(output_dir / "val_species.json", val_species)
    write_json(output_dir / "test_species.json", test_species)
    write_json(output_dir / "eval_c_species.json", eval_c_species)
    candidates_df.to_csv(output_dir / "candidate_species.csv", index=False)
    zero_shot_out.to_csv(output_dir / "zero_shot_queries.csv", index=False)

    per_source = {}
    if "source" in matched_rows.columns:
        source_counts = (
            matched_rows.groupby(["source", "split"]).size().unstack(fill_value=0)
        )
        per_source = {
            str(source): {str(split): int(count) for split, count in row.items()}
            for source, row in source_counts.iterrows()
        }

    not_in_tree = all_rows[~all_rows["in_tree"]]
    manifest: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "tree_file": str(args.tree_file),
        "taxonomy_csv": str(args.taxonomy_csv) if args.taxonomy_csv else None,
        "output_dir": str(output_dir),
        "amplicon": args.amplicon,
        "reference_splits": args.reference_splits,
        "raw_split_rows": raw_split_counts,
        "retained_split_rows_after_amplicon_filter": retained_split_counts,
        "tree_tip_count": len(tree_labels),
        "all_retained_rows": int(len(all_rows)),
        "rows_matching_tree": int(len(matched_rows)),
        "species_matching_tree": int(matched_rows["tree_label"].nunique()),
        "reference_rows_matching_tree": int(len(reference)),
        "reference_species_matching_tree": int(reference["tree_label"].nunique()),
        "train_reference_rows_matching_tree": int(len(train_reference)),
        "train_reference_species_matching_tree": int(train_reference["tree_label"].nunique()),
        "eval_c_rows_matching_tree": int(len(eval_c)),
        "zero_shot_eval_c_rows": int(len(zero_shot_queries)),
        "zero_shot_eval_c_species": int(zero_shot_queries["tree_label"].nunique()),
        "species_sequences_json_species": len(species_sequences),
        "species_sequences_json_sequences": int(sum(len(v) for v in species_sequences.values())),
        "train_species_sequences_json_species": len(train_species_sequences),
        "train_species_sequences_json_sequences": int(sum(len(v) for v in train_species_sequences.values())),
        "species_info_entries": len(species_info),
        "candidate_taxonomy_coverage": {
            column: int(candidates_df[column].map(nonempty).sum())
            for column in ["genus_name", "family_name", "order_name", "class_name", "taxid"]
            if column in candidates_df.columns
        },
        "per_source_retained_rows_matching_tree": per_source,
        "dropped_retained_species_not_in_tree": int(not_in_tree["tree_label"].nunique()),
        "dropped_retained_rows_not_in_tree": int(len(not_in_tree)),
        "not_in_tree_species_sample": sorted(set(not_in_tree["species_name"].astype(str)))[:50],
        "outputs": {
            "species_sequences_json": str(output_dir / "species_sequences.json"),
            "train_species_sequences_json": str(output_dir / "train_species_sequences.json"),
            "species_info_json": str(output_dir / "species_info.json"),
            "val_species_json": str(output_dir / "val_species.json"),
            "test_species_json": str(output_dir / "test_species.json"),
            "eval_c_species_json": str(output_dir / "eval_c_species.json"),
            "candidate_species_csv": str(output_dir / "candidate_species.csv"),
            "zero_shot_queries_csv": str(output_dir / "zero_shot_queries.csv"),
        },
        "notes": [
            "species_sequences.json uses tree labels with underscores, matching TAXDNA input format.",
            "zero_shot_queries.csv contains Eval C species that have no sequence in the selected reference splits.",
            "Species-level zero-shot evaluation must score candidates from the tree; closed-set train labels are insufficient.",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/edna/processed_12s_multisource"),
        help="Processed 12S split directory.",
    )
    parser.add_argument(
        "--tree-file",
        type=Path,
        default=Path("data/phylo/actinopt_12k_treePL.tre"),
        help="Newick tree whose leaf labels define the open candidate universe.",
    )
    parser.add_argument(
        "--taxonomy-csv",
        type=Path,
        default=Path("data/phylo/PFC_taxonomy.csv"),
        help="Optional taxonomy table used to fill genus/family/order for all tree candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/edna/stalder_inputs/multisource"),
        help="Output directory.",
    )
    parser.add_argument(
        "--reference-splits",
        nargs="+",
        default=["supervised_train", "supervised_val", "supervised_test"],
        choices=list(SPLIT_FILES),
        help="Splits treated as known-reference species for sequence database construction.",
    )
    parser.add_argument(
        "--amplicon",
        choices=["none", "teleo", "mifish"],
        default="none",
        help="Optional exact primer-bounded amplicon extraction before building JSONs.",
    )
    args = parser.parse_args()

    manifest = build_inputs(args)
    print(
        "Wrote {out}: {species} reference species, {seqs} reference sequences, "
        "{queries} zero-shot Eval C queries across {query_species} species.".format(
            out=args.output_dir,
            species=manifest["species_sequences_json_species"],
            seqs=manifest["species_sequences_json_sequences"],
            queries=manifest["zero_shot_eval_c_rows"],
            query_species=manifest["zero_shot_eval_c_species"],
        )
    )


if __name__ == "__main__":
    main()
