#!/usr/bin/env python3
"""Build strict Fish Tree DNA-to-tree input directories from clean COI splits."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import dendropy
import pandas as pd


QUERY_SPLITS = {
    "eval_c": "eval_c_query.csv",
    "seen_test": "seen_test.csv",
    "unseen_genera": "unseen_genera_query.csv",
}


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def tree_label(species_name: object) -> str:
    return str(species_name).strip().replace(" ", "_")


def species_name_from_label(label: str) -> str:
    return label.replace("_", " ")


def genus_from_label(label: str) -> str:
    return label.split("_", 1)[0]


def clean_seq(value: object) -> str:
    return "".join(ch for ch in str(value).upper().strip() if ch in "ACGTN")


def load_tree_labels(tree_file: Path) -> list[str]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick", preserve_underscores=True)
    labels = [
        taxon.label.strip("'\"")
        for taxon in tree.taxon_namespace
        if taxon.label and "_" in taxon.label
    ]
    return sorted(dict.fromkeys(labels))


def taxonomy_lookup(taxonomy_csv: Path) -> dict[str, dict[str, str]]:
    taxonomy = pd.read_csv(taxonomy_csv)
    lookup: dict[str, dict[str, str]] = {}
    for _, row in taxonomy.iterrows():
        if not nonempty(row.get("genus.species")):
            continue
        label = tree_label(row["genus.species"])
        lookup[label] = {
            "species_name": str(row["genus.species"]).strip(),
            "genus_name": str(row["genus"]).strip() if nonempty(row.get("genus")) else genus_from_label(label),
            "family_name": str(row["family"]).strip() if nonempty(row.get("family")) else "",
            "order_name": str(row["order"]).strip() if nonempty(row.get("order")) else "",
            "class_name": str(row["class"]).strip() if nonempty(row.get("class")) else "",
        }
    return lookup


def split_taxonomy_rows(frames: list[pd.DataFrame]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for frame in frames:
        for _, row in frame.iterrows():
            if not nonempty(row.get("species_name")):
                continue
            label = tree_label(row["species_name"])
            lookup[label] = {
                "species_name": str(row["species_name"]).strip(),
                "genus_name": str(row["genus_name"]).strip() if nonempty(row.get("genus_name")) else genus_from_label(label),
                "family_name": str(row["family_name"]).strip() if nonempty(row.get("family_name")) else "",
                "order_name": str(row["order_name"]).strip() if nonempty(row.get("order_name")) else "",
                "class_name": "",
            }
    return lookup


def species_sequences(frame: pd.DataFrame, allowed_labels: set[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for _, row in frame.iterrows():
        label = tree_label(row["species_name"])
        seq = clean_seq(row["nucleotides"])
        if label in allowed_labels and seq:
            grouped[label].append(seq)
    return {label: seqs for label, seqs in sorted(grouped.items()) if seqs}


def query_frame(frame: pd.DataFrame, split_name: str, allowed_labels: set[str]) -> pd.DataFrame:
    rows = []
    for idx, row in frame.reset_index(drop=True).iterrows():
        label = tree_label(row["species_name"])
        seq = clean_seq(row["nucleotides"])
        if label not in allowed_labels or not seq:
            continue
        processid = str(row["processid"]) if nonempty(row.get("processid")) else f"{split_name}:{label}:{idx}"
        rows.append(
            {
                "processid": processid,
                "tree_label": label,
                "species_name": str(row["species_name"]).strip(),
                "genus_name": str(row["genus_name"]).strip() if nonempty(row.get("genus_name")) else genus_from_label(label),
                "family_name": str(row["family_name"]).strip() if nonempty(row.get("family_name")) else "",
                "order_name": str(row["order_name"]).strip() if nonempty(row.get("order_name")) else "",
                "nucleotides": seq,
                "split": split_name,
            }
        )
    return pd.DataFrame(rows)


def candidate_table(
    tree_labels: list[str],
    reference_sequences: dict[str, list[str]],
    queries_by_split: dict[str, pd.DataFrame],
    tax_lookup: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for label in tree_labels:
        tax = tax_lookup.get(label, {})
        row = {
            "tree_label": label,
            "species_name": tax.get("species_name") or species_name_from_label(label),
            "genus_from_label": genus_from_label(label),
            "genus_name": tax.get("genus_name") or genus_from_label(label),
            "family_name": tax.get("family_name") or "",
            "order_name": tax.get("order_name") or "",
            "class_name": tax.get("class_name") or "",
            "has_reference_sequence": int(label in reference_sequences),
            "reference_sequence_count": int(len(reference_sequences.get(label, []))),
        }
        for split_name, queries in queries_by_split.items():
            split_counts = queries["tree_label"].value_counts().to_dict() if len(queries) else {}
            row[f"has_{split_name}_query"] = int(label in split_counts)
            row[f"{split_name}_query_rows"] = int(split_counts.get(label, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def species_info_from_candidates(candidates: pd.DataFrame) -> dict[str, dict[str, str]]:
    info = {}
    for _, row in candidates.iterrows():
        label = str(row["tree_label"])
        info[label] = {
            "species": str(row["species_name"]),
            "genus": str(row["genus_name"]) if nonempty(row.get("genus_name")) else genus_from_label(label),
            "family": str(row["family_name"]) if nonempty(row.get("family_name")) else "",
            "order": str(row["order_name"]) if nonempty(row.get("order_name")) else "",
            "class": str(row["class_name"]) if nonempty(row.get("class_name")) else "",
            "genus_name": str(row["genus_name"]) if nonempty(row.get("genus_name")) else genus_from_label(label),
            "family_name": str(row["family_name"]) if nonempty(row.get("family_name")) else "",
            "order_name": str(row["order_name"]) if nonempty(row.get("order_name")) else "",
            "class_name": str(row["class_name"]) if nonempty(row.get("class_name")) else "",
        }
    return info


def write_input_dir(
    out_dir: Path,
    split_name: str,
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    species_info: dict[str, dict[str, str]],
    reference_sequences: dict[str, list[str]],
    val_species: list[str],
    args: argparse.Namespace,
    source_manifest: dict[str, object],
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "species_sequences.json").write_text(json.dumps(reference_sequences, sort_keys=True) + "\n")
    (out_dir / "train_species_sequences.json").write_text(json.dumps(reference_sequences, sort_keys=True) + "\n")
    (out_dir / "species_info.json").write_text(json.dumps(species_info, sort_keys=True) + "\n")
    (out_dir / "val_species.json").write_text(json.dumps(val_species, indent=2, sort_keys=True) + "\n")
    candidates.to_csv(out_dir / "candidate_species.csv", index=False)
    query.to_csv(out_dir / "zero_shot_queries.csv", index=False)
    (out_dir / "eval_c_species.json").write_text(
        json.dumps(sorted(query["tree_label"].unique().tolist()), indent=2) + "\n"
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/build_fish_tree_phylo_inputs.py",
        "source_split_dir": str(args.split_dir),
        "source_manifest": source_manifest,
        "tree_file": str(args.tree_file),
        "taxonomy_csv": str(args.taxonomy_csv),
        "query_split": split_name,
        "candidate_species": int(len(candidates)),
        "reference_species": int(len(reference_sequences)),
        "reference_sequences": int(sum(len(seqs) for seqs in reference_sequences.values())),
        "val_species": int(len(val_species)),
        "zero_shot_query_rows": int(len(query)),
        "zero_shot_query_species": int(query["tree_label"].nunique()),
        "outputs": {
            "species_sequences_json": str(out_dir / "species_sequences.json"),
            "train_species_sequences_json": str(out_dir / "train_species_sequences.json"),
            "species_info_json": str(out_dir / "species_info.json"),
            "candidate_species_csv": str(out_dir / "candidate_species.csv"),
            "zero_shot_queries_csv": str(out_dir / "zero_shot_queries.csv"),
            "val_species_json": str(out_dir / "val_species.json"),
        },
        "notes": [
            "Eval C holdout species are absent from species_sequences and train_species_sequences.",
            "Reference validation species are predeclared in val_species.json.",
            "Candidate species include all Fish Tree tips with taxonomy where available.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("data/phylo/fish_tree_clean_splits"))
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--taxonomy-csv", type=Path, default=Path("data/phylo/PFC_taxonomy.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/phylo/fish_tree_clean_phylo_inputs"))
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((args.split_dir / "manifest.json").read_text())
    tree_labels = load_tree_labels(args.tree_file)
    tree_label_set = set(tree_labels)

    reference_train = pd.read_csv(args.split_dir / "reference_train.csv")
    reference_val = pd.read_csv(args.split_dir / "reference_val.csv")
    split_frames = [reference_train, reference_val]
    queries_by_split = {}
    for split_name, filename in QUERY_SPLITS.items():
        frame = pd.read_csv(args.split_dir / filename)
        split_frames.append(frame)
        queries_by_split[split_name] = query_frame(frame, split_name, tree_label_set)

    tax_lookup = taxonomy_lookup(args.taxonomy_csv)
    tax_lookup.update({key: {**tax_lookup.get(key, {}), **value} for key, value in split_taxonomy_rows(split_frames).items()})

    reference_frame = pd.concat([reference_train, reference_val], ignore_index=True)
    reference_sequences = species_sequences(reference_frame, tree_label_set)
    eval_c_labels = set(queries_by_split["eval_c"]["tree_label"].unique())
    leaked = sorted(eval_c_labels & set(reference_sequences))
    if leaked:
        raise SystemExit(f"Eval C leakage detected in reference sequences: {leaked[:10]}")

    val_species = sorted(set(query_frame(reference_val, "reference_val", tree_label_set)["tree_label"]))
    candidates = candidate_table(tree_labels, reference_sequences, queries_by_split, tax_lookup)
    species_info = species_info_from_candidates(candidates)

    manifests = {}
    for split_name, query in queries_by_split.items():
        manifests[split_name] = write_input_dir(
            args.output_root / split_name,
            split_name,
            query,
            candidates,
            species_info,
            reference_sequences,
            val_species,
            args,
            source_manifest,
        )

    top_manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split_dir": str(args.split_dir),
        "output_root": str(args.output_root),
        "tree_file": str(args.tree_file),
        "taxonomy_csv": str(args.taxonomy_csv),
        "tree_candidate_species": int(len(tree_labels)),
        "reference_species": int(len(reference_sequences)),
        "reference_sequences": int(sum(len(seqs) for seqs in reference_sequences.values())),
        "query_manifests": {name: str(args.output_root / name / "manifest.json") for name in manifests},
        "query_counts": {
            name: {
                "rows": int(len(query)),
                "species": int(query["tree_label"].nunique()),
            }
            for name, query in queries_by_split.items()
        },
    }
    (args.output_root / "manifest.json").write_text(json.dumps(top_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(top_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
