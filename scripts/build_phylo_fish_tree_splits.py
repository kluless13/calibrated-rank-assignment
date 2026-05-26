#!/usr/bin/env python3
"""Build strict clean COI Fish Tree splits with predeclared Eval C holdouts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd


def tree_species(tree_file: Path) -> set[str]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick", preserve_underscores=True)
    return {
        taxon.label.strip("'\"").replace("_", " ")
        for taxon in tree.taxon_namespace
        if taxon.label
    }


def pick_eval_c_species(
    train_fish: pd.DataFrame,
    min_species_per_genus: int,
    holdout_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    species_by_genus = (
        train_fish[["genus_name", "species_name"]]
        .dropna()
        .drop_duplicates()
        .groupby("genus_name")["species_name"]
        .apply(lambda values: sorted(set(values)))
    )
    for genus, species in species_by_genus.items():
        if len(species) < min_species_per_genus:
            continue
        n_holdout = max(1, int(np.floor(len(species) * holdout_fraction)))
        n_holdout = min(n_holdout, len(species) - 1)
        chosen = sorted(rng.choice(species, size=n_holdout, replace=False).tolist())
        for name in chosen:
            rows.append({
                "species_name": name,
                "genus_name": genus,
                "eligible_genus_species": len(species),
                "holdout_fraction": holdout_fraction,
                "selection_seed": seed,
            })
    return pd.DataFrame(rows).sort_values(["genus_name", "species_name"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_clean"))
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/phylo/fish_tree_clean_splits"))
    parser.add_argument("--min-species-per-genus", type=int, default=3)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_names = tree_species(args.tree_file)

    train = pd.read_csv(args.data_dir / "supervised_train.csv")
    val = pd.read_csv(args.data_dir / "supervised_val.csv")
    test = pd.read_csv(args.data_dir / "supervised_test.csv")
    unseen = pd.read_csv(args.data_dir / "unseen.csv")

    train_fish = train[train["species_name"].isin(tree_names)].copy()
    val_fish = val[val["species_name"].isin(tree_names)].copy()
    test_fish = test[test["species_name"].isin(tree_names)].copy()
    unseen_fish = unseen[unseen["species_name"].isin(tree_names)].copy()

    holdout = pick_eval_c_species(
        train_fish,
        min_species_per_genus=args.min_species_per_genus,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    holdout_species = set(holdout["species_name"].astype(str))

    reference_train = train_fish[~train_fish["species_name"].isin(holdout_species)].copy()
    reference_val = val_fish[~val_fish["species_name"].isin(holdout_species)].copy()
    seen_test = test_fish[~test_fish["species_name"].isin(holdout_species)].copy()
    eval_c_query = pd.concat(
        [
            train_fish[train_fish["species_name"].isin(holdout_species)],
            val_fish[val_fish["species_name"].isin(holdout_species)],
            test_fish[test_fish["species_name"].isin(holdout_species)],
        ],
        ignore_index=True,
    )

    reference_species = set(reference_train["species_name"].astype(str))
    eval_c_species = set(eval_c_query["species_name"].astype(str))
    unseen_species = set(unseen_fish["species_name"].astype(str))
    overlap_ref_eval_c = sorted(reference_species & eval_c_species)
    overlap_ref_unseen = sorted(reference_species & unseen_species)

    outputs = {
        "reference_train_csv": args.output_dir / "reference_train.csv",
        "reference_val_csv": args.output_dir / "reference_val.csv",
        "seen_test_csv": args.output_dir / "seen_test.csv",
        "eval_c_query_csv": args.output_dir / "eval_c_query.csv",
        "unseen_genera_query_csv": args.output_dir / "unseen_genera_query.csv",
        "eval_c_holdout_species_csv": args.output_dir / "eval_c_holdout_species.csv",
    }
    reference_train.to_csv(outputs["reference_train_csv"], index=False)
    reference_val.to_csv(outputs["reference_val_csv"], index=False)
    seen_test.to_csv(outputs["seen_test_csv"], index=False)
    eval_c_query.to_csv(outputs["eval_c_query_csv"], index=False)
    unseen_fish.to_csv(outputs["unseen_genera_query_csv"], index=False)
    holdout.to_csv(outputs["eval_c_holdout_species_csv"], index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "tree_file": str(args.tree_file),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "min_species_per_genus": args.min_species_per_genus,
        "holdout_fraction": args.holdout_fraction,
        "tree_species": len(tree_names),
        "reference_train_sequences": int(len(reference_train)),
        "reference_train_species": int(reference_train["species_name"].nunique()),
        "reference_train_genera": int(reference_train["genus_name"].nunique()),
        "reference_val_sequences": int(len(reference_val)),
        "reference_val_species": int(reference_val["species_name"].nunique()),
        "seen_test_sequences": int(len(seen_test)),
        "seen_test_species": int(seen_test["species_name"].nunique()),
        "eval_c_query_sequences": int(len(eval_c_query)),
        "eval_c_query_species": int(eval_c_query["species_name"].nunique()),
        "eval_c_query_genera": int(eval_c_query["genus_name"].nunique()),
        "unseen_genera_query_sequences": int(len(unseen_fish)),
        "unseen_genera_query_species": int(unseen_fish["species_name"].nunique()),
        "unseen_genera_query_genera": int(unseen_fish["genus_name"].nunique()),
        "overlap_reference_eval_c_species": overlap_ref_eval_c,
        "overlap_reference_unseen_species": overlap_ref_unseen,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "notes": [
            "Eval C holdout species are removed from reference train, reference val, and seen test before model training.",
            "Eval C query can use all sequences from those held-out species because none are used for sequence-model fitting.",
            "The unseen_genera_query split is inherited from data/processed_clean/unseen.csv.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if overlap_ref_eval_c or overlap_ref_unseen:
        raise SystemExit("Split leakage detected; see manifest overlaps.")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
