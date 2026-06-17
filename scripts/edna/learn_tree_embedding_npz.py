#!/usr/bin/env python3
"""Learn and save shared NPZ tree embeddings for TAXDNA-style experiments."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from phylo_zero_shot_common import (
    learn_tree_embeddings,
    load_zero_shot_inputs,
    ordered_candidate_labels,
    read_tree,
    save_tree_embedding_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--tree-epochs", type=int, default=300)
    parser.add_argument("--tree-pairs-per-epoch", type=int, default=768)
    parser.add_argument("--tree-negatives", type=int, default=32)
    parser.add_argument("--tree-max-distance-sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inputs = load_zero_shot_inputs(args.input_dir)
    _, taxa = read_tree(args.tree_file)
    candidate_labels = ordered_candidate_labels(inputs.candidate_species, set(taxa))
    summary = {
        "input_dir": str(args.input_dir),
        "tree_file": str(args.tree_file),
        "output_npz": str(args.output_npz),
        "candidate_count": len(candidate_labels),
        "embed_dim": args.embed_dim,
        "tree_epochs": args.tree_epochs,
        "tree_pairs_per_epoch": args.tree_pairs_per_epoch,
        "tree_negatives": args.tree_negatives,
        "tree_max_distance_sample": args.tree_max_distance_sample,
        "seed": args.seed,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = learn_tree_embeddings(
        tree_file=args.tree_file,
        labels=candidate_labels,
        embed_dim=args.embed_dim,
        epochs=args.tree_epochs,
        pairs_per_epoch=args.tree_pairs_per_epoch,
        negatives_per_anchor=args.tree_negatives,
        sample_size_for_max_distance=args.tree_max_distance_sample,
        device=device,
        seed=args.seed,
    )
    metadata = {
        **summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
    }
    save_tree_embedding_npz(args.output_npz, candidate_labels, embeddings, metadata)
    print(f"Wrote {args.output_npz}")


if __name__ == "__main__":
    main()
