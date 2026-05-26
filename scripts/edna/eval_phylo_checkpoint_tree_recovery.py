#!/usr/bin/env python3
"""Evaluate a PhyloMamba checkpoint against Fish Tree distances."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from phylo_zero_shot_common import extract_embeddings, load_zero_shot_inputs
from train_12s_phylo_mamba import PhyloMamba


def node_depths(tree: dendropy.Tree) -> dict[object, float]:
    root = tree.seed_node
    depths: dict[object, float] = {root: 0.0}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.child_node_iter():
            edge_length = child.edge_length if child.edge_length is not None else 0.0
            depths[child] = depths[node] + float(edge_length)
            stack.append(child)
    return depths


def ancestors_to_root(node: object) -> list[object]:
    ancestors = []
    current = node
    while current is not None:
        ancestors.append(current)
        current = current.parent_node
    return ancestors


def tree_distance(
    label_a: str,
    label_b: str,
    taxon_nodes: dict[str, object],
    depths: dict[object, float],
    ancestor_lists: dict[str, list[object]],
    ancestor_sets: dict[str, set[object]],
) -> float:
    node_a = taxon_nodes[label_a]
    node_b = taxon_nodes[label_b]
    ancestors_b = ancestor_sets[label_b]
    mrca = None
    for node in ancestor_lists[label_a]:
        if node in ancestors_b:
            mrca = node
            break
    if mrca is None:
        return float("nan")
    return depths[node_a] + depths[node_b] - 2 * depths[mrca]


def sample_pairs(labels_a: list[str], labels_b: list[str] | None, max_pairs: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    if labels_b is None:
        total = len(labels_a) * (len(labels_a) - 1) // 2
        if total <= max_pairs:
            return [(labels_a[i], labels_a[j]) for i, j in combinations(range(len(labels_a)), 2)]
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < max_pairs:
            i = int(rng.integers(0, len(labels_a)))
            j = int(rng.integers(0, len(labels_a) - 1))
            if j >= i:
                j += 1
            pairs.add((min(i, j), max(i, j)))
        return [(labels_a[i], labels_a[j]) for i, j in sorted(pairs)]

    total = len(labels_a) * len(labels_b)
    if total <= max_pairs:
        return [(a, b) for a in labels_a for b in labels_b]
    pairs_cross: set[tuple[int, int]] = set()
    while len(pairs_cross) < max_pairs:
        pairs_cross.add((int(rng.integers(0, len(labels_a))), int(rng.integers(0, len(labels_b)))))
    return [(labels_a[i], labels_b[j]) for i, j in sorted(pairs_cross)]


def species_query_frame(species_sequences: dict[str, list[str]], seqs_per_species: int) -> pd.DataFrame:
    rows = []
    for label, seqs in species_sequences.items():
        for idx, seq in enumerate(seqs[:seqs_per_species]):
            rows.append({"processid": f"{label}:{idx}", "tree_label": label, "nucleotides": seq})
    return pd.DataFrame(rows)


def average_species_embeddings(frame: pd.DataFrame, embeddings: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label, indices in frame.groupby("tree_label").groups.items():
        out[str(label)] = embeddings[list(indices)].mean(axis=0)
    return out


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.dot(a, b) / denom)


def correlation(rows: list[dict[str, object]], distance_column: str) -> dict[str, float | int]:
    arr = np.array(
        [
            (float(row[distance_column]), float(row["tree_distance"]))
            for row in rows
            if np.isfinite(row[distance_column]) and np.isfinite(row["tree_distance"])
        ],
        dtype=np.float64,
    )
    if len(arr) < 3:
        return {"n_pairs": int(len(arr)), "pearson_r": 0.0, "spearman_r": 0.0}
    pr, pp = pearsonr(arr[:, 0], arr[:, 1])
    sr, sp = spearmanr(arr[:, 0], arr[:, 1])
    return {
        "n_pairs": int(len(arr)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=700)
    parser.add_argument("--pooling", choices=["legacy_mean", "masked_mean", "last_token"], default="masked_mean")
    parser.add_argument("--seqs-per-species", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_zero_shot_inputs(args.input_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PhyloMamba(embed_dim=args.embed_dim, pooling=args.pooling)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    reference_frame = species_query_frame(inputs.train_species_sequences, args.seqs_per_species)
    reference_embeddings = extract_embeddings(
        model,
        reference_frame,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    reference_by_species = average_species_embeddings(reference_frame, reference_embeddings)

    query_frame = inputs.zero_shot_queries[["processid", "tree_label", "nucleotides"]].copy()
    query_embeddings = extract_embeddings(
        model,
        query_frame,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    query_by_species = average_species_embeddings(query_frame, query_embeddings)

    tree = dendropy.Tree.get(path=str(args.tree_file), schema="newick", preserve_underscores=True)
    taxon_nodes = {
        node.taxon.label.strip("'\""): node
        for node in tree.leaf_node_iter()
        if node.taxon is not None and node.taxon.label
    }
    depths = node_depths(tree)
    ancestor_lists = {label: ancestors_to_root(node) for label, node in taxon_nodes.items()}
    ancestor_sets = {label: set(nodes) for label, nodes in ancestor_lists.items()}

    reference_labels = sorted(set(reference_by_species) & set(taxon_nodes))
    query_labels = sorted(set(query_by_species) & set(taxon_nodes))
    pair_sets = {
        "reference_reference": sample_pairs(reference_labels, None, args.max_pairs, args.seed),
        "zero_shot_zero_shot": sample_pairs(query_labels, None, args.max_pairs, args.seed + 1),
        "zero_shot_reference": sample_pairs(query_labels, reference_labels, args.max_pairs, args.seed + 2),
    }

    all_rows = []
    metrics = {}
    for set_name, pairs in pair_sets.items():
        rows = []
        for label_a, label_b in pairs:
            emb_a = query_by_species.get(label_a, reference_by_species.get(label_a))
            emb_b = query_by_species.get(label_b, reference_by_species.get(label_b))
            if emb_a is None or emb_b is None:
                continue
            tdist = tree_distance(label_a, label_b, taxon_nodes, depths, ancestor_lists, ancestor_sets)
            row = {
                "pair_set": set_name,
                "label_a": label_a,
                "label_b": label_b,
                "tree_distance": tdist,
                "embedding_cosine_distance": cosine_distance(emb_a, emb_b),
            }
            rows.append(row)
            all_rows.append(row)
        metrics[set_name] = correlation(rows, "embedding_cosine_distance")

    pair_path = args.output_dir / "sampled_tree_recovery_pairs.csv"
    pd.DataFrame(all_rows).to_csv(pair_path, index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "tree_file": str(args.tree_file),
        "embed_dim": args.embed_dim,
        "max_seq_len": args.max_seq_len,
        "pooling": args.pooling,
        "seqs_per_species": args.seqs_per_species,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "reference_species": int(len(reference_labels)),
        "zero_shot_species": int(len(query_labels)),
        "pair_distances_csv": str(pair_path),
        "metrics": metrics,
    }
    out_path = args.output_dir / "tree_recovery_metrics.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
