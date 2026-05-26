#!/usr/bin/env python3
"""Sequence-only Fish Tree distance baselines for cleaned COI splits."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


DNA = "ACGT"


def kmer_index(k: int) -> dict[str, int]:
    kmers = [""]
    for _ in range(k):
        kmers = [prefix + base for prefix in kmers for base in DNA]
    return {kmer: idx for idx, kmer in enumerate(kmers)}


def kmer_features(seqs: list[str], index: dict[str, int], k: int) -> np.ndarray:
    vec = np.zeros(len(index), dtype=np.float32)
    for seq in seqs:
        text = str(seq).upper()
        for i in range(0, max(0, len(text) - k + 1)):
            idx = index.get(text[i : i + k])
            if idx is not None:
                vec[idx] += 1.0
    total = float(vec.sum())
    if total > 0:
        vec /= total
    return vec


def representative_sequence(seqs: list[str]) -> str:
    if not seqs:
        return ""
    lengths = np.array([len(str(seq)) for seq in seqs])
    median_len = np.median(lengths)
    best = int(np.argmin(np.abs(lengths - median_len)))
    return str(seqs[best]).upper()


def p_distance(seq_a: str, seq_b: str) -> float:
    n = min(len(seq_a), len(seq_b))
    valid = 0
    diff = 0
    for a, b in zip(seq_a[:n], seq_b[:n]):
        if a in DNA and b in DNA:
            valid += 1
            diff += int(a != b)
    if valid == 0:
        return float("nan")
    return diff / valid


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return 1.0 - float(np.dot(a, b) / denom)


def binary_jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    union = int(np.logical_or(aa, bb).sum())
    if union == 0:
        return float("nan")
    intersection = int(np.logical_and(aa, bb).sum())
    return 1.0 - intersection / union


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
    species_a: str,
    species_b: str,
    taxon_nodes: dict[str, object],
    depths: dict[object, float],
    ancestor_lists: dict[str, list[object]],
    ancestor_sets: dict[str, set[object]],
) -> float:
    node_a = taxon_nodes[species_a]
    node_b = taxon_nodes[species_b]
    ancestors_b = ancestor_sets[species_b]
    mrca = None
    for node in ancestor_lists[species_a]:
        if node in ancestors_b:
            mrca = node
            break
    if mrca is None:
        return float("nan")
    return depths[node_a] + depths[node_b] - 2 * depths[mrca]


def sample_pairs(items_a: list[str], items_b: list[str] | None, max_pairs: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    if items_b is None:
        total = len(items_a) * (len(items_a) - 1) // 2
        if total <= max_pairs:
            return [(items_a[i], items_a[j]) for i, j in combinations(range(len(items_a)), 2)]
        seen: set[tuple[int, int]] = set()
        while len(seen) < max_pairs:
            i = int(rng.integers(0, len(items_a)))
            j = int(rng.integers(0, len(items_a) - 1))
            if j >= i:
                j += 1
            pair = (min(i, j), max(i, j))
            seen.add(pair)
        return [(items_a[i], items_a[j]) for i, j in sorted(seen)]

    total = len(items_a) * len(items_b)
    if total <= max_pairs:
        return [(a, b) for a in items_a for b in items_b]
    seen_cross: set[tuple[int, int]] = set()
    while len(seen_cross) < max_pairs:
        seen_cross.add((int(rng.integers(0, len(items_a))), int(rng.integers(0, len(items_b)))))
    return [(items_a[i], items_b[j]) for i, j in sorted(seen_cross)]


def correlation(values: list[tuple[float, float]]) -> dict[str, float | int]:
    arr = np.array([(x, y) for x, y in values if np.isfinite(x) and np.isfinite(y)], dtype=np.float64)
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
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_clean"))
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phylo_tree_distance_baselines_clean"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seqs-per-species", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree = dendropy.Tree.get(path=str(args.tree_file), schema="newick")
    taxon_nodes = {
        node.taxon.label.replace("_", " "): node
        for node in tree.leaf_node_iter()
        if node.taxon is not None and node.taxon.label
    }
    depths = node_depths(tree)
    ancestor_lists = {species: ancestors_to_root(node) for species, node in taxon_nodes.items()}
    ancestor_sets = {species: set(nodes) for species, nodes in ancestor_lists.items()}

    train = pd.read_csv(args.data_dir / "supervised_train.csv")
    unseen = pd.read_csv(args.data_dir / "unseen.csv")
    train = train[train["species_name"].isin(taxon_nodes)].copy()
    unseen = unseen[unseen["species_name"].isin(taxon_nodes)].copy()

    feature_index = kmer_index(args.k)
    features: dict[str, np.ndarray] = {}
    representatives: dict[str, str] = {}
    source_split: dict[str, str] = {}

    for split_name, frame in [("train", train), ("unseen", unseen)]:
        for species, sub in frame.groupby("species_name"):
            seqs = sub["nucleotides"].dropna().astype(str).head(args.seqs_per_species).tolist()
            if not seqs:
                continue
            features[species] = kmer_features(seqs, feature_index, args.k)
            representatives[species] = representative_sequence(seqs)
            source_split[species] = split_name

    train_species = sorted(set(train["species_name"]) & set(features))
    unseen_species = sorted(set(unseen["species_name"]) & set(features))
    pair_sets = {
        "train_train": sample_pairs(train_species, None, args.max_pairs, args.seed),
        "unseen_unseen": sample_pairs(unseen_species, None, args.max_pairs, args.seed + 1),
        "unseen_train": sample_pairs(unseen_species, train_species, args.max_pairs, args.seed + 2),
    }

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    rows = []
    for set_name, pairs in pair_sets.items():
        method_values = {
            "kmer_cosine": [],
            "kmer_jaccard": [],
            "raw_p_distance": [],
        }
        for species_a, species_b in pairs:
            tree_dist = tree_distance(
                species_a,
                species_b,
                taxon_nodes=taxon_nodes,
                depths=depths,
                ancestor_lists=ancestor_lists,
                ancestor_sets=ancestor_sets,
            )
            if not np.isfinite(tree_dist):
                continue
            dist_values = {
                "kmer_cosine": cosine_distance(features[species_a], features[species_b]),
                "kmer_jaccard": binary_jaccard_distance(features[species_a], features[species_b]),
                "raw_p_distance": p_distance(representatives[species_a], representatives[species_b]),
            }
            for method, seq_dist in dist_values.items():
                method_values[method].append((seq_dist, tree_dist))
            rows.append({
                "pair_set": set_name,
                "species_a": species_a,
                "species_b": species_b,
                "split_a": source_split.get(species_a),
                "split_b": source_split.get(species_b),
                "tree_distance": tree_dist,
                **dist_values,
            })
        metrics[set_name] = {
            method: correlation(values)
            for method, values in method_values.items()
        }

    pair_path = args.output_dir / "sampled_pair_distances.csv"
    pd.DataFrame(rows).to_csv(pair_path, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "tree_file": str(args.tree_file),
        "output_dir": str(args.output_dir),
        "k": args.k,
        "seqs_per_species": args.seqs_per_species,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "train_fish_sequences": int(len(train)),
        "train_fish_species": int(len(train_species)),
        "unseen_fish_sequences": int(len(unseen)),
        "unseen_fish_species": int(len(unseen_species)),
        "pair_distances_csv": str(pair_path),
        "metrics": metrics,
        "notes": [
            "These are sequence-only distance baselines against Fish Tree of Life distances.",
            "No learned model is used; this is the control for phylogenetic tree-recovery reporting.",
        ],
    }
    out_path = args.output_dir / "tree_distance_baseline_metrics.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
