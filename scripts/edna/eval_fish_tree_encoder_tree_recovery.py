#!/usr/bin/env python3
"""Evaluate non-Mamba fish-tree encoder checkpoints against tree distances."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phylo_zero_shot_common import extract_embeddings, load_zero_shot_inputs  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402
from train_fish_tree_encoder_benchmark import apply_model_config, build_model  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def node_depths(tree: dendropy.Tree) -> dict[object, float]:
    depths: dict[object, float] = {tree.seed_node: 0.0}
    stack = [tree.seed_node]
    while stack:
        node = stack.pop()
        for child in node.child_node_iter():
            depths[child] = depths[node] + float(child.edge_length or 0.0)
            stack.append(child)
    return depths


def ancestors_to_root(node: object) -> list[object]:
    out = []
    current = node
    while current is not None:
        out.append(current)
        current = current.parent_node
    return out


def tree_distance(
    label_a: str,
    label_b: str,
    taxon_nodes: dict[str, object],
    depths: dict[object, float],
    ancestor_lists: dict[str, list[object]],
    ancestor_sets: dict[str, set[object]],
) -> float:
    for node in ancestor_lists[label_a]:
        if node in ancestor_sets[label_b]:
            return float(depths[taxon_nodes[label_a]] + depths[taxon_nodes[label_b]] - 2 * depths[node])
    return float("nan")


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
    return {
        str(label): embeddings[list(indices)].mean(axis=0)
        for label, indices in frame.groupby("tree_label").groups.items()
    }


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
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--seqs-per-species", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading run manifest from {args.run_manifest}")
    run_manifest = json.loads(args.run_manifest.read_text())
    apply_model_config(args, run_manifest["model_config"])
    logger.log(f"Loading zero-shot inputs from {args.input_dir}")
    inputs = load_zero_shot_inputs(args.input_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"Using device: {device}")

    logger.log(f"Loading checkpoint {args.checkpoint}")
    model = build_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    reference_frame = species_query_frame(inputs.train_species_sequences, args.seqs_per_species)
    logger.log(f"Extracting reference embeddings for {len(reference_frame)} sequences")
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
    logger.log(f"Extracting query embeddings for {len(query_frame)} sequences")
    query_embeddings = extract_embeddings(
        model,
        query_frame,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    query_by_species = average_species_embeddings(query_frame, query_embeddings)

    logger.log(f"Loading tree from {args.tree_file}")
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
    logger.log(
        f"Scoring pair sets with {len(reference_labels)} reference species and "
        f"{len(query_labels)} zero-shot species"
    )

    all_rows = []
    metrics = {}
    for set_name, pairs in pair_sets.items():
        logger.log(f"Scoring {set_name}: {len(pairs)} sampled pairs")
        rows = []
        for label_a, label_b in pairs:
            emb_a = query_by_species.get(label_a, reference_by_species.get(label_a))
            emb_b = query_by_species.get(label_b, reference_by_species.get(label_b))
            if emb_a is None or emb_b is None:
                continue
            row = {
                "pair_set": set_name,
                "label_a": label_a,
                "label_b": label_b,
                "tree_distance": tree_distance(label_a, label_b, taxon_nodes, depths, ancestor_lists, ancestor_sets),
                "embedding_cosine_distance": cosine_distance(emb_a, emb_b),
            }
            rows.append(row)
            all_rows.append(row)
        metrics[set_name] = correlation(rows, "embedding_cosine_distance")

    pair_path = args.output_dir / "sampled_tree_recovery_pairs.csv"
    logger.log(f"Writing sampled pair distances to {pair_path}")
    pd.DataFrame(all_rows).to_csv(pair_path, index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": run_manifest.get("model"),
        "model_config": run_manifest.get("model_config"),
        "input_dir": str(args.input_dir),
        "checkpoint": str(args.checkpoint),
        "run_manifest": str(args.run_manifest),
        "output_dir": str(args.output_dir),
        "tree_file": str(args.tree_file),
        "max_seq_len": args.max_seq_len,
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
    logger.log(f"Writing metrics to {out_path}")
    logger.done(Path(__file__).name)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
