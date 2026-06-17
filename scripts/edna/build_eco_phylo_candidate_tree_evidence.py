#!/usr/bin/env python3
"""Build inference-safe tree-neighborhood evidence for Eco-Phylo candidates.

This script does not use the true query label as a feature. It summarizes the
candidate set returned for each method/sample/query:

- candidate distance to the top-ranked candidate on the reference tree;
- candidate distance to the retrieved candidate neighborhood;
- whether candidates agree with the top candidate at genus/family/order;
- top-k taxonomic and tree-distance spread.

The output is a separate join table keyed by method/sample/query/candidate so
the large candidate feature table does not need to be rewritten.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dendropy
import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSTERIOR_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "eco_phylo_posterior"
    / "candidate_level"
)
DEFAULT_CANDIDATE_FEATURES = DEFAULT_POSTERIOR_DIR / "eco_phylo_candidate_features_top5.csv.gz"
DEFAULT_TREE_FILE = ROOT / "data" / "phylo" / "actinopt_12k_treePL.tre"
KEY_COLS = ["method", "sample_id", "query_processid", "candidate_tree_label"]
GROUP_COLS = ["method", "sample_id", "query_processid"]
RANKS = ("genus", "family", "order")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def canonical_tree_label(value: object) -> str:
    return clean(value).replace(" ", "_")


def load_tree_distances(tree_file: Path) -> tuple[
    dict[str, object],
    dict[object, float],
    dict[str, list[object]],
    dict[str, set[object]],
]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick")
    taxon_nodes: dict[str, object] = {}
    for node in tree.leaf_node_iter():
        if node.taxon is None or not node.taxon.label:
            continue
        label = canonical_tree_label(node.taxon.label)
        taxon_nodes[label] = node
    depths = {tree.seed_node: 0.0}
    stack = [tree.seed_node]
    while stack:
        node = stack.pop()
        for child in node.child_node_iter():
            depths[child] = depths[node] + float(child.edge_length or 0.0)
            stack.append(child)

    def ancestors(node: object) -> list[object]:
        out = []
        current = node
        while current is not None:
            out.append(current)
            current = current.parent_node
        return out

    ancestor_lists = {label: ancestors(node) for label, node in taxon_nodes.items()}
    ancestor_sets = {label: set(nodes) for label, nodes in ancestor_lists.items()}
    return taxon_nodes, depths, ancestor_lists, ancestor_sets


class TreeDistanceCache:
    def __init__(
        self,
        taxon_nodes: dict[str, object],
        depths: dict[object, float],
        ancestor_lists: dict[str, list[object]],
        ancestor_sets: dict[str, set[object]],
    ) -> None:
        self.taxon_nodes = taxon_nodes
        self.depths = depths
        self.ancestor_lists = ancestor_lists
        self.ancestor_sets = ancestor_sets
        self.cache: dict[tuple[str, str], float] = {}

    def distance(self, a: object, b: object) -> float:
        label_a = canonical_tree_label(a)
        label_b = canonical_tree_label(b)
        if not label_a or not label_b or label_a not in self.taxon_nodes or label_b not in self.taxon_nodes:
            return float("nan")
        if label_a == label_b:
            return 0.0
        key = tuple(sorted((label_a, label_b)))
        if key in self.cache:
            return self.cache[key]
        for node in self.ancestor_lists[label_a]:
            if node in self.ancestor_sets[label_b]:
                dist = float(
                    self.depths[self.taxon_nodes[label_a]]
                    + self.depths[self.taxon_nodes[label_b]]
                    - 2 * self.depths[node]
                )
                self.cache[key] = dist
                return dist
        self.cache[key] = float("nan")
        return float("nan")


def group_key(frame: pd.DataFrame) -> pd.Series:
    return frame["sample_id"].astype(str) + "\t" + frame["query_processid"].astype(str)


def read_candidate_rows(
    path: Path,
    max_query_groups: int,
    random_state: int,
    chunksize: int,
    logger: ProgressLogger,
) -> pd.DataFrame:
    usecols = [
        "method",
        "sample_id",
        "query_processid",
        "candidate_tree_label",
        "candidate_rank",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
    ]
    if max_query_groups <= 0:
        logger.log(f"Reading all candidate rows from {rel(path)}")
        return pd.read_csv(path, usecols=usecols, low_memory=False)

    logger.log(f"Selecting complete sample/query groups: max_query_groups={max_query_groups:,}")
    group_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=["sample_id", "query_processid"], chunksize=chunksize):
        group_frames.append(chunk.drop_duplicates())
    groups = pd.concat(group_frames, ignore_index=True).drop_duplicates()
    if len(groups) > max_query_groups:
        groups = groups.sample(n=max_query_groups, random_state=random_state)
    selected = set(group_key(groups))
    logger.log(f"Selected {len(selected):,} sample/query groups")

    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        keep = group_key(chunk).isin(selected)
        if keep.any():
            frames.append(chunk.loc[keep].copy())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)
    logger.log(f"Read {len(out):,} candidate rows after filtering")
    return out


def finite_stats(values: list[float]) -> tuple[float, float, float]:
    finite = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(finite.min()), float(finite.mean()), float(finite.max())


def build_rows(frame: pd.DataFrame, distances: TreeDistanceCache, logger: ProgressLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = frame.copy()
    frame["candidate_rank_num"] = pd.to_numeric(frame["candidate_rank"], errors="coerce").fillna(999_999)
    frame["candidate_tree_label"] = frame["candidate_tree_label"].map(canonical_tree_label)
    frame = frame.sort_values(GROUP_COLS + ["candidate_rank_num", "candidate_tree_label"])
    groups = frame.groupby(GROUP_COLS, sort=False, dropna=False)
    logger.log(f"Building tree-neighborhood evidence for {len(groups):,} method/sample/query groups")

    for idx, (group_key_values, group) in enumerate(groups, start=1):
        group = group.drop_duplicates("candidate_tree_label", keep="first").copy()
        labels = group["candidate_tree_label"].tolist()
        top_label = labels[0] if labels else ""
        group_pair_distances: list[float] = []
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                group_pair_distances.append(distances.distance(left, right))
        group_min, group_mean, group_max = finite_stats(group_pair_distances)

        unique_counts = {
            f"topk_unique_{rank}_count": int(group[f"candidate_{rank}"].map(clean).replace("", np.nan).nunique(dropna=True))
            for rank in RANKS
        }
        top_tax = {
            rank: clean(group.iloc[0].get(f"candidate_{rank}", ""))
            for rank in RANKS
        }
        for row in group.itertuples(index=False):
            candidate = canonical_tree_label(getattr(row, "candidate_tree_label"))
            candidate_distances = [
                distances.distance(candidate, other)
                for other in labels
                if other != candidate
            ]
            candidate_min, candidate_mean, candidate_max = finite_stats(candidate_distances)
            top_distance = distances.distance(candidate, top_label)
            out: dict[str, Any] = {
                "method": getattr(row, "method"),
                "sample_id": getattr(row, "sample_id"),
                "query_processid": getattr(row, "query_processid"),
                "candidate_tree_label": candidate,
                "tree_evidence_available": int(np.isfinite(top_distance) or np.isfinite(candidate_mean)),
                "candidate_is_top1": int(candidate == top_label),
                "tree_distance_to_top1_candidate": top_distance,
                "tree_distance_to_candidate_set_min": candidate_min,
                "tree_distance_to_candidate_set_mean": candidate_mean,
                "tree_distance_to_candidate_set_max": candidate_max,
                "topk_pairwise_tree_distance_min": group_min,
                "topk_pairwise_tree_distance_mean": group_mean,
                "topk_pairwise_tree_distance_max": group_max,
                **unique_counts,
            }
            for rank in RANKS:
                candidate_value = clean(getattr(row, f"candidate_{rank}"))
                top_value = top_tax[rank]
                values = [clean(value) for value in group[f"candidate_{rank}"].tolist()]
                comparable = [value for value in values if value]
                out[f"same_{rank}_as_top1"] = int(bool(candidate_value and top_value and candidate_value == top_value))
                out[f"same_{rank}_topk_fraction"] = (
                    float(sum(value == candidate_value for value in comparable) / len(comparable))
                    if candidate_value and comparable
                    else float("nan")
                )
            rows.append(out)
        if idx % 100_000 == 0:
            logger.log(f"Processed {idx:,}/{len(groups):,} groups; distance cache={len(distances.cache):,} pairs")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        *KEY_COLS,
        "tree_evidence_available",
        "candidate_is_top1",
        "tree_distance_to_top1_candidate",
        "tree_distance_to_candidate_set_min",
        "tree_distance_to_candidate_set_mean",
        "tree_distance_to_candidate_set_max",
        "topk_pairwise_tree_distance_min",
        "topk_pairwise_tree_distance_mean",
        "topk_pairwise_tree_distance_max",
        "topk_unique_genus_count",
        "topk_unique_family_count",
        "topk_unique_order_count",
        "same_genus_as_top1",
        "same_genus_topk_fraction",
        "same_family_as_top1",
        "same_family_topk_fraction",
        "same_order_as_top1",
        "same_order_topk_fraction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-features", type=Path, default=DEFAULT_CANDIDATE_FEATURES)
    parser.add_argument("--tree-file", type=Path, default=DEFAULT_TREE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_POSTERIOR_DIR)
    parser.add_argument("--output-name", default="eco_phylo_candidate_tree_neighborhood_evidence.csv")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-query-groups", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=1206)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = read_candidate_rows(
        path=args.candidate_features,
        max_query_groups=args.max_query_groups,
        random_state=args.random_state,
        chunksize=args.chunksize,
        logger=logger,
    )
    logger.log(f"Loading fish tree from {rel(args.tree_file)}")
    taxon_nodes, depths, ancestor_lists, ancestor_sets = load_tree_distances(args.tree_file)
    distance_cache = TreeDistanceCache(taxon_nodes, depths, ancestor_lists, ancestor_sets)
    rows = build_rows(frame, distance_cache, logger)

    output_path = args.output_dir / args.output_name
    logger.log(f"Writing tree-neighborhood evidence to {rel(output_path)}")
    write_csv(output_path, rows)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_features": rel(args.candidate_features),
        "tree_file": rel(args.tree_file),
        "output_csv": rel(output_path),
        "input_candidate_rows": int(len(frame)),
        "output_rows": int(len(rows)),
        "max_query_groups": int(args.max_query_groups),
        "random_state": int(args.random_state),
        "distance_cache_pairs": int(len(distance_cache.cache)),
        "notes": [
            "Inference-safe: does not use the true query taxon as a feature.",
            "Tree distances are between retrieved candidate labels, not query truth and candidate.",
        ],
    }
    manifest_name = output_path.name
    if manifest_name.endswith(".csv.gz"):
        manifest_name = manifest_name[: -len(".csv.gz")] + "_manifest.json"
    elif manifest_name.endswith(".csv"):
        manifest_name = manifest_name[: -len(".csv")] + "_manifest.json"
    else:
        manifest_name = manifest_name + "_manifest.json"
    manifest_path = args.output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {rel(manifest_path)}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
