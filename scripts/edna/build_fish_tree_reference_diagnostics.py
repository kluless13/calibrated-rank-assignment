#!/usr/bin/env python3
"""Build nearest-reference and rank-coverage diagnostics for fish-tree queries."""
from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
RANKS = ["genus", "family", "order"]


def canonical_tree_label(label: object) -> str:
    return clean(label).replace(" ", "_")


def clean(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def load_tree_distances(tree_file: Path) -> tuple[dict[str, object], dict[object, float], dict[str, list[object]], dict[str, set[object]]]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick")
    taxon_nodes: dict[str, object] = {}
    for node in tree.leaf_node_iter():
        if node.taxon is None or not node.taxon.label:
            continue
        raw_label = clean(node.taxon.label)
        taxon_nodes[raw_label] = node
        taxon_nodes[canonical_tree_label(raw_label)] = node
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


def tree_distance(
    a: str,
    b: str,
    taxon_nodes: dict[str, object],
    depths: dict[object, float],
    ancestor_lists: dict[str, list[object]],
    ancestor_sets: dict[str, set[object]],
) -> float:
    if a not in taxon_nodes or b not in taxon_nodes:
        return float("nan")
    for node in ancestor_lists[a]:
        if node in ancestor_sets[b]:
            return float(depths[taxon_nodes[a]] + depths[taxon_nodes[b]] - 2 * depths[node])
    return float("nan")


def load_reference_labels(input_dir: Path) -> set[str]:
    raw = json.loads((input_dir / "train_species_sequences.json").read_text())
    return {canonical_tree_label(label) for label, seqs in raw.items() if seqs}


def parse_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [clean(item) for item in value]
    text = clean(value)
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            return [clean(item) for item in parsed]
    return []


def add_topk_rank_hits(merged: pd.DataFrame, candidate_tax: dict[str, dict[str, object]]) -> pd.DataFrame:
    if "top_tree_labels" not in merged.columns:
        return merged
    out = merged.copy()
    for rank in ["species", "genus", "family", "order"]:
        for k in [1, 5, 10]:
            col = f"{rank}_top{k}"
            if col not in out.columns:
                out[col] = False
    rank_true_cols = {
        "species": "true_tree_label",
        "genus": "true_genus_name",
        "family": "true_family_name",
        "order": "true_order_name",
    }
    for idx, row in out.iterrows():
        top_labels = [canonical_tree_label(label) for label in parse_list(row.get("top_tree_labels", ""))]
        for rank, true_col in rank_true_cols.items():
            target = canonical_tree_label(row.get(true_col, "")) if rank == "species" else clean(row.get(true_col, ""))
            if not target:
                continue
            for k in [1, 5, 10]:
                top_k = top_labels[:k]
                if rank == "species":
                    hit = target in top_k
                else:
                    hit = any(clean(candidate_tax.get(label, {}).get(f"{rank}_name", "")) == target for label in top_k)
                out.at[idx, f"{rank}_top{k}"] = bool(hit)
    return out


def summarize_with_predictions(
    diag: pd.DataFrame,
    per_query_csv: Path,
    output_dir: Path,
    name: str,
    candidate_tax: dict[str, dict[str, object]],
) -> dict[str, object]:
    pred = pd.read_csv(per_query_csv)
    merged = diag.merge(pred, on="processid", how="inner")
    merged = add_topk_rank_hits(merged, candidate_tax)
    merged["nearest_reference_bin"] = "missing"
    finite_mask = np.isfinite(merged["nearest_reference_tree_distance"].astype(float))
    finite_distances = merged.loc[finite_mask, "nearest_reference_tree_distance"].astype(float)
    if len(finite_distances) > 0:
        n_unique = int(finite_distances.nunique(dropna=True))
        if n_unique <= 1:
            merged.loc[finite_mask, "nearest_reference_bin"] = "all_finite"
        else:
            n_bins = min(5, n_unique)
            bins = pd.qcut(
                finite_distances.rank(method="first"),
                q=n_bins,
                duplicates="drop",
            )
            merged.loc[finite_mask, "nearest_reference_bin"] = bins.astype(str).to_numpy()
    rows = []
    for bin_name, sub in merged.groupby("nearest_reference_bin", dropna=False):
        row = {
            "prediction_set": name,
            "nearest_reference_bin": str(bin_name),
            "n_query": int(len(sub)),
            "nearest_reference_tree_distance_min": finite_or_nan(sub["nearest_reference_tree_distance"].min()),
            "nearest_reference_tree_distance_median": finite_or_nan(sub["nearest_reference_tree_distance"].median()),
            "nearest_reference_tree_distance_max": finite_or_nan(sub["nearest_reference_tree_distance"].max()),
        }
        for rank in ["species", "genus", "family", "order"]:
            for k in [1, 5, 10]:
                col = f"{rank}_top{k}"
                if col in sub.columns:
                    row[col] = float(sub[col].astype(bool).mean())
        rows.append(row)
    out_csv = output_dir / f"{name}_nearest_reference_bins.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return {"name": name, "per_query_csv": str(per_query_csv), "bin_summary_csv": str(out_csv)}


def finite_or_nan(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument(
        "--prediction-per-query",
        action="append",
        default=[],
        help="Optional NAME=path/to/zero_shot_candidate_per_query.csv for bin summaries.",
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading queries from {args.input_dir / 'zero_shot_queries.csv'}")
    queries = pd.read_csv(args.input_dir / "zero_shot_queries.csv")
    logger.log(f"Loading candidates from {args.input_dir / 'candidate_species.csv'}")
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    logger.log(f"Loading reference labels from {args.input_dir / 'train_species_sequences.json'}")
    reference_labels = load_reference_labels(args.input_dir)
    logger.log(f"Loading tree distances from {args.tree_file}")
    taxon_nodes, depths, ancestor_lists, ancestor_sets = load_tree_distances(args.tree_file)

    candidate_tax = candidates.set_index("tree_label").to_dict(orient="index")
    ref_by_rank = {rank: set() for rank in RANKS}
    for label in reference_labels:
        row = candidate_tax.get(label, {})
        for rank in RANKS:
            value = clean(row.get(f"{rank}_name", ""))
            if value:
                ref_by_rank[rank].add(value)

    rows = []
    ref_list = sorted(reference_labels)
    logger.log(f"Computing nearest-reference diagnostics for {len(queries)} queries and {len(ref_list)} reference species")
    for _, query in queries.iterrows():
        true_label = canonical_tree_label(query.get("tree_label", ""))
        nearest_label = ""
        nearest_distance = float("nan")
        if true_label in taxon_nodes:
            best = []
            for ref_label in ref_list:
                dist = tree_distance(true_label, ref_label, taxon_nodes, depths, ancestor_lists, ancestor_sets)
                if np.isfinite(dist):
                    best.append((dist, ref_label))
            if best:
                nearest_distance, nearest_label = min(best)
        row = {
            "processid": clean(query.get("processid", "")),
            "true_tree_label": true_label,
            "true_species_name": clean(query.get("species_name", "")),
            "true_genus_name": clean(query.get("genus_name", "")),
            "true_family_name": clean(query.get("family_name", "")),
            "true_order_name": clean(query.get("order_name", "")),
            "true_species_has_reference": true_label in reference_labels,
            "nearest_reference_tree_label": nearest_label,
            "nearest_reference_tree_distance": float(nearest_distance),
        }
        for rank in RANKS:
            value = clean(query.get(f"{rank}_name", ""))
            row[f"true_{rank}_represented_in_reference"] = bool(value and value in ref_by_rank[rank])
        rows.append(row)

    diag = pd.DataFrame(rows)
    diag_path = args.output_dir / "reference_diagnostics_per_query.csv"
    logger.log(f"Writing per-query diagnostics to {diag_path}")
    diag.to_csv(diag_path, index=False)

    summaries = []
    for item in args.prediction_per_query:
        if "=" not in item:
            raise SystemExit("--prediction-per-query must be NAME=path")
        name, path_text = item.split("=", 1)
        path = Path(path_text)
        if path.exists():
            logger.log(f"Building nearest-reference bin summary for {name}: {path}")
            summaries.append(summarize_with_predictions(diag, path, args.output_dir, name, candidate_tax))
        else:
            logger.log(f"Skipping missing prediction summary input for {name}: {path}")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "tree_file": str(args.tree_file),
        "query_count": int(len(diag)),
        "reference_species_count": int(len(reference_labels)),
        "per_query_csv": str(diag_path),
        "prediction_summaries": summaries,
    }
    manifest_path = args.output_dir / "reference_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
