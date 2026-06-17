#!/usr/bin/env python3
"""Score EPA-ng/pplacer jplace outputs for Paper 1 placement diagnostics.

This first scorer intentionally reports conservative placement-derived rank
signals rather than claiming a full Fernando-style PCP score. It maps the top
jplace edge to the descendant reference clade and asks whether that clade
contains the query's true species/genus/family/order. This gives us an immediate
shared table for calibration and rank-backoff while a stricter PCP adapter is
developed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import dendropy
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


RANKS = ("species", "genus", "family", "order")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TREE_FILE = Path("data/phylo/actinopt_12k_treePL.tre")


@dataclass(frozen=True)
class PlacementRun:
    split: str
    method: str
    jplace_path: Path
    split_root: Path


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def canonical_label(value: object) -> str:
    return clean(value).replace(" ", "_")


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def jplace_tree_to_dendropy(tree_text: str) -> dendropy.Tree:
    # jplace stores edge numbers as Newick curly-brace annotations, e.g.
    # A:0.1{17}. DendroPy cannot read that directly, so convert them to BEAST-
    # style comments that DendroPy can expose as annotations.
    converted = re.sub(r"\{(\d+)\}", r"[&edge_num=\1]", tree_text)
    return dendropy.Tree.get(
        data=converted,
        schema="newick",
        preserve_underscores=True,
        extract_comment_metadata=True,
    )


def edge_descendant_labels(tree_text: str) -> dict[int, set[str]]:
    tree = jplace_tree_to_dendropy(tree_text)
    out: dict[int, set[str]] = {}
    for node in tree.postorder_node_iter():
        edge_num = None
        for annotation in node.annotations:
            if annotation.name == "edge_num":
                edge_num = int(annotation.value)
                break
        if edge_num is None:
            continue
        labels = {
            canonical_label(leaf.taxon.label)
            for leaf in node.leaf_iter()
            if leaf.taxon is not None and leaf.taxon.label
        }
        out[edge_num] = labels
    return out


def load_tree_distance_index(
    tree_file: Path,
) -> tuple[dict[str, object], dict[object, float], dict[str, list[object]], dict[str, set[object]]]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick", preserve_underscores=True)
    taxon_nodes: dict[str, object] = {}
    for node in tree.leaf_node_iter():
        if node.taxon is None or not node.taxon.label:
            continue
        label = canonical_label(node.taxon.label)
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


def min_tree_distance_to_labels(
    true_label: str,
    candidate_labels: Iterable[str],
    taxon_nodes: dict[str, object],
    depths: dict[object, float],
    ancestor_lists: dict[str, list[object]],
    ancestor_sets: dict[str, set[object]],
) -> tuple[float, str]:
    best_distance = float("nan")
    best_label = ""
    for candidate in candidate_labels:
        distance = tree_distance(true_label, candidate, taxon_nodes, depths, ancestor_lists, ancestor_sets)
        if not math.isfinite(distance):
            continue
        if not math.isfinite(best_distance) or distance < best_distance:
            best_distance = distance
            best_label = candidate
    return best_distance, best_label


def load_candidate_taxonomy(split_root: Path) -> dict[str, dict[str, str]]:
    candidate_path = split_root / "candidate_species.csv"
    if not candidate_path.exists():
        manifest_path = split_root / "placement_input_manifest.json"
        if manifest_path.exists():
            input_dir = read_json(manifest_path).get("input_dir")
            if input_dir:
                candidate_path = ROOT / str(input_dir) / "candidate_species.csv"
    if not candidate_path.exists():
        return {}
    candidates = pd.read_csv(candidate_path)
    out: dict[str, dict[str, str]] = {}
    for _, row in candidates.iterrows():
        label = canonical_label(row.get("tree_label", ""))
        if not label:
            continue
        out[label] = {
            "species": label,
            "genus": clean(row.get("genus_name", "")) or clean(row.get("genus_from_label", "")),
            "family": clean(row.get("family_name", "")),
            "order": clean(row.get("order_name", "")),
        }
    return out


def load_query_taxonomy(split_root: Path) -> dict[str, dict[str, str]]:
    query_path = split_root / "query_manifest.csv"
    if not query_path.exists():
        query_path = split_root / "zero_shot_queries.csv"
    queries = pd.read_csv(query_path)
    out: dict[str, dict[str, str]] = {}
    for _, row in queries.iterrows():
        aliases = [
            clean(row.get("query_id", "")),
            clean(row.get("processid", "")),
        ]
        aliases = [alias for alias in aliases if alias]
        if not aliases:
            fallback_col = queries.columns[0]
            fallback = clean(row.get(fallback_col, ""))
            aliases = [fallback] if fallback else []
        if not aliases:
            continue
        true_label = canonical_label(row.get("tree_label", row.get("true_tree_label", "")))
        record = {
            "processid": aliases[0],
            "species": true_label,
            "genus": clean(row.get("genus_name", row.get("true_genus_name", ""))),
            "family": clean(row.get("family_name", row.get("true_family_name", ""))),
            "order": clean(row.get("order_name", row.get("true_order_name", ""))),
        }
        for alias in aliases:
            out[alias] = record
    return out


def placement_names(placement: dict) -> list[str]:
    names = placement.get("n") or placement.get("nm") or []
    out: list[str] = []
    for item in names:
        if isinstance(item, str):
            out.append(clean(item))
        elif isinstance(item, list) and item:
            out.append(clean(item[0]))
    return [name for name in out if name]


def top_placement(placement: dict, fields: list[str]) -> dict[str, float | int | None]:
    placements = placement.get("p") or []
    if not placements:
        return {"edge_num": None, "lwr": None, "like_weight_ratio": None}
    try:
        edge_idx = fields.index("edge_num")
    except ValueError:
        edge_idx = 0
    lwr_idx = None
    for candidate in ("like_weight_ratio", "lwr"):
        if candidate in fields:
            lwr_idx = fields.index(candidate)
            break
    if lwr_idx is None:
        # Most jplace variants put likelihood weight ratio in the third field,
        # but only use this fallback if the placement is long enough.
        lwr_idx = 2 if len(placements[0]) > 2 else None

    def lwr_value(row: list[object]) -> float:
        if lwr_idx is None or lwr_idx >= len(row):
            return 0.0
        try:
            return float(row[lwr_idx])
        except (TypeError, ValueError):
            return 0.0

    best = max(placements, key=lwr_value)
    edge_num = int(best[edge_idx]) if edge_idx < len(best) else None
    lwr = lwr_value(best)
    return {"edge_num": edge_num, "lwr": lwr, "like_weight_ratio": lwr}


def most_specific_rank(row: dict[str, object]) -> str:
    for rank in RANKS:
        if bool(row.get(f"{rank}_in_placed_clade", False)):
            return rank
    return "none"


def broadest_rank(row: dict[str, object]) -> str:
    for rank in reversed(RANKS):
        if bool(row.get(f"{rank}_in_placed_clade", False)):
            return rank
    return "none"


def lwr_bucket(value: object) -> str:
    try:
        lwr = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(lwr):
        return "missing"
    if lwr >= 0.99:
        return "0.99-1.00"
    if lwr >= 0.90:
        return "0.90-0.99"
    if lwr >= 0.70:
        return "0.70-0.90"
    if lwr >= 0.50:
        return "0.50-0.70"
    return "0.00-0.50"


def clade_size_bucket(value: object) -> str:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return "missing"
    if size <= 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 25:
        return "6-25"
    if size <= 100:
        return "26-100"
    return "101+"


def score_run(
    run: PlacementRun,
    tree_file: Path,
    taxon_nodes: dict[str, object],
    depths: dict[object, float],
    ancestor_lists: dict[str, list[object]],
    ancestor_sets: dict[str, set[object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = read_json(run.jplace_path)
    fields = list(data.get("fields", []))
    edge_to_labels = edge_descendant_labels(str(data["tree"]))
    candidate_tax = load_candidate_taxonomy(run.split_root)
    query_tax = load_query_taxonomy(run.split_root)
    reference_labels = sorted({label for labels in edge_to_labels.values() for label in labels})
    nearest_reference_cache: dict[str, tuple[float, str]] = {}
    placed_clade_distance_cache: dict[tuple[str, int], tuple[float, str]] = {}

    rows: list[dict[str, object]] = []
    missing_queries = 0
    for placement in data.get("placements", []):
        names = placement_names(placement)
        if not names:
            continue
        top = top_placement(placement, fields)
        edge_num = top["edge_num"]
        descendant_labels = edge_to_labels.get(int(edge_num), set()) if edge_num is not None else set()
        descendant_tax = [candidate_tax.get(label, {}) for label in descendant_labels]
        descendant_values = {
            "species": set(descendant_labels),
            "genus": {clean(row.get("genus", "")) for row in descendant_tax if clean(row.get("genus", ""))},
            "family": {clean(row.get("family", "")) for row in descendant_tax if clean(row.get("family", ""))},
            "order": {clean(row.get("order", "")) for row in descendant_tax if clean(row.get("order", ""))},
        }
        for name in names:
            truth = query_tax.get(name)
            if truth is None:
                missing_queries += 1
                truth = {"species": "", "genus": "", "family": "", "order": ""}
            true_species = canonical_label(truth.get("species", ""))
            nearest_reference_distance = float("nan")
            nearest_reference_label = ""
            placed_min_distance = float("nan")
            placed_nearest_label = ""
            if true_species:
                if true_species not in nearest_reference_cache:
                    nearest_reference_cache[true_species] = min_tree_distance_to_labels(
                        true_species,
                        reference_labels,
                        taxon_nodes,
                        depths,
                        ancestor_lists,
                        ancestor_sets,
                    )
                nearest_reference_distance, nearest_reference_label = nearest_reference_cache[true_species]
                if edge_num is not None:
                    cache_key = (true_species, int(edge_num))
                    if cache_key not in placed_clade_distance_cache:
                        placed_clade_distance_cache[cache_key] = min_tree_distance_to_labels(
                            true_species,
                            descendant_labels,
                            taxon_nodes,
                            depths,
                            ancestor_lists,
                            ancestor_sets,
                        )
                    placed_min_distance, placed_nearest_label = placed_clade_distance_cache[cache_key]
            excess_tree_distance = (
                placed_min_distance - nearest_reference_distance
                if math.isfinite(placed_min_distance) and math.isfinite(nearest_reference_distance)
                else float("nan")
            )
            row: dict[str, object] = {
                "split": run.split,
                "method": run.method,
                "processid": name,
                "top_edge_num": edge_num,
                "top_lwr": top["lwr"],
                "top_lwr_bucket": lwr_bucket(top["lwr"]),
                "descendant_leaf_count": len(descendant_labels),
                "descendant_leaf_count_bucket": clade_size_bucket(len(descendant_labels)),
                "nearest_reference_tree_label": nearest_reference_label,
                "nearest_reference_tree_distance": nearest_reference_distance,
                "nearest_placed_clade_tree_label": placed_nearest_label,
                "placement_min_tree_distance_to_placed_clade": placed_min_distance,
                "placement_excess_tree_distance_vs_nearest_reference": excess_tree_distance,
                "jplace_file": str(run.jplace_path),
            }
            for rank in RANKS:
                target = truth.get(rank, "")
                hit = bool(target and target in descendant_values[rank])
                row[f"{rank}_in_placed_clade"] = hit
            row["most_specific_placed_rank"] = most_specific_rank(row)
            row["broadest_placed_rank"] = broadest_rank(row)
            # Backward-compatible name, corrected to mean the deepest taxonomic
            # claim supported by the placed clade: species before genus/family/order.
            row["deepest_placed_rank"] = row["most_specific_placed_rank"]
            rows.append(row)

    summary = {
        "split": run.split,
        "method": run.method,
        "jplace_file": str(run.jplace_path),
        "n_queries_scored": len(rows),
        "missing_query_metadata": missing_queries,
        "n_edges_with_descendants": len(edge_to_labels),
        "tree_file": str(tree_file),
    }
    for rank in RANKS:
        summary[f"{rank}_in_placed_clade_rate"] = (
            sum(bool(row[f"{rank}_in_placed_clade"]) for row in rows) / len(rows) if rows else None
        )
    finite_distances = [
        float(row["placement_min_tree_distance_to_placed_clade"])
        for row in rows
        if math.isfinite(float(row["placement_min_tree_distance_to_placed_clade"]))
    ]
    finite_excess = [
        float(row["placement_excess_tree_distance_vs_nearest_reference"])
        for row in rows
        if math.isfinite(float(row["placement_excess_tree_distance_vs_nearest_reference"]))
    ]
    summary["placement_min_tree_distance_median"] = float(pd.Series(finite_distances).median()) if finite_distances else None
    summary["placement_min_tree_distance_mean"] = float(pd.Series(finite_distances).mean()) if finite_distances else None
    summary["placement_excess_tree_distance_median"] = float(pd.Series(finite_excess).median()) if finite_excess else None
    summary["placement_excess_tree_distance_mean"] = float(pd.Series(finite_excess).mean()) if finite_excess else None
    for rank in list(RANKS) + ["none"]:
        summary[f"most_specific_rank_{rank}_rate"] = (
            sum(row["most_specific_placed_rank"] == rank for row in rows) / len(rows) if rows else None
        )
    for threshold in (0.5, 0.7, 0.9):
        high = [row for row in rows if row.get("top_lwr") is not None and float(row["top_lwr"]) >= threshold]
        summary[f"lwr_ge_{threshold}_n"] = len(high)
        for rank in RANKS:
            summary[f"lwr_ge_{threshold}_{rank}_rate"] = (
                sum(bool(row[f"{rank}_in_placed_clade"]) for row in high) / len(high) if high else None
            )
    return rows, summary


def finite_median(values: Iterable[object]) -> float | None:
    finite = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    return float(pd.Series(finite).median()) if finite else None


def summarize_groups(rows: list[dict[str, object]], group_fields: list[str]) -> list[dict[str, object]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[dict[str, object]] = []
    for keys, sub in df.groupby(group_fields, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {field: key for field, key in zip(group_fields, keys)}
        row["n_queries"] = int(len(sub))
        row["top_lwr_median"] = finite_median(sub["top_lwr"])
        row["descendant_leaf_count_median"] = finite_median(sub["descendant_leaf_count"])
        row["placement_min_tree_distance_median"] = finite_median(sub["placement_min_tree_distance_to_placed_clade"])
        row["placement_excess_tree_distance_median"] = finite_median(sub["placement_excess_tree_distance_vs_nearest_reference"])
        for rank in RANKS:
            row[f"{rank}_in_placed_clade_rate"] = float(sub[f"{rank}_in_placed_clade"].astype(bool).mean())
        for rank in list(RANKS) + ["none"]:
            row[f"most_specific_rank_{rank}_rate"] = float((sub["most_specific_placed_rank"] == rank).mean())
        out.append(row)
    return out


def discover_runs(root: Path) -> list[PlacementRun]:
    runs: list[PlacementRun] = []
    for split_root in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "logs"):
        split = split_root.name
        candidates = [
            ("epa_ng", split_root / "epa_ng" / "epa_result.jplace"),
            ("pplacer", split_root / "pplacer" / "pplacer.jplace"),
            ("apples", split_root / "apples" / "apples.jplace"),
        ]
        for method, path in candidates:
            if path.exists():
                runs.append(PlacementRun(split=split, method=method, jplace_path=path, split_root=split_root))
    return runs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--placement-root",
        type=Path,
        default=Path("results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/phylo_placement"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
    )
    parser.add_argument("--tree-file", type=Path, default=DEFAULT_TREE_FILE)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    logger.log(f"Loading full tree for placement-distance diagnostics from {args.tree_file}")
    taxon_nodes, depths, ancestor_lists, ancestor_sets = load_tree_distance_index(args.tree_file)
    logger.log(f"Loaded {len(taxon_nodes)} tree tips")
    logger.log(f"Discovering jplace runs under {args.placement_root}")
    runs = discover_runs(args.placement_root) if args.placement_root.exists() else []
    logger.log(f"Discovered {len(runs)} placement run(s)")
    per_query_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for run in runs:
        logger.log(f"Scoring {run.method} {run.split}: {run.jplace_path}")
        rows, summary = score_run(run, args.tree_file, taxon_nodes, depths, ancestor_lists, ancestor_sets)
        per_query_rows.extend(rows)
        summary_rows.append(summary)
        logger.log(
            f"Scored {summary['n_queries_scored']} queries for {run.method} {run.split}; "
            f"missing metadata={summary['missing_query_metadata']}"
        )

    per_query_path = args.output_dir / "placement_rank_diagnostics_per_query.csv"
    summary_path = args.output_dir / "placement_rank_diagnostics_summary.csv"
    logger.log(f"Writing {per_query_path}")
    write_csv(per_query_path, per_query_rows)
    logger.log(f"Writing {summary_path}")
    write_csv(summary_path, summary_rows)

    lwr_summary_path = args.output_dir / "placement_lwr_rank_summary.csv"
    lwr_rows = summarize_groups(per_query_rows, ["split", "method", "top_lwr_bucket"])
    logger.log(f"Writing {lwr_summary_path}")
    write_csv(lwr_summary_path, lwr_rows)

    clade_size_summary_path = args.output_dir / "placement_clade_size_rank_summary.csv"
    clade_rows = summarize_groups(per_query_rows, ["split", "method", "descendant_leaf_count_bucket"])
    logger.log(f"Writing {clade_size_summary_path}")
    write_csv(clade_size_summary_path, clade_rows)

    rank_backoff_path = args.output_dir / "placement_rank_backoff_summary.csv"
    rank_rows = summarize_groups(per_query_rows, ["split", "method", "most_specific_placed_rank"])
    logger.log(f"Writing {rank_backoff_path}")
    write_csv(rank_backoff_path, rank_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "placement_root": str(args.placement_root),
        "output_dir": str(args.output_dir),
        "tree_file": str(args.tree_file),
        "runs_scored": len(runs),
        "per_query_rows": len(per_query_rows),
        "summary_rows": len(summary_rows),
        "lwr_summary_rows": len(lwr_rows),
        "clade_size_summary_rows": len(clade_rows),
        "rank_backoff_rows": len(rank_rows),
        "caution": "Placed-clade rank containment and tree-distance-to-placed-clade are Fernando-adjacent diagnostics, not full Fernando PCP.",
    }
    manifest_path = args.output_dir / "placement_rank_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
