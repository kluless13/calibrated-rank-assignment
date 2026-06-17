#!/usr/bin/env python3
"""Build Fernando-like sister-clade placement diagnostics for Paper 1.

Fernando et al. score percentage of correct placement (PCP) by comparing the
sister clade of a placed species in the placement tree against the sister clade
in the full reference tree. Our current clean splits and jplace adapter are not
an exact reproduction of their backbone-completeness experiment, so this script
emits a clearly labelled PCP-like diagnostic:

1. For each query, find the nearest non-empty sister support set in the full
   fish tree after restricting to the reference taxa present in that split.
2. For the top jplace edge, compare both sides of the placed edge against that
   expected sister support set.
3. Report descendant-side exact sister-set matches, any-overlap matches,
   Jaccard overlap, and high-LWR subsets. The complement side is kept only as
   an orientation diagnostic because treating the whole complementary tree as a
   match makes overlap statistics uninformative.

These outputs are useful for Fernando alignment, but they must not be reported
as exact Fernando PCP unless the backbone-completeness protocol is matched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import dendropy
import pandas as pd

from progress_logging import ProgressLogger, default_log_path
from score_fish_tree_placement_outputs import (
    DEFAULT_TREE_FILE,
    ROOT,
    PlacementRun,
    canonical_label,
    discover_runs,
    edge_descendant_labels,
    load_query_taxonomy,
    lwr_bucket,
    placement_names,
    read_json,
    top_placement,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_full_tree_support(tree_file: Path) -> tuple[dict[str, object], dict[object, set[str]]]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick", preserve_underscores=True)
    label_to_node: dict[str, object] = {}
    descendant_cache: dict[object, set[str]] = {}
    for node in tree.postorder_node_iter():
        labels: set[str] = set()
        if node.is_leaf():
            if node.taxon is not None and node.taxon.label:
                label = canonical_label(node.taxon.label)
                labels.add(label)
                label_to_node[label] = node
        else:
            for child in node.child_node_iter():
                labels.update(descendant_cache.get(child, set()))
        descendant_cache[node] = labels
    return label_to_node, descendant_cache


def expected_sister_support(
    true_label: str,
    label_to_node: dict[str, object],
    descendant_cache: dict[object, set[str]],
    reference_labels: set[str],
) -> tuple[set[str], int, int]:
    """Find the closest full-tree sister support represented in the backbone.

    If the immediate sister taxon/clade is absent from this split's reference
    tree, climb to the next ancestor until at least one represented reference
    taxon is found outside the query lineage.
    """
    node = label_to_node.get(true_label)
    if node is None:
        return set(), 0, 0
    depth = 0
    current = node
    while current.parent_node is not None:
        parent = current.parent_node
        support: set[str] = set()
        for sibling in parent.child_node_iter():
            if sibling is current:
                continue
            support.update(descendant_cache.get(sibling, set()) & reference_labels)
        if support:
            return support, depth, len(descendant_cache.get(parent, set()))
        current = parent
        depth += 1
    return set(), depth, len(descendant_cache.get(current, set()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return float("nan")
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def side_metrics(expected: set[str], side: set[str]) -> dict[str, object]:
    overlap = expected & side
    return {
        "overlap_count": len(overlap),
        "jaccard": jaccard(expected, side),
        "exact_match": bool(expected and expected == side),
        "any_overlap": bool(overlap),
        "precision": len(overlap) / len(side) if side else float("nan"),
        "recall": len(overlap) / len(expected) if expected else float("nan"),
    }


def score_run(
    run: PlacementRun,
    label_to_node: dict[str, object],
    descendant_cache: dict[object, set[str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    data = read_json(run.jplace_path)
    fields = list(data.get("fields", []))
    edge_to_labels = edge_descendant_labels(str(data["tree"]))
    reference_labels = sorted({label for labels in edge_to_labels.values() for label in labels})
    reference_set = set(reference_labels)
    query_tax = load_query_taxonomy(run.split_root)
    expected_cache: dict[str, tuple[set[str], int, int]] = {}
    rows: list[dict[str, object]] = []
    missing_queries = 0

    for placement in data.get("placements", []):
        names = placement_names(placement)
        if not names:
            continue
        top = top_placement(placement, fields)
        edge_num = top["edge_num"]
        descendant_side = edge_to_labels.get(int(edge_num), set()) if edge_num is not None else set()
        complement_side = reference_set - descendant_side
        for name in names:
            truth = query_tax.get(name)
            if truth is None:
                missing_queries += 1
                true_label = ""
            else:
                true_label = canonical_label(truth.get("species", ""))
            if true_label not in expected_cache:
                expected_cache[true_label] = expected_sister_support(
                    true_label,
                    label_to_node,
                    descendant_cache,
                    reference_set,
                )
            expected, climb_depth, parent_clade_size = expected_cache[true_label]
            descendant_metrics = side_metrics(expected, descendant_side)
            complement_metrics = side_metrics(expected, complement_side)
            if (
                math.isfinite(float(descendant_metrics["jaccard"]))
                and math.isfinite(float(complement_metrics["jaccard"]))
                and float(complement_metrics["jaccard"]) > float(descendant_metrics["jaccard"])
            ):
                best_orientation_side = "complement"
                best_orientation = complement_metrics
            else:
                best_orientation_side = "descendant"
                best_orientation = descendant_metrics
            rows.append(
                {
                    "split": run.split,
                    "method": run.method,
                    "processid": name,
                    "top_edge_num": edge_num,
                    "top_lwr": top["lwr"],
                    "top_lwr_bucket": lwr_bucket(top["lwr"]),
                    "true_tree_label": true_label,
                    "expected_sister_support_count": len(expected),
                    "expected_sister_climb_depth": climb_depth,
                    "expected_parent_clade_size": parent_clade_size,
                    "placed_descendant_count": len(descendant_side),
                    "placed_complement_count": len(complement_side),
                    "best_orientation_side": best_orientation_side,
                    "pcp_like_exact_sister_match": bool(descendant_metrics["exact_match"]),
                    "pcp_like_any_sister_overlap": bool(descendant_metrics["any_overlap"]),
                    "pcp_like_best_jaccard": descendant_metrics["jaccard"],
                    "pcp_like_best_precision": descendant_metrics["precision"],
                    "pcp_like_best_recall": descendant_metrics["recall"],
                    "descendant_sister_jaccard": descendant_metrics["jaccard"],
                    "descendant_sister_precision": descendant_metrics["precision"],
                    "descendant_sister_recall": descendant_metrics["recall"],
                    "complement_sister_jaccard": complement_metrics["jaccard"],
                    "complement_sister_precision": complement_metrics["precision"],
                    "complement_sister_recall": complement_metrics["recall"],
                    "orientation_best_jaccard": best_orientation["jaccard"],
                    "orientation_best_any_sister_overlap": bool(best_orientation["any_overlap"]),
                    "caution": "Fernando-like sister-clade diagnostic; not exact Fernando PCP.",
                    "jplace_file": str(run.jplace_path),
                }
            )

    summary: dict[str, object] = {
        "split": run.split,
        "method": run.method,
        "jplace_file": str(run.jplace_path),
        "n_queries_scored": len(rows),
        "missing_query_metadata": missing_queries,
        "n_reference_labels": len(reference_set),
        "caution": "Fernando-like sister-clade diagnostic; not exact Fernando PCP.",
    }
    add_summary_metrics(summary, rows)
    return rows, summary


def finite_values(rows: Iterable[dict[str, object]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def add_summary_metrics(summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    n = len(rows)
    supported = [row for row in rows if int(row.get("expected_sister_support_count", 0)) > 0]
    summary["n_queries_with_expected_sister_support"] = len(supported)
    summary["expected_sister_support_rate"] = len(supported) / n if n else None
    for subset_name, subset in {
        "all": rows,
        "supported": supported,
        "lwr_ge_0p9": [row for row in supported if safe_float(row.get("top_lwr")) >= 0.9],
    }.items():
        denom = len(subset)
        summary[f"{subset_name}_n"] = denom
        summary[f"{subset_name}_exact_sister_match_rate"] = (
            sum(bool(row.get("pcp_like_exact_sister_match")) for row in subset) / denom if denom else None
        )
        summary[f"{subset_name}_any_sister_overlap_rate"] = (
            sum(bool(row.get("pcp_like_any_sister_overlap")) for row in subset) / denom if denom else None
        )
        jaccards = finite_values(subset, "pcp_like_best_jaccard")
        precisions = finite_values(subset, "pcp_like_best_precision")
        recalls = finite_values(subset, "pcp_like_best_recall")
        summary[f"{subset_name}_best_jaccard_median"] = float(pd.Series(jaccards).median()) if jaccards else None
        summary[f"{subset_name}_best_jaccard_mean"] = float(pd.Series(jaccards).mean()) if jaccards else None
        summary[f"{subset_name}_best_precision_median"] = float(pd.Series(precisions).median()) if precisions else None
        summary[f"{subset_name}_best_recall_median"] = float(pd.Series(recalls).median()) if recalls else None


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def summarize_buckets(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["method"]), str(row["top_lwr_bucket"]))].append(row)
    out: list[dict[str, object]] = []
    for (split, method, bucket), sub in sorted(grouped.items()):
        summary: dict[str, object] = {
            "split": split,
            "method": method,
            "top_lwr_bucket": bucket,
        }
        add_summary_metrics(summary, sub)
        out.append(summary)
    return out


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
    logger.log(f"Loading full tree sister-clade support from {args.tree_file}")
    label_to_node, descendant_cache = load_full_tree_support(args.tree_file)
    logger.log(f"Loaded {len(label_to_node)} labelled full-tree tips")
    runs = discover_runs(args.placement_root) if args.placement_root.exists() else []
    logger.log(f"Discovered {len(runs)} placement run(s)")

    per_query_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for run in runs:
        logger.log(f"Building Fernando-like sister-clade diagnostics for {run.method} {run.split}")
        rows, summary = score_run(run, label_to_node, descendant_cache)
        per_query_rows.extend(rows)
        summary_rows.append(summary)
        logger.log(f"Scored {summary['n_queries_scored']} query rows for {run.method} {run.split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "placement_pcp_like_per_query.csv"
    summary_path = args.output_dir / "placement_pcp_like_summary.csv"
    lwr_path = args.output_dir / "placement_pcp_like_lwr_summary.csv"
    logger.log(f"Writing {per_query_path}")
    write_csv(per_query_path, per_query_rows)
    logger.log(f"Writing {summary_path}")
    write_csv(summary_path, summary_rows)
    lwr_rows = summarize_buckets(per_query_rows)
    logger.log(f"Writing {lwr_path}")
    write_csv(lwr_path, lwr_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_fernando_like_pcp_diagnostics.py",
        "placement_root": str(args.placement_root),
        "tree_file": str(args.tree_file),
        "output_dir": str(args.output_dir),
        "runs_scored": len(runs),
        "per_query_rows": len(per_query_rows),
        "summary_rows": len(summary_rows),
        "lwr_summary_rows": len(lwr_rows),
        "caution": "Fernando-like sister-clade diagnostic; not exact Fernando PCP or backbone-completeness reproduction.",
    }
    manifest_path = args.output_dir / "placement_pcp_like_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
