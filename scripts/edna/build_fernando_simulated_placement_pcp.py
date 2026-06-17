#!/usr/bin/env python3
"""Build Fernando-style PCP diagnostics from simulated jplace trees.

Fernando et al. compute PCP by comparing the sister of each placed species in a
placement tree against the sister of that species in the full reference tree.
Their R script uses `ips::sister()` for both trees.

This script mirrors that idea for our EPA-ng jplace files:

1. Read the reference tree embedded in a jplace file.
2. Graft each query onto its top-LWR jplace edge to create a simulated
   placement tree.
3. Compare the sister support set for the placed query against the sister
   support set for the true species in the full fish tree.

Two levels are emitted:

- sequence_level: every query sequence is inserted as a unique query leaf.
- species_representative: one highest-LWR query is inserted per true species.

This is closer to Fernando's PCP than placed-clade containment, but it is still
not an exact reproduction until we run matched backbone-completeness sampling
and official placement-tree generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dendropy
import pandas as pd

from progress_logging import ProgressLogger, default_log_path
from score_fish_tree_placement_outputs import (
    DEFAULT_TREE_FILE,
    ROOT,
    PlacementRun,
    canonical_label,
    discover_runs,
    jplace_tree_to_dendropy,
    load_query_taxonomy,
    lwr_bucket,
    placement_names,
    read_json,
    top_placement,
)


@dataclass(frozen=True)
class QueryPlacement:
    processid: str
    true_label: str
    query_leaf_label: str
    edge_num: int
    lwr: float


def clean_query_leaf(value: str) -> str:
    return canonical_label(value).replace("|", "_").replace(":", "_").replace(";", "_")


def edge_node_map(tree: dendropy.Tree) -> dict[int, dendropy.Node]:
    out: dict[int, dendropy.Node] = {}
    for node in tree.postorder_node_iter():
        for annotation in node.annotations:
            if annotation.name == "edge_num":
                out[int(annotation.value)] = node
                break
    return out


def descendant_labels(node: dendropy.Node) -> set[str]:
    labels: set[str] = set()
    for leaf in node.leaf_iter():
        if leaf.taxon is not None and leaf.taxon.label:
            labels.add(canonical_label(leaf.taxon.label))
    return labels


def all_tree_labels(tree: dendropy.Tree) -> set[str]:
    labels: set[str] = set()
    for leaf in tree.leaf_node_iter():
        if leaf.taxon is not None and leaf.taxon.label:
            labels.add(canonical_label(leaf.taxon.label))
    return labels


def sister_support(tree: dendropy.Tree, label: str) -> set[str]:
    node = tree.find_node_with_taxon_label(label)
    if node is None:
        return set()
    parent = node.parent_node
    if parent is None:
        return set()
    support: set[str] = set()
    for sibling in parent.child_node_iter():
        if sibling is node:
            continue
        support.update(descendant_labels(sibling))
    return support


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def parse_query_placements(run: PlacementRun) -> list[QueryPlacement]:
    data = read_json(run.jplace_path)
    fields = list(data.get("fields", []))
    query_tax = load_query_taxonomy(run.split_root)
    placements: list[QueryPlacement] = []
    for placement in data.get("placements", []):
        top = top_placement(placement, fields)
        edge_num = top.get("edge_num")
        lwr = top.get("lwr")
        if edge_num is None:
            continue
        for name in placement_names(placement):
            truth = query_tax.get(name)
            if truth is None:
                continue
            true_label = canonical_label(truth.get("species", ""))
            if not true_label:
                continue
            placements.append(
                QueryPlacement(
                    processid=name,
                    true_label=true_label,
                    query_leaf_label=clean_query_leaf(f"query__{true_label}__{name}"),
                    edge_num=int(edge_num),
                    lwr=float(lwr or 0.0),
                )
            )
    return placements


def species_representatives(placements: list[QueryPlacement]) -> list[QueryPlacement]:
    best: dict[str, QueryPlacement] = {}
    for placement in placements:
        current = best.get(placement.true_label)
        if current is None or placement.lwr > current.lwr:
            best[placement.true_label] = QueryPlacement(
                processid=placement.processid,
                true_label=placement.true_label,
                query_leaf_label=placement.true_label,
                edge_num=placement.edge_num,
                lwr=placement.lwr,
            )
    return list(best.values())


def graft_queries(jplace_tree_text: str, placements: list[QueryPlacement]) -> dendropy.Tree:
    tree = jplace_tree_to_dendropy(jplace_tree_text)
    edge_nodes = edge_node_map(tree)
    grouped: dict[int, list[QueryPlacement]] = defaultdict(list)
    for placement in placements:
        grouped[placement.edge_num].append(placement)

    taxon_namespace = tree.taxon_namespace
    for edge_num, edge_placements in grouped.items():
        child = edge_nodes.get(edge_num)
        if child is None:
            continue
        parent = child.parent_node
        insertion = dendropy.Node()
        insertion.edge_length = (child.edge_length or 0.0) / 2.0 if child.edge_length is not None else None
        child.edge_length = (child.edge_length or 0.0) / 2.0 if child.edge_length is not None else child.edge_length
        if parent is None:
            old_seed = tree.seed_node
            new_seed = dendropy.Node()
            tree.seed_node = new_seed
            new_seed.add_child(old_seed)
            new_seed.add_child(insertion)
        else:
            parent.remove_child(child, suppress_unifurcations=False)
            parent.add_child(insertion)
            insertion.add_child(child)
        for placement in edge_placements:
            taxon = taxon_namespace.require_taxon(label=placement.query_leaf_label)
            leaf = dendropy.Node(taxon=taxon)
            insertion.add_child(leaf)
    return tree


def score_placements(
    run: PlacementRun,
    placements: list[QueryPlacement],
    level: str,
    full_tree: dendropy.Tree,
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    if not placements:
        return [], {}, ""
    jplace = read_json(run.jplace_path)
    simulated = graft_queries(str(jplace["tree"]), placements)
    represented = all_tree_labels(simulated)
    rows: list[dict[str, object]] = []
    for placement in placements:
        full_sister = sister_support(full_tree, placement.true_label)
        represented_full_sister = full_sister & represented
        placed_sister = sister_support(simulated, placement.query_leaf_label)
        full_exact = bool(full_sister and placed_sister == full_sister)
        represented_exact = bool(represented_full_sister and placed_sister == represented_full_sister)
        overlap = placed_sister & represented_full_sister
        rows.append(
            {
                "split": run.split,
                "method": run.method,
                "pcp_level": level,
                "processid": placement.processid,
                "true_tree_label": placement.true_label,
                "query_leaf_label": placement.query_leaf_label,
                "top_edge_num": placement.edge_num,
                "top_lwr": placement.lwr,
                "top_lwr_bucket": lwr_bucket(placement.lwr),
                "full_sister_count": len(full_sister),
                "represented_full_sister_count": len(represented_full_sister),
                "placed_sister_count": len(placed_sister),
                "pcp_full_exact": full_exact,
                "pcp_represented_exact": represented_exact,
                "pcp_represented_any_overlap": bool(overlap),
                "pcp_represented_overlap_count": len(overlap),
                "pcp_represented_jaccard": jaccard(placed_sister, represented_full_sister),
                "pcp_represented_precision": len(overlap) / len(placed_sister) if placed_sister else float("nan"),
                "pcp_represented_recall": (
                    len(overlap) / len(represented_full_sister) if represented_full_sister else float("nan")
                ),
                "caution": "Simulated jplace-tree PCP diagnostic; not exact Fernando backbone-completeness reproduction.",
                "jplace_file": str(run.jplace_path),
            }
        )
    summary = summarize_rows(rows, run, level)
    simulated_newick = simulated.as_string(schema="newick", suppress_annotations=True).strip()
    return rows, summary, simulated_newick


def finite_values(rows: list[dict[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize_rows(rows: list[dict[str, object]], run: PlacementRun, level: str) -> dict[str, object]:
    n = len(rows)
    represented = [row for row in rows if int(row.get("represented_full_sister_count", 0)) > 0]
    out: dict[str, object] = {
        "split": run.split,
        "method": run.method,
        "pcp_level": level,
        "n_rows": n,
        "n_with_represented_full_sister": len(represented),
        "represented_full_sister_rate": len(represented) / n if n else None,
        "jplace_file": str(run.jplace_path),
        "caution": "Simulated jplace-tree PCP diagnostic; not exact Fernando backbone-completeness reproduction.",
    }
    for prefix, subset in {
        "all": rows,
        "represented": represented,
        "lwr_ge_0p9": [row for row in represented if float(row.get("top_lwr", 0.0)) >= 0.9],
    }.items():
        denom = len(subset)
        out[f"{prefix}_n"] = denom
        out[f"{prefix}_pcp_full_exact_rate"] = (
            sum(bool(row.get("pcp_full_exact")) for row in subset) / denom if denom else None
        )
        out[f"{prefix}_pcp_represented_exact_rate"] = (
            sum(bool(row.get("pcp_represented_exact")) for row in subset) / denom if denom else None
        )
        out[f"{prefix}_pcp_represented_any_overlap_rate"] = (
            sum(bool(row.get("pcp_represented_any_overlap")) for row in subset) / denom if denom else None
        )
        jaccards = finite_values(subset, "pcp_represented_jaccard")
        out[f"{prefix}_pcp_represented_jaccard_median"] = (
            float(pd.Series(jaccards).median()) if jaccards else None
        )
        out[f"{prefix}_pcp_represented_jaccard_mean"] = float(pd.Series(jaccards).mean()) if jaccards else None
    return out


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
    parser.add_argument("--tree-file", type=Path, default=DEFAULT_TREE_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
    )
    parser.add_argument(
        "--tree-output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/fernando_simulated_placement_trees"),
    )
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    logger.log(f"Loading full reference tree from {args.tree_file}")
    full_tree = dendropy.Tree.get(path=str(args.tree_file), schema="newick", preserve_underscores=True)
    runs = discover_runs(args.placement_root) if args.placement_root.exists() else []
    logger.log(f"Discovered {len(runs)} jplace run(s)")

    per_query_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    args.tree_output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        logger.log(f"Parsing top placements for {run.method} {run.split}")
        sequence_placements = parse_query_placements(run)
        levels = {
            "sequence_level": sequence_placements,
            "species_representative": species_representatives(sequence_placements),
        }
        for level, placements in levels.items():
            logger.log(f"Scoring {level} PCP for {run.method} {run.split}: {len(placements)} placements")
            rows, summary, newick = score_placements(run, placements, level, full_tree)
            per_query_rows.extend(rows)
            summary_rows.append(summary)
            tree_path = args.tree_output_dir / f"{run.split}_{run.method}_{level}.nwk"
            tree_path.write_text(newick + "\n")
            logger.log(f"Wrote simulated placement tree {tree_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = args.output_dir / "placement_simulated_tree_pcp_per_query.csv"
    summary_path = args.output_dir / "placement_simulated_tree_pcp_summary.csv"
    logger.log(f"Writing {per_query_path}")
    write_csv(per_query_path, per_query_rows)
    logger.log(f"Writing {summary_path}")
    write_csv(summary_path, summary_rows)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_fernando_simulated_placement_pcp.py",
        "placement_root": str(args.placement_root),
        "tree_file": str(args.tree_file),
        "output_dir": str(args.output_dir),
        "tree_output_dir": str(args.tree_output_dir),
        "runs_scored": len(runs),
        "per_query_rows": len(per_query_rows),
        "summary_rows": len(summary_rows),
        "caution": "Simulated jplace-tree PCP diagnostic; not exact Fernando backbone-completeness reproduction.",
    }
    manifest_path = args.output_dir / "placement_simulated_tree_pcp_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
