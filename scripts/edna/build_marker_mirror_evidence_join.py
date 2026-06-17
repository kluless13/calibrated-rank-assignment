#!/usr/bin/env python3
"""Build an evidence-join table for MarkerMirror candidates.

MarkerMirror gives a cross-marker candidate list.  This script adds the first
downstream evidence layer:

- candidate-list geometry: score, margins, rank support, top-k ambiguity;
- same-marker sequence checks: compare the query sequence against candidate
  references from the query marker, when available;
- reference/resolvability flags: whether a candidate has query-marker and
  target-marker references, and whether the query sequence is exact-ambiguous;
- optional tree-neighborhood evidence: candidate distance to the top candidate
  in the fish tree when both labels are present.

For a 12S query with 16S candidates, same-marker sequence evidence means
12S-vs-12S evidence for each candidate species if that candidate has a 12S
reference.  The script deliberately does not align 12S directly to 16S.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import Logger, canonical_tree_label, clean_sequence, load_species_json, load_taxonomy

RANKS = ("species", "genus", "family", "order")
TOP_KS = (5, 10, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rankings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--marker-a-name", default="12S")
    parser.add_argument("--marker-a-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource")
    parser.add_argument("--marker-b-name", default="16S")
    parser.add_argument("--marker-b-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource")
    parser.add_argument("--max-a-per-species", type=int, default=8)
    parser.add_argument("--max-b-per-species", type=int, default=8)
    parser.add_argument("--tree-file", type=Path, default=ROOT / "data" / "phylo" / "actinopt_12k_treePL.tre")
    parser.add_argument("--disable-tree", action="store_true")
    parser.add_argument(
        "--marker-resolvability-table",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_marker_resolvability_by_species.csv",
        help="Optional per-species marker-resolvability table from build_marker_mirror_marker_resolvability.py.",
    )
    parser.add_argument("--marker-resolvability-identity", type=float, default=0.99)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def load_marker_sequences(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    specs = [
        (args.marker_a_name, args.marker_a_input_dir, args.max_a_per_species),
        (args.marker_b_name, args.marker_b_input_dir, args.max_b_per_species),
    ]
    markers: dict[str, dict[str, Any]] = {}
    for name, input_dir, max_per_species in specs:
        raw = load_species_json(input_dir / "train_species_sequences.json", max_per_species)
        cleaned = {label: [clean_sequence(seq) for seq in seqs] for label, seqs in raw.items()}
        exact_index: dict[str, set[str]] = defaultdict(set)
        for label, seqs in cleaned.items():
            for seq in seqs:
                if seq:
                    exact_index[seq].add(label)
        markers[name] = {
            "input_dir": input_dir,
            "sequences": cleaned,
            "exact_index": exact_index,
            "taxonomy": load_taxonomy(input_dir / "candidate_species.csv"),
        }
    return markers


def p_distance(query: str, reference: str) -> tuple[float, int, int]:
    n = min(len(query), len(reference))
    if n <= 0:
        return math.nan, 0, 0
    comparable = 0
    mismatches = 0
    for a, b in zip(query[:n], reference[:n]):
        if a not in "ACGT" or b not in "ACGT":
            continue
        comparable += 1
        mismatches += int(a != b)
    if comparable == 0:
        return math.nan, 0, 0
    return mismatches / comparable, comparable, mismatches


def best_same_marker_evidence(
    query_seq: str,
    candidate_label: str,
    marker: str,
    markers: dict[str, dict[str, Any]],
    cache: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (marker, query_seq, candidate_label)
    if key in cache:
        return cache[key]
    candidate_seqs = markers[marker]["sequences"].get(candidate_label, [])
    if not query_seq or not candidate_seqs:
        out = {
            "same_marker_sequence_available": 0,
            "same_marker_best_pdistance": math.nan,
            "same_marker_best_identity": math.nan,
            "same_marker_best_overlap_bases": 0,
            "same_marker_best_mismatches": math.nan,
            "same_marker_exact_match": 0,
        }
        cache[key] = out
        return out
    best: tuple[float, int, int] | None = None
    for ref in candidate_seqs:
        dist, comparable, mismatches = p_distance(query_seq, ref)
        if not math.isfinite(dist):
            continue
        current = (dist, comparable, mismatches)
        if best is None or (current[0], -current[1], current[2]) < (best[0], -best[1], best[2]):
            best = current
    if best is None:
        out = {
            "same_marker_sequence_available": 1,
            "same_marker_best_pdistance": math.nan,
            "same_marker_best_identity": math.nan,
            "same_marker_best_overlap_bases": 0,
            "same_marker_best_mismatches": math.nan,
            "same_marker_exact_match": 0,
        }
    else:
        dist, comparable, mismatches = best
        out = {
            "same_marker_sequence_available": 1,
            "same_marker_best_pdistance": float(dist),
            "same_marker_best_identity": float(1.0 - dist),
            "same_marker_best_overlap_bases": int(comparable),
            "same_marker_best_mismatches": int(mismatches),
            "same_marker_exact_match": int(dist == 0.0),
        }
    cache[key] = out
    return out


def load_tree_cache(tree_file: Path, enabled: bool, logger: Logger) -> dict[str, Any] | None:
    if not enabled:
        return None
    try:
        import dendropy
    except Exception as exc:
        logger.log(f"DendroPy unavailable; tree evidence disabled: {exc}")
        return None
    if not tree_file.exists():
        logger.log(f"Tree file missing; tree evidence disabled: {tree_file}")
        return None
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick")
    nodes: dict[str, Any] = {}
    for node in tree.leaf_node_iter():
        if node.taxon is not None and node.taxon.label:
            nodes[canonical_tree_label(node.taxon.label)] = node
    depths = {tree.seed_node: 0.0}
    stack = [tree.seed_node]
    while stack:
        node = stack.pop()
        for child in node.child_node_iter():
            depths[child] = depths[node] + float(child.edge_length or 0.0)
            stack.append(child)

    def ancestors(node: Any) -> list[Any]:
        out = []
        current = node
        while current is not None:
            out.append(current)
            current = current.parent_node
        return out

    ancestor_lists = {label: ancestors(node) for label, node in nodes.items()}
    ancestor_sets = {label: set(items) for label, items in ancestor_lists.items()}
    logger.log(f"Loaded tree labels={len(nodes)}")
    return {"nodes": nodes, "depths": depths, "ancestor_lists": ancestor_lists, "ancestor_sets": ancestor_sets, "distance_cache": {}}


def tree_distance(left: str, right: str, cache: dict[str, Any] | None) -> float:
    if cache is None:
        return math.nan
    left = canonical_tree_label(left)
    right = canonical_tree_label(right)
    nodes = cache["nodes"]
    if not left or not right or left not in nodes or right not in nodes:
        return math.nan
    if left == right:
        return 0.0
    key = tuple(sorted((left, right)))
    dist_cache = cache["distance_cache"]
    if key in dist_cache:
        return float(dist_cache[key])
    for node in cache["ancestor_lists"][left]:
        if node in cache["ancestor_sets"][right]:
            dist = float(cache["depths"][nodes[left]] + cache["depths"][nodes[right]] - 2 * cache["depths"][node])
            dist_cache[key] = dist
            return dist
    return math.nan


def load_resolvability_table(path: Path, identity: float, logger: Logger) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        logger.log(f"Marker resolvability table missing; feature disabled: {path}")
        return {}
    frame = pd.read_csv(path)
    if "identity" in frame.columns:
        frame = frame[np.isclose(frame["identity"].astype(float), float(identity))]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        marker = str(row.get("marker", ""))
        label = str(row.get("tree_label", ""))
        if not marker or not label:
            continue
        out[(marker, label)] = row.to_dict()
    logger.log(f"Loaded marker resolvability rows={len(out)} identity={identity} from {path}")
    return out


def resolvability_features(prefix: str, marker: str, label: str, table: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    row = table.get((marker, label), {})
    return {
        f"{prefix}_resolvability_rank": row.get("deepest_supported_rank", ""),
        f"{prefix}_resolvability_neighbor_species_count": row.get("neighbor_species_count", math.nan),
        f"{prefix}_resolvability_species_supported": row.get("species_oracle_supported", math.nan),
        f"{prefix}_resolvability_genus_supported": row.get("genus_oracle_supported", math.nan),
        f"{prefix}_resolvability_family_supported": row.get("family_oracle_supported", math.nan),
        f"{prefix}_resolvability_order_supported": row.get("order_oracle_supported", math.nan),
    }


def add_query_level_features(frame: pd.DataFrame, markers: dict[str, dict[str, Any]], logger: Logger) -> pd.DataFrame:
    out_frames = []
    for idx, (_, group) in enumerate(frame.groupby(["model", "direction", "split", "query_id"], sort=False), start=1):
        group = group.sort_values("candidate_rank").copy()
        scores = group["score"].to_numpy(dtype=float)
        top = group.iloc[0]
        top_label = str(top["candidate_tree_label"])
        query_marker = str(top["query_marker"])
        query_label = str(top["query_tree_label"])
        query_idx = int(top["query_seq_index"])
        query_seqs = markers[query_marker]["sequences"].get(query_label, [])
        query_seq = query_seqs[query_idx] if 0 <= query_idx < len(query_seqs) else ""
        exact_species = markers[query_marker]["exact_index"].get(query_seq, set()) if query_seq else set()

        group["top1_score"] = float(scores[0])
        group["score_margin_to_top1"] = float(scores[0]) - group["score"].astype(float)
        group["score_margin_1_2"] = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
        group["score_margin_1_5"] = float(scores[0] - scores[min(4, len(scores) - 1)])
        group["score_margin_1_10"] = float(scores[0] - scores[min(9, len(scores) - 1)])
        group["query_sequence_length"] = len(query_seq)
        group["query_exact_sequence_species_count"] = len(exact_species)
        group["query_exact_sequence_ambiguous"] = int(len(exact_species) > 1)
        group["candidate_is_top1"] = (group["candidate_tree_label"].astype(str) == top_label).astype(int)

        for k in TOP_KS:
            subset = group[group["candidate_rank"] <= k]
            group[f"top{k}_score_mean"] = float(subset["score"].mean()) if len(subset) else math.nan
            group[f"top{k}_score_std"] = float(subset["score"].std(ddof=0)) if len(subset) else math.nan
            for rank in RANKS:
                values = subset[f"candidate_{rank}"].astype(str).replace({"nan": "", "None": ""})
                unique = values[values != ""].nunique()
                group[f"top{k}_{rank}_unique"] = int(unique)
                counts = values.value_counts()
                group[f"top{k}_{rank}_mode_count"] = int(counts.iloc[0]) if len(counts) else 0
                group[f"top{k}_{rank}_mode_fraction"] = float(counts.iloc[0] / max(len(subset), 1)) if len(counts) else 0.0
                group[f"candidate_top{k}_{rank}_support"] = group[f"candidate_{rank}"].astype(str).map(counts).fillna(0).astype(int)
                group[f"candidate_top{k}_{rank}_support_fraction"] = group[f"candidate_top{k}_{rank}_support"] / max(len(subset), 1)
        out_frames.append(group)
        if idx == 1 or idx % 500 == 0:
            logger.log(f"Added query-level features for {idx} query groups")
    return pd.concat(out_frames, ignore_index=True)


def add_candidate_evidence(
    frame: pd.DataFrame,
    markers: dict[str, dict[str, Any]],
    tree_cache: dict[str, Any] | None,
    resolvability: dict[tuple[str, str], dict[str, Any]],
    logger: Logger,
) -> pd.DataFrame:
    seq_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    out = []
    top1_by_query = frame[frame["candidate_rank"] == 1].set_index(["model", "direction", "split", "query_id"])["candidate_tree_label"].astype(str).to_dict()
    for idx, row in frame.iterrows():
        query_marker = str(row["query_marker"])
        target_marker = str(row["target_marker"])
        query_label = str(row["query_tree_label"])
        query_idx = int(row["query_seq_index"])
        candidate_label = str(row["candidate_tree_label"])
        query_seqs = markers[query_marker]["sequences"].get(query_label, [])
        query_seq = query_seqs[query_idx] if 0 <= query_idx < len(query_seqs) else ""
        same_marker = best_same_marker_evidence(query_seq, candidate_label, query_marker, markers, seq_cache)
        top_key = (row["model"], row["direction"], row["split"], row["query_id"])
        top_label = top1_by_query.get(top_key, "")
        item = row.to_dict()
        item.update(same_marker)
        item["candidate_has_query_marker_reference"] = int(candidate_label in markers[query_marker]["sequences"])
        item["candidate_query_marker_reference_count"] = int(len(markers[query_marker]["sequences"].get(candidate_label, [])))
        item["candidate_has_target_marker_reference"] = int(candidate_label in markers[target_marker]["sequences"])
        item["candidate_target_marker_reference_count"] = int(len(markers[target_marker]["sequences"].get(candidate_label, [])))
        item["candidate_has_both_marker_references"] = int(
            item["candidate_has_query_marker_reference"] and item["candidate_has_target_marker_reference"]
        )
        item["tree_distance_to_top1_candidate"] = tree_distance(candidate_label, top_label, tree_cache)
        item.update(resolvability_features("query_marker", query_marker, query_label, resolvability))
        item.update(resolvability_features("candidate_query_marker", query_marker, candidate_label, resolvability))
        item.update(resolvability_features("candidate_target_marker", target_marker, candidate_label, resolvability))
        out.append(item)
        if idx == 0 or (idx + 1) % 50000 == 0:
            logger.log(f"Added candidate evidence rows={idx + 1}/{len(frame)}")
    return pd.DataFrame(out)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(["model", "direction", "split"], dropna=False):
        model, direction, split = key
        top = group[group["candidate_rank"] == 1]
        rows.append(
            {
                "model": model,
                "direction": direction,
                "split": split,
                "n_candidate_rows": int(len(group)),
                "n_query": int(group["query_id"].nunique()),
                "top1_same_marker_sequence_available_rate_pct": 100.0 * float(top["same_marker_sequence_available"].mean()) if len(top) else math.nan,
                "top1_same_marker_exact_match_rate_pct": 100.0 * float(top["same_marker_exact_match"].mean()) if len(top) else math.nan,
                "top1_same_marker_identity_mean": float(top["same_marker_best_identity"].mean()) if len(top) else math.nan,
                "top1_candidate_has_both_marker_references_rate_pct": 100.0 * float(top["candidate_has_both_marker_references"].mean()) if len(top) else math.nan,
                "query_exact_sequence_ambiguous_rate_pct": 100.0 * float(top["query_exact_sequence_ambiguous"].mean()) if len(top) else math.nan,
                "query_marker_species_resolvable_rate_pct": 100.0 * float(top["query_marker_resolvability_species_supported"].mean()) if len(top) and "query_marker_resolvability_species_supported" in top else math.nan,
                "top1_candidate_query_marker_species_resolvable_rate_pct": 100.0 * float(top["candidate_query_marker_resolvability_species_supported"].mean()) if len(top) and "candidate_query_marker_resolvability_species_supported" in top else math.nan,
                "top1_candidate_target_marker_species_resolvable_rate_pct": 100.0 * float(top["candidate_target_marker_resolvability_species_supported"].mean()) if len(top) and "candidate_target_marker_resolvability_species_supported" in top else math.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_evidence_join.log")
    logger.log(f"Arguments: {vars(args)}")
    markers = load_marker_sequences(args)
    tree_cache = load_tree_cache(args.tree_file, not args.disable_tree, logger)
    resolvability = load_resolvability_table(args.marker_resolvability_table, args.marker_resolvability_identity, logger)

    frame = pd.read_csv(args.candidate_rankings)
    logger.log(f"Loaded candidate rankings rows={len(frame)} queries={frame['query_id'].nunique()}")
    joined = add_query_level_features(frame, markers, logger)
    joined = add_candidate_evidence(joined, markers, tree_cache, resolvability, logger)
    summary = summarize(joined)

    out_path = args.output_dir / "marker_mirror_evidence_join.csv.gz"
    joined.to_csv(out_path, index=False)
    summary.to_csv(args.output_dir / "marker_mirror_evidence_join_summary.csv", index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_rankings": str(args.candidate_rankings),
        "rows": int(len(joined)),
        "queries": int(joined["query_id"].nunique()),
        "tree_evidence_enabled": tree_cache is not None,
        "marker_resolvability_table": str(args.marker_resolvability_table),
        "marker_resolvability_identity": args.marker_resolvability_identity,
        "marker_resolvability_rows": len(resolvability),
        "claim_boundary": "Evidence join for candidate ranking and calibration; no final assignment claim by itself.",
    }
    (args.output_dir / "marker_mirror_evidence_join_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote {out_path} rows={len(joined)}")


if __name__ == "__main__":
    main()
