#!/usr/bin/env python3
"""Audit whether same-marker support expands MarkerMirror candidate coverage.

MarkerMirror currently asks a 12S query to retrieve 16S reference species. That
cannot recover a species that is absent from the 16S target reference. This
script builds a simple union-candidate diagnostic:

    MarkerMirror 12S->16S top-k candidates
    + same-marker 12S k-mer top-k candidates
    -> query-level support by species/genus/family/order.

The same-marker arm is deliberately labelled as a k-mer audit, not a final
BLAST/VSEARCH result. It is meant to quantify whether adding another candidate
source helps the rank/no-call pipeline before we train more models.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import Logger

RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff-evidence-table",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "production_handoff_fullref_all_queries_12s_to_16s"
        / "evidence_handoff"
        / "marker_mirror_candidate_generator_evidence_handoff.csv.gz",
    )
    parser.add_argument(
        "--query-table",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "zero_shot_queries.csv",
    )
    parser.add_argument(
        "--same-marker-reference-dir",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource",
    )
    parser.add_argument(
        "--target-marker-reference-dir",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--ngram", type=int, default=7)
    parser.add_argument("--max-ref-sequences-per-species", type=int, default=12)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def normalize_sequence(value: object) -> str:
    return "".join(ch for ch in str(value).upper() if ch in {"A", "C", "G", "T"})


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def load_species_info(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for label, row in raw.items():
        out[str(label)] = {
            "species": str(label),
            "species_name": clean(row.get("species_name")) or str(label).replace("_", " "),
            "genus": clean(row.get("genus")) or clean(row.get("genus_name")),
            "family": clean(row.get("family")) or clean(row.get("family_name")),
            "order": clean(row.get("order")) or clean(row.get("order_name")),
        }
    return out


def load_same_marker_reference(input_dir: Path, max_per_species: int) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    species_sequences = json.loads((input_dir / "species_sequences.json").read_text(encoding="utf-8"))
    species_info = load_species_info(input_dir / "species_info.json")
    rows: list[dict[str, Any]] = []
    for species, sequences in species_sequences.items():
        info = species_info.get(species, {"species": species})
        used = 0
        seen_sequences: set[str] = set()
        for sequence in sequences:
            seq = normalize_sequence(sequence)
            if not seq or seq in seen_sequences:
                continue
            seen_sequences.add(seq)
            rows.append(
                {
                    "ref_sequence_id": f"{species}:{used}",
                    "candidate_tree_label": species,
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "candidate_species": info.get("species_name", species.replace("_", " ")),
                    "candidate_genus": info.get("genus", ""),
                    "candidate_family": info.get("family", ""),
                    "candidate_order": info.get("order", ""),
                }
            )
            used += 1
            if used >= max_per_species:
                break
    return pd.DataFrame(rows), species_info


def query_taxonomy(queries: pd.DataFrame) -> pd.DataFrame:
    out = queries.copy()
    out["query_id"] = out["processid"].astype(str)
    out["query_tree_label"] = out["tree_label"].astype(str)
    out["query_species"] = out["species_name"].astype(str)
    out["query_genus"] = out["genus_name"].map(clean)
    out["query_family"] = out["family_name"].map(clean)
    out["query_order"] = out["order_name"].map(clean)
    out["query_sequence"] = out["nucleotides"].map(normalize_sequence)
    return out[
        [
            "query_id",
            "source",
            "query_tree_label",
            "query_species",
            "query_genus",
            "query_family",
            "query_order",
            "query_sequence",
        ]
    ]


def rank_hits(candidate_rows: pd.DataFrame, query: pd.Series) -> dict[str, bool]:
    hits = {}
    values = {
        "species": "candidate_tree_label",
        "genus": "candidate_genus",
        "family": "candidate_family",
        "order": "candidate_order",
    }
    targets = {
        "species": clean(query["query_tree_label"]),
        "genus": clean(query["query_genus"]),
        "family": clean(query["query_family"]),
        "order": clean(query["query_order"]),
    }
    for rank, col in values.items():
        target = targets[rank]
        hits[rank] = bool(target and col in candidate_rows.columns and (candidate_rows[col].astype(str) == target).any())
    return hits


def marker_mirror_query_support(handoff: pd.DataFrame, target_species: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = handoff.groupby("query_id", sort=False)
    for query_id, group in grouped:
        first = group.iloc[0]
        row: dict[str, Any] = {
            "query_id": query_id,
            "query_tree_label": first.get("query_tree_label", ""),
            "query_species": first.get("query_species", ""),
            "query_genus": first.get("query_genus", ""),
            "query_family": first.get("query_family", ""),
            "query_order": first.get("query_order", ""),
            "query_source": str(query_id).split(":", 1)[0],
            "marker_mirror_top50_species_hit": bool(group["match_species"].fillna(False).astype(bool).any()),
            "marker_mirror_top50_genus_hit": bool(group["match_genus"].fillna(False).astype(bool).any()),
            "marker_mirror_top50_family_hit": bool(group["match_family"].fillna(False).astype(bool).any()),
            "marker_mirror_top50_order_hit": bool(group["match_order"].fillna(False).astype(bool).any()),
            "query_species_present_in_16s_reference": str(first.get("query_tree_label", "")) in target_species,
            "marker_mirror_top1_score": float(first.get("top1_score", np.nan)),
            "marker_mirror_top1_same_marker_available": bool(first.get("same_marker_sequence_available", 0)),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    return out


def build_same_marker_candidates(
    queries: pd.DataFrame,
    ref: pd.DataFrame,
    top_k: int,
    ngram: int,
    logger: Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.log(
        f"Fitting same-marker char {ngram}-mer TF-IDF reference_sequences={len(ref)} queries={len(queries)}"
    )
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram, ngram),
        lowercase=False,
        norm="l2",
        min_df=1,
        dtype=np.float32,
    )
    ref_x = vectorizer.fit_transform(ref["sequence"].astype(str).tolist())
    query_x = vectorizer.transform(queries["query_sequence"].astype(str).tolist())
    n_neighbors = min(max(top_k * 4, top_k), len(ref))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(ref_x)
    logger.log(f"Running same-marker nearest-neighbor search n_neighbors={n_neighbors}")
    distances, indices = nn.kneighbors(query_x, return_distance=True)

    candidate_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for q_pos, query in queries.reset_index(drop=True).iterrows():
        seen_species: set[str] = set()
        selected: list[dict[str, Any]] = []
        for dist, idx in zip(distances[q_pos], indices[q_pos]):
            ref_row = ref.iloc[int(idx)].to_dict()
            species = str(ref_row["candidate_tree_label"])
            if species in seen_species:
                continue
            seen_species.add(species)
            rank = len(selected) + 1
            record = {
                "query_id": query["query_id"],
                "candidate_rank": rank,
                "candidate_source": "same_marker_12s_kmer",
                "score": 1.0 - float(dist),
                **{k: ref_row[k] for k in [
                    "candidate_tree_label",
                    "candidate_species",
                    "candidate_genus",
                    "candidate_family",
                    "candidate_order",
                    "sequence_length",
                ]},
            }
            selected.append(record)
            if len(selected) >= top_k:
                break
        cand = pd.DataFrame(selected)
        candidate_rows.extend(selected)
        support = {
            "query_id": query["query_id"],
            "same_marker_top50_candidate_count": int(len(cand)),
            "same_marker_top1_score": float(cand.iloc[0]["score"]) if not cand.empty else math.nan,
        }
        support.update({f"same_marker_top50_{rank}_hit": hit for rank, hit in rank_hits(cand, query).items()})
        support_rows.append(support)
    return pd.DataFrame(candidate_rows), pd.DataFrame(support_rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata = [("all", "all", frame)]
    if "source" in frame.columns:
        for value, group in frame.groupby("source", dropna=False):
            strata.append(("query_source", value, group))
    if "query_species_present_in_16s_reference" in frame.columns:
        for value, group in frame.groupby("query_species_present_in_16s_reference", dropna=False):
            strata.append(("species_present_in_16s_reference", str(bool(value)), group))
    if "query_species_present_in_12s_reference" in frame.columns:
        for value, group in frame.groupby("query_species_present_in_12s_reference", dropna=False):
            strata.append(("species_present_in_12s_reference", str(bool(value)), group))

    for stratum, value, group in strata:
        row: dict[str, Any] = {"stratum": stratum, "value": value, "n_query": int(len(group))}
        for source in ("marker_mirror", "same_marker", "union"):
            for rank in RANKS:
                col = f"{source}_top50_{rank}_hit"
                if col in group.columns:
                    row[f"{source}_top50_{rank}_hit_pct"] = pct(group[col].fillna(False).astype(bool).sum(), len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_union_candidate_support.log")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    handoff = pd.read_csv(args.handoff_evidence_table)
    queries = query_taxonomy(pd.read_csv(args.query_table))
    ref, same_marker_info = load_same_marker_reference(args.same_marker_reference_dir, args.max_ref_sequences_per_species)
    target_species = set(json.loads((args.target_marker_reference_dir / "species_sequences.json").read_text(encoding="utf-8")))
    logger.log(f"Loaded handoff_rows={len(handoff)} queries={len(queries)} same_marker_ref_sequences={len(ref)}")

    mm_support = marker_mirror_query_support(handoff, target_species)
    same_marker_candidates, same_support = build_same_marker_candidates(queries, ref, args.top_k, args.ngram, logger)
    same_marker_reference_species = set(ref["candidate_tree_label"].astype(str))
    queries["query_species_present_in_12s_reference"] = queries["query_tree_label"].isin(same_marker_reference_species)

    per_query = queries.merge(mm_support, on="query_id", how="left").merge(same_support, on="query_id", how="left")
    for rank in RANKS:
        mm_col = f"marker_mirror_top50_{rank}_hit"
        sm_col = f"same_marker_top50_{rank}_hit"
        per_query[mm_col] = per_query[mm_col].fillna(False).astype(bool)
        per_query[sm_col] = per_query[sm_col].fillna(False).astype(bool)
        per_query[f"union_top50_{rank}_hit"] = per_query[mm_col] | per_query[sm_col]

    summary = summarize(per_query)
    per_query_path = args.output_dir / "marker_mirror_union_candidate_support_per_query.csv"
    summary_path = args.output_dir / "marker_mirror_union_candidate_support_summary.csv"
    candidates_path = args.output_dir / "marker_mirror_same_marker_kmer_candidates_top50.csv.gz"
    per_query.to_csv(per_query_path, index=False)
    summary.to_csv(summary_path, index=False)
    same_marker_candidates.to_csv(candidates_path, index=False)

    manifest = {
        "generated_by": "scripts/edna/build_marker_mirror_union_candidate_support.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "handoff_evidence_table": str(args.handoff_evidence_table),
        "query_table": str(args.query_table),
        "same_marker_reference_dir": str(args.same_marker_reference_dir),
        "target_marker_reference_dir": str(args.target_marker_reference_dir),
        "top_k": args.top_k,
        "ngram": args.ngram,
        "max_ref_sequences_per_species": args.max_ref_sequences_per_species,
        "n_query": int(len(per_query)),
        "n_same_marker_ref_sequences": int(len(ref)),
        "outputs": {
            "per_query": str(per_query_path),
            "summary": str(summary_path),
            "same_marker_candidates": str(candidates_path),
        },
        "claim_boundary": "Same-marker arm is TF-IDF character k-mer candidate support, not final BLAST/VSEARCH alignment evidence.",
    }
    manifest_path = args.output_dir / "marker_mirror_union_candidate_support_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(f"Wrote {summary_path}")
    logger.log(f"Wrote {per_query_path}")
    logger.log(f"Wrote {candidates_path}")


if __name__ == "__main__":
    main()
