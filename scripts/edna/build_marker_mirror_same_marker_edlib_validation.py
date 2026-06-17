#!/usr/bin/env python3
"""Validate same-marker 12S candidates with edlib alignment scoring.

Exp 103 used a character-kmer nearest-neighbor audit to show that same-marker
12S evidence greatly expands high-rank support for the MarkerMirror union
pipeline.  This script tests the next claim boundary: whether that support
survives after scoring the same candidate pool with an alignment/edit-distance
backend.

This is not full all-vs-all BLAST/VSEARCH search.  It reranks the existing
same-marker top-k candidate pool by best edlib HW identity against reference
sequences for each candidate species.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--same-marker-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_same_marker_kmer_candidates_top50.csv.gz",
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
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--max-ref-sequences-per-species", type=int, default=12)
    parser.add_argument("--report-top-ks", default="1,5,10,50")
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def normalize_sequence(value: Any) -> str:
    return "".join(ch for ch in str(value).upper() if ch in {"A", "C", "G", "T"})


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_queries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "query_id": frame["processid"].astype(str),
            "source": frame["source"].astype(str),
            "query_tree_label": frame["tree_label"].astype(str),
            "query_species": frame["species_name"].astype(str),
            "query_genus": frame["genus_name"].map(clean),
            "query_family": frame["family_name"].map(clean),
            "query_order": frame["order_name"].map(clean),
            "query_sequence": frame["nucleotides"].map(normalize_sequence),
        }
    )
    return out


def load_reference_sequences(input_dir: Path, max_per_species: int) -> dict[str, list[str]]:
    raw = json.loads((input_dir / "species_sequences.json").read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for species, sequences in raw.items():
        seen: set[str] = set()
        out[str(species)] = []
        for sequence in sequences:
            seq = normalize_sequence(sequence)
            if not seq or seq in seen:
                continue
            seen.add(seq)
            out[str(species)].append(seq)
            if len(out[str(species)]) >= max_per_species:
                break
    return out


def edlib_identity(query: str, reference: str) -> tuple[float, int, str]:
    import edlib  # type: ignore[import-not-found]

    if not query or not reference:
        return math.nan, -1, "missing_sequence"

    best_identity = -math.inf
    best_distance = -1
    best_orientation = ""
    for orientation, qseq, tseq, denom in (
        ("query_to_reference", query, reference, len(query)),
        ("reference_to_query", reference, query, len(reference)),
    ):
        if denom <= 0:
            continue
        result = edlib.align(qseq, tseq, mode="HW", task="locations")
        distance = int(result.get("editDistance", -1))
        if distance < 0:
            continue
        identity = max(0.0, min(1.0, 1.0 - (float(distance) / float(denom))))
        if identity > best_identity:
            best_identity = identity
            best_distance = distance
            best_orientation = orientation

    if not math.isfinite(best_identity):
        return math.nan, -1, "unaligned"
    return best_identity, best_distance, best_orientation


def best_species_alignment(query_sequence: str, references: list[str]) -> dict[str, Any]:
    best = {
        "edlib_best_identity": math.nan,
        "edlib_best_edit_distance": -1,
        "edlib_best_orientation": "",
        "edlib_best_reference_length": 0,
        "edlib_reference_sequence_count_scored": 0,
    }
    for reference in references:
        identity, distance, orientation = edlib_identity(query_sequence, reference)
        best["edlib_reference_sequence_count_scored"] += 1
        current = best["edlib_best_identity"]
        if math.isnan(current) or (not math.isnan(identity) and identity > current):
            best.update(
                {
                    "edlib_best_identity": identity,
                    "edlib_best_edit_distance": distance,
                    "edlib_best_orientation": orientation,
                    "edlib_best_reference_length": len(reference),
                }
            )
    return best


def hit_flags(group: pd.DataFrame, query: pd.Series, top_k: int) -> dict[str, bool]:
    subset = group.head(top_k)
    targets = {
        "species": clean(query["query_tree_label"]),
        "genus": clean(query["query_genus"]),
        "family": clean(query["query_family"]),
        "order": clean(query["query_order"]),
    }
    columns = {
        "species": "candidate_tree_label",
        "genus": "candidate_genus",
        "family": "candidate_family",
        "order": "candidate_order",
    }
    return {
        rank: bool(targets[rank] and (subset[columns[rank]].astype(str) == targets[rank]).any())
        for rank in RANKS
    }


def support_tables(candidates: pd.DataFrame, queries: pd.DataFrame, top_ks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_by_id = queries.set_index("query_id")
    rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []

    for source_name, rank_col in [
        ("same_marker_kmer_original", "candidate_rank"),
        ("same_marker_edlib_rerank", "edlib_candidate_rank"),
    ]:
        ranked = candidates.sort_values(["query_id", rank_col, "candidate_rank"]).copy()
        for query_id, group in ranked.groupby("query_id", sort=False):
            if query_id not in query_by_id.index:
                continue
            query = query_by_id.loc[query_id]
            query_row: dict[str, Any] = {"query_id": query_id, "candidate_source_eval": source_name}
            for top_k in top_ks:
                flags = hit_flags(group, query, top_k)
                for rank, value in flags.items():
                    query_row[f"top{top_k}_{rank}_hit"] = bool(value)
            first = group.iloc[0]
            query_row.update(
                {
                    "top1_candidate_tree_label": first.get("candidate_tree_label", ""),
                    "top1_candidate_genus": first.get("candidate_genus", ""),
                    "top1_candidate_family": first.get("candidate_family", ""),
                    "top1_candidate_order": first.get("candidate_order", ""),
                    "top1_kmer_score": float(first.get("score", math.nan)),
                    "top1_edlib_identity": float(first.get("edlib_best_identity", math.nan)),
                }
            )
            per_query_rows.append(query_row)

        per_query = pd.DataFrame([row for row in per_query_rows if row["candidate_source_eval"] == source_name])
        for top_k in top_ks:
            metric_row: dict[str, Any] = {
                "candidate_source_eval": source_name,
                "top_k": int(top_k),
                "n_queries": int(len(per_query)),
            }
            for rank in RANKS:
                metric_row[f"{rank}_hit_pct"] = float(per_query[f"top{top_k}_{rank}_hit"].mean() * 100.0)
            rows.append(metric_row)

    return pd.DataFrame(rows), pd.DataFrame(per_query_rows)


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    script_name = Path(__file__).name
    logger.start(script_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import edlib  # noqa: F401  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit("edlib is required. Install with `python3 -m pip install edlib`.") from exc

    top_ks = [int(value.strip()) for value in args.report_top_ks.split(",") if value.strip()]
    logger.log(f"Loading queries {rel(args.query_table)}")
    queries = load_queries(args.query_table)
    if args.limit_queries:
        queries = queries.head(args.limit_queries).copy()
    query_ids = set(queries["query_id"])
    logger.log(f"Queries rows={len(queries):,}")

    logger.log(f"Loading candidates {rel(args.same_marker_candidates)}")
    candidates = pd.read_csv(args.same_marker_candidates)
    candidates = candidates[candidates["query_id"].isin(query_ids)].copy()
    logger.log(f"Candidate rows={len(candidates):,}")

    logger.log(f"Loading reference sequences from {rel(args.same_marker_reference_dir)}")
    references = load_reference_sequences(args.same_marker_reference_dir, args.max_ref_sequences_per_species)
    logger.log(f"Reference species={len(references):,}")

    query_sequences = queries.set_index("query_id")["query_sequence"].to_dict()
    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates.itertuples(index=False), start=1):
        if idx % 25000 == 0:
            logger.log(f"Scored {idx:,}/{len(candidates):,} candidate rows")
        row_dict = row._asdict()
        query_sequence = query_sequences.get(row_dict["query_id"], "")
        candidate_label = clean(row_dict.get("candidate_tree_label"))
        row_dict.update(best_species_alignment(query_sequence, references.get(candidate_label, [])))
        scored_rows.append(row_dict)

    scored = pd.DataFrame(scored_rows)
    scored = scored.sort_values(
        ["query_id", "edlib_best_identity", "score", "candidate_rank"],
        ascending=[True, False, False, True],
    ).copy()
    scored["edlib_candidate_rank"] = scored.groupby("query_id").cumcount() + 1

    candidate_path = args.output_dir / "marker_mirror_same_marker_edlib_candidates_top50.csv.gz"
    summary_path = args.output_dir / "marker_mirror_same_marker_edlib_support_summary.csv"
    per_query_path = args.output_dir / "marker_mirror_same_marker_edlib_support_per_query.csv"
    manifest_path = args.output_dir / "marker_mirror_same_marker_edlib_validation_manifest.json"

    summary, per_query = support_tables(scored, queries, top_ks)
    scored.to_csv(candidate_path, index=False)
    summary.to_csv(summary_path, index=False)
    per_query.to_csv(per_query_path, index=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "same_marker_candidates": rel(args.same_marker_candidates),
            "query_table": rel(args.query_table),
            "same_marker_reference_dir": rel(args.same_marker_reference_dir),
        },
        "outputs": {
            "alignment_candidates": rel(candidate_path),
            "support_summary": rel(summary_path),
            "per_query": rel(per_query_path),
        },
        "n_queries": int(len(queries)),
        "n_candidate_rows": int(len(scored)),
        "max_ref_sequences_per_species": int(args.max_ref_sequences_per_species),
        "backend": "edlib HW bidirectional edit-distance identity over existing k-mer top-k pool",
        "claim_boundary": "alignment validation/rerank of candidate pool, not full all-vs-all BLAST/VSEARCH replacement",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote {rel(candidate_path)}")
    logger.log(f"Wrote {rel(summary_path)}")
    logger.log(f"Wrote {rel(per_query_path)}")
    logger.log(f"Wrote {rel(manifest_path)}")
    logger.done(script_name)


if __name__ == "__main__":
    main()
