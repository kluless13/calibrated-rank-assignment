#!/usr/bin/env python3
"""Build exact-identity 12S resolvability/oracle maps.

The output answers: from this 12S reference alone, what is the deepest
taxonomic rank that an exact sequence can support?

This is intentionally model-free. If multiple species share an exact normalized
12S sequence, no sequence-only model can honestly distinguish those species
without extra evidence such as geography, co-occurrence, another marker, or a
no-call/higher-rank policy.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
RANK_ORDER = ["species", "genus", "family", "order", "unresolved"]
RANK_SCORE = {rank: idx for idx, rank in enumerate(RANK_ORDER)}
TAXONOMIC_RANKS = ["species", "genus", "family", "order"]


def clean_text(value: object) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def normalize_sequence(seq: object, mode: str) -> str:
    text = str(seq).upper()
    if mode == "acgtn":
        keep = set("ACGTN")
    elif mode == "acgt":
        keep = set("ACGT")
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    return "".join(ch for ch in text if ch in keep)


def rank_from_cluster(labels: set[str], species_info: dict[str, dict[str, Any]]) -> str:
    if not labels:
        return "unresolved"
    if len(labels) == 1:
        return "species"
    rank_values: dict[str, set[str]] = {}
    for rank in ["genus", "family", "order"]:
        values = set()
        for label in labels:
            info = species_info.get(label, {})
            if rank == "species":
                value = info.get("species") or label.replace("_", " ")
            else:
                value = info.get(rank) or info.get(f"{rank}_name")
            value_text = clean_text(value)
            if value_text:
                values.add(value_text)
        rank_values[rank] = values
    for rank in ["genus", "family", "order"]:
        if len(rank_values[rank]) == 1:
            return rank
    return "unresolved"


def rank_value(label: str, rank: str, species_info: dict[str, dict[str, Any]]) -> str:
    info = species_info.get(label, {})
    if rank == "species":
        return clean_text(info.get("species") or label.replace("_", " "))
    return clean_text(info.get(rank) or info.get(f"{rank}_name"))


def load_species_info(input_dir: Path, labels: set[str]) -> dict[str, dict[str, Any]]:
    info_path = input_dir / "species_info.json"
    if info_path.exists():
        raw = json.loads(info_path.read_text())
    else:
        raw = {}
    candidate_path = input_dir / "candidate_species.csv"
    if candidate_path.exists():
        rows = pd.read_csv(candidate_path).set_index("tree_label").to_dict(orient="index")
    else:
        rows = {}
    out: dict[str, dict[str, Any]] = {}
    for label in labels | set(raw) | set(rows):
        row = rows.get(label, {})
        info = raw.get(label, {}).copy()
        info.setdefault("species", clean_text(row.get("species_name", "") or label.replace("_", " ")))
        for rank in ["genus", "family", "order"]:
            info.setdefault(rank, clean_text(row.get(f"{rank}_name", "")))
            info.setdefault(f"{rank}_name", clean_text(row.get(f"{rank}_name", "")))
        out[label] = info
    return out


def load_species_sequences(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text())
    return {str(label): [str(seq) for seq in seqs] for label, seqs in raw.items()}


def build_clusters(
    species_sequences: dict[str, list[str]],
    species_info: dict[str, dict[str, Any]],
    normalization: str,
    min_length: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, list[str]]]:
    cluster_labels: dict[str, set[str]] = defaultdict(set)
    cluster_counts: Counter[str] = Counter()
    species_to_sequences: dict[str, list[str]] = defaultdict(list)
    for label, seqs in species_sequences.items():
        for seq in seqs:
            normalized = normalize_sequence(seq, normalization)
            if len(normalized) < min_length:
                continue
            cluster_labels[normalized].add(label)
            cluster_counts[normalized] += 1
            species_to_sequences[label].append(normalized)

    cluster_rows = []
    cluster_meta: dict[str, dict[str, Any]] = {}
    for idx, (sequence, labels) in enumerate(cluster_labels.items()):
        labels_sorted = sorted(labels)
        deepest = rank_from_cluster(labels, species_info)
        row = {
            "cluster_id": f"exact_{idx:08d}",
            "normalized_sequence": sequence,
            "sequence_length": len(sequence),
            "record_count": int(cluster_counts[sequence]),
            "species_count": len(labels_sorted),
            "species_labels": json.dumps(labels_sorted),
            "deepest_supported_rank": deepest,
        }
        for rank in ["genus", "family", "order"]:
            values = sorted({rank_value(label, rank, species_info) for label in labels if rank_value(label, rank, species_info)})
            row[f"{rank}_count"] = len(values)
            row[f"{rank}_values"] = json.dumps(values)
        cluster_rows.append(row)
        cluster_meta[sequence] = row

    return pd.DataFrame(cluster_rows), cluster_meta, dict(species_to_sequences)


def best_rank(ranks: list[str]) -> str:
    if not ranks:
        return "unresolved"
    return min(ranks, key=lambda rank: RANK_SCORE.get(rank, 999))


def worst_rank(ranks: list[str]) -> str:
    if not ranks:
        return "unresolved"
    return max(ranks, key=lambda rank: RANK_SCORE.get(rank, 999))


def species_summary(
    species_to_sequences: dict[str, list[str]],
    cluster_meta: dict[str, dict[str, Any]],
    species_info: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for label, sequences in sorted(species_to_sequences.items()):
        ranks = [cluster_meta[seq]["deepest_supported_rank"] for seq in sequences if seq in cluster_meta]
        exact_clusters = sorted(set(sequences))
        rows.append({
            "tree_label": label,
            "species_name": rank_value(label, "species", species_info),
            "genus_name": rank_value(label, "genus", species_info),
            "family_name": rank_value(label, "family", species_info),
            "order_name": rank_value(label, "order", species_info),
            "sequence_record_count": len(sequences),
            "exact_cluster_count": len(exact_clusters),
            "best_supported_rank": best_rank(ranks),
            "worst_supported_rank": worst_rank(ranks),
            "species_resolvable_any": any(rank == "species" for rank in ranks),
            "species_resolvable_all": bool(ranks) and all(rank == "species" for rank in ranks),
            "rank_trace": json.dumps(Counter(ranks), sort_keys=True),
        })
    return pd.DataFrame(rows)


def query_oracle_summary(
    input_dir: Path,
    reference_cluster_meta: dict[str, dict[str, Any]],
    observed_cluster_meta: dict[str, dict[str, Any]],
    species_info: dict[str, dict[str, Any]],
    normalization: str,
    min_length: int,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    query_path = input_dir / "zero_shot_queries.csv"
    if not query_path.exists():
        return None, None
    queries = pd.read_csv(query_path)
    rows = []
    for _, query in queries.iterrows():
        true_label = clean_text(query.get("tree_label", ""))
        normalized = normalize_sequence(query.get("nucleotides", ""), normalization)
        reference_meta = reference_cluster_meta.get(normalized) if len(normalized) >= min_length else None
        observed_meta = observed_cluster_meta.get(normalized) if len(normalized) >= min_length else None
        cluster_labels = set(json.loads(observed_meta["species_labels"])) if observed_meta is not None else set()
        deepest = observed_meta["deepest_supported_rank"] if observed_meta is not None else "unresolved"
        row = {
            "processid": clean_text(query.get("processid", "")),
            "true_tree_label": true_label,
            "true_species_name": clean_text(query.get("species_name", "")) or rank_value(true_label, "species", species_info),
            "true_genus_name": clean_text(query.get("genus_name", "")) or rank_value(true_label, "genus", species_info),
            "true_family_name": clean_text(query.get("family_name", "")) or rank_value(true_label, "family", species_info),
            "true_order_name": clean_text(query.get("order_name", "")) or rank_value(true_label, "order", species_info),
            "normalized_sequence_length": len(normalized),
            "reference_exact_cluster_found": reference_meta is not None,
            "observed_exact_cluster_found": observed_meta is not None,
            "true_species_in_observed_exact_cluster": true_label in cluster_labels,
            "observed_exact_cluster_species_count": len(cluster_labels),
            "deepest_supported_rank": deepest,
        }
        for rank in TAXONOMIC_RANKS:
            row[f"{rank}_oracle_supported"] = RANK_SCORE[deepest] <= RANK_SCORE[rank] and true_label in cluster_labels
        rows.append(row)

    out = pd.DataFrame(rows)
    metrics: dict[str, Any] = {
        "query_count": int(len(out)),
        "reference_exact_cluster_found_rate": float(out["reference_exact_cluster_found"].mean()) if len(out) else None,
        "observed_exact_cluster_found_rate": float(out["observed_exact_cluster_found"].mean()) if len(out) else None,
        "true_species_in_observed_exact_cluster_rate": float(out["true_species_in_observed_exact_cluster"].mean()) if len(out) else None,
        "deepest_supported_rank_counts": out["deepest_supported_rank"].value_counts(dropna=False).to_dict(),
    }
    for rank in TAXONOMIC_RANKS:
        metrics[f"{rank}_oracle_supported_rate"] = float(out[f"{rank}_oracle_supported"].mean()) if len(out) else None
    return out, metrics


def rank_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    counts = frame[column].value_counts(dropna=False).to_dict()
    return {rank: int(counts.get(rank, 0)) for rank in RANK_ORDER}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", choices=["acgt", "acgtn"], default="acgt")
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading species sequences from {args.input_dir / 'species_sequences.json'}")
    species_sequences = load_species_sequences(args.input_dir / "species_sequences.json")
    labels = set(species_sequences)
    logger.log(f"Loaded {len(species_sequences)} species sequence groups")
    species_info = load_species_info(args.input_dir, labels)
    logger.log("Building exact sequence clusters")
    clusters, cluster_meta, species_to_sequences = build_clusters(
        species_sequences,
        species_info,
        args.normalization,
        args.min_length,
    )
    logger.log("Building species resolvability summary")
    species = species_summary(species_to_sequences, cluster_meta, species_info)

    observed_species_sequences = {label: seqs[:] for label, seqs in species_sequences.items()}
    query_path = args.input_dir / "zero_shot_queries.csv"
    if query_path.exists():
        logger.log(f"Adding zero-shot query sequences from {query_path}")
        queries = pd.read_csv(query_path)
        for _, query in queries.iterrows():
            label = clean_text(query.get("tree_label", ""))
            seq = clean_text(query.get("nucleotides", ""))
            if label and seq:
                observed_species_sequences.setdefault(label, []).append(seq)
                species_info.setdefault(label, {})
                species_info[label].setdefault("species", clean_text(query.get("species_name", "")) or label.replace("_", " "))
                for rank in ["genus", "family", "order"]:
                    species_info[label].setdefault(rank, clean_text(query.get(f"{rank}_name", "")))
                    species_info[label].setdefault(f"{rank}_name", clean_text(query.get(f"{rank}_name", "")))
    logger.log("Building observed exact clusters including queries")
    observed_clusters, observed_cluster_meta, _ = build_clusters(
        observed_species_sequences,
        species_info,
        args.normalization,
        args.min_length,
    )
    logger.log("Building query oracle resolvability")
    query_oracle, query_metrics = query_oracle_summary(
        args.input_dir,
        cluster_meta,
        observed_cluster_meta,
        species_info,
        args.normalization,
        args.min_length,
    )

    cluster_csv = args.output_dir / "exact_sequence_clusters.csv"
    species_csv = args.output_dir / "species_resolvability.csv"
    logger.log(f"Writing exact clusters to {cluster_csv}")
    clusters.to_csv(cluster_csv, index=False)
    logger.log(f"Writing species resolvability to {species_csv}")
    species.to_csv(species_csv, index=False)
    query_csv = None
    if query_oracle is not None:
        query_csv = args.output_dir / "zero_shot_query_oracle_resolvability.csv"
        logger.log(f"Writing query oracle resolvability to {query_csv}")
        query_oracle.to_csv(query_csv, index=False)

    rank_summary_rows = []
    for column in ["best_supported_rank", "worst_supported_rank"]:
        counts = rank_counts(species, column)
        total = len(species)
        for rank in RANK_ORDER:
            rank_summary_rows.append({
                "level": "species",
                "summary": column,
                "rank": rank,
                "count": counts[rank],
                "fraction": counts[rank] / total if total else None,
            })
    cluster_counts = rank_counts(clusters, "deepest_supported_rank")
    for rank in RANK_ORDER:
        rank_summary_rows.append({
            "level": "exact_cluster",
            "summary": "deepest_supported_rank",
            "rank": rank,
            "count": cluster_counts[rank],
            "fraction": cluster_counts[rank] / len(clusters) if len(clusters) else None,
        })
    rank_summary_csv = args.output_dir / "rank_resolvability_summary.csv"
    logger.log(f"Writing rank summary to {rank_summary_csv}")
    pd.DataFrame(rank_summary_rows).to_csv(rank_summary_csv, index=False)
    observed_cluster_csv = args.output_dir / "observed_exact_sequence_clusters_with_queries.csv"
    logger.log(f"Writing observed clusters to {observed_cluster_csv}")
    observed_clusters.to_csv(observed_cluster_csv, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "normalization": args.normalization,
        "min_length": args.min_length,
        "species_with_sequences": int(len(species)),
        "sequence_records_used": int(sum(len(v) for v in species_to_sequences.values())),
        "exact_cluster_count": int(len(clusters)),
        "observed_exact_cluster_count_with_queries": int(len(observed_clusters)),
        "cluster_csv": str(cluster_csv),
        "observed_cluster_csv": str(observed_cluster_csv),
        "species_csv": str(species_csv),
        "rank_summary_csv": str(rank_summary_csv),
        "query_oracle_csv": str(query_csv) if query_csv else None,
        "query_oracle_metrics": query_metrics,
        "species_best_rank_counts": rank_counts(species, "best_supported_rank"),
        "species_worst_rank_counts": rank_counts(species, "worst_supported_rank"),
        "cluster_rank_counts": cluster_counts,
        "claim_boundary": (
            "Exact-identity resolvability is a marker-information diagnostic, not a model result. "
            "Near-exact clustering should be run separately before final paper claims."
        ),
    }
    manifest_path = args.output_dir / "resolvability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
