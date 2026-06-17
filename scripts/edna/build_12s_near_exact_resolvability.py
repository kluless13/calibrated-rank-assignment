#!/usr/bin/env python3
"""Build VSEARCH near-exact 12S resolvability maps."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
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
    keep = set("ACGT") if mode == "acgt" else set("ACGTN")
    return "".join(ch for ch in str(seq).upper() if ch in keep)


def rank_value(label: str, rank: str, species_info: dict[str, dict[str, Any]]) -> str:
    info = species_info.get(label, {})
    if rank == "species":
        return clean_text(info.get("species") or label.replace("_", " "))
    return clean_text(info.get(rank) or info.get(f"{rank}_name"))


def rank_from_labels(labels: set[str], species_info: dict[str, dict[str, Any]]) -> str:
    if not labels:
        return "unresolved"
    if len(labels) == 1:
        return "species"
    for rank in ["genus", "family", "order"]:
        values = {rank_value(label, rank, species_info) for label in labels}
        values = {value for value in values if value}
        if len(values) == 1:
            return rank
    return "unresolved"


def load_species_info(input_dir: Path, labels: set[str]) -> dict[str, dict[str, Any]]:
    info_path = input_dir / "species_info.json"
    raw = json.loads(info_path.read_text()) if info_path.exists() else {}
    rows = {}
    candidate_path = input_dir / "candidate_species.csv"
    if candidate_path.exists():
        rows = pd.read_csv(candidate_path).set_index("tree_label").to_dict(orient="index")
    out: dict[str, dict[str, Any]] = {}
    for label in labels | set(raw) | set(rows):
        row = rows.get(label, {})
        info = raw.get(label, {}).copy()
        info.setdefault("species", clean_text(row.get("species_name", "")) or label.replace("_", " "))
        for rank in ["genus", "family", "order"]:
            info.setdefault(rank, clean_text(row.get(f"{rank}_name", "")))
            info.setdefault(f"{rank}_name", clean_text(row.get(f"{rank}_name", "")))
        out[label] = info
    return out


def load_records(input_dir: Path, normalization: str, min_length: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    species_sequences = json.loads((input_dir / "species_sequences.json").read_text())
    labels = set(species_sequences)
    query_path = input_dir / "zero_shot_queries.csv"
    queries = pd.read_csv(query_path) if query_path.exists() else pd.DataFrame()
    if len(queries):
        labels.update(clean_text(value) for value in queries["tree_label"].tolist())
    species_info = load_species_info(input_dir, {label for label in labels if label})

    records = []
    idx = 0
    for label, seqs in species_sequences.items():
        for seq in seqs:
            normalized = normalize_sequence(seq, normalization)
            if len(normalized) >= min_length:
                records.append({
                    "record_id": f"ref_{idx}",
                    "tree_label": label,
                    "sequence": normalized,
                    "source": "reference",
                    "processid": "",
                })
                idx += 1
    for _, row in queries.iterrows():
        label = clean_text(row.get("tree_label", ""))
        normalized = normalize_sequence(row.get("nucleotides", ""), normalization)
        if label and len(normalized) >= min_length:
            species_info.setdefault(label, {})
            species_info[label].setdefault("species", clean_text(row.get("species_name", "")) or label.replace("_", " "))
            for rank in ["genus", "family", "order"]:
                species_info[label].setdefault(rank, clean_text(row.get(f"{rank}_name", "")))
                species_info[label].setdefault(f"{rank}_name", clean_text(row.get(f"{rank}_name", "")))
            records.append({
                "record_id": f"query_{idx}",
                "tree_label": label,
                "sequence": normalized,
                "source": "query",
                "processid": clean_text(row.get("processid", "")),
            })
            idx += 1
    return records, species_info


def write_fasta(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(f">{record['record_id']}\n{record['sequence']}\n")


def parse_uc(path: Path) -> dict[str, str]:
    record_to_cluster = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            rec_type = parts[0]
            if rec_type not in {"S", "H"}:
                continue
            cluster_id = parts[1]
            record_id = parts[8]
            record_to_cluster[record_id] = cluster_id
    return record_to_cluster


def run_vsearch(records: list[dict[str, Any]], identity: float, threads: int, workdir: Path) -> dict[str, str]:
    fasta = workdir / "records.fasta"
    uc = workdir / f"clusters_{identity:.3f}.uc"
    centroids = workdir / f"centroids_{identity:.3f}.fasta"
    write_fasta(records, fasta)
    subprocess.run(
        [
            "vsearch",
            "--cluster_fast",
            str(fasta),
            "--id",
            str(identity),
            "--uc",
            str(uc),
            "--centroids",
            str(centroids),
            "--threads",
            str(threads),
            "--quiet",
        ],
        check=True,
    )
    return parse_uc(uc)


def summarize_threshold(
    records: list[dict[str, Any]],
    species_info: dict[str, dict[str, Any]],
    record_to_cluster: dict[str, str],
    identity: float,
    output_dir: Path,
) -> dict[str, Any]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cluster_id = record_to_cluster.get(record["record_id"])
        if cluster_id is not None:
            by_cluster[cluster_id].append(record)

    cluster_rows = []
    cluster_rank: dict[str, str] = {}
    cluster_labels: dict[str, set[str]] = {}
    cluster_has_reference: dict[str, bool] = {}
    for cluster_id, members in sorted(by_cluster.items(), key=lambda item: int(item[0])):
        labels = {member["tree_label"] for member in members}
        deepest = rank_from_labels(labels, species_info)
        cluster_rank[cluster_id] = deepest
        cluster_labels[cluster_id] = labels
        cluster_has_reference[cluster_id] = any(member["source"] == "reference" for member in members)
        row = {
            "identity": identity,
            "cluster_id": cluster_id,
            "record_count": len(members),
            "reference_record_count": sum(member["source"] == "reference" for member in members),
            "query_record_count": sum(member["source"] == "query" for member in members),
            "species_count": len(labels),
            "deepest_supported_rank": deepest,
            "species_labels": json.dumps(sorted(labels)),
        }
        for rank in ["genus", "family", "order"]:
            values = sorted({rank_value(label, rank, species_info) for label in labels if rank_value(label, rank, species_info)})
            row[f"{rank}_count"] = len(values)
            row[f"{rank}_values"] = json.dumps(values)
        cluster_rows.append(row)

    query_rows = []
    for record in records:
        if record["source"] != "query":
            continue
        cluster_id = record_to_cluster.get(record["record_id"], "")
        deepest = cluster_rank.get(cluster_id, "unresolved")
        labels = cluster_labels.get(cluster_id, set())
        row = {
            "identity": identity,
            "processid": record["processid"],
            "true_tree_label": record["tree_label"],
            "cluster_id": cluster_id,
            "cluster_has_reference": cluster_has_reference.get(cluster_id, False),
            "true_species_in_cluster": record["tree_label"] in labels,
            "cluster_species_count": len(labels),
            "deepest_supported_rank": deepest,
        }
        for rank in TAXONOMIC_RANKS:
            row[f"{rank}_oracle_supported"] = RANK_SCORE[deepest] <= RANK_SCORE[rank] and record["tree_label"] in labels
        query_rows.append(row)

    clusters = pd.DataFrame(cluster_rows)
    queries = pd.DataFrame(query_rows)
    ident_label = str(identity).replace(".", "p")
    cluster_csv = output_dir / f"clusters_id{ident_label}.csv"
    query_csv = output_dir / f"query_oracle_id{ident_label}.csv"
    clusters.to_csv(cluster_csv, index=False)
    queries.to_csv(query_csv, index=False)

    metrics = {
        "identity": identity,
        "cluster_count": int(len(clusters)),
        "record_count": int(len(records)),
        "query_count": int(len(queries)),
        "cluster_rank_counts": clusters["deepest_supported_rank"].value_counts(dropna=False).to_dict(),
        "query_rank_counts": queries["deepest_supported_rank"].value_counts(dropna=False).to_dict() if len(queries) else {},
        "query_cluster_has_reference_rate": float(queries["cluster_has_reference"].mean()) if len(queries) else None,
        "query_species_oracle_supported_rate": float(queries["species_oracle_supported"].mean()) if len(queries) else None,
        "query_genus_oracle_supported_rate": float(queries["genus_oracle_supported"].mean()) if len(queries) else None,
        "query_family_oracle_supported_rate": float(queries["family_oracle_supported"].mean()) if len(queries) else None,
        "query_order_oracle_supported_rate": float(queries["order_oracle_supported"].mean()) if len(queries) else None,
        "cluster_csv": str(cluster_csv),
        "query_csv": str(query_csv),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--identities", default="0.99,0.98,0.97,0.95")
    parser.add_argument("--normalization", choices=["acgt", "acgtn"], default="acgt")
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    if not shutil.which("vsearch"):
        raise SystemExit("vsearch is required for near-exact resolvability")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identities = [float(value) for value in args.identities.split(",") if value.strip()]
    logger.log(f"Loading records from {args.input_dir}")
    records, species_info = load_records(args.input_dir, args.normalization, args.min_length)
    logger.log(f"Loaded {len(records)} records across {len(species_info)} taxonomy labels")
    metrics = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for identity in identities:
            logger.log(f"Running VSEARCH near-exact clustering at identity={identity}")
            record_to_cluster = run_vsearch(records, identity, args.threads, tmpdir)
            logger.log(f"Summarizing clusters at identity={identity}")
            metrics.append(summarize_threshold(records, species_info, record_to_cluster, identity, args.output_dir))

    summary_csv = args.output_dir / "near_exact_resolvability_summary.csv"
    logger.log(f"Writing near-exact summary to {summary_csv}")
    pd.DataFrame(metrics).to_csv(summary_csv, index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "normalization": args.normalization,
        "min_length": args.min_length,
        "identities": identities,
        "record_count": len(records),
        "summary_csv": str(summary_csv),
        "metrics": metrics,
        "claim_boundary": (
            "Near-exact clustering is sensitive to marker window comparability. "
            "Use Teleo/MiFish-window-normalized inputs for final claims."
        ),
    }
    manifest_path = args.output_dir / "near_exact_resolvability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
