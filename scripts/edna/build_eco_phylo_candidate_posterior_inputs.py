#!/usr/bin/env python3
"""Build candidate-level Eco-Phylo posterior inputs for Global_eDNA.

The earlier Eco-Phylo posterior prototype used one row per method/query: each
method had already compressed its evidence down to a single top-1 candidate.
That is too weak for a real posterior. This builder expands existing prediction
files into top-k candidate rows and attaches candidate-specific evidence:

- sequence/model score and original sequence-only score when available;
- BLAST candidate pident/rank evidence when the candidate is present in BLAST;
- RLS and OBIS occurrence counts for the sample/site;
- marker-resolvability oracle labels for the query;
- candidate taxonomy/reference metadata;
- correctness by species/genus/family/order.

It does not invent p-distance, range, or co-occurrence scores when those are
not present. Missing evidence is marked explicitly so the next model can learn
from availability rather than from silent zeros.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_global_edna_geographic_prior_rerank import (  # noqa: E402
    build_sample_priors,
    load_rls_priors,
    load_species_map,
)
from eval_global_edna_occurrence_prior_rerank import load_occurrence_priors  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "edna" / "real_edna_queries" / "global_tropical_multisource_teleo"
DEFAULT_METHODS = ROOT / "configs" / "runs" / "2026-05-31-paper1-merged-global-edna-calibration-methods.json"
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "eco_phylo_posterior"
    / "candidate_level"
)
DEFAULT_BLAST = (
    ROOT
    / "results"
    / "edna"
    / "global_tropical_validation"
    / "blast_train_reference"
    / "global_edna_blast_zero_shot_predictions.csv"
)
DEFAULT_OBIS_PRIOR = (
    ROOT
    / "data"
    / "edna"
    / "raw"
    / "real_edna"
    / "global_obis_range_prior_site20_pad05"
    / "obis_site_prior_counts.csv"
)
DEFAULT_RLS_SPECIES = Path("/Users/kluless/Downloads/Global_eDNA/data/RLS/RLS_species_NEW.csv")
DEFAULT_QUERY_ORACLE = (
    ROOT
    / "results"
    / "edna"
    / "resolvability"
    / "multisource_teleo_exact_acgt"
    / "zero_shot_query_oracle_resolvability.csv"
)
RANKS = ("species", "genus", "family", "order")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def clean(value: object) -> str:
    if not nonempty(value):
        return ""
    return str(value).strip()


def normalize_label(value: object) -> str:
    return clean(value).replace(" ", "_")


def as_float(value: object) -> float | None:
    if not nonempty(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool01(value: object) -> int | str:
    if not nonempty(value):
        return ""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    try:
        return int(float(text) != 0)
    except ValueError:
        return ""


def parse_json_list(value: object) -> list[Any]:
    if not nonempty(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    for sep in ("|", ",", ";"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def parse_scores(value: object, n: int) -> list[float | None]:
    values = parse_json_list(value)
    scores: list[float | None] = []
    for item in values[:n]:
        scores.append(as_float(item))
    scores.extend([None] * max(0, n - len(scores)))
    return scores


def split_label(value: object) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return "calibration" if int(digest[:8], 16) % 2 == 0 else "evaluation"


def rank_value(label: str | None, rank: str, taxonomy: dict[str, dict[str, object]]) -> str:
    if not label:
        return ""
    label = label.replace(" ", "_")
    if rank == "species":
        return label
    if rank == "genus":
        value = taxonomy.get(label, {}).get("genus_name")
        return clean(value) or label.split("_", 1)[0]
    return clean(taxonomy.get(label, {}).get(f"{rank}_name"))


def true_rank_value(row: pd.Series, rank: str, taxonomy: dict[str, dict[str, object]]) -> str:
    if rank == "species":
        return normalize_label(row.get("true_tree_label"))
    value = clean(row.get(f"true_{rank}_name"))
    if value:
        return value
    return rank_value(normalize_label(row.get("true_tree_label")), rank, taxonomy)


def method_metadata(method_name: str) -> dict[str, Any]:
    if method_name.startswith("ssm_"):
        encoder = "ssm"
    elif method_name.startswith("cnn_"):
        encoder = "cnn"
    elif method_name.startswith("blast_"):
        encoder = "blast"
    else:
        encoder = "unknown"

    if "sequence_only" in method_name:
        context = "sequence_only"
    elif "rls_obis" in method_name and "cooccurrence" not in method_name:
        context = "sequence_plus_geography_occurrence"
    elif "rls_geo" in method_name:
        context = "sequence_plus_geography"
    elif "same_sample_cooccurrence" in method_name:
        context = "sequence_plus_same_sample_cooccurrence"
    elif "cooccurrence" in method_name:
        context = "learned_cooccurrence"
    else:
        context = "unknown"

    if "rls_obis" in method_name:
        prior_source = "rls_obis"
    elif "rls_geo" in method_name:
        prior_source = "rls"
    elif "fishglob" in method_name:
        prior_source = "fishglob_public_50k"
    elif "same_sample" in method_name:
        prior_source = "same_sample"
    else:
        prior_source = ""

    prior_weight: float | str = ""
    for token, weight in (
        ("w001", 0.01),
        ("w005", 0.05),
        ("w010", 0.10),
        ("w020", 0.20),
        ("w025", 0.25),
        ("w050", 0.50),
        ("w100", 1.0),
        ("w200", 2.0),
        ("w1", 1.0),
        ("w5", 5.0),
    ):
        if token in method_name:
            prior_weight = weight
            break

    if context == "sequence_only":
        evidence_family = "sequence"
    elif "cooccurrence" in context:
        evidence_family = "sequence_plus_ecology"
    elif "geography" in context:
        evidence_family = "sequence_plus_geography"
    else:
        evidence_family = "other"

    return {
        "encoder": encoder,
        "context": context,
        "prior_source": prior_source,
        "prior_weight": prior_weight,
        "evidence_family": evidence_family,
        "sequence_score_available": 1,
        "tree_candidate_evidence_available": int(encoder in {"ssm", "cnn"}),
        "blast_candidate_evidence_available": int(encoder == "blast"),
        "geography_prior_available": int(prior_source in {"rls", "rls_obis"}),
        "occurrence_prior_available": int(prior_source == "rls_obis"),
        "cooccurrence_prior_available": int("cooccurrence" in context),
    }


def extra_methods() -> list[dict[str, str]]:
    return [
        {
            "method": "blast_sequence_only",
            "predictions": "results/edna/global_tropical_validation/blast_train_reference/global_edna_blast_zero_shot_predictions.csv",
        },
        {
            "method": "blast_rls_geo_w1",
            "predictions": "results/edna/global_tropical_validation/blast_train_reference_rls_geo_w1/geographic_prior_reranked_predictions.csv",
        },
        {
            "method": "blast_rls_geo_w5",
            "predictions": "results/edna/global_tropical_validation/blast_train_reference_rls_geo_w5/geographic_prior_reranked_predictions.csv",
        },
        {
            "method": "ssm_rls_geo_site20_w005",
            "predictions": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_rls_geo_site20_w005/geographic_prior_reranked_predictions.csv",
        },
        {
            "method": "ssm_rls_obis_site20_w005",
            "predictions": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_rls_obis_site20_w005/combined_prior_reranked_predictions.csv",
        },
        {
            "method": "ssm_same_sample_cooccurrence_w005",
            "predictions": "results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_cooccurrence_w005/sample_cooccurrence_reranked_predictions.csv",
        },
    ]


def merge_predictions(predictions_path: Path, sample_map: pd.DataFrame, split_by: str) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path)
    if "query_processid" not in pred.columns:
        pred["query_processid"] = pred.get("processid")

    sample_cols = [
        "sample_id",
        "query_processid",
        "true_tree_label",
        "true_species_name",
        "true_genus_name",
        "true_family_name",
        "true_order_name",
        "read_count",
        "asv_count",
        split_by,
    ]
    sample_cols = [column for column in sample_cols if column in sample_map.columns]
    if "sample_id" in pred.columns:
        merged = pred.merge(
            sample_map[sample_cols],
            on=["sample_id", "query_processid"],
            how="left",
            suffixes=("", "_sample"),
        )
        for column in ("true_tree_label", "true_species_name", "true_genus_name", "true_family_name", "true_order_name"):
            sample_column = f"{column}_sample"
            if sample_column in merged.columns:
                merged[column] = merged[column].where(merged[column].map(nonempty), merged[sample_column])
    else:
        pred_small = pred.drop_duplicates("query_processid")
        merged = sample_map[sample_cols].merge(pred_small, on="query_processid", how="left", suffixes=("_sample", ""))
        for column in ("true_tree_label", "true_species_name", "true_genus_name", "true_family_name", "true_order_name"):
            sample_column = f"{column}_sample"
            if sample_column in merged.columns:
                merged[column] = merged[sample_column]
    return merged


def load_query_oracle(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for _, row in table.iterrows():
        processid = clean(row.get("processid"))
        if not processid:
            continue
        out[processid] = {
            "marker_normalized_sequence_length": row.get("normalized_sequence_length", ""),
            "marker_reference_exact_cluster_found": as_bool01(row.get("reference_exact_cluster_found")),
            "marker_observed_exact_cluster_found": as_bool01(row.get("observed_exact_cluster_found")),
            "marker_true_species_in_observed_exact_cluster": as_bool01(row.get("true_species_in_observed_exact_cluster")),
            "marker_observed_exact_cluster_species_count": row.get("observed_exact_cluster_species_count", ""),
            "marker_deepest_supported_rank": clean(row.get("deepest_supported_rank")),
            "marker_species_oracle_supported": as_bool01(row.get("species_oracle_supported")),
            "marker_genus_oracle_supported": as_bool01(row.get("genus_oracle_supported")),
            "marker_family_oracle_supported": as_bool01(row.get("family_oracle_supported")),
            "marker_order_oracle_supported": as_bool01(row.get("order_oracle_supported")),
        }
    return out


def load_blast_lookup(path: Path, logger: ProgressLogger) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        logger.log(f"BLAST lookup file missing: {rel(path)}")
        return {}
    logger.log(f"Loading BLAST candidate lookup from {rel(path)}")
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for chunk in pd.read_csv(path, chunksize=5000):
        for _, row in chunk.iterrows():
            processid = clean(row.get("processid")) or clean(row.get("query_processid"))
            if not processid:
                continue
            labels = [normalize_label(label) for label in parse_json_list(row.get("top_tree_labels")) if nonempty(label)]
            scores = parse_scores(row.get("top_scores"), len(labels))
            row_lookup: dict[str, dict[str, Any]] = {}
            for idx, (label, score) in enumerate(zip(labels, scores), start=1):
                row_lookup[label] = {
                    "blast_candidate_rank": idx,
                    "blast_candidate_pident": score,
                    "blast_candidate_in_top50": 1,
                }
            lookup[processid] = row_lookup
    logger.log(f"Loaded BLAST candidate evidence for {len(lookup):,} queries")
    return lookup


def make_prior_lookups(
    input_dir: Path,
    sample_map: pd.DataFrame,
    rls_species_csv: Path,
    obis_prior_counts: Path,
    site_column: str,
    radius_km: float,
    logger: ProgressLogger,
) -> tuple[dict[str, tuple[Counter[str], str]], dict[str, Counter[str]]]:
    rls_sample_priors: dict[str, tuple[Counter[str], str]] = {}
    if rls_species_csv.exists():
        try:
            species_map = load_species_map(input_dir / "candidate_species.csv")
            rls_by_site, rls_geo_rows = load_rls_priors(rls_species_csv, species_map, site_column)
            rls_sample_priors = build_sample_priors(sample_map, rls_by_site, rls_geo_rows, site_column, radius_km)
            logger.log(
                "Loaded RLS candidate priors for "
                f"{sum(1 for counts, _ in rls_sample_priors.values() if counts):,} samples"
            )
        except Exception as exc:  # pragma: no cover - defensive audit logging
            logger.log(f"RLS prior loading failed; continuing without RLS counts: {exc}")
    else:
        logger.log(f"RLS species table not found; continuing without RLS counts: {rls_species_csv}")

    obis_site_priors: dict[str, Counter[str]] = {}
    if obis_prior_counts.exists():
        try:
            obis_site_priors = load_occurrence_priors(obis_prior_counts)
            logger.log(f"Loaded OBIS occurrence priors for {len(obis_site_priors):,} site groups")
        except Exception as exc:  # pragma: no cover - defensive audit logging
            logger.log(f"OBIS prior loading failed; continuing without OBIS counts: {exc}")
    else:
        logger.log(f"OBIS prior-count table not found; continuing without OBIS counts: {rel(obis_prior_counts)}")
    return rls_sample_priors, obis_site_priors


def add_method_score_normalizers(
    path: Path,
    method_stats: dict[str, dict[str, float]],
    logger: ProgressLogger,
) -> Path:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    logger.log(f"Adding method-normalized candidate score columns to {rel(path)}")
    with gzip.open(path, "rt", newline="") as src, gzip.open(tmp_path, "wt", newline="") as dst:
        reader = csv.DictReader(src)
        assert reader.fieldnames is not None
        fieldnames = list(reader.fieldnames) + [
            "candidate_score_method_z",
            "sequence_only_score_method_z",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            method = row.get("method", "")
            stats = method_stats.get(method, {})
            for source_col, out_col, mean_key, std_key in (
                ("candidate_score", "candidate_score_method_z", "candidate_score_mean", "candidate_score_std"),
                (
                    "sequence_only_score",
                    "sequence_only_score_method_z",
                    "sequence_only_score_mean",
                    "sequence_only_score_std",
                ),
            ):
                value = as_float(row.get(source_col))
                mean = stats.get(mean_key)
                std = stats.get(std_key)
                row[out_col] = "" if value is None or mean is None or not std else (value - mean) / std
            writer.writerow(row)
    tmp_path.replace(path)
    return path


def fieldnames() -> list[str]:
    base = [
        "method",
        "encoder",
        "context",
        "prior_source",
        "prior_weight",
        "evidence_family",
        "predictions",
        "sample_id",
        "query_processid",
        "split_by",
        "split_key",
        "calibration_split",
        "read_count",
        "asv_count",
        "true_tree_label",
        "candidate_tree_label",
        "candidate_rank",
        "candidate_score",
        "candidate_score_method_z",
        "candidate_score_delta_from_top1",
        "candidate_score_ratio_to_top1",
        "top_candidate_count",
        "score_direction",
        "sequence_only_score",
        "sequence_only_score_method_z",
        "sequence_only_rank",
        "sequence_only_available",
        "blast_candidate_pident",
        "blast_candidate_rank",
        "blast_candidate_in_top50",
        "pdistance_available",
        "candidate_pdistance",
        "rls_prior_count",
        "rls_prior_log1p",
        "rls_prior_supported",
        "rls_prior_source",
        "obis_prior_count",
        "obis_prior_log1p",
        "obis_prior_supported",
        "obis_prior_source",
        "geographic_prior_candidate_count",
        "occurrence_prior_candidate_count",
        "rls_prior_candidate_count",
        "obis_prior_candidate_count",
        "sample_query_count",
        "sample_support_candidates",
        "candidate_species_name",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
        "candidate_has_reference_sequence",
        "candidate_reference_sequence_count",
        "true_species",
        "true_genus",
        "true_family",
        "true_order",
        "marker_normalized_sequence_length",
        "marker_reference_exact_cluster_found",
        "marker_observed_exact_cluster_found",
        "marker_true_species_in_observed_exact_cluster",
        "marker_observed_exact_cluster_species_count",
        "marker_deepest_supported_rank",
        "marker_species_oracle_supported",
        "marker_genus_oracle_supported",
        "marker_family_oracle_supported",
        "marker_order_oracle_supported",
    ]
    for rank in RANKS:
        if f"candidate_{rank}" not in base:
            base.append(f"candidate_{rank}")
        base.extend([f"{rank}_eligible", f"{rank}_candidate_correct"])
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--methods-json", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blast-predictions", type=Path, default=DEFAULT_BLAST)
    parser.add_argument("--rls-species-csv", type=Path, default=DEFAULT_RLS_SPECIES)
    parser.add_argument("--obis-prior-counts", type=Path, default=DEFAULT_OBIS_PRIOR)
    parser.add_argument("--query-oracle", type=Path, default=DEFAULT_QUERY_ORACLE)
    parser.add_argument("--split-by", default="site20")
    parser.add_argument("--site-column", default="site20")
    parser.add_argument("--radius-km", type=float, default=250.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-rows", type=int, default=1000)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Reading sample map from {rel(args.input_dir / 'sample_query_map.csv')}")
    sample_map = pd.read_csv(args.input_dir / "sample_query_map.csv")
    if args.split_by not in sample_map.columns:
        raise SystemExit(f"--split-by column not found in sample map: {args.split_by}")
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")
    query_oracle = load_query_oracle(args.query_oracle)
    logger.log(f"Loaded query-level marker oracle rows: {len(query_oracle):,}")
    blast_lookup = load_blast_lookup(args.blast_predictions, logger)
    rls_sample_priors, obis_site_priors = make_prior_lookups(
        input_dir=args.input_dir,
        sample_map=sample_map,
        rls_species_csv=args.rls_species_csv,
        obis_prior_counts=args.obis_prior_counts,
        site_column=args.site_column,
        radius_km=args.radius_km,
        logger=logger,
    )

    methods = json.loads(args.methods_json.read_text()) + extra_methods()
    candidate_path = args.output_dir / f"eco_phylo_candidate_features_top{args.top_k}.csv.gz"
    sample_path = args.output_dir / f"eco_phylo_candidate_features_top{args.top_k}_sample.csv"
    inventory_path = args.output_dir / "eco_phylo_candidate_method_inventory.csv"
    manifest_path = args.output_dir / "eco_phylo_candidate_feature_manifest.json"
    fields = fieldnames()

    method_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    total_rows = 0
    method_stats_accumulator: dict[str, dict[str, list[float]]] = {}

    logger.log(f"Writing candidate features to {rel(candidate_path)}")
    with gzip.open(candidate_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for method in methods:
            method_name = method["method"]
            predictions_path = ROOT / method["predictions"]
            if not predictions_path.exists():
                logger.log(f"Skipping missing predictions for {method_name}: {rel(predictions_path)}")
                continue

            meta = method_metadata(method_name)
            logger.log(f"Expanding top-{args.top_k} candidates for {method_name}")
            merged = merge_predictions(predictions_path, sample_map, args.split_by)
            method_candidate_rows = 0
            method_query_rows = 0
            rows_with_sequence_only = 0
            rows_with_blast = 0
            rows_with_rls = 0
            rows_with_obis = 0
            stats = method_stats_accumulator.setdefault(
                method_name, {"candidate_score": [], "sequence_only_score": []}
            )

            for _, row in merged.iterrows():
                labels = [normalize_label(label) for label in parse_json_list(row.get("top_tree_labels")) if nonempty(label)]
                if not labels:
                    continue
                scores = parse_scores(row.get("top_scores"), len(labels))
                method_query_rows += 1
                top_score = scores[0] if scores else None
                query_id = clean(row.get("query_processid")) or clean(row.get("processid"))
                sample_id = clean(row.get("sample_id"))
                split_key = clean(row.get(args.split_by))
                true_label = normalize_label(row.get("true_tree_label"))
                sequence_only_labels = [
                    normalize_label(label)
                    for label in parse_json_list(row.get("top_tree_labels_sequence_only"))
                    if nonempty(label)
                ]
                sequence_only_scores = parse_scores(row.get("top_scores_sequence_only"), len(sequence_only_labels))
                sequence_lookup = {
                    label: {"rank": idx, "score": score}
                    for idx, (label, score) in enumerate(zip(sequence_only_labels, sequence_only_scores), start=1)
                }
                if not sequence_lookup:
                    sequence_lookup = {
                        label: {"rank": idx, "score": score}
                        for idx, (label, score) in enumerate(zip(labels, scores), start=1)
                    }
                rls_counts, rls_source = rls_sample_priors.get(sample_id, (Counter(), "none"))
                site_value = clean(row.get(args.site_column))
                obis_counts = obis_site_priors.get(site_value, Counter())
                obis_source = args.site_column if obis_counts else "none"
                blast_for_query = blast_lookup.get(query_id, {})
                oracle = query_oracle.get(query_id, {})

                for rank_idx, (candidate_label, candidate_score) in enumerate(
                    zip(labels[: args.top_k], scores[: args.top_k]),
                    start=1,
                ):
                    sequence = sequence_lookup.get(candidate_label, {})
                    sequence_score = sequence.get("score")
                    sequence_rank = sequence.get("rank", "")
                    blast_evidence = blast_for_query.get(candidate_label, {})
                    rls_count = float(rls_counts.get(candidate_label, 0.0))
                    obis_count = float(obis_counts.get(candidate_label, 0.0))
                    candidate_tax = taxonomy.get(candidate_label, {})
                    out: dict[str, Any] = {
                        "method": method_name,
                        **meta,
                        "predictions": rel(predictions_path),
                        "sample_id": sample_id,
                        "query_processid": query_id,
                        "split_by": args.split_by,
                        "split_key": split_key,
                        "calibration_split": split_label(split_key),
                        "read_count": row.get("read_count", ""),
                        "asv_count": row.get("asv_count", ""),
                        "true_tree_label": true_label,
                        "candidate_tree_label": candidate_label,
                        "candidate_rank": rank_idx,
                        "candidate_score": candidate_score,
                        "candidate_score_delta_from_top1": (
                            "" if top_score is None or candidate_score is None else float(top_score) - float(candidate_score)
                        ),
                        "candidate_score_ratio_to_top1": (
                            ""
                            if top_score in {None, 0}
                            or candidate_score is None
                            else float(candidate_score) / float(top_score)
                        ),
                        "top_candidate_count": len(labels),
                        "score_direction": "higher_is_better_by_output_order",
                        "sequence_only_score": sequence_score,
                        "sequence_only_rank": sequence_rank,
                        "sequence_only_available": int(bool(sequence)),
                        "blast_candidate_pident": blast_evidence.get("blast_candidate_pident", ""),
                        "blast_candidate_rank": blast_evidence.get("blast_candidate_rank", ""),
                        "blast_candidate_in_top50": blast_evidence.get("blast_candidate_in_top50", 0),
                        "pdistance_available": 0,
                        "candidate_pdistance": "",
                        "rls_prior_count": rls_count,
                        "rls_prior_log1p": math.log1p(rls_count),
                        "rls_prior_supported": int(rls_count > 0),
                        "rls_prior_source": rls_source,
                        "obis_prior_count": obis_count,
                        "obis_prior_log1p": math.log1p(obis_count),
                        "obis_prior_supported": int(obis_count > 0),
                        "obis_prior_source": obis_source,
                        "geographic_prior_candidate_count": row.get("geographic_prior_candidate_count", ""),
                        "occurrence_prior_candidate_count": row.get("occurrence_prior_candidate_count", ""),
                        "rls_prior_candidate_count": row.get("rls_prior_candidate_count", ""),
                        "obis_prior_candidate_count": row.get("obis_prior_candidate_count", ""),
                        "sample_query_count": row.get("sample_query_count", ""),
                        "sample_support_candidates": row.get("sample_support_candidates", ""),
                        "candidate_species_name": clean(candidate_tax.get("species_name")),
                        "candidate_genus": rank_value(candidate_label, "genus", taxonomy),
                        "candidate_family": rank_value(candidate_label, "family", taxonomy),
                        "candidate_order": rank_value(candidate_label, "order", taxonomy),
                        "candidate_has_reference_sequence": candidate_tax.get("has_reference_sequence", ""),
                        "candidate_reference_sequence_count": candidate_tax.get("reference_sequence_count", ""),
                        **oracle,
                    }
                    for rank in RANKS:
                        true_value = true_rank_value(row, rank, taxonomy)
                        candidate_value = rank_value(candidate_label, rank, taxonomy)
                        out[f"true_{rank}"] = true_value
                        out[f"candidate_{rank}"] = candidate_value
                        out[f"{rank}_eligible"] = int(bool(true_value and candidate_value))
                        out[f"{rank}_candidate_correct"] = int(bool(true_value and candidate_value and true_value == candidate_value))

                    writer.writerow(out)
                    total_rows += 1
                    method_candidate_rows += 1
                    if len(sample_rows) < args.sample_rows:
                        sample_rows.append(out.copy())
                    if candidate_score is not None:
                        stats["candidate_score"].append(float(candidate_score))
                    if sequence_score is not None:
                        stats["sequence_only_score"].append(float(sequence_score))
                    rows_with_sequence_only += int(bool(sequence))
                    rows_with_blast += int(bool(blast_evidence))
                    rows_with_rls += int(rls_count > 0)
                    rows_with_obis += int(obis_count > 0)

            method_rows.append(
                {
                    "method": method_name,
                    **meta,
                    "predictions": rel(predictions_path),
                    "query_rows": method_query_rows,
                    "candidate_rows": method_candidate_rows,
                    "candidate_rows_with_sequence_only_score": rows_with_sequence_only,
                    "candidate_rows_with_blast_candidate_evidence": rows_with_blast,
                    "candidate_rows_with_rls_prior": rows_with_rls,
                    "candidate_rows_with_obis_prior": rows_with_obis,
                }
            )
            logger.log(f"Added {method_candidate_rows:,} candidate rows for {method_name}")

    method_stats: dict[str, dict[str, float]] = {}
    for method_name, values in method_stats_accumulator.items():
        row_stats: dict[str, float] = {}
        for column, column_values in values.items():
            if column_values:
                series = pd.Series(column_values, dtype="float64")
                row_stats[f"{column}_mean"] = float(series.mean())
                row_stats[f"{column}_std"] = float(series.std(ddof=0)) or 1.0
        method_stats[method_name] = row_stats
        for row in method_rows:
            if row["method"] == method_name:
                row.update(row_stats)

    add_method_score_normalizers(candidate_path, method_stats, logger)

    logger.log(f"Writing candidate feature sample to {rel(sample_path)}")
    with sample_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample_rows)

    logger.log(f"Writing candidate method inventory to {rel(inventory_path)}")
    inventory_fields = [
        "method",
        "encoder",
        "context",
        "prior_source",
        "prior_weight",
        "evidence_family",
        "predictions",
        "query_rows",
        "candidate_rows",
        "candidate_rows_with_sequence_only_score",
        "candidate_rows_with_blast_candidate_evidence",
        "candidate_rows_with_rls_prior",
        "candidate_rows_with_obis_prior",
        "candidate_score_mean",
        "candidate_score_std",
        "sequence_only_score_mean",
        "sequence_only_score_std",
    ]
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(method_rows)

    manifest = {
        "generated_by": rel(Path(__file__)),
        "input_dir": rel(args.input_dir),
        "methods_json": rel(args.methods_json),
        "output_dir": rel(args.output_dir),
        "top_k": args.top_k,
        "split_by": args.split_by,
        "site_column": args.site_column,
        "query_oracle": rel(args.query_oracle),
        "blast_predictions": rel(args.blast_predictions),
        "rls_species_csv": str(args.rls_species_csv),
        "obis_prior_counts": rel(args.obis_prior_counts),
        "methods_seen": len(method_rows),
        "candidate_feature_rows": total_rows,
        "outputs": {
            "candidate_features": rel(candidate_path),
            "candidate_features_sample": rel(sample_path),
            "method_inventory": rel(inventory_path),
        },
        "notes": [
            "This is a candidate-level posterior input table, not a final posterior model.",
            "p-distance evidence is marked unavailable for Global_eDNA 12S because no candidate-level 12S p-distance table exists yet.",
            "RLS and OBIS counts are attached as candidate-specific evidence where local priors are available.",
            "BLAST pident/rank evidence is attached only when the candidate appears in the BLAST top-50 for that query.",
            "Default top-k is intentionally compact because local disk is nearly full; rerun with larger top-k on Vast storage if needed.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {rel(manifest_path)}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
