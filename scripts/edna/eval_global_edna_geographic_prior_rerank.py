#!/usr/bin/env python3
"""Rerank Global_eDNA predictions with an independent RLS geographic prior."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RLS_META_COLUMNS = {
    "SurveyID",
    "Station",
    "SiteCode",
    "Site name",
    "SiteName",
    "SiteLat",
    "SiteLong",
    "Country",
    "State",
    "Location",
    "Ecoregion",
    "Province",
    "province",
    "Realm",
    "realm",
    "SurveyDate",
    "Depth",
    "site35",
    "site30",
    "site25",
    "site20",
    "site10",
    "site5",
    "coral_cover",
}


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def parse_labels(value: object) -> list[str]:
    if not nonempty(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip().replace(" ", "_") for item in parsed if nonempty(item)]
        except json.JSONDecodeError:
            pass
    for sep in ["|", ",", ";"]:
        if sep in text:
            return [part.strip().replace(" ", "_") for part in text.split(sep) if part.strip()]
    return [text.replace(" ", "_")]


def parse_scores(value: object, n: int) -> list[float]:
    if not nonempty(value):
        return [0.0] * n
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                scores = [float(item) for item in parsed[:n]]
                return scores + [0.0] * max(0, n - len(scores))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    scores = []
    for sep in ["|", ",", ";"]:
        if sep in text:
            for part in text.split(sep)[:n]:
                try:
                    scores.append(float(part))
                except ValueError:
                    scores.append(0.0)
            break
    if not scores:
        try:
            scores = [float(text)]
        except ValueError:
            scores = [0.0]
    return scores + [0.0] * max(0, n - len(scores))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def load_species_map(candidate_path: Path) -> dict[str, str]:
    candidates = pd.read_csv(candidate_path)
    species_map: dict[str, str] = {}
    for _, row in candidates.iterrows():
        if nonempty(row.get("species_name")) and nonempty(row.get("tree_label")):
            species_map[str(row["species_name"]).strip()] = str(row["tree_label"]).strip()
    return species_map


def row_species_counts(row: pd.Series, species_columns: list[str], species_map: dict[str, str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for col in species_columns:
        try:
            value = float(row[col])
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            counts[species_map[col]] += value
    return counts


def load_rls_priors(rls_species_csv: Path, species_map: dict[str, str], site_column: str) -> tuple[dict[str, Counter[str]], list[dict[str, object]]]:
    rls = pd.read_csv(rls_species_csv, sep=";", encoding="utf-8-sig")
    species_columns = [col for col in rls.columns if col not in RLS_META_COLUMNS and col in species_map]
    if not species_columns:
        raise SystemExit(f"No candidate species columns matched in {rls_species_csv}")

    by_site: dict[str, Counter[str]] = {}
    geo_rows: list[dict[str, object]] = []
    for _, row in rls.iterrows():
        counts = row_species_counts(row, species_columns, species_map)
        if not counts:
            continue
        if site_column in rls.columns and nonempty(row.get(site_column)):
            by_site.setdefault(str(row[site_column]), Counter()).update(counts)
        lat = pd.to_numeric(pd.Series([row.get("SiteLat")]), errors="coerce").iloc[0]
        lon = pd.to_numeric(pd.Series([row.get("SiteLong")]), errors="coerce").iloc[0]
        if not pd.isna(lat) and not pd.isna(lon):
            geo_rows.append({"lat": float(lat), "lon": float(lon), "counts": counts})
    return by_site, geo_rows


def build_sample_priors(
    sample_map: pd.DataFrame,
    by_site: dict[str, Counter[str]],
    geo_rows: list[dict[str, object]],
    site_column: str,
    radius_km: float,
) -> dict[str, tuple[Counter[str], str]]:
    sample_cols = ["sample_id", "latitude_start_clean", "longitude_start_clean", site_column]
    sample_cols = [col for col in sample_cols if col in sample_map.columns]
    samples = sample_map[sample_cols].drop_duplicates("sample_id")
    priors: dict[str, tuple[Counter[str], str]] = {}
    for _, sample in samples.iterrows():
        sample_id = str(sample["sample_id"])
        if site_column in sample.index and nonempty(sample.get(site_column)):
            site_value = str(sample[site_column])
            if site_value in by_site:
                priors[sample_id] = (by_site[site_value], site_column)
                continue
        lat = pd.to_numeric(pd.Series([sample.get("latitude_start_clean")]), errors="coerce").iloc[0]
        lon = pd.to_numeric(pd.Series([sample.get("longitude_start_clean")]), errors="coerce").iloc[0]
        counts: Counter[str] = Counter()
        if not pd.isna(lat) and not pd.isna(lon):
            for geo in geo_rows:
                if haversine_km(float(lat), float(lon), float(geo["lat"]), float(geo["lon"])) <= radius_km:
                    counts.update(geo["counts"])
        source = f"radius_{radius_km:g}km" if counts else "none"
        priors[sample_id] = (counts, source)
    return priors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rls-species-csv",
        type=Path,
        default=Path("/Users/kluless/Downloads/Global_eDNA/data/RLS/RLS_species_NEW.csv"),
    )
    parser.add_argument("--site-column", default="site35")
    parser.add_argument("--radius-km", type=float, default=250.0)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--absence-penalty", type=float, default=0.0)
    parser.add_argument("--output-top-k", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    sample_map = pd.read_csv(args.sample_query_map)
    species_map = load_species_map(args.input_dir / "candidate_species.csv")
    by_site, geo_rows = load_rls_priors(args.rls_species_csv, species_map, args.site_column)
    sample_priors = build_sample_priors(sample_map, by_site, geo_rows, args.site_column, args.radius_km)

    pred_by_query = {}
    for _, row in predictions.iterrows():
        labels = parse_labels(row.get("top_tree_labels"))
        scores = parse_scores(row.get("top_scores"), len(labels))
        pred_by_query[str(row["processid"])] = (labels, scores)

    rows = []
    matched_rows = 0
    source_counts: Counter[str] = Counter()
    for _, map_row in sample_map.iterrows():
        sample_id = str(map_row["sample_id"])
        query_id = str(map_row["query_processid"])
        labels, seq_scores = pred_by_query.get(query_id, ([], []))
        prior_counts, prior_source = sample_priors.get(sample_id, (Counter(), "none"))
        source_counts[prior_source] += 1
        if prior_counts:
            matched_rows += 1

        combined = []
        for label, score in zip(labels, seq_scores):
            prior_count = prior_counts.get(label, 0.0)
            prior_bonus = args.prior_weight * math.log1p(prior_count)
            penalty = args.absence_penalty if prior_count <= 0 else 0.0
            combined.append(float(score) + prior_bonus - penalty)
        order = sorted(range(len(labels)), key=lambda idx: combined[idx], reverse=True)[: args.output_top_k]
        ranked_labels = [labels[idx] for idx in order]
        ranked_scores = [combined[idx] for idx in order]
        rows.append({
            "sample_id": sample_id,
            "query_processid": query_id,
            "processid": query_id,
            "true_tree_label": map_row.get("true_tree_label"),
            "true_species_name": map_row.get("true_species_name"),
            "top_tree_labels": json.dumps(ranked_labels),
            "top_scores": json.dumps([round(score, 8) for score in ranked_scores]),
            "top_tree_labels_sequence_only": json.dumps(labels[: args.output_top_k]),
            "top_scores_sequence_only": json.dumps([round(score, 8) for score in seq_scores[: args.output_top_k]]),
            "pred_tree_label": ranked_labels[0] if ranked_labels else None,
            "pred_score": ranked_scores[0] if ranked_scores else None,
            "geographic_prior_candidate_count": int(len(prior_counts)),
            "geographic_prior_source": prior_source,
        })

    out_path = args.output_dir / "geographic_prior_reranked_predictions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    validation_dir = args.output_dir / "global_edna_validation"
    subprocess.run(
        [
            sys.executable,
            "scripts/edna/eval_global_edna_sample_validation.py",
            "--input-dir",
            str(args.input_dir),
            "--predictions",
            str(out_path),
            "--sample-query-map",
            str(args.sample_query_map),
            "--output-dir",
            str(validation_dir),
        ],
        check=True,
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": str(args.predictions),
        "sample_query_map": str(args.sample_query_map),
        "input_dir": str(args.input_dir),
        "rls_species_csv": str(args.rls_species_csv),
        "reranked_predictions": str(out_path),
        "validation_dir": str(validation_dir),
        "site_column": args.site_column,
        "radius_km": args.radius_km,
        "prior_weight": args.prior_weight,
        "absence_penalty": args.absence_penalty,
        "output_top_k": args.output_top_k,
        "rows": int(len(rows)),
        "rows_with_nonempty_geographic_prior": int(matched_rows),
        "rls_site_prior_groups": int(len(by_site)),
        "rls_geo_rows_with_candidates": int(len(geo_rows)),
        "prior_source_counts": dict(source_counts),
        "notes": [
            "This uses the independent RLS visual census matrix bundled with Global_eDNA.",
            "It is a transparent geographic prior re-ranker, not a learned ecological model.",
        ],
    }
    (args.output_dir / "geographic_prior_rerank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
