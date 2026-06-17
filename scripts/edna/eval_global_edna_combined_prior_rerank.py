#!/usr/bin/env python3
"""Rerank Global_eDNA predictions with combined RLS and OBIS occurrence priors."""
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_global_edna_geographic_prior_rerank import (  # noqa: E402
    build_sample_priors,
    load_rls_priors,
    load_species_map,
    nonempty,
    parse_labels,
    parse_scores,
)
from eval_global_edna_occurrence_prior_rerank import load_occurrence_priors  # noqa: E402


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
    parser.add_argument("--occurrence-prior-counts", type=Path, required=True)
    parser.add_argument("--site-column", default="site20")
    parser.add_argument("--radius-km", type=float, default=250.0)
    parser.add_argument("--rls-weight", type=float, default=0.05)
    parser.add_argument("--obis-weight", type=float, default=0.05)
    parser.add_argument("--absence-penalty", type=float, default=0.0)
    parser.add_argument("--output-top-k", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    sample_map = pd.read_csv(args.sample_query_map)
    if args.site_column not in sample_map.columns:
        raise SystemExit(f"Sample map is missing site column {args.site_column!r}")

    species_map = load_species_map(args.input_dir / "candidate_species.csv")
    rls_by_site, rls_geo_rows = load_rls_priors(args.rls_species_csv, species_map, args.site_column)
    rls_sample_priors = build_sample_priors(sample_map, rls_by_site, rls_geo_rows, args.site_column, args.radius_km)
    obis_site_priors = load_occurrence_priors(args.occurrence_prior_counts)

    pred_by_query = {}
    for _, row in predictions.iterrows():
        labels = parse_labels(row.get("top_tree_labels"))
        scores = parse_scores(row.get("top_scores"), len(labels))
        pred_by_query[str(row["processid"])] = (labels, scores)

    rows = []
    rls_rows = 0
    obis_rows = 0
    source_counts: Counter[str] = Counter()
    for _, map_row in sample_map.iterrows():
        sample_id = str(map_row["sample_id"])
        query_id = str(map_row["query_processid"])
        labels, seq_scores = pred_by_query.get(query_id, ([], []))

        rls_counts, rls_source = rls_sample_priors.get(sample_id, (Counter(), "none"))
        if rls_counts:
            rls_rows += 1
        site_value = str(map_row[args.site_column]) if nonempty(map_row.get(args.site_column)) else ""
        obis_counts = obis_site_priors.get(site_value, Counter())
        obis_source = args.site_column if obis_counts else "none"
        if obis_counts:
            obis_rows += 1
        source_counts[f"rls={rls_source};obis={obis_source}"] += 1

        combined = []
        for label, score in zip(labels, seq_scores):
            rls_count = rls_counts.get(label, 0.0)
            obis_count = obis_counts.get(label, 0.0)
            prior_bonus = (
                args.rls_weight * math.log1p(rls_count)
                + args.obis_weight * math.log1p(obis_count)
            )
            penalty = args.absence_penalty if rls_count <= 0 and obis_count <= 0 else 0.0
            combined.append(float(score) + prior_bonus - penalty)

        order = sorted(range(len(labels)), key=lambda idx: combined[idx], reverse=True)[: args.output_top_k]
        ranked_labels = [labels[idx] for idx in order]
        ranked_scores = [combined[idx] for idx in order]
        rows.append(
            {
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
                "rls_prior_candidate_count": int(len(rls_counts)),
                "rls_prior_source": rls_source,
                "obis_prior_candidate_count": int(len(obis_counts)),
                "obis_prior_source": obis_source,
            }
        )

    out_path = args.output_dir / "combined_prior_reranked_predictions.csv"
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
        "occurrence_prior_counts": str(args.occurrence_prior_counts),
        "reranked_predictions": str(out_path),
        "validation_dir": str(validation_dir),
        "site_column": args.site_column,
        "radius_km": args.radius_km,
        "rls_weight": args.rls_weight,
        "obis_weight": args.obis_weight,
        "absence_penalty": args.absence_penalty,
        "output_top_k": args.output_top_k,
        "rows": int(len(rows)),
        "rows_with_nonempty_rls_prior": int(rls_rows),
        "rows_with_nonempty_obis_prior": int(obis_rows),
        "rls_site_prior_groups": int(len(rls_by_site)),
        "obis_site_prior_groups": int(len(obis_site_priors)),
        "prior_source_counts": dict(source_counts),
        "notes": [
            "This is a transparent combined RLS plus OBIS occurrence-prior re-ranker.",
            "It should be interpreted alongside RLS-only, OBIS-only, and prior-only controls.",
        ],
    }
    (args.output_dir / "combined_prior_rerank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
