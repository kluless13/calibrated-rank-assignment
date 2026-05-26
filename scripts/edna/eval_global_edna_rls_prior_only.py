#!/usr/bin/env python3
"""Evaluate an RLS geographic prior alone on Global_eDNA sample queries."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rls-species-csv",
        type=Path,
        default=Path("/Users/kluless/Downloads/Global_eDNA/data/RLS/RLS_species_NEW.csv"),
    )
    parser.add_argument("--site-column", default="site20")
    parser.add_argument("--radius-km", type=float, default=250.0)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_map = pd.read_csv(args.sample_query_map)
    species_map = load_species_map(args.input_dir / "candidate_species.csv")
    by_site, geo_rows = load_rls_priors(args.rls_species_csv, species_map, args.site_column)
    sample_priors = build_sample_priors(sample_map, by_site, geo_rows, args.site_column, args.radius_km)

    rows = []
    nonempty = 0
    for _, map_row in sample_map.iterrows():
        sample_id = str(map_row["sample_id"])
        query_id = str(map_row["query_processid"])
        counts, source = sample_priors.get(sample_id, ({}, "none"))
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: args.top_k]
        labels = [label for label, _ in ranked]
        scores = [float(score) for _, score in ranked]
        if labels:
            nonempty += 1
        rows.append({
            "sample_id": sample_id,
            "query_processid": query_id,
            "processid": query_id,
            "true_tree_label": map_row.get("true_tree_label"),
            "true_species_name": map_row.get("true_species_name"),
            "top_tree_labels": json.dumps(labels),
            "top_scores": json.dumps(scores),
            "pred_tree_label": labels[0] if labels else None,
            "pred_score": scores[0] if scores else None,
            "prior_candidate_count": int(len(counts)),
            "prior_source": source,
        })

    pred_path = args.output_dir / "rls_prior_only_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    validation_dir = args.output_dir / "global_edna_validation"
    subprocess.run(
        [
            sys.executable,
            "scripts/edna/eval_global_edna_sample_validation.py",
            "--input-dir",
            str(args.input_dir),
            "--predictions",
            str(pred_path),
            "--sample-query-map",
            str(args.sample_query_map),
            "--output-dir",
            str(validation_dir),
        ],
        check=True,
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "sample_query_map": str(args.sample_query_map),
        "rls_species_csv": str(args.rls_species_csv),
        "site_column": args.site_column,
        "radius_km": args.radius_km,
        "top_k": args.top_k,
        "predictions": str(pred_path),
        "validation_dir": str(validation_dir),
        "rows": int(len(rows)),
        "rows_with_nonempty_prior": int(nonempty),
        "notes": [
            "This is an ecological prior-only baseline; no sequence information is used.",
            "It is useful for checking whether sequence+prior improvements are just prior leakage.",
        ],
    }
    (args.output_dir / "rls_prior_only_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
