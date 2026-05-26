#!/usr/bin/env python3
"""Evaluate an external occurrence prior alone on Global_eDNA sample queries."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_global_edna_occurrence_prior_rerank import load_occurrence_priors, nonempty  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--occurrence-prior-counts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-column", default="site20")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_map = pd.read_csv(args.sample_query_map)
    if args.site_column not in sample_map.columns:
        raise SystemExit(f"Sample map is missing site column {args.site_column!r}")
    occurrence_priors = load_occurrence_priors(args.occurrence_prior_counts)

    rows = []
    nonempty_rows = 0
    source_counts: Counter[str] = Counter()
    for _, map_row in sample_map.iterrows():
        sample_id = str(map_row["sample_id"])
        query_id = str(map_row["query_processid"])
        site_value = str(map_row[args.site_column]) if nonempty(map_row.get(args.site_column)) else ""
        counts = occurrence_priors.get(site_value, Counter())
        source = args.site_column if counts else "none"
        source_counts[source] += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: args.top_k]
        labels = [label for label, _ in ranked]
        scores = [float(score) for _, score in ranked]
        if labels:
            nonempty_rows += 1
        rows.append(
            {
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
            }
        )

    pred_path = args.output_dir / "occurrence_prior_only_predictions.csv"
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
        "occurrence_prior_counts": str(args.occurrence_prior_counts),
        "site_column": args.site_column,
        "top_k": args.top_k,
        "predictions": str(pred_path),
        "validation_dir": str(validation_dir),
        "rows": int(len(rows)),
        "rows_with_nonempty_prior": int(nonempty_rows),
        "occurrence_site_prior_groups": int(len(occurrence_priors)),
        "prior_source_counts": dict(source_counts),
        "notes": [
            "This is an external occurrence prior-only baseline; no sequence information is used.",
            "It measures how much geographic occurrence context can explain the real eDNA labels by itself.",
        ],
    }
    (args.output_dir / "occurrence_prior_only_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
