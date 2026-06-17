#!/usr/bin/env python3
"""Evaluate same-sample co-occurrence prior without the query's own sequence score.

For each ASV/query, this builds a candidate list from other ASVs in the same
sample. It uses sequence predictions only to form the sample context and then
excludes the current query's own top labels. This is a prior-only decomposition
arm, not a final inference model.
"""
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

from eval_sample_cooccurrence_rerank import nonempty, parse_labels  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-top-k", type=int, default=10)
    parser.add_argument("--output-top-k", type=int, default=50)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Loading sequence predictions from {args.predictions}")
    predictions = pd.read_csv(args.predictions)
    logger.log(f"Loading sample map from {args.sample_query_map}")
    sample_map = pd.read_csv(args.sample_query_map)

    pred_by_query: dict[str, list[str]] = {}
    for _, row in predictions.iterrows():
        query_id = str(row.get("processid", row.get("query_processid", "")))
        if not nonempty(query_id):
            continue
        pred_by_query[query_id] = parse_labels(row.get("top_tree_labels"))[: args.support_top_k]
    logger.log(f"Loaded top-{args.support_top_k} support labels for {len(pred_by_query)} queries")

    rows = []
    empty_context = 0
    for sample_id, sub in sample_map.groupby("sample_id", sort=False, dropna=False):
        query_ids = [str(value) for value in sub["query_processid"].dropna().astype(str)]
        sample_support = Counter()
        own_support: dict[str, Counter[str]] = {}
        for query_id in query_ids:
            labels = pred_by_query.get(query_id, [])
            own_support[query_id] = Counter(labels)
            sample_support.update(labels)

        for _, map_row in sub.iterrows():
            query_id = str(map_row["query_processid"])
            support = sample_support.copy()
            support.subtract(own_support.get(query_id, Counter()))
            support = Counter({label: count for label, count in support.items() if count > 0})
            ranked = sorted(support.items(), key=lambda item: (-item[1], item[0]))[: args.output_top_k]
            labels = [label for label, _count in ranked]
            scores = [float(count) for _label, count in ranked]
            if not labels:
                empty_context += 1
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
                    "sample_context_candidates": int(len(support)),
                    "sample_query_count": int(len(query_ids)),
                }
            )

    pred_path = args.output_dir / "sample_cooccurrence_prior_only_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    logger.log(f"Wrote prior-only predictions to {pred_path}")

    validation_dir = args.output_dir / "global_edna_validation"
    logger.log(f"Running Global_eDNA validation into {validation_dir}")
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
        "sequence_predictions_used_as_sample_context": str(args.predictions),
        "predictions": str(pred_path),
        "validation_dir": str(validation_dir),
        "support_top_k": args.support_top_k,
        "output_top_k": args.output_top_k,
        "rows": int(len(rows)),
        "rows_with_empty_context_after_excluding_own_query": int(empty_context),
        "notes": [
            "This is a co-occurrence prior-only baseline for the current query.",
            "The current query's own top labels are excluded from its candidate list.",
            "Other ASVs' sequence predictions still define the sample context, so this is a same-sample community prior rather than an external ecological database.",
        ],
    }
    manifest_path = args.output_dir / "sample_cooccurrence_prior_only_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
