#!/usr/bin/env python3
"""Rerank Global_eDNA predictions with an external occurrence-table prior."""
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


def load_occurrence_priors(prior_counts_csv: Path) -> dict[str, Counter[str]]:
    priors: dict[str, Counter[str]] = {}
    counts = pd.read_csv(prior_counts_csv)
    required = {"site_value", "tree_label", "occurrence_count"}
    missing = sorted(required - set(counts.columns))
    if missing:
        raise SystemExit(f"Occurrence prior table missing columns: {missing}")
    for _, row in counts.iterrows():
        if not (nonempty(row.get("site_value")) and nonempty(row.get("tree_label"))):
            continue
        try:
            count = float(row.get("occurrence_count", 0.0))
        except (TypeError, ValueError):
            count = 0.0
        if count <= 0:
            continue
        site_value = str(row["site_value"])
        label = str(row["tree_label"]).strip()
        priors.setdefault(site_value, Counter())[label] += count
    return priors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--occurrence-prior-counts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-column", default="site20")
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--absence-penalty", type=float, default=0.0)
    parser.add_argument("--output-top-k", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    sample_map = pd.read_csv(args.sample_query_map)
    if args.site_column not in sample_map.columns:
        raise SystemExit(f"Sample map is missing site column {args.site_column!r}")
    occurrence_priors = load_occurrence_priors(args.occurrence_prior_counts)

    pred_by_query = {}
    for _, row in predictions.iterrows():
        labels = parse_labels(row.get("top_tree_labels"))
        scores = parse_scores(row.get("top_scores"), len(labels))
        pred_by_query[str(row["processid"])] = (labels, scores)

    rows = []
    rows_with_prior = 0
    source_counts: Counter[str] = Counter()
    for _, map_row in sample_map.iterrows():
        sample_id = str(map_row["sample_id"])
        query_id = str(map_row["query_processid"])
        labels, seq_scores = pred_by_query.get(query_id, ([], []))
        site_value = str(map_row[args.site_column]) if nonempty(map_row.get(args.site_column)) else ""
        prior_counts = occurrence_priors.get(site_value, Counter())
        prior_source = args.site_column if prior_counts else "none"
        source_counts[prior_source] += 1
        if prior_counts:
            rows_with_prior += 1

        combined = []
        for label, score in zip(labels, seq_scores):
            prior_count = prior_counts.get(label, 0.0)
            prior_bonus = args.prior_weight * math.log1p(prior_count)
            penalty = args.absence_penalty if prior_count <= 0 else 0.0
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
                "occurrence_prior_candidate_count": int(len(prior_counts)),
                "occurrence_prior_source": prior_source,
            }
        )

    out_path = args.output_dir / "occurrence_prior_reranked_predictions.csv"
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
        "occurrence_prior_counts": str(args.occurrence_prior_counts),
        "reranked_predictions": str(out_path),
        "validation_dir": str(validation_dir),
        "site_column": args.site_column,
        "prior_weight": args.prior_weight,
        "absence_penalty": args.absence_penalty,
        "output_top_k": args.output_top_k,
        "rows": int(len(rows)),
        "rows_with_nonempty_occurrence_prior": int(rows_with_prior),
        "occurrence_site_prior_groups": int(len(occurrence_priors)),
        "prior_source_counts": dict(source_counts),
        "notes": [
            "This is a transparent external occurrence-prior re-ranker.",
            "It should be interpreted separately from sequence-only and RLS priors because OBIS sampling effort is heterogeneous.",
        ],
    }
    (args.output_dir / "occurrence_prior_rerank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
