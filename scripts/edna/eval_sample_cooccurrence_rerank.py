#!/usr/bin/env python3
"""Rerank per-ASV predictions using support from other ASVs in the same sample."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-top-k", type=int, default=10)
    parser.add_argument("--output-top-k", type=int, default=50)
    parser.add_argument("--support-weight", type=float, default=0.05)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    sample_map = pd.read_csv(args.sample_query_map)

    pred_by_query = {}
    for _, row in predictions.iterrows():
        labels = parse_labels(row.get("top_tree_labels"))
        scores = parse_scores(row.get("top_scores"), len(labels))
        pred_by_query[str(row["processid"])] = (labels, scores)

    rows = []
    for sample_id, sub in sample_map.groupby("sample_id", dropna=False):
        query_ids = [str(value) for value in sub["query_processid"].dropna().astype(str)]
        support = Counter()
        query_support_labels = {}
        for query_id in query_ids:
            labels, _ = pred_by_query.get(query_id, ([], []))
            labels = labels[: args.support_top_k]
            query_support_labels[query_id] = Counter(labels)
            support.update(labels)
        for _, map_row in sub.iterrows():
            query_id = str(map_row["query_processid"])
            labels, seq_scores = pred_by_query.get(query_id, ([], []))
            own_support = query_support_labels.get(query_id, Counter())
            combined = []
            for label, score in zip(labels, seq_scores):
                other_support = max(0, support.get(label, 0) - own_support.get(label, 0))
                combined.append(float(score) + args.support_weight * math.log1p(other_support))
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
                "pred_tree_label": ranked_labels[0] if ranked_labels else None,
                "pred_score": ranked_scores[0] if ranked_scores else None,
                "sample_support_candidates": int(len(support)),
            })

    out_path = args.output_dir / "sample_cooccurrence_reranked_predictions.csv"
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
        "reranked_predictions": str(out_path),
        "validation_dir": str(validation_dir),
        "support_top_k": args.support_top_k,
        "output_top_k": args.output_top_k,
        "support_weight": args.support_weight,
        "rows": int(len(rows)),
        "notes": [
            "This is a transparent same-sample co-occurrence posterior, not a learned ecological model.",
            "Support for a query excludes that query's own top-k labels.",
        ],
    }
    (args.output_dir / "sample_cooccurrence_rerank_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
