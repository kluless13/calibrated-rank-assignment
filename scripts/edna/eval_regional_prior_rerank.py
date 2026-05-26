#!/usr/bin/env python3
"""Rerank sequence-model species candidates with simple regional priors."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def parse_list(value: object) -> list[str]:
    if not nonempty(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    for sep in ["|", ",", ";"]:
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


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
    parts = []
    for sep in ["|", ",", ";"]:
        if sep in text:
            parts = [part.strip() for part in text.split(sep) if part.strip()]
            break
    if not parts:
        parts = [text]
    scores = []
    for part in parts[:n]:
        try:
            scores.append(float(part))
        except ValueError:
            scores.append(0.0)
    return scores + [0.0] * max(0, n - len(scores))


def load_prior_counts(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    priors = pd.read_csv(path)
    by_event: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in priors.iterrows():
        key = (str(row["source_id"]), str(row["eventID"]))
        try:
            counts = json.loads(row.get("regional_candidate_counts_json", "{}"))
        except json.JSONDecodeError:
            counts = {}
        by_event[key] = {str(label): float(count) for label, count in counts.items()}
    return by_event


def rerank_row(
    row: pd.Series,
    prior_counts: dict[str, float],
    prior_weight: float,
    absence_penalty: float,
    top_k: int,
) -> tuple[list[str], list[float], list[float]]:
    labels = parse_list(row.get("top_tree_labels"))
    scores = parse_scores(row.get("top_scores"), len(labels))
    combined = []
    for label, score in zip(labels, scores):
        prior_count = prior_counts.get(label, 0.0)
        prior_bonus = prior_weight * math.log1p(prior_count)
        penalty = absence_penalty if prior_count <= 0 else 0.0
        combined.append(float(score) + prior_bonus - penalty)
    order = sorted(range(len(labels)), key=lambda idx: combined[idx], reverse=True)[:top_k]
    return [labels[idx] for idx in order], [scores[idx] for idx in order], [combined[idx] for idx in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--regional-priors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, help="Optional Stalder input dir for immediate metric evaluation.")
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--absence-penalty", type=float, default=0.0)
    parser.add_argument("--source-id-column", default="source_id")
    parser.add_argument("--event-id-column", default="eventID")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    prior_by_event = load_prior_counts(args.regional_priors)

    out_rows = []
    matched_events = 0
    for _, row in predictions.iterrows():
        source_id = str(row.get(args.source_id_column, ""))
        event_id = str(row.get(args.event_id_column, ""))
        prior_counts = prior_by_event.get((source_id, event_id), {})
        if prior_counts:
            matched_events += 1
        labels, seq_scores, combined_scores = rerank_row(
            row,
            prior_counts=prior_counts,
            prior_weight=args.prior_weight,
            absence_penalty=args.absence_penalty,
            top_k=args.top_k,
        )
        out = row.to_dict()
        out["top_tree_labels_sequence_only"] = row.get("top_tree_labels")
        out["top_scores_sequence_only"] = row.get("top_scores")
        out["top_tree_labels"] = json.dumps(labels)
        out["top_scores"] = json.dumps([round(score, 8) for score in combined_scores])
        out["top_sequence_scores_after_rerank"] = json.dumps([round(score, 8) for score in seq_scores])
        out["pred_tree_label"] = labels[0] if labels else None
        out["pred_score"] = combined_scores[0] if combined_scores else None
        out["regional_prior_candidate_count"] = len(prior_counts)
        out["regional_prior_matched_event"] = bool(prior_counts)
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    reranked_path = args.output_dir / "regional_prior_reranked_predictions.csv"
    out_df.to_csv(reranked_path, index=False)

    metrics_dir = None
    if args.input_dir:
        metrics_dir = args.output_dir / "zero_shot_metrics"
        subprocess.run(
            [
                sys.executable,
                "scripts/edna/eval_zero_shot_candidate_predictions.py",
                "--input-dir",
                str(args.input_dir),
                "--predictions",
                str(reranked_path),
                "--output-dir",
                str(metrics_dir),
            ],
            check=True,
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": str(args.predictions),
        "regional_priors": str(args.regional_priors),
        "reranked_predictions": str(reranked_path),
        "metrics_dir": str(metrics_dir) if metrics_dir else None,
        "rows": int(len(out_df)),
        "rows_with_matched_regional_prior_event": int(matched_events),
        "prior_weight": args.prior_weight,
        "absence_penalty": args.absence_penalty,
        "top_k": args.top_k,
        "notes": [
            "This is a transparent prior re-ranker, not a learned co-occurrence model.",
            "Rows need source_id and eventID to receive event-specific regional priors.",
        ],
    }
    manifest_path = args.output_dir / "regional_prior_rerank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {reranked_path}")


if __name__ == "__main__":
    main()
