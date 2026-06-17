#!/usr/bin/env python3
"""Evaluate Stalder-style zero-shot species candidate predictions.

Prediction CSV contract:
  - must identify each query by `processid`
  - either provide one column containing ranked labels:
      `top_tree_labels` or `top_k_tree_labels` as JSON/list/comma/pipe text
    or provide rank columns:
      `rank_1`, `rank_2`, ...
    or provide a single top-1 column:
      `pred_tree_label`, `prediction`, or `tree_label_pred`

Tree labels use the TAXDNA/Stalder convention: Genus_species with underscores.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
RANKS = ["species", "genus", "family", "order"]
TOPK_DEFAULT = [1, 5, 10]


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def normalize_label(value: object) -> str | None:
    if not nonempty(value):
        return None
    text = str(value).strip().strip("'\"")
    text = text.replace(" ", "_")
    return text or None


def split_ranked_labels(value: object) -> list[str]:
    if not nonempty(value):
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [label for label in (normalize_label(item) for item in parsed) if label]
        except json.JSONDecodeError:
            pass
    parts = [part for part in re.split(r"[|,;]", text) if part.strip()]
    return [label for label in (normalize_label(part) for part in parts) if label]


def find_rank_columns(columns: list[str]) -> list[str]:
    def rank_key(column: str) -> int:
        match = re.search(r"(\d+)$", column)
        return int(match.group(1)) if match else 10**9

    rank_cols = [
        column
        for column in columns
        if re.fullmatch(r"(rank|top|pred|prediction)_?\d+", column)
    ]
    return sorted(rank_cols, key=rank_key)


def predictions_for_row(row: pd.Series, rank_columns: list[str]) -> list[str]:
    for column in ["top_tree_labels", "top_k_tree_labels", "ranked_tree_labels"]:
        if column in row.index and nonempty(row[column]):
            labels = split_ranked_labels(row[column])
            if labels:
                return labels
    if rank_columns:
        labels = [normalize_label(row[column]) for column in rank_columns if column in row.index]
        return [label for label in labels if label]
    for column in ["pred_tree_label", "prediction", "tree_label_pred", "rank_1", "top_1"]:
        if column in row.index and nonempty(row[column]):
            label = normalize_label(row[column])
            return [label] if label else []
    return []


def genus_from_tree_label(label: str | None) -> str | None:
    if not label:
        return None
    return label.split("_", 1)[0]


def rank_value(label: str | None, rank: str, taxonomy: dict[str, dict[str, object]]) -> str | None:
    if not label:
        return None
    if rank == "species":
        return label
    if rank == "genus":
        tax_value = taxonomy.get(label, {}).get("genus_name")
        return str(tax_value) if nonempty(tax_value) else genus_from_tree_label(label)
    column = f"{rank}_name"
    value = taxonomy.get(label, {}).get(column)
    return str(value) if nonempty(value) else None


def first_hit_rank(
    true_label: str,
    predictions: list[str],
    rank: str,
    taxonomy: dict[str, dict[str, object]],
    true_rank_value: object | None = None,
) -> int | None:
    if rank == "species":
        true_value = rank_value(true_label, rank, taxonomy)
    else:
        true_value = str(true_rank_value).strip() if nonempty(true_rank_value) else rank_value(true_label, rank, taxonomy)
    if not true_value:
        return None
    for idx, pred_label in enumerate(predictions, start=1):
        pred_value = rank_value(pred_label, rank, taxonomy)
        if pred_value and pred_value == true_value:
            return idx
    return None


def safe_div(numer: int, denom: int) -> float | None:
    if denom == 0:
        return None
    return numer / denom


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Stalder-style input directory containing zero_shot_queries.csv and candidate_species.csv.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Prediction CSV to evaluate. Omit with --write-template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=TOPK_DEFAULT,
        help="Top-k thresholds for species and rank metrics.",
    )
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a blank prediction template and exit.",
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    query_path = args.input_dir / "zero_shot_queries.csv"
    candidate_path = args.input_dir / "candidate_species.csv"
    if not query_path.exists() or not candidate_path.exists():
        raise SystemExit(f"Missing zero_shot_queries.csv or candidate_species.csv in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading queries from {query_path}")
    queries = pd.read_csv(query_path)
    logger.log(f"Loading candidates from {candidate_path}")
    candidates = pd.read_csv(candidate_path)
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")
    logger.log(f"Loaded {len(queries)} queries and {len(candidates)} candidates")

    if args.write_template:
        template = queries[[
            "processid",
            "tree_label",
            "species_name",
            "genus_name",
            "family_name",
            "order_name",
        ]].copy()
        template = template.rename(columns={"tree_label": "true_tree_label"})
        template["top_tree_labels"] = ""
        template["score_trace"] = ""
        template_path = args.output_dir / "zero_shot_prediction_template.csv"
        template.to_csv(template_path, index=False)
        manifest = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(args.input_dir),
            "template_csv": str(template_path),
            "query_count": int(len(template)),
            "candidate_count": int(len(candidates)),
            "notes": [
                "Fill top_tree_labels with ranked Genus_species labels as JSON/list/comma/pipe text.",
                "At minimum, provide a top-1 label in pred_tree_label or top_tree_labels.",
            ],
        }
        (args.output_dir / "zero_shot_prediction_template_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        logger.log(f"Wrote prediction template to {template_path}")
        logger.done(Path(__file__).name)
        print(f"Wrote {template_path}.")
        return

    if args.predictions is None:
        raise SystemExit("--predictions is required unless --write-template is set")

    logger.log(f"Loading predictions from {args.predictions}")
    predictions = pd.read_csv(args.predictions)
    if "processid" not in predictions.columns:
        raise SystemExit("Prediction CSV must contain processid")

    rank_columns = find_rank_columns(predictions.columns.tolist())
    logger.log(f"Evaluating {len(predictions)} prediction rows with top-k {args.top_k}")
    pred_by_processid = {}
    for _, row in predictions.iterrows():
        pred_by_processid[str(row["processid"])] = predictions_for_row(row, rank_columns)

    per_query_rows = []
    for _, query in queries.iterrows():
        processid = str(query["processid"])
        true_label = normalize_label(query["tree_label"])
        ranked = pred_by_processid.get(processid, [])
        row = {
            "processid": processid,
            "true_tree_label": true_label,
            "true_species_name": query.get("species_name"),
            "true_genus_name": query.get("genus_name"),
            "true_family_name": query.get("family_name"),
            "true_order_name": query.get("order_name"),
            "n_predictions": len(ranked),
            "top_prediction": ranked[0] if ranked else None,
        }
        for rank in RANKS:
            query_rank_value = query.get(f"{rank}_name")
            hit = first_hit_rank(true_label, ranked, rank, taxonomy, query_rank_value) if true_label else None
            row[f"{rank}_first_hit_rank"] = hit
            for k in args.top_k:
                row[f"{rank}_top{k}"] = bool(hit is not None and hit <= k)
        per_query_rows.append(row)

    per_query = pd.DataFrame(per_query_rows)
    per_query_path = args.output_dir / "zero_shot_candidate_per_query.csv"
    logger.log(f"Writing per-query metrics to {per_query_path}")
    per_query.to_csv(per_query_path, index=False)

    metrics: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "predictions": str(args.predictions),
        "query_count": int(len(per_query)),
        "queries_with_predictions": int((per_query["n_predictions"] > 0).sum()),
        "candidate_count": int(len(candidates)),
        "top_k": args.top_k,
        "metrics": {},
        "per_query_csv": str(per_query_path),
    }
    for rank in RANKS:
        rank_metrics = {}
        eligible = per_query[f"{rank}_first_hit_rank"].notna() | (per_query["n_predictions"] > 0)
        denom = int(eligible.sum())
        rank_metrics["eligible_queries"] = denom
        for k in args.top_k:
            rank_metrics[f"top{k}"] = safe_div(int(per_query[f"{rank}_top{k}"].sum()), denom)
        first_hits = per_query[f"{rank}_first_hit_rank"].dropna().astype(int)
        rank_metrics["mean_first_hit_rank"] = float(first_hits.mean()) if len(first_hits) else math.nan
        metrics["metrics"][rank] = rank_metrics

    metrics_path = args.output_dir / "zero_shot_candidate_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing aggregate metrics to {metrics_path}")
    logger.done(Path(__file__).name)
    print(f"Wrote {metrics_path} and {per_query_path}.")


if __name__ == "__main__":
    main()
