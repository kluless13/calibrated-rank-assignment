#!/usr/bin/env python3
"""Evaluate open-candidate predictions against Global_eDNA sample species sets."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RANKS = ["species", "genus", "family", "order"]


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


def rank_value(label: str | None, rank: str, taxonomy: dict[str, dict[str, object]]) -> str | None:
    if not label:
        return None
    if rank == "species":
        return label
    if rank == "genus":
        value = taxonomy.get(label, {}).get("genus_name")
        return str(value) if nonempty(value) else label.split("_", 1)[0]
    value = taxonomy.get(label, {}).get(f"{rank}_name")
    return str(value) if nonempty(value) else None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def safe_div(numer: int, denom: int) -> float | None:
    return None if denom == 0 else numer / denom


def summarize(rows: pd.DataFrame, prefix: str) -> dict[str, object]:
    if rows.empty:
        return {}
    summary: dict[str, object] = {"samples": int(len(rows))}
    for column in rows.columns:
        if not column.startswith(prefix):
            continue
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if len(values):
            summary[column] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_map_path = args.sample_query_map or (args.input_dir / "sample_query_map.csv")
    sample_map = pd.read_csv(sample_map_path)
    predictions = pd.read_csv(args.predictions)
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")

    pred_by_processid = {
        str(row["processid"]): parse_labels(row.get("top_tree_labels"))
        for _, row in predictions.iterrows()
        if "processid" in row.index
    }
    pred_by_sample_query = {}
    if {"sample_id", "query_processid"}.issubset(predictions.columns):
        pred_by_sample_query = {
            (str(row["sample_id"]), str(row["query_processid"])): parse_labels(row.get("top_tree_labels"))
            for _, row in predictions.iterrows()
        }

    asv_rows = []
    for _, row in sample_map.iterrows():
        processid = str(row["query_processid"])
        sample_id = str(row["sample_id"])
        true_label = str(row["true_tree_label"])
        ranked = pred_by_sample_query.get((sample_id, processid), pred_by_processid.get(processid, []))
        out = row.to_dict()
        out["n_predictions"] = len(ranked)
        out["top_prediction"] = ranked[0] if ranked else None
        for rank in RANKS:
            true_value = rank_value(true_label, rank, taxonomy)
            first_hit = None
            if true_value:
                for idx, label in enumerate(ranked, start=1):
                    if rank_value(label, rank, taxonomy) == true_value:
                        first_hit = idx
                        break
            out[f"{rank}_first_hit_rank"] = first_hit
            for k in args.top_k:
                out[f"{rank}_top{k}"] = bool(first_hit is not None and first_hit <= k)
        asv_rows.append(out)
    asv_eval = pd.DataFrame(asv_rows)
    asv_eval_path = args.output_dir / "global_edna_asv_level_validation.csv"
    asv_eval.to_csv(asv_eval_path, index=False)

    sample_rows = []
    grouped = sample_map.groupby("sample_id", dropna=False)
    for sample_id, sub in grouped:
        row: dict[str, object] = {"sample_id": sample_id, "query_rows": int(len(sub))}
        if "source_region_dir" in sub.columns:
            row["source_region_dir"] = "|".join(sorted(set(sub["source_region_dir"].dropna().astype(str))))
        for rank in RANKS:
            true_values = {
                value
                for label in sub["true_tree_label"].dropna().astype(str)
                if (value := rank_value(label, rank, taxonomy))
            }
            row[f"{rank}_true_richness"] = len(true_values)
            for k in args.top_k:
                pred_values: set[str] = set()
                for processid in sub["query_processid"].dropna().astype(str):
                    labels = pred_by_sample_query.get((str(sample_id), str(processid)), pred_by_processid.get(str(processid), []))
                    for label in labels[:k]:
                        value = rank_value(label, rank, taxonomy)
                        if value:
                            pred_values.add(value)
                row[f"{rank}_top{k}_pred_richness"] = len(pred_values)
                row[f"{rank}_top{k}_jaccard"] = jaccard(true_values, pred_values)
                row[f"{rank}_top{k}_recall"] = safe_div(len(true_values & pred_values), len(true_values))
                row[f"{rank}_top{k}_precision"] = safe_div(len(true_values & pred_values), len(pred_values))
        sample_rows.append(row)
    sample_eval = pd.DataFrame(sample_rows)
    sample_eval_path = args.output_dir / "global_edna_sample_level_validation.csv"
    sample_eval.to_csv(sample_eval_path, index=False)

    asv_metrics = {}
    for rank in RANKS:
        denom = int((asv_eval["n_predictions"] > 0).sum())
        asv_metrics[rank] = {
            "assigned_rows": denom,
            **{
                f"top{k}": safe_div(int(asv_eval[f"{rank}_top{k}"].sum()), denom)
                for k in args.top_k
            },
        }

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "predictions": str(args.predictions),
        "sample_query_map": str(sample_map_path),
        "asv_level_csv": str(asv_eval_path),
        "sample_level_csv": str(sample_eval_path),
        "sample_count": int(sample_eval["sample_id"].nunique()) if len(sample_eval) else 0,
        "sample_query_rows": int(len(sample_map)),
        "prediction_rows": int(len(predictions)),
        "top_k": args.top_k,
        "asv_metrics": asv_metrics,
        "sample_metric_summary": summarize(sample_eval, ""),
        "notes": [
            "Ground truth is the published Global_eDNA table assignment, not independent visual census.",
            "Sample-level top-k uses the union of per-ASV top-k predicted taxa within each sample.",
        ],
    }
    metrics_path = args.output_dir / "global_edna_validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Wrote {metrics_path}, {asv_eval_path}, and {sample_eval_path}.")


if __name__ == "__main__":
    main()
