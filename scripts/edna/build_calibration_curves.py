#!/usr/bin/env python3
"""Build assignment-rate/accuracy curves for no-call calibration."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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


def safe_div(numer: int, denom: int) -> float | None:
    return None if denom == 0 else numer / denom


def load_zero_shot_candidate(predictions: Path, input_dir: Path) -> pd.DataFrame:
    pred = pd.read_csv(predictions)
    candidates = pd.read_csv(input_dir / "candidate_species.csv")
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")
    rows = []
    for _, row in pred.iterrows():
        labels = parse_labels(row.get("top_tree_labels"))
        pred_label = labels[0] if labels else None
        true_label = str(row.get("true_tree_label") or row.get("tree_label") or "").replace(" ", "_")
        try:
            score = float(row.get("pred_score"))
        except (TypeError, ValueError):
            score = np.nan
        out = {
            "processid": row.get("processid"),
            "score": score,
            "assigned": bool(pred_label),
        }
        for rank in RANKS:
            true_value = rank_value(true_label, rank, taxonomy)
            pred_value = rank_value(pred_label, rank, taxonomy)
            out[f"{rank}_eligible"] = bool(true_value and pred_value)
            out[f"{rank}_correct"] = bool(true_value and pred_value and true_value == pred_value)
        rows.append(out)
    return pd.DataFrame(rows)


def load_rank_prediction(predictions: Path, prefix: str, score_column: str | None) -> pd.DataFrame:
    pred = pd.read_csv(predictions)
    rows = []
    for _, row in pred.iterrows():
        if score_column:
            score = float(row.get(score_column, np.nan))
            if score_column.endswith("distance"):
                score = -score
        elif prefix == "blast":
            score = float(row.get("blast_pident", np.nan))
        elif prefix == "knn":
            score = -float(row.get("knn_distance", np.nan))
        else:
            score = float(row.get(f"{prefix}_conf_species", np.nan))
        out = {
            "processid": row.get("processid"),
            "score": score,
            "assigned": bool(row.get("has_hit", True)),
        }
        for rank in RANKS:
            true_value = row.get(f"true_{rank}")
            pred_value = row.get(f"{prefix}_pred_{rank}")
            if prefix == "blast":
                pred_value = row.get(f"blast_pred_{rank}")
            out[f"{rank}_eligible"] = nonempty(true_value) and nonempty(pred_value)
            out[f"{rank}_correct"] = nonempty(true_value) and nonempty(pred_value) and str(true_value) == str(pred_value)
        rows.append(out)
    return pd.DataFrame(rows)


def build_curve(scored: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    scored = scored.copy()
    scored = scored[scored["score"].notna()].copy()
    thresholds = sorted(set(float(scored["score"].quantile(q)) for q in quantiles), reverse=True)
    rows = []
    n_query = len(scored)
    for threshold in thresholds:
        assigned = scored[scored["assigned"] & (scored["score"] >= threshold)]
        row: dict[str, object] = {
            "threshold": threshold,
            "n_query": n_query,
            "n_assigned": int(len(assigned)),
            "assignment_rate": safe_div(int(len(assigned)), n_query),
        }
        for rank in RANKS:
            eligible = assigned[assigned[f"{rank}_eligible"]]
            row[f"{rank}_n"] = int(len(eligible))
            row[f"{rank}_accuracy"] = safe_div(int(eligible[f"{rank}_correct"].sum()), int(len(eligible)))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["zero_shot_candidate", "rank_prediction"], required=True)
    parser.add_argument("--input-dir", type=Path, help="Required for zero_shot_candidate mode.")
    parser.add_argument("--prefix", default="blast", help="Prediction prefix for rank_prediction mode: blast, knn, direct.")
    parser.add_argument("--score-column", help="Optional explicit score column. Distance columns are negated.")
    parser.add_argument("--quantiles", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "zero_shot_candidate":
        if args.input_dir is None:
            raise SystemExit("--input-dir is required for zero_shot_candidate mode")
        scored = load_zero_shot_candidate(args.predictions, args.input_dir)
    else:
        scored = load_rank_prediction(args.predictions, args.prefix, args.score_column)

    scored_path = args.output_dir / "calibration_scored_rows.csv"
    scored.to_csv(scored_path, index=False)
    curve = build_curve(scored, args.quantiles)
    curve_path = args.output_dir / "calibration_curve.csv"
    curve.to_csv(curve_path, index=False)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": str(args.predictions),
        "mode": args.mode,
        "input_dir": str(args.input_dir) if args.input_dir else None,
        "prefix": args.prefix,
        "score_column": args.score_column,
        "quantiles": args.quantiles,
        "scored_rows_csv": str(scored_path),
        "curve_csv": str(curve_path),
        "rows": int(len(scored)),
    }
    manifest_path = args.output_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {curve_path} and {scored_path}.")


if __name__ == "__main__":
    main()
