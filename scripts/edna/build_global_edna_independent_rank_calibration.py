#!/usr/bin/env python3
"""Evaluate Global_eDNA rank/no-call thresholds on held-out sites.

The existing calibration curves are useful diagnostics, but they learn and
evaluate thresholds on the same Global_eDNA table. This script makes the next
step explicit: split samples by a deterministic site key, learn score
thresholds on calibration sites, and evaluate the locked thresholds on held-out
sites.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "edna" / "real_edna_queries" / "global_tropical_multisource_teleo"
DEFAULT_METHODS = ROOT / "configs" / "runs" / "2026-05-31-paper1-merged-global-edna-calibration-methods.json"
DEFAULT_OUT_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "global_edna_independent_rank_calibration"
RANKS = ("species", "genus", "family", "order")
TARGET_ACCURACIES = (50.0, 70.0, 80.0, 90.0, 95.0)
QUANTILES = tuple(np.linspace(0.0, 0.95, 20))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None", ""}


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
    for sep in ("|", ",", ";"):
        if sep in text:
            return [part.strip().replace(" ", "_") for part in text.split(sep) if part.strip()]
    return [text.replace(" ", "_")]


def rank_value(label: str | None, rank: str, taxonomy: dict[str, dict[str, object]]) -> str | None:
    if not label:
        return None
    label = label.replace(" ", "_")
    if rank == "species":
        return label
    if rank == "genus":
        value = taxonomy.get(label, {}).get("genus_name")
        return str(value) if nonempty(value) else label.split("_", 1)[0]
    value = taxonomy.get(label, {}).get(f"{rank}_name")
    return str(value) if nonempty(value) else None


def true_rank_value(row: pd.Series, rank: str, taxonomy: dict[str, dict[str, object]]) -> str | None:
    if rank == "species":
        value = row.get("true_tree_label")
        return str(value).replace(" ", "_") if nonempty(value) else None
    column = f"true_{rank}_name"
    if column in row and nonempty(row.get(column)):
        return str(row.get(column))
    return rank_value(str(row.get("true_tree_label", "")).replace(" ", "_"), rank, taxonomy)


def split_label(value: object) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return "calibration" if int(digest[:8], 16) % 2 == 0 else "evaluation"


def score_predictions(
    predictions_path: Path,
    sample_map: pd.DataFrame,
    taxonomy: dict[str, dict[str, object]],
    split_by: str,
) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path)
    if "query_processid" not in pred.columns:
        pred["query_processid"] = pred["processid"]

    sample_cols = [
        "sample_id",
        "query_processid",
        "true_tree_label",
        "true_species_name",
        "true_genus_name",
        "true_family_name",
        "true_order_name",
        split_by,
    ]
    sample_cols = [col for col in sample_cols if col in sample_map.columns]

    if "sample_id" in pred.columns:
        merged = pred.merge(
            sample_map[sample_cols],
            on=["sample_id", "query_processid"],
            how="left",
            suffixes=("", "_sample"),
        )
        for column in ("true_tree_label", "true_species_name"):
            sample_column = f"{column}_sample"
            if sample_column in merged.columns:
                merged[column] = merged[column].where(merged[column].map(nonempty), merged[sample_column])
    else:
        pred_small = pred.drop_duplicates("query_processid")
        merged = sample_map[sample_cols].merge(pred_small, on="query_processid", how="left", suffixes=("_sample", ""))
        if "true_tree_label_sample" in merged.columns:
            merged["true_tree_label"] = merged["true_tree_label_sample"]
        if "true_species_name_sample" in merged.columns:
            merged["true_species_name"] = merged["true_species_name_sample"]

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        labels = parse_labels(row.get("top_tree_labels"))
        pred_label = labels[0] if labels else None
        try:
            score = float(row.get("pred_score"))
        except (TypeError, ValueError):
            score = np.nan
        out: dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "query_processid": row.get("query_processid"),
            "split_key": row.get(split_by),
            "split": split_label(row.get(split_by)),
            "score": score,
            "assigned": bool(pred_label),
        }
        for rank in RANKS:
            true_value = true_rank_value(row, rank, taxonomy)
            pred_value = rank_value(pred_label, rank, taxonomy)
            out[f"{rank}_eligible"] = bool(true_value and pred_value)
            out[f"{rank}_correct"] = bool(true_value and pred_value and true_value == pred_value)
        rows.append(out)
    return pd.DataFrame(rows)


def curve(scored: pd.DataFrame, rank: str) -> pd.DataFrame:
    scored = scored[scored["score"].notna()].copy()
    if scored.empty:
        return pd.DataFrame()
    thresholds = sorted(set(float(scored["score"].quantile(q)) for q in QUANTILES), reverse=True)
    rows: list[dict[str, Any]] = []
    n_query = len(scored)
    for threshold in thresholds:
        assigned = scored[scored["assigned"] & (scored["score"] >= threshold)]
        eligible = assigned[assigned[f"{rank}_eligible"]]
        rows.append(
            {
                "threshold": threshold,
                "n_query": int(n_query),
                "n_assigned": int(len(assigned)),
                "assignment_rate_pct": 100.0 * len(assigned) / n_query if n_query else 0.0,
                "rank_n": int(len(eligible)),
                "rank_accuracy_pct": 100.0 * eligible[f"{rank}_correct"].sum() / len(eligible) if len(eligible) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_threshold(scored: pd.DataFrame, rank: str, threshold: float) -> dict[str, Any]:
    scored = scored[scored["score"].notna()].copy()
    assigned = scored[scored["assigned"] & (scored["score"] >= threshold)]
    eligible = assigned[assigned[f"{rank}_eligible"]]
    return {
        "eval_n_query": int(len(scored)),
        "eval_n_assigned": int(len(assigned)),
        "eval_assignment_rate_pct": 100.0 * len(assigned) / len(scored) if len(scored) else 0.0,
        "eval_rank_n": int(len(eligible)),
        "eval_rank_accuracy_pct": 100.0 * eligible[f"{rank}_correct"].sum() / len(eligible) if len(eligible) else np.nan,
    }


def method_metadata(method_name: str) -> dict[str, Any]:
    encoder = "ssm" if method_name.startswith("ssm_") else "cnn" if method_name.startswith("cnn_") else "unknown"
    if "sequence_only" in method_name:
        context = "sequence_only"
    elif "cooccurrence" in method_name:
        context = "learned_cooccurrence"
    else:
        context = "unknown"
    if "rls_obis" in method_name:
        prior_source = "rls_obis"
    elif "fishglob" in method_name:
        prior_source = "fishglob_public_50k"
    else:
        prior_source = ""
    prior_weight = ""
    for token in ("w025", "w050", "w100", "w200"):
        if token in method_name:
            prior_weight = {"w025": 0.25, "w050": 0.50, "w100": 1.0, "w200": 2.0}[token]
            break
    return {
        "encoder": encoder,
        "context": context,
        "prior_source": prior_source,
        "prior_weight": prior_weight,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--methods-json", type=Path, default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split-by", default="site20")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Reading sample map from {args.input_dir / 'sample_query_map.csv'}")
    sample_map = pd.read_csv(args.input_dir / "sample_query_map.csv")
    if args.split_by not in sample_map.columns:
        raise SystemExit(f"--split-by column not found in sample map: {args.split_by}")
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    taxonomy = candidates.set_index("tree_label").to_dict(orient="index")
    methods = json.loads(args.methods_json.read_text())

    all_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for method in methods:
        method_name = method["method"]
        predictions_path = ROOT / method["predictions"]
        if not predictions_path.exists():
            logger.log(f"Skipping missing predictions for {method_name}: {rel(predictions_path)}")
            continue
        logger.log(f"Scoring {method_name}")
        scored = score_predictions(predictions_path, sample_map, taxonomy, args.split_by)
        method_rows.append(
            {
                "method": method_name,
                **method_metadata(method_name),
                "rows": int(len(scored)),
                "calibration_rows": int((scored["split"] == "calibration").sum()),
                "evaluation_rows": int((scored["split"] == "evaluation").sum()),
                "split_by": args.split_by,
                "predictions": rel(predictions_path),
            }
        )
        calibration = scored[scored["split"] == "calibration"]
        evaluation = scored[scored["split"] == "evaluation"]
        for rank in RANKS:
            calibration_curve = curve(calibration, rank)
            for target in TARGET_ACCURACIES:
                eligible = calibration_curve[calibration_curve["rank_accuracy_pct"] >= target].copy()
                meta = {
                    "method": method_name,
                    **method_metadata(method_name),
                    "rank": rank,
                    "target_accuracy_pct": target,
                    "split_by": args.split_by,
                }
                if eligible.empty:
                    all_rows.append(
                        {
                            **meta,
                            "status": "no_calibration_threshold",
                            "threshold": "",
                            "calibration_assignment_rate_pct": 0.0,
                            "calibration_rank_accuracy_pct": "",
                            "calibration_n_assigned": 0,
                            "eval_assignment_rate_pct": 0.0,
                            "eval_rank_accuracy_pct": "",
                            "eval_n_assigned": 0,
                        }
                    )
                    continue
                best = eligible.sort_values(["assignment_rate_pct", "rank_accuracy_pct"], ascending=[False, False]).iloc[0]
                eval_stats = evaluate_threshold(evaluation, rank, float(best["threshold"]))
                all_rows.append(
                    {
                        **meta,
                        "status": "available",
                        "threshold": best["threshold"],
                        "calibration_assignment_rate_pct": best["assignment_rate_pct"],
                        "calibration_rank_accuracy_pct": best["rank_accuracy_pct"],
                        "calibration_n_assigned": best["n_assigned"],
                        "eval_assignment_rate_pct": eval_stats["eval_assignment_rate_pct"],
                        "eval_rank_accuracy_pct": eval_stats["eval_rank_accuracy_pct"],
                        "eval_n_assigned": eval_stats["eval_n_assigned"],
                    }
                )

    summary_path = args.output_dir / "global_edna_independent_rank_calibration_summary.csv"
    logger.log(f"Writing {summary_path}")
    write_csv(
        summary_path,
        all_rows,
        [
            "method",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "rank",
            "target_accuracy_pct",
            "split_by",
            "status",
            "threshold",
            "calibration_assignment_rate_pct",
            "calibration_rank_accuracy_pct",
            "calibration_n_assigned",
            "eval_assignment_rate_pct",
            "eval_rank_accuracy_pct",
            "eval_n_assigned",
        ],
    )

    method_path = args.output_dir / "global_edna_independent_rank_calibration_methods.csv"
    logger.log(f"Writing {method_path}")
    write_csv(
        method_path,
        method_rows,
        [
            "method",
            "encoder",
            "context",
            "prior_source",
            "prior_weight",
            "rows",
            "calibration_rows",
            "evaluation_rows",
            "split_by",
            "predictions",
        ],
    )

    manifest = {
        "generated_by": rel(Path(__file__)),
        "input_dir": rel(args.input_dir),
        "methods_json": rel(args.methods_json),
        "output_dir": rel(args.output_dir),
        "split_by": args.split_by,
        "target_accuracies_pct": TARGET_ACCURACIES,
        "quantiles": QUANTILES,
        "outputs": {
            "summary": rel(summary_path),
            "methods": rel(method_path),
        },
        "notes": [
            "Thresholds are learned on calibration site groups and evaluated on held-out site groups.",
            "This is a stronger diagnostic than same-table curves, but it is still not an external dataset validation.",
            "Rows report top-1 score-threshold rank accuracy, not top-k assignment accuracy.",
        ],
    }
    manifest_path = args.output_dir / "global_edna_independent_rank_calibration_manifest.json"
    logger.log(f"Writing {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
