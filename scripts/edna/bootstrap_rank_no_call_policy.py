#!/usr/bin/env python3
"""Bootstrap the missing-reference-aware rank/no-call policy."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_rank_adaptive_calibration import CONSENSUS_POLICY_FEATURES, RANKS
from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "rank_adaptive_calibration"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assign_policy(evaluation: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    assigned_rank = []
    assigned_correct = []
    species_call_false = []
    for _, query in evaluation.iterrows():
        chosen = "no_call"
        correct = False
        for rank in RANKS:
            feature = CONSENSUS_POLICY_FEATURES[rank]
            threshold = thresholds.get(rank)
            value = query.get(feature)
            if threshold is None or pd.isna(value):
                continue
            if float(value) >= threshold:
                chosen = rank
                correct = bool(query.get(f"{rank}_top1", False))
                break
        assigned_rank.append(chosen)
        assigned_correct.append(correct)
        species_call_false.append(chosen == "species" and not correct)
    return pd.DataFrame(
        {
            "assigned_rank": assigned_rank,
            "correct": assigned_correct,
            "species_call_false": species_call_false,
        }
    )


def summarize_policy(policy: pd.DataFrame) -> dict[str, float | int]:
    assigned = policy[policy["assigned_rank"] != "no_call"]
    species_calls = policy[policy["assigned_rank"] == "species"]
    out: dict[str, float | int] = {
        "n": int(len(policy)),
        "n_assigned": int(len(assigned)),
        "coverage": float(len(assigned) / len(policy)) if len(policy) else np.nan,
        "assigned_precision": float(assigned["correct"].mean()) if len(assigned) else np.nan,
        "species_calls": int(len(species_calls)),
        "species_call_precision": float(species_calls["correct"].mean()) if len(species_calls) else np.nan,
        "false_species_call_rate_all_queries": float(policy["species_call_false"].mean()) if len(policy) else np.nan,
    }
    for rank in RANKS + ["no_call"]:
        out[f"assigned_{rank}_count"] = int((policy["assigned_rank"] == rank).sum())
    return out


def quantiles(values: list[float]) -> tuple[float, float]:
    clean = np.array([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
        return np.nan, np.nan
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def bootstrap(policy: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(policy)
    metrics = {
        "coverage": [],
        "assigned_precision": [],
        "species_call_precision": [],
        "false_species_call_rate_all_queries": [],
    }
    if n == 0:
        return {f"{key}_ci95_low": np.nan for key in metrics} | {f"{key}_ci95_high": np.nan for key in metrics}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        summary = summarize_policy(policy.iloc[idx].reset_index(drop=True))
        for key in metrics:
            metrics[key].append(float(summary[key]))
    out: dict[str, float] = {}
    for key, values in metrics.items():
        low, high = quantiles(values)
        out[f"{key}_ci95_low"] = low
        out[f"{key}_ci95_high"] = high
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=CALIBRATION_DIR)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--prediction-set", action="append", default=["cnn_seed1206", "blast", "vsearch", "kmer"])
    parser.add_argument("--evaluation-split", action="append", default=["eval_c", "unseen_genera"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    per_query = pd.read_csv(args.calibration_dir / "calibration_per_query.csv")
    thresholds = pd.read_csv(args.calibration_dir / "missing_reference_aware_thresholds.csv")

    rows: list[dict[str, object]] = []
    logger.log(
        f"Bootstrapping target={args.target_precision} for {args.prediction_set} "
        f"on {args.evaluation_split} with n={args.n_bootstrap}"
    )
    for prediction_set in args.prediction_set:
        threshold_rows = thresholds[
            (thresholds["prediction_set"].astype(str) == str(prediction_set))
            & (pd.to_numeric(thresholds["target_precision"], errors="coerce") == args.target_precision)
        ]
        if threshold_rows.empty:
            logger.log(f"Skipping {prediction_set}: no thresholds")
            continue
        threshold_by_rank = {
            str(row["rank"]): float(row["threshold"])
            for _, row in threshold_rows.iterrows()
            if not pd.isna(row.get("threshold"))
        }
        if not threshold_by_rank:
            logger.log(f"Skipping {prediction_set}: no finite thresholds")
            continue
        for split in args.evaluation_split:
            evaluation = per_query[
                (per_query["prediction_set"].astype(str) == str(prediction_set))
                & (per_query["split"].astype(str) == str(split))
            ].copy()
            if evaluation.empty:
                logger.log(f"Skipping {prediction_set} {split}: no evaluation rows")
                continue
            policy = assign_policy(evaluation, threshold_by_rank)
            summary = summarize_policy(policy)
            boot = bootstrap(policy, args.n_bootstrap, args.seed)
            row = {
                "prediction_set": prediction_set,
                "evaluation_split": split,
                "target_precision": args.target_precision,
                "n_bootstrap": args.n_bootstrap,
                **summary,
                **boot,
            }
            rows.append(row)
            logger.log(
                f"{prediction_set} {split}: precision={summary['assigned_precision']:.4f} "
                f"coverage={summary['coverage']:.4f} false_species_rate={summary['false_species_call_rate_all_queries']:.4f}"
            )

    out_path = args.output_dir / "missing_reference_aware_policy_bootstrap.csv"
    logger.log(f"Writing bootstrap summary to {out_path}")
    write_csv(
        out_path,
        rows,
        [
            "prediction_set",
            "evaluation_split",
            "target_precision",
            "n_bootstrap",
            "n",
            "n_assigned",
            "coverage",
            "coverage_ci95_low",
            "coverage_ci95_high",
            "assigned_precision",
            "assigned_precision_ci95_low",
            "assigned_precision_ci95_high",
            "species_calls",
            "species_call_precision",
            "species_call_precision_ci95_low",
            "species_call_precision_ci95_high",
            "false_species_call_rate_all_queries",
            "false_species_call_rate_all_queries_ci95_low",
            "false_species_call_rate_all_queries_ci95_high",
            "assigned_species_count",
            "assigned_genus_count",
            "assigned_family_count",
            "assigned_order_count",
            "assigned_no_call_count",
        ],
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/bootstrap_rank_no_call_policy.py",
        "target_precision": args.target_precision,
        "prediction_sets": args.prediction_set,
        "evaluation_splits": args.evaluation_split,
        "n_bootstrap": args.n_bootstrap,
        "output_csv": str(out_path),
        "claim_boundary": "Bootstrap is over query rows for the locked seen-test-derived missing-reference-aware policy.",
    }
    manifest_path = args.output_dir / "missing_reference_aware_policy_bootstrap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Wrote {manifest_path}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
