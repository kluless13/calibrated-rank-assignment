#!/usr/bin/env python3
"""Overlay reference-gap warnings on Paper 1 production-v1 assignments.

This script does not train a new model. It asks whether existing v2
reference-gap detector probabilities are useful as reason labels on top of the
current conservative COI production-v1 assignment policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from progress_logging import ProgressLogger


RANKS = ("species", "genus", "family")
SPLITS = ("seen_test", "eval_c", "unseen_genera")


@dataclass(frozen=True)
class DetectorRun:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/production_v1"),
    )
    parser.add_argument(
        "--reference-gap-root",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/reference_gap_detector"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables"),
    )
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
        .fillna(False)
    )


def detector_runs(root: Path) -> list[DetectorRun]:
    names = [
        "coi_mlp_seed1301_v2_candidate_evidence_target095",
        "coi_mlp_seed1301_v2_candidate_evidence_target099",
        "coi_mlp_seed1301_v2_candidate_evidence_notree_target095",
    ]
    runs: list[DetectorRun] = []
    for name in names:
        path = root / name
        if (path / "reference_gap_predictions.csv").exists() and (
            path / "reference_gap_thresholds.csv"
        ).exists():
            runs.append(DetectorRun(name=name, path=path))
    return runs


def load_thresholds(detector: DetectorRun) -> dict[str, float]:
    thresholds = pd.read_csv(detector.path / "reference_gap_thresholds.csv")
    return {str(row["rank"]): float(row["threshold"]) for _, row in thresholds.iterrows()}


def rank_warning(row: pd.Series, thresholds: dict[str, float]) -> bool:
    rank = str(row.get("production_assigned_rank", ""))
    if rank not in RANKS:
        return False
    prob = row.get(f"gap_p_{rank}")
    if pd.isna(prob):
        return False
    return float(prob) >= thresholds[rank]


def strongest_warning(row: pd.Series, thresholds: dict[str, float]) -> tuple[str, float]:
    best_rank = "none"
    best_prob = 0.0
    for rank in RANKS:
        prob = row.get(f"gap_p_{rank}")
        if pd.isna(prob):
            continue
        prob_f = float(prob)
        if prob_f >= thresholds[rank] and prob_f > best_prob:
            best_rank = rank
            best_prob = prob_f
    return best_rank, best_prob


def summarize_split(prod: pd.DataFrame, gap: pd.DataFrame, detector: DetectorRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = load_thresholds(detector)
    normal_gap = gap[gap["split"].isin(SPLITS)].copy()
    rows: list[dict[str, object]] = []
    examples: list[pd.DataFrame] = []

    for split in SPLITS:
        split_prod = prod[split].copy()
        split_gap = normal_gap[normal_gap["split"] == split].copy()
        merged = split_prod.merge(
            split_gap[["processid", "gap_p_species", "gap_p_genus", "gap_p_family"]],
            on="processid",
            how="left",
        )
        merged["split"] = split
        merged["production_assigned_correct"] = bool_series(merged["production_assigned_correct"])
        assigned = merged["production_assigned_rank"].astype(str).ne("no_call")
        for rank in RANKS:
            merged[f"{rank}_gap_warning"] = (
                merged[f"gap_p_{rank}"].fillna(0.0).astype(float) >= thresholds[rank]
            )
        rank_warn = merged.apply(lambda row: rank_warning(row, thresholds), axis=1)
        strongest = merged.apply(lambda row: strongest_warning(row, thresholds), axis=1)
        merged["assigned_rank_gap_warning"] = rank_warn
        merged["strongest_gap_warning_rank"] = [x[0] for x in strongest]
        merged["strongest_gap_warning_probability"] = [x[1] for x in strongest]
        any_warning = merged[[f"{rank}_gap_warning" for rank in RANKS]].any(axis=1)
        more_specific_warning = pd.Series(False, index=merged.index)
        more_specific_warning |= merged["production_assigned_rank"].astype(str).eq("genus") & merged[
            "species_gap_warning"
        ]
        more_specific_warning |= merged["production_assigned_rank"].astype(str).eq("family") & (
            merged["species_gap_warning"] | merged["genus_gap_warning"]
        )
        more_specific_warning |= merged["production_assigned_rank"].astype(str).eq("order") & (
            merged["species_gap_warning"] | merged["genus_gap_warning"] | merged["family_gap_warning"]
        )

        gap_aware_assigned = assigned & ~rank_warn
        if gap_aware_assigned.any():
            gap_aware_precision = float(merged.loc[gap_aware_assigned, "production_assigned_correct"].mean())
        else:
            gap_aware_precision = float("nan")
        original_precision = (
            float(merged.loc[assigned, "production_assigned_correct"].mean()) if assigned.any() else float("nan")
        )
        wrong_assigned = assigned & ~merged["production_assigned_correct"]
        correct_assigned = assigned & merged["production_assigned_correct"]
        broader_assignment = assigned & merged["production_assigned_rank"].astype(str).isin(["genus", "family", "order"])

        rows.append(
            {
                "detector": detector.name,
                "split": split,
                "n_queries": int(len(merged)),
                "original_assigned": int(assigned.sum()),
                "original_coverage_pct": float(assigned.mean() * 100.0),
                "original_assigned_precision_pct": original_precision * 100.0,
                "gap_warning_on_assigned_count": int((assigned & rank_warn).sum()),
                "gap_warning_on_assigned_pct": float((assigned & rank_warn).mean() * 100.0),
                "any_gap_warning_on_assigned_count": int((assigned & any_warning).sum()),
                "any_gap_warning_on_assigned_pct": float((assigned & any_warning).mean() * 100.0),
                "species_gap_warning_on_assigned_pct": float((assigned & merged["species_gap_warning"]).mean() * 100.0),
                "genus_gap_warning_on_assigned_pct": float((assigned & merged["genus_gap_warning"]).mean() * 100.0),
                "family_gap_warning_on_assigned_pct": float((assigned & merged["family_gap_warning"]).mean() * 100.0),
                "more_specific_gap_warning_on_broader_assignment_pct": (
                    float((broader_assignment & more_specific_warning).sum() / broader_assignment.sum() * 100.0)
                    if broader_assignment.any()
                    else float("nan")
                ),
                "wrong_assigned_count": int(wrong_assigned.sum()),
                "wrong_assigned_caught_by_rank_warning_count": int((wrong_assigned & rank_warn).sum()),
                "wrong_assigned_caught_by_rank_warning_pct": (
                    float((wrong_assigned & rank_warn).sum() / wrong_assigned.sum() * 100.0)
                    if wrong_assigned.any()
                    else float("nan")
                ),
                "wrong_assigned_caught_by_any_warning_pct": (
                    float((wrong_assigned & any_warning).sum() / wrong_assigned.sum() * 100.0)
                    if wrong_assigned.any()
                    else float("nan")
                ),
                "correct_assigned_warning_rate_pct": (
                    float((correct_assigned & rank_warn).sum() / correct_assigned.sum() * 100.0)
                    if correct_assigned.any()
                    else float("nan")
                ),
                "gap_aware_assigned": int(gap_aware_assigned.sum()),
                "gap_aware_coverage_pct": float(gap_aware_assigned.mean() * 100.0),
                "gap_aware_assigned_precision_pct": gap_aware_precision * 100.0,
                "coverage_delta_pct": float((gap_aware_assigned.mean() - assigned.mean()) * 100.0),
                "precision_delta_pct": float((gap_aware_precision - original_precision) * 100.0),
                "species_threshold": thresholds.get("species"),
                "genus_threshold": thresholds.get("genus"),
                "family_threshold": thresholds.get("family"),
                "claim_boundary": (
                    "Diagnostic overlay only. Rank-specific reference-gap warnings are used to "
                    "abstain from existing production-v1 genus/family/species calls in this "
                    "table; this is not yet the default production policy."
                ),
            }
        )

        example_cols = [
            "processid",
            "split",
            "true_tree_label",
            "true_genus",
            "true_family",
            "true_order",
            "pred_species",
            "pred_genus",
            "pred_family",
            "pred_order",
            "production_assigned_rank",
            "production_assigned_label",
            "production_assigned_correct",
            "gap_p_species",
            "gap_p_genus",
            "gap_p_family",
            "species_gap_warning",
            "genus_gap_warning",
            "family_gap_warning",
            "assigned_rank_gap_warning",
            "strongest_gap_warning_rank",
            "strongest_gap_warning_probability",
        ]
        ex = (
            merged[assigned & rank_warn]
            .sort_values(["production_assigned_correct", "strongest_gap_warning_probability"], ascending=[True, False])
            .head(25)
        )
        if not ex.empty:
            ex = ex.assign(detector=detector.name)
            examples.append(ex[["detector"] + example_cols])

    return pd.DataFrame(rows), pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    script_name = Path(__file__).name
    logger.start(script_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prod = {}
    for split in SPLITS:
        path = args.production_root / split / "production_v1_assignments.csv"
        logger.log(f"Loading production assignments {path}")
        prod[split] = pd.read_csv(path)

    summaries: list[pd.DataFrame] = []
    examples: list[pd.DataFrame] = []
    runs = detector_runs(args.reference_gap_root)
    if not runs:
        raise FileNotFoundError(f"No reference-gap detector runs found under {args.reference_gap_root}")

    for detector in runs:
        pred_path = detector.path / "reference_gap_predictions.csv"
        logger.log(f"Loading reference-gap predictions {pred_path}")
        gap = pd.read_csv(pred_path)
        summary, example = summarize_split(prod, gap, detector)
        summaries.append(summary)
        if not example.empty:
            examples.append(example)

    summary_df = pd.concat(summaries, ignore_index=True)
    examples_df = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()
    summary_path = args.output_dir / "gap_warning_overlay_summary.csv"
    examples_path = args.output_dir / "gap_warning_overlay_examples.csv"
    manifest_path = args.output_dir / "gap_warning_overlay_manifest.json"
    logger.log(f"Writing {summary_path}")
    summary_df.to_csv(summary_path, index=False)
    logger.log(f"Writing {examples_path}")
    examples_df.to_csv(examples_path, index=False)
    manifest = {
        "generated_by": str(Path(__file__)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "production_root": str(args.production_root),
        "reference_gap_root": str(args.reference_gap_root),
        "detectors": [run.name for run in runs],
        "summary_rows": int(len(summary_df)),
        "example_rows": int(len(examples_df)),
        "outputs": {
            "summary": str(summary_path),
            "examples": str(examples_path),
        },
        "claim_boundary": (
            "Diagnostic overlay of existing reference-gap probabilities on existing "
            "production-v1 assignments; no new model is trained."
        ),
    }
    logger.log(f"Writing {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.done(script_name)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
