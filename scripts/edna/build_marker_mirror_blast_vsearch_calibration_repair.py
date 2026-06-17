#!/usr/bin/env python3
"""Calibration-transfer repair diagnostics for MarkerMirror + BLAST/VSEARCH.

Exp 108 showed that a learned listwise compiler improved high-coverage order
diagnostics but did not lock a target-0.99 production threshold. Exp 109/110
then replaced the k-mer same-marker arm with full VSEARCH and BLASTN candidate
generation. This script asks the next narrower question:

Can simple production-available BLAST/VSEARCH list evidence transfer under
species-split calibration/evaluation better than the previous learned compiler?

It builds candidate-list policy rows from top-1 and top-10-mode BLASTN/VSEARCH
taxa, then evaluates global thresholds, source-stratified thresholds, and
Wilson-lower-bound thresholds across repeated query-species splits.

Labels are used only to choose/evaluate thresholds. Candidate list features are
production-available.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("genus", "family", "order")
TOP_KS = (1, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-table",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "zero_shot_queries.csv",
    )
    parser.add_argument(
        "--marker-mirror-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "union_candidate_rank_policy"
        / "marker_mirror_union_production_candidates.csv.gz",
    )
    parser.add_argument(
        "--blast-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_same_marker_blast_candidates_top50.csv.gz",
    )
    parser.add_argument(
        "--vsearch-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_same_marker_vsearch_candidates_top50.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--targets", default="0.95,0.99")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--min-calibration-assignments", type=int, default=25)
    parser.add_argument("--min-source-calibration-assignments", type=int, default=20)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return math.nan
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return float((centre - margin) / denom)


def load_queries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "query_id": frame["processid"].astype(str),
            "source": frame["source"].astype(str),
            "query_tree_label": frame["tree_label"].astype(str),
            "query_species": frame["species_name"].astype(str),
            "query_genus": frame["genus_name"].map(clean),
            "query_family": frame["family_name"].map(clean),
            "query_order": frame["order_name"].map(clean),
        }
    )


def mode_stats(values: pd.Series) -> tuple[str, int, float, int]:
    cleaned = [clean(value) for value in values if clean(value)]
    if not cleaned:
        return "", 0, math.nan, 0
    value, count = Counter(cleaned).most_common(1)[0]
    return value, int(count), float(count) / float(len(cleaned)), int(len(set(cleaned)))


def source_score_column(source: str, frame: pd.DataFrame) -> str:
    if source == "blast" and "blast_identity" in frame.columns:
        return "blast_identity"
    if source == "vsearch" and "vsearch_identity" in frame.columns:
        return "vsearch_identity"
    return "score"


def source_features(candidates: pd.DataFrame, prefix: str) -> pd.DataFrame:
    score_col = source_score_column(prefix, candidates)
    rows: list[dict[str, Any]] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        group = group.sort_values("candidate_rank").copy()
        row: dict[str, Any] = {
            "query_id": query_id,
            f"{prefix}_candidate_count": int(len(group)),
        }
        score = pd.to_numeric(group[score_col], errors="coerce")
        row[f"{prefix}_top1_score"] = float(score.iloc[0]) if len(score) else math.nan
        if prefix == "blast":
            row[f"{prefix}_top1_bitscore"] = float(pd.to_numeric(group["blast_bitscore"], errors="coerce").iloc[0])
            row[f"{prefix}_top1_alignment_length"] = float(
                pd.to_numeric(group["blast_alignment_length"], errors="coerce").iloc[0]
            )
        if prefix == "vsearch":
            row[f"{prefix}_top1_alignment_length"] = float(
                pd.to_numeric(group["vsearch_alignment_length"], errors="coerce").iloc[0]
            )
        for top_k in TOP_KS:
            top = group.head(top_k)
            for rank in RANKS:
                col = f"candidate_{rank}"
                mode, count, fraction, unique = mode_stats(top[col]) if col in top.columns else ("", 0, math.nan, 0)
                row[f"{prefix}_top{top_k}_{rank}_mode"] = mode
                row[f"{prefix}_top{top_k}_{rank}_mode_count"] = count
                row[f"{prefix}_top{top_k}_{rank}_mode_fraction"] = fraction
                row[f"{prefix}_top{top_k}_{rank}_unique"] = unique
        rows.append(row)
    return pd.DataFrame(rows)


def build_policy_rows(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in features.itertuples(index=False):
        base = row._asdict()
        query_id = base["query_id"]
        source = base["source"]
        for rank in RANKS:
            truth = clean(base.get(f"query_{rank}"))
            for prefix in ("blast", "vsearch"):
                for top_k in TOP_KS:
                    taxon = clean(base.get(f"{prefix}_top{top_k}_{rank}_mode"))
                    if not taxon:
                        continue
                    mode_fraction = float(base.get(f"{prefix}_top{top_k}_{rank}_mode_fraction") or 0.0)
                    top1_score = float(base.get(f"{prefix}_top1_score") or 0.0)
                    confidence = top1_score * mode_fraction
                    rows.append(
                        {
                            "query_id": query_id,
                            "source": source,
                            "rank": rank,
                            "policy": f"{prefix}_top{top_k}_mode",
                            "assigned_taxon": taxon,
                            "confidence": confidence,
                            "correct": bool(taxon and taxon == truth),
                        }
                    )

            for top_k in TOP_KS:
                blast_taxon = clean(base.get(f"blast_top{top_k}_{rank}_mode"))
                vsearch_taxon = clean(base.get(f"vsearch_top{top_k}_{rank}_mode"))
                if blast_taxon and blast_taxon == vsearch_taxon:
                    blast_conf = float(base.get("blast_top1_score") or 0.0) * float(
                        base.get(f"blast_top{top_k}_{rank}_mode_fraction") or 0.0
                    )
                    vsearch_conf = float(base.get("vsearch_top1_score") or 0.0) * float(
                        base.get(f"vsearch_top{top_k}_{rank}_mode_fraction") or 0.0
                    )
                    rows.append(
                        {
                            "query_id": query_id,
                            "source": source,
                            "rank": rank,
                            "policy": f"blast_vsearch_agree_top{top_k}_mode",
                            "assigned_taxon": blast_taxon,
                            "confidence": min(blast_conf, vsearch_conf),
                            "correct": bool(blast_taxon == truth),
                        }
                    )

                mm_taxon = clean(base.get(f"mm_top{top_k}_{rank}_mode"))
                if mm_taxon and mm_taxon == blast_taxon == vsearch_taxon:
                    mm_conf = float(base.get("mm_top1_score") or 0.0) * float(
                        base.get(f"mm_top{top_k}_{rank}_mode_fraction") or 0.0
                    )
                    blast_conf = float(base.get("blast_top1_score") or 0.0) * float(
                        base.get(f"blast_top{top_k}_{rank}_mode_fraction") or 0.0
                    )
                    vsearch_conf = float(base.get("vsearch_top1_score") or 0.0) * float(
                        base.get(f"vsearch_top{top_k}_{rank}_mode_fraction") or 0.0
                    )
                    rows.append(
                        {
                            "query_id": query_id,
                            "source": source,
                            "rank": rank,
                            "policy": f"mm_blast_vsearch_agree_top{top_k}_mode",
                            "assigned_taxon": mm_taxon,
                            "confidence": min(mm_conf, blast_conf, vsearch_conf),
                            "correct": bool(mm_taxon == truth),
                        }
                    )
    return pd.DataFrame(rows)


def choose_threshold(frame: pd.DataFrame, target: float, min_assignments: int, use_wilson: bool) -> float:
    valid = frame[frame["assigned_taxon"].map(clean).astype(bool) & frame["confidence"].notna()].copy()
    if len(valid) < min_assignments:
        return math.inf
    grouped = (
        valid.assign(correct_int=valid["correct"].astype(int))
        .groupby("confidence", as_index=False)
        .agg(n=("correct_int", "size"), successes=("correct_int", "sum"))
        .sort_values("confidence", ascending=False)
    )
    grouped["cum_n"] = grouped["n"].cumsum()
    grouped["cum_successes"] = grouped["successes"].cumsum()
    grouped = grouped[grouped["cum_n"] >= min_assignments].copy()
    if grouped.empty:
        return math.inf
    if use_wilson:
        grouped["criterion"] = [
            wilson_lower(int(successes), int(total))
            for successes, total in zip(grouped["cum_successes"], grouped["cum_n"])
        ]
    else:
        grouped["criterion"] = grouped["cum_successes"] / grouped["cum_n"]
    passing = grouped[grouped["criterion"] >= target]
    if passing.empty:
        return math.inf
    # Sorted high-to-low; the last passing threshold gives maximum calibration
    # coverage while still meeting the requested calibration criterion.
    return float(passing.iloc[-1]["confidence"])


def evaluate(frame: pd.DataFrame, threshold: float, eval_query_count: int) -> dict[str, Any]:
    if math.isinf(threshold):
        assigned = frame.iloc[0:0].copy()
    else:
        assigned = frame[(frame["assigned_taxon"].map(clean).astype(bool)) & (frame["confidence"] >= threshold)].copy()
    n_assigned = len(assigned)
    n_correct = int(assigned["correct"].sum()) if n_assigned else 0
    return {
        "threshold": threshold,
        "n_eval_queries": int(eval_query_count),
        "n_assigned": int(n_assigned),
        "n_correct": int(n_correct),
        "coverage_pct": pct(n_assigned, eval_query_count),
        "assigned_precision": n_correct / n_assigned if n_assigned else math.nan,
        "false_species_call_rate": 0.0,
    }


def split_species(frame: pd.DataFrame, rng: np.random.Generator, calibration_fraction: float) -> tuple[set[str], set[str]]:
    species = np.array(sorted(frame["query_tree_label"].map(clean).dropna().unique()))
    rng.shuffle(species)
    split = max(1, min(len(species) - 1, int(round(len(species) * calibration_fraction))))
    return set(species[:split]), set(species[split:])


def evaluate_policy_splits(
    policies: pd.DataFrame,
    query_frame: pd.DataFrame,
    targets: list[float],
    repeats: int,
    calibration_fraction: float,
    seed: int,
    min_calibration_assignments: int,
    min_source_calibration_assignments: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    query_meta = query_frame[["query_id", "query_tree_label"]].drop_duplicates()
    policies = policies.merge(query_meta, on="query_id", how="left")
    per_split_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for repeat in range(repeats):
        cal_species, eval_species = split_species(query_frame, rng, calibration_fraction)
        cal_query_ids = set(query_frame[query_frame["query_tree_label"].isin(cal_species)]["query_id"])
        eval_query_ids = set(query_frame[query_frame["query_tree_label"].isin(eval_species)]["query_id"])
        eval_query_count = len(eval_query_ids)
        cal = policies[policies["query_id"].isin(cal_query_ids)].copy()
        ev = policies[policies["query_id"].isin(eval_query_ids)].copy()

        for (policy, rank), cal_group in cal.groupby(["policy", "rank"], sort=False):
            eval_group = ev[(ev["policy"] == policy) & (ev["rank"] == rank)].copy()
            for target in targets:
                for strategy, use_wilson in [("global_precision", False), ("global_wilson95", True)]:
                    threshold = choose_threshold(cal_group, target, min_calibration_assignments, use_wilson)
                    result = evaluate(eval_group, threshold, eval_query_count)
                    per_split_rows.append(
                        {
                            "repeat": repeat,
                            "strategy": strategy,
                            "policy": policy,
                            "rank": rank,
                            "target": target,
                            "target_met": bool(
                                result["n_assigned"] > 0
                                and not math.isnan(result["assigned_precision"])
                                and result["assigned_precision"] >= target
                            ),
                            **result,
                        }
                    )
                    threshold_rows.append(
                        {
                            "repeat": repeat,
                            "strategy": strategy,
                            "policy": policy,
                            "rank": rank,
                            "target": target,
                            "source": "__global__",
                            "threshold": threshold,
                        }
                    )

                for strategy, use_wilson in [("source_stratified_precision", False), ("source_stratified_wilson95", True)]:
                    global_threshold = choose_threshold(cal_group, target, min_calibration_assignments, use_wilson)
                    assigned_parts: list[pd.DataFrame] = []
                    thresholds_by_source: dict[str, float] = {}
                    for source, eval_source_group in eval_group.groupby("source", sort=False):
                        cal_source_group = cal_group[cal_group["source"] == source]
                        source_threshold = choose_threshold(
                            cal_source_group,
                            target,
                            min_source_calibration_assignments,
                            use_wilson,
                        )
                        threshold = max(global_threshold, source_threshold) if not math.isinf(source_threshold) else global_threshold
                        thresholds_by_source[str(source)] = threshold
                        if not math.isinf(threshold):
                            assigned_parts.append(eval_source_group[eval_source_group["confidence"] >= threshold].copy())
                    assigned = pd.concat(assigned_parts, ignore_index=True) if assigned_parts else eval_group.iloc[0:0].copy()
                    n_assigned = len(assigned)
                    n_correct = int(assigned["correct"].sum()) if n_assigned else 0
                    assigned_precision = n_correct / n_assigned if n_assigned else math.nan
                    per_split_rows.append(
                        {
                            "repeat": repeat,
                            "strategy": strategy,
                            "policy": policy,
                            "rank": rank,
                            "target": target,
                            "threshold": math.nan,
                            "n_eval_queries": int(eval_query_count),
                            "n_assigned": int(n_assigned),
                            "n_correct": int(n_correct),
                            "coverage_pct": pct(n_assigned, eval_query_count),
                            "assigned_precision": assigned_precision,
                            "false_species_call_rate": 0.0,
                            "target_met": bool(
                                n_assigned > 0 and not math.isnan(assigned_precision) and assigned_precision >= target
                            ),
                        }
                    )
                    for source, threshold in thresholds_by_source.items():
                        threshold_rows.append(
                            {
                                "repeat": repeat,
                                "strategy": strategy,
                                "policy": policy,
                                "rank": rank,
                                "target": target,
                                "source": source,
                                "threshold": threshold,
                            }
                        )
    return pd.DataFrame(per_split_rows), pd.DataFrame(threshold_rows)


def summarize(per_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in per_split.groupby(["strategy", "policy", "rank", "target"], sort=False):
        strategy, policy, rank, target = keys
        rows.append(
            {
                "strategy": strategy,
                "policy": policy,
                "rank": rank,
                "target": float(target),
                "repeats": int(len(group)),
                "mean_coverage_pct": float(group["coverage_pct"].mean()),
                "median_coverage_pct": float(group["coverage_pct"].median()),
                "mean_assigned_precision": float(group["assigned_precision"].mean()),
                "median_assigned_precision": float(group["assigned_precision"].median()),
                "target_met_rate_pct": float(group["target_met"].mean() * 100.0),
                "mean_n_assigned": float(group["n_assigned"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["target", "rank", "target_met_rate_pct", "mean_assigned_precision", "mean_coverage_pct"],
        ascending=[False, True, False, False, False],
    )


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    script_name = Path(__file__).name
    logger.start(script_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [float(value.strip()) for value in args.targets.split(",") if value.strip()]

    logger.log("Loading queries and candidate tables")
    queries = load_queries(args.query_table)
    mm = pd.read_csv(args.marker_mirror_candidates)
    mm = mm[mm["candidate_source"] == "marker_mirror_12s_to_16s"].copy()
    blast = pd.read_csv(args.blast_candidates)
    vsearch = pd.read_csv(args.vsearch_candidates)

    logger.log("Building list features")
    features = queries.copy()
    for prefix, frame in [
        ("mm", mm),
        ("blast", blast),
        ("vsearch", vsearch),
    ]:
        features = features.merge(source_features(frame, prefix), on="query_id", how="left")
    feature_path = args.output_dir / "marker_mirror_blast_vsearch_calibration_repair_features.csv"
    features.to_csv(feature_path, index=False)

    logger.log("Building policy rows")
    policies = build_policy_rows(features)
    policy_path = args.output_dir / "marker_mirror_blast_vsearch_calibration_repair_policy_rows.csv.gz"
    policies.to_csv(policy_path, index=False)
    logger.log(f"Policy rows={len(policies):,}")

    logger.log("Evaluating repeated species-split calibration transfer")
    per_split, thresholds = evaluate_policy_splits(
        policies,
        queries,
        targets,
        args.repeats,
        args.calibration_fraction,
        args.seed,
        args.min_calibration_assignments,
        args.min_source_calibration_assignments,
    )
    summary = summarize(per_split)

    summary_path = args.output_dir / "marker_mirror_blast_vsearch_calibration_repair_summary.csv"
    per_split_path = args.output_dir / "marker_mirror_blast_vsearch_calibration_repair_per_split.csv"
    threshold_path = args.output_dir / "marker_mirror_blast_vsearch_calibration_repair_thresholds.csv"
    manifest_path = args.output_dir / "marker_mirror_blast_vsearch_calibration_repair_manifest.json"
    summary.to_csv(summary_path, index=False)
    per_split.to_csv(per_split_path, index=False)
    thresholds.to_csv(threshold_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "query_table": rel(args.query_table),
            "marker_mirror_candidates": rel(args.marker_mirror_candidates),
            "blast_candidates": rel(args.blast_candidates),
            "vsearch_candidates": rel(args.vsearch_candidates),
        },
        "outputs": {
            "features": rel(feature_path),
            "policy_rows": rel(policy_path),
            "summary": rel(summary_path),
            "per_split": rel(per_split_path),
            "thresholds": rel(threshold_path),
        },
        "targets": targets,
        "repeats": int(args.repeats),
        "calibration_fraction": float(args.calibration_fraction),
        "claim_boundary": "Calibration-transfer repair diagnostics over BLAST/VSEARCH-backed union candidate lists; labels are used only for threshold calibration and evaluation.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote {rel(summary_path)}")
    logger.log(f"Wrote {rel(per_split_path)}")
    logger.log(f"Wrote {rel(threshold_path)}")
    logger.log(f"Wrote {rel(manifest_path)}")
    logger.done(script_name)


if __name__ == "__main__":
    main()
