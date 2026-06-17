#!/usr/bin/env python3
"""Nested high-coverage order repair for MarkerMirror + BLAST/VSEARCH.

Exp 111 found a conservative order policy that transfers cleanly but covers
only about a quarter of queries. This follow-up asks whether stricter threshold
locking can rescue higher-coverage BLAST/VSEARCH order policies under an outer
species-split evaluation.

Labels are used only for calibration/evaluation. The policy rows themselves are
production-available candidate-list evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = Path(__file__).stem
DEFAULT_POLICIES = (
    "blast_top10_mode",
    "vsearch_top10_mode",
    "blast_vsearch_agree_top10_mode",
    "mm_blast_vsearch_agree_top1_mode",
)
MULTI_SOURCE_POLICIES = {
    "blast_vsearch_agree_top10_mode",
    "mm_blast_vsearch_agree_top1_mode",
    "mm_blast_vsearch_agree_top10_mode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-table",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource" / "zero_shot_queries.csv",
    )
    parser.add_argument(
        "--policy-rows",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "source_tables"
        / "marker_mirror_blast_vsearch_calibration_repair_policy_rows.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--rank", choices=("genus", "family", "order"), default="order")
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix. Defaults to marker_mirror_high_coverage_<rank>_repair.",
    )
    parser.add_argument("--outer-repeats", type=int, default=50)
    parser.add_argument("--inner-repeats", type=int, default=20)
    parser.add_argument("--fit-fraction", type=float, default=0.50)
    parser.add_argument("--inner-calibration-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--min-calibration-assignments", type=int, default=25)
    parser.add_argument("--min-source-calibration-assignments", type=int, default=20)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
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


def load_query_meta(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "query_id": frame["processid"].astype(str),
            "source": frame["source"].astype(str),
            "query_tree_label": frame["tree_label"].astype(str),
        }
    )


def split_species(
    species: list[str] | np.ndarray,
    rng: np.random.Generator,
    fraction: float,
) -> tuple[set[str], set[str]]:
    values = np.array(sorted(clean(value) for value in species if clean(value)))
    rng.shuffle(values)
    split = max(1, min(len(values) - 1, int(round(len(values) * fraction))))
    return set(values[:split]), set(values[split:])


def choose_threshold(frame: pd.DataFrame, target: float, min_assignments: int, use_wilson: bool) -> float:
    valid = frame[
        frame["assigned_taxon"].map(clean).astype(bool)
        & frame["confidence"].notna()
        & np.isfinite(pd.to_numeric(frame["confidence"], errors="coerce"))
    ].copy()
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
    return float(passing.iloc[-1]["confidence"])


def lock_global_thresholds(
    fit: pd.DataFrame,
    policy: str,
    fit_species: set[str],
    rng: np.random.Generator,
    inner_repeats: int,
    inner_fraction: float,
    target: float,
    min_assignments: int,
    use_wilson: bool,
) -> tuple[float, list[float]]:
    thresholds: list[float] = []
    species = sorted(fit_species)
    for _ in range(inner_repeats):
        cal_species, _ = split_species(species, rng, inner_fraction)
        cal = fit[(fit["policy"] == policy) & fit["query_tree_label"].isin(cal_species)]
        threshold = choose_threshold(cal, target, min_assignments, use_wilson)
        if math.isfinite(threshold):
            thresholds.append(threshold)
    locked = max(thresholds) if thresholds else math.inf
    return locked, thresholds


def lock_source_thresholds(
    fit: pd.DataFrame,
    policy: str,
    fit_species: set[str],
    sources: list[str],
    rng: np.random.Generator,
    inner_repeats: int,
    inner_fraction: float,
    target: float,
    min_global_assignments: int,
    min_source_assignments: int,
    use_wilson: bool,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    global_locked, _ = lock_global_thresholds(
        fit=fit,
        policy=policy,
        fit_species=fit_species,
        rng=rng,
        inner_repeats=inner_repeats,
        inner_fraction=inner_fraction,
        target=target,
        min_assignments=min_global_assignments,
        use_wilson=use_wilson,
    )
    species = sorted(fit_species)
    source_thresholds: dict[str, list[float]] = {source: [] for source in sources}
    for _ in range(inner_repeats):
        cal_species, _ = split_species(species, rng, inner_fraction)
        cal = fit[(fit["policy"] == policy) & fit["query_tree_label"].isin(cal_species)]
        for source in sources:
            threshold = choose_threshold(
                cal[cal["source"] == source],
                target,
                min_source_assignments,
                use_wilson,
            )
            if math.isfinite(threshold):
                source_thresholds[source].append(threshold)
    locked: dict[str, float] = {}
    for source, values in source_thresholds.items():
        source_locked = max(values) if values else math.inf
        if math.isinf(global_locked) and math.isinf(source_locked):
            locked[source] = math.inf
        elif math.isinf(source_locked):
            locked[source] = global_locked
        elif math.isinf(global_locked):
            locked[source] = source_locked
        else:
            locked[source] = max(global_locked, source_locked)
    return locked, source_thresholds


def evaluate_global(
    ev: pd.DataFrame,
    policy: str,
    threshold: float,
    eval_query_count: int,
    target: float,
) -> dict[str, Any]:
    if math.isinf(threshold):
        assigned = ev.iloc[0:0].copy()
    else:
        assigned = ev[(ev["policy"] == policy) & (ev["confidence"] >= threshold)].copy()
    n_assigned = len(assigned)
    n_correct = int(assigned["correct"].sum()) if n_assigned else 0
    return {
        "threshold": threshold,
        "n_eval_queries": int(eval_query_count),
        "n_assigned": int(n_assigned),
        "n_correct": int(n_correct),
        "coverage_pct": pct(n_assigned, eval_query_count),
        "assigned_precision": n_correct / n_assigned if n_assigned else math.nan,
        "target_met": bool(n_assigned > 0 and n_correct / n_assigned >= target),
    }


def evaluate_source(
    ev: pd.DataFrame,
    policy: str,
    thresholds: dict[str, float],
    eval_query_count: int,
    target: float,
) -> dict[str, Any]:
    assigned_parts: list[pd.DataFrame] = []
    for source, threshold in thresholds.items():
        if math.isinf(threshold):
            continue
        source_rows = ev[(ev["policy"] == policy) & (ev["source"] == source)]
        assigned_parts.append(source_rows[source_rows["confidence"] >= threshold].copy())
    assigned = pd.concat(assigned_parts, ignore_index=True) if assigned_parts else ev.iloc[0:0].copy()
    n_assigned = len(assigned)
    n_correct = int(assigned["correct"].sum()) if n_assigned else 0
    return {
        "threshold": math.nan,
        "n_eval_queries": int(eval_query_count),
        "n_assigned": int(n_assigned),
        "n_correct": int(n_correct),
        "coverage_pct": pct(n_assigned, eval_query_count),
        "assigned_precision": n_correct / n_assigned if n_assigned else math.nan,
        "target_met": bool(n_assigned > 0 and n_correct / n_assigned >= target),
    }


def nested_evaluate(
    policies: pd.DataFrame,
    query_meta: pd.DataFrame,
    policy_names: list[str],
    rank: str,
    target: float,
    outer_repeats: int,
    inner_repeats: int,
    fit_fraction: float,
    inner_fraction: float,
    seed: int,
    min_calibration_assignments: int,
    min_source_calibration_assignments: int,
    logger: ProgressLogger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    species = sorted(query_meta["query_tree_label"].map(clean).dropna().unique())
    sources = sorted(policies["source"].map(clean).dropna().unique())
    per_split_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    strategies = [
        ("nested_global_max_precision", False, "global"),
        ("nested_global_max_wilson95", True, "global"),
        ("nested_source_max_precision", False, "source"),
        ("nested_source_max_wilson95", True, "source"),
    ]

    for repeat in range(outer_repeats):
        if repeat % max(1, outer_repeats // 10) == 0:
            logger.log(f"Outer split {repeat + 1}/{outer_repeats}")
        fit_species, eval_species = split_species(species, rng, fit_fraction)
        fit_query_ids = set(query_meta[query_meta["query_tree_label"].isin(fit_species)]["query_id"])
        eval_query_ids = set(query_meta[query_meta["query_tree_label"].isin(eval_species)]["query_id"])
        fit = policies[policies["query_id"].isin(fit_query_ids)].copy()
        ev = policies[policies["query_id"].isin(eval_query_ids)].copy()
        eval_query_count = len(eval_query_ids)

        for policy in policy_names:
            for strategy, use_wilson, scope in strategies:
                if scope == "global":
                    threshold, inner_thresholds = lock_global_thresholds(
                        fit=fit,
                        policy=policy,
                        fit_species=fit_species,
                        rng=rng,
                        inner_repeats=inner_repeats,
                        inner_fraction=inner_fraction,
                        target=target,
                        min_assignments=min_calibration_assignments,
                        use_wilson=use_wilson,
                    )
                    result = evaluate_global(ev, policy, threshold, eval_query_count, target)
                    per_split_rows.append(
                        {
                            "repeat": repeat,
                            "strategy": strategy,
                            "policy": policy,
                            "rank": rank,
                            "target": target,
                            "inner_finite_threshold_count": int(len(inner_thresholds)),
                            "inner_threshold_max": threshold,
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
                            "inner_finite_threshold_count": int(len(inner_thresholds)),
                        }
                    )
                else:
                    thresholds, source_inner = lock_source_thresholds(
                        fit=fit,
                        policy=policy,
                        fit_species=fit_species,
                        sources=sources,
                        rng=rng,
                        inner_repeats=inner_repeats,
                        inner_fraction=inner_fraction,
                        target=target,
                        min_global_assignments=min_calibration_assignments,
                        min_source_assignments=min_source_calibration_assignments,
                        use_wilson=use_wilson,
                    )
                    result = evaluate_source(ev, policy, thresholds, eval_query_count, target)
                    per_split_rows.append(
                        {
                            "repeat": repeat,
                            "strategy": strategy,
                            "policy": policy,
                            "rank": rank,
                            "target": target,
                            "inner_finite_threshold_count": int(
                                sum(len(values) for values in source_inner.values())
                            ),
                            "inner_threshold_max": max(
                                [value for value in thresholds.values() if math.isfinite(value)],
                                default=math.inf,
                            ),
                            **result,
                        }
                    )
                    for source, threshold in thresholds.items():
                        threshold_rows.append(
                            {
                                "repeat": repeat,
                                "strategy": strategy,
                                "policy": policy,
                                "rank": rank,
                                "target": target,
                                "source": source,
                                "threshold": threshold,
                                "inner_finite_threshold_count": int(len(source_inner.get(source, []))),
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
                "outer_repeats": int(len(group)),
                "mean_coverage_pct": float(group["coverage_pct"].mean()),
                "median_coverage_pct": float(group["coverage_pct"].median()),
                "mean_assigned_precision": float(group["assigned_precision"].mean()),
                "median_assigned_precision": float(group["assigned_precision"].median()),
                "target_met_rate_pct": float(group["target_met"].mean() * 100.0),
                "mean_n_assigned": float(group["n_assigned"].mean()),
                "min_assigned_precision": float(group["assigned_precision"].min(skipna=True)),
                "recommended_multi_source": bool(policy in MULTI_SOURCE_POLICIES),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        [
            "target_met_rate_pct",
            "recommended_multi_source",
            "mean_coverage_pct",
            "mean_assigned_precision",
        ],
        ascending=[False, False, False, False],
    )


def choose_recommended(summary: pd.DataFrame) -> pd.Series:
    stable = summary[(summary["target_met_rate_pct"] >= 100.0) & summary["recommended_multi_source"]].copy()
    if stable.empty:
        stable = summary[summary["target_met_rate_pct"] >= 100.0].copy()
    if stable.empty:
        stable = summary.copy()
    stable = stable.sort_values(
        ["target_met_rate_pct", "recommended_multi_source", "mean_coverage_pct", "mean_assigned_precision"],
        ascending=[False, False, False, False],
    )
    return stable.iloc[0]


def lock_full_thresholds(
    policies: pd.DataFrame,
    query_meta: pd.DataFrame,
    policy: str,
    strategy: str,
    rank: str,
    target: float,
    inner_repeats: int,
    inner_fraction: float,
    seed: int,
    min_calibration_assignments: int,
    min_source_calibration_assignments: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    all_species = set(query_meta["query_tree_label"].map(clean).dropna().unique())
    sources = sorted(policies["source"].map(clean).dropna().unique())
    use_wilson = "wilson95" in strategy
    rows: list[dict[str, Any]] = []
    if "source" in strategy:
        thresholds, source_inner = lock_source_thresholds(
            fit=policies,
            policy=policy,
            fit_species=all_species,
            sources=sources,
            rng=rng,
            inner_repeats=inner_repeats,
            inner_fraction=inner_fraction,
            target=target,
            min_global_assignments=min_calibration_assignments,
            min_source_assignments=min_source_calibration_assignments,
            use_wilson=use_wilson,
        )
        for source, threshold in thresholds.items():
            rows.append(
                {
                    "strategy": strategy,
                    "policy": policy,
                    "rank": rank,
                    "target": target,
                    "source": source,
                    "threshold": threshold,
                    "inner_finite_threshold_count": len(source_inner.get(source, [])),
                }
            )
        return thresholds, pd.DataFrame(rows)
    threshold, inner = lock_global_thresholds(
        fit=policies,
        policy=policy,
        fit_species=all_species,
        rng=rng,
        inner_repeats=inner_repeats,
        inner_fraction=inner_fraction,
        target=target,
        min_assignments=min_calibration_assignments,
        use_wilson=use_wilson,
    )
    rows.append(
        {
            "strategy": strategy,
            "policy": policy,
            "rank": rank,
            "target": target,
            "source": "__global__",
            "threshold": threshold,
            "inner_finite_threshold_count": len(inner),
        }
    )
    return {"__global__": threshold}, pd.DataFrame(rows)


def build_full_assignments(
    policies: pd.DataFrame,
    query_meta: pd.DataFrame,
    policy: str,
    thresholds: dict[str, float],
    strategy: str,
    rank: str,
) -> pd.DataFrame:
    subset = policies[policies["policy"] == policy].copy()
    assigned_query_ids: set[str] = set()
    if "__global__" in thresholds:
        threshold = thresholds["__global__"]
        if math.isfinite(threshold):
            assigned_query_ids = set(subset[subset["confidence"] >= threshold]["query_id"])
    else:
        for source, threshold in thresholds.items():
            if math.isinf(threshold):
                continue
            part = subset[(subset["source"] == source) & (subset["confidence"] >= threshold)].copy()
            assigned_query_ids.update(part["query_id"])
    row_by_query = (
        subset.sort_values(["query_id", "confidence"], ascending=[True, False])
        .drop_duplicates("query_id")
        .set_index("query_id")
    )
    rows: list[dict[str, Any]] = []
    for meta in query_meta.itertuples(index=False):
        row = row_by_query.loc[meta.query_id] if meta.query_id in row_by_query.index else None
        if row is not None:
            threshold = thresholds.get(row.source, thresholds.get("__global__", math.inf))
            is_assigned = bool(meta.query_id in assigned_query_ids and row.confidence >= threshold)
            assigned_label = row.assigned_taxon if is_assigned else ""
            confidence = row.confidence
            correct = bool(row.correct) if is_assigned else np.nan
            reason = (
                f"{policy}_{strategy}_threshold_met"
                if is_assigned
                else f"{policy}_{strategy}_below_threshold"
            )
        else:
            threshold = thresholds.get(meta.source, thresholds.get("__global__", math.inf))
            is_assigned = False
            assigned_label = ""
            confidence = math.nan
            correct = np.nan
            reason = f"{policy}_{strategy}_no_policy_candidate_row"
        rows.append(
            {
                "query_id": meta.query_id,
                "source": meta.source,
                "decision_mode": f"marker_mirror_blastn_vsearch_high_coverage_{rank}_diagnostic",
                "policy": policy,
                "strategy": strategy,
                "assigned_rank": rank if is_assigned else "no_call",
                "assigned_label": assigned_label,
                "confidence": confidence,
                "locked_threshold": threshold,
                "correct": correct,
                "assignment_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def summarize_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    assigned = assignments[assignments["assigned_rank"] != "no_call"].copy()
    known = assigned[assigned["correct"].notna()]
    n_correct = int(known["correct"].astype(bool).sum()) if len(known) else 0
    return pd.DataFrame(
        [
            {
                "policy": assignments["policy"].iloc[0] if len(assignments) else "",
                "strategy": assignments["strategy"].iloc[0] if len(assignments) else "",
                "n_queries": int(assignments["query_id"].nunique()),
                "n_assigned": int(len(assigned)),
                "coverage_pct": pct(len(assigned), assignments["query_id"].nunique()),
                "assigned_precision_pct": pct(n_correct, len(known)) if len(known) else math.nan,
                "n_correct": n_correct,
                "n_incorrect": int(len(known) - n_correct),
                "n_assigned_known_truth": int(len(known)),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    log_file = args.log_file or (
        ROOT / "results" / "paper1_phylo_calibrated_assignment" / "logs" / f"{SCRIPT_NAME}.log"
    )
    logger = ProgressLogger(log_file)
    logger.start(SCRIPT_NAME)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy_names = [value.strip() for value in args.policies.split(",") if value.strip()]
    logger.log(f"Loading query meta: {rel(args.query_table)}")
    query_meta = load_query_meta(args.query_table)
    logger.log(f"Loading policy rows: {rel(args.policy_rows)}")
    policies = pd.read_csv(args.policy_rows)
    policies = policies.merge(query_meta[["query_id", "query_tree_label"]], on="query_id", how="left")
    policies = policies[(policies["rank"] == args.rank) & policies["policy"].isin(policy_names)].copy()
    policies["confidence"] = pd.to_numeric(policies["confidence"], errors="coerce")
    policies["correct"] = policies["correct"].astype(bool)
    logger.log(f"Evaluating {len(policies):,} {args.rank} policy rows across {len(policy_names)} policies")

    per_split, thresholds = nested_evaluate(
        policies=policies,
        query_meta=query_meta,
        policy_names=policy_names,
        rank=args.rank,
        target=args.target,
        outer_repeats=args.outer_repeats,
        inner_repeats=args.inner_repeats,
        fit_fraction=args.fit_fraction,
        inner_fraction=args.inner_calibration_fraction,
        seed=args.seed,
        min_calibration_assignments=args.min_calibration_assignments,
        min_source_calibration_assignments=args.min_source_calibration_assignments,
        logger=logger,
    )
    summary = summarize(per_split)
    recommended = choose_recommended(summary)
    logger.log(
        "Recommended diagnostic row: "
        f"{recommended['strategy']} / {recommended['policy']} "
        f"coverage={recommended['mean_coverage_pct']:.2f}% "
        f"precision={recommended['mean_assigned_precision']:.4f} "
        f"target_met={recommended['target_met_rate_pct']:.1f}%"
    )

    locked_thresholds, locked_threshold_rows = lock_full_thresholds(
        policies=policies,
        query_meta=query_meta,
        policy=str(recommended["policy"]),
        strategy=str(recommended["strategy"]),
        rank=args.rank,
        target=args.target,
        inner_repeats=args.inner_repeats,
        inner_fraction=args.inner_calibration_fraction,
        seed=args.seed + 999,
        min_calibration_assignments=args.min_calibration_assignments,
        min_source_calibration_assignments=args.min_source_calibration_assignments,
    )
    assignments = build_full_assignments(
        policies=policies,
        query_meta=query_meta,
        policy=str(recommended["policy"]),
        thresholds=locked_thresholds,
        strategy=str(recommended["strategy"]),
        rank=args.rank,
    )
    assignment_summary = summarize_assignments(assignments)

    prefix = args.output_prefix or f"marker_mirror_high_coverage_{args.rank}_repair"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    per_split_path = args.output_dir / f"{prefix}_per_split.csv"
    threshold_path = args.output_dir / f"{prefix}_thresholds.csv"
    locked_threshold_path = args.output_dir / f"{prefix}_locked_thresholds.csv"
    assignment_path = args.output_dir / f"{prefix}_assignments.csv"
    assignment_summary_path = args.output_dir / f"{prefix}_assignment_summary.csv"
    manifest_path = args.output_dir / f"{prefix}_manifest.json"

    summary.to_csv(summary_path, index=False)
    per_split.to_csv(per_split_path, index=False)
    thresholds.to_csv(threshold_path, index=False)
    locked_threshold_rows.to_csv(locked_threshold_path, index=False)
    assignments.to_csv(assignment_path, index=False)
    assignment_summary.to_csv(assignment_summary_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": rel(Path(__file__)),
        "inputs": {
            "query_table": rel(args.query_table),
            "policy_rows": rel(args.policy_rows),
        },
        "outputs": {
            "summary": rel(summary_path),
            "per_split": rel(per_split_path),
            "thresholds": rel(threshold_path),
            "locked_thresholds": rel(locked_threshold_path),
            "assignments": rel(assignment_path),
            "assignment_summary": rel(assignment_summary_path),
        },
        "target": float(args.target),
        "outer_repeats": int(args.outer_repeats),
        "inner_repeats": int(args.inner_repeats),
        "recommended": recommended.to_dict(),
        "claim_boundary": f"Nested species-split diagnostic for high-coverage {args.rank}/no-call repair. Treat recommended assignments as diagnostic until independently locked for deployment.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in [
        summary_path,
        per_split_path,
        threshold_path,
        locked_threshold_path,
        assignment_path,
        assignment_summary_path,
        manifest_path,
    ]:
        logger.log(f"Wrote {rel(path)}")
    logger.done(SCRIPT_NAME)


if __name__ == "__main__":
    main()
