#!/usr/bin/env python3
"""Build hierarchical set-valued MarkerMirror candidate diagnostics.

The single-label family/genus repair did not transfer cleanly. This script asks
a different question: can the pipeline return a small set of plausible taxa at a
rank, calibrated on held-out species splits, rather than forcing one label?

Labels are used only for calibration/evaluation. Candidate sets are built from
production-available MarkerMirror, BLASTN, and VSEARCH candidate lists.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("genus", "family", "order")
SOURCE_POLICIES = (
    "blast",
    "vsearch",
    "marker_mirror",
    "blast_vsearch_union",
    "blast_vsearch_intersection",
    "all_union",
    "all_intersection",
)
SOURCE_MAP = {
    "blast": ("blast",),
    "vsearch": ("vsearch",),
    "marker_mirror": ("marker_mirror",),
    "blast_vsearch_union": ("blast", "vsearch"),
    "blast_vsearch_intersection": ("blast", "vsearch"),
    "all_union": ("marker_mirror", "blast", "vsearch"),
    "all_intersection": ("marker_mirror", "blast", "vsearch"),
}


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
        / "remote_runs"
        / "2026-06-03"
        / "rtx_pro_6000"
        / "marker_mirror_12s_production_v1"
        / "vast_full_all_queries_20260603"
        / "features"
        / "marker_mirror_candidates_for_policy.csv.gz",
    )
    parser.add_argument(
        "--blast-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "remote_runs"
        / "2026-06-03"
        / "rtx_pro_6000"
        / "marker_mirror_12s_production_v1"
        / "vast_full_all_queries_20260603"
        / "blast"
        / "marker_mirror_same_marker_blast_candidates_top50.csv.gz",
    )
    parser.add_argument(
        "--vsearch-candidates",
        type=Path,
        default=ROOT
        / "results"
        / "remote_runs"
        / "2026-06-03"
        / "rtx_pro_6000"
        / "marker_mirror_12s_production_v1"
        / "vast_full_all_queries_20260603"
        / "vsearch"
        / "marker_mirror_same_marker_vsearch_candidates_top50.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables",
    )
    parser.add_argument("--top-ks", default="1,3,5,10,20,50")
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--calibration-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--min-calibration-queries", type=int, default=200)
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


def split_species(
    species: Iterable[str],
    rng: np.random.Generator,
    fraction: float,
) -> tuple[set[str], set[str]]:
    values = np.array(sorted(clean(value) for value in species if clean(value)))
    rng.shuffle(values)
    split = max(1, min(len(values) - 1, int(round(len(values) * fraction))))
    return set(values[:split]), set(values[split:])


def load_queries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "query_id": frame["processid"].astype(str),
            "source": frame["source"].astype(str),
            "query_tree_label": frame["tree_label"].astype(str),
            "query_species": frame["species_name"].map(clean),
            "query_genus": frame["genus_name"].map(clean),
            "query_family": frame["family_name"].map(clean),
            "query_order": frame["order_name"].map(clean),
        }
    )


def load_candidates(path: Path, source_name: str) -> pd.DataFrame:
    columns = [
        "query_id",
        "candidate_rank",
        "candidate_genus",
        "candidate_family",
        "candidate_order",
    ]
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    frame["candidate_source"] = source_name
    for rank in RANKS:
        frame[f"candidate_{rank}"] = frame[f"candidate_{rank}"].map(clean)
    frame["candidate_rank"] = pd.to_numeric(frame["candidate_rank"], errors="coerce")
    return frame.dropna(subset=["candidate_rank"]).sort_values(
        ["query_id", "candidate_source", "candidate_rank"]
    )


def unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = clean(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def source_rank_sets(candidates: pd.DataFrame, top_ks: list[int]) -> dict[tuple[str, str, int], list[str]]:
    result: dict[tuple[str, str, int], list[str]] = {}
    for source, source_frame in candidates.groupby("candidate_source", sort=False):
        source_frame = source_frame.sort_values("candidate_rank")
        for rank in RANKS:
            values = source_frame[f"candidate_{rank}"].tolist()
            for top_k in top_ks:
                result[(source, rank, top_k)] = unique_preserve(values[:top_k])
    return result


def combine_sets(
    per_source: dict[tuple[str, str, int], list[str]],
    policy: str,
    rank: str,
    top_k: int,
) -> list[str]:
    sources = SOURCE_MAP[policy]
    lists = [per_source.get((source, rank, top_k), []) for source in sources]
    if policy.endswith("_intersection"):
        if not lists or any(not values for values in lists):
            return []
        shared = set(lists[0])
        for values in lists[1:]:
            shared &= set(values)
        return [value for value in lists[0] if value in shared]
    merged: list[str] = []
    for values in lists:
        merged.extend(values)
    return unique_preserve(merged)


def build_candidate_sets(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    top_ks: list[int],
    logger: ProgressLogger,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meta = queries.set_index("query_id")
    grouped = candidates.groupby("query_id", sort=False)
    for idx, (query_id, group) in enumerate(grouped):
        if idx % 500 == 0:
            logger.log(f"Building candidate sets query {idx + 1:,}/{grouped.ngroups:,}")
        if query_id not in meta.index:
            continue
        query = meta.loc[query_id]
        per_source = source_rank_sets(group, top_ks)
        for rank in RANKS:
            truth = clean(query[f"query_{rank}"])
            for policy in SOURCE_POLICIES:
                for top_k in top_ks:
                    labels = combine_sets(per_source, policy, rank, top_k)
                    rows.append(
                        {
                            "query_id": query_id,
                            "source": query["source"],
                            "query_tree_label": query["query_tree_label"],
                            "rank": rank,
                            "policy": policy,
                            "top_k": int(top_k),
                            "candidate_set": "|".join(labels),
                            "set_size": int(len(labels)),
                            "truth_label": truth,
                            "contains_truth": bool(truth and truth in labels),
                            "nonempty": bool(labels),
                        }
                    )
    return pd.DataFrame(rows)


def choose_policy(
    calibration: pd.DataFrame,
    rank: str,
    target: float,
    strategy: str,
    min_queries: int,
) -> pd.Series | None:
    subset = calibration[(calibration["rank"] == rank) & calibration["nonempty"]].copy()
    total_queries = int(calibration[calibration["rank"] == rank]["query_id"].nunique())
    rows: list[dict[str, Any]] = []
    for keys, group in subset.groupby(["policy", "top_k"], sort=False):
        policy, top_k = keys
        n = int(len(group))
        if n < min_queries:
            continue
        successes = int(group["contains_truth"].sum())
        coverage = successes / n if n else math.nan
        wilson = wilson_lower(successes, n)
        criterion = wilson if strategy == "wilson95" else coverage
        rows.append(
            {
                "policy": policy,
                "top_k": int(top_k),
                "calibration_n": n,
                "calibration_total_queries": total_queries,
                "calibration_emission_coverage_pct": pct(n, total_queries),
                "calibration_coverage": coverage,
                "calibration_wilson95": wilson,
                "criterion": criterion,
                "calibration_mean_set_size": float(group["set_size"].mean()),
                "calibration_median_set_size": float(group["set_size"].median()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    passing = frame[frame["criterion"] >= target].copy()
    if passing.empty:
        return None
    passing = passing.sort_values(
        ["calibration_mean_set_size", "calibration_median_set_size", "top_k", "calibration_coverage"],
        ascending=[True, True, True, False],
    )
    return passing.iloc[0]


def evaluate_policy(
    frame: pd.DataFrame,
    rank: str,
    policy: str,
    top_k: int,
    target: float,
    total_eval_queries: int,
) -> dict[str, Any]:
    subset = frame[
        (frame["rank"] == rank)
        & (frame["policy"] == policy)
        & (frame["top_k"].astype(int) == int(top_k))
        & frame["nonempty"]
    ].copy()
    n = int(len(subset))
    successes = int(subset["contains_truth"].sum()) if n else 0
    return {
        "n_eval_queries": int(total_eval_queries),
        "n_emitted_sets": n,
        "n_containing_truth": successes,
        "set_coverage": successes / n if n else math.nan,
        "emission_coverage_pct": pct(n, total_eval_queries),
        "full_query_truth_coverage_pct": pct(successes, total_eval_queries),
        "target_met": bool(n > 0 and successes / n >= target),
        "mean_set_size": float(subset["set_size"].mean()) if n else math.nan,
        "median_set_size": float(subset["set_size"].median()) if n else math.nan,
        "p90_set_size": float(subset["set_size"].quantile(0.90)) if n else math.nan,
        "max_set_size": int(subset["set_size"].max()) if n else 0,
    }


def nested_evaluate(
    candidate_sets: pd.DataFrame,
    queries: pd.DataFrame,
    target: float,
    repeats: int,
    calibration_fraction: float,
    seed: int,
    min_queries: int,
    logger: ProgressLogger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    species = sorted(queries["query_tree_label"].map(clean).dropna().unique())
    strategies = ("observed", "wilson95")
    per_split_rows: list[dict[str, Any]] = []
    chosen_rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        if repeat % max(1, repeats // 10) == 0:
            logger.log(f"Nested split {repeat + 1}/{repeats}")
        cal_species, eval_species = split_species(species, rng, calibration_fraction)
        cal_ids = set(queries[queries["query_tree_label"].isin(cal_species)]["query_id"])
        eval_ids = set(queries[queries["query_tree_label"].isin(eval_species)]["query_id"])
        cal = candidate_sets[candidate_sets["query_id"].isin(cal_ids)].copy()
        ev = candidate_sets[candidate_sets["query_id"].isin(eval_ids)].copy()
        for rank in RANKS:
            for strategy in strategies:
                chosen = choose_policy(cal, rank, target, strategy, min_queries)
                if chosen is None:
                    row = {
                        "repeat": repeat,
                        "rank": rank,
                        "strategy": strategy,
                        "target": target,
                        "policy": "",
                        "top_k": math.nan,
                        "calibration_n": 0,
                        "calibration_total_queries": len(cal_ids),
                        "calibration_emission_coverage_pct": math.nan,
                        "calibration_coverage": math.nan,
                        "calibration_wilson95": math.nan,
                        "calibration_mean_set_size": math.nan,
                        "n_eval_queries": len(eval_ids),
                        "n_emitted_sets": 0,
                        "n_containing_truth": 0,
                        "set_coverage": math.nan,
                        "emission_coverage_pct": 0.0,
                        "full_query_truth_coverage_pct": 0.0,
                        "target_met": False,
                        "mean_set_size": math.nan,
                        "median_set_size": math.nan,
                        "p90_set_size": math.nan,
                        "max_set_size": 0,
                    }
                else:
                    metrics = evaluate_policy(
                        ev,
                        rank,
                        chosen["policy"],
                        int(chosen["top_k"]),
                        target,
                        total_eval_queries=len(eval_ids),
                    )
                    row = {
                        "repeat": repeat,
                        "rank": rank,
                        "strategy": strategy,
                        "target": target,
                        "policy": chosen["policy"],
                        "top_k": int(chosen["top_k"]),
                        "calibration_n": int(chosen["calibration_n"]),
                        "calibration_total_queries": int(chosen["calibration_total_queries"]),
                        "calibration_emission_coverage_pct": float(
                            chosen["calibration_emission_coverage_pct"]
                        ),
                        "calibration_coverage": float(chosen["calibration_coverage"]),
                        "calibration_wilson95": float(chosen["calibration_wilson95"]),
                        "calibration_mean_set_size": float(chosen["calibration_mean_set_size"]),
                        **metrics,
                    }
                per_split_rows.append(row)
                chosen_rows.append({key: row[key] for key in row if key.startswith("calibration") or key in {
                    "repeat",
                    "rank",
                    "strategy",
                    "target",
                    "policy",
                    "top_k",
                }})
    return pd.DataFrame(per_split_rows), pd.DataFrame(chosen_rows)


def summarize(per_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in per_split.groupby(["rank", "strategy"], sort=False):
        rank, strategy = keys
        rows.append(
            {
                "rank": rank,
                "strategy": strategy,
                "outer_repeats": int(len(group)),
                "target": float(group["target"].iloc[0]),
                "target_met_rate_pct": float(group["target_met"].mean() * 100.0),
                "mean_set_coverage_pct": float(group["set_coverage"].mean() * 100.0),
                "min_set_coverage_pct": float(group["set_coverage"].min() * 100.0),
                "mean_emission_coverage_pct": float(group["emission_coverage_pct"].mean()),
                "mean_full_query_truth_coverage_pct": float(
                    group["full_query_truth_coverage_pct"].mean()
                ),
                "mean_set_size": float(group["mean_set_size"].mean()),
                "median_mean_set_size": float(group["mean_set_size"].median()),
                "mean_p90_set_size": float(group["p90_set_size"].mean()),
                "most_common_policy": clean(group["policy"].mode().iloc[0]) if len(group["policy"].mode()) else "",
                "most_common_top_k": int(group["top_k"].mode().iloc[0]) if len(group["top_k"].mode()) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rank", "target_met_rate_pct", "mean_set_size"],
        ascending=[True, False, True],
    )


def choose_full_policy(summary: pd.DataFrame, per_split: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank, group in summary.groupby("rank", sort=False):
        stable = group[group["target_met_rate_pct"] >= 100.0].copy()
        choice_pool = stable if not stable.empty else group.copy()
        choice = choice_pool.sort_values(
            ["target_met_rate_pct", "mean_set_size", "mean_set_coverage_pct"],
            ascending=[False, True, False],
        ).iloc[0]
        choices = per_split[(per_split["rank"] == rank) & (per_split["strategy"] == choice["strategy"])]
        policy = clean(choices["policy"].mode().iloc[0]) if len(choices["policy"].mode()) else ""
        top_k = int(choices["top_k"].mode().iloc[0]) if len(choices["top_k"].mode()) else 0
        rows.append(
            {
                "rank": rank,
                "strategy": choice["strategy"],
                "policy": policy,
                "top_k": top_k,
                "target_met_rate_pct": choice["target_met_rate_pct"],
                "mean_set_coverage_pct": choice["mean_set_coverage_pct"],
                "mean_emission_coverage_pct": choice["mean_emission_coverage_pct"],
                "mean_full_query_truth_coverage_pct": choice["mean_full_query_truth_coverage_pct"],
                "mean_set_size": choice["mean_set_size"],
                "recommendation": "diagnostic_set_mode" if choice["target_met_rate_pct"] >= 100.0 else "do_not_enable",
            }
        )
    return pd.DataFrame(rows)


def build_full_assignments(candidate_sets: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for row in chosen.itertuples(index=False):
        subset = candidate_sets[
            (candidate_sets["rank"] == row.rank)
            & (candidate_sets["policy"] == row.policy)
            & (candidate_sets["top_k"].astype(int) == int(row.top_k))
        ].copy()
        subset["decision_mode"] = "marker_mirror_hierarchical_candidate_set_diagnostic"
        subset["recommendation"] = row.recommendation
        parts.append(subset)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_policy_grid(candidate_sets: pd.DataFrame) -> pd.DataFrame:
    total_queries = int(candidate_sets["query_id"].nunique())
    rows: list[dict[str, Any]] = []
    for keys, group in candidate_sets.groupby(["rank", "policy", "top_k"], sort=False):
        rank, policy, top_k = keys
        emitted = group[group["nonempty"]].copy()
        n_emitted = int(len(emitted))
        successes = int(emitted["contains_truth"].sum()) if n_emitted else 0
        rows.append(
            {
                "rank": rank,
                "policy": policy,
                "top_k": int(top_k),
                "n_queries": total_queries,
                "n_emitted_sets": n_emitted,
                "emission_coverage_pct": pct(n_emitted, total_queries),
                "set_coverage_pct": pct(successes, n_emitted) if n_emitted else math.nan,
                "full_query_truth_coverage_pct": pct(successes, total_queries),
                "mean_set_size": float(emitted["set_size"].mean()) if n_emitted else math.nan,
                "median_set_size": float(emitted["set_size"].median()) if n_emitted else math.nan,
                "p90_set_size": float(emitted["set_size"].quantile(0.90)) if n_emitted else math.nan,
                "max_set_size": int(emitted["set_size"].max()) if n_emitted else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rank", "set_coverage_pct", "emission_coverage_pct", "mean_set_size"],
        ascending=[True, False, False, True],
    )


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or args.output_dir / "marker_mirror_hierarchical_candidate_sets.log")
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top_ks = [int(value.strip()) for value in args.top_ks.split(",") if value.strip()]

    logger.log(f"Loading queries: {rel(args.query_table)}")
    queries = load_queries(args.query_table)
    logger.log("Loading candidate lists")
    marker = load_candidates(args.marker_mirror_candidates, "marker_mirror")
    blast = load_candidates(args.blast_candidates, "blast")
    vsearch = load_candidates(args.vsearch_candidates, "vsearch")
    candidates = pd.concat([marker, blast, vsearch], ignore_index=True)
    logger.log(
        f"Loaded candidates rows={len(candidates):,} "
        f"queries={candidates['query_id'].nunique():,}"
    )

    candidate_sets = build_candidate_sets(queries, candidates, top_ks, logger)
    logger.log(f"Built candidate-set rows={len(candidate_sets):,}")
    policy_grid = summarize_policy_grid(candidate_sets)
    per_split, chosen = nested_evaluate(
        candidate_sets=candidate_sets,
        queries=queries,
        target=args.target,
        repeats=args.repeats,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        min_queries=args.min_calibration_queries,
        logger=logger,
    )
    summary = summarize(per_split)
    full_policy = choose_full_policy(summary, per_split)
    assignments = build_full_assignments(candidate_sets, full_policy)

    prefix = "marker_mirror_hierarchical_candidate_sets"
    candidate_sets_path = args.output_dir / f"{prefix}_policy_rows.csv.gz"
    policy_grid_path = args.output_dir / f"{prefix}_policy_grid_summary.csv"
    per_split_path = args.output_dir / f"{prefix}_per_split.csv"
    chosen_path = args.output_dir / f"{prefix}_chosen_policies.csv"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    full_policy_path = args.output_dir / f"{prefix}_full_policy.csv"
    assignments_path = args.output_dir / f"{prefix}_assignments.csv.gz"
    manifest_path = args.output_dir / f"{prefix}_manifest.json"

    candidate_sets.to_csv(candidate_sets_path, index=False)
    policy_grid.to_csv(policy_grid_path, index=False)
    per_split.to_csv(per_split_path, index=False)
    chosen.to_csv(chosen_path, index=False)
    summary.to_csv(summary_path, index=False)
    full_policy.to_csv(full_policy_path, index=False)
    assignments.to_csv(assignments_path, index=False)
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
            "policy_rows": rel(candidate_sets_path),
            "policy_grid_summary": rel(policy_grid_path),
            "per_split": rel(per_split_path),
            "chosen_policies": rel(chosen_path),
            "summary": rel(summary_path),
            "full_policy": rel(full_policy_path),
            "assignments": rel(assignments_path),
        },
        "target": float(args.target),
        "repeats": int(args.repeats),
        "top_ks": top_ks,
        "claim_boundary": "Set-valued diagnostic over production-available candidate lists. This is not a single-label family/genus production policy.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in [
        candidate_sets_path,
        policy_grid_path,
        per_split_path,
        chosen_path,
        summary_path,
        full_policy_path,
        assignments_path,
        manifest_path,
    ]:
        logger.log(f"Wrote {rel(path)}")
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
