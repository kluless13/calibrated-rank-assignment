#!/usr/bin/env python3
"""Build source tables for Paper 1 phylo-calibrated barcode assignment.

This script intentionally works from copied result artifacts rather than live
training directories. It is meant to make the current Paper 1 state auditable:
retrieval metrics, tree-recovery metrics, tree-distance diagnostics, reference
diagnostics, and candidate-ablation/rank-backoff summaries.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
REMOTE_2026_05_30 = ROOT / "results" / "remote_runs" / "2026-05-30" / "rtx_pro_6000"
REMOTE_2026_05_31 = ROOT / "results" / "remote_runs" / "2026-05-31" / "rtx_pro_6000"
PAPER1_2026_05_30 = REMOTE_2026_05_30 / "paper1_phylo_calibrated_assignment"
PAPER1_2026_05_31 = REMOTE_2026_05_31 / "paper1_phylo_calibrated_assignment"
ENCODERS_2026_05_30 = REMOTE_2026_05_30 / "paper1_encoder_benchmarks"
INPUT_ROOT = ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs"
LOCAL_PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
OUT_DIR = LOCAL_PAPER1 / "source_tables"

RANKS = ("species", "genus", "family", "order")
SPLITS = ("eval_c", "seen_test", "unseen_genera")
TOP_KS = (1, 5, 10)


@dataclass(frozen=True)
class RunInfo:
    method: str
    split: str
    family: str
    seed: str = ""
    variant: str = ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percent(value: object) -> float | str:
    if value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value_f):
        return ""
    return 100.0 * value_f


def finite_float(value: object) -> float | str:
    if value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value_f):
        return ""
    return value_f


def strip_split_suffix(name: str) -> tuple[str, str]:
    for split in ("unseen_genera", "seen_test"):
        suffix = f"_{split}"
        if name.endswith(suffix):
            return name[: -len(suffix)], split
    return name, "eval_c"


def classify_mamba_dir(name: str) -> RunInfo | None:
    prefix = "coi_fish_tree_clean_phylo_mamba_"
    if not name.startswith(prefix):
        return None
    base, split = strip_split_suffix(name)
    variant = base.removeprefix(prefix)
    seed = "1206"
    for candidate_seed in ("1207", "1208"):
        suffix = f"_seed{candidate_seed}"
        if variant.endswith(suffix):
            variant = variant[: -len(suffix)]
            seed = candidate_seed
            break
    method = f"mamba_{variant}_seed{seed}"
    return RunInfo(method=method, split=split, family="neural", seed=seed, variant=variant)


def classify_encoder_dir(name: str) -> RunInfo | None:
    if not name.startswith("coi_") or "_tree_recovery_" in name:
        return None
    base, split = strip_split_suffix(name)
    if not base.endswith("_seed1206"):
        return None
    encoder = base.removeprefix("coi_").removesuffix("_seed1206")
    return RunInfo(
        method=f"{encoder}_seed1206",
        split=split,
        family="neural",
        seed="1206",
        variant=encoder,
    )


def classify_baseline_path(path: Path) -> RunInfo | None:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part.startswith("baselines_") and idx + 1 < len(parts):
            return RunInfo(
                method=parts[idx + 1],
                split=part.removeprefix("baselines_"),
                family="sequence_baseline",
            )
        if part.startswith("negative_controls_seed") and idx + 1 < len(parts):
            stem = part.removeprefix("negative_controls_seed")
            seed, split = stem.split("_", 1)
            control = parts[idx + 1]
            return RunInfo(
                method=f"{control}_seed{seed}",
                split=split,
                family="negative_control",
                seed=seed,
                variant=control,
            )
    return None


def classify_prediction_root(root: Path) -> RunInfo | None:
    if root.parent == ENCODERS_2026_05_30:
        return classify_encoder_dir(root.name)
    if root.parent == REMOTE_2026_05_30:
        return classify_mamba_dir(root.name)
    if PAPER1_2026_05_30 in root.parents or PAPER1_2026_05_31 in root.parents:
        candidate = root / "zero_shot_metrics" / "zero_shot_candidate_metrics.json"
        if candidate.exists():
            return classify_baseline_path(candidate)
    return None


def classify_tree_recovery_dir(name: str) -> RunInfo | None:
    for split in ("unseen_genera", "eval_c"):
        suffix = f"_tree_recovery_{split}"
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base.startswith("coi_fish_tree_clean_phylo_mamba_"):
                info = classify_mamba_dir(base)
                if info:
                    return RunInfo(info.method, split, info.family, info.seed, info.variant)
            if base.startswith("coi_"):
                info = classify_encoder_dir(base)
                if info:
                    return RunInfo(info.method, split, info.family, info.seed, info.variant)
    return None


def metric_json_paths() -> list[Path]:
    paths: list[Path] = []
    if REMOTE_2026_05_30.exists():
        paths.extend(REMOTE_2026_05_30.glob("coi_fish_tree_clean_phylo_mamba*/zero_shot_metrics/zero_shot_candidate_metrics.json"))
    if ENCODERS_2026_05_30.exists():
        paths.extend(ENCODERS_2026_05_30.glob("coi_*seed1206*/zero_shot_metrics/zero_shot_candidate_metrics.json"))
    if PAPER1_2026_05_31.exists():
        paths.extend(PAPER1_2026_05_31.glob("baselines_*/*/zero_shot_metrics/zero_shot_candidate_metrics.json"))
    if PAPER1_2026_05_30.exists():
        paths.extend(PAPER1_2026_05_30.glob("negative_controls_seed*/*/zero_shot_metrics/zero_shot_candidate_metrics.json"))
        # Keep older baselines only when the refreshed 2026-05-31 copy is absent.
        if not PAPER1_2026_05_31.exists():
            paths.extend(PAPER1_2026_05_30.glob("baselines_*/*/zero_shot_metrics/zero_shot_candidate_metrics.json"))
    return sorted(set(paths))


def tree_recovery_paths() -> list[Path]:
    paths: list[Path] = []
    if REMOTE_2026_05_30.exists():
        paths.extend(REMOTE_2026_05_30.glob("coi_fish_tree_clean_phylo_mamba*_tree_recovery_*/tree_recovery_metrics.json"))
    if ENCODERS_2026_05_30.exists():
        paths.extend(ENCODERS_2026_05_30.glob("coi_*_tree_recovery_*/tree_recovery_metrics.json"))
    return sorted(set(paths))


def prediction_roots() -> list[Path]:
    roots: list[Path] = []
    for metrics_path in metric_json_paths():
        root = metrics_path.parents[1]
        if (root / "zero_shot_candidate_predictions.csv").exists():
            roots.append(root)
    return sorted(set(roots))


def build_retrieval_metrics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in metric_json_paths():
        info = classify_prediction_root(path.parents[1])
        if not info:
            continue
        metrics = read_json(path).get("metrics", {})
        for rank in RANKS:
            rank_metrics = metrics.get(rank, {})
            rows.append(
                {
                    "method": info.method,
                    "method_family": info.family,
                    "split": info.split,
                    "seed": info.seed,
                    "variant": info.variant,
                    "rank": rank,
                    "top1_pct": percent(rank_metrics.get("top1")),
                    "top5_pct": percent(rank_metrics.get("top5")),
                    "top10_pct": percent(rank_metrics.get("top10")),
                    "eligible_queries": rank_metrics.get("eligible_queries", ""),
                    "mean_first_hit_rank": finite_float(rank_metrics.get("mean_first_hit_rank")),
                    "source_file": rel(path),
                }
            )
    return rows


def build_tree_recovery_metrics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in tree_recovery_paths():
        info = classify_tree_recovery_dir(path.parent.name)
        if not info:
            continue
        metrics = read_json(path).get("metrics", {})
        for comparison, values in metrics.items():
            rows.append(
                {
                    "method": info.method,
                    "method_family": info.family,
                    "split": info.split,
                    "seed": info.seed,
                    "variant": info.variant,
                    "comparison": comparison,
                    "pearson_r": finite_float(values.get("pearson_r")),
                    "spearman_r": finite_float(values.get("spearman_r")),
                    "n_pairs": values.get("n_pairs", ""),
                    "source_file": rel(path),
                }
            )
    return rows


def quantile_bins(values: pd.Series, n_bins: int = 10) -> pd.Series:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(["missing"] * len(values), index=values.index)
    if finite.nunique() < 2:
        return pd.Series(["all_finite" if pd.notna(v) else "missing" for v in values], index=values.index)
    out = pd.Series(["missing"] * len(values), index=values.index, dtype="object")
    out.loc[finite.index] = pd.qcut(finite, q=min(n_bins, finite.nunique()), duplicates="drop").astype(str)
    return out


def build_tree_distance_diagnostics() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    bin_rows: list[dict[str, object]] = []
    sample_summary_rows: list[dict[str, object]] = []
    for metrics_path in tree_recovery_paths():
        info = classify_tree_recovery_dir(metrics_path.parent.name)
        if not info:
            continue
        pairs_path = metrics_path.parent / "sampled_tree_recovery_pairs.csv"
        if not pairs_path.exists():
            continue
        pairs = pd.read_csv(pairs_path)
        pairs["tree_distance"] = pd.to_numeric(pairs["tree_distance"], errors="coerce")
        pairs["embedding_cosine_distance"] = pd.to_numeric(pairs["embedding_cosine_distance"], errors="coerce")
        for pair_set, group in pairs.groupby("pair_set", dropna=False):
            group = group.dropna(subset=["tree_distance", "embedding_cosine_distance"]).copy()
            if group.empty:
                continue
            corr = group[["tree_distance", "embedding_cosine_distance"]].corr(method="pearson").iloc[0, 1]
            spear = group[["tree_distance", "embedding_cosine_distance"]].corr(method="spearman").iloc[0, 1]
            sample_summary_rows.append(
                {
                    "method": info.method,
                    "split": info.split,
                    "pair_set": pair_set,
                    "n_pairs": len(group),
                    "sample_pearson_r": finite_float(corr),
                    "sample_spearman_r": finite_float(spear),
                    "tree_distance_median": group["tree_distance"].median(),
                    "embedding_distance_median": group["embedding_cosine_distance"].median(),
                    "source_file": rel(pairs_path),
                }
            )
            x = group["tree_distance"].to_numpy()
            y = group["embedding_cosine_distance"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            group["residual"] = y - (slope * x + intercept)
            group["tree_distance_bin"] = quantile_bins(group["tree_distance"])
            for bin_label, bin_group in group.groupby("tree_distance_bin", dropna=False):
                bin_rows.append(
                    {
                        "method": info.method,
                        "split": info.split,
                        "pair_set": pair_set,
                        "tree_distance_bin": bin_label,
                        "n_pairs": len(bin_group),
                        "tree_distance_min": bin_group["tree_distance"].min(),
                        "tree_distance_median": bin_group["tree_distance"].median(),
                        "tree_distance_max": bin_group["tree_distance"].max(),
                        "embedding_distance_mean": bin_group["embedding_cosine_distance"].mean(),
                        "embedding_distance_median": bin_group["embedding_cosine_distance"].median(),
                        "linear_residual_mean": bin_group["residual"].mean(),
                        "linear_residual_median": bin_group["residual"].median(),
                        "source_file": rel(pairs_path),
                    }
                )
    return bin_rows, sample_summary_rows


def load_candidate_taxonomy(split: str) -> dict[str, dict[str, str]]:
    path = INPUT_ROOT / split / "candidate_species.csv"
    if not path.exists():
        return {}
    rows = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, dict[str, str]] = {}
    for _, row in rows.iterrows():
        out[row["tree_label"]] = {
            "species": row.get("species_name", ""),
            "genus": row.get("genus_name", ""),
            "family": row.get("family_name", ""),
            "order": row.get("order_name", ""),
        }
    return out


def parse_list(value: object) -> list:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    return parsed if isinstance(parsed, list) else []


def rank_match(label: str, true_values: dict[str, str], taxonomy: dict[str, dict[str, str]], rank: str) -> bool:
    if rank == "species":
        return label == true_values["tree_label"]
    return taxonomy.get(label, {}).get(rank, "") == true_values.get(rank, "")


def first_hit(labels: list[str], true_values: dict[str, str], taxonomy: dict[str, dict[str, str]], rank: str) -> int | None:
    for idx, label in enumerate(labels, start=1):
        if rank_match(label, true_values, taxonomy, rank):
            return idx
    return None


def build_neighborhood_and_ablation() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    neighborhood_rows: list[dict[str, object]] = []
    ablation_acc: dict[
        tuple[str, str, str, str, str, str, str],
        dict[str, object],
    ] = {}
    for root in prediction_roots():
        info = classify_prediction_root(root)
        if not info:
            continue
        if info.family == "negative_control":
            continue
        if info.method.startswith("mamba_") and info.method != "mamba_cosine512_seqval_seed1206":
            continue
        pred_path = root / "zero_shot_candidate_predictions.csv"
        taxonomy = load_candidate_taxonomy(info.split)
        if not taxonomy:
            continue
        candidate_count = len(taxonomy)
        rank_value_counts: dict[str, dict[str, int]] = {rank: {} for rank in ("genus", "family", "order")}
        for tax in taxonomy.values():
            for rank in ("genus", "family", "order"):
                value = tax.get(rank, "")
                rank_value_counts[rank][value] = rank_value_counts[rank].get(value, 0) + 1
        predictions = pd.read_csv(pred_path)
        by_rank_counts: dict[str, list[float]] = {rank: [] for rank in ("genus", "family", "order")}
        by_rank_expected: dict[str, list[float]] = {rank: [] for rank in ("genus", "family", "order")}
        topk_hits: dict[tuple[str, int], list[float]] = {
            (rank, k): [] for rank in ("genus", "family", "order") for k in TOP_KS
        }
        for _, row in predictions.iterrows():
            labels = [str(label) for label in parse_list(row.get("top_tree_labels"))]
            true_values = {
                "tree_label": str(row.get("true_tree_label", "")),
                "species": str(row.get("species_name", "")),
                "genus": str(row.get("genus_name", "")),
                "family": str(row.get("family_name", "")),
                "order": str(row.get("order_name", "")),
            }
            for rank in ("genus", "family", "order"):
                if candidate_count:
                    expected_count = rank_value_counts[rank].get(true_values.get(rank, ""), 0)
                    by_rank_expected[rank].append(expected_count / candidate_count)
                for k in TOP_KS:
                    top = labels[:k]
                    if top:
                        topk_hits[(rank, k)].append(
                            sum(rank_match(label, true_values, taxonomy, rank) for label in top) / len(top)
                        )
            ablation_specs = {
                "hide_true_species": {"drop_rank": "species", "target_ranks": ("genus", "family", "order")},
                "hide_true_genus": {"drop_rank": "genus", "target_ranks": ("family", "order")},
                "hide_true_family": {"drop_rank": "family", "target_ranks": ("order",)},
            }
            for ablation, spec in ablation_specs.items():
                drop_rank = spec["drop_rank"]
                filtered = [
                    label for label in labels if not rank_match(label, true_values, taxonomy, drop_rank)
                ]
                for target_rank in spec["target_ranks"]:
                    hit = first_hit(filtered, true_values, taxonomy, target_rank)
                    key = (
                        info.method,
                        info.family,
                        info.split,
                        ablation,
                        target_rank,
                        rel(pred_path),
                        info.seed,
                    )
                    acc = ablation_acc.setdefault(
                        key,
                        {
                            "n_queries": 0,
                            "top1": 0,
                            "top5": 0,
                            "top10": 0,
                            "hit_ranks": [],
                        },
                    )
                    acc["n_queries"] = int(acc["n_queries"]) + 1
                    acc["top1"] = int(acc["top1"]) + int(hit is not None and hit <= 1)
                    acc["top5"] = int(acc["top5"]) + int(hit is not None and hit <= 5)
                    acc["top10"] = int(acc["top10"]) + int(hit is not None and hit <= 10)
                    if hit is not None:
                        acc["hit_ranks"].append(hit)
        for rank in ("genus", "family", "order"):
            expected_rate = float(np.mean(by_rank_expected[rank])) if by_rank_expected[rank] else math.nan
            for k in TOP_KS:
                observed = float(np.mean(topk_hits[(rank, k)])) if topk_hits[(rank, k)] else math.nan
                neighborhood_rows.append(
                    {
                        "method": info.method,
                        "method_family": info.family,
                        "split": info.split,
                        "rank": rank,
                        "top_k": k,
                        "mean_topk_same_rank_fraction": finite_float(observed),
                        "mean_candidate_pool_same_rank_fraction": finite_float(expected_rate),
                        "enrichment_vs_candidate_pool": finite_float(
                            observed / expected_rate if expected_rate and not math.isnan(expected_rate) else math.nan
                        ),
                        "n_queries": len(predictions),
                        "source_file": rel(pred_path),
                    }
                )
    ablation_summary = summarize_ablation(ablation_acc)
    return neighborhood_rows, ablation_summary


def summarize_ablation(
    acc: dict[tuple[str, str, str, str, str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for keys, values in acc.items():
        method, family, split, ablation, rank, source_file, seed = keys
        n_queries = int(values["n_queries"])
        hit_ranks = values["hit_ranks"]
        hit_array = np.array(hit_ranks, dtype=float) if hit_ranks else np.array([], dtype=float)
        out.append(
            {
                "method": method,
                "method_family": family,
                "seed": seed,
                "split": split,
                "ablation": ablation,
                "target_rank": rank,
                "n_queries": n_queries,
                "top1_pct": 100.0 * int(values["top1"]) / n_queries if n_queries else "",
                "top5_pct": 100.0 * int(values["top5"]) / n_queries if n_queries else "",
                "top10_pct": 100.0 * int(values["top10"]) / n_queries if n_queries else "",
                "mean_first_hit_rank": float(hit_array.mean()) if len(hit_array) else "",
                "median_first_hit_rank": float(np.median(hit_array)) if len(hit_array) else "",
                "source_file": source_file,
            }
        )
    return out


def build_reference_diagnostics_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root in (LOCAL_PAPER1, PAPER1_2026_05_31, PAPER1_2026_05_30):
        if not root.exists():
            continue
        for path in sorted(root.glob("reference_diagnostics_*/*_nearest_reference_bins.csv")):
            split = path.parent.name.removeprefix("reference_diagnostics_")
            method = path.name.removesuffix("_nearest_reference_bins.csv")
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    row.update({"split": split, "method": method, "source_file": rel(path)})
                    rows.append(row)
        # Prefer the refreshed diagnostics root when present.
        if rows:
            break
    return rows


def build_manifest(files_written: list[Path], row_counts: dict[str, int]) -> None:
    manifest = {
        "generated_by": "scripts/edna/build_paper1_source_tables.py",
        "output_dir": rel(OUT_DIR),
        "inputs": {
            "remote_2026_05_30": rel(REMOTE_2026_05_30),
            "remote_2026_05_31": rel(REMOTE_2026_05_31),
            "fish_tree_inputs": rel(INPUT_ROOT),
        },
        "files": [
            {"path": rel(path), "rows": row_counts.get(path.name, 0)}
            for path in files_written
        ],
        "notes": [
            "Tree-distance diagnostics use sampled pairs emitted by the tree-recovery evaluator.",
            "Candidate-ablation tables are post-hoc top-50 rank-backoff diagnostics; they do not retrain models.",
            "Reference-diagnostic nearest-reference distances use normalized tree labels because source tree leaves use spaces while clean input packs use underscores.",
        ],
    }
    path = OUT_DIR / "source_table_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    logger = ProgressLogger(default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: list[tuple[str, list[dict[str, object]], list[str]]] = []
    logger.log("Building retrieval metrics")
    outputs.append(
        (
            "retrieval_metrics.csv",
            build_retrieval_metrics(),
            [
                "method",
                "method_family",
                "split",
                "seed",
                "variant",
                "rank",
                "top1_pct",
                "top5_pct",
                "top10_pct",
                "eligible_queries",
                "mean_first_hit_rank",
                "source_file",
            ],
        )
    )
    logger.log("Building tree recovery metrics")
    outputs.append(
        (
            "tree_recovery_metrics.csv",
            build_tree_recovery_metrics(),
            [
                "method",
                "method_family",
                "split",
                "seed",
                "variant",
                "comparison",
                "pearson_r",
                "spearman_r",
                "n_pairs",
                "source_file",
            ],
        )
    )
    logger.log("Building tree-distance diagnostics")
    tree_bins, tree_sample_summary = build_tree_distance_diagnostics()
    outputs.append(
        (
            "tree_distance_bin_summary.csv",
            tree_bins,
            [
                "method",
                "split",
                "pair_set",
                "tree_distance_bin",
                "n_pairs",
                "tree_distance_min",
                "tree_distance_median",
                "tree_distance_max",
                "embedding_distance_mean",
                "embedding_distance_median",
                "linear_residual_mean",
                "linear_residual_median",
                "source_file",
            ],
        )
    )
    outputs.append(
        (
            "tree_distance_sample_summary.csv",
            tree_sample_summary,
            [
                "method",
                "split",
                "pair_set",
                "n_pairs",
                "sample_pearson_r",
                "sample_spearman_r",
                "tree_distance_median",
                "embedding_distance_median",
                "source_file",
            ],
        )
    )
    logger.log("Building neighborhood and candidate-ablation diagnostics")
    neighborhood, ablation = build_neighborhood_and_ablation()
    outputs.append(
        (
            "neighborhood_preservation.csv",
            neighborhood,
            [
                "method",
                "method_family",
                "split",
                "rank",
                "top_k",
                "mean_topk_same_rank_fraction",
                "mean_candidate_pool_same_rank_fraction",
                "enrichment_vs_candidate_pool",
                "n_queries",
                "source_file",
            ],
        )
    )
    logger.log("Building reference diagnostics summary")
    outputs.append(
        (
            "candidate_ablation_rank_backoff.csv",
            ablation,
            [
                "method",
                "method_family",
                "seed",
                "split",
                "ablation",
                "target_rank",
                "n_queries",
                "top1_pct",
                "top5_pct",
                "top10_pct",
                "mean_first_hit_rank",
                "median_first_hit_rank",
                "source_file",
            ],
        )
    )
    outputs.append(
        (
            "reference_diagnostics_summary.csv",
            build_reference_diagnostics_summary(),
            [
                "split",
                "method",
                "prediction_set",
                "nearest_reference_bin",
                "n_query",
                "nearest_reference_tree_distance_min",
                "nearest_reference_tree_distance_median",
                "nearest_reference_tree_distance_max",
                "species_top1",
                "species_top5",
                "species_top10",
                "genus_top1",
                "genus_top5",
                "genus_top10",
                "family_top1",
                "family_top5",
                "family_top10",
                "order_top1",
                "order_top5",
                "order_top10",
                "source_file",
            ],
        )
    )

    row_counts: dict[str, int] = {}
    files_written: list[Path] = []
    for name, rows, fieldnames in outputs:
        path = OUT_DIR / name
        logger.log(f"Writing {name} with {len(rows)} rows")
        write_csv(path, rows, fieldnames)
        row_counts[name] = len(rows)
        files_written.append(path)
    build_manifest(files_written, row_counts)
    logger.log("Wrote source_table_manifest.json")

    for name, count in row_counts.items():
        print(f"{name}: {count} rows")
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
