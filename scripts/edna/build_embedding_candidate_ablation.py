#!/usr/bin/env python3
"""Build full-candidate rank-backoff diagnostics from saved query embeddings."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from phylo_zero_shot_common import load_query_embedding_npz, load_tree_embedding_npz  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RANKS = ["species", "genus", "family", "order"]
TOP_KS = [1, 5, 10]


@dataclass(frozen=True)
class EmbeddingRun:
    path: Path
    method: str
    seed: str
    split: str


def parse_run(path: Path) -> EmbeddingRun | None:
    name = path.parent.name
    parts = name.split("_")
    if len(parts) < 3 or parts[0] != "coi":
        return None
    method = parts[1]
    seed = parts[2].removeprefix("seed")
    split = "_".join(parts[3:]) if len(parts) > 3 else "eval_c"
    return EmbeddingRun(path=path, method=method, seed=seed, split=split)


def resolve_tree_npz(raw_path: str, search_roots: list[Path]) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    for root in search_roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    basename = path.parent.name
    filename = path.name
    for root in search_roots:
        matches = list(root.glob(f"**/{basename}/{filename}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve tree embedding NPZ from {raw_path!r}")


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def load_candidate_taxonomy(input_dir: Path, candidate_labels: list[str]) -> pd.DataFrame:
    species_info_path = input_dir / "species_info.json"
    species_info = json.loads(species_info_path.read_text()) if species_info_path.exists() else {}
    candidate_path = input_dir / "candidate_species.csv"
    candidate_rows = {}
    if candidate_path.exists():
        candidate_rows = pd.read_csv(candidate_path).set_index("tree_label").to_dict(orient="index")

    rows: list[dict[str, str]] = []
    for label in candidate_labels:
        info = species_info.get(label, {})
        row = candidate_rows.get(label, {})
        genus = clean(info.get("genus")) or clean(row.get("genus_name")) or clean(row.get("genus_from_label"))
        family = clean(info.get("family")) or clean(row.get("family_name"))
        order = clean(info.get("order")) or clean(row.get("order_name"))
        rows.append(
            {
                "tree_label": label,
                "species": label,
                "genus": genus,
                "family": family,
                "order": order,
            }
        )
    return pd.DataFrame(rows)


def load_queries(input_dir: Path, processids: list[str]) -> pd.DataFrame:
    queries = pd.read_csv(input_dir / "zero_shot_queries.csv")
    queries["processid"] = queries["processid"].astype(str)
    by_processid = queries.set_index("processid", drop=False)
    missing = [processid for processid in processids if processid not in by_processid.index]
    if missing:
        raise RuntimeError(f"{len(missing)} query processids were not found in {input_dir}/zero_shot_queries.csv")
    ordered = by_processid.loc[processids].reset_index(drop=True)
    return ordered


def query_rank_values(queries: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["species"] = queries["tree_label"].astype(str)
    out["genus"] = queries.get("genus_name", "").map(clean)
    out["family"] = queries.get("family_name", "").map(clean)
    out["order"] = queries.get("order_name", "").map(clean)
    return out


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def build_mask_index(candidate_tax: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    index: dict[str, dict[str, np.ndarray]] = {}
    for rank in RANKS:
        values = candidate_tax[rank].astype(str).to_numpy()
        rank_index: dict[str, np.ndarray] = {}
        for value in sorted(set(values)):
            if value:
                rank_index[value] = values == value
        index[rank] = rank_index
    return index


def score_run(run: EmbeddingRun, search_roots: list[Path]) -> list[dict[str, object]]:
    processids, query_embeddings, metadata = load_query_embedding_npz(run.path)
    input_dir = Path(metadata["input_dir"])
    tree_npz = resolve_tree_npz(str(metadata["tree_embedding_npz"]), search_roots)
    candidate_labels, tree_embeddings, _ = load_tree_embedding_npz(tree_npz)

    queries = load_queries(input_dir, processids)
    query_tax = query_rank_values(queries)
    candidate_tax = load_candidate_taxonomy(input_dir, candidate_labels)

    q = normalize(query_embeddings.astype(np.float32))
    c = normalize(tree_embeddings.astype(np.float32))
    ablations = {
        "none": [],
        "hide_true_species": ["species"],
        "hide_true_genus": ["genus"],
        "hide_true_family": ["family"],
    }
    valid_targets = {
        "none": RANKS,
        "hide_true_species": ["genus", "family", "order"],
        "hide_true_genus": ["family", "order"],
        "hide_true_family": ["order"],
    }

    rows: list[dict[str, object]] = []
    candidate_rank_values = {rank: candidate_tax[rank].astype(str).to_numpy() for rank in RANKS}
    mask_index = build_mask_index(candidate_tax)
    all_allowed = np.ones(len(candidate_tax), dtype=bool)
    block_size = 2048

    for ablation, hidden_ranks in ablations.items():
        candidate_counts: list[int] = []
        allowed_masks: list[np.ndarray] = []
        for q_idx in range(len(query_tax)):
            keep = np.ones(len(candidate_tax), dtype=bool)
            for hidden_rank in hidden_ranks:
                value = clean(query_tax.iloc[q_idx][hidden_rank])
                if value:
                    keep &= candidate_rank_values[hidden_rank] != value
            allowed_masks.append(keep if hidden_ranks else all_allowed)
            candidate_counts.append(int(keep.sum()) if hidden_ranks else len(candidate_tax))

        for target_rank in valid_targets[ablation]:
            first_ranks: list[int] = []
            top_hits = {k: 0 for k in TOP_KS}
            eligible = 0
            for start in range(0, len(query_tax), block_size):
                end = min(start + block_size, len(query_tax))
                scores = q[start:end] @ c.T
                for local_idx, row_scores in enumerate(scores):
                    q_idx = start + local_idx
                    target = clean(query_tax.iloc[q_idx][target_rank])
                    if not target:
                        continue
                    target_mask = mask_index[target_rank].get(target)
                    if target_mask is None:
                        continue
                    allowed = allowed_masks[q_idx]
                    target_allowed = allowed & target_mask
                    if not target_allowed.any():
                        continue
                    eligible += 1
                    best_target_score = float(row_scores[target_allowed].max())
                    rank = int(np.count_nonzero(row_scores[allowed] > best_target_score) + 1)
                    first_ranks.append(rank)
                    for k in TOP_KS:
                        if rank <= k:
                            top_hits[k] += 1
            if eligible == 0:
                continue
            rows.append(
                {
                    "method": run.method,
                    "seed": run.seed,
                    "split": run.split,
                    "ablation": ablation,
                    "target_rank": target_rank,
                    "n_queries": eligible,
                    "candidate_count_mean": float(np.mean(candidate_counts)),
                    "candidate_count_min": int(np.min(candidate_counts)),
                    "candidate_count_max": int(np.max(candidate_counts)),
                    "top1_pct": 100.0 * top_hits[1] / eligible,
                    "top5_pct": 100.0 * top_hits[5] / eligible,
                    "top10_pct": 100.0 * top_hits[10] / eligible,
                    "mean_first_hit_rank": float(np.mean(first_ranks)) if first_ranks else None,
                    "median_first_hit_rank": float(np.median(first_ranks)) if first_ranks else None,
                    "source_file": str(run.path),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "seed",
        "split",
        "ablation",
        "target_rank",
        "n_queries",
        "candidate_count_mean",
        "candidate_count_min",
        "candidate_count_max",
        "top1_pct",
        "top5_pct",
        "top10_pct",
        "mean_first_hit_rank",
        "median_first_hit_rank",
        "source_file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-embedding-root",
        type=Path,
        default=Path("results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/source_tables/full_candidate_embedding_ablation.csv"),
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[
            Path("."),
            Path("results/remote_runs/2026-05-30/rtx_pro_6000"),
            Path("results/remote_runs/2026-05-31/rtx_pro_6000"),
        ],
    )
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    rows: list[dict[str, object]] = []
    logger.log(f"Searching embedding root {args.query_embedding_root}")
    for path in sorted(args.query_embedding_root.glob("*/query_embeddings.npz")):
        run = parse_run(path)
        if run is None:
            logger.log(f"Skipping unrecognized embedding run: {path}")
            continue
        logger.log(f"Scoring candidate ablations for {run.method} seed={run.seed} split={run.split}")
        rows.extend(score_run(run, args.search_root))
        logger.log(f"Current ablation rows: {len(rows)}")
    logger.log(f"Writing candidate-ablation table to {args.output_csv}")
    write_csv(args.output_csv, rows)
    manifest = {
        "query_embedding_root": str(args.query_embedding_root),
        "output_csv": str(args.output_csv),
        "rows": len(rows),
    }
    args.output_csv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {args.output_csv.with_suffix('.manifest.json')}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
