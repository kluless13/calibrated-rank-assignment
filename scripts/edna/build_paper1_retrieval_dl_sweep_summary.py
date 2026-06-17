#!/usr/bin/env python3
"""Summarize Paper 1 retrieval-DL sweep outputs into source tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(Path(__file__).resolve().parent))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


DEFAULT_INPUT_ROOT = (
    ROOT
    / "results"
    / "remote_runs"
    / "2026-06-02"
    / "rtx_pro_6000"
    / "paper1_retrieval_dl_sweep"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"

SPLIT_SUFFIXES = {
    "_seen_test": "seen_test",
    "_unseen_genera": "unseen_genera",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_from_run_dir(name: str) -> tuple[str, str]:
    for suffix, split in SPLIT_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], split
    return name, "eval_c"


def arm_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    args = manifest.get("args", {})
    split = manifest.get("sequence_train_val_split", {})
    return {
        "model_type": args.get("model_type") or manifest.get("model"),
        "loss_mode": args.get("loss_mode"),
        "seed": args.get("seed") or manifest.get("seed"),
        "embed_dim": args.get("embed_dim"),
        "train_epochs": args.get("train_epochs"),
        "validation_mode": args.get("validation_mode"),
        "train_sequences": split.get("train_sequences"),
        "val_sequences": split.get("val_sequences"),
        "train_species": split.get("train_species"),
        "val_species": split.get("val_species"),
    }


def build_retrieval_rows(input_root: Path, logger: ProgressLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manifests = sorted(input_root.glob("*/run_manifest.json"))
    logger.log(f"Scanning {len(manifests)} run manifests for retrieval metrics")
    for manifest_path in manifests:
        run_dir = manifest_path.parent
        arm, split = split_from_run_dir(run_dir.name)
        manifest = read_json(manifest_path)
        meta = arm_metadata(manifest)
        metrics_path = run_dir / "zero_shot_metrics_top50" / "zero_shot_candidate_metrics.json"
        if not metrics_path.exists():
            metrics_path = run_dir / "zero_shot_metrics" / "zero_shot_candidate_metrics.json"
        if not metrics_path.exists():
            logger.log(f"Skipping {run_dir.name}: no zero-shot metrics found")
            continue
        metrics = read_json(metrics_path)
        for rank, values in sorted(metrics.get("metrics", {}).items()):
            rows.append(
                {
                    "arm": arm,
                    "run_dir": run_dir.name,
                    "split": split,
                    "rank": rank,
                    **meta,
                    "candidate_count": metrics.get("candidate_count"),
                    "query_count": metrics.get("query_count"),
                    "queries_with_predictions": metrics.get("queries_with_predictions"),
                    "eligible_queries": values.get("eligible_queries"),
                    "top1": values.get("top1"),
                    "top5": values.get("top5"),
                    "top10": values.get("top10"),
                    "top50": values.get("top50"),
                    "mean_first_hit_rank": values.get("mean_first_hit_rank"),
                    "metrics_json": str(metrics_path.relative_to(ROOT)),
                }
            )
    logger.log(f"Built {len(rows)} retrieval metric rows")
    return rows


def build_training_rows(input_root: Path, logger: ProgressLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    histories = sorted(input_root.glob("*/train_history.json"))
    logger.log(f"Scanning {len(histories)} training histories")
    for history_path in histories:
        run_dir = history_path.parent
        manifest_path = run_dir / "run_manifest.json"
        meta = arm_metadata(read_json(manifest_path)) if manifest_path.exists() else {}
        history = read_json(history_path)
        for entry in history:
            row = {"arm": run_dir.name, **meta, **entry}
            row["train_history_json"] = str(history_path.relative_to(ROOT))
            rows.append(row)
    logger.log(f"Built {len(rows)} training history rows")
    return rows


def split_from_tree_dir(name: str) -> tuple[str, str] | None:
    for suffix, split in (
        ("_tree_recovery_eval_c", "eval_c"),
        ("_tree_recovery_unseen_genera", "unseen_genera"),
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)], split
    return None


def build_tree_rows(input_root: Path, logger: ProgressLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_paths = sorted(input_root.glob("*_tree_recovery_*/tree_recovery_metrics.json"))
    logger.log(f"Scanning {len(metric_paths)} tree-recovery metric files")
    for metrics_path in metric_paths:
        parsed = split_from_tree_dir(metrics_path.parent.name)
        if parsed is None:
            logger.log(f"Skipping unrecognized tree-recovery directory {metrics_path.parent.name}")
            continue
        arm, split = parsed
        manifest_path = input_root / arm / "run_manifest.json"
        meta = arm_metadata(read_json(manifest_path)) if manifest_path.exists() else {}
        metrics = read_json(metrics_path)
        for pair_set, values in sorted(metrics.get("metrics", {}).items()):
            rows.append(
                {
                    "arm": arm,
                    "split": split,
                    "pair_set": pair_set,
                    **meta,
                    "pearson_r": values.get("pearson_r"),
                    "pearson_p": values.get("pearson_p"),
                    "spearman_r": values.get("spearman_r"),
                    "spearman_p": values.get("spearman_p"),
                    "n_pairs": values.get("n_pairs"),
                    "tree_recovery_json": str(metrics_path.relative_to(ROOT)),
                }
            )
    logger.log(f"Built {len(rows)} tree-recovery rows")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    input_root = args.input_root
    output_dir = args.output_dir
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    retrieval_rows = build_retrieval_rows(input_root, logger)
    training_rows = build_training_rows(input_root, logger)
    tree_rows = build_tree_rows(input_root, logger)

    write_csv(
        output_dir / "retrieval_dl_sweep_summary.csv",
        retrieval_rows,
        [
            "arm",
            "run_dir",
            "split",
            "rank",
            "model_type",
            "loss_mode",
            "seed",
            "embed_dim",
            "train_epochs",
            "validation_mode",
            "train_sequences",
            "val_sequences",
            "train_species",
            "val_species",
            "candidate_count",
            "query_count",
            "queries_with_predictions",
            "eligible_queries",
            "top1",
            "top5",
            "top10",
            "top50",
            "mean_first_hit_rank",
            "metrics_json",
        ],
    )
    write_csv(
        output_dir / "retrieval_dl_sweep_training_history.csv",
        training_rows,
        [
            "arm",
            "model_type",
            "loss_mode",
            "seed",
            "embed_dim",
            "train_epochs",
            "epoch",
            "train_loss",
            "val_loss",
            "train_contrastive",
            "val_contrastive",
            "train_cosine",
            "val_cosine",
            "train_triplet",
            "val_triplet",
            "train_history_json",
        ],
    )
    write_csv(
        output_dir / "retrieval_dl_sweep_tree_recovery.csv",
        tree_rows,
        [
            "arm",
            "split",
            "pair_set",
            "model_type",
            "loss_mode",
            "seed",
            "embed_dim",
            "train_epochs",
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
            "n_pairs",
            "tree_recovery_json",
        ],
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_root": str(input_root.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "retrieval_rows": len(retrieval_rows),
        "training_rows": len(training_rows),
        "tree_recovery_rows": len(tree_rows),
        "outputs": [
            "retrieval_dl_sweep_summary.csv",
            "retrieval_dl_sweep_training_history.csv",
            "retrieval_dl_sweep_tree_recovery.csv",
        ],
    }
    manifest_path = output_dir / "retrieval_dl_sweep_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    logger.log(f"Wrote {manifest_path.relative_to(ROOT)}")
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
