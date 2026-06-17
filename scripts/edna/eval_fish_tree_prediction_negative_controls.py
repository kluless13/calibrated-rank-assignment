#!/usr/bin/env python3
"""Negative controls for fish-tree ranked candidate predictions."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]


def parse_labels(value: object) -> list[str]:
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return []
    return [part.strip() for part in text.split(",") if part.strip()]


def write_control_predictions(input_csv: Path, output_csv: Path, candidate_labels: list[str], mode: str, seed: int) -> None:
    rng = random.Random(seed)
    df = pd.read_csv(input_csv)
    if mode == "shuffled_labels":
        shuffled = candidate_labels[:]
        rng.shuffle(shuffled)
        mapping = dict(zip(candidate_labels, shuffled, strict=True))
        top_labels = []
        for value in df["top_tree_labels"]:
            labels = [mapping.get(label, label) for label in parse_labels(value)]
            top_labels.append(labels)
    elif mode == "random_ranked":
        max_k = max((len(parse_labels(value)) for value in df["top_tree_labels"]), default=50)
        top_labels = [rng.sample(candidate_labels, k=min(max_k, len(candidate_labels))) for _ in range(len(df))]
    else:
        raise ValueError(mode)

    out = df.copy()
    out["top_tree_labels"] = [json.dumps(labels) for labels in top_labels]
    out["top_scores"] = [json.dumps(list(reversed(range(len(labels))))) for labels in top_labels]
    out["pred_tree_label"] = [labels[0] if labels else "" for labels in top_labels]
    out["pred_score"] = [len(labels) if labels else None for labels in top_labels]
    out.to_csv(output_csv, index=False)


def evaluate(input_dir: Path, predictions: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/edna/eval_zero_shot_candidate_predictions.py",
            "--input-dir",
            str(input_dir),
            "--predictions",
            str(predictions),
            "--output-dir",
            str(output_dir / "zero_shot_metrics"),
            "--top-k",
            "1",
            "5",
            "10",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading candidates from {args.input_dir / 'candidate_species.csv'}")
    candidates = pd.read_csv(args.input_dir / "candidate_species.csv")
    candidate_labels = candidates["tree_label"].astype(str).tolist()
    logger.log(f"Loaded {len(candidate_labels)} candidate labels")
    outputs = {}
    for mode in ["shuffled_labels", "random_ranked"]:
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        prediction_csv = mode_dir / "zero_shot_candidate_predictions.csv"
        logger.log(f"Writing {mode} control predictions to {prediction_csv}")
        write_control_predictions(args.predictions, prediction_csv, candidate_labels, mode, args.seed)
        logger.log(f"Evaluating {mode} control predictions")
        evaluate(args.input_dir, prediction_csv, mode_dir)
        outputs[mode] = {
            "prediction_csv": str(prediction_csv),
            "metrics_json": str(mode_dir / "zero_shot_metrics" / "zero_shot_candidate_metrics.json"),
        }
        logger.log(f"Finished {mode} control")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "source_predictions": str(args.predictions),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "controls": outputs,
    }
    manifest_path = args.output_dir / "negative_control_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
