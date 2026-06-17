#!/usr/bin/env python3
"""Collect Paper 1 candidate-reranker outputs into source tables."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_INPUT_ROOT = PAPER1 / "candidate_reranker"
DEFAULT_OUTPUT_DIR = PAPER1 / "source_tables"


def load_reranker(run_dir: Path, logger: ProgressLogger) -> pd.DataFrame | None:
    summary_path = run_dir / "candidate_reranker_summary.csv"
    manifest_path = run_dir / "candidate_reranker_manifest.json"
    if not summary_path.exists():
        logger.log(f"Skipping {run_dir}: missing candidate_reranker_summary.csv")
        return None
    logger.log(f"Loading {summary_path}")
    frame = pd.read_csv(summary_path)
    frame.insert(0, "reranker", run_dir.name)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        frame.insert(1, "feature_count", len(manifest.get("features", [])))
        frame.insert(2, "top_k", manifest.get("top_k"))
        frame.insert(3, "claim_boundary", manifest.get("claim_boundary", ""))
    else:
        frame.insert(1, "feature_count", pd.NA)
        frame.insert(2, "top_k", pd.NA)
        frame.insert(3, "claim_boundary", "")
    frame["top1_accuracy_delta_vs_baseline"] = frame["reranker_top1_accuracy"] - frame["baseline_top1_accuracy"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    frames: list[pd.DataFrame] = []
    if not args.input_root.exists():
        raise RuntimeError(f"Missing input root: {args.input_root}")
    for run_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        frame = load_reranker(run_dir, logger)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No reranker summaries found under {args.input_root}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "candidate_reranker_summary.csv"
    manifest_path = args.output_dir / "candidate_reranker_summary_manifest.json"
    logger.log(f"Writing {summary_path}")
    out.to_csv(summary_path, index=False)
    manifest = {
        "generated_by": "scripts/edna/build_paper1_candidate_reranker_summary.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "summary": str(summary_path),
        "rerankers": sorted(out["reranker"].dropna().unique().tolist()),
        "claim_boundary": (
            "Candidate-reranker rows are model-development outputs. They show "
            "whether candidate-level DL improves top-k candidate ordering, but "
            "are not a final production rank/no-call policy until calibration "
            "transfers on held-out splits."
        ),
    }
    logger.log(f"Writing {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
