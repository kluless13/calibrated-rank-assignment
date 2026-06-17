#!/usr/bin/env python3
"""Collect Paper 1 reference-gap detector outputs into source tables."""
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
DEFAULT_INPUT_ROOT = PAPER1 / "reference_gap_detector"
DEFAULT_OUTPUT_DIR = PAPER1 / "source_tables"


def load_detector(detector_dir: Path, logger: ProgressLogger) -> pd.DataFrame | None:
    summary_path = detector_dir / "reference_gap_summary.csv"
    manifest_path = detector_dir / "reference_gap_manifest.json"
    threshold_path = detector_dir / "reference_gap_thresholds.csv"
    if not summary_path.exists():
        logger.log(f"Skipping {detector_dir}: missing reference_gap_summary.csv")
        return None
    logger.log(f"Loading {summary_path}")
    frame = pd.read_csv(summary_path)
    frame.insert(0, "detector", detector_dir.name)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        frame.insert(1, "target_gap_precision", manifest.get("target_gap_precision"))
        frame.insert(2, "feature_count", len(manifest.get("features", [])))
        frame.insert(3, "claim_boundary", manifest.get("claim_boundary", ""))
    else:
        frame.insert(1, "target_gap_precision", pd.NA)
        frame.insert(2, "feature_count", pd.NA)
        frame.insert(3, "claim_boundary", "")
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)[["rank", "threshold", "calibration_precision", "calibration_recall"]]
        thresholds = thresholds.rename(
            columns={
                "threshold": "rank_threshold",
                "calibration_precision": "threshold_calibration_precision",
                "calibration_recall": "threshold_calibration_recall",
            }
        )
        frame = frame.merge(thresholds, on="rank", how="left")
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
    for detector_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        frame = load_detector(detector_dir, logger)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No detector summaries found under {args.input_root}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "reference_gap_detector_summary.csv"
    manifest_path = args.output_dir / "reference_gap_detector_summary_manifest.json"
    logger.log(f"Writing {summary_path}")
    out.to_csv(summary_path, index=False)
    manifest = {
        "generated_by": "scripts/edna/build_paper1_reference_gap_detector_summary.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "summary": str(summary_path),
        "detectors": sorted(out["detector"].dropna().unique().tolist()),
        "claim_boundary": (
            "Reference-gap detector rows are diagnostic model-development "
            "outputs. The no-counts target-0.99 detector is the more honest "
            "per-query evidence run; count-feature runs can encode synthetic "
            "strict-pack identity and should not be used as a headline result."
        ),
    }
    logger.log(f"Writing {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"summary": str(summary_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
