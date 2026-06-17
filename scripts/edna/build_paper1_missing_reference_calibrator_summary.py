#!/usr/bin/env python3
"""Collect Paper 1 missing-reference-aware calibrator summaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from progress_logging import ProgressLogger


ROOT = Path(__file__).resolve().parents[2]
PAPER1 = ROOT / "results" / "paper1_phylo_calibrated_assignment"
DEFAULT_INPUT_ROOT = PAPER1 / "dl_evidence_rank_backoff"
DEFAULT_OUTPUT_DIR = PAPER1 / "source_tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(args.log_file)
    script = Path(__file__).name
    logger.start(script)
    rows: list[pd.DataFrame] = []
    for path in sorted(args.input_root.glob("*/missing_ref_aware_summary.csv")):
        run = path.parent.name
        logger.log(f"Loading {path}")
        frame = pd.read_csv(path)
        frame.insert(0, "calibrator", run)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No missing_ref_aware_summary.csv files found under {args.input_root}")
    summary = pd.concat(rows, ignore_index=True, sort=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "missing_reference_aware_calibrator_summary.csv"
    manifest = args.output_dir / "missing_reference_aware_calibrator_summary_manifest.json"
    logger.log(f"Writing {output}")
    summary.to_csv(output, index=False)
    payload = {
        "generated_by": str(Path(__file__)),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "output": str(output),
        "rows": int(len(summary)),
        "calibrators": sorted(summary["calibrator"].dropna().unique().tolist()),
        "claim_boundary": (
            "Model-development summary for missing-reference-aware DL rank/no-call "
            "calibrators. These rows do not replace production-v1 unless explicitly "
            "promoted after strict transfer review."
        ),
    }
    logger.log(f"Writing {manifest}")
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    logger.done(script)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
