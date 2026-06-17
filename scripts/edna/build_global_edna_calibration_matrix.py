#!/usr/bin/env python3
"""Build no-call calibration curves for the Global_eDNA method matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from progress_logging import ProgressLogger, default_log_path
from summarize_global_edna_benchmarks import DEFAULT_METHODS


ROOT = Path(__file__).resolve().parents[2]


def safe_name(method: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/edna/real_edna_queries/global_tropical_multisource_teleo"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/edna/global_tropical_validation/calibration"))
    parser.add_argument("--methods-json", type=Path)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = DEFAULT_METHODS
    if args.methods_json:
        logger.log(f"Loading methods from {args.methods_json}")
        methods = json.loads(args.methods_json.read_text())
    logger.log(f"Building calibration matrix for {len(methods)} methods")

    outputs = []
    skipped = []
    for method in methods:
        pred_path = Path(method["predictions"])
        if not pred_path.exists():
            logger.log(f"Skipping {method['method']}: missing predictions {pred_path}")
            skipped.append({"method": method["method"], "predictions": str(pred_path), "reason": "missing_predictions"})
            continue
        out_dir = args.output_dir / safe_name(method["method"])
        logger.log(f"Running calibration curves for {method['method']} -> {out_dir}")
        subprocess.run(
            [
                sys.executable,
                "scripts/edna/build_calibration_curves.py",
                "--predictions",
                str(pred_path),
                "--output-dir",
                str(out_dir),
                "--mode",
                "zero_shot_candidate",
                "--input-dir",
                str(args.input_dir),
                "--log-file",
                str(out_dir / "calibration.log"),
            ],
            check=True,
        )
        outputs.append(
            {
                "method": method["method"],
                "predictions": str(pred_path),
                "curve_csv": str(out_dir / "calibration_curve.csv"),
                "scored_rows_csv": str(out_dir / "calibration_scored_rows.csv"),
            }
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "outputs": outputs,
        "skipped": skipped,
    }
    manifest_path = args.output_dir / "global_edna_calibration_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.log(f"Writing manifest to {manifest_path}")
    logger.done(Path(__file__).name)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
