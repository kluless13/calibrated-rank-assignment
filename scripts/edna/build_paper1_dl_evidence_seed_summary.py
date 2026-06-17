#!/usr/bin/env python3
"""Aggregate Paper 1 COI DL evidence model seed-repeat outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "dl_evidence_rank_backoff"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
SEED_RE = re.compile(r"seed(\d+)")


def seed_from_path(path: Path) -> int | None:
    match = SEED_RE.search(path.name)
    return int(match.group(1)) if match else None


def load_table(path: Path, seed: int, model_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(0, "seed", seed)
    frame.insert(1, "model_dir", str(model_dir))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="coi_mlp_seed*_pdistance")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    model_dirs = sorted(path for path in args.model_root.glob(args.pattern) if path.is_dir())
    logger.log(f"Scanning {len(model_dirs)} model directories under {args.model_root}")
    for model_dir in model_dirs:
        seed = seed_from_path(model_dir)
        if seed is None:
            logger.log(f"Skipping {model_dir}: no seed in directory name")
            continue
        summary_path = model_dir / "coi_dl_evidence_rank_backoff_summary.csv"
        bootstrap_path = model_dir / "coi_dl_evidence_rank_backoff_bootstrap.csv"
        if summary_path.exists():
            logger.log(f"Loading {summary_path}")
            summaries.append(load_table(summary_path, seed, model_dir))
        if bootstrap_path.exists():
            logger.log(f"Loading {bootstrap_path}")
            bootstraps.append(load_table(bootstrap_path, seed, model_dir))

    if not summaries:
        raise RuntimeError(f"No summary tables found under {args.model_root}")

    summary = pd.concat(summaries, ignore_index=True, sort=False)
    summary_out = args.output_dir / "dl_evidence_seed_summary.csv"
    logger.log(f"Writing {summary_out}")
    summary.to_csv(summary_out, index=False)

    bootstrap_out = args.output_dir / "dl_evidence_seed_bootstrap_summary.csv"
    if bootstraps:
        bootstrap = pd.concat(bootstraps, ignore_index=True, sort=False)
        logger.log(f"Writing {bootstrap_out}")
        bootstrap.to_csv(bootstrap_out, index=False)
    else:
        bootstrap_out = None

    manifest = {
        "generated_by": "scripts/edna/build_paper1_dl_evidence_seed_summary.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_root": str(args.model_root),
        "pattern": args.pattern,
        "model_dirs": [str(path) for path in model_dirs],
        "outputs": {
            "summary": str(summary_out),
            "bootstrap": str(bootstrap_out) if bootstrap_out else None,
        },
        "claim_boundary": (
            "Aggregates already-trained COI DL evidence model seed-repeat "
            "summaries. It does not retrain or recalibrate models."
        ),
    }
    manifest_out = args.output_dir / "dl_evidence_seed_summary_manifest.json"
    logger.log(f"Writing {manifest_out}")
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
