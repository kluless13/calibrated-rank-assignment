#!/usr/bin/env python3
"""Aggregate strict hidden-reference DL decision-layer application outputs."""

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
DEFAULT_INPUT_ROOT = (
    ROOT
    / "results"
    / "paper1_phylo_calibrated_assignment"
    / "dl_evidence_rank_backoff"
    / "strict_dl_apply"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "paper1_phylo_calibrated_assignment" / "source_tables"
PACK_RE = re.compile(r"^(?P<split>eval_c|unseen_genera)_hide_(?P<hidden_rank>species|genus|family)_seed(?P<seed>\d+)$")


def load_summary(path: Path) -> dict[str, object] | None:
    match = PACK_RE.match(path.name)
    if not match:
        return None
    summary_path = path / "coi_dl_evidence_applied_summary.csv"
    manifest_path = path / "coi_dl_evidence_applied_manifest.json"
    if not summary_path.exists():
        return None
    frame = pd.read_csv(summary_path)
    if frame.empty:
        return None
    row = frame.iloc[0].to_dict()
    row.update(
        {
            "base_split": match.group("split"),
            "hidden_rank": match.group("hidden_rank"),
            "seed": int(match.group("seed")),
            "source_dir": str(path),
            "manifest": str(manifest_path) if manifest_path.exists() else "",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for path in sorted(args.input_root.iterdir()) if args.input_root.exists() else []:
        if not path.is_dir():
            continue
        row = load_summary(path)
        if row is not None:
            rows.append(row)
            logger.log(f"Loaded {path}")
    if not rows:
        raise RuntimeError(f"No strict DL apply summaries found under {args.input_root}")

    out = pd.DataFrame(rows).sort_values(["base_split", "hidden_rank", "seed"]).reset_index(drop=True)
    summary_out = args.output_dir / "dl_evidence_strict_apply_summary.csv"
    logger.log(f"Writing {summary_out}")
    out.to_csv(summary_out, index=False)

    manifest = {
        "generated_by": "scripts/edna/build_paper1_dl_strict_apply_summary.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "row_count": int(len(out)),
        "outputs": {"summary": str(summary_out)},
        "claim_boundary": (
            "Aggregates the species-disabled COI DL decision layer applied to "
            "strict hidden-reference executable pipeline runs. These are stress "
            "tests, not calibration-training rows."
        ),
    }
    manifest_out = args.output_dir / "dl_evidence_strict_apply_summary_manifest.json"
    logger.log(f"Writing {manifest_out}")
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
