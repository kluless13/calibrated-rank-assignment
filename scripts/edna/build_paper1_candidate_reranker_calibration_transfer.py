#!/usr/bin/env python3
"""Build cross-split calibration-transfer diagnostics for candidate rerankers."""
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


def threshold_for_precision(frame: pd.DataFrame, target_precision: float) -> dict[str, float | int | None]:
    scored = frame[["selection_score", "selection_correct"]].dropna().sort_values(
        "selection_score", ascending=False
    )
    n = int(len(scored))
    if n == 0:
        return {"threshold": None, "precision": None, "coverage": 0.0, "assigned": 0, "correct": 0}

    correct = scored["selection_correct"].astype(float).cumsum()
    assigned = pd.Series(range(1, n + 1), index=scored.index, dtype=float)
    precision = correct / assigned
    ok = precision >= target_precision
    if not bool(ok.any()):
        return {"threshold": None, "precision": None, "coverage": 0.0, "assigned": 0, "correct": 0}

    # Last qualifying row is the lowest threshold that keeps the target precision.
    idx = ok[ok].index[-1]
    assigned_n = int(assigned.loc[idx])
    correct_n = int(correct.loc[idx])
    return {
        "threshold": float(scored.loc[idx, "selection_score"]),
        "precision": float(precision.loc[idx]),
        "coverage": assigned_n / n,
        "assigned": assigned_n,
        "correct": correct_n,
    }


def apply_threshold(frame: pd.DataFrame, threshold: float | None) -> dict[str, float | int | None]:
    n = int(len(frame))
    if n == 0 or threshold is None:
        return {"precision": None, "coverage": 0.0, "assigned": 0, "correct": 0, "n": n}
    selected = frame[frame["selection_score"] >= threshold]
    assigned = int(len(selected))
    correct = int(selected["selection_correct"].fillna(0).astype(float).sum())
    precision = correct / assigned if assigned else None
    return {
        "precision": precision,
        "coverage": assigned / n if n else 0.0,
        "assigned": assigned,
        "correct": correct,
        "n": n,
    }


def summarize_run(run_dir: Path, target_precision: float, logger: ProgressLogger) -> list[dict[str, object]]:
    pred_path = run_dir / "candidate_reranker_selected_predictions.csv"
    manifest_path = run_dir / "candidate_reranker_manifest.json"
    if not pred_path.exists():
        logger.log(f"Skipping {run_dir}: missing candidate_reranker_selected_predictions.csv")
        return []

    logger.log(f"Loading {pred_path}")
    frame = pd.read_csv(pred_path)
    required = {"split", "selection_rank", "selection_score", "selection_correct"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        logger.log(f"Skipping {run_dir}: missing columns {missing}")
        return []

    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, object]] = []
    for rank in sorted(frame["selection_rank"].dropna().unique().tolist()):
        rank_frame = frame[frame["selection_rank"] == rank].copy()
        for source_split in sorted(rank_frame["split"].dropna().unique().tolist()):
            source_frame = rank_frame[rank_frame["split"] == source_split]
            source_stats = threshold_for_precision(source_frame, target_precision)
            for target_split in sorted(rank_frame["split"].dropna().unique().tolist()):
                target_frame = rank_frame[rank_frame["split"] == target_split]
                target_stats = apply_threshold(target_frame, source_stats["threshold"])
                rows.append(
                    {
                        "reranker": run_dir.name,
                        "rank": rank,
                        "target_precision": target_precision,
                        "source_split": source_split,
                        "target_split": target_split,
                        "threshold": source_stats["threshold"],
                        "source_precision": source_stats["precision"],
                        "source_coverage": source_stats["coverage"],
                        "source_assigned": source_stats["assigned"],
                        "source_correct": source_stats["correct"],
                        "target_precision_observed": target_stats["precision"],
                        "target_coverage": target_stats["coverage"],
                        "target_assigned": target_stats["assigned"],
                        "target_correct": target_stats["correct"],
                        "target_n": target_stats["n"],
                        "feature_count": len(manifest.get("features", [])) if manifest else pd.NA,
                        "claim_boundary": (
                            "Threshold-transfer diagnostics use saved selected predictions only. "
                            "They audit calibration stability and do not retrain or re-rank candidates."
                        ),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    if not args.input_root.exists():
        raise RuntimeError(f"Missing input root: {args.input_root}")

    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        rows.extend(summarize_run(run_dir, args.target_precision, logger))
    if not rows:
        raise RuntimeError(f"No calibration-transfer rows built under {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    table_path = args.output_dir / "candidate_reranker_calibration_transfer.csv"
    manifest_path = args.output_dir / "candidate_reranker_calibration_transfer_manifest.json"
    logger.log(f"Writing {table_path}")
    out.to_csv(table_path, index=False)
    manifest = {
        "generated_by": "scripts/edna/build_paper1_candidate_reranker_calibration_transfer.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root),
        "target_precision": args.target_precision,
        "table": str(table_path),
        "rerankers": sorted(out["reranker"].dropna().unique().tolist()),
        "claim_boundary": (
            "This table tests whether thresholds fitted on one split transfer to another "
            "using selected prediction rows. It is a calibration audit, not a final "
            "production operating policy."
        ),
    }
    logger.log(f"Writing {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps({"table": str(table_path), "manifest": str(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
