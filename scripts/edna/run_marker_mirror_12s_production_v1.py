#!/usr/bin/env python3
"""Run the MarkerMirror 12S production-v1 order/no-call chain.

This is the orchestration wrapper for the current conservative MarkerMirror
handoff:

    12S FASTA/CSV
      -> MarkerMirror 12S->16S candidates
      -> same-marker BLASTN candidates
      -> same-marker VSEARCH candidates
      -> production-available source feature table
      -> stable all-source order/no-call policy

The final policy is order-only. It emits no species/genus/family calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_marker_mirror_blast_vsearch_calibration_repair import source_features  # noqa: E402
from progress_logging import ProgressLogger, default_log_path  # noqa: E402
from scripts.edna.run_marker_mirror_candidate_generator import load_queries  # noqa: E402


DEFAULT_MARKER_MIRROR_CHECKPOINT = Path(
    "results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/"
    "nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best_seed1903/"
    "marker_mirror_shared_projection_head.pt"
)
DEFAULT_TARGET_CACHE = Path(
    "results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/cache/"
    "marker_mirror_16s_nt_v2_50m_fullref_embeddings.npz"
)
DEFAULT_THRESHOLDS = Path(
    "results/paper1_phylo_calibrated_assignment/source_tables/"
    "marker_mirror_blast_vsearch_calibration_repair_thresholds.csv"
)
DEFAULT_HIGH_COVERAGE_THRESHOLDS = Path(
    "results/paper1_phylo_calibrated_assignment/source_tables/"
    "marker_mirror_high_coverage_order_repair_locked_thresholds.csv"
)
DEFAULT_REFERENCE_DIR = Path("data/edna/stalder_inputs/multisource")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="12S FASTA or CSV input.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/inference_run"),
    )
    parser.add_argument("--query-id-column")
    parser.add_argument("--sequence-column")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--marker-mirror-checkpoint", type=Path, default=DEFAULT_MARKER_MIRROR_CHECKPOINT)
    parser.add_argument("--target-embedding-cache", type=Path, default=DEFAULT_TARGET_CACHE)
    parser.add_argument("--same-marker-reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--high-coverage-thresholds", type=Path, default=DEFAULT_HIGH_COVERAGE_THRESHOLDS)
    parser.add_argument(
        "--decision-mode",
        choices=("stable_order", "high_coverage_order"),
        default="stable_order",
        help=(
            "Final order/no-call mode. stable_order is the conservative default; "
            "high_coverage_order is the Exp 117 BLASTN/VSEARCH top-10 diagnostic mode."
        ),
    )
    parser.add_argument("--blastn-bin", default="blastn")
    parser.add_argument("--makeblastdb-bin", default="makeblastdb")
    parser.add_argument("--vsearch-bin", default="vsearch")
    parser.add_argument("--dry-run", action="store_true", help="Write query/dependency/plan files but do not run tools.")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dependency_report(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    def add(name: str, kind: str, value: str | Path, required: bool = True) -> None:
        resolved = ""
        exists = False
        if kind == "binary":
            resolved = shutil.which(str(value)) or ""
            exists = bool(resolved)
        else:
            path = Path(value)
            resolved = str(path)
            exists = path.exists()
        rows.append(
            {
                "name": name,
                "kind": kind,
                "requested": str(value),
                "resolved": resolved,
                "required": bool(required),
                "available": bool(exists),
            }
        )
        if required and not exists:
            missing.append(name)

    add("python", "binary", args.python)
    add("blastn", "binary", args.blastn_bin)
    add("makeblastdb", "binary", args.makeblastdb_bin)
    add("vsearch", "binary", args.vsearch_bin)
    add("input", "path", args.input)
    add("same_marker_reference_dir", "path", args.same_marker_reference_dir)
    add("marker_mirror_checkpoint", "path", args.marker_mirror_checkpoint)
    add("target_embedding_cache", "path", args.target_embedding_cache, required=False)
    add("thresholds", "path", args.thresholds, required=args.decision_mode == "stable_order")
    add(
        "high_coverage_thresholds",
        "path",
        args.high_coverage_thresholds,
        required=args.decision_mode == "high_coverage_order",
    )
    return pd.DataFrame(rows), missing


def write_query_table(args: argparse.Namespace, output_dir: Path, logger: ProgressLogger) -> Path:
    query_args = argparse.Namespace(
        query_id_column=args.query_id_column,
        sequence_column=args.sequence_column,
        limit=args.limit,
    )
    queries = load_queries(args.input, query_args)
    out = pd.DataFrame(
        {
            "processid": queries["query_id"].astype(str),
            "source": "user_input",
            "tree_label": queries.get("tree_label", "").map(clean) if "tree_label" in queries.columns else "",
            "species_name": queries.get("species", "").map(clean) if "species" in queries.columns else "",
            "genus_name": queries.get("genus", "").map(clean) if "genus" in queries.columns else "",
            "family_name": queries.get("family", "").map(clean) if "family" in queries.columns else "",
            "order_name": queries.get("order", "").map(clean) if "order" in queries.columns else "",
            "nucleotides": queries["nucleotides"].astype(str),
        }
    )
    path = output_dir / "input_queries" / "zero_shot_queries.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.log(f"Wrote normalized 12S query table {rel(path)} rows={len(out)}")
    return path


def run_step(name: str, command: list[str], logger: ProgressLogger) -> float:
    logger.log(f"START step={name}")
    logger.log(" ".join(command))
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start
    logger.log(f"DONE step={name} seconds={elapsed:.3f}")
    return elapsed


def marker_mirror_policy_candidates(path: Path, output_dir: Path, logger: ProgressLogger) -> Path:
    frame = pd.read_csv(path)
    frame["candidate_source"] = "marker_mirror_12s_to_16s"
    output = output_dir / "marker_mirror_candidates_for_policy.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    logger.log(f"Wrote MarkerMirror policy candidates {rel(output)} rows={len(frame)}")
    return output


def load_production_queries(query_table: Path) -> pd.DataFrame:
    frame = pd.read_csv(query_table)
    return pd.DataFrame(
        {
            "query_id": frame["processid"].astype(str),
            "source": frame["source"].astype(str) if "source" in frame.columns else "user_input",
            "query_tree_label": frame["tree_label"].map(clean) if "tree_label" in frame.columns else "",
            "query_species": frame["species_name"].map(clean) if "species_name" in frame.columns else "",
            "query_genus": frame["genus_name"].map(clean) if "genus_name" in frame.columns else "",
            "query_family": frame["family_name"].map(clean) if "family_name" in frame.columns else "",
            "query_order": frame["order_name"].map(clean) if "order_name" in frame.columns else "",
        }
    )


def build_feature_table(
    query_table: Path,
    marker_mirror_candidates: Path,
    blast_candidates: Path,
    vsearch_candidates: Path,
    output_dir: Path,
    logger: ProgressLogger,
) -> Path:
    logger.log("Building production-available all-source feature table")
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_production_queries(query_table)
    for prefix, path in [
        ("mm", marker_mirror_candidates),
        ("blast", blast_candidates),
        ("vsearch", vsearch_candidates),
    ]:
        frame = pd.read_csv(path)
        features = features.merge(source_features(frame, prefix), on="query_id", how="left")
        logger.log(f"Merged {prefix} features from {rel(path)} rows={len(frame)}")
    path = output_dir / "marker_mirror_12s_production_features.csv"
    features.to_csv(path, index=False)
    logger.log(f"Wrote production feature table {rel(path)} rows={len(features)}")
    return path


def pct(num: float, denom: float) -> float:
    return 100.0 * float(num) / float(denom) if denom else math.nan


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def high_coverage_threshold(thresholds_path: Path) -> tuple[float, dict[str, Any]]:
    thresholds = pd.read_csv(thresholds_path)
    subset = thresholds[
        (thresholds["strategy"] == "nested_global_max_wilson95")
        & (thresholds["policy"] == "blast_vsearch_agree_top10_mode")
        & (thresholds["rank"] == "order")
        & (thresholds["source"] == "__global__")
    ].copy()
    if subset.empty:
        raise ValueError(
            "No Exp 117 nested_global_max_wilson95 / "
            "blast_vsearch_agree_top10_mode / order threshold found."
        )
    threshold = pd.to_numeric(subset["threshold"], errors="coerce").dropna()
    if threshold.empty:
        raise ValueError("Exp 117 high-coverage threshold is not finite.")
    row = subset.iloc[0].to_dict()
    return float(threshold.iloc[0]), row


def high_coverage_reason(row: pd.Series, agreement: bool, confidence: float, threshold: float) -> str:
    blast_order = clean(row.get("blast_top10_order_mode"))
    vsearch_order = clean(row.get("vsearch_top10_order_mode"))
    if agreement and confidence >= threshold:
        return "blast_vsearch_top10_order_agreement_high_coverage"
    if agreement:
        return "blast_vsearch_top10_order_agreement_below_high_coverage_threshold"
    if not blast_order or not vsearch_order:
        return "missing_blast_or_vsearch_top10_order"
    return "blast_vsearch_top10_order_conflict"


def build_high_coverage_order_policy(
    features_path: Path,
    thresholds_path: Path,
    output_dir: Path,
    logger: ProgressLogger,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading high-coverage features: {rel(features_path)}")
    features = pd.read_csv(features_path)
    threshold, threshold_row = high_coverage_threshold(thresholds_path)
    logger.log(f"Using high-coverage order threshold={threshold:.6f}")

    rows: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        blast_order = clean(row.get("blast_top10_order_mode"))
        vsearch_order = clean(row.get("vsearch_top10_order_mode"))
        truth_order = clean(row.get("query_order"))
        blast_confidence = numeric(row.get("blast_top1_score")) * numeric(row.get("blast_top10_order_mode_fraction"))
        vsearch_confidence = numeric(row.get("vsearch_top1_score")) * numeric(
            row.get("vsearch_top10_order_mode_fraction")
        )
        confidence = min(blast_confidence, vsearch_confidence)
        if not math.isfinite(confidence):
            confidence = math.nan
        agreement = bool(blast_order and blast_order == vsearch_order)
        assigned = bool(agreement and math.isfinite(confidence) and confidence >= threshold)
        assigned_taxon = blast_order if assigned else ""
        known_truth = bool(truth_order)
        rows.append(
            {
                "query_id": clean(row.get("query_id")),
                "source": clean(row.get("source")),
                "query_species": clean(row.get("query_species")),
                "query_genus": clean(row.get("query_genus")),
                "query_family": clean(row.get("query_family")),
                "query_order": truth_order,
                "policy": "blast_vsearch_agree_top10_mode_nested_global_max_wilson95",
                "assigned_rank": "order" if assigned else "no_call",
                "assigned_taxon": assigned_taxon,
                "confidence": confidence,
                "correct": bool(assigned and assigned_taxon == truth_order) if known_truth else math.nan,
                "false_species_call": False,
                "reason_code": high_coverage_reason(row, agreement, confidence, threshold),
                "locked_threshold": threshold,
                "mm_top1_order": clean(row.get("mm_top1_order_mode")),
                "mm_top1_score": row.get("mm_top1_score"),
                "blast_top1_order": clean(row.get("blast_top1_order_mode")),
                "blast_top1_score": row.get("blast_top1_score"),
                "blast_top10_order": blast_order,
                "blast_top10_order_fraction": row.get("blast_top10_order_mode_fraction"),
                "vsearch_top1_order": clean(row.get("vsearch_top1_order_mode")),
                "vsearch_top1_score": row.get("vsearch_top1_score"),
                "vsearch_top10_order": vsearch_order,
                "vsearch_top10_order_fraction": row.get("vsearch_top10_order_mode_fraction"),
            }
        )
    assignments = pd.DataFrame(rows)
    production = pd.DataFrame(
        {
            "query_id": assignments["query_id"],
            "source": assignments["source"],
            "decision_mode": "marker_mirror_blastn_vsearch_high_coverage_order_v1",
            "policy": assignments["policy"],
            "assigned_rank": assignments["assigned_rank"],
            "assigned_label": assignments["assigned_taxon"],
            "confidence": assignments["confidence"],
            "assignment_reason": assignments["reason_code"],
            "locked_threshold": assignments["locked_threshold"],
            "mm_top1_order": assignments["mm_top1_order"],
            "mm_top1_score": assignments["mm_top1_score"],
            "blast_top1_order": assignments["blast_top1_order"],
            "blast_top1_score": assignments["blast_top1_score"],
            "blast_top10_order": assignments["blast_top10_order"],
            "blast_top10_order_fraction": assignments["blast_top10_order_fraction"],
            "vsearch_top1_order": assignments["vsearch_top1_order"],
            "vsearch_top1_score": assignments["vsearch_top1_score"],
            "vsearch_top10_order": assignments["vsearch_top10_order"],
            "vsearch_top10_order_fraction": assignments["vsearch_top10_order_fraction"],
        }
    )
    assigned = assignments[assignments["assigned_rank"] != "no_call"]
    assigned_known = assigned[assigned["correct"].notna()]
    n_correct = int(assigned_known["correct"].astype(bool).sum()) if len(assigned_known) else math.nan
    summary = pd.DataFrame(
        [
            {
                "policy": "blast_vsearch_agree_top10_mode_nested_global_max_wilson95",
                "n_queries": int(len(assignments)),
                "n_assigned": int(len(assigned)),
                "coverage_pct": pct(len(assigned), len(assignments)),
                "assigned_precision_pct": pct(n_correct, len(assigned_known)) if len(assigned_known) else math.nan,
                "false_species_call_rate_pct": 0.0,
                "n_no_call": int(len(assignments) - len(assigned)),
                "n_correct": n_correct,
                "n_incorrect": int(len(assigned_known) - n_correct) if len(assigned_known) else math.nan,
                "n_assigned_known_truth": int(len(assigned_known)),
                "locked_threshold": threshold,
            }
        ]
    )
    reason_counts = (
        assignments.groupby(["policy", "reason_code"], dropna=False)
        .size()
        .reset_index(name="n_queries")
        .sort_values(["policy", "n_queries"], ascending=[True, False])
    )

    assignments_path = output_dir / "marker_mirror_high_coverage_order_policy_assignments.csv"
    production_path = output_dir / "marker_mirror_high_coverage_order_policy_production_assignments.csv"
    summary_path = output_dir / "marker_mirror_high_coverage_order_policy_summary.csv"
    reasons_path = output_dir / "marker_mirror_high_coverage_order_policy_reason_counts.csv"
    manifest_path = output_dir / "marker_mirror_high_coverage_order_policy_manifest.json"
    assignments.to_csv(assignments_path, index=False)
    production.to_csv(production_path, index=False)
    summary.to_csv(summary_path, index=False)
    reason_counts.to_csv(reasons_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/edna/run_marker_mirror_12s_production_v1.py",
        "decision_mode": "high_coverage_order",
        "inputs": {
            "features": rel(features_path),
            "thresholds": rel(thresholds_path),
        },
        "threshold_row": threshold_row,
        "outputs": {
            "assignments": rel(assignments_path),
            "production_assignments": rel(production_path),
            "summary": rel(summary_path),
            "reason_counts": rel(reasons_path),
        },
        "claim_boundary": (
            "High-coverage order mode emits order/no-call only from BLASTN/VSEARCH "
            "top-10 order agreement. It is a research diagnostic mode, not species "
            "identification or field-eDNA validation."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"High-coverage summary:\n{summary.to_string(index=False)}")
    return production_path


def planned_commands(args: argparse.Namespace, query_table: Path) -> dict[str, list[str]]:
    marker_mirror_dir = args.output_dir / "marker_mirror"
    blast_dir = args.output_dir / "blast"
    vsearch_dir = args.output_dir / "vsearch"
    policy_dir = args.output_dir / "stable_order_policy"
    high_policy_dir = args.output_dir / "high_coverage_order_policy"
    marker_mirror_command = [
        args.python,
        "scripts/edna/run_marker_mirror_candidate_generator.py",
        "--input",
        str(args.input),
        "--output-dir",
        str(marker_mirror_dir),
        "--query-marker",
        "12S",
        "--target-marker",
        "16S",
        "--checkpoint",
        str(args.marker_mirror_checkpoint),
        "--target-embedding-cache",
        str(args.target_embedding_cache),
        "--top-k",
        str(args.top_k),
        "--device",
        args.device,
    ]
    if args.query_id_column:
        marker_mirror_command.extend(["--query-id-column", args.query_id_column])
    if args.sequence_column:
        marker_mirror_command.extend(["--sequence-column", args.sequence_column])
    if args.limit:
        marker_mirror_command.extend(["--limit", str(args.limit)])

    return {
        "marker_mirror": marker_mirror_command,
        "blast": [
            args.python,
            "scripts/edna/build_marker_mirror_same_marker_blast_candidates.py",
            "--query-table",
            str(query_table),
            "--same-marker-reference-dir",
            str(args.same_marker_reference_dir),
            "--output-dir",
            str(blast_dir),
            "--top-k",
            str(args.top_k),
            "--threads",
            str(args.threads),
            "--blastn-bin",
            args.blastn_bin,
            "--makeblastdb-bin",
            args.makeblastdb_bin,
        ],
        "vsearch": [
            args.python,
            "scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py",
            "--query-table",
            str(query_table),
            "--same-marker-reference-dir",
            str(args.same_marker_reference_dir),
            "--output-dir",
            str(vsearch_dir),
            "--top-k",
            str(args.top_k),
            "--threads",
            str(args.threads),
            "--vsearch-bin",
            args.vsearch_bin,
        ],
        "stable_policy": [
            args.python,
            "scripts/edna/build_marker_mirror_stable_order_policy.py",
            "--features",
            str(args.output_dir / "features" / "marker_mirror_12s_production_features.csv"),
            "--thresholds",
            str(args.thresholds),
            "--output-dir",
            str(policy_dir),
        ],
        "high_coverage_policy": [
            "internal",
            "build_high_coverage_order_policy",
            "--features",
            str(args.output_dir / "features" / "marker_mirror_12s_production_features.csv"),
            "--thresholds",
            str(args.high_coverage_thresholds),
            "--output-dir",
            str(high_policy_dir),
        ],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    query_table = write_query_table(args, args.output_dir, logger)
    dependencies, missing = dependency_report(args)
    dependency_path = args.output_dir / "marker_mirror_12s_production_dependency_report.csv"
    dependencies.to_csv(dependency_path, index=False)
    logger.log(f"Wrote dependency report {rel(dependency_path)}")

    commands = planned_commands(args, query_table)
    plan_path = args.output_dir / "marker_mirror_12s_production_plan.json"
    plan_path.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote run plan {rel(plan_path)}")

    timings: dict[str, float] = {}
    if args.dry_run or missing:
        status = "dry_run" if args.dry_run else "blocked_missing_dependencies"
        todo_rows = [
            {
                "status": status,
                "decision_mode": args.decision_mode,
                "missing_dependencies": ",".join(missing),
                "next_action": (
                    "Install missing local tools or run this wrapper on Vast; "
                    "all-source order/no-call requires MarkerMirror, BLASTN, and VSEARCH outputs."
                ),
            }
        ]
        todo_path = args.output_dir / "marker_mirror_12s_production_next_actions.csv"
        csv_write(todo_path, todo_rows)
        logger.log(f"Stopping before execution status={status} missing={missing}")
    else:
        timings["marker_mirror_seconds"] = run_step("marker_mirror", commands["marker_mirror"], logger)
        mm_candidates = marker_mirror_policy_candidates(
            args.output_dir / "marker_mirror" / "marker_mirror_candidate_generator_candidates.csv",
            args.output_dir / "features",
            logger,
        )
        timings["blast_seconds"] = run_step("blast", commands["blast"], logger)
        timings["vsearch_seconds"] = run_step("vsearch", commands["vsearch"], logger)
        feature_path = build_feature_table(
            query_table=query_table,
            marker_mirror_candidates=mm_candidates,
            blast_candidates=args.output_dir / "blast" / "marker_mirror_same_marker_blast_candidates_top50.csv.gz",
            vsearch_candidates=args.output_dir / "vsearch" / "marker_mirror_same_marker_vsearch_candidates_top50.csv.gz",
            output_dir=args.output_dir / "features",
            logger=logger,
        )
        commands["stable_policy"][3] = str(feature_path)
        commands["high_coverage_policy"][3] = str(feature_path)
        if args.decision_mode == "stable_order":
            timings["stable_policy_seconds"] = run_step("stable_policy", commands["stable_policy"], logger)
            production_src = (
                args.output_dir
                / "stable_order_policy"
                / "marker_mirror_stable_order_policy_production_assignments.csv"
            )
        else:
            start = time.perf_counter()
            production_src = build_high_coverage_order_policy(
                features_path=feature_path,
                thresholds_path=args.high_coverage_thresholds,
                output_dir=args.output_dir / "high_coverage_order_policy",
                logger=logger,
            )
            timings["high_coverage_policy_seconds"] = time.perf_counter() - start
        production_dst = args.output_dir / "marker_mirror_12s_production_assignments.csv"
        shutil.copy2(production_src, production_dst)
        logger.log(f"Wrote final production assignments {rel(production_dst)}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/edna/run_marker_mirror_12s_production_v1.py",
        "input": str(args.input),
        "query_table": rel(query_table),
        "output_dir": rel(args.output_dir),
        "dry_run": bool(args.dry_run),
        "missing_dependencies": missing,
        "dependency_report": rel(dependency_path),
        "plan": rel(plan_path),
        "decision_mode": args.decision_mode,
        "timings": timings,
        "claim_boundary": (
            "MarkerMirror 12S production-v1 emits order/no-call assignments only. "
            "stable_order is the conservative default; high_coverage_order is an "
            "explicit research diagnostic mode. Neither mode is a species identifier."
        ),
    }
    manifest_path = args.output_dir / "marker_mirror_12s_production_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.log(f"Wrote manifest {rel(manifest_path)}")
    logger.done(Path(__file__).name)


if __name__ == "__main__":
    main()
