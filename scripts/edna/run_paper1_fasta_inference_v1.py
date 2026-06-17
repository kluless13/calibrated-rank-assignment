#!/usr/bin/env python3
"""Run Paper 1 production-v1 inference from arbitrary COI FASTA/CSV input.

This is the user-facing wrapper around the current research pipeline:

1. parse FASTA or CSV specimen sequences;
2. build a temporary Stalder-style input pack from the production reference;
3. export query embeddings with the locked CNN seed1206 checkpoint;
4. run vector retrieval plus train-reference p-distance reranking;
5. apply locked rank/no-call thresholds;
6. write specimen-facing calls plus an auditable manifest.

If the input does not include known labels, correctness/precision fields are
reported as unavailable rather than treating assignments as wrong.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILES = [
    "candidate_species.csv",
    "species_info.json",
    "species_sequences.json",
    "train_species_sequences.json",
    "val_species.json",
]
RANKS = ("species", "genus", "family", "order")
DEFAULT_CHECKPOINT = Path(
    "results/remote_runs/2026-05-30/rtx_pro_6000/paper1_encoder_benchmarks/"
    "coi_cnn_seed1206/cnn_tree_encoder_best.pt"
)
DEFAULT_RUN_MANIFEST = Path(
    "results/remote_runs/2026-05-30/rtx_pro_6000/paper1_encoder_benchmarks/"
    "coi_cnn_seed1206/run_manifest.json"
)
DEFAULT_TREE_EMBEDDINGS = Path(
    "results/remote_runs/2026-05-30/rtx_pro_6000/"
    "coi_fish_tree_clean_phylo_mamba_hier512_seqval/tree_embeddings.npz"
)
DEFAULT_DL_MODEL_DIR = Path(
    "results/paper1_phylo_calibrated_assignment/"
    "dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance"
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def clean_sequence(value: Any) -> str:
    return "".join(base for base in str(value).upper() if base in "ACGTN")


def normalize_tree_label(value: Any) -> str:
    return clean(value).replace(" ", "_")


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def unique_ids(raw_ids: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for index, raw in enumerate(raw_ids, start=1):
        base = clean(raw) or f"query_{index:06d}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count + 1}")
    return out


def parse_fasta(path: Path, limit: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    rows.append({"processid": current_id, "nucleotides": clean_sequence("".join(chunks))})
                    if limit is not None and len(rows) >= limit:
                        return pd.DataFrame(rows)
                current_id = line[1:].strip().split()[0] or f"query_{len(rows) + 1:06d}"
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None and (limit is None or len(rows) < limit):
            rows.append({"processid": current_id, "nucleotides": clean_sequence("".join(chunks))})
    if not rows:
        raise RuntimeError(f"No FASTA records found in {path}")
    frame = pd.DataFrame(rows)
    frame["processid"] = unique_ids(frame["processid"].astype(str).tolist())
    return frame


def parse_csv_input(
    path: Path,
    id_column: str | None,
    sequence_column: str | None,
    limit: int | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if limit is not None:
        frame = frame.head(limit).copy()
    columns = set(frame.columns)
    seq_col = sequence_column or first_existing(columns, ["nucleotides", "sequence", "seq", "barcode"])
    if seq_col is None or seq_col not in frame.columns:
        raise RuntimeError(
            f"Could not find a sequence column in {path}. "
            "Use --sequence-column or provide one of: nucleotides, sequence, seq, barcode."
        )
    id_col = id_column or first_existing(columns, ["processid", "query_id", "sample_id", "id"])
    out = pd.DataFrame()
    raw_ids = frame[id_col].astype(str).tolist() if id_col and id_col in frame.columns else ["" for _ in range(len(frame))]
    out["processid"] = unique_ids(raw_ids)
    out["nucleotides"] = frame[seq_col].map(clean_sequence)
    for output_col, candidates in [
        ("tree_label", ["tree_label", "species_tree_label"]),
        ("species_name", ["species_name", "species"]),
        ("genus_name", ["genus_name", "genus"]),
        ("family_name", ["family_name", "family"]),
        ("order_name", ["order_name", "order"]),
    ]:
        source = first_existing(columns, candidates)
        out[output_col] = frame[source].map(clean) if source and source in frame.columns else ""
    missing_tree = out["tree_label"].map(clean) == ""
    has_species = out["species_name"].map(clean) != ""
    out.loc[missing_tree & has_species, "tree_label"] = out.loc[missing_tree & has_species, "species_name"].map(
        normalize_tree_label
    )
    return out


def finalize_query_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = frame.copy()
    for column in ["tree_label", "species_name", "genus_name", "family_name", "order_name"]:
        if column not in out.columns:
            out[column] = ""
    out["tree_label"] = out["tree_label"].map(normalize_tree_label)
    out["nucleotides"] = out["nucleotides"].map(clean_sequence)
    out = out[out["nucleotides"].map(bool)].copy()
    if out.empty:
        raise RuntimeError("All input sequences were empty after A/C/G/T/N cleaning.")
    out["split"] = split_name
    return out[
        [
            "processid",
            "tree_label",
            "species_name",
            "genus_name",
            "family_name",
            "order_name",
            "nucleotides",
            "split",
        ]
    ]


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def build_input_pack(
    queries: pd.DataFrame,
    reference_input_dir: Path,
    output_dir: Path,
    input_source: Path,
    split_name: str,
    copy_reference_files: bool,
    logger: ProgressLogger,
) -> Path:
    pack_dir = output_dir / "inference_input_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    for filename in REFERENCE_FILES:
        src = reference_input_dir / filename
        if not src.exists():
            if filename == "val_species.json":
                continue
            raise RuntimeError(f"Reference input pack is missing {src}")
        link_or_copy(src, pack_dir / filename, copy_reference_files)

    query_path = pack_dir / "zero_shot_queries.csv"
    logger.log(f"Writing query table to {query_path}")
    queries.to_csv(query_path, index=False)

    source_manifest_path = reference_input_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text()) if source_manifest_path.exists() else {}
    manifest = {
        "builder": "scripts/edna/run_paper1_fasta_inference_v1.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "query_split": split_name,
        "zero_shot_query_rows": int(len(queries)),
        "input_source": str(input_source),
        "reference_input_dir": str(reference_input_dir),
        "source_reference_manifest": source_manifest,
        "notes": [
            "Temporary inference input pack for the Paper 1 production-v1 pipeline.",
            "Reference files are linked by default; use --copy-reference-files for a materialized pack.",
            "Correctness metrics are only meaningful when input rows include known taxonomy labels.",
        ],
        "outputs": {
            "zero_shot_queries_csv": str(query_path),
            "candidate_species_csv": str(pack_dir / "candidate_species.csv"),
            "species_info_json": str(pack_dir / "species_info.json"),
            "species_sequences_json": str(pack_dir / "species_sequences.json"),
            "train_species_sequences_json": str(pack_dir / "train_species_sequences.json"),
        },
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return pack_dir


def run_step(name: str, command: list[str], logger: ProgressLogger) -> float:
    logger.log(f"START step={name}")
    logger.log(" ".join(command))
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - start
    logger.log(f"DONE step={name} seconds={elapsed:.3f}")
    return elapsed


def read_pipeline_metric(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    sub = frame[frame["metric"].astype(str) == metric]
    if sub.empty:
        return None
    value = pd.to_numeric(sub["value"], errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def first_production_dir(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    for candidate in candidates:
        if (candidate / "production_v1_assignments.csv").exists():
            return candidate
    raise RuntimeError(f"No production_v1_assignments.csv found under {root}")


def safe_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) and math.isfinite(float(numeric)) else None


def build_user_outputs(
    queries: pd.DataFrame,
    pipeline_dir: Path,
    production_root: Path,
    output_dir: Path,
    timings: dict[str, float],
    logger: ProgressLogger,
    decision_mode: str = "production_thresholds",
    dl_decision_dir: Path | None = None,
) -> dict[str, Any]:
    production_dir = first_production_dir(production_root)
    if decision_mode == "production_thresholds":
        decision_dir = production_dir
        decision = pd.read_csv(production_dir / "production_v1_assignments.csv")
        rank_col = "production_assigned_rank"
        label_col = "production_assigned_label"
        reason_col = "production_assignment_reason"
        correct_col = "production_assigned_correct"
    elif decision_mode == "dl_mlp_species_disabled":
        if dl_decision_dir is None:
            raise RuntimeError("dl_decision_dir is required for decision_mode=dl_mlp_species_disabled")
        decision_dir = dl_decision_dir
        decision_path = dl_decision_dir / "coi_dl_evidence_applied_predictions.csv"
        if not decision_path.exists():
            raise RuntimeError(f"Missing DL decision output: {decision_path}")
        decision = pd.read_csv(decision_path)
        rank_col = "dl_assigned_rank"
        label_col = "dl_assigned_label"
        reason_col = "dl_assignment_reason"
        correct_col = "dl_assigned_correct"
    else:
        raise RuntimeError(f"Unsupported decision_mode={decision_mode}")
    predictions = pd.read_csv(pipeline_dir / "pipeline_candidate_predictions.csv")
    query_meta = queries[["processid", "nucleotides"]].copy()
    query_meta["sequence_length"] = query_meta["nucleotides"].astype(str).str.len()

    merged = decision.merge(
        predictions[
            [
                "processid",
                "top_tree_labels",
                "top_scores",
                "top_pdistances",
                "pred_tree_label",
                "pred_pdistance",
            ]
        ],
        on="processid",
        how="left",
        suffixes=("", "_candidate"),
    ).merge(query_meta[["processid", "sequence_length"]], on="processid", how="left")
    truth_cols = ["true_tree_label", "true_genus", "true_family", "true_order"]
    has_truth = merged[truth_cols].apply(lambda row: any(clean(value) for value in row), axis=1)
    user = pd.DataFrame(
        {
            "query_id": merged["processid"],
            "sequence_length": merged["sequence_length"],
            "decision_mode": decision_mode,
            "assigned_rank": merged[rank_col],
            "assigned_label": merged[label_col],
            "assignment_reason": merged[reason_col],
            "pred_species": merged["pred_species"],
            "pred_genus": merged["pred_genus"],
            "pred_family": merged["pred_family"],
            "pred_order": merged["pred_order"],
            "pred_score": merged["pred_score"],
            "pred_pdistance": merged.get("pred_pdistance_candidate", merged.get("pred_pdistance", "")),
            "top_tree_labels": merged["top_tree_labels"],
            "top_scores": merged["top_scores"],
            "top_pdistances": merged["top_pdistances"],
            "has_known_truth": has_truth,
            "known_tree_label": merged["true_tree_label"],
            "known_genus": merged["true_genus"],
            "known_family": merged["true_family"],
            "known_order": merged["true_order"],
            "assigned_correct_if_known": merged[correct_col],
        }
    )

    assignments_path = output_dir / "inference_assignments.csv"
    logger.log(f"Writing specimen-facing assignments to {assignments_path}")
    user.to_csv(assignments_path, index=False)

    n_queries = int(len(user))
    assigned = user[user["assigned_rank"] != "no_call"].copy()
    known_assigned = assigned[assigned["assigned_correct_if_known"].notna()].copy()
    summary: dict[str, Any] = {
        "n_queries": n_queries,
        "n_assigned": int(len(assigned)),
        "coverage": float(len(assigned) / n_queries) if n_queries else None,
        "known_truth_queries": int(has_truth.sum()),
        "known_truth_assigned_count": int(len(known_assigned)),
        "assigned_precision_if_known": (
            float(known_assigned["assigned_correct_if_known"].astype(bool).mean())
            if len(known_assigned)
            else None
        ),
        "embedding_export_seconds": timings["embedding_export_seconds"],
        "pipeline_seconds": timings["pipeline_seconds"],
        "production_packaging_seconds": timings["production_packaging_seconds"],
        "dl_decision_seconds": timings.get("dl_decision_seconds"),
        "total_seconds": timings["total_seconds"],
        "total_ms_per_query": 1000.0 * timings["total_seconds"] / n_queries if n_queries else None,
        "vector_retrieval_seconds": read_pipeline_metric(pipeline_dir / "pipeline_summary.csv", "vector_search_seconds"),
        "rerank_seconds": read_pipeline_metric(pipeline_dir / "pipeline_summary.csv", "rerank_seconds"),
        "decision_mode": decision_mode,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    for rank in list(RANKS) + ["no_call"]:
        summary[f"assigned_{rank}_count"] = int((user["assigned_rank"] == rank).sum())

    summary_path = output_dir / "inference_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return {
        "assignments": str(assignments_path),
        "summary": str(summary_path),
        "summary_values": summary,
        "production_dir": str(production_dir),
        "decision_dir": str(decision_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-fasta", type=Path)
    group.add_argument("--input-csv", type=Path)
    parser.add_argument("--id-column")
    parser.add_argument("--sequence-column")
    parser.add_argument("--split-name", default="inference")
    parser.add_argument("--limit", type=int, help="Optional row/record limit for smoke tests.")
    parser.add_argument(
        "--reference-input-dir",
        type=Path,
        default=Path("data/phylo/fish_tree_clean_phylo_inputs/eval_c"),
        help="Reference pack carrying candidates, taxonomy, and train-reference sequences.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--tree-embedding-npz", type=Path, default=DEFAULT_TREE_EMBEDDINGS)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/pipeline_calibration/pipeline_mode_thresholds.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper1_phylo_calibrated_assignment/production_v1_cli/inference_run"),
    )
    parser.add_argument("--prediction-set", default="cnn_seed1206")
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument(
        "--decision-mode",
        choices=["production_thresholds", "dl_mlp_species_disabled"],
        default="production_thresholds",
        help=(
            "Final rank/no-call decision layer. The default uses locked hand "
            "thresholds; dl_mlp_species_disabled applies the trained evidence "
            "MLP with species calls disabled."
        ),
    )
    parser.add_argument("--dl-model-dir", type=Path, default=DEFAULT_DL_MODEL_DIR)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rerank-top-candidates", type=int, default=25)
    parser.add_argument("--copy-reference-files", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)

    input_source = args.input_fasta or args.input_csv
    assert input_source is not None
    if args.input_fasta:
        logger.log(f"Parsing FASTA input {args.input_fasta}")
        raw_queries = parse_fasta(args.input_fasta, limit=args.limit)
    else:
        logger.log(f"Parsing CSV input {args.input_csv}")
        raw_queries = parse_csv_input(args.input_csv, args.id_column, args.sequence_column, limit=args.limit)
    queries = finalize_query_frame(raw_queries, split_name=args.split_name)
    input_pack = build_input_pack(
        queries=queries,
        reference_input_dir=args.reference_input_dir,
        output_dir=output_dir,
        input_source=input_source,
        split_name=args.split_name,
        copy_reference_files=args.copy_reference_files,
        logger=logger,
    )

    embedding_dir = output_dir / "embedding_export"
    pipeline_dir = output_dir / "pipeline_run"
    production_root = output_dir / "production_v1"
    dl_decision_dir = output_dir / "dl_mlp_species_disabled_decision"

    embedding_seconds = run_step(
        "embedding_export",
        [
            args.python,
            "scripts/edna/train_fish_tree_encoder_benchmark.py",
            "predict",
            "--input-dir",
            str(input_pack),
            "--tree-file",
            "data/phylo/actinopt_12k_treePL.tre",
            "--output-dir",
            str(embedding_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--run-manifest",
            str(args.run_manifest),
            "--tree-embedding-npz",
            str(args.tree_embedding_npz),
            "--write-query-embeddings",
            "--predict-batch-size",
            str(args.predict_batch_size),
            "--num-workers",
            "4",
        ],
        logger,
    )
    pipeline_seconds = run_step(
        "vector_retrieval_pdistance_rerank",
        [
            args.python,
            "scripts/edna/run_paper1_coi_pipeline.py",
            "--query-embeddings",
            str(embedding_dir / "query_embeddings.npz"),
            "--prediction-set",
            args.prediction_set,
            "--target-precision",
            str(args.target_precision),
            "--output-dir",
            str(pipeline_dir),
            "--top-k",
            str(args.top_k),
            "--retrieval-mode",
            "exact",
            "--rerank-mode",
            "p_distance",
            "--rerank-top-candidates",
            str(args.rerank_top_candidates),
            "--assignment-source",
            "reranked",
        ],
        logger,
    )
    production_seconds = run_step(
        "locked_rank_no_call_packaging",
        [
            args.python,
            "scripts/edna/run_paper1_production_v1.py",
            "--target-precision",
            str(args.target_precision),
            "--thresholds",
            str(args.thresholds),
            "--output-root",
            str(production_root),
            "--input-run-dir",
            str(pipeline_dir),
        ],
        logger,
    )
    dl_decision_seconds = 0.0
    active_dl_decision_dir: Path | None = None
    if args.decision_mode == "dl_mlp_species_disabled":
        active_dl_decision_dir = dl_decision_dir
        dl_decision_seconds = run_step(
            "dl_mlp_species_disabled_decision",
            [
                args.python,
                "scripts/edna/apply_paper1_coi_evidence_model.py",
                "--pipeline-run-dir",
                str(pipeline_dir),
                "--model-dir",
                str(args.dl_model_dir),
                "--output-dir",
                str(dl_decision_dir),
                "--target-precision",
                str(args.target_precision),
                "--policy",
                "species_disabled",
                "--batch-size",
                str(max(1024, args.predict_batch_size)),
            ],
            logger,
        )
    total_seconds = embedding_seconds + pipeline_seconds + production_seconds + dl_decision_seconds
    user_outputs = build_user_outputs(
        queries=queries,
        pipeline_dir=pipeline_dir,
        production_root=production_root,
        output_dir=output_dir,
        timings={
            "embedding_export_seconds": embedding_seconds,
            "pipeline_seconds": pipeline_seconds,
            "production_packaging_seconds": production_seconds,
            "dl_decision_seconds": dl_decision_seconds if args.decision_mode == "dl_mlp_species_disabled" else None,
            "total_seconds": total_seconds,
        },
        logger=logger,
        decision_mode=args.decision_mode,
        dl_decision_dir=active_dl_decision_dir,
    )
    manifest = {
        "generated_by": "scripts/edna/run_paper1_fasta_inference_v1.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_source": str(input_source),
        "reference_input_dir": str(args.reference_input_dir),
        "checkpoint": str(args.checkpoint),
        "run_manifest": str(args.run_manifest),
        "tree_embedding_npz": str(args.tree_embedding_npz),
        "prediction_set": args.prediction_set,
        "target_precision": float(args.target_precision),
        "decision_mode": args.decision_mode,
        "dl_model_dir": str(args.dl_model_dir) if args.decision_mode == "dl_mlp_species_disabled" else None,
        "query_count": int(len(queries)),
        "outputs": {
            "input_pack": str(input_pack),
            "embedding_dir": str(embedding_dir),
            "pipeline_dir": str(pipeline_dir),
            "production_root": str(production_root),
            "dl_decision_dir": str(active_dl_decision_dir) if active_dl_decision_dir else None,
            **user_outputs,
        },
        "claim_boundary": (
            "Production-v1 CLI for COI barcode inference. It uses the locked "
            "CNN seed1206 model, vector candidate retrieval, train-reference "
            "p-distance reranking, and an explicit final decision layer. "
            "Assignments for unlabeled specimens have no precision estimate "
            "unless known taxonomy labels are supplied in the input."
        ),
    }
    manifest_path = output_dir / "inference_manifest.json"
    logger.log(f"Writing {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
