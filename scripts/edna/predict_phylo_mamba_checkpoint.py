#!/usr/bin/env python3
"""Run a trained MarineMamba DNA-to-tree checkpoint on open-candidate queries."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phylo_zero_shot_common import (  # noqa: E402
    extract_embeddings,
    load_tree_embedding_npz,
    save_query_embedding_npz,
    load_zero_shot_inputs,
    ordered_candidate_labels,
    ranked_predictions,
    read_tree,
    write_prediction_csv,
)
from train_12s_phylo_mamba import PhyloMamba, set_global_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tree-embedding-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--pooling", choices=["legacy_mean", "masked_mean", "last_token"], default="masked_mean")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--write-query-embeddings", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inputs = load_zero_shot_inputs(args.input_dir)
    _, taxa = read_tree(args.tree_file)
    candidate_labels = ordered_candidate_labels(inputs.candidate_species, set(taxa))
    loaded_labels, tree_embeddings, tree_metadata = load_tree_embedding_npz(args.tree_embedding_npz)
    if loaded_labels != candidate_labels:
        raise RuntimeError("Tree embedding labels do not match current input candidate labels.")

    model = PhyloMamba(embed_dim=tree_embeddings.shape[1], d_model=args.d_model, pooling=args.pooling)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    query_embeddings = extract_embeddings(
        model,
        inputs.zero_shot_queries,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    ranked_labels, ranked_scores = ranked_predictions(
        query_embeddings,
        candidate_labels,
        tree_embeddings,
        top_k=args.top_k,
    )
    prediction_csv = args.output_dir / "zero_shot_candidate_predictions.csv"
    write_prediction_csv(prediction_csv, inputs.zero_shot_queries, ranked_labels, ranked_scores)
    query_embedding_npz = None
    if args.write_query_embeddings:
        query_embedding_npz = args.output_dir / "query_embeddings.npz"
        save_query_embedding_npz(
            query_embedding_npz,
            inputs.zero_shot_queries,
            query_embeddings,
            {
                "input_dir": str(args.input_dir),
                "checkpoint": str(args.checkpoint),
                "tree_embedding_npz": str(args.tree_embedding_npz),
                "tree_embedding_metadata": tree_metadata,
                "model": "phylo_mamba",
                "max_seq_len": args.max_seq_len,
            },
        )

    metrics_dir = None
    if not args.skip_eval and "tree_label" in inputs.zero_shot_queries.columns:
        metrics_dir = args.output_dir / "zero_shot_metrics"
        subprocess.run(
            [
                sys.executable,
                "scripts/edna/eval_zero_shot_candidate_predictions.py",
                "--input-dir",
                str(args.input_dir),
                "--predictions",
                str(prediction_csv),
                "--output-dir",
                str(metrics_dir),
            ],
            check=True,
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "tree_file": str(args.tree_file),
        "checkpoint": str(args.checkpoint),
        "tree_embedding_npz": str(args.tree_embedding_npz),
        "tree_embedding_metadata": tree_metadata,
        "output_dir": str(args.output_dir),
        "prediction_csv": str(prediction_csv),
        "query_embedding_npz": str(query_embedding_npz) if query_embedding_npz else None,
        "metrics_dir": str(metrics_dir) if metrics_dir else None,
        "candidate_count": len(candidate_labels),
        "query_count": int(len(inputs.zero_shot_queries)),
        "device": device,
        "args": vars(args),
    }
    (args.output_dir / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"Wrote {prediction_csv}")


if __name__ == "__main__":
    main()
