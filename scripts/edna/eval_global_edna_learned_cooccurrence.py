#!/usr/bin/env python3
"""Apply an NPZ-trained learned co-occurrence model to Global_eDNA predictions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phylo_zero_shot_common import load_query_embedding_npz, load_tree_embedding_npz  # noqa: E402
from train_npz_cooccurrence_model import NpzCoOccurrenceModel  # noqa: E402


def nonempty(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def load_model(checkpoint: Path, num_species: int, device: str) -> NpzCoOccurrenceModel:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    key = "context_embedding.weight"
    if key not in state:
        raise RuntimeError(f"Checkpoint missing {key}")
    hidden_dim, checkpoint_species = state[key].shape
    if checkpoint_species != num_species:
        raise RuntimeError(
            f"Checkpoint species dimension {checkpoint_species} does not match tree embeddings {num_species}."
        )
    model = NpzCoOccurrenceModel(num_species=num_species, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def top_predictions(logits: torch.Tensor, labels: list[str], top_k: int) -> tuple[list[list[str]], list[list[float]]]:
    k = min(top_k, len(labels))
    values, indices = torch.topk(logits, k=k, dim=1)
    values_np = values.detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy()
    ranked_labels = [[labels[int(idx)] for idx in row] for row in indices_np]
    ranked_scores = [[float(value) for value in row] for row in values_np]
    return ranked_labels, ranked_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sample-query-map", type=Path, required=True)
    parser.add_argument("--query-embedding-npz", type=Path, required=True)
    parser.add_argument("--tree-embedding-npz", type=Path, required=True)
    parser.add_argument("--cooccurrence-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-temp", type=float, default=0.05)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-samples", type=int, help="Optional cap for smoke tests.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_map = pd.read_csv(args.sample_query_map)
    processids, query_embeddings_np, query_metadata = load_query_embedding_npz(args.query_embedding_npz)
    labels, tree_embeddings_np, tree_metadata = load_tree_embedding_npz(args.tree_embedding_npz)
    if query_embeddings_np.shape[1] != tree_embeddings_np.shape[1]:
        raise RuntimeError(
            f"Query embedding dim {query_embeddings_np.shape[1]} does not match tree embedding dim {tree_embeddings_np.shape[1]}."
        )
    processid_to_idx = {processid: idx for idx, processid in enumerate(processids)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_embeddings = F.normalize(torch.tensor(tree_embeddings_np, dtype=torch.float32, device=device), dim=1)
    query_embeddings = F.normalize(torch.tensor(query_embeddings_np, dtype=torch.float32, device=device), dim=1)
    model = load_model(args.cooccurrence_checkpoint, num_species=len(labels), device=device)

    rows = []
    missing_queries = 0
    sample_count = 0
    source_counts: Counter[str] = Counter()
    for sample_id, sub in sample_map.groupby("sample_id", sort=False, dropna=False):
        if args.max_samples is not None and sample_count >= args.max_samples:
            break
        sample_count += 1
        indices = []
        map_rows = []
        for _, row in sub.iterrows():
            query_id = str(row["query_processid"])
            idx = processid_to_idx.get(query_id)
            if idx is None:
                missing_queries += 1
                continue
            indices.append(idx)
            map_rows.append(row)
        if not indices:
            continue

        emb = query_embeddings[torch.tensor(indices, dtype=torch.long, device=device)]
        sequence_logits = emb @ class_embeddings.T
        class_probabilities = torch.softmax(sequence_logits / args.sequence_temp, dim=1).clamp_min(1e-12)
        with torch.no_grad():
            coocc_logits = model(class_probabilities)
            if len(indices) > 1:
                context = (coocc_logits.sum(dim=0, keepdim=True) - coocc_logits) / (len(indices) - 1)
                source = "learned_cooccurrence"
            else:
                context = torch.zeros_like(coocc_logits)
                source = "single_query_no_context"
            final_logits = torch.log(class_probabilities) + args.context_weight * context
            ranked_labels, ranked_scores = top_predictions(final_logits, labels, args.top_k)

        source_counts[source] += len(map_rows)
        for row, top_labels, top_scores in zip(map_rows, ranked_labels, ranked_scores):
            query_id = str(row["query_processid"])
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "query_processid": query_id,
                    "processid": query_id,
                    "true_tree_label": row.get("true_tree_label"),
                    "true_species_name": row.get("true_species_name"),
                    "top_tree_labels": json.dumps(top_labels),
                    "top_scores": json.dumps([round(score, 8) for score in top_scores]),
                    "pred_tree_label": top_labels[0] if top_labels else None,
                    "pred_score": top_scores[0] if top_scores else None,
                    "cooccurrence_source": source,
                    "sample_query_count": int(len(indices)),
                }
            )

    pred_path = args.output_dir / "learned_cooccurrence_predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    validation_dir = args.output_dir / "global_edna_validation"
    subprocess.run(
        [
            sys.executable,
            "scripts/edna/eval_global_edna_sample_validation.py",
            "--input-dir",
            str(args.input_dir),
            "--predictions",
            str(pred_path),
            "--sample-query-map",
            str(args.sample_query_map),
            "--output-dir",
            str(validation_dir),
        ],
        check=True,
    )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir),
        "sample_query_map": str(args.sample_query_map),
        "query_embedding_npz": str(args.query_embedding_npz),
        "query_embedding_metadata": query_metadata,
        "tree_embedding_npz": str(args.tree_embedding_npz),
        "tree_embedding_metadata": tree_metadata,
        "cooccurrence_checkpoint": str(args.cooccurrence_checkpoint),
        "predictions": str(pred_path),
        "validation_dir": str(validation_dir),
        "sequence_temp": args.sequence_temp,
        "context_weight": args.context_weight,
        "top_k": args.top_k,
        "sample_count_processed": sample_count,
        "prediction_rows": len(rows),
        "missing_query_rows": missing_queries,
        "source_counts": dict(source_counts),
        "device": device,
        "notes": [
            "This applies a learned TAXDNA-style co-occurrence correction in the fixed NPZ tree embedding space.",
            "Rows from single-query samples receive no community context.",
        ],
    }
    (args.output_dir / "learned_cooccurrence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
