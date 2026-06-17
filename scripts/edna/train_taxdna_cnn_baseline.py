#!/usr/bin/env python3
"""Train a TAXDNA-style two-convolution CNN baseline on our 12S zero-shot inputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phylo_zero_shot_common import (  # noqa: E402
    CHAR_VOCAB,
    SequenceEmbeddingDataset,
    dry_run_summary,
    extract_embeddings,
    learn_tree_embeddings,
    load_tree_embedding_npz,
    load_zero_shot_inputs,
    ordered_candidate_labels,
    ranked_predictions,
    read_tree,
    save_query_embedding_npz,
    reference_dataframe,
    save_tree_embedding_npz,
    split_reference_dataframe,
    write_prediction_csv,
)


class TaxDnaCnn(nn.Module):
    """Public TAXDNA-style CNN: token embedding, two Conv1d layers, flatten, MLP."""

    def __init__(self, target_embed_dim: int, max_seq_len: int, token_emb_dim: int = 16):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(len(CHAR_VOCAB), token_emb_dim)
        self.conv_1 = nn.Conv1d(token_emb_dim, 2 * token_emb_dim, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv1d(2 * token_emb_dim, 4 * token_emb_dim, kernel_size=5, dilation=5, padding=10)
        self.seq_mlp = nn.Sequential(
            nn.Linear(4 * token_emb_dim * max_seq_len, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, target_embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens)
        x = torch.swapaxes(x, 1, 2)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = torch.flatten(x, start_dim=1)
        return self.seq_mlp(x)

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward(tokens), dim=1)


def train_sequence_model(
    model: TaxDnaCnn,
    train_dataset: SequenceEmbeddingDataset,
    val_dataset: SequenceEmbeddingDataset,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    num_workers: int,
) -> Path:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=len(train_dataset) >= batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
    checkpoint = output_dir / "taxdna_cnn_best.pt"
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for tokens, targets in train_loader:
            tokens = tokens.to(device)
            targets = F.normalize(targets.to(device), dim=1)
            pred = model.get_embeddings(tokens)
            loss = 1 - F.cosine_similarity(pred, targets, dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for tokens, targets in val_loader:
                tokens = tokens.to(device)
                targets = F.normalize(targets.to(device), dim=1)
                pred = model.get_embeddings(tokens)
                loss = 1 - F.cosine_similarity(pred, targets, dim=1).mean()
                val_loss += float(loss.item()) * len(tokens)
                val_n += len(tokens)
        avg_train = total_loss / max(1, batches)
        avg_val = val_loss / max(1, val_n)
        scheduler.step(avg_val)
        print(f"  epoch {epoch + 1}/{epochs}: train_cos_dist={avg_train:.4f} val_cos_dist={avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--token-emb-dim", type=int, default=16)
    parser.add_argument("--tree-epochs", type=int, default=300)
    parser.add_argument("--tree-pairs-per-epoch", type=int, default=768)
    parser.add_argument("--tree-negatives", type=int, default=32)
    parser.add_argument("--tree-max-distance-sample", type=int, default=500)
    parser.add_argument("--tree-embedding-npz", type=Path)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--train-only-reference", action="store_true")
    parser.add_argument("--write-query-embeddings", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_zero_shot_inputs(args.input_dir, train_only=args.train_only_reference)

    if args.dry_run:
        summary = dry_run_summary(inputs, args.tree_file, args.max_seq_len, args.train_only_reference)
        summary["model"] = "taxdna_style_cnn"
        summary["token_emb_dim"] = args.token_emb_dim
        summary["flattened_conv_features"] = 4 * args.token_emb_dim * args.max_seq_len
        out = args.output_dir / "dry_run_summary.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {out}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, taxa = read_tree(args.tree_file)
    candidate_labels = ordered_candidate_labels(inputs.candidate_species, set(taxa))
    species_to_idx = {label: idx for idx, label in enumerate(candidate_labels)}
    ref_df = reference_dataframe(inputs.species_sequences, species_to_idx)
    if len(ref_df) < 2:
        raise RuntimeError("Need at least two reference sequences to train.")

    if args.tree_embedding_npz and args.tree_embedding_npz.exists():
        loaded_labels, tree_embeddings, _ = load_tree_embedding_npz(args.tree_embedding_npz)
        if loaded_labels != candidate_labels:
            raise RuntimeError("Loaded tree embedding labels do not match current candidate labels.")
    else:
        tree_embeddings = learn_tree_embeddings(
            tree_file=args.tree_file,
            labels=candidate_labels,
            embed_dim=args.embed_dim,
            epochs=args.tree_epochs,
            pairs_per_epoch=args.tree_pairs_per_epoch,
            negatives_per_anchor=args.tree_negatives,
            sample_size_for_max_distance=args.tree_max_distance_sample,
            device=device,
            seed=args.seed,
        )
        args.tree_embedding_npz = args.output_dir / "tree_embeddings.npz"
        save_tree_embedding_npz(
            args.tree_embedding_npz,
            candidate_labels,
            tree_embeddings,
            {
                "tree_file": str(args.tree_file),
                "embed_dim": args.embed_dim,
                "tree_epochs": args.tree_epochs,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    train_df, val_df, split_manifest = split_reference_dataframe(
        ref_df,
        input_dir=args.input_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    train_dataset = SequenceEmbeddingDataset(train_df, tree_embeddings, args.max_seq_len)
    val_dataset = SequenceEmbeddingDataset(val_df, tree_embeddings, args.max_seq_len)
    model = TaxDnaCnn(
        target_embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        token_emb_dim=args.token_emb_dim,
    )
    checkpoint = train_sequence_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        num_workers=args.num_workers,
    )

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
                "checkpoint": str(checkpoint),
                "tree_embedding_npz": str(args.tree_embedding_npz),
                "model": "taxdna_style_cnn",
                "max_seq_len": args.max_seq_len,
            },
        )

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
        "model": "taxdna_style_cnn",
        "input_dir": str(args.input_dir),
        "tree_file": str(args.tree_file),
        "tree_embedding_npz": str(args.tree_embedding_npz),
        "checkpoint": str(checkpoint),
        "prediction_csv": str(prediction_csv),
        "query_embedding_npz": str(query_embedding_npz) if query_embedding_npz else None,
        "metrics_dir": str(metrics_dir),
        "candidate_count": len(candidate_labels),
        "reference_sequences": len(ref_df),
        "sequence_train_val_split": split_manifest,
        "zero_shot_queries": len(inputs.zero_shot_queries),
        "args": vars(args),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Wrote predictions to {prediction_csv}")


if __name__ == "__main__":
    main()
