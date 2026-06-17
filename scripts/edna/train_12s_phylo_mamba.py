#!/usr/bin/env python3
"""Train MarineMamba/BarcodeMamba for Stalder-style 12S DNA-to-tree embeddings."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
    reference_dataframe,
    save_tree_embedding_npz,
    save_query_embedding_npz,
    split_reference_dataframe,
    write_prediction_csv,
)


HIERARCHICAL_LOSS_MODES = {"hierarchical_contrastive", "hierarchical_hybrid"}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_rank_id_arrays(input_dir: Path, candidate_labels: list[str]) -> dict[str, np.ndarray]:
    info_path = input_dir / "species_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Hierarchical loss requires {info_path}")
    species_info = json.loads(info_path.read_text())
    candidate_path = input_dir / "candidate_species.csv"
    candidate_rows = {}
    if candidate_path.exists():
        import pandas as pd

        candidate_rows = pd.read_csv(candidate_path).set_index("tree_label").to_dict(orient="index")

    def clean_rank_value(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return None if not text or text.lower() == "nan" else text

    def rank_value(label: str, rank: str) -> str | None:
        value = clean_rank_value(species_info.get(label, {}).get(rank))
        if value:
            return value
        row = candidate_rows.get(label, {})
        if rank == "genus":
            value = clean_rank_value(row.get("genus_name")) or clean_rank_value(row.get("genus_from_label"))
        elif rank == "family":
            value = clean_rank_value(row.get("family_name"))
        elif rank == "order":
            value = clean_rank_value(row.get("order_name"))
        else:
            value = None
        return value

    arrays: dict[str, np.ndarray] = {}
    for rank in ["genus", "family", "order"]:
        value_to_id: dict[str, int] = {}
        ids: list[int] = []
        for label in candidate_labels:
            value = rank_value(label, rank)
            if not value:
                ids.append(-1)
                continue
            if value not in value_to_id:
                value_to_id[value] = len(value_to_id)
            ids.append(value_to_id[value])
        arrays[rank] = np.array(ids, dtype=np.int64)
    return arrays


def import_barcode_mamba() -> type[nn.Module]:
    repo = Path("BarcodeMamba")
    if not repo.exists():
        raise RuntimeError(
            "BarcodeMamba directory is missing. Clone/install it before running training, "
            "or run this script with --dry-run only."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from utils.barcode_mamba import BarcodeMamba

    return BarcodeMamba


class PhyloMamba(nn.Module):
    def __init__(self, embed_dim: int, d_model: int = 384, pooling: str = "masked_mean"):
        super().__init__()
        if pooling not in {"legacy_mean", "masked_mean", "last_token"}:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        self.pooling = pooling
        BarcodeMamba = import_barcode_mamba()
        self.backbone = BarcodeMamba(
            d_model=d_model,
            n_layer=2,
            d_inner=d_model * 4,
            vocab_size=8,
            resid_dropout=0.0,
            embed_dropout=0.1,
            residual_in_fp32=True,
            pad_vocab_size_multiple=8,
            mamba_ver="mamba2",
            n_classes=8,
            use_head="pretrain",
            layer={"d_model": d_model, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 48},
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, embed_dim),
        )

    def pool_hidden(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooling == "legacy_mean":
            return hidden.mean(dim=1)
        if self.pooling == "last_token":
            return hidden[:, -1]

        mask = tokens.ne(CHAR_VOCAB["[PAD]"]).unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone.get_hidden_states(tokens)
        return self.proj(self.pool_hidden(hidden, tokens))

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward(tokens), dim=1)


def train_sequence_model(
    model: PhyloMamba,
    train_dataset: SequenceEmbeddingDataset,
    val_dataset: SequenceEmbeddingDataset,
    candidate_embeddings: np.ndarray,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_mode: str,
    temperature: float,
    cosine_weight: float,
    contrastive_weight: float,
    rank_id_arrays: dict[str, np.ndarray] | None,
    species_positive_weight: float,
    genus_positive_weight: float,
    family_positive_weight: float,
    order_positive_weight: float,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    checkpoint = output_dir / "phylo_mamba_best.pt"
    best_val = float("inf")
    candidate_tensor = F.normalize(torch.tensor(candidate_embeddings, dtype=torch.float32, device=device), dim=1)
    rank_tensors = {
        rank: torch.tensor(values, dtype=torch.long, device=device)
        for rank, values in (rank_id_arrays or {}).items()
    }

    def hierarchical_targets(target_indices: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(logits)
        rank_weights = [
            ("order", order_positive_weight),
            ("family", family_positive_weight),
            ("genus", genus_positive_weight),
        ]
        for rank, weight in rank_weights:
            rank_ids = rank_tensors.get(rank)
            if rank_ids is None or weight <= 0:
                continue
            target_rank = rank_ids[target_indices].unsqueeze(1)
            valid = target_rank.ge(0) & rank_ids.unsqueeze(0).ge(0)
            match = target_rank.eq(rank_ids.unsqueeze(0)) & valid
            weights = torch.maximum(weights, match.to(logits.dtype) * weight)
        weights.scatter_(1, target_indices.unsqueeze(1), species_positive_weight)
        row_sums = weights.sum(dim=1, keepdim=True)
        missing = row_sums.squeeze(1).le(0)
        if missing.any():
            weights[missing, target_indices[missing]] = 1.0
            row_sums = weights.sum(dim=1, keepdim=True)
        return weights / row_sums.clamp_min(1e-8)

    def compute_loss(
        pred: torch.Tensor,
        targets: torch.Tensor,
        target_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = F.normalize(pred, dim=1)
        targets = F.normalize(targets, dim=1)
        cosine_loss = 1 - F.cosine_similarity(pred, targets, dim=1).mean()
        contrastive_loss = torch.zeros((), device=pred.device)
        if loss_mode in {"contrastive", "hybrid"} | HIERARCHICAL_LOSS_MODES:
            if target_indices is None:
                raise RuntimeError("Contrastive losses require species indices.")
            logits = pred @ candidate_tensor.T / temperature
            if loss_mode in HIERARCHICAL_LOSS_MODES:
                soft_targets = hierarchical_targets(target_indices, logits)
                contrastive_loss = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                contrastive_loss = F.cross_entropy(logits, target_indices)
        if loss_mode == "cosine":
            loss = cosine_loss
        elif loss_mode in {"contrastive", "hierarchical_contrastive"}:
            loss = contrastive_loss
        else:
            loss = cosine_weight * cosine_loss + contrastive_weight * contrastive_loss
        return loss, {
            "cosine": float(cosine_loss.detach().item()),
            "contrastive": float(contrastive_loss.detach().item()),
        }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cosine = 0.0
        total_contrastive = 0.0
        batches = 0
        for batch in train_loader:
            if len(batch) == 3:
                tokens, targets, target_indices = batch
                target_indices = target_indices.to(device)
            else:
                tokens, targets = batch
                target_indices = None
            tokens = tokens.to(device)
            targets = targets.to(device)
            pred = model.forward(tokens)
            loss, loss_parts = compute_loss(pred, targets, target_indices)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_cosine += loss_parts["cosine"]
            total_contrastive += loss_parts["contrastive"]
            batches += 1
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_cosine = 0.0
        val_contrastive = 0.0
        val_batches = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    tokens, targets, target_indices = batch
                    target_indices = target_indices.to(device)
                else:
                    tokens, targets = batch
                    target_indices = None
                tokens = tokens.to(device)
                targets = targets.to(device)
                pred = model.forward(tokens)
                loss, loss_parts = compute_loss(pred, targets, target_indices)
                val_loss += float(loss.item()) * len(tokens)
                val_cosine += loss_parts["cosine"]
                val_contrastive += loss_parts["contrastive"]
                val_batches += 1
                val_n += len(tokens)
        avg_train = total_loss / max(1, batches)
        avg_train_cosine = total_cosine / max(1, batches)
        avg_train_contrastive = total_contrastive / max(1, batches)
        avg_val = val_loss / max(1, val_n)
        avg_val_cosine = val_cosine / max(1, val_batches)
        avg_val_contrastive = val_contrastive / max(1, val_batches)
        print(
            f"  epoch {epoch + 1}/{epochs}: "
            f"train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"train_cos={avg_train_cosine:.4f} val_cos={avg_val_cosine:.4f} "
            f"train_nce={avg_train_contrastive:.4f} val_nce={avg_val_contrastive:.4f}"
        )
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
    parser.add_argument("--tree-epochs", type=int, default=300)
    parser.add_argument("--tree-pairs-per-epoch", type=int, default=768)
    parser.add_argument("--tree-negatives", type=int, default=32)
    parser.add_argument("--tree-max-distance-sample", type=int, default=500)
    parser.add_argument("--tree-embedding-npz", type=Path)
    parser.add_argument("--train-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--pooling", choices=["legacy_mean", "masked_mean", "last_token"], default="masked_mean")
    parser.add_argument(
        "--loss-mode",
        choices=["cosine", "contrastive", "hybrid", "hierarchical_contrastive", "hierarchical_hybrid"],
        default="cosine",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--species-positive-weight", type=float, default=1.0)
    parser.add_argument("--genus-positive-weight", type=float, default=0.25)
    parser.add_argument("--family-positive-weight", type=float, default=0.1)
    parser.add_argument("--order-positive-weight", type=float, default=0.03)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--validation-mode",
        choices=["auto", "species_file", "random_species", "random_sequence"],
        default="auto",
        help=(
            "How to split reference sequences for model selection. Use random_sequence "
            "for clean fish-tree reruns where val_species.json would hold out most "
            "reference species from training."
        ),
    )
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
        out = args.output_dir / "dry_run_summary.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {out}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(args.seed)
    from phylo_zero_shot_common import read_tree

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
        validation_mode=args.validation_mode,
    )
    rank_id_arrays = build_rank_id_arrays(args.input_dir, candidate_labels) if args.loss_mode in HIERARCHICAL_LOSS_MODES else None
    needs_species_index = args.loss_mode in {"contrastive", "hybrid"} | HIERARCHICAL_LOSS_MODES
    train_dataset = SequenceEmbeddingDataset(
        train_df,
        tree_embeddings,
        args.max_seq_len,
        return_species_index=needs_species_index,
    )
    val_dataset = SequenceEmbeddingDataset(
        val_df,
        tree_embeddings,
        args.max_seq_len,
        return_species_index=needs_species_index,
    )
    model = PhyloMamba(embed_dim=args.embed_dim, pooling=args.pooling)
    checkpoint = train_sequence_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        candidate_embeddings=tree_embeddings,
        output_dir=args.output_dir,
        epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_mode=args.loss_mode,
        temperature=args.temperature,
        cosine_weight=args.cosine_weight,
        contrastive_weight=args.contrastive_weight,
        rank_id_arrays=rank_id_arrays,
        species_positive_weight=args.species_positive_weight,
        genus_positive_weight=args.genus_positive_weight,
        family_positive_weight=args.family_positive_weight,
        order_positive_weight=args.order_positive_weight,
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
                "model": "phylo_mamba",
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
        "model": "phylo_mamba",
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
