#!/usr/bin/env python3
"""Train or run non-Mamba barcode encoders for fish-tree assignment.

This script keeps the Paper 1 objective fixed while swapping only the sequence
encoder. It supports lightweight CNN, biLSTM, and Transformer encoders trained
to map COI barcodes into the same species-tree embedding space used by the
MarineMamba runs.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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
    reference_dataframe,
    save_query_embedding_npz,
    save_tree_embedding_npz,
    split_reference_dataframe,
    write_prediction_csv,
)


HIERARCHICAL_LOSS_MODES = {"hierarchical_contrastive", "hierarchical_hybrid"}
MODEL_TYPES = {"cnn", "bilstm", "transformer"}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_rank(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() == "nan" else text


def build_rank_id_arrays(input_dir: Path, candidate_labels: list[str]) -> dict[str, np.ndarray]:
    info_path = input_dir / "species_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Hierarchical loss requires {info_path}")
    species_info = json.loads(info_path.read_text())
    candidate_rows = {}
    candidate_path = input_dir / "candidate_species.csv"
    if candidate_path.exists():
        candidate_rows = pd.read_csv(candidate_path).set_index("tree_label").to_dict(orient="index")

    def rank_value(label: str, rank: str) -> str | None:
        value = clean_rank(species_info.get(label, {}).get(rank))
        if value:
            return value
        row = candidate_rows.get(label, {})
        if rank == "genus":
            return clean_rank(row.get("genus_name")) or clean_rank(row.get("genus_from_label"))
        if rank == "family":
            return clean_rank(row.get("family_name"))
        if rank == "order":
            return clean_rank(row.get("order_name"))
        return None

    arrays: dict[str, np.ndarray] = {}
    for rank in ["genus", "family", "order"]:
        value_to_id: dict[str, int] = {}
        ids = []
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


class MaskedPoolingMixin:
    @staticmethod
    def masked_mean(hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(CHAR_VOCAB["[PAD]"]).unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    @staticmethod
    def masked_max(hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(CHAR_VOCAB["[PAD]"]).unsqueeze(-1)
        masked = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min)
        pooled = masked.max(dim=1).values
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


class CnnBarcodeEncoder(nn.Module, MaskedPoolingMixin):
    def __init__(self, embed_dim: int, d_model: int, token_emb_dim: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(len(CHAR_VOCAB), token_emb_dim, padding_idx=CHAR_VOCAB["[PAD]"])
        self.conv = nn.Sequential(
            nn.Conv1d(token_emb_dim, d_model, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens).transpose(1, 2)
        hidden = self.conv(x).transpose(1, 2)
        pooled = torch.cat([self.masked_mean(hidden, tokens), self.masked_max(hidden, tokens)], dim=1)
        return self.proj(pooled)

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward(tokens), dim=1)


class BiLstmBarcodeEncoder(nn.Module):
    def __init__(self, embed_dim: int, d_model: int, token_emb_dim: int, num_layers: int, dropout: float):
        super().__init__()
        hidden_size = max(16, d_model // 2)
        self.token_embedding = nn.Embedding(len(CHAR_VOCAB), token_emb_dim, padding_idx=CHAR_VOCAB["[PAD]"])
        self.lstm = nn.LSTM(
            input_size=token_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        lengths = tokens.ne(CHAR_VOCAB["[PAD]"]).sum(dim=1).clamp_min(1).cpu()
        x = self.token_embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.proj(hidden)

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward(tokens), dim=1)


class TransformerBarcodeEncoder(nn.Module, MaskedPoolingMixin):
    def __init__(
        self,
        embed_dim: int,
        d_model: int,
        max_seq_len: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("--d-model must be divisible by --num-heads")
        self.token_embedding = nn.Embedding(len(CHAR_VOCAB), d_model, padding_idx=CHAR_VOCAB["[PAD]"])
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        key_padding_mask = tokens.eq(CHAR_VOCAB["[PAD]"])
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.proj(self.masked_mean(hidden, tokens))

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward(tokens), dim=1)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model_type == "cnn":
        return CnnBarcodeEncoder(
            embed_dim=args.embed_dim,
            d_model=args.d_model,
            token_emb_dim=args.token_emb_dim,
            dropout=args.dropout,
        )
    if args.model_type == "bilstm":
        return BiLstmBarcodeEncoder(
            embed_dim=args.embed_dim,
            d_model=args.d_model,
            token_emb_dim=args.token_emb_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if args.model_type == "transformer":
        return TransformerBarcodeEncoder(
            embed_dim=args.embed_dim,
            d_model=args.d_model,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model type: {args.model_type}")


def train_sequence_model(
    model: nn.Module,
    train_dataset: SequenceEmbeddingDataset,
    val_dataset: SequenceEmbeddingDataset,
    candidate_embeddings: np.ndarray,
    rank_id_arrays: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
    device: str,
) -> Path:
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=len(train_dataset) >= args.batch_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs)
    checkpoint = args.output_dir / f"{args.model_type}_tree_encoder_best.pt"
    best_val = float("inf")
    candidate_tensor = F.normalize(torch.tensor(candidate_embeddings, dtype=torch.float32, device=device), dim=1)
    rank_tensors = {
        rank: torch.tensor(values, dtype=torch.long, device=device)
        for rank, values in (rank_id_arrays or {}).items()
    }

    def hierarchical_targets(target_indices: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        weights = torch.zeros_like(logits)
        for rank, weight in [
            ("order", args.order_positive_weight),
            ("family", args.family_positive_weight),
            ("genus", args.genus_positive_weight),
        ]:
            rank_ids = rank_tensors.get(rank)
            if rank_ids is None or weight <= 0:
                continue
            target_rank = rank_ids[target_indices].unsqueeze(1)
            valid = target_rank.ge(0) & rank_ids.unsqueeze(0).ge(0)
            match = target_rank.eq(rank_ids.unsqueeze(0)) & valid
            weights = torch.maximum(weights, match.to(logits.dtype) * weight)
        weights.scatter_(1, target_indices.unsqueeze(1), args.species_positive_weight)
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
        if args.loss_mode in {"contrastive", "hybrid"} | HIERARCHICAL_LOSS_MODES:
            if target_indices is None:
                raise RuntimeError("Contrastive losses require species indices.")
            logits = pred @ candidate_tensor.T / args.temperature
            if args.loss_mode in HIERARCHICAL_LOSS_MODES:
                soft_targets = hierarchical_targets(target_indices, logits)
                contrastive_loss = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                contrastive_loss = F.cross_entropy(logits, target_indices)
        if args.loss_mode == "cosine":
            loss = cosine_loss
        elif args.loss_mode in {"contrastive", "hierarchical_contrastive"}:
            loss = contrastive_loss
        else:
            loss = args.cosine_weight * cosine_loss + args.contrastive_weight * contrastive_loss
        return loss, {
            "cosine": float(cosine_loss.detach().item()),
            "contrastive": float(contrastive_loss.detach().item()),
        }

    history = []
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = train_cosine = train_contrastive = 0.0
        train_batches = 0
        for batch in train_loader:
            if len(batch) == 3:
                tokens, targets, target_indices = batch
                target_indices = target_indices.to(device)
            else:
                tokens, targets = batch
                target_indices = None
            tokens = tokens.to(device)
            targets = targets.to(device)
            pred = model(tokens)
            loss, parts = compute_loss(pred, targets, target_indices)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_cosine += parts["cosine"]
            train_contrastive += parts["contrastive"]
            train_batches += 1
        scheduler.step()

        model.eval()
        val_loss = val_cosine = val_contrastive = 0.0
        val_batches = val_n = 0
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
                pred = model(tokens)
                loss, parts = compute_loss(pred, targets, target_indices)
                val_loss += float(loss.item()) * len(tokens)
                val_cosine += parts["cosine"]
                val_contrastive += parts["contrastive"]
                val_batches += 1
                val_n += len(tokens)
        avg_train = train_loss / max(1, train_batches)
        avg_val = val_loss / max(1, val_n)
        record = {
            "epoch": epoch + 1,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "train_cosine": train_cosine / max(1, train_batches),
            "val_cosine": val_cosine / max(1, val_batches),
            "train_contrastive": train_contrastive / max(1, train_batches),
            "val_contrastive": val_contrastive / max(1, val_batches),
        }
        history.append(record)
        print(
            f"  epoch {epoch + 1}/{args.train_epochs}: "
            f"train_loss={record['train_loss']:.4f} val_loss={record['val_loss']:.4f} "
            f"train_cos={record['train_cosine']:.4f} val_cos={record['val_cosine']:.4f} "
            f"train_nce={record['train_contrastive']:.4f} val_nce={record['val_contrastive']:.4f}",
            flush=True,
        )
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), checkpoint)

    (args.output_dir / "train_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return checkpoint


def load_common_inputs(args: argparse.Namespace):
    inputs = load_zero_shot_inputs(args.input_dir, train_only=args.train_only_reference)
    _, taxa = read_tree(args.tree_file)
    candidate_labels = ordered_candidate_labels(inputs.candidate_species, set(taxa))
    species_to_idx = {label: idx for idx, label in enumerate(candidate_labels)}
    return inputs, candidate_labels, species_to_idx


def load_or_build_tree_embeddings(args: argparse.Namespace, candidate_labels: list[str], device: str) -> np.ndarray:
    if args.tree_embedding_npz and args.tree_embedding_npz.exists():
        loaded_labels, tree_embeddings, _ = load_tree_embedding_npz(args.tree_embedding_npz)
        if loaded_labels != candidate_labels:
            raise RuntimeError("Loaded tree embedding labels do not match current candidate labels.")
        return tree_embeddings
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
    return tree_embeddings


def predict_and_evaluate(
    model: nn.Module,
    inputs,
    candidate_labels: list[str],
    tree_embeddings: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> tuple[Path, Path, Path | None]:
    query_embeddings = extract_embeddings(
        model,
        inputs.zero_shot_queries,
        max_seq_len=args.max_seq_len,
        batch_size=args.predict_batch_size or args.batch_size,
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
                "model": args.model_type,
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
    return prediction_csv, metrics_dir, query_embedding_npz


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=sorted(MODEL_TYPES), default="cnn")
    parser.add_argument("--max-seq-len", type=int, default=700)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--token-emb-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tree-embedding-npz", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--train-only-reference", action="store_true")
    parser.add_argument("--write-query-embeddings", action="store_true")
    parser.add_argument("--predict-batch-size", type=int)


def model_config(args: argparse.Namespace) -> dict[str, object]:
    keys = [
        "model_type",
        "max_seq_len",
        "embed_dim",
        "d_model",
        "token_emb_dim",
        "num_layers",
        "num_heads",
        "dropout",
    ]
    return {key: getattr(args, key) for key in keys}


def apply_model_config(args: argparse.Namespace, config: dict[str, object]) -> None:
    for key, value in config.items():
        setattr(args, key, value)


def command_train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_zero_shot_inputs(args.input_dir, train_only=args.train_only_reference)
    if args.dry_run:
        summary = dry_run_summary(inputs, args.tree_file, args.max_seq_len, args.train_only_reference)
        summary["model_config"] = model_config(args)
        out = args.output_dir / "dry_run_summary.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(args.seed)
    inputs, candidate_labels, species_to_idx = load_common_inputs(args)
    ref_df = reference_dataframe(inputs.species_sequences, species_to_idx)
    if len(ref_df) < 2:
        raise RuntimeError("Need at least two reference sequences to train.")
    tree_embeddings = load_or_build_tree_embeddings(args, candidate_labels, device)
    train_df, val_df, split_manifest = split_reference_dataframe(
        ref_df,
        input_dir=args.input_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        validation_mode=args.validation_mode,
    )
    needs_species_index = args.loss_mode in {"contrastive", "hybrid"} | HIERARCHICAL_LOSS_MODES
    rank_id_arrays = build_rank_id_arrays(args.input_dir, candidate_labels) if args.loss_mode in HIERARCHICAL_LOSS_MODES else None
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
    model = build_model(args)
    checkpoint = train_sequence_model(model, train_dataset, val_dataset, tree_embeddings, rank_id_arrays, args, device)
    args.checkpoint = checkpoint
    prediction_csv, metrics_dir, query_embedding_npz = predict_and_evaluate(
        model,
        inputs,
        candidate_labels,
        tree_embeddings,
        args,
        device,
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": f"fish_tree_{args.model_type}_encoder",
        "model_config": model_config(args),
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


def command_predict(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = json.loads(args.run_manifest.read_text())
    apply_model_config(args, run_manifest["model_config"])
    args.batch_size = args.predict_batch_size or 128
    if args.tree_embedding_npz is None:
        args.tree_embedding_npz = Path(run_manifest["tree_embedding_npz"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_global_seed(args.seed)
    inputs, candidate_labels, _ = load_common_inputs(args)
    loaded_labels, tree_embeddings, _ = load_tree_embedding_npz(args.tree_embedding_npz)
    if loaded_labels != candidate_labels:
        raise RuntimeError("Loaded tree embedding labels do not match current candidate labels.")
    model = build_model(args).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    prediction_csv, metrics_dir, query_embedding_npz = predict_and_evaluate(
        model,
        inputs,
        candidate_labels,
        tree_embeddings,
        args,
        device,
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": f"fish_tree_{args.model_type}_encoder",
        "source_run_manifest": str(args.run_manifest),
        "input_dir": str(args.input_dir),
        "tree_file": str(args.tree_file),
        "tree_embedding_npz": str(args.tree_embedding_npz),
        "checkpoint": str(args.checkpoint),
        "prediction_csv": str(prediction_csv),
        "query_embedding_npz": str(query_embedding_npz) if query_embedding_npz else None,
        "metrics_dir": str(metrics_dir),
        "zero_shot_queries": len(inputs.zero_shot_queries),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(f"Wrote predictions to {prediction_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an encoder and evaluate on --input-dir.")
    add_common_args(train_parser)
    train_parser.add_argument("--tree-epochs", type=int, default=300)
    train_parser.add_argument("--tree-pairs-per-epoch", type=int, default=768)
    train_parser.add_argument("--tree-negatives", type=int, default=32)
    train_parser.add_argument("--tree-max-distance-sample", type=int, default=500)
    train_parser.add_argument("--train-epochs", type=int, default=40)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=5e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument(
        "--loss-mode",
        choices=["cosine", "contrastive", "hybrid", "hierarchical_contrastive", "hierarchical_hybrid"],
        default="cosine",
    )
    train_parser.add_argument("--temperature", type=float, default=0.07)
    train_parser.add_argument("--cosine-weight", type=float, default=0.1)
    train_parser.add_argument("--contrastive-weight", type=float, default=1.0)
    train_parser.add_argument("--species-positive-weight", type=float, default=1.0)
    train_parser.add_argument("--genus-positive-weight", type=float, default=0.25)
    train_parser.add_argument("--family-positive-weight", type=float, default=0.1)
    train_parser.add_argument("--order-positive-weight", type=float, default=0.03)
    train_parser.add_argument("--val-fraction", type=float, default=0.1)
    train_parser.add_argument(
        "--validation-mode",
        choices=["auto", "species_file", "random_species", "random_sequence"],
        default="random_sequence",
    )
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Run a trained encoder on another split.")
    add_common_args(predict_parser)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--run-manifest", type=Path, required=True)
    predict_parser.set_defaults(func=command_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
