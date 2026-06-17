#!/usr/bin/env python3
"""Shared utilities for 12S zero-shot DNA-to-tree-embedding experiments."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import dendropy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CHAR_VOCAB = {"[PAD]": 0, "[UNK]": 1, "A": 2, "C": 3, "G": 4, "T": 5, "N": 6}


@dataclass(frozen=True)
class ZeroShotInputs:
    input_dir: Path
    species_sequences: dict[str, list[str]]
    train_species_sequences: dict[str, list[str]]
    zero_shot_queries: pd.DataFrame
    candidate_species: pd.DataFrame
    manifest: dict[str, object]


def clean_seq(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper().strip() if ch in "ACGTN")


def tokenize_char(seq: object, max_seq_len: int) -> list[int]:
    tokens = [CHAR_VOCAB.get(ch, CHAR_VOCAB["N"]) for ch in clean_seq(seq)]
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    return [CHAR_VOCAB["[PAD]"]] * (max_seq_len - len(tokens)) + tokens


def load_zero_shot_inputs(input_dir: Path, train_only: bool = False) -> ZeroShotInputs:
    species_path = input_dir / "species_sequences.json"
    train_species_path = input_dir / "train_species_sequences.json"
    query_path = input_dir / "zero_shot_queries.csv"
    candidate_path = input_dir / "candidate_species.csv"
    manifest_path = input_dir / "manifest.json"
    missing = [
        path
        for path in [species_path, train_species_path, query_path, candidate_path, manifest_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    species_sequences = json.loads(species_path.read_text())
    train_species_sequences = json.loads(train_species_path.read_text())
    if train_only:
        species_sequences = train_species_sequences

    return ZeroShotInputs(
        input_dir=input_dir,
        species_sequences={
            str(label): [clean_seq(seq) for seq in seqs if clean_seq(seq)]
            for label, seqs in species_sequences.items()
        },
        train_species_sequences={
            str(label): [clean_seq(seq) for seq in seqs if clean_seq(seq)]
            for label, seqs in train_species_sequences.items()
        },
        zero_shot_queries=pd.read_csv(query_path),
        candidate_species=pd.read_csv(candidate_path),
        manifest=json.loads(manifest_path.read_text()),
    )


def read_tree(tree_file: Path) -> tuple[dendropy.Tree, dict[str, dendropy.Taxon]]:
    tree = dendropy.Tree.get(path=str(tree_file), schema="newick", preserve_underscores=True)
    labels = {
        taxon.label.strip("'\""): taxon
        for taxon in tree.taxon_namespace
        if taxon.label
    }
    return tree, labels


def ordered_candidate_labels(candidate_species: pd.DataFrame, tree_labels: set[str]) -> list[str]:
    candidates = [
        str(label)
        for label in candidate_species["tree_label"].dropna().astype(str).tolist()
        if str(label) in tree_labels
    ]
    return list(dict.fromkeys(candidates))


def reference_dataframe(species_sequences: dict[str, list[str]], species_to_idx: dict[str, int]) -> pd.DataFrame:
    rows = []
    for label, seqs in species_sequences.items():
        if label not in species_to_idx:
            continue
        for i, seq in enumerate(seqs):
            rows.append({
                "tree_label": label,
                "species_index": species_to_idx[label],
                "sequence_index": i,
                "nucleotides": seq,
                "seq_len": len(seq),
            })
    return pd.DataFrame(rows)


def split_reference_dataframe(
    ref_df: pd.DataFrame,
    input_dir: Path,
    val_fraction: float,
    seed: int,
    validation_mode: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if validation_mode not in {"auto", "species_file", "random_species", "random_sequence"}:
        raise ValueError(f"Unknown validation mode: {validation_mode}")

    val_path = input_dir / "val_species.json"
    if validation_mode in {"auto", "species_file"} and val_path.exists():
        val_species = set(str(label) for label in json.loads(val_path.read_text()))
        val_mask = ref_df["tree_label"].isin(val_species)
        if val_mask.any() and (~val_mask).any():
            train_df = ref_df[~val_mask].copy()
            val_df = ref_df[val_mask].copy()
            return train_df, val_df, {
                "mode": "val_species_json",
                "val_species_json": str(val_path),
                "train_sequences": int(len(train_df)),
                "val_sequences": int(len(val_df)),
                "train_species": int(train_df["tree_label"].nunique()),
                "val_species": int(val_df["tree_label"].nunique()),
            }
        if validation_mode == "species_file":
            raise RuntimeError(f"Cannot build non-empty train/val split from {val_path}")

    rng = np.random.default_rng(seed)
    if validation_mode == "random_sequence":
        indices = np.arange(len(ref_df))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_fraction))
        n_val = min(n_val, len(indices) - 1)
        val_indices = set(indices[:n_val].tolist())
        val_mask = ref_df.index.isin(val_indices)
        train_df = ref_df[~val_mask].copy()
        val_df = ref_df[val_mask].copy()
        return train_df, val_df, {
            "mode": "random_sequence_split",
            "val_fraction": val_fraction,
            "seed": seed,
            "train_sequences": int(len(train_df)),
            "val_sequences": int(len(val_df)),
            "train_species": int(train_df["tree_label"].nunique()),
            "val_species": int(val_df["tree_label"].nunique()),
        }

    species = np.array(sorted(ref_df["tree_label"].unique()))
    n_val = max(1, int(len(species) * val_fraction))
    val_species = set(rng.choice(species, size=min(n_val, len(species) - 1), replace=False))
    val_mask = ref_df["tree_label"].isin(val_species)
    train_df = ref_df[~val_mask].copy()
    val_df = ref_df[val_mask].copy()
    return train_df, val_df, {
        "mode": "random_species_split",
        "val_fraction": val_fraction,
        "seed": seed,
        "train_sequences": int(len(train_df)),
        "val_sequences": int(len(val_df)),
        "train_species": int(train_df["tree_label"].nunique()),
        "val_species": int(val_df["tree_label"].nunique()),
    }


class SequenceEmbeddingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_embeddings: np.ndarray,
        max_seq_len: int,
        return_species_index: bool = False,
    ):
        self.seqs = df["nucleotides"].tolist()
        self.indices = df["species_index"].astype(int).tolist()
        self.target_embeddings = target_embeddings
        self.max_seq_len = max_seq_len
        self.return_species_index = return_species_index

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(tokenize_char(self.seqs[idx], self.max_seq_len), dtype=torch.long)
        target = torch.tensor(self.target_embeddings[self.indices[idx]], dtype=torch.float32)
        if self.return_species_index:
            return tokens, target, torch.tensor(self.indices[idx], dtype=torch.long)
        return tokens, target


class QueryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_seq_len: int):
        self.seqs = df["nucleotides"].tolist()
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(tokenize_char(self.seqs[idx], self.max_seq_len), dtype=torch.long)


def estimate_max_tree_distance(pdm: dendropy.PhylogeneticDistanceMatrix, labels: list[str], taxa: dict[str, dendropy.Taxon], sample_size: int) -> float:
    sample = [label for label in labels[:sample_size] if label in taxa]
    max_dist = 0.0
    for i, label_a in enumerate(sample):
        for label_b in sample[i + 1:]:
            distance = float(pdm(taxa[label_a], taxa[label_b]))
            if distance > max_dist:
                max_dist = distance
    return max_dist or 1.0


def learn_tree_embeddings(
    tree_file: Path,
    labels: list[str],
    embed_dim: int,
    epochs: int,
    pairs_per_epoch: int,
    negatives_per_anchor: int,
    sample_size_for_max_distance: int,
    device: str,
    seed: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tree, taxa = read_tree(tree_file)
    pdm = tree.phylogenetic_distance_matrix()
    max_dist = estimate_max_tree_distance(pdm, labels, taxa, sample_size_for_max_distance)

    n_species = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    embeddings = torch.nn.Embedding(n_species, embed_dim).to(device)
    torch.nn.init.normal_(embeddings.weight, std=0.1)
    optimizer = torch.optim.AdamW(embeddings.parameters(), lr=0.01, weight_decay=1e-4)

    for epoch in range(epochs):
        anchors = torch.randperm(n_species)[:min(pairs_per_epoch, n_species)]
        total_loss = 0.0
        n_steps = 0
        for anchor_tensor in anchors:
            anchor_idx = int(anchor_tensor.item())
            anchor_label = labels[anchor_idx]
            if anchor_label not in taxa:
                continue
            neg_indices = torch.randint(0, n_species, (negatives_per_anchor,))
            batch_indices = torch.cat([torch.tensor([anchor_idx]), neg_indices]).to(device)
            batch_emb = embeddings(batch_indices)
            anchor_emb = batch_emb[0:1]
            other_emb = batch_emb[1:]
            pred_dist = 1 - F.cosine_similarity(anchor_emb.expand_as(other_emb), other_emb)
            target_dist = []
            for idx_tensor in neg_indices:
                other_label = labels[int(idx_tensor.item())]
                try:
                    dist = float(pdm(taxa[anchor_label], taxa[other_label])) / max_dist
                except Exception:
                    dist = 0.5
                target_dist.append(min(dist, 1.0))
            target = torch.tensor(target_dist, dtype=torch.float32, device=device)
            loss = F.mse_loss(pred_dist, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_steps += 1
        if (epoch + 1) % max(1, epochs // 5) == 0:
            avg = total_loss / max(1, n_steps)
            print(f"  tree-embedding epoch {epoch + 1}/{epochs}: loss={avg:.6f}")

    with torch.no_grad():
        return F.normalize(embeddings.weight.detach(), dim=1).cpu().numpy()


def save_tree_embedding_npz(path: Path, labels: list[str], embeddings: np.ndarray, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=np.array(labels, dtype=object),
        embeddings=embeddings.astype(np.float32),
        metadata=json.dumps(metadata, sort_keys=True),
    )


def load_tree_embedding_npz(path: Path) -> tuple[list[str], np.ndarray, dict[str, object]]:
    payload = np.load(path, allow_pickle=True)
    labels = [str(label) for label in payload["labels"].tolist()]
    embeddings = payload["embeddings"].astype(np.float32)
    metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}
    return labels, embeddings, metadata


def save_query_embedding_npz(
    path: Path,
    queries: pd.DataFrame,
    embeddings: np.ndarray,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        processids=queries["processid"].astype(str).to_numpy(dtype=object),
        embeddings=embeddings.astype(np.float32),
        metadata=json.dumps(metadata, sort_keys=True, default=str),
    )


def load_query_embedding_npz(path: Path) -> tuple[list[str], np.ndarray, dict[str, object]]:
    payload = np.load(path, allow_pickle=True)
    processids = [str(processid) for processid in payload["processids"].tolist()]
    embeddings = payload["embeddings"].astype(np.float32)
    metadata = json.loads(str(payload["metadata"])) if "metadata" in payload else {}
    return processids, embeddings, metadata


def extract_embeddings(model: torch.nn.Module, queries: pd.DataFrame, max_seq_len: int, batch_size: int, device: str, num_workers: int) -> np.ndarray:
    loader = DataLoader(
        QueryDataset(queries, max_seq_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for tokens in loader:
            emb = model.get_embeddings(tokens.to(device))
            outputs.append(emb.detach().cpu().numpy())
    return np.vstack(outputs)


def ranked_predictions(
    query_embeddings: np.ndarray,
    candidate_labels: list[str],
    candidate_embeddings: np.ndarray,
    top_k: int,
) -> tuple[list[list[str]], list[list[float]]]:
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    cand_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
    scores = query_norm @ cand_norm.T
    top_k = min(top_k, len(candidate_labels))
    top_indices = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    ranked_labels: list[list[str]] = []
    ranked_scores: list[list[float]] = []
    for row_idx, indices in enumerate(top_indices):
        ordered = indices[np.argsort(-scores[row_idx, indices])]
        ranked_labels.append([candidate_labels[int(idx)] for idx in ordered])
        ranked_scores.append([float(scores[row_idx, int(idx)]) for idx in ordered])
    return ranked_labels, ranked_scores


def write_prediction_csv(
    output_path: Path,
    queries: pd.DataFrame,
    ranked_labels: list[list[str]],
    ranked_scores: list[list[float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (_, query), labels, scores in zip(queries.iterrows(), ranked_labels, ranked_scores):
        rows.append({
            "processid": query["processid"],
            "true_tree_label": query["tree_label"],
            "species_name": query.get("species_name"),
            "genus_name": query.get("genus_name"),
            "family_name": query.get("family_name"),
            "order_name": query.get("order_name"),
            "eventID": query.get("eventID"),
            "top_tree_labels": json.dumps(labels),
            "top_scores": json.dumps([round(score, 8) for score in scores]),
            "pred_tree_label": labels[0] if labels else None,
            "pred_score": scores[0] if scores else None,
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def dry_run_summary(inputs: ZeroShotInputs, tree_file: Path, max_seq_len: int, train_only: bool) -> dict[str, object]:
    _, taxa = read_tree(tree_file)
    tree_labels = set(taxa)
    candidate_labels = ordered_candidate_labels(inputs.candidate_species, tree_labels)
    reference_df = reference_dataframe(inputs.species_sequences, {label: i for i, label in enumerate(candidate_labels)})
    seq_lengths = reference_df["seq_len"] if len(reference_df) else pd.Series(dtype=float)
    query_lengths = inputs.zero_shot_queries["nucleotides"].map(lambda seq: len(clean_seq(seq)))
    return {
        "input_dir": str(inputs.input_dir),
        "tree_file": str(tree_file),
        "tree_tip_count": len(tree_labels),
        "candidate_count": len(candidate_labels),
        "reference_species": len(inputs.species_sequences),
        "reference_sequences_in_candidate_tree": int(len(reference_df)),
        "zero_shot_queries": int(len(inputs.zero_shot_queries)),
        "zero_shot_query_species": int(inputs.zero_shot_queries["tree_label"].nunique()),
        "train_only_reference": train_only,
        "max_seq_len": max_seq_len,
        "reference_seq_len_min": int(seq_lengths.min()) if len(seq_lengths) else None,
        "reference_seq_len_median": float(seq_lengths.median()) if len(seq_lengths) else None,
        "reference_seq_len_max": int(seq_lengths.max()) if len(seq_lengths) else None,
        "reference_sequences_over_max_len": int((seq_lengths > max_seq_len).sum()) if len(seq_lengths) else 0,
        "query_seq_len_min": int(query_lengths.min()) if len(query_lengths) else None,
        "query_seq_len_median": float(query_lengths.median()) if len(query_lengths) else None,
        "query_seq_len_max": int(query_lengths.max()) if len(query_lengths) else None,
        "query_sequences_over_max_len": int((query_lengths > max_seq_len).sum()) if len(query_lengths) else 0,
    }
