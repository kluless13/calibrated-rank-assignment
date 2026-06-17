#!/usr/bin/env python3
"""Train a TAXDNA-style learned co-occurrence model from NPZ tree embeddings."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phylo_zero_shot_common import load_tree_embedding_npz


@dataclass(frozen=True)
class CoOccGroup:
    group_id: str
    indices: list[int]
    has_reference_sequence: list[bool]


class NpzCoOccurrenceModel(nn.Module):
    """Low-rank community-context model matching TAXDNA's co-occurrence shape."""

    def __init__(self, num_species: int, hidden_dim: int):
        super().__init__()
        self.context_embedding = nn.Linear(num_species, hidden_dim, bias=False)
        self.subject_embedding = nn.Linear(hidden_dim, num_species, bias=False)

    def forward(self, class_probabilities: torch.Tensor) -> torch.Tensor:
        return self.subject_embedding(self.context_embedding(class_probabilities))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_groups(path: Path, label_to_idx: dict[str, int], min_species: int) -> tuple[list[CoOccGroup], dict[str, int]]:
    raw = json.loads(path.read_text())
    groups: list[CoOccGroup] = []
    skipped = {"too_small": 0, "unknown_species": 0}
    for group_id, species in raw.items():
        indices: list[int] = []
        has_reference: list[bool] = []
        unknown = 0
        for label, sequences in species.items():
            idx = label_to_idx.get(str(label))
            if idx is None:
                unknown += 1
                continue
            indices.append(idx)
            has_reference.append(bool(sequences))
        if len(indices) < min_species:
            skipped["too_small"] += 1
            skipped["unknown_species"] += unknown
            continue
        groups.append(CoOccGroup(str(group_id), indices, has_reference))
        skipped["unknown_species"] += unknown
    return groups, skipped


def group_probabilities(
    group: CoOccGroup,
    class_embeddings: torch.Tensor,
    kernel_temp: float,
    add_noise: bool,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_indices = torch.tensor(group.indices, dtype=torch.long, device=class_embeddings.device)
    group_embeddings = class_embeddings[target_indices]
    if add_noise and noise_std > 0:
        no_ref_mask = torch.tensor(
            [not flag for flag in group.has_reference_sequence],
            dtype=torch.bool,
            device=class_embeddings.device,
        )
        if bool(no_ref_mask.any()):
            noise = torch.randn_like(group_embeddings[no_ref_mask]) * noise_std
            group_embeddings = group_embeddings.clone()
            group_embeddings[no_ref_mask] = F.normalize(group_embeddings[no_ref_mask] + noise, dim=1)

    sims = F.normalize(group_embeddings, dim=1) @ F.normalize(class_embeddings, dim=1).T
    class_probabilities = torch.softmax(sims / kernel_temp, dim=1).clamp_min(1e-12)
    return class_probabilities, target_indices


def leave_one_out_logits(class_probabilities: torch.Tensor, coocc_logits: torch.Tensor) -> torch.Tensor:
    if coocc_logits.shape[0] <= 1:
        context = torch.zeros_like(coocc_logits)
    else:
        context = (coocc_logits.sum(dim=0, keepdim=True) - coocc_logits) / (coocc_logits.shape[0] - 1)
    return torch.log(class_probabilities) + context


def run_epoch(
    model: NpzCoOccurrenceModel,
    groups: list[CoOccGroup],
    class_embeddings: torch.Tensor,
    kernel_temp: float,
    add_noise: bool,
    noise_std: float,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    losses = []
    random.shuffle(groups) if optimizer is not None else None
    model.train(optimizer is not None)
    for group in groups:
        class_probabilities, target_indices = group_probabilities(
            group,
            class_embeddings,
            kernel_temp=kernel_temp,
            add_noise=add_noise,
            noise_std=noise_std,
        )
        coocc_logits = model(class_probabilities)
        final_logits = leave_one_out_logits(class_probabilities, coocc_logits)
        loss = F.cross_entropy(final_logits, target_indices)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-embedding-npz", type=Path, required=True)
    parser.add_argument("--cooccurrence-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--kernel-temp", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-species-per-group", type=int, default=2)
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--max-train-groups", type=int, help="Optional cap for smoke tests.")
    parser.add_argument("--seed", type=int, default=1206)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    labels, class_embeddings_np, tree_metadata = load_tree_embedding_npz(args.tree_embedding_npz)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    groups, skipped = load_groups(args.cooccurrence_json, label_to_idx, args.min_species_per_group)
    loaded_group_count = len(groups)
    if args.max_train_groups and len(groups) > args.max_train_groups:
        random.Random(args.seed).shuffle(groups)
        groups = groups[: args.max_train_groups]
    if len(groups) < 2 and not args.dry_run:
        raise RuntimeError("Need at least two co-occurrence groups to train.")

    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_fraction)) if len(groups) > 1 else 0
    val_groups = groups[:n_val]
    train_groups = groups[n_val:] if n_val else groups
    summary = {
        "tree_embedding_npz": str(args.tree_embedding_npz),
        "cooccurrence_json": str(args.cooccurrence_json),
        "candidate_species": len(labels),
        "available_groups_before_cap": loaded_group_count,
        "loaded_groups": len(groups),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "skipped_groups": skipped,
        "min_species_per_group": args.min_species_per_group,
        "max_train_groups": args.max_train_groups,
        "tree_embedding_metadata": tree_metadata,
    }
    if args.dry_run:
        out = args.output_dir / "dry_run_summary.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        print(f"Wrote {out}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_embeddings = torch.tensor(class_embeddings_np, dtype=torch.float32, device=device)
    model = NpzCoOccurrenceModel(num_species=len(labels), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = float("inf")
    checkpoint = args.output_dir / "npz_cooccurrence_model.pt"
    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model,
            train_groups,
            class_embeddings,
            kernel_temp=args.kernel_temp,
            add_noise=args.add_noise,
            noise_std=args.noise_std,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model,
                val_groups,
                class_embeddings,
                kernel_temp=args.kernel_temp,
                add_noise=False,
                noise_std=0.0,
                optimizer=None,
            )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        metric = val_loss if not np.isnan(val_loss) else train_loss
        if metric < best_val:
            best_val = metric
            torch.save(model.state_dict(), checkpoint)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "npz_taxdna_style_cooccurrence",
        "checkpoint": str(checkpoint),
        "output_dir": str(args.output_dir),
        "args": vars(args),
        "summary": summary,
        "history": history,
        "device": device,
        "notes": [
            "This model matches TAXDNA's low-rank co-occurrence shape but consumes fixed NPZ tree embeddings.",
            "Use with sequence query embeddings from the same tree embedding space.",
        ],
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(f"Wrote {checkpoint}")


if __name__ == "__main__":
    main()
