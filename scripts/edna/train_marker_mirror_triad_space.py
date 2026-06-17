#!/usr/bin/env python3
"""Train a shared 12S/16S/COI MarkerMirror species-space prototype.

This tests the practical version of:

    12S -> 16S -> COI

Rather than hard-chaining predictions, the model learns one shared projection
head over frozen DNA foundation embeddings using all available pairwise marker
overlaps.  It then evaluates direct cross-marker retrieval in both directions
for 12S/16S, 12S/COI, and 16S/COI.

This is candidate-retrieval evidence only.  Species calls still require the
downstream rank/no-call calibration layer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.edna.train_marker_mirror_bridge import (
    Logger,
    ProjectionHead,
    clean_sequence,
    contrastive_loss,
    embed_sequences,
    evaluate,
    flatten,
    load_hf,
    load_species_json,
    load_taxonomy,
    sample_species_batch,
    split_species,
    taxonomy_soft_contrastive_loss,
)


RANKS_FOR_SELECTION = ("genus", "family", "order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--marker-a-name", default="12S")
    parser.add_argument("--marker-a-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource")
    parser.add_argument("--marker-b-name", default="16S")
    parser.add_argument("--marker-b-input-dir", type=Path, default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource")
    parser.add_argument("--marker-c-name", default="COI")
    parser.add_argument("--marker-c-input-dir", type=Path, default=ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs" / "eval_c")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best",
    )
    parser.add_argument("--max-per-species", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps-per-epoch", type=int, default=90)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-strategy", choices=("random", "taxonomy_hard"), default="taxonomy_hard")
    parser.add_argument("--loss-mode", choices=("hard_ce", "taxonomy_soft"), default="taxonomy_soft")
    parser.add_argument("--same-species-weight", type=float, default=1.0)
    parser.add_argument("--same-genus-weight", type=float, default=0.20)
    parser.add_argument("--same-family-weight", type=float, default=0.08)
    parser.add_argument("--same-order-weight", type=float, default=0.03)
    parser.add_argument("--val-steps", type=int, default=20)
    parser.add_argument("--restore-best-retrieval", action="store_true")
    parser.add_argument("--retrieval-eval-every", type=int, default=10)
    parser.add_argument("--retrieval-selection-k", type=int, default=10)
    parser.add_argument("--retrieval-selection-ranks", default="genus,family,order")
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def load_marker_inputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    specs = [
        (args.marker_a_name, args.marker_a_input_dir),
        (args.marker_b_name, args.marker_b_input_dir),
        (args.marker_c_name, args.marker_c_input_dir),
    ]
    out: dict[str, dict[str, Any]] = {}
    for name, input_dir in specs:
        species = load_species_json(input_dir / "train_species_sequences.json", args.max_per_species)
        out[name] = {"input_dir": input_dir, "species": species}
    return out


def pair_names(marker_names: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for i, left in enumerate(marker_names):
        for right in marker_names[i + 1 :]:
            pairs.append((left, right))
    return pairs


def pair_species(markers: dict[str, dict[str, Any]], left: str, right: str, labels: list[str]) -> list[str]:
    left_species = markers[left]["species"]
    right_species = markers[right]["species"]
    return [label for label in labels if label in left_species and label in right_species and left_species[label] and right_species[label]]


def loss_fn(
    zx: torch.Tensor,
    zy: torch.Tensor,
    batch_species: list[str],
    tax: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.loss_mode == "taxonomy_soft":
        return taxonomy_soft_contrastive_loss(
            zx,
            zy,
            batch_species,
            tax,
            args.temperature,
            args.same_species_weight,
            args.same_genus_weight,
            args.same_family_weight,
            args.same_order_weight,
        )
    return contrastive_loss(zx, zy, args.temperature)


def evaluate_direction(
    split: str,
    source_name: str,
    target_name: str,
    labels: list[str],
    markers: dict[str, dict[str, Any]],
    shared_head: ProjectionHead | None,
    tax: dict[str, dict[str, str]],
    device: str,
    model_name: str,
) -> list[dict[str, Any]]:
    source = markers[source_name]
    target = markers[target_name]
    mask = source["rows"]["tree_label"].isin(labels).to_numpy()
    return evaluate(
        f"{split}_{source_name}_to_{target_name}",
        source["rows"][mask].reset_index(drop=True),
        target["rows"],
        source["emb"][mask],
        target["emb"],
        shared_head,
        shared_head,
        tax,
        device,
        model_name,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_triad_space.log")
    logger.log(f"Arguments: {vars(args)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    markers = load_marker_inputs(args)
    marker_names = list(markers)
    pairs = pair_names(marker_names)
    union_species = sorted(set().union(*(set(markers[name]["species"]) for name in marker_names)))
    splits = split_species(union_species, args.seed)
    logger.log(f"Union species={len(union_species)} train/val/test={[len(splits[k]) for k in ['train','val','test']]}")
    for left, right in pairs:
        logger.log(
            f"Pair {left}<->{right} overlap total/train/val/test="
            f"{len(pair_species(markers, left, right, union_species))}/"
            f"{len(pair_species(markers, left, right, splits['train']))}/"
            f"{len(pair_species(markers, left, right, splits['val']))}/"
            f"{len(pair_species(markers, left, right, splits['test']))}"
        )

    tax: dict[str, dict[str, str]] = {}
    for marker in markers.values():
        tax.update(load_taxonomy(marker["input_dir"] / "candidate_species.csv"))

    tokenizer, model, masked_lm = load_hf(args.model, args.device, logger)
    for name, marker in markers.items():
        marker["rows"] = flatten(marker["species"], sorted(marker["species"]), name)
        marker["emb"] = embed_sequences(
            marker["rows"]["nucleotides"].map(clean_sequence).tolist(),
            tokenizer,
            model,
            masked_lm,
            args.device,
            args.embed_batch_size,
            args.max_length,
            logger,
            name,
        )
        by_species: dict[str, list[int]] = defaultdict(list)
        for i, label in enumerate(marker["rows"]["tree_label"]):
            by_species[str(label)].append(i)
        marker["by_species"] = by_species

    dim = next(iter(markers.values()))["emb"].shape[1]
    shared_head = ProjectionHead(dim, args.hidden_dim, args.projection_dim).to(args.device)
    opt = torch.optim.AdamW(shared_head.parameters(), lr=args.lr, weight_decay=1e-4)
    retrieval_selection_ranks = [rank.strip() for rank in args.retrieval_selection_ranks.split(",") if rank.strip()]
    pair_train = {pair: pair_species(markers, pair[0], pair[1], splits["train"]) for pair in pairs}
    pair_val = {pair: pair_species(markers, pair[0], pair[1], splits["val"]) for pair in pairs}
    train_pairs = [pair for pair in pairs if pair_train[pair]]
    if not train_pairs:
        raise ValueError("No train pair overlaps available")

    best_retrieval_score = -math.inf
    best_retrieval_epoch = 0
    best_state: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        shared_head.train()
        losses = []
        for step in range(args.steps_per_epoch):
            left, right = train_pairs[step % len(train_pairs)]
            batch_pool = pair_train[(left, right)]
            batch_species = sample_species_batch(rng, batch_pool, tax, args.batch_size, args.batch_strategy)
            idx_left = [int(rng.choice(markers[left]["by_species"][label])) for label in batch_species]
            idx_right = [int(rng.choice(markers[right]["by_species"][label])) for label in batch_species]
            z_left = torch.tensor(markers[left]["emb"][idx_left], device=args.device)
            z_right = torch.tensor(markers[right]["emb"][idx_right], device=args.device)
            loss = loss_fn(shared_head(z_left), shared_head(z_right), batch_species, tax, args)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_losses = []
        if args.val_steps > 0:
            shared_head.eval()
            with torch.no_grad():
                for step in range(args.val_steps):
                    left, right = train_pairs[step % len(train_pairs)]
                    batch_pool = pair_val.get((left, right), [])
                    if not batch_pool:
                        continue
                    batch_species = sample_species_batch(rng, batch_pool, tax, args.batch_size, args.batch_strategy)
                    idx_left = [int(rng.choice(markers[left]["by_species"][label])) for label in batch_species]
                    idx_right = [int(rng.choice(markers[right]["by_species"][label])) for label in batch_species]
                    z_left = torch.tensor(markers[left]["emb"][idx_left], device=args.device)
                    z_right = torch.tensor(markers[right]["emb"][idx_right], device=args.device)
                    val_losses.append(float(loss_fn(shared_head(z_left), shared_head(z_right), batch_species, tax, args).cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses)) if val_losses else np.nan}

        if (
            args.restore_best_retrieval
            and args.retrieval_eval_every > 0
            and (epoch == 1 or epoch % args.retrieval_eval_every == 0 or epoch == args.epochs)
        ):
            val_metric_rows = []
            for left, right in pairs:
                labels = pair_val.get((left, right), [])
                if not labels:
                    continue
                val_metric_rows.extend(evaluate_direction("val_checkpoint", left, right, labels, markers, shared_head, tax, args.device, f"triad_shared_{left}_to_{right}"))
                val_metric_rows.extend(evaluate_direction("val_checkpoint", right, left, labels, markers, shared_head, tax, args.device, f"triad_shared_{right}_to_{left}"))
            score_values = [
                float(metric["topk_accuracy_pct"])
                for metric in val_metric_rows
                if metric["rank"] in retrieval_selection_ranks and int(metric["k"]) == args.retrieval_selection_k
            ]
            retrieval_score = float(np.mean(score_values)) if score_values else float("nan")
            row["val_retrieval_score"] = retrieval_score
            logger.log(
                "epoch="
                f"{epoch} val_retrieval_score={retrieval_score:.4f} "
                f"pairs={','.join(f'{a}<->{b}' for a,b in pairs)} "
                f"ranks={','.join(retrieval_selection_ranks)} k={args.retrieval_selection_k}"
            )
            if np.isfinite(retrieval_score) and retrieval_score > best_retrieval_score:
                best_retrieval_score = retrieval_score
                best_retrieval_epoch = epoch
                best_state = {"shared_head": copy.deepcopy(shared_head.state_dict())}
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            logger.log(f"epoch={epoch} loss={row['loss']:.4f} val_loss={row['val_loss']:.4f}")

    restored_best_retrieval = False
    if args.restore_best_retrieval and best_state is not None:
        shared_head.load_state_dict(best_state["shared_head"])
        restored_best_retrieval = True
        logger.log(f"Restored best retrieval checkpoint epoch={best_retrieval_epoch} score={best_retrieval_score:.4f}")

    metric_rows = []
    for split_name, split_labels in splits.items():
        for left, right in pairs:
            labels = pair_species(markers, left, right, split_labels)
            if not labels:
                continue
            metric_rows.extend(evaluate_direction(split_name, left, right, labels, markers, None, tax, args.device, "frozen_nt_triad_space"))
            metric_rows.extend(evaluate_direction(split_name, left, right, labels, markers, shared_head, tax, args.device, f"triad_shared_{left}_to_{right}"))
            metric_rows.extend(evaluate_direction(split_name, right, left, labels, markers, None, tax, args.device, "frozen_nt_triad_space"))
            metric_rows.extend(evaluate_direction(split_name, right, left, labels, markers, shared_head, tax, args.device, f"triad_shared_{right}_to_{left}"))

    pd.DataFrame(history).to_csv(args.output_dir / "marker_mirror_triad_training_history.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "marker_mirror_triad_retrieval_metrics.csv", index=False)
    torch.save({"shared_head": shared_head.state_dict(), "args": vars(args)}, args.output_dir / "marker_mirror_triad_projection_head.pt")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "markers": marker_names,
        "pairs": pairs,
        "union_species": len(union_species),
        "splits": {key: len(value) for key, value in splits.items()},
        "pair_overlaps": {
            f"{left}<->{right}": {
                "total": len(pair_species(markers, left, right, union_species)),
                "train": len(pair_species(markers, left, right, splits["train"])),
                "val": len(pair_species(markers, left, right, splits["val"])),
                "test": len(pair_species(markers, left, right, splits["test"])),
            }
            for left, right in pairs
        },
        "restored_best_retrieval": restored_best_retrieval,
        "best_retrieval_epoch": best_retrieval_epoch,
        "best_retrieval_score": best_retrieval_score if math.isfinite(best_retrieval_score) else None,
        "claim_boundary": "Tri-marker shared-space prototype; candidate retrieval evidence only, not final assignment.",
    }
    (args.output_dir / "marker_mirror_triad_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log("Completed tri-marker MarkerMirror species-space prototype")


if __name__ == "__main__":
    main()
