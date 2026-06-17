#!/usr/bin/env python3
"""Train a shared 12S/16S MarkerMirror species-space prototype.

This is the follow-up to the directional MarkerMirror probes.  Instead of
learning one projection head per marker direction, this script learns one shared
projection head on top of frozen DNA foundation embeddings and evaluates both
marker directions:

    marker A query -> marker B species prototypes
    marker B query -> marker A species prototypes

The goal is to test whether 12S and 16S can live in one ribosomal species
space, not to produce a final calibrated species assignment system.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument(
        "--marker-a-input-dir",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource",
        help="First marker input directory, e.g. 12S multisource.",
    )
    parser.add_argument(
        "--marker-b-input-dir",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "16s_multisource",
        help="Second marker input directory, e.g. 16S multisource.",
    )
    parser.add_argument("--marker-a-name", default="12S")
    parser.add_argument("--marker-b-name", default="16S")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best",
    )
    parser.add_argument("--max-a-per-species", type=int, default=4)
    parser.add_argument("--max-b-per-species", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
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
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def loss_fn(
    za: torch.Tensor,
    zb: torch.Tensor,
    batch_species: list[str],
    tax: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.loss_mode == "taxonomy_soft":
        return taxonomy_soft_contrastive_loss(
            za,
            zb,
            batch_species,
            tax,
            args.temperature,
            args.same_species_weight,
            args.same_genus_weight,
            args.same_family_weight,
            args.same_order_weight,
        )
    return contrastive_loss(za, zb, args.temperature)


def evaluate_direction(
    split_name: str,
    query_rows: pd.DataFrame,
    reference_rows: pd.DataFrame,
    query_emb: np.ndarray,
    reference_emb: np.ndarray,
    shared_head: ProjectionHead,
    tax: dict[str, dict[str, str]],
    device: str,
    model_name: str,
) -> list[dict[str, Any]]:
    return evaluate(
        split_name,
        query_rows,
        reference_rows,
        query_emb,
        reference_emb,
        shared_head,
        shared_head,
        tax,
        device,
        model_name,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_shared_space.log")
    logger.log(f"Arguments: {vars(args)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    marker_a = load_species_json(args.marker_a_input_dir / "train_species_sequences.json", args.max_a_per_species)
    marker_b = load_species_json(args.marker_b_input_dir / "train_species_sequences.json", args.max_b_per_species)
    overlap = sorted([label for label in set(marker_a) & set(marker_b) if marker_a[label] and marker_b[label]])
    if not overlap:
        raise ValueError("No overlap species between marker inputs")
    tax = load_taxonomy(args.marker_a_input_dir / "candidate_species.csv")
    tax.update(load_taxonomy(args.marker_b_input_dir / "candidate_species.csv"))
    splits = split_species(overlap, args.seed)
    logger.log(f"Overlap species={len(overlap)} train/val/test={[len(splits[k]) for k in ['train','val','test']]}")

    rows_a = flatten(marker_a, overlap, args.marker_a_name)
    rows_b = flatten(marker_b, overlap, args.marker_b_name)
    tokenizer, model, masked_lm = load_hf(args.model, args.device, logger)
    emb_a = embed_sequences(
        rows_a["nucleotides"].map(clean_sequence).tolist(),
        tokenizer,
        model,
        masked_lm,
        args.device,
        args.embed_batch_size,
        args.max_length,
        logger,
        args.marker_a_name,
    )
    emb_b = embed_sequences(
        rows_b["nucleotides"].map(clean_sequence).tolist(),
        tokenizer,
        model,
        masked_lm,
        args.device,
        args.embed_batch_size,
        args.max_length,
        logger,
        args.marker_b_name,
    )

    dim = emb_a.shape[1]
    shared_head = ProjectionHead(dim, args.hidden_dim, args.projection_dim).to(args.device)
    opt = torch.optim.AdamW(shared_head.parameters(), lr=args.lr, weight_decay=1e-4)

    a_by_species: dict[str, list[int]] = defaultdict(list)
    b_by_species: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(rows_a["tree_label"]):
        a_by_species[str(label)].append(i)
    for i, label in enumerate(rows_b["tree_label"]):
        b_by_species[str(label)].append(i)

    retrieval_selection_ranks = [rank.strip() for rank in args.retrieval_selection_ranks.split(",") if rank.strip()]
    best_retrieval_score = -math.inf
    best_retrieval_epoch = 0
    best_state: dict[str, Any] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        shared_head.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch_species = sample_species_batch(rng, splits["train"], tax, args.batch_size, args.batch_strategy)
            idx_a = [int(rng.choice(a_by_species[label])) for label in batch_species]
            idx_b = [int(rng.choice(b_by_species[label])) for label in batch_species]
            za = torch.tensor(emb_a[idx_a], device=args.device)
            zb = torch.tensor(emb_b[idx_b], device=args.device)
            loss = loss_fn(shared_head(za), shared_head(zb), batch_species, tax, args)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_losses = []
        if splits["val"] and args.val_steps > 0:
            shared_head.eval()
            with torch.no_grad():
                for _ in range(args.val_steps):
                    batch_species = sample_species_batch(rng, splits["val"], tax, args.batch_size, args.batch_strategy)
                    idx_a = [int(rng.choice(a_by_species[label])) for label in batch_species]
                    idx_b = [int(rng.choice(b_by_species[label])) for label in batch_species]
                    za = torch.tensor(emb_a[idx_a], device=args.device)
                    zb = torch.tensor(emb_b[idx_b], device=args.device)
                    val_losses.append(float(loss_fn(shared_head(za), shared_head(zb), batch_species, tax, args).cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses)) if val_losses else np.nan}

        if (
            args.restore_best_retrieval
            and args.retrieval_eval_every > 0
            and (epoch == 1 or epoch % args.retrieval_eval_every == 0 or epoch == args.epochs)
        ):
            mask_a_val = rows_a["tree_label"].isin(splits["val"]).to_numpy()
            mask_b_val = rows_b["tree_label"].isin(splits["val"]).to_numpy()
            val_metric_rows = []
            val_metric_rows.extend(
                evaluate_direction(
                    "val_checkpoint",
                    rows_a[mask_a_val].reset_index(drop=True),
                    rows_b,
                    emb_a[mask_a_val],
                    emb_b,
                    shared_head,
                    tax,
                    args.device,
                    f"shared_space_{args.marker_a_name}_to_{args.marker_b_name}",
                )
            )
            val_metric_rows.extend(
                evaluate_direction(
                    "val_checkpoint",
                    rows_b[mask_b_val].reset_index(drop=True),
                    rows_a,
                    emb_b[mask_b_val],
                    emb_a,
                    shared_head,
                    tax,
                    args.device,
                    f"shared_space_{args.marker_b_name}_to_{args.marker_a_name}",
                )
            )
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
                f"directions={args.marker_a_name}->{args.marker_b_name},{args.marker_b_name}->{args.marker_a_name} "
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
    for split_name, labels in splits.items():
        mask_a = rows_a["tree_label"].isin(labels).to_numpy()
        mask_b = rows_b["tree_label"].isin(labels).to_numpy()
        metric_rows.extend(
            evaluate(
                f"{split_name}_{args.marker_a_name}_to_{args.marker_b_name}",
                rows_a[mask_a].reset_index(drop=True),
                rows_b,
                emb_a[mask_a],
                emb_b,
                None,
                None,
                tax,
                args.device,
                "frozen_nt_shared_space",
            )
        )
        metric_rows.extend(
            evaluate_direction(
                f"{split_name}_{args.marker_a_name}_to_{args.marker_b_name}",
                rows_a[mask_a].reset_index(drop=True),
                rows_b,
                emb_a[mask_a],
                emb_b,
                shared_head,
                tax,
                args.device,
                f"shared_space_{args.marker_a_name}_to_{args.marker_b_name}",
            )
        )
        metric_rows.extend(
            evaluate(
                f"{split_name}_{args.marker_b_name}_to_{args.marker_a_name}",
                rows_b[mask_b].reset_index(drop=True),
                rows_a,
                emb_b[mask_b],
                emb_a,
                None,
                None,
                tax,
                args.device,
                "frozen_nt_shared_space",
            )
        )
        metric_rows.extend(
            evaluate_direction(
                f"{split_name}_{args.marker_b_name}_to_{args.marker_a_name}",
                rows_b[mask_b].reset_index(drop=True),
                rows_a,
                emb_b[mask_b],
                emb_a,
                shared_head,
                tax,
                args.device,
                f"shared_space_{args.marker_b_name}_to_{args.marker_a_name}",
            )
        )

    pd.DataFrame(history).to_csv(args.output_dir / "marker_mirror_shared_training_history.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "marker_mirror_shared_retrieval_metrics.csv", index=False)
    torch.save(
        {"shared_head": shared_head.state_dict(), "args": vars(args)},
        args.output_dir / "marker_mirror_shared_projection_head.pt",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "marker_a": args.marker_a_name,
        "marker_b": args.marker_b_name,
        "overlap_species": len(overlap),
        "splits": {key: len(value) for key, value in splits.items()},
        "restored_best_retrieval": restored_best_retrieval,
        "best_retrieval_epoch": best_retrieval_epoch,
        "best_retrieval_score": best_retrieval_score if math.isfinite(best_retrieval_score) else None,
        "claim_boundary": "Shared-head 12S/16S MarkerMirror prototype; candidate retrieval evidence only, not final assignment.",
    }
    (args.output_dir / "marker_mirror_shared_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log("Completed shared MarkerMirror species-space prototype")


if __name__ == "__main__":
    main()
