#!/usr/bin/env python3
"""LoRA fine-tune for the MarkerMirror 12S->COI bridge.

This is the heavier follow-up to ``train_marker_mirror_bridge.py``.  The frozen
probe showed that projection heads can turn Nucleotide Transformer embeddings
into a useful cross-marker signal.  This script tests the next question: does a
small adapter on the foundation model improve held-out 12S-to-COI retrieval?
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_marker_mirror_bridge import (
    Logger,
    ProjectionHead,
    contrastive_loss,
    evaluate,
    flatten,
    load_species_json,
    load_taxonomy,
    sample_species_batch,
    split_species,
    taxonomy_soft_contrastive_loss,
)


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")
TOP_KS = (1, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument(
        "--coi-input-dir",
        type=Path,
        default=ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs" / "eval_c",
    )
    parser.add_argument(
        "--marker-input-dir",
        type=Path,
        default=ROOT / "data" / "edna" / "stalder_inputs" / "multisource",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "marker_mirror_bridge"
        / "nt_v2_50m_multisource_lora_taxonomy_soft_retrieval_best",
    )
    parser.add_argument("--max-coi-per-species", type=int, default=4)
    parser.add_argument("--max-marker-per-species", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-strategy", choices=("random", "taxonomy_hard"), default="taxonomy_hard")
    parser.add_argument("--loss-mode", choices=("hard_ce", "taxonomy_soft"), default="taxonomy_soft")
    parser.add_argument("--same-genus-weight", type=float, default=0.20)
    parser.add_argument("--same-species-weight", type=float, default=1.0)
    parser.add_argument("--same-family-weight", type=float, default=0.08)
    parser.add_argument("--same-order-weight", type=float, default=0.03)
    parser.add_argument("--val-steps", type=int, default=12)
    parser.add_argument("--retrieval-eval-every", type=int, default=4)
    parser.add_argument("--retrieval-selection-k", type=int, default=10)
    parser.add_argument("--retrieval-selection-ranks", default="genus,family,order")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="query,key,value")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def load_hf_lora(args: argparse.Namespace, logger: Logger) -> tuple[Any, Any, bool]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    logger.log(f"Loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    logger.log(f"Loading backbone {args.model}")
    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        masked_lm = False
    except Exception as exc:
        logger.log(f"AutoModel failed ({exc}); using AutoModelForMaskedLM")
        model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
        masked_lm = True

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.to(args.device)
    return tokenizer, model, masked_lm


def tokenize(tokenizer: Any, sequences: list[str], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    return {key: value.to(args.device) for key, value in encoded.items()}


def pooled_backbone(
    model: Any,
    encoded: dict[str, torch.Tensor],
    masked_lm: bool,
) -> torch.Tensor:
    if masked_lm:
        out = model(**encoded, output_hidden_states=True)
        hidden = out.hidden_states[-1]
    else:
        out = model(**encoded)
        hidden = getattr(out, "last_hidden_state", out[0])
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return nn.functional.normalize(pooled, dim=-1)


def encode_sequences(
    rows: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    masked_lm: bool,
    head: ProjectionHead,
    args: argparse.Namespace,
    logger: Logger,
    label: str,
) -> np.ndarray:
    outputs = []
    model.eval()
    head.eval()
    n_batches = math.ceil(len(rows) / args.eval_batch_size)
    with torch.no_grad():
        for start in range(0, len(rows), args.eval_batch_size):
            batch = rows["nucleotides"].iloc[start : start + args.eval_batch_size].tolist()
            pooled = pooled_backbone(model, tokenize(tokenizer, batch, args), masked_lm)
            outputs.append(head(pooled).cpu().numpy().astype("float32"))
            idx = start // args.eval_batch_size + 1
            if idx == 1 or idx == n_batches or idx % 25 == 0:
                logger.log(f"Encoded {label} batch {idx}/{n_batches}")
    return np.vstack(outputs)


def loss_for_mode(
    z12: torch.Tensor,
    zcoi: torch.Tensor,
    batch_species: list[str],
    tax: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.loss_mode == "taxonomy_soft":
        return taxonomy_soft_contrastive_loss(
            z12,
            zcoi,
            batch_species,
            tax,
            args.temperature,
            args.same_species_weight,
            args.same_genus_weight,
            args.same_family_weight,
            args.same_order_weight,
        )
    return contrastive_loss(z12, zcoi, args.temperature)


def cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def restore_state(module: nn.Module, state: dict[str, torch.Tensor], device: str) -> None:
    module.load_state_dict({key: value.to(device) for key, value in state.items()})


def retrieval_score(
    metric_rows: list[dict[str, Any]],
    ranks: list[str],
    k: int,
) -> float:
    values = [
        float(row["topk_accuracy_pct"])
        for row in metric_rows
        if str(row["rank"]) in ranks and int(row["k"]) == k and str(row["model"]) == "marker_mirror_lora"
    ]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_lora_bridge.log")
    logger.log(f"Arguments: {vars(args)}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    coi = load_species_json(args.coi_input_dir / "train_species_sequences.json", args.max_coi_per_species)
    marker = load_species_json(args.marker_input_dir / "train_species_sequences.json", args.max_marker_per_species)
    overlap = sorted([label for label in set(coi) & set(marker) if coi[label] and marker[label]])
    tax = load_taxonomy(args.marker_input_dir / "candidate_species.csv")
    tax.update(load_taxonomy(args.coi_input_dir / "candidate_species.csv"))
    splits = split_species(overlap, args.seed)
    logger.log(f"Overlap species={len(overlap)} train/val/test={[len(splits[k]) for k in ['train','val','test']]}")

    all_species = overlap
    coi_rows = flatten(coi, all_species, "coi")
    marker_rows = flatten(marker, all_species, "marker")
    coi_by_species: dict[str, list[int]] = defaultdict(list)
    marker_by_species: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(coi_rows["tree_label"]):
        coi_by_species[str(label)].append(i)
    for i, label in enumerate(marker_rows["tree_label"]):
        marker_by_species[str(label)].append(i)

    tokenizer, model, masked_lm = load_hf_lora(args, logger)
    with torch.no_grad():
        probe = pooled_backbone(
            model,
            tokenize(tokenizer, [str(coi_rows["nucleotides"].iloc[0])], args),
            masked_lm,
        )
    dim = int(probe.shape[1])
    coi_head = ProjectionHead(dim, args.hidden_dim, args.projection_dim).to(args.device)
    marker_head = ProjectionHead(dim, args.hidden_dim, args.projection_dim).to(args.device)

    head_params = list(coi_head.parameters()) + list(marker_head.parameters())
    model_params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": model_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    train_species = splits["train"]
    val_species = splits["val"]
    rng = np.random.default_rng(args.seed)
    selection_ranks = [rank.strip() for rank in args.retrieval_selection_ranks.split(",") if rank.strip()]
    history = []
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        coi_head.train()
        marker_head.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch_species = sample_species_batch(rng, train_species, tax, args.batch_size, args.batch_strategy)
            coi_idx = [rng.choice(coi_by_species[label]) for label in batch_species]
            marker_idx = [rng.choice(marker_by_species[label]) for label in batch_species]
            coi_seq = coi_rows["nucleotides"].iloc[coi_idx].tolist()
            marker_seq = marker_rows["nucleotides"].iloc[marker_idx].tolist()
            zc = coi_head(pooled_backbone(model, tokenize(tokenizer, coi_seq, args), masked_lm))
            zm = marker_head(pooled_backbone(model, tokenize(tokenizer, marker_seq, args), masked_lm))
            loss = loss_for_mode(zm, zc, batch_species, tax, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params + head_params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_losses = []
        model.eval()
        coi_head.eval()
        marker_head.eval()
        if val_species and args.val_steps > 0:
            with torch.no_grad():
                for _ in range(args.val_steps):
                    batch_species = sample_species_batch(rng, val_species, tax, args.batch_size, args.batch_strategy)
                    coi_idx = [rng.choice(coi_by_species[label]) for label in batch_species]
                    marker_idx = [rng.choice(marker_by_species[label]) for label in batch_species]
                    coi_seq = coi_rows["nucleotides"].iloc[coi_idx].tolist()
                    marker_seq = marker_rows["nucleotides"].iloc[marker_idx].tolist()
                    zc = coi_head(pooled_backbone(model, tokenize(tokenizer, coi_seq, args), masked_lm))
                    zm = marker_head(pooled_backbone(model, tokenize(tokenizer, marker_seq, args), masked_lm))
                    val_losses.append(float(loss_for_mode(zm, zc, batch_species, tax, args).cpu()))

        row: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_loss": float(np.mean(val_losses)) if val_losses else np.nan,
        }
        if epoch == 1 or epoch % args.retrieval_eval_every == 0 or epoch == args.epochs:
            val_rows = marker_rows[marker_rows["tree_label"].isin(val_species)].reset_index(drop=True)
            val_marker_emb = encode_sequences(val_rows, tokenizer, model, masked_lm, marker_head, args, logger, "val_marker")
            coi_emb = encode_sequences(coi_rows, tokenizer, model, masked_lm, coi_head, args, logger, "coi_reference")
            metrics = evaluate(
                "val_checkpoint",
                val_rows,
                coi_rows,
                val_marker_emb,
                coi_emb,
                None,
                None,
                tax,
                args.device,
                "marker_mirror_lora",
            )
            score = retrieval_score(metrics, selection_ranks, args.retrieval_selection_k)
            row["val_retrieval_score"] = score
            logger.log(
                f"epoch={epoch} val_retrieval_score={score:.4f} "
                f"ranks={','.join(selection_ranks)} k={args.retrieval_selection_k}"
            )
            if np.isfinite(score) and score > best_score:
                from peft import get_peft_model_state_dict

                best_score = score
                best_epoch = epoch
                best_state = {
                    "lora": copy.deepcopy({k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}),
                    "coi_head": cpu_state(coi_head),
                    "marker_head": cpu_state(marker_head),
                }
        history.append(row)
        logger.log(f"epoch={epoch} loss={row['loss']:.4f} val_loss={row['val_loss']:.4f}")

    restored_best_retrieval = False
    if best_state is not None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, {k: v.to(args.device) for k, v in best_state["lora"].items()})
        restore_state(coi_head, best_state["coi_head"], args.device)
        restore_state(marker_head, best_state["marker_head"], args.device)
        restored_best_retrieval = True
        logger.log(f"Restored best retrieval checkpoint epoch={best_epoch} score={best_score:.4f}")

    coi_emb = encode_sequences(coi_rows, tokenizer, model, masked_lm, coi_head, args, logger, "coi_reference_final")
    metric_rows = []
    for split_name, labels in splits.items():
        query_rows = marker_rows[marker_rows["tree_label"].isin(labels)].reset_index(drop=True)
        marker_emb = encode_sequences(query_rows, tokenizer, model, masked_lm, marker_head, args, logger, f"{split_name}_marker_final")
        metric_rows.extend(
            evaluate(
                split_name,
                query_rows,
                coi_rows,
                marker_emb,
                coi_emb,
                None,
                None,
                tax,
                args.device,
                "marker_mirror_lora",
            )
        )

    pd.DataFrame(history).to_csv(args.output_dir / "marker_mirror_lora_training_history.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "marker_mirror_retrieval_metrics.csv", index=False)
    model.save_pretrained(args.output_dir / "lora_adapter")
    torch.save(
        {"coi_head": coi_head.state_dict(), "marker_head": marker_head.state_dict(), "args": vars(args)},
        args.output_dir / "marker_mirror_lora_projection_heads.pt",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "overlap_species": len(overlap),
        "splits": {key: len(value) for key, value in splits.items()},
        "best_retrieval_epoch": best_epoch,
        "best_retrieval_score": best_score if math.isfinite(best_score) else None,
        "restored_best_retrieval": restored_best_retrieval,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "claim_boundary": "LoRA feasibility run for cross-marker bridge; compare against frozen projection-head MarkerMirror before making claims.",
    }
    (args.output_dir / "marker_mirror_lora_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log("Completed LoRA MarkerMirror bridge run")


if __name__ == "__main__":
    main()
