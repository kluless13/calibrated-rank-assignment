#!/usr/bin/env python3
"""Probe a Hugging Face DNA foundation model as a barcode candidate generator.

This is a compact reconnaissance script, not a production benchmark. It embeds
one representative reference barcode per tree species plus sampled query
barcodes, retrieves nearest references by cosine similarity, and reports
species/genus/family/order top-k support. The goal is to decide whether a
pretrained DNA foundation model is worth integrating into the Paper 1 evidence
schema as another candidate generator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "data" / "phylo" / "fish_tree_clean_phylo_inputs"
RANKS = ("species", "genus", "family", "order")
TOP_KS = (1, 5, 10)


class Logger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
        help="Hugging Face model id.",
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["seen_test", "eval_c", "unseen_genera"],
        choices=["seen_test", "eval_c", "unseen_genera"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results"
        / "paper1_phylo_calibrated_assignment"
        / "hf_foundation_probe"
        / "nt_v2_50m_frozen",
    )
    parser.add_argument("--max-queries-per-split", type=int, default=1200)
    parser.add_argument("--max-reference-species", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean_sequence(value: object) -> str:
    text = str(value or "").upper()
    text = re.sub("[^ACGT]", "A", text)
    return text


def sequence_quality(seq: str) -> tuple[int, int]:
    raw = str(seq or "").upper()
    acgt = sum(1 for char in raw if char in {"A", "C", "G", "T"})
    return acgt, len(raw)


def choose_representative(sequences: list[str]) -> str:
    if not sequences:
        return ""
    return max(sequences, key=sequence_quality)


def load_reference_rows(input_root: Path, split: str, max_species: int, seed: int) -> pd.DataFrame:
    split_dir = input_root / split
    with (split_dir / "train_species_sequences.json").open("r", encoding="utf-8") as handle:
        species_sequences: dict[str, list[str]] = json.load(handle)
    candidates = pd.read_csv(split_dir / "candidate_species.csv")
    candidates["tree_label"] = candidates["tree_label"].astype(str)
    meta = candidates.set_index("tree_label").to_dict(orient="index")
    labels = sorted(species_sequences)
    if max_species and max_species < len(labels):
        rng = np.random.default_rng(seed)
        labels = sorted(rng.choice(labels, size=max_species, replace=False).tolist())
    rows: list[dict[str, Any]] = []
    for label in labels:
        seq = choose_representative(species_sequences.get(label, []))
        if not seq:
            continue
        info = meta.get(label, {})
        rows.append(
            {
                "tree_label": label,
                "species": str(info.get("species_name") or label.replace("_", " ")),
                "genus": str(info.get("genus_name") or label.split("_")[0]),
                "family": str(info.get("family_name") or ""),
                "order": str(info.get("order_name") or ""),
                "nucleotides": clean_sequence(seq),
            }
        )
    return pd.DataFrame(rows)


def load_query_rows(input_root: Path, split: str, max_queries: int, seed: int) -> pd.DataFrame:
    queries = pd.read_csv(input_root / split / "zero_shot_queries.csv")
    if max_queries and max_queries < len(queries):
        queries = queries.sample(n=max_queries, random_state=seed).sort_index()
    queries = queries.rename(
        columns={
            "tree_label": "true_tree_label",
            "species_name": "true_species",
            "genus_name": "true_genus",
            "family_name": "true_family",
            "order_name": "true_order",
        }
    )
    keep = [
        "processid",
        "true_tree_label",
        "true_species",
        "true_genus",
        "true_family",
        "true_order",
        "nucleotides",
    ]
    queries = queries[keep].copy()
    queries["nucleotides"] = queries["nucleotides"].map(clean_sequence)
    return queries


def load_hf_model(model_name: str, device_arg: str, logger: Logger):
    import torch
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if device_arg == "auto" and torch.cuda.is_available() else device_arg
    if device == "auto":
        device = "cpu"
    logger.log(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.log(f"Loading model: {model_name} on {device}")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        masked_lm = False
    except Exception as exc:
        logger.log(f"AutoModel failed ({exc}); trying AutoModelForMaskedLM")
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        masked_lm = True
    model.eval()
    model.to(device)
    return tokenizer, model, device, masked_lm


def embed_sequences(
    sequences: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    masked_lm: bool,
    batch_size: int,
    max_length: int,
    logger: Logger,
) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    n_batches = math.ceil(len(sequences) / batch_size)
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if masked_lm:
                output = model(**encoded, output_hidden_states=True)
                hidden = output.hidden_states[-1]
            else:
                output = model(**encoded)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    hidden = output[0]
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            vectors.append(pooled.detach().cpu().numpy().astype("float32"))
            batch_idx = (start // batch_size) + 1
            if batch_idx == 1 or batch_idx == n_batches or batch_idx % 25 == 0:
                logger.log(f"Embedded batch {batch_idx}/{n_batches}")
    return np.vstack(vectors)


def evaluate_split(
    split: str,
    refs: pd.DataFrame,
    queries: pd.DataFrame,
    ref_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    output_dir: Path,
    logger: Logger,
) -> list[dict[str, Any]]:
    logger.log(f"Scoring split={split} refs={len(refs)} queries={len(queries)}")
    sims = query_embeddings @ ref_embeddings.T
    max_k = max(TOP_KS)
    top_idx = np.argpartition(-sims, kth=min(max_k, sims.shape[1] - 1), axis=1)[:, :max_k]
    ordered = np.take_along_axis(top_idx, np.argsort(-np.take_along_axis(sims, top_idx, axis=1), axis=1), axis=1)

    pred_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    rank_hits = {rank: {k: 0 for k in TOP_KS} for rank in RANKS}

    ref_records = refs.reset_index(drop=True)
    query_records = queries.reset_index(drop=True)
    for row_i, indices in enumerate(ordered):
        query = query_records.iloc[row_i]
        top_refs = ref_records.iloc[indices]
        pred_row: dict[str, Any] = {
            "split": split,
            "processid": query["processid"],
            "true_tree_label": query["true_tree_label"],
        }
        for k in TOP_KS:
            subset = top_refs.iloc[:k]
            rank_hits["species"][k] += int(str(query["true_tree_label"]) in set(subset["tree_label"].astype(str)))
            for rank in ("genus", "family", "order"):
                true_value = str(query[f"true_{rank}"])
                rank_hits[rank][k] += int(true_value in set(subset[rank].astype(str)))
        for k in TOP_KS:
            pred_row[f"top{k}_tree_labels"] = json.dumps(top_refs.iloc[:k]["tree_label"].astype(str).tolist())
        pred_rows.append(pred_row)

    for rank in RANKS:
        for k in TOP_KS:
            metric_rows.append(
                {
                    "split": split,
                    "model": "hf_dna_foundation_frozen",
                    "rank": rank,
                    "k": k,
                    "n_query": len(queries),
                    "topk_accuracy_pct": 100.0 * rank_hits[rank][k] / max(len(queries), 1),
                }
            )
    pd.DataFrame(pred_rows).to_csv(output_dir / f"{split}_hf_foundation_predictions.csv", index=False)
    return metric_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "hf_foundation_probe.log")
    logger.log("Starting HF DNA foundation retrieval probe")
    logger.log(f"Arguments: {vars(args)}")

    tokenizer, model, device, masked_lm = load_hf_model(args.model, args.device, logger)
    all_metrics: list[dict[str, Any]] = []
    manifests: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "splits": args.splits,
        "max_queries_per_split": args.max_queries_per_split,
        "max_reference_species": args.max_reference_species,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
    }

    for split in args.splits:
        logger.log(f"Loading split {split}")
        refs = load_reference_rows(args.input_root, split, args.max_reference_species, args.seed)
        queries = load_query_rows(args.input_root, split, args.max_queries_per_split, args.seed)
        logger.log(f"Split {split}: {len(refs)} reference species, {len(queries)} queries")
        ref_embeddings = embed_sequences(
            refs["nucleotides"].tolist(),
            tokenizer,
            model,
            device,
            masked_lm,
            args.batch_size,
            args.max_length,
            logger,
        )
        query_embeddings = embed_sequences(
            queries["nucleotides"].tolist(),
            tokenizer,
            model,
            device,
            masked_lm,
            args.batch_size,
            args.max_length,
            logger,
        )
        metrics = evaluate_split(split, refs, queries, ref_embeddings, query_embeddings, args.output_dir, logger)
        all_metrics.extend(metrics)
        manifests[split] = {"n_refs": int(len(refs)), "n_queries": int(len(queries))}

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(args.output_dir / "hf_foundation_retrieval_metrics.csv", index=False)
    with (args.output_dir / "hf_foundation_probe_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2)
    logger.log("Completed HF DNA foundation retrieval probe")


if __name__ == "__main__":
    main()
