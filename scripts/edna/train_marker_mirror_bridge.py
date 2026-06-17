#!/usr/bin/env python3
"""Train a cross-marker COI<->12S bridge on frozen DNA foundation embeddings.

This is a fast "breakthrough probe" for MarkerMirror/BarcodeBridge:

    12S/eDNA fragment -> shared marker embedding -> nearest COI barcode species

The model learns small projection heads on top of frozen Hugging Face DNA
foundation embeddings. It is intentionally lightweight so we can decide whether
cross-marker marker-bridging is worth a full LoRA/backbone fine-tune.
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


ROOT = Path(__file__).resolve().parents[2]
RANKS = ("species", "genus", "family", "order")
TOP_KS = (1, 5, 10)


class Logger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


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
        / "nt_v2_50m_multisource",
    )
    parser.add_argument("--max-coi-per-species", type=int, default=4)
    parser.add_argument("--max-marker-per-species", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--batch-strategy",
        choices=("random", "taxonomy_hard"),
        default="random",
        help="Training batch sampler. taxonomy_hard packs close relatives into the same contrastive batch.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("hard_ce", "taxonomy_soft", "tree_soft"),
        default="hard_ce",
        help="Contrastive target. taxonomy_soft gives close relatives some probability mass.",
    )
    parser.add_argument("--tree-file", type=Path, default=ROOT / "data" / "phylo" / "actinopt_12k_treePL.tre")
    parser.add_argument(
        "--tree-soft-scale",
        type=float,
        default=0.0,
        help="Patristic distance scale for tree_soft loss. <=0 estimates from train pairs.",
    )
    parser.add_argument("--tree-soft-quantile", type=float, default=0.10)
    parser.add_argument("--tree-soft-sample-pairs", type=int, default=50000)
    parser.add_argument("--same-genus-weight", type=float, default=0.20)
    parser.add_argument("--same-species-weight", type=float, default=1.0)
    parser.add_argument("--same-family-weight", type=float, default=0.08)
    parser.add_argument("--same-order-weight", type=float, default=0.03)
    parser.add_argument(
        "--sequences-per-species-per-batch",
        type=int,
        default=1,
        help="Use >1 to create multi-positive batches with several COI/12S sequences per sampled species.",
    )
    parser.add_argument("--val-steps", type=int, default=20)
    parser.add_argument(
        "--restore-best-val",
        action="store_true",
        help="Restore projection heads from the epoch with the lowest validation loss before final retrieval evaluation.",
    )
    parser.add_argument(
        "--restore-best-retrieval",
        action="store_true",
        help="Restore projection heads from the epoch with the best validation retrieval score.",
    )
    parser.add_argument("--retrieval-eval-every", type=int, default=0)
    parser.add_argument("--retrieval-selection-k", type=int, default=10)
    parser.add_argument("--retrieval-selection-ranks", default="genus,family,order")
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def clean_sequence(value: object) -> str:
    text = str(value or "").upper()
    return re.sub("[^ACGT]", "A", text)


def choose_sequences(sequences: list[str], max_n: int) -> list[str]:
    cleaned = [clean_sequence(seq) for seq in sequences if str(seq or "").strip()]
    cleaned = sorted(set(cleaned), key=lambda seq: (-sum(c in "ACGT" for c in seq), -len(seq), seq))
    return cleaned[:max_n]


def load_species_json(path: Path, max_per_species: int) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, list[str]] = json.load(handle)
    return {label: choose_sequences(seqs, max_per_species) for label, seqs in raw.items()}


def load_taxonomy(candidate_csv: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_csv(candidate_csv)
    frame["tree_label"] = frame["tree_label"].astype(str)
    tax = {}
    for _, row in frame.iterrows():
        label = str(row["tree_label"])
        tax[label] = {
            "species": str(row.get("species_name") or label.replace("_", " ")),
            "genus": str(row.get("genus_name") or label.split("_")[0]),
            "family": str(row.get("family_name") or ""),
            "order": str(row.get("order_name") or ""),
        }
    return tax


def canonical_tree_label(label: object) -> str:
    text = str(label or "").strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text.replace(" ", "_")


def load_tree_distance_cache(tree_file: Path) -> dict[str, Any]:
    import dendropy

    tree = dendropy.Tree.get(path=str(tree_file), schema="newick")
    taxon_nodes: dict[str, Any] = {}
    for node in tree.leaf_node_iter():
        if node.taxon is None or not node.taxon.label:
            continue
        label = canonical_tree_label(node.taxon.label)
        if label:
            taxon_nodes[label] = node
    depths = {tree.seed_node: 0.0}
    stack = [tree.seed_node]
    while stack:
        node = stack.pop()
        for child in node.child_node_iter():
            depths[child] = depths[node] + float(child.edge_length or 0.0)
            stack.append(child)

    def ancestors(node: Any) -> list[Any]:
        out = []
        current = node
        while current is not None:
            out.append(current)
            current = current.parent_node
        return out

    ancestor_lists = {label: ancestors(node) for label, node in taxon_nodes.items()}
    ancestor_sets = {label: set(nodes) for label, nodes in ancestor_lists.items()}
    return {
        "taxon_nodes": taxon_nodes,
        "depths": depths,
        "ancestor_lists": ancestor_lists,
        "ancestor_sets": ancestor_sets,
        "distance_cache": {},
    }


def tree_distance(label_a: object, label_b: object, tree_cache: dict[str, Any] | None) -> float:
    if tree_cache is None:
        return float("nan")
    a = canonical_tree_label(label_a)
    b = canonical_tree_label(label_b)
    taxon_nodes = tree_cache["taxon_nodes"]
    if not a or not b or a not in taxon_nodes or b not in taxon_nodes:
        return float("nan")
    if a == b:
        return 0.0
    key = tuple(sorted((a, b)))
    cache = tree_cache["distance_cache"]
    if key in cache:
        return float(cache[key])
    ancestor_lists = tree_cache["ancestor_lists"]
    ancestor_sets = tree_cache["ancestor_sets"]
    depths = tree_cache["depths"]
    for node in ancestor_lists[a]:
        if node in ancestor_sets[b]:
            dist = float(depths[taxon_nodes[a]] + depths[taxon_nodes[b]] - 2 * depths[node])
            cache[key] = dist
            return dist
    cache[key] = float("nan")
    return float("nan")


def estimate_tree_soft_scale(
    species: list[str],
    tree_cache: dict[str, Any] | None,
    rng: np.random.Generator,
    n_pairs: int,
    quantile: float,
) -> float:
    if tree_cache is None or len(species) < 2:
        return 1.0
    values = []
    labels = np.array(species)
    for _ in range(n_pairs):
        left, right = rng.choice(labels, size=2, replace=False)
        dist = tree_distance(str(left), str(right), tree_cache)
        if np.isfinite(dist) and dist > 0:
            values.append(dist)
    if not values:
        return 1.0
    q = float(np.quantile(np.asarray(values, dtype=float), min(max(quantile, 0.01), 0.99)))
    return max(q, 1e-6)


def split_species(labels: list[str], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    labels = np.array(sorted(labels))
    rng.shuffle(labels)
    n = len(labels)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    return {
        "train": sorted(labels[:n_train].tolist()),
        "val": sorted(labels[n_train : n_train + n_val].tolist()),
        "test": sorted(labels[n_train + n_val :].tolist()),
    }


def load_hf(model_name: str, device: str, logger: Logger):
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    logger.log(f"Loading tokenizer {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.log(f"Loading model {model_name}")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        masked_lm = False
    except Exception as exc:
        logger.log(f"AutoModel failed ({exc}); using AutoModelForMaskedLM")
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        masked_lm = True
    model.eval().to(device)
    return tokenizer, model, masked_lm


def embed_sequences(
    sequences: list[str],
    tokenizer: Any,
    model: Any,
    masked_lm: bool,
    device: str,
    batch_size: int,
    max_length: int,
    logger: Logger,
    label: str,
) -> np.ndarray:
    vectors = []
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
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if masked_lm:
                out = model(**encoded, output_hidden_states=True)
                hidden = out.hidden_states[-1]
            else:
                out = model(**encoded)
                hidden = getattr(out, "last_hidden_state", out[0])
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
            idx = start // batch_size + 1
            if idx == 1 or idx == n_batches or idx % 25 == 0:
                logger.log(f"Embedded {label} batch {idx}/{n_batches}")
    return np.vstack(vectors)


def flatten(species_to_sequences: dict[str, list[str]], species: list[str], marker: str) -> pd.DataFrame:
    rows = []
    for label in species:
        for idx, seq in enumerate(species_to_sequences[label]):
            rows.append({"tree_label": label, "seq_index": idx, "marker": marker, "nucleotides": seq})
    return pd.DataFrame(rows)


def contrastive_loss(z12: torch.Tensor, zcoi: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z12 @ zcoi.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.T, labels))


def taxonomy_soft_targets(
    batch_species: list[str],
    tax: dict[str, dict[str, str]],
    device: str,
    same_species_weight: float,
    same_genus_weight: float,
    same_family_weight: float,
    same_order_weight: float,
) -> torch.Tensor:
    n = len(batch_species)
    target = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i, left in enumerate(batch_species):
        left_tax = tax.get(left, {})
        for j, right in enumerate(batch_species):
            if i == j:
                target[i, j] = 1.0
                continue
            if left == right:
                target[i, j] = max(target[i, j], same_species_weight)
            right_tax = tax.get(right, {})
            if left_tax.get("genus") and left_tax.get("genus") == right_tax.get("genus"):
                target[i, j] = max(target[i, j], same_genus_weight)
            if left_tax.get("family") and left_tax.get("family") == right_tax.get("family"):
                target[i, j] = max(target[i, j], same_family_weight)
            if left_tax.get("order") and left_tax.get("order") == right_tax.get("order"):
                target[i, j] = max(target[i, j], same_order_weight)
    return target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)


def taxonomy_soft_contrastive_loss(
    z12: torch.Tensor,
    zcoi: torch.Tensor,
    batch_species: list[str],
    tax: dict[str, dict[str, str]],
    temperature: float,
    same_species_weight: float,
    same_genus_weight: float,
    same_family_weight: float,
    same_order_weight: float,
) -> torch.Tensor:
    logits = z12 @ zcoi.T / temperature
    target = taxonomy_soft_targets(
        batch_species,
        tax,
        str(z12.device),
        same_species_weight,
        same_genus_weight,
        same_family_weight,
        same_order_weight,
    )
    loss_forward = -(target * nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_reverse = -(target.T * nn.functional.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (loss_forward + loss_reverse)


def tree_soft_targets(
    batch_species: list[str],
    tree_cache: dict[str, Any] | None,
    tree_soft_scale: float,
    device: str,
) -> torch.Tensor:
    n = len(batch_species)
    target = torch.zeros((n, n), dtype=torch.float32, device=device)
    for i, left in enumerate(batch_species):
        for j, right in enumerate(batch_species):
            if i == j:
                target[i, j] = 1.0
                continue
            dist = tree_distance(left, right, tree_cache)
            if np.isfinite(dist):
                target[i, j] = float(math.exp(-dist / max(tree_soft_scale, 1e-8)))
    return target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)


def tree_soft_contrastive_loss(
    z12: torch.Tensor,
    zcoi: torch.Tensor,
    batch_species: list[str],
    tree_cache: dict[str, Any] | None,
    tree_soft_scale: float,
    temperature: float,
) -> torch.Tensor:
    logits = z12 @ zcoi.T / temperature
    target = tree_soft_targets(batch_species, tree_cache, tree_soft_scale, str(z12.device))
    loss_forward = -(target * nn.functional.log_softmax(logits, dim=1)).sum(dim=1).mean()
    loss_reverse = -(target.T * nn.functional.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (loss_forward + loss_reverse)


def marker_mirror_loss(
    z12: torch.Tensor,
    zcoi: torch.Tensor,
    batch_species: list[str],
    tax: dict[str, dict[str, str]],
    args: argparse.Namespace,
    tree_cache: dict[str, Any] | None,
    tree_soft_scale: float,
) -> torch.Tensor:
    if args.loss_mode == "tree_soft":
        return tree_soft_contrastive_loss(
            z12,
            zcoi,
            batch_species,
            tree_cache,
            tree_soft_scale,
            args.temperature,
        )
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


def sample_species_batch(
    rng: np.random.Generator,
    species: list[str],
    tax: dict[str, dict[str, str]],
    batch_size: int,
    strategy: str,
) -> list[str]:
    if not species:
        return []
    target = min(batch_size, len(species))
    if strategy == "random":
        return rng.choice(species, size=target, replace=False).tolist()

    family_groups: dict[str, list[str]] = defaultdict(list)
    order_groups: dict[str, list[str]] = defaultdict(list)
    species_set = set(species)
    for label in species:
        row = tax.get(label, {})
        family = str(row.get("family") or "")
        order = str(row.get("order") or "")
        if family:
            family_groups[family].append(label)
        if order:
            order_groups[order].append(label)

    groups = [members for members in family_groups.values() if len(members) >= 2]
    if not groups:
        groups = [members for members in order_groups.values() if len(members) >= 2]
    if not groups:
        return rng.choice(species, size=target, replace=False).tolist()

    chosen = rng.choice(len(groups))
    batch = list(rng.choice(groups[int(chosen)], size=min(target, len(groups[int(chosen)])), replace=False))
    if len(batch) < target:
        order = str(tax.get(batch[0], {}).get("order") or "") if batch else ""
        fill_pool = [label for label in order_groups.get(order, []) if label not in set(batch)]
        if len(fill_pool) < target - len(batch):
            fill_pool.extend([label for label in species if label not in set(batch) and label not in set(fill_pool)])
        if fill_pool:
            batch.extend(rng.choice(fill_pool, size=min(target - len(batch), len(fill_pool)), replace=False).tolist())
    # Defend against accidental duplicate labels because duplicates create
    # false negatives in the batch contrastive loss.
    deduped = []
    seen = set()
    for label in batch:
        if label in species_set and label not in seen:
            deduped.append(label)
            seen.add(label)
    if len(deduped) < target:
        fill = [label for label in species if label not in seen]
        deduped.extend(rng.choice(fill, size=min(target - len(deduped), len(fill)), replace=False).tolist())
    return deduped[:target]


def expand_multi_positive_batch(
    rng: np.random.Generator,
    batch_species: list[str],
    coi_by_species: dict[str, list[int]],
    marker_by_species: dict[str, list[int]],
    repeats: int,
) -> tuple[list[str], list[int], list[int]]:
    expanded_species = []
    coi_idx = []
    marker_idx = []
    n_repeats = max(1, repeats)
    for label in batch_species:
        for _ in range(n_repeats):
            expanded_species.append(label)
            coi_idx.append(int(rng.choice(coi_by_species[label])))
            marker_idx.append(int(rng.choice(marker_by_species[label])))
    return expanded_species, coi_idx, marker_idx


def evaluate(
    split: str,
    marker_rows: pd.DataFrame,
    coi_rows: pd.DataFrame,
    marker_emb: np.ndarray,
    coi_emb: np.ndarray,
    marker_head: ProjectionHead | None,
    coi_head: ProjectionHead | None,
    tax: dict[str, dict[str, str]],
    device: str,
    name: str,
) -> list[dict[str, Any]]:
    def project(head: ProjectionHead | None, arr: np.ndarray) -> np.ndarray:
        if head is None:
            z = torch.tensor(arr, device=device)
            return nn.functional.normalize(z, dim=-1).cpu().numpy()
        head.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(arr), 512):
                z = torch.tensor(arr[start : start + 512], device=device)
                outs.append(head(z).cpu().numpy())
        return np.vstack(outs)

    q = project(marker_head, marker_emb)
    c = project(coi_head, coi_emb)
    # COI species prototypes.
    proto_rows = []
    proto_vecs = []
    for label, idxs in coi_rows.groupby("tree_label").groups.items():
        proto_rows.append({"tree_label": label, **tax.get(label, {})})
        proto_vecs.append(c[list(idxs)].mean(axis=0))
    proto = pd.DataFrame(proto_rows)
    proto_arr = np.vstack(proto_vecs).astype("float32")
    proto_arr /= np.linalg.norm(proto_arr, axis=1, keepdims=True).clip(min=1e-8)
    sims = q @ proto_arr.T
    max_k = max(TOP_KS)
    top_idx = np.argpartition(-sims, kth=min(max_k, sims.shape[1] - 1), axis=1)[:, :max_k]
    ordered = np.take_along_axis(top_idx, np.argsort(-np.take_along_axis(sims, top_idx, axis=1), axis=1), axis=1)
    hits = {rank: {k: 0 for k in TOP_KS} for rank in RANKS}
    for row_i, indices in enumerate(ordered):
        query = marker_rows.iloc[row_i]
        top = proto.iloc[indices]
        true_label = str(query["tree_label"])
        true_tax = tax.get(true_label, {"species": true_label.replace("_", " "), "genus": true_label.split("_")[0], "family": "", "order": ""})
        for k in TOP_KS:
            subset = top.iloc[:k]
            hits["species"][k] += int(true_label in set(subset["tree_label"].astype(str)))
            for rank in ("genus", "family", "order"):
                hits[rank][k] += int(true_tax.get(rank, "") in set(subset[rank].astype(str)))
    rows = []
    for rank in RANKS:
        for k in TOP_KS:
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "rank": rank,
                    "k": k,
                    "n_query": len(marker_rows),
                    "topk_accuracy_pct": 100.0 * hits[rank][k] / max(len(marker_rows), 1),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.sequences_per_species_per_batch > 1 and args.loss_mode != "taxonomy_soft":
        raise ValueError("--sequences-per-species-per-batch > 1 requires --loss-mode taxonomy_soft")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.log_file or args.output_dir / "marker_mirror_bridge.log")
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
    tree_cache = None
    tree_soft_scale = args.tree_soft_scale
    if args.loss_mode == "tree_soft":
        logger.log(f"Loading tree distances from {args.tree_file}")
        tree_cache = load_tree_distance_cache(args.tree_file)
        n_in_tree = sum(1 for label in overlap if canonical_tree_label(label) in tree_cache["taxon_nodes"])
        logger.log(f"Tree labels available for overlap species: {n_in_tree}/{len(overlap)}")
        if tree_soft_scale <= 0:
            tree_soft_scale = estimate_tree_soft_scale(
                splits["train"],
                tree_cache,
                np.random.default_rng(args.seed + 17),
                args.tree_soft_sample_pairs,
                args.tree_soft_quantile,
            )
        logger.log(f"Using tree_soft_scale={tree_soft_scale:.6f}")

    all_species = overlap
    coi_rows = flatten(coi, all_species, "coi")
    marker_rows = flatten(marker, all_species, "marker")
    tokenizer, model, masked_lm = load_hf(args.model, args.device, logger)
    coi_emb = embed_sequences(coi_rows["nucleotides"].tolist(), tokenizer, model, masked_lm, args.device, args.embed_batch_size, args.max_length, logger, "coi")
    marker_emb = embed_sequences(marker_rows["nucleotides"].tolist(), tokenizer, model, masked_lm, args.device, args.embed_batch_size, args.max_length, logger, "marker")

    dim = coi_emb.shape[1]
    coi_head = ProjectionHead(dim, args.hidden_dim, args.projection_dim).to(args.device)
    marker_head = ProjectionHead(dim, args.hidden_dim, args.projection_dim).to(args.device)
    opt = torch.optim.AdamW(list(coi_head.parameters()) + list(marker_head.parameters()), lr=args.lr, weight_decay=1e-4)

    coi_by_species = defaultdict(list)
    marker_by_species = defaultdict(list)
    for i, label in enumerate(coi_rows["tree_label"]):
        coi_by_species[str(label)].append(i)
    for i, label in enumerate(marker_rows["tree_label"]):
        marker_by_species[str(label)].append(i)
    train_species = splits["train"]
    val_species = splits["val"]
    rng = np.random.default_rng(args.seed)
    retrieval_selection_ranks = [rank.strip() for rank in args.retrieval_selection_ranks.split(",") if rank.strip()]
    history = []
    best_val_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    best_retrieval_score = -math.inf
    best_retrieval_epoch = 0
    best_retrieval_state: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        coi_head.train()
        marker_head.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            sampled_species = sample_species_batch(
                rng,
                train_species,
                tax,
                max(1, args.batch_size // max(1, args.sequences_per_species_per_batch)),
                args.batch_strategy,
            )
            batch_species, coi_idx, marker_idx = expand_multi_positive_batch(
                rng,
                sampled_species,
                coi_by_species,
                marker_by_species,
                args.sequences_per_species_per_batch,
            )
            zc = torch.tensor(coi_emb[coi_idx], device=args.device)
            zm = torch.tensor(marker_emb[marker_idx], device=args.device)
            loss = marker_mirror_loss(marker_head(zm), coi_head(zc), batch_species, tax, args, tree_cache, tree_soft_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_losses = []
        if val_species and args.val_steps > 0:
            coi_head.eval()
            marker_head.eval()
            with torch.no_grad():
                for _ in range(args.val_steps):
                    sampled_species = sample_species_batch(
                        rng,
                        val_species,
                        tax,
                        max(1, args.batch_size // max(1, args.sequences_per_species_per_batch)),
                        args.batch_strategy,
                    )
                    batch_species, coi_idx, marker_idx = expand_multi_positive_batch(
                        rng,
                        sampled_species,
                        coi_by_species,
                        marker_by_species,
                        args.sequences_per_species_per_batch,
                    )
                    zc = torch.tensor(coi_emb[coi_idx], device=args.device)
                    zm = torch.tensor(marker_emb[marker_idx], device=args.device)
                    val_losses.append(
                        float(
                            marker_mirror_loss(
                                marker_head(zm),
                                coi_head(zc),
                                batch_species,
                                tax,
                                args,
                                tree_cache,
                                tree_soft_scale,
                            ).cpu()
                        )
                    )
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses)) if val_losses else np.nan}
        history.append(row)
        if args.restore_best_val and not math.isnan(row["val_loss"]) and row["val_loss"] < best_val_loss:
            best_val_loss = float(row["val_loss"])
            best_epoch = epoch
            best_state = {
                "coi_head": copy.deepcopy(coi_head.state_dict()),
                "marker_head": copy.deepcopy(marker_head.state_dict()),
            }
        if (
            args.restore_best_retrieval
            and args.retrieval_eval_every > 0
            and (epoch == 1 or epoch % args.retrieval_eval_every == 0 or epoch == args.epochs)
        ):
            q_rows = marker_rows[marker_rows["tree_label"].isin(val_species)].reset_index(drop=True)
            q_emb = marker_emb[marker_rows["tree_label"].isin(val_species).to_numpy()]
            val_metric_rows = evaluate(
                "val_checkpoint",
                q_rows,
                coi_rows,
                q_emb,
                coi_emb,
                marker_head,
                coi_head,
                tax,
                args.device,
                "marker_mirror_projection",
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
                f"ranks={','.join(retrieval_selection_ranks)} k={args.retrieval_selection_k}"
            )
            if np.isfinite(retrieval_score) and retrieval_score > best_retrieval_score:
                best_retrieval_score = retrieval_score
                best_retrieval_epoch = epoch
                best_retrieval_state = {
                    "coi_head": copy.deepcopy(coi_head.state_dict()),
                    "marker_head": copy.deepcopy(marker_head.state_dict()),
                }
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            logger.log(f"epoch={epoch} loss={row['loss']:.4f} val_loss={row['val_loss']:.4f}")

    restored_best_val = False
    restored_best_retrieval = False
    if args.restore_best_retrieval and best_retrieval_state is not None:
        coi_head.load_state_dict(best_retrieval_state["coi_head"])
        marker_head.load_state_dict(best_retrieval_state["marker_head"])
        restored_best_retrieval = True
        logger.log(
            f"Restored best retrieval checkpoint epoch={best_retrieval_epoch} "
            f"score={best_retrieval_score:.4f}"
        )
    elif args.restore_best_val and best_state is not None:
        coi_head.load_state_dict(best_state["coi_head"])
        marker_head.load_state_dict(best_state["marker_head"])
        restored_best_val = True
        logger.log(f"Restored best validation checkpoint epoch={best_epoch} val_loss={best_val_loss:.4f}")

    metric_rows = []
    for split_name, labels in splits.items():
        q_rows = marker_rows[marker_rows["tree_label"].isin(labels)].reset_index(drop=True)
        q_emb = marker_emb[marker_rows["tree_label"].isin(labels).to_numpy()]
        metric_rows.extend(evaluate(split_name, q_rows, coi_rows, q_emb, coi_emb, None, None, tax, args.device, "frozen_nt_cross_marker"))
        metric_rows.extend(evaluate(split_name, q_rows, coi_rows, q_emb, coi_emb, marker_head, coi_head, tax, args.device, "marker_mirror_projection"))

    pd.DataFrame(history).to_csv(args.output_dir / "marker_mirror_training_history.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "marker_mirror_retrieval_metrics.csv", index=False)
    torch.save(
        {"coi_head": coi_head.state_dict(), "marker_head": marker_head.state_dict(), "args": vars(args)},
        args.output_dir / "marker_mirror_projection_heads.pt",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "overlap_species": len(overlap),
        "splits": {key: len(value) for key, value in splits.items()},
        "restored_best_val": restored_best_val,
        "restored_best_retrieval": restored_best_retrieval,
        "best_val_epoch": best_epoch,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "best_retrieval_epoch": best_retrieval_epoch,
        "best_retrieval_score": best_retrieval_score if math.isfinite(best_retrieval_score) else None,
        "tree_soft_scale": tree_soft_scale if args.loss_mode == "tree_soft" else None,
        "claim_boundary": "Fast cross-marker bridge probe. Projection heads only; not a full foundation-model fine-tune yet.",
    }
    (args.output_dir / "marker_mirror_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log("Completed marker mirror bridge probe")


if __name__ == "__main__":
    main()
