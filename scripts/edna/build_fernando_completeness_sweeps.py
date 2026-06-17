#!/usr/bin/env python3
"""Prepare Fernando-style backbone-completeness sweep inputs.

This builds local input directories for fish COI placement sweeps at fixed
backbone completeness levels. Each replicate has:

- candidate_species.csv over the COI species universe;
- train_species_sequences.json for backbone species;
- species_sequences.json for the same COI universe;
- zero_shot_queries.csv with one representative COI query per held-out species;
- split_manifest.json documenting sampling and counts.

The generated inputs can be consumed by
`scripts/edna/prepare_fish_tree_placement_inputs.py` and the Vast wrapper
`experiments/paper1_phylo_calibrated_assignment/runs/07_vast_fernando_completeness_sweeps.sh`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("data/phylo/fish_tree_clean_phylo_inputs/seen_test")
DEFAULT_OUTPUT_ROOT = Path("data/phylo/fernando_completeness_sweeps")
DEFAULT_COMPLETENESS = (0.99, 0.80, 0.60, 0.40, 0.20)
DNA = set("ACGTN")


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def clean_sequence(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in DNA)


def sequence_score(seq: str) -> tuple[int, int, int]:
    acgt = sum(1 for ch in seq if ch in "ACGT")
    return acgt, -seq.count("N"), len(seq)


def best_sequence(seqs: list[object], min_acgt: int) -> str:
    cleaned = [clean_sequence(seq) for seq in seqs]
    cleaned = [seq for seq in cleaned if sequence_score(seq)[0] >= min_acgt]
    return max(cleaned, key=sequence_score) if cleaned else ""


def sample_random(labels: list[str], keep_n: int, rng: np.random.Generator) -> set[str]:
    if keep_n >= len(labels):
        return set(labels)
    return set(rng.choice(labels, size=keep_n, replace=False).tolist())


def sample_stratified(candidates: pd.DataFrame, keep_n: int, rng: np.random.Generator) -> set[str]:
    if keep_n >= len(candidates):
        return set(candidates["tree_label"])
    groups = []
    for _, group in candidates.groupby("family_name", dropna=False):
        groups.append(group["tree_label"].astype(str).tolist())
    selected: set[str] = set()
    fractional: list[tuple[float, list[str], int]] = []
    total = len(candidates)
    for group in groups:
        exact = keep_n * len(group) / total
        base = min(len(group), int(math.floor(exact)))
        if base > 0:
            selected.update(rng.choice(group, size=base, replace=False).tolist())
        fractional.append((exact - base, group, base))
    remaining = keep_n - len(selected)
    for _, group, _ in sorted(fractional, key=lambda item: item[0], reverse=True):
        if remaining <= 0:
            break
        available = [label for label in group if label not in selected]
        if not available:
            continue
        selected.add(str(rng.choice(available)))
        remaining -= 1
    if len(selected) < keep_n:
        available = [label for label in candidates["tree_label"].astype(str) if label not in selected]
        selected.update(rng.choice(available, size=keep_n - len(selected), replace=False).tolist())
    return selected


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_query_csv(
    path: Path,
    heldout_labels: list[str],
    candidate_by_label: dict[str, dict[str, object]],
    representatives: dict[str, str],
    split_name: str,
) -> int:
    fieldnames = [
        "processid",
        "tree_label",
        "species_name",
        "genus_name",
        "family_name",
        "order_name",
        "nucleotides",
        "split",
    ]
    rows = []
    for label in heldout_labels:
        seq = representatives.get(label, "")
        if not seq:
            continue
        item = candidate_by_label[label]
        rows.append(
            {
                "processid": f"{split_name}__{label}",
                "tree_label": label,
                "species_name": item.get("species_name", ""),
                "genus_name": item.get("genus_name", ""),
                "family_name": item.get("family_name", ""),
                "order_name": item.get("order_name", ""),
                "nucleotides": seq,
                "split": split_name,
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_sweeps(args: argparse.Namespace) -> None:
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    input_root = args.input_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    logger.log(f"Loading source species from {input_root}")
    candidates = pd.read_csv(input_root / "candidate_species.csv")
    species_sequences = load_json(input_root / "species_sequences.json")
    if not isinstance(species_sequences, dict):
        raise TypeError("species_sequences.json must be a species->sequences object")

    candidates = candidates[candidates["tree_label"].astype(str).isin(species_sequences)].copy()
    candidates = candidates[candidates["has_reference_sequence"].astype(int) == 1].copy()
    candidates["tree_label"] = candidates["tree_label"].astype(str)
    candidates = candidates.sort_values("tree_label").reset_index(drop=True)
    logger.log(f"Using {len(candidates)} COI species with reference sequences")

    representatives = {
        label: best_sequence(seqs, args.min_acgt)
        for label, seqs in species_sequences.items()
        if label in set(candidates["tree_label"])
    }
    representatives = {label: seq for label, seq in representatives.items() if seq}
    candidates = candidates[candidates["tree_label"].isin(representatives)].copy()
    candidate_by_label = {str(row["tree_label"]): dict(row) for _, row in candidates.iterrows()}
    labels = candidates["tree_label"].astype(str).tolist()
    logger.log(f"{len(labels)} species have representative sequences with >= {args.min_acgt} ACGT bases")

    summary_rows: list[dict[str, object]] = []
    for scheme in args.schemes:
        for completeness in args.completeness:
            keep_n = max(1, min(len(labels), int(round(len(labels) * completeness))))
            for rep in range(1, args.replicates + 1):
                seed = args.seed + rep + int(completeness * 10000) + (100000 if scheme == "family_stratified" else 0)
                rng = np.random.default_rng(seed)
                if scheme == "random":
                    backbone = sample_random(labels, keep_n, rng)
                elif scheme == "family_stratified":
                    backbone = sample_stratified(candidates, keep_n, rng)
                else:
                    raise ValueError(f"Unsupported scheme: {scheme}")
                heldout = sorted(label for label in labels if label not in backbone)
                split_name = f"{scheme}_c{int(round(completeness * 100)):02d}_rep{rep:02d}"
                split_dir = output_root / split_name
                split_dir.mkdir(parents=True, exist_ok=True)
                logger.log(
                    f"Writing {split_name}: backbone={len(backbone)} heldout={len(heldout)} seed={seed}"
                )
                candidates.to_csv(split_dir / "candidate_species.csv", index=False)
                train_sequences = {label: species_sequences[label] for label in sorted(backbone)}
                write_json(split_dir / "train_species_sequences.json", train_sequences)
                write_json(split_dir / "species_sequences.json", {label: species_sequences[label] for label in labels})
                write_json(split_dir / "species_info.json", candidate_by_label)
                query_count = write_query_csv(
                    split_dir / "zero_shot_queries.csv",
                    heldout,
                    candidate_by_label,
                    representatives,
                    split_name,
                )
                write_json(split_dir / "val_species.json", [])
                write_json(split_dir / "test_species.json", heldout)
                write_json(split_dir / "eval_c_species.json", heldout)
                manifest = {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "scheme": scheme,
                    "completeness": completeness,
                    "replicate": rep,
                    "seed": seed,
                    "input_root": str(input_root),
                    "tree_file": str(args.tree_file),
                    "species_universe": len(labels),
                    "backbone_species": len(backbone),
                    "heldout_species": len(heldout),
                    "query_rows": query_count,
                    "min_acgt": args.min_acgt,
                    "claim_boundary": "Fernando-style sweep input; exact comparability requires running the same placement/scoring protocol.",
                }
                write_json(split_dir / "manifest.json", manifest)
                summary_rows.append(
                    {
                        "split_name": split_name,
                        "scheme": scheme,
                        "completeness": completeness,
                        "replicate": rep,
                        "seed": seed,
                        "species_universe": len(labels),
                        "backbone_species": len(backbone),
                        "heldout_species": len(heldout),
                        "query_rows": query_count,
                        "input_dir": str(split_dir),
                    }
                )

    summary_path = output_root / "sweep_manifest.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "split_name",
            "scheme",
            "completeness",
            "replicate",
            "seed",
            "species_universe",
            "backbone_species",
            "heldout_species",
            "query_rows",
            "input_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(
        output_root / "manifest.json",
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generated_by": "scripts/edna/build_fernando_completeness_sweeps.py",
            "input_root": str(input_root),
            "output_root": str(output_root),
            "tree_file": str(args.tree_file),
            "schemes": args.schemes,
            "completeness": args.completeness,
            "replicates": args.replicates,
            "rows": len(summary_rows),
            "notes": [
                "Uses the clean COI species universe with reference sequences.",
                "Each held-out species receives one representative COI query sequence.",
                "Random and family-stratified schemes are Fernando-style, not an exact copy of their data universe.",
            ],
        },
    )
    logger.log(f"Wrote {summary_path} with {len(summary_rows)} sweep definitions")
    logger.done(Path(__file__).name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tree-file", type=Path, default=Path("data/phylo/actinopt_12k_treePL.tre"))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--min-acgt", type=int, default=400)
    parser.add_argument("--schemes", nargs="+", default=["random", "family_stratified"])
    parser.add_argument("--completeness", nargs="+", type=float, default=list(DEFAULT_COMPLETENESS))
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()
    build_sweeps(args)


if __name__ == "__main__":
    main()
