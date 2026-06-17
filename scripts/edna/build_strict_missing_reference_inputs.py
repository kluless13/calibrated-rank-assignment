#!/usr/bin/env python3
"""Build strict missing-reference input packs for Paper 1.

The existing candidate-ablation tables hide taxa after rankings are produced.
These packs remove taxa before tree embeddings, sequence training, and
candidate retrieval are built. That is the stricter experiment needed for
missing-reference claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_DIR))

from progress_logging import ProgressLogger, default_log_path  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RANK_COLUMN = {
    "species": "tree_label",
    "genus": "genus_name",
    "family": "family_name",
    "order": "order_name",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() == "nan" else text


def rank_values_for_queries(queries: pd.DataFrame, hide_rank: str) -> set[str]:
    if hide_rank == "species":
        return set(queries["tree_label"].dropna().astype(str))
    column = RANK_COLUMN[hide_rank]
    return {clean_value(value) for value in queries[column].tolist() if clean_value(value)}


def hidden_labels_for_rank(candidates: pd.DataFrame, queries: pd.DataFrame, hide_rank: str) -> tuple[set[str], set[str]]:
    hidden_values = rank_values_for_queries(queries, hide_rank)
    if hide_rank == "species":
        return hidden_values, hidden_values
    column = RANK_COLUMN[hide_rank]
    hidden = set(
        candidates.loc[candidates[column].map(clean_value).isin(hidden_values), "tree_label"].dropna().astype(str)
    )
    return hidden, hidden_values


def prune_sequence_map(raw: dict[str, list[str]], hidden_labels: set[str]) -> dict[str, list[str]]:
    return {
        str(label): seqs
        for label, seqs in raw.items()
        if str(label) not in hidden_labels and isinstance(seqs, list) and seqs
    }


def prune_val_species(path: Path, hidden_labels: set[str]) -> list[str]:
    if not path.exists():
        return []
    raw = read_json(path)
    return [str(label) for label in raw if str(label) not in hidden_labels]


def build_split(base_dir: Path, output_dir: Path, split: str, hide_rank: str, logger: ProgressLogger) -> dict[str, Any]:
    source_dir = base_dir / split
    candidates = pd.read_csv(source_dir / "candidate_species.csv")
    queries = pd.read_csv(source_dir / "zero_shot_queries.csv")
    species_sequences = read_json(source_dir / "species_sequences.json")
    train_sequences = read_json(source_dir / "train_species_sequences.json")
    manifest = read_json(source_dir / "manifest.json")
    species_info = read_json(source_dir / "species_info.json")

    hidden_labels, hidden_values = hidden_labels_for_rank(candidates, queries, hide_rank)
    kept_candidates = candidates[~candidates["tree_label"].astype(str).isin(hidden_labels)].copy()
    pruned_species_sequences = prune_sequence_map(species_sequences, hidden_labels)
    pruned_train_sequences = prune_sequence_map(train_sequences, hidden_labels)
    pruned_val_species = prune_val_species(source_dir / "val_species.json", hidden_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    kept_candidates.to_csv(output_dir / "candidate_species.csv", index=False)
    queries.to_csv(output_dir / "zero_shot_queries.csv", index=False)
    write_json(output_dir / "species_sequences.json", pruned_species_sequences)
    write_json(output_dir / "train_species_sequences.json", pruned_train_sequences)
    write_json(output_dir / "species_info.json", species_info)
    write_json(output_dir / "val_species.json", pruned_val_species)
    if (source_dir / "eval_c_species.json").exists():
        shutil.copy2(source_dir / "eval_c_species.json", output_dir / "eval_c_species.json")

    hidden_summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/edna/build_strict_missing_reference_inputs.py",
        "source_input_dir": str(source_dir),
        "split": split,
        "hide_rank": hide_rank,
        "hidden_rank_value_count": len(hidden_values),
        "hidden_candidate_species": len(hidden_labels),
        "query_rows": int(len(queries)),
        "query_species": int(queries["tree_label"].nunique()),
        "source_candidate_species": int(candidates["tree_label"].nunique()),
        "kept_candidate_species": int(kept_candidates["tree_label"].nunique()),
        "source_species_sequences_species": len(species_sequences),
        "kept_species_sequences_species": len(pruned_species_sequences),
        "source_train_species": len(train_sequences),
        "kept_train_species": len(pruned_train_sequences),
        "source_val_species": len(read_json(source_dir / "val_species.json")) if (source_dir / "val_species.json").exists() else 0,
        "kept_val_species": len(pruned_val_species),
        "candidate_true_species_present": int(queries["tree_label"].astype(str).isin(set(kept_candidates["tree_label"].astype(str))).sum()),
        "notes": [
            "Hidden taxa are removed before tree embedding and encoder training.",
            "zero_shot_queries.csv is intentionally unchanged so the scorer can evaluate rank backoff.",
            "species_info.json is retained for query taxonomy and candidate rank metadata; candidate_species.csv controls the candidate tree.",
        ],
        "source_manifest": manifest,
    }
    write_json(output_dir / "manifest.json", hidden_summary)
    write_json(output_dir / "hidden_labels.json", sorted(hidden_labels))
    write_json(output_dir / "hidden_rank_values.json", sorted(hidden_values))
    logger.log(
        f"Built {output_dir}: hide={hide_rank} hidden_candidates={len(hidden_labels)} "
        f"kept_candidates={kept_candidates['tree_label'].nunique()} kept_train={len(pruned_train_sequences)}"
    )
    return hidden_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/phylo/fish_tree_clean_phylo_inputs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/phylo/paper1_strict_missing_reference_inputs"),
    )
    parser.add_argument("--splits", nargs="+", default=["eval_c", "unseen_genera"])
    parser.add_argument("--hide-ranks", nargs="+", choices=sorted(RANK_COLUMN), default=["species", "genus", "family"])
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    logger.start(Path(__file__).name)
    rows: list[dict[str, Any]] = []
    for split in args.splits:
        for hide_rank in args.hide_ranks:
            name = f"{split}_hide_{hide_rank}"
            summary = build_split(args.base_dir, args.output_root / name, split, hide_rank, logger)
            rows.append(
                {
                    "name": name,
                    "split": split,
                    "hide_rank": hide_rank,
                    "input_dir": str(args.output_root / name),
                    "hidden_candidate_species": summary["hidden_candidate_species"],
                    "kept_candidate_species": summary["kept_candidate_species"],
                    "kept_train_species": summary["kept_train_species"],
                    "query_rows": summary["query_rows"],
                    "query_species": summary["query_species"],
                }
            )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/edna/build_strict_missing_reference_inputs.py",
        "base_dir": str(args.base_dir),
        "output_root": str(args.output_root),
        "splits": args.splits,
        "hide_ranks": args.hide_ranks,
        "n_packs": len(rows),
        "notes": [
            "These packs are for strict missing-reference validation.",
            "Training and candidate tree construction should use these input dirs directly.",
            "Do not mix predictions from non-pruned candidate trees with these strict packs.",
        ],
    }
    write_json(args.output_root / "manifest.json", manifest)
    summary_path = args.output_root / "strict_missing_reference_manifest.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "name",
            "split",
            "hide_rank",
            "input_dir",
            "hidden_candidate_species",
            "kept_candidate_species",
            "kept_train_species",
            "query_rows",
            "query_species",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.log(f"Wrote {summary_path} with {len(rows)} rows")
    logger.done(Path(__file__).name)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
