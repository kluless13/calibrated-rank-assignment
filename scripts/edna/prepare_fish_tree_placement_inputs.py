#!/usr/bin/env python3
"""Prepare clean COI fish-tree inputs for phylogenetic placement baselines.

The placement baselines are intentionally reference-backed: the backbone tree
is pruned to species that have training/reference COI sequences. Eval C and
unseen-genera query species are therefore absent from the backbone sequence
database, matching the constraint faced by BLAST/VSEARCH/k-mer baselines.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import dendropy
from Bio import Phylo, SeqIO

from progress_logging import ProgressLogger, default_log_path


ROOT = Path(__file__).resolve().parents[2]
DNA = set("ACGTN")


def clean_sequence(seq: object) -> str:
    return "".join(ch for ch in str(seq).upper() if ch in DNA)


def sequence_score(seq: str) -> tuple[int, int, int]:
    acgt = sum(1 for ch in seq if ch in "ACGT")
    n_count = seq.count("N")
    return acgt, -n_count, len(seq)


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def load_candidate_labels(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["tree_label"]) for row in reader}


def load_queries(path: Path, max_queries: int | None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_queries is not None:
        rows = rows[:max_queries]
    return rows


def choose_reference_sequences(
    train_sequences: dict[str, list[object]],
    candidate_labels: set[str],
    min_ref_acgt: int,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for label, seqs in train_sequences.items():
        if label not in candidate_labels:
            continue
        cleaned = [clean_sequence(seq) for seq in seqs]
        cleaned = [seq for seq in cleaned if sequence_score(seq)[0] >= min_ref_acgt]
        if not cleaned:
            continue
        selected[label] = max(cleaned, key=sequence_score)
    return selected


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with path.open("w") as handle:
        for label, seq in records:
            handle.write(f">{label}\n{seq}\n")


def prune_tree(tree_path: Path, keep_labels: set[str], output_path: Path) -> dict[str, int]:
    tree = Phylo.read(tree_path, "newick")
    terminals = list(tree.get_terminals())
    initial_tips = len(terminals)
    missing_name = 0
    for tip in terminals:
        if not tip.name:
            missing_name += 1
            continue
        if tip.name not in keep_labels:
            tree.prune(tip)
    final_tips = len(tree.get_terminals())
    Phylo.write(tree, output_path, "newick")
    return {
        "initial_tips": initial_tips,
        "final_tips": final_tips,
        "missing_name_tips": missing_name,
    }


def prepare(args: argparse.Namespace) -> None:
    logger = getattr(args, "logger", ProgressLogger())
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.log(f"Preparing placement inputs from {input_dir}")
    candidate_labels = load_candidate_labels(input_dir / "candidate_species.csv")
    logger.log(f"Loaded {len(candidate_labels)} candidate species labels")
    train_sequences = load_json(input_dir / "train_species_sequences.json")
    if not isinstance(train_sequences, dict):
        raise TypeError("train_species_sequences.json must contain a species->sequences object")
    selected = choose_reference_sequences(train_sequences, candidate_labels, args.min_ref_acgt)
    if not selected:
        raise RuntimeError(f"No reference sequences selected from {input_dir}")
    logger.log(f"Selected {len(selected)} reference species with >= {args.min_ref_acgt} ACGT bases")

    queries = load_queries(input_dir / "zero_shot_queries.csv", args.max_queries)
    logger.log(f"Loaded {len(queries)} candidate query rows before sequence filtering")
    query_records: list[tuple[str, str]] = []
    query_manifest_rows: list[dict[str, object]] = []
    for idx, row in enumerate(queries):
        seq = clean_sequence(row["nucleotides"])
        if sequence_score(seq)[0] < args.min_query_acgt:
            continue
        processid = str(row.get("processid") or f"query_{idx}")
        tree_label = str(row["tree_label"])
        query_id = f"query_{idx}|{processid}|{tree_label}"
        query_records.append((query_id, seq))
        query_manifest_rows.append(
            {
                "query_id": query_id,
                "processid": processid,
                "tree_label": tree_label,
                "species_name": row.get("species_name", ""),
                "genus_name": row.get("genus_name", ""),
                "family_name": row.get("family_name", ""),
                "order_name": row.get("order_name", ""),
                "split": row.get("split", ""),
                "acgt_bases": sequence_score(seq)[0],
                "length": len(seq),
            }
        )

    ref_records = sorted(selected.items())
    logger.log(f"Writing {len(ref_records)} reference records and {len(query_records)} query records")
    write_fasta(ref_records, output_dir / "reference_unaligned.fasta")
    write_fasta(query_records, output_dir / "query_unaligned.fasta")

    logger.log(f"Pruning tree {args.tree} to selected reference species")
    tree_stats = prune_tree(Path(args.tree), set(selected), output_dir / "reference_tree.nwk")
    logger.log(f"Pruned tree stats: {tree_stats}")

    with (output_dir / "query_manifest.csv").open("w", newline="") as handle:
        fieldnames = [
            "query_id",
            "processid",
            "tree_label",
            "species_name",
            "genus_name",
            "family_name",
            "order_name",
            "split",
            "acgt_bases",
            "length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(query_manifest_rows)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "tree": str(args.tree),
        "reference_species": len(ref_records),
        "queries": len(query_records),
        "min_ref_acgt": args.min_ref_acgt,
        "min_query_acgt": args.min_query_acgt,
        "max_queries": args.max_queries,
        "tree_stats": tree_stats,
        "notes": [
            "Backbone tree is pruned to species with training/reference COI sequences.",
            "Held-out Eval C/unseen-genera species are not injected as backbone leaves unless they have train references.",
            "This makes placement baselines reference-backed rather than known-candidate-tree ranking.",
        ],
    }
    (output_dir / "placement_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.log(f"Wrote placement input manifest to {output_dir / 'placement_input_manifest.json'}")


def split_added_alignment(args: argparse.Namespace) -> None:
    logger = getattr(args, "logger", ProgressLogger())
    combined = Path(args.combined)
    query_out = Path(args.query_output)
    ref_out = Path(args.reference_output) if args.reference_output else None
    query_out.parent.mkdir(parents=True, exist_ok=True)
    logger.log(f"Splitting added alignment {combined}")
    query_records = []
    ref_records = []
    for record in SeqIO.parse(combined, "fasta"):
        if record.id.startswith(args.query_prefix):
            query_records.append(record)
        else:
            ref_records.append(record)
    if not query_records:
        raise RuntimeError(f"No query records with prefix {args.query_prefix!r} in {combined}")
    SeqIO.write(query_records, query_out, "fasta")
    if ref_out is not None:
        ref_out.parent.mkdir(parents=True, exist_ok=True)
        SeqIO.write(ref_records, ref_out, "fasta")
    logger.log(f"Split alignment into {len(query_records)} query and {len(ref_records)} reference records")

    manifest = {
        "combined": str(combined),
        "query_output": str(query_out),
        "reference_output": str(ref_out) if ref_out else None,
        "query_records": len(query_records),
        "reference_records": len(ref_records),
    }
    (query_out.parent / "split_added_alignment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.log(f"Wrote split alignment manifest to {query_out.parent / 'split_added_alignment_manifest.json'}")


def deroot_tree(args: argparse.Namespace) -> None:
    logger = getattr(args, "logger", ProgressLogger())
    input_tree = Path(args.input_tree)
    output_tree = Path(args.output_tree)
    output_tree.parent.mkdir(parents=True, exist_ok=True)
    logger.log(f"Derooting tree {input_tree} for APPLES branch-length re-estimation")
    tree = dendropy.Tree.get(path=str(input_tree), schema="newick", preserve_underscores=True)
    before_children = len(tree.seed_node.child_nodes())
    tree.deroot()
    tree.is_rooted = False
    after_children = len(tree.seed_node.child_nodes())
    tree.write(
        path=str(output_tree),
        schema="newick",
        suppress_rooting=True,
        suppress_annotations=True,
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_tree": str(input_tree),
        "output_tree": str(output_tree),
        "before_root_children": before_children,
        "after_root_children": after_children,
        "purpose": "APPLES default branch-length re-estimation fails on some rooted two-child trees; derooting keeps the unrooted topology for APPLES placement.",
    }
    (output_tree.parent / "apples_tree_deroot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.log(f"Wrote derooted APPLES tree to {output_tree}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--input-dir", required=True)
    prep.add_argument("--tree", default="data/phylo/actinopt_12k_treePL.tre")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--min-ref-acgt", type=int, default=400)
    prep.add_argument("--min-query-acgt", type=int, default=300)
    prep.add_argument("--max-queries", type=int, default=None)
    prep.add_argument("--log-file", type=Path)
    prep.set_defaults(func=prepare)

    split = subparsers.add_parser("split-added-alignment")
    split.add_argument("--combined", required=True)
    split.add_argument("--query-output", required=True)
    split.add_argument("--reference-output", default=None)
    split.add_argument("--query-prefix", default="query_")
    split.add_argument("--log-file", type=Path)
    split.set_defaults(func=split_added_alignment)

    deroot = subparsers.add_parser("deroot-tree")
    deroot.add_argument("--input-tree", required=True)
    deroot.add_argument("--output-tree", required=True)
    deroot.add_argument("--log-file", type=Path)
    deroot.set_defaults(func=deroot_tree)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logger = ProgressLogger(args.log_file or default_log_path(ROOT, Path(__file__).stem))
    args.logger = logger
    logger.start(f"{Path(__file__).name} {args.command}")
    args.func(args)
    logger.done(f"{Path(__file__).name} {args.command}")


if __name__ == "__main__":
    main()
