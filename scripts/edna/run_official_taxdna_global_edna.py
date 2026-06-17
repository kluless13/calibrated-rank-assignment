#!/usr/bin/env python3
"""Run released TAXDNA inference on local Global_eDNA SWARM representatives."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def add_taxdna_to_path(repo: Path) -> None:
    sys.path.insert(0, str(repo.resolve()))
    sys.path.insert(0, str((repo / "src").resolve()))


def import_taxdna_modules() -> tuple[object, object, object, object, object]:
    try:
        import torch
        from ete3 import TreeNode
        from predict import predict
        from taxDNA.models.co_occ_model import CoOccModel
        from taxDNA.models.dna_model import DNAModel
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            "Missing official TAXDNA dependency "
            f"{missing!r}. Install the official package dependencies first, e.g. "
            "`python -m pip install -e data/edna/raw/stalder_taxdna`."
        ) from exc
    return torch, TreeNode, predict, CoOccModel, DNAModel


def infer_site_label(path: Path) -> str:
    name = path.stem
    suffix = "_teleo_table_motu_repr"
    return name.removesuffix(suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxdna-repo",
        type=Path,
        default=Path("data/edna/raw/stalder_taxdna"),
        help="Resolved official TAXDNA repository clone.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/edna/raw/real_edna/global_tropical_swarm_representatives"),
        help="Directory of SWARM representative FASTA files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/edna/stalder_reproduction/official_taxdna_global_edna"),
    )
    parser.add_argument(
        "--dna-checkpoint",
        type=Path,
        default=Path(
            "trained_models/dna_models/val_RivFishTIME/ha2bbpcc/checkpoints/"
            "best-val-epoch=12-val_loss=0.12.ckpt"
        ),
    )
    parser.add_argument(
        "--cooccurrence-checkpoint",
        type=Path,
        default=Path(
            "trained_models/co_occ_models/train_community/3zhw69kx/checkpoints/"
            "epoch=19-step=86040.ckpt"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. Use auto to select CUDA when available.",
    )
    args = parser.parse_args()

    taxdna_repo = args.taxdna_repo.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not taxdna_repo.exists():
        raise SystemExit(f"Official TAXDNA repo not found: {taxdna_repo}")
    if not input_dir.exists():
        raise SystemExit(f"Input FASTA directory not found: {input_dir}")

    add_taxdna_to_path(taxdna_repo)
    torch, TreeNode, predict, CoOccModel, DNAModel = import_taxdna_modules()

    original_cwd = Path.cwd()
    os.chdir(taxdna_repo)
    try:
        dna_ckpt = args.dna_checkpoint
        coocc_ckpt = args.cooccurrence_checkpoint
        if not dna_ckpt.is_absolute():
            dna_ckpt = taxdna_repo / dna_ckpt
        if not coocc_ckpt.is_absolute():
            coocc_ckpt = taxdna_repo / coocc_ckpt

        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"

        tree = TreeNode("data/fish_tree_of_life/actinopt_single.tree")
        all_species = tree.get_leaf_names()
        with open("data/species_info/species_info.json", "r") as handle:
            species_info = json.load(handle)

        co_occ_model = CoOccModel.load_from_checkpoint(coocc_ckpt, map_location=device)
        co_occ_model.eval()
        dna_model = DNAModel.load_from_checkpoint(dna_ckpt, map_location=device)
        hyper_params = torch.load(dna_ckpt, map_location=device, weights_only=True)["hyper_parameters"]
        dna_model.eval()

        rows: list[dict[str, object]] = []
        combined_frames: list[pd.DataFrame] = []
        for fasta in sorted(input_dir.glob("*.fasta")):
            site_label = infer_site_label(fasta)
            predictions = predict(
                swarm_file=fasta,
                dna_model=dna_model,
                co_occ_model=co_occ_model,
                hyper_params=hyper_params,
                species_info=species_info,
                all_species=all_species,
            )
            predictions.insert(0, "site_label", site_label)
            predictions.insert(1, "input_fasta", fasta.name)

            out_csv = output_dir / f"{fasta.stem}.official_taxdna_predictions.csv"
            predictions.to_csv(out_csv, index=False)
            combined_frames.append(predictions)
            rows.append(
                {
                    "input_fasta": str(fasta),
                    "site_label": site_label,
                    "output_csv": str(out_csv),
                    "representative_sequences": int(len(predictions)),
                    "total_abundance": int(predictions["Abundance"].sum()),
                    "unique_predicted_species": int(predictions["Species"].nunique()),
                    "unique_predicted_genera": int(predictions["Genus"].nunique()),
                    "unique_predicted_families": int(predictions["Family"].nunique()),
                }
            )

        combined_csv = output_dir / "official_taxdna_global_edna_predictions.csv"
        if combined_frames:
            pd.concat(combined_frames, ignore_index=True).to_csv(combined_csv, index=False)

        manifest = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "taxdna_repo": str(taxdna_repo),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "device": device,
            "dna_checkpoint": str(dna_ckpt),
            "cooccurrence_checkpoint": str(coocc_ckpt),
            "combined_csv": str(combined_csv),
            "n_fastas": len(rows),
            "runs": rows,
        }
        manifest_path = output_dir / "official_taxdna_global_edna_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {manifest_path}")
        print(f"Wrote {combined_csv}")
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
