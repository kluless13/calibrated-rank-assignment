# MarineMamba

MarineMamba contains barcode-modeling workflows for marine COI data, fish 12S/eDNA data, and Fish Tree of Life DNA-to-tree experiments. The repository is organized around reproducible scripts, run ledgers, and audit notes. Large datasets, model checkpoints, and result directories are intentionally kept out of git.

## Repository Layout

- `scripts/` - COI data preparation, baseline evaluation, curriculum models, phylogenetic runs, and result summaries.
- `scripts/edna/` - 12S/eDNA ingestion, split building, Stalder/TAXDNA-style inputs, real eDNA validation, BLAST/geographic/ecological baselines, calibration, and DNA-to-tree training utilities.
- `scripts/figures/` - source-table generation utilities for downstream plotting.
- `configs/runs/` - dated run ledgers and launch plans. These record commands, expected inputs, outputs, and status without storing large artifacts.
- `docs/` - audits, workflow notes, dataset notes, interpretation notes, and active-track status.
- `data/` - local data workspace. Most contents are ignored; see `data/README.md`.
- `results/` - local and copied run outputs. Ignored by git.

## Tracked Versus Generated Artifacts

Tracked:

- Source code under `scripts/`.
- Documentation under `docs/`.
- Run ledgers under `configs/runs/`.
- Small repository metadata and the Fish Tree Newick file already present in `data/phylo/`.

Ignored:

- Raw BOLD, Mitohelper, Mare-MAGE, Dryad, GBIF, OBIS, and other downloaded data.
- Processed split CSVs and JSON input packs.
- Checkpoints and model weights.
- Result directories and calibration outputs.

## Main Workflows

### COI Cleaning And Splits

```bash
python3 scripts/fetch_bold_marine.py
python3 scripts/02_clean_and_split.py
python3 scripts/build_clean_splits.py
```

### COI Curriculum And Eval C

```bash
python3 scripts/09_multihead_hierarchical.py --data-dir data/processed_clean --output-dir results/coi_multihead
python3 scripts/11_curriculum_6mer.py --data-dir data/processed_clean --output-dir results/coi_6mer
python3 scripts/eval_c_stalder_protocol.py --data-dir data/processed_clean --output-dir results/coi_eval_c
```

### 12S/eDNA Inputs And Validation

```bash
python3 scripts/edna/build_12s_splits.py --help
python3 scripts/edna/build_multisource_12s_splits.py --help
python3 scripts/edna/build_stalder_sequence_inputs.py --help
python3 scripts/edna/eval_global_edna_sample_validation.py --help
```

### DNA-To-Tree Training

```bash
python3 scripts/edna/train_12s_phylo_mamba.py --help
python3 scripts/edna/predict_phylo_mamba_checkpoint.py --help
python3 scripts/edna/eval_phylo_checkpoint_tree_recovery.py --help
```

### Strict Fish Tree Input Pack

```bash
python3 scripts/build_phylo_fish_tree_splits.py
python3 scripts/build_fish_tree_phylo_inputs.py
```

The generated Fish Tree input directories are ignored by git. The launch details are recorded in `configs/runs/2026-05-26-fish-tree-clean-phylo-rerun-plan.json`.

## Run Ledgers

Use the dated JSON files in `configs/runs/` as the current source for exact commands and output paths. They are designed to survive cleanup of local `data/` and `results/` directories.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some workflows also require BarcodeMamba, BLAST/VSEARCH, CUDA-enabled PyTorch, or remote GPU instances. Those requirements are noted in the relevant script help text and run ledgers.

## License

MIT
