# Clean COI Fish Tree Experiment

This folder is the orchestration layer for the strict COI Fish Tree DNA-to-tree
rerun. It is separate from the 12S/eDNA TAXDNA-style track.

The goal is to test whether a sequence encoder trained on COI barcodes can
recover Fish Tree of Life structure when the evaluation species are not used for
sequence-model training.

## Inputs

- Split manifest: `data/phylo/fish_tree_clean_splits/manifest.json`
- Eval C input: `data/phylo/fish_tree_clean_phylo_inputs/eval_c`
- Seen-test input: `data/phylo/fish_tree_clean_phylo_inputs/seen_test`
- Held-out-genera input: `data/phylo/fish_tree_clean_phylo_inputs/unseen_genera`
- Tree: `data/phylo/actinopt_12k_treePL.tre`
- Taxonomy: `data/phylo/PFC_taxonomy.csv`

## Run Wrappers

- `runs/01_local_checks.sh`: rerun dry-run checks and refresh source tables/plots.
- `runs/02_vast_cosine_dim384.sh`: historical strict clean-split COI
  DNA-to-tree baseline.
- `runs/03_pull_vast_results.sh`: copy clean fish-tree outputs back from Vast
  hosts.
- `runs/04_refresh_local_outputs.sh`: regenerate the canonical ledger and
  figure drafts after copied results are available.
- `runs/05_vast_hier512_seqval.sh`: corrected stronger tree-space run using
  random-sequence validation so training is not starved of reference species.
- `runs/06_vast_cosine512_seed_repeats.sh`: cosine512 seed-repeat wrapper for
  robustness checks.

## Claim Boundary

Legacy fish-tree results remain useful as historical evidence, but this track is
the clean rerun for paper-facing claims. Eval C query species are removed before
sequence-model training, and the held-out-genera split is stricter still.

Current paper-facing interpretation is maintained in
`experiments/paper1_phylo_calibrated_assignment/`. This folder is now the lower
level orchestration layer, not the main narrative doc.
