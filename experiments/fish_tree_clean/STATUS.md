# Clean COI Fish Tree Status

Generated on 2026-05-26.

## Ready Locally

- Strict input directories are prepared under `data/phylo/fish_tree_clean_phylo_inputs/`.
- Dry-run checks completed for Eval C, seen-test, and held-out-genera inputs.
- Sequence-only tree-distance baselines exist at `results/phylo_tree_distance_baselines_clean/`.
- The run plan is captured in `configs/runs/2026-05-26-fish-tree-clean-phylo-rerun-plan.json`.
- Executable wrappers are now available under `experiments/fish_tree_clean/runs/`.

## Input Counts

- candidate tree species: 11,638
- reference species: 3,839
- reference sequences: 62,866
- Eval C query species: 531
- Eval C query sequences: 11,594
- seen-test species: 3,839
- seen-test sequences: 15,763
- held-out-genera species: 614
- held-out-genera sequences: 9,148

## Current Baselines

The clean sequence-only distance baseline is weak relative to the historical
learned tree-recovery result:

- train-train 6-mer cosine Pearson/Spearman: 0.278 / 0.224
- unseen-train 6-mer cosine Pearson/Spearman: 0.260 / 0.221
- unseen-unseen 6-mer cosine Pearson/Spearman: 0.371 / 0.239

This makes the clean learned rerun worth doing: a strong result would not be a
trivial k-mer-distance artifact.

## Remaining Work

- Launch `runs/02_vast_cosine_dim384.sh` on a GPU host.
- Pull results with `runs/03_pull_vast_results.sh`.
- Refresh ledger/figures with `runs/04_refresh_local_outputs.sh`.
- Decide whether to run a second objective only after the cosine rerun is known.
