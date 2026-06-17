#!/usr/bin/env bash
set -euo pipefail

python3 scripts/edna/train_12s_phylo_mamba.py \
  --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
  --output-dir results/dry_runs/fish_tree_clean_eval_c_phylo_mamba \
  --tree-file data/phylo/actinopt_12k_treePL.tre \
  --max-seq-len 700 \
  --embed-dim 384 \
  --dry-run

python3 scripts/edna/train_12s_phylo_mamba.py \
  --input-dir data/phylo/fish_tree_clean_phylo_inputs/seen_test \
  --output-dir results/dry_runs/fish_tree_clean_seen_test_phylo_mamba \
  --tree-file data/phylo/actinopt_12k_treePL.tre \
  --max-seq-len 700 \
  --embed-dim 384 \
  --dry-run

python3 scripts/edna/train_12s_phylo_mamba.py \
  --input-dir data/phylo/fish_tree_clean_phylo_inputs/unseen_genera \
  --output-dir results/dry_runs/fish_tree_clean_unseen_genera_phylo_mamba \
  --tree-file data/phylo/actinopt_12k_treePL.tre \
  --max-seq-len 700 \
  --embed-dim 384 \
  --dry-run

python3 scripts/summarize_results_ledger.py
python3 scripts/figures/build_source_tables.py
python3 scripts/figures/plot_source_tables.py
