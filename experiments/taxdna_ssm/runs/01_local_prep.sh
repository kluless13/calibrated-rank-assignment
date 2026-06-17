#!/usr/bin/env bash
set -euo pipefail

python3 scripts/edna/train_12s_phylo_mamba.py \
  --input-dir data/edna/stalder_inputs/multisource_teleo \
  --output-dir results/edna/taxdna_ssm/dry_run_mamba_multisource_teleo \
  --max-seq-len 128 \
  --dry-run

python3 scripts/edna/train_taxdna_cnn_baseline.py \
  --input-dir data/edna/stalder_inputs/multisource_teleo \
  --output-dir results/edna/taxdna_ssm/dry_run_cnn_multisource_teleo \
  --max-seq-len 128 \
  --dry-run

python3 scripts/edna/train_12s_phylo_mamba.py \
  --input-dir data/edna/stalder_inputs/multisource \
  --output-dir results/edna/taxdna_ssm/dry_run_mamba_multisource \
  --max-seq-len 2048 \
  --dry-run

python3 scripts/edna/train_taxdna_cnn_baseline.py \
  --input-dir data/edna/stalder_inputs/multisource \
  --output-dir results/edna/taxdna_ssm/dry_run_cnn_multisource \
  --max-seq-len 2048 \
  --dry-run

python3 scripts/edna/build_taxdna_cooccurrence_inputs.py \
  --input-dir data/edna/real_edna_queries/global_tropical_multisource_teleo \
  --output-dir data/edna/cooccurrence_inputs/taxdna_ssm \
  --rls-group-column SurveyID \
  --obis-site-column site20 \
  --min-species-per-group 2

python3 scripts/edna/export_global_edna_swarm_representatives.py \
  --global-edna-dir /Users/kluless/Downloads/Global_eDNA \
  --output-dir data/edna/raw/real_edna/global_tropical_swarm_representatives \
  --min-length 20
