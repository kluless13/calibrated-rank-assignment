#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

python3 scripts/edna/run_paper1_production_v1.py \
  --target-precision 0.99 \
  --input-run-dir results/paper1_phylo_calibrated_assignment/pipeline_runs/coi_cnn_seed1206_seen_test_target099_pdistance_experimental \
  --input-run-dir results/paper1_phylo_calibrated_assignment/pipeline_runs/coi_cnn_seed1206_eval_c_target099_pdistance_experimental \
  --input-run-dir results/paper1_phylo_calibrated_assignment/pipeline_runs/coi_cnn_seed1206_unseen_genera_target099_pdistance_experimental
