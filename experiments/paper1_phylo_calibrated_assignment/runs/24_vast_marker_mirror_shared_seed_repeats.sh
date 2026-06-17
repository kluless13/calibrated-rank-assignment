#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

ROOT="results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/shared_space_seed_repeats"
mkdir -p "${ROOT}"
QUEUE_LOG="${ROOT}/shared_space_seed_repeats.log"

{
  echo "[$(date -Is)] Starting shared 12S/16S MarkerMirror seed repeats"
  for SEED in 1902 1903; do
    OUT="results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best_seed${SEED}"
    if [[ -f "${OUT}/marker_mirror_shared_retrieval_metrics.csv" ]]; then
      echo "[$(date -Is)] seed=${SEED} already complete: ${OUT}"
      continue
    fi
    mkdir -p "${OUT}"
    echo "[$(date -Is)] seed=${SEED} launching"
    python3 -u scripts/edna/train_marker_mirror_shared_space.py \
      --marker-a-input-dir data/edna/stalder_inputs/multisource \
      --marker-b-input-dir data/edna/stalder_inputs/16s_multisource \
      --marker-a-name 12S \
      --marker-b-name 16S \
      --output-dir "${OUT}" \
      --batch-strategy taxonomy_hard \
      --loss-mode taxonomy_soft \
      --restore-best-retrieval \
      --retrieval-eval-every 10 \
      --retrieval-selection-ranks genus,family,order \
      --retrieval-selection-k 10 \
      --epochs 120 \
      --seed "${SEED}" \
      --log-file "${OUT}/marker_mirror_shared_space.log"
    echo "[$(date -Is)] seed=${SEED} done"
  done
  echo "[$(date -Is)] Completed shared 12S/16S MarkerMirror seed repeats"
} >> "${QUEUE_LOG}" 2>&1
