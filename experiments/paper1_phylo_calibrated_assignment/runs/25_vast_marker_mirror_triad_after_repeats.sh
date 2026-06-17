#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

WAIT_ROOT="results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/shared_space_seed_repeats"
OUT="results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best"
LOG_ROOT="results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/triad_after_repeats"
mkdir -p "${OUT}" "${LOG_ROOT}"
QUEUE_LOG="${LOG_ROOT}/triad_after_repeats.log"

{
  echo "[$(date -Is)] Triad queue started"
  if [[ -f "${WAIT_ROOT}/shared_space_seed_repeats.pid" ]]; then
    WAIT_PID="$(cat "${WAIT_ROOT}/shared_space_seed_repeats.pid")"
    while ps -p "${WAIT_PID}" >/dev/null 2>&1; do
      echo "[$(date -Is)] Waiting for shared seed-repeat queue PID ${WAIT_PID}"
      sleep 60
    done
    echo "[$(date -Is)] Shared seed-repeat queue finished or exited"
  else
    echo "[$(date -Is)] No shared seed-repeat PID found; launching triad immediately"
  fi

  if [[ -f "${OUT}/marker_mirror_triad_retrieval_metrics.csv" ]]; then
    echo "[$(date -Is)] Triad output already exists: ${OUT}"
    exit 0
  fi

  echo "[$(date -Is)] Launching 12S/16S/COI tri-marker shared-space prototype"
  python3 -u scripts/edna/train_marker_mirror_triad_space.py \
    --marker-a-name 12S \
    --marker-a-input-dir data/edna/stalder_inputs/multisource \
    --marker-b-name 16S \
    --marker-b-input-dir data/edna/stalder_inputs/16s_multisource \
    --marker-c-name COI \
    --marker-c-input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
    --output-dir "${OUT}" \
    --batch-strategy taxonomy_hard \
    --loss-mode taxonomy_soft \
    --restore-best-retrieval \
    --retrieval-eval-every 10 \
    --retrieval-selection-ranks genus,family,order \
    --retrieval-selection-k 10 \
    --epochs 120 \
    --seed 2001 \
    --log-file "${OUT}/marker_mirror_triad_space.log"
  echo "[$(date -Is)] Completed 12S/16S/COI tri-marker shared-space prototype"
} >> "${QUEUE_LOG}" 2>&1
