#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

RUN_ROOT="results/edna/taxdna_ssm"
WRAPPER_LOG="${RUN_ROOT}/broad_cnn_seed_repeats_wrapper.log"
WRAPPER_PID="${RUN_ROOT}/broad_cnn_seed_repeats_wrapper.pid"
TREE_NPZ="${RUN_ROOT}/multisource_tree_embeddings.npz"
mkdir -p "${RUN_ROOT}/logs"

if [[ -f "${WRAPPER_PID}" ]] && ps -p "$(cat "${WRAPPER_PID}")" >/dev/null 2>&1; then
  echo "broad CNN seed-repeat wrapper is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  if [[ ! -f "${TREE_NPZ}" ]]; then
    .venv/bin/python -u scripts/edna/learn_tree_embedding_npz.py \
      --input-dir data/edna/stalder_inputs/multisource \
      --output-npz "${TREE_NPZ}" \
      --embed-dim 128 \
      --tree-epochs 300 \
      > "${RUN_ROOT}/logs/multisource_tree_embeddings_broad_cnn_seed_repeats.log" 2>&1
  fi

  for seed in 1207 1208; do
    cnn_out="${RUN_ROOT}/multisource_cnn_seed${seed}"
    if [[ ! -f "${cnn_out}/run_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/train_taxdna_cnn_baseline.py \
        --input-dir data/edna/stalder_inputs/multisource \
        --output-dir "${cnn_out}" \
        --max-seq-len 2048 \
        --embed-dim 128 \
        --tree-embedding-npz "${TREE_NPZ}" \
        --train-epochs 50 \
        --batch-size 8 \
        --top-k 50 \
        --seed "${seed}" \
        --write-query-embeddings \
        > "${RUN_ROOT}/logs/multisource_cnn_seed${seed}.log" 2>&1
    fi
  done
) > "${WRAPPER_LOG}" 2>&1 &

echo $! > "${WRAPPER_PID}"
echo "launched broad CNN seed-repeat wrapper PID $(cat "${WRAPPER_PID}")"
