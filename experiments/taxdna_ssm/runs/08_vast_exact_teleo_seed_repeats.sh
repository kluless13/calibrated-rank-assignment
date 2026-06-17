#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

RUN_ROOT="results/edna/taxdna_ssm"
WRAPPER_LOG="${RUN_ROOT}/exact_teleo_seed_repeats_wrapper.log"
WRAPPER_PID="${RUN_ROOT}/exact_teleo_seed_repeats_wrapper.pid"
TREE_NPZ="${RUN_ROOT}/multisource_teleo_tree_embeddings.npz"
mkdir -p "${RUN_ROOT}/logs"

if [[ -f "${WRAPPER_PID}" ]] && ps -p "$(cat "${WRAPPER_PID}")" >/dev/null 2>&1; then
  echo "exact Teleo seed-repeat wrapper is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  if [[ ! -f "${TREE_NPZ}" ]]; then
    .venv/bin/python -u scripts/edna/learn_tree_embedding_npz.py \
      --input-dir data/edna/stalder_inputs/multisource_teleo \
      --output-npz "${TREE_NPZ}" \
      --embed-dim 128 \
      --tree-epochs 300 \
      > "${RUN_ROOT}/logs/multisource_teleo_tree_embeddings_seed_repeats.log" 2>&1
  fi

  for seed in 1207 1208; do
    ssm_out="${RUN_ROOT}/multisource_teleo_ssm_contrastive_seed${seed}"
    cnn_out="${RUN_ROOT}/multisource_teleo_cnn_seed${seed}"

    if [[ ! -f "${ssm_out}/run_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/train_12s_phylo_mamba.py \
        --input-dir data/edna/stalder_inputs/multisource_teleo \
        --output-dir "${ssm_out}" \
        --max-seq-len 128 \
        --embed-dim 128 \
        --tree-embedding-npz "${TREE_NPZ}" \
        --train-epochs 50 \
        --batch-size 64 \
        --loss-mode contrastive \
        --temperature 0.07 \
        --pooling masked_mean \
        --top-k 50 \
        --seed "${seed}" \
        --write-query-embeddings \
        > "${RUN_ROOT}/logs/multisource_teleo_ssm_contrastive_seed${seed}.log" 2>&1
    fi

    if [[ ! -f "${cnn_out}/run_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/train_taxdna_cnn_baseline.py \
        --input-dir data/edna/stalder_inputs/multisource_teleo \
        --output-dir "${cnn_out}" \
        --max-seq-len 128 \
        --embed-dim 128 \
        --tree-embedding-npz "${TREE_NPZ}" \
        --train-epochs 50 \
        --batch-size 16 \
        --top-k 50 \
        --seed "${seed}" \
        --write-query-embeddings \
        > "${RUN_ROOT}/logs/multisource_teleo_cnn_seed${seed}.log" 2>&1
    fi
  done
) > "${WRAPPER_LOG}" 2>&1 &

echo $! > "${WRAPPER_PID}"
echo "launched exact Teleo seed-repeat wrapper PID $(cat "${WRAPPER_PID}")"
