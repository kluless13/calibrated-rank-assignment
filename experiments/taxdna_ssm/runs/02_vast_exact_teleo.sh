#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

RUN_ROOT="results/edna/taxdna_ssm"
WRAPPER_LOG="${RUN_ROOT}/exact_teleo_wrapper.log"
WRAPPER_PID="${RUN_ROOT}/exact_teleo_wrapper.pid"
mkdir -p "${RUN_ROOT}"

if [[ -f "${WRAPPER_PID}" ]] && ps -p "$(cat "${WRAPPER_PID}")" >/dev/null 2>&1; then
  echo "exact Teleo wrapper is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba
  mkdir -p "${RUN_ROOT}/logs"

  .venv/bin/python -u scripts/edna/learn_tree_embedding_npz.py \
    --input-dir data/edna/stalder_inputs/multisource_teleo \
    --output-npz "${RUN_ROOT}/multisource_teleo_tree_embeddings.npz" \
    --embed-dim 128 \
    --tree-epochs 300 \
    > "${RUN_ROOT}/logs/multisource_teleo_tree_embeddings.log" 2>&1

  .venv/bin/python -u scripts/edna/train_12s_phylo_mamba.py \
    --input-dir data/edna/stalder_inputs/multisource_teleo \
    --output-dir "${RUN_ROOT}/multisource_teleo_ssm_contrastive" \
    --max-seq-len 128 \
    --embed-dim 128 \
    --tree-embedding-npz "${RUN_ROOT}/multisource_teleo_tree_embeddings.npz" \
    --train-epochs 50 \
    --batch-size 64 \
    --loss-mode contrastive \
    --temperature 0.07 \
    --pooling masked_mean \
    --top-k 50 \
    --write-query-embeddings \
    > "${RUN_ROOT}/logs/multisource_teleo_ssm_contrastive.log" 2>&1

  .venv/bin/python -u scripts/edna/train_taxdna_cnn_baseline.py \
    --input-dir data/edna/stalder_inputs/multisource_teleo \
    --output-dir "${RUN_ROOT}/multisource_teleo_cnn" \
    --max-seq-len 128 \
    --embed-dim 128 \
    --tree-embedding-npz "${RUN_ROOT}/multisource_teleo_tree_embeddings.npz" \
    --train-epochs 50 \
    --batch-size 16 \
    --top-k 50 \
    --write-query-embeddings \
    > "${RUN_ROOT}/logs/multisource_teleo_cnn.log" 2>&1
) > "${WRAPPER_LOG}" 2>&1 &

echo $! > "${WRAPPER_PID}"
echo "launched exact Teleo wrapper PID $(cat "${WRAPPER_PID}")"
