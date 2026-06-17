#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

ROOT="results/paper1_phylo_calibrated_assignment/encoder_benchmarks"
QUEUE_LOG="${ROOT}/encoder_benchmark_queue.log"
QUEUE_PID="${ROOT}/encoder_benchmark_queue.pid"
mkdir -p "${ROOT}/logs"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 encoder benchmark queue is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  TREE_NPZ="results/coi_fish_tree_clean_phylo_mamba_hier512_seqval/tree_embeddings.npz"
  if [[ ! -f "${TREE_NPZ}" ]]; then
    TREE_NPZ="results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval/tree_embeddings.npz"
  fi
  if [[ ! -f "${TREE_NPZ}" ]]; then
    echo "Missing shared tree embedding NPZ" >&2
    exit 1
  fi

  run_model () {
    local model_type="$1"
    local d_model="$2"
    local token_dim="$3"
    local layers="$4"
    local heads="$5"
    local batch="$6"
    local lr="$7"
    local run="${ROOT}/coi_${model_type}_seed1206"

    mkdir -p "${run}/logs"
    echo "[$(date -Iseconds)] ${model_type} train/eval_c"
    if [[ ! -f "${run}/run_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/train_fish_tree_encoder_benchmark.py train \
        --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
        --output-dir "${run}" \
        --tree-file data/phylo/actinopt_12k_treePL.tre \
        --tree-embedding-npz "${TREE_NPZ}" \
        --model-type "${model_type}" \
        --max-seq-len 700 \
        --embed-dim 512 \
        --d-model "${d_model}" \
        --token-emb-dim "${token_dim}" \
        --num-layers "${layers}" \
        --num-heads "${heads}" \
        --dropout 0.1 \
        --train-epochs 40 \
        --batch-size "${batch}" \
        --lr "${lr}" \
        --loss-mode cosine \
        --validation-mode random_sequence \
        --val-fraction 0.1 \
        --top-k 50 \
        --num-workers 8 \
        --seed 1206 \
        > "${run}/logs/train_eval_c.log" 2>&1
    fi

    for split in seen_test unseen_genera; do
      local out="${run}_${split}"
      echo "[$(date -Iseconds)] ${model_type} predict/${split}"
      if [[ ! -f "${out}/run_manifest.json" ]]; then
        .venv/bin/python -u scripts/edna/train_fish_tree_encoder_benchmark.py predict \
          --input-dir "data/phylo/fish_tree_clean_phylo_inputs/${split}" \
          --output-dir "${out}" \
          --tree-file data/phylo/actinopt_12k_treePL.tre \
          --tree-embedding-npz "${TREE_NPZ}" \
          --checkpoint "${run}/${model_type}_tree_encoder_best.pt" \
          --run-manifest "${run}/run_manifest.json" \
          --predict-batch-size 256 \
          --top-k 50 \
          --num-workers 8 \
          --seed 1206 \
          > "${run}/logs/predict_${split}.log" 2>&1
      fi
    done
  }

  run_model cnn 256 32 2 8 128 5e-4
  run_model bilstm 256 32 2 8 128 5e-4
  run_model transformer 256 32 2 8 96 3e-4

  echo "[$(date -Iseconds)] Paper 1 encoder benchmark queue complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 encoder benchmark queue PID $(cat "${QUEUE_PID}")"
