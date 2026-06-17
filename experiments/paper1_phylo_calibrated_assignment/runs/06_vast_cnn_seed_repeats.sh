#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

ROOT="results/paper1_phylo_calibrated_assignment/cnn_seed_repeats"
QUEUE_LOG="${ROOT}/cnn_seed_repeats.log"
QUEUE_PID="${ROOT}/cnn_seed_repeats.pid"
mkdir -p "${ROOT}/logs"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 CNN seed-repeat queue is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  TREE_NPZ="results/coi_fish_tree_clean_phylo_mamba_hier512_seqval/tree_embeddings.npz"
  if [[ ! -f "${TREE_NPZ}" ]]; then
    echo "Missing shared tree embedding NPZ: ${TREE_NPZ}" >&2
    exit 1
  fi

  run_seed () {
    local seed="$1"
    local run="${ROOT}/coi_cnn_seed${seed}"

    mkdir -p "${run}/logs"
    echo "[$(date -Iseconds)] cnn seed ${seed} train/eval_c"
    if [[ ! -f "${run}/run_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/train_fish_tree_encoder_benchmark.py train \
        --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
        --output-dir "${run}" \
        --tree-file data/phylo/actinopt_12k_treePL.tre \
        --tree-embedding-npz "${TREE_NPZ}" \
        --model-type cnn \
        --max-seq-len 700 \
        --embed-dim 512 \
        --d-model 256 \
        --token-emb-dim 32 \
        --num-layers 2 \
        --num-heads 8 \
        --dropout 0.1 \
        --train-epochs 40 \
        --batch-size 128 \
        --lr 5e-4 \
        --loss-mode cosine \
        --validation-mode random_sequence \
        --val-fraction 0.1 \
        --top-k 50 \
        --num-workers 2 \
        --seed "${seed}" \
        --write-query-embeddings \
        > "${run}/logs/train_eval_c.log" 2>&1
    fi

    for split in seen_test unseen_genera; do
      local out="${run}_${split}"
      echo "[$(date -Iseconds)] cnn seed ${seed} predict/${split}"
      if [[ ! -f "${out}/run_manifest.json" ]]; then
        .venv/bin/python -u scripts/edna/train_fish_tree_encoder_benchmark.py predict \
          --input-dir "data/phylo/fish_tree_clean_phylo_inputs/${split}" \
          --output-dir "${out}" \
          --tree-file data/phylo/actinopt_12k_treePL.tre \
          --tree-embedding-npz "${TREE_NPZ}" \
          --checkpoint "${run}/cnn_tree_encoder_best.pt" \
          --run-manifest "${run}/run_manifest.json" \
          --predict-batch-size 256 \
          --top-k 50 \
          --num-workers 2 \
          --seed "${seed}" \
          --write-query-embeddings \
          > "${run}/logs/predict_${split}.log" 2>&1
      fi
    done

    for split in eval_c unseen_genera; do
      local input_dir="data/phylo/fish_tree_clean_phylo_inputs/${split}"
      local out="${run}_tree_recovery_${split}"
      echo "[$(date -Iseconds)] cnn seed ${seed} tree recovery/${split}"
      if [[ ! -f "${out}/tree_recovery_metrics.json" ]]; then
        .venv/bin/python -u scripts/edna/eval_fish_tree_encoder_tree_recovery.py \
          --input-dir "${input_dir}" \
          --checkpoint "${run}/cnn_tree_encoder_best.pt" \
          --run-manifest "${run}/run_manifest.json" \
          --output-dir "${out}" \
          --tree-file data/phylo/actinopt_12k_treePL.tre \
          --seqs-per-species 10 \
          --max-pairs 50000 \
          --batch-size 256 \
          --num-workers 2 \
          --seed "${seed}" \
          > "${run}/logs/tree_recovery_${split}.log" 2>&1
      fi
    done
  }

  .venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda")
PY

  run_seed 1207
  run_seed 1208

  echo "[$(date -Iseconds)] Paper 1 CNN seed-repeat queue complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 CNN seed-repeat queue PID $(cat "${QUEUE_PID}")"
