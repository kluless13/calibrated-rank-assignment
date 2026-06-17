#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

ROOT="${ROOT:-results/paper1_phylo_calibrated_assignment/strict_missing_reference_cnn}"
INPUT_MANIFEST="${INPUT_MANIFEST:-data/phylo/paper1_strict_missing_reference_inputs/strict_missing_reference_manifest.csv}"
TREE_FILE="${TREE_FILE:-data/phylo/actinopt_12k_treePL.tre}"
MAX_PACKS="${MAX_PACKS:-0}"
PACK_FILTER="${PACK_FILTER:-}"
SEED="${SEED:-1206}"
QUEUE_PID="${ROOT}/strict_missing_reference_cnn.pid"
QUEUE_LOG="${ROOT}/strict_missing_reference_cnn.log"

mkdir -p "${ROOT}/logs"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 strict missing-reference CNN queue is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  if [[ ! -f "${INPUT_MANIFEST}" ]]; then
    echo "Missing strict input manifest: ${INPUT_MANIFEST}" >&2
    exit 1
  fi

  .venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda")
PY

  count=0
  tail -n +2 "${INPUT_MANIFEST}" | while IFS=, read -r name split hide_rank input_dir hidden_candidate_species kept_candidate_species kept_train_species query_rows query_species; do
    if [[ -n "${PACK_FILTER}" && "${name}" != *"${PACK_FILTER}"* ]]; then
      continue
    fi
    count=$((count + 1))
    if [[ "${MAX_PACKS}" != "0" && "${count}" -gt "${MAX_PACKS}" ]]; then
      break
    fi

    run="${ROOT}/${name}_cnn_seed${SEED}"
    mkdir -p "${run}/logs"
    echo "[$(date -Iseconds)] strict missing-reference train/eval ${name} split=${split} hide=${hide_rank} kept_train=${kept_train_species}"

    if [[ ! -f "${run}/run_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/train_fish_tree_encoder_benchmark.py train \
        --input-dir "${input_dir}" \
        --output-dir "${run}" \
        --tree-file "${TREE_FILE}" \
        --model-type cnn \
        --max-seq-len 700 \
        --embed-dim 512 \
        --d-model 256 \
        --token-emb-dim 32 \
        --num-layers 2 \
        --num-heads 8 \
        --dropout 0.1 \
        --tree-epochs 300 \
        --tree-pairs-per-epoch 768 \
        --tree-negatives 32 \
        --tree-max-distance-sample 500 \
        --train-epochs 40 \
        --batch-size 128 \
        --lr 5e-4 \
        --loss-mode cosine \
        --validation-mode random_sequence \
        --val-fraction 0.1 \
        --top-k 50 \
        --num-workers 2 \
        --seed "${SEED}" \
        --write-query-embeddings \
        > "${run}/logs/train_${name}.log" 2>&1
    fi
  done

  echo "[$(date -Iseconds)] Paper 1 strict missing-reference CNN queue complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 strict missing-reference CNN queue PID $(cat "${QUEUE_PID}")"
