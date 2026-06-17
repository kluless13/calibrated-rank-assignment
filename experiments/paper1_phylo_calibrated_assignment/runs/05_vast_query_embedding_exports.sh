#!/usr/bin/env bash
set -euo pipefail

ROOT="results/paper1_phylo_calibrated_assignment/query_embeddings"
ENC_ROOT="results/paper1_phylo_calibrated_assignment/encoder_benchmarks"
LOG_DIR="${ROOT}/logs"
WRAPPER_PID="${ROOT}/query_embedding_exports.pid"
WRAPPER_LOG="${LOG_DIR}/query_embedding_exports.log"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-256}"

mkdir -p "${LOG_DIR}"

if [[ -f "${WRAPPER_PID}" ]] && kill -0 "$(cat "${WRAPPER_PID}")" 2>/dev/null; then
  echo "Paper 1 query embedding export queue is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

run_export() {
  local model_type="$1"
  local run_name="$2"
  local checkpoint_name="$3"
  local split="$4"

  local input_dir="data/phylo/fish_tree_clean_phylo_inputs/${split}"
  local source_run="${ENC_ROOT}/${run_name}"
  local output_dir="${ROOT}/${run_name}_${split}"
  local log_file="${LOG_DIR}/${run_name}_${split}.log"

  if [[ -f "${output_dir}/query_embeddings.npz" && -f "${output_dir}/run_manifest.json" ]]; then
    echo "[$(date -Iseconds)] skip existing ${run_name} ${split}"
    return
  fi

  echo "[$(date -Iseconds)] export ${run_name} ${split}"
  .venv/bin/python -u scripts/edna/train_fish_tree_encoder_benchmark.py predict \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --model-type "${model_type}" \
    --checkpoint "${source_run}/${checkpoint_name}" \
    --run-manifest "${source_run}/run_manifest.json" \
    --write-query-embeddings \
    --predict-batch-size "${PREDICT_BATCH_SIZE}" \
    > "${log_file}" 2>&1
}

run_queue() {
  echo "[$(date -Iseconds)] Paper 1 query embedding export queue start"
  .venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda")
PY

  for split in eval_c seen_test unseen_genera; do
    run_export cnn coi_cnn_seed1206 cnn_tree_encoder_best.pt "${split}"
    run_export bilstm coi_bilstm_seed1206 bilstm_tree_encoder_best.pt "${split}"
    run_export transformer coi_transformer_seed1206 transformer_tree_encoder_best.pt "${split}"
  done

  echo "[$(date -Iseconds)] Paper 1 query embedding export queue complete"
}

nohup bash -c "$(declare -f run_export); $(declare -f run_queue); ROOT='${ROOT}'; ENC_ROOT='${ENC_ROOT}'; LOG_DIR='${LOG_DIR}'; PREDICT_BATCH_SIZE='${PREDICT_BATCH_SIZE}'; run_queue" > "${WRAPPER_LOG}" 2>&1 &
echo $! > "${WRAPPER_PID}"
echo "launched Paper 1 query embedding export queue PID $(cat "${WRAPPER_PID}")"
