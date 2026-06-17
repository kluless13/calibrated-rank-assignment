#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

ROOT="results/paper1_phylo_calibrated_assignment/encoder_benchmarks"
QUEUE_PID="${ROOT}/encoder_benchmark_queue.pid"
FOLLOW_ROOT="${ROOT}/tree_recovery_followup"
FOLLOW_LOG="${FOLLOW_ROOT}/tree_recovery_followup.log"
FOLLOW_PID="${FOLLOW_ROOT}/tree_recovery_followup.pid"
mkdir -p "${FOLLOW_ROOT}/logs"

if [[ -f "${FOLLOW_PID}" ]] && ps -p "$(cat "${FOLLOW_PID}")" >/dev/null 2>&1; then
  echo "Encoder tree-recovery follow-up is already running: PID $(cat "${FOLLOW_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  if [[ -f "${QUEUE_PID}" ]]; then
    queue_pid="$(cat "${QUEUE_PID}")"
    echo "[$(date -Iseconds)] waiting for encoder benchmark queue PID ${queue_pid}"
    while ps -p "${queue_pid}" >/dev/null 2>&1; do
      sleep 60
    done
  fi

  for model_type in cnn bilstm transformer; do
    run="${ROOT}/coi_${model_type}_seed1206"
    checkpoint="${run}/${model_type}_tree_encoder_best.pt"
    manifest="${run}/run_manifest.json"
    if [[ ! -f "${checkpoint}" || ! -f "${manifest}" ]]; then
      echo "[$(date -Iseconds)] missing ${model_type} checkpoint or manifest; skipping"
      continue
    fi

    for split in eval_c unseen_genera; do
      input_dir="data/phylo/fish_tree_clean_phylo_inputs/${split}"
      out="${run}_tree_recovery_${split}"
      if [[ ! -f "${out}/tree_recovery_metrics.json" ]]; then
        echo "[$(date -Iseconds)] ${model_type} tree recovery ${split}"
        .venv/bin/python -u scripts/edna/eval_fish_tree_encoder_tree_recovery.py \
          --input-dir "${input_dir}" \
          --checkpoint "${checkpoint}" \
          --run-manifest "${manifest}" \
          --output-dir "${out}" \
          --tree-file data/phylo/actinopt_12k_treePL.tre \
          --seqs-per-species 10 \
          --max-pairs 50000 \
          --batch-size 256 \
          --num-workers 8 \
          --seed 1206 \
          > "${FOLLOW_ROOT}/logs/${model_type}_${split}_tree_recovery.log" 2>&1
      fi
    done
  done

  echo "[$(date -Iseconds)] encoder tree-recovery follow-up complete"
) > "${FOLLOW_LOG}" 2>&1 &

echo $! > "${FOLLOW_PID}"
echo "launched encoder tree-recovery follow-up PID $(cat "${FOLLOW_PID}")"
