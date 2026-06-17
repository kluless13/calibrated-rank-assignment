#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  cd "${REPO_ROOT}"
elif [[ -d /workspace/marinemamba ]]; then
  cd /workspace/marinemamba
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

PY="${PYTHON:-/venv/main/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

ROOT="results/paper1_phylo_calibrated_assignment"
RERANK_ROOT="${ROOT}/candidate_reranker"
CAL_ROOT="${ROOT}/candidate_assignment_calibrator"
QUEUE_ROOT="${ROOT}/candidate_assignment_calibrators"
QUEUE_LOG="${QUEUE_ROOT}/candidate_assignment_calibrators.log"
QUEUE_PID="${QUEUE_ROOT}/candidate_assignment_calibrators.pid"
mkdir -p "${QUEUE_ROOT}" "${CAL_ROOT}"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 candidate-assignment calibrator queue is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -u
  cd /workspace/marinemamba

  echo "[$(date -Iseconds)] Paper 1 candidate-assignment calibrator queue starting"
  "${PY}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda", flush=True)
PY

  declare -a SPECS=(
    "pointwise_tree10|coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10"
    "pairwise_tree10|coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10_pairwise"
  )

  for spec in "${SPECS[@]}"; do
    label="${spec%%|*}"
    reranker="${spec##*|}"
    selected="${RERANK_ROOT}/${reranker}/candidate_reranker_selected_predictions.csv"
    out="${CAL_ROOT}/${reranker}_assignment_calibrator"
    if [[ ! -f "${selected}" ]]; then
      echo "Missing selected predictions for ${label}: ${selected}" >&2
      exit 1
    fi
    if [[ -f "${out}/candidate_assignment_calibrator_manifest.json" ]]; then
      echo "[$(date -Iseconds)] SKIP candidate assignment calibrator ${label}"
      continue
    fi
    echo "[$(date -Iseconds)] RUN candidate assignment calibrator ${label}"
    "${PY}" -u scripts/edna/train_paper1_candidate_assignment_calibrator.py \
      --selected-predictions "${selected}" \
      --output-dir "${out}" \
      --seed 1301 \
      --target-precision 0.99 \
      --epochs 80 \
      --patience 12 \
      --batch-size 4096 \
      --hidden-dim 96 \
      --dropout 0.1 \
      --log-file "${QUEUE_ROOT}/${label}_assignment_calibrator.log"
    echo "[$(date -Iseconds)] DONE candidate assignment calibrator ${label}"
  done

  echo "[$(date -Iseconds)] Paper 1 candidate-assignment calibrator queue complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 candidate-assignment calibrator queue PID $(cat "${QUEUE_PID}")"
