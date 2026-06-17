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
PIPELINE_ROOT="${ROOT}/pipeline_runs"
RERANK_ROOT="${ROOT}/candidate_reranker"
BASELINE_ROOT="${ROOT}/classical_baselines"
QUEUE_ROOT="${ROOT}/retrieval_dl_listwise_tree_reranker"
QUEUE_LOG="${QUEUE_ROOT}/retrieval_dl_listwise_tree_reranker.log"
QUEUE_PID="${QUEUE_ROOT}/retrieval_dl_listwise_tree_reranker.pid"
TREE_FILE="data/phylo/actinopt_12k_treePL.tre"
ARM="coi_cnn_retrieval_hybrid_seed1301"
OUT="${RERANK_ROOT}/${ARM}_top50_blast_vsearch_tree10_listwise"
mkdir -p "${QUEUE_ROOT}" "${RERANK_ROOT}"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 retrieval-DL listwise tree reranker is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -u
  cd /workspace/marinemamba

  echo "[$(date -Iseconds)] Paper 1 retrieval-DL listwise tree reranker starting"
  "${PY}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda", flush=True)
PY

  if [[ ! -f "${TREE_FILE}" ]]; then
    echo "Missing tree file: ${TREE_FILE}" >&2
    exit 1
  fi
  for split in seen_test eval_c unseen_genera; do
    if [[ ! -f "${PIPELINE_ROOT}/${ARM}_${split}_target099_pdistance_experimental/pipeline_manifest.json" ]]; then
      echo "Missing pipeline run for arm=${ARM} split=${split}" >&2
      exit 1
    fi
    for method in blast vsearch; do
      path="${BASELINE_ROOT}/baselines_${split}/${method}/zero_shot_candidate_predictions.csv"
      if [[ ! -f "${path}" ]]; then
        echo "Missing baseline evidence: ${path}" >&2
        exit 1
      fi
    done
  done

  if [[ -f "${OUT}/candidate_reranker_manifest.json" ]]; then
    echo "[$(date -Iseconds)] SKIP listwise reranker arm=${ARM}"
    exit 0
  fi

  "${PY}" -u scripts/edna/train_paper1_candidate_listwise_reranker.py \
    --output-dir "${OUT}" \
    --run "seen_test=${PIPELINE_ROOT}/${ARM}_seen_test_target099_pdistance_experimental" \
    --run "eval_c=${PIPELINE_ROOT}/${ARM}_eval_c_target099_pdistance_experimental" \
    --run "unseen_genera=${PIPELINE_ROOT}/${ARM}_unseen_genera_target099_pdistance_experimental" \
    --baseline-root "${BASELINE_ROOT}" \
    --baseline-methods blast vsearch \
    --tree-neighborhood-features \
    --tree-file "${TREE_FILE}" \
    --tree-neighborhood-size 10 \
    --seed 1301 \
    --top-k 50 \
    --target-precision 0.99 \
    --epochs 80 \
    --patience 12 \
    --group-batch-size 512 \
    --batch-size 8192 \
    --hidden-dim 128 \
    --dropout 0.1 \
    --log-file "${QUEUE_ROOT}/${ARM}_tree10_listwise.log"

  echo "[$(date -Iseconds)] Paper 1 retrieval-DL listwise tree reranker complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 retrieval-DL listwise tree reranker PID $(cat "${QUEUE_PID}")"
