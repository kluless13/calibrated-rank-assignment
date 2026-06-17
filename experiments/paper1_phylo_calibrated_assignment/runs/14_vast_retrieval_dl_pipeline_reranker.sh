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
SWEEP_ROOT="${ROOT}/retrieval_dl_sweep"
PIPELINE_ROOT="${ROOT}/pipeline_runs"
RERANK_ROOT="${ROOT}/candidate_reranker"
BASELINE_ROOT="${ROOT}/classical_baselines"
QUEUE_ROOT="${ROOT}/retrieval_dl_pipeline_reranker"
QUEUE_LOG="${QUEUE_ROOT}/retrieval_dl_pipeline_reranker.log"
QUEUE_PID="${QUEUE_ROOT}/retrieval_dl_pipeline_reranker.pid"
mkdir -p "${QUEUE_ROOT}" "${PIPELINE_ROOT}" "${RERANK_ROOT}"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 retrieval-DL pipeline/reranker queue is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -u
  cd /workspace/marinemamba

  echo "[$(date -Iseconds)] Paper 1 retrieval-DL pipeline/reranker queue starting"
  "${PY}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda", flush=True)
PY

  for split in eval_c seen_test unseen_genera; do
    for method in blast vsearch; do
      path="${BASELINE_ROOT}/baselines_${split}/${method}/zero_shot_candidate_predictions.csv"
      if [[ ! -f "${path}" ]]; then
        echo "Missing baseline evidence: ${path}" >&2
        exit 1
      fi
    done
  done

  run_pipeline () {
    local arm="$1"
    local split="$2"
    local suffix=""
    if [[ "${split}" == "seen_test" ]]; then
      suffix="_seen_test"
    elif [[ "${split}" == "unseen_genera" ]]; then
      suffix="_unseen_genera"
    fi
    local emb="${SWEEP_ROOT}/${arm}${suffix}/query_embeddings.npz"
    local out="${PIPELINE_ROOT}/${arm}_${split}_target099_pdistance_experimental"
    local log="${QUEUE_ROOT}/${arm}_${split}_pipeline.log"
    if [[ ! -f "${emb}" ]]; then
      echo "Missing query embeddings: ${emb}" >&2
      exit 1
    fi
    if [[ -f "${out}/pipeline_manifest.json" ]]; then
      echo "[$(date -Iseconds)] SKIP pipeline arm=${arm} split=${split}"
      return 0
    fi
    echo "[$(date -Iseconds)] START pipeline arm=${arm} split=${split}"
    "${PY}" -u scripts/edna/run_paper1_coi_pipeline.py \
      --query-embeddings "${emb}" \
      --output-dir "${out}" \
      --top-k 50 \
      --rerank-mode p_distance \
      --assignment-source reranked \
      --log-file "${log}"
    echo "[$(date -Iseconds)] DONE pipeline arm=${arm} split=${split}"
  }

  train_reranker () {
    local arm="$1"
    local out="${RERANK_ROOT}/${arm}_top50_blast_vsearch"
    local log="${QUEUE_ROOT}/${arm}_reranker.log"
    if [[ -f "${out}/candidate_reranker_manifest.json" ]]; then
      echo "[$(date -Iseconds)] SKIP reranker arm=${arm}"
      return 0
    fi
    echo "[$(date -Iseconds)] START reranker arm=${arm}"
    "${PY}" -u scripts/edna/train_paper1_candidate_reranker.py \
      --output-dir "${out}" \
      --run "seen_test=${PIPELINE_ROOT}/${arm}_seen_test_target099_pdistance_experimental" \
      --run "eval_c=${PIPELINE_ROOT}/${arm}_eval_c_target099_pdistance_experimental" \
      --run "unseen_genera=${PIPELINE_ROOT}/${arm}_unseen_genera_target099_pdistance_experimental" \
      --baseline-root "${BASELINE_ROOT}" \
      --baseline-methods blast vsearch \
      --seed 1301 \
      --top-k 50 \
      --target-precision 0.99 \
      --epochs 80 \
      --patience 12 \
      --batch-size 8192 \
      --hidden-dim 128 \
      --dropout 0.1 \
      --log-file "${log}"
    echo "[$(date -Iseconds)] DONE reranker arm=${arm}"
  }

  for arm in \
    coi_cnn_retrieval_contrastive_seed1301 \
    coi_cnn_retrieval_hybrid_seed1301
  do
    for split in eval_c seen_test unseen_genera; do
      run_pipeline "${arm}" "${split}"
    done
    train_reranker "${arm}"
  done

  echo "[$(date -Iseconds)] Paper 1 retrieval-DL pipeline/reranker queue complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 retrieval-DL pipeline/reranker queue PID $(cat "${QUEUE_PID}")"
