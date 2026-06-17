#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  cd "${REPO_ROOT}"
elif [[ -d /workspace/marinemamba ]]; then
  cd /workspace/marinemamba
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

ROOT="results/paper1_phylo_calibrated_assignment"
LOG_DIR="${ROOT}/pipeline_runs/logs"
mkdir -p "${LOG_DIR}"

run_pipeline() {
  local label="$1"
  local query_embeddings="$2"
  local output_dir="$3"
  shift 3
  echo "[$(date -Iseconds)] executable COI pipeline ${label}"
  "${PYTHON}" -u scripts/edna/run_paper1_coi_pipeline.py \
    --query-embeddings "${query_embeddings}" \
    --output-dir "${output_dir}" \
    --log-file "${LOG_DIR}/${output_dir##*/}.log" \
    "$@"
}

EVAL_C_EMBEDDINGS="results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings/coi_cnn_seed1206_eval_c/query_embeddings.npz"
SEEN_TEST_EMBEDDINGS="results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings/coi_cnn_seed1206_seen_test/query_embeddings.npz"
UNSEEN_GENERA_EMBEDDINGS="results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings/coi_cnn_seed1206_unseen_genera/query_embeddings.npz"

run_pipeline \
  "eval_c exact calibrated" \
  "${EVAL_C_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_eval_c_target099"

run_pipeline \
  "seen_test exact calibration" \
  "${SEEN_TEST_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_seen_test_target099"

run_pipeline \
  "unseen_genera exact calibrated" \
  "${UNSEEN_GENERA_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_unseen_genera_target099"

run_pipeline \
  "eval_c HNSW calibrated" \
  "${EVAL_C_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_eval_c_target099_hnsw" \
  --retrieval-mode hnsw

run_pipeline \
  "seen_test HNSW calibration" \
  "${SEEN_TEST_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_seen_test_target099_hnsw" \
  --retrieval-mode hnsw

run_pipeline \
  "unseen_genera HNSW calibrated" \
  "${UNSEEN_GENERA_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_unseen_genera_target099_hnsw" \
  --retrieval-mode hnsw

run_pipeline \
  "eval_c p-distance rerank experimental" \
  "${EVAL_C_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_eval_c_target099_pdistance_experimental" \
  --rerank-mode p_distance \
  --assignment-source reranked

run_pipeline \
  "seen_test p-distance rerank calibration" \
  "${SEEN_TEST_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_seen_test_target099_pdistance_experimental" \
  --rerank-mode p_distance \
  --assignment-source reranked

run_pipeline \
  "unseen_genera p-distance rerank experimental" \
  "${UNSEEN_GENERA_EMBEDDINGS}" \
  "${ROOT}/pipeline_runs/coi_cnn_seed1206_unseen_genera_target099_pdistance_experimental" \
  --rerank-mode p_distance \
  --assignment-source reranked

echo "[$(date -Iseconds)] executable COI pipeline source summary"
"${PYTHON}" -u scripts/edna/build_paper1_pipeline_run_summary.py \
  --log-file "${LOG_DIR}/build_paper1_pipeline_run_summary.log"
"${PYTHON}" -u scripts/edna/build_paper1_pipeline_benchmarks.py \
  --log-file "${LOG_DIR}/build_paper1_pipeline_benchmarks.log"
"${PYTHON}" -u scripts/edna/build_paper1_end_to_end_summary.py \
  --log-file "${LOG_DIR}/build_paper1_end_to_end_summary.log"

echo "[$(date -Iseconds)] executable COI pipeline complete"
