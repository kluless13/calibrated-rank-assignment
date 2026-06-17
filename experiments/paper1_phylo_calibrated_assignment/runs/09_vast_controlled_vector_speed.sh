#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

ROOT="${ROOT:-results/paper1_phylo_calibrated_assignment/controlled_vector_speed}"
QUEUE_PID="${ROOT}/controlled_vector_speed.pid"
QUEUE_LOG="${ROOT}/controlled_vector_speed.log"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-2}"
THREADS="${THREADS:-8}"

mkdir -p "${ROOT}/logs"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 controlled vector speed queue is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  echo "[$(date -Iseconds)] controlled vector speed benchmark start"
  .venv/bin/python -u scripts/edna/build_controlled_vector_speed_benchmark.py \
    --output-dir results/paper1_phylo_calibrated_assignment/source_tables \
    --repeats "${REPEATS}" \
    --warmup "${WARMUP}" \
    --hnsw-m 16 32 \
    --hnsw-ef-search 50 \
    --threads "${THREADS}" \
    --log-file "${ROOT}/logs/build_controlled_vector_speed_benchmark.log"

  .venv/bin/python -u scripts/edna/build_paper1_pipeline_benchmarks.py \
    --log-file "${ROOT}/logs/build_paper1_pipeline_benchmarks.log"
  .venv/bin/python -u scripts/edna/build_paper1_end_to_end_summary.py \
    --log-file "${ROOT}/logs/build_paper1_end_to_end_summary.log"
  echo "[$(date -Iseconds)] controlled vector speed benchmark complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 controlled vector speed queue PID $(cat "${QUEUE_PID}")"
