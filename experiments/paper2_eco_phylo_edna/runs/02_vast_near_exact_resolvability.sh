#!/usr/bin/env bash
set -euo pipefail

ROOT="results/edna/resolvability_near_exact"
LOG_DIR="${ROOT}/logs"
WRAPPER_PID="${ROOT}/near_exact_resolvability.pid"
WRAPPER_LOG="${LOG_DIR}/near_exact_resolvability.log"

mkdir -p "${LOG_DIR}"

if [[ -f "${WRAPPER_PID}" ]] && kill -0 "$(cat "${WRAPPER_PID}")" 2>/dev/null; then
  echo "near-exact 12S resolvability already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

run_queue() {
  echo "[$(date -Iseconds)] near-exact 12S resolvability start"
  command -v vsearch
  vsearch --version 2>&1 | head -n 2 || true

  for dataset in multisource multisource_teleo rcrux_cleaned mitohelper_full_teleo; do
    input_dir="data/edna/stalder_inputs/${dataset}"
    output_dir="${ROOT}/${dataset}_near_exact_acgt"
    if [[ ! -f "${output_dir}/near_exact_resolvability_manifest.json" ]]; then
      echo "[$(date -Iseconds)] ${dataset}"
      .venv/bin/python -u scripts/edna/build_12s_near_exact_resolvability.py \
        --input-dir "${input_dir}" \
        --output-dir "${output_dir}" \
        --identities 0.99,0.98,0.97,0.95 \
        --normalization acgt \
        --min-length 30 \
        --threads 64 \
        > "${LOG_DIR}/${dataset}.log" 2>&1
    else
      echo "[$(date -Iseconds)] ${dataset} already complete"
    fi
  done
  echo "[$(date -Iseconds)] near-exact 12S resolvability complete"
}

nohup bash -c "$(declare -f run_queue); ROOT='${ROOT}'; LOG_DIR='${LOG_DIR}'; run_queue" > "${WRAPPER_LOG}" 2>&1 &
echo $! > "${WRAPPER_PID}"
echo "launched near-exact 12S resolvability PID $(cat "${WRAPPER_PID}")"

