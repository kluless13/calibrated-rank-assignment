#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-results/paper1_phylo_calibrated_assignment/fernando_completeness_sweeps}"
INPUT_ROOT="${INPUT_ROOT:-data/phylo/fernando_completeness_sweeps}"
LOG_DIR="${ROOT}/logs"
WRAPPER_PID="${ROOT}/fernando_apples_queue.pid"
WRAPPER_LOG="${LOG_DIR}/fernando_apples_queue.log"
THREADS="${THREADS:-8}"
MAX_SWEEPS="${MAX_SWEEPS:-0}"
SWEEP_FILTER="${SWEEP_FILTER:-}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-86400}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DISABLE_REESTIMATION="${DISABLE_REESTIMATION:-0}"

mkdir -p "${LOG_DIR}"
export PATH="$(pwd)/.venv/bin:${PATH}"

if [[ -f "${WRAPPER_PID}" ]] && kill -0 "$(cat "${WRAPPER_PID}")" 2>/dev/null; then
  echo "Fernando APPLES sweep queue is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

wait_for_alignment() {
  local split="$1"
  local split_root="${ROOT}/${split}"
  local waited=0
  while [[ ! -f "${split_root}/reference_msa.fasta" || ! -f "${split_root}/query_msa.fasta" || ! -f "${split_root}/reference_tree.nwk" ]]; do
    if [[ "${waited}" -ge "${WAIT_TIMEOUT_SECONDS}" ]]; then
      echo "[$(date -Iseconds)] timeout waiting for prepared alignment for ${split}" >&2
      return 1
    fi
    echo "[$(date -Iseconds)] waiting for EPA-ng prepared alignment for ${split} (${waited}s)"
    sleep "${POLL_SECONDS}"
    waited=$((waited + POLL_SECONDS))
  done
}

make_apples_tree() {
  local split_root="$1"
  local apples_tree="${split_root}/reference_tree_unrooted_for_apples.nwk"
  if [[ -f "${apples_tree}" ]]; then
    return 0
  fi
  .venv/bin/python -u scripts/edna/prepare_fish_tree_placement_inputs.py deroot-tree \
    --input-tree "${split_root}/reference_tree.nwk" \
    --output-tree "${apples_tree}"
}

run_split_apples() {
  local split="$1"
  local split_root="${ROOT}/${split}"
  local apples_root="${split_root}/apples"
  local apples_tree="${split_root}/reference_tree_unrooted_for_apples.nwk"
  mkdir -p "${apples_root}"

  wait_for_alignment "${split}"
  make_apples_tree "${split_root}"

  if [[ -f "${apples_root}/apples.jplace" ]]; then
    echo "[$(date -Iseconds)] APPLES already complete ${split}"
    return 0
  fi

  echo "[$(date -Iseconds)] APPLES ${split}"
  local apples_args=(
    -s "${split_root}/reference_msa.fasta" \
    -q "${split_root}/query_msa.fasta" \
    -t "${apples_tree}" \
    -o "${apples_root}/apples.jplace" \
    -T "${THREADS}"
  )
  if [[ "${DISABLE_REESTIMATION}" == "1" ]]; then
    apples_args+=(-D)
  fi
  .venv/bin/run_apples.py "${apples_args[@]}" > "${LOG_DIR}/${split}_apples.log" 2>&1
}

run_queue() {
  echo "[$(date -Iseconds)] Fernando APPLES sweep queue start"
  echo "[$(date -Iseconds)] DISABLE_REESTIMATION=${DISABLE_REESTIMATION}"
  .venv/bin/run_apples.py --version || true

  local count=0
  while IFS=, read -r split_name scheme completeness replicate seed species_universe backbone_species heldout_species query_rows input_dir; do
    if [[ "${split_name}" == "split_name" ]]; then
      continue
    fi
    if [[ -n "${SWEEP_FILTER}" && "${split_name}" != *"${SWEEP_FILTER}"* ]]; then
      continue
    fi
    count=$((count + 1))
    if [[ "${MAX_SWEEPS}" != "0" && "${count}" -gt "${MAX_SWEEPS}" ]]; then
      echo "[$(date -Iseconds)] reached MAX_SWEEPS=${MAX_SWEEPS}; stopping queue"
      break
    fi
    echo "[$(date -Iseconds)] APPLES sweep ${count}: ${split_name} (${scheme}, completeness=${completeness}, heldout=${heldout_species})"
    run_split_apples "${split_name}"
  done < "${INPUT_ROOT}/sweep_manifest.csv"

  echo "[$(date -Iseconds)] Fernando APPLES sweep queue complete"
}

nohup bash -c "$(declare -f wait_for_alignment); $(declare -f make_apples_tree); $(declare -f run_split_apples); $(declare -f run_queue); ROOT='${ROOT}'; INPUT_ROOT='${INPUT_ROOT}'; LOG_DIR='${LOG_DIR}'; THREADS='${THREADS}'; MAX_SWEEPS='${MAX_SWEEPS}'; SWEEP_FILTER='${SWEEP_FILTER}'; WAIT_TIMEOUT_SECONDS='${WAIT_TIMEOUT_SECONDS}'; POLL_SECONDS='${POLL_SECONDS}'; DISABLE_REESTIMATION='${DISABLE_REESTIMATION}'; run_queue" > "${WRAPPER_LOG}" 2>&1 &
echo $! > "${WRAPPER_PID}"
echo "launched Fernando APPLES sweep queue PID $(cat "${WRAPPER_PID}")"
