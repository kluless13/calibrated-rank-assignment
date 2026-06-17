#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-results/paper1_phylo_calibrated_assignment/fernando_completeness_sweeps}"
INPUT_ROOT="${INPUT_ROOT:-data/phylo/fernando_completeness_sweeps}"
LOG_DIR="${ROOT}/logs"
WRAPPER_PID="${ROOT}/fernando_sweep_queue.pid"
WRAPPER_LOG="${LOG_DIR}/fernando_sweep_queue.log"
TREE="${TREE:-data/phylo/actinopt_12k_treePL.tre}"
THREADS="${THREADS:-32}"
MAX_SWEEPS="${MAX_SWEEPS:-0}"
SWEEP_FILTER="${SWEEP_FILTER:-}"

mkdir -p "${LOG_DIR}"

if [[ -f "${WRAPPER_PID}" ]] && kill -0 "$(cat "${WRAPPER_PID}")" 2>/dev/null; then
  echo "Fernando completeness sweep queue is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

run_split() {
  local split="$1"
  local input_dir="${INPUT_ROOT}/${split}"
  local split_root="${ROOT}/${split}"
  mkdir -p "${split_root}"

  echo "[$(date -Iseconds)] prepare ${split}"
  .venv/bin/python -u scripts/edna/prepare_fish_tree_placement_inputs.py prepare \
    --input-dir "${input_dir}" \
    --tree "${TREE}" \
    --output-dir "${split_root}" \
    > "${LOG_DIR}/${split}_prepare.log" 2>&1

  if [[ ! -f "${split_root}/reference_msa.fasta" ]]; then
    echo "[$(date -Iseconds)] mafft reference ${split}"
    mafft --auto --thread "${THREADS}" "${split_root}/reference_unaligned.fasta" \
      > "${split_root}/reference_msa.fasta" \
      2> "${LOG_DIR}/${split}_mafft_reference.log"
  fi

  if [[ ! -f "${split_root}/query_msa.fasta" ]]; then
    echo "[$(date -Iseconds)] mafft addfragments ${split}"
    mafft --thread "${THREADS}" --keeplength --addfragments "${split_root}/query_unaligned.fasta" "${split_root}/reference_msa.fasta" \
      > "${split_root}/combined_added_alignment.fasta" \
      2> "${LOG_DIR}/${split}_mafft_addfragments.log"
    .venv/bin/python -u scripts/edna/prepare_fish_tree_placement_inputs.py split-added-alignment \
      --combined "${split_root}/combined_added_alignment.fasta" \
      --query-output "${split_root}/query_msa.fasta" \
      --reference-output "${split_root}/reference_msa_from_combined.fasta" \
      > "${LOG_DIR}/${split}_split_added_alignment.log" 2>&1
  fi

  if [[ ! -f "${split_root}/epa_ng/epa_result.jplace" ]]; then
    echo "[$(date -Iseconds)] EPA-ng ${split}"
    rm -rf "${split_root}/epa_ng"
    mkdir -p "${split_root}/epa_ng"
    /workspace/phylo-placement/bin/epa-ng \
      --ref-msa "${split_root}/reference_msa.fasta" \
      --tree "${split_root}/reference_tree.nwk" \
      --query "${split_root}/query_msa.fasta" \
      --model GTR+G \
      --threads "${THREADS}" \
      --outdir "${split_root}/epa_ng" \
      > "${LOG_DIR}/${split}_epa_ng.log" 2>&1
  fi
}

run_queue() {
  echo "[$(date -Iseconds)] Fernando completeness sweep queue start"
  command -v mafft && mafft --version || true
  /workspace/phylo-placement/bin/epa-ng --version || true

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
    echo "[$(date -Iseconds)] sweep ${count}: ${split_name} (${scheme}, completeness=${completeness}, heldout=${heldout_species})"
    run_split "${split_name}"
  done < "${INPUT_ROOT}/sweep_manifest.csv"

  echo "[$(date -Iseconds)] Fernando completeness sweep queue complete"
}

nohup bash -c "$(declare -f run_split); $(declare -f run_queue); ROOT='${ROOT}'; INPUT_ROOT='${INPUT_ROOT}'; LOG_DIR='${LOG_DIR}'; TREE='${TREE}'; THREADS='${THREADS}'; MAX_SWEEPS='${MAX_SWEEPS}'; SWEEP_FILTER='${SWEEP_FILTER}'; run_queue" > "${WRAPPER_LOG}" 2>&1 &
echo $! > "${WRAPPER_PID}"
echo "launched Fernando completeness sweep queue PID $(cat "${WRAPPER_PID}")"
