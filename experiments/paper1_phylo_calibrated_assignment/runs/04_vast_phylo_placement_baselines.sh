#!/usr/bin/env bash
set -euo pipefail

ROOT="results/paper1_phylo_calibrated_assignment/phylo_placement"
LOG_DIR="${ROOT}/logs"
WRAPPER_PID="${ROOT}/phylo_placement_queue.pid"
WRAPPER_LOG="${LOG_DIR}/phylo_placement_queue.log"
TREE="data/phylo/actinopt_12k_treePL.tre"
THREADS="${THREADS:-32}"
PPLACER_REFPKG="${PPLACER_REFPKG:-}"
PPLACER_STATS="${PPLACER_STATS:-}"

mkdir -p "${LOG_DIR}"

if [[ -f "${WRAPPER_PID}" ]] && kill -0 "$(cat "${WRAPPER_PID}")" 2>/dev/null; then
  echo "Paper 1 phylogenetic placement queue is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

run_split() {
  local split="$1"
  local split_root="${ROOT}/${split}"
  mkdir -p "${split_root}"

  echo "[$(date -Iseconds)] prepare ${split}"
  .venv/bin/python -u scripts/edna/prepare_fish_tree_placement_inputs.py prepare \
    --input-dir "data/phylo/fish_tree_clean_phylo_inputs/${split}" \
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

  if [[ ! -f "${split_root}/pplacer/pplacer.jplace" ]]; then
    if [[ -n "${PPLACER_REFPKG}" ]]; then
      echo "[$(date -Iseconds)] pplacer ${split} with refpkg"
      rm -rf "${split_root}/pplacer"
      mkdir -p "${split_root}/pplacer"
      /workspace/phylo-placement/bin/pplacer \
        -c "${PPLACER_REFPKG}" \
        -j "${THREADS}" \
        -o "${split_root}/pplacer/pplacer.jplace" \
        "${split_root}/query_msa.fasta" \
        > "${LOG_DIR}/${split}_pplacer.log" 2>&1
    elif [[ -n "${PPLACER_STATS}" ]]; then
      echo "[$(date -Iseconds)] pplacer ${split} with stats"
      rm -rf "${split_root}/pplacer"
      mkdir -p "${split_root}/pplacer"
      /workspace/phylo-placement/bin/pplacer \
        -t "${split_root}/reference_tree.nwk" \
        -r "${split_root}/reference_msa.fasta" \
        -s "${PPLACER_STATS}" \
        -m GTR \
        -j "${THREADS}" \
        -o "${split_root}/pplacer/pplacer.jplace" \
        "${split_root}/query_msa.fasta" \
        > "${LOG_DIR}/${split}_pplacer.log" 2>&1
    else
      echo "[$(date -Iseconds)] skip pplacer ${split}: set PPLACER_REFPKG or PPLACER_STATS to run a valid pplacer comparator" \
        > "${LOG_DIR}/${split}_pplacer.log"
    fi
  fi
}

run_queue() {
  echo "[$(date -Iseconds)] Paper 1 phylogenetic placement queue start"
  command -v mafft && mafft --version || true
  /workspace/phylo-placement/bin/epa-ng --version || true
  /workspace/phylo-placement/bin/pplacer --version || true

  for split in eval_c seen_test unseen_genera; do
    run_split "${split}"
  done

  echo "[$(date -Iseconds)] Paper 1 phylogenetic placement queue complete"
}

nohup bash -c "$(declare -f run_split); $(declare -f run_queue); ROOT='${ROOT}'; LOG_DIR='${LOG_DIR}'; TREE='${TREE}'; THREADS='${THREADS}'; PPLACER_REFPKG='${PPLACER_REFPKG}'; PPLACER_STATS='${PPLACER_STATS}'; run_queue" > "${WRAPPER_LOG}" 2>&1 &
echo $! > "${WRAPPER_PID}"
echo "launched Paper 1 phylogenetic placement queue PID $(cat "${WRAPPER_PID}")"
