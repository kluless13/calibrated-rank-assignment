#!/usr/bin/env bash
set -euo pipefail

ROOT="results/paper1_phylo_calibrated_assignment"
LOG_DIR="${ROOT}/logs"
WRAPPER_PID="${ROOT}/baseline_control_queue.pid"
WRAPPER_LOG="${LOG_DIR}/baseline_control_queue.log"

mkdir -p "${LOG_DIR}"

if [[ -f "${WRAPPER_PID}" ]] && kill -0 "$(cat "${WRAPPER_PID}")" 2>/dev/null; then
  echo "Paper 1 baseline/control queue is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

run_queue() {
  echo "[$(date -Iseconds)] Paper 1 baseline/control queue start"
  echo "[$(date -Iseconds)] Tool versions"
  command -v blastn && blastn -version | head -n 2 || true
  command -v vsearch && vsearch --version 2>&1 | head -n 2 || true

  for split in eval_c seen_test unseen_genera; do
    input_dir="data/phylo/fish_tree_clean_phylo_inputs/${split}"
    out_dir="${ROOT}/baselines_${split}"
    if [[ ! -f "${out_dir}/fish_tree_candidate_baseline_manifest.json" ]]; then
      echo "[$(date -Iseconds)] Baselines for ${split}"
      .venv/bin/python -u scripts/edna/eval_fish_tree_candidate_baselines.py \
        --input-dir "${input_dir}" \
        --output-dir "${out_dir}" \
        --methods kmer blast vsearch \
        --k 6 \
        --top-k 50 \
        --batch-size 256 \
        --threads 64 \
        --blast-task megablast \
        --blast-max-target-seqs 500 \
        --vsearch-min-id 0.5 \
        --vsearch-maxaccepts 500 \
        > "${LOG_DIR}/baselines_${split}.log" 2>&1
    else
      echo "[$(date -Iseconds)] Baselines for ${split} already complete"
    fi
  done

  for seed_suffix in "" "_seed1207" "_seed1208"; do
    seed_name="seed1206"
    if [[ "${seed_suffix}" == "_seed1207" ]]; then
      seed_name="seed1207"
    elif [[ "${seed_suffix}" == "_seed1208" ]]; then
      seed_name="seed1208"
    fi
    base="results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval${seed_suffix}"
    for split in eval_c seen_test unseen_genera; do
      input_dir="data/phylo/fish_tree_clean_phylo_inputs/${split}"
      pred_dir="${base}"
      if [[ "${split}" == "seen_test" ]]; then
        pred_dir="${base}_seen_test"
      elif [[ "${split}" == "unseen_genera" ]]; then
        pred_dir="${base}_unseen_genera"
      fi
      pred_csv="${pred_dir}/zero_shot_candidate_predictions.csv"
      out_dir="${ROOT}/negative_controls_${seed_name}_${split}"
      if [[ -f "${pred_csv}" && ! -f "${out_dir}/negative_control_manifest.json" ]]; then
        echo "[$(date -Iseconds)] Negative controls for ${seed_name} ${split}"
        .venv/bin/python -u scripts/edna/eval_fish_tree_prediction_negative_controls.py \
          --input-dir "${input_dir}" \
          --predictions "${pred_csv}" \
          --output-dir "${out_dir}" \
          --seed 1206 \
          > "${LOG_DIR}/negative_controls_${seed_name}_${split}.log" 2>&1
      fi
    done
  done

  for split in eval_c seen_test unseen_genera; do
    input_dir="data/phylo/fish_tree_clean_phylo_inputs/${split}"
    out_dir="${ROOT}/reference_diagnostics_${split}"
    args=()
    if [[ "${split}" == "eval_c" ]]; then
      args+=(--prediction-per-query "neural_seed1206=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval/zero_shot_metrics/zero_shot_candidate_per_query.csv")
      args+=(--prediction-per-query "neural_seed1207=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1207/zero_shot_metrics/zero_shot_candidate_per_query.csv")
      args+=(--prediction-per-query "neural_seed1208=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1208/zero_shot_metrics/zero_shot_candidate_per_query.csv")
    elif [[ "${split}" == "seen_test" ]]; then
      args+=(--prediction-per-query "neural_seed1206=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seen_test/zero_shot_metrics/zero_shot_candidate_per_query.csv")
      args+=(--prediction-per-query "neural_seed1207=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1207_seen_test/zero_shot_metrics/zero_shot_candidate_per_query.csv")
      args+=(--prediction-per-query "neural_seed1208=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1208_seen_test/zero_shot_metrics/zero_shot_candidate_per_query.csv")
    else
      args+=(--prediction-per-query "neural_seed1206=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_unseen_genera/zero_shot_metrics/zero_shot_candidate_per_query.csv")
      args+=(--prediction-per-query "neural_seed1207=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1207_unseen_genera/zero_shot_metrics/zero_shot_candidate_per_query.csv")
      args+=(--prediction-per-query "neural_seed1208=results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed1208_unseen_genera/zero_shot_metrics/zero_shot_candidate_per_query.csv")
    fi
    for method in kmer blast vsearch; do
      path="${ROOT}/baselines_${split}/${method}/zero_shot_metrics/zero_shot_candidate_per_query.csv"
      if [[ -f "${path}" ]]; then
        args+=(--prediction-per-query "${method}=${path}")
      fi
    done
    echo "[$(date -Iseconds)] Reference diagnostics for ${split}"
    .venv/bin/python -u scripts/edna/build_fish_tree_reference_diagnostics.py \
      --input-dir "${input_dir}" \
      --output-dir "${out_dir}" \
      "${args[@]}" \
      > "${LOG_DIR}/reference_diagnostics_${split}.log" 2>&1
  done

  echo "[$(date -Iseconds)] Paper 1 baseline/control queue complete"
}

nohup bash -c "$(declare -f run_queue); ROOT='${ROOT}'; LOG_DIR='${LOG_DIR}'; run_queue" > "${WRAPPER_LOG}" 2>&1 &
echo $! > "${WRAPPER_PID}"
echo "launched Paper 1 baseline/control queue PID $(cat "${WRAPPER_PID}")"
