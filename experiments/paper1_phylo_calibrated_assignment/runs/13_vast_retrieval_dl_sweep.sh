#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

PY="${PYTHON:-/venv/main/bin/python}"
ROOT="results/paper1_phylo_calibrated_assignment/retrieval_dl_sweep"
QUEUE_LOG="${ROOT}/retrieval_dl_sweep.log"
QUEUE_PID="${ROOT}/retrieval_dl_sweep.pid"
TREE_NPZ="results/coi_fish_tree_clean_phylo_mamba_hier512_seqval/tree_embeddings.npz"
mkdir -p "${ROOT}/logs"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "Paper 1 retrieval-DL sweep is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -u
  cd /workspace/marinemamba

  echo "[$(date -Iseconds)] Paper 1 retrieval-DL sweep starting"
  "${PY}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no_cuda", flush=True)
PY

  if [[ ! -f "${TREE_NPZ}" ]]; then
    echo "Missing shared tree embedding NPZ: ${TREE_NPZ}" >&2
    exit 1
  fi

  run_arm () {
    local run_name="$1"
    local model_type="$2"
    local loss_mode="$3"
    local epochs="$4"
    local batch="$5"
    local lr="$6"
    local d_model="$7"
    local token_dim="$8"
    local layers="$9"
    local heads="${10}"
    local seed="${11}"
    local out="${ROOT}/${run_name}"

    mkdir -p "${out}/logs"
    echo "[$(date -Iseconds)] START arm=${run_name} model=${model_type} loss=${loss_mode}"

    if [[ ! -f "${out}/run_manifest.json" && ! -f "${out}/FAILED" ]]; then
      "${PY}" -u scripts/edna/train_fish_tree_encoder_benchmark.py train \
        --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
        --output-dir "${out}" \
        --tree-file data/phylo/actinopt_12k_treePL.tre \
        --tree-embedding-npz "${TREE_NPZ}" \
        --model-type "${model_type}" \
        --max-seq-len 700 \
        --embed-dim 512 \
        --d-model "${d_model}" \
        --token-emb-dim "${token_dim}" \
        --num-layers "${layers}" \
        --num-heads "${heads}" \
        --dropout 0.1 \
        --train-epochs "${epochs}" \
        --batch-size "${batch}" \
        --lr "${lr}" \
        --weight-decay 0.01 \
        --loss-mode "${loss_mode}" \
        --temperature 0.07 \
        --cosine-weight 0.1 \
        --contrastive-weight 1.0 \
        --species-positive-weight 1.0 \
        --genus-positive-weight 0.30 \
        --family-positive-weight 0.12 \
        --order-positive-weight 0.04 \
        --validation-mode random_sequence \
        --val-fraction 0.1 \
        --top-k 50 \
        --num-workers 8 \
        --seed "${seed}" \
        --write-query-embeddings \
        --predict-batch-size 256 \
        > "${out}/logs/train_eval_c.log" 2>&1
      local status=$?
      if [[ "${status}" -ne 0 ]]; then
        echo "[$(date -Iseconds)] FAIL arm=${run_name} train status=${status}" | tee "${out}/FAILED"
        return 0
      fi
    fi

    for split in seen_test unseen_genera; do
      local split_out="${out}_${split}"
      mkdir -p "${split_out}/logs"
      echo "[$(date -Iseconds)] PREDICT arm=${run_name} split=${split}"
      if [[ ! -f "${split_out}/run_manifest.json" && ! -f "${split_out}/FAILED" ]]; then
        "${PY}" -u scripts/edna/train_fish_tree_encoder_benchmark.py predict \
          --input-dir "data/phylo/fish_tree_clean_phylo_inputs/${split}" \
          --output-dir "${split_out}" \
          --tree-file data/phylo/actinopt_12k_treePL.tre \
          --tree-embedding-npz "${TREE_NPZ}" \
          --checkpoint "${out}/${model_type}_tree_encoder_best.pt" \
          --run-manifest "${out}/run_manifest.json" \
          --predict-batch-size 256 \
          --top-k 50 \
          --num-workers 8 \
          --seed "${seed}" \
          --write-query-embeddings \
          > "${split_out}/logs/predict_${split}.log" 2>&1
        local status=$?
        if [[ "${status}" -ne 0 ]]; then
          echo "[$(date -Iseconds)] FAIL arm=${run_name} split=${split} predict status=${status}" | tee "${split_out}/FAILED"
        fi
      fi
    done

    for split in eval_c unseen_genera; do
      local input_dir="data/phylo/fish_tree_clean_phylo_inputs/${split}"
      local recovery_out="${out}_tree_recovery_${split}"
      mkdir -p "${recovery_out}/logs"
      echo "[$(date -Iseconds)] TREE_RECOVERY arm=${run_name} split=${split}"
      if [[ ! -f "${recovery_out}/tree_recovery_metrics.json" && ! -f "${recovery_out}/FAILED" ]]; then
        "${PY}" -u scripts/edna/eval_fish_tree_encoder_tree_recovery.py \
          --input-dir "${input_dir}" \
          --checkpoint "${out}/${model_type}_tree_encoder_best.pt" \
          --run-manifest "${out}/run_manifest.json" \
          --output-dir "${recovery_out}" \
          --tree-file data/phylo/actinopt_12k_treePL.tre \
          --seqs-per-species 10 \
          --max-pairs 50000 \
          --batch-size 256 \
          --num-workers 8 \
          --seed "${seed}" \
          > "${recovery_out}/logs/tree_recovery_${split}.log" 2>&1
        local status=$?
        if [[ "${status}" -ne 0 ]]; then
          echo "[$(date -Iseconds)] FAIL arm=${run_name} split=${split} tree_recovery status=${status}" | tee "${recovery_out}/FAILED"
        fi
      fi
    done

    echo "[$(date -Iseconds)] DONE arm=${run_name}"
  }

  run_arm "coi_cnn_retrieval_contrastive_seed1301" "cnn" "contrastive" 60 160 5e-4 256 32 2 8 1301
  run_arm "coi_cnn_retrieval_hybrid_seed1301" "cnn" "hybrid" 60 160 5e-4 256 32 2 8 1301
  run_arm "coi_cnn_retrieval_hier_contrastive_seed1301" "cnn" "hierarchical_contrastive" 60 160 5e-4 256 32 2 8 1301
  run_arm "coi_transformer_retrieval_hier_contrastive_seed1301" "transformer" "hierarchical_contrastive" 50 96 3e-4 256 32 2 8 1301

  echo "[$(date -Iseconds)] Paper 1 retrieval-DL sweep complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched Paper 1 retrieval-DL sweep PID $(cat "${QUEUE_PID}")"
