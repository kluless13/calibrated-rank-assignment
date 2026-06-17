#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

QUEUE_ROOT="results/high_leverage_2026_05_30"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"
QUEUE_PID="${QUEUE_ROOT}/queue.pid"
mkdir -p "${QUEUE_ROOT}" results/edna/taxdna_ssm/logs

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "high-leverage queue is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  RUN_ROOT="results/edna/taxdna_ssm"
  INPUT_DIR="data/edna/real_edna_queries/global_tropical_multisource_teleo"
  SAMPLE_MAP="${INPUT_DIR}/sample_query_map.csv"
  TREE_NPZ="${RUN_ROOT}/multisource_teleo_tree_embeddings.npz"
  FISHGLOB_JSON="data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_cooccurrence.json"
  FISHGLOB_DIR="${RUN_ROOT}/multisource_teleo_npz_cooccurrence_fishglob_public_50k"

  echo "[$(date -Iseconds)] Task 1: FISHGLOB public co-occurrence model"
  if [[ ! -f "${FISHGLOB_DIR}/run_manifest.json" ]]; then
    .venv/bin/python -u scripts/edna/train_npz_cooccurrence_model.py \
      --tree-embedding-npz "${TREE_NPZ}" \
      --cooccurrence-json "${FISHGLOB_JSON}" \
      --output-dir "${FISHGLOB_DIR}" \
      --hidden-dim 256 \
      --kernel-temp 0.05 \
      --epochs 8 \
      --lr 0.001 \
      --val-fraction 0.05 \
      --max-train-groups 50000 \
      --add-noise \
      > "${RUN_ROOT}/logs/multisource_teleo_npz_cooccurrence_fishglob_public_50k.log" 2>&1
  else
    echo "FISHGLOB model already complete; skipping training."
  fi

  if [[ ! -f "${RUN_ROOT}/global_edna_multisource_teleo_ssm_contrastive_predictions/query_embeddings.npz" ]]; then
    .venv/bin/python -u scripts/edna/predict_phylo_mamba_checkpoint.py \
      --input-dir "${INPUT_DIR}" \
      --checkpoint "${RUN_ROOT}/multisource_teleo_ssm_contrastive/phylo_mamba_best.pt" \
      --tree-embedding-npz "${TREE_NPZ}" \
      --output-dir "${RUN_ROOT}/global_edna_multisource_teleo_ssm_contrastive_predictions" \
      --max-seq-len 128 \
      --pooling masked_mean \
      --batch-size 256 \
      --top-k 50 \
      --write-query-embeddings \
      --skip-eval \
      > "${RUN_ROOT}/logs/global_edna_multisource_teleo_ssm_contrastive_predictions.log" 2>&1
  fi

  if [[ ! -f "${RUN_ROOT}/global_edna_multisource_teleo_cnn_predictions/query_embeddings.npz" ]]; then
    .venv/bin/python -u scripts/edna/predict_taxdna_cnn_checkpoint.py \
      --input-dir "${INPUT_DIR}" \
      --checkpoint "${RUN_ROOT}/multisource_teleo_cnn/taxdna_cnn_best.pt" \
      --tree-embedding-npz "${TREE_NPZ}" \
      --output-dir "${RUN_ROOT}/global_edna_multisource_teleo_cnn_predictions" \
      --max-seq-len 128 \
      --batch-size 256 \
      --top-k 50 \
      --write-query-embeddings \
      --skip-eval \
      > "${RUN_ROOT}/logs/global_edna_multisource_teleo_cnn_predictions.log" 2>&1
  fi

  for pair in "025 0.25" "050 0.50" "100 1.00" "200 2.00"; do
    set -- ${pair}
    label="$1"
    weight="$2"

    SSM_OUT="${RUN_ROOT}/global_edna_multisource_teleo_ssm_fishglob_learned_cooccurrence_w${label}"
    if [[ ! -f "${SSM_OUT}/learned_cooccurrence_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/eval_global_edna_learned_cooccurrence.py \
        --input-dir "${INPUT_DIR}" \
        --sample-query-map "${SAMPLE_MAP}" \
        --query-embedding-npz "${RUN_ROOT}/global_edna_multisource_teleo_ssm_contrastive_predictions/query_embeddings.npz" \
        --tree-embedding-npz "${TREE_NPZ}" \
        --cooccurrence-checkpoint "${FISHGLOB_DIR}/npz_cooccurrence_model.pt" \
        --output-dir "${SSM_OUT}" \
        --sequence-temp 0.05 \
        --context-weight "${weight}" \
        --top-k 50 \
        > "${RUN_ROOT}/logs/global_edna_multisource_teleo_ssm_fishglob_learned_cooccurrence_w${label}.log" 2>&1
    fi

    CNN_OUT="${RUN_ROOT}/global_edna_multisource_teleo_cnn_fishglob_learned_cooccurrence_w${label}"
    if [[ ! -f "${CNN_OUT}/learned_cooccurrence_manifest.json" ]]; then
      .venv/bin/python -u scripts/edna/eval_global_edna_learned_cooccurrence.py \
        --input-dir "${INPUT_DIR}" \
        --sample-query-map "${SAMPLE_MAP}" \
        --query-embedding-npz "${RUN_ROOT}/global_edna_multisource_teleo_cnn_predictions/query_embeddings.npz" \
        --tree-embedding-npz "${TREE_NPZ}" \
        --cooccurrence-checkpoint "${FISHGLOB_DIR}/npz_cooccurrence_model.pt" \
        --output-dir "${CNN_OUT}" \
        --sequence-temp 0.05 \
        --context-weight "${weight}" \
        --top-k 50 \
        > "${RUN_ROOT}/logs/global_edna_multisource_teleo_cnn_fishglob_learned_cooccurrence_w${label}.log" 2>&1
    fi
  done

  echo "[$(date -Iseconds)] Task 2: fish-tree cosine-512 diagnostic"
  FISH_RUN="results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval"
  FISH_TREE_NPZ="results/coi_fish_tree_clean_phylo_mamba_hier512_seqval/tree_embeddings.npz"
  mkdir -p "${FISH_RUN}/logs"
  cp "${FISH_TREE_NPZ}" "${FISH_RUN}/tree_embeddings.npz"
  if [[ ! -f "${FISH_RUN}/run_manifest.json" ]]; then
    .venv/bin/python -u scripts/edna/train_12s_phylo_mamba.py \
      --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
      --output-dir "${FISH_RUN}" \
      --tree-file data/phylo/actinopt_12k_treePL.tre \
      --max-seq-len 700 \
      --embed-dim 512 \
      --tree-embedding-npz "${FISH_RUN}/tree_embeddings.npz" \
      --train-epochs 50 \
      --batch-size 64 \
      --lr 5e-4 \
      --pooling masked_mean \
      --loss-mode cosine \
      --validation-mode random_sequence \
      --val-fraction 0.1 \
      --top-k 50 \
      --num-workers 8 \
      --seed 1206 \
      > "${FISH_RUN}/logs/train_eval_c_cosine512_seqval.log" 2>&1
  else
    echo "Fish-tree cosine-512 model already complete; skipping training."
  fi

  .venv/bin/python -u scripts/edna/predict_phylo_mamba_checkpoint.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/seen_test \
    --checkpoint "${FISH_RUN}/phylo_mamba_best.pt" \
    --tree-embedding-npz "${FISH_RUN}/tree_embeddings.npz" \
    --output-dir "${FISH_RUN}_seen_test" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --max-seq-len 700 \
    --pooling masked_mean \
    --batch-size 256 \
    --top-k 50 \
    --num-workers 8 \
    --seed 1206 \
    > "${FISH_RUN}/logs/predict_seen_test.log" 2>&1

  .venv/bin/python -u scripts/edna/predict_phylo_mamba_checkpoint.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/unseen_genera \
    --checkpoint "${FISH_RUN}/phylo_mamba_best.pt" \
    --tree-embedding-npz "${FISH_RUN}/tree_embeddings.npz" \
    --output-dir "${FISH_RUN}_unseen_genera" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --max-seq-len 700 \
    --pooling masked_mean \
    --batch-size 256 \
    --top-k 50 \
    --num-workers 8 \
    --seed 1206 \
    > "${FISH_RUN}/logs/predict_unseen_genera.log" 2>&1

  .venv/bin/python -u scripts/edna/eval_phylo_checkpoint_tree_recovery.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
    --checkpoint "${FISH_RUN}/phylo_mamba_best.pt" \
    --output-dir "${FISH_RUN}_tree_recovery_eval_c" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --embed-dim 512 \
    --max-seq-len 700 \
    --pooling masked_mean \
    --seqs-per-species 10 \
    --max-pairs 50000 \
    --batch-size 256 \
    --num-workers 8 \
    --seed 1206 \
    > "${FISH_RUN}/logs/tree_recovery_eval_c.log" 2>&1

  .venv/bin/python -u scripts/edna/eval_phylo_checkpoint_tree_recovery.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/unseen_genera \
    --checkpoint "${FISH_RUN}/phylo_mamba_best.pt" \
    --output-dir "${FISH_RUN}_tree_recovery_unseen_genera" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --embed-dim 512 \
    --max-seq-len 700 \
    --pooling masked_mean \
    --seqs-per-species 10 \
    --max-pairs 50000 \
    --batch-size 256 \
    --num-workers 8 \
    --seed 1206 \
    > "${FISH_RUN}/logs/tree_recovery_unseen_genera.log" 2>&1

  echo "[$(date -Iseconds)] High-leverage queue complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched high-leverage queue PID $(cat "${QUEUE_PID}")"
