#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

RUN_ROOT="results/edna/taxdna_ssm"
INPUT_DIR="data/edna/real_edna_queries/global_tropical_multisource_teleo"
SAMPLE_MAP="${INPUT_DIR}/sample_query_map.csv"
TREE_NPZ="${RUN_ROOT}/multisource_teleo_tree_embeddings.npz"
COOCC_JSON="data/edna/cooccurrence_inputs/taxdna_ssm/rls_obis_taxdna_cooccurrence.json"
COOCC_DIR="${RUN_ROOT}/multisource_teleo_npz_cooccurrence_rls_obis"
WRAPPER_LOG="${RUN_ROOT}/learned_cooccurrence_wrapper.log"
WRAPPER_PID="${RUN_ROOT}/learned_cooccurrence_wrapper.pid"

mkdir -p "${RUN_ROOT}/logs"

if [[ -f "${WRAPPER_PID}" ]] && ps -p "$(cat "${WRAPPER_PID}")" >/dev/null 2>&1; then
  echo "learned co-occurrence wrapper is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  .venv/bin/python -u scripts/edna/train_npz_cooccurrence_model.py \
    --tree-embedding-npz "${TREE_NPZ}" \
    --cooccurrence-json "${COOCC_JSON}" \
    --output-dir "${COOCC_DIR}" \
    --hidden-dim 256 \
    --kernel-temp 0.05 \
    --epochs 20 \
    --lr 0.001 \
    --add-noise \
    > "${RUN_ROOT}/logs/multisource_teleo_npz_cooccurrence_rls_obis.log" 2>&1

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

  for pair in "025 0.25" "050 0.50" "100 1.00" "200 2.00"; do
    set -- ${pair}
    label="$1"
    weight="$2"

    .venv/bin/python -u scripts/edna/eval_global_edna_learned_cooccurrence.py \
      --input-dir "${INPUT_DIR}" \
      --sample-query-map "${SAMPLE_MAP}" \
      --query-embedding-npz "${RUN_ROOT}/global_edna_multisource_teleo_ssm_contrastive_predictions/query_embeddings.npz" \
      --tree-embedding-npz "${TREE_NPZ}" \
      --cooccurrence-checkpoint "${COOCC_DIR}/npz_cooccurrence_model.pt" \
      --output-dir "${RUN_ROOT}/global_edna_multisource_teleo_ssm_learned_cooccurrence_w${label}" \
      --sequence-temp 0.05 \
      --context-weight "${weight}" \
      --top-k 50 \
      > "${RUN_ROOT}/logs/global_edna_multisource_teleo_ssm_learned_cooccurrence_w${label}.log" 2>&1

    .venv/bin/python -u scripts/edna/eval_global_edna_learned_cooccurrence.py \
      --input-dir "${INPUT_DIR}" \
      --sample-query-map "${SAMPLE_MAP}" \
      --query-embedding-npz "${RUN_ROOT}/global_edna_multisource_teleo_cnn_predictions/query_embeddings.npz" \
      --tree-embedding-npz "${TREE_NPZ}" \
      --cooccurrence-checkpoint "${COOCC_DIR}/npz_cooccurrence_model.pt" \
      --output-dir "${RUN_ROOT}/global_edna_multisource_teleo_cnn_learned_cooccurrence_w${label}" \
      --sequence-temp 0.05 \
      --context-weight "${weight}" \
      --top-k 50 \
      > "${RUN_ROOT}/logs/global_edna_multisource_teleo_cnn_learned_cooccurrence_w${label}.log" 2>&1
  done
) > "${WRAPPER_LOG}" 2>&1 &

echo $! > "${WRAPPER_PID}"
echo "launched learned co-occurrence wrapper PID $(cat "${WRAPPER_PID}")"
