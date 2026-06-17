#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

RUN_ROOT="results/coi_fish_tree_clean_phylo_mamba_hier512_seqval"
WRAPPER_LOG="${RUN_ROOT}/wrapper.log"
WRAPPER_PID="${RUN_ROOT}/wrapper.pid"
mkdir -p "${RUN_ROOT}/logs"

if [[ -f "${WRAPPER_PID}" ]] && ps -p "$(cat "${WRAPPER_PID}")" >/dev/null 2>&1; then
  echo "corrected fish-tree wrapper is already running: PID $(cat "${WRAPPER_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  .venv/bin/python -u scripts/edna/train_12s_phylo_mamba.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
    --output-dir "${RUN_ROOT}" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --max-seq-len 700 \
    --embed-dim 512 \
    --tree-epochs 600 \
    --tree-pairs-per-epoch 4096 \
    --tree-negatives 64 \
    --tree-max-distance-sample 3000 \
    --train-epochs 50 \
    --batch-size 64 \
    --lr 5e-4 \
    --pooling masked_mean \
    --loss-mode hierarchical_hybrid \
    --temperature 0.05 \
    --cosine-weight 0.2 \
    --contrastive-weight 1.0 \
    --validation-mode random_sequence \
    --val-fraction 0.1 \
    --top-k 50 \
    --num-workers 8 \
    --seed 1206 \
    > "${RUN_ROOT}/logs/train_eval_c_hier512_seqval.log" 2>&1

  .venv/bin/python -u scripts/edna/predict_phylo_mamba_checkpoint.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/seen_test \
    --checkpoint "${RUN_ROOT}/phylo_mamba_best.pt" \
    --tree-embedding-npz "${RUN_ROOT}/tree_embeddings.npz" \
    --output-dir "${RUN_ROOT}_seen_test" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --max-seq-len 700 \
    --pooling masked_mean \
    --batch-size 256 \
    --top-k 50 \
    --num-workers 8 \
    --seed 1206 \
    > "${RUN_ROOT}/logs/predict_seen_test.log" 2>&1

  .venv/bin/python -u scripts/edna/predict_phylo_mamba_checkpoint.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/unseen_genera \
    --checkpoint "${RUN_ROOT}/phylo_mamba_best.pt" \
    --tree-embedding-npz "${RUN_ROOT}/tree_embeddings.npz" \
    --output-dir "${RUN_ROOT}_unseen_genera" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --max-seq-len 700 \
    --pooling masked_mean \
    --batch-size 256 \
    --top-k 50 \
    --num-workers 8 \
    --seed 1206 \
    > "${RUN_ROOT}/logs/predict_unseen_genera.log" 2>&1

  .venv/bin/python -u scripts/edna/eval_phylo_checkpoint_tree_recovery.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/eval_c \
    --checkpoint "${RUN_ROOT}/phylo_mamba_best.pt" \
    --output-dir "${RUN_ROOT}_tree_recovery_eval_c" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --embed-dim 512 \
    --max-seq-len 700 \
    --pooling masked_mean \
    --seqs-per-species 10 \
    --max-pairs 50000 \
    --batch-size 256 \
    --num-workers 8 \
    --seed 1206 \
    > "${RUN_ROOT}/logs/tree_recovery_eval_c.log" 2>&1

  .venv/bin/python -u scripts/edna/eval_phylo_checkpoint_tree_recovery.py \
    --input-dir data/phylo/fish_tree_clean_phylo_inputs/unseen_genera \
    --checkpoint "${RUN_ROOT}/phylo_mamba_best.pt" \
    --output-dir "${RUN_ROOT}_tree_recovery_unseen_genera" \
    --tree-file data/phylo/actinopt_12k_treePL.tre \
    --embed-dim 512 \
    --max-seq-len 700 \
    --pooling masked_mean \
    --seqs-per-species 10 \
    --max-pairs 50000 \
    --batch-size 256 \
    --num-workers 8 \
    --seed 1206 \
    > "${RUN_ROOT}/logs/tree_recovery_unseen_genera.log" 2>&1
) > "${WRAPPER_LOG}" 2>&1 &

echo $! > "${WRAPPER_PID}"
echo "launched corrected fish-tree wrapper PID $(cat "${WRAPPER_PID}")"
