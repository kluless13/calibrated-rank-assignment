#!/usr/bin/env bash
set -euo pipefail

cd /workspace/marinemamba

QUEUE_ROOT="results/coi_fish_tree_clean_phylo_mamba_cosine512_seed_repeats"
QUEUE_LOG="${QUEUE_ROOT}/wrapper.log"
QUEUE_PID="${QUEUE_ROOT}/wrapper.pid"
mkdir -p "${QUEUE_ROOT}/logs"

if [[ -f "${QUEUE_PID}" ]] && ps -p "$(cat "${QUEUE_PID}")" >/dev/null 2>&1; then
  echo "COI cosine512 seed-repeat wrapper is already running: PID $(cat "${QUEUE_PID}")"
  exit 0
fi

(
  set -euo pipefail
  cd /workspace/marinemamba

  FISH_TREE_NPZ="results/coi_fish_tree_clean_phylo_mamba_hier512_seqval/tree_embeddings.npz"
  if [[ ! -f "${FISH_TREE_NPZ}" ]]; then
    echo "Missing tree embedding NPZ: ${FISH_TREE_NPZ}" >&2
    exit 1
  fi

  for seed in 1207 1208; do
    FISH_RUN="results/coi_fish_tree_clean_phylo_mamba_cosine512_seqval_seed${seed}"
    mkdir -p "${FISH_RUN}/logs"
    cp "${FISH_TREE_NPZ}" "${FISH_RUN}/tree_embeddings.npz"

    echo "[$(date -Iseconds)] Training COI cosine512 seed ${seed}"
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
        --seed "${seed}" \
        > "${FISH_RUN}/logs/train_eval_c_cosine512_seqval.log" 2>&1
    else
      echo "Seed ${seed} model already complete; skipping training."
    fi

    echo "[$(date -Iseconds)] Evaluating COI cosine512 seed ${seed}"
    if [[ ! -f "${FISH_RUN}_seen_test/zero_shot_metrics.json" ]]; then
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
        --seed "${seed}" \
        > "${FISH_RUN}/logs/predict_seen_test.log" 2>&1
    fi

    if [[ ! -f "${FISH_RUN}_unseen_genera/zero_shot_metrics.json" ]]; then
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
        --seed "${seed}" \
        > "${FISH_RUN}/logs/predict_unseen_genera.log" 2>&1
    fi

    if [[ ! -f "${FISH_RUN}_tree_recovery_eval_c/tree_recovery_metrics.json" ]]; then
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
        --seed "${seed}" \
        > "${FISH_RUN}/logs/tree_recovery_eval_c.log" 2>&1
    fi

    if [[ ! -f "${FISH_RUN}_tree_recovery_unseen_genera/tree_recovery_metrics.json" ]]; then
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
        --seed "${seed}" \
        > "${FISH_RUN}/logs/tree_recovery_unseen_genera.log" 2>&1
    fi
  done

  echo "[$(date -Iseconds)] COI cosine512 seed-repeat queue complete"
) > "${QUEUE_LOG}" 2>&1 &

echo $! > "${QUEUE_PID}"
echo "launched COI cosine512 seed-repeat wrapper PID $(cat "${QUEUE_PID}")"
