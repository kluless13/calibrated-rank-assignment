# Query Embedding Export TODO

Current copied COI Paper 1 outputs now include reusable per-query embedding
arrays for the CNN/biLSTM/Transformer encoder benchmarks and CNN seed repeats.
Mamba query embeddings remain missing.

Observed locally:

- Mamba COI runs have `tree_embeddings.npz` and top-50
  `zero_shot_candidate_predictions.csv`.
- CNN/biLSTM/Transformer COI benchmark runs now have `query_embeddings.npz`
  for Eval C, seen-test, and unseen-genera under
  `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings/`.
- CNN seed repeats 1207 and 1208 also have `query_embeddings.npz` for Eval C,
  seen-test, and unseen-genera under
  `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/cnn_seed_repeats/`.
- Mamba query-embedding export was probed on the current Vast image, but
  `mamba-ssm` did not build cleanly against PyTorch 2.12 / CUDA 13.

Why this matters:

- Post-hoc candidate-ablation can be done from saved top-50 rankings, and this
  is now in `candidate_ablation_rank_backoff.csv`.
- Strict candidate-ablation, embedding-derived tree reconstruction, and
  placement-style adapters need full query embeddings or full candidate score
  vectors, not only top-50 lists.

Remaining export jobs:

1. Mamba cosine512 seed1206/1207/1208:
   - Eval C
   - seen-test
   - unseen-genera

Required output per run:

- `query_embeddings.npz`
- `candidate_embeddings.npz` or a manifest pointing to `tree_embeddings.npz`
- query metadata with `processid`, true species/genus/family/order, split
- candidate metadata with tree label and taxonomy
- full or chunked query-by-candidate score matrix if feasible

Claim boundary:

- Full-candidate embedding ablations are now available for
  CNN/biLSTM/Transformer and CNN seed repeats.
- Mamba candidate-ablation tables should still be described as post-hoc
  rank-backoff diagnostics over saved top-50 predictions until Mamba query
  embeddings are exported.
- None of the current candidate-ablation tables are strict retrained
  tree-pruned experiments.
