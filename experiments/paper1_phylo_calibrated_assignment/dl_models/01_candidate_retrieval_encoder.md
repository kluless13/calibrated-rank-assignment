# DL 01: Candidate Retrieval Encoder

## Question

Can a learned barcode encoder retrieve plausible tree-aware candidate species
fast enough to serve as the front end of biodiversity inference?

## Why This Is Not Novel Alone

Fast learned/vector barcode retrieval already exists in nearby work such as
BarcodeBERT, BarcodeMamba, DNABERT-S, TaxoTagger, and learned DNA sequence
search systems. We should not claim that vector barcode search itself is new.

## Our Contribution Boundary

The retrieval encoder is useful only as one layer in an uncertainty-aware
pipeline:

```text
query barcode
  -> vector top-k candidates
  -> classical/tree/evidence reranking
  -> calibrated rank/no-call
```

The novel test is not just speed. It is whether fast retrieval preserves enough
candidate diversity for missing-reference rank-backoff.

## Current Status

Done:

- CNN, biLSTM, Transformer, and Mamba-style candidate rankings exist.
- HNSW vector timing exists.
- Raw sequence production-v1 timing exists.
- Retrieval-DL sweep completed on Vast and was copied locally.
- Runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/13_vast_retrieval_dl_sweep.sh`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_dl_sweep_summary.csv`,
  `retrieval_dl_sweep_tree_recovery.csv`, and
  `retrieval_dl_sweep_training_history.csv`.
- Arms:
  - `coi_cnn_retrieval_contrastive_seed1301`;
  - `coi_cnn_retrieval_hybrid_seed1301`;
  - `coi_cnn_retrieval_hier_contrastive_seed1301`;
  - `coi_transformer_retrieval_hier_contrastive_seed1301`.

Result summary:

| Arm | Held-out top50 species/genus/family/order | Unseen-genera top50 species/genus/family/order | Held-out tree Pearson/Spearman |
|---|---:|---:|---:|
| CNN contrastive | 61.2 / 96.2 / 97.4 / 98.2% | 34.1 / 62.9 / 86.1 / 90.5% | 0.561 / 0.489 |
| CNN hybrid | 60.8 / 95.6 / 97.4 / 98.0% | 34.0 / 60.3 / 86.4 / 91.6% | 0.573 / 0.507 |
| CNN hierarchical contrastive | 53.6 / 84.8 / 95.4 / 97.1% | 22.1 / 54.3 / 85.1 / 89.8% | 0.585 / 0.587 |
| Transformer hierarchical contrastive | 25.5 / 54.0 / 65.9 / 72.4% | 10.9 / 27.7 / 50.1 / 59.6% | 0.630 / 0.575 |

Interpretation: contrastive/hybrid CNNs are the practical retrieval candidates.
The hierarchical losses are valuable as tree-geometry probes, but they reduce
candidate recall. The Transformer arm is not competitive as a first-pass
retriever in this sweep.

Key current speed:

- exact vector: 0.397 ms/query on controlled Vast timing;
- HNSW m16/ef50: 0.00475 ms/query.

## Next Experiments

1. Run the vector-first rank/no-call pipeline on the best retrieval-DL query
   embeddings as a diagnostic, with thresholds recalibrated before any final
   claim.
2. Compare recall, latency, and memory against BLAST/VSEARCH/k-mer under a
   consistent top-k candidate protocol.
3. Add BLAST/VSEARCH identity features to the top-k candidate reranker so the
   DL layer learns when sequence similarity and tree-space disagree.
4. Stress test reference size with synthetic 5x/10x/25x expansion.
5. Add pretrained barcode encoder baselines only if they can be evaluated under
   the same candidate tree and rank/no-call protocol.
