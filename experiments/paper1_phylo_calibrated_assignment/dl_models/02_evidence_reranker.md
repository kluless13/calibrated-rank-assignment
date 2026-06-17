# DL 02: Evidence Reranker

## Question

Can a trainable reranker improve the top-k candidate order by combining neural,
classical, and tree evidence?

## Motivation

BLAST and p-distance are strong because they compare sequences directly.
Neural embeddings are useful because they provide fast retrieval and tree-space
geometry. A reranker should not discard either source.

## Candidate Features

First candidate-level feature set:

- vector cosine score and rank;
- p-distance / identity to train-reference sequence;
- BLAST/VSEARCH identity when added;
- candidate taxonomy and top-k consensus;
- candidate tree distance to nearest retrieved reference;
- placement confidence or placed-clade diagnostics where available;
- reference availability flags.

## Current Status

First diagnostic model complete:

- p-distance reranking exists as a deterministic stage;
- candidate-level eDNA posterior infrastructure exists;
- trainable COI top-k candidate reranker has been trained:
  `scripts/edna/train_paper1_candidate_reranker.py`;
- compact source-table collector:
  `scripts/edna/build_paper1_candidate_reranker_summary.py`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/candidate_reranker_summary.csv`.

Second-stage BLAST/VSEARCH-aware run complete:

- runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/14_vast_retrieval_dl_pipeline_reranker.sh`;
- copied log root:
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_retrieval_dl_pipeline_reranker/retrieval_dl_pipeline_reranker/`;
- arms:
  `coi_cnn_retrieval_contrastive_seed1301` and
  `coi_cnn_retrieval_hybrid_seed1301`;
- added evidence:
  BLAST and VSEARCH top-50 candidate presence, rank, score, and score delta;
- purpose:
  test whether the first-stage retrieval-DL embeddings become stronger when
  the second stage fuses neural retrieval, p-distance, and classical sequence
  evidence.

BLAST/VSEARCH-aware reranker results:

| Arm | Split | Rank | Reranker Top-1 | p-distance Top-1 | Delta | Calibrated Coverage | Calibrated Precision |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN contrastive | held-out fish | genus | 83.1% | 49.1% | +34.0 pp | 67.3% | 91.5% |
| CNN contrastive | held-out fish | family | 94.3% | 88.0% | +6.3 pp | 96.8% | 96.6% |
| CNN contrastive | unseen-genera | family | 70.5% | 68.9% | +1.6 pp | 85.7% | 80.1% |
| CNN contrastive | unseen-genera | order | 85.5% | 84.9% | +0.6 pp | 99.9% | 85.5% |
| CNN hybrid | held-out fish | genus | 80.7% | 48.2% | +32.5 pp | 68.9% | 92.2% |
| CNN hybrid | held-out fish | family | 95.1% | 87.6% | +7.5 pp | 95.5% | 97.3% |
| CNN hybrid | unseen-genera | family | 72.0% | 69.4% | +2.6 pp | 78.0% | 85.6% |
| CNN hybrid | unseen-genera | order | 86.1% | 85.2% | +0.9 pp | 99.8% | 86.2% |

Species-level rows stay at 0.0% in held-out missing-reference splits because
the true species is not supported by the candidate/reference setup. That is a
feature of the evaluation, not a failure to optimize species labels.

Tree10 reranker results:

| Arm | Split | Rank | Reranker Top-1 | p-distance Top-1 | Delta | Calibrated Coverage | Calibrated Precision |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN contrastive tree10 | held-out fish | genus | 83.7% | 49.1% | +34.6 pp | 60.1% | 93.7% |
| CNN contrastive tree10 | held-out fish | family | 94.7% | 88.0% | +6.7 pp | 98.6% | 95.9% |
| CNN contrastive tree10 | unseen-genera | family | 71.6% | 68.9% | +2.7 pp | 94.1% | 75.2% |
| CNN contrastive tree10 | unseen-genera | order | 85.3% | 84.9% | +0.3 pp | 99.9% | 85.4% |
| CNN hybrid tree10 | held-out fish | genus | 84.0% | 48.2% | +35.8 pp | 74.5% | 93.2% |
| CNN hybrid tree10 | held-out fish | family | 95.4% | 87.6% | +7.8 pp | 98.7% | 96.5% |
| CNN hybrid tree10 | unseen-genera | family | 72.5% | 69.4% | +3.1 pp | 93.5% | 77.0% |
| CNN hybrid tree10 | unseen-genera | order | 85.8% | 85.2% | +0.5 pp | 99.8% | 85.9% |

Focused query-listwise tree10 result:

| Arm | Split | Rank | Reranker Top-1 | Pointwise Tree10 Top-1 | Result |
|---|---|---:|---:|---:|---|
| CNN hybrid listwise tree10 | held-out fish | genus | 69.4% | 84.0% | worse |
| CNN hybrid listwise tree10 | held-out fish | family | 88.0% | 95.4% | worse |
| CNN hybrid listwise tree10 | unseen-genera | family | 65.0% | 72.5% | worse |
| CNN hybrid listwise tree10 | unseen-genera | order | 83.2% | 85.8% | worse |

Focused query-pairwise tree10 result:

| Arm | Split | Rank | Reranker Top-1 | Pointwise Tree10 Top-1 | Result |
|---|---|---:|---:|---:|---|
| CNN hybrid pairwise tree10 | held-out fish | genus | 84.1% | 84.0% | similar |
| CNN hybrid pairwise tree10 | held-out fish | family | 95.3% | 95.4% | similar |
| CNN hybrid pairwise tree10 | unseen-genera | family | 72.4% | 72.5% | similar |
| CNN hybrid pairwise tree10 | unseen-genera | order | 86.0% | 85.8% | slight gain |

Calibration-transfer audit for the strongest pointwise hybrid tree10 reranker,
using seen-test target-0.99 thresholds:

| Target Split | Rank | Observed Precision | Coverage | Interpretation |
|---|---:|---:|---:|---|
| held-out fish | genus | 93.2% | 74.5% | useful but below target |
| held-out fish | family | 96.5% | 98.7% | useful but below target |
| held-out fish | order | 97.0% | 100.0% | useful but below target |
| unseen-genera | family | 77.0% | 93.5% | not production-safe |
| unseen-genera | order | 85.9% | 99.8% | not production-safe |

Independent selected-assignment calibrator:

| Input Reranker | Target Split | Rank | Observed Precision | Coverage | Result |
|---|---|---:|---:|---:|---|
| pointwise tree10 | held-out fish | genus | 93.7% | 62.3% | below target |
| pointwise tree10 | held-out fish | family | 96.4% | 98.8% | below target |
| pointwise tree10 | unseen-genera | family | 75.9% | 95.2% | not production-safe |
| pointwise tree10 | unseen-genera | order | 85.8% | 99.9% | not production-safe |
| pairwise tree10 | held-out fish | genus | 94.4% | 58.7% | below target |
| pairwise tree10 | unseen-genera | family | 72.6% | 99.6% | not production-safe |
| pairwise tree10 | unseen-genera | order | 86.1% | 99.9% | not production-safe |

First COI MLP top-50 reranker result:

| Split | Rank | Reranker Top-1 | p-distance Top-1 | Delta | Calibrated Coverage | Calibrated Precision |
|---|---:|---:|---:|---:|---:|---:|
| held-out fish | genus | 49.7% | 38.4% | +11.3 pp | 12.7% | 98.2% |
| held-out fish | family | 87.3% | 84.1% | +3.2 pp | 70.8% | 98.1% |
| held-out fish | order | 95.5% | 95.2% | +0.3 pp | 96.0% | 97.8% |
| unseen-genera | family | 71.3% | 69.1% | +2.2 pp | 50.0% | 96.1% |
| unseen-genera | order | 87.7% | 88.3% | -0.6 pp | 91.4% | 93.6% |

Interpretation: candidate-level DL gives a real ordering improvement for
genus/family in the held-out fish split and family/order in unseen-genera. The
tree10 version lowers calibration loss and improves top-1 genus/family ordering
again, so tree-neighborhood evidence is useful. Its target-0.99 threshold
transfer is still below the production standard, especially on unseen-genera
family precision, so it is an evidence-fusion result rather than the headline
operating point. The first query-listwise version is a negative result: it
reduced training loss under a listwise objective but did not improve the actual
rank-level operating metrics. Pairwise training is mixed, not a clear
replacement. The calibration-transfer and selected-assignment calibrator audits
are also clear: the reranker should currently be used as a
candidate-ordering/evidence-fusion layer, not as the final rank/no-call
decision maker.

## Next Experiments

1. Build a more realistic missing-reference calibration design; small
   seen-test-calibrated MLPs are not enough.
2. Evaluate whether reranker scores help the production p-distance policy as
   soft evidence rather than replacing thresholds.
3. Keep the reranker separate from final rank/no-call calibration to avoid
   hiding error sources.
