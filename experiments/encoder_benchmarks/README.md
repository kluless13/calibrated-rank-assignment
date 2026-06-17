# Encoder Benchmarks

This folder tracks architecture-agnostic sequence encoder comparisons for the
MarineMamba research program.

The goal is not to prove that one architecture is universally best. The goal is
to separate:

- sequence evidence,
- tree-space objective,
- ecological posterior,
- calibration/no-call layer,
- reference and marker limitations.

## Core Question

Which sequence encoder is most useful inside a phylogeny-aware, ecology-aware,
calibrated biodiversity inference system?

## Candidate Encoders

Baseline encoders:

- k-mer nearest-neighbor / logistic features
- BLAST/VSEARCH nearest-hit
- CNN/TAXDNA-style encoder

Neural encoders:

- Mamba/SSM encoder
- LSTM/biLSTM encoder
- small Transformer encoder
- S5/state-space encoder
- pretrained DNA encoder, if feasible later

## Fair Comparison Contract

Every encoder should be evaluated with the same:

- input splits,
- candidate species universe,
- tree embeddings,
- top-k metrics,
- rank metrics,
- calibration/no-call protocol,
- ecological posterior,
- reference diagnostics.

The architecture should be the only changed component.

## Workstream Links

Paper 1:

- compare encoders on COI fish-tree recovery and calibrated assignment.

Merged Paper 1 eDNA work package:

- compare encoders inside the Eco-Phylo eDNA posterior.

Paper 3:

- compare marker-specific encoders in shared COI/12S tree space.

## Immediate Priority

Paper 1 baselines, negative controls, CNN/biLSTM/Transformer runs, query
embedding exports, and CNN seed repeats are now complete. The next priority is
not a broader architecture sweep; it is to put the completed encoders through
the same retrieval/calibration/placement scoring layer.

Near-term:

1. Use saved embeddings for vector-first retrieval benchmarks.
2. Compare vector-first retrieval against BLAST/VSEARCH/k-mer on speed,
   top-k recall, and rank-adaptive accuracy.
3. Score EPA-ng/pplacer/APPLES-style placement outputs through the same
   rank/tree-distance layer.
4. Add S5/BarcodeBERT/DNABERT-style models only after the current scoring layer
   is stable and the added architecture answers a specific question.
