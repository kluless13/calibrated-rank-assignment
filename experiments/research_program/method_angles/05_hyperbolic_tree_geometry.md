# Hyperbolic Tree Geometry

## Core Idea

Trees expand exponentially with depth. Hyperbolic space is designed for this
kind of hierarchy. Instead of mapping barcode sequences into Euclidean species
coordinates, train an encoder into hyperbolic tree space and compare distortion,
placement, retrieval, and rank/no-call behavior.

## Literature Boundary

Hyperbolic and tree-aware phylogenetic embedding already exists:

- Poincare embeddings for hierarchical representations:
  https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hie
- DEPP maps sequences to embeddings that preserve tree distances and then uses
  distance-based phylogenetic placement:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10198656/
- H-DEPP extends this idea with hyperbolic embeddings:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/
- Hyperbolic phylogenetic tree embedding work studies how well tree distances
  can be represented in hyperbolic space:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8058397/

Therefore, "hyperbolic phylogenetic embeddings" are not novel by themselves.

## Our Gap

Our test would be different:

- short barcode/eDNA queries, not full-gene species-tree extension only;
- production-style candidate retrieval plus rank/no-call;
- strict missing-reference validation;
- comparison to BLAST/VSEARCH/k-mer, EPA-ng, APPLES, and Euclidean tree-space
  encoders.

The question is whether hyperbolic geometry improves the downstream inference
system, not whether it can embed a tree.

## Experiments To Run

1. Fit hyperbolic species coordinates for the fish tree.
2. Train CNN/SSM/Transformer barcode encoders into hyperbolic coordinates.
3. Compare tree-distance distortion against Euclidean embeddings.
4. Compare candidate retrieval and missing-reference rank-backoff.
5. Compare against DEPP/H-DEPP-style distance placement where feasible.

## Current Evidence

Euclidean tree-space encoders already recover meaningful tree geometry. The
question is whether hyperbolic space gives lower distortion or better
missing-reference rank behavior.

## Success Criterion

Hyperbolic becomes paper-central only if it improves at least one of:

- tree-distance distortion;
- nearest-neighbor/clade preservation;
- unseen-genera rank-backoff;
- candidate calibration under missing references.

Otherwise it stays a methods extension, not the main paper.

