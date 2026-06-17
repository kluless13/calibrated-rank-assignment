# Method Angles

This folder tracks five publication-grade method angles for MarineMamba. The
common goal is inference about biodiversity from imperfect molecular evidence:
missing references, marker ambiguity, phylogeny, ecology, and uncertainty.

The important boundary is that most ingredients already exist somewhere:
BLAST/VSEARCH/k-mer search, vector DNA search, phylogenetic placement,
tree-aware neural embeddings, ecological priors, and selective classification.
The novelty target is not any one ingredient in isolation. The stronger target
is a pipeline that makes these evidence sources work together and returns the
deepest defensible taxonomic claim instead of a forced species label.

## Five Angles

1. [Fast Vector-First Barcode Retrieval](01_fast_vector_first_barcode_retrieval.md)
2. [Candidate-Level Eco-Phylo Posterior](02_candidate_level_eco_phylo_posterior.md)
3. [Rank-Adaptive No-Call Calibration](03_rank_adaptive_no_call_calibration.md)
4. [Multi-Marker Shared Tree Space](04_multi_marker_shared_tree_space.md)
5. [Hyperbolic Tree Geometry](05_hyperbolic_tree_geometry.md)

## Build Order

1. Finish production v1 for COI saved-embedding inference.
2. Turn the candidate-level Eco-Phylo posterior into a candidate-reranking model,
   not only a diagnostic scorer.
3. Recalibrate rank/no-call policies under true split transfer.
4. Use COI-rich supervision to shape 12S/eDNA candidate inference through the
   shared fish tree.
5. Try hyperbolic objectives only after Euclidean tree-space and placement
   comparators are fully summarized.

