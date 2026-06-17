# DEPP / H-DEPP Notes For Paper 1

DEPP and H-DEPP are close predecessors. They should shape both the method and
the claim boundary.

## What DEPP Does

DEPP does not simply memorize a lookup table of species labels. It learns a
function:

> aligned gene sequence -> coordinate in a tree-distance-preserving space

The reference species tree provides the training target. For reference species,
DEPP knows:

- the input single-gene sequence,
- the species-tree position,
- the pairwise path distances between that species and other backbone species.

The neural model is trained so that distances between learned sequence
embeddings approximate distances on the species tree. Once trained, a new query
sequence is embedded into the same space. Distances from that query embedding to
backbone species are then used by a distance-based placement method such as
APPLES.

So DEPP is better described as:

> supervised metric learning from gene sequence space into species-tree
> distance space.

It is not just memorizing a tree map, although memorization is a real risk if
the evaluation split is not strict. For our work, strict held-out species and
held-out genera are essential.

## What H-DEPP Changes

H-DEPP keeps the same broad idea but changes the geometry.

DEPP uses Euclidean embeddings. H-DEPP uses hyperbolic embeddings because trees
are naturally hierarchical and branching. Hyperbolic space has negative
curvature and expands exponentially with radius, which lets it represent
tree-like structures with lower distortion than fixed-dimensional Euclidean
space in many settings.

In simple terms:

- Euclidean space is flat. Large trees can require many dimensions to preserve
  distances well.
- Hyperbolic space is curved. It gives exponentially more room as you move away
  from the origin, which fits branching hierarchies.

H-DEPP therefore asks:

> If the target is a tree, should the learned sequence embedding live in a
> tree-friendly geometry rather than ordinary Euclidean space?

Their own result is nuanced: hyperbolic embeddings can reduce distance
distortion, but that does not always guarantee better final placement.

## What We Should Learn From Them

### 1. Separate Distance Learning From Placement

Our current models directly rank candidate species from sequence-to-tree
embedding distance. DEPP suggests a cleaner decomposition:

1. learn sequence -> tree-distance space;
2. estimate distances from query to backbone species;
3. use a placement or rank/no-call layer on top.

This would make our method easier to compare with APPLES, EPA-ng, pplacer, and
SEPP.

### 2. Evaluate Distance Distortion Explicitly

Pearson/Spearman tree recovery is useful but incomplete. We should add:

- distance-bin error,
- relative distance-ratio error,
- near-neighbor preservation,
- clade/rank residuals,
- placement/rank accuracy as a function of distance distortion.

### 3. Treat Geometry As An Ablation

Paper 1 should not jump straight to "Mamba is the method." Better ablations:

- Euclidean tree-space objective,
- hyperbolic tree-space objective,
- rank-aware/hierarchical objective,
- sequence similarity baselines,
- phylogenetic placement baselines.

### 4. Keep The Barcode Assignment Layer Distinct

DEPP is about species-tree extension/placement. Our distinct added layer can be:

- candidate species ranking,
- genus/family/order recovery,
- calibrated rank/no-call,
- reference-gap diagnostics.

That is where our Paper 1 can become different from DEPP rather than a weaker
reinvention.

## How This Affects Paper 1

We should not claim:

> neural sequence-to-tree placement is new.

We can aim to show:

> in COI barcode assignment, tree-space learning plus calibrated rank/no-call
> evaluation gives a practical framework for biodiversity inference under
> missing references.

The direct next benchmark is to run phylogenetic placement baselines and score
them through the same rank/tree-distance layer.

## Current Paper 1 Translation

The first local source tables now partially implement the DEPP/H-DEPP lesson:

- `tree_recovery_metrics.csv` records end-to-end tree-distance recovery;
- `tree_distance_bin_summary.csv` and `tree_distance_sample_summary.csv`
  expose distance distortion by true tree-distance range;
- `candidate_ablation_rank_backoff.csv` begins to separate placement/ranking
  from the taxonomic decision layer by asking what rank is still recoverable
  when true species/genus/family candidates are hidden from saved rankings.

What remains:

- run EPA-ng/pplacer and score their placements through the same layer;
- add a true DEPP-style or hyperbolic ablation only after the classical
  placement comparator is working;
- replace post-hoc top-50 candidate ablation with strict tree-pruned candidate
  and/or retraining experiments if we want a strong missing-candidate claim.

## Sources

- DEPP: <https://pubmed.ncbi.nlm.nih.gov/35485976/>
- DEPP full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10198656/>
- H-DEPP: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/>
