# Placement Comparator Decision

Decision: keep EPA-ng as the completed likelihood-placement comparator, keep
pplacer blocked unless a valid refpkg/stats model is supplied, and use official
APPLES 2.0.11 for the Fernando-style completeness-sweep comparator. Keep the
local APPLES-like distance-placement diagnostic as a lightweight proxy on the
clean held-out splits, not as an official APPLES result.

## Why Not pplacer Next

pplacer is not invalid scientifically, but our current run is invalid
operationally: it requires either a reference package or a model/stats file.
Without that, pplacer does not produce a placement result and should not be
reported.

Fixing pplacer would require extra model/refpkg construction and validation
before it can answer the current paper question. That is secondary because:

- EPA-ng already gives us a likelihood-based placement comparator on the clean
  splits.
- Fernando 2025 used EPA-ng and APPLES, so APPLES-style distance placement is
  the more direct missing comparator.
- pplacer would be useful as an additional likelihood-placement check, but it
  does not close the Fernando-aligned gap as directly as APPLES.

## Why APPLES-Style Was Added

APPLES is distance-based phylogenetic placement: given a fixed reference tree
and query-to-reference distances, it places the query on the tree by optimizing
a distance objective. This directly complements EPA-ng:

- EPA-ng tests alignment/likelihood placement.
- APPLES-style placement tests distance-to-tree placement.
- Neural tree-space encoders can then be compared against both classical
  placement families.

This matters because Paper 1 is not trying to prove that a neural encoder
replaces classical methods. It is testing whether fast retrieval and
tree-aware/rank-aware interpretation add value under missing and ambiguous
barcode evidence.

## Current Implementation

The local APPLES-like diagnostic has now been run for Eval C, seen-test, and
unseen-genera with:

- prepared EPA-ng placement inputs;
- aligned query/reference COI sequences;
- VSEARCH top-25 candidate neighborhoods;
- local p-distance scoring;
- the same Paper 1 rank and nearest-reference diagnostic layer.

Current nearest-reference match rates:

- Eval C: 54.4%.
- seen-test: 78.8%.
- unseen-genera: 22.1%.

These rows are useful because they give us a distance-placement proxy on the
same clean held-out splits, but they are not official APPLES and not exact
Fernando PCP.

Official APPLES has also now been run on the Fernando-style completeness
sweeps:

- official APPLES 2.0.11;
- default branch-length re-estimation;
- derooted reduced trees because APPLES default failed on rooted two-child
  reduced backbones;
- 30 matched sweeps: random and family-stratified 99/80/60/40/20% completeness,
  3 replicates each;
- final source tables under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_final_30/`.

Final diagnostic averages in that matrix:

- APPLES placed-clade genus/family/order: 32.8 / 57.2 / 65.6%.
- EPA-ng placed-clade genus/family/order: 17.3 / 45.2 / 57.0%.
- APPLES sister-clade any-overlap/exact: 42.5 / 21.4%.
- EPA-ng sister-clade any-overlap/exact: 14.8 / 3.2%.

## Future Implementation Choice

Use two levels in the manuscript:

1. Official APPLES 2.0.11 for the matched Fernando-style completeness sweeps.
2. The local APPLES-like distance-placement baseline only as a same-split
   lightweight diagnostic/proxy. This must stay labelled as APPLES-like, not
   official APPLES.

The local baseline should:

- use the same placement inputs already prepared for EPA-ng;
- compute query-to-reference distances from aligned COI sequences;
- restrict candidate placement edges to the neighborhood of the nearest
  reference leaves for tractability;
- score placements through the same Paper 1 placement/rank/no-call layer.

## Claim Boundary

The paper can now say:

> EPA-ng and official APPLES 2.0.11 were run on a Fernando-style matched
> completeness-sweep matrix built from our public fish COI setup.

It should not say:

> We reproduced Fernando's exact APPLES result or exact PCP implementation.

## References

- APPLES: Scalable Distance-Based Phylogenetic Placement with or without
  Alignments, Systematic Biology.
- APPLES-2 / distance-based phylogenetic placement with statistical support.
- Fernando, Fu, and Adamowicz 2025 fish COI backbone placement with EPA-ng and
  APPLES.
