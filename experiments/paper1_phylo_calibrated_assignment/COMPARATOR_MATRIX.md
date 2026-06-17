# Paper 1 Comparator Matrix

Paper 1 should be positioned against three neighboring literatures:

1. classical barcode and phylogenetic placement methods,
2. neural phylogenetic reconstruction methods,
3. tree and hyperbolic representation learning.

Not all of these are equally direct baselines. The key is to separate
executable benchmarks from conceptual comparators.

## Core Claim Boundary

Paper 1 is not only a model leaderboard. The defensible core question is:

> Can barcode sequence encoders learn a species-tree coordinate system, and can
> that coordinate system support calibrated rank/no-call assignment when exact
> species evidence is missing?

That puts the paper between barcode assignment and phylogenetic placement.

## Uniqueness Audit

The broad idea is not unique:

- Classical phylogenetic placement already places query sequences onto fixed
  reference trees.
- DEPP already frames single-gene sequence-to-species-tree placement as a
  supervised deep-learning problem.
- A recent fish-specific study already tested COI barcode placement onto a fish
  backbone tree with EPA-ng and APPLES.

Therefore Paper 1 should not claim:

> nobody has used a sequence to place a taxon onto a species tree.

It should claim a narrower contribution only if our experiments support it:

> a transparent barcode-to-tree benchmark for fish COI that compares neural
> encoders, classical similarity, and phylogenetic placement under held-out
> species/genera, then converts tree-space evidence into calibrated
> species/genus/family/order/no-call assignment.

The likely novelty is the integration and evaluation protocol, not the existence
of phylogenetic placement itself.

## Comparator Tiers

### Tier 1: Direct Benchmarks

These should be benchmarked in the Paper 1 results if feasible.

#### BLAST / VSEARCH / k-mer

Status: implemented.

Role:

- classical reference-backed assignment;
- strong higher-rank retrieval baseline;
- exposes the species-reference limitation when true held-out species are not
  in the reference sequence database.

Required metrics:

- species/genus/family/order top-k;
- no-call/calibration curves;
- nearest-reference distance;
- tree-distance prediction from similarity.

#### Phylogenetic Placement: pplacer, EPA-ng, SEPP

Status: runner and input-preparation scripts exist. EPA-ng has completed for
Eval C, seen-test, and unseen-genera. Outputs have been copied/scored with
placed-clade containment, LWR-binned rank summaries, rank-backoff summaries,
and tree-distance-to-placed-clade diagnostics. Pplacer is not a valid result
yet because it needs a reference package or stats/model file.

Role:

- closest classical field for "where does this sequence belong on a fixed
  reference tree?";
- directly tests whether neural barcode-to-tree embeddings add value over
  likelihood/alignment placement.

Why it matters:

- pplacer/EPA-ng/SEPP solve a nearby problem with explicit evolutionary models.
- If they perform strongly, our claim must shift toward speed, calibration,
  missing-reference behavior, or rank-aware outputs.
- If they struggle on short COI barcode fragments or scale poorly to this
  tree, that is important evidence.

Implementation sketch:

1. Export the clean fish reference tree to Newick.
2. Build a reference COI multiple sequence alignment for training/reference
   species.
3. Align Eval C and unseen-genera queries to the reference MSA.
4. Run pplacer and/or EPA-ng on the fixed reference tree.
5. Map placement edges to nearest candidate species/clades.
6. Score:
   - tree-distance error,
   - nearest clade rank,
   - species/genus/family/order top-k where possible,
   - abstention/no-call based on placement uncertainty.

This is a high-value Paper 1 addition.

Current implementation artifacts:

- `scripts/edna/prepare_fish_tree_placement_inputs.py`
- `experiments/paper1_phylo_calibrated_assignment/runs/04_vast_phylo_placement_baselines.sh`

Required before claims:

- keep EPA-ng as the completed likelihood-placement comparator;
- keep pplacer blocked unless a valid refpkg/stats model is supplied;
- implement APPLES or an explicitly labelled APPLES-like distance-placement
  comparator next;
- score placement output through the same tree-distance/rank/no-call metrics.

#### Fish COI Backbone Placement

Status: prior work exists; benchmark alignment needed.

Fernando, Fu, and Adamowicz tested placement of COI barcode sequences onto a
Fish Tree of Life-style backbone using EPA-ng and APPLES while varying backbone
tree completeness and species representation.

Role:

- directly overlaps with our biological domain and marker;
- sets a minimum bar for any COI tree-placement claim;
- motivates testing our neural/tree-space methods against EPA-ng/APPLES-style
  placement under the same held-out splits.

Paper 1 implication:

- We should cite this work directly.
- Our paper should not present fish COI placement as new by itself.
- Our added value must come from the calibrated neural/classical comparison,
  missing-reference behavior, rank/no-call decisions, and transparent
  tree-geometry diagnostics.

#### DEPP / H-DEPP

Status: prior work exists; very close conceptually.

DEPP learns a neural mapping from single-gene sequences into a space where
distances preserve a reference species tree, then uses distance-based placement.
H-DEPP extends this idea with hyperbolic embeddings.

Role:

- closest neural predecessor to our sequence-to-tree objective;
- makes it unsafe to claim "neural sequence-to-tree placement" as our unique
  invention;
- gives us a strong design and evaluation reference.

Paper 1 implication:

- We should compare conceptually at minimum.
- If feasible, we should benchmark DEPP/H-DEPP or reproduce the DEPP-style
  objective on our clean fish COI split.
- Our distinct angle should be barcode/eDNA assignment behavior: candidate
  retrieval, taxonomic rank recovery, calibration, no-call, and reference-gap
  diagnostics.

### Tier 2: Conceptual / Partial Benchmarks

These should be compared in Related Work and only benchmarked if the interface
can be made fair.

#### Deep Learning for Phylogenetic Reconstruction

Examples:

- CNN quartet/topology inference from alignments,
- Phyloformer-style neural phylogenetic reconstruction.

Role:

- shows that neural models can learn phylogenetic signal;
- distinguishes our setting from de novo tree reconstruction.

Why not a direct baseline:

- these methods usually infer tree topology or pairwise distances from
  alignments/sets of sequences;
- Paper 1 is single-query barcode placement into a known species tree with
  candidate/rank/no-call outputs.

How to use:

- cite as evidence that neural phylogenetic signal learning is plausible;
- compare problem formulation, not necessarily raw accuracy;
- optionally add a small distance-reconstruction comparison if inputs can be
  adapted without changing the question.

### Tier 3: Representation Geometry / Method Extensions

These should shape the method section and ablations.

#### Tree / Hyperbolic Embeddings

Examples:

- Poincare embeddings for hierarchical representation,
- hyperbolic phylogenetic tree embeddings,
- hyperbolic/deep phylogenetic placement.

Role:

- supports the idea that tree geometry is a real representational target;
- motivates Euclidean-vs-hyperbolic tree-space ablations.

Paper 1 use:

- current runs use Euclidean/cosine tree embeddings;
- next method ablation can compare:
  - Euclidean tree embeddings,
  - hyperbolic tree embeddings,
  - rank/hierarchy loss,
  - mixed tree-distance + taxonomic-rank loss.

This is a method-extension path, not required before the first tree-geometry
benchmark is complete.

## What Goes In Paper 1

Yes, all three areas belong in Paper 1, but with different jobs:

- BLAST/VSEARCH/k-mer: direct baselines, already in results.
- pplacer/EPA-ng/SEPP: direct classical placement baselines to add if feasible.
- neural phylogenetic reconstruction: related work and problem contrast.
- hyperbolic/tree embeddings: method motivation and possible ablation.

## Apples-To-Apples Rules

The comparisons are not automatically apples-to-apples. They become fair only
if we force the same biological question and score the outputs through the same
evaluation layer.

### Fair Direct Comparison

These can be made close to apples-to-apples:

- BLAST/VSEARCH/k-mer,
- EPA-ng / pplacer / SEPP,
- APPLES-style distance placement,
- DEPP / H-DEPP or a faithful DEPP-style reproduction,
- our CNN/biLSTM/Transformer/Mamba tree-space encoders.

Required constraints:

- same clean COI splits:
  - Eval C held-out species,
  - seen-test reference-backed species,
  - unseen-genera held-out genera;
- same reference sequence set for methods that require reference sequences;
- same candidate species tree for methods that require a tree;
- true held-out query sequences absent from the reference/training sequence
  database;
- same taxonomy normalization;
- same final scoring:
  - tree-distance error,
  - nearest candidate/clade,
  - species/genus/family/order top-k where the method can produce rankings,
  - calibrated species/genus/family/order/no-call.

### Not Naturally Apples-To-Apples

These are not direct score-table competitors unless adapted carefully:

- CNN quartet/topology inference,
- Phyloformer-style de novo tree reconstruction,
- generic tree/hyperbolic embedding papers.

They answer different questions:

- infer topology from a set/alignment of sequences;
- estimate a full distance matrix/tree;
- embed tree-like structure.

For Paper 1 they should mainly define context and motivate diagnostics, not be
forced into the main benchmark table unless we build a fair adapter.

### Output Harmonization

Different method families output different objects:

- BLAST/VSEARCH/k-mer output ranked reference hits or similarities.
- EPA-ng/pplacer/SEPP output edge placements and placement confidence.
- APPLES/DEPP-style methods output tree-distance-based placements.
- Our neural encoders output candidate-species scores/distances.

To compare them, Paper 1 needs a shared scoring adapter:

1. Convert every method output into either:
   - ranked candidate species, or
   - a placed edge/clade on the same reference tree.
2. Derive the deepest defensible taxonomic rank from that output.
3. Score all methods on the same query table.
4. Keep method-native confidence as a calibration feature, but do not pretend
   all confidence scores mean the same thing.

This makes the comparison honest: the scientific question is shared, while the
method-specific outputs remain transparent.

## Immediate Action List

1. Finish/copy the active EPA-ng/pplacer queue.
2. Inspect resulting `jplace` files and implement the shared placement scorer.
3. Add Fernando-style metrics:
   - PCP-like placement correctness,
   - placement support / LWR bins,
   - tree-distance placement error,
   - deepest defensible taxonomic rank.
4. Add APPLES or an APPLES-equivalent distance-placement baseline if feasible.
5. Build vector-first retrieval source tables from copied neural embeddings.
6. Evaluate whether DEPP/H-DEPP can be run as-is or whether we should implement
   a faithful DEPP-style baseline using our existing tree embeddings and COI
   split.
7. Only then decide whether hyperbolic/tree-embedding ablations are needed.

## Literature Anchors

- pplacer: <https://pmc.ncbi.nlm.nih.gov/articles/PMC3098090/>
- EPA-ng: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6368480/>
- SEPP: <https://pubmed.ncbi.nlm.nih.gov/22174280/>
- Fish COI backbone placement with EPA-ng/APPLES: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11706799/>
- DEPP: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10198656/>
- H-DEPP: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/>
- CNN topology inference: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8204903/>
- Phyloformer: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11965795/>
- Poincare embeddings: <https://papers.neurips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations>
- Hyperbolic phylogenetic placement: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/>
- Hyperbolic phylogenetic tree embeddings: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8058397/>
