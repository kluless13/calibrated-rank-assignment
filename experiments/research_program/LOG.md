# Research Program Log

## 2026-05-30

Decision: organize the work around the goal:

> Inference about biodiversity from imperfect molecular evidence.

Agreed build order:

1. Finish MarineMamba-Phylo validation.
2. Build COI rank-adaptive calibration/no-call.
3. Add reference/missingness diagnostics.
4. Formalize Eco-Phylo Posterior for 12S/eDNA.
5. Start Multi-Marker Shared Tree Space.
6. Keep hyperbolic geometry as an optional mathematical method track after
   baselines and controls.

Original paper structure decision, now superseded:

- Paper 1 combines MarineMamba-Phylo and rank-adaptive calibrated assignment.
- Paper 2 covers Eco-Phylo eDNA.
- Paper 3 covers Multi-Marker Shared Tree Space.
- Stalder/TAXDNA reproduction remains a supporting comparator track under
  `experiments/stalder_reproduction/`.

Updated decision on 2026-05-31:

- Merge Paper 2 into Paper 1.
- The merged Paper 1 is now the flagship methods paper:
  uncertainty-aware molecular biodiversity inference from short marker
  sequences.
- COI is the tree-space/retrieval/missing-reference benchmark.
- 12S/eDNA is the marker-ambiguity/ecological-context stress test.
- Multi-marker shared tree space remains the next separate paper/workstream.

Conceptual chain:

1. Barcode sequences can learn biological structure.
2. That structure can be evaluated as real species-tree recovery.
3. Tree-space scores can become calibrated rank assignment.
4. Reference and marker diagnostics explain uncertainty.
5. Ecological priors can improve real eDNA interpretation.
6. COI and 12S can be bridged through shared species coordinates.

## Method Contribution Status

We are on the verge of unique method contributions, but the language should
remain conditional until baselines, negative controls, and calibration are
complete.

The strongest candidate contribution is not "Mamba for barcodes." It is:

> biodiversity inference from imperfect molecular evidence by mapping marker
> sequences into biological coordinate systems, then calibrating taxonomic
> claims under reference gaps, marker ambiguity, phylogeny, and ecological
> context.

This contribution should survive encoder swaps. Mamba/SSM is one sequence
encoder; CNN, LSTM, Transformer, S5, k-mer, BLAST, and VSEARCH should be treated
as comparable sequence-evidence modules.

## What The Program Answers

1. Can short marker sequences encode biological relatedness rather than only
   closed-set labels?
2. Can a model place held-out taxa into a real fish species tree?
3. When a species label is not defensible, what rank is defensible?
4. Can we distinguish model error from marker ambiguity and missing references?
5. Can ecological context improve real eDNA interpretation without becoming
   hidden bias?
6. Can COI-rich reference structure help 12S/eDNA through a shared species-tree
   space?

## Why This Is Useful

- It turns barcode ML from forced classification into calibrated biological
  inference.
- It makes uncertainty actionable for biodiversity surveys.
- It shows when reference curation, additional markers, or ecological context
  are required.
- It supports practical eDNA reporting: species when supported, genus/family/
  order when not, and no-call when evidence is insufficient.

## Current Paper Map

Paper 1:

- Uncertainty-aware barcode and eDNA inference.
- Includes COI tree-space retrieval, Fernando-style placement comparators,
  vector-first learned retrieval, rank-adaptive calibration, reference
  diagnostics, 12S resolvability ceilings, and ecological-context eDNA
  validation.

Former Paper 2:

- Merged into Paper 1 as the 12S/eDNA work package.

Next separate paper:

- Multi-marker shared tree space.
- Tests whether COI and 12S can be connected through species-tree coordinates.

Optional method ablation:

- Hyperbolic/tree-geometry objective if cosine/tree-space results and baselines
  justify a mathematical geometry track.

## Strong 12S/eDNA Experiment Set

The strongest 12S/eDNA claim should not be "SSM solves species-level 12S."
Species-level 12S can only be solved when the necessary evidence exists:

1. the 12S fragment uniquely identifies the species;
2. geography/range eliminates sequence-equivalent alternatives;
3. community co-occurrence makes one candidate much more plausible;
4. another marker, such as COI, provides discriminative evidence;
5. the system abstains from species and reports genus/family honestly.

Therefore the merged Paper 1 eDNA experiments should ask what evidence is
sufficient:

1. 12S resolvability/oracle upper-bound.
2. Sequence vs tree vs geography vs co-occurrence decomposition.
3. Rank-adaptive species/genus/family/order/no-call.
4. Multi-marker shared species-tree space.
5. Reference-gap/active curation map.

Detailed plan:

- `experiments/paper2_eco_phylo_edna/STRONG_EXPERIMENTS.md`

Current status update:

- Paper 1 BLAST/VSEARCH/k-mer baselines, negative controls, diagnostics,
  CNN/biLSTM/Transformer benchmarks, CNN seed repeats, query-embedding exports,
  and first calibration/source tables are complete.
- Paper 1 EPA-ng/pplacer placement comparators are the current active Vast job.
- Paper 1 vector-first retrieval is now a tool direction, but not a novelty
  claim by itself because learned/vector barcode retrieval already has prior
  art.
- Paper 1 vector-first exact-cosine source tables have been generated locally
  from copied CNN/biLSTM/Transformer embeddings and CNN repeat embeddings.
- Paper 1 nearest-reference diagnostics have been fixed and regenerated with
  real tree-distance bins.
- Paper 1 placement-output scoring has an initial `jplace` parser/scorer
  scaffold ready for copied EPA-ng/pplacer outputs.
- Former Paper 2 exact and near-exact 12S resolvability maps are complete and
  now support the merged Paper 1 marker-information-ceiling argument.

Near-term timing:

- Finish/copy EPA-ng/pplacer placement outputs.
- Implement the placement scoring adapter and Fernando-style metrics.
- Build vector-first retrieval source tables from copied embeddings.
- Keep Eco-Phylo posterior ablations queued as part of the merged Paper 1 after
  the placement layer is stable.
- Begin multi-marker shared-tree inventory locally before any new GPU training.
