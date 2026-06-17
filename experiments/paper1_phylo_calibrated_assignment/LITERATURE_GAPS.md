# Paper 1 Literature Gap Audit

> **NOVELTY CORRECTION (2026-06-17):** Superseded for novelty claims by
> `NOVELTY_AND_PRIOR_ART.md` (corrected after a verified literature audit). The
> gaps below were written before verifying Stalder 2025 (phylo-distance fish
> embedding, **same tree**), kf2vec 2025 (already ran the k-mer baseline), and
> BarcodeBERT 2026 / DNABERT-S (already cluster barcode embeddings to recover
> species). The embedding, baseline, and clustering are **prior art**; genuine
> novelty = open-set DETECT + integration on fish COI + the marker boundary.

This document records what appears to be genuinely open after checking the
closest literature. It is intentionally conservative: "not found yet" does not
mean "does not exist."

## What Is Already Done In The Literature

### 1. Query Sequence Placement On Fixed Trees

This is not new.

Classical phylogenetic placement tools such as pplacer, EPA-ng, SEPP, APPLES,
and related methods already address the question:

> Given a reference tree and a query sequence, where does the query belong on
> the tree?

Implication for Paper 1:

- We must benchmark or directly compare against these tools.
- We should not claim that "placing barcode sequences on trees" is new.

### 2. Fish COI Placement Onto A Fish Backbone

This is also not new.

Fernando, Fu, and Adamowicz tested COI barcode placement on a ray-finned fish
backbone tree using EPA-ng and APPLES. Their work varied backbone completeness
and sampling strategy, then placed missing species back onto reduced trees.
EPA-ng achieved roughly 70-78% correct placement across completeness settings,
and high-confidence placements performed better.

Implication for Paper 1:

- This is the closest direct ecological/evolutionary comparator.
- Our novelty cannot be "COI can place fish on a tree."
- Their evaluation is placement accuracy against reference-tree sister clades,
  not calibrated species/genus/family/order/no-call assignment.

### 3. Neural Single-Gene To Species-Tree Placement

This is not new either.

DEPP learns a neural mapping from single-gene sequences into Euclidean space so
that embedding distances approximate species-tree path distances. H-DEPP extends
the idea into hyperbolic space. C-DEPP addresses scaling via a tree-aware
ensemble.

Implication for Paper 1:

- Sequence-to-tree metric learning already exists.
- Hyperbolic geometry for this exact placement family already exists.
- Our unique contribution must be narrower than "neural tree embeddings."

### 4. Probabilistic / Uncertainty-Aware Taxonomic Assignment

This also exists.

PROTAX explicitly models known species, species without reference sequences, and
unknown species within the taxonomy. Work on taxonomic placement uncertainty and
reference-database limitations already argues that uncertainty must be reported
at multiple ranks.

Implication for Paper 1:

- Rank uncertainty and missing-reference reasoning are not new in isolation.
- Our contribution must integrate this with tree-space barcode embeddings and
direct placement baselines.

### 5. Ecology-Aware eDNA Assignment

TAXDNA already combines sequence evidence, phylogenetic embeddings,
co-occurrence, and geographic modulation for zero-shot 12S/eDNA assignment.

Implication for merged Paper 1:

- We should not frame the broad "sequence + tree + ecology improves eDNA" idea
  as ours.
- Our unique angle must be reliability, calibration, marker information
  ceilings, and transparent evidence decomposition.

### 6. Fast Learned / Vector Barcode Retrieval

Fast learned retrieval is not new by itself.

Examples now identified:

- LISA proposed learned indexes for DNA sequence search.
- LV Barcoding used locality-sensitive hashing for rapid DNA barcode
  identification.
- BarcodeBERT reports BLAST-comparable species-level barcode classification
  while running much faster.
- DNABERT-S learns species-aware DNA embeddings.
- TaxoTagger exposes a vector-database / semantic-search interface for DNA
  barcode identification.
- BarcodeMamba and BarcodeMamba+ position state-space models as efficient
  barcode encoders.

Implication for Paper 1:

- We should not claim that "embedding DNA barcodes and doing vector search" is
  new.
- We also should not claim that "a faster BLAST replacement" alone is our
  contribution.
- Vector-first retrieval is best treated as one execution mode inside a larger
  uncertainty-aware biodiversity inference system.

What still looks underdeveloped:

- vector retrieval tied directly to a real species-tree coordinate system;
- explicit missing-reference and out-of-candidate evaluation;
- rank-adaptive species/genus/family/order/no-call rather than forced species
  classification;
- direct comparison to EPA-ng/pplacer/APPLES and Fernando-style fish COI
  placement metrics;
- reference-gap diagnostics and active curation signals.

## Gaps That Still Look Real

### Gap 1: Unified Barcode-To-Tree Benchmark Across Neural, Similarity, And Placement Methods

I do not yet see a benchmark that evaluates all of these under one clean COI
split:

- BLAST/VSEARCH/k-mer,
- EPA-ng/pplacer/SEPP/APPLES-style placement,
- DEPP/H-DEPP-style neural placement,
- CNN/biLSTM/Transformer/Mamba barcode encoders,
- same held-out species and held-out genera,
- same candidate tree and taxonomy,
- same rank/no-call scoring layer.

Why this matters:

- Existing fish placement work compares placement tools, not barcode neural
  encoders and calibrated taxonomic outputs.
- Barcode foundation/model papers usually focus on taxonomic classification or
  retrieval, not external species-tree geometry and placement-style scoring.

Potential contribution:

> a transparent fish COI benchmark for sequence-to-tree biodiversity inference
> under missing-reference conditions.

### Gap 2: Rank-Adaptive Output After Tree Placement

Phylogenetic placement methods output edge placements/confidence. Barcode
classifiers output labels or probabilities. But real biodiversity reporting
often needs:

- species if justified,
- genus if species is ambiguous,
- family/order if only higher rank is justified,
- no-call if evidence is insufficient.

I do not yet see a fish COI placement benchmark that converts every method into
the same calibrated species/genus/family/order/no-call decision.

Potential contribution:

> placement and barcode assignment should be evaluated by the deepest
> defensible taxonomic rank, not forced species labels alone.

### Gap 3: Out-Of-Candidate / Missing-Reference Behavior As A First-Class Metric

Many works acknowledge missing references. Fewer make it the central evaluation
object:

- true species absent from sequence reference DB,
- true genus absent from sequence reference DB,
- true species present in candidate tree but absent from sequence references,
- true species absent from candidate tree entirely,
- biased clade sampling.

Fernando et al. tested backbone completeness and biased sampling for placement.
PROTAX models missing/unknown taxa probabilistically. The missing edge for our
work is to combine these ideas with neural tree-space embeddings and calibrated
rank/no-call outputs.

Potential contribution:

> reference gaps can be measured and converted into rank-aware uncertainty
> rather than treated as a caveat after top-1 assignment.

### Gap 4: Geometry Diagnostics For Barcode Embeddings

DEPP/H-DEPP evaluate distance preservation and placement. Barcode model papers
usually evaluate species prediction/retrieval. What seems underdeveloped is a
barcode-focused diagnostic suite:

- learned distance vs true tree distance bins,
- near-neighbor preservation,
- clade/rank residual maps,
- local distortion around sparse reference regions,
- species/genus/family/order recovery as a function of tree-distance error,
- comparison to BLAST/VSEARCH/k-mer distances as tree-distance predictors.

Potential contribution:

> barcode embeddings should be audited as biological coordinate systems, not
> only as classifiers.

### Gap 5: Method-Agnostic Evidence Fusion For Barcode Inference

Classical sequence similarity is often excellent when close references exist.
Tree-space neural encoders may help where exact references are missing.
Phylogenetic placement gives edge-level tree evidence. These sources are rarely
combined as calibrated evidence streams.

Potential contribution:

> a rank-aware posterior that combines sequence similarity, neural tree-space
> distance, placement confidence, nearest-reference diagnostics, and taxonomy.

This is more valuable than trying to make one architecture "win."

### Gap 6: Active Reference Curation From Model Uncertainty

Reference-database limitation papers identify missing taxa, mislabels, conflicts,
and low marker resolution. What seems less developed is a direct action layer:

- which missing species/genera most reduce uncertainty if sequenced?
- which clades cause the largest false species calls?
- which references appear mislabeled or inconsistent with tree placement?
- where does adding one COI reference collapse many genus/family-only calls to
  species-level calls?

Potential contribution:

> use calibrated barcode/tree uncertainty to prioritize reference-library
> curation.

### Gap 7: Vector-First Retrieval For Uncertain Biodiversity Inference

The fast learned-BLAST idea is only novel if it is not presented as vector
search alone.

The target system is:

1. encode reference barcodes into a vector index;
2. retrieve top-k candidates quickly for each query;
3. rerank only the top-k with identity/alignment/tree-aware scores;
4. calibrate the final output to species/genus/family/order/no-call;
5. report whether the limiting factor is marker ambiguity, missing references,
   low confidence, or conflicting evidence.

Potential contribution:

> fast vector retrieval as the front end for calibrated, tree-aware barcode
> inference under uncertain and incomplete molecular evidence.

This is distinct from BarcodeBERT/TaxoTagger-style fast classification only if
we demonstrate uncertainty handling and missing-reference behavior, not just
speed.

## Ranked Opportunity For Our Current Paper 1

### Strongest

Unified benchmark + rank-adaptive missing-reference evaluation.

This directly connects our current results, the classical baselines, DEPP/H-DEPP,
and fish COI placement literature.

### Strong

Tree-geometry diagnostic suite for barcode embeddings.

This is publishable only if paired with the comparator benchmark. Alone it is a
useful analysis, but probably not enough.

### Strong But Larger

Active reference curation from uncertainty.

This could become a second methods contribution or a strong extension section.

### Strong Tool Direction

Vector-first barcode retrieval with rank-adaptive tree-aware reranking.

This becomes publishable if benchmarked on both speed and scientific reliability:

- top-k candidate recall before reranking,
- speed and memory against BLAST/VSEARCH/k-mer,
- final rank-aware accuracy after reranking,
- false species-call rate when references are missing,
- comparison to Fernando-style EPA-ng/APPLES placement metrics.

### Riskier

Hyperbolic MarineMamba / hyperbolic barcode embeddings.

H-DEPP already exists, so this is not novel unless we connect it to barcode
rank/no-call assignment and show it changes biological decisions.

### Not Enough

Mamba vs CNN architecture leaderboard.

This is currently weak as a novelty route because the CNN result is strong and
architecture superiority is not the central scientific problem.

## Questions We Can Answer With Current And Running Experiments

1. Do neural barcode encoders recover species-tree geometry for held-out fish
   species and held-out genera?
2. How do neural tree-space encoders compare to BLAST/VSEARCH/k-mer under the
   same clean missing-reference split?
3. How do EPA-ng and pplacer behave on our exact Eval C/seen-test/unseen-genera
   setup?
4. Does a placement-style method produce more reliable rank-level calls than
   neural candidate ranking?
5. When the true species lacks a sequence reference, should the correct output
   be species, genus, family, order, or no-call?

## Current Local Progress Against The Gaps

Source tables now exist for the parts we can support from copied outputs:

- unified retrieval metrics across neural encoders, sequence baselines, and
  negative controls;
- tree-recovery metrics for Mamba/CNN/biLSTM/Transformer;
- sampled true-tree-distance versus learned-distance bin diagnostics;
- neighborhood preservation/enrichment against the candidate pool;
- post-hoc candidate-ablation/rank-backoff over saved top-50 predictions.

This advances Gap 1 and Gap 4, and gives an initial empirical handle on Gap 2
and Gap 3. It does not yet close them because:

- EPA-ng placement has now been copied/scored for Eval C, seen-test, and
  unseen-genera, but pplacer/SEPP/APPLES-style comparators are still missing;
- nearest-reference tree distances are fixed locally, but the copied remote
  diagnostics still preserve the older missing-distance artifacts;
- candidate-ablation is post-hoc over saved rankings, not a strict retrained
  out-of-candidate experiment;
- rank/no-call calibration still needs a cleaner validation protocol.

Vector-first source tables now also exist:

- `vector_first_retrieval_metrics.csv`
- `vector_first_runtime_comparison.csv`
- `ann_vector_retrieval_metrics.csv`
- `ann_vector_runtime_comparison.csv`
- `ann_vector_recall_against_exact.csv`

These advance Gap 7 as local exact-cosine and HNSW ANN proxies for learned
vector retrieval, but they do not yet close it because controlled hardware
timing, larger reference stress tests, and top-k reranking/calibration still
need to be implemented.

Reference diagnostics now also have fixed nearest-reference tree-distance bins
after normalizing the space-separated Fish Tree leaf labels to the underscore
labels used in the clean input packs. This gives Gap 3 a stronger empirical
handle: we can ask how retrieval and rank recovery change as the nearest
reference gets farther away on the tree.

Placement-output scoring is implemented in
`scripts/edna/score_fish_tree_placement_outputs.py`. EPA-ng now emits
placed-clade rank-containment, LWR-binned reliability, rank-backoff, and
tree-distance-to-placed-clade diagnostics for all three clean splits. This is
Fernando-adjacent, but it is not yet a full Fernando-style PCP score.

## Experiments To Fill The Gap

### Experiment A: Paper 1 Main Benchmark

Methods:

- BLAST,
- VSEARCH,
- k-mer,
- EPA-ng,
- pplacer,
- CNN,
- biLSTM,
- Transformer,
- Mamba.

Splits:

- seen-test,
- Eval C held-out species,
- unseen-genera.

Metrics:

- species/genus/family/order top-k,
- tree-distance error,
- nearest-reference distance,
- false species-call rate,
- deepest defensible rank,
- no-call/risk-coverage curves.

### Experiment B: Placement Output Adapter

Convert EPA-ng/pplacer edge placements into:

- nearest candidate species,
- placed clade,
- LWR/confidence,
- deepest common rank between placement neighborhood and true query.

Then compare against neural/classical candidate rankings.

### Experiment C: Candidate/Reference Ablation

Remove from the candidate/reference layer:

- true species,
- true genus,
- true family.

Score whether each method backs off correctly instead of overclaiming.

### Experiment D: Tree Geometry Diagnostics

For neural embeddings and sequence-similarity baselines:

- distance-bin residuals,
- clade residual maps,
- same-genus/family/order enrichment,
- nearest-neighbor preservation,
- correlation with true species-tree distance.

### Experiment E: Reference Curation Map

For every ambiguous query:

- identify nearest unresolved clade,
- estimate which missing reference would reduce ambiguity,
- rank clades by expected gain from new COI sequencing.

### Experiment F: Vector-First Retrieval Benchmark

Build a reference embedding index from the fish COI candidate database.

Methods:

- CNN embeddings,
- Mamba embeddings when available,
- Transformer/biLSTM embeddings,
- k-mer vector index,
- BLAST/VSEARCH/k-mer as speed/accuracy baselines.

Metrics:

- index-build time,
- query latency,
- memory footprint,
- top-k recall of true species / genus / family / order,
- final rank-adaptive accuracy after top-k reranking,
- false species-call rate when the true species/genus is absent.

## Claim If The Experiments Work

The strongest honest claim would be:

> Fish COI barcode inference is best treated as calibrated tree-aware evidence
> integration under missing references. Existing placement tools, sequence
> similarity, and neural encoders each solve different parts of the problem; a
> useful biodiversity system should report the deepest defensible rank and
> expose reference gaps rather than force species labels.

That is more novel than "MarineMamba beats BLAST" and more robust than "Mamba is
the best encoder."

## 2026-06-02 Deep-Dive Update

A more detailed literature-to-method audit is now recorded in:

- `experiments/paper1_phylo_calibrated_assignment/PIPELINE_NOVELTY_DEEP_DIVE.md`

The practical conclusion is that our differentiating work should be an
evidence compiler: a common candidate schema, calibrated rank/no-call decisions,
reason codes, and active reference-curation signals. Neural encoders, BLAST,
VSEARCH, EPA-ng, APPLES, and eDNA priors become evidence streams inside that
compiler rather than standalone novelty claims.

## Key Sources

- pplacer: <https://pmc.ncbi.nlm.nih.gov/articles/PMC3098090/>
- EPA-ng: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6368480/>
- Fish COI backbone placement: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11706799/>
- DEPP: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10198656/>
- C-DEPP: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11193062/>
- H-DEPP: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/>
- TAXDNA: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013776>
- PROTAX: <https://academic.oup.com/bioinformatics/article/32/19/2920/2196481>
- Reference-database challenges: <https://colab.ws/articles/10.1111%2F1755-0998.13746>
- LISA learned sequence search: <https://arxiv.org/abs/1910.04728>
- LV Barcoding: <https://arxiv.org/abs/1407.3348>
- BarcodeBERT: <https://pmc.ncbi.nlm.nih.gov/articles/PMC13008329/>
- DNABERT-S: <https://arxiv.org/abs/2402.08777>
- TaxoTagger: <https://mycoai.github.io/taxotagger/latest/>
- BarcodeMamba: <https://arxiv.org/abs/2412.11084>
