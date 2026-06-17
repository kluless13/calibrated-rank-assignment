# Calibrated rank-adaptive biodiversity inference from DNA barcodes under missing references

**Working draft — Paper 1 (Experiment 1).** Target venue: *Molecular Ecology
Resources*. This is the full COI pipeline paper; the 12S cross-marker bridging
work (MarkerMirror) and the learned Eco-Phylo posterior at scale are deferred to
Paper 2 and appear here only as the marker-ceiling boundary (§4.6). Numbers are
drawn from the tracked source tables; see [SOURCE_TABLES.md](SOURCE_TABLES.md).

---

## Abstract

DNA barcoding and environmental DNA (eDNA) are transforming biodiversity
monitoring, but standard taxonomic assignment assumes a reasonably complete
reference database — an assumption that fails for most of life, where the great
majority of species remain unsequenced or undescribed. Confronted with a query
whose true species is absent from the reference set, conventional pipelines
assign the nearest catalogued species, producing a confident but wrong label. We
reframe assignment as **calibrated, rank-adaptive inference under missing
references**: the goal is the deepest taxonomic rank the evidence can defend —
species, genus, family, or order — or an explicit *no-call*, with a measured
false-species-call rate. We present an evidence-compiler pipeline that fuses fast
vector retrieval, classical sequence comparison, a tree-aware learned embedding
with open-set novelty detection, and reference-gap diagnostics into one
calibrated decision, and converts abstentions into ranked reference-curation
priorities. On leakage-audited fish COI splits, a conservative operating point
reaches 95.8% coverage at 93.0% precision with **zero false species calls** on
held-out species, and **0.0% false species calls survive prospective,
species-disjoint calibration across 30 repeats**. The tree-aware embedding
recovers phylogenetic distance with Pearson 0.91 — confirmed against a k-mer
baseline (0.38) and a shuffled-tree negative control (0.09) — and detects
genus-level novelty at AUROC 0.84. We benchmark unsupervised rediscovery
honestly: classical alignment clustering remains the gold standard at species
level, our embedding wins at family level, and at matched cluster granularity the
learned representation ties VSEARCH — but only by trading away the tree geometry
that powers placement and novelty detection. The contribution is the integrated,
calibrated, missing-reference-aware system and its honest characterisation, not a
new embedding or a claim to beat alignment.

---

## 1. Introduction

DNA barcoding identifies organisms from a short, standardised stretch of DNA, and
environmental DNA (eDNA) extends this to detecting species from traces shed into
water, soil, or air. Both rest on the same final step: comparing a query barcode
against a **reference database** of barcodes from identified specimens. That step
silently assumes the reference is adequately populated for the query's taxon.

For most biodiversity this assumption is false. The overwhelming majority of
species on Earth are undescribed or unsequenced, and eDNA disproportionately
samples under-referenced taxa. When the true species is absent from the
reference, a forced top-1 assignment is not merely uncertain — it is a confident
wrong answer, and such answers propagate into biodiversity inventories,
conservation status assessments, and management decisions. The failure is
systemic and under-acknowledged: the tooling is optimised for the case where the
answer is in the database.

We argue that assignment under incomplete references is not a classification
problem but an **inference problem under missing information**. The scientifically
correct output is the deepest taxonomic rank the available evidence can defend —
and an honest *no-call* when even the coarsest rank is unsupported — accompanied
by a calibrated, *measured* error rate. A doctor who cannot identify the precise
pathogen names the most specific category the evidence supports rather than
inventing a strain; biodiversity assignment should do the same.

This paper makes the following contributions:

1. **A reframing and an end-to-end pipeline** that returns calibrated
   species/genus/family/order/no-call decisions by fusing multiple, separately
   measured streams of evidence (§2).
2. **A missing-reference evaluation regime** — leakage-audited splits plus
   strict stress tests that hide species, genera, or families before training —
   that makes abstention testable (§3, §4.5).
3. **A conservative operating point with a measured 0% false-species-call rate**,
   shown to hold under *prospective, species-disjoint* calibration (§4.1).
4. **An honest characterisation of the learned representation**: it genuinely
   encodes phylogeny (verified with negative controls; §4.2), detects open-set
   novelty (§4.3), and sits on a measurable tree-vs-species Pareto frontier where
   matching classical species clustering costs the tree geometry (§4.4).
5. **Active reference-curation**: turning abstentions into a ranked list of which
   references to sequence next (§4.7).

We are explicit about prior art. Learned barcode-to-tree embeddings (Stalder et
al. 2025; DEPP, Jiang et al. 2023), vector barcode retrieval (TaxoTagger; LISA;
BarcodeBERT), and unsupervised sequence clustering (BIN; ABGD; ASAP; and neural
variants) all exist. We use them; we do not claim to have invented them, and we
do not claim to beat alignment-based methods at fine-grained clustering. Our
novelty is the integration, the calibrated missing-reference regime, and the
open-set detection axis.

## 2. The pipeline

The system is an **evidence compiler** (Figure 1). A query barcode passes through
ordered stations, each contributing a distinct, independently measurable kind of
evidence, before a final calibrated decision:

1. **Fast vector candidate retrieval.** A learned encoder maps the barcode to a
   fixed-length vector; approximate nearest-neighbour search returns candidate
   references in well under a millisecond per query (§4.1), so the pipeline scales
   to large samples.
2. **Classical sequence checks.** BLAST, VSEARCH, and pairwise p-distance measure
   how close the retrieved candidates truly are — the field's trusted, precise
   workhorses, retained as strong baselines and as evidence, not discarded.
3. **Tree-aware evidence and open-set novelty (DETECT).** The encoder is trained
   so that embedding distance approximates phylogenetic distance; a query is
   placed relative to its neighbourhood on the tree, and a detector flags whether
   it looks like *something not in the reference at all*.
4. **Reference-gap and marker-resolvability diagnostics.** The pipeline estimates
   where the reference is thin and whether the marker can even resolve species in
   this region.
5. **Calibrated decision.** All streams are fused into a
   species/genus/family/order/no-call output at a chosen precision target, with
   **reason codes** explaining the call.
6. **Active curation.** Each no-call is converted into a concrete
   recommendation: which references to add to resolve it.

The design principle is **evidence separation**: each stream is measured on its
own before fusion, which is what makes the system auditable and its error rate
calibratable.

## 3. Evaluation design

**Splits.** All experiments use fish COI sequences mapped to the Fish Tree of
Life (Rabosky et al. 2018; 11,638 species). Splits are audited to be leakage-free
— species are removed before training, with exact-sequence and process-ID
deduplication, and an overlap audit confirming zero intersection. Two held-out
regimes test progressively harder novelty:

- **Eval C** — 531 species held out entirely (11,594 reads), with their genera
  still represented (tests species-level novelty within known genera).
- **Unseen genera** — 614 species across genera held out (9,148 reads; tests
  genus-level novelty).
- A **reference** set of 3,839 species supplies the searchable library.

**Missing-reference stress tests.** Beyond holding species out, we additionally
*prune the candidate set* before training: separate runs hide all species, all
genera, or all families of the query, forcing the model to operate when a given
rank is unsupported (§4.5).

**Prospective calibration.** Decision thresholds are fit on one set of species
and applied to a *disjoint* set never used to set them, repeated 30 times, so
reported operating points reflect transfer to genuinely unseen taxa rather than
in-sample tuning (§4.1).

**Metrics.** Coverage (fraction of queries assigned rather than no-called),
assigned precision (fraction of assignments correct at their rank),
false-species-call rate (fraction confidently assigned a wrong species), tree
recovery (Pearson correlation between embedding distance and tree distance),
novelty AUROC, and adjusted mutual information (AMI) for clustering. Terms are
defined in the Glossary.

## 4. Results

### 4.1 Calibrated rank/no-call with zero false species calls

At a conservative target precision (0.99) with p-distance reranking, the pipeline
reaches **95.8% coverage at 93.0% assigned precision on Eval C, and 92.3% /
83.9% on unseen genera, with a false-species-call rate of 0.0% in both** — under
missing references it backs off to a correct broader rank rather than inventing a
species.

Crucially, this holds **prospectively**. Fitting per-rank thresholds on a
calibration set of species and applying them to a *disjoint* evaluation set
(30 repeats) yields 0.923 coverage, 0.900 assigned precision, and a
false-species-call rate of **0.0% that survives every single repeat** (Figure 2).
Assignments concentrate at genus, family, and order with explicit no-calls; the
system never confidently names a species it cannot support. This prospective,
species-disjoint result is the central claim of the paper: the safety property is
not an artefact of in-sample threshold tuning.

Retrieval underpinning this is fast: 0.40 ms/query for exact vector search and
0.0048 ms/query with an HNSW index, exclusive of downstream reranking.

### 4.2 The tree-aware embedding encodes real phylogeny

The encoder's notion of barcode similarity aligns with the fish phylogeny:
embedding distance correlates with tree distance at **Pearson 0.91** on held-out
species. Two controls confirm this is genuine evolutionary signal, not a
sequence-similarity artefact (Figure 3):

- **k-mer baseline.** Raw 6-mer cosine distance on the same Eval C split recovers
  the tree at only 0.375 — the learned embedding is 2.4× better.
- **Shuffled-tree negative control.** Retraining the encoder on a randomly
  permuted tree collapses true-tree recovery from 0.919 to **0.094**, exactly as
  expected if the 0.91 reflects real structure the model learned from correct
  targets.

This phylogenetic geometry is what lets the system place queries — including
those whose species is absent — near their true relatives, and is the basis for
both rank back-off and novelty detection.

### 4.3 Open-set novelty detection (DETECT)

The pipeline can recognise that a query comes from outside the reference rather
than forcing it into a known slot (Figure 4). Genus-level novelty (queries from
held-out genera) is detected at **AUROC 0.84**; a multi-feature detector trained
across the species split reaches 0.77; species-level novelty — distinguishing a
held-out species from its known congeners — is the hard limit at 0.68. Reliable
detection of higher-rank novelty is the capability that makes principled
abstention, and ultimately honest discovery, possible. This open-set axis is the
one cleanly novel result at the model level.

### 4.4 Where learned representations sit: an honest rediscovery benchmark

We benchmark **unsupervised species rediscovery** — clustering reads from
held-out species and asking whether clusters recover true taxa — against
established tools (Figure 6). At cluster count fixed to the true species number
(KMeans, k = 531), classical alignment clustering leads at the fine ranks:
VSEARCH 0.915 and cd-hit 0.886 species AMI versus our embedding's 0.874, while a
frozen invertebrate-trained foundation model (BarcodeBERT) reaches only 0.492.
**Our embedding wins at family level (0.756 vs 0.720 / 0.692)**, capturing coarse
tree structure that identity-threshold clustering fragments.

The k = 531 comparison, however, *understates* the learned representation, because
species delimitation naturally over-segments: intraspecific variation splits a
true species across several clusters, and the best classical results themselves
use ~1,200 clusters for 531 true species. When our embedding is allowed to
over-segment to a matched granularity via blind thresholding (1,229 clusters), its
species AMI rises to **0.915 — tying VSEARCH's 0.915 (1,203 clusters)** (Figure 7).
At matched cluster granularity, the learned representation is on par with the
classical gold standard at species clustering.

The honest caveat is the frontier (§4.4 continued, Figure 8): the configuration
that reaches 0.915 species AMI is the species-leaning variant, whose tree recovery
falls to ~0.59. Sweeping the loss weighting between tree-distance and
species-contrastive objectives traces a strict Pareto frontier — keep tree
recovery ≥ 0.85 and species clustering caps near 0.86; match VSEARCH at species
and tree recovery collapses to ≤ 0.66. **No single weighting holds both.** A
representation that matches alignment at species clustering is effectively a
species-contrastive model that has discarded the tree geometry on which placement
and novelty detection depend. The model component is therefore a frontier, not a
do-everything embedding — a finding, not a defeat.

### 4.5 Missing-reference stress: abstention is principled

When a rank's references are hidden before training, the model does not
hallucinate that rank — it collapses to zero there and recovers the next broader
rank (Figure 5). On Eval C: hiding species drops species retrieval to 0% while
genus/family/order remain recoverable (41.8 / 62.9 / 83.9% top-10); hiding genera
drops species and genus to 0% while family/order persist (56.3 / 75.1%); hiding
families leaves only order (40.9%). The pattern is identical on unseen genera.
This is direct evidence that the rank/no-call policy is grounded in available
evidence rather than overconfident extrapolation.

### 4.6 Classical and placement comparators, and the marker ceiling

We retain classical methods as strong baselines, not foils: BLAST recovers family
at 98.1% (Eval C) top-10, and we ran Fernando-*style* completeness sweeps with
EPA-ng and official APPLES 2.0.11 (30 sweeps over random and family-stratified
backbones). We report these as matched-protocol comparators and explicitly do not
claim exact reproduction of any prior placement-completeness protocol.

The **marker ceiling** motivates the whole rank-adaptive stance. For the shorter
12S region used in much fish eDNA, species-level resolution is frequently
impossible regardless of method — the marker simply does not carry the
information. We therefore treat 12S/eDNA as a boundary: genus and family are
reachable, species is marker-limited, and forcing species there would be
precisely the error this paper is designed to avoid. The cross-marker bridging
that partially addresses this (MarkerMirror) is the subject of Paper 2.

### 4.7 Active reference-curation

Because abstentions are informative, the pipeline ranks them: each no-call carries
a reason code (species absent from reference, marker-limited, retrieval-weak) and
a recommended next action, producing a prioritised list of references whose
addition would resolve the most queries. This converts "I don't know" from a dead
end into a directed sequencing agenda — the value-of-information layer that makes
the honest system also a useful one.

## 5. Discussion

**What this changes.** Treating assignment as calibrated rank-adaptive inference
makes biodiversity calls that are safe under the realistic condition of an
incomplete reference: the system reports the deepest defensible rank with a
measured, controllable error rate and abstains rather than fabricating species.
The prospective, species-disjoint 0% false-species result is what makes this
deployable rather than merely aspirational.

**Conceded priorities.** We build on, and do not claim to have invented: learned
barcode-to-tree embeddings (Stalder et al. 2025, on the same fish tree; DEPP and
its descendants), the learned-vs-k-mer representation comparison (kf2vec), vector
barcode retrieval (TaxoTagger, LISA, BarcodeBERT), and unsupervised species
delimitation (BIN, ABGD, ASAP, and neural clustering). We also do not claim to
beat BLAST or VSEARCH — at species clustering they remain the gold standard, and
where we match them we do so only by sacrificing the tree geometry. The
contribution is the integration, the calibrated missing-reference regime, the
open-set detection axis, and the active-curation layer.

**Limits.** Species-level eDNA on the shorter 12S marker is intentionally out of
scope as a species claim; the cross-marker and learned-posterior extensions are
deferred to Paper 2. The encoder here is a CNN; we frame results as
encoder-agnostic and do not claim a particular architecture is best. The current
system is a research pipeline, not a deployed service.

**Future work.** A conformal-prediction layer would turn the calibrated
thresholds into formal coverage guarantees; cross-marker bridging and a learned
eco-phylogenetic posterior extend the approach to multi-marker eDNA (Paper 2).

## 6. Figures

- **Figure 1** — `fig_pipeline_architecture` — the evidence-compiler pipeline.
- **Figure 2** — `fig4_prospective_calibration` — species-disjoint operating
  point; 0% false species across 30 repeats.
- **Figure 3** — `fig1_place_audit_controls` — tree recovery vs k-mer baseline and
  shuffled-tree control.
- **Figure 4** — `fig_detect_novelty` — open-set novelty AUROC by rank.
- **Figure 5** — `fig_missing_reference_collapse` — rank collapse under hidden
  references.
- **Figure 6** — `fig2_rediscovery_headtohead` — classical vs neural species
  rediscovery.
- **Figure 7** — `fig_rediscovery_granularity` — species AMI vs cluster
  granularity; embedding ties VSEARCH at matched ~1.2k clusters.
- **Figure 8** — `fig3_tree_species_frontier` — the tree-vs-species Pareto
  frontier.

All figures are in `manuscript_assets/experiment1/figures/` (PNG + PDF) and are
regenerated by `scripts/figures/plot_experiment1_figures.py` and
`scripts/figures/plot_manuscript_figures.py`.

## 7. Key references (to be completed)

Rabosky et al. 2018 (Fish Tree of Life); Stalder et al. 2025 (fish 12S
phylogenetic embedding); Jiang et al. 2023 (DEPP); BarcodeBERT; kf2vec;
Ratnasingham & Hebert 2013 (BIN); Puillandre et al. (ABGD); Barbera et al.
(EPA-ng); Balaban et al. (APPLES); TaxoTagger; LISA. Full bibliography pending.

---

## Glossary

- **DNA barcode** — a short standardised DNA region used for species
  identification (e.g. COI in animals).
- **eDNA** — environmental DNA: genetic material shed into the environment,
  enabling detection without observing the organism.
- **Reference database** — the catalogue of barcodes from identified specimens
  that queries are compared against.
- **Rank / rank-adaptive** — the taxonomic level (species, genus, family, order)
  of an answer; rank-adaptive means choosing the deepest defensible level.
- **No-call / abstention** — declining to assign because evidence is
  insufficient.
- **Coverage** — fraction of queries assigned (not no-called).
- **Assigned precision** — fraction of assignments correct at their reported rank.
- **False-species-call rate** — fraction of queries confidently assigned a wrong
  species; the error driven to zero here.
- **Prospective / species-disjoint calibration** — fitting thresholds on one set
  of species and testing on a non-overlapping set, so results reflect transfer to
  unseen taxa.
- **Tree recovery (Pearson)** — correlation between embedding distance and
  phylogenetic-tree distance; 1.0 is perfect, ~0 is none.
- **AUROC** — area under the ROC curve; 0.5 is chance, 1.0 is perfect separation
  (here, known vs novel).
- **AMI (adjusted mutual information)** — clustering agreement with true labels;
  0 is random, 1 is perfect.
- **Pareto frontier** — the set of achievable trade-offs where improving one
  objective (species clustering) necessarily worsens another (tree recovery).
- **HNSW** — Hierarchical Navigable Small World, a fast approximate
  nearest-neighbour index.
