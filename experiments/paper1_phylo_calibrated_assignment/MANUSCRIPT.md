# Calibrated rank-adaptive biodiversity inference from DNA barcodes under missing references

**Working draft.** Target venue: *Ecological Informatics* (Elsevier). This is the
full COI pipeline paper; 12S cross-marker bridging and a learned eco-phylogenetic
posterior at scale are out of scope here and appear only as the marker-ceiling
boundary (§4.6), noted as directions for future work. Numbers are drawn from the
tracked source tables; see [SOURCE_TABLES.md](SOURCE_TABLES.md).

**Authors:** _[Author One]¹, [Author Two]², … [corresponding author ✉]_ — *to be
completed (names, ORCIDs, order).*
**Affiliations:** _¹[Institution]; ²[Institution] — to be completed._
**Corresponding author:** _[name, email] — to be completed._

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

DNA barcoding identifies organisms from a short, standardised stretch of DNA
(Hebert et al. 2003), and environmental DNA (eDNA) extends this to detecting
species from traces shed into water, soil, or air. Both rest on the same final step: comparing a query barcode
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

The value of reliable higher-rank assignment under incomplete references is
increasingly recognised — Villon et al. (2026) show a neural classifier
outperforming reference-based tools at genus and family when the query species is
absent — but existing approaches remain closed-set and uncalibrated; we treat the
problem as calibrated, open-set inference (positioned fully in §2).

This paper makes the following contributions:

1. **A reframing and an end-to-end pipeline** that returns calibrated
   species/genus/family/order/no-call decisions by fusing multiple, separately
   measured streams of evidence (§3.1).
2. **A missing-reference evaluation regime** — leakage-audited splits plus
   strict stress tests that hide species, genera, or families before training —
   that makes abstention testable (§3.5, §4.5).
3. **A conservative operating point with a measured 0% false-species-call rate**,
   shown to hold under *prospective, species-disjoint* calibration (§4.1).
4. **An honest characterisation of the learned representation**: it genuinely
   encodes phylogeny (verified with negative controls; §4.2), detects open-set
   novelty (§4.3), and sits on a measurable tree-vs-species Pareto frontier where
   matching classical species clustering costs the tree geometry (§4.4).
5. **Active reference-curation**: turning abstentions into a ranked list of which
   references to sequence next (§4.7).

We are explicit about prior art. Learned barcode-to-tree embeddings (Stalder et
al. 2025; DEPP, Jiang et al. 2023), the learned-vs-k-mer representation comparison
(kf2vec, Rachtman et al. 2025), neural barcode foundation models and vector
retrieval (BarcodeBERT, Millan Arias et al. 2026; DNABERT-S, Zhou et al. 2025;
TaxoTagger), probabilistic rank-aware assignment (PROTAX, Somervuo et al. 2016;
IDTAXA, Murali et al. 2018), and unsupervised sequence clustering (BIN,
Ratnasingham & Hebert 2013; ABGD; ASAP) all exist. We use them; we do not claim
to have invented them, and we do not claim to beat alignment-based methods at
fine-grained clustering. Our novelty is the integration, the calibrated
missing-reference regime, and the open-set detection axis.

## 2. Related work

We organise prior art by the axis of our pipeline it touches, and state plainly
where priority belongs to others.

**Tree-distance barcode embeddings.** Learning a representation in which sequence
distance approximates phylogenetic-tree distance is established. Stalder et al.
(2025) embed fish 12S eDNA barcodes into a phylogenetic space on the same Fish
Tree of Life we use, combine it with species co-occurrence for zero-shot
annotation of unknown sequences, and report a probability-calibration analysis;
DEPP (Jiang et al. 2023) learns to place sequences on a species tree. We adopt
this idea rather than claim it; our tree-geometry encoder is a component, and its
distinctive use here is to power open-set detection and rank back-off, not to
advance placement per se.

**Learned vs. k-mer representations.** kf2vec (Rachtman et al. 2025) already
established that learned barcode embeddings outperform raw k-mer frequencies for
distance and placement. Our k-mer control (§4.2) is a confirmation of that finding
on fish COI, not a new result.

**Open-set novelty detection.** Recognising that a query lies *outside* the
reference — rather than forcing it into the nearest known class — is the relatively
open axis. Fujisawa & Imai (2026) benchmark out-of-distribution detectors for
insect COI barcoding and show that detection degrades sharply on short fragments,
a caveat we inherit at the 12S marker ceiling (§4.6). No prior barcode method,
to our knowledge, integrates open-set detection into a calibrated rank/no-call
decision, and this is our lead model-level contribution (§4.3).

**Neural clustering and species delimitation.** Unsupervised recovery of species
from barcode embeddings exists in neural form (BarcodeBERT, Millan Arias et al.
2026; DNABERT-S, Zhou et al. 2025) and, classically, as identity-threshold or
gap-based delimitation (BIN, Ratnasingham & Hebert 2013; ABGD, Puillandre et al.
2012; ASAP, Puillandre et al. 2021), which we use as strong baselines (§4.4). We
concede that clustering-based rediscovery is not where we win.

**Closed-set neural assignment under missing references.** Closest to our
*motivation*, Villon et al. (2026) show that a closed-set, position-aware CNN
softmax classifier outperforms reference-based tools (Kraken2, OBITools, Lolo) at
genus- and family-level assignment when the query species is absent from training.
But their classifier is trained on a fixed set of families and genera: it cannot
abstain, cannot recognise a taxon outside its training classes, and reports no
calibrated error rate — indeed, because every test taxon's genus and family are
still training classes, a query from a genuinely unseen family is forced into a
known class with a confident softmax score, which is exactly the failure mode we
measure and prevent (§4.3, §4.5). We reframe the same problem as calibrated,
open-set inference: the deepest defensible rank or an explicit no-call with a
measured false-species-call rate, plus tree-geometry placement, novelty detection,
and an active-curation loop.

**Probabilistic and rank-aware assignment.** Probabilistic taxonomic classifiers
report rank-wise confidence (PROTAX, Somervuo et al. 2016; IDTAXA, Murali et al.
2018), and Zito et al. (2023) use Bayesian species-sampling priors to allow
unobserved taxa to be discovered at each rank — conceptually the nearest prior
work to rank-adaptive assignment with abstention. These are not deep-learning
systems and are not integrated with tree-geometry placement, a measured
false-species-call rate, or reference-curation; our contribution is that
integration under an audited missing-reference regime.

## 3. Methods

### 3.1 The pipeline architecture

The system is an **evidence compiler** (Figure 1). A query barcode passes through
ordered stations, each contributing a distinct, independently measurable kind of
evidence, before a final calibrated decision:

1. **Fast vector candidate retrieval.** A learned encoder maps the barcode to a
   fixed-length vector; approximate nearest-neighbour search returns candidate
   references in well under a millisecond per query (§4.1), so the pipeline scales
   to large samples.
2. **Classical sequence checks.** BLAST (Camacho et al. 2009), VSEARCH (Rognes et
   al. 2016), and pairwise p-distance measure how close the retrieved candidates
   truly are — the field's trusted, precise workhorses, retained as strong
   baselines and as evidence, not discarded.
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

### 3.2 Sequence encoder and training objective

The learned encoder is a character-level convolutional network. Nucleotide tokens
are embedded (32-dimensional, with a padding token), passed through three stacked
dilated 1-D convolutions (256 channels; kernel sizes 7, 5, 3 with dilations 1, 2,
4 respectively, each followed by GELU and dropout 0.1) to give a multi-scale
receptive field over the barcode, then reduced by concatenated masked mean- and
max-pooling and projected through a two-layer head (512→256→512, GELU, dropout) to
a **512-dimensional** embedding. We frame results as encoder-agnostic: the CNN is
the strongest encoder we tested, but the objective, not the architecture, is the
claim.

The training **target** is a per-species tree embedding: for the fish species on
the Fish Tree of Life we pre-learn 512-D vectors whose cosine distances match
patristic tree distances (gradient descent, AdamW). The encoder is then trained to
map a sequence to its species' tree vector under a composite objective,

> L = w_tree · L_cosine + w_species · L_contrastive,

where L_cosine = 1 − cos(prediction, tree target) is the tree-distance regression
term and L_contrastive is an InfoNCE loss (temperature 0.07) over candidate
reference species. Sweeping the weight ratio w_tree : w_species traces the
tree-vs-species Pareto frontier reported in §4.4; the tree-only anchor sets
w_species = 0. Training uses AdamW (learning rate 5×10⁻⁴, weight decay 0.01),
cosine-annealed over 40 epochs, batch size 64, fixed seed 1206.

### 3.3 Data, taxonomy mapping, and splits

Sequences come from a BOLD Teleostei COI pull (318,829 sequences spanning 23,663
nominal species). Because the training target requires a position on the Fish Tree
of Life (Rabosky et al. 2018; ~11,638 species with genetic data on the
time-calibrated actinopterygian backbone), we restrict to species present on that
tree, then deduplicate on exact sequence and on process ID and run a leakage audit
that removes held-out species before training and confirms zero intersection
between reference and evaluation sets. The resulting modelled set is partitioned
into a **reference** library of 3,839 species (the searchable set) and two
held-out regimes of increasing difficulty:

- **Eval C** — 531 species held out entirely (11,594 reads), with their genera
  still represented (species-level novelty within known genera).
- **Unseen genera** — 614 species whose genera are held out (9,148 reads;
  genus-level novelty).

*(This single lineage — raw BOLD pull → Fish-Tree-mapped subset → dedup and
leakage audit → the three splits — is the authoritative dataset description; the
earlier "318K sequences / 23,663 species" figure refers to the raw pull, not the
modelled set.)*

**Missing-reference stress tests.** Beyond holding species out, we additionally
*prune the candidate set* before training: separate runs hide all species, all
genera, or all families of the query, forcing the model to operate when a given
rank is entirely unsupported (§4.5).

### 3.4 Open-set novelty detector (DETECT)

Novelty detection uses features computed from a query's reference neighbourhood:
the top-1 similarity, the top-1-to-next margin, the genus consensus among the top
neighbours, the mean top-k similarity, and the number of reference neighbours
retained in the top-k (`ref_top1`, `ref_margin`, `ref_genus_consensus`,
`ref_topk_mean`, `ref_n_ref_in_topk`). We report both single-feature AUROC
(reference-only) and a multi-feature logistic detector trained and evaluated on
disjoint calibration and evaluation species so that reported detection reflects
transfer to unseen taxa (§4.3).

### 3.5 Calibration and evaluation protocol

**Prospective calibration.** Per-rank decision thresholds are fit at a target
assigned-precision of 0.99 on one set of species and applied to a *disjoint* set
never used to set them, repeated 30 times, so reported operating points reflect
transfer to genuinely unseen taxa rather than in-sample tuning (§4.1).

**Metrics.** Coverage (fraction of queries assigned rather than no-called),
assigned precision (fraction of assignments correct at their rank),
false-species-call rate (fraction confidently assigned a wrong species), tree
recovery (Pearson correlation between embedding distance and tree distance),
novelty AUROC, and adjusted mutual information (AMI) for clustering. Terms are
defined in the Glossary.

### 3.6 Software, versions, and reproducibility

Classical comparators are BLAST+ (Camacho et al. 2009), VSEARCH 2.28 (Rognes et
al. 2016), CD-HIT (Fu et al. 2012), and, for placement, EPA-ng (Barbera et al.
2019) and official APPLES 2.0.11 (Balaban et al. 2020). Neural models are
implemented in PyTorch; the fixed random seed (1206) is used throughout, and all
run commands, versions, and output paths are recorded in the dated ledgers under
`configs/runs/`. Code and derived split manifests are released with the paper.

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
established tools (Figure 5). At cluster count fixed to the true species number
(KMeans, k = 531), classical alignment clustering leads at the fine ranks:
VSEARCH 0.915 and cd-hit (Fu et al. 2012) 0.886 species AMI versus our
embedding's 0.874, while a frozen invertebrate-trained foundation model
(BarcodeBERT) reaches only 0.492.
**Our embedding wins at family level (0.756 vs 0.720 / 0.692)**, capturing coarse
tree structure that identity-threshold clustering fragments.

The k = 531 comparison, however, *understates* the learned representation, because
species delimitation naturally over-segments: intraspecific variation splits a
true species across several clusters, and the best classical results themselves
use ~1,200 clusters for 531 true species. When our embedding is allowed to
over-segment to a matched granularity via blind thresholding (1,229 clusters), its
species AMI rises to **0.915 — tying VSEARCH's 0.915 (1,203 clusters)** (Figure 6).
At matched cluster granularity, the learned representation is on par with the
classical gold standard at species clustering.

The honest caveat is the frontier (§4.4 continued, Figure 7): the configuration
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
rank (Figure 8). On Eval C: hiding species drops species retrieval to 0% while
genus/family/order remain recoverable (41.8 / 62.9 / 83.9% top-10); hiding genera
drops species and genus to 0% while family/order persist (56.3 / 75.1%); hiding
families leaves only order (40.9%). The pattern is identical on unseen genera.
This is direct evidence that the rank/no-call policy is grounded in available
evidence rather than overconfident extrapolation. A fixed-class classifier cannot
exhibit this behaviour: with no class outside its training taxa, a query from a
hidden family is necessarily forced into a known family with a confident score —
the failure mode our detector (§4.3) and back-off measure and prevent.

### 4.6 Classical and placement comparators, and the marker ceiling

We retain classical methods as strong baselines, not foils: BLAST recovers family
at 98.1% (Eval C) top-10, and we ran Fernando-*style* (Fernando et al. 2025)
completeness sweeps with EPA-ng (Barbera et al. 2019) and official APPLES 2.0.11
(Balaban et al. 2020; 30 sweeps over random and family-stratified backbones). We
report these as matched-protocol comparators and explicitly do not claim exact
reproduction of any prior placement-completeness protocol.

The **marker ceiling** motivates the whole rank-adaptive stance. For the shorter
12S region used in much fish eDNA, species-level resolution is frequently
impossible regardless of method — the marker simply does not carry the
information. We therefore treat 12S/eDNA as a boundary: genus and family are
reachable, species is marker-limited, and forcing species there would be
precisely the error this paper is designed to avoid. Cross-marker bridging that
could partially address this — mapping a short marker into the representation of a
more informative one — is a natural direction for future work.

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
its descendants), the learned-vs-k-mer representation comparison (kf2vec), neural
barcode foundation models and vector retrieval (BarcodeBERT, DNABERT-S,
TaxoTagger), probabilistic rank-aware assignment (PROTAX, IDTAXA), and
unsupervised species delimitation (BIN, ABGD, ASAP, and neural clustering). We also do not claim to
beat BLAST or VSEARCH — at species clustering they remain the gold standard, and
where we match them we do so only by sacrificing the tree geometry. The
contribution is the integration, the calibrated missing-reference regime, the
open-set detection axis, and the active-curation layer.

**Limits.** Species-level eDNA on the shorter 12S marker is intentionally out of
scope as a species claim; the cross-marker and learned-posterior extensions are
left to future work. The encoder here is a CNN; we frame results as
encoder-agnostic and do not claim a particular architecture is best. The current
system is a research pipeline, not a deployed service.

**Future work.** A conformal-prediction layer would turn the calibrated
thresholds into formal coverage guarantees; cross-marker bridging and a learned
eco-phylogenetic posterior would extend the approach to multi-marker eDNA.

## 6. Figures

- **Figure 1** — `fig_pipeline_architecture` — the evidence-compiler pipeline (§3.1).
- **Figure 2** — `fig4_prospective_calibration` — species-disjoint operating
  point; 0% false species across 30 repeats (§4.1).
- **Figure 3** — `fig1_place_audit_controls` — tree recovery vs k-mer baseline and
  shuffled-tree control (§4.2).
- **Figure 4** — `fig_detect_novelty` — open-set novelty AUROC by rank (§4.3).
- **Figure 5** — `fig2_rediscovery_headtohead` — classical vs neural species
  rediscovery (§4.4).
- **Figure 6** — `fig_rediscovery_granularity` — species AMI vs cluster
  granularity; embedding ties VSEARCH at matched ~1.2k clusters (§4.4).
- **Figure 7** — `fig3_tree_species_frontier` — the tree-vs-species Pareto
  frontier (§4.4).
- **Figure 8** — `fig_missing_reference_collapse` — rank collapse under hidden
  references (§4.5).

All figures are in `manuscript_assets/experiment1/figures/` (PNG + PDF) and are
regenerated by `scripts/figures/plot_experiment1_figures.py` and
`scripts/figures/plot_manuscript_figures.py`.

## Declarations

**Data availability.** Sequence data derive from public BOLD Teleostei COI records
and the Fish Tree of Life (Rabosky et al. 2018; https://doi.org/10.1038/s41586-018-0273-1).
The derived, leakage-audited split manifests (with per-record BOLD/process IDs) and
all source tables underlying the reported figures are released with the code
repository; raw accession lists will be deposited on acceptance. _[BOLD accession
list / archive DOI — to be completed.]_

**Code availability.** All analysis code, run ledgers, and figure scripts are in
the project repository (`calibrated-rank-assignment`); a tagged release with a
Zenodo DOI will accompany submission. _[Zenodo DOI — to be completed.]_

**Funding.** _[Funding sources / grant numbers — to be completed.]_

**Author contributions.** _[CRediT roles — to be completed.]_

**Declaration of competing interest.** _The authors declare no competing
interests._ _[Confirm at submission.]_

## 7. References

*Reference style: Elsevier (name–date). Journal titles abbreviated; to be
finalised against the Ecological Informatics guide via a CSL style at submission.*

Balaban, M., Sarmashghi, S., Mirarab, S., 2020. APPLES: scalable distance-based phylogenetic placement with or without alignments. Syst. Biol. 69, 566–578. https://doi.org/10.1093/sysbio/syz063

Barbera, P., Kozlov, A.M., Czech, L., Morel, B., Darriba, D., Flouri, T., Stamatakis, A., 2019. EPA-ng: massively parallel evolutionary placement of genetic sequences. Syst. Biol. 68, 365–369. https://doi.org/10.1093/sysbio/syy054

Camacho, C., Coulouris, G., Avagyan, V., Ma, N., Papadopoulos, J., Bealer, K., Madden, T.L., 2009. BLAST+: architecture and applications. BMC Bioinformatics 10, 421. https://doi.org/10.1186/1471-2105-10-421

Fernando, M.A.T.M., Fu, J., Adamowicz, S.J., 2025. Testing phylogenetic placement accuracy of DNA barcode sequences on a fish backbone tree: implications of backbone tree completeness and species representation. Ecol. Evol. 15, e70817. https://doi.org/10.1002/ece3.70817

Fu, L., Niu, B., Zhu, Z., Wu, S., Li, W., 2012. CD-HIT: accelerated for clustering the next-generation sequencing data. Bioinformatics 28, 3150–3152. https://doi.org/10.1093/bioinformatics/bts565

Fujisawa, T., Imai, T., 2026. Performance and limitations of out-of-distribution detection for insect DNA barcoding. Ecol. Evol. 16, e73112. https://doi.org/10.1002/ece3.73112

Hebert, P.D.N., Cywinska, A., Ball, S.L., deWaard, J.R., 2003. Biological identifications through DNA barcodes. Proc. R. Soc. B 270, 313–321. https://doi.org/10.1098/rspb.2002.2218

Jiang, Y., Balaban, M., Zhu, Q., Mirarab, S., 2023. DEPP: deep learning enables extending species trees using single genes. Syst. Biol. 72, 17–34. https://doi.org/10.1093/sysbio/syac031

Li, W., Godzik, A., 2006. Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences. Bioinformatics 22, 1658–1659. https://doi.org/10.1093/bioinformatics/btl158

Millan Arias, P., Sadjadi, N., Safari, M., Gong, Z., Wang, A.T., Haurum, J.B., Zarubiieva, I., Steinke, D., Kari, L., Chang, A.X., Lowe, S.C., Taylor, G.W., 2026. BarcodeBERT: transformers for biodiversity analyses. Bioinform. Adv. 6, vbag054. https://doi.org/10.1093/bioadv/vbag054

Murali, A., Bhargava, A., Wright, E.S., 2018. IDTAXA: a novel approach for accurate taxonomic classification of microbiome sequences. Microbiome 6, 140. https://doi.org/10.1186/s40168-018-0521-5

Puillandre, N., Brouillet, S., Achaz, G., 2021. ASAP: assemble species by automatic partitioning. Mol. Ecol. Resour. 21, 609–620. https://doi.org/10.1111/1755-0998.13281

Puillandre, N., Lambert, A., Brouillet, S., Achaz, G., 2012. ABGD, automatic barcode gap discovery for primary species delimitation. Mol. Ecol. 21, 1864–1877. https://doi.org/10.1111/j.1365-294X.2011.05239.x

Rabosky, D.L., Chang, J., Title, P.O., Cowman, P.F., Sallan, L., Friedman, M., Eastman, J.M., Brown, J.W., Alfaro, M.E., Wainwright, P.C., et al., 2018. An inverse latitudinal gradient in speciation rate for marine fishes. Nature 559, 392–395. https://doi.org/10.1038/s41586-018-0273-1

Rachtman, E., Jiang, Y., Mirarab, S., 2025. Machine learning enables alignment-free distance calculation and phylogenetic placement using k-mer frequencies (kf2vec). Mol. Ecol. Resour. e70055. https://doi.org/10.1111/1755-0998.70055

Ratnasingham, S., Hebert, P.D.N., 2013. A DNA-based registry for all animal species: the Barcode Index Number (BIN) system. PLoS ONE 8, e66213. https://doi.org/10.1371/journal.pone.0066213

Rognes, T., Flouri, T., Nichols, B., Quince, C., Mahé, F., 2016. VSEARCH: a versatile open source tool for metagenomics. PeerJ 4, e2584. https://doi.org/10.7717/peerj.2584

Somervuo, P., Koskela, S., Pennanen, J., Nilsson, R.H., Ovaskainen, O., 2016. Unbiased probabilistic taxonomic classification for DNA barcoding. Bioinformatics 32, 2920–2927. https://doi.org/10.1093/bioinformatics/btw346

Stalder, S., Sanchez, T., Volpi, M., Manel, S., Mouillot, D., Auber, A., Bruno, M., Marques, V., Albouy, C., Pellissier, L., 2025. Zero-shot deep learning for the annotation of unknown eDNA sequences using co-occurrences and phylogenetic embeddings. PLoS Comput. Biol. 21, e1013776. https://doi.org/10.1371/journal.pcbi.1013776

Villon, S., Mangeas, M., Berteaux-Lecellier, V., Vigliola, L., Lecellier, G., 2026. Fine-grained assignment of unknown marine eDNA sequences using neural networks. Biology (Basel) 15, 285. https://doi.org/10.3390/biology15030285

Zhou, Z., Wu, W., Ho, H., Wang, J., Shi, L., Davuluri, R.V., Wang, Z., Liu, H., 2025. DNABERT-S: pioneering species differentiation with species-aware DNA embeddings. Bioinformatics btaf188. https://doi.org/10.1093/bioinformatics/btaf188

Zito, A., Rigon, T., Dunson, D.B., 2023. Inferring taxonomic placement from DNA barcoding aiding in discovery of new taxa. Methods Ecol. Evol. 14, 529–542. https://doi.org/10.1111/2041-210X.14009

TaxoTagger, 2024. MycoAI: semantic search for DNA barcode taxonomy identification (software). https://github.com/MycoAI/taxotagger

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
