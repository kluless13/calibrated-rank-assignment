# Merged Paper 1 Manuscript Outline

> **NOVELTY CORRECTION (2026-06-17):** Superseded for novelty claims by
> `NOVELTY_AND_PRIOR_ART.md` (corrected after a verified literature audit).
> Concede priority on the embedding (Stalder 2025 — same fish tree; DEPP 2023),
> the k-mer baseline (kf2vec 2025), and neural clustering (BarcodeBERT 2026,
> DNABERT-S, BIN/ABGD/ASAP). Genuine novelty = open-set DETECT + 3-way integration
> on fish COI + the genus-knowable/species-limited boundary.

## Working Title

Fast tree-aware molecular biodiversity inference with calibrated rank assignment
under missing and ambiguous evidence.

## Central Thesis

Biodiversity inference from short marker sequences should not be forced species
classification. A defensible system should combine candidate retrieval,
tree-aware scoring, reference diagnostics, marker-resolvability limits,
ecological context where justified, and calibrated species/genus/family/order/
no-call output.

For a shorter coauthor-facing storyline that includes the latest MarkerMirror
and active-curation results, see
`experiments/paper1_phylo_calibrated_assignment/PAPER_STORYLINE.md`.

## What Is New

The individual ingredients are not all new:

- BLAST/VSEARCH/k-mer already retrieve similar sequences.
- EPA-ng/pplacer/APPLES already place sequences onto trees.
- Fernando 2025 already tests fish COI placement onto a fish backbone.
- DEPP/H-DEPP already learn sequence-to-tree-distance placements.
- BarcodeBERT/TaxoTagger/LISA already explore fast learned/vector sequence
  retrieval.
- TAXDNA already combines 12S sequence, tree, and ecological context.

The intended contribution is the bridge:

> one benchmarked inference pipeline that asks what taxonomic rank is justified
> under missing references, marker ambiguity, tree evidence, and ecological
> context.

## Paper Structure

### 1. Problem Definition

Molecular biodiversity assignment is an evidence problem:

- the true species may be absent from the reference database;
- the marker may not resolve species;
- similar species may share the same marker signal;
- ecological context may help or bias;
- a forced species label can be less scientific than a calibrated genus/family
  call.

### 2. Pipeline

```text
marker sequence
  -> fast candidate retrieval
  -> optional classical/tree reranking
  -> tree-aware score
  -> reference/missingness diagnostics
  -> marker-resolvability diagnostics
  -> ecological prior when justified
  -> calibrated species/genus/family/order/no-call
```

Two execution modes:

- accuracy-first: classical/placement baselines included directly;
- vector-first: learned embedding index retrieves candidates quickly, then
  reranking/calibration interprets them.

### 3. COI Benchmark

Purpose:

- test tree-space recovery and missing-reference behavior in an information-rich
  marker.

Methods:

- CNN / biLSTM / Transformer / Mamba encoders;
- BLAST / VSEARCH / k-mer;
- EPA-ng / pplacer / APPLES-style placement;
- negative controls;
- vector-first retrieval.

Metrics:

- species/genus/family/order top-k;
- tree-distance recovery;
- nearest-reference distance bins;
- rank-backoff when true species/genus/family is hidden;
- runtime / candidate recall for vector-first retrieval;
- false species-call and no-call behavior.

### 4. 12S/eDNA Resolvability

Purpose:

- show when species-level assignment is theoretically unsupported by the marker.

Experiments:

- exact identity resolvability;
- near-exact identity thresholds;
- species/genus/family/order oracle support;
- marker/window-aware filtering if feasible.

### 5. 12S/eDNA Evidence Decomposition

Purpose:

- test what evidence actually improves eDNA inference.

Arms:

- sequence only;
- tree only;
- ecology/geography/co-occurrence only;
- sequence + tree;
- sequence + ecology;
- sequence + tree + ecology;
- rank/no-call calibration for each arm.

### 6. Rank-Adaptive Assignment

Purpose:

- convert model scores into defensible biodiversity claims.

Outputs:

- species;
- genus;
- family;
- order;
- no-call.

Required:

- independent threshold calibration;
- risk/coverage curves;
- false species-call rate;
- comparison across neural, classical, placement, and ecological arms.

### 7. Reference-Gap And Curation Diagnostics

Purpose:

- identify when new reference sequencing would improve inference.

Outputs:

- nearest unresolved clades;
- candidate groups causing ambiguity;
- missing species/genera/families that most reduce uncertainty;
- model/reference disagreement flags.

## Main Claims If Supported

1. Short marker sequences carry useful tree/taxonomic signal, but exact species
   assignment depends strongly on reference completeness and marker resolution.
2. Classical sequence tools remain extremely strong when close references exist.
3. Learned vector retrieval can be a fast front end, but speed is not enough:
   calibration and rank-aware inference are the scientific contribution.
4. 12S/eDNA species-level assignment is often underdetermined by sequence alone.
5. A unified rank-adaptive pipeline reduces overconfident species claims and
   gives more defensible biodiversity outputs.
6. Cross-marker candidate generation is useful when paired with same-marker
   classical alignment. MarkerMirror alone gives weak high-rank support on the
   full 12S query set, but MarkerMirror + BLASTN reaches 9.5 / 92.1 / 95.3 /
   99.7% top50 support for species/genus/family/order. This supports the
   "candidate set first, calibrated rank second" framing.
7. The active-curation layer turns no-calls into reference/evidence priorities,
   identifying which missing 12S/16S references or target-marker curation fixes
   would most improve future inference.

## Claims To Avoid

- We invented vector barcode search.
- We invented phylogenetic placement of fish COI.
- Mamba is necessarily the best encoder.
- 12S sequence-only models solve species-level eDNA assignment.
- Ecology always improves assignment.
- A top-1 species prediction is automatically a valid biodiversity claim.

## Immediate Remaining Work

1. Improve calibration transfer for the BLAST/VSEARCH-backed union candidate
   lists. Current source agreement is production-safe but conservative; learned
   listwise models improve coverage but do not yet lock target-0.99.
2. Decide which operating points become manuscript claims:
   - COI p-distance rank/no-call,
   - COI DL species-disabled decision layer,
   - MarkerMirror + BLASTN/VSEARCH candidate support,
   - species-disabled eDNA Eco-Phylo posterior.
3. Refresh manuscript assets and figure/table plans from the latest source
   tables.
4. Wire reason codes into the FASTA/CSV CLI output, not only source tables.
5. If field-eDNA claims are central, run final independent eDNA validation or
   keep eDNA as a conservative extension rather than a species-level claim.
6. Use Exp131 active-curation outputs to choose targeted reference additions
   before running another family/genus attempt.
7. Keep the merged source-table ledger current:
   `merged_12s_resolvability_summary.csv`,
   `merged_12s_zero_shot_model_metrics.csv`,
   `merged_global_edna_asv_metrics.csv`, and
   `merged_global_edna_sample_metrics.csv`.
7. Build figure source tables and draft panels.
