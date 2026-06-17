# Coauthor Brief: Paper 1

> **NOVELTY CORRECTION (2026-06-17):** Novelty claims in this doc are superseded by
> `NOVELTY_AND_PRIOR_ART.md` (corrected after a verified literature audit). Key
> concessions: the tree-distance embedding is **prior art** (Stalder 2025 — same
> fish tree; DEPP 2023); the k-mer-vs-learned baseline is a **replication** (kf2vec
> 2025); unsupervised clustering / "neural species delimitation" **already exists**
> (BarcodeBERT 2026, DNABERT-S, BIN/ABGD/ASAP/GMYC). Genuine novelty = the
> **open-set DETECT axis**, the **3-way integration on fish COI**, and the
> **quantified genus-knowable / species-limited boundary**. Do not claim the
> embedding, the baseline, or clustering as novel.

> **SCOPE (Paper 1):** Paper 1 is the **COI calibrated rank-adaptive pipeline** —
> the full draft is in [MANUSCRIPT.md](MANUSCRIPT.md). Everything central to the
> coauthor review for this submission is that COI pipeline. The 12S cross-marker
> bridging (MarkerMirror) and the learned Eco-Phylo posterior described later in
> this brief are **Paper 2** material; in Paper 1 they appear only as the
> marker-ceiling boundary (why species-level 12S/eDNA is intentionally not
> claimed). Read MANUSCRIPT.md and [COAUTHOR_PLAIN_LANGUAGE.md](COAUTHOR_PLAIN_LANGUAGE.md)
> first; the sections below are the fuller results dossier, Paper 1 then Paper 2.

Working title:

> Calibrated rank-adaptive biodiversity inference from DNA barcodes under missing
> references.

## One-Sentence Story

Short barcode and eDNA evidence should not be treated as forced species
classification. We are building and evaluating a fast candidate-retrieval and
rank-adaptive inference pipeline that reports the deepest defensible taxonomic
rank: species, genus, family, order, or no-call.

## What Is New Here

The novelty is not vector search alone and not "deep learning beats BLAST".
Those are not defensible claims.

The stronger contribution is the integrated pipeline:

1. fast vector-first candidate retrieval;
2. classical sequence/placement comparators retained as strong baselines;
3. tree-aware learned embeddings and/or tree-distance diagnostics;
4. conservative p-distance reranking over retrieved candidates;
5. calibrated rank/no-call assignment;
6. strict missing-reference stress tests where species/genus/family references
   are removed before model training and candidate construction;
7. for 12S/eDNA, an Eco-Phylo posterior that fuses sequence, tree-neighborhood,
   geography/co-occurrence, and marker-resolvability evidence.

The scientific point is that biodiversity inference is an evidence problem
under missing references and marker ambiguity, not just a classifier benchmark.

## Current Headline Results

For the shortest MarkerMirror-specific handoff, use
`experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_COAUTHOR_ONE_PAGER.md`
and the manuscript package under
`results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/`.
Draft figures are under
`results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/`.
Slide-ready tables and the five-slide outline are under
`results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/`.
Manuscript captions, results/methods paragraphs, and claim-boundary text are
under
`results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`.
The same text directory now also contains
`marker_mirror_manuscript_section_outline.md` and
`marker_mirror_manuscript_section_checklist.csv`, which propose where the
MarkerMirror section belongs in Paper 1 and which assets to use.
For the next family/genus evidence path, see
`marker_mirror_family_genus_next_evidence_plan.md` in that same directory.
For the paper-level narrative that integrates COI, MarkerMirror, rank/no-call,
and active curation, use
`experiments/paper1_phylo_calibrated_assignment/PAPER_STORYLINE.md`.

### Strongest New Result: MarkerMirror + BLASTN Candidate Support

For the 12S/16S MarkerMirror track, the best result is no longer just "a neural
cross-marker model finds related candidates." We now have a union candidate
generator:

```text
12S query
  -> MarkerMirror cross-marker 12S->16S candidates
  -> same-marker 12S BLASTN candidates
  -> union candidate list
```

On 3,566 full-query 12S sequences:

- MarkerMirror alone top50: 9.5 / 39.9 / 59.8 / 76.3% for
  species/genus/family/order.
- Same-marker BLASTN top50: 0.0 / 90.7 / 95.1 / 99.4%.
- MarkerMirror + BLASTN union top50: 9.5 / 92.1 / 95.3 / 99.7%.

Interpretation: species remains low because the held-out query species are
absent from the current same-marker 12S reference table by design. But the
pipeline very often recovers the correct genus, family, or order. This is a
strong candidate-generation result for rank-aware biodiversity inference:
when species evidence is missing, the system can still assemble a defensible
high-rank candidate set instead of forcing an unsupported species label.

VSEARCH independently confirms the same pattern: MarkerMirror + VSEARCH union
top50 is 9.5 / 91.8 / 95.1 / 99.6%.

The current order/no-call layer is also improving:

- conservative MarkerMirror + BLASTN + VSEARCH top-1 order agreement assigns
  880/3,566 queries, 24.7% coverage, at 99.7% diagnostic precision;
- the new nested high-coverage diagnostic uses BLASTN/VSEARCH top-10 order
  agreement and reaches 57.2% held-out coverage at 99.8% precision, meeting the
  99% target in all 50 species-split repeats;
- locked on the full labelled table, that high-coverage diagnostic assigns
  2,513/3,566 queries, 70.5% coverage, at 99.8% precision.

This is still order-level inference, not species identification. The value is
that the pipeline can make many high-confidence high-rank calls while refusing
unsupported species calls.

The wrapper now exposes this as two modes:

- conservative default: fewer calls, strongest caution;
- high-coverage order mode: more order calls, still explicitly order/no-call
  only and still research/diagnostic rather than field-deployed production.

### Fast Candidate Retrieval

Controlled CNN seed1206 Eval C vector timing on the Vast RTX host:

- exact vector search: 0.397 ms/query;
- HNSW m16/ef50: 0.00475 ms/query;
- HNSW m32/ef50: 0.00513 ms/query.

Interpretation: the vector-first layer is fast enough to be a practical
candidate generator. These timings do not include downstream reranking or
calibration.

### Rank/No-Call Pipeline

CNN seed1206 with target-0.99 seen-test-derived p-distance rerank calibration:

- Eval C: 95.8% coverage, 93.0% assigned precision, 0.0% false species calls.
- Unseen-genera: 92.3% coverage, 83.9% assigned precision, 0.0% false species
  calls.

Interpretation: at a conservative operating point, the system backs off from
species calls rather than hallucinating species under missing-reference
conditions.

### First DL Decision Layer

We trained the first small neural evidence model above the current pipeline. It
does not replace BLAST, vector retrieval, or p-distance; it learns when their
combined evidence supports genus/family/order/no-call.

Species-disabled target-0.99 result:

- held-out fish: 94.2% coverage, 97.4% assigned precision, 0.0% false species
  calls. Bootstrap 95% intervals: coverage 93.8-94.7%, precision 97.1-97.7%.
- unseen-genera: 88.5% coverage, 93.5% assigned precision, 0.0% false species
  calls. Bootstrap 95% intervals: coverage 87.8-89.2%, precision 93.0-94.0%.

This improves precision over the current hand-threshold production-v1 policy,
with lower coverage. It is a promising decision-layer result, not yet the final
production default. It is now integrated as an optional FASTA/CSV CLI mode:
`--decision-mode dl_mlp_species_disabled`.

Seed repeats are stable across three MLP seeds:

- held-out fish: 94.2-96.0% coverage, 97.1-97.4% assigned precision, 0.0%
  false species calls.
- unseen-genera: 88.5-91.3% coverage, 92.9-93.5% assigned precision, 0.0%
  false species calls.

### Production CLI Status

The COI production-v1 path now runs from specimen-style FASTA or CSV input:

```text
FASTA/CSV sequence
  -> CNN barcode-to-tree embedding
  -> vector candidate retrieval
  -> top-k p-distance reranking
  -> calibrated species/genus/family/order/no-call
```

Smoke tests passed on the Vast RTX PRO 6000:

- CSV with 16 known-label rows: 100.0% coverage, 87.5% precision if known,
  0 species calls.
- FASTA with 8 unlabeled rows: 100.0% coverage, precision unavailable by
  design, 0 species calls.
- DL decision mode, CSV with 16 known-label rows: 100.0% coverage, 100.0%
  precision if known, 0 species calls.
- DL decision mode, FASTA with 8 unlabeled rows: 100.0% coverage, precision
  unavailable by design, 0 species calls.

This is still a research CLI, not a deployed API, but the core command-line
inference path now exists.

### Strict Missing-Reference Validation

All six strict pruned CNN runs completed. These runs remove references before
training and candidate construction.

- Eval C hide species: species top10 0.0; genus/family/order top10
  41.8 / 62.9 / 83.9.
- Eval C hide genus: species/genus top10 0.0; family/order top10 56.3 / 75.1.
- Eval C hide family: species/genus/family top10 0.0; order top10 40.9.
- Unseen-genera hide species: species top10 0.0; family/order top10
  53.3 / 82.9; genus is essentially unsupported.
- Unseen-genera hide genus: species/genus top10 0.0; family/order top10
  47.4 / 80.7.
- Unseen-genera hide family: species/genus/family top10 0.0; order top10 51.0.

Interpretation: when a rank is absent from the evidence, the model cannot and
should not force that rank. Broader ranks can remain recoverable.

### Classical Comparators

BLAST/k-mer/VSEARCH remain strong when close references exist. This is why the
paper should not be framed as neural methods replacing classical methods.

Current end-to-end ledger highlights:

- BLAST family top10: Eval C 98.1%, unseen-genera 83.0%.
- APPLES-like local distance placement nearest-reference match:
  Eval C 54.4%, unseen-genera 22.1%.
- Fernando-style completeness sweeps are now complete for both EPA-ng and
  official APPLES 2.0.11: 30 sweeps covering random and family-stratified
  99/80/60/40/20% backbones with 3 replicates each.
- In that completed sweep matrix, APPLES is stronger than EPA-ng in our
  diagnostics:
  - placed-clade genus/family/order: APPLES 32.8 / 57.2 / 65.6% versus EPA-ng
    17.3 / 45.2 / 57.0%;
  - sister-clade any-overlap/exact: APPLES 42.5 / 21.4% versus EPA-ng
    14.8 / 3.2%.

Interpretation: we are now close enough to say we ran a Fernando-style
classical placement protocol on our public setup. We should not say exact
Fernando reproduction, because the reference set, backbone construction, and
PCP implementation are not identical.

### 12S/eDNA Eco-Phylo Posterior

The full candidate-level posterior now includes direct 12S sequence evidence
and inference-safe candidate tree-neighborhood evidence over 6,995,880
candidate rows.

Species is intentionally disabled because species-level thresholds do not
transfer. The current safe policy is genus -> family -> order -> no-call.

- target-90: 51.7% held-out assignment at 84.3% accuracy.
- target-95: 40.3% held-out assignment at 94.3% accuracy.
- target-95 composition: 4,852 genus calls, 7,094 family calls, 864 order calls.
- Nested threshold-stability over 30 calibration resplits: target-95 averages
  40.2% held-out assignment at 94.3% accuracy.
- True nested posterior fit, where the model is fit on 70% of calibration
  groups and thresholds are learned on the remaining calibration groups:
  species still fails; family/order transfer well individually; mixed
  species-disabled target-95 assigns 38.9% of held-out queries at 93.4%
  accuracy.

Interpretation: this supports conservative higher-rank eDNA inference, not
species-level eDNA assignment.

## What We Should Claim

Claim:

> A barcode/eDNA assignment system should combine fast candidate retrieval,
> tree-aware evidence, classical sequence checks, and calibrated rank/no-call
> decisions so that missing or ambiguous molecular evidence produces the
> deepest defensible taxonomic claim rather than a forced species label.

Supported subclaims:

- vector indexing can make learned candidate retrieval very fast;
- tree-aware neural encoders preserve useful higher-rank signal;
- conservative p-distance reranking can eliminate false species calls in the
  current target-0.99 operating point by backing off to broader ranks;
- strict missing-reference tests show hidden ranks collapse as expected while
  broader ranks remain partly recoverable;
- 12S/eDNA should be evaluated as marker ambiguity plus evidence integration,
  not sequence-only species classification;
- a species-disabled Eco-Phylo posterior can make high-confidence higher-rank
  eDNA calls while avoiding unsupported species calls.
- the current MarkerMirror 12S wrapper now has two order/no-call modes:
  conservative `stable_order` and explicit diagnostic `high_coverage_order`.
  Family/genus repair has been tested but is not stable enough to enable.
- set-valued family/genus output was also tested and still does not reach the
  99% target without very large, low-utility candidate sets.
- lineage/reference-coverage features were tested as the next new evidence
  source. They did not stabilize family/genus transfer, so the next legitimate
  family/genus attempt needs alignment-backed marker-resolvability or
  sample-aware context rather than another threshold wrapper.
- VSEARCH-backed marker-resolvability is now source-tabled. At 99% identity,
  12S query oracle support is 77.9 / 95.2 / 99.6 / 99.7% for
  species/genus/family/order, but this is marker-ceiling evidence rather than
  an enabled family/genus policy.
- A learned policy using production-available VSEARCH cluster features was also
  tested and still did not stabilize family/genus transfer.
- The new active-curation layer ranks what to do after an abstention. The
  largest action category is adding both 12S and 16S species references for
  532 species groups covering 1,928 queries. The clearest model/target-curation
  failure is `Trichiurus_lepturus`, where the 16S target is present but
  MarkerMirror retrieval is weak.

## What We Should Not Claim

Do not claim:

- deep learning replaces BLAST;
- species-level assignment is solved;
- vector search for barcodes is novel by itself;
- local APPLES-like distance output on the clean splits is official APPLES;
- current Fernando-style sweep diagnostics are exact Fernando PCP;
- Mamba is the best architecture.

## Key Files To Review

Start here:

- `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_COAUTHOR_ONE_PAGER.md`
- `experiments/paper1_phylo_calibrated_assignment/README.md`
- `experiments/paper1_phylo_calibrated_assignment/PIPELINE.md`
- `experiments/paper1_phylo_calibrated_assignment/SOURCE_TABLES.md`
- `experiments/paper1_phylo_calibrated_assignment/CLAIM_BOUNDARIES.md`
- `experiments/paper1_phylo_calibrated_assignment/MANUSCRIPT_ASSETS.md`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_actions.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_species.csv`

Key source tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_end_to_end_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_coi_method_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_vector_index_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/controlled_vector_speed_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/strict_missing_reference_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/strict_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/pipeline_calibration/pipeline_mode_policy_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_summary.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/claim_evidence_map.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/figure_plan.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/missing_results_checklist.csv`

## Remaining Decisions

1. Decide whether the completed Fernando-style sweep diagnostics are sufficient
   comparator context, or whether exact Fernando PCP implementation is required.
2. Decide whether the local APPLES-like p-distance diagnostic is still useful as
   a lightweight same-split proxy, or whether the manuscript should show only
   official APPLES from the completeness sweeps.
3. Choose the final manuscript operating point:
   - current strongest conservative point is CNN seed1206 p-distance rerank,
     target 0.99.
4. Decide how central the 12S/eDNA extension is in the first manuscript:
   - main result if we want a unified biodiversity-evidence paper;
   - supplementary/second result if we want Paper 1 to stay COI-focused.
5. Decide whether to improve eDNA posterior calibration so mixed
   species-disabled target-95 reaches the target on held-out groups, or report
   the current 93.4% held-out operating point honestly.

## Suggested Coauthor Questions

- Is the main thesis clear and different enough from Fernando/DEPP/TAXDNA?
- Are we comfortable making rank/no-call the core contribution rather than
  species top-1 accuracy?
- Are the official APPLES/EPA-ng matched Fernando-style completeness sweep
  outputs sufficient, or do we need an even closer reproduction of Fernando's
  exact PCP implementation before submission?
- Should 12S/eDNA be part of the main paper or treated as an extension?
- Which operating point is most useful for biodiversity-monitoring users?
