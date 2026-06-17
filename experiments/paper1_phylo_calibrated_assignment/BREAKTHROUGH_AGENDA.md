# Breakthrough Agenda

Last updated: 2026-06-03

> **NOVELTY CORRECTION (2026-06-17):** Superseded for novelty claims by
> `NOVELTY_AND_PRIOR_ART.md` (corrected after a verified literature audit). The
> tree-distance embedding (Stalder 2025 / DEPP 2023), the k-mer baseline (kf2vec
> 2025), and neural barcode clustering (BarcodeBERT 2026, DNABERT-S, BIN/ABGD/ASAP)
> are **prior art**. Genuine novelty = open-set DETECT + integration on fish COI +
> the marker-resolution boundary.

## Current Thesis

The breakthrough path is not "a neural model beats BLAST." The field already
has strong sequence search, strong phylogenetic placement, barcode foundation
models, and probabilistic taxonomy. The missing method is an auditable
evidence compiler for biodiversity inference:

```text
multiple candidate generators
  -> one evidence table
  -> calibrated smallest defensible taxonomic node
  -> reason codes
  -> active reference-curation priorities
```

This reframes barcode/eDNA identification as a high-stakes uncertainty problem,
closer to medical triage or sensor fusion than normal closed-set
classification.

## Outside-Field Ideas To Import

### 1. Hierarchical Conformal / Selective Prediction

Sources:

- Angelopoulos and Bates, conformal prediction:
  https://arxiv.org/abs/2107.07511
- Geifman and El-Yaniv, selective classification:
  https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- El-Yaniv and Wiener, risk-coverage foundations:
  https://jmlr.csail.mit.edu/papers/v11/el-yaniv10a.html
- Open-world recognition survey:
  https://ojs.aaai.org/index.php/AAAI/article/view/5054

Adaptation:

- Treat species/genus/family/order/no-call as a hierarchical selective
  prediction problem.
- Calibrate the smallest node that satisfies empirical risk, rather than
  manually choosing confidence thresholds.
- Report risk-coverage curves by rank and by missing-reference stress pack.

Why this could be novel:

- Existing barcode tools usually return a label or a score.
- Existing placement tools return a placement.
- We can return a calibrated taxonomic claim with an explicit "I do not know"
  boundary.

### 2. Weak Supervision / Label-Model Evidence Fusion

Sources:

- Snorkel weak supervision:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7075849/
- Snorkel/data programming:
  https://arxiv.org/abs/1711.10160

Adaptation:

- Treat BLAST, VSEARCH, k-mer, vector retrieval, EPA-ng/APPLES, tree embedding,
  range priors, co-occurrence priors, and reference-gap detectors as noisy
  labeling functions.
- Learn source reliabilities and conflicts instead of hand-weighting every
  evidence stream.
- Emit a final rank/no-call decision plus evidence provenance.

Why this could be novel:

- The pipeline becomes a principled evidence compiler, not just an ensemble.
- It creates a clean way to add future encoders without rewriting the method.

### 3. Dempster-Shafer / Sensor-Fusion Reasoning

Sources:

- Dempster-Shafer in GIS uncertainty:
  https://www.sciencedirect.com/science/article/abs/pii/S0957417405003064
- Dempster-Shafer topic overview:
  https://www.nature.com/nature-index/topics/l4/dempster-shafer-theory-and-uncertainty-modeling

Adaptation:

- Let evidence sources assign mass to species, genus, family, order, unknown,
  or conflict.
- Use conflict itself as a reason code.
- This is especially natural for eDNA where sequence, tree, geography, and
  co-occurrence can disagree.

Why this could be novel:

- It separates lack of evidence from conflicting evidence.
- That distinction matters biologically: missing references are different from
  marker ambiguity and different again from ecological implausibility.

### 4. Active Reference Curation / Value Of Information

Sources:

- Reference library limitations in metabarcoding/barcoding:
  https://pubmed.ncbi.nlm.nih.gov/36478393/
- Species-distribution data fusion:
  https://pubs.usgs.gov/publication/70193265

Adaptation:

- For each no-call or broader-rank call, identify which missing species/genus
  references would most reduce uncertainty.
- Rank taxa for sequencing priority by expected reduction in no-call/conflict.
- Tie the model directly to biodiversity infrastructure improvement.

Why this could be novel:

- Most tools use reference libraries; few tell users how to improve them.
- This converts uncertainty into an actionable sequencing plan.

### 5. Foundation Models As Cross-Marker Bridges

Sources:

- Nucleotide Transformer:
  https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
- DNABERT-S:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10896361/
- BarcodeBERT:
  https://arxiv.org/abs/2311.02401
- BarcodeMamba:
  https://arxiv.org/abs/2412.11084

Adaptation:

- Frozen foundation embeddings can be tested as fast candidate generators.
- Fine-tuning or contrastive adaptation may be useful, but only if it improves
  calibrated evidence decisions, not only retrieval.
- The most interesting direction is now cross-marker translation: learn a
  shared COI<->12S/eDNA coordinate system so a low-resolution 12S fragment can
  be compared against COI-rich species/tree evidence.

Probe results:

- A frozen Nucleotide Transformer v2-50M probe was run on Vast with one
  reference barcode per fish-tree species and 1,200 sampled queries per split.
- Results are in
  `results/paper1_phylo_calibrated_assignment/source_tables/hf_foundation_probe_retrieval_metrics.csv`.
- Frozen embeddings show useful higher-rank signal but are weaker than our
  current COI/classical baselines:
  - held-out fish top-10 genus/family/order: 38.4 / 59.8 / 74.2%;
  - unseen-genera top-10 family/order: 43.5 / 69.7%.
- The first MarkerMirror / BarcodeBridge probes trained small projection heads
  on frozen Nucleotide Transformer embeddings for species with both COI and 12S
  data. The split is species-level: bridge training used 674 overlap species
  and evaluated on 145 held-out overlap species.
- Results are in
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`.
- Held-out cross-marker top-10 retrieval improved from frozen NT to the best
  learned projection with taxonomy-soft rank targets:
  - species: 0.7 -> 9.8%;
  - genus: 3.5 -> 20.7%;
  - family: 24.2 -> 54.5%;
  - order: 43.0 -> 72.5%.
- A first actual tree-distance soft-target run was added after the rank-soft
  runs. It used patristic distances from `actinopt_12k_treePL.tre`, but did not
  beat taxonomy-soft rank targets:
  - tree-soft scale-25 held-out top-10 species/genus/family/order:
    8.4 / 18.3 / 48.3 / 68.1%.
- A retrieval-aligned checkpointing run selected the taxonomy-soft checkpoint
  with best validation genus/family/order top-10. It gives the strongest
  held-out family/order bridge so far:
  - species/genus/family/order: 8.5 / 18.8 / 55.4 / 75.4%.
- A first LoRA adapter run on Nucleotide Transformer query/key/value modules
  was completed after installing PEFT on Vast. It trained and restored the best
  retrieval checkpoint, but held-out retrieval dropped:
  - species/genus/family/order: 4.0 / 7.5 / 28.8 / 50.1%.
- A first 12S->16S ribosomal MarkerMirror run was completed after building a
  bounded NCBI 16S reference. It used 502 overlap species and a held-out
  species split of 351 train / 75 validation / 76 test. Held-out top-10
  improved strongly over frozen NT:
  - frozen NT species/genus/family/order: 18.4 / 23.0 / 34.9 / 49.7%;
  - learned 12S->16S bridge: 33.9 / 45.4 / 67.1 / 73.4%.
- The reverse 16S->12S bridge is stronger still:
  - frozen NT species/genus/family/order: 11.9 / 17.8 / 32.6 / 48.2%;
  - learned 16S->12S bridge: 44.4 / 54.8 / 70.4 / 76.3%.
- The first shared 12S/16S species-space prototype is now the strongest
  MarkerMirror result:
  - shared 12S->16S species/genus/family/order: 42.1 / 50.0 / 68.5 / 81.5%;
  - shared 16S->12S species/genus/family/order: 64.3 / 71.3 / 78.3 / 85.3%.
- Shared 12S/16S seed repeats are stable:
  - 12S->16S mean/range species top-10: 43.4%, range 42.1-45.4%;
  - 16S->12S mean/range species top-10: 66.4%, range 63.3-71.5%;
  - genus/family/order are also stable and remain high.
- A first 12S/16S/COI tri-marker shared-space prototype completed. It improves
  every tested direction over frozen NT, but it is not the new lead bridge:
  held-out 12S->COI top-10 is 2.7 / 12.7 / 34.8 / 58.6% for
  species/genus/family/order, below the best direct 12S->COI run at
  8.5 / 18.8 / 55.4 / 75.4%. The useful lesson is that 16S is a strong
  complementary eDNA marker, while COI remains better as a downstream
  barcode/tree anchor unless the triad objective is redesigned.
- The first integrated MarkerMirror evidence compiler is now a strong
  breakthrough candidate. It uses shared 12S/16S MarkerMirror candidates plus
  same-marker sequence checks, candidate-list ambiguity/support, reference
  availability, and tree-neighborhood evidence. On held-out test at target
  0.99, learned 12S->16S logistic gives 55.0% coverage, 99.4% assigned
  precision, 0.0% false species-call rate, and 100.0% species precision;
  learned 16S->12S HGB gives 75.0% coverage, 99.1% assigned precision, 0.0%
  false species-call rate, and 100.0% species precision.
- This integrated result now has a three-seed stability check. At target 0.99,
  the best learned calibrator per seed averages 51.0% coverage / 98.9%
  assigned precision for 12S->16S and 71.1% coverage / 98.7% assigned
  precision for 16S->12S. False species-call rates remain near zero. This is
  the strongest current "breakthrough" material because it turns cross-marker
  retrieval into calibrated biodiversity evidence rather than a naked top-k
  candidate list.
- The first explicit marker-resolvability hardening pass is complete. Exact
  and 0.99 proxy ambiguity features for 12S/16S are now source-tabled and
  wired into the seed1903 evidence compiler. They did not improve the best
  target-0.99 row, but they make the pipeline more scientifically honest: a
  species call can now be audited against marker ambiguity, not only model
  confidence.
- The first production-handoff script now exists:
  `scripts/edna/run_marker_mirror_candidate_generator.py`. A tiny 12S->16S
  CPU smoke and a full-reference GPU smoke both wrote candidate/summary/manifest
  files. The GPU smoke used 32 held-out 12S queries against the full
  1,865-species 16S reference, wrote 1,600 candidate rows, and recovered the
  known target in top-50 at 25.0 / 59.4 / 78.1 / 84.4% for
  species/genus/family/order. This turns MarkerMirror from table-only evidence
  into an executable candidate-generator layer, but final rank/no-call
  integration remains to be done.
- Cache-backed MarkerMirror repeat inference now works. The cache-write pass
  created a 2,971-row 16S foundation-embedding cache, the cache-read pass loaded
  it, and both passes produced identical candidate tables. This makes repeated
  specimen-style inference feasible without recomputing the full 16S reference.
- The first production-style evidence handoff now works:
  `scripts/edna/build_marker_mirror_candidate_generator_handoff.py` converts
  executable candidate-generator rows into evidence-compiler rows. The smoke
  produced 1,600 evidence rows with 97 production numeric features, same-marker
  sequence evidence, tree-neighborhood distances, and marker-resolvability
  features. The next gap is applying a trained integrated rank/no-call
  calibrator to these rows.
- The first full-query candidate-generator-to-rank/no-call apply path now runs:
  3,566 12S queries, 178,300 top-50 candidate/evidence rows, and locked
  validation thresholds from the offline integrated calibrator. This is an
  important engineering milestone but not a final operating point. Species calls
  failed transfer; species-disabled logistic gives 6.5% coverage at 93.1%
  assigned precision and 0 false species calls.

Interpretation:

- Do not pivot the paper around generic frozen DNA foundation embeddings.
- The cross-marker bridge is a stronger breakthrough prospect than generic
  frozen retrieval because it addresses a real eDNA bottleneck: 12S often lacks
  species resolution while COI has richer barcode/tree evidence.
- The first projection-only bridge is not yet a finished result. Taxonomy-hard
  batches improve held-out species/genus retrieval, taxonomy-soft rank targets
  improve family/order retrieval, and retrieval-aligned checkpointing improves
  family/order again. Best-validation loss restore underperforms, and the first
  actual tree-distance target underperforms taxonomy-soft. The first LoRA run
  also underperforms, so the next backbone-adaptation step should require a
  better objective, frozen-teacher regularization, multi-positive training, or
  more paired data rather than simply running longer.
- The 12S/16S shared-space result is cleaner at species/genus/family than
  12S->COI and stronger than the separate 12S/16S directional heads. COI
  remains important as a barcode/tree anchor and comparator, but the eDNA marker
  bridge does not need CytB/18S yet.
- The new high-value claim is no longer just "candidate retrieval improves."
  It is: cross-marker candidate generation becomes reliable when it is checked
  by sequence/reference/tree evidence before returning a calibrated
  species/genus/family/order/no-call decision.

## Ranked Breakthrough Bets

1. MarkerMirror / BarcodeBridge: cross-marker 12S/eDNA-to-COI/tree-space
   foundation-model bridge.
2. 12S/16S MarkerMirror: add 16S as the second eDNA marker and learn a shared
   ribosomal species/tree space, with COI retained only as an anchor/comparator
   when useful.
3. Hierarchical conformal evidence compiler.
4. Active reference-curation / value-of-information layer.
5. Weak-supervision label model over evidence sources.
6. Evidence-conflict reason codes using sensor-fusion ideas.

## Immediate Work

1. Build a hierarchical conformal prototype from existing COI production
   assignment, reason-code, and gap-warning tables.
   - First prototype complete:
     `results/paper1_phylo_calibrated_assignment/source_tables/hierarchical_selective_rank_summary.csv`.
   - Consensus-only target-0.99 gives held-out fish 92.0% coverage at 98.0%
     precision and unseen-genera 89.9% coverage at 94.1% precision, mostly by
     collapsing to order-level calls.
   - Interpretation: selective prediction is the right formal frame, but the
     scoring function needs richer evidence fusion before it becomes a
     breakthrough operating point.
2. Build an active reference-curation table from no-calls, broader-rank calls,
   reference-gap probabilities, and top-k ambiguity.
3. Normalize candidate generators into a single "labeling function" matrix.
4. Start the next MarkerMirror model:
   - keep the held-out species split;
   - use taxonomy-soft rank targets as the current best objective;
   - select checkpoints by validation retrieval metrics, not only loss;
   - consider a hybrid rank-soft + tree-distance regularizer only if it improves
     validation retrieval;
   - compare projection-only against LoRA/backbone fine-tuning only after the
     checkpointing/objective is stable;
   - current LoRA result is negative, so prioritize objective/data changes over
     longer adapter training.
5. Build the 12S/16S MarkerMirror reference layer:
   - first audit complete:
     `results/paper1_phylo_calibrated_assignment/source_tables/marker_16s_local_source_audit.csv`;
   - NCBI query plan and bounded reference build complete:
     `scripts/edna/build_16s_reference_from_ncbi.py`;
   - first 16S reference root:
     `data/edna/stalder_inputs/16s_multisource/`;
   - current bounded build has 1,865 16S species, 502 overlapping 12S, and 319
     overlapping 12S+COI;
   - first 12S->16S bridge run is complete:
     held-out top-10 species/genus/family/order 33.9 / 45.4 / 67.1 / 73.4%;
   - reverse 16S->12S bridge is complete:
     held-out top-10 species/genus/family/order 44.4 / 54.8 / 70.4 / 76.3%;
   - shared 12S/16S species-space training completed and is strongest:
     12S->16S top-10 42.1 / 50.0 / 68.5 / 81.5 and 16S->12S top-10
     64.3 / 71.3 / 78.3 / 85.3;
   - seed repeats completed and are stable;
  - tri-marker 12S/16S/COI shared-space training completed. It improves over
    frozen NT in every direction, but 12S/16S remains the lead bridge and COI
    remains a downstream barcode/tree anchor;
  - candidate-reranker/rank-no-call integration has now been tested across
    three shared-space seeds;
  - near-exact marker-resolvability prototype, full-reference production-handoff
    testing, cached reference embeddings, evidence-handoff rows, integrated
    calibrator application, and calibration-transfer diagnostics have been
    added;
  - current transfer diagnosis: the full handoff has only 26.6% query-species
    coverage in the 16S target reference, versus 100.0% in the controlled
    validation split. The next method step is reference-aware calibration and
    rank/no-call policy, not more blind candidate generation;
  - independent reference-aware validation now exists. Repeated species splits
    show target-0.99 gates average 4.13% held-out coverage at 98.27% precision
    and meet target in 70% of repeats. This is promising but not a locked
    production threshold;
  - union candidate support now exists. MarkerMirror-only full-query top50 is
    9.5 / 39.9 / 59.8 / 76.3% for species/genus/family/order; union with
    same-marker 12S k-mer candidates is 9.5 / 91.7 / 95.1 / 99.6%;
  - the first production-style union rank/no-call diagnostic is complete. The
    union candidate table has 355,231 rows without hidden labels. Top-1 source
    agreement at family/order only assigns 25.2% of queries at 98.4% precision
    with 0 species calls. Score gates show high-rank signal but are not locked
    production thresholds;
  - the first learned HGB compiler over top-1 union features is complete and is
    a useful negative result. It does not beat the simpler diagnostics:
    order-only target-0.99 averages 67.4% coverage at 98.5% precision and meets
    target in 44% of species-split repeats;
  - reason-code and reference-curation tables now exist for the union pipeline.
    The largest bucket is high-rank union support to genus: 2,249/3,566
    full-query rows. Conservative source agreement emits 621 family calls at
    98.1% precision and 277 order calls at 99.3% precision. The curation table
    separates missing-marker-reference cases from retrieval/model failures such
    as `Trichiurus_lepturus`, where the species exists in 16S but is rarely
    recovered at species level;
  - same-marker edlib validation now exists. It reranks the existing 12S k-mer
    top50 candidate pool with bidirectional edlib HW edit-distance identity.
    Top10 support is 0.0 / 87.8 / 94.3 / 98.8% for species/genus/family/order,
    so the high-rank same-marker signal survives alignment-backed scoring. This
    is still not full all-vs-all BLAST/VSEARCH candidate generation;
  - the first list-level selective compiler now exists. It improves the
    high-coverage order diagnostic over top-1 HGB: order-only target-0.99
    averages 83.1% coverage at 98.8% precision, but target is met in only 56%
    of species-split repeats. This is a promising diagnostic, not the final
    production policy;
  - VSEARCH same-marker candidate generation now exists. VSEARCH global top50
    support is 0.0 / 90.4 / 94.9 / 99.4% for species/genus/family/order, and
    MarkerMirror + VSEARCH union top50 support is 9.5 / 91.8 / 95.1 / 99.6%.
    This replaces the k-mer audit as the stronger claim-facing same-marker
    candidate arm;
  - BLASTN same-marker candidate generation now exists. BLASTN local top50
    support is 0.0 / 90.7 / 95.1 / 99.4%, and MarkerMirror + BLASTN union
    top50 support is 9.5 / 92.1 / 95.3 / 99.7%. This confirms the high-rank
    same-marker result under local alignment too;
  - the first BLAST/VSEARCH-backed calibration-transfer repair now exists.
    All-source top1 order agreement is stable at target-0.99: 24.8% coverage,
    99.6% precision, target met in 100% of 50 repeats. Higher-coverage order
    policies are close but not locked: about 56.1-69.0% coverage at about
    99.4-99.5% mean precision, target met in 82-94% of repeats;
  - Exp 112 has converted the stable all-source order diagnostic into a
    production-style assignment/reason-code table. The conservative max-repeat
    threshold assigns 24.7% of full-query 12S rows at 99.7% precision with 0
    false species calls;
  - the stable policy also now emits a label-stripped production assignment
    payload. The remaining engineering gap is orchestration from arbitrary 12S
    FASTA into MarkerMirror, BLASTN, VSEARCH, and the shared feature table;
  - Exp 114 has started closing that gap: the one-command 12S wrapper exists,
    dry-run smoke writes a normalized query table/dependency report/run plan,
    and the BLASTN stage smoke succeeds locally. Local full execution is blocked
    only by missing VSEARCH;
  - Exp 115 completed the full wrapper on Vast for all 3,566 current 12S query
    rows. The chain now runs end to end: MarkerMirror -> BLASTN -> VSEARCH ->
    feature table -> stable order/no-call. Output: 880 order calls, 2,686
    no-calls, 99.7% diagnostic precision, 0 false species calls;
  - Exp 116 completed an unlabeled FASTA smoke of the same wrapper on Vast. Two
    taxonomy-stripped FASTA records produced 1 order call and 1 no-call, with
    precision/correctness fields intentionally blank. This verifies that the
    CLI can emit production-style assignments and reason codes without hidden
    labels; it is not an accuracy estimate;
  - a concise usage note now exists in
    `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md`;
  - Exp 117 found a stable high-coverage order repair diagnostic:
    BLASTN/VSEARCH top-10 order agreement with nested global Wilson95 locking
    gives 57.2% mean held-out coverage, 99.8% mean precision, target-0.99 met
    in 100% of 50 outer repeats, and a full-table locked diagnostic of
    2,513/3,566 order calls at 99.8% labelled precision;
  - Exp 118 exposes the Exp 117 policy as explicit
    `--decision-mode high_coverage_order` in the 12S wrapper. Vast smokes passed
    for labelled and unlabeled inputs in both stable and high-coverage modes;
  - Exp 119 tested family/genus repair. No family/genus policy met target-0.99
    in all 50 species-split repeats, so those ranks stay disabled in the
    wrapper;
  - Exp 120 created the coauthor-facing MarkerMirror one-pager:
    `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_COAUTHOR_ONE_PAGER.md`;
  - Exp 121 tested set-valued family/genus output and still found no stable
    target-0.99 family/genus result, so the next family/genus attempt needs new
    evidence rather than another threshold wrapper;
  - Exp 122 created a MarkerMirror manuscript-facing package under
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/`
    with candidate-support, order-policy, rank-boundary, runtime, figure-plan,
    and methods-blurb files;
  - Exp 123 rendered MarkerMirror figure drafts under
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/`
    with candidate-support, order-policy, rank-boundary, runtime, slide-ready
    summary, and manifest files;
  - Exp 124 created slide-ready tables and a five-slide outline under
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/`;
  - Exp 125 created manuscript captions, results/methods paragraphs, claim
    boundary text, and a caption inventory under
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`;
  - Exp 126 created a manuscript section outline, section checklist, and
    manifest in the same text directory, placing MarkerMirror as a focused
    12S/eDNA extension section after the core COI and missing-reference
    results;
  - Exp 127 created the MarkerMirror next-evidence audit and lineage coverage
    table. The P0 paths for any future family/genus attempt are
    lineage-specific reference coverage, alignment-backed marker-resolvability,
    and active reference-curation/value-of-information;
  - Exp 128 tested lineage/reference-coverage features directly in a nested
    species-split policy diagnostic. This did not unlock stable family/genus:
    target-0.99 genus averaged 17.7% coverage at 97.6% precision and met
    target in 42% of repeats, while target-0.99 family averaged 87.4% coverage
    at 98.0% precision and met target in 10% of repeats. Keep family/genus
    disabled until genuinely new evidence is added;
  - Exp 129 completed the VSEARCH-backed marker-resolvability replacement for
    the rare-kmer proxy. At 0.99 identity, 12S query oracle support is
    77.9 / 95.2 / 99.6 / 99.7% for species/genus/family/order, but only 19.6%
    of query clusters contain a current reference. This is a better
    marker-ceiling diagnostic, not an enabled family/genus policy;
  - Exp 130 tested production-available VSEARCH cluster features as a policy
    input. This still did not unlock family/genus: target-0.99 family averaged
    57.3% coverage at 95.5% precision and met target in 44% of repeats, while
    target-0.99 genus averaged 11.5% coverage at 87.8% precision and met target
    in 38% of repeats;
  - Exp 131 built the active reference-curation/value-of-information layer:
    `marker_mirror_active_reference_value_species.csv`,
    `marker_mirror_active_reference_value_lineage.csv`, and
    `marker_mirror_active_reference_value_actions.csv`. The largest action is
    adding both 12S and 16S species references for 532 species groups covering
    1,928 queries. This is a curation-priority artifact, not an enabled
    family/genus policy;
  - next breakthrough work should either turn the slide/text package into a
    polished deck/full manuscript draft after user review or identify a
    genuinely new evidence source for family/genus beyond the Exp 127 P0 list;
  - if the near-exact rows become claim-facing, replace the proxy backend with
    VSEARCH/edlib;
   - CytB, 18S, and non-fish markers are out of scope for this paper unless
     12S/16S coverage proves insufficient.

## Claim Boundary

Breakthrough material means:

> The system does not merely identify barcodes. It explains the deepest
> biodiversity claim supported by incomplete molecular evidence and tells users
> what reference data would most improve that claim.

This is stronger and more defensible than architecture novelty.
