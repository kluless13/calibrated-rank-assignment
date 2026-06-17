# Current Results: Paper 1

Last updated: 2026-06-03

> **NOVELTY CORRECTION (2026-06-17):** For novelty claims see
> `NOVELTY_AND_PRIOR_ART.md` (corrected after a verified literature audit).
> Concede priority on the embedding (Stalder 2025 / DEPP 2023), the k-mer baseline
> (kf2vec 2025), and neural clustering (BarcodeBERT 2026, DNABERT-S, BIN/ABGD/ASAP).
> Genuine novelty = open-set DETECT + 3-way integration on fish COI + the
> genus-knowable/species-limited boundary. The results numbers below remain valid.

This is the central "start here" result ledger for the merged Paper 1 work.
Use it for orientation before opening the detailed logs/source tables.

## One-Line Thesis

Short barcode and eDNA evidence should not be forced into species labels. The
pipeline should retrieve candidates quickly, check them with classical and
tree-aware evidence, then return the deepest defensible rank:

```text
species / genus / family / order / no-call
```

## Current Pipeline

```text
marker sequence
  -> fast vector or classical candidate retrieval
  -> BLAST/p-distance/placement checks
  -> tree-space and reference-gap evidence
  -> marker-resolvability evidence
  -> ecological/geographic/co-occurrence evidence for eDNA
  -> calibrated rank/no-call decision
```

## Headline Status

| Area | Status | Meaning |
|---|---|---|
| COI tree/retrieval | Strong | Tree-aware barcode inference and missing-reference rank backoff are the most mature results. |
| Vector retrieval | Strong speed evidence | HNSW/exact vector search is fast enough to be a practical candidate generator. |
| Retrieval-DL encoder sweep | Complete diagnostic | CNN contrastive/hybrid encoders are useful fast candidate generators; hierarchical losses improve tree geometry but hurt retrieval ordering. |
| Frozen DNA foundation probe | Diagnostic, not strong enough alone | Nucleotide Transformer v2-50M has higher-rank signal but is weaker than current COI/classical baselines as a frozen candidate generator. |
| Classical comparators | Strong and necessary | BLAST/k-mer/VSEARCH/APPLES remain powerful baselines; we should not claim neural replacement. |
| Missing-reference validation | Strong | Strict pruned tests show unsupported ranks collapse and broader ranks remain partly recoverable. |
| DL evidence/rank-backoff | Precision-first positive result | MLP calibrators over vector+p-distance and reference-gap evidence improve conservative precision when species is disabled, but coverage drops. |
| Candidate-level DL reranker | Positive diagnostic evidence | BLAST/VSEARCH-aware top-k neural scoring strongly improves genus/family ordering, but calibration is not stable enough for production. |
| Reference-gap detector | Improved diagnostic layer | Candidate-level evidence substantially improves hidden species/genus/family gap detection, but supported-case false warning rates are still too high for default production use. |
| Reason-code overlay | New explanatory layer | Production-v1 assignments now have source-table reason codes joining rank/no-call decisions with reference-gap and ambiguity warnings. |
| Hierarchical selective prototype | Useful frame, not final | Consensus-only selective calibration formalizes risk/coverage but collapses to broad order calls at high targets. |
| MarkerMirror cross-marker bridge | Promising breakthrough prospect | Learned projection heads turn frozen foundation embeddings into real cross-marker signal. The 12S->16S bridge is now the cleanest species/genus result; full-reference candidate-generation handoff now runs on GPU. |
| 12S/eDNA resolvability | Useful | 12S has a real species-resolution ceiling; species-level forced calls are often unjustified. |
| eDNA evidence decomposition | Available | Sequence-only, ecology-only, sequence+ecology, and learned co-occurrence arms are now separated. |
| Eco-Phylo posterior | Full sequence+tree candidate run | Candidate-level scoring now includes direct 12S sequence evidence and inference-safe candidate tree-neighborhood evidence. |
| eDNA rank/no-call | Conservative higher-rank policy | Species is disabled; genus/family/order backoff gives the current honest eDNA operating point. |
| Production v1 | Research CLI works for COI FASTA/CSV | Locked p-distance-rerank thresholds and an optional DL decision layer now write final assignment, summary, and manifest files from saved embeddings, split sequences, or specimen-style FASTA/CSV input. |

## COI / Fish Barcode Results

### Vector Speed

Controlled CNN seed1206 Eval C vector timing on the Vast RTX host:

- exact vector search: 0.397 ms/query;
- HNSW m16/ef50: 0.00475 ms/query;
- HNSW m32/ef50: 0.00513 ms/query.

Interpretation: vector search is the fast first-pass candidate layer. BLAST is
the strong classical comparator/check, not the fast part of the final tool.

### Retrieval-DL Encoder Sweep

Four retrieval-first encoders were trained on the RTX PRO 6000 and copied into
the local archive:

- source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_dl_sweep_summary.csv`,
  `retrieval_dl_sweep_tree_recovery.csv`, and
  `retrieval_dl_sweep_training_history.csv`;
- copied result root:
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_retrieval_dl_sweep/`.

Best candidate-generator read:

| Arm | Held-out top50 species/genus/family/order | Unseen-genera top50 species/genus/family/order | Tree recovery held-out Pearson/Spearman | Tree recovery unseen-genera Pearson/Spearman |
|---|---:|---:|---:|---:|
| CNN contrastive | 61.2 / 96.2 / 97.4 / 98.2% | 34.1 / 62.9 / 86.1 / 90.5% | 0.561 / 0.489 | 0.548 / 0.487 |
| CNN hybrid | 60.8 / 95.6 / 97.4 / 98.0% | 34.0 / 60.3 / 86.4 / 91.6% | 0.573 / 0.507 | 0.548 / 0.490 |
| CNN hierarchical contrastive | 53.6 / 84.8 / 95.4 / 97.1% | 22.1 / 54.3 / 85.1 / 89.8% | 0.585 / 0.587 | 0.575 / 0.569 |
| Transformer hierarchical contrastive | 25.5 / 54.0 / 65.9 / 72.4% | 10.9 / 27.7 / 50.1 / 59.6% | 0.630 / 0.575 | 0.591 / 0.539 |

Interpretation: CNN contrastive/hybrid are the useful retrieval-front-end
candidates. Hierarchical/tree-shaped losses are informative because they improve
tree-recovery correlations, but they currently sacrifice too much candidate
recall. Transformer is not competitive as a first-pass retriever in this sweep.
This is a model-development result, not a replacement for BLAST/VSEARCH or the
current production rank/no-call policy.

### Frozen DNA Foundation Probe

I ran a frozen HuggingFace Nucleotide Transformer v2-50M probe on Vast to test
whether a pretrained DNA foundation model is immediately useful as another
candidate generator:

- model: `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`;
- setup: one representative reference barcode per fish-tree species;
- queries: 1,200 sampled queries per split;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/hf_foundation_probe_retrieval_metrics.csv`.

Top-10 result:

- seen-test species/genus/family/order: 25.1 / 37.6 / 57.3 / 73.3%;
- held-out fish species/genus/family/order: 0.0 / 38.4 / 59.8 / 74.2%;
- unseen-genera family/order: 43.5 / 69.7%.

Interpretation: frozen foundation embeddings contain useful higher-rank signal,
but they are not competitive enough to become the core method. If used, they
should be another evidence stream or should be fine-tuned against our
tree/rank objective.

### MarkerMirror / BarcodeBridge Probe

This is the first explicit attempt at a more novel model-development lane:
learn a cross-marker bridge between COI barcodes and 12S/eDNA fragments.

- script: `scripts/edna/train_marker_mirror_bridge.py`;
- model backbone: frozen
  `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`;
- training: small learned projection heads over species with both COI and 12S
  data;
- species-level split: 674 train, 144 validation, 145 held-out test overlap
  species;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`;
- copied result root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_multisource/`.

Held-out cross-marker top-10 result, 12S query to COI species prototypes:

| Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| Frozen NT cross-marker | 0.7% | 3.5% | 24.2% | 43.0% |
| MarkerMirror projection, random negatives | 7.8% | 16.9% | 49.0% | 65.9% |
| MarkerMirror projection, taxonomy-hard negatives | 10.3% | 20.4% | 50.7% | 70.4% |
| MarkerMirror projection, taxonomy-soft rank targets | 9.8% | 20.7% | 54.5% | 72.5% |
| MarkerMirror projection, taxonomy-soft best-val restore | 4.4% | 7.1% | 26.7% | 51.1% |
| MarkerMirror projection, tree-distance soft targets scale 25 | 8.4% | 18.3% | 48.3% | 68.1% |
| MarkerMirror projection, taxonomy-soft retrieval-best checkpoint | 8.5% | 18.8% | 55.4% | 75.4% |
| MarkerMirror LoRA, taxonomy-soft retrieval-best checkpoint | 4.0% | 7.5% | 28.8% | 50.1% |
| MarkerMirror projection, taxonomy-soft multi-positive retrieval-best checkpoint | 9.9% | 17.3% | 47.0% | 69.3% |

Seen/training-split diagnostic, same top-10 metric:

| Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| Frozen NT cross-marker | 0.9% | 2.4% | 16.4% | 34.2% |
| MarkerMirror projection, random negatives | 97.1% | 97.2% | 98.2% | 98.8% |
| MarkerMirror projection, taxonomy-hard negatives | 99.8% | 99.8% | 99.9% | 99.9% |
| MarkerMirror projection, taxonomy-soft rank targets | 99.1% | 99.2% | 100.0% | 100.0% |
| MarkerMirror projection, taxonomy-soft retrieval-best checkpoint | 99.3% | 99.4% | 99.9% | 99.9% |
| MarkerMirror projection, tree-distance soft targets scale 25 | 94.3% | 94.9% | 97.1% | 98.1% |
| MarkerMirror LoRA, taxonomy-soft retrieval-best checkpoint | 19.3% | 24.1% | 39.7% | 53.3% |
| MarkerMirror projection, taxonomy-soft multi-positive retrieval-best checkpoint | 79.0% | 80.6% | 89.1% | 93.7% |

Interpretation of seen rows: the projection-head bridge has enough capacity to
align trained species almost perfectly. The hard part is not fitting seen
species; it is transferring that marker bridge to held-out species. Use the
held-out table for claims and the seen table for diagnosing capacity/overfit.

Interpretation: this is not a finished classifier, but it is the clearest new
prospect so far. Frozen foundation embeddings alone do not align 12S to COI
well. A simple cross-marker objective creates measurable held-out signal. The
taxonomy-hard run improves species/genus, while taxonomy-soft rank targets
improve family/order. Retrieval-aligned checkpointing gives the best held-out
family/order result so far. The best-validation-loss checkpoint underperforms,
so contrastive loss alone is not the right selection criterion. The first
actual tree-distance soft-target run does not improve the bridge. The next
experiment tested a small LoRA adapter on query/key/value attention projections.
It trained cleanly and restored the best retrieval checkpoint, but held-out
retrieval fell below the projection-only bridge. Treat this as a negative
result: projection-head alignment is the current best MarkerMirror evidence,
and heavier foundation-model adaptation needs a better objective or more paired
data before it is worth scaling. The first multi-positive projection run is a
mixed objective result: it improves held-out species top-10 versus
taxonomy-soft retrieval-best, but loses the family/order strength that makes
the current bridge most useful.

First 12S-to-16S bridge result, same source table:

| Marker Pair / Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| 12S query to 16S, frozen NT cross-marker | 18.4% | 23.0% | 34.9% | 49.7% |
| 12S query to 16S, MarkerMirror taxonomy-soft retrieval-best | 33.9% | 45.4% | 67.1% | 73.4% |
| 16S query to 12S, frozen NT cross-marker | 11.9% | 17.8% | 32.6% | 48.2% |
| 16S query to 12S, MarkerMirror taxonomy-soft retrieval-best | 44.4% | 54.8% | 70.4% | 76.3% |

Interpretation: the 12S/16S bridge is now the strongest MarkerMirror evidence,
especially in the reverse 16S->12S direction. It is much stronger than the
current 12S->COI bridge at species/genus/family and slightly stronger at order.
This makes 12S/16S the right eDNA marker expansion to prioritize. COI should
stay in the paper as a barcode/tree anchor and comparator, not as the only
bridge target.

Shared 12S/16S species-space result:

- run:
  `nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_shared_retrieval_metrics.csv`;
- split: same 351 train, 75 validation, 76 held-out test species;
- checkpoint selection: combined validation genus/family/order top-10 across
  both directions;
- best checkpoint: epoch 60, validation score 81.7221.

Held-out shared-space top-10:

| Direction / Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| 12S -> 16S, frozen NT | 15.2% | 19.2% | 31.1% | 51.3% |
| 12S -> 16S, shared MarkerMirror | 42.1% | 50.0% | 68.5% | 81.5% |
| 16S -> 12S, frozen NT | 7.0% | 14.7% | 30.1% | 50.4% |
| 16S -> 12S, shared MarkerMirror | 64.3% | 71.3% | 78.3% | 85.3% |

Interpretation: the shared 12S/16S space is now the strongest MarkerMirror
result. It improves over the separate directional bridges in both directions.
This makes the next model-development question seed stability and rank/no-call
integration, not whether the bridge signal exists.

Shared 12S/16S seed-repeat stability:

- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_shared_retrieval_metrics.csv`;
- rows/runs: 432 rows across 3 shared-space runs;
- held-out 12S->16S top-10 mean species/genus/family/order:
  43.4 / 50.7 / 68.1 / 77.6%;
- held-out 12S->16S range:
  species 42.1-45.4, genus 49.0-53.0, family 65.6-70.2, order 72.9-81.5%;
- held-out 16S->12S top-10 mean species/genus/family/order:
  66.4 / 73.9 / 81.9 / 86.4%;
- held-out 16S->12S range:
  species 63.3-71.5, genus 71.3-78.5, family 78.3-86.8, order 84.2-89.6%.

Interpretation: the 12S/16S shared-space signal is stable enough to justify
downstream candidate-reranker/rank-no-call integration. Seed repeats do not
remove the claim boundary: this is still candidate retrieval, not final species
assignment.

Tri-marker 12S/16S/COI shared-space result:

- run:
  `nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_triad_retrieval_metrics.csv`;
- pair overlaps total/train/validation/test:
  - 12S/16S: 502 / 364 / 66 / 72;
  - 12S/COI: 963 / 669 / 164 / 130;
  - 16S/COI: 607 / 424 / 95 / 88;
- checkpoint selection: combined validation genus/family/order top-10 across
  all marker directions;
- best checkpoint: epoch 110, validation score 47.9459.

Held-out tri-marker top-10:

| Direction / Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| 12S -> 16S, frozen NT | 11.9% | 19.6% | 34.3% | 42.7% |
| 12S -> 16S, triad MarkerMirror | 31.1% | 47.2% | 65.0% | 73.1% |
| 16S -> 12S, frozen NT | 11.1% | 14.1% | 29.6% | 43.7% |
| 16S -> 12S, triad MarkerMirror | 51.9% | 65.9% | 77.8% | 89.6% |
| 12S -> COI, frozen NT | 0.0% | 1.8% | 12.7% | 38.1% |
| 12S -> COI, triad MarkerMirror | 2.7% | 12.7% | 34.8% | 58.6% |
| 16S -> COI, frozen NT | 0.0% | 4.0% | 18.7% | 46.0% |
| 16S -> COI, triad MarkerMirror | 5.3% | 20.7% | 48.7% | 64.7% |

Interpretation: the triad improves strongly over frozen embeddings in every
direction, so it is real cross-marker evidence. It does not yet beat the best
direct 12S->COI MarkerMirror bridge, whose held-out top-10 is
8.5 / 18.8 / 55.4 / 75.4% for species/genus/family/order. The current
boundary is therefore: 12S/16S shared space is the lead MarkerMirror result;
COI is still best treated as a downstream barcode/tree anchor and comparator
unless a later triad objective improves 12S->COI transfer.

MarkerMirror full-reference candidate export:

- script:
  `scripts/edna/export_marker_mirror_candidate_rankings.py`;
- run:
  shared 12S/16S seed1903;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_rankings_shared_seed1903_summary.csv`;
- full candidate table:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/marker_mirror_candidate_rankings.csv.gz`;
- candidate rows: 294,700.

This table ranks against the full target marker library, not only the 502
overlap species used by the aggregate training metric. It is therefore harder
and more realistic. Held-out learned top-k against the full target library:

| Direction | k | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|---:|
| 12S -> 16S | 1 | 16.6% | 27.8% | 37.1% | 42.1% |
| 12S -> 16S | 10 | 35.1% | 44.4% | 59.6% | 68.5% |
| 12S -> 16S | 50 | 52.0% | 60.6% | 80.1% | 87.8% |
| 16S -> 12S | 1 | 27.1% | 41.0% | 59.0% | 66.0% |
| 16S -> 12S | 10 | 54.2% | 65.3% | 79.2% | 86.8% |
| 16S -> 12S | 50 | 74.3% | 80.6% | 90.3% | 93.8% |

Interpretation: this is the first MarkerMirror artifact that can feed the
pipeline's candidate-generation stage. It is still not a final assignment
policy.

Executable MarkerMirror candidate-generator handoff:

- script:
  `scripts/edna/run_marker_mirror_candidate_generator.py`;
- source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_fullref_smoke_summary.csv`
  and
  `marker_mirror_candidate_generator_fullref_smoke_manifest_summary.csv`;
- cache source table:
  `marker_mirror_candidate_generator_cache_smoke_summary.csv`;
- archived output:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`;
- local output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`.

This run used 32 held-out 12S queries, the full 1,865-species 16S reference, and
top-50 candidates. It wrote 1,600 candidate rows. Known-target top-50 recovery
was 25.0 / 59.4 / 78.1 / 84.4% for species/genus/family/order. Interpretation:
the specimen-style candidate-generator path works on GPU and can produce a
realistic candidate list, but final assignment still requires same-marker
sequence checks, tree/reference/resolvability evidence, and calibrated
rank/no-call.

Cache-backed repeat inference now works: the cache-write pass wrote a 2,971-row
16S foundation-embedding cache, the cache-read pass loaded it, and the
candidate tables were exactly identical.

Candidate-generator evidence handoff:

- script:
  `scripts/edna/build_marker_mirror_candidate_generator_handoff.py`;
- source tables:
  `marker_mirror_candidate_generator_evidence_handoff_summary.csv`,
  `marker_mirror_candidate_generator_evidence_handoff_manifest_summary.csv`,
  and `marker_mirror_candidate_generator_evidence_handoff_feature_inventory.csv`;
- local output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_cache_read_12s_to_16s/evidence_handoff/`.

The handoff produced 1,600 evidence rows for 32 queries, with tree evidence
enabled, 3,502 marker-resolvability rows available, and 97 production numeric
features. Top-1 candidates had same-marker 12S reference evidence available for
40.6% of queries. This is now rank/no-call-ready input, but it is not yet a
final assignment because the trained integrated calibrator is not serialized and
applied to these specimen-style rows yet.

Full-query production-style handoff:

- source tables:
  `marker_mirror_candidate_generator_handoff_summary.csv`,
  `marker_mirror_candidate_generator_evidence_handoff_summary.csv`, and
  `marker_mirror_candidate_generator_rank_apply_summary.csv`;
- run:
  all 3,566 12S query rows, cached full 1,865-species 16S reference, top-50;
- candidate/evidence rows:
  178,300.

Known-target top-50 recovery on the full query table:

| Species | Genus | Family | Order |
|---:|---:|---:|---:|
| 9.5% | 39.9% | 59.8% | 76.3% |

First rank/no-call apply result:

| Policy | Coverage | Assigned Precision | Species Calls | False Species Call Rate |
|---|---:|---:|---:|---:|
| logistic target-0.99, species enabled | 7.0% | 85.5% | 20 | 0.56% |
| logistic target-0.99, species disabled | 6.5% | 93.1% | 0 | 0.00% |
| HGB target-0.99, species disabled | 4.7% | 71.1% | 0 | 0.00% |

Interpretation: the end-to-end apply path now exists, but calibration does not
transfer at the nominal 0.99 target. Species calls should remain disabled for
MarkerMirror production-style assignment. The current strongest use remains
candidate/evidence generation, with rank/no-call as a diagnostic layer rather
than a manuscript-ready production claim.

Calibration-transfer diagnosis:

- source tables:
  `marker_mirror_calibration_transfer_cohort_summary.csv`,
  `marker_mirror_calibration_transfer_handoff_strata.csv`,
  `marker_mirror_calibration_transfer_feature_drift.csv`, and
  `marker_mirror_calibration_transfer_top_feature_drift.csv`;
- script:
  `scripts/edna/build_marker_mirror_calibration_transfer_diagnostics.py`.

The transfer gap is now measured. In the controlled validation split, every
query species is present in the 16S target reference and top-50 recovery is
47.6 / 60.8 / 74.3 / 81.1% for species/genus/family/order. In the full
production-style handoff, only 26.6% of query species have a 16S target
reference, and top-50 recovery drops to 9.5 / 39.9 / 59.8 / 76.3%. When the
true species is present in the 16S reference, top-50 species recovery is 35.8%;
when it is absent, species recovery is necessarily 0.0% but genus/family/order
still carry useful signal. This supports the pipeline framing: MarkerMirror is
a cross-marker candidate/evidence layer, and the final assignment policy must
be reference-aware and willing to stop at genus, family, order, or no-call.

Reference-aware policy repair diagnostic:

- source tables:
  `marker_mirror_reference_aware_policy_summary.csv` and
  `marker_mirror_reference_aware_policy_manifest.json`;
- script:
  `scripts/edna/build_marker_mirror_reference_aware_policy.py`.

Starting from the species-disabled logistic apply row, a production-safe gate on
the MarkerMirror top-1 score improves the full-query assignment tradeoff:

| Policy | Coverage | Assigned Precision | False Species Call Rate |
|---|---:|---:|---:|
| baseline species-disabled logistic | 6.48% | 93.07% | 0.00% |
| top1 score >= 0.620484 | 5.83% | 95.67% | 0.00% |
| top1 score >= 0.697663 | 3.25% | 100.00% | 0.00% |

This is still a diagnostic sweep fitted on the labelled handoff, not an
independent production calibration. It does show that evidence-aware abstention
can recover precision without changing the candidate generator.

Independent reference-aware validation:

- source tables:
  `marker_mirror_reference_aware_policy_validation_summary.csv`,
  `marker_mirror_reference_aware_policy_validation_per_split.csv`, and
  `marker_mirror_reference_aware_policy_validation_manifest.json`;
- script:
  `scripts/edna/build_marker_mirror_reference_aware_policy_validation.py`.

Across 50 repeated query-species splits, thresholds are chosen on calibration
species and evaluated on held-out species. The target-0.95 policy averages
5.79% held-out coverage at 94.39% assigned precision; only 48% of repeats meet
the 95% target. The stricter target-0.99 policy averages 4.13% held-out coverage
at 98.27% assigned precision; 70% of repeats meet the 99% target, with 5th-95th
percentile precision 90.97-100.00%. Source-holdout checks are also mixed:
MitoHelper and rCRUX are reasonable, but Mare-MAGE is small and unstable at the
0.95 target. Conclusion: the reference-aware gate is promising, but not yet a
locked production threshold.

MarkerMirror-only rank/no-call diagnostic:

- script:
  `scripts/edna/build_marker_mirror_candidate_rank_policy.py`;
- source directory:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/rank_policy/`.

Validation-fitted top-1 score thresholds did not transfer cleanly to held-out
test. At target 0.95, learned 16S->12S assigned 36.1% of test queries at 88.5%
assigned precision, and learned 12S->16S assigned only 5.6% at 76.5%.
Conclusion: MarkerMirror is useful as candidate generation evidence, but rank
calibration must use richer features: top-k consensus, margins, sequence
similarity, tree/rank evidence, and reference-gap diagnostics.

MarkerMirror feature-based rank calibrator:

- script:
  `scripts/edna/train_marker_mirror_rank_calibrator.py`;
- source directory:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/rank_calibrator/`.

The feature-based calibrator uses top-k consensus, score margins, and
taxonomic ambiguity features. It improves the diagnostic frame but still does
not produce the final policy. At target 0.95 on held-out test, learned
12S->16S reaches 100.0% assigned precision but only 1.3% coverage; learned
16S->12S reaches 20.8% coverage but only 76.7% assigned precision. This
supports the next design decision: MarkerMirror should feed an integrated
evidence compiler, not act alone.

MarkerMirror integrated evidence compiler prototype:

- evidence join script:
  `scripts/edna/build_marker_mirror_evidence_join.py`;
- integrated calibrator script:
  `scripts/edna/train_marker_mirror_integrated_rank_calibrator.py`;
- source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_evidence_join_shared_seed1903_summary.csv`,
  `marker_mirror_integrated_rank_logistic_shared_seed1903_summary.csv`,
  `marker_mirror_integrated_rank_hgb_shared_seed1903_summary.csv`, and
  `marker_mirror_integrated_rank_best_shared_seed1903_summary.csv`.

This is the first working version of:

```text
MarkerMirror candidates
  -> same-marker sequence/reference check
  -> candidate-list ambiguity and support
  -> tree-neighborhood evidence
  -> candidate-level rank probabilities
  -> calibrated species/genus/family/order/no-call
```

Held-out best learned MarkerMirror integrated rows:

| Direction | Target | Calibrator | Coverage | Assigned Precision | False Species Calls | Species Calls | Species Precision |
|---|---:|---|---:|---:|---:|---:|---:|
| 12S -> 16S | 0.90 | HGB | 60.3% | 91.8% | 1.3% | 161 | 97.5% |
| 12S -> 16S | 0.95 | HGB | 56.0% | 95.9% | 1.3% | 161 | 97.5% |
| 12S -> 16S | 0.99 | logistic | 55.0% | 99.4% | 0.0% | 157 | 100.0% |
| 16S -> 12S | 0.90 | HGB | 84.7% | 87.7% | 4.2% | 113 | 94.7% |
| 16S -> 12S | 0.95 | HGB | 79.2% | 93.9% | 3.5% | 112 | 95.5% |
| 16S -> 12S | 0.99 | HGB | 75.0% | 99.1% | 0.0% | 106 | 100.0% |

Interpretation: this is the first strong proof that the pipeline design works
for MarkerMirror. The win is not "MarkerMirror score alone"; it is
MarkerMirror candidate generation plus same-marker reference evidence,
candidate ambiguity, and tree-neighborhood features. Claim boundary: this is a
held-out species diagnostic on the current 12S/16S reference, not yet an eDNA
field-validation result or a seed-repeated production policy.

### 12S/16S MarkerMirror Expansion

Scope decision: the eDNA marker expansion is now limited to 12S and 16S. COI
remains an existing barcode/tree anchor and comparator, but CytB, 18S, and
non-fish markers are out of scope for the current manuscript.

Current 16S reference status:

- script: `scripts/edna/build_16s_reference_from_ncbi.py`;
- output root: `data/edna/stalder_inputs/16s_multisource/`;
- NCBI query returned 40,034 candidate Actinopterygii mitochondrial 16S/rrnL
  records;
- bounded 5,000-record fetch produced 4,673 usable records and 1,865 species;
- overlap with existing 12S: 502 species;
- overlap with existing COI: 607 species;
- three-way 12S/16S/COI overlap: 319 species;
- taxonomy enrichment filled family/order for 1,133 of 1,865 species.

First `12S -> 16S` and reverse `16S -> 12S` bridge status:

- run:
  `nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best`;
- copied root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best/`;
- split: 351 train, 75 validation, 76 held-out test species;
- held-out top-10 species/genus/family/order:
  33.9 / 45.4 / 67.1 / 73.4%;
- frozen NT baseline on the same split:
  18.4 / 23.0 / 34.9 / 49.7%.
- reverse `16S -> 12S` held-out top-10 species/genus/family/order:
  44.4 / 54.8 / 70.4 / 76.3%;
- reverse frozen NT baseline:
  11.9 / 17.8 / 32.6 / 48.2%.

Interpretation: the learned ribosomal bridge improves strongly over frozen
foundation embeddings and gives better species/genus/family transfer than the
12S->COI bridge. The shared 12S/16S species-space prototype is stronger than
the directional heads. Next step is seed repeat/curation check and integration
with candidate reranking plus rank/no-call calibration, not more marker sprawl.

### Rank/No-Call Pipeline

CNN seed1206 with target-0.99 seen-test-derived p-distance rerank calibration:

- Eval C: 95.8% coverage, 93.0% assigned precision, 0.0% false species calls.
- Unseen-genera: 92.3% coverage, 83.9% assigned precision, 0.0% false species calls.

Interpretation: the conservative operating point backs off from species calls
instead of hallucinating species when evidence is missing.

Production v1 now packages this operating point:

- script: `scripts/edna/run_paper1_production_v1.py`;
- raw-sequence wrapper:
  `scripts/edna/run_paper1_raw_sequence_production_v1.py`;
- FASTA/CSV inference CLI:
  `scripts/edna/run_paper1_fasta_inference_v1.py`;
- optional DL decision adapter:
  `scripts/edna/apply_paper1_coi_evidence_model.py`;
- wrapper: `experiments/paper1_phylo_calibrated_assignment/runs/12_run_production_v1.sh`;
- output summary:
  `results/paper1_phylo_calibrated_assignment/production_v1/production_v1_summary_all.csv`;
- raw sequence timing summary:
  `results/paper1_phylo_calibrated_assignment/source_tables/raw_sequence_production_v1_summary.csv`;
- CLI doc:
  `experiments/paper1_phylo_calibrated_assignment/PRODUCTION_CLI_V1.md`;
- caveat: FASTA/CSV research CLI works, but this is not a deployed API/web
  product yet.

Raw split-sequence timing on the Vast RTX PRO 6000:

- Eval C: 46.77 seconds for 11,594 queries, 4.03 ms/query.
- seen-test: 49.80 seconds for 15,763 queries, 3.16 ms/query.
- unseen-genera: 47.12 seconds for 9,148 queries, 5.15 ms/query.

These timings include embedding export, exact vector retrieval, top-k
p-distance reranking, and locked rank/no-call packaging.

FASTA/CSV CLI smoke tests on the Vast RTX PRO 6000:

- CSV with 16 known-label rows: 100.0% coverage, 87.5% precision if known,
  0 species calls, 13.29 seconds total.
- FASTA with 8 unlabeled rows: 100.0% coverage, precision unavailable by
  design, 0 species calls, 13.07 seconds total.
- DL decision mode, CSV with 16 known-label rows: 100.0% coverage, 100.0%
  precision if known, 0 species calls, 14.08 seconds total.
- DL decision mode, FASTA with 8 unlabeled rows: 100.0% coverage, precision
  unavailable by design, 0 species calls, 17.16 seconds total.

These smokes show the interface works. They are not headline benchmarks because
small-batch timing is dominated by Python/Torch startup.

### Strict Missing-Reference Validation

All six strict pruned CNN runs completed:

- Eval C hide species: species top10 0.0; genus/family/order top10 41.8 / 62.9 / 83.9.
- Eval C hide genus: species/genus top10 0.0; family/order top10 56.3 / 75.1.
- Eval C hide family: species/genus/family top10 0.0; order top10 40.9.
- Unseen-genera hide species: species top10 0.0; family/order top10 53.3 / 82.9; genus essentially unsupported.
- Unseen-genera hide genus: species/genus top10 0.0; family/order top10 47.4 / 80.7.
- Unseen-genera hide family: species/genus/family top10 0.0; order top10 51.0.

Interpretation: this supports the core rank-backoff logic.

### DL Evidence / Rank-Backoff Model

First trainable decision-layer model:

- script: `scripts/edna/train_paper1_coi_evidence_model.py`;
- roadmap: `experiments/paper1_phylo_calibrated_assignment/DL_MODEL_ROADMAP.md`;
- output root:
  `results/paper1_phylo_calibrated_assignment/dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/`.
- inference adapter:
  `scripts/edna/apply_paper1_coi_evidence_model.py`;
- optional CLI mode:
  `--decision-mode dl_mlp_species_disabled`.

The model is a small MLP trained on seen-test vector+p-distance evidence. It
predicts whether each rank is supported, calibrates thresholds on held-out
seen-test rows, and evaluates on held-out fish and unseen-genera.

Current conservative species-disabled target-0.99 result:

- held-out fish: 94.2% coverage, 97.4% assigned precision, 0.0% false species
  calls. Bootstrap 95% intervals: coverage 93.8-94.7%, precision 97.1-97.7%.
- unseen-genera: 88.5% coverage, 93.5% assigned precision, 0.0% false species
  calls. Bootstrap 95% intervals: coverage 87.8-89.2%, precision 93.0-94.0%.

Seed-repeat stability across MLP seeds 1206/1207/1208:

- held-out fish: coverage 94.2-96.0%, assigned precision 97.1-97.4%,
  0.0% false species-call rate.
- unseen-genera: coverage 88.5-91.3%, assigned precision 92.9-93.5%,
  0.0% false species-call rate.
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/dl_evidence_seed_summary.csv`.

Strict hidden-reference stress test for the same species-disabled DL layer:

- held-out fish hidden species/genus/family: coverage 58.0 / 57.9 / 16.1%;
  assigned precision 88.5 / 82.5 / 81.5%; species calls 0 / 0 / 0.
- unseen-genera hidden species/genus/family: coverage 50.6 / 52.9 / 41.9%;
  assigned precision 91.3 / 89.5 / 72.7%; species calls 0 / 0 / 0.
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/dl_evidence_strict_apply_summary.csv`.

Compared with hand-threshold production-v1:

- held-out fish: 95.8% coverage, 93.0% assigned precision, 0.0% false species
  calls.
- unseen-genera: 92.3% coverage, 83.9% assigned precision, 0.0% false species
  calls.

Interpretation: the first DL layer gives a real precision gain at lower
coverage. Species remains disabled for the conservative missing-reference
claim because species-enabled mode makes a small number of false species calls.
The DL layer is now integrated as an optional production CLI decision mode, but
the hand-threshold production-v1 policy remains the simpler default operating
point. The strict hidden-taxonomy tests are complete and show the DL layer is
conservative, but not sufficient by itself for hidden-family/reference-gap
diagnosis.

Missing-reference-aware calibrator:

- script: `scripts/edna/train_paper1_missing_reference_aware_calibrator.py`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/missing_reference_aware_calibrator_summary.csv`;
- output root:
  `results/paper1_phylo_calibrated_assignment/dl_evidence_rank_backoff/coi_mlp_seed1401_missing_reference_aware_v2_gap/`.

This version trains on normal supported seen-test rows plus strict held-out
hidden species/genus/family rows, using production-v1 evidence plus v2
reference-gap probabilities. It avoids split labels, roles, and candidate-count
leakage as features.

Best conservative species-disabled target-0.99 result:

- held-out fish: 91.0% coverage, 98.3% assigned precision, 0 species calls;
- unseen-genera: 79.7% coverage, 95.7% assigned precision, 0 species calls.

Strict unseen-genera stress at target 0.99:

- hidden species: 37.3% coverage, 94.7% assigned precision;
- hidden genus: 37.9% coverage, 94.8% assigned precision;
- hidden family: 28.6% coverage, 78.7% assigned precision.

Interpretation: this is a real precision-first DL improvement and the first
rank/no-call model trained with missing-reference positives. It is not yet the
default policy because it sacrifices coverage and hidden-family precision is
still not manuscript-grade. Production-v1 remains the default; this calibrator
is a strong optional precision/reasoning layer.

### Candidate-Level DL Reranker

Candidate-level COI top-50 rerankers:

- script: `scripts/edna/train_paper1_candidate_reranker.py`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/candidate_reranker_summary.csv`;
- output roots:
  `results/paper1_phylo_calibrated_assignment/candidate_reranker/coi_mlp_seed1206_top50/`,
  `coi_cnn_retrieval_contrastive_seed1301_top50_blast_vsearch/`, and
  `coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch/`.

Top-1 candidate ordering versus current p-distance order:

- held-out fish genus: 49.7% vs 38.4%, +11.3 percentage points.
- held-out fish family: 87.3% vs 84.1%, +3.2 percentage points.
- held-out fish order: 95.5% vs 95.2%, +0.3 percentage points.
- unseen-genera family: 71.3% vs 69.1%, +2.2 percentage points.
- unseen-genera order: 87.7% vs 88.3%, -0.6 percentage points.

Target-0.99 calibration transfer:

- held-out fish genus/family/order: assigned precision 98.2 / 98.1 / 97.8%.
- unseen-genera family/order: assigned precision 96.1 / 93.6%.

BLAST/VSEARCH-aware second-stage reranker:

- held-out fish genus top-1: 80.7-83.1% versus 48.2-49.1% p-distance order.
- held-out fish family top-1: 94.3-95.1% versus 87.6-88.0% p-distance order.
- unseen-genera family top-1: 70.5-72.0% versus 68.9-69.4% p-distance order.
- unseen-genera order top-1: 85.5-86.1% versus 84.9-85.2% p-distance order.

Tree-neighborhood second-stage reranker:

- held-out fish genus top-1: 83.7-84.0% versus 48.2-49.1% p-distance order.
- held-out fish family top-1: 94.7-95.4% versus 87.6-88.0% p-distance order.
- unseen-genera family top-1: 71.6-72.5% versus 68.9-69.4% p-distance order.
- unseen-genera order top-1: 85.3-85.8% versus 84.9-85.2% p-distance order.

Focused listwise tree-neighborhood reranker:

- held-out fish genus top-1: 69.4% versus 84.0% for pointwise tree10.
- held-out fish family top-1: 88.0% versus 95.4% for pointwise tree10.
- unseen-genera family top-1: 65.0% versus 72.5% for pointwise tree10.
- unseen-genera order top-1: 83.2% versus 85.8% for pointwise tree10.

Focused pairwise tree-neighborhood reranker:

- held-out fish genus top-1: 84.1% versus 84.0% for pointwise tree10.
- held-out fish family top-1: 95.3% versus 95.4% for pointwise tree10.
- unseen-genera family top-1: 72.4% versus 72.5% for pointwise tree10.
- unseen-genera order top-1: 86.0% versus 85.8% for pointwise tree10.

Cross-split calibration transfer for the strongest pointwise hybrid tree10
reranker, using seen-test-derived target-0.99 thresholds:

- held-out fish genus: 93.2% precision at 74.5% coverage.
- held-out fish family: 96.5% precision at 98.7% coverage.
- held-out fish order: 97.0% precision at 100.0% coverage.
- unseen-genera family: 77.0% precision at 93.5% coverage.
- unseen-genera order: 85.9% precision at 99.8% coverage.

Independent selected-candidate assignment calibrators:

- pointwise-selected calibrator:
  held-out fish genus/family/order precision 93.7 / 96.4 / 97.0%,
  unseen-genera family/order precision 75.9 / 85.8%.
- pairwise-selected calibrator:
  held-out fish genus/family/order precision 94.4 / 95.4 / 96.9%,
  unseen-genera family/order precision 72.6 / 86.1%.

Interpretation: candidate-level learning is now a real evidence-fusion result:
vector retrieval, p-distance, taxonomic consensus, and BLAST/VSEARCH evidence
can substantially improve genus/family candidate ordering. Inference-safe
tree-neighborhood evidence improves this further. It is not yet a production
rank/no-call layer because target-0.99 calibration transfer is still below the
conservative production standard. The first listwise reranker was a useful
negative result: direct query-list optimization did not beat the pointwise
tree10 model. Pairwise training was mixed, not a clean replacement.
The calibration-transfer audit and independent calibrator both confirm the same
boundary: seen-test thresholds are not safe enough for unseen-genera
family/order calls. The next version needs a more realistic missing-reference
calibration design before any DL reranker replaces the production operating
point.

### Reference-Gap Detector

Reference-gap detectors ask whether the candidate reference set supports a
species/genus/family claim, or whether the tool should warn that the reference
library is probably missing the right taxon at that rank.

Current source table:

- `results/paper1_phylo_calibrated_assignment/source_tables/reference_gap_detector_summary.csv`.

The first honest no-counts detector avoided global candidate-set size features
because those can encode synthetic strict-pack identity. It showed that hidden
species cases are detectable, but hidden genus/family recall was weak.

Candidate-evidence v2 adds inference-safe candidate-list evidence:

- script: `scripts/edna/train_paper1_reference_gap_detector_v2.py`;
- tree-aware target-0.95 run:
  `results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_target095/`;
- tree-aware target-0.99 run:
  `results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_target099/`;
- no-tree ablation:
  `results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_notree_target095/`.

Strict unseen-genera gap recall at target 0.95:

| Detector | Hidden species species gap | Hidden genus genus gap | Hide-family genus gap | Hide-family family gap |
|---|---:|---:|---:|---:|
| no-counts v1 | 92.8% | 5.7% | 23.5% | 7.0% |
| v2 no-tree | 95.9% | 30.8% | 38.8% | 27.2% |
| v2 tree-aware | 95.4% | 30.0% | 41.3% | 32.6% |

The v2 tree-aware target-0.99 setting is more conservative:

- hidden species species-gap recall: 79.7%;
- hidden genus genus-gap recall: 10.1%;
- hide-family genus-gap recall: 19.0%;
- hide-family family-gap recall: 18.9%.

Normal supported false warning rates are the limiting factor. At target 0.95,
the tree-aware v2 detector flags supported unseen-genera species/genus/family
as gaps at 34.3 / 10.7 / 7.0%, and supported held-out fish at
15.3 / 3.4 / 2.4%. Target 0.99 lowers the supported-case warning rates but
also loses strict-gap recall.

Interpretation: candidate-level evidence is a real improvement for explaining
missing-reference stress, especially genus/family gaps. It is not yet the final
production reason layer because warnings on supported normal cases are still
too frequent. The next step is to use these scores as soft diagnostic evidence
inside a missing-reference-aware rank/no-call policy, not as a standalone
decision rule.

Production-v1 gap-warning overlay:

- script: `scripts/edna/build_paper1_gap_warning_overlay.py`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/gap_warning_overlay_summary.csv`;
- examples:
  `results/paper1_phylo_calibrated_assignment/source_tables/gap_warning_overlay_examples.csv`.

This overlay does not train a model. It joins v2 reference-gap probabilities to
the existing production-v1 COI assignments. Hard rank-specific warning
abstention barely changes production precision: with tree-aware target 0.95,
assigned-rank warnings cover 0.04% of held-out fish assignments and 0.34% of
unseen-genera assignments, moving precision by only +0.02 and +0.12 percentage
points. The more useful signal is explanatory: more-specific gap warnings
explain 12.6% of held-out broader-rank assignments and 29.1% of unseen-genera
broader-rank assignments, and catch 25.0% of wrong held-out assignments and
41.3% of wrong unseen-genera assignments under the any-warning view.

Interpretation: v2 gap scores are better as reason labels and active-curation
signals than as a hard abstention rule over the already conservative
production-v1 assignments.

Production-v1 reason-code overlay:

- script: `scripts/edna/build_paper1_reason_code_overlay.py`;
- source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/production_reason_code_summary.csv`,
  `production_reason_code_examples.csv`, and
  `production_reason_code_assignments.csv`.

This is the first evidence-accounting layer over the production COI outputs. It
adds labels such as `species_not_supported_at_operating_point`,
`possible_missing_species_reference`, `genus_ambiguous_in_top10`, and
`no_call_no_rank_met`. The reason-code summary shows no-call rows are strongly
enriched for reference-gap warnings: species-gap warnings appear in 78.4% of
held-out fish no-calls and 96.7% of unseen-genera no-calls.

Interpretation: this is not a new accuracy metric. It is a tool-facing
explanation layer, and it moves the pipeline closer to a distinct contribution:
not just assigning a rank, but explaining why a deeper rank was not defensible.

### Hierarchical Selective Rank Prototype

Inspired by conformal prediction/selective classification, I built a compact
prototype that calibrates per-rank score thresholds on seen-test and then
chooses the deepest supported species/genus/family/order/no-call decision:

- script: `scripts/edna/build_hierarchical_selective_rank_prototype.py`;
- tables:
  `hierarchical_selective_rank_summary.csv`,
  `hierarchical_selective_rank_thresholds.csv`,
  `hierarchical_selective_rank_examples.csv`.

Best high-target read from this simple consensus-score prototype:

- held-out fish target-0.99: 92.0% coverage at 98.0% precision;
- unseen-genera target-0.99: 89.9% coverage at 94.1% precision;
- both mostly collapse to order-level calls.

Interpretation: this validates selective prediction as a formal frame, but it
is not a new winning operating point. The next version needs candidate-level
evidence fusion: alignment/p-distance, vector score, placement confidence,
reference-gap probability, and tree-neighborhood features in one calibrated
scorer.

### Fernando-Style Placement Comparators

Completed 30 matched completeness sweeps for EPA-ng and official APPLES 2.0.11
on our public setup.

Final diagnostic averages:

- APPLES placed-clade genus/family/order: 32.8 / 57.2 / 65.6%.
- EPA-ng placed-clade genus/family/order: 17.3 / 45.2 / 57.0%.
- APPLES sister-clade any-overlap/exact: 42.5 / 21.4%.
- EPA-ng sister-clade any-overlap/exact: 14.8 / 3.2%.

Claim boundary: this is Fernando-style matched completeness testing, not exact
Fernando PCP reproduction.

## 12S / eDNA Results

### 12S Resolvability

Near-exact 99% identity query oracle support:

- multisource species/genus/family/order: 77.9 / 95.2 / 99.6 / 99.7.
- multisource Teleo: 70.7 / 90.9 / 97.3 / 100.0.
- rCRUX cleaned: 95.4 / 100.0 / 100.0 / 100.0.
- Mitohelper full Teleo: 70.8 / 93.4 / 97.5 / 99.6.

Near-exact 95% identity query oracle support:

- multisource species/genus/family/order: 38.3 / 73.8 / 92.0 / 98.2.
- multisource Teleo: 42.7 / 71.2 / 89.4 / 99.6.
- rCRUX cleaned: 54.8 / 98.2 / 100.0 / 100.0.
- Mitohelper full Teleo: 33.2 / 69.0 / 87.8 / 99.5.

Interpretation: species-level 12S assignment has an information ceiling. A
forced species label is often not scientifically justified.

### Broad Multisource 12S Model Results

Top-10 species/genus/family/order:

- CNN seed1206: 0.17 / 18.62 / 36.37 / 53.20.
- CNN seed1207: 0.20 / 15.70 / 31.74 / 46.72.
- CNN seed1208: 0.14 / 10.71 / 23.16 / 35.89.
- SSM contrastive: 0.31 / 25.55 / 45.79 / 60.18.

Interpretation: SSM is currently stronger than CNN on broad 12S higher-rank
retrieval, but species-level retrieval remains extremely hard.

### Global_eDNA Evidence Decomposition

Current Global_eDNA forced top-k picture:

- BLAST sequence-only remains strongest for forced species/family/order top-k.
- SSM sequence/tree-only carries useful higher-rank signal but does not beat BLAST on forced top-k.
- Geography and co-occurrence can help selected ranks/arms but can also degrade results.
- Pure geography-only and pure co-occurrence-only are now explicit controls.

Best current Global_eDNA integrated/context rows:

- SSM + RLS/OBIS geography: genus top10 31.8%.
- BLAST + RLS geography: family top10 50.5%.
- SSM + same-sample co-occurrence: order top10 55.2%.

### eDNA Rank/No-Call Calibration

Site-heldout threshold transfer now exists.

Best current modest operating point:

- SSM + RLS/OBIS learned co-occurrence, weight 0.25:
  - assigns 8.9% of held-out rows;
  - family accuracy 59.6%;
  - order accuracy 74.1%.

No current method reaches a 70%+ calibration target for species/genus/family/order
under the top-1 score-threshold policy.

Interpretation: the current eDNA rank/no-call layer is not yet a positive
high-accuracy claim. It tells us the next method layer must be a stronger
Eco-Phylo posterior.

### Eco-Phylo Posterior First Pass

Created a local posterior-prep artifact and a first held-out posterior scorer
from existing copied Global_eDNA outputs only:

- 18 existing SSM/CNN sequence-only and learned co-occurrence methods;
- 1,049,382 query-method rows in a compressed posterior feature table;
- method-design rows for sequence score, tree/candidate evidence, geography/
  co-occurrence arm labels, marker-resolvability ceilings, and calibration
  availability;
- rank-correctness summaries over the deterministic `site20`
  calibration/evaluation split;
- rank-specific posterior models for species/genus/family/order;
- thresholded rank/no-call evaluation on held-out `site20` groups.

First-pass held-out result:

- species: no reliable threshold, correctly backing away from species calls;
- genus: tiny calibration-site thresholds do not transfer;
- family: at target-60 calibration, held-out assignment is 6.9% with 48.6%
  family accuracy;
- order: at target-60 calibration, held-out assignment is 10.1% with 60.4%
  order accuracy.

This is a useful negative/diagnostic result, not a headline win. The posterior
model learns signal row-wise, but its thresholds do not transfer better than the
best current single-method operating point:

- SSM + RLS/OBIS learned co-occurrence, weight 0.25:
  - assigns 8.9% of held-out rows;
  - family accuracy 59.6%;
  - order accuracy 74.1%.

Interpretation: the next posterior needs candidate-level evidence, not just
top-1 method-arm outputs. Specifically, it should expose per-candidate sequence
score, tree distance, BLAST/p-distance evidence, range/co-occurrence prior
weight, marker-resolvability group, and calibrated probability before ranking.

### Eco-Phylo Candidate Posterior

Built a candidate-level posterior table and ran the full scorer on Vast:

- 24 evidence arms, including SSM/CNN sequence-only, learned RLS/OBIS
  co-occurrence, public FishGLOB co-occurrence, BLAST sequence-only,
  BLAST + RLS geography, SSM + RLS geography, SSM + RLS/OBIS geography, and
  SSM same-sample co-occurrence;
- top-5 candidates per arm;
- 6,995,880 candidate rows in the full input table;
- candidate-specific fields for model score, sequence-only score where
  available, BLAST pident/rank where available, RLS count, OBIS count,
  reference support, taxonomy, marker-resolvability flags, and rank correctness;
- full scorer trained on calibration `site20` groups and evaluated on held-out
  `site20` groups.

Full held-out candidate-posterior result:

- species: target-50 gives 11.3% assignment at 48.3% accuracy; target-60 gives
  1.9% assignment at 57.1%; no reliable 70%+ species threshold;
- genus: target-80 gives 3.9% assignment at 80.5%; no reliable 90%+ genus
  threshold;
- family: target-90 gives 28.9% assignment at 86.1%; target-95 gives 16.0%
  assignment at 95.2%;
- order: target-90 gives 22.7% assignment at 90.1%; target-95 gives 4.9%
  assignment at 95.1%;
- mixed rank-backoff: target-90 gives 31.8% assignment at 86.6%; target-95 gives
  16.4% assignment at 95.3%.

Interpretation: candidate-level fusion is the right direction and shows useful
high-confidence family/order/rank-backoff signal. It is now a real full-table
result, but it still does not solve species-level eDNA assignment. The next
version should add direct sequence/tree evidence and stronger
site/region-independent calibration.

### Candidate 12S Sequence Evidence

Built a direct train-reference 12S sequence-evidence table for the candidate
posterior:

- 584,894 unique query/candidate pairs;
- 145,785 pairs have train-reference 12S evidence;
- evidence is best ungapped sliding p-distance/identity against
  `train_species_sequences.json`;
- candidates without train-reference 12S remain explicitly unavailable.

Sampled 10,000 complete-query posterior rerun with this evidence:

- genus target-80: 39.4% assignment at 83.7% accuracy;
- genus target-95: 6.4% assignment at 98.6% accuracy;
- order target-90: 46.0% assignment at 88.9% accuracy;
- order target-95: 37.3% assignment at 93.6% accuracy;
- mixed rank-backoff target-95: 38.9% assignment at 93.6%.

Interpretation: adding direct sequence distance is promising, especially for
genus/order coverage. This first sequence-aware run was only a sampled
diagnostic. Also note that this is best ungapped p-distance, not BLAST and not
full gapped alignment.

### Candidate Tree-Neighborhood Evidence

Added an inference-safe tree-neighborhood evidence layer for the candidate
posterior. This does not use the true query species as a feature. It summarizes
the retrieved candidate set itself:

- candidate distance to the top-ranked candidate on the fish reference tree;
- candidate distance to the retrieved candidate neighborhood;
- genus/family/order agreement with the top candidate;
- top-k taxonomic and tree-distance spread.

Sampled 10,000 complete-query posterior rerun with both direct 12S sequence
evidence and tree-neighborhood evidence:

- genus target-90: 19.7% assignment at 97.0% accuracy;
- genus target-95: 11.9% assignment at 98.3% accuracy;
- family target-90: 42.2% assignment at 92.9% accuracy;
- family target-95: 30.2% assignment at 97.3% accuracy;
- order target-90: 8.5% assignment at 99.1% accuracy;
- order target-95: 7.3% assignment at 99.5% accuracy.

Important caveat: the same sampled run shows species thresholds overfit the
calibration side and fail on held-out groups. That poisons naive mixed
rank-backoff because species is tried first. For eDNA, species should remain
disabled unless an independently calibrated species threshold transfers. The
useful result here is higher-rank signal, not species-level assignment.

Full Vast sequence+tree posterior completed on all 6,995,880 candidate rows:

- species: target-50 gives 48.5% assignment at 45.7% accuracy; target-60 gives
  37.6% assignment at 51.5%; target-90/95 still assign 12.5% but only reach
  50.7%;
- genus: target-90 gives 19.8% assignment at 97.5% accuracy; target-95 gives
  15.2% assignment at 98.6%;
- family: target-90 gives 47.3% assignment at 85.6%; target-95 gives 35.2%
  assignment at 95.1%;
- order: target-90 gives 42.0% assignment at 92.7%; target-95 gives 33.1%
  assignment at 97.0%;
- naive species-first mixed rank-backoff target-95 gives 42.9% assignment at
  80.4% accuracy because species calls still leak through.

Interpretation: the full sequence+tree posterior materially strengthens the
higher-rank eDNA story, especially genus, family, and order. It also confirms
the species-level boundary: species thresholds do not transfer, so species must
be disabled or separately calibrated before any eDNA rank-backoff claim.

Species-disabled rank-backoff, using genus -> family -> order only:

- target-90: 51.7% held-out assignment at 84.3% accuracy;
- target-95: 40.3% held-out assignment at 94.3% accuracy;
- target-95 composition: 4,852 genus calls, 7,094 family calls, 864 order calls.

Interpretation: this is the most honest current eDNA posterior operating point.
It reports high-confidence higher-rank biodiversity evidence while avoiding
unsupported species calls.

Nested threshold-stability check over 30 calibration resplits:

- target-90 on original held-out eDNA groups: 51.7% mean assignment at 84.2%
  mean accuracy;
- target-95 on original held-out eDNA groups: 40.2% mean assignment at 94.3%
  mean accuracy;
- target-95 held-out accuracy 5th-95th percentile: 93.9-94.9%.

Interpretation: the species-disabled target-95 policy is reasonably stable as a
thresholding policy. This is not a full nested model retrain; posterior weights
are still trained on the original calibration groups.

True nested posterior fit:

- model fit on 70% of calibration groups;
- thresholds learned on the remaining calibration groups;
- evaluation on the original held-out eDNA groups.

Species still fails under this stricter split:

- species target-95 assigns 2.3% of held-out rows at only 1.3% accuracy.

Higher ranks remain useful:

- genus target-90: 21.2% assignment at 94.1% accuracy;
- family target-95: 34.0% assignment at 95.4% accuracy;
- order target-95: 34.4% assignment at 96.7% accuracy.

True nested species-disabled rank-backoff:

- target-90: 48.8% held-out assignment at 85.9% accuracy;
- target-95: 38.9% held-out assignment at 93.4% accuracy;
- target-95 composition: 5,174 genus calls, 5,749 family calls, 1,463 order
  calls.

Additional true nested stability repeats:

- rep1 target-95: 38.5% held-out assignment at 95.4% accuracy;
- rep2 target-95: 54.9% held-out assignment at 84.5% accuracy.

Interpretation: the true nested runs confirm the scientific boundary but also
show that this eDNA posterior is not yet stable enough for a headline target-95
claim. We can use it as evidence that conservative higher-rank eDNA inference
is possible, especially family/order, but the current mixed species-disabled
policy needs stricter calibration or better features before becoming a final
operating point.

## MarkerMirror Integrated Evidence Stability

Shared 12S/16S MarkerMirror is now validated beyond a single seed. Candidate
ranking exports were repeated for the original shared-space run and seed 1902,
then joined with the same sequence/reference/tree evidence used for seed 1903.

Source tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_evidence_join_seed_repeat_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_best_target099.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_target099_stability.csv`

Target-0.99 learned MarkerMirror stability across three seeds:

| Direction | Coverage Mean | Coverage Range | Assigned Precision Mean | Precision Range | False Species Calls Mean | Species Precision Mean |
|---|---:|---:|---:|---:|---:|---:|
| 12S -> 16S | 51.0% | 48.7-55.0% | 98.9% | 97.3-100.0% | 0.11% | 99.8% |
| 16S -> 12S | 71.1% | 69.1-75.0% | 98.7% | 98.0-99.1% | 0.47% | 99.3% |

Interpretation: this is the strongest MarkerMirror evidence so far. The
pipeline does not rely on the bridge score alone. It uses MarkerMirror for
candidate generation, then checks candidates against same-marker sequence
evidence, top-k taxonomic consensus, marker-reference availability, exact
sequence ambiguity, and tree-neighborhood evidence before returning a calibrated
species/genus/family/order/no-call decision.

Caveat: the current seed-repeat stability table used exact-sequence ambiguity.
A follow-up seed1903 prototype now adds explicit reference-only marker
resolvability at exact and 0.99 proxy identity. The 0.99 proxy suggests 91.8%
of 12S species and 92.0% of 16S species remain species-resolvable in the
current reference, with most ambiguous cases collapsing to genus. Adding these
features did not change the seed1903 target-0.99 operating point, so the table
is currently an auditable ambiguity layer rather than a performance gain.

Additional source tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_marker_resolvability_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_resolvability_calibrator_seed1903_best_target099.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_production_handoff_next_actions.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_smoke_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_smoke_manifest_summary.csv`

Important caveat: the 0.99 rows use a rare-kmer prefix-identity proxy, not
VSEARCH/edlib clustering. Use this as a compiler prototype until an
alignment-backed backend is run.

Production handoff progress:
`scripts/edna/run_marker_mirror_candidate_generator.py` now runs a bounded
12S->16S FASTA/CSV-style candidate-generation smoke test and writes
candidate/summary/manifest files. This is not final species identification; it
is the executable candidate-generator layer that must next feed the evidence
compiler and rank/no-call calibration.

MarkerMirror union candidate-support audit:

- script: `scripts/edna/build_marker_mirror_union_candidate_support.py`;
- source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_candidate_support_summary.csv`,
  `marker_mirror_union_candidate_support_per_query.csv`, and
  `marker_mirror_same_marker_kmer_candidates_top50.csv.gz`;
- method: union the full-query MarkerMirror 12S->16S top50 candidates with a
  same-marker 12S character-kmer top50 candidate list.

Result:

| Candidate source | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| MarkerMirror 12S->16S top50 | 9.5% | 39.9% | 59.8% | 76.3% |
| Same-marker 12S k-mer top50 | 0.0% | 89.5% | 94.9% | 99.5% |
| Union top50 | 9.5% | 91.7% | 95.1% | 99.6% |

Interpretation:
same-marker 12S support does not restore species calls because the full-query
species are absent from the current 12S reference sequence table, and only
26.6% are present in the 16S reference. It does solve a different part of the
pipeline: high-rank candidate support becomes very strong. The next production
direction is therefore a union candidate generator plus calibrated rank/no-call,
not a forced species classifier.

MarkerMirror union rank/no-call diagnostic:

- script: `scripts/edna/build_marker_mirror_union_rank_policy.py`;
- production candidate table:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_candidate_rank_policy/marker_mirror_union_production_candidates.csv.gz`;
- source tables:
  `marker_mirror_union_static_policy_summary.csv`,
  `marker_mirror_union_score_gate_validation_summary.csv`, and
  `marker_mirror_union_score_gate_validation_per_split.csv`.

The production candidate table has 355,231 rows and does not require hidden
labels. Evaluation labels are used only in the diagnostic summaries.

Static top-1 source-agreement policies:

| Policy | Coverage | Assigned Precision | Species Calls |
|---|---:|---:|---:|
| deepest genus/family/order agreement | 25.9% | 95.2% | 0 |
| family/order agreement only | 25.2% | 98.4% | 0 |

Repeated species-split score-gate validation:

| Gate | Mean Coverage | Mean Precision | Target Met Rate |
|---|---:|---:|---:|
| same-marker family, target 0.95 | 92.7% | 94.8% | 64.0% |
| same-marker family, target 0.99 | 32.8% | 98.8% | 32.0% |
| same-marker order, target 0.95 | 99.4% | 94.1% | 70.0% |
| same-marker order, target 0.99 | 68.5% | 99.0% | 54.0% |

Interpretation:
source agreement is conservative and clean enough to be useful immediately as a
diagnostic. The score gates show strong high-rank signal, but the target-met
rates are not stable enough to call them locked production thresholds. The next
method step is a better calibrated evidence compiler over the union candidate
features, not another blind candidate generator.

Learned union evidence compiler:

- script: `scripts/edna/train_marker_mirror_union_evidence_compiler.py`;
- source tables:
  `marker_mirror_union_evidence_compiler_summary.csv`,
  `marker_mirror_union_evidence_compiler_family_order_summary.csv`,
  `marker_mirror_union_evidence_compiler_order_summary.csv`, and
  `marker_mirror_union_evidence_compiler_features.csv`;
- method: HGB classifiers over 102 production-available top-1 union features,
  with train/calibration/evaluation split by query species.

Result:

| Compiler | Target | Mean Coverage | Mean Precision | Target Met Rate |
|---|---:|---:|---:|---:|
| family+order HGB | 0.95 | 99.1% | 91.8% | 34.0% |
| family+order HGB | 0.99 | 72.2% | 97.9% | 20.0% |
| order-only HGB | 0.95 | 99.0% | 94.6% | 76.0% |
| order-only HGB | 0.99 | 67.4% | 98.5% | 44.0% |

Interpretation:
the learned compiler is not yet better than the simple diagnostics. It is still
valuable because it tells us the current feature set/model does not solve
calibration transfer. The strongest clean union result remains family/order
source agreement at 25.2% coverage and 98.4% precision. The strongest
high-coverage high-rank signal remains simple same-marker order score gating,
not the HGB compiler.

MarkerMirror union reason-code and reference-curation layer:

- script: `scripts/edna/build_marker_mirror_union_reason_codes.py`;
- source tables:
  `marker_mirror_union_reason_code_summary.csv`,
  `marker_mirror_union_reason_code_by_source.csv`,
  `marker_mirror_union_reason_code_per_query.csv`, and
  `marker_mirror_union_reference_curation_priorities.csv`;
- method: join union candidate support, top-1 source agreement, reference
  availability, and static family/order assignments into query-level reason
  codes.

Main diagnostic breakdown:

| Primary reason | Queries | Interpretation |
|---|---:|---|
| high-rank union support to genus | 2,249 / 3,566 | The union list usually contains the right genus even when species is not supportable. |
| conservative family source agreement | 621 / 3,566 | Family can be emitted with species disabled; precision is 98.1%. |
| conservative order source agreement | 277 / 3,566 | Order can be emitted with species disabled; precision is 99.3%. |
| species present in 16S but not retrieved | 145 / 3,566 | This is a real cross-marker retrieval/model failure mode. |
| species candidate available, needs calibration | 139 / 3,566 | Species appears in the candidate set but should not yet be called automatically. |
| species absent from both current 12S/16S references | 135 / 3,566 | Reference curation is the limiting factor. |

Top curation priorities include `Epinephelus_coioides`, `Gadus_morhua`,
`Pareiorhaphis_hystrix`, and `Acrossocheilus_paradoxus`, where high-rank
support is strong but species-level reference coverage blocks a species call.
The top retrieval-failure priority is `Trichiurus_lepturus`: the species exists
in the 16S reference, but MarkerMirror rarely recovers it at species level.

Interpretation:
the pipeline is now able to say *why* it cannot make a species call. Most
failures are not generic "model is bad" failures. They separate into current
reference gaps, high-rank-only molecular support, and a smaller set of
cross-marker retrieval misses. This is exactly the evidence-compiler story:
candidate generation should feed rank/no-call and active reference curation,
not forced species labels.

Same-marker edlib alignment validation:

- script: `scripts/edna/build_marker_mirror_same_marker_edlib_validation.py`;
- source tables:
  `marker_mirror_same_marker_edlib_support_summary.csv`,
  `marker_mirror_same_marker_edlib_support_per_query.csv`,
  `marker_mirror_same_marker_edlib_candidates_top50.csv.gz`, and
  `marker_mirror_same_marker_edlib_validation_manifest.json`;
- remote run:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_edlib_validation/edlib_same_marker_validation/`;
- method: score and rerank the existing same-marker 12S k-mer top50 candidate
  pool using bidirectional edlib HW edit-distance identity against up to 12
  reference sequences per candidate species.

Result:

| Same-marker source | Top-k | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|---:|
| k-mer original | 1 | 0.0% | 74.2% | 90.2% | 93.5% |
| edlib rerank | 1 | 0.0% | 73.0% | 89.4% | 92.3% |
| k-mer original | 10 | 0.0% | 86.9% | 93.9% | 98.7% |
| edlib rerank | 10 | 0.0% | 87.8% | 94.3% | 98.8% |
| k-mer original | 50 | 0.0% | 89.5% | 94.9% | 99.5% |
| edlib rerank | 50 | 0.0% | 89.5% | 94.9% | 99.5% |

Interpretation:
alignment-backed scoring preserves the high-rank same-marker signal. This
reduces the risk that Exp 103 was only a k-mer artifact. It still does not
restore species support because the current same-marker reference pool lacks
the query species by design. This is edlib validation of the candidate pool,
not full all-vs-all BLAST/VSEARCH replacement.

List-level selective compiler:

- script: `scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py`;
- source tables:
  `marker_mirror_union_listwise_selective_family_order_summary.csv`,
  `marker_mirror_union_listwise_selective_order_summary.csv`,
  `marker_mirror_union_listwise_selective_family_order_per_split.csv`,
  `marker_mirror_union_listwise_selective_order_per_split.csv`, and matching
  threshold/feature tables;
- method: train HGB rank selectors from 223 production-available list-level
  features: MarkerMirror list concentration, k-mer list concentration,
  edlib-reranked list concentration, source agreement, score margins, and
  edit-distance identities. Validation uses repeated query-species
  train/calibration/evaluation splits.

Result:

| Compiler | Target | Mean Coverage | Mean Precision | Target Met Rate |
|---|---:|---:|---:|---:|
| family+order listwise HGB | 0.95 | 94.3% | 90.5% | 18.0% |
| family+order listwise HGB | 0.99 | 83.8% | 98.2% | 26.0% |
| order-only listwise HGB | 0.95 | 94.0% | 93.1% | 48.0% |
| order-only listwise HGB | 0.99 | 83.1% | 98.8% | 56.0% |

Interpretation:
the list-level compiler is better than the previous top-1 HGB for high-coverage
order calls: order-only target-0.99 improves from 67.4% coverage at 98.5%
precision to 83.1% coverage at 98.8% precision. It is still not a locked
target-0.99 production operating point because only 56% of species-split repeats
meet the 99% precision target. The production-safe layer remains conservative
family/order source agreement plus reason codes; the listwise compiler is a
promising diagnostic/high-coverage candidate, not the final assignment policy.

VSEARCH same-marker candidate generation:

- scripts:
  `scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py` and
  `scripts/edna/build_marker_mirror_union_vsearch_candidate_support.py`;
- source tables:
  `marker_mirror_same_marker_vsearch_support_summary.csv`,
  `marker_mirror_same_marker_vsearch_support_per_query.csv`,
  `marker_mirror_same_marker_vsearch_candidates_top50.csv.gz`,
  `marker_mirror_union_vsearch_candidate_support_summary.csv`,
  `marker_mirror_union_vsearch_candidate_support_per_query.csv`, and matching
  manifests;
- remote run:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_vsearch_same_marker/vsearch_same_marker_full/`;
- method: VSEARCH 2.27 `--usearch_global` over 3,566 full-query 12S sequences
  against 12,593 current 12S reference sequences, keeping top-50 species-level
  candidates.

Same-marker VSEARCH support:

| Top-k | Species | Genus | Family | Order |
|---:|---:|---:|---:|---:|
| 1 | 0.0% | 76.2% | 90.5% | 92.1% |
| 5 | 0.0% | 86.7% | 92.6% | 93.8% |
| 10 | 0.0% | 88.4% | 93.3% | 94.6% |
| 50 | 0.0% | 90.4% | 94.9% | 99.4% |

MarkerMirror + VSEARCH top50 union support:

| Candidate source | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| MarkerMirror 12S->16S | 9.5% | 39.9% | 59.8% | 76.3% |
| same-marker 12S VSEARCH global | 0.0% | 90.4% | 94.9% | 99.4% |
| union | 9.5% | 91.8% | 95.1% | 99.6% |

Interpretation:
the high-rank same-marker story survives a real VSEARCH global-alignment
candidate-generation run. This is stronger than the k-mer audit and stronger
than merely reranking the k-mer pool with edlib. It still does not solve
species-level 12S because the query species are absent from the current
same-marker reference table by design. VSEARCH global alignment is not BLAST
local alignment, but it is now a proper classical same-marker candidate source
inside the union pipeline.

BLAST same-marker candidate generation:

- scripts:
  `scripts/edna/build_marker_mirror_same_marker_blast_candidates.py` and
  `scripts/edna/build_marker_mirror_union_blast_candidate_support.py`;
- source tables:
  `marker_mirror_same_marker_blast_support_summary.csv`,
  `marker_mirror_same_marker_blast_support_per_query.csv`,
  `marker_mirror_same_marker_blast_candidates_top50.csv.gz`,
  `marker_mirror_union_blast_candidate_support_summary.csv`,
  `marker_mirror_union_blast_candidate_support_per_query.csv`, and matching
  manifests;
- remote run:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_blast_same_marker/blast_same_marker_full/`;
- method: NCBI BLASTN 2.12.0 local alignment over 3,566 full-query 12S
  sequences against 12,593 current 12S reference sequences, keeping top-50
  species-level candidates.

Same-marker BLASTN support:

| Top-k | Species | Genus | Family | Order |
|---:|---:|---:|---:|---:|
| 1 | 0.0% | 77.3% | 93.1% | 94.7% |
| 5 | 0.0% | 88.3% | 94.4% | 95.5% |
| 10 | 0.0% | 89.3% | 94.7% | 99.1% |
| 50 | 0.0% | 90.7% | 95.1% | 99.4% |

MarkerMirror + BLASTN top50 union support:

| Candidate source | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| MarkerMirror 12S->16S | 9.5% | 39.9% | 59.8% | 76.3% |
| same-marker 12S BLASTN local | 0.0% | 90.7% | 95.1% | 99.4% |
| union | 9.5% | 92.1% | 95.3% | 99.7% |

Interpretation:
BLASTN local alignment confirms the same high-rank candidate-generation story
seen with VSEARCH. The union improves genus/order support slightly over the
VSEARCH union while leaving species unchanged because the current same-marker
reference table does not contain the held-out query species by design.

BLAST/VSEARCH calibration-transfer repair diagnostic:

- script:
  `scripts/edna/build_marker_mirror_blast_vsearch_calibration_repair.py`;
- source tables:
  `marker_mirror_blast_vsearch_calibration_repair_summary.csv`,
  `marker_mirror_blast_vsearch_calibration_repair_per_split.csv`,
  `marker_mirror_blast_vsearch_calibration_repair_thresholds.csv`,
  `marker_mirror_blast_vsearch_calibration_repair_features.csv`, and
  `marker_mirror_blast_vsearch_calibration_repair_policy_rows.csv.gz`;
- method: 50 repeated query-species calibration/evaluation splits over
  production-available BLASTN/VSEARCH/MarkerMirror list evidence.

Best target-0.99 rows:

| Policy | Rank | Strategy | Coverage | Precision | Target Met |
|---|---|---|---:|---:|---:|
| MarkerMirror+BLAST+VSEARCH top1 agreement | order | global precision | 24.8% | 99.6% | 100% |
| VSEARCH top10 mode | order | source-stratified precision | 56.1% | 99.5% | 94% |
| BLAST top10 mode | order | source-stratified Wilson95 | 69.0% | 99.4% | 82% |
| BLAST+VSEARCH top10 agreement | order | source-stratified Wilson95 | 68.4% | 99.4% | 86% |

Interpretation:
this repairs the conservative production-safe order layer, but not the whole
rank/no-call problem. All-source top1 agreement gives a stable target-0.99
order-call policy at about 24.8% coverage. The raw high-coverage Exp 111 rows
meet target-0.99 in 82-94% rather than 100% of the 50 species-split repeats.
Use Exp 117's nested repair for the stronger high-coverage order result.

Stable order policy handoff:

- script:
  `scripts/edna/build_marker_mirror_stable_order_policy.py`;
- source tables:
  `marker_mirror_stable_order_policy_assignments.csv`,
  `marker_mirror_stable_order_policy_production_assignments.csv`,
  `marker_mirror_stable_order_policy_summary.csv`,
  `marker_mirror_stable_order_policy_by_source.csv`,
  `marker_mirror_stable_order_policy_reason_counts.csv`, and
  `marker_mirror_stable_order_policy_manifest.json`;
- policy: emit an order call only when MarkerMirror, BLASTN, and VSEARCH all
  agree on the top-1 order.

Applied over all 3,566 full-query 12S rows:

| Policy | Coverage | Precision | False Species Calls |
|---|---:|---:|---:|
| unthresholded all-source top1 order agreement | 24.8% | 99.7% | 0 |
| max-repeat target-0.99 threshold | 24.7% | 99.7% | 0 |

The max-repeat threshold is the most conservative global precision threshold
seen across the 50 Exp 111 repeats. It removes only 6 additional queries here,
so the practical behavior is almost identical to the simpler all-source
agreement rule.

The production-facing handoff table,
`marker_mirror_stable_order_policy_production_assignments.csv`, strips
truth/correctness columns and keeps only query id, source, assigned rank/label,
confidence, source top-1 order evidence, and the reason code. This is not yet
an arbitrary-FASTA 12S CLI because it still assumes the MarkerMirror, BLASTN,
and VSEARCH feature table has already been generated.

12S orchestration wrapper status:

- script:
  `scripts/edna/run_marker_mirror_12s_production_v1.py`;
- dry-run smoke:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/`;
- local dependency result:
  BLASTN and makeblastdb are available; VSEARCH is missing;
- local partial smoke:
  the BLASTN same-marker stage ran on 2 normalized 12S queries and wrote 100
  top-50 candidate rows.

Interpretation:
the one-command wrapper skeleton is real and dependency-gated locally. It can
plan the full chain and run the BLASTN leg locally, but complete local
all-source execution still needs VSEARCH installed.

Vast full one-command run:

- local copied output:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/`;
- input:
  all 3,566 current 12S query rows;
- stages completed:
  MarkerMirror 12S->16S, BLASTN same-marker search, VSEARCH same-marker search,
  all-source feature table, stable order/no-call policy;
- final production assignments:
  880 order calls and 2,686 no-calls;
- diagnostic labelled performance:
  24.7% coverage, 99.7% precision, 0 false species calls;
- runtime:
  15.0 s MarkerMirror, 254.6 s BLASTN, 48.8 s VSEARCH, 1.6 s stable policy.

Interpretation:
the MarkerMirror 12S production-v1 wrapper is now an executable end-to-end
research pipeline on Vast. It is still conservative order/no-call only, not a
species identifier.

Unlabeled FASTA smoke:

- local copied output:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/`;
- input:
  2 FASTA records with taxonomy labels stripped;
- stages completed:
  MarkerMirror, BLASTN, VSEARCH, feature table, stable order/no-call policy;
- final production assignments:
  1 order call and 1 no-call;
- diagnostic precision:
  unavailable/blank, as expected, because no truth labels were supplied.

CLI usage doc:
`experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md`.

## High-Coverage Order Repair Diagnostic

Exp 117 adds a nested species-split repair diagnostic for the higher-coverage
BLAST/VSEARCH order policies.

Source tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_high_coverage_order_repair_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_high_coverage_order_repair_assignment_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_high_coverage_order_repair_assignments.csv`

Best nested row:

- policy: BLASTN/VSEARCH top-10 order agreement;
- thresholding: nested global Wilson95 locking;
- held-out species-split coverage: 57.2%;
- held-out assigned precision: 99.8%;
- target-0.99 met in 100% of 50 outer repeats;
- minimum repeat precision: 99.3%.

Full-table locked diagnostic:

- 2,513/3,566 queries receive an order call;
- coverage: 70.5%;
- labelled precision: 99.8%;
- remaining rows: 1,053 no-calls.

Interpretation: this is the strongest high-coverage 12S order/no-call result so
far. Exp 118 exposes it as explicit `high_coverage_order`; it remains a
diagnostic/research mode, not the default production CLI policy.

Exp 118 wiring status:

- `scripts/edna/run_marker_mirror_12s_production_v1.py` now supports
  `--decision-mode stable_order` and `--decision-mode high_coverage_order`;
- `stable_order` remains the default;
- `high_coverage_order` is explicit, order-only, and diagnostic/research mode.

Vast smoke outputs:

- labelled stable mode: 4 rows, 1 order call, 3 no-calls, 100% diagnostic
  precision on assigned rows;
- labelled high-coverage mode: 4 rows, 3 order calls, 1 no-call, 100%
  diagnostic precision on assigned rows;
- unlabeled stable mode: 2 rows, 1 order call, 1 no-call, precision blank by
  design;
- unlabeled high-coverage mode: 2 rows, 2 order calls, precision blank by
  design.

Exp 119 family/genus repair status:

- generalized `scripts/edna/build_marker_mirror_high_coverage_order_repair.py`
  with `--rank genus|family|order`;
- created `marker_mirror_high_coverage_rank_repair_comparison.csv`;
- no family/genus row met target-0.99 in 100% of 50 species-split repeats;
- best family all-policy row: 35.5% mean coverage, 99.35% mean precision,
  target met in 94% of repeats;
- best genus all-policy row: 7.8% mean coverage, 99.79% mean precision,
  target met in 98% of repeats;
- conclusion: keep family/genus disabled in the 12S wrapper. Order is the only
  high-coverage rank currently stable enough for an explicit diagnostic mode.

Exp 121 set-valued family/genus diagnostic:

- script: `scripts/edna/build_marker_mirror_hierarchical_candidate_sets.py`;
- source tables:
  `marker_mirror_hierarchical_candidate_sets_summary.csv`,
  `marker_mirror_hierarchical_candidate_sets_policy_grid_summary.csv`,
  `marker_mirror_hierarchical_candidate_sets_assignments.csv.gz`;
- this tested a different strategy: emit a small set of plausible taxa instead
  of one forced family/genus label;
- result: still no stable 99% family/genus solution;
- best full-query family set coverage was 95.4% using `all_union` top50, but
  the mean set size was 34.4 families;
- best full-query genus set coverage was 92.4% using `all_union` top50, with
  mean set size 79.6 genera;
- the only stable target-0.99 set-valued rank remains order. Observed
  `all_intersection` top1 order sets met target in 100% of repeats with 25.0%
  emission coverage and set size 1; Wilson95 order sets reached higher emission
  but met target in only 90% of repeats.

Interpretation: family/genus are not merely failing because we forced a single
label. Under the current evidence, even broad set-valued family/genus output
cannot deliver reliable 99% coverage without becoming too large to be useful.
This strengthens the claim boundary: order/no-call is the current defensible 12S
MarkerMirror output.

Exp 122 manuscript-facing MarkerMirror package:

- script: `scripts/edna/build_marker_mirror_manuscript_assets.py`;
- output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/`;
- files:
  `marker_mirror_candidate_support_table.csv`,
  `marker_mirror_order_policy_table.csv`,
  `marker_mirror_rank_boundary_table.csv`,
  `marker_mirror_runtime_table.csv`,
  `marker_mirror_figure_plan.csv`,
  `marker_mirror_methods_blurb.md`, and
  `marker_mirror_manuscript_asset_manifest.json`;
- log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_assets.log`.

Interpretation: this is a manuscript planning package, not a new metric. It
turns the MarkerMirror candidate-support, order/no-call, rank-boundary, and
runtime rows into a compact figure/table/methods bundle for coauthor review.

Exp 123 MarkerMirror figure drafts:

- script: `scripts/edna/build_marker_mirror_manuscript_figures.py`;
- output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/`;
- files:
  `marker_mirror_candidate_support_bars.png` / `.pdf`,
  `marker_mirror_order_policy_tradeoff.png` / `.pdf`,
  `marker_mirror_rank_boundary.png` / `.pdf`,
  `marker_mirror_runtime_breakdown.png` / `.pdf`,
  `marker_mirror_slide_ready_summary.md`, and
  `marker_mirror_manuscript_figure_manifest.json`;
- log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_figures.log`.

Interpretation: these are draft visuals for coauthor/manuscript discussion,
not new benchmark rows. The claim boundary is unchanged: order/no-call is the
current defensible 12S MarkerMirror output.

Exp 124 MarkerMirror slide-ready package:

- script: `scripts/edna/build_marker_mirror_slide_tables.py`;
- output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/`;
- files:
  `marker_mirror_candidate_support_slide_table.csv` / `.md`,
  `marker_mirror_order_policy_slide_table.csv` / `.md`,
  `marker_mirror_rank_boundary_slide_table.csv` / `.md`,
  `marker_mirror_runtime_slide_table.csv` / `.md`,
  `marker_mirror_slide_package_outline.md`, and
  `marker_mirror_slide_tables_manifest.json`;
- log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_slide_tables.log`.

Interpretation: this is a five-slide outline and table package for coauthor or
deck assembly. It repackages existing MarkerMirror metrics and does not change
the claim boundary.

Exp 125 MarkerMirror manuscript text package:

- script: `scripts/edna/build_marker_mirror_manuscript_text.py`;
- output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`;
- files:
  `marker_mirror_figure_captions.md`,
  `marker_mirror_results_paragraph.md`,
  `marker_mirror_methods_paragraph.md`,
  `marker_mirror_claim_boundary_box.md`,
  `marker_mirror_caption_inventory.csv`, and
  `marker_mirror_manuscript_text_manifest.json`;
- log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_text.log`.

Interpretation: this is manuscript drafting text generated from the same
source tables and figures. It is not a new result and does not change the
order/no-call claim boundary.

Exp 126 MarkerMirror manuscript section outline:

- script: `scripts/edna/build_marker_mirror_manuscript_section_outline.py`;
- output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`;
- files:
  `marker_mirror_manuscript_section_outline.md`,
  `marker_mirror_manuscript_section_checklist.csv`, and
  `marker_mirror_manuscript_section_manifest.json`;
- log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_section_outline.log`.

Interpretation: this is a manuscript organization layer that integrates the
Exp 122-125 tables, figures, and text into a Paper 1 section plan. It is not a
new benchmark and does not change the order/no-call claim boundary.

Exp 127 MarkerMirror family/genus next-evidence audit:

- script: `scripts/edna/build_marker_mirror_next_evidence_audit.py`;
- source-table outputs:
  `marker_mirror_next_evidence_source_audit.csv`,
  `marker_mirror_reference_coverage_by_lineage.csv`, and
  `marker_mirror_next_evidence_source_manifest.json`;
- manuscript text output:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_family_genus_next_evidence_plan.md`;
- log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_next_evidence_audit.log`.

Interpretation: this is a planning and data-availability audit, not a new
rank/no-call result. The P0 paths are lineage-specific reference coverage,
alignment-backed marker-resolvability, and active reference-curation/value of
information. Geography and co-occurrence are reserved for sample-aware eDNA
mode because arbitrary FASTA input lacks site/sample context.

Exp 128 MarkerMirror lineage/reference-coverage policy diagnostic:

- script:
  `scripts/edna/build_marker_mirror_reference_coverage_policy_diagnostic.py`;
- source-table outputs:
  `marker_mirror_reference_coverage_policy_diagnostic_summary.csv`,
  `marker_mirror_reference_coverage_policy_diagnostic_per_split.csv`,
  `marker_mirror_reference_coverage_policy_diagnostic_thresholds.csv`,
  `marker_mirror_reference_coverage_policy_diagnostic_features.csv`,
  `marker_mirror_reference_coverage_policy_diagnostic_lineage_features.csv`,
  and `marker_mirror_reference_coverage_policy_diagnostic_manifest.json`;
- local output root:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/reference_coverage_policy_diagnostic/`;
- method:
  join production-available BLASTN/VSEARCH/MarkerMirror policy rows to
  lineage-level reference coverage features, train rank-specific HGB
  correctness models, calibrate on species-disjoint rows, and evaluate on
  held-out species.

Summary over 50 species-split repeats:

| Rank | Target | Mean coverage | Mean precision | Target met | Decision |
|---|---:|---:|---:|---:|---|
| Order | 0.95 | 99.9% | 94.6% | 72% | Diagnostic only |
| Family | 0.95 | 99.2% | 93.4% | 58% | Diagnostic only |
| Genus | 0.99 | 17.7% | 97.6% | 42% | Diagnostic only |
| Genus | 0.95 | 60.4% | 92.8% | 24% | Diagnostic only |
| Order | 0.99 | 96.6% | 96.2% | 14% | Diagnostic only |
| Family | 0.99 | 87.4% | 98.0% | 10% | Diagnostic only |

Interpretation: lineage/reference-coverage features alone do not stabilize
family/genus transfer. This is a useful negative result and keeps the current
12S wrapper order/no-call only. The next family/genus attempt needs
alignment-backed marker-resolvability, active reference-curation evidence, or
sample-aware geography/co-occurrence, not another wrapper around the same
policy rows.

Exp 129 VSEARCH-backed marker-resolvability diagnostic:

- script:
  `scripts/edna/build_12s_near_exact_resolvability.py`;
- remote/copy root:
  `results/remote_runs/2026-06-04/rtx_pro_6000/marker_mirror_vsearch_resolvability_20260604/`;
- source-table outputs:
  `marker_mirror_vsearch_resolvability_summary.csv`,
  `marker_mirror_vsearch_resolvability_cluster_rank_counts.csv`,
  `marker_mirror_vsearch_resolvability_query_oracle_rates.csv`,
  `marker_mirror_vsearch_resolvability_12s_summary.csv`,
  `marker_mirror_vsearch_resolvability_16s_summary.csv`,
  and `marker_mirror_vsearch_resolvability_manifest.json`;
- method:
  VSEARCH `cluster_fast` near-exact clustering at 0.99, 0.98, 0.97, and 0.95
  identity for 12S and 16S marker references.

12S query oracle support:

| Identity | Query clusters with reference | Species | Genus | Family | Order |
|---:|---:|---:|---:|---:|---:|
| 0.99 | 19.6% | 77.9% | 95.2% | 99.6% | 99.7% |
| 0.98 | 33.3% | 64.0% | 90.8% | 98.8% | 99.3% |
| 0.97 | 43.4% | 53.5% | 83.5% | 96.3% | 98.8% |
| 0.95 | 57.9% | 38.3% | 73.8% | 92.0% | 98.2% |

16S rows are reference-cluster summaries only because the current
`16s_multisource` table has no held-out `zero_shot_queries.csv`. At 0.99
identity, 16S reference clusters are mostly species-level
(`1876/1988` clusters), with 93 genus-level, 16 family-level, and 3 unresolved
clusters.

Interpretation: Exp 129 replaces the rare-kmer resolvability proxy with an
alignment-backed VSEARCH diagnostic. It confirms that 12S has strong high-rank
oracle support but limited species/reference support under current benchmark
coverage. This is evidence-source hardening only; it does not enable
family/genus/species calls without a separately validated policy.

Exp 130 VSEARCH-resolvability-aware policy diagnostic:

- script:
  `scripts/edna/build_marker_mirror_vsearch_resolvability_policy_diagnostic.py`;
- source-table outputs:
  `marker_mirror_vsearch_resolvability_policy_diagnostic_summary.csv`,
  `marker_mirror_vsearch_resolvability_policy_diagnostic_per_split.csv`,
  `marker_mirror_vsearch_resolvability_policy_diagnostic_thresholds.csv`,
  `marker_mirror_vsearch_resolvability_policy_diagnostic_features.csv`,
  `marker_mirror_vsearch_resolvability_policy_diagnostic_query_features.csv`,
  and `marker_mirror_vsearch_resolvability_policy_diagnostic_manifest.json`;
- local output root:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/vsearch_resolvability_policy_diagnostic/`;
- method:
  join production-available VSEARCH cluster features from Exp 129 to existing
  MarkerMirror/BLASTN/VSEARCH policy rows, excluding the hidden oracle-support
  labels, then train and evaluate rank-specific HGB policy rows with
  species-disjoint calibration/evaluation.

Summary over 50 species-split repeats:

| Rank | Target | Mean coverage | Mean precision | Target met | Decision |
|---|---:|---:|---:|---:|---|
| Order | 0.95 | 84.0% | 95.7% | 78% | Diagnostic only |
| Family | 0.95 | 74.7% | 94.8% | 70% | Diagnostic only |
| Order | 0.99 | 71.3% | 96.3% | 52% | Diagnostic only |
| Family | 0.99 | 57.3% | 95.5% | 44% | Diagnostic only |
| Genus | 0.95 | 24.3% | 88.7% | 42% | Diagnostic only |
| Genus | 0.99 | 11.5% | 87.8% | 38% | Diagnostic only |

Interpretation: production-available VSEARCH resolvability features do not
stabilize family/genus transfer when used as another learned policy input. This
is a useful negative result. Exp 129 remains marker-ceiling evidence, but the
12S wrapper stays order/no-call only.

Exp 131 MarkerMirror active reference-curation/value-of-information:

- Script:
  `scripts/edna/build_marker_mirror_active_reference_value.py`.
- Source tables:
  `marker_mirror_active_reference_value_species.csv`,
  `marker_mirror_active_reference_value_lineage.csv`,
  `marker_mirror_active_reference_value_actions.csv`,
  and `marker_mirror_active_reference_value_manifest.json`.
- Scope:
  795 species groups, 698 lineage rows, and 7 action categories built from
  existing reason-code, policy, and VSEARCH-resolvability tables.
- Largest action category:
  add both 12S and 16S species references where the benchmark evidence suggests
  high expected value: 532 species groups and 1,928 queries.
- Other major action categories:
  add reference then validate genus/family rather than species
  (147 species groups, 685 queries), and add same-marker 12S references then
  revalidate (89 species groups, 690 queries).
- Top active targets:
  `Trichiurus_lepturus`, `Epinephelus_coioides`, `Oryzias_latipes`,
  `Gadus_morhua`, `Acrossocheilus_paradoxus`, and
  `Pareiorhaphis_hystrix`.

Interpretation: Exp 131 converts failure analysis into an active curation
priority map. It does not enable family/genus/species. It identifies what new
reference evidence would need to change before another rank-lift attempt is
scientifically different from the failed threshold, set-valued,
lineage-coverage, and VSEARCH-cluster-feature repairs.

## What We Can Claim Now

- Fast vector retrieval is a practical candidate-generation layer.
- Classical sequence methods remain strong and should be retained.
- Tree-aware learned embeddings and rank/no-call decisions are useful under
  missing references.
- Strict missing-reference validation supports rank backoff.
- 12S species-level assignment is often marker-limited.
- eDNA assignment needs evidence integration, not sequence-only species forcing.
- MarkerMirror + same-marker BLAST/VSEARCH can assemble very strong high-rank
  12S candidate sets, and all-source agreement can make conservative order
  calls with high transfer precision. The stable order policy now has an
  explicit assignment/reason-code source table.
- A stricter nested BLAST/VSEARCH top-10 agreement diagnostic can raise
  order-level coverage substantially while preserving target-0.99 transfer in
  repeated species splits.
- MarkerMirror can now emit active reference-curation priorities: it can tell
  users which missing/weak marker references or target-marker curation fixes
  would most likely improve the evidence, while keeping unsupported ranks
  disabled.

## What We Should Not Claim

- Deep learning replaces BLAST.
- Mamba is the best architecture.
- Species-level eDNA is solved.
- Current eDNA rank/no-call is final.
- Fernando is exactly reproduced.
- Vector barcode search is novel by itself.

## Immediate Next Work

1. Improve calibration:
   - Exp111 is now at 50 repeats and Exp112 has an explicit conservative
     order/no-call handoff,
   - Exp117 gives a stronger high-coverage order diagnostic and Exp118 exposes
     it as explicit `high_coverage_order`,
   - Exp119 tested family/genus repair; keep those ranks disabled unless a new
     calibration strategy beats the current transfer failure.
   - Exp121 tested set-valued family/genus output; the failure remains, so the
     next family/genus attempt needs new evidence, not only new thresholding.
2. Add remaining candidate evidence:
   - explicit per-candidate co-occurrence/range weights where available.
3. Harden production v1 from raw split-sequence timing into arbitrary FASTA
   input plus a simple CLI/API wrapper.
4. Use the Exp125 manuscript text package for coauthor/manuscript review, or
   add genuinely new evidence before attempting family/genus again.
5. Use Exp131 active-curation tables to choose any future targeted reference
   additions before attempting family/genus again.
6. Choose final COI operating point for the main paper.

## Start-Here Files

- `experiments/paper1_phylo_calibrated_assignment/README.md`
- `experiments/paper1_phylo_calibrated_assignment/PIPELINE.md`
- `experiments/paper1_phylo_calibrated_assignment/SOURCE_TABLES.md`
- `experiments/paper1_phylo_calibrated_assignment/CLAIM_BOUNDARIES.md`
- `experiments/paper1_phylo_calibrated_assignment/COAUTHOR_BRIEF.md`
- `experiments/paper1_phylo_calibrated_assignment/PRODUCTION_PIPELINE_V1.md`
- `experiments/research_program/method_angles/README.md`

## Source Tables

- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_end_to_end_summary.csv`
- `results/paper1_phylo_calibrated_assignment/production_v1/production_v1_summary_all.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_coi_method_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_edna_method_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/merged_12s_zero_shot_model_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/edna_evidence_decomposition_matrix.csv`
- `results/paper1_phylo_calibrated_assignment/global_edna_independent_rank_calibration/global_edna_independent_rank_calibration_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_method_design.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_rank_correctness_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_query_features.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_model_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_features_top5.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_method_inventory.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_model_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_top5.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`
