# MarkerMirror / BarcodeBridge Prospect

Last updated: 2026-06-03

## One-Line Idea

Train a cross-marker DNA model that maps an eDNA marker fragment into another
marker/species space, so low-resolution molecular evidence can borrow structure
from richer or complementary references without pretending a single marker has
species-level resolution.

## Why This Could Matter

COI, 12S, and 16S solve different parts of the biodiversity problem:

- COI often has stronger species-level barcode resolution and richer barcode
  reference coverage.
- 12S/eDNA is common in environmental monitoring but often cannot uniquely
  resolve species.
- 16S is also eDNA-compatible and biologically closer to 12S than COI, making
  it a cleaner ribosomal bridge marker for the current paper.
- A direct 12S classifier can only learn what 12S contains. A bridge can ask a
  different question: where does this fragment land in another marker/species
  coordinate system?

The useful claim would be:

> We made a cross-marker DNA bridge that lets eDNA fragments retrieve
> complementary marker/species candidates and improves candidate recovery over
> frozen foundation embeddings.

## What Is Different From Existing Work

This is not "embedding search for DNA"; that already exists. This is also not a
standard multi-marker classifier.

The intended novelty is:

- paired marker-to-marker contrastive training;
- held-out species evaluation, not only seen-species classification;
- retrieval into COI/tree-space from 12S/eDNA queries;
- downstream use inside calibrated rank/no-call biodiversity inference.

Claim boundary: this needs a deeper formal literature audit before any
publication claim. Current status is "promising direct prospect," not
"proven novel field first."

## Adjacent Work To Respect

Recent and adjacent work already covers important pieces:

- DNACSE uses contrastive learning to improve DNA barcode embeddings from
  genomic language models.
- DeepCOI is a language-model framework for COI metabarcoding assignment.
- OceanOmics `eDNABERT-S_12S` fine-tunes DNABERT-S on 12S eDNA ASVs.
- TaxoTagger is a semantic-search/vector-database tool for DNA barcode
  identification.
- BarcodeBERT, BarcodeMamba, and DNABERT-S show that barcode/foundation-model
  sequence embeddings are an active area.

Therefore our claim cannot be "we made neural barcode embeddings" or "we made
vector search for barcodes." The defensible prospect is narrower and stronger:
paired cross-marker bridging, especially 12S/16S ribosomal marker alignment,
plus calibrated rank/no-call inference.

## First Probe

Script:

- `scripts/edna/train_marker_mirror_bridge.py`

Backbone:

- frozen `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`

Training:

- COI input:
  `data/phylo/fish_tree_clean_phylo_inputs/eval_c/train_species_sequences.json`
- 12S input:
  `data/edna/stalder_inputs/multisource/train_species_sequences.json`
- overlap species with both markers: 963
- species-level split: 674 train, 144 validation, 145 held-out test
- model: COI projection head + 12S projection head
- objective: symmetric contrastive COI<->12S InfoNCE

Outputs:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_multisource/`

## First Result

Held-out 12S query to COI species prototype retrieval, top-10:

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

These seen/training rows are useful for diagnosing capacity: the
projection-head bridge can align species it was trained on almost perfectly,
whereas frozen NT cannot. They are not the manuscript generalization result.
The scientific test is the held-out species split above.

Training loss dropped from 4.578 to 0.902 over 80 epochs in the random-negative
run. In the taxonomy-hard run, loss dropped from 4.573 to 0.519 over 120
epochs, while validation loss rose from 4.573 to 5.133. In the taxonomy-soft
run, final held-out family/order improved, but restoring the lowest validation
loss checkpoint underperformed the final epoch. In the tree-distance run, broad
auto-scale targets were too diffuse, and scale-25 tree targets learned but did
not beat taxonomy-soft retrieval. Retrieval-aligned checkpointing selected
epoch 110 and produced the best family/order result so far. The first LoRA
adapter run trained cleanly and loss dropped, but held-out retrieval
underperformed the projection-only bridge; naive adapter tuning appears to
overfit or distort the cross-marker geometry under the current data/objective.
The first multi-positive run treated same-species alternatives as positives.
It improved species over the taxonomy-soft retrieval-best checkpoint but
reduced family/order, so it is a useful objective diagnostic rather than the
new best bridge.

## 12S/16S Probe

After the first COI-anchor probes, we built a bounded Actinopterygii
mitochondrial 16S reference and ran a first 12S query to 16S species-prototype
bridge. This is now the cleaner near-term eDNA marker expansion.

Reference:

- 1,865 16S species from a bounded NCBI nuccore fetch;
- 502 species overlapping the existing 12S multisource set;
- species-level split: 351 train, 75 validation, 76 held-out test species;
- run:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best/`.

Held-out 12S query to 16S species prototype retrieval, top-10:

| Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| Frozen NT cross-marker | 18.4% | 23.0% | 34.9% | 49.7% |
| MarkerMirror projection, taxonomy-soft retrieval-best checkpoint | 33.9% | 45.4% | 67.1% | 73.4% |

Held-out reverse 16S query to 12S species prototype retrieval, top-10:

| Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| Frozen NT cross-marker | 11.9% | 17.8% | 32.6% | 48.2% |
| MarkerMirror projection, taxonomy-soft retrieval-best checkpoint | 44.4% | 54.8% | 70.4% | 76.3% |

Seen/training-split diagnostic, same top-10 metric:

| Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| Frozen NT cross-marker | 13.5% | 19.2% | 32.6% | 50.3% |
| MarkerMirror projection, taxonomy-soft retrieval-best checkpoint | 100.0% | 100.0% | 100.0% | 100.0% |

Interpretation: both ribosomal directions beat frozen foundation embeddings,
and the reverse 16S->12S run is the strongest MarkerMirror result so far at
species/genus/family. This supports the scope decision: focus the eDNA marker
expansion on 12S/16S, and keep COI as an anchor/comparator rather than adding
every possible marker.

## Shared 12S/16S Species Space

The next prototype trained one shared projection head for both 12S and 16S,
rather than separate directional projection heads. This asks whether the two
ribosomal markers can live in one species-space.

Run:

- `nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best`;
- same 502 overlap species;
- split: 351 train, 75 validation, 76 held-out test species;
- checkpoint selected by combined validation genus/family/order top-10 across
  both 12S->16S and 16S->12S directions;
- best checkpoint: epoch 60, validation score 81.7221.

Held-out shared-space top-10:

| Direction / Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| 12S -> 16S, frozen NT | 15.2% | 19.2% | 31.1% | 51.3% |
| 12S -> 16S, shared MarkerMirror | 42.1% | 50.0% | 68.5% | 81.5% |
| 16S -> 12S, frozen NT | 7.0% | 14.7% | 30.1% | 50.4% |
| 16S -> 12S, shared MarkerMirror | 64.3% | 71.3% | 78.3% | 85.3% |

Interpretation: this is the strongest MarkerMirror result so far. The shared
12S/16S species-space improves over the separate directional bridges in both
directions. This is still candidate retrieval, not final calibrated species
assignment, but it is now strong enough to justify seed repeats and integration
with the rank/no-call pipeline.

Seed repeats:

| Direction | Species Mean | Species Range | Genus Mean | Genus Range | Family Mean | Family Range | Order Mean | Order Range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12S -> 16S | 43.4% | 42.1-45.4% | 50.7% | 49.0-53.0% | 68.1% | 65.6-70.2% | 77.6% | 72.9-81.5% |
| 16S -> 12S | 66.4% | 63.3-71.5% | 73.9% | 71.3-78.5% | 81.9% | 78.3-86.8% | 86.4% | 84.2-89.6% |

Interpretation: the shared 12S/16S result is stable across three seeds. The
16S->12S direction is consistently stronger, and the 12S->16S direction remains
useful. This is now the lead candidate-generation result.

## Tri-Marker 12S/16S/COI Shared Space

The tri-marker prototype tested the user's chain question without hard-coding a
fragile chain such as `12S -> 16S -> COI`. It trained one species-space across
12S, 16S, and COI, then evaluated every direction on held-out species.

Run:

- `nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best`

Pair overlaps total/train/validation/test:

- 12S/16S: 502 / 364 / 66 / 72;
- 12S/COI: 963 / 669 / 164 / 130;
- 16S/COI: 607 / 424 / 95 / 88.

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

Interpretation: the triad creates real learned cross-marker signal, because it
beats frozen embeddings in every tested direction. It is not the new lead
12S->COI result: direct 12S->COI MarkerMirror still has stronger held-out
top-10 at 8.5 / 18.8 / 55.4 / 75.4% for species/genus/family/order. The
current scientific read is that 16S is a strong complementary eDNA marker, but
forcing COI into the same simple shared head makes 12S->COI transfer harder.

## Candidate-Generation Export

The first per-query candidate table is now available for the shared 12S/16S
seed1903 model:

- script:
  `scripts/edna/export_marker_mirror_candidate_rankings.py`;
- candidate table:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/marker_mirror_candidate_rankings.csv.gz`;
- summary:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_rankings_shared_seed1903_summary.csv`;
- rows: 294,700 candidate rows.

This export ranks against the full target marker library. That is a harder and
more pipeline-relevant setting than the overlap-only aggregate metric used
during training.

Held-out learned full-reference candidate retrieval:

| Direction | k | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|---:|
| 12S -> 16S | 10 | 35.1% | 44.4% | 59.6% | 68.5% |
| 12S -> 16S | 50 | 52.0% | 60.6% | 80.1% | 87.8% |
| 16S -> 12S | 10 | 54.2% | 65.3% | 79.2% | 86.8% |
| 16S -> 12S | 50 | 74.3% | 80.6% | 90.3% | 93.8% |

The first MarkerMirror-only rank/no-call diagnostic is also complete:

- script:
  `scripts/edna/build_marker_mirror_candidate_rank_policy.py`;
- output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/rank_policy/`.

Simple validation-fitted top-1 score thresholds do not transfer cleanly. This
is useful because it tells us the next integration layer needs richer evidence:
top-k consensus, score margins, sequence similarity, tree/rank evidence, and
reference-gap diagnostics.

A first feature-based calibrator over top-k consensus and score-margin features
was also tested:

- script:
  `scripts/edna/train_marker_mirror_rank_calibrator.py`;
- output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/rank_calibrator/`.

It confirms the same boundary. Learned 12S->16S can be made very conservative
and precise, but coverage collapses. Learned 16S->12S gives more assignments,
but held-out precision is still below the target. The next version needs to
join MarkerMirror candidates with external evidence instead of relying only on
candidate-list geometry.

## Integrated Evidence Prototype

The next version joined MarkerMirror candidates with evidence that the final
pipeline can actually use:

- same-marker sequence checks;
- candidate reference availability;
- candidate-list score margins and taxonomic consensus;
- exact sequence ambiguity flags;
- tree distance to the top candidate.

Scripts:

- `scripts/edna/build_marker_mirror_evidence_join.py`;
- `scripts/edna/train_marker_mirror_integrated_rank_calibrator.py`.

Best held-out learned rows:

| Direction | Target | Calibrator | Coverage | Assigned Precision | False Species Calls | Species Precision |
|---|---:|---|---:|---:|---:|---:|
| 12S -> 16S | 0.99 | logistic | 55.0% | 99.4% | 0.0% | 100.0% |
| 16S -> 12S | 0.99 | HGB | 75.0% | 99.1% | 0.0% | 100.0% |

Seed-repeat handoff is also complete. Candidate exports, evidence joins, and
integrated rank calibrators were repeated for the original shared-space run
and seed 1902, then merged with seed 1903:

| Direction | Target | Seeds | Coverage Mean | Assigned Precision Mean | False Species Calls Mean | Species Precision Mean |
|---|---:|---:|---:|---:|---:|---:|
| 12S -> 16S | 0.99 | 3 | 51.0% | 98.9% | 0.11% | 99.8% |
| 16S -> 12S | 0.99 | 3 | 71.1% | 98.7% | 0.47% | 99.3% |

Source tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_summary.csv`;
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_best_target099.csv`;
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_target099_stability.csv`.

Interpretation: this is the first strong MarkerMirror pipeline result. It
does not mean a neural marker bridge alone solves species assignment. It means
the bridge becomes valuable when it is used as candidate generation and then
checked against sequence/reference/tree evidence before making a rank-calibrated
claim.

Caveat: current ambiguity support is exact-sequence ambiguity plus top-k
candidate consensus. The near-exact marker-resolvability layer remains a
planned hardening step.

## Resolvability Hardening

The first explicit 12S/16S resolvability layer is now implemented:

- script:
  `scripts/edna/build_marker_mirror_marker_resolvability.py`;
- compiler hook:
  `scripts/edna/build_marker_mirror_evidence_join.py --marker-resolvability-table`;
- source tables:
  `marker_mirror_marker_resolvability_summary.csv`,
  `marker_mirror_evidence_join_resolvability_seed1903_summary.csv`, and
  `marker_mirror_resolvability_calibrator_seed1903_best_target099.csv`.

At 0.99 proxy identity:

| Marker | Species-Resolvable | Genus-Or-Better | Family-Or-Better |
|---|---:|---:|---:|
| 12S | 91.8% | 98.2% | 99.8% |
| 16S | 92.0% | 98.6% | 99.8% |

Adding these features to the seed1903 integrated compiler did not change the
best target-0.99 held-out operating point. That is acceptable: the point of
this layer is to make marker ambiguity explicit and auditable before the
production handoff.

Caveat: the 0.99 rows are a rare-kmer prefix-identity proxy, not
alignment-backed clustering. Run VSEARCH/edlib before treating the near-exact
rows as final marker ceilings.

## Production Handoff Prototype

MarkerMirror now has a specimen-facing research candidate-generator script:

- `scripts/edna/run_marker_mirror_candidate_generator.py`.

It accepts FASTA/CSV input, embeds 12S or 16S query fragments with the frozen
Nucleotide Transformer backbone, projects them through the shared MarkerMirror
head, searches target-marker species prototypes, and writes top-k candidate
rows plus a manifest.

Smoke test:

- output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_smoke_12s_to_16s/`;
- input:
  one 12S query from `multisource/zero_shot_queries.csv`;
- target:
  25 16S species, top-5 candidates;
- result:
  candidate/summary/manifest files written successfully.

Full-reference GPU smoke:

- output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`;
- archived copy:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`;
- input:
  32 held-out 12S queries from `multisource/zero_shot_queries.csv`;
- target:
  full 16S reference, 1,865 species, top-50 candidates;
- result:
  1,600 candidate rows; known-target top-50 species/genus/family/order recovery
  25.0 / 59.4 / 78.1 / 84.4%.

Cache-backed repeat inference:

- cache:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/cache/marker_mirror_16s_nt_v2_50m_fullref_embeddings.npz`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_cache_smoke_summary.csv`;
- result:
  cache-write status `written`, cache-read status `loaded`, and identical
  candidate tables.

This is deliberately candidate-only. The next hardening step is connection to
the evidence compiler.

Evidence handoff:

- script:
  `scripts/edna/build_marker_mirror_candidate_generator_handoff.py`;
- output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_cache_read_12s_to_16s/evidence_handoff/`;
- source tables:
  `marker_mirror_candidate_generator_evidence_handoff_summary.csv`,
  `marker_mirror_candidate_generator_evidence_handoff_manifest_summary.csv`,
  and `marker_mirror_candidate_generator_evidence_handoff_feature_inventory.csv`;
- result:
  1,600 evidence rows, 97 production numeric features, tree evidence enabled,
  and marker-resolvability features joined.

This moves MarkerMirror from candidate-only output to evidence-compiler input.
The remaining handoff gap is applying a trained/serialized integrated
rank/no-call calibrator to these specimen-style evidence rows.

Full-query production-style handoff:

- output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_all_queries_12s_to_16s/`;
- source tables:
  `marker_mirror_candidate_generator_handoff_summary.csv`,
  `marker_mirror_candidate_generator_evidence_handoff_summary.csv`, and
  `marker_mirror_candidate_generator_rank_apply_summary.csv`;
- result:
  3,566 12S query rows, 178,300 top-50 candidate/evidence rows.

Known-target top-50 recovery over the full query table is 9.5 / 39.9 / 59.8 /
76.3% for species/genus/family/order. The rank/no-call apply path now runs, but
the nominal target-0.99 calibration does not transfer: species-enabled logistic
falls to 85.5% assigned precision, while species-disabled logistic gives 6.5%
coverage at 93.1% assigned precision and 0 false species calls. Keep
MarkerMirror as candidate/evidence generation until calibration transfer is
improved.

Calibration-transfer diagnosis:

- script:
  `scripts/edna/build_marker_mirror_calibration_transfer_diagnostics.py`;
- source tables:
  `marker_mirror_calibration_transfer_cohort_summary.csv`,
  `marker_mirror_calibration_transfer_handoff_strata.csv`,
  `marker_mirror_calibration_transfer_feature_drift.csv`, and
  `marker_mirror_calibration_transfer_top_feature_drift.csv`.

The weak transfer is now explained by the evaluation population, not just by
threshold noise. The controlled validation split has 100.0% query-species
coverage in the 16S target reference and top-50 recovery of 47.6 / 60.8 / 74.3
/ 81.1% for species/genus/family/order. The full production-style handoff has
only 26.6% query-species coverage in the 16S target reference and top-50
recovery of 9.5 / 39.9 / 59.8 / 76.3%. When the query species is present in the
16S target reference, top-50 species recovery is 35.8%; when absent, species
recovery is necessarily 0.0%, while genus/family/order remain useful. The next
method step should therefore be reference-aware calibration, not more blind
candidate generation.

Reference-aware policy repair diagnostic:

- script:
  `scripts/edna/build_marker_mirror_reference_aware_policy.py`;
- source table:
  `marker_mirror_reference_aware_policy_summary.csv`.

Production-safe gates over the existing species-disabled assignment row improve
precision by abstaining more aggressively. The baseline species-disabled
logistic row is 6.48% coverage at 93.07% assigned precision. A top-1
MarkerMirror score gate of 0.620484 gives 5.83% coverage at 95.67% assigned
precision; a stricter gate of 0.697663 gives 3.25% coverage at 100.00%
assigned precision. These are labelled-handoff diagnostics, not independent
production calibration, but they show the intended behavior: the system can use
candidate/evidence strength to abstain rather than force a weak taxonomic
claim.

Independent reference-aware validation:

- script:
  `scripts/edna/build_marker_mirror_reference_aware_policy_validation.py`;
- source tables:
  `marker_mirror_reference_aware_policy_validation_summary.csv` and
  `marker_mirror_reference_aware_policy_validation_per_split.csv`.

Across 50 repeated query-species splits, target-0.95 gates average 5.79%
held-out coverage at 94.39% precision and meet the target in 48% of repeats.
Target-0.99 gates average 4.13% held-out coverage at 98.27% precision and meet
target in 70% of repeats. Source-holdout validation is mixed: MitoHelper and
rCRUX behave reasonably, while the small Mare-MAGE subset is unstable. This
keeps MarkerMirror in the "promising reference-aware abstention" category, not
the "locked production assignment threshold" category.

Union candidate-support audit:

- script:
  `scripts/edna/build_marker_mirror_union_candidate_support.py`;
- source tables:
  `marker_mirror_union_candidate_support_summary.csv`,
  `marker_mirror_union_candidate_support_per_query.csv`, and
  `marker_mirror_same_marker_kmer_candidates_top50.csv.gz`.

This audit combines the production-style MarkerMirror 12S->16S top50 candidate
list with a same-marker 12S character-kmer top50 candidate list. The result is
clear: species support does not improve because the query species are absent
from the current 12S sequence reference and only 26.6% are present in the 16S
target reference. High-rank support improves sharply. MarkerMirror-only top50
support is 9.5 / 39.9 / 59.8 / 76.3% for species/genus/family/order; same-marker
12S k-mer support is 0.0 / 89.5 / 94.9 / 99.5%; the union list is 9.5 / 91.7 /
95.1 / 99.6%.

This shifts the practical MarkerMirror design: use cross-marker retrieval for
species opportunities when target-marker reference coverage exists, use
same-marker evidence for robust genus/family/order support, and let the
rank/no-call compiler decide the deepest defensible claim. The same-marker arm
in this first audit is k-mer only; the later BLASTN and VSEARCH sections below
supersede it for claim-facing same-marker alignment evidence.

Union rank/no-call diagnostic:

- script:
  `scripts/edna/build_marker_mirror_union_rank_policy.py`;
- output root:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_candidate_rank_policy/`;
- source tables:
  `marker_mirror_union_static_policy_summary.csv`,
  `marker_mirror_union_score_gate_validation_summary.csv`, and
  `marker_mirror_union_score_gate_validation_per_split.csv`.

This creates a production-style union candidate table with 355,231 rows and no
hidden labels, then evaluates labels only in separate diagnostics. The cleanest
static policy is top-1 source agreement at family/order only: 25.2% coverage,
98.4% assigned precision, and no species calls. Same-marker score gates can
cover far more queries, but species-split validation shows they are not locked:
family target-0.95 averages 92.7% coverage at 94.8% precision, and order
target-0.99 averages 68.5% coverage at 99.0% precision. The next step is a
calibrated evidence compiler over the union features, not more blind
candidate-generation training.

Learned union evidence compiler:

- script:
  `scripts/edna/train_marker_mirror_union_evidence_compiler.py`;
- source tables:
  `marker_mirror_union_evidence_compiler_summary.csv`,
  `marker_mirror_union_evidence_compiler_family_order_summary.csv`,
  `marker_mirror_union_evidence_compiler_order_summary.csv`, and
  `marker_mirror_union_evidence_compiler_features.csv`.

The first HGB compiler uses 102 production-available top-1 union features with
train/calibration/evaluation species splits. It is not yet the answer. Mixed
family/order target-0.99 averages 72.2% coverage at 97.9% precision and meets
target in 20% of repeats. Order-only target-0.99 averages 67.4% coverage at
98.5% precision and meets target in 44% of repeats. This is worse than the
simpler same-marker order score gate and less clean than family/order source
agreement. The next model should use richer candidate-level/list-level evidence
or a proper hierarchical selective/conformal frame, not just another top-1 HGB.

Reason-code and curation-priority layer:

- script:
  `scripts/edna/build_marker_mirror_union_reason_codes.py`;
- source tables:
  `marker_mirror_union_reason_code_summary.csv`,
  `marker_mirror_union_reason_code_by_source.csv`,
  `marker_mirror_union_reason_code_per_query.csv`, and
  `marker_mirror_union_reference_curation_priorities.csv`.

This turns the union pipeline into an explanation layer. The largest bucket is
not species-level success or total failure: 2,249/3,566 queries have genus-level
union support but no justified species call. Conservative source agreement
emits 621 family calls at 98.1% precision and 277 order calls at 99.3%
precision. The curation table separates "add marker reference for this species"
cases from "the species exists in the target marker but MarkerMirror failed to
retrieve it" cases. That distinction is important: the former is reference
curation, the latter is model/candidate-generator improvement.

Caveat: current 12S reference-gap labels are tied to the current full-query
zero-shot/reference setup. Treat them as current-table diagnostics, not final
claims about all possible 12S reference databases.

Same-marker edlib validation:

- script:
  `scripts/edna/build_marker_mirror_same_marker_edlib_validation.py`;
- source tables:
  `marker_mirror_same_marker_edlib_support_summary.csv`,
  `marker_mirror_same_marker_edlib_support_per_query.csv`,
  `marker_mirror_same_marker_edlib_candidates_top50.csv.gz`, and
  `marker_mirror_same_marker_edlib_validation_manifest.json`.

The edlib run scored 176,931 existing same-marker candidate rows on Vast. It
does not perform all-vs-all search; it validates and reranks the k-mer top50
pool. The result is reassuring: top10 edlib-reranked support is 0.0 / 87.8 /
94.3 / 98.8% for species/genus/family/order, matching or slightly improving
the original k-mer top10 high-rank support. This reduces the risk that the
same-marker signal is merely a k-mer artifact, but full BLAST/VSEARCH-style
candidate generation remains a separate comparator/hardening step.

List-level selective compiler:

- script:
  `scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py`;
- source tables:
  `marker_mirror_union_listwise_selective_family_order_summary.csv`,
  `marker_mirror_union_listwise_selective_order_summary.csv`, and matching
  per-split/threshold/feature tables.

This is the second learned union compiler. It uses list-level evidence rather
than only top-1 features. It improves the high-coverage order diagnostic:
order-only target-0.99 averages 83.1% coverage at 98.8% precision, compared
with 67.4% coverage at 98.5% for the first top-1 HGB. It still is not the final
answer because target-0.99 is met in only 56% of species-split repeats. Treat
it as evidence that list-level modeling helps, but keep source agreement plus
reason codes as the current production-safe layer.

VSEARCH same-marker candidate generation:

- scripts:
  `scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py` and
  `scripts/edna/build_marker_mirror_union_vsearch_candidate_support.py`;
- source tables:
  `marker_mirror_same_marker_vsearch_support_summary.csv`,
  `marker_mirror_same_marker_vsearch_candidates_top50.csv.gz`, and
  `marker_mirror_union_vsearch_candidate_support_summary.csv`.

This is the stronger same-marker classical arm. VSEARCH global alignment over
3,566 query sequences and 12,593 current 12S reference sequences gives
same-marker top50 support of 0.0 / 90.4 / 94.9 / 99.4% for
species/genus/family/order. The MarkerMirror + VSEARCH union top50 support is
9.5 / 91.8 / 95.1 / 99.6%. This makes the high-rank union result more
defensible than the original k-mer audit. Species remains blocked by current
reference design/coverage, not by the lack of a candidate generator.

BLASTN same-marker candidate generation:

- scripts:
  `scripts/edna/build_marker_mirror_same_marker_blast_candidates.py` and
  `scripts/edna/build_marker_mirror_union_blast_candidate_support.py`;
- source tables:
  `marker_mirror_same_marker_blast_support_summary.csv`,
  `marker_mirror_same_marker_blast_candidates_top50.csv.gz`, and
  `marker_mirror_union_blast_candidate_support_summary.csv`.

BLASTN local alignment over the same 3,566 query sequences and 12,593 current
12S reference sequences gives same-marker top50 support of 0.0 / 90.7 / 95.1 /
99.4% for species/genus/family/order. The MarkerMirror + BLASTN union top50
support is 9.5 / 92.1 / 95.3 / 99.7%. This is the strictest classical
same-marker arm we have for claim-facing candidate generation. It confirms that
the high-rank union result is not a k-mer artifact and not specific to VSEARCH
global alignment.

BLAST/VSEARCH calibration-transfer repair:

- script:
  `scripts/edna/build_marker_mirror_blast_vsearch_calibration_repair.py`;
- source table:
  `marker_mirror_blast_vsearch_calibration_repair_summary.csv`.

The conservative repair is all-source top1 order agreement. Across 50
query-species calibration/evaluation repeats, it averages 24.8% coverage at
99.6% precision and meets target-0.99 in 100% of repeats. Higher-coverage order
diagnostics are close but not locked: BLAST top10 source-stratified Wilson95
averages 69.0% coverage at 99.4% precision and meets target-0.99 in 82% of
repeats. This means the production-safe MarkerMirror layer can emit conservative
order calls, while the broader high-coverage rank/no-call compiler still needs
better calibration transfer.

The stricter Exp 117 nested repair supersedes the raw high-coverage Exp 111
rows for order-level claims. BLASTN/VSEARCH top-10 order agreement with nested
global Wilson95 locking reaches 57.2% mean held-out coverage at 99.8% precision
and meets target-0.99 in all 50 outer repeats.

Exp 112 turns that stable diagnostic into an assignment/reason-code table:
`marker_mirror_stable_order_policy_assignments.csv`. The conservative
max-repeat threshold assigns 880/3,566 full-query 12S rows, 24.7% coverage, at
99.7% precision with 0 false species calls. This is currently the strongest
production-style MarkerMirror behavior: conservative order/no-call, not
species-level identification.

The same script now writes
`marker_mirror_stable_order_policy_production_assignments.csv`, which strips
truth and correctness columns. That is the handoff payload for a future
arbitrary-FASTA MarkerMirror wrapper after MarkerMirror, BLASTN, and VSEARCH
candidate features have been generated.

The first arbitrary-input 12S wrapper now exists:
`scripts/edna/run_marker_mirror_12s_production_v1.py`. The dry-run smoke wrote
a normalized query table, dependency report, run plan, next-action table, and
manifest under `marker_mirror_12s_production_v1/dry_run_smoke/`. Local BLASTN
smoke succeeded on 2 normalized queries and wrote 100 candidate rows. The full
all-source order/no-call wrapper is dependency-gated locally because VSEARCH is
not installed.

The full wrapper has now run on Vast for all 3,566 current 12S queries:
`results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/`.
It completed MarkerMirror, BLASTN, VSEARCH, feature-table construction, and the
stable order/no-call policy end to end. Final output is 880 order calls and
2,686 no-calls, with 99.7% diagnostic precision and 0 false species calls on
the labelled benchmark table.

An unlabeled FASTA smoke has also completed:
`results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/`.
It emits 1 order call and 1 no-call over 2 FASTA records with taxonomy stripped,
and correctly leaves precision/correctness blank because no truth labels were
available. Usage is summarized in
`experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md`.

Exp 117 improves the high-coverage order/no-call path. A nested species-split
repair over BLASTN/VSEARCH top-10 order agreement reaches 57.2% mean held-out
coverage at 99.8% mean precision and meets target-0.99 in all 50 outer repeats.
The locked full-table diagnostic assigns 2,513/3,566 rows, 70.5% coverage, at
99.8% labelled precision. This is the strongest high-coverage order result so
far, but it remains a diagnostic/research mode rather than the default
conservative behavior.

Exp 118 exposes that policy as `--decision-mode high_coverage_order` in
`scripts/edna/run_marker_mirror_12s_production_v1.py`. Vast smokes passed for
labelled and unlabeled inputs in both `stable_order` and `high_coverage_order`
modes. The default remains conservative `stable_order`.

Exp 119 tested the analogous repair at family and genus using all available
BLAST/VSEARCH/MarkerMirror policy rows. Neither rank transferred cleanly:
family met target-0.99 in 94% of repeats and genus in 98%. These are useful
diagnostics, but not enabled decisions.

Exp 121 tested a different path: set-valued family/genus outputs from the full
candidate lists. This also failed to unlock family/genus. Family reaches only
95.4% full-query set coverage with a mean set size of 34.4 families; genus
reaches 92.4% with a mean set size of 79.6 genera. The limitation is therefore
evidence-level, not just single-label thresholding.

Exp 128 tested lineage/reference-coverage features from the next-evidence
audit as a new policy input. This was useful but still not enough: target-0.99
family averaged 87.4% coverage at 98.0% precision and met target in only 10%
of species-split repeats; target-0.99 genus averaged 17.7% coverage at 97.6%
precision and met target in 42% of repeats. This keeps family/genus disabled
and moves the next serious attempt toward alignment-backed marker-resolvability
or sample-aware ecological/curation evidence.

Exp 129 completed that alignment-backed marker-resolvability hardening step
with VSEARCH clustering. At 0.99 identity, 12S query oracle support is
77.9 / 95.2 / 99.6 / 99.7% for species/genus/family/order, but only 19.6% of
query clusters contain a current reference. At 0.95 identity, species support
falls to 38.3% while order remains 98.2%. This is strong evidence that 12S has
a high-rank marker ceiling and reference-coverage bottleneck; it still does
not enable family/genus without a validated policy.

Exp 130 tested whether the production-available part of that VSEARCH
resolvability evidence can improve the learned policy. Hidden oracle-support
columns were excluded. The answer is still no for family/genus: target-0.99
family averages 57.3% coverage at 95.5% precision and target met in 44% of
repeats; target-0.99 genus averages 11.5% coverage at 87.8% precision and
target met in 38% of repeats.

Exp 131 turns the failure analysis into an active reference-curation layer.
It ranks 795 species groups and 698 lineage rows by value-of-information using
reason codes, BLASTN/VSEARCH/MarkerMirror policy outputs, production-available
VSEARCH cluster evidence, and labelled VSEARCH oracle diagnostics for curation
triage only. The largest action category is adding both 12S and 16S species
references for 532 species groups covering 1,928 queries. `Trichiurus_lepturus`
is the top model/target-curation failure because 16S is present but
MarkerMirror retrieval is weak. This is not an enabled family/genus policy; it
tells us what evidence must change before another rank-lift attempt is worth
running.

## Interpretation

This is a positive signal:

- frozen foundation embeddings do not naturally align COI and 12S well;
- a small supervised bridge creates measurable held-out cross-marker retrieval;
- the gain is strongest at family/order, matching the biological reality that
  12S carries stronger high-rank than species-rank information.

This is not enough yet:

- train retrieval is much higher than held-out retrieval, so generalization is
  the bottleneck;
- species-level held-out retrieval is still low;
- hard negatives help species/genus retrieval;
- taxonomy-soft rank targets help family/order retrieval;
- best-validation restore is not reliable because validation loss is not yet
  aligned with downstream retrieval quality;
- retrieval-aligned checkpointing is useful and should be standard for this
  model family;
- actual tree-distance targets are not automatically better than rank labels;
- naive LoRA/backbone adaptation is not automatically better than projection
  heads and should not be escalated without a better paired-data or
  regularized objective;
- multi-positive same-species targets can help species-level transfer but
  currently weaken high-rank retrieval;
- the current bridge still has no reference-gap calibration or eDNA posterior
  integration;
- high-coverage order repair now transfers in repeated species splits, but it
  does not solve genus/species identification.
- lineage/reference coverage alone does not solve family/genus transfer.
- VSEARCH-backed marker-resolvability strengthens the marker-ceiling evidence,
  but does not by itself solve family/genus transfer.
- VSEARCH-resolvability-aware learned policies still do not solve family/genus
  transfer.
- Active reference-curation is now source-tabled, so the system can point to
  concrete reference/evidence improvements instead of only abstaining.

## Next Experiments

1. Retrieval-aligned MarkerMirror:
   evaluate validation retrieval every N epochs and select by genus/family/order
   top-k, not by contrastive loss.
2. Hybrid MarkerMirror:
   combine taxonomy-soft rank targets with a small tree-distance regularizer,
   rather than replacing rank targets entirely.
3. Multi-positive MarkerMirror:
   support many COI and many 12S sequences per species without collapsing them
   into one prototype too early.
4. LoRA/backbone fine-tune:
   first gated run is complete and negative. Do not scale adapter training
   until the objective is improved, e.g. multi-positive species batches,
   frozen-teacher regularization, or candidate-posterior supervision.
5. Pipeline integration:
   feed MarkerMirror candidates into the existing rank/no-call evidence
   compiler and compare against 12S sequence-only, BLAST/VSEARCH, and Eco-Phylo
   posterior candidates.
6. Shared 12S/16S species space:
   first run is complete and currently strongest. Next: seed repeats and
   candidate-reranker/rank-calibration integration.
7. Family/genus evidence:
   do not rerun threshold-only, set-only, lineage-coverage-only, or
   resolvability-only repair. The next enabling attempt needs a new information
   source, such as sample-aware ecology/co-occurrence. Exp 131 now covers the
   active reference-curation audit, so future family/genus work should either
   add those references or introduce sample-aware context rather than reranking
   the same evidence again.

## Breakthrough Target

The strong target is not "12S species classification."

The strong target is:

> A marker-bridging model that turns ambiguous eDNA fragments into COI/tree
> candidate evidence, then returns the deepest defensible taxonomic rank with
> uncertainty and reason codes.
