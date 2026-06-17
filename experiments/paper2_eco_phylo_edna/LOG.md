# Former Paper 2 / Merged Paper 1 eDNA Work Package Log

## 2026-06-02 Eco-Phylo Posterior Prep

Added and ran:

- `scripts/edna/build_eco_phylo_posterior_ablation.py`
- `scripts/edna/run_eco_phylo_posterior.py`

New outputs:

- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_query_features.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_query_features_sample.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_method_design.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_rank_correctness_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_manifest.json`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_selected_predictions.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_model_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_method_selection_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/eco_phylo_posterior_model_manifest.json`

What changed:

- The scattered Global_eDNA prediction outputs are now consolidated into a
  posterior-ready feature table.
- The full query-method table has 1,049,382 rows across 18 existing SSM/CNN
  sequence-only and learned co-occurrence methods.
- Each row carries sequence/model score, top candidate labels, candidate
  taxonomy/reference evidence, deterministic `site20` calibration/evaluation
  split, 99% near-exact 12S marker-resolvability ceiling, ecological arm labels,
  and species/genus/family/order top-1 correctness.
- Method-design and rank-summary tables make explicit which posterior terms are
  available for each method/rank and which site-heldout calibration thresholds
  currently transfer.
- The first posterior scorer trains species/genus/family/order probability
  models on calibration `site20` groups, chooses the best existing evidence arm
  per query/rank, and evaluates thresholded rank/no-call calls on held-out
  `site20` groups.

First-pass held-out posterior result:

- species: no reliable threshold;
- genus: tiny calibration-site thresholds fail to transfer;
- family target-60 calibration point: 6.9% held-out assignment, 48.6% family
  accuracy;
- order target-60 calibration point: 10.1% held-out assignment, 60.4% order
  accuracy;
- mixed rank-backoff target-60 point: 14.6% held-out assignment, 51.1% assigned
  rank accuracy.

Interpretation:

- The first posterior is implemented and auditable, but it is not yet a positive
  result.
- It does not beat the current strongest single-method calibration: SSM +
  RLS/OBIS learned co-occurrence at weight 0.25, which assigns 8.9% of held-out
  rows with 59.6% family accuracy and 74.1% order accuracy.
- The next version should be candidate-level: expose per-candidate sequence
  score, BLAST/p-distance evidence, tree distance, marker-resolvability group,
  geography/range prior, and co-occurrence weight before ranking. The current
  top-1 method-arm posterior can only choose among already-compressed outputs.

## 2026-06-02 Candidate-Level Eco-Phylo Posterior

Added and ran:

- `scripts/edna/build_eco_phylo_candidate_posterior_inputs.py`
- `scripts/edna/run_eco_phylo_candidate_posterior.py`

New outputs:

- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_features_top5.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_features_top5_sample.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_method_inventory.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_feature_manifest.json`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_selected_predictions.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_model_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_method_selection_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_model_manifest.json`

What changed:

- The posterior input is now candidate-level instead of top-1 method-level.
- The full top-5 candidate table has 6,995,880 rows across 24 evidence arms.
- Candidate rows include model score, sequence-only score where available,
  BLAST pident/rank where available, RLS/OBIS counts, reference/taxonomy
  metadata, marker-resolvability flags, and species/genus/family/order
  correctness.
- Local scoring was first run on 10,000 complete sample/query groups as a
  disk-safe prototype.
- The full scorer was then run on Vast over all 6,995,880 candidate rows and
  copied back locally.

Full held-out result:

- species: target-50 gives 11.3% assignment at 48.3% accuracy; target-60 gives
  1.9% assignment at 57.1%; no reliable 70%+ threshold;
- genus: target-80 gives 3.9% assignment at 80.5%; no reliable 90%+ threshold;
- family: target-90 gives 28.9% assignment at 86.1%; target-95 gives 16.0%
  assignment at 95.2%;
- order: target-90 gives 22.7% assignment at 90.1%; target-95 gives 4.9%
  assignment at 95.1%;
- mixed rank-backoff: target-90 gives 31.8% assignment at 86.6%; target-95 gives
  16.4% assignment at 95.3%.

Interpretation:

- Candidate-level fusion is the correct direction and is more informative than
  the top-1 method-arm posterior.
- This is now a useful full-table family/order/rank-backoff result, but it is
  not a species-level eDNA solution.
- Calibration still drifts for some ranks, and the current table lacks explicit
  candidate tree-distance and 12S p-distance/alignment features.
- Next steps: add candidate tree-distance/p-distance evidence and use a
  stronger calibration design.

## 2026-06-02 Candidate 12S Sequence Evidence

Added and ran:

- `scripts/edna/build_eco_phylo_candidate_sequence_evidence.py`

Updated:

- `scripts/edna/run_eco_phylo_candidate_posterior.py`

New outputs:

- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_top5.csv.gz`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_summary.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_manifest.json`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_rank_backoff_summary.csv`

What changed:

- Built direct train-reference 12S evidence for 584,894 unique
  query/candidate pairs.
- Evidence is available for 145,785 pairs; 439,109 pairs lack a train-reference
  12S sequence for the candidate species.
- Distance is best ungapped sliding p-distance/identity against
  `train_species_sequences.json`; it is not BLAST and not a full gapped
  alignment.
- The posterior scorer can now join this evidence with `--sequence-evidence`.

Sampled 10,000 complete-query diagnostic:

- genus target-80: 39.4% held-out assignment at 83.7% accuracy;
- genus target-95: 6.4% held-out assignment at 98.6% accuracy;
- order target-90: 46.0% held-out assignment at 88.9% accuracy;
- order target-95: 37.3% held-out assignment at 93.6% accuracy;
- mixed rank-backoff target-95: 38.9% held-out assignment at 93.6% accuracy.

Interpretation:

- Direct sequence-distance evidence appears useful, especially for genus/order
  coverage, but this is only a sampled diagnostic.
- The full sequence-aware scorer still needs to run on Vast. The current Vast
  endpoint refused SSH during this step.

## 2026-06-02 Candidate Tree-Neighborhood Evidence

Added:

- `scripts/edna/build_eco_phylo_candidate_tree_evidence.py`

Updated:

- `scripts/edna/run_eco_phylo_candidate_posterior.py` now accepts
  `--tree-evidence`.

The tree-neighborhood evidence is inference-safe. It does not use the true
query species as a feature. It summarizes the retrieved candidate set:

- candidate distance to the top-ranked candidate on the fish tree;
- candidate distance to the candidate neighborhood;
- genus/family/order agreement with the top candidate;
- top-k taxonomic and tree-distance spread.

Local sampled build:

- input: 10,000 complete sample/query groups;
- candidate rows: 1,200,000;
- method/sample/query groups: 240,000;
- output:
  `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level/eco_phylo_candidate_tree_neighborhood_evidence_sample10k.csv`.

Local sampled sequence+tree posterior:

- output:
  `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_sample/`;
- genus target-90: 19.7% assignment at 97.0% held-out accuracy;
- genus target-95: 11.9% assignment at 98.3%;
- family target-90: 42.2% assignment at 92.9%;
- family target-95: 30.2% assignment at 97.3%;
- order target-90: 8.5% assignment at 99.1%;
- order target-95: 7.3% assignment at 99.5%.

Caveat:

- species thresholds overfit the calibration side and fail on held-out groups;
- naive species-first rank-backoff is therefore not acceptable yet;
- for the eDNA pipeline, species must be disabled unless a nested
  fit/threshold/evaluation calibration proves that it transfers.

Full Vast sequence+tree posterior:

- full tree-neighborhood evidence completed on Vast:
  6,995,880 candidate rows, 175,912 cached tree-distance pairs;
- full posterior output:
  `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/`;
- copied remote archive:
  `results/remote_runs/2026-06-02/rtx_pro_6000/eco_phylo_candidate_level_sequence_tree_evidence_full/`.

Full held-out operating points:

- species target-50: 48.5% assignment at 45.7% accuracy;
- species target-60: 37.6% assignment at 51.5%;
- genus target-90: 19.8% assignment at 97.5%;
- genus target-95: 15.2% assignment at 98.6%;
- family target-90: 47.3% assignment at 85.6%;
- family target-95: 35.2% assignment at 95.1%;
- order target-90: 42.0% assignment at 92.7%;
- order target-95: 33.1% assignment at 97.0%;
- naive species-first mixed rank-backoff target-95: 42.9% assignment at 80.4%.

Interpretation:

- full sequence+tree fusion strengthens the higher-rank eDNA evidence;
- species thresholds still fail transfer and should be disabled or separately
  calibrated for rank-backoff claims;
- next concrete step is a species-disabled rank-backoff summary.

Species-disabled rank-backoff:

- added `scripts/edna/build_species_disabled_edna_rank_backoff.py`;
- output:
  `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`;
- rank order: genus -> family -> order;
- species is intentionally disabled because species thresholds do not transfer.

Held-out operating points:

- target-90: 51.7% assignment at 84.3% accuracy;
- target-95: 40.3% assignment at 94.3% accuracy;
- target-95 composition: 4,852 genus calls, 7,094 family calls, 864 order calls.

Interpretation:

- this is the cleanest current Eco-Phylo posterior operating point;
- it supports defensible higher-rank eDNA assignment rather than forced species
  labels.

## 2026-06-02 Evidence-Decomposition Consolidation

Added and ran:

- `scripts/edna/build_edna_evidence_decomposition.py`

New source tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/edna_evidence_decomposition_matrix.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/edna_evidence_best_by_rank.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/edna_rank_no_call_operating_points.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/edna_evidence_decomposition_manifest.json`

What changed:

- Global_eDNA evidence arms are now in one manuscript-facing matrix instead of
  scattered metric files.
- The table separates classical BLAST sequence-only, neural sequence/tree-only,
  geography-only, same-sample co-occurrence-only, sequence + geography,
  sequence + same-sample co-occurrence, learned RLS/OBIS co-occurrence, and
  learned public FISHGLOB co-occurrence.
- The pipeline ledgers now include these eDNA tables through
  `pipeline_component_status.csv`, `pipeline_next_actions.csv`, and
  `pipeline_end_to_end_summary.csv`.

Current best Global_eDNA ASV top-10 rows by broad evidence type:

- BLAST sequence-only remains strongest on forced species/family/order top-k:
  species 34.0%, genus 51.4%, family 65.1%, order 62.5%.
- SSM sequence/tree-only is weaker than BLAST for forced species calls but
  gives usable higher-rank signal: species 11.4%, genus 27.9%, family 49.8%,
  order 54.8%.
- Sequence + geography/co-occurrence gives some rank-specific improvements over
  SSM sequence-only, especially genus/order in selected arms, but does not beat
  BLAST sequence-only on this Global_eDNA forced top-k table.
- Pure geography and pure co-occurrence are now explicit negative/context
  controls rather than hidden implementation details.

Rank/no-call status:

- `edna_rank_no_call_operating_points.csv` is diagnostic only. It uses current
  Global_eDNA score-threshold curves, not an independent calibration split.
- At the current diagnostic thresholds, only a small family/order operating
  point reaches 50% rank accuracy, and no 70/80/90/95% operating point is
  available. This is a useful warning: real 12S/eDNA rank/no-call claims need
  a stronger calibrated posterior or an explicit held-out calibration design.
- Added site-heldout threshold transfer with
  `scripts/edna/build_global_edna_independent_rank_calibration.py`. Outputs:
  `results/paper1_phylo_calibrated_assignment/global_edna_independent_rank_calibration/`.
  Thresholds are learned on deterministic `site20` calibration groups and
  evaluated on held-out `site20` groups.
- Site-heldout result: SSM + RLS/OBIS learned co-occurrence at weight 0.25 has
  the best 50%-target operating point, assigning 8.9% of held-out rows with
  59.6% family accuracy and 74.1% order accuracy. No method reaches a 70%+
  calibration target for species/genus/family/order under this top-1 score
  threshold policy.

Interpretation:

- The eDNA work now supports the paper's reliability argument better than an
  architecture leaderboard: BLAST is still a strong forced assignment baseline,
  12S species calls remain hard, ecology is helpful but fragile, and rank/no-call
  needs a stronger posterior before it becomes a positive manuscript claim.

## 2026-05-31 Merge Status

This workstream is now part of the merged Paper 1 manuscript. Keep this folder
as the implementation/audit trail for 12S/eDNA resolvability, Global_eDNA
validation, and Eco-Phylo posterior experiments. Do not frame it as an
independent Paper 2 unless the eDNA work later grows beyond the merged
uncertainty-aware inference paper.

## 2026-05-30

Decision retained inside merged Paper 1: the 12S/eDNA work should answer how
sequence models, tree-space candidates, and ecological priors combine under
real eDNA ambiguity.

Claim boundary:

- We should not claim that SSMs solve species-level 12S assignment by sequence
  alone.
- The stronger claim is that a sequence encoder is useful inside a
  phylogeny-aware, ecology-aware, calibrated eDNA system.
- The encoder should be swappable: SSM/Mamba, CNN, LSTM, Transformer, S5,
  k-mer, BLAST, VSEARCH, and later pretrained DNA encoders.

Current state:

- TAXDNA-style SSM/CNN experiments have completed for exact-Teleo and broad
  multisource 12S.
- Global_eDNA validation exists.
- Learned co-occurrence reranking exists for RLS/OBIS and public FISHGLOB.
- Official Stalder/TAXDNA exact retraining remains blocked by unavailable
  processed sequence/co-occurrence JSONs.

Results so far:

- Broad multisource 12S top10 species/genus/family/order:
  - SSM contrastive: 0.31 / 25.55 / 45.79 / 60.18
  - CNN: 0.17 / 18.62 / 36.37 / 53.20
- Exact-Teleo top10 species/genus/family/order:
  - SSM contrastive seed1206: 0.14 / 34.64 / 60.89 / 70.67
  - SSM contrastive seed1207: 0.28 / 32.68 / 64.11 / 77.51
  - SSM contrastive seed1208: 1.12 / 31.56 / 61.87 / 74.72
  - CNN seed1206: 0.42 / 18.30 / 37.71 / 66.06
  - CNN seed1207: 0.14 / 22.49 / 40.78 / 64.53
  - CNN seed1208: 0.14 / 39.66 / 62.15 / 82.26
- Global_eDNA ASV top10 species/genus/family/order:
  - SSM sequence-only: 4.28 / 19.28 / 34.77 / 42.95
  - CNN sequence-only: 0.02 / 0.36 / 6.14 / 8.51
  - SSM + RLS/OBIS learned co-occurrence w0.25:
    5.88 / 22.24 / 37.62 / 45.02
  - SSM + public FISHGLOB learned co-occurrence w0.25:
    2.83 / 12.23 / 25.71 / 39.56

What this answers:

- Sequence-only 12S species-level open-candidate assignment remains very hard.
- Higher-rank 12S/eDNA assignment is more meaningful and more stable.
- Ecological context can improve some real Global_eDNA metrics, but it must be
  ablated carefully because it can also bias or degrade predictions depending
  on the source.

What actually solves species-level 12S, from first principles:

- unique 12S sequence signal when it exists;
- longer sequence or another marker when 12S is not resolvable;
- geography/range when sequence alternatives are geographically impossible;
- co-occurrence/community context when one candidate is ecologically more
  plausible;
- explicit no-call or higher-rank assignment when evidence is insufficient.

How this is unique/useful:

- It reframes eDNA assignment as evidence integration rather than sequence
  classification alone.
- It can show when a species-level claim is supported, when only genus/family/
  order is supported, and when no-call is the honest output.
- It creates a transparent, encoder-agnostic alternative to black-box ecological
  reranking.

Remarkable experiments to run:

1. 12S resolvability/oracle upper-bound:
   - cluster exact/near-exact 12S sequences;
   - estimate whether species/genus/family/order is theoretically resolvable;
   - report the information ceiling for sequence-only assignment.
2. Sequence/tree/ecology decomposition:
   - sequence only;
   - tree only;
   - geography only;
   - co-occurrence only;
   - sequence + tree;
   - sequence + geography;
   - sequence + co-occurrence;
   - sequence + tree + ecology.
3. Rank-adaptive assignment:
   - compare SSM, CNN, LSTM, Transformer, S5, BLAST, VSEARCH, and k-mer.
4. Reference-gap/active curation map:
   - identify which missing references would most reduce uncertainty.

These experiments are tracked in:

- `experiments/paper2_eco_phylo_edna/STRONG_EXPERIMENTS.md`

Timing:

- 12S resolvability/oracle upper-bound can start immediately without GPU.
- Rank-adaptive no-call curves can start as soon as comparable prediction files
  are available from Paper 1 and the 12S/eDNA runs.
- Eco-Phylo posterior ablation is the next merged Paper 1 eDNA protocol step
  after the current Paper 1 validation queue is copied back.
- Multi-marker shared tree space should start as data inventory first, then GPU
  training later.
- Reference-gap curation map should follow the resolvability map and
  rank-adaptive calibration.

## 2026-05-30 Exact 12S Resolvability First Pass

Added:

- `scripts/edna/build_12s_resolvability_map.py`
- `experiments/paper2_eco_phylo_edna/runs/01_local_12s_resolvability.sh`

Outputs:

- `results/edna/resolvability/`
- overview table:
  `results/edna/resolvability/resolvability_overview.csv`

First-pass exact-identity results:

- multisource:
  - species with sequences: 1,637
  - species best-rank species-resolvable: 100.0%
  - zero-shot query species oracle support after including held-out query
    sequences as observed marker evidence: 100.0%
  - reference exact-match rate for zero-shot queries: 0.0%
- multisource Teleo:
  - species with sequences: 949
  - species best-rank species-resolvable: 92.5%
  - zero-shot query species oracle support: 76.3%
  - genus/family/order oracle support: 95.9% / 99.7% / 100.0%
  - reference exact-match rate for zero-shot queries: 22.1%
- rCRUX cleaned:
  - species with sequences: 163
  - species best-rank species-resolvable: 100.0%
  - zero-shot query species oracle support: 100.0%
  - reference exact-match rate for zero-shot queries: 0.0%
- Mitohelper full Teleo:
  - species with sequences: 717
  - species best-rank species-resolvable: 92.3%
  - zero-shot query species oracle support: 78.4%
  - genus/family/order oracle support: 94.4% / 99.6% / 99.6%
  - reference exact-match rate for zero-shot queries: 18.1%

Interpretation:

- Exact identity already shows a real information ceiling in the Teleo-style
  datasets: roughly 21-24% of held-out query species are not species-resolvable
  by exact 12S sequence alone.
- The broad multisource and rCRUX exact-identity results are likely too
  permissive because sequences may not be normalized to a single primer/window
  and exact full-fragment identity can make nearly every record unique.
- Therefore, the paper-grade version needs a near-exact and amplicon-aware pass
  before final claims.

Next resolvability step:

1. Add VSEARCH/BLAST near-exact clustering at identity thresholds such as
   99%, 98%, 97%, and 95%.
2. Where possible, restrict or normalize to comparable Teleo/MiFish windows.
3. Compare exact and near-exact oracle ceilings.

Near-exact preparation:

- Added `scripts/edna/build_12s_near_exact_resolvability.py`.
- Added Vast runner:
  `experiments/paper2_eco_phylo_edna/runs/02_vast_near_exact_resolvability.sh`.
- Copied both to the Vast host used for the 2026-05-30 runs.

Near-exact launch:

- After the Paper 1 baseline/control queue completed and was copied locally,
  the near-exact 12S resolvability runner was launched on the Vast RTX PRO 6000
  host.
- Remote PID: 80835.
- Remote output root:
  `results/edna/resolvability_near_exact/`.
- Datasets:
  - multisource,
  - multisource_teleo,
  - rCRUX cleaned,
  - Mitohelper full Teleo.
- Identity thresholds:
  - 99%,
  - 98%,
  - 97%,
  - 95%.

Near-exact completion:

- Completed on the Vast RTX PRO 6000 host and copied locally to:
  `results/remote_runs/2026-05-30/rtx_pro_6000/resolvability_near_exact/`.

Near-exact query oracle support at 99% identity:

- multisource species/genus/family/order:
  77.9 / 95.2 / 99.6 / 99.7
- multisource Teleo species/genus/family/order:
  70.7 / 90.9 / 97.3 / 100.0
- rCRUX cleaned species/genus/family/order:
  95.4 / 100.0 / 100.0 / 100.0
- Mitohelper full Teleo species/genus/family/order:
  70.8 / 93.4 / 97.5 / 99.6

Near-exact query oracle support at 95% identity:

- multisource species/genus/family/order:
  38.3 / 73.8 / 92.0 / 98.2
- multisource Teleo species/genus/family/order:
  42.7 / 71.2 / 89.4 / 99.6
- rCRUX cleaned species/genus/family/order:
  54.8 / 98.2 / 100.0 / 100.0
- Mitohelper full Teleo species/genus/family/order:
  33.2 / 69.0 / 87.8 / 99.5

Interpretation:

- The near-exact result strengthens the 12S information-ceiling claim.
- At strict 99% identity, Teleo-style datasets still leave roughly 29% of
  held-out query species unresolved at species rank.
- At 95% identity, species-level support drops to roughly 33-43% for Teleo-
  style datasets, while order remains almost fully supported.
- This gives the merged Paper 1 a strong biological claim: many 12S/eDNA
  observations are not species-identifiable from sequence evidence alone, even
  before model architecture is considered.

Next actions:

1. Consolidate scattered reranking scripts into a single Eco-Phylo Posterior
   protocol.
2. Build full ablation matrix.
3. Strengthen calibration/no-call reporting.
4. Audit potential ecological leakage and document priors.
5. Decide which Global_eDNA figures are robust enough for the paper.
6. Add encoder-swap experiments after the posterior protocol is stable.

Full sequence+tree posterior threshold stability:

- Added the species-disabled nested calibration test in
  `scripts/edna/build_species_disabled_nested_calibration.py`.
- The policy intentionally disables species and backs off genus -> family ->
  order.
- Across 30 calibration resplits, the original held-out Global_eDNA groups at
  target-95 average 40.2% assignment at 94.3% accuracy; the 5th-95th percentile
  held-out accuracy range is 93.9-94.9%.
- This supports a conservative higher-rank eDNA claim. It does not support
  species-level eDNA calls, and it is not yet a full nested posterior retrain.

True nested full sequence+tree posterior:

- Completed the stricter nested posterior fit on Vast.
- Model fit: 70% of calibration groups.
- Threshold learning: remaining calibration groups.
- Evaluation: original held-out Global_eDNA groups.
- Species remains unsupported: species target-95 assigns 2.3% at 1.3% held-out
  accuracy.
- Higher ranks transfer:
  - genus target-90: 21.2% assignment at 94.1% accuracy;
  - family target-95: 34.0% assignment at 95.4% accuracy;
  - order target-95: 34.4% assignment at 96.7% accuracy.
- Mixed species-disabled rank-backoff:
  - target-90: 48.8% assignment at 85.9% accuracy;
  - target-95: 38.9% assignment at 93.4% accuracy.
- Additional nested fit70 repeats:
  - rep1 target-95: 38.5% assignment at 95.4% accuracy;
  - rep2 target-95: 54.9% assignment at 84.5% accuracy.
- Interpretation: report this as promising higher-rank eDNA evidence, not as a
  final calibrated operating point. The rep2 failure means the current mixed
  species-disabled target-95 policy is unstable across nested refits.
