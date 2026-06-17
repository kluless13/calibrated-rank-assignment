# Paper 1 Source Tables

All manuscript numbers should come from these generated tables, not directly
from logs.

For the current human-readable summary, start with:

- `experiments/paper1_phylo_calibrated_assignment/CURRENT_RESULTS.md`

## End-To-End Summary

- `pipeline_end_to_end_summary.csv`
- `pipeline_run_summary.csv`
- `../pipeline_calibration/pipeline_mode_policy_summary.csv`
- `../pipeline_calibration/pipeline_mode_thresholds.csv`
- `../production_v1/production_v1_summary_all.csv`
- `raw_sequence_production_v1_summary.csv`
- `../dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/coi_dl_evidence_rank_backoff_summary.csv`
- `../dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/coi_dl_evidence_thresholds.csv`
- `../dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/coi_dl_evidence_rank_backoff_bootstrap.csv`
- `dl_evidence_seed_summary.csv`
- `dl_evidence_seed_bootstrap_summary.csv`
- `dl_evidence_strict_apply_summary.csv`
- `candidate_reranker_summary.csv`
- `candidate_reranker_calibration_transfer.csv`
- `candidate_assignment_calibrator_summary.csv`
- `reference_gap_detector_summary.csv`
- `production_reason_code_summary.csv`
- `production_reason_code_examples.csv`
- `hf_foundation_probe_retrieval_metrics.csv`
- `marker_mirror_retrieval_metrics.csv`
- `marker_mirror_shared_retrieval_metrics.csv`
- `marker_mirror_triad_retrieval_metrics.csv`
- `marker_mirror_candidate_rankings_shared_seed1903_summary.csv`
- `marker_mirror_rank_policy_shared_seed1903_summary.csv`
- `marker_mirror_rank_calibrator_shared_seed1903_summary.csv`
- `marker_mirror_evidence_join_shared_seed1903_summary.csv`
- `marker_mirror_integrated_rank_logistic_shared_seed1903_summary.csv`
- `marker_mirror_integrated_rank_hgb_shared_seed1903_summary.csv`
- `marker_mirror_integrated_rank_best_shared_seed1903_summary.csv`
- `marker_mirror_candidate_generator_fullref_smoke_summary.csv`
- `marker_mirror_candidate_generator_fullref_smoke_manifest_summary.csv`
- `marker_mirror_candidate_generator_cache_smoke_summary.csv`
- `marker_mirror_candidate_generator_handoff_summary.csv`
- `marker_mirror_candidate_generator_handoff_manifest_summary.csv`
- `marker_mirror_candidate_generator_evidence_handoff_summary.csv`
- `marker_mirror_candidate_generator_evidence_handoff_manifest_summary.csv`
- `marker_mirror_candidate_generator_evidence_handoff_feature_inventory.csv`
- `marker_mirror_candidate_generator_rank_apply_summary.csv`
- `marker_mirror_candidate_generator_rank_apply_thresholds.csv`
- `marker_mirror_candidate_generator_rank_apply_manifest_summary.csv`
- `marker_mirror_calibration_transfer_cohort_summary.csv`
- `marker_mirror_calibration_transfer_handoff_strata.csv`
- `marker_mirror_calibration_transfer_feature_drift.csv`
- `marker_mirror_calibration_transfer_top_feature_drift.csv`
- `marker_mirror_calibration_transfer_diagnostics_manifest.json`
- `marker_mirror_reference_aware_policy_summary.csv`
- `marker_mirror_reference_aware_policy_manifest.json`
- `marker_mirror_reference_aware_policy_validation_summary.csv`
- `marker_mirror_reference_aware_policy_validation_per_split.csv`
- `marker_mirror_reference_aware_policy_validation_manifest.json`
- `marker_mirror_union_candidate_support_summary.csv`
- `marker_mirror_union_candidate_support_per_query.csv`
- `marker_mirror_union_candidate_support_manifest.json`
- `marker_mirror_same_marker_kmer_candidates_top50.csv.gz`
- `marker_mirror_union_static_policy_summary.csv`
- `marker_mirror_union_score_gate_validation_summary.csv`
- `marker_mirror_union_score_gate_validation_per_split.csv`
- `marker_mirror_union_rank_policy_manifest.json`
- `marker_mirror_union_evidence_compiler_summary.csv`
- `marker_mirror_union_evidence_compiler_family_order_summary.csv`
- `marker_mirror_union_evidence_compiler_order_summary.csv`
- `marker_mirror_union_evidence_compiler_per_split.csv`
- `marker_mirror_union_evidence_compiler_thresholds.csv`
- `marker_mirror_union_evidence_compiler_features.csv`
- `marker_mirror_union_evidence_compiler_manifest.json`
- `marker_mirror_union_reason_code_summary.csv`
- `marker_mirror_union_reason_code_by_source.csv`
- `marker_mirror_union_reason_code_per_query.csv`
- `marker_mirror_union_reference_curation_priorities.csv`
- `marker_mirror_union_reason_code_manifest.json`
- `marker_mirror_same_marker_edlib_support_summary.csv`
- `marker_mirror_same_marker_edlib_support_per_query.csv`
- `marker_mirror_same_marker_edlib_candidates_top50.csv.gz`
- `marker_mirror_same_marker_edlib_validation_manifest.json`
- `marker_mirror_union_listwise_selective_family_order_summary.csv`
- `marker_mirror_union_listwise_selective_family_order_per_split.csv`
- `marker_mirror_union_listwise_selective_family_order_thresholds.csv`
- `marker_mirror_union_listwise_selective_family_order_features.csv`
- `marker_mirror_union_listwise_selective_order_summary.csv`
- `marker_mirror_union_listwise_selective_order_per_split.csv`
- `marker_mirror_union_listwise_selective_order_thresholds.csv`
- `marker_mirror_union_listwise_selective_order_features.csv`
- `marker_mirror_union_listwise_selective_manifest.json`
- `marker_mirror_same_marker_vsearch_support_summary.csv`
- `marker_mirror_same_marker_vsearch_support_per_query.csv`
- `marker_mirror_same_marker_vsearch_candidates_top50.csv.gz`
- `marker_mirror_same_marker_vsearch_manifest.json`
- `marker_mirror_union_vsearch_candidate_support_summary.csv`
- `marker_mirror_union_vsearch_candidate_support_per_query.csv`
- `marker_mirror_union_vsearch_candidate_support_manifest.json`
- `marker_16s_local_source_audit.csv`
- `marker_16s_ncbi_query_plan.json`
- `marker_16s_reference_manifest.json`
- `marker_16s_reference_accessions.csv`
- `marker_16s_overlap_summary.csv`
- `marker_16s_candidate_taxonomy_enrichment.csv`
- `hierarchical_selective_rank_summary.csv`
- `hierarchical_selective_rank_thresholds.csv`
- `hierarchical_selective_rank_examples.csv`
- `retrieval_dl_sweep_summary.csv`
- `retrieval_dl_sweep_tree_recovery.csv`
- `retrieval_dl_sweep_training_history.csv`
- `../../remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/*/inference_summary.csv`
- `../manuscript_assets/figure_plan.csv`
- `../manuscript_assets/table_plan.csv`
- `../manuscript_assets/claim_evidence_map.csv`
- `../manuscript_assets/pipeline_operating_points.csv`
- `../manuscript_assets/missing_results_checklist.csv`
- `../manuscript_assets/marker_mirror/marker_mirror_candidate_support_table.csv`
- `../manuscript_assets/marker_mirror/marker_mirror_order_policy_table.csv`
- `../manuscript_assets/marker_mirror/marker_mirror_rank_boundary_table.csv`
- `../manuscript_assets/marker_mirror/marker_mirror_runtime_table.csv`
- `../manuscript_assets/marker_mirror/marker_mirror_figure_plan.csv`

Use for the high-level pipeline table:

- best current candidate retrieval rows;
- vector-index speed rows;
- APPLES-like placement rows;
- rank/no-call precision and false species-call rows;
- executable vector-first/rank-adaptive pipeline run rows;
- pure eDNA evidence-decomposition rows.

`pipeline_run_summary.csv` distinguishes `retrieval_mode`, `rerank_mode`, and
`assignment_source`. Treat HNSW rows as approximate speed checks. Treat raw
`*_pdistance_experimental` summaries as pre-production rows unless interpreted
through `pipeline_calibration/` or `production_v1/`.

Use `pipeline_calibration/` for seen-test-derived operating thresholds over the
executable modes. These rows are the safer source for claims about p-distance
reranking than the raw `*_pdistance_experimental` run summaries.

Use `production_v1/` for the packaged current COI operating point. These rows
apply locked p-distance-rerank thresholds and emit final assignment manifests.

Use `raw_sequence_production_v1_summary.csv` for raw split-sequence timing and
final assignment summaries through the current research pipeline.

Use `dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/` for the first DL
decision-layer result. Treat it as a model-development result until seed
repeats and strict hidden-taxonomy tests are added. The current adapter and
FASTA/CSV CLI integration are smoke-tested, but the DL layer is not yet the
default manuscript operating point.

Use `missing_reference_aware_calibrator_summary.csv` for the mixed
missing-reference-aware rank/no-call calibrator. It trains on normal supported
rows plus strict hidden-reference rows and uses v2 reference-gap probabilities
as soft evidence. Treat it as a precision-first model-development result until
seed repeats and production reason-label integration are complete.

Use `reference_gap_detector_summary.csv` for the first diagnostic
reference-gap detector family. The current strongest diagnostic rows are the
seed1301 v2 candidate-evidence detectors, which use candidate-list,
p-distance, taxonomic-diversity, reference-density, and tree-neighborhood
features without global candidate-count or split-specific leakage. Treat these
as diagnostic warning-layer rows, not final production rank/no-call policy
rows. Treat count-feature variants as diagnostics only because global
candidate-set size can encode synthetic strict-pack identity.

Use `gap_warning_overlay_summary.csv` and `gap_warning_overlay_examples.csv`
for the production-v1 reason-label overlay. These tables join existing v2
reference-gap probabilities onto existing production-v1 assignments. They are
diagnostic explanation/curation rows, not a newly trained policy.

Use `production_reason_code_summary.csv`,
`production_reason_code_examples.csv`, and
`production_reason_code_assignments.csv` for the first user-facing reason-code
overlay. These tables translate the current production-v1 rank/no-call
assignments plus v2 reference-gap probabilities into labels such as
`species_not_supported_at_operating_point`, `possible_missing_species_reference`,
`genus_ambiguous_in_top10`, and `no_call_no_rank_met`. They are explanatory
source tables, not independent accuracy metrics.

Use `retrieval_dl_sweep_summary.csv`, `retrieval_dl_sweep_tree_recovery.csv`,
and `retrieval_dl_sweep_training_history.csv` for the first retrieval-DL
encoder sweep. Treat these as model-development rows: CNN contrastive/hybrid
are useful candidate generators, while hierarchical/tree-shaped objectives are
tree-geometry probes rather than production rank/no-call operating points.

Use `candidate_reranker_summary.csv` for diagnostic top-50 COI candidate
rerankers. It now includes the first MLP reranker and the BLAST/VSEARCH-aware
retrieval-DL second-stage rerankers, including tree10 neighborhood-feature
variants. Treat these rows as model-development evidence-fusion results, not
the production rank/no-call operating point.

Use `candidate_reranker_calibration_transfer.csv` for cross-split
threshold-transfer diagnostics. It asks whether thresholds fitted on one split
survive on another split. Treat it as the stricter source for deciding whether
a reranker is ready for rank/no-call decisions.

Use `candidate_assignment_calibrator_summary.csv` for independent selected-rank
calibration over candidate-reranker outputs. Treat these rows as a test of
whether a second DL layer stabilizes rank/no-call thresholds; current rows are
model-development results, not the production operating point.

Use `marker_mirror_retrieval_metrics.csv` for the cross-marker MarkerMirror /
BarcodeBridge probes. It compares frozen Nucleotide Transformer cross-marker
retrieval against learned projection heads on species-level train/validation/test
splits. It now includes 12S->COI random-negative, taxonomy-hard, taxonomy-soft,
tree-distance, retrieval-aligned checkpoint, LoRA adapter, and multi-positive
variants, plus the first 12S->16S ribosomal bridge. Treat this as breakthrough
prospect evidence, not as a production assignment result. The current source
table has 684 rows across 10 runs after the reverse 16S->12S run was merged.

Use `marker_16s_local_source_audit.csv` for the first 12S/16S MarkerMirror
audit. It shows local Scotian Shelf 16S occurrence metadata exists, but no
clean local 16S species-reference sequence table is available yet. Treat this
as a data-readiness table, not a model result.

Use `marker_16s_ncbi_query_plan.json`, `marker_16s_reference_manifest.json`,
`marker_16s_reference_accessions.csv`, `marker_16s_overlap_summary.csv`, and
`marker_16s_candidate_taxonomy_enrichment.csv` for the first 12S/16S
MarkerMirror reference build. The bounded NCBI fetch produced 1,865 16S
species and 502 species overlapping the existing 12S multisource set. Treat
this as an initial reference construction artifact; taxonomy family/order are
partially enriched, not fully curated.

The first 12S->16S MarkerMirror result is in the same
`marker_mirror_retrieval_metrics.csv` table under run
`nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best`. Held-out top-10
species/genus/family/order are 33.9 / 45.4 / 67.1 / 73.4% for the learned
bridge versus 18.4 / 23.0 / 34.9 / 49.7% for frozen NT on the same split.
The reverse 16S->12S run is
`nt_v2_50m_16s_to_12s_taxonomy_soft_retrieval_best`; held-out top-10
species/genus/family/order are 44.4 / 54.8 / 70.4 / 76.3% for the learned
bridge versus 11.9 / 17.8 / 32.6 / 48.2% for frozen NT.

Use `marker_mirror_shared_retrieval_metrics.csv` for the shared 12S/16S
species-space prototype. It currently has 432 rows from three shared-head runs:
the original seed plus seed1902 and seed1903. Held-out top-10 mean
12S->16S species/genus/family/order are 43.4 / 50.7 / 68.1 / 77.6%; held-out
top-10 mean 16S->12S are 66.4 / 73.9 / 81.9 / 86.4%. Treat this as the
strongest MarkerMirror candidate-retrieval result so far, pending rank/no-call
integration.

Use `marker_mirror_triad_retrieval_metrics.csv` for the first 12S/16S/COI
shared-space prototype. It currently has 432 rows from one tri-marker run.
Held-out top-10 triad rows are: 12S->16S 31.1 / 47.2 / 65.0 / 73.1%,
16S->12S 51.9 / 65.9 / 77.8 / 89.6%, 12S->COI 2.7 / 12.7 / 34.8 / 58.6%,
and 16S->COI 5.3 / 20.7 / 48.7 / 64.7% for species/genus/family/order.
Treat this as useful cross-marker evidence, but not as a replacement for the
best direct 12S->COI bridge or the shared 12S/16S lead result.

Use `marker_mirror_candidate_rankings_shared_seed1903_summary.csv` for the
first full-reference candidate-generation export from the shared 12S/16S
MarkerMirror model. The compressed per-query candidate table is under
`../marker_mirror_bridge/candidate_rankings_shared_seed1903/marker_mirror_candidate_rankings.csv.gz`.
Unlike the aggregate training metric, this export ranks against the full target
marker library. Held-out learned top-50 full-reference top-k is 52.0 / 60.6 /
80.1 / 87.8% for 12S->16S and 74.3 / 80.6 / 90.3 / 93.8% for 16S->12S
species/genus/family/order. Treat this as candidate-generation evidence for
the pipeline.

Use `marker_mirror_candidate_generator_fullref_smoke_summary.csv` and
`marker_mirror_candidate_generator_fullref_smoke_manifest_summary.csv` for the
executable MarkerMirror candidate-generator handoff. This is a specimen-style
FASTA/CSV path, not the offline full-test-set export. The first GPU smoke used
32 held-out 12S queries, the full 1,865-species 16S reference, and top-50
candidates. Known-target top-50 recovery was 25.0 / 59.4 / 78.1 / 84.4% for
species/genus/family/order. Treat this as proof that the candidate-generation
entry point runs and emits auditable outputs, not as a final assignment policy.

Use `marker_mirror_candidate_generator_cache_smoke_summary.csv` for the
cache-backed repeat-inference check. The write pass created a 2,971-row 16S
foundation-embedding cache, the read pass loaded it, and both passes produced
identical candidate tables. Treat this as implementation/testing evidence, not
a new biological metric.

Use `marker_mirror_candidate_generator_evidence_handoff_summary.csv`,
`marker_mirror_candidate_generator_evidence_handoff_manifest_summary.csv`, and
`marker_mirror_candidate_generator_evidence_handoff_feature_inventory.csv` for
the first production-style evidence handoff. It takes executable
candidate-generator rows and adds same-marker sequence checks,
reference-availability flags, tree-neighborhood evidence, candidate-list
support/margins, and marker-resolvability features. Treat this as
rank/no-call-ready input, not as final assignment output.

Use `marker_mirror_candidate_generator_rank_apply_summary.csv`,
`marker_mirror_candidate_generator_rank_apply_thresholds.csv`, and
`marker_mirror_candidate_generator_rank_apply_manifest_summary.csv` for the
first end-to-end candidate-generator-to-rank/no-call apply diagnostics. Current
best honest row is logistic target-0.99 with species disabled: 6.5% coverage,
93.1% assigned precision, and 0 false species calls over all 3,566 12S query
rows. Treat this as a transfer diagnostic, not a final production operating
point, because it falls short of the nominal 0.99 target.

Use `marker_mirror_rank_policy_shared_seed1903_summary.csv`
for the first MarkerMirror-only rank/no-call diagnostic. It fits score
thresholds on validation and applies them to test. Treat it as a negative
calibration result: simple top-1 score thresholds are not enough for production
rank/no-call.

Use `marker_mirror_rank_calibrator_shared_seed1903_summary.csv`
for the feature-based MarkerMirror rank/no-call diagnostic. It trains per-rank
classifiers from top-k support, score margins, and ambiguity features, then
fits thresholds on validation. Treat it as a second diagnostic calibration
result: useful for evidence design, but not ready as the final rank/no-call
policy.

Use `marker_mirror_evidence_join_shared_seed1903_summary.csv` for the first
MarkerMirror evidence-join table summary. The full joined table is under
`../marker_mirror_bridge/candidate_rankings_shared_seed1903/evidence_join/marker_mirror_evidence_join.csv.gz`.
It adds candidate-list support/margins, same-marker sequence evidence,
reference availability, exact-sequence ambiguity flags, and tree distance to
the top candidate.

Use `marker_mirror_integrated_rank_logistic_shared_seed1903_summary.csv`,
`marker_mirror_integrated_rank_hgb_shared_seed1903_summary.csv`, and
`marker_mirror_integrated_rank_best_shared_seed1903_summary.csv` for the first
integrated MarkerMirror rank/no-call prototype. The best held-out target-0.99
rows are 12S->16S logistic at 55.0% coverage, 99.4% assigned precision, and
0.0% false species-call rate; and 16S->12S HGB at 75.0% coverage, 99.1%
assigned precision, and 0.0% false species-call rate. Treat this as a strong
pipeline diagnostic, pending seed repeats and field/eDNA validation.

Use `paper1_production_v1_cli/*/inference_summary.csv` and
`inference_assignments.csv` for FASTA/CSV CLI smoke checks. These are interface
tests, not headline benchmark rows. Current archived smoke roots include the
hand-threshold mode and the `dl_mlp_species_disabled` decision mode under
`results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/`.

Use `manuscript_assets/` for writing plans and claim-to-evidence tracking. Those
files are planning artifacts, not new results.

## COI Retrieval And Tree Geometry

- `pipeline_coi_method_benchmark.csv`
- `retrieval_metrics.csv`
- `tree_recovery_metrics.csv`
- `retrieval_dl_sweep_summary.csv`
- `retrieval_dl_sweep_tree_recovery.csv`
- `retrieval_dl_sweep_training_history.csv`
- `tree_distance_bin_summary.csv`
- `neighborhood_preservation.csv`
- `reference_diagnostics_summary.csv`

Use for:

- species/genus/family/order top-k retrieval;
- tree-recovery Pearson/Spearman;
- nearest-reference and reference-gap diagnostics;
- model/classical comparisons.
- retrieval-DL candidate-generator and tree-geometry development sweeps.

## Placement Comparators

- `pipeline_placement_benchmark.csv`
- `placement_rank_diagnostics_summary.csv`
- `placement_rank_diagnostics_per_query.csv`
- `apples_like_distance_placement_summary.csv`
- `placement_tree_error_summary.csv`
- `placement_pcp_like_summary.csv`
- `placement_pcp_like_lwr_summary.csv`
- `placement_simulated_tree_pcp_summary.csv`
- `placement_simulated_tree_pcp_per_query.csv`
- `fernando_completeness_final_30/placement_rank_diagnostics_summary.csv`
- `fernando_completeness_final_30/placement_pcp_like_summary.csv`
- `fernando_completeness_final_30/placement_simulated_tree_pcp_summary.csv`

Use for:

- EPA-ng placed-clade containment;
- APPLES-like local distance-placement diagnostics;
- official APPLES 2.0.11 and EPA-ng matched completeness-sweep diagnostics;
- nearest-reference tree-error summaries;
- Fernando-like edge-to-sister diagnostics;
- simulated-placement-tree PCP-style diagnostics.

Use `fernando_completeness_final_30/` for the completed 30-sweep public
Fernando-style matrix. Do not use these tables to claim exact Fernando PCP
unless a future table explicitly says `fernando_pcp`.

## Rank/No-Call Calibration

- `missing_reference_aware_policy_summary.csv`
- `missing_reference_aware_policy_bootstrap.csv`
- `missing_reference_aware_thresholds.csv`
- `prospective_rank_adaptive_policy_summary.csv`
- `calibration_per_query.csv`
- `strict_missing_reference_summary.csv`
- `strict_rank_backoff_summary.csv`
- `../production_v1/production_v1_summary_all.csv`
- `raw_sequence_production_v1_summary.csv`
- `../dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/coi_dl_evidence_rank_backoff_summary.csv`
- `../dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/coi_dl_evidence_rank_backoff_bootstrap.csv`
- `../dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/coi_dl_evidence_predictions.csv`
- `dl_evidence_seed_summary.csv`
- `dl_evidence_seed_bootstrap_summary.csv`
- `dl_evidence_strict_apply_summary.csv`
- `candidate_reranker_summary.csv`
- `candidate_reranker_calibration_transfer.csv`
- `candidate_assignment_calibrator_summary.csv`
- `reference_gap_detector_summary.csv`
- `../dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance_apply_smoke_eval_c/coi_dl_evidence_applied_summary.csv`
- `../../remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/*/inference_summary.csv`

Use for:

- candidate locked rank/no-call policy;
- assigned precision;
- coverage;
- false species-call rate;
- bootstrap confidence intervals.
- strict tree-pruned forced top-k metrics.
- descriptive strict rank-backoff interpretation: deepest supported rank when
  species/genus/family references are removed before training and candidate
  construction.
- production-v1 final COI rank/no-call assignments and summary.
- raw split-sequence to final rank/no-call latency.
- DL species-enabled and species-disabled rank/no-call policy summaries.
- DL seed-repeat stability.
- DL strict hidden-reference stress testing.
- diagnostic candidate-level DL reranking over top-50 COI candidates.
- cross-split calibration transfer for candidate-level DL rerankers.
- independent selected-candidate DL assignment calibration.
- diagnostic reference-gap detection under strict hidden-reference packs.
- production-v1 reason-code overlays for why a query was assigned to a rank or
  left as no-call.
- DL apply-adapter smoke summaries.
- FASTA/CSV CLI smoke summaries and specimen-facing assignment files.

Current headline policy is CNN seed1206 at target 0.99.

## Vector Retrieval

- `ann_vector_retrieval_metrics.csv`
- `ann_vector_runtime_comparison.csv`
- `ann_vector_recall_against_exact.csv`
- `ann_vector_stress_runtime.csv`
- `controlled_vector_speed_benchmark.csv`
- `controlled_vector_speed_detail.csv`

Use for:

- exact vector vs HNSW recall;
- local latency;
- index size;
- synthetic larger-reference stress timing.
- repeat-based median latency and spread.
- controlled Vast HNSW/exact vector timing for CNN seed1206 Eval C.

Synthetic-expanded stress rows are not biological retrieval accuracy.

## 12S/eDNA

- `merged_12s_resolvability_summary.csv`
- `merged_12s_zero_shot_model_metrics.csv`
- `merged_global_edna_asv_metrics.csv`
- `merged_global_edna_sample_metrics.csv`
- `merged_edna_evidence_arm_status.csv`
- `merged_global_edna_calibration_curves.csv`
- `edna_evidence_decomposition_matrix.csv`
- `edna_evidence_best_by_rank.csv`
- `edna_rank_no_call_operating_points.csv`
- `edna_evidence_decomposition_manifest.json`
- `../global_edna_independent_rank_calibration/global_edna_independent_rank_calibration_summary.csv`
- `../global_edna_independent_rank_calibration/global_edna_independent_rank_calibration_methods.csv`
- `../global_edna_independent_rank_calibration/global_edna_independent_rank_calibration_manifest.json`
- `../eco_phylo_posterior/eco_phylo_posterior_query_features.csv.gz`
- `../eco_phylo_posterior/eco_phylo_posterior_query_features_sample.csv`
- `../eco_phylo_posterior/eco_phylo_posterior_method_design.csv`
- `../eco_phylo_posterior/eco_phylo_posterior_rank_correctness_summary.csv`
- `../eco_phylo_posterior/eco_phylo_posterior_manifest.json`
- `../eco_phylo_posterior/eco_phylo_posterior_selected_predictions.csv.gz`
- `../eco_phylo_posterior/eco_phylo_posterior_model_summary.csv`
- `../eco_phylo_posterior/eco_phylo_posterior_operating_points.csv`
- `../eco_phylo_posterior/eco_phylo_posterior_method_selection_summary.csv`
- `../eco_phylo_posterior/eco_phylo_posterior_rank_backoff_summary.csv`
- `../eco_phylo_posterior/eco_phylo_posterior_model_manifest.json`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_features_top5.csv.gz`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_features_top5_sample.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_method_inventory.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_feature_manifest.json`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_selected_predictions.csv.gz`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_model_summary.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_operating_points.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_method_selection_summary.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_posterior_model_manifest.json`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_top5.csv.gz`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_summary.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_12s_sequence_evidence_manifest.json`
- `../eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_selected_predictions.csv.gz`
- `../eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_model_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_operating_points.csv`
- `../eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_method_selection_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_evidence_sample/eco_phylo_candidate_posterior_model_manifest.json`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_tree_neighborhood_evidence_sample10k.csv`
- `../eco_phylo_posterior/candidate_level/eco_phylo_candidate_tree_neighborhood_evidence_sample10k_manifest.json`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_sample/eco_phylo_candidate_posterior_selected_predictions.csv.gz`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_sample/eco_phylo_candidate_posterior_model_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_sample/eco_phylo_candidate_posterior_operating_points.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_sample/eco_phylo_candidate_posterior_method_selection_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_sample/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_sample/eco_phylo_candidate_posterior_model_manifest.json`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_selected_predictions.csv.gz`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_model_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_operating_points.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_method_selection_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_model_manifest.json`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_species_disabled_rank_backoff_assignments.csv.gz`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_candidate_posterior_species_disabled_rank_backoff_manifest.json`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_per_repeat.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_thresholds.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_manifest.json`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_selected_predictions.csv.gz`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_model_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_operating_points.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/eco_phylo_candidate_posterior_model_manifest.json`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep1/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`
- `../eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep2/eco_phylo_candidate_posterior_species_disabled_rank_backoff_summary.csv`

Use for:

- 12S marker resolvability ceilings;
- SSM/CNN 12S candidate retrieval;
- Global_eDNA sequence-only and context-aware validation;
- pure geography-only and same-sample co-occurrence-only decomposition arms.
- learned co-occurrence versus geography/context controls;
- diagnostic eDNA rank/no-call operating points.
- posterior-ready evidence features for the next Eco-Phylo scoring layer.

Important caveat: `edna_rank_no_call_operating_points.csv` is diagnostic only.
It is built from current validation curves, not an independent calibration
split. Use it to design the next posterior/calibration experiment, not as a
final manuscript operating-point claim.

The `global_edna_independent_rank_calibration/` tables are stronger because
thresholds are learned on deterministic `site20` calibration groups and
evaluated on held-out `site20` groups. Current results are still not a positive
high-accuracy rank/no-call claim: only modest family/order operating points
transfer, so the next eDNA step is a stronger Eco-Phylo posterior.

The `eco_phylo_posterior/` tables now include both design/prototype inputs and
the first held-out posterior scorer. The input tables consolidate existing
Global_eDNA prediction scores, candidate/tree evidence, ecological arm labels,
12S marker-resolvability ceilings, deterministic calibration/evaluation split
labels, and rank correctness. The model tables are an initial top-1 method-arm
posterior, not the final Eco-Phylo system: first-pass thresholds do not yet beat
the strongest single-method SSM + RLS/OBIS family/order calibration.

The `eco_phylo_posterior/candidate_level/` tables are the next posterior layer.
They expand 24 evidence arms into 6,995,880 top-5 candidate rows with
candidate-specific model score, sequence-only score where available, BLAST
pident/rank evidence where available, RLS/OBIS counts, reference/taxonomy
metadata, and rank correctness. The candidate-posterior model tables are now a
full-table Vast run over all candidate rows. Current full-table operating
points are useful for family/order/rank-backoff claims, but species/genus remain
limited and the method still needs explicit candidate tree-distance and 12S
p-distance/alignment features before becoming the final Eco-Phylo posterior.

The `eco_phylo_candidate_12s_sequence_evidence_*` tables add direct
train-reference 12S evidence for candidate rows. They cover 584,894 unique
query/candidate pairs, with evidence available for 145,785 pairs. The distance
is best ungapped sliding p-distance/identity against
`train_species_sequences.json`, not BLAST and not a full affine-gap alignment.
The `candidate_level_sequence_evidence_sample/` posterior output is a local
10,000 complete-query sampled diagnostic. It should not replace the full-table
sequence+tree posterior now available under
`candidate_level_sequence_tree_evidence_full/`.

The `eco_phylo_candidate_tree_neighborhood_evidence_sample10k.csv` table adds
inference-safe tree-neighborhood features for the same 10,000 sampled
complete-query groups. It uses distances and taxonomic agreement among
retrieved candidates only, not true-query-to-candidate tree distance. The
`candidate_level_sequence_tree_evidence_sample/` posterior diagnostic improves
held-out genus/family/order operating points, but species thresholds overfit and
must not be used as a final species-call policy.

The `candidate_level_sequence_tree_evidence_full/` tables are the full-table
Vast sequence+tree posterior over all 6,995,880 candidate rows. They improve
high-confidence genus/family/order operating points, but the full run confirms
that species thresholds still do not transfer. Use these tables for
higher-rank eDNA posterior evidence and for designing a species-disabled
rank-backoff policy.

The `eco_phylo_candidate_posterior_species_disabled_rank_backoff_*` tables are
the current honest eDNA rank-backoff policy. They intentionally remove species
and apply genus -> family -> order thresholds. On held-out groups, target-95
assigns 40.3% of queries at 94.3% accuracy.

The `eco_phylo_species_disabled_nested_calibration_*` tables stress-test that
policy by repeatedly resplitting calibration groups into threshold-fit and
calibration-holdout groups. Across 30 resplits, target-95 on the original
held-out eDNA groups averages 40.2% assignment at 94.3% accuracy. This is a
threshold-stability test, not a full nested posterior retrain.

The `candidate_level_sequence_tree_evidence_nested_fit70_rep0/` tables are the
stricter true nested posterior run. The model is fit on 70% of calibration
groups, thresholds are learned on the remaining calibration groups, and
evaluation is on the original held-out eDNA groups. Species still fails under
this split. Family/order target-95 transfer well individually, while mixed
species-disabled target-95 assigns 38.9% of held-out queries at 93.4% accuracy.
The additional `rep1` and `rep2` nested refits give species-disabled target-95
assignment/accuracy of 38.5%/95.4% and 54.9%/84.5%, respectively. Treat these
as a stability warning: the posterior is useful for method development, but the
current mixed eDNA policy is not final.

## MarkerMirror Integrated Evidence

For a concise coauthor-facing summary of the MarkerMirror evidence package, use
`experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_COAUTHOR_ONE_PAGER.md`.

- `marker_mirror_evidence_join_seed_repeat_summary.csv`
- `marker_mirror_integrated_rank_seed_repeat_summary.csv`
- `marker_mirror_integrated_rank_seed_repeat_best_target099.csv`
- `marker_mirror_integrated_rank_seed_repeat_target099_stability.csv`
- `marker_mirror_marker_resolvability_by_species.csv`
- `marker_mirror_marker_resolvability_summary.csv`
- `marker_mirror_marker_resolvability_backend.csv`
- `marker_mirror_evidence_join_resolvability_seed1903_summary.csv`
- `marker_mirror_resolvability_calibrator_seed1903_summary.csv`
- `marker_mirror_resolvability_calibrator_seed1903_best_target099.csv`
- `marker_mirror_production_handoff_next_actions.csv`
- `marker_mirror_candidate_generator_smoke_summary.csv`
- `marker_mirror_candidate_generator_smoke_manifest_summary.csv`
- `marker_mirror_union_candidate_support_summary.csv`
- `marker_mirror_union_candidate_support_per_query.csv`
- `marker_mirror_union_candidate_support_manifest.json`
- `marker_mirror_same_marker_kmer_candidates_top50.csv.gz`
- `marker_mirror_union_static_policy_summary.csv`
- `marker_mirror_union_score_gate_validation_summary.csv`
- `marker_mirror_union_score_gate_validation_per_split.csv`
- `marker_mirror_union_rank_policy_manifest.json`
- `marker_mirror_union_evidence_compiler_summary.csv`
- `marker_mirror_union_evidence_compiler_family_order_summary.csv`
- `marker_mirror_union_evidence_compiler_order_summary.csv`
- `marker_mirror_union_evidence_compiler_per_split.csv`
- `marker_mirror_union_evidence_compiler_thresholds.csv`
- `marker_mirror_union_evidence_compiler_features.csv`
- `marker_mirror_union_evidence_compiler_manifest.json`

Use these for the shared 12S/16S MarkerMirror pipeline handoff. The tables
combine full-reference candidate exports for seeds 1901, 1902, and 1903 with
same-marker sequence checks, candidate-list consensus/margins, marker-reference
availability, exact-sequence ambiguity, and tree-neighborhood evidence. The
target-0.99 stability table is the claim-facing summary: learned MarkerMirror
achieves 51.0% mean held-out coverage at 98.9% assigned precision for
12S->16S, and 71.1% mean held-out coverage at 98.7% assigned precision for
16S->12S. False species-call rates remain near zero across seeds.

Caveat: these are controlled held-out marker-reference tests, not yet real
field eDNA validation. The new marker-resolvability tables make the ambiguity
layer explicit for 12S and 16S. The exact rows are direct exact-sequence
ambiguity; the 0.99 rows use a rare-kmer prefix-identity proxy because
VSEARCH/edlib is not yet wired into this local pipeline. The seed1903
resolvability-enhanced calibrator has the same best target-0.99 operating point
as the previous seed1903 compiler, so the current value is auditability and
claim-boundary clarity rather than improved precision/coverage.

The candidate-generator smoke tables are executable-handoff tests, not
performance evidence. They show that
`scripts/edna/run_marker_mirror_candidate_generator.py` can parse
specimen-style input and write top-k 12S->16S candidates plus a manifest. The
smoke run used only 25 target species for speed.

Use `marker_mirror_union_candidate_support_summary.csv` for the first union
candidate-support audit. It combines the full-query MarkerMirror 12S->16S top50
candidate list with a same-marker 12S character-kmer top50 candidate list. This
does not improve species support because the full-query species are absent from
the current 12S reference sequence table and only 26.6% are present in the 16S
reference. It does substantially improve high-rank support: MarkerMirror-only
top50 support is 9.5 / 39.9 / 59.8 / 76.3% for species/genus/family/order,
while the union candidate list is 9.5 / 91.7 / 95.1 / 99.6%. Treat the
same-marker arm as a historical k-mer audit. For claim-facing same-marker
evidence, use the BLASTN and VSEARCH tables below.

Use `marker_mirror_union_static_policy_summary.csv` and
`marker_mirror_union_score_gate_validation_summary.csv` for the first
production-style union rank/no-call diagnostic. The production candidate table
is written under
`../marker_mirror_bridge/union_candidate_rank_policy/marker_mirror_union_production_candidates.csv.gz`
and contains 355,231 candidate rows without hidden labels. The strongest
simple static policy is top-1 source agreement at family/order only: 25.2%
coverage, 98.4% assigned precision, and 0 false species calls. Same-marker
score gates are higher coverage but not locked: repeated species-split
validation gives family target-0.95 mean 92.7% coverage at 94.8% precision, and
order target-0.99 mean 68.5% coverage at 99.0% precision. Treat these as
rank/no-call diagnostics, not field-eDNA production thresholds.

Use `marker_mirror_union_evidence_compiler_*` for the first learned union
evidence compiler. It trains HGB classifiers over 102 production-available
top-1 union features using train/calibration/evaluation species splits. This is
a useful negative/diagnostic result: the mixed family/order compiler is not
stable enough, and the order-only compiler does not beat the simpler score
gate. Order-only target-0.99 averages 67.4% coverage at 98.5% precision and
meets target in 44% of repeats. The better claim-facing diagnostics remain
family/order source agreement for conservative precision and same-marker
score-gated order for high-rank signal.

Use `marker_mirror_union_reason_code_*` for the first MarkerMirror union
explanation layer. The query-level table joins union support, top-1 source
agreement, current marker-reference availability, and static family/order
assignments. The summary shows that 2,249/3,566 queries have genus-level union
support but no species call, 621 can emit conservative family calls at 98.1%
precision, and 277 can emit conservative order calls at 99.3% precision.
Reference-curation priorities rank species where high-rank support is strong
but species-level marker coverage blocks a more specific call. Treat current
12S reference-gap labels as current-table/split-design diagnostics, not a final
statement about all possible 12S databases.

Use `marker_mirror_same_marker_edlib_*` for alignment-backed validation of the
same-marker candidate pool. The run scored 176,931 existing 12S k-mer candidate
rows with bidirectional edlib HW identity against up to 12 reference sequences
per candidate species. Edlib reranking preserves the high-rank signal:
top-10 support is 0.0 / 87.8 / 94.3 / 98.8% for species/genus/family/order,
versus 0.0 / 86.9 / 93.9 / 98.7% for the original k-mer order. This validates
the candidate pool but is not full all-vs-all BLAST/VSEARCH search.

Use `marker_mirror_union_listwise_selective_*` for the second learned union
compiler. Unlike the first top-1 HGB, this uses list-level concentration,
source-agreement, score-margin, and edlib identity features. The order-only
target-0.99 row is the best high-coverage learned diagnostic so far: 83.1%
coverage at 98.8% mean precision, with target met in 56% of species-split
repeats. It improves on the previous top-1 HGB order-only row but still does
not lock a 99% production threshold. Treat it as a promising diagnostic and
not as the final rank/no-call policy.

Use `marker_mirror_same_marker_vsearch_*` and
`marker_mirror_union_vsearch_candidate_support_*` for the VSEARCH-backed
same-marker arm. VSEARCH global alignment over 3,566 queries and 12,593 current
12S reference sequences produced 178,204 top-50 species-level candidate rows.
Same-marker VSEARCH top50 support is 0.0 / 90.4 / 94.9 / 99.4% for
species/genus/family/order. The MarkerMirror + VSEARCH union top50 support is
9.5 / 91.8 / 95.1 / 99.6%. This is claim-facing classical same-marker candidate
generation, with the caveat that it is VSEARCH global alignment, not BLAST local
alignment, and cannot recover species absent from the current 12S reference.

Use `marker_mirror_same_marker_blast_*` and
`marker_mirror_union_blast_candidate_support_*` for the BLASTN-backed
same-marker arm. BLASTN local alignment over the same 3,566 queries and 12,593
current 12S reference sequences produced 177,726 top-50 species-level candidate
rows. Same-marker BLASTN top50 support is 0.0 / 90.7 / 95.1 / 99.4% for
species/genus/family/order. The MarkerMirror + BLASTN union top50 support is
9.5 / 92.1 / 95.3 / 99.7%. This is now the strictest same-marker classical
candidate-generation table available for MarkerMirror union claims.

Use `marker_mirror_blast_vsearch_calibration_repair_*` for the first
BLAST/VSEARCH-backed calibration-transfer repair diagnostic. It evaluates
production-available top1/top10 list policies over 50 query-species
calibration/evaluation repeats. The stable target-0.99 row is all-source
top1 order agreement: 24.8% mean coverage, 99.6% mean precision, target met in
100% of repeats. High-coverage order rows are promising but not locked:
BLAST top10 source-stratified Wilson95 gives 69.0% coverage at 99.4% precision
and target met in 82% of repeats. Use Exp 117
`marker_mirror_high_coverage_order_repair_*` below for the stricter
high-coverage order diagnostic.

Use `marker_mirror_stable_order_policy_*` for the explicit production-style
handoff of the stable all-source order policy. The tables are:

- `marker_mirror_stable_order_policy_assignments.csv`
- `marker_mirror_stable_order_policy_production_assignments.csv`
- `marker_mirror_stable_order_policy_summary.csv`
- `marker_mirror_stable_order_policy_by_source.csv`
- `marker_mirror_stable_order_policy_reason_counts.csv`
- `marker_mirror_stable_order_policy_manifest.json`

The assignment table emits order calls only when MarkerMirror, BLASTN, and
VSEARCH agree on top-1 order. The unthresholded policy assigns 886/3,566
queries, 24.8% coverage, at 99.7% assigned precision with 0 false species
calls. The max-repeat target-0.99 threshold assigns 880/3,566 queries, 24.7%
coverage, at 99.7% assigned precision with 0 false species calls. Treat the
thresholded row as the conservative handoff candidate.

`marker_mirror_stable_order_policy_production_assignments.csv` is the
label-stripped handoff table for downstream CLI/API work. It contains query id,
source, decision mode, assigned rank/label, confidence, reason code, threshold,
and the top-1 order evidence from MarkerMirror, BLASTN, and VSEARCH. It does
not contain truth labels or correctness columns.

Use `marker_mirror_high_coverage_order_repair_*` for the Exp 117 nested
high-coverage order repair diagnostic. The tables are:

- `marker_mirror_high_coverage_order_repair_summary.csv`
- `marker_mirror_high_coverage_order_repair_per_split.csv`
- `marker_mirror_high_coverage_order_repair_thresholds.csv`
- `marker_mirror_high_coverage_order_repair_locked_thresholds.csv`
- `marker_mirror_high_coverage_order_repair_assignments.csv`
- `marker_mirror_high_coverage_order_repair_assignment_summary.csv`
- `marker_mirror_high_coverage_order_repair_manifest.json`

The best nested target-0.99 row is BLASTN/VSEARCH top-10 order agreement with
nested global Wilson95 locking: 57.2% mean held-out coverage, 99.8% mean
assigned precision, target met in 100% of 50 outer species-split repeats, and
minimum repeat precision 99.3%. The full-table locked diagnostic assigns
2,513/3,566 rows, 70.5% coverage, at 99.8% labelled precision. Treat this as a
high-coverage diagnostic. Exp 118 wires it into the CLI as explicit
`--decision-mode high_coverage_order`; it is not the default mode.

Use the following files for the Exp 119 family/genus repair check:

- `marker_mirror_high_coverage_rank_repair_comparison.csv`
- `marker_mirror_high_coverage_family_repair_all_policies_summary.csv`
- `marker_mirror_high_coverage_family_repair_all_policies_per_split.csv`
- `marker_mirror_high_coverage_family_repair_all_policies_assignments.csv`
- `marker_mirror_high_coverage_genus_repair_all_policies_summary.csv`
- `marker_mirror_high_coverage_genus_repair_all_policies_per_split.csv`
- `marker_mirror_high_coverage_genus_repair_all_policies_assignments.csv`

Neither family nor genus reached stable target-0.99 transfer across all 50
species-split repeats. Keep these as diagnostic source tables only.

Use the following files for the Exp 121 hierarchical candidate-set diagnostic:

- `marker_mirror_hierarchical_candidate_sets_policy_rows.csv.gz`
- `marker_mirror_hierarchical_candidate_sets_policy_grid_summary.csv`
- `marker_mirror_hierarchical_candidate_sets_per_split.csv`
- `marker_mirror_hierarchical_candidate_sets_chosen_policies.csv`
- `marker_mirror_hierarchical_candidate_sets_summary.csv`
- `marker_mirror_hierarchical_candidate_sets_full_policy.csv`
- `marker_mirror_hierarchical_candidate_sets_assignments.csv.gz`
- `marker_mirror_hierarchical_candidate_sets_manifest.json`

This is a set-valued alternative to single-label family/genus repair. It shows
that family/genus still do not reach stable target-0.99 under current evidence.
Treat it as a claim-boundary diagnostic.

Use the following files for the Exp 122 MarkerMirror manuscript-facing package:

- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/marker_mirror_candidate_support_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/marker_mirror_order_policy_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/marker_mirror_rank_boundary_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/marker_mirror_runtime_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/marker_mirror_figure_plan.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/marker_mirror_methods_blurb.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/marker_mirror_manuscript_asset_manifest.json`

These are writing assets built from the source tables above, not new metrics.
Use them for the MarkerMirror manuscript table, figure plan, runtime callouts,
and short methods text.

Use the following files for the Exp 123 MarkerMirror figure drafts:

- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_candidate_support_bars.png`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_candidate_support_bars.pdf`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_order_policy_tradeoff.png`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_order_policy_tradeoff.pdf`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_rank_boundary.png`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_rank_boundary.pdf`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_runtime_breakdown.png`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_runtime_breakdown.pdf`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_slide_ready_summary.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/marker_mirror_manuscript_figure_manifest.json`

These are rendered drafts from Exp 122 source tables, not separate source
metrics.

Use the following files for the Exp 124 slide-ready MarkerMirror package:

- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_candidate_support_slide_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_candidate_support_slide_table.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_order_policy_slide_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_order_policy_slide_table.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_rank_boundary_slide_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_rank_boundary_slide_table.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_runtime_slide_table.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_runtime_slide_table.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_slide_package_outline.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/marker_mirror_slide_tables_manifest.json`

These repackage the Exp 122/123 material into slide tables and a five-slide
outline. They are not additional measurements.

Use the following files for the Exp 125 MarkerMirror manuscript text package:

- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_figure_captions.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_results_paragraph.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_methods_paragraph.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_claim_boundary_box.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_caption_inventory.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_text_manifest.json`

These are manuscript text snippets and caption inventory rows derived from
existing source metrics. They are not additional measurements.

Use the following files for the Exp 126 MarkerMirror manuscript section
outline:

- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_section_outline.md`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_section_checklist.csv`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_section_manifest.json`

These are manuscript organization and readiness files derived from existing
MarkerMirror text, figure, and source-table assets. They are not additional
measurements.

Use the following files for the Exp 127 MarkerMirror next-evidence audit:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_next_evidence_source_audit.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_by_lineage.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_next_evidence_source_manifest.json`
- `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_family_genus_next_evidence_plan.md`

These files summarize which genuinely new evidence sources exist for a future
family/genus attempt. They do not enable family/genus calls.

Use the following files for the Exp 128 MarkerMirror lineage/reference-coverage
policy diagnostic:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_per_split.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_thresholds.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_features.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_lineage_features.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_manifest.json`

These files test whether lineage-level reference coverage features can repair
family/genus calibration. They are diagnostic only. No family/genus row met a
stable target-0.99 transfer criterion, so the wrapper remains order/no-call.

Use the following files for the Exp 129 VSEARCH-backed marker-resolvability
diagnostic:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_cluster_rank_counts.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_query_oracle_rates.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_12s_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_16s_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_manifest.json`

The copied full output tree is:

- `results/remote_runs/2026-06-04/rtx_pro_6000/marker_mirror_vsearch_resolvability_20260604/`

These files replace the previous rare-kmer marker-resolvability proxy with a
VSEARCH-backed near-exact clustering diagnostic. They do not enable
family/genus/species calls by themselves.

Use the following files for the Exp 130 VSEARCH-resolvability-aware policy
diagnostic:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_per_split.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_thresholds.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_features.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_query_features.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_manifest.json`

These files join production-available VSEARCH cluster features to the existing
MarkerMirror/BLASTN/VSEARCH policy rows. Hidden oracle-support columns are not
used as features. No family/genus target-0.99 row transferred cleanly.

Use the following files for the Exp 131 MarkerMirror active
reference-curation/value-of-information layer:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_species.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_lineage.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_actions.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_manifest.json`

These files rank species groups, lineages, and action categories by which
reference or evidence additions would most plausibly improve MarkerMirror's
current no-call/high-rank-only behavior. The diagnostic VSEARCH oracle columns
use benchmark labels for curation triage only; they are not production inference
features and do not enable family/genus/species calls.

Use the following files under
`marker_mirror_12s_production_v1/dry_run_smoke/` for the first one-command 12S
orchestration wrapper smoke:

- `input_queries/zero_shot_queries.csv`
- `marker_mirror_12s_production_dependency_report.csv`
- `marker_mirror_12s_production_plan.json`
- `marker_mirror_12s_production_next_actions.csv`
- `marker_mirror_12s_production_manifest.json`
- `blast_smoke/marker_mirror_same_marker_blast_candidates_top50.csv.gz`
- `blast_smoke/marker_mirror_same_marker_blast_manifest.json`

The dependency report shows BLASTN and makeblastdb available locally, with
VSEARCH missing. The BLAST smoke table has 100 candidate rows from 2 normalized
12S query rows.

Use the following copied Vast full-run directory for the complete one-command
MarkerMirror 12S production-v1 wrapper result:

- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/marker_mirror_12s_production_assignments.csv`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/marker_mirror_12s_production_manifest.json`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/features/marker_mirror_12s_production_features.csv`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/stable_order_policy/marker_mirror_stable_order_policy_summary.csv`

This full run covers 3,566 12S queries and emits 880 order calls plus 2,686
no-calls. Diagnostic labelled precision is 99.7% with 0 false species calls.

Use the following copied Vast unlabeled FASTA smoke directory to verify
production behavior without truth labels:

- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/marker_mirror_12s_production_assignments.csv`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/marker_mirror_12s_production_manifest.json`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/stable_order_policy/marker_mirror_stable_order_policy_summary.csv`

This smoke uses two FASTA records with taxonomy stripped. It emits 1 order call
and 1 no-call, and the precision/correctness fields are blank because no truth
labels were supplied.

Use the following copied Vast smoke directories for the Exp 118 explicit
decision-mode wiring:

- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_stable_labelled_20260604/`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_high_coverage_labelled_20260604/`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_stable_unlabeled_20260604/`
- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_high_coverage_unlabeled_20260604/`

Each directory includes `marker_mirror_12s_production_assignments.csv`,
`marker_mirror_12s_production_manifest.json`, and the relevant policy summary
subdirectory. The labelled high-coverage smoke assigned 3/4 rows at 100%
diagnostic precision. The unlabeled high-coverage smoke assigned 2/2 rows and
left precision/correctness blank by design.

## Pipeline Status

- `pipeline_component_status.csv`
- `pipeline_best_by_task.csv`
- `pipeline_next_actions.csv`

Use for internal status and planning. `pipeline_best_by_task.csv` is
descriptive and should not be treated as a claim-ready leaderboard without the
claim-boundary notes.
