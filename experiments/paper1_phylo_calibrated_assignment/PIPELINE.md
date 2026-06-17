# Paper 1 Pipeline

## Goal

Build an uncertainty-aware molecular biodiversity inference pipeline:

```text
marker sequence
  -> candidate retrieval
  -> tree/placement/reference diagnostics
  -> optional ecological context
  -> calibrated species/genus/family/order/no-call
```

The pipeline is not an SSM-only system. It is model-agnostic and compares
classical tools, neural encoders, vector retrieval, phylogenetic placement, and
eDNA ecological/context arms under one benchmark ledger.

The DL track is now split into five explicit layers:

- candidate retrieval encoder;
- evidence reranker;
- rank/no-call calibrator;
- reference-gap detector;
- eDNA Eco-Phylo posterior.

Roadmap:

- `experiments/paper1_phylo_calibrated_assignment/DL_MODEL_ROADMAP.md`

## Current Pipeline Stages

### 1. Candidate Retrieval

Implemented sources:

- BLAST
- VSEARCH
- k-mer
- CNN / biLSTM / Transformer / Mamba candidate rankings
- exact vector-first cosine retrieval over saved embeddings
- HNSW ANN vector-index retrieval over saved embeddings

Current tables:

- `retrieval_metrics.csv`
- `vector_first_retrieval_metrics.csv`
- `vector_first_runtime_comparison.csv`
- `ann_vector_retrieval_metrics.csv`
- `ann_vector_runtime_comparison.csv`
- `ann_vector_recall_against_exact.csv`
- `ann_vector_stress_runtime.csv`
- `controlled_vector_speed_benchmark.csv`
- `controlled_vector_speed_detail.csv`

Current repeat-based controlled speed signal on local hardware:

- CNN seed1206 Eval C exact vector search median: 0.091 ms/query.
- CNN seed1206 Eval C HNSW m16/ef50 median: 0.043 ms/query.
- CNN seed1206 Eval C HNSW m32/ef50 median: 0.045 ms/query.

These are vector-retrieval-only timings. They do not include downstream
reranking, tree diagnostics, or rank/no-call calibration.

For the 12S/16S MarkerMirror branch, the strongest current candidate-generation
result is the union of cross-marker candidates and same-marker classical
alignment:

- MarkerMirror 12S->16S top50: 9.5 / 39.9 / 59.8 / 76.3% for
  species/genus/family/order.
- Same-marker 12S BLASTN top50: 0.0 / 90.7 / 95.1 / 99.4%.
- MarkerMirror + BLASTN top50 union: 9.5 / 92.1 / 95.3 / 99.7%.
- MarkerMirror + VSEARCH top50 union independently gives
  9.5 / 91.8 / 95.1 / 99.6%.

Interpretation: this is not species identification, because the held-out query
species are absent from the current same-marker reference by design. It is a
strong high-rank candidate-generation layer for rank-aware 12S/eDNA inference.

The current executable 12S wrapper uses this evidence for order/no-call only:

- default `stable_order`: 880/3,566 order calls at 99.7% diagnostic labelled
  precision;
- explicit `high_coverage_order`: nested BLASTN/VSEARCH top-10 order agreement
  gives 57.2% held-out coverage at 99.8% precision with target-0.99 met in all
  50 species-split repeats;
- family/genus/species are disabled. Threshold repair, set-valued output, and
  lineage/reference-coverage diagnostics have not produced stable target-0.99
  transfer for those ranks. VSEARCH-backed marker-resolvability now exists as
  a marker-ceiling diagnostic, and production-available VSEARCH cluster
  features have also been tested as policy inputs, but neither enables
  family/genus.
- active reference-curation/value-of-information is now source-tabled. The
  pipeline can rank which missing 12S/16S references or target-marker curation
  fixes are likely to improve current no-calls, while still keeping unsupported
  ranks disabled.

Target-host rerun wrapper:

- `experiments/paper1_phylo_calibrated_assignment/runs/09_vast_controlled_vector_speed.sh`

### Executable COI Pipeline

Implemented:

- `scripts/edna/run_paper1_coi_pipeline.py`
- `scripts/edna/build_paper1_pipeline_run_summary.py`
- `experiments/paper1_phylo_calibrated_assignment/runs/10_run_executable_coi_pipeline.sh`

This is the current runnable path:

```text
saved query embeddings
  -> exact vector or HNSW candidate retrieval
  -> optional train-reference p-distance reranking over top-k candidates
  -> top-10 taxonomic consensus features
  -> seen-test-derived missing-reference-aware thresholds
  -> species/genus/family/order/no-call assignment
```

Current calibrated executable CNN seed1206 target-0.99 results use exact
vector retrieval and `assignment_source=vector`:

- Eval C: 96.1% coverage, 90.0% assigned precision, 1.61% false species-call
  rate over all queries, 0.108 ms/query vector search.
- unseen-genera: 93.7% coverage, 83.7% assigned precision, 0.066% false
  species-call rate over all queries, 0.095 ms/query vector search.

Current HNSW executable rows are approximate vector-index checks using the same
rank/no-call policy:

- Eval C: 96.4% coverage, 89.5% assigned precision, 1.61% false species-call
  rate, 0.038 ms/query vector search.
- unseen-genera: 94.3% coverage, 82.8% assigned precision, 0.066% false
  species-call rate, 0.073 ms/query vector search.

Current p-distance rerank rows are experimental because the policy thresholds
were calibrated on vector ordering, not reranked ordering:

- Eval C: 96.7% coverage, 92.0% assigned precision, 0.73% false species-call
  rate, 0.184 ms/query candidate stage.
- unseen-genera: 93.8% coverage, 83.1% assigned precision, 0.219% false
  species-call rate, 0.186 ms/query candidate stage.

Rerank-specific calibration now exists:

- `scripts/edna/calibrate_paper1_pipeline_modes.py`
- `results/paper1_phylo_calibrated_assignment/pipeline_calibration/pipeline_mode_policy_summary.csv`
- `results/paper1_phylo_calibrated_assignment/pipeline_calibration/pipeline_mode_thresholds.csv`

At target 0.99, seen-test-derived p-distance rerank thresholds produce:

- Eval C: 95.8% coverage, 93.0% assigned precision, 0.0% false species-call
  rate.
- unseen-genera: 92.3% coverage, 83.9% assigned precision, 0.0% false
  species-call rate.

This calibrated p-distance mode makes no species calls at target 0.99. That is
not a weakness; it is the rank-adaptive policy doing what it should under
missing-reference uncertainty. It shifts claims toward genus/family/order unless
species evidence is strong enough.

Current executable pipeline tables:

- `pipeline_run_summary.csv`
- `pipeline_runs/*/pipeline_candidate_predictions.csv`
- `pipeline_runs/*/pipeline_rank_assignments.csv`
- `pipeline_runs/*/pipeline_summary.csv`

Production v1 packaging now exists:

- `scripts/edna/run_paper1_production_v1.py`
- `scripts/edna/run_paper1_raw_sequence_production_v1.py`
- `scripts/edna/run_paper1_fasta_inference_v1.py`
- `scripts/edna/build_paper1_raw_sequence_production_summary.py`
- `experiments/paper1_phylo_calibrated_assignment/runs/12_run_production_v1.sh`
- `results/paper1_phylo_calibrated_assignment/production_v1/production_v1_summary_all.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/raw_sequence_production_v1_summary.csv`
- `experiments/paper1_phylo_calibrated_assignment/PRODUCTION_CLI_V1.md`
- `experiments/paper1_phylo_calibrated_assignment/PRODUCTION_PIPELINE_V1.md`

Production v1 applies locked rerank-specific thresholds to existing
p-distance-reranked CNN seed1206 pipeline runs. At target 0.99 it gives:

- Eval C: 95.8% coverage, 93.0% assigned precision, 0.0% false species-call
  rate.
- unseen-genera: 92.3% coverage, 83.9% assigned precision, 0.0% false
  species-call rate.

This is the current strongest COI production-style operating point. Raw
split-sequence execution now works through
`run_paper1_raw_sequence_production_v1.py`, and specimen-style FASTA/CSV
execution now works through `run_paper1_fasta_inference_v1.py`. The remaining
packaging gap is a polished API/web wrapper, not the core inference path.

Current raw split-sequence Vast timing:

- Eval C: 46.77 seconds for 11,594 queries, 4.03 ms/query.
- seen-test: 49.80 seconds for 15,763 queries, 3.16 ms/query.
- unseen-genera: 47.12 seconds for 9,148 queries, 5.15 ms/query.

FASTA/CSV CLI smoke tests on Vast:

- CSV known-label smoke: 16 queries, 100.0% coverage, 87.5% precision if
  known, 0 species calls.
- FASTA unlabeled smoke: 8 queries, 100.0% coverage, precision unavailable by
  design, 0 species calls.
- DL decision-layer CSV smoke: 16 queries, 100.0% coverage, 100.0% precision
  if known, 0 species calls.
- DL decision-layer FASTA smoke: 8 queries, 100.0% coverage, precision
  unavailable by design, 0 species calls.

Caveat: the claim-ready executable rows are the exact-vector calibrated rows and
the separately calibrated p-distance rows. HNSW is a speed/approximation check.
p-distance reranking uses only `train_species_sequences.json`, so it avoids
held-out sequence leakage. The executable path still does not include ecological
context. The FASTA/CSV CLI is still a research CLI, not a deployed product.
Official APPLES and strict tree-pruned CNN validation now exist as
separate comparator/stress-test tracks, not inside the executable pipeline
runner.

### DL Evidence Rank-Backoff Layer

First trainable decision-layer script:

- `scripts/edna/train_paper1_coi_evidence_model.py`
- inference adapter: `scripts/edna/apply_paper1_coi_evidence_model.py`
- optional FASTA/CSV CLI mode: `--decision-mode dl_mlp_species_disabled`

This model learns per-rank correctness probabilities from the current
vector+p-distance evidence traces. It is trained on seen-test rows and
calibrated on held-out seen-test rows.

Species-disabled target-0.99 result:

- Eval C: 94.2% coverage, 97.4% assigned precision, 0.0% false species-call
  rate. Bootstrap 95% intervals: coverage 93.8-94.7%, precision 97.1-97.7%.
- unseen-genera: 88.5% coverage, 93.5% assigned precision, 0.0% false
  species-call rate. Bootstrap 95% intervals: coverage 87.8-89.2%, precision
  93.0-94.0%.

Seed-repeat range across MLP seeds 1206/1207/1208:

- Eval C: 94.2-96.0% coverage, 97.1-97.4% assigned precision, 0.0% false
  species-call rate.
- unseen-genera: 88.5-91.3% coverage, 92.9-93.5% assigned precision, 0.0%
  species-call rate.

Interpretation: this improves precision over hand-threshold production-v1 at
lower coverage. It is integrated into the FASTA/CSV CLI as an optional
species-disabled decision mode. It is not yet the default production policy
because strict hidden-taxonomy tests are still needed.

### 2. Tree Geometry

Implemented diagnostics:

- zero-shot/reference tree recovery
- zero-shot/zero-shot tree recovery
- sampled tree-distance versus embedding-distance bins
- neighborhood preservation
- reference-distance bins

Current tables:

- `tree_recovery_metrics.csv`
- `tree_distance_bin_summary.csv`
- `tree_distance_sample_summary.csv`
- `neighborhood_preservation.csv`
- `reference_diagnostics_summary.csv`

### 3. Missing-Reference / Rank Backoff

Implemented diagnostics and policy scaffolds:

- hide true species from candidate ranking
- hide true genus
- hide true family
- full-candidate embedding ablation where query embeddings exist
- seen-test-to-heldout threshold transfer
- missing-reference-aware consensus rank/no-call policy

Current tables:

- `candidate_ablation_rank_backoff.csv`
- `full_candidate_embedding_ablation.csv`
- `full_candidate_embedding_ablation_cnn_seed_repeats.csv`
- `prospective_rank_adaptive_policy_summary.csv`
- `missing_reference_aware_policy_summary.csv`
- `strict_missing_reference_summary.csv`

Current candidate locked operating point:

- CNN seed1206 with the 0.99 missing-reference-aware consensus policy:
  - Eval C: 90.0% assigned precision at 96.1% coverage.
  - unseen-genera: 83.7% assigned precision at 93.7% coverage.

Important caveat: the older candidate/ranking ablations are post-hoc. Strict
missing-reference claims should now use the completed pruned CNN runs where the
hidden taxon is absent from training/reference construction.

Strict validation prep:

- Input packs now exist under
  `data/phylo/paper1_strict_missing_reference_inputs/`.
- The packs remove hidden species/genus/family taxa before candidate-tree
  construction and before training/reference sequence construction.
- Current packs cover Eval C and unseen-genera with species/genus/family hidden.
- Vast/Linux runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/08_vast_strict_missing_reference_cnn.sh`.
- Current status: all six strict retrained CNN runs completed for Eval C and
  unseen-genera species/genus/family-hidden packs. Use
  `strict_missing_reference_summary.csv` and `strict_rank_backoff_summary.csv`
  for manuscript-facing strict missing-reference numbers.

### 4. Phylogenetic Placement

Current status:

- EPA-ng completed and scored for Eval C, seen-test, and unseen-genera.
- Current placed-clade containment diagnostics by species/genus/family/order:
  - Eval C: 0.0 / 45.9 / 67.8 / 74.3.
  - seen-test: 26.2 / 42.4 / 63.6 / 72.1.
  - unseen-genera: 0.0 / 0.0 / 37.0 / 51.0.
- pplacer is not valid yet because it needs a reference package or model stats
  file.
- Local APPLES-like p-distance placement has been run and scored for all three
  clean splits. It is a labelled distance-placement diagnostic, not official
  APPLES.
- Official APPLES 2.0.11 has been run and scored on the matched
  Fernando-style completeness sweeps.
- Placement comparator decision is now locked: keep EPA-ng as the completed
  likelihood-placement comparator, keep APPLES-like rows clearly labelled for
  clean split diagnostics, use official APPLES for the matched sweep matrix,
  and leave pplacer blocked unless a valid refpkg/stats model is supplied.
- Fernando-adjacent scoring now has two layers:
  - edge-to-sister diagnostics from the top EPA-ng jplace edge;
  - simulated-placement-tree PCP diagnostics, where each query is grafted onto
    its top-LWR edge before comparing sister support to the full fish tree.
- Matched Fernando-style completeness sweeps have completed for random and
  family-stratified 99/80/60/40/20% backbones with 3 replicates. Final outputs
  are in `fernando_completeness_final_30/`.

Current tables:

- `placement_rank_diagnostics_summary.csv`
- `placement_rank_diagnostics_per_query.csv`
- `placement_lwr_rank_summary.csv`
- `placement_clade_size_rank_summary.csv`
- `placement_rank_backoff_summary.csv`
- `apples_like_distance_placement_summary.csv`
- `placement_tree_error_summary.csv`
- `placement_pcp_like_summary.csv`
- `placement_simulated_tree_pcp_summary.csv`

Important caveat: current placement scoring is still not exact Fernando PCP.
The final sweep table is the closest current diagnostic, but exact Fernando
comparability would still need their exact species universe, preprocessing,
backbone construction, and PCP implementation.

### 5. 12S/eDNA Marker Ambiguity And Ecology

Implemented evidence:

- exact and near-exact 12S resolvability
- SSM/CNN 12S zero-shot candidate retrieval
- Global_eDNA ASV and sample-level validation
- learned co-occurrence reranking from RLS/OBIS and public FISHGLOB
- geography/range-only prior baselines from RLS and OBIS
- same-sample co-occurrence-only baseline that excludes the current query's own
  sequence evidence
- diagnostic Global_eDNA calibration curves

Current tables:

- `merged_12s_resolvability_summary.csv`
- `merged_12s_zero_shot_model_metrics.csv`
- `merged_global_edna_asv_metrics.csv`
- `merged_global_edna_sample_metrics.csv`
- `merged_global_edna_calibration_curves.csv`
- `merged_edna_evidence_arm_status.csv`

### 6. Pipeline Benchmark Ledger

Implemented:

- `scripts/edna/build_paper1_pipeline_benchmarks.py`

Outputs:

- `pipeline_component_status.csv`
- `pipeline_coi_method_benchmark.csv`
- `pipeline_placement_benchmark.csv`
- `pipeline_vector_index_benchmark.csv`
- `pipeline_edna_method_benchmark.csv`
- `pipeline_best_by_task.csv`
- `pipeline_next_actions.csv`
- `pipeline_end_to_end_summary.csv`

`pipeline_best_by_task.csv` reports best observed values. It does not mean the
method is claim-ready. Single-seed rows, split-transfer calibration without
confidence intervals, and partial placement outputs remain preliminary.

## Build Commands

```bash
python3 scripts/edna/build_paper1_source_tables.py
python3 scripts/edna/build_merged_paper1_edna_source_tables.py
python3 scripts/edna/score_fish_tree_placement_outputs.py \
  --placement-root results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/phylo_placement \
  --output-dir results/paper1_phylo_calibrated_assignment/source_tables
python3 scripts/edna/eval_apples_like_distance_placement.py \
  --splits eval_c seen_test unseen_genera \
  --candidate-source vsearch \
  --candidate-top-k 25
python3 scripts/edna/build_placement_tree_error_tables.py
python3 scripts/edna/bootstrap_rank_no_call_policy.py \
  --target-precision 0.99 \
  --n-bootstrap 1000
bash experiments/paper1_phylo_calibrated_assignment/runs/10_run_executable_coi_pipeline.sh
python3 scripts/edna/calibrate_paper1_pipeline_modes.py
bash experiments/paper1_phylo_calibrated_assignment/runs/12_run_production_v1.sh
python3 scripts/edna/build_paper1_pipeline_benchmarks.py
python3 scripts/edna/build_paper1_end_to_end_summary.py
```

## Logging Standard

From 2026-05-31 onward, Paper 1 scripts should expose progress while they run.

Python scripts should use `scripts/edna/progress_logging.py` and write
timestamped messages to both stdout and a file. The default local path is:

```text
results/paper1_phylo_calibrated_assignment/logs/{script_name}.log
```

Scripts that support a custom destination accept `--log-file`. Vast wrapper
scripts should continue to write a wrapper PID plus one log per long phase under
their result root, for example:

```text
results/paper1_phylo_calibrated_assignment/.../logs/*.log
```

This is now part of the experiment contract: every new long-running script
should log start, major stages, output paths, failures when caught, and done.

## What Is Left

1. Decide whether the completed Fernando-style sweep diagnostics are enough for
   comparator context, or whether we need to implement Fernando's exact PCP
   workflow.
2. Decide whether to add neural tree-space encoders to the same reduced-backbone
   completeness sweep matrix.
3. Add raw FASTA-to-embedding inference and end-to-end timing to production v1.
   Current speed rows cover vector retrieval and controlled retrieval stress
   tests, but not every downstream rerank/calibration step.
4. Improve the eDNA posterior if we want the mixed species-disabled target-95
   policy to reach 95% on held-out groups. The true nested run is complete and
   currently lands at 93.4% held-out accuracy for that mixed policy.
