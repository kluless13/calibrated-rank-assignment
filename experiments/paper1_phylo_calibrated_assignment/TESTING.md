# Paper 1 Testing

This file records how the current Paper 1 pipeline is tested. The aim is to
make every manuscript number traceable to a script, input, output table, and
claim boundary.

## Test Contract

Every new long-running Paper 1 script should:

- write timestamped progress logs;
- emit a manifest JSON where practical;
- write source tables under
  `results/paper1_phylo_calibrated_assignment/source_tables/` or a clearly
  named result root;
- state whether outputs are publication metrics, diagnostics, or stress tests.

## Recent Checks

### Exp 103 MarkerMirror Union Candidate Support

Commands run:

```bash
python3 -m py_compile scripts/edna/build_marker_mirror_union_candidate_support.py
python3 scripts/edna/build_marker_mirror_union_candidate_support.py
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_candidate_support_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_candidate_support_per_query.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_kmer_candidates_top50.csv.gz`

Claim boundary:
the same-marker arm is a TF-IDF character-kmer candidate-support audit. It is
not a final BLAST/VSEARCH/edlib alignment comparator.

### Exp 104 MarkerMirror Union Rank Policy

Commands run:

```bash
python3 -m py_compile scripts/edna/build_marker_mirror_union_rank_policy.py
python3 scripts/edna/build_marker_mirror_union_rank_policy.py
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_candidate_rank_policy/marker_mirror_union_production_candidates.csv.gz`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_static_policy_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_score_gate_validation_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_score_gate_validation_per_split.csv`

Claim boundary:
the production candidate table does not require hidden labels. The rank-policy
summaries are diagnostics; the score gates are species-split validated but not
field-eDNA production thresholds.

### Exp 105 MarkerMirror Union Evidence Compiler

Commands run:

```bash
python3 -m py_compile scripts/edna/train_marker_mirror_union_evidence_compiler.py
python3 scripts/edna/train_marker_mirror_union_evidence_compiler.py
python3 scripts/edna/train_marker_mirror_union_evidence_compiler.py \
  --enabled-ranks order \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_evidence_compiler_order_only \
  --log-file results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_evidence_compiler_order_only/marker_mirror_union_evidence_compiler_order_only.log
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_evidence_compiler_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_evidence_compiler_family_order_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_evidence_compiler_order_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_evidence_compiler_features.csv`

Claim boundary:
this is a diagnostic learned compiler. It did not beat the simpler source-
agreement or score-gate diagnostics, so it should not be described as the
production calibration solution.

### Exp 106 MarkerMirror Union Reason Codes

Commands run:

```bash
python3 -m py_compile scripts/edna/build_marker_mirror_union_reason_codes.py
python3 scripts/edna/build_marker_mirror_union_reason_codes.py \
  --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_union_reason_codes.log
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_reason_code_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_reason_code_by_source.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_reason_code_per_query.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_reference_curation_priorities.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_reason_code_manifest.json`

Claim boundary:
this is an explanation and curation-priority layer, not a new classifier.
Current 12S reference-gap labels are current-table/split-design diagnostics.
The same-marker candidate source is still a k-mer audit until replaced or
validated with BLAST/VSEARCH/edlib-style alignment evidence.

### Exp 107 Same-Marker Edlib Validation

Commands run locally/remotely:

```bash
python3 -m py_compile scripts/edna/build_marker_mirror_same_marker_edlib_validation.py
ssh -p 23156 root@194.14.47.19 'python3 -m pip install edlib'
ssh -p 23156 root@194.14.47.19 'cd /workspace/marinemamba && python3 -m py_compile scripts/edna/build_marker_mirror_same_marker_edlib_validation.py'
ssh -p 23156 root@194.14.47.19 'cd /workspace/marinemamba && python3 scripts/edna/build_marker_mirror_same_marker_edlib_validation.py --limit-queries 10 --output-dir results/paper1_phylo_calibrated_assignment/source_tables/edlib_smoke --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_same_marker_edlib_validation_smoke.log'
```

Full run:

```bash
python3 scripts/edna/build_marker_mirror_same_marker_edlib_validation.py \
  --output-dir results/paper1_phylo_calibrated_assignment/source_tables/edlib_same_marker_validation \
  --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_same_marker_edlib_validation_full.progress.log
```

The full run was executed on Vast and copied locally from:
`results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_edlib_validation/edlib_same_marker_validation/`.

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_edlib_support_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_edlib_support_per_query.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_edlib_candidates_top50.csv.gz`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_edlib_validation_manifest.json`

Claim boundary:
edlib reranks and validates the existing k-mer top50 candidate pool. It is not
full all-vs-all BLAST/VSEARCH search. The top10 edlib-reranked result is
0.0/87.8/94.3/98.8% species/genus/family/order support.

### Exp 108 MarkerMirror Union Listwise Selective Compiler

Commands run:

```bash
python3 -m py_compile scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py
python3 scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py \
  --repeats 10 \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler_smoke \
  --log-file results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler_smoke/marker_mirror_union_listwise_selective_compiler_smoke.log
python3 scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py \
  --repeats 50 \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler \
  --log-file results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler/marker_mirror_union_listwise_selective_compiler.log
python3 scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py \
  --enabled-ranks order \
  --repeats 50 \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler_order_only \
  --log-file results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler_order_only/marker_mirror_union_listwise_selective_compiler_order_only.log
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_listwise_selective_family_order_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_listwise_selective_order_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_listwise_selective_family_order_per_split.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_listwise_selective_order_per_split.csv`

Claim boundary:
the listwise compiler improves high-coverage order diagnostics but is not a
locked production policy. Order-only target-0.99 averages 83.1% coverage at
98.8% precision and meets target in 56% of species-split repeats.

### Exp 109 VSEARCH Same-Marker Candidate Generation

Commands run:

```bash
ssh -p 23156 root@194.14.47.19 'apt-get update -qq && apt-get install -y vsearch'
python3 -m py_compile \
  scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py \
  scripts/edna/build_marker_mirror_union_vsearch_candidate_support.py
scp -P 23156 scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py scripts/edna/progress_logging.py root@194.14.47.19:/workspace/marinemamba/scripts/edna/
ssh -p 23156 root@194.14.47.19 'cd /workspace/marinemamba && python3 -m py_compile scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py && python3 scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py --limit-queries 10 --output-dir results/paper1_phylo_calibrated_assignment/source_tables/vsearch_smoke --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_same_marker_vsearch_candidates_smoke.log --threads 16 --force'
ssh -p 23156 root@194.14.47.19 'cd /workspace/marinemamba; OUT=results/paper1_phylo_calibrated_assignment/source_tables/vsearch_same_marker_full; mkdir -p "$OUT"; nohup python3 -u scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py --output-dir "$OUT" --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_same_marker_vsearch_candidates_full.progress.log --threads 64 --force > "$OUT/build_marker_mirror_same_marker_vsearch_candidates_full.log" 2>&1 &'
python3 scripts/edna/build_marker_mirror_union_vsearch_candidate_support.py \
  --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_union_vsearch_candidate_support.log
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_vsearch_support_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_vsearch_support_per_query.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_vsearch_candidates_top50.csv.gz`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_vsearch_candidate_support_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_vsearch_candidate_support_per_query.csv`

Claim boundary:
VSEARCH global alignment is now a classical same-marker candidate source. It is
not BLAST local alignment, and it cannot recover species absent from the current
12S reference table.

### Exp 110 BLASTN Same-Marker Candidate Generation

Commands run:

```bash
ssh -p 23156 root@194.14.47.19 'apt-get update -qq && apt-get install -y ncbi-blast+'
python3 -m py_compile \
  scripts/edna/build_marker_mirror_same_marker_blast_candidates.py \
  scripts/edna/build_marker_mirror_union_blast_candidate_support.py
scp -P 23156 \
  scripts/edna/build_marker_mirror_same_marker_blast_candidates.py \
  scripts/edna/build_marker_mirror_union_blast_candidate_support.py \
  scripts/edna/progress_logging.py \
  root@194.14.47.19:/workspace/marinemamba/scripts/edna/
ssh -p 23156 root@194.14.47.19 'cd /workspace/marinemamba && python3 -m py_compile scripts/edna/build_marker_mirror_same_marker_blast_candidates.py scripts/edna/build_marker_mirror_union_blast_candidate_support.py && python3 -u scripts/edna/build_marker_mirror_same_marker_blast_candidates.py --limit-queries 10 --threads 8 --output-dir results/paper1_phylo_calibrated_assignment/source_tables/blast_same_marker_smoke --force --log-file results/paper1_phylo_calibrated_assignment/source_tables/blast_same_marker_smoke/blast_same_marker_smoke.log'
ssh -p 23156 root@194.14.47.19 'cd /workspace/marinemamba && mkdir -p results/paper1_phylo_calibrated_assignment/source_tables/blast_same_marker_full && nohup python3 -u scripts/edna/build_marker_mirror_same_marker_blast_candidates.py --threads 64 --output-dir results/paper1_phylo_calibrated_assignment/source_tables/blast_same_marker_full --force --log-file results/paper1_phylo_calibrated_assignment/source_tables/blast_same_marker_full/blast_same_marker_full.progress.log > results/paper1_phylo_calibrated_assignment/source_tables/blast_same_marker_full/blast_same_marker_full.stdout.log 2>&1 &'
python3 scripts/edna/build_marker_mirror_union_blast_candidate_support.py \
  --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_union_blast_candidate_support.log
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_blast_support_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_blast_support_per_query.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_same_marker_blast_candidates_top50.csv.gz`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_blast_candidate_support_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_blast_candidate_support_per_query.csv`

Claim boundary:
BLASTN local alignment is now a classical same-marker candidate source. It
cannot recover species absent from the current 12S reference table, and it is
not by itself a calibrated rank/no-call policy.

### Exp 111 BLAST/VSEARCH Calibration-Transfer Repair

Commands run:

```bash
python3 -m py_compile \
  scripts/edna/build_marker_mirror_blast_vsearch_calibration_repair.py
python3 scripts/edna/build_marker_mirror_blast_vsearch_calibration_repair.py \
  --repeats 50 \
  --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_blast_vsearch_calibration_repair.log
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_blast_vsearch_calibration_repair_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_blast_vsearch_calibration_repair_per_split.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_blast_vsearch_calibration_repair_thresholds.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_blast_vsearch_calibration_repair_features.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_blast_vsearch_calibration_repair_policy_rows.csv.gz`

Claim boundary:
This is a 50-repeat calibration-transfer diagnostic over production-available
BLAST/VSEARCH/MarkerMirror list features. It supports a conservative
all-source-agreement order policy, but the higher-coverage order rows are not
locked production thresholds.

### Exp 112 Stable Order Policy Handoff

Commands run:

```bash
python3 -m py_compile \
  scripts/edna/build_marker_mirror_stable_order_policy.py
python3 scripts/edna/build_marker_mirror_stable_order_policy.py
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_stable_order_policy_assignments.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_stable_order_policy_production_assignments.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_stable_order_policy_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_stable_order_policy_by_source.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_stable_order_policy_reason_counts.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_stable_order_policy_manifest.json`
- `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_stable_order_policy.log`

Result:
The unthresholded all-source top1 order policy assigns 886/3,566 full-query 12S
rows, 24.8% coverage, at 99.7% assigned precision and 0 false species calls.
The conservative max-repeat target-0.99 threshold assigns 880/3,566 rows, 24.7%
coverage, at 99.7% assigned precision and 0 false species calls.

Claim boundary:
This is the first explicit assignment/reason-code handoff for the stable order
policy. It is conservative and order-only; it does not solve species-level
identification or the higher-coverage rank/no-call calibration problem.
The production assignment output is label-stripped, but it still requires the
precomputed MarkerMirror/BLASTN/VSEARCH feature table upstream.

### Exp 114 MarkerMirror 12S Production Wrapper Dry-Run

Commands run:

```bash
python3 -m py_compile \
  scripts/edna/run_marker_mirror_12s_production_v1.py
python3 scripts/edna/run_marker_mirror_12s_production_v1.py \
  --input data/edna/stalder_inputs/multisource/zero_shot_queries.csv \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke \
  --limit 2 \
  --dry-run
python3 scripts/edna/build_marker_mirror_same_marker_blast_candidates.py \
  --query-table results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/input_queries/zero_shot_queries.csv \
  --same-marker-reference-dir data/edna/stalder_inputs/multisource \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/blast_smoke \
  --top-k 50 \
  --threads 4 \
  --log-file results/paper1_phylo_calibrated_assignment/logs/run_marker_mirror_12s_production_v1_blast_smoke.log
```

Outputs checked:

- `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/input_queries/zero_shot_queries.csv`
- `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/marker_mirror_12s_production_dependency_report.csv`
- `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/marker_mirror_12s_production_plan.json`
- `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/marker_mirror_12s_production_next_actions.csv`
- `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/marker_mirror_12s_production_manifest.json`
- `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/blast_smoke/marker_mirror_same_marker_blast_candidates_top50.csv.gz`

Result:
The wrapper normalized 2 existing 12S rows, wrote a complete plan for
MarkerMirror -> BLASTN -> VSEARCH -> feature table -> stable order policy, and
correctly reported that local VSEARCH is missing. BLASTN and makeblastdb are
available locally; the BLASTN stage smoke completed and produced 100 top-50
candidate rows.

Claim boundary:
This is a dependency-gated orchestration smoke, not a full production run. The
complete all-source order/no-call wrapper requires VSEARCH locally or execution
on Vast.

### Exp 115 MarkerMirror 12S Production Wrapper Full Vast Run

Command launched on Vast:

```bash
cd /workspace/marinemamba
python3 -u scripts/edna/run_marker_mirror_12s_production_v1.py \
  --input data/edna/stalder_inputs/multisource/zero_shot_queries.csv \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/vast_full_all_queries_20260603 \
  --device cuda \
  --threads 32
```

Copied local output root:

- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/`

Outputs checked:

- `marker_mirror_12s_production_assignments.csv`
- `marker_mirror_12s_production_manifest.json`
- `features/marker_mirror_12s_production_features.csv`
- `stable_order_policy/marker_mirror_stable_order_policy_summary.csv`
- `blast/marker_mirror_same_marker_blast_candidates_top50.csv.gz`
- `vsearch/marker_mirror_same_marker_vsearch_candidates_top50.csv.gz`

Result:
The full wrapper completed for all 3,566 current 12S query rows. It emitted 880
order calls and 2,686 no-calls. Because this benchmark input has known labels,
the diagnostic summary reports 24.7% coverage, 99.7% precision, and 0 false
species calls for the conservative max-repeat threshold. Runtime components:
15.0 s MarkerMirror, 254.6 s BLASTN, 48.8 s VSEARCH, 1.6 s stable policy.

Claim boundary:
This proves the current MarkerMirror 12S production-v1 research wrapper can run
end to end on Vast. It remains order/no-call only and should not be presented
as species-level identification.

### Exp 116 MarkerMirror 12S Unlabeled FASTA Smoke

Input created:

- `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/unlabeled_fasta_smoke_input/unlabeled_12s_queries.fa`

Command run on Vast:

```bash
cd /workspace/marinemamba
python3 -u scripts/edna/run_marker_mirror_12s_production_v1.py \
  --input results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/unlabeled_fasta_smoke_input/unlabeled_12s_queries.fa \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604 \
  --device cuda \
  --threads 16
```

Copied local output root:

- `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/`

Result:
The wrapper completed over 2 unlabeled FASTA records. It emitted 1 order call
and 1 no-call. The diagnostic precision/correctness fields are blank because no
truth labels were supplied, which is the expected production behavior.

Claim boundary:
This is an unlabeled smoke test of output behavior, not an accuracy estimate.

## Local Smoke Test

Run this before trusting a local refresh:

```bash
python3 -m py_compile \
  scripts/edna/eval_apples_like_distance_placement.py \
  scripts/edna/build_placement_tree_error_tables.py \
  scripts/edna/build_fernando_like_pcp_diagnostics.py \
  scripts/edna/build_fernando_simulated_placement_pcp.py \
  scripts/edna/build_fernando_completeness_sweeps.py \
  scripts/edna/build_controlled_vector_speed_benchmark.py \
  scripts/edna/bootstrap_rank_no_call_policy.py \
  scripts/edna/build_ann_vector_stress_benchmark.py \
  scripts/edna/build_strict_missing_reference_inputs.py \
  scripts/edna/build_strict_missing_reference_summary.py \
  scripts/edna/build_strict_rank_backoff_summary.py \
  scripts/edna/run_paper1_coi_pipeline.py \
  scripts/edna/build_paper1_pipeline_run_summary.py \
  scripts/edna/calibrate_paper1_pipeline_modes.py \
  scripts/edna/run_paper1_production_v1.py \
  scripts/edna/run_paper1_raw_sequence_production_v1.py \
  scripts/edna/run_paper1_fasta_inference_v1.py \
  scripts/edna/train_paper1_coi_evidence_model.py \
  scripts/edna/apply_paper1_coi_evidence_model.py \
  scripts/edna/build_paper1_dl_evidence_seed_summary.py \
  scripts/edna/build_paper1_dl_strict_apply_summary.py \
  scripts/edna/train_paper1_missing_reference_aware_calibrator.py \
  scripts/edna/build_paper1_missing_reference_calibrator_summary.py \
  scripts/edna/train_paper1_candidate_reranker.py \
  scripts/edna/build_paper1_candidate_reranker_summary.py \
  scripts/edna/train_paper1_reference_gap_detector.py \
  scripts/edna/train_paper1_reference_gap_detector_v2.py \
  scripts/edna/build_paper1_reference_gap_detector_summary.py \
  scripts/edna/build_paper1_gap_warning_overlay.py \
  scripts/edna/build_paper1_retrieval_dl_sweep_summary.py \
  scripts/edna/build_paper1_raw_sequence_production_summary.py \
  scripts/edna/build_paper1_manuscript_assets.py \
  scripts/edna/build_marker_mirror_manuscript_assets.py \
  scripts/edna/build_marker_mirror_manuscript_figures.py \
  scripts/edna/build_marker_mirror_slide_tables.py \
  scripts/edna/build_marker_mirror_manuscript_text.py \
  scripts/edna/build_marker_mirror_manuscript_section_outline.py \
  scripts/edna/build_marker_mirror_next_evidence_audit.py \
  scripts/edna/eval_global_edna_sample_cooccurrence_prior_only.py \
  scripts/edna/build_merged_paper1_edna_source_tables.py \
  scripts/edna/build_rank_adaptive_calibration.py \
  scripts/edna/build_species_disabled_edna_rank_backoff.py \
  scripts/edna/build_species_disabled_nested_calibration.py \
  scripts/edna/build_paper1_pipeline_benchmarks.py \
  scripts/edna/build_paper1_end_to_end_summary.py \
  scripts/edna/run_marker_mirror_candidate_generator.py \
  scripts/edna/build_marker_mirror_union_candidate_support.py \
  scripts/edna/build_marker_mirror_union_rank_policy.py \
  scripts/edna/train_marker_mirror_union_evidence_compiler.py \
  scripts/edna/build_marker_mirror_union_reason_codes.py \
  scripts/edna/build_marker_mirror_same_marker_edlib_validation.py \
  scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py \
  scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py \
  scripts/edna/build_marker_mirror_union_vsearch_candidate_support.py
```

## Rebuild Order

Use this order when regenerating local Paper 1 tables:

```bash
python3 scripts/edna/eval_apples_like_distance_placement.py \
  --splits eval_c seen_test unseen_genera \
  --candidate-source vsearch \
  --candidate-top-k 25

python3 scripts/edna/build_placement_tree_error_tables.py

python3 scripts/edna/build_fernando_like_pcp_diagnostics.py

python3 scripts/edna/build_fernando_simulated_placement_pcp.py

python3 scripts/edna/bootstrap_rank_no_call_policy.py \
  --target-precision 0.99 \
  --n-bootstrap 1000

python3 scripts/edna/build_ann_vector_stress_benchmark.py \
  --candidate-multipliers 1 5 10 25 \
  --hnsw-m 16 32 \
  --hnsw-ef-search 50 \
  --threads 8

python3 scripts/edna/build_controlled_vector_speed_benchmark.py \
  --repeats 5 \
  --warmup 1 \
  --hnsw-m 16 32 \
  --hnsw-ef-search 50 \
  --threads 8

python3 scripts/edna/build_strict_missing_reference_inputs.py
python3 scripts/edna/build_strict_missing_reference_summary.py
python3 scripts/edna/build_strict_rank_backoff_summary.py

bash experiments/paper1_phylo_calibrated_assignment/runs/10_run_executable_coi_pipeline.sh

python3 scripts/edna/calibrate_paper1_pipeline_modes.py
bash experiments/paper1_phylo_calibrated_assignment/runs/12_run_production_v1.sh
python3 scripts/edna/build_paper1_raw_sequence_production_summary.py
python3 scripts/edna/train_paper1_candidate_reranker.py \
  --epochs 50 \
  --patience 8 \
  --batch-size 4096 \
  --hidden-dim 96 \
  --dropout 0.1 \
  --cpu
python3 scripts/edna/build_paper1_candidate_reranker_summary.py
python3 scripts/edna/train_paper1_reference_gap_detector.py \
  --output-dir results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1206_no_counts_target099 \
  --target-gap-precision 0.99 \
  --epochs 100 \
  --patience 15 \
  --batch-size 1024 \
  --hidden-dim 64 \
  --dropout 0.1 \
  --cpu
python3 scripts/edna/build_paper1_reference_gap_detector_summary.py
python3 scripts/edna/build_paper1_gap_warning_overlay.py
python3 scripts/edna/build_paper1_missing_reference_calibrator_summary.py
python3 scripts/edna/build_paper1_retrieval_dl_sweep_summary.py
python3 scripts/edna/build_paper1_manuscript_assets.py

python3 scripts/edna/eval_global_edna_sample_cooccurrence_prior_only.py \
  --input-dir data/edna/real_edna_queries/global_tropical_multisource_teleo \
  --sample-query-map data/edna/real_edna_queries/global_tropical_multisource_teleo/sample_query_map.csv \
  --predictions results/remote_runs/2026-05-26/rtx_pro/global_edna_multisource_teleo_hier_strong_seed1207_predictions/zero_shot_candidate_predictions.csv \
  --output-dir results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_sample_cooccurrence_prior_only_top10

python3 scripts/edna/build_merged_paper1_edna_source_tables.py
python3 scripts/edna/build_species_disabled_edna_rank_backoff.py
python3 scripts/edna/build_species_disabled_nested_calibration.py
python3 scripts/edna/build_paper1_pipeline_benchmarks.py
python3 scripts/edna/build_paper1_end_to_end_summary.py
```

## Current Passing Checks

- EPA-ng placement outputs exist for Eval C, seen-test, and unseen-genera.
- APPLES-like local distance placement has been generated for all three splits.
- Matched nearest-reference tree-error tables have been generated.
- Fernando-like sister-clade diagnostics have been generated. These are not
  exact Fernando PCP.
- Simulated-placement-tree PCP diagnostics have been generated. These are
  closer to Fernando's generated-tree sister comparison, but still not exact
  Fernando PCP.
- Fernando-style completeness-sweep input packs have been generated for random
  and family-stratified 99/80/60/40/20% backbones with 3 replicates each.
  Local smoke preparation passed for `random_c99_rep01`.
- Fernando-style completeness sweeps have completed on Vast for both EPA-ng and
  official APPLES 2.0.11: 30 EPA-ng jplace outputs and 30 APPLES jplace
  outputs. Final scored source tables are under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_final_30/`.
- Missing-reference-aware rank/no-call policy has seen-test-derived thresholds,
  held-out Eval C/unseen-genera results, and bootstrap confidence intervals.
- ANN vector retrieval has exact/HNSW rows and synthetic 1x/5x/10x/25x stress
  timing rows.
- Controlled vector retrieval has repeat-based exact-vector and HNSW median
  latency rows for CNN seed1206 Eval C on the Vast RTX host.
- Executable COI pipeline runs exist for CNN seed1206 target-0.99 on Eval C and
  unseen-genera. The runner now regenerates exact-vector calibrated rows, HNSW
  approximate-vector rows, and train-reference p-distance rerank experimental
  rows for Eval C, seen-test, and unseen-genera.
- Rerank-specific seen-test calibration has been generated for executable
  pipeline modes. At target 0.99, calibrated p-distance reranking produces zero
  false species calls on Eval C and unseen-genera by backing off to
  genus/family/order instead of forcing species.
- Production v1 packaging has been generated for the current conservative COI
  operating point under
  `results/paper1_phylo_calibrated_assignment/production_v1/`.
- Raw split-sequence production-v1 runs have completed on the Vast RTX PRO
  6000 for seen-test, Eval C, and unseen-genera. Source table:
  `raw_sequence_production_v1_summary.csv`.
- FASTA/CSV production-v1 CLI smoke tests have passed on the Vast RTX PRO
  6000:
  - CSV known-label smoke: 16 queries, output copied to
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_eval_c_known/`.
  - FASTA unlabeled smoke: 8 queries, output copied to
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_unlabeled_fasta/`.
  - Unlabeled precision is correctly reported as unavailable rather than
    incorrect.
- First COI DL evidence/rank-backoff model trains and evaluates locally:
  `python3 scripts/edna/train_paper1_coi_evidence_model.py --epochs 80 --patience 12 --batch-size 1024 --hidden-dim 64 --dropout 0.1 --cpu`.
  Species-disabled target-0.99 improves held-out assigned precision while
  keeping false species calls at zero.
- The COI DL evidence/rank-backoff model can now be applied to existing
  pipeline runs with
  `scripts/edna/apply_paper1_coi_evidence_model.py`. Local adapter smoke on the
  held-out fish p-distance pipeline reproduced 94.2% coverage and 97.4%
  assigned precision with species disabled.
- FASTA/CSV CLI DL decision-mode smokes have passed on the Vast RTX PRO 6000:
  - CSV known-label smoke: 16 queries, 100.0% coverage, 100.0% precision if
    known, 0 species calls, copied to
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_eval_c_known_dl_rerun/`.
  - FASTA unlabeled smoke: 8 queries, 100.0% coverage, precision unavailable,
    0 species calls, copied to
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_unlabeled_fasta_dl/`.
- MarkerMirror candidate-generator smokes have passed:
  - CPU bounded smoke: one 12S query, 25 16S target species, top-5 candidates.
  - Vast GPU full-reference smoke: 32 held-out 12S queries, 1,865 16S target
    species, top-50 candidates, 1,600 candidate rows, copied to
    `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`.
  - Cache-backed repeat inference: write pass created the 2,971-row 16S
    embedding cache, read pass loaded it, and the candidate tables were exactly
    identical.
  - Candidate-generator evidence handoff: 1,600 cache-backed candidate rows were
    converted into evidence rows with 97 production numeric features, tree
    evidence, same-marker sequence checks, and marker-resolvability features.
  - Full-query candidate-generator handoff: 3,566 12S query rows produced
    178,300 candidate/evidence rows against the cached 16S reference.
  - Integrated rank/no-call apply path runs on the full-query handoff, but
    target-0.99 calibration does not transfer cleanly; species-disabled logistic
    is the current honest diagnostic row.
- COI DL evidence/rank-backoff seed repeats have been generated for MLP seeds
  1206, 1207, and 1208. Source tables:
  `dl_evidence_seed_summary.csv` and
  `dl_evidence_seed_bootstrap_summary.csv`.
- COI DL evidence/rank-backoff has been applied to all six strict hidden
  species/genus/family executable pipeline runs. Source table:
  `dl_evidence_strict_apply_summary.csv`.
- Diagnostic COI top-50 candidate rerankers have been trained and
  source-tabled. Source table: `candidate_reranker_summary.csv`. The
  BLAST/VSEARCH-aware retrieval-DL rerankers strongly improve held-out
  genus/family candidate ordering, but target-0.99 calibration transfer is
  still below production standard.
- Exp 66 refresh checks passed:
  `build_paper1_candidate_reranker_summary.py`,
  `build_paper1_pipeline_run_summary.py`,
  `build_paper1_pipeline_benchmarks.py`,
  `build_paper1_end_to_end_summary.py`, and
  `build_paper1_manuscript_assets.py`.
- Exp 67 tree-neighborhood reranker checks passed:
  remote syntax/compile preflight, queue completion on Vast, local copy, and
  refresh of `build_paper1_candidate_reranker_summary.py`,
  `build_paper1_pipeline_benchmarks.py`,
  `build_paper1_end_to_end_summary.py`, and
  `build_paper1_manuscript_assets.py`.
- Exp 68 query-listwise tree10 reranker checks passed:
  local/remote compile preflight, queue completion on Vast, local copy, and
  refresh of `build_paper1_candidate_reranker_summary.py`,
  `build_paper1_pipeline_benchmarks.py`,
  `build_paper1_end_to_end_summary.py`, and
  `build_paper1_manuscript_assets.py`. Result was negative versus the
  pointwise tree10 reranker, so it is kept as model-development evidence rather
  than a production candidate.
- Exp 69 calibration-transfer checks passed:
  `build_paper1_candidate_reranker_calibration_transfer.py` compiled and ran
  locally with progress logging, writing
  `candidate_reranker_calibration_transfer.csv` and its manifest. This audit
  confirms candidate-level reranker thresholds do not yet transfer cleanly
  enough for production rank/no-call use.
- Exp 70 pairwise tree10 reranker checks passed:
  local compile, local runner syntax, remote compile, remote runner syntax,
  queue completion on Vast, local copy, and refresh of
  `build_paper1_candidate_reranker_summary.py`,
  `build_paper1_candidate_reranker_calibration_transfer.py`,
  `build_paper1_pipeline_benchmarks.py`,
  `build_paper1_end_to_end_summary.py`, and
  `build_paper1_manuscript_assets.py`. Result was mixed versus pointwise
  tree10, so it is kept as a diagnostic alternative.
- Exp 71 selected-assignment calibrator checks passed:
  `train_paper1_candidate_assignment_calibrator.py` and
  `build_paper1_candidate_assignment_calibrator_summary.py` compiled locally
  and remotely, the Vast queue completed, outputs were copied locally, and
  ledgers were refreshed. Result did not meet missing-reference production
  calibration targets.
- Exp 72 reference-gap follow-up checks passed:
  target-0.95 and target-0.99 no-counts seed1301 runs completed locally with
  progress logging, `build_paper1_reference_gap_detector_summary.py` refreshed
  `reference_gap_detector_summary.csv`, and the pipeline/manuscript ledgers
  refreshed.
- Exp 73 candidate-evidence reference-gap v2 checks passed:
  target-0.95 and target-0.99 tree-aware runs completed locally with progress
  logging, `build_paper1_reference_gap_detector_summary.py` refreshed
  `reference_gap_detector_summary.csv`, and the pipeline/end-to-end/manuscript
  ledgers refreshed.
- Exp 74 candidate-evidence reference-gap no-tree ablation checks passed:
  target-0.95 no-tree run completed locally with progress logging,
  `build_paper1_reference_gap_detector_summary.py` refreshed
  `reference_gap_detector_summary.csv`, and the pipeline/end-to-end/manuscript
  ledgers refreshed.
- Exp 75 gap-warning overlay checks passed:
  `scripts/edna/build_paper1_gap_warning_overlay.py` compiled and ran locally,
  then wrote `gap_warning_overlay_summary.csv`,
  `gap_warning_overlay_examples.csv`, and
  `gap_warning_overlay_manifest.json`. This is a diagnostic overlay over
  existing production-v1 assignments and v2 reference-gap probabilities; no new
  model was trained.
- Exp 76 missing-reference-aware calibrator checks passed:
  `scripts/edna/train_paper1_missing_reference_aware_calibrator.py` compiled
  and trained locally with progress logging. Calibration loss dropped from
  0.499 to 0.245 over 100 epochs. The collector
  `scripts/edna/build_paper1_missing_reference_calibrator_summary.py` compiled
  and wrote `missing_reference_aware_calibrator_summary.csv` plus manifest.
- First diagnostic COI reference-gap detector has been trained and
  source-tabled. Current strongest diagnostic run:
  `reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_target095/`.
  Source table: `reference_gap_detector_summary.csv`. It is not
  production-ready: candidate-level evidence improves hidden species/genus/
  family gap recall, but normal supported false-warning rates remain too high
  for a standalone reason layer.
- Retrieval-DL sweep has completed for four arms:
  CNN contrastive, CNN hybrid, CNN hierarchical contrastive, and Transformer
  hierarchical contrastive. Outputs were copied to
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_retrieval_dl_sweep/`.
  Source tables:
  `retrieval_dl_sweep_summary.csv`,
  `retrieval_dl_sweep_tree_recovery.csv`, and
  `retrieval_dl_sweep_training_history.csv`.
- Manuscript asset inventories have been generated under
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/`.
- Strict missing-reference input packs exist for Eval C and unseen-genera with
  species/genus/family hidden before candidate tree and training construction.
- Strict missing-reference retrained CNN runs have completed for all 6 Eval
  C/unseen-genera species/genus/family-hidden packs.
- Strict rank-backoff source rows have been generated from the completed
  strict runs.
- Hidden-species scoring regression passes:
  `python3 -m unittest tests.test_eval_zero_shot_candidate_predictions`.
- Global_eDNA evidence decomposition now has sequence/tree, learned
  co-occurrence, geography-only, and same-sample co-occurrence-only arms.
- Full sequence+tree Eco-Phylo posterior outputs exist for all 6,995,880
  candidate rows.
- Species-disabled eDNA rank-backoff has been generated. On held-out groups,
  target-95 assigns 40.3% of queries at 94.3% accuracy, with species calls
  intentionally disabled.
- Nested threshold-stability testing over 30 calibration resplits has been
  generated. On the original held-out eDNA groups, target-95 averages 40.2%
  assignment at 94.3% accuracy, with 5th-95th percentile accuracy 93.9-94.9%.
- True nested eDNA posterior fit has been generated. The model is fit on 70%
  of calibration groups, thresholds are learned on the remaining calibration
  groups, and evaluation is on original held-out groups. Species still fails;
  family/order target-95 transfer individually, and mixed species-disabled
  target-95 assigns 38.9% of held-out queries at 93.4% accuracy.
- MarkerMirror 12S/16S integrated evidence has completed a three-seed
  candidate-export/evidence-join/calibration stability pass. Target-0.99
  learned MarkerMirror averages 51.0% coverage at 98.9% assigned precision for
  12S->16S and 71.1% coverage at 98.7% assigned precision for 16S->12S.
- MarkerMirror marker-resolvability prototype has been generated and wired into
  the seed1903 evidence compiler. At 0.99 proxy identity, 12S and 16S are each
  about 92% species-resolvable in the current references. The enhanced
  seed1903 compiler gives the same best target-0.99 operating point as the
  prior seed1903 compiler.
- MarkerMirror production-handoff research CLI has been added:
  `scripts/edna/run_marker_mirror_candidate_generator.py`. It passed a bounded
  CPU smoke test with one 12S query, 25 16S target species, and top-5
  candidates. Outputs are under
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_smoke_12s_to_16s/`.
- MarkerMirror full-query production-style handoff has been generated and
  diagnosed. The full run covers 3,566 12S query rows and 178,300 top-50
  candidate/evidence rows. The integrated rank/no-call apply path runs, but the
  nominal target-0.99 threshold does not transfer. The calibration-transfer
  diagnostic source tables show that the full handoff has only 26.6% query
  species coverage in the 16S target reference, versus 100.0% in the controlled
  validation split.
- MarkerMirror reference-aware policy repair sweep has been generated. The
  best higher-coverage production-safe diagnostic gate is top-1 MarkerMirror
  score >= 0.620484, which gives 5.83% coverage at 95.67% assigned precision
  with species calls disabled. A stricter top-1 score gate >= 0.697663 gives
  3.25% coverage at 100.00% assigned precision. These are labelled-handoff
  diagnostics, not locked independent production thresholds.
- MarkerMirror independent reference-aware validation has been generated.
  Across 50 query-species splits, target-0.95 gates average 5.79% held-out
  coverage at 94.39% precision and meet the target in 48% of repeats.
  Target-0.99 gates average 4.13% held-out coverage at 98.27% precision and
  meet target in 70% of repeats. Source-holdout checks show MitoHelper and
  rCRUX are more stable than the small Mare-MAGE subset.
- MarkerMirror high-coverage order repair has been generated. Exp 117 evaluates
  BLASTN/VSEARCH top-10 order agreement with nested species-split threshold
  locking. The best row uses nested global Wilson95 thresholds and reaches
  57.2% mean held-out coverage at 99.8% mean precision, with target-0.99 met in
  100% of 50 outer repeats. The full-table locked diagnostic assigns
  2,513/3,566 rows at 99.8% labelled precision. This is a high-coverage
  diagnostic, not the default CLI mode.
- MarkerMirror explicit decision-mode wiring has been smoked on Vast. The 12S
  wrapper now supports `stable_order` and `high_coverage_order`. On a 4-row
  labelled smoke, stable mode assigned 1 order call and high-coverage mode
  assigned 3 order calls; both were 100% precise on assigned rows. On 2-row
  unlabeled FASTA smokes, stable mode assigned 1 order call and high-coverage
  mode assigned 2 order calls, with precision/correctness blank by design.
- MarkerMirror family/genus high-coverage repair has been tested with nested
  species-split calibration across all available BLAST/VSEARCH/MarkerMirror
  policy rows. Family reached 35.5% mean coverage at 99.35% mean precision but
  met target-0.99 in only 94% of repeats. Genus reached 7.8% mean coverage at
  99.79% mean precision but met target in only 98% of repeats. Neither rank is
  enabled in the wrapper.
- MarkerMirror hierarchical candidate sets have been tested as a different
  family/genus strategy. Full-query family set coverage peaks at 95.4% only with
  a mean set size of 34.4 families; genus peaks at 92.4% with a mean set size of
  79.6 genera. This confirms that family/genus remain unsupported at target-0.99
  under current evidence.
- MarkerMirror manuscript assets have been generated with progress logging:
  `scripts/edna/build_marker_mirror_manuscript_assets.py` compiled and wrote
  candidate-support, order-policy, rank-boundary, runtime, figure-plan, methods,
  and manifest files under
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/`.
  These are writing assets, not new metrics.
- MarkerMirror figure drafts have been generated with progress logging:
  `scripts/edna/build_marker_mirror_manuscript_figures.py` compiled and wrote
  PNG/PDF candidate-support, order-policy, rank-boundary, and runtime figures
  plus `marker_mirror_slide_ready_summary.md` under
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/`.
  These are rendered drafts from existing metrics.
- MarkerMirror slide-ready tables have been generated with progress logging:
  `scripts/edna/build_marker_mirror_slide_tables.py` compiled and wrote CSV/MD
  slide tables plus `marker_mirror_slide_package_outline.md` under
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/`.
  These are coauthor/deck assembly artifacts, not new metrics.
- MarkerMirror manuscript text snippets have been generated with progress
  logging: `scripts/edna/build_marker_mirror_manuscript_text.py` compiled and
  wrote figure captions, results text, methods text, claim-boundary text, and a
  caption inventory under
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`.
  These are manuscript drafting artifacts, not new metrics.
- MarkerMirror manuscript section outline has been generated with progress
  logging: `scripts/edna/build_marker_mirror_manuscript_section_outline.py`
  compiled and wrote `marker_mirror_manuscript_section_outline.md`,
  `marker_mirror_manuscript_section_checklist.csv`, and
  `marker_mirror_manuscript_section_manifest.json` under
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`.
  These are manuscript organization artifacts, not new metrics.
- MarkerMirror next-evidence audit has been generated with progress logging:
  `scripts/edna/build_marker_mirror_next_evidence_audit.py` compiled and wrote
  `marker_mirror_next_evidence_source_audit.csv`,
  `marker_mirror_reference_coverage_by_lineage.csv`,
  `marker_mirror_next_evidence_source_manifest.json`, and
  `marker_mirror_family_genus_next_evidence_plan.md`. This is a planning
  artifact for future family/genus work, not an enabled rank/no-call result.
- MarkerMirror lineage/reference-coverage policy diagnostic has been generated
  with progress logging:
  `scripts/edna/build_marker_mirror_reference_coverage_policy_diagnostic.py`
  compiled, completed a 50-repeat species-split diagnostic, and wrote
  `marker_mirror_reference_coverage_policy_diagnostic_*.csv/json` source
  tables. A two-repeat `/tmp` smoke also passed after filtering inactive
  rank-specific feature columns. The result is diagnostic only: lineage
  reference coverage alone did not unlock stable family/genus transfer.
- MarkerMirror VSEARCH-backed marker-resolvability diagnostic has completed on
  Vast with progress logging:
  `scripts/edna/build_12s_near_exact_resolvability.py` ran 12S and 16S
  VSEARCH `cluster_fast` at 0.99, 0.98, 0.97, and 0.95 identity. Outputs were
  copied under
  `results/remote_runs/2026-06-04/rtx_pro_6000/marker_mirror_vsearch_resolvability_20260604/`
  and summarized into `marker_mirror_vsearch_resolvability_*.csv/json` source
  tables. This is alignment-backed evidence-source hardening, not an enabled
  family/genus policy.
- MarkerMirror VSEARCH-resolvability-aware policy diagnostic has been generated
  with progress logging:
  `scripts/edna/build_marker_mirror_vsearch_resolvability_policy_diagnostic.py`
  compiled and completed 50 species-split repeats. It wrote
  `marker_mirror_vsearch_resolvability_policy_diagnostic_*.csv/json` source
  tables. Hidden oracle-support columns were excluded from model features. The
  result is diagnostic only: VSEARCH cluster features did not unlock stable
  family/genus transfer.
- MarkerMirror active reference-curation/value-of-information tables have been
  generated with progress logging:
  `scripts/edna/build_marker_mirror_active_reference_value.py` compiled and
  completed. It wrote `marker_mirror_active_reference_value_*.csv/json` source
  tables and logs to
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_active_reference_value.log`.
  This is a curation-priority diagnostic, not an enabled family/genus policy.

## Known Non-Passing / Partial Checks

- Exact Fernando PCP has not been reproduced. The completed
  `fernando_completeness_final_30/` tables are matched Fernando-style
  diagnostics for our public setup, not an exact reproduction of Fernando's
  reference set, backbone extraction, or PCP implementation.
- pplacer is blocked unless a valid `PPLACER_REFPKG` or `PPLACER_STATS` model
  is supplied.
- Vector timing now includes controlled Vast timing for CNN seed1206 Eval C.
  Broader deployment claims still need larger reference stress tests and
  end-to-end rerank/calibration latency.
- p-distance rerank rows without `pipeline_calibration/` thresholds are
  experimental. The calibrated p-distance rows should be interpreted alongside
  the strict tree-pruned CNN validation; they are rank-backoff evidence, not
  proof of species-level assignment.
- Production v1 can now run from clean split `zero_shot_queries.csv` sequences
  and from specimen-style FASTA/CSV input. API/web packaging is not yet
  implemented.
- Older eDNA top-1 score-threshold calibration is diagnostic only. The stronger
  eDNA claim should use the full sequence+tree Eco-Phylo posterior with species
  disabled.
- The eDNA nested check is threshold stability only. It is not a full nested
  posterior model retrain, so the final manuscript should state that boundary.
- The true nested eDNA posterior run is a stronger check, but its mixed
  species-disabled target-95 policy lands below 95% held-out accuracy. Use the
  exact held-out accuracy, not the calibration target name, when writing.
- Older MarkerMirror 0.99 resolvability rows used a rare-kmer prefix-identity
  proxy. Claim-facing marker-ceiling discussion should now use the Exp129
  VSEARCH-backed source tables instead.
- MarkerMirror is wired into a 12S research wrapper for conservative
  order/no-call output, but not into a deployed production API or species-level
  identifier. Species calls remain disabled because calibration transfer failed
  on the full-query table. The higher-coverage Exp 117 order repair is now
  exposed as an explicit high-coverage diagnostic mode, but it is still
  order-only and not the default. Current MarkerMirror evidence is controlled
  marker-reference validation plus production-style handoff diagnostics, not
  production field-eDNA validation.
- Exp131 active reference-curation tables are action-priority diagnostics, not
  assignment probabilities.

## GPU Needed Later

No GPU is needed for the current table/documentation pass.

GPU is needed only for:

- new neural retrains or seed repeats;
- Mamba query-embedding export on a compatible CUDA/PyTorch image;
- pretrained barcode encoder benchmarks;
- any strict tree-pruned retraining experiment where hidden taxa are removed
  before representation learning.
