# MarkerMirror 12S CLI

Last updated: 2026-06-04

## Purpose

`scripts/edna/run_marker_mirror_12s_production_v1.py` is the current
research CLI for conservative 12S order/no-call inference.

```text
12S FASTA/CSV
  -> MarkerMirror 12S->16S candidate generation
  -> BLASTN same-marker candidate generation
  -> VSEARCH same-marker candidate generation
  -> all-source feature table
  -> stable order/no-call policy
```

It does not emit species, genus, or family calls. The only final calls are:

- `order`, when MarkerMirror, BLASTN, and VSEARCH agree on top-1 order and the
  locked threshold is met;
- `no_call`, otherwise, with a reason code.

The default decision mode is conservative:

```bash
--decision-mode stable_order
```

An explicit research diagnostic mode is also available:

```bash
--decision-mode high_coverage_order
```

`high_coverage_order` uses BLASTN/VSEARCH top-10 order agreement from Exp 117.
It remains order/no-call only and should be described as a high-coverage
diagnostic mode, not species identification.

## Dependencies

Required:

- Python environment with the repo dependencies;
- BLASTN and makeblastdb;
- VSEARCH;
- MarkerMirror projection head:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best_seed1903/marker_mirror_shared_projection_head.pt`;
- 16S target embedding cache:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/cache/marker_mirror_16s_nt_v2_50m_fullref_embeddings.npz`;
- locked threshold table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_blast_vsearch_calibration_repair_thresholds.csv`.

Local status: BLASTN and makeblastdb are installed; VSEARCH is not installed
locally. Full execution currently runs on Vast.

## Run

Example on Vast:

```bash
cd /workspace/marinemamba
python3 -u scripts/edna/run_marker_mirror_12s_production_v1.py \
  --input path/to/queries.fa \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/my_run \
  --decision-mode stable_order \
  --device cuda \
  --threads 32
```

High-coverage diagnostic mode:

```bash
cd /workspace/marinemamba
python3 -u scripts/edna/run_marker_mirror_12s_production_v1.py \
  --input path/to/queries.fa \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/my_high_coverage_run \
  --decision-mode high_coverage_order \
  --device cuda \
  --threads 32
```

For a local dependency/plan check:

```bash
python3 scripts/edna/run_marker_mirror_12s_production_v1.py \
  --input path/to/queries.fa \
  --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run \
  --dry-run
```

## Main Outputs

- `marker_mirror_12s_production_assignments.csv`: final order/no-call table,
  including `assignment_reason` for the emitted reason code.
- `marker_mirror_12s_production_manifest.json`: run manifest and timings.
- `marker_mirror_12s_production_dependency_report.csv`: tool/data availability.
- `marker_mirror_12s_production_plan.json`: commands for each stage.
- `features/marker_mirror_12s_production_features.csv`: all-source feature table.
- `blast/marker_mirror_same_marker_blast_candidates_top50.csv.gz`: BLASTN candidates.
- `vsearch/marker_mirror_same_marker_vsearch_candidates_top50.csv.gz`: VSEARCH candidates.
- `marker_mirror/marker_mirror_candidate_generator_candidates.csv`: MarkerMirror candidates.
- `stable_order_policy/marker_mirror_stable_order_policy_summary.csv`: policy summary.

## Current Validation

Full labelled Vast run:

- output root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/`;
- input: 3,566 current 12S query rows;
- final output: 880 order calls and 2,686 no-calls;
- diagnostic precision: 99.7%;
- false species calls: 0;
- runtime: 15.0 s MarkerMirror, 254.6 s BLASTN, 48.8 s VSEARCH, 1.6 s stable policy.

High-coverage diagnostic not yet exposed as default CLI behavior:

- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_high_coverage_order_repair_summary.csv`;
- best nested row: BLASTN/VSEARCH top-10 order agreement with nested global
  Wilson95 locking;
- held-out species-split result: 57.2% coverage at 99.8% assigned precision,
  target-0.99 met in 100% of 50 outer repeats;
- full-table locked diagnostic: 2,513/3,566 order calls, 70.5% coverage, 99.8%
  labelled precision.

This diagnostic is now exposed as explicit `--decision-mode high_coverage_order`
but is not the default CLI behavior.

Vast smoke tests after wiring:

- labelled stable mode:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_stable_labelled_20260604/`;
- labelled high-coverage mode:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_high_coverage_labelled_20260604/`;
- unlabeled stable mode:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_stable_unlabeled_20260604/`;
- unlabeled high-coverage mode:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_high_coverage_unlabeled_20260604/`.

Unlabeled FASTA smoke:

- output root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/`;
- input: 2 FASTA records with no taxonomy labels;
- final output: 1 order call and 1 no-call;
- diagnostic precision fields are blank because no truth labels were supplied.

## Claim Boundary

This is an executable research pipeline for conservative high-rank 12S
inference. It is not species-level identification, not field-eDNA validation,
and not a deployed production API.

Family/genus are intentionally disabled. Threshold repair, set-valued output,
and lineage/reference-coverage policy diagnostics have all been tested without
stable target-0.99 transfer. Enabling those ranks requires genuinely new
evidence, such as alignment-backed marker-resolvability or sample-aware
geography/co-occurrence, not only a different threshold on the current tables.
VSEARCH-backed marker-resolvability now exists as a source-table diagnostic,
but it is not itself an enabled family/genus policy.
Production-available VSEARCH cluster features were also tested in a learned
policy diagnostic and still did not stabilize family/genus transfer.

Active reference-curation/value-of-information tables now exist as source
tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_species.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_actions.csv`

These are not CLI assignment outputs. They are benchmark-side curation
priorities for deciding which missing 12S/16S references, target-marker fixes,
or multi-marker/context additions would most likely improve future no-calls.
