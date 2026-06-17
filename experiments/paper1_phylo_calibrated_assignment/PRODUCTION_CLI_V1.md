# Production CLI V1

Last updated: 2026-06-02

## What This Is

`scripts/edna/run_paper1_fasta_inference_v1.py` is the first user-facing COI
inference wrapper for the Paper 1 pipeline.

It accepts specimen sequences as FASTA or CSV and runs:

```text
FASTA/CSV sequences
  -> temporary inference input pack
  -> CNN seed1206 barcode-to-tree embedding export
  -> exact vector candidate retrieval
  -> train-reference p-distance reranking over top candidates
  -> explicit decision layer
  -> species/genus/family/order/no-call output
```

Decision layers currently available:

- `production_thresholds`:
  locked target-0.99 hand-threshold rank/no-call policy.
- `dl_mlp_species_disabled`:
  trained MLP over vector+p-distance evidence, with species calls disabled.

This is a research CLI, not a deployed API. It is the correct next step between
split-table experiments and a real tool.

## Example Commands

CSV input:

```bash
python3 scripts/edna/run_paper1_fasta_inference_v1.py \
  --input-csv specimens.csv \
  --id-column sample_id \
  --sequence-column sequence \
  --output-dir results/paper1_phylo_calibrated_assignment/production_v1_cli/my_run
```

FASTA input:

```bash
python3 scripts/edna/run_paper1_fasta_inference_v1.py \
  --input-fasta specimens.fasta \
  --output-dir results/paper1_phylo_calibrated_assignment/production_v1_cli/my_fasta_run
```

On the Vast image, use the Torch environment explicitly:

```bash
/venv/main/bin/python scripts/edna/run_paper1_fasta_inference_v1.py \
  --input-fasta specimens.fasta \
  --output-dir results/paper1_phylo_calibrated_assignment/production_v1_cli/my_fasta_run \
  --python /venv/main/bin/python
```

DL decision-layer mode:

```bash
python3 scripts/edna/run_paper1_fasta_inference_v1.py \
  --input-csv specimens.csv \
  --sequence-column sequence \
  --decision-mode dl_mlp_species_disabled \
  --output-dir results/paper1_phylo_calibrated_assignment/production_v1_cli/my_dl_run
```

## Input Contract

FASTA:

- header ID becomes `query_id`;
- sequence lines are cleaned to A/C/G/T/N.

CSV:

- required sequence column: `nucleotides`, `sequence`, `seq`, or `barcode`;
- optional ID column: `processid`, `query_id`, `sample_id`, or `id`;
- optional known labels for evaluation only:
  `tree_label`, `species_name`, `genus_name`, `family_name`, `order_name`.

If known labels are absent, the CLI still makes assignments. It does not report
precision, because correctness is not knowable for unlabeled specimens.

## Output Contract

Main user-facing files:

- `inference_assignments.csv`
- `inference_summary.csv`
- `inference_manifest.json`

Audit files:

- `inference_input_pack/zero_shot_queries.csv`
- `embedding_export/query_embeddings.npz`
- `pipeline_run/pipeline_candidate_predictions.csv`
- `pipeline_run/pipeline_rank_assignments.csv`
- `production_v1/*/production_v1_assignments.csv`
- `dl_mlp_species_disabled_decision/coi_dl_evidence_applied_predictions.csv`
  when `--decision-mode dl_mlp_species_disabled` is used

Important `inference_assignments.csv` columns:

- `query_id`
- `decision_mode`
- `assigned_rank`
- `assigned_label`
- `assignment_reason`
- `pred_species`, `pred_genus`, `pred_family`, `pred_order`
- `top_tree_labels`
- `top_scores`
- `top_pdistances`
- `has_known_truth`
- `assigned_correct_if_known`

## Smoke Tests

Vast endpoint used:

- `ssh -p 23156 root@194.14.47.19`

CSV smoke with 16 known-label rows:

- output copied to
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_eval_c_known/`;
- coverage: 100.0%;
- assigned precision if known: 87.5%;
- no species calls;
- total time: 13.29 seconds.

FASTA smoke with 8 unlabeled rows:

- output copied to
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_unlabeled_fasta/`;
- coverage: 100.0%;
- assigned precision: unavailable, correctly, because no labels were supplied;
- no species calls;
- total time: 13.07 seconds.

DL decision-layer CSV smoke with 16 known-label rows:

- output copied to
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_eval_c_known_dl_rerun/`;
- coverage: 100.0%;
- assigned precision if known: 100.0%;
- no species calls;
- total time: 14.08 seconds.

DL decision-layer FASTA smoke with 8 unlabeled rows:

- output copied to
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_unlabeled_fasta_dl/`;
- coverage: 100.0%;
- assigned precision: unavailable, correctly, because no labels were supplied;
- no species calls;
- total time: 17.16 seconds.

These are smoke tests, not headline accuracy benchmarks. Small-batch timing is
dominated by Python/Torch startup and model loading. Use the full raw-sequence
timing table for throughput claims.

## Claim Boundary

Safe:

> The current COI pipeline can run from arbitrary FASTA/CSV specimen sequences
> into auditable species/genus/family/order/no-call assignments.

Not safe:

- this is a deployed web/API product;
- this supports 12S/eDNA inputs;
- unlabeled specimen precision is measurable without known labels;
- the CLI proves species-level assignment is solved.
