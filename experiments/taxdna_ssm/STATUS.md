# TAXDNA-Style SSM Status

Generated on 2026-05-26.

## Ready Locally

- Experiment protocol: `experiments/taxdna_ssm/protocol.yaml`
- Local prep runner: `experiments/taxdna_ssm/runs/01_local_prep.sh`
- Vast exact-Teleo runner: `experiments/taxdna_ssm/runs/02_vast_exact_teleo.sh`
- Vast broad-12S runner: `experiments/taxdna_ssm/runs/03_vast_broad_12s.sh`
- Local Global_eDNA ecological-prior runner: `experiments/taxdna_ssm/runs/04_local_global_edna_priors.sh`
- Vast learned co-occurrence runner: `experiments/taxdna_ssm/runs/05_vast_global_edna_learned_cooccurrence.sh`
- Run ledger: `configs/runs/2026-05-26-taxdna-ssm-track.json`

## Input Validation

Exact Teleo input:

- path: `data/edna/stalder_inputs/multisource_teleo`
- candidate species: 11,638
- reference species: 949
- reference sequences: 1,507
- zero-shot query species: 327
- zero-shot queries: 716
- max sequence length used: 128
- sequences over max length: 0

Broad 12S input:

- path: `data/edna/stalder_inputs/multisource`
- candidate species: 11,638
- reference species: 1,637
- reference sequences: 15,592
- zero-shot query species: 795
- zero-shot queries: 3,566
- max sequence length used: 2,048
- sequences over max length: 0

## SWARM / Real eDNA Preprocessing

Global_eDNA local files contain published SWARM MOTU tables, not raw FASTQ/FASTA
reads. The local prep exports these published SWARM representatives to FASTA for
TAXDNA-style downstream prediction:

- output: `data/edna/raw/real_edna/global_tropical_swarm_representatives`
- source MOTU tables: 14
- representative FASTA records: 116,997

This is not a de novo SWARM rerun from raw reads.

## Co-Occurrence Inputs

TAXDNA-shaped co-occurrence JSONs were built from independent ecological data:

- RLS survey groups: 2,047
- OBIS site-cell groups: 28
- combined groups: 2,075
- combined unique species: 1,553
- combined unique species with local reference sequences: 206

Outputs:

- `data/edna/cooccurrence_inputs/taxdna_ssm/rls_taxdna_cooccurrence.json`
- `data/edna/cooccurrence_inputs/taxdna_ssm/obis_taxdna_cooccurrence.json`
- `data/edna/cooccurrence_inputs/taxdna_ssm/rls_obis_taxdna_cooccurrence.json`

## Real eDNA Ecological-Prior Matrix

The current Global_eDNA matrix includes:

- Mamba sequence-only
- Mamba plus same-sample co-occurrence rerank
- Mamba plus RLS geographic prior
- Mamba plus OBIS occurrence prior
- Mamba plus RLS and OBIS priors
- RLS prior only
- OBIS prior only
- BLAST train-reference baseline

Summary table:

- `results/edna/global_tropical_validation/summary/global_edna_method_overall_metrics.csv`
- `results/edna/global_tropical_validation/summary/global_edna_method_stratified_metrics.csv`

## Learned Co-Occurrence Bridge

The NPZ-native learned co-occurrence bridge is now implemented:

- training: `scripts/edna/train_npz_cooccurrence_model.py`
- SSM prediction with query embedding export: `scripts/edna/predict_phylo_mamba_checkpoint.py`
- CNN prediction with query embedding export: `scripts/edna/predict_taxdna_cnn_checkpoint.py`
- evaluation: `scripts/edna/eval_global_edna_learned_cooccurrence.py`

This keeps the tree embedding space fixed across CNN and SSM arms, so the model
comparison is still controlled.

Remaining work is GPU execution after exact-Teleo CNN/SSM checkpoints are
available in `results/edna/taxdna_ssm/`.

Smoke checks completed locally:

- `scripts/edna/learn_tree_embedding_npz.py --dry-run`
- `scripts/edna/train_npz_cooccurrence_model.py --dry-run`
- one-epoch NPZ co-occurrence training on 5 groups
- learned co-occurrence evaluator on 2 synthetic Global_eDNA samples

Additional local support:

- `scripts/edna/build_global_edna_calibration_matrix.py` builds no-call calibration curves for every available Global_eDNA method.
- Learned co-occurrence evaluation is configured to sweep context weights 0.25, 0.50, 1.00, and 2.00 for both SSM and CNN.
- Vast training wrappers now launch one background wrapper PID per track and run model arms sequentially to avoid same-GPU contention.
- Result pull/refresh wrappers are prepared:
  - `experiments/taxdna_ssm/runs/06_pull_vast_results.sh`
  - `experiments/taxdna_ssm/runs/07_refresh_local_outputs.sh`
- Figure source tables and plot drafts now include Global_eDNA ecological-prior and calibration outputs:
  - `results/figures/source_data/figure_global_edna_prior_matrix.csv`
  - `results/figures/source_data/figure_global_edna_calibration.csv`
  - `results/figures/plots/global_edna_prior_matrix.pdf`
  - `results/figures/plots/global_edna_sample_jaccard.pdf`
  - `results/figures/plots/global_edna_calibration.pdf`
