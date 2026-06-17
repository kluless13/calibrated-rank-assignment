# TAXDNA-Style SSM Experiment

This folder is a thin orchestration layer for a controlled TAXDNA-style experiment.
The purpose is to keep the biological setup fixed and compare the DNA sequence
encoder:

- TAXDNA-style CNN baseline
- MarineMamba/BarcodeMamba SSM encoder

The reusable code remains in `scripts/edna/`. Input data remain in
`data/edna/stalder_inputs/`, `data/edna/real_edna_queries/`, and
`data/edna/cooccurrence_inputs/`. Outputs should go under
`results/edna/taxdna_ssm/`.

## Protocol Shape

1. Build or verify Stalder/TAXDNA-style sequence inputs.
2. Learn or reuse the same phylogenetic tree embeddings.
3. Train sequence-to-tree models with the same input/candidate universe.
4. Evaluate sequence-only zero-shot candidate ranking.
5. Apply ecological context as separate ablations:
   - RLS prior only
   - OBIS prior only
   - sequence plus RLS
   - sequence plus OBIS
   - sequence plus RLS and OBIS
   - learned co-occurrence model when trained
6. Report species, genus, family, and order top-k metrics separately.

## Run Wrappers

- `runs/01_local_prep.sh`: local dry-runs plus SWARM representative and co-occurrence JSON export.
- `runs/02_vast_exact_teleo.sh`: shared tree embedding, exact-Teleo CNN, exact-Teleo SSM.
- `runs/03_vast_broad_12s.sh`: shared tree embedding, broad-12S CNN, broad-12S SSM.
- `runs/04_local_global_edna_priors.sh`: transparent RLS/OBIS prior controls plus calibration curves.
- `runs/05_vast_global_edna_learned_cooccurrence.sh`: NPZ learned co-occurrence training plus CNN/SSM real-eDNA community-context evaluation with context-weight sweep.
- `runs/06_pull_vast_results.sh`: copy TAXDNA-style outputs from the H200 and RTX Vast hosts into `results/remote_runs/`.
- `runs/07_refresh_local_outputs.sh`: regenerate Global_eDNA summaries, calibration tables, canonical ledger, source tables, and plot drafts after results are copied back.

## Figure Outputs

Source tables are generated under `results/figures/source_data/` and plot drafts
under `results/figures/plots/`. The current figure set covers COI Eval C, 12S
reference Eval C, BLAST abstention, dataset coverage, fish tree recovery, and
Global_eDNA ecological-prior/calibration results.

## Active Inputs

- Broad 12S input: `data/edna/stalder_inputs/multisource`
- Closest Teleo input: `data/edna/stalder_inputs/multisource_teleo`
- Real eDNA query input: `data/edna/real_edna_queries/global_tropical_multisource_teleo`
- Co-occurrence JSONs: `data/edna/cooccurrence_inputs/taxdna_ssm`
- OBIS occurrence prior: `data/edna/raw/real_edna/global_obis_range_prior_site20_pad05`

## Claim Boundary

This is a TAXDNA-style controlled reproduction with an SSM substitution. It is
not an exact reproduction unless the exact Stalder reference database, tree,
community data, and preprocessing are used.
