# Stalder/TAXDNA Reproduction

This folder is for an exact-as-possible reproduction of Stalder et al.'s
TAXDNA setup before swapping the DNA encoder for MarineMamba/BarcodeMamba.
It is intentionally separate from `experiments/taxdna_ssm/`, which is our
independent TAXDNA-style experiment.

## Goal

Reproduce the official TAXDNA baseline first, then change only the sequence
encoder.

The controlled comparison should hold fixed:

- official TAXDNA tree
- official candidate species universe
- official reference sequences
- official seen/unseen species split
- official co-occurrence data
- official metrics and reporting ranks

Only after the official CNN baseline is reproducible should we swap the DNA
network for the SSM encoder.

## Official Assets

Primary source:

- `https://gitlab.renkulab.io/dnai/TAXDNA.git`
- Renku project/app: `https://renkulab.io/v2/projects/sdsc/taxdna`

The current local clone at `data/edna/raw/stalder_taxdna` has had Git LFS
resolved. It now includes the official pretrained checkpoints, tree, species
metadata, and validation-species file. It does not include all training and
co-occurrence data referenced by the official config files.

Minimum official assets needed for exact baseline retraining/re-evaluation:

- `data/fish_tree_of_life/actinopt_single.tree`
- `data/species_info/species_info.json`
- `data/val_species/val_species.json`
- `data/sequence_data/species_sequences.json`
- `trained_models/tree_embedder/**/checkpoints/*.ckpt`
- `trained_models/dna_models/**/checkpoints/*.ckpt`
- `trained_models/co_occ_models/**/checkpoints/*.ckpt`
- the reference sequence JSON used by TAXDNA DNA training
- the co-occurrence JSONs referenced by official configs, especially:
  - `range_maps_marine.json`
  - `reef_life_survey.json`
  - `FISHGLOB.json`
  - `maestro.json`
  - the combined community/range-map validation split

If the missing data is obtained manually, place it into the official clone with
the same relative paths used by the config files, then rerun the readiness audit.

## Public Source Check

Checked on 2026-05-30:

- Git LFS payload from the official GitLab/Renku mirror resolves all tracked
  LFS files.
- The public Renku project metadata points to the original public repo:
  `https://gitlab.datascience.ch/stalder30/TAXDNA`.
- The original public repo has 28 commits, one public branch, no tags, and no
  missing data folders in tree, public history, or LFS pointers.
- Public releases, packages, issues, merge requests, and wiki are empty or do
  not expose the missing data.
- Public container images from both Renku and the original GitLab registry were
  pulled and layer-scanned; they do not bundle the missing
  `data/sequence_data` or `data/co_occurrence_data` folders.

Detailed audit:
`experiments/stalder_reproduction/OFFICIAL_ASSET_DEEP_DIVE.md`.

Conclusion: the official pretrained app can be used, but exact retraining of
TAXDNA is blocked unless the missing reference/co-occurrence training data are
made available separately.

## Public Reconstruction

We also keep a public reconstruction track for Stalder-style experiments where
the processed official JSON exports are unavailable. This is not exact TAXDNA
retraining, but it uses public source datasets and emits the same JSON shape
expected by TAXDNA co-occurrence training.

Current reconstructed source:

- FISHGLOB public bottom-trawl survey data.
- Output:
  `data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_cooccurrence.json`
- Manifest:
  `data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_manifest.json`
- Rebuild wrapper:
  `experiments/stalder_reproduction/runs/04_build_public_fishglob_cooccurrence.sh`

Details and claim boundary are in `PUBLIC_RECONSTRUCTION_PLAN.md`.

## Local Commands

Check what is currently present:

```bash
bash experiments/stalder_reproduction/runs/02_audit_assets.sh
```

Try to fetch official LFS assets:

```bash
bash experiments/stalder_reproduction/runs/01_fetch_official_assets.sh
```

The fetch script requires `git-lfs`. On macOS:

```bash
brew install git-lfs
git lfs install
```

Run official released TAXDNA inference on our Global_eDNA SWARM representative
FASTA files:

```bash
bash experiments/stalder_reproduction/runs/03_run_official_global_edna_inference.sh
```

This requires the official TAXDNA Python dependencies, e.g.:

```bash
python -m pip install -e data/edna/raw/stalder_taxdna
```

Build public FISHGLOB co-occurrence JSON:

```bash
bash experiments/stalder_reproduction/runs/04_build_public_fishglob_cooccurrence.sh
```

This requires `Rscript` for converting the public FISHGLOB RData export.

## Replication Stages

1. Asset readiness: all official pointers resolved, all referenced data present.
2. Official baseline: run TAXDNA CNN/tree/co-occurrence and recover Table 1/2-style metrics.
3. Encoder swap: keep all official data fixed and replace only the CNN DNA model with MarineMamba/BarcodeMamba.
4. Report:
   - single-sequence only
   - community co-occurrence
   - range-map co-occurrence
   - combined co-occurrence
   - seen and unseen species
   - species/genus/family/order top-1
   - calibration
   - real eDNA agreement against the traditional pipeline

## Claim Boundary

Until the readiness check passes with official LFS/data assets, this is not an
exact Stalder reproduction. It is only a reproduction scaffold.
