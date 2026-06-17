# Stalder/TAXDNA Reproduction Status

Last updated: 2026-05-30

## Done

- Created isolated reproduction track under `experiments/stalder_reproduction/`.
- Installed `git-lfs` locally.
- Pulled official TAXDNA Git LFS assets from `https://gitlab.renkulab.io/dnai/TAXDNA.git`.
- Resolved the core LFS pointers:
  - official tree
  - species info
  - validation species file
  - tree embedder checkpoint
  - DNA model checkpoints
  - co-occurrence model checkpoints
- Wrote asset inventory to `data/edna/raw/stalder_taxdna_manifest.json`.
- Wrote readiness report to `results/edna/stalder_reproduction/asset_readiness.json`.
- Checked the public Renku project metadata/API for linked data connectors:
  none are exposed for `sdsc/taxdna`.
- Checked the original public GitLab repository at
  `https://gitlab.datascience.ch/stalder30/TAXDNA`:
  - it is public and has 28 commits;
  - public refs contain only `main`;
  - there are no public tags;
  - the missing data folders are not in the current repository tree;
  - the missing data folders do not appear as deleted files in public history;
  - `git lfs ls-files --all` has no LFS pointers for the missing processed
    JSONs.
- Checked GitLab releases/packages: no public releases or packages are listed.
- Checked public issues, merge requests, and wiki: no missing-data pointers.
- Checked the public container registries:
  - `registry.renkulab.io/dnai/taxdna:latest`
  - `registry.renkulab.io/dnai/taxdna:61fdb4a`
  - `gitlab-registry.datascience.ch/stalder30/taxdna:latest`
  The image layers only contain the app/package/runtime pieces checked so far;
  they do not bundle the missing official training data folders.
- Added a detailed evidence trail at
  `experiments/stalder_reproduction/OFFICIAL_ASSET_DEEP_DIVE.md`.
- Added `scripts/edna/run_official_taxdna_global_edna.py` and
  `runs/03_run_official_global_edna_inference.sh` so we can run the released
  official TAXDNA app on our Global_eDNA SWARM representative FASTAs once the
  official Python dependencies are installed.
- Added a public reconstruction path for one empirical co-occurrence source:
  FISHGLOB. The builder is
  `scripts/edna/build_public_fishglob_taxdna_json.py`, with a rebuild wrapper at
  `runs/04_build_public_fishglob_cooccurrence.sh`.
- FISHGLOB public source files were staged on the Vast instance and converted
  into a TAXDNA-shaped co-occurrence JSON. The generated local output is ignored
  by git under `data/edna/cooccurrence_inputs/stalder_public/`.
- After local disk was cleared, the public FISHGLOB RData files and converted
  CSV were copied locally under
  `data/edna/raw/stalder_public_sources/fishglob/`.
- Current public FISHGLOB reconstruction summary:
  - rows read: 2,931,610
  - rows matched to current candidate tree labels: 2,279,934
  - haul-level co-occurrence groups: 214,658
  - unique matched candidate species: 1,211
  - matched candidate species with current reference sequences: 150
  - median group size: 10 species

## Current Blocker

The official Git/LFS checkout now has the pretrained app assets, but it does not
include the data directories referenced by the official TAXDNA config files. The
public Renku/GitLab/container surfaces checked so far do not expose these
folders. The FISHGLOB source data are public and reconstructable, but the
official processed TAXDNA JSONs and reference sequence JSON are still missing.

Missing paths:

- `data/co_occurrence_data/reef_life_survey.json`
- `data/co_occurrence_data/FISHGLOB.json`
- `data/co_occurrence_data/maestro.json`
- `data/co_occurrence_data/range_maps_marine.json`
- `data/sequence_data/species_sequences.json`
- `data/co_occurrence_data/RivFishTIME.json`
- `data/co_occurrence_data/freshwater_drainage_basins.json`
- `data/co_occurrence_data/range_maps_freshwater.json`
- `data/community_and_marine_range_maps/val_species.json`
- `data/reef_life_survey/val_species.json`
- `data/FISHGLOB/val_species.json`
- `data/maestro/val_species.json`
- `data/range_maps/val_species_marine.json`
- `data/RivFishTIME/val_species.json`
- `data/freshwater_drainage_basins/val_species.json`
- `data/range_maps/val_species_freshwater.json`

## Next Action

For exact Stalder/TAXDNA retraining, retrieve the missing official data
directories from a live Renku session export if those files are mounted there,
or from the authors/SDSC. The first priority for a marine reproduction is:

1. `data/co_occurrence_data/reef_life_survey.json`
2. `data/co_occurrence_data/FISHGLOB.json`
3. `data/co_occurrence_data/maestro.json`
4. `data/co_occurrence_data/range_maps_marine.json`
5. `data/sequence_data/species_sequences.json`
6. `data/community_and_marine_range_maps/val_species.json`

After these are present, rerun:

```bash
bash experiments/stalder_reproduction/runs/02_audit_assets.sh
```

In parallel, use the now-resolved official pretrained checkpoints for an
official TAXDNA inference-only baseline on our Global_eDNA SWARM representatives.
That is not exact retraining, but it is a valid official-app comparison because
it uses their released tree, species metadata, DNA checkpoint, and co-occurrence
checkpoint.

Local note: the current base Python is missing `lightning`, so the official
inference wrapper has been syntax-checked but not executed locally yet.

For public-source reconstruction, next build targets are:

1. train the learned NPZ co-occurrence module on the reconstructed FISHGLOB JSON;
2. fold this into the Global_eDNA reranking comparison;
3. reconstruct or document RLS, OBIS/range-map, DATRAS/MEDITS, and MAESTRO
   inputs separately from exact TAXDNA reproduction.
