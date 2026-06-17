# Official TAXDNA Asset Request

Date: 2026-05-30

This is the minimal missing-data request for an exact Stalder/TAXDNA
reproduction. The public Git/LFS checkout has the released pretrained
checkpoints, tree, species metadata, and validation-species file. It does not
contain all config-referenced data needed to retrain or exactly re-evaluate the
official pipeline.

## Official Project Checked

- Renku project: `https://renkulab.io/v2/projects/sdsc/taxdna`
- GitLab/Renku repository: `https://gitlab.renkulab.io/dnai/TAXDNA.git`
- GitLab source mirror: `https://gitlab.datascience.ch/stalder30/TAXDNA`
- Container image: `gitlab-registry.datascience.ch/stalder30/taxdna:latest`

## Assets Already Resolved Locally

- `data/fish_tree_of_life/actinopt_single.tree`
- `data/species_info/species_info.json`
- `data/val_species/val_species.json`
- `trained_models/tree_embedder/**/checkpoints/*.ckpt`
- `trained_models/dna_models/**/checkpoints/*.ckpt`
- `trained_models/co_occ_models/**/checkpoints/*.ckpt`

## Missing Required Assets

These are required by official config files or training scripts:

- `data/sequence_data/species_sequences.json`
- `data/co_occurrence_data/reef_life_survey.json`
- `data/co_occurrence_data/FISHGLOB.json`
- `data/co_occurrence_data/maestro.json`
- `data/co_occurrence_data/range_maps_marine.json`
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

## Why These Are Needed

- `species_sequences.json` is the official 12S/Teleo sequence reference used to
  train/re-evaluate the DNA encoder.
- `data/co_occurrence_data/*.json` are the official community/range-map
  co-occurrence sources used to train/evaluate the ecological modulation model.
- The dataset-specific `val_species*.json` files define official held-out
  validation species for the corresponding co-occurrence setting.

## Current Claim Boundary

Without these folders, we can run official pretrained TAXDNA inference and our
own TAXDNA-style experiments. We cannot honestly claim an exact official
training reproduction or an exact encoder-swap reproduction yet.

## Public Reconstruction Note

One missing co-occurrence input is partially reconstructable from public source
data. We built a FISHGLOB-derived TAXDNA-shaped JSON from the public FISHGLOB
compiled RData:

- `data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_cooccurrence.json`
- `data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_manifest.json`

This is useful for a public Stalder-style experiment, but it is not a substitute
for the official processed `data/co_occurrence_data/FISHGLOB.json` if the goal
is exact retraining.
