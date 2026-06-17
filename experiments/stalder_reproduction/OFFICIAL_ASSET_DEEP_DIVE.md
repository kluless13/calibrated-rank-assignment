# Official TAXDNA Asset Deep Dive

Last updated: 2026-05-30

## Goal

Find the official processed TAXDNA assets needed for exact Stalder-style
retraining and encoder-swap experiments:

- `data/sequence_data/species_sequences.json`
- `data/co_occurrence_data/reef_life_survey.json`
- `data/co_occurrence_data/FISHGLOB.json`
- `data/co_occurrence_data/maestro.json`
- `data/co_occurrence_data/range_maps_marine.json`
- `data/co_occurrence_data/RivFishTIME.json`
- `data/co_occurrence_data/freshwater_drainage_basins.json`
- `data/co_occurrence_data/range_maps_freshwater.json`
- dataset-specific `val_species*.json` files referenced by the official DNA
  model configs.

## Public Surfaces Checked

### Renku project page

- Project: `https://renkulab.io/p/sdsc/taxdna`
- Renku project id: `01JJPK91DD7VSXHPC9WH2XVGY6`
- The rendered project metadata points to source repository:
  `https://gitlab.datascience.ch/stalder30/TAXDNA`
- No public data connector links exposing `data/sequence_data` or
  `data/co_occurrence_data` were visible from the project metadata.

### Renku/GitLab mirror

- Mirror: `https://gitlab.renkulab.io/dnai/TAXDNA.git`
- Local clone: `data/edna/raw/stalder_taxdna`
- Public branch API reports only `main`.
- Public tag API returns no tags.
- Git LFS tracks official tree/species metadata/model checkpoints, but not the
  missing reference sequence or co-occurrence JSONs.
- Recursive repository tree does not include `data/sequence_data` or
  `data/co_occurrence_data`.
- Raw URL checks for missing files returned 404.
- Public releases/packages did not expose missing data.
- Public pipeline jobs only build the app image; no artifact archive exposes the
  missing data.

### Original public GitLab repository

- Original repo: `https://gitlab.datascience.ch/stalder30/TAXDNA`
- API project id: `719`
- Local clone: `data/edna/raw/stalder_taxdna_original`
- Repository is public and has 28 commits.
- Public refs:
  - `refs/heads/main` only
  - no tags
- Current public file listing contains 103 files. It includes:
  - `data/fish_tree_of_life/actinopt_single.tree`
  - `data/species_info/species_info.json`
  - `data/val_species/val_species.json`
  - pretrained DNA/tree/co-occurrence model checkpoints
  - official configs referencing the missing JSONs
- Current public file listing does not include:
  - `data/sequence_data/species_sequences.json`
  - `data/co_occurrence_data/*.json`
  - dataset-specific folders such as `data/FISHGLOB/`,
    `data/reef_life_survey/`, `data/range_maps/`, or `data/maestro/`
- `git log --all --name-status` over the missing paths found no deleted copies
  in history.
- `git lfs ls-files --all` found no LFS pointers for the missing processed
  files.
- Public releases/packages are empty.
- Public issues/merge requests are empty.
- Public wiki is empty.

### Container registries

Checked with `skopeo` on the Vast host, without running the images.

1. `registry.renkulab.io/dnai/taxdna:latest`
   - Pullable public image.
   - Layer scan found no `species_sequences`, `sequence_data`,
     `co_occurrence`, `FISHGLOB`, `reef_life`, `range_maps`, `RivFishTIME`,
     `maestro`, or `freshwater_drainage` paths.

2. `registry.renkulab.io/dnai/taxdna:61fdb4a`
   - Older public Renku image tag.
   - Layer scan found no missing processed data paths.

3. `gitlab-registry.datascience.ch/stalder30/taxdna:latest`
   - Original project container image.
   - Layer scan shows only `/code/pyproject.toml`, `/code/src/taxDNA/...`,
     package build output, and dependencies.
   - No `data/sequence_data`, `data/co_occurrence_data`, model checkpoints, or
     processed training JSONs are bundled.

## Interpretation

The official code, pretrained model checkpoints, tree, species metadata, and one
generic validation species file are public. The processed reference sequence and
co-occurrence JSONs required for exact retraining are not exposed through the
public repositories, Git history, LFS pointers, Renku metadata visible without
login, releases/packages, pipeline artifacts, or public container layers checked
so far.

The strongest honest claim boundary remains:

- Exact Stalder/TAXDNA pretrained-app inference is possible with released
  checkpoints and our own query data.
- Exact Stalder/TAXDNA retraining or exact encoder swap is blocked until the
  missing processed JSONs are obtained.
- Public Stalder-style reconstruction is possible by rebuilding equivalent
  inputs from public sources, but it is not the same as using the official
  processed TAXDNA JSON exports.

## What Is Still Worth Trying

1. Launch a Renku session from the public project UI and inspect the live mounted
   workspace for non-Git data volumes or session-init downloads.
2. Ask the authors/SDSC for the processed data export listed above.
3. Continue the public reconstruction route:
   - FISHGLOB: already reconstructed locally into TAXDNA-shaped co-occurrence
     JSON.
   - Reef Life Survey: reconstruct from public/API/export source if available.
   - Range maps: reconstruct from OBIS/IUCN/AquaMaps-style occurrence/range
     sources, documenting differences.
   - DATRAS/MEDITS/MAESTRO/RivFishTIME/freshwater drainage basins: treat each as
     separate provenance-controlled public reconstructions or access requests.
