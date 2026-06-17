# Public Stalder-Style Reconstruction Plan

Date: 2026-05-30

This track is separate from exact TAXDNA reproduction. The exact reproduction is
still blocked by missing official processed data files. This reconstruction uses
public source datasets and emits TAXDNA-shaped inputs so we can run a transparent
Stalder-style comparison.

## Why This Exists

Stalder et al. report a pipeline that combines:

- a Teleo 12S reference database,
- a fish phylogeny,
- empirical community co-occurrence data,
- range-map co-occurrence data,
- real eDNA validation against a traditional pipeline.

The official Git/LFS checkout includes trained models, tree files, and species
metadata, but not the processed `data/sequence_data/species_sequences.json` or
`data/co_occurrence_data/*.json` files referenced by the official training
configs. The paper says the method source code is public, and it names public or
request-based source datasets, but the processed TAXDNA JSON exports are not
exposed in the public Renku/GitLab assets.

## Public Sources

- TAXDNA paper: `https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013776`
- TAXDNA project: `https://renkulab.io/v2/projects/sdsc/taxdna`
- TAXDNA GitLab: `https://gitlab.renkulab.io/dnai/TAXDNA.git`
- FISHGLOB compiled data: `https://github.com/fishglob/FishGlob_data/tree/main/outputs/Compiled_data`
- FISHGLOB Zenodo record: `https://zenodo.org/records/10218308`
- OBIS: `https://obis.org`
- DATRAS: `https://datras.ices.dk`
- RLS: `https://reeflifesurvey.com`

## Completed Public Reconstruction Work

FISHGLOB was first staged and converted on the Vast instance because the local
Mac disk was tight and the public file is RData. After local disk was cleared,
the public source files and converted CSV were copied into the workspace.

Local staged inputs:

- `data/edna/raw/stalder_public_sources/fishglob/FishGlob_public_clean.RData`
- `data/edna/raw/stalder_public_sources/fishglob/FishGlob_public_metadata_clean.RData`
- `data/edna/raw/stalder_public_sources/fishglob/FishGlob_public_clean_taxdna_columns.csv.gz`

Generated TAXDNA-shaped output:

- local:
  `data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_cooccurrence.json`
- local:
  `data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_manifest.json`
- remote:
  `/workspace/marinemamba/data/edna/cooccurrence_inputs/stalder_public/fishglob_public_taxdna_cooccurrence.json`

Current FISHGLOB reconstruction summary:

- rows read: 2,931,610
- matched rows to current candidate tree labels: 2,279,934
- haul-level co-occurrence groups: 214,658
- unique matched candidate species: 1,211
- unique matched species with current reference sequences: 150
- median group size: 10 species
- max group size: 41 species

## Rebuild Command

On a machine with `Rscript` available:

```bash
bash experiments/stalder_reproduction/runs/04_build_public_fishglob_cooccurrence.sh
```

The script downloads the public FISHGLOB RData if needed, converts only the
columns required for co-occurrence grouping, and runs:

```bash
python scripts/edna/build_public_fishglob_taxdna_json.py \
  --fishglob-csv data/edna/raw/stalder_public_sources/fishglob/FishGlob_public_clean_taxdna_columns.csv.gz \
  --input-dir data/edna/real_edna_queries/global_tropical_multisource_teleo \
  --output-dir data/edna/cooccurrence_inputs/stalder_public \
  --min-species-per-group 2 \
  --encoding latin1
```

## Remaining Reconstruction Gaps

- Official `species_sequences.json`: not public as a processed file. The paper
  says it combines NCBI sequences and authors' own Sanger sequencing, so an
  exact rebuild needs author files or accessions for the custom sequences.
- Official `reef_life_survey.json`: we have local RLS-derived inputs from the
  Global_eDNA work, but not the authors' processed TAXDNA JSON.
- Official `maestro.json`: not yet reconstructed.
- Official marine range maps: we have OBIS-derived site/range priors, but not
  the authors' OBIS + Gaspar + convex-hull/depth-filtered 1-degree range-map
  export.
- DATRAS can be public, MEDITS is request-based according to the paper, so the
  full empirical community source cannot be reconstructed exactly without
  access.

## Claim Boundary

This public reconstruction lets us say:

- we built TAXDNA-shaped co-occurrence inputs from public FISHGLOB source data;
- we can train/evaluate a public Stalder-style ecological context module;
- we can compare MarineMamba/SSM and CNN sequence encoders under this public
  reconstruction.

It does not let us say:

- exact Stalder/TAXDNA retraining;
- exact recovery of their Table 1/2 metrics;
- exact encoder swap with all official splits/data fixed.
