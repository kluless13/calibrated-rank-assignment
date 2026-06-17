# 12S/16S MarkerMirror

## Core Idea

The next MarkerMirror extension is to add 16S as the second eDNA marker. This
keeps the project scoped around fish eDNA while testing whether two ribosomal
markers can support stronger rank-aware inference than 12S alone.

Scope decision, 2026-06-03:

```text
near-term marker expansion = 12S + 16S
do not expand into CytB / 18S / plant / fungal markers for this paper
```

COI remains useful as a reference/tree anchor and comparator, but the eDNA
marker expansion itself should be 12S/16S.

The clean model is not a hard chain:

```text
12S -> 16S -> COI
```

The cleaner model is a shared species/tree space. In the tight eDNA version:

```text
12S encoder -> shared fish species/tree space
16S encoder -> shared fish species/tree space
```

When we want to connect the eDNA bridge back to the COI tree/barcode system,
COI can be an anchor:

```text
12S encoder -> shared species/tree space
16S encoder -> shared species/tree space
COI encoder -> optional COI/tree anchor
```

Then inference can use marker-specific evidence while still returning the same
species/genus/family/order/no-call output.

## Why 16S Is Interesting

16S is another mitochondrial ribosomal marker. For fish eDNA, it is closer in
biological character to 12S than COI is, while still adding independent marker
evidence. That makes it a plausible bridge marker:

- 12S: short, common in fish eDNA, often species-ambiguous.
- 16S: also eDNA-compatible, sometimes complementary to 12S.
- COI: richer barcode tradition and reference structure, often more
  discriminative, but protein-coding and less directly comparable to 12S.

## Literature Boundary

Multi-marker metabarcoding is already established. Recent fish studies use 12S,
16S, COI, and sometimes CytB together, compare marker-specific recovery, and
validate records by redundant detections across markers.

What we have not yet found is the specific learned bridge:

> train 12S and 16S encoders into a shared fish species/tree coordinate system,
> optionally anchored to COI/tree evidence, then use that bridge for rank-aware
> eDNA inference.

So the novelty claim should not be "we discovered 16S" or "we invented
multi-marker metabarcoding." The claim would be:

> Multi-marker species-space alignment turns heterogeneous marker evidence into
> a calibrated biodiversity inference system.

## Other Markers To Know

For fish/marine animal eDNA and barcoding:

- COI: standard animal barcode, high species resolution, strong BOLD/GenBank
  tradition.
- 12S: common fish eDNA/metabarcoding marker.
- 16S: common mitochondrial marker for metazoans and fish eDNA panels.
- CytB: mitochondrial cytochrome b; used in some fish eDNA/multimarker panels.
- 18S: broad eukaryotic marker, useful for protists/metazoan community context,
  usually lower species resolution for fish.
- 28S/LSU: common in some eukaryote/fungal/nematode contexts.
- ITS: fungal marker, not a fish marker.
- rbcL/matK/trnL/23S: plants, algae, photosynthetic microbes; useful if we
  later broaden beyond fish.

For MarineMamba Paper 1, the practical expansion set is:

```text
12S + 16S
```

COI is retained as an existing barcode/tree anchor. CytB and 18S are explicitly
out of scope unless the 12S/16S experiment fails for a data-coverage reason.

## Current Local Status

We now have an initial clean 16S `species_sequences.json` comparable to our COI
and 12S inputs. It is an NCBI pilot/reference-build, not yet a fully curated
publication-grade 16S database.

Local audit:

- script: `scripts/edna/audit_16s_marker_sources.py`
- table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_16s_local_source_audit.csv`
- result: Scotian Shelf contains 2,592 16S occurrence records covering 46
  species, but no actual sequence strings in the local GBIF/OBIS export.

Reference construction:

- script: `scripts/edna/build_16s_reference_from_ncbi.py`
- query plan:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_16s_ncbi_query_plan.json`
- reference manifest:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_16s_reference_manifest.json`
- output root:
  `data/edna/stalder_inputs/16s_multisource/`
- NCBI query returned 40,034 candidate Actinopterygii mitochondrial 16S/rrnL
  records.
- Bounded 5,000-record fetch produced:
  - 4,673 usable sequence records;
  - 1,865 species;
  - 502 species overlapping existing 12S;
  - 607 species overlapping COI;
  - 319 species overlapping 12S+COI.

Taxonomy enrichment:

- script: `scripts/edna/enrich_16s_reference_taxonomy.py`
- table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_16s_candidate_taxonomy_enrichment.csv`
- family/order filled for 1,133 of 1,865 16S species using existing 12S/COI
  candidate tables.

This is enough to run a first 12S/16S MarkerMirror experiment, and the first
held-out 12S->16S bridge is now complete.

First 12S->16S bridge:

- run:
  `nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best`;
- copied root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best/`;
- split: 351 train, 75 validation, 76 held-out test species;
- held-out top-10 species/genus/family/order:
  33.9 / 45.4 / 67.1 / 73.4%;
- frozen NT baseline on the same split:
  18.4 / 23.0 / 34.9 / 49.7%.

Interpretation: learned 12S->16S alignment is substantially better than frozen
foundation embeddings and gives much cleaner species/genus retrieval than the
current 12S->COI bridge. This strengthens the 12S/16S focus, but it is still a
bridge-candidate result, not a calibrated final assignment system.

Reverse 16S->12S bridge:

- run:
  `nt_v2_50m_16s_to_12s_taxonomy_soft_retrieval_best`;
- copied root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_16s_to_12s_taxonomy_soft_retrieval_best/`;
- split: same 351 train, 75 validation, 76 held-out test species;
- held-out top-10 species/genus/family/order:
  44.4 / 54.8 / 70.4 / 76.3%;
- frozen NT baseline on the same split:
  11.9 / 17.8 / 32.6 / 48.2%.

Interpretation: the reverse bridge is even stronger than 12S->16S. The next
step should be a shared 12S/16S species-space prototype plus seed/curation
checks, not expansion into more markers.

Shared 12S/16S species-space prototype:

- run:
  `nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_shared_retrieval_metrics.csv`;
- split: same 351 train, 75 validation, 76 held-out test species;
- checkpoint: epoch 60 selected by combined validation genus/family/order
  top-10 across both directions;
- held-out 12S->16S top-10 species/genus/family/order:
  42.1 / 50.0 / 68.5 / 81.5%;
- held-out 16S->12S top-10 species/genus/family/order:
  64.3 / 71.3 / 78.3 / 85.3%.

Interpretation: the shared marker space beats the separate directional bridges.
This is now the lead MarkerMirror result. It remains candidate retrieval, so
the next step is seed repeats and rank/no-call integration.

Seed-repeat stability:

- three shared-space runs are now merged into
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_shared_retrieval_metrics.csv`;
- 12S->16S held-out top-10 mean species/genus/family/order:
  43.4 / 50.7 / 68.1 / 77.6%;
- 12S->16S ranges:
  species 42.1-45.4, genus 49.0-53.0, family 65.6-70.2, order 72.9-81.5%;
- 16S->12S held-out top-10 mean species/genus/family/order:
  66.4 / 73.9 / 81.9 / 86.4%;
- 16S->12S ranges:
  species 63.3-71.5, genus 71.3-78.5, family 78.3-86.8, order 84.2-89.6%.

Interpretation: the shared 12S/16S signal is stable enough to move downstream
into candidate reranking and rank/no-call calibration.

Tri-marker bridge:

- script:
  `scripts/edna/train_marker_mirror_triad_space.py`;
- active run:
  `nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best`;
- pair overlaps total/train/val/test:
  - 12S/16S: 502 / 364 / 66 / 72;
  - 12S/COI: 963 / 669 / 164 / 130;
  - 16S/COI: 607 / 424 / 95 / 88.

The triad test answers the user's chain question without creating a brittle
hard chain: it tests whether one 12S/16S/COI shared species-space improves
12S->COI/tree candidate retrieval compared with the older direct 12S->COI
bridge.

Result:

- run:
  `nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best`;
- copied root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best/`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_triad_retrieval_metrics.csv`;
- best checkpoint: epoch 110, combined validation score 47.9459.

Held-out tri-marker top-10:

| Direction / Model | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| 12S -> 16S, frozen NT | 11.9% | 19.6% | 34.3% | 42.7% |
| 12S -> 16S, triad MarkerMirror | 31.1% | 47.2% | 65.0% | 73.1% |
| 16S -> 12S, frozen NT | 11.1% | 14.1% | 29.6% | 43.7% |
| 16S -> 12S, triad MarkerMirror | 51.9% | 65.9% | 77.8% | 89.6% |
| 12S -> COI, frozen NT | 0.0% | 1.8% | 12.7% | 38.1% |
| 12S -> COI, triad MarkerMirror | 2.7% | 12.7% | 34.8% | 58.6% |
| 16S -> COI, frozen NT | 0.0% | 4.0% | 18.7% | 46.0% |
| 16S -> COI, triad MarkerMirror | 5.3% | 20.7% | 48.7% | 64.7% |

Conclusion: the triad is useful evidence but not the lead bridge. It proves
that one shared head can align three marker families better than frozen NT, but
it weakens 12S->COI relative to the best direct 12S->COI run. Keep the
12S/16S shared space as the lead MarkerMirror result, and use COI as a
downstream barcode/tree anchor unless a later architecture explicitly improves
12S->COI.

Pipeline handoff:

- shared 12S/16S candidate rankings have now been exported against the full
  target marker library for seeds 1901, 1902, and 1903;
- candidate rows:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed190*/marker_mirror_candidate_rankings.csv.gz`;
- summary:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_rankings_shared_seed1903_summary.csv`;
- top-1 score thresholds alone are not enough for rank/no-call, so the next
  handoff should join these candidates with sequence similarity, tree/rank
  evidence, and reference-gap features.
- the integrated evidence handoff is now complete. Best held-out seed1903
  target-0.99 rows are 12S->16S at 55.0% coverage / 99.4% assigned precision /
  0.0% false species-call rate, and 16S->12S at 75.0% coverage / 99.1%
  assigned precision / 0.0% false species-call rate.
- seed-repeat handoff is complete across seeds 1901, 1902, and 1903. At
  target 0.99, learned MarkerMirror averages 51.0% coverage / 98.9% assigned
  precision for 12S->16S and 71.1% coverage / 98.7% assigned precision for
  16S->12S, with near-zero false species-call rates.
- a seed1903 resolvability-enhanced handoff is complete. Explicit 12S/16S
  exact/0.99-proxy resolvability features were added to the evidence compiler.
  They did not change the best seed1903 target-0.99 operating point, but they
  make marker ambiguity auditable.

## Build Plan

1. Build a fish 16S reference library.
   - Source candidates: GenBank records, MitoFish/mitogenomes, and curated
     multi-marker reference libraries where available.
   - Required output:
     `data/edna/stalder_inputs/16s_multisource/species_sequences.json`.
   - Also build:
     `candidate_species.csv`, `species_info.json`, `manifest.json`.
   - Status: first bounded NCBI build complete; larger/curated build remains
     optional after the first bridge result.

2. Compute overlap sets:
   - `12S ∩ 16S`
   - optional anchor audit: `12S ∩ 16S ∩ COI`
   - Status: first overlap table complete through the reference manifest.

3. Run bridge baselines:
   - `12S -> 16S`;
   - `16S -> 12S`;
   - shared `12S + 16S -> species/tree space`;
   - optional anchor comparison to existing `12S -> COI` MarkerMirror results.
   - Status: both directional 12S/16S runs and the first shared-space prototype
     completed and copied; the shared-space prototype is strongest so far.

4. Evaluate:
   - species/genus/family/order top-k;
   - held-out species;
   - held-out genera where enough overlap exists;
   - marker-specific ambiguity/resolvability;
   - whether 16S improves high-rank or species-rank transfer.

5. Integrate with the pipeline:
   - 12S query produces candidates in shared marker space;
   - 16S references supply bridge evidence;
   - COI/tree evidence can remain a downstream anchor/comparator;
   - final decision remains rank/no-call with reason codes.

## Claim Boundary

Do not claim:

- multi-marker metabarcoding is new;
- 16S is new;
- 12S can always become species-level via 16S;
- the bridge works before a clean 16S reference set and held-out test exist.

Potential claim if results hold:

> A 12S/16S species-space bridge lets ambiguous fish eDNA evidence borrow
> structure across ribosomal markers, improving calibrated high-rank
> biodiversity inference under marker ambiguity and missing references.
