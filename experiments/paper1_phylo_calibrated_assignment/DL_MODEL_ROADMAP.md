# DL Model Roadmap

Last updated: 2026-06-02

## Principle

The goal is not to prove that deep learning beats BLAST everywhere. BLAST,
VSEARCH, k-mer search, APPLES, and EPA-ng are strong because they solve
well-defined sequence-similarity or placement problems with mature algorithms.

The DL opportunity is narrower and more valuable:

> learn which evidence streams support which biodiversity claim, then return
> species/genus/family/order/no-call instead of forcing a species label.

## Literature Boundary

The closest prior nodes are:

- DEPP / H-DEPP / C-DEPP: neural sequence-to-species-tree placement.
- PROTAX / PROTAX-GPU: probabilistic taxonomic placement with missing/unknown
  species outcomes.
- BarcodeBERT / BarcodeMamba / DNABERT-S / TaxoTagger: fast learned barcode
  representation and retrieval/classification.
- TAXDNA: 12S sequence + phylogeny + range/community priors for eDNA.
- Fernando-style fish COI placement: classical EPA-ng/APPLES placement on a
  ray-finned fish backbone.

Our DL work should therefore avoid claims that are already taken:

- "neural placement onto trees is new";
- "vector barcode retrieval is new";
- "uncertainty-aware taxonomy is new";
- "sequence + tree + ecology is new."

The useful bridge is an integrated, auditable system that combines these pieces
under missing-reference and marker-ambiguity stress tests.

## Five Trainable Layers

1. Candidate retrieval encoder
   - Learn fast vector candidates from barcode sequences.
   - Must be compared to BLAST/VSEARCH/k-mer and existing barcode encoders.
   - Status: first retrieval-DL sweep complete. CNN contrastive/hybrid are
     useful candidate generators; hierarchical losses and the Transformer arm
     improve tree-shape correlations but currently hurt candidate recall.
   - Doc: `dl_models/01_candidate_retrieval_encoder.md`

2. Evidence reranker / fusion model
   - Learn how to combine vector score, p-distance, BLAST/VSEARCH identity,
     tree distance, and placement confidence over top-k candidates.
   - Status: first diagnostic COI top-50 candidate reranker trained and
     source-tabled. It improves genus/family candidate ordering, but calibrated
     target-0.99 precision does not transfer well enough to become production.
   - Doc: `dl_models/02_evidence_reranker.md`

3. Rank/no-call calibrator
   - Learn whether evidence supports species, genus, family, order, or no-call.
   - First model completed: COI MLP over vector+p-distance evidence.
   - Status: trained, seed-repeated, CLI-integrated, and strict-tested.
   - Doc: `dl_models/03_rank_no_call_calibrator.md`

4. Reference-gap detector
   - Learn when the true species/genus/family is likely absent or unsupported.
   - Should explain why a call backs off.
   - Status: first diagnostic COI MLP trained and source-tabled. The honest
     no-counts target-0.99 run is conservative but low-recall; use it as a
     development diagnostic, not a production reason layer yet.
   - Doc: `dl_models/04_reference_gap_detector.md`

5. eDNA Eco-Phylo posterior
   - Learn candidate-level posterior over sequence, tree-neighborhood,
     geography/range, co-occurrence, and marker-resolvability evidence.
   - Status: candidate-level 12S/eDNA posterior exists with true nested
     evaluation; needs stronger calibration/features before it is headline.
   - Doc: `dl_models/05_edna_eco_phylo_posterior.md`

## First Completed DL Result

Script:

- `scripts/edna/train_paper1_coi_evidence_model.py`
- inference adapter: `scripts/edna/apply_paper1_coi_evidence_model.py`
- production CLI mode: `--decision-mode dl_mlp_species_disabled`

Inputs:

- current CNN seed1206 vector + p-distance pipeline outputs;
- trained on seen-test rows;
- thresholds calibrated on held-out seen-test rows;
- evaluated on held-out fish and unseen-genera.

Best current conservative result:

| Policy | Split | Coverage | Assigned Precision | False Species Calls |
|---|---:|---:|---:|---:|
| MLP species-disabled target-0.99 | held-out fish | 94.2% | 97.4% | 0.0% |
| MLP species-disabled target-0.99 | unseen-genera | 88.5% | 93.5% | 0.0% |

Bootstrap 95% intervals:

- held-out fish: coverage 93.8-94.7%, assigned precision 97.1-97.7%;
- unseen-genera: coverage 87.8-89.2%, assigned precision 93.0-94.0%.

Comparison to current hand-threshold production-v1:

| Policy | Split | Coverage | Assigned Precision | False Species Calls |
|---|---:|---:|---:|---:|
| hand threshold target-0.99 | held-out fish | 95.8% | 93.0% | 0.0% |
| hand threshold target-0.99 | unseen-genera | 92.3% | 83.9% | 0.0% |

Interpretation:

- the first DL layer improves precision materially;
- it does so with slightly lower coverage;
- species must remain disabled for the conservative missing-reference claim;
- this is now integrated as an optional FASTA/CSV CLI decision layer;
- the hand-threshold production-v1 policy remains the simpler default until DL
  seed repeats and strict-pack tests are complete.

Integration smoke tests on the Vast RTX PRO 6000:

- CSV known-label CLI smoke: 16/16 assigned, 100.0% precision if known, 0
  species calls.
- FASTA unlabeled CLI smoke: 8/8 assigned, precision unavailable by design, 0
  species calls.

Seed-repeat stability across MLP seeds 1206/1207/1208:

- held-out fish: coverage 94.2-96.0%, assigned precision 97.1-97.4%, 0.0%
  false species-call rate.
- unseen-genera: coverage 88.5-91.3%, assigned precision 92.9-93.5%, 0.0%
  false species-call rate.

Strict hidden-reference stress test for seed1206, species-disabled target-0.99:

| Split | Hidden Rank | Coverage | Assigned Precision | Species Calls |
|---|---:|---:|---:|---:|
| held-out fish | species | 58.0% | 88.5% | 0 |
| held-out fish | genus | 57.9% | 82.5% | 0 |
| held-out fish | family | 16.1% | 81.5% | 0 |
| unseen-genera | species | 50.6% | 91.3% | 0 |
| unseen-genera | genus | 52.9% | 89.5% | 0 |
| unseen-genera | family | 41.9% | 72.7% | 0 |

Interpretation: the current DL calibrator remains conservative under strict
missing-reference stress, but family-hidden cases expose the need for an
explicit reference-gap detector and/or candidate-level reranker.

## Immediate Next Work

1. Wire the reason-code overlay into the FASTA/CSV CLI output, not only source
   tables. Current source-table prototype:
   `scripts/edna/build_paper1_reason_code_overlay.py`.
2. Build an active-curation table from production assignments, strict
   hidden-reference failures, and v2 reference-gap scores:
   which genera/families would benefit most from additional reference barcodes.
3. Normalize all candidate generators into one evidence schema:
   vector/BLAST/VSEARCH/k-mer/EPA-ng/APPLES -> candidate rows with score,
   taxonomy, tree/placement evidence, and reason-code inputs.
4. Keep production-v1 as the default manuscript operating point unless a
   combined DL policy transfers under strict hidden-reference stress with no
   false species calls.
5. Extend the same evidence-compiler/reason-code layer to 12S/eDNA. Species
   remains disabled for eDNA unless independent thresholds transfer.
