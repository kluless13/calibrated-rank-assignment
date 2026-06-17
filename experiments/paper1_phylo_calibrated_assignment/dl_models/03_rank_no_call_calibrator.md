# DL 03: Rank/No-Call Calibrator

## Question

Given current candidate evidence, can a model learn whether the defensible
claim is species, genus, family, order, or no-call?

## Why This Matters

The manuscript contribution is strongest when we stop forcing species labels.
A learned calibrator can improve the decision layer without pretending that the
sequence marker contains species-level information it does not have.

## First Completed Model

Script:

- `scripts/edna/train_paper1_coi_evidence_model.py`
- inference adapter: `scripts/edna/apply_paper1_coi_evidence_model.py`
- FASTA/CSV CLI mode: `--decision-mode dl_mlp_species_disabled`

Model:

- small PyTorch MLP;
- inputs: vector scores, score margins, p-distance traces, top-k taxonomic
  consensus;
- outputs: per-rank correctness probabilities.

Training protocol:

- train on seen-test rows;
- calibrate thresholds on held-out seen-test rows;
- evaluate on held-out fish and unseen-genera.

## Current Result

Species-disabled target-0.99:

| Split | Coverage | Assigned Precision | False Species Calls |
|---|---:|---:|---:|
| held-out fish | 94.2% | 97.4% | 0.0% |
| unseen-genera | 88.5% | 93.5% | 0.0% |

Bootstrap 95% intervals:

- held-out fish: coverage 93.8-94.7%, assigned precision 97.1-97.7%;
- unseen-genera: coverage 87.8-89.2%, assigned precision 93.0-94.0%.

Compared with hand-threshold production-v1:

| Split | Coverage | Assigned Precision | False Species Calls |
|---|---:|---:|---:|
| held-out fish | 95.8% | 93.0% | 0.0% |
| unseen-genera | 92.3% | 83.9% | 0.0% |

## Interpretation

The MLP gives a real precision gain but reduces coverage. Species-enabled mode
starts making a small number of species calls, including false species calls on
missing-reference splits. For the conservative claim, species remains disabled.

The species-disabled model is now wired into
`scripts/edna/run_paper1_fasta_inference_v1.py` as an optional decision layer.
Vast smoke tests passed for both known-label CSV input and unlabeled FASTA
input. This confirms tool integration, not publication-level accuracy by
itself.

Seed repeats across MLP seeds 1206/1207/1208 are stable:

- held-out fish: coverage 94.2-96.0%, assigned precision 97.1-97.4%, 0.0%
  false species-call rate.
- unseen-genera: coverage 88.5-91.3%, assigned precision 92.9-93.5%, 0.0%
  false species-call rate.

Strict hidden-reference stress test for seed1206:

| Split | Hidden Rank | Coverage | Assigned Precision | Species Calls |
|---|---:|---:|---:|---:|
| held-out fish | species | 58.0% | 88.5% | 0 |
| held-out fish | genus | 57.9% | 82.5% | 0 |
| held-out fish | family | 16.1% | 81.5% | 0 |
| unseen-genera | species | 50.6% | 91.3% | 0 |
| unseen-genera | genus | 52.9% | 89.5% | 0 |
| unseen-genera | family | 41.9% | 72.7% | 0 |

This confirms that the calibrator is conservative under strict missing
reference stress. It also shows the current MLP is not enough by itself:
family-hidden stress especially needs explicit reference-gap detection.

## Selected-Candidate Calibration Follow-Up

After the retrieval-DL reranker work, we trained an independent assignment
calibrator over selected top-50 candidate-reranker outputs.

Scripts:

- `scripts/edna/train_paper1_candidate_assignment_calibrator.py`
- `scripts/edna/build_paper1_candidate_assignment_calibrator_summary.py`

Source table:

- `results/paper1_phylo_calibrated_assignment/source_tables/candidate_assignment_calibrator_summary.csv`

Result:

| Input Reranker | Split | Rank | Precision | Coverage |
|---|---|---:|---:|---:|
| pointwise tree10 | held-out fish | genus | 93.7% | 62.3% |
| pointwise tree10 | held-out fish | family | 96.4% | 98.8% |
| pointwise tree10 | unseen-genera | family | 75.9% | 95.2% |
| pointwise tree10 | unseen-genera | order | 85.8% | 99.9% |
| pairwise tree10 | held-out fish | genus | 94.4% | 58.7% |
| pairwise tree10 | unseen-genera | family | 72.6% | 99.6% |
| pairwise tree10 | unseen-genera | order | 86.1% | 99.9% |

Interpretation:

The selected-candidate calibrator does not fix missing-reference transfer. It
is useful evidence that the current failure is not just a missing neural layer
on top of the reranker; we need a better calibration design with realistic
missing-reference positives and normal supported negatives.

## Missing-Reference-Aware Calibrator

We then trained the deployment-matched follow-up:

- script:
  `scripts/edna/train_paper1_missing_reference_aware_calibrator.py`;
- source-table collector:
  `scripts/edna/build_paper1_missing_reference_calibrator_summary.py`;
- output root:
  `results/paper1_phylo_calibrated_assignment/dl_evidence_rank_backoff/coi_mlp_seed1401_missing_reference_aware_v2_gap/`;
- source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/missing_reference_aware_calibrator_summary.csv`.

Training protocol:

- train/calibrate on normal supported seen-test rows plus strict held-out fish
  hidden species/genus/family rows;
- evaluate on normal held-out fish, normal unseen-genera, and strict
  unseen-genera hidden species/genus/family rows;
- use production-v1 evidence plus v2 reference-gap probabilities as soft
  features;
- avoid split labels, roles, and candidate-count leakage as features.

Training loss behaved normally: calibration loss dropped from 0.499 at epoch 1
to 0.245 at epoch 100.

Species-disabled target-0.99:

| Split | Coverage | Assigned Precision | Species Calls |
|---|---:|---:|---:|
| held-out fish | 91.0% | 98.3% | 0 |
| unseen-genera | 79.7% | 95.7% | 0 |

Strict unseen-genera stress at target-0.99:

| Hidden Rank | Coverage | Assigned Precision | Species Calls |
|---|---:|---:|---:|
| species | 37.3% | 94.7% | 0 |
| genus | 37.9% | 94.8% | 0 |
| family | 28.6% | 78.7% | 0 |

Interpretation:

This is a real precision-first improvement over the first MLP: it learns from
missing-reference positives and reaches higher normal unseen-genera precision.
The cost is coverage. Hidden-family precision is better than the first MLP but
still too weak for a final production claim. Keep this as an optional
precision/reasoning layer, not the default policy.

## Next Experiments

1. Add explicit reason labels to production output:
   likely missing species/genus/family, weak candidate evidence, or broader
   rank is supported.
2. Use reranker scores as soft evidence inside the conservative production
   policy rather than replacing thresholds directly.
3. Repeat the missing-reference-aware calibrator with seed repeats only if we
   decide it is a candidate manuscript operating point.
4. Keep species disabled unless independent species thresholds transfer cleanly.
