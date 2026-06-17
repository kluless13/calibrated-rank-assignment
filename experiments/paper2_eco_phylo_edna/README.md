# Former Paper 2: Eco-Phylo eDNA Work Package

Status:

> Merged into Paper 1 as the 12S/eDNA marker-ambiguity and ecological-context
> evidence package.

This folder remains as an implementation and audit trail. It should not be
treated as a separate manuscript unless the eDNA work grows beyond the merged
paper scope.

Working title:

> Phylogeny- and ecology-aware eDNA assignment from imperfect 12S evidence.

## Core Question

How much biodiversity inference is supported by 12S sequence evidence alone,
and when do phylogeny, geography, co-occurrence, calibration, and rank-adaptive
abstention turn ambiguous molecular evidence into defensible taxonomic claims?

In the merged manuscript, this becomes the stress test for the Paper 1 thesis:

> What rank of biodiversity claim is justified when marker evidence is
> intrinsically ambiguous?

## Claim Boundary Relative To TAXDNA

Stalder et al. already answer the broad first-order question of whether 12S
sequence evidence can be improved by adding phylogenetic embeddings,
co-occurrence/range information, and real eDNA validation. Their TAXDNA paper
uses a 12S reference database, a fish phylogeny, range/community
co-occurrence data, and global eDNA samples, and reports improved zero-shot
species assignment when ecological/phylogenetic context is added.

This track should therefore not be framed as:

> Can sequence + tree + ecology improve eDNA assignment?

That is too close to TAXDNA.

Our distinct contribution is a reliability framework:

> When is a species, genus, family, order, or no-call assignment justified by
> the available molecular, phylogenetic, reference, and ecological evidence?

Closest prior work:

- Stalder et al. 2025, PLOS Computational Biology:
  <https://doi.org/10.1371/journal.pcbi.1013776>

## Concept

For eDNA, the sequence alone is often not enough. Many 12S fragments are short,
conserved, or missing close references. The assignment should be a posterior:

```text
score(species | sequence, tree, ecology)
  = sequence score
  + tree/candidate proximity
  + co-occurrence or range prior
  - calibration/reference penalty
```

Each component must be separately measurable.

The sequence encoder should be swappable. The paper should be able to compare
SSM/Mamba, CNN, LSTM, Transformer, S5, and similarity baselines under the same
tree-space and ecological-posterior protocol.

The posterior is not the whole contribution. The key is deciding which
taxonomic claim is warranted:

```text
species if sequence + tree + ecology support species
genus   if species is underdetermined but genus is supported
family  if genus is underdetermined but family is supported
order   if family is underdetermined but order is supported
no-call if evidence is insufficient
```

## Why It Matters

Real biodiversity surveys do not need overconfident species guesses. They need
defensible evidence about what taxa are present, at the deepest rank supported
by the molecular, phylogenetic, and ecological data.

## Gaps Addressed

- eDNA models often over-focus on species top-1.
- Species-level 12S assignment can be biologically impossible when the marker
  does not distinguish candidate species.
- Ecological context is powerful but can be hard to reproduce or audit.
- Sequence-only neural models do not explain marker ambiguity or reference
  missingness.
- TAXDNA-style systems show that integrated evidence helps, but they do not
  make resolvability ceilings, rank-adaptive abstention, and reference-gap
  diagnostics the main scientific object.
- Exact Stalder/TAXDNA retraining remains separate from this track because the
  processed inputs are not currently all public in the repo/LFS assets we have
  found.

## Existing Implementation

Primary implementation tracks:

- `experiments/taxdna_ssm/`
- `experiments/stalder_reproduction/`

Important scripts:

- `scripts/edna/train_12s_phylo_mamba.py`
- `scripts/edna/train_taxdna_cnn_baseline.py`
- `scripts/edna/train_npz_cooccurrence_model.py`
- `scripts/edna/eval_global_edna_learned_cooccurrence.py`
- `scripts/edna/eval_global_edna_sample_validation.py`
- `scripts/edna/build_global_edna_calibration_matrix.py`

Important local results:

- `results/remote_runs/2026-05-30/rtx_pro_6000/taxdna_ssm/`

## Current Evidence

Completed:

- exact-Teleo SSM vs CNN
- broad multisource 12S SSM vs CNN
- Global_eDNA sequence-only validation
- RLS/OBIS learned co-occurrence
- public FISHGLOB learned co-occurrence reconstruction

Current strongest broad-12S signal:

- SSM improves over CNN on higher-rank open-candidate retrieval in the broad
  multisource setting.
- Species-level 12S open-candidate assignment remains difficult and should be
  handled with calibration/no-call and ecological context.
- Exact and near-exact 12S resolvability runs show that many Teleo-style
  held-out query species are not species-resolvable from sequence evidence
  alone, especially at more permissive identity thresholds.
- Global_eDNA evidence-decomposition source tables now separate sequence-only,
  geography-only, co-occurrence-only, sequence + geography, sequence +
  co-occurrence, and learned co-occurrence arms:
  `results/paper1_phylo_calibrated_assignment/source_tables/edna_evidence_decomposition_matrix.csv`.
- Diagnostic eDNA rank/no-call operating points now exist, but they are not
  claim-ready because they are derived from the current validation table rather
  than an independent calibration split:
  `results/paper1_phylo_calibrated_assignment/source_tables/edna_rank_no_call_operating_points.csv`.
- Site-heldout eDNA rank/no-call threshold transfer now exists:
  `results/paper1_phylo_calibrated_assignment/global_edna_independent_rank_calibration/global_edna_independent_rank_calibration_summary.csv`.
  It shows that the current top-1 score-threshold policy is still weak for
  high-accuracy eDNA rank/no-call: only modest family/order operating points
  transfer, and no 70%+ target is available.

## Remaining Work

Merged-paper critical:

1. Make the 12S resolvability/oracle ceiling paper-grade:
   - exact identity,
   - near-exact identity,
   - marker/window-aware filtering where possible,
   - species/genus/family/order oracle support.
2. Formalize the posterior into one repeatable script/config.
3. Run evidence decomposition ablations:
   - sequence only,
   - ecology only,
   - tree only,
   - sequence + tree,
   - sequence + ecology,
   - sequence + tree + ecology.
   First manuscript-facing Global_eDNA matrix is complete; the remaining gap is
   a single pre-registered Eco-Phylo posterior rather than a collection of
   rerank outputs.
4. Separate ASV-level and sample/site-level validation.
5. Add calibration/no-call curves for every method arm.
   Diagnostic curves and a site-heldout transfer test exist for current SSM/CNN
   arms; a stronger posterior is still required before positive eDNA rank/no-call
   claims.
6. Build rank-adaptive assignment:
   - species,
   - genus,
   - family,
   - order,
   - no-call.
7. Add reference-gap/active-curation diagnostics:
   - which missing references would reduce uncertainty most,
   - where sequence ambiguity is irreducible without another marker.
8. Keep Stalder exact reproduction separate from public Stalder-style
   reconstruction.
9. Add encoder-swap experiments once the posterior protocol is stable.
