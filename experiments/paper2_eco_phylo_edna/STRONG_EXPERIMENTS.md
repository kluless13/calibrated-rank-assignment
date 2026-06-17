# Strong Experiments For The 12S/eDNA Track

This file tracks the experiments that turn the 12S/eDNA work from "which model
scores higher?" into a broader scientific claim:

> Species-level eDNA assignment is not a sequence classification problem alone;
> it is an evidence integration problem under marker ambiguity, missing
> references, phylogeny, and ecology.

## Where These Experiments Belong

- These experiments now belong inside the merged Paper 1 manuscript as the
  12S/eDNA marker-ambiguity and ecological-context stress test.
- The COI work remains the Paper 1 tree-space, retrieval, missing-reference,
  and rank-calibration foundation.
- Paper 3 owns the multi-marker shared tree-space extension.
- `experiments/encoder_benchmarks/` owns the architecture-swap protocol.

## 1. 12S Resolvability Map

Status:

- Exact-identity pass: complete locally.
- Near-exact VSEARCH clustering pass: complete on Vast and copied locally.
- Current status: strong first result exists; needs marker/window-aware
  refinement before final paper figures.

Question:

- For every fish species in our 12S reference, is species-level assignment even
  theoretically possible from the marker?

Experiment:

1. Normalize 12S sequences by primer/window/source where needed.
2. Cluster sequences by exact identity and near-exact identity.
3. For each species, ask whether the sequence cluster uniquely identifies:
   - species,
   - genus,
   - family,
   - order,
   - or none of the above.
4. Compute an oracle upper bound: even a perfect sequence-only model cannot
   distinguish species that share the same or near-identical 12S signal.

Why this is remarkable:

- It turns poor species-level 12S accuracy into a biological result rather than
  only a model failure.
- It gives reviewers a ceiling: what 12S can and cannot support before ecology,
  another marker, or no-call is added.

Inputs already available:

- `data/edna/stalder_inputs/multisource/`
- `data/edna/stalder_inputs/multisource_teleo/`
- `data/edna/processed_12s_multisource/`
- `data/edna/processed_12s_rcrux_cleaned/`
- `data/edna/processed_12s_mare_mage/`

When:

- Already started and completed for the first exact and near-exact passes.
- Exact outputs:
  `results/edna/resolvability/`
- Near-exact outputs:
  `results/remote_runs/2026-05-30/rtx_pro_6000/resolvability_near_exact/`

Current evidence:

- Exact identity already showed that Teleo-style datasets have a species-level
  ceiling: roughly 21-24% of held-out query species were not species-resolvable
  by exact 12S sequence alone.
- Near-exact 99% identity query oracle support:
  - multisource species/genus/family/order:
    77.9 / 95.2 / 99.6 / 99.7
  - multisource Teleo:
    70.7 / 90.9 / 97.3 / 100.0
  - rCRUX cleaned:
    95.4 / 100.0 / 100.0 / 100.0
  - Mitohelper full Teleo:
    70.8 / 93.4 / 97.5 / 99.6
- Near-exact 95% identity query oracle support:
  - multisource species/genus/family/order:
    38.3 / 73.8 / 92.0 / 98.2
  - multisource Teleo:
    42.7 / 71.2 / 89.4 / 99.6
  - rCRUX cleaned:
    54.8 / 98.2 / 100.0 / 100.0
  - Mitohelper full Teleo:
    33.2 / 69.0 / 87.8 / 99.5

Remaining:

- Make the result marker/window aware where possible.
- Separate biological ambiguity from preprocessing/window heterogeneity.
- Convert exact/near-exact summaries into figure-ready source tables.

## 2. Sequence vs Tree vs Ecology Information Decomposition

Status:

- Manuscript-facing evidence-decomposition tables are now built.
- Not yet consolidated into one repeatable pre-registered Eco-Phylo Posterior
  protocol.

Question:

- Where does correct species assignment come from?

Experiment matrix:

- sequence only;
- tree only;
- geography only;
- co-occurrence only;
- sequence + tree;
- sequence + geography;
- sequence + co-occurrence;
- sequence + tree + ecology.

Evaluate on:

- held-out 12S candidate retrieval;
- Global_eDNA ASV-level validation;
- Global_eDNA sample/site-level validation.

Why this is remarkable:

- It tells us what actually solves eDNA assignment.
- If species accuracy improves only when ecology is added, the paper becomes
  about integrated biodiversity inference rather than model architecture.
- If ecology harms some sites or ranks, that is also valuable because it reveals
  bias/transfer limits.

Inputs already available:

- SSM/CNN predictions in `results/remote_runs/2026-05-30/.../taxdna_ssm/`
- RLS/OBIS co-occurrence JSONs.
- public FISHGLOB co-occurrence reconstruction.
- Global_eDNA validation query inputs.

When:

- Can start locally now because the main prediction and co-occurrence outputs
  are available.
- First step is to consolidate scattered rerank scripts into one
  `Eco-Phylo Posterior` protocol.
- GPU is not required unless we add new encoder training.

Current evidence available:

- SSM/CNN sequence-only Global_eDNA validation exists.
- RLS/OBIS occurrence/range priors exist.
- public FISHGLOB learned co-occurrence reconstruction exists.
- learned co-occurrence reranking exists for SSM and CNN.
- calibration matrix scaffolding exists under
  `results/edna/global_tropical_validation/calibration/`.
- Consolidated source tables exist under
  `results/paper1_phylo_calibrated_assignment/source_tables/`:
  - `edna_evidence_decomposition_matrix.csv`,
  - `edna_evidence_best_by_rank.csv`,
  - `edna_rank_no_call_operating_points.csv`.

Remaining:

- Convert the consolidated table into a pre-registered Eco-Phylo Posterior
  protocol with fixed weights/thresholds.
- Keep ASV-level and sample/site-level validation separate.
- Audit ecological leakage and site/range assumptions.

## 3. Rank-Adaptive Assignment

Status:

- Paper 1 COI first pass is implemented and run.
- Diagnostic and site-heldout merged Paper 1 12S/eDNA rank/no-call tables now
  exist.
- The result is not yet a positive rank/no-call claim: current top-1
  score-threshold policies only transfer at modest family/order operating
  points.

Question:

- If 12S cannot support species, can we still make reliable genus/family/order
  calls?

Experiment:

1. Build confidence curves for each method.
2. Output species, genus, family, order, or no-call.
3. Compare:
   - SSM/Mamba,
   - CNN,
   - LSTM,
   - Transformer,
   - S5 if implemented cleanly,
   - BLAST,
   - VSEARCH,
   - k-mer,
   - Eco-Phylo posterior.

Why this is remarkable:

- This is what real eDNA users need.
- A correct family call is better science than a fake species call.
- It turns model outputs into defensible biodiversity statements.

When:

- Started for Paper 1 after the baseline queue produced comparable prediction
  files.
- Reuse the same calibration/no-call machinery across COI and 12S.
- Mostly local CPU work once predictions exist.

Current implementation:

- `scripts/edna/build_rank_adaptive_calibration.py`
- Paper 1 outputs:
  `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/`

Current finding from COI:

- Rank-adaptive assignment is useful, but top1-margin-only confidence is too
  strict for some hard splits.
- The next pass should include top-k evidence, tree-distance diagnostics,
  nearest-reference bins, and missing-reference status.

Remaining for the merged Paper 1 eDNA work package:

- Apply rank-adaptive/no-call curves to:
  - SSM 12S,
  - CNN 12S,
  - BLAST/VSEARCH/k-mer 12S baselines,
  - Eco-Phylo posterior outputs.
- Use the resolvability ceiling to prevent unjustified species calls.
- Build a stronger pre-registered Eco-Phylo posterior and then rerun
  site-heldout calibration before reporting eDNA rank/no-call as a positive
  result.

## 4. Multi-Marker Shared Tree Space

Status:

- Planned, not started as an experiment.
- Belongs mainly to Paper 3 after the merged Paper 1 foundations are stable.

Question:

- Can COI help 12S through the species tree?

Experiment:

```text
COI encoder -> shared species-tree space
12S encoder -> shared species-tree space
```

Then test whether 12S predictions improve when the species coordinate system is
shaped by COI-rich data.

Why this is remarkable:

- It is the cleanest way for COI to help eDNA without falsely claiming that COI
  sequence motifs transfer into 12S.
- The transfer happens through species/tree coordinates, not raw sequence
  similarity.

When:

- Start after Paper 1 tree-space validation and merged Paper 1 eDNA
  resolvability mapping.
- First step is local data inventory: shared species IDs, taxonomy, and tree
  labels.
- Training needs GPU later.

Primary owner:

- `experiments/paper3_multimarker_shared_tree_space/`

## 5. Reference Gap / Active Curation Experiment

Status:

- Planned, not started.
- Now unblocked conceptually by the exact/near-exact resolvability outputs and
  Paper 1 reference diagnostics.

Question:

- Which missing reference sequences would most improve eDNA assignment?

Experiment:

1. For each ambiguous 12S query, identify unresolved species groups.
2. Estimate whether uncertainty is caused by:
   - no species-level 12S signal,
   - missing reference sequence,
   - missing close relative,
   - taxonomy/synonym problem,
   - geography/ecology conflict.
3. Simulate adding references.
4. Rank species/genera where new reference sequencing would most reduce
   uncertainty.

Why this is remarkable:

- It turns the model into a tool for biodiversity database improvement.
- It makes reference curation a measurable scientific output rather than a
  vague caveat.

When:

- Start after the 12S resolvability map and reference diagnostics exist.
- Mostly local CPU work.
- Works best after rank-adaptive calibration is available.

Next implementation idea:

- Start with the 12S near-exact clusters.
- For each ambiguous query cluster, mark whether uncertainty is caused by:
  - multiple same-rank species sharing marker signal,
  - no reference for the true species,
  - missing close relatives,
  - broad taxonomic ambiguity,
  - geography/ecology conflict.
- Summarize which species/genera would most reduce uncertainty if added to the
  reference library.

## Priority Order

Updated status-aware order:

1. Convert exact/near-exact 12S resolvability into figure-ready source tables.
2. Consolidate the Eco-Phylo posterior ablation into one repeatable protocol.
3. Extend rank-adaptive/no-call curves from COI to 12S/eDNA outputs.
4. Build the reference-gap/active-curation map from ambiguous 12S clusters.
5. Start multi-marker shared tree-space inventory for Paper 3.

## Immediate Next Step

Do not run another generic 12S model yet. The next merged Paper 1 eDNA step is
to formalize the Eco-Phylo Posterior ablation and connect it to the
resolvability ceiling and rank-adaptive/no-call outputs.

Concrete next local artifact:

- `scripts/edna/build_eco_phylo_posterior_ablation.py`
- output table:
  `results/edna/global_tropical_validation/eco_phylo_posterior_ablation/`
