# Paper 1: Uncertainty-Aware Barcode And eDNA Inference

Working title:

> Fast tree-aware molecular biodiversity inference with calibrated rank
> assignment under missing and ambiguous evidence.

Start here for current numbers:

- `experiments/paper1_phylo_calibrated_assignment/CURRENT_RESULTS.md`
- `experiments/paper1_phylo_calibrated_assignment/PRODUCTION_PIPELINE_V1.md`
- `experiments/paper1_phylo_calibrated_assignment/PIPELINE_NOVELTY_DEEP_DIVE.md`
- `experiments/paper1_phylo_calibrated_assignment/BREAKTHROUGH_AGENDA.md`
- `experiments/research_program/method_angles/README.md`

## Strongest Results Right Now

These are the results that currently carry the most weight.

1. **MarkerMirror + BLASTN builds a very strong high-rank 12S candidate set.**
   On 3,566 full-query 12S sequences, MarkerMirror + BLASTN top50 union support
   is 9.5 / 92.1 / 95.3 / 99.7% for species/genus/family/order. VSEARCH
   independently supports the same result at 9.5 / 91.8 / 95.1 / 99.6%.
   Species remains low because the current same-marker reference table lacks
   the held-out query species by design; genus/family/order recovery is the
   important result.
2. **The executable MarkerMirror 12S wrapper can make high-confidence order
   calls and abstain otherwise.** The conservative mode emits 880 order calls
   over 3,566 full-query 12S rows at 99.7% diagnostic precision. The explicit
   high-coverage order diagnostic reaches 57.2% held-out coverage at 99.8%
   precision with target-0.99 met in 50/50 species-split repeats. Family/genus
   remain disabled; lineage/reference-coverage and VSEARCH-backed
   marker-resolvability features were tested, including production-available
   VSEARCH cluster features, and did not stabilize those ranks.
3. **The conservative COI rank/no-call pipeline avoids false species calls.**
   CNN seed1206 with p-distance rerank target-0.99 gives 95.8% coverage /
   93.0% precision on held-out fish and 92.3% coverage / 83.9% precision on
   unseen genera, with 0.0% false species calls in both.
4. **A small DL decision layer improves conservative COI precision.**
   Species-disabled MLP target-0.99 gives 94.2% coverage / 97.4% precision on
   held-out fish and 88.5% coverage / 93.5% precision on unseen genera, with
   0.0% false species calls.
5. **Vector retrieval is fast enough for a practical first-pass candidate
   generator.** Controlled Vast timing reached 0.397 ms/query for exact vector
   search and about 0.005 ms/query for HNSW. This is candidate retrieval only,
   not the full pipeline.
6. **The 12S/eDNA Eco-Phylo posterior supports conservative higher-rank calls.**
   Species is disabled because species thresholds do not transfer cleanly; the
   species-disabled target-95 posterior assigns about 40% of held-out queries at
   about 94% accuracy.

## Core Question

What taxonomic claim is justified by an imperfect marker sequence when the
reference database, marker resolution, tree position, and ecological context may
all be incomplete?

The output should not be forced species classification. It should be:

```text
species / genus / family / order / no-call
```

with a reason for uncertainty.

## Merged Manuscript Decision

Paper 1 now absorbs the former Paper 2 workstream. The merged paper has one
scientific spine:

> Biodiversity inference from short marker sequences should be treated as
> uncertainty-aware evidence integration, not forced species classification.

The two empirical regimes have different jobs:

- COI fish barcode work tests whether short barcodes and retrieval/classical
  tools can recover species-tree structure, detect missing references, and
  support calibrated rank/no-call decisions.
- 12S/eDNA work tests the harder real-world setting where the marker itself is
  often species-ambiguous and ecological/geographic context may be needed to
  justify deeper taxonomic claims.

This merge strengthens the paper because the COI track alone sits close to
Fernando 2025 and DEPP/H-DEPP. The unified paper becomes about a broader
missing-evidence problem: deciding what biodiversity claim is scientifically
defensible.

## Concept

Most barcode models predict labels. This work asks whether a sequence encoder
can learn or use a biological coordinate system: the fish species tree. The
encoder is not assumed to be Mamba: CNN, biLSTM, Transformer, SSM, pretrained
barcode models, and classical similarity methods should all be tested under the
same tree-space / rank-calibration protocol.

The output should not only be "species X". It should support:

- candidate species retrieval,
- tree-distance error,
- genus/family/order recovery,
- calibrated rank assignment,
- no-call when evidence is insufficient.

For 12S/eDNA, the same logic extends to marker resolvability and ecological
context:

- sequence-only evidence,
- tree/candidate proximity,
- geography/range or co-occurrence priors,
- marker information ceiling,
- sample/site-level validation.

The current 12S/eDNA evidence layer is now consolidated into manuscript-facing
source tables. Use
`results/paper1_phylo_calibrated_assignment/source_tables/edna_evidence_decomposition_matrix.csv`
for the full Global_eDNA evidence-arm matrix and
`edna_evidence_best_by_rank.csv` for the best forced top-k arms by rank. Use
`edna_rank_no_call_operating_points.csv` only as a diagnostic design table until
independent eDNA calibration exists. A first site-heldout calibration transfer
now exists under
`results/paper1_phylo_calibrated_assignment/global_edna_independent_rank_calibration/`;
it shows the current top-1 score-threshold policy is not strong enough for
high-accuracy eDNA rank/no-call claims. The stronger method step is now the
full candidate-level Eco-Phylo posterior with direct 12S sequence evidence and
candidate tree-neighborhood evidence. Its current safe policy disables species
and backs off genus -> family -> order.

## Working Pipeline

Paper 1 should now be treated as a barcode retrieval and inference pipeline,
not as a single neural architecture paper.

The literature boundary is strict: phylogenetic placement, neural tree
placement, vector barcode retrieval, probabilistic taxonomic uncertainty, and
12S+ecology have all been studied separately. Our differentiating target is the
evidence-accounting layer that combines these pieces and reports the deepest
defensible rank with reason codes.

The working system is:

1. Candidate generation:
   - BLAST,
   - VSEARCH,
   - k-mer search,
   - EPA-ng / pplacer / APPLES-style placement where feasible.
2. Neural tree-space encoder:
   - CNN / biLSTM / Transformer / Mamba encoders map COI barcodes into a
     species-tree coordinate space;
   - for 12S/eDNA, sequence encoders map ambiguous marker evidence into the
     same candidate species/taxonomy space where possible.
3. Evidence fusion:
   - sequence similarity,
   - classical placement score,
   - neural embedding distance,
   - tree distance,
   - reference completeness / nearest-reference diagnostics,
   - marker resolvability,
   - ecological/geographic/co-occurrence priors for eDNA.
4. Rank-adaptive decision:
   - species if evidence supports species,
   - otherwise genus,
   - otherwise family/order,
   - otherwise no-call.
5. Diagnostics:
   - ambiguous marker,
   - missing reference,
   - low confidence,
   - multiple near-equivalent candidates,
   - clade/rank where the evidence remains reliable.

This is the "learned BLAST-like" direction: not replacing BLAST, but adding a
learned tree-space and calibrated rank/no-call layer around strong classical
candidate generation.

There are two useful execution modes:

### Accuracy-First Mode

Use classical methods as the broad candidate generator:

1. BLAST/VSEARCH/k-mer/EPA/APPLES generate candidate lists.
2. Neural tree-space scores and reference diagnostics rerank or interpret the
   candidates.
3. Rank-adaptive calibration decides species/genus/family/order/no-call.

This mode is best for paper validation because it compares against the strongest
classical tools directly.

### Vector-First Mode

Use learned embeddings as the fast candidate generator:

1. Encode every reference barcode once.
2. Store reference embeddings in an approximate-nearest-neighbor vector index.
3. Encode a query barcode once and retrieve top-k candidates by vector search.
4. Rerank only the top-k candidates with identity/alignment/tree-aware scores.
5. Apply rank-adaptive confidence and reference-gap detection.

This mode is the practical "fast learned BLAST" tool. It should be evaluated
separately on:

- retrieval speed,
- memory footprint,
- top-k candidate recall before reranking,
- final rank-adaptive accuracy after reranking,
- false species-call rate when references are missing.

The goal is not to make neural embeddings imitate every BLAST score. The goal is
to use a fast learned index to recover a small, biologically plausible candidate
set, then spend classical alignment/tree computation only where it matters.

Current controlled vector timing exists for CNN seed1206 Eval C on the Vast
RTX host:

- exact vector median: 0.397 ms/query;
- HNSW m16/ef50 median: 0.00475 ms/query;
- HNSW m32/ef50 median: 0.00513 ms/query.

These are vector-retrieval-only timings. They are not final deployment speed
claims and do not include downstream reranking/calibration, but they show that
the vector-first candidate layer can be very fast once indexed.

Current executable COI pipeline exists:

- `scripts/edna/run_paper1_coi_pipeline.py`
- `experiments/paper1_phylo_calibrated_assignment/runs/10_run_executable_coi_pipeline.sh`
- source summary: `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_run_summary.csv`

CNN seed1206 with exact vector retrieval and the target-0.99
missing-reference-aware policy gives:

- Eval C: 96.1% coverage, 90.0% assigned precision, 1.61% false species-call
  rate, 0.108 ms/query vector search.
- unseen-genera: 93.7% coverage, 83.7% assigned precision, 0.066% false
  species-call rate, 0.095 ms/query vector search.

The same executable path now has two additional experimental modes:

- HNSW candidate retrieval: faster approximate retrieval with slightly shifted
  assignment metrics. Current Eval C is 0.038 ms/query at 89.5% assigned
  precision; unseen-genera is 0.073 ms/query at 82.8% assigned precision.
- train-reference p-distance reranking over retrieved candidates: uses only
  `train_species_sequences.json`, not held-out query species sequences. It
  improves Eval C precision/false species-call rate in the raw inherited-
  threshold run (92.0% precision, 0.73% false species-call rate) but slightly
  worsens unseen-genera false species calls before rerank-specific calibration.

Rerank-specific calibration has now been added:

- `scripts/edna/calibrate_paper1_pipeline_modes.py` learns thresholds on
  seen-test pipeline rows for each executable mode.
- At target 0.99, calibrated p-distance reranking gives Eval C 95.8% coverage,
  93.0% assigned precision, and 0.0% false species-call rate.
- On unseen-genera it gives 92.3% coverage, 83.9% assigned precision, and 0.0%
  false species-call rate.

This is scientifically useful because the calibrated p-distance mode refuses to
make species calls at target 0.99 under missing-reference conditions, while
still assigning many queries to genus/family/order.

The first trainable DL decision layer has now been added:

- `scripts/edna/train_paper1_coi_evidence_model.py`;
- inference adapter: `scripts/edna/apply_paper1_coi_evidence_model.py`;
- optional FASTA/CSV CLI mode: `--decision-mode dl_mlp_species_disabled`;
- roadmap:
  `experiments/paper1_phylo_calibrated_assignment/DL_MODEL_ROADMAP.md`;
- output:
  `results/paper1_phylo_calibrated_assignment/dl_evidence_rank_backoff/coi_mlp_seed1206_pdistance/`.

This MLP learns from vector scores, p-distance traces, and taxonomic consensus
features. In species-disabled target-0.99 mode it gives:

- Eval C: 94.2% coverage, 97.4% assigned precision, 0.0% false species-call
  rate. Bootstrap 95% intervals: coverage 93.8-94.7%, precision 97.1-97.7%.
- unseen-genera: 88.5% coverage, 93.5% assigned precision, 0.0% false
  species-call rate. Bootstrap 95% intervals: coverage 87.8-89.2%, precision
  93.0-94.0%.

This is a promising decision-layer improvement over hand thresholds, but it is
not yet the default production mode. It is now integrated and smoke-tested as
an optional species-disabled decision layer. Seed repeats across MLP seeds
1206/1207/1208 are stable:

- Eval C: 94.2-96.0% coverage, 97.1-97.4% assigned precision, 0.0% false
  species-call rate.
- unseen-genera: 88.5-91.3% coverage, 92.9-93.5% assigned precision, 0.0%
  false species-call rate.

Strict-pack tests are still needed before making it the manuscript default.

Production v1 now packages this operating point:

- script: `scripts/edna/run_paper1_production_v1.py`;
- raw split-sequence wrapper:
  `scripts/edna/run_paper1_raw_sequence_production_v1.py`;
- FASTA/CSV specimen CLI:
  `scripts/edna/run_paper1_fasta_inference_v1.py`;
- wrapper: `experiments/paper1_phylo_calibrated_assignment/runs/12_run_production_v1.sh`;
- summary: `results/paper1_phylo_calibrated_assignment/production_v1/production_v1_summary_all.csv`;
- CLI usage doc:
  `experiments/paper1_phylo_calibrated_assignment/PRODUCTION_CLI_V1.md`.

Production v1 now runs from saved embeddings, clean split sequence tables, and
specimen-style FASTA/CSV inputs. The remaining hardening step is API/web
packaging and external demo polish, not the core command-line inference path.

Strict missing-reference retraining now directly stress-tests that behavior.
For CNN seed1206, all six strict pruned packs completed:

- Eval C hide species: species top10 0.0, genus/family/order top10
  41.8 / 62.9 / 83.9.
- Eval C hide genus: species/genus top10 0.0, family/order top10
  56.3 / 75.1.
- Eval C hide family: species/genus/family top10 0.0, order top10 40.9.
- unseen-genera hide species: species top10 0.0, family/order top10
  53.3 / 82.9; genus remains essentially unsupported.
- unseen-genera hide genus: species/genus top10 0.0, family/order top10
  47.4 / 80.7.
- unseen-genera hide family: species/genus/family top10 0.0, order top10
  51.0.

This is exactly the rank-backoff pattern we want: when a rank is not supported
by the candidate/reference evidence, the system should not force that rank.

This is the first actual pipeline run. It is not just a ledger row, but it
still lacks ecological context inside the same executable path and larger
deployment-scale speed stress tests.

## Fernando-Style Classical Placement Status

We are now much closer to a Fernando et al.-style protocol than before:

- 30 matched backbone-completeness sweeps have been generated and run:
  random and family-stratified sampling at 99/80/60/40/20% completeness with 3
  replicates each.
- EPA-ng and official APPLES 2.0.11 have both been run on the same sweep
  matrix.
- Outputs have been copied and scored under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_final_30/`.

Final public-setup placement diagnostics:

- APPLES placed-clade genus/family/order: 32.8 / 57.2 / 65.6%.
- EPA-ng placed-clade genus/family/order: 17.3 / 45.2 / 57.0%.
- APPLES sister-clade any-overlap/exact: 42.5 / 21.4%.
- EPA-ng sister-clade any-overlap/exact: 14.8 / 3.2%.

This supports the statement that we ran a Fernando-inspired completeness
protocol with the same broad families of classical placement methods. It is
not an exact Fernando reproduction because our reference set, tree extraction,
reduced-backbone construction, and PCP implementation are not identical to
Fernando et al.'s released workflow. In writing, use "Fernando-style" or
"Fernando-inspired matched completeness sweeps", not "we reproduced Fernando".

## Why It Matters

Biodiversity inference is often made from incomplete references and imperfect
markers. A forced species label can be misleading. A tree-aware, calibrated
system can say what the molecular evidence actually supports.

## Gaps Addressed

- Barcode foundation models rarely evaluate real species-tree geometry.
- Phylogenetic placement methods are not usually framed as practical calibrated
  barcode/eDNA assignment systems.
- Neural barcode classifiers often force labels rather than reporting the
  deepest defensible rank.
- Reference-library coverage is usually treated as a caveat, not as a measured
  uncertainty driver.
- 12S/eDNA species assignment is often evaluated as top-1 prediction even when
  the marker cannot theoretically resolve species.
- Ecology-aware systems such as TAXDNA show context helps, but marker
  resolvability, evidence decomposition, and rank/no-call reliability should be
  first-class evaluation objects.

## Fernando 2025 Comparator

Fernando, Fu, and Adamowicz 2025 is a required direct comparator, not a side
citation. They already tested COI barcode placement onto a fish backbone tree
using EPA-ng and APPLES under backbone-completeness and species-representation
ablations.

Paper 1 must therefore not claim that fish COI phylogenetic placement itself is
new. Our contribution must be narrower and more useful:

- compare neural tree-space encoders against BLAST/VSEARCH/k-mer and
  EPA-ng/pplacer/APPLES-style placement on the same held-out splits;
- convert candidate scores into species/genus/family/order/no-call decisions;
- measure missing-reference behavior directly by hiding true species/genera;
- report tree-distance error and rank-backoff, not only forced species top-k;
- make the reference-completeness and calibration diagnostics explicit.
- validate missing-reference behavior strictly by removing hidden taxa before
  candidate-tree construction and encoder training, not only after rankings.

Fernando-aligned outputs:

- EPA-ng outputs have been parsed and scored for all three clean splits.
- A labelled local APPLES-like p-distance placement diagnostic has been run for
  all three clean splits.
- LWR-binned EPA-ng summaries, placed-clade containment, placement tree
  distance, and rank-backoff summaries exist.
- A Fernando-like edge-to-sister diagnostic exists.
- A closer simulated-placement-tree PCP diagnostic now exists: each EPA-ng
  top-LWR edge is grafted into a simulated placement tree, then sister support
  is compared against the full fish tree.
- Matched completeness-sweep input packs were generated under
  `data/phylo/fernando_completeness_sweeps/`; the Vast/Linux EPA-ng runner is
  `experiments/paper1_phylo_calibrated_assignment/runs/07_vast_fernando_completeness_sweeps.sh`,
  and the official APPLES runner is
  `experiments/paper1_phylo_calibrated_assignment/runs/11_vast_fernando_apples_sweeps.sh`.
- All 30 matched sweeps have completed for EPA-ng and official APPLES 2.0.11.
  Final scored outputs are under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_final_30/`.
- The first disabled-re-estimation APPLES trial was moved aside and is not used
  as the official comparator.
- Strict missing-reference input packs have been generated under
  `data/phylo/paper1_strict_missing_reference_inputs/`; the Vast/Linux runner is
  `experiments/paper1_phylo_calibrated_assignment/runs/08_vast_strict_missing_reference_cnn.sh`.
- Local APPLES-like rows remain separate p-distance diagnostics and must not be
  described as Fernando's APPLES. Official APPLES rows come only from the
  completed APPLES 2.0.11 sweep outputs.

Current placement status:

- EPA-ng has completed and been scored for Eval C, seen-test, and
  unseen-genera.
- Local APPLES-like distance placement has completed and been scored for Eval C,
  seen-test, and unseen-genera.
- Current placed-clade containment diagnostics by species/genus/family/order:
  - Eval C: 0.0 / 45.9 / 67.8 / 74.3.
  - seen-test: 26.2 / 42.4 / 63.6 / 72.1.
  - unseen-genera: 0.0 / 0.0 / 37.0 / 51.0.
- Current APPLES-like nearest-reference match rates:
  - Eval C: 54.4.
  - seen-test: 78.8.
  - unseen-genera: 22.1.
- The scorer now also emits LWR-binned rank summaries, rank-backoff summaries,
  and tree-distance-to-placed-clade medians:
  - Eval C median placement distance 56.1, median excess over nearest reference
    32.3.
  - seen-test median placement distance 54.4, median excess 54.4.
  - unseen-genera median placement distance 192.7, median excess 107.0.
- Fernando-like sister-clade exact match rates for EPA-ng are currently low:
  - Eval C: 7.1%.
  - seen-test: 4.7%.
  - unseen-genera: 0.6%.
  These values are not comparable to Fernando's PCP yet; they indicate that our
  current edge-to-sister adapter and split design do not reproduce the
  Fernando completeness protocol.
- Simulated-placement-tree species-representative PCP-like rates are:
  - Eval C exact/overlap: 7.3 / 22.1.
  - seen-test exact/overlap: 24.0 / 50.4.
  - unseen-genera exact/overlap: 0.2 / 45.8.
  This is closer to Fernando's sister-clade scoring than placed-clade
  containment, but it is still not a full Fernando PCP score because our
  current query universe and backbone-completeness protocol differ.
- Completed Fernando-style 30-sweep matrix:
  - APPLES placed-clade genus/family/order: 32.8 / 57.2 / 65.6%.
  - EPA-ng placed-clade genus/family/order: 17.3 / 45.2 / 57.0%.
  - APPLES sister-clade any-overlap/exact: 42.5 / 21.4%.
  - EPA-ng sister-clade any-overlap/exact: 14.8 / 3.2%.
  These are the closest Fernando-style comparator outputs in the repo, but they
  are not exact Fernando PCP.
- pplacer currently fails because the command needs a valid tree model/stats or
  reference package (`-s` or `-c`), so pplacer is not a valid comparator yet.
- Placement comparator decision: keep EPA-ng as the likelihood-placement
  comparator, keep the local APPLES-like diagnostic clearly labelled, leave
  pplacer blocked unless a valid refpkg/stats model is supplied, and report
  official APPLES only from the completed APPLES 2.0.11 sweep outputs.

Detailed notes:

- `experiments/paper1_phylo_calibrated_assignment/FERNANDO_2025_POSITIONING.md`
- `experiments/paper1_phylo_calibrated_assignment/PIPELINE.md`
- `experiments/paper1_phylo_calibrated_assignment/PLACEMENT_COMPARATOR_DECISION.md`

## Existing Implementation

Primary implementation track:

- `experiments/fish_tree_clean/`
- `experiments/taxdna_ssm/`
- `experiments/paper2_eco_phylo_edna/`
- `experiments/stalder_reproduction/`

Important scripts:

- `scripts/edna/progress_logging.py`
- `scripts/edna/train_12s_phylo_mamba.py`
- `scripts/edna/train_fish_tree_encoder_benchmark.py`
- `scripts/edna/eval_phylo_checkpoint_tree_recovery.py`
- `scripts/edna/eval_zero_shot_candidate_predictions.py`
- `scripts/phylo_tree_distance_baselines.py`
- `scripts/summarize_results_ledger.py`
- `scripts/edna/build_12s_resolvability_map.py`
- `scripts/edna/build_12s_near_exact_resolvability.py`
- `scripts/edna/train_npz_cooccurrence_model.py`
- `scripts/edna/eval_global_edna_learned_cooccurrence.py`
- `scripts/edna/build_global_edna_calibration_matrix.py`
- `scripts/edna/build_ann_vector_retrieval_benchmark.py`
- `scripts/edna/build_ann_vector_stress_benchmark.py`
- `scripts/edna/eval_apples_like_distance_placement.py`
- `scripts/edna/build_placement_tree_error_tables.py`
- `scripts/edna/build_fernando_like_pcp_diagnostics.py`
- `scripts/edna/bootstrap_rank_no_call_policy.py`
- `scripts/edna/run_paper1_coi_pipeline.py`
- `scripts/edna/build_paper1_pipeline_run_summary.py`
- `scripts/edna/calibrate_paper1_pipeline_modes.py`
- `scripts/edna/build_paper1_end_to_end_summary.py`

Logging rule:

- Paper 1 Python scripts should log progress with
  `scripts/edna/progress_logging.py`.
- Default logs go to
  `results/paper1_phylo_calibrated_assignment/logs/{script_name}.log`.
- Vast wrappers should keep writing wrapper PIDs and phase logs under their
  result-specific `logs/` folders.

Important local results:

- `results/remote_runs/2026-05-30/rtx_pro_6000/coi_fish_tree_clean_phylo_mamba_cosine512_seqval*`
- `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/`
- `results/paper1_phylo_calibrated_assignment/source_tables/`
- `results/remote_runs/2026-05-30/rtx_pro_6000/taxdna_ssm/`
- `results/remote_runs/2026-05-30/rtx_pro_6000/resolvability_near_exact/`

Comparator matrix:

- `experiments/paper1_phylo_calibrated_assignment/COMPARATOR_MATRIX.md`
- `experiments/paper1_phylo_calibrated_assignment/DEPP_HDEPP_NOTES.md`
- `experiments/paper1_phylo_calibrated_assignment/LITERATURE_GAPS.md`
- `experiments/paper1_phylo_calibrated_assignment/MANUSCRIPT_ASSETS.md`

## Current Evidence

### COI: Tree-Space And Missing-Reference Benchmark

Clean COI cosine512 seed repeats support a real signal, but they do not yet
make a complete paper by themselves.

Eval C zero-shot/reference tree recovery:

- seed1206: Pearson 0.914, Spearman 0.859
- seed1207: Pearson 0.916, Spearman 0.862
- seed1208: Pearson 0.921, Spearman 0.867

Unseen-genera zero-shot/reference tree recovery:

- seed1206: Pearson 0.859, Spearman 0.821
- seed1207: Pearson 0.858, Spearman 0.820
- seed1208: Pearson 0.860, Spearman 0.824

Encoder benchmark and repeat status:

- CNN tree recovery is stronger than the current Mamba run and is stable across
  seeds:
  - Eval C zero-shot/reference: Pearson 0.938, Spearman 0.885
  - unseen-genera zero-shot/reference: Pearson 0.925, Spearman 0.875
  - seed1207 Eval C zero-shot/reference: Pearson 0.944, Spearman 0.891
  - seed1208 Eval C zero-shot/reference: Pearson 0.941, Spearman 0.887
  - seed1207 unseen-genera zero-shot/reference: Pearson 0.916, Spearman 0.866
  - seed1208 unseen-genera zero-shot/reference: Pearson 0.908, Spearman 0.865
- Mamba remains better than CNN on Eval C species top10 retrieval:
  - Mamba: 10.84
  - CNN: 7.94
- CNN is stronger than Mamba on unseen-genera neural retrieval:
  - CNN species/genus/family/order top10: 7.77 / 29.68 / 77.74 / 90.97
  - Mamba species/genus/family/order top10: 5.37 / 21.69 / 62.92 / 81.74

Classical baseline finding:

- BLAST/VSEARCH/k-mer dominate higher-rank retrieval when nearby reference
  sequences exist.
- They score 0.00 species top10 on Eval C and unseen-genera because the true
  species are absent from the reference sequence database.

Current scientific read:

- Sequence-to-tree supervision works for fish COI barcode embeddings.
- The effect is not Mamba-specific.
- The current strongest result is tree-distance recovery for held-out taxa,
  not exact species classification.
- Vector-first retrieval is promising as a practical speed layer: local exact
  cosine search over saved CNN embeddings retrieves all Eval C queries in about
  one second while preserving the same candidate-ranking metrics as the neural
  zero-shot outputs. This is a tool-engineering result for now, not a
  controlled speed claim.

### 12S/eDNA: Marker Ambiguity And Ecological Context

Completed 12S/eDNA evidence:

- exact-Teleo SSM vs CNN;
- broad multisource 12S SSM vs CNN;
- Global_eDNA sequence-only validation;
- RLS/OBIS learned co-occurrence;
- public FISHGLOB learned co-occurrence reconstruction;
- exact and near-exact 12S resolvability maps.

Current 12S/eDNA read:

- species-level open-candidate 12S assignment is hard and often biologically
  underdetermined;
- higher-rank 12S/eDNA assignment is more stable and more scientifically
  defensible;
- ecological context can help, but it must be ablated because context can also
  bias or degrade predictions;
- near-exact 12S resolvability provides the biological reason rank/no-call
  output is necessary.

Near-exact query oracle support at 99% identity:

- multisource species/genus/family/order: 77.9 / 95.2 / 99.6 / 99.7
- multisource Teleo species/genus/family/order: 70.7 / 90.9 / 97.3 / 100.0
- rCRUX cleaned species/genus/family/order: 95.4 / 100.0 / 100.0 / 100.0
- Mitohelper full Teleo species/genus/family/order: 70.8 / 93.4 / 97.5 / 99.6

Near-exact query oracle support at 95% identity:

- multisource species/genus/family/order: 38.3 / 73.8 / 92.0 / 98.2
- multisource Teleo species/genus/family/order: 42.7 / 71.2 / 89.4 / 99.6
- rCRUX cleaned species/genus/family/order: 54.8 / 98.2 / 100.0 / 100.0
- Mitohelper full Teleo species/genus/family/order: 33.2 / 69.0 / 87.8 / 99.5

## Source Tables

Current local source tables:

- `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/tree_recovery_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/tree_distance_bin_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/tree_distance_sample_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/neighborhood_preservation.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/candidate_ablation_rank_backoff.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/full_candidate_embedding_ablation.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/full_candidate_embedding_ablation_cnn_seed_repeats.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/vector_first_retrieval_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/vector_first_runtime_comparison.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/ann_vector_runtime_comparison.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/ann_vector_stress_runtime.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/apples_like_distance_placement_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/placement_tree_error_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/placement_pcp_like_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_end_to_end_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/merged_12s_resolvability_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/merged_12s_zero_shot_model_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/merged_global_edna_asv_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/merged_global_edna_sample_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/merged_edna_evidence_arm_status.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/merged_global_edna_calibration_curves.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_component_status.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_coi_method_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_placement_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_edna_method_benchmark.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_best_by_task.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/pipeline_next_actions.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/reference_diagnostics_summary.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/coverage_calibration_curves.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/rank_adaptive_policy_summary.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/prospective_rank_thresholds.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/prospective_rank_adaptive_policy_summary.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/missing_reference_aware_policy_bootstrap.csv`

Builder:

- `scripts/edna/build_paper1_source_tables.py`
- `scripts/edna/build_merged_paper1_edna_source_tables.py`
- `scripts/edna/build_paper1_pipeline_benchmarks.py`
- `scripts/edna/build_paper1_end_to_end_summary.py`

Current diagnostic use:

- retrieval tables support the main method/baseline comparison;
- tree-distance bin tables support the claim that embeddings can be audited as
  biological coordinate systems, not just classifiers;
- neighborhood preservation tables quantify whether top candidates are enriched
  for true genus/family/order relative to the whole candidate tree;
- candidate-ablation tables prototype the missing-reference/rank-backoff
  question by hiding true species/genus/family from saved top-50 rankings.
- vector-first tables benchmark exact cosine search over saved embeddings as a
  dependency-light proxy for the fast retrieval layer. HNSW ANN tables add
  recall against exact vector search, and the stress table expands the
  reference catalog synthetically up to 25x for speed/memory testing. These
  timings are still local source-table timings and should be rerun under a
  controlled environment before publication claims; synthetic-expanded rows are
  not biological retrieval accuracy.
- merged 12S/eDNA tables summarize marker-information ceilings, 12S zero-shot
  SSM/CNN metrics, and Global_eDNA ASV/sample-level validation so the former
  Paper 2 evidence is now part of the same manuscript ledger.
- `merged_edna_evidence_arm_status.csv` is the guardrail table for the Stalder
  comparison: sequence/tree encoder, learned co-occurrence, pure
  geography/range-only, and same-sample co-occurrence-only arms are now
  present.
- `merged_global_edna_calibration_curves.csv` gives first diagnostic
  Global_eDNA rank/no-call curves for current SSM/CNN sequence-only and learned
  co-occurrence predictions. These are top-1 score-threshold curves and are not
  the final eDNA rank/no-call protocol.
- `eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/` is the
  current strongest eDNA posterior layer. Species-disabled target-95 backoff
  assigns 40.3% of held-out eDNA queries at 94.3% accuracy, and 30 nested
  threshold resplits average 40.2% assignment at 94.3% accuracy.
- `eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/`
  is the stricter true nested posterior check. The model is fit on 70% of
  calibration groups and thresholds are learned on the remaining calibration
  groups. Species still fails; family/order target-95 transfer individually,
  while mixed species-disabled target-95 assigns 38.9% of held-out queries at
  93.4% accuracy.
- pipeline tables provide the paper-level benchmark ledger. They summarize what
  is available, what is partial, best observed task metrics, and next actions.
  `pipeline_best_by_task.csv` is intentionally descriptive; it is not a list of
  claim-ready winners.

Rank-adaptive calibration:

- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/`
  now includes Mamba seeds, CNN, biLSTM, Transformer, BLAST, VSEARCH, k-mer,
  and negative controls.
- The calibration folder now includes two levels:
  - same-split diagnostic curves;
  - seen-test-to-heldout prospective threshold transfer for Eval C and
    unseen-genera;
  - missing-reference-aware consensus thresholds learned on seen-test and
    evaluated on Eval C/unseen-genera.
- The prospective pass is more honest than the same-split curves, and it shows
  why final calibration is nontrivial: thresholds learned on seen-test do not
  reliably preserve their nominal precision under missing-species and
  unseen-genera shifts. The current tables include Wilson 95% intervals for
  assigned precision.
- The current best candidate locked policy is the 0.99 missing-reference-aware
  consensus setting. For CNN seed1206, it gives:
  - Eval C: 90.0% assigned precision at 96.1% coverage, with 187 species,
    969 genus, 7922 family, 2061 order, and 455 no-call assignments.
  - unseen-genera: 83.7% assigned precision at 93.7% coverage, with 6 species,
    305 genus, 5606 family, 2652 order, and 579 no-call assignments.
  Bootstrap intervals and false species-call estimates now exist in
  `missing_reference_aware_policy_bootstrap.csv`. This is still a candidate
  locked operating point, not the final publication policy; it needs a final
  operating-point decision and a strict independent validation or tree-pruned
  retraining protocol.

Current caution:

- candidate-ablation is post-hoc over saved top-50 lists; it is not a retrained
  tree-pruned experiment;
- reference diagnostics exist for BLAST/VSEARCH/k-mer, but the
  `nearest_reference_tree_distance` label-normalization bug is fixed in the
  local regenerated diagnostics. Eval C and unseen-genera now have real
  nearest-reference tree-distance bins; seen-test is zero-distance by design
  because the query species are represented in the reference set;
- CNN/biLSTM/Transformer query embeddings are now copied locally, and CNN repeat
  embeddings are copied for seeds 1207/1208;
- Mamba query embeddings remain blocked on this Vast image because
  `mamba-ssm` did not build cleanly against PyTorch 2.12 / CUDA 13.

## Publication Readiness

Current status:

- strong merged-paper foundation,
- testing/source-table documentation is now in place,
- official APPLES/EPA-ng Fernando-style completeness sweeps are complete,
- strict independent missing-reference validation is complete for the CNN
  seed1206 pruned packs,
- still not a complete publishable contribution until manuscript figures,
  final claim boundaries, eDNA calibration, and end-to-end speed framing are
  tightened.

Why not enough yet:

- Literature review is still too thin around phylogenetic placement,
  barcode/tree reconstruction, distance-based barcode methods, and calibrated
  abstention.
- CNN outperforming Mamba on tree geometry means the novelty cannot be
  "MarineMamba is best."
- Tree-recovery Pearson/Spearman is useful but not enough by itself; we need
  richer tree-geometry diagnostics.
- Calibration now includes split-transfer tests, Wilson intervals, a
  candidate locked missing-reference policy, and bootstrap intervals, but final
  operating thresholds still need an independent validation protocol.
- EPA-ng, local APPLES-like placement, and official APPLES 2.0.11
  Fernando-style completeness sweeps now exist because Fernando 2025 already
  covers the classical fish-COI placement question. Exact Fernando PCP remains
  open only if direct reproduction is needed.
- We need to show why the benchmark changes what biodiversity inference can do:
  fast candidate retrieval, tree-aware uncertainty, missing-reference behavior,
  rank-backoff, and no-call.
- TAXDNA already covers the broad "sequence + tree + ecology helps eDNA" idea,
  so our 12S/eDNA contribution must be resolvability ceilings, transparent
  evidence decomposition, and calibrated rank/no-call reliability.

Potential publishable claim after more work:

> Short marker sequences support different levels of biodiversity inference
> depending on marker resolution, reference completeness, tree position, and
> ecological context. A defensible molecular biodiversity system should combine
> fast candidate retrieval, tree-aware scoring, reference diagnostics,
> ecological priors where justified, and calibrated rank/no-call assignment.

## Remaining Work

Paper-critical:

1. Finish COI placement comparators:
   - EPA-ng: completed for Eval C, seen-test, and unseen-genera;
   - local APPLES-like distance placement: completed for Eval C, seen-test,
     and unseen-genera;
   - pplacer: blocked until a valid refpkg/stats model is supplied;
   - official APPLES: completed on the Fernando-style completeness sweeps;
   - Fernando-style scoring: completed for our public setup, but not exact
     Fernando PCP.
2. Extend placement-output scoring:
   - `jplace` parsing, LWR-binned accuracy, tree-distance-to-placed-clade,
     and species/genus/family/order/rank-backoff now exist in
     `scripts/edna/score_fish_tree_placement_outputs.py`;
   - Fernando-like sister-clade diagnostics now exist in
     `placement_pcp_like_summary.csv`;
   - completed matched sweep diagnostics are in
     `fernando_completeness_final_30/`;
   - still add exact Fernando PCP only if we choose to reproduce their exact
     scoring definition.
3. Expand COI tree-geometry analysis:
   - learned distance vs true tree distance scatter/bin tables: first source
     tables done;
   - residuals by clade, genus, family, and nearest-reference distance;
   - neighborhood preservation: first source table done;
   - same-genus/same-family/same-order enrichment,
   - embedding-derived tree reconstruction against the reference tree.
4. Add direct tree-distance baselines:
   - k-mer distance vs tree distance,
   - BLAST distance/identity vs tree distance,
   - VSEARCH similarity vs tree distance,
   - compare these to neural embedding distances.
5. Convert vector-first retrieval into a benchmark:
   - reference embedding index,
   - query latency,
   - top-k recall,
   - synthetic large-reference stress timing: first table done,
   - reranked rank-adaptive output,
   - speed/memory comparison against BLAST/VSEARCH/k-mer.
6. Make 12S resolvability paper-grade:
   - exact identity,
   - near-exact identity,
   - marker/window-aware filtering where possible,
   - species/genus/family/order oracle support.
   - first merged source table done:
     `merged_12s_resolvability_summary.csv`.
7. Build the unified evidence-decomposition matrix:
   - sequence only,
   - tree only,
   - ecology/geography only: first pure-prior rows done,
   - sequence + tree,
   - sequence + ecology,
   - sequence + tree + ecology,
   - calibrated rank/no-call for each arm.
8. Separate ASV-level and sample/site-level Global_eDNA validation.
9. Add reference-gap/active-curation diagnostics:
   - which missing references would reduce uncertainty most,
   - where sequence ambiguity is irreducible without another marker.
   - Exp131 adds active reference-curation/value-of-information source tables:
     `marker_mirror_active_reference_value_species.csv`,
     `marker_mirror_active_reference_value_lineage.csv`, and
     `marker_mirror_active_reference_value_actions.csv`.
10. Export Mamba query embeddings on a compatible image if they are needed for
   full-candidate ablation and embedding-derived tree reconstruction.
11. Refresh rank-adaptive calibration with an independent validation protocol.
12. Strengthen literature review around:
   - BarcodeBERT/BarcodeMamba and barcode foundation models,
   - classical barcode assignment and phylogenetic placement,
   - pplacer/EPA-ng/SEPP as direct fixed-tree placement baselines,
   - neural phylogenetic reconstruction as a neighboring but different task,
   - Euclidean and hyperbolic tree embeddings as representation geometry,
   - tree-aware or hierarchy-aware classification,
   - selective classification / abstention / calibration.
13. Build final figure panels after the above claims are stable.

Optional method ablation:

- hyperbolic/tree-geometry target if cosine512 remains stable against baselines.

Encoder benchmark details:

- `experiments/paper1_phylo_calibrated_assignment/ENCODER_BENCHMARKS.md`
