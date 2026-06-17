# Paper 1 Log

## Status Overview

- Latest as of 2026-06-02: production-v1 remains the conservative COI
  operating point; DL rank/no-call is a precision-first optional mode; the
  first candidate-level reranker, reference-gap detector, and retrieval-DL
  encoder sweep are now complete as diagnostic model-development results. None
  replaces production-v1 yet.
- Exp 63 done: check. First diagnostic reference-gap detector source-tabled.
- Exp 64 done: check. First diagnostic candidate-level reranker source-tabled.
- Exp 65 done: check. Retrieval-DL encoder sweep copied and source-tabled.
- Exp 66 done: check. Retrieval-DL pipeline diagnostics plus BLAST/VSEARCH-aware
  candidate reranker training completed on Vast, outputs were copied locally,
  and source tables were refreshed.
- Exp 67 done: check. Retrieval-DL candidate reranker with explicit
  tree-neighborhood features completed on Vast, outputs were copied locally,
  and source tables were refreshed.
- Exp 68 done: check. Focused query-listwise tree10 reranker completed on
  Vast, outputs were copied locally, and source tables were refreshed. It is a
  useful negative result: listwise training did not beat the pointwise tree10
  reranker and should not replace the current best diagnostic reranker.
- Exp 69 done: check. Added a cross-split calibration-transfer source table for
  candidate rerankers. It confirms the pointwise tree10 reranker is useful for
  ordering but still not stable enough for production rank/no-call thresholds.
- Exp 70 done: check. Pairwise tree-neighborhood reranker completed on Vast,
  outputs were copied locally, and source tables were refreshed. It improves a
  few precision-transfer numbers but does not clearly replace pointwise tree10.
- Exp 71 done: check. Independent selected-candidate assignment calibrators
  trained for pointwise tree10 and pairwise tree10 reranker outputs. They did
  not solve hard missing-reference calibration transfer.
- Exp 72 done: check. No-counts reference-gap detector follow-up ran locally at
  target 0.95 and 0.99, then source tables and ledgers were refreshed.
- Exp 73 done: check. Candidate-evidence reference-gap detector v2 ran locally
  at target 0.95 and 0.99, using candidate-list, p-distance, taxonomic
  diversity, reference-density, and tree-neighborhood features without global
  candidate-count or split-specific leakage.
- Exp 74 done: check. Candidate-evidence reference-gap detector v2 no-tree
  ablation ran locally at target 0.95, then source tables and ledgers were
  refreshed. Candidate-list evidence is the main gain; tree-neighborhood
  features help hide-family genus/family recall but are not uniformly better.
- Exp 75 done: check. Built a production-v1 gap-warning overlay source table.
  It joins existing v2 reference-gap probabilities onto production-v1 COI
  assignments to test warning/explanation behavior without training a new
  model.
- Exp 76 done: check. Trained a missing-reference-aware rank/no-call
  calibrator using normal supported rows plus strict hidden-reference rows and
  v2 reference-gap probabilities as soft evidence.
- Exp 77 done: check. Completed and copied eDNA Eco-Phylo posterior nested
  stability repeats for rep1 and rep2. Species-disabled target-95 results are
  rep0 38.9% assignment at 93.4% accuracy, rep1 38.5% at 95.4%, and rep2
  54.9% at 84.5%. This is a useful stability warning: eDNA higher-rank
  posterior evidence is promising, but the current mixed target-95 policy is
  not ready as a headline operating point.
- Exp 78 done: check. Added the literature-to-method deep dive and first
  production reason-code overlay. New docs/tables:
  `PIPELINE_NOVELTY_DEEP_DIVE.md`,
  `scripts/edna/build_paper1_reason_code_overlay.py`,
  `production_reason_code_summary.csv`,
  `production_reason_code_examples.csv`, and
  `production_reason_code_assignments.csv`. This moves the pipeline toward an
  evidence-accounting tool: broader-rank support, no-call, likely missing
  reference, and top-k ambiguity are now explicit source-table labels.
- Exp 79 done: check. Added `BREAKTHROUGH_AGENDA.md` and ran a frozen
  Nucleotide Transformer v2-50M GPU probe on Vast as an outside-model candidate
  generator. It has useful higher-rank signal but is weaker than current
  COI/classical baselines: held-out fish top-10 genus/family/order is
  38.4 / 59.8 / 74.2%, and unseen-genera top-10 family/order is 43.5 / 69.7%.
  Treat pretrained foundation models as optional evidence streams or fine-tune
  candidates, not as the core breakthrough by themselves.
- Exp 80 done: check. Added the first hierarchical selective-rank prototype
  using production-v1, reason-code, and reference-gap tables. The simple
  consensus-score version is useful as a formal frame but not a final policy:
  target-0.99 gives held-out fish 92.0% coverage at 98.0% precision and
  unseen-genera 89.9% coverage at 94.1%, mostly by collapsing to order-level
  calls. Next step is richer evidence fusion, not more consensus thresholding.
- Exp 81 done: check. MarkerMirror shared 12S/16S seed repeats completed,
  copied, and source-tabled. Across three runs, held-out 12S->16S top-10
  mean species/genus/family/order is 43.4 / 50.7 / 68.1 / 77.6%, and
  16S->12S is 66.4 / 73.9 / 81.9 / 86.4%. This is the lead
  candidate-generation result.
- Exp 82 done: check. Tri-marker 12S/16S/COI shared-space run completed,
  copied, and source-tabled. It beats frozen NT in every direction, including
  held-out 12S->COI 2.7 / 12.7 / 34.8 / 58.6% versus frozen
  0.0 / 1.8 / 12.7 / 38.1%, but it does not beat the best direct 12S->COI
  bridge. Keep 12S/16S as the lead result and COI as downstream anchor.
- Exp 83 done: check. Added and ran
  `scripts/edna/export_marker_mirror_candidate_rankings.py` on Vast for shared
  12S/16S seed1903. It produced 294,700 full-reference candidate rows copied
  locally under
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/`.
  Full-reference learned held-out top-50 is 52.0 / 60.6 / 80.1 / 87.8% for
  12S->16S and 74.3 / 80.6 / 90.3 / 93.8% for 16S->12S
  species/genus/family/order.
- Exp 84 done: check. Added and ran
  `scripts/edna/build_marker_mirror_candidate_rank_policy.py`. Simple
  validation-fitted top-1 score thresholds do not transfer cleanly to held-out
  test, so MarkerMirror should be treated as candidate-generation evidence
  until it is joined with sequence similarity, tree/rank evidence, and
  reference-gap features.
- Exp 85 done: check. Added and ran
  `scripts/edna/train_marker_mirror_rank_calibrator.py`, a feature-based
  diagnostic over top-k support, score margins, and taxonomic ambiguity. It is
  still not production-ready: learned 12S->16S target-0.95 test precision is
  100.0% but at only 1.3% coverage, while learned 16S->12S target-0.95 covers
  20.8% at 76.7% precision. This confirms the next layer must integrate
  MarkerMirror with sequence/tree/reference-gap evidence.
- Exp 86 done: check. Added and ran
  `scripts/edna/build_marker_mirror_evidence_join.py`, producing a 294,700-row
  evidence table with MarkerMirror candidates, same-marker sequence checks,
  candidate-list ambiguity/support, reference availability, and tree distance
  to the top candidate.
- Exp 87 done: check. Added and ran
  `scripts/edna/train_marker_mirror_integrated_rank_calibrator.py` with
  logistic and HGB candidate-level calibrators. This is the first strong
  MarkerMirror pipeline result. At target 0.99 on held-out test, learned
  12S->16S logistic gives 55.0% coverage, 99.4% assigned precision, 0.0%
  false species-call rate, and 100.0% species precision; learned 16S->12S HGB
  gives 75.0% coverage, 99.1% assigned precision, 0.0% false species-call
  rate, and 100.0% species precision.
- Current next method step: validate the integrated MarkerMirror pipeline. The
  first evidence compiler works on seed1903. Next repeat candidate export and
  integrated calibration on the other shared-space seeds, then add explicit
  12S/16S resolvability and field/eDNA evidence where available. Do not claim
  production readiness until seed stability and external validation are done.

- Exp 1 done: check. Current Vast Paper 1 host used for the latest production
  CLI smoke is:
  `ssh -p 23156 root@194.14.47.19`.
- Exp 2 done: check. Vast runtime has BLAST, VSEARCH, MAFFT, EPA-ng, pplacer,
  and a Python `.venv`; SEPP is installed but currently secondary because of a
  DendroPy compatibility issue.
- Exp 3 done: check. Paper 1 BLAST/VSEARCH/k-mer baselines completed on Vast,
  diagnostics binning was patched, diagnostics reran cleanly, and outputs were
  copied locally under
  `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/`.
- Exp 4 done: check. Added the phylogenetic-placement input/runner scripts for
  EPA-ng and pplacer:
  `scripts/edna/prepare_fish_tree_placement_inputs.py` and
  `experiments/paper1_phylo_calibrated_assignment/runs/04_vast_phylo_placement_baselines.sh`.
- Exp 5 done: check. EPA-ng placement baseline queue completed on the fresh
  Vast host as PID 4299. All three split outputs were copied locally and scored.
- Exp 6 done: check. Local Paper 1 source tables were built from copied outputs
  under `results/paper1_phylo_calibrated_assignment/source_tables/`.
- Exp 7 done: check. CNN/biLSTM/Transformer query-embedding export queue
  completed on the fresh Vast host and all 9 model/split `query_embeddings.npz`
  files were copied locally under
  `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/query_embeddings/`.
- Exp 8 done: check. CNN seed-repeat queue completed on the fresh Vast host
  using
  `experiments/paper1_phylo_calibrated_assignment/runs/06_vast_cnn_seed_repeats.sh`.
  Seeds 1207 and 1208 were trained, predictions/query embeddings were exported
  for Eval C, seen-test, and unseen-genera, and Eval C/unseen-genera
  tree-recovery follow-ups completed. Outputs were copied locally under
  `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/cnn_seed_repeats/`.
  CNN tree recovery is stable across seeds:
  - seed1207 Eval C zero-shot/reference: Pearson 0.944, Spearman 0.891;
    unseen-genera: Pearson 0.916, Spearman 0.866.
  - seed1208 Eval C zero-shot/reference: Pearson 0.941, Spearman 0.887;
    unseen-genera: Pearson 0.908, Spearman 0.865.
- Exp 9 done: check. Added full-candidate embedding ablation diagnostics from
  copied CNN/biLSTM/Transformer query embeddings using
  `scripts/edna/build_embedding_candidate_ablation.py`. Output:
  `results/paper1_phylo_calibrated_assignment/source_tables/full_candidate_embedding_ablation.csv`.
- Exp 10 done: check. Extended full-candidate embedding ablation to CNN repeat
  seeds 1207 and 1208. Output:
  `results/paper1_phylo_calibrated_assignment/source_tables/full_candidate_embedding_ablation_cnn_seed_repeats.csv`.
  Under Eval C with the true species hidden, CNN genus/family/order top10 was:
  seed1207 70.14 / 90.58 / 96.27; seed1208 72.25 / 91.52 / 96.48.
- Exp 11 blocked. Mamba query-embedding export support was probed on the fresh
  Vast host. `BarcodeMamba` cloned and imports, but `PhyloMamba` instantiation
  needs `mamba_ssm`. Installing `mamba-ssm` against the current PyTorch 2.12 /
  CUDA 13 environment triggered long native CUDA builds across many
  architectures, so the build was stopped to avoid wasting GPU/Vast time. Use a
  known BarcodeMamba-compatible CUDA/PyTorch image for this export later.
- Exp 12 done: check. Paper 1 README and Fernando positioning now explicitly
  frame the work as a learned BLAST-like barcode retrieval and rank-adaptive
  inference pipeline, with Fernando 2025 treated as a required direct
  comparator rather than optional related work.
- Exp 13 done: check. Paper 1 README now separates the pipeline into
  accuracy-first mode and vector-first mode. Accuracy-first mode keeps
  BLAST/VSEARCH/k-mer/EPA/APPLES as direct validation comparators. Vector-first
  mode is the practical fast learned-BLAST direction: encode references once,
  retrieve top-k candidates through an embedding index, then rerank only those
  candidates with alignment/tree-aware scores and rank-adaptive confidence.
- Exp 14 done: check. Literature-gap audit updated after online check:
  vector barcode retrieval is not novel by itself because related work exists
  around learned DNA sequence search, LSH barcode retrieval, BarcodeBERT,
  DNABERT-S, TaxoTagger, and BarcodeMamba. The defensible novelty is the full
  uncertainty-aware system: fast vector retrieval plus tree-aware reranking,
  Fernando-style placement comparators, rank-adaptive no-call, and
  missing-reference diagnostics.
- Exp 15 done: check. Added and ran the local vector-first retrieval benchmark
  scaffold:
  `scripts/edna/build_vector_first_retrieval_benchmark.py`. It generated
  `vector_first_retrieval_metrics.csv`, `vector_first_runtime_comparison.csv`,
  and `vector_first_retrieval_manifest.json` under
  `results/paper1_phylo_calibrated_assignment/source_tables/`. This is exact
  cosine search over saved embeddings, not an ANN index yet.
  Early timing signal on local exact cosine search is strong but must be rerun
  under controlled hardware before publication claims:
  - Eval C CNN seed1208 vector search: 1.01 s total, 0.087 ms/query.
  - Eval C BLAST runtime from Vast baseline: 1553 s total, 134 ms/query.
- Exp 47 done: check. The executable COI pipeline now supports exact vector
  retrieval, HNSW approximate retrieval, and train-reference p-distance
  reranking. The runner regenerates all nine current CNN seed1206 target-0.99
  rows for Eval C, seen-test, and unseen-genera. Exact-vector rows remain the
  primary calibrated operating point; HNSW rows are speed/approximation checks;
  p-distance raw rows are experimental, and Exp 48 adds rerank-specific
  calibration.
- Exp 48 done: check. Added and ran
  `scripts/edna/calibrate_paper1_pipeline_modes.py`, which learns executable
  pipeline thresholds on seen-test rows and evaluates the locked thresholds on
  Eval C and unseen-genera. Output:
  `results/paper1_phylo_calibrated_assignment/pipeline_calibration/`.
  At target 0.99, calibrated p-distance reranking gives Eval C 95.8% coverage,
  93.0% assigned precision, and 0.0% false species-call rate; unseen-genera
  92.3% coverage, 83.9% assigned precision, and 0.0% false species-call rate.
  This mode makes no species calls at target 0.99, which is an honest
  rank-backoff behavior rather than a species-level win.
- Exp 49 done: check. Added and ran
  `scripts/edna/build_paper1_manuscript_assets.py`, which packages current
  source tables into writing-facing inventories under
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/`.
  Added `MANUSCRIPT_ASSETS.md` to keep these as planning/source inventories,
  not new results.
  - Eval C k-mer runtime from Vast baseline: 58.7 s total, 5.07 ms/query.
  Early retrieval signal for CNN seed1208:
  - Eval C top10 species/genus/family/order: 9.44 / 72.63 / 91.52 / 96.48.
  - Eval C top50 species/genus/family/order: 55.68 / 88.63 / 95.87 / 97.37.
  - unseen-genera top10 species/genus/family/order:
    6.45 / 31.15 / 79.62 / 90.21.
- Exp 50 done: check. Strict missing-reference CNN validation completed on
  Vast for all six Eval C/unseen-genera species/genus/family-hidden packs.
  Outputs were copied locally under
  `results/remote_runs/2026-06-01/rtx_pro_6000/paper1_strict_missing_reference_cnn/`.
  Source tables were refreshed with
  `scripts/edna/build_strict_missing_reference_summary.py`. The key pattern is
  clean: hidden ranks collapse to zero, while broader ranks remain partly
  recoverable.
- Exp 51 done: check. Added
  `scripts/edna/build_strict_rank_backoff_summary.py` and generated
  `strict_rank_backoff_summary.csv`. The descriptive backoff table reports:
  Eval C hide species -> genus; Eval C hide genus -> family; Eval C hide
  family -> order; unseen-genera hide species/genus -> family; unseen-genera
  hide family -> order.
- Exp 52 done: check. Added regression coverage for the hidden-species scoring
  bug:
  `tests/test_eval_zero_shot_candidate_predictions.py`. The evaluator now
  verifies that a species absent from the candidate table can still score
  genus/family/order from query metadata. Verified with
  `python3 -m unittest tests.test_eval_zero_shot_candidate_predictions`.
- Exp 53 done: check. Installed `hnswlib` on the Vast host and reran controlled
  vector-speed timing with exact cosine plus HNSW. Outputs were copied locally
  under
  `results/remote_runs/2026-06-01/rtx_pro_6000/paper1_controlled_vector_speed/`
  and source ledgers were refreshed. Controlled CNN seed1206 Eval C timing:
  exact vector 0.397 ms/query; HNSW m16/ef50 0.00475 ms/query; HNSW m32/ef50
  0.00513 ms/query.
- Exp 54 running. Official APPLES 2.0.11 was installed on the Vast host and
  added to the Fernando completeness-sweep track via
  `experiments/paper1_phylo_calibrated_assignment/runs/11_vast_fernando_apples_sweeps.sh`.
  APPLES default branch-length re-estimation failed on the rooted two-child
  reduced tree, so the runner now writes `reference_tree_unrooted_for_apples.nwk`
  with `prepare_fish_tree_placement_inputs.py deroot-tree` before calling
  `run_apples.py` without `-D`. The earlier disabled-re-estimation trial outputs
  were moved aside as `apples_disabled_reestimation_20260601` and are not final
  APPLES results. The full default-APPLES queue is running with 8 threads and
  trails the EPA-ng queue by waiting for prepared reference/query alignments.
  The placement scorer now discovers `apples/apples.jplace` outputs. APPLES
  warnings about zero pendant edges/internal-node placement are retained in the
  logs and should be treated as method diagnostics, not run failures.
- Exp 55 partial: check. Copied the first 12 matched Fernando-style sweeps
  locally under
  `results/remote_runs/2026-06-01/rtx_pro_6000/paper1_fernando_completeness_sweeps/`
  and scored 24 placement runs (EPA-ng + official APPLES). Partial outputs are
  isolated under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_partial_12/`
  so they do not overwrite the clean split source tables. The main pipeline
  benchmark and end-to-end ledgers were refreshed after the copy. On these
  partial random completeness sweeps, official APPLES is materially stronger than
  EPA-ng in Fernando-like diagnostics: placed-clade genus/family/order rates
  average 34.7 / 58.4 / 67.6% for APPLES versus 18.4 / 45.6 / 57.7% for
  EPA-ng; simulated-placement-tree represented sister-overlap averages 20.9%
  sequence-level and 27.8% species-representative for APPLES versus 6.1% and
  14.0% for EPA-ng. These are partial comparator diagnostics, not final
  Fernando PCP until all 30 sweeps finish.
- Exp 56 partial: check. Copied and scored 19/30 matched Fernando-style sweeps
  (all random sweeps plus the first family-stratified 99% block and one 80%
  replicate). Outputs are isolated under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_partial_19/`
  and pipeline/end-to-end ledgers were refreshed. The partial signal still
  favours official APPLES over EPA-ng: placed-clade genus/family/order rates
  average 34.6 / 58.0 / 66.7% for APPLES versus 19.0 / 48.0 / 59.8% for
  EPA-ng; simulated-placement-tree represented sister-overlap averages 19.6%
  sequence-level and 26.8% species-representative for APPLES versus 5.8% and
  13.6% for EPA-ng. These are still partial diagnostics, not final Fernando PCP.
- Exp 57 partial: check. Copied and scored 24/30 matched Fernando-style sweeps
  after the additional family-stratified 60% and 80% replicates completed on
  the Vast host. Outputs are isolated under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_partial_24/`
  and the pipeline/end-to-end ledgers were refreshed. The partial signal remains
  stable: placed-clade genus/family/order rates average 34.5 / 57.8 / 66.6% for
  APPLES versus 19.1 / 48.1 / 59.6% for EPA-ng; simulated-placement-tree
  represented sister-overlap averages 19.6% sequence-level and 26.6%
  species-representative for APPLES versus 6.0% and 13.8% for EPA-ng. These are
  still partial comparator diagnostics, not final Fernando PCP.
- Exp 58 done: check. Completed, copied, and scored all 30 matched
  Fernando-style completeness sweeps for both EPA-ng and official APPLES 2.0.11.
  Final copied outputs are under
  `results/remote_runs/2026-06-01/rtx_pro_6000/paper1_fernando_completeness_sweeps/`;
  final source-table snapshots are under
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_final_30/`.
  The final matrix contains 60 placement runs and 92,592 placed-query rows for
  rank/sister diagnostics, plus 185,184 rows for the two simulated
  placement-tree PCP-style modes. APPLES remains clearly stronger than EPA-ng
  in these comparator diagnostics: placed-clade genus/family/order averages are
  32.8 / 57.2 / 65.6% for APPLES versus 17.3 / 45.2 / 57.0% for EPA-ng;
  sister-clade any-overlap/exact-match averages are 42.5 / 21.4% for APPLES
  versus 14.8 / 3.2% for EPA-ng. Simulated-placement-tree represented
  sister-overlap averages are 18.0% sequence-level and 26.8%
  species-representative for APPLES versus 5.4% and 15.2% for EPA-ng. These are
  full matched sweep diagnostics for our public setup, but still not an exact
  reproduction of Fernando PCP.
- Exp 59 done: check. Consolidated the merged 12S/eDNA evidence-decomposition
  layer with `scripts/edna/build_edna_evidence_decomposition.py` and refreshed
  the pipeline ledgers. New source tables:
  `edna_evidence_decomposition_matrix.csv`,
  `edna_evidence_best_by_rank.csv`, and
  `edna_rank_no_call_operating_points.csv`. The Global_eDNA evidence arms now
  explicitly separate BLAST sequence-only, neural sequence/tree-only,
  geography-only, same-sample co-occurrence-only, sequence + geography,
  sequence + co-occurrence, learned RLS/OBIS co-occurrence, and learned public
  FISHGLOB co-occurrence. Current diagnostic result: BLAST remains strongest on
  forced Global_eDNA top-k assignment, while SSM/context arms are useful for
  evidence decomposition but not yet a claim-ready calibrated eDNA posterior.
  The eDNA rank/no-call table is diagnostic only; independent calibration or a
  pre-registered threshold policy is still required.
- Exp 60 done: check. Added and ran
  `scripts/edna/build_global_edna_independent_rank_calibration.py`, which learns
  Global_eDNA rank/no-call score thresholds on deterministic `site20`
  calibration groups and evaluates them on held-out `site20` groups. Outputs are
  under
  `results/paper1_phylo_calibrated_assignment/global_edna_independent_rank_calibration/`.
  The result is informative but not yet a positive eDNA rank/no-call claim:
  SSM + RLS/OBIS learned co-occurrence at weight 0.25 assigns 8.9% of held-out
  rows at 59.6% family accuracy and 74.1% order accuracy under the 50%-target
  threshold. No current method reaches a 70%+ calibration target for
  species/genus/family/order with this top-1 score-threshold policy.
- Exp 61 done: check. Added the specimen-facing COI production CLI:
  `scripts/edna/run_paper1_fasta_inference_v1.py`, plus
  `PRODUCTION_CLI_V1.md`. Patched `run_paper1_production_v1.py` so unlabeled
  FASTA/CSV specimens no longer get counted as incorrect; precision is reported
  only when known labels are supplied. Verified on Vast endpoint
  `ssh -p 23156 root@194.14.47.19` with two smokes:
  - CSV known-label smoke: 16 queries, 100.0% coverage, 87.5% precision if
    known, 0 species calls, output copied under
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_eval_c_known/`.
  - FASTA unlabeled smoke: 8 queries, 100.0% coverage, precision unavailable
    by design, 0 species calls, output copied under
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_unlabeled_fasta/`.
  This means the core COI inference path now runs from arbitrary FASTA/CSV
  specimen-style input. The remaining production gap is API/web packaging and
  external demo polish, not the command-line inference path.
- Exp 62 done: check. Began the DL model track. Added
  `scripts/edna/train_paper1_coi_evidence_model.py`, plus
  `DL_MODEL_ROADMAP.md` and five component docs under `dl_models/`. The first
  model is a PyTorch MLP over current CNN seed1206 vector+p-distance evidence.
  It trains on seen-test rows, calibrates thresholds on held-out seen-test rows,
  and evaluates on held-out fish plus unseen-genera. Species-disabled
  target-0.99 results:
  - held-out fish: 94.2% coverage, 97.4% assigned precision, 0.0% false species
    calls. Bootstrap 95% intervals: coverage 93.8-94.7%, precision 97.1-97.7%.
  - unseen-genera: 88.5% coverage, 93.5% assigned precision, 0.0% false species
    calls. Bootstrap 95% intervals: coverage 87.8-89.2%, precision 93.0-94.0%.
  Compared with hand-threshold production-v1, this improves precision
  materially but lowers coverage. Species remains disabled for the conservative
  missing-reference claim.
- Exp 16 done: check. Fixed nearest-reference diagnostics by normalizing tree
  labels with spaces to the underscore labels used by clean input packs.
  Regenerated local diagnostics under
  `results/paper1_phylo_calibrated_assignment/reference_diagnostics_*` and
  rebuilt source tables. Eval C and unseen-genera now have real
  nearest-reference tree-distance bins instead of all-missing bins.
- Exp 17 done: check. Added initial placement-output scoring scaffold:
  `scripts/edna/score_fish_tree_placement_outputs.py`. It parses jplace files
  when available and emits placed-clade species/genus/family/order containment
  diagnostics plus LWR-threshold summaries. This is not yet Fernando PCP, but it
  is the first shared placement-to-rank adapter.
- Exp 18 done: check. Merged the former Paper 2 into Paper 1 at the manuscript
  level. COI is now the tree-space/retrieval/missing-reference benchmark; 12S
  and Global_eDNA are the marker-ambiguity/ecological-context stress test.
  Added `MERGED_MANUSCRIPT_OUTLINE.md` to keep the merged paper focused around
  one thesis instead of becoming two papers glued together.
- Exp 19 done: check. Added the merged 12S/eDNA source-table builder
  `scripts/edna/build_merged_paper1_edna_source_tables.py` and generated:
  - `merged_12s_resolvability_summary.csv` with exact and near-exact marker
    oracle support across multisource, Teleo, rCRUX, and Mitohelper datasets;
  - `merged_12s_zero_shot_model_metrics.csv` with SSM/CNN 12S zero-shot
    species/genus/family/order metrics;
  - `merged_global_edna_asv_metrics.csv` and
    `merged_global_edna_sample_metrics.csv` with Global_eDNA sequence-only and
    learned co-occurrence validation summaries.
  - `merged_edna_evidence_arm_status.csv` to record which Stalder-adjacent
    evidence-decomposition arms exist and which are still missing.
  These tables make the merged-paper evidence ledger explicit: COI is the
  tree-space/missing-reference benchmark; 12S/eDNA is the marker-resolution and
  ecology-aware inference stress test.
- Exp 20 done: check. Ran diagnostic Global_eDNA calibration/no-call curves for
  18 current SSM/CNN sequence-only and learned co-occurrence prediction sets
  using
  `configs/runs/2026-05-31-paper1-merged-global-edna-calibration-methods.json`.
  Outputs are under
  `results/paper1_phylo_calibrated_assignment/global_edna_rank_calibration/`,
  and the merged source-table builder now exports
  `merged_global_edna_calibration_curves.csv` with 720 rank/coverage rows.
  Caveat: this is top-1 score-threshold calibration over current Global_eDNA
  prediction rows, not the final independent eDNA rank/no-call protocol.
- Exp 21 superseded: check. Eval C EPA-ng placement completed on the Vast queue
  and was copied locally under
  `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/phylo_placement/eval_c/`.
  The placement scorer was fixed to map EPA-ng query IDs back to
  `query_manifest.csv` and to load taxonomy from the original clean input pack.
  Current Eval C placed-clade containment diagnostic:
  - species: 0.0 by design because held-out species are absent from the
    reference placement tree;
  - genus/family/order: 45.9 / 67.8 / 74.3.
  Pplacer is not yet valid: the current command fails with
  `please specify a tree model with -s or -c`, so pplacer needs a refpkg/stats
  command fix before it can be used as a comparator.
- Exp 22 done: check. Patched
  `experiments/paper1_phylo_calibrated_assignment/runs/04_vast_phylo_placement_baselines.sh`
  so future pplacer runs are skipped unless `PPLACER_REFPKG` or
  `PPLACER_STATS` is explicitly provided. This prevents invalid pplacer output
  from being mistaken for a failed biological result. EPA-ng remains the active
  placement comparator for the current queue. Exp 26 supersedes this with all
  three EPA-ng splits completed and scored.
- Exp 23 done: check. Added the pipeline-level benchmark layer:
  `scripts/edna/build_paper1_pipeline_benchmarks.py` and
  `PIPELINE.md`. The builder generated:
  - `pipeline_component_status.csv`
  - `pipeline_coi_method_benchmark.csv`
  - `pipeline_placement_benchmark.csv`
  - `pipeline_edna_method_benchmark.csv`
  - `pipeline_best_by_task.csv`
  - `pipeline_next_actions.csv`
  This is the current paper-level benchmark ledger. It joins the existing
  source tables into one pipeline view: retrieval, tree geometry,
  missing-reference/rank-backoff, placement, vector speed, 12S/eDNA evidence,
  calibration status, and remaining work. `pipeline_best_by_task.csv` reports
  best observed values, not claim-ready method winners.
- Exp 24 done: check. Added a shared timestamped progress logger:
  `scripts/edna/progress_logging.py`. Active Paper 1 Python scripts now log
  start/stage/output/done messages to stdout and default log files under
  `results/paper1_phylo_calibrated_assignment/logs/`, with `--log-file`
  support where useful. Patched scripts include source-table builders,
  placement scoring, vector-first retrieval, rank/eDNA calibration,
  candidate-ablation diagnostics, reference diagnostics, baseline evaluation,
  negative controls, zero-shot prediction evaluation, 12S exact/near-exact
  resolvability, and encoder tree recovery.
  Verified with `py_compile` and a fresh run of
  `scripts/edna/build_paper1_pipeline_benchmarks.py`.
- Exp 25 done: check. Added and ran the ANN vector-store speed benchmark:
  `scripts/edna/build_ann_vector_retrieval_benchmark.py`. It benchmarks exact
  cosine retrieval against HNSW cosine indexes over 15 saved COI embedding runs
  and writes:
  - `ann_vector_retrieval_metrics.csv`
  - `ann_vector_runtime_comparison.csv`
  - `ann_vector_recall_against_exact.csv`
  - `pipeline_vector_index_benchmark.csv`
  Fastest local HNSW rows are around 0.03-0.04 ms/query on the current copied
  embedding set, with top-10 recall against exact vector search averaging
  about 0.98-0.995 depending on HNSW parameters. This is strong evidence for
  the vector-first retrieval layer, but final speed claims still need
  controlled hardware and larger reference stress tests.
- Exp 26 done: check. EPA-ng placement completed for Eval C, seen-test, and
  unseen-genera. Copied remote placement outputs from Vast into
  `results/remote_runs/2026-05-31/rtx_pro_6000/paper1_phylo_calibrated_assignment/phylo_placement/`,
  reran `scripts/edna/score_fish_tree_placement_outputs.py`, and regenerated
  the pipeline ledger. Current placed-clade containment diagnostics:
  - Eval C species/genus/family/order: 0.0 / 45.9 / 67.8 / 74.3.
  - seen-test species/genus/family/order: 26.2 / 42.4 / 63.6 / 72.1.
  - unseen-genera species/genus/family/order: 0.0 / 0.0 / 37.0 / 51.0.
  Pplacer remains invalid on all splits with
  `please specify a tree model with -s or -c`; keep pplacer blocked unless a
  valid refpkg/stats file is provided or replace it with APPLES-style distance
  placement. These EPA-ng scores are still placed-clade containment, not full
  Fernando PCP.
- Exp 27 done: check. Upgraded
  `scripts/edna/score_fish_tree_placement_outputs.py` from a containment-only
  adapter to a Fernando-adjacent placement diagnostic. It now adds:
  - per-query nearest reference tree distance;
  - minimum tree distance from the true species to the placed clade;
  - excess distance over the nearest available reference;
  - corrected most-specific rank-backoff labels;
  - LWR-binned and clade-size-binned rank summaries.
  New source tables:
  - `placement_lwr_rank_summary.csv`
  - `placement_clade_size_rank_summary.csv`
  - `placement_rank_backoff_summary.csv`
  The pipeline ledger now records placement-distance medians and marks full
  Fernando PCP as partial rather than absent.
- Exp 28 done: check. Upgraded rank-adaptive calibration from same-split-only
  diagnostics to include seen-test-to-heldout threshold transfer:
  `scripts/edna/build_rank_adaptive_calibration.py` now accepts multiple copied
  result roots, deduplicates prediction sets, learns thresholds on
  `seen_test`, and evaluates them on Eval C and unseen-genera. New outputs:
  - `prospective_rank_thresholds.csv`
  - `prospective_rank_adaptive_policy_summary.csv`
  The combined calibration now covers 51 prediction sets and 620,585 query
  rows. The policy rows now include Wilson 95% intervals. First pass signal:
  at target 0.9 using confidence margin, CNN seed1206 gives prospective
  assigned precision/coverage of 77.0% / 100.0% on Eval C
  (95% CI 76.2-77.8%) and 74.2% / 100.0% on unseen-genera
  (95% CI 73.3-75.1%). This is scientifically useful because it shows naive
  seen-test thresholds do not preserve nominal precision under missing-species
  and unseen-genera shifts; final no-call claims need a locked policy or a
  pre-registered bootstrap protocol.
- Exp 29 done: check. Added a missing-reference-aware consensus rank/no-call
  policy to `scripts/edna/build_rank_adaptive_calibration.py`. It learns
  species, genus, family, and order thresholds on seen-test using relative
  margin plus top-10 taxonomic consensus, then evaluates those locked
  thresholds on Eval C and unseen-genera. New outputs:
  - `missing_reference_aware_thresholds.csv`
  - `missing_reference_aware_policy_summary.csv`
  Candidate locked operating point: CNN seed1206 at target 0.99 gives Eval C
  assigned precision/coverage of 90.0% / 96.1% and unseen-genera
  assigned precision/coverage of 83.7% / 93.7%. Assigned-rank counts are
  species/genus/family/order/no-call = 187/969/7922/2061/455 for Eval C and
  6/305/5606/2652/579 for unseen-genera. This is stronger than naive
  confidence-margin transfer, but still needs bootstrap stability and an
  explicit false species-call tolerance before it is treated as final.
- Exp 30 done: check. Locked the placement-comparator decision in
  `PLACEMENT_COMPARATOR_DECISION.md`: EPA-ng stays as the completed
  likelihood-placement comparator; pplacer remains blocked unless a valid
  refpkg/stats model is provided; APPLES-style distance placement is the next
  comparator because it is the direct Fernando-aligned missing piece. This
  avoids spending time building a pplacer refpkg before closing the more
  relevant EPA-ng/APPLES comparison.
- Exp 31 done: check. Added and ran
  `scripts/edna/build_ann_vector_stress_benchmark.py` for controlled
  vector-index speed stress testing. Output:
  `results/paper1_phylo_calibrated_assignment/source_tables/ann_vector_stress_runtime.csv`.
  The run used CNN seed1206 Eval C embeddings and synthetic reference-catalog
  expansion at 1x, 5x, 10x, and 25x. This produced HNSW speed/memory rows up
  to 290,950 candidates. Fastest observed rows stayed under 0.09 ms/query, with
  25x rows around 0.055-0.069 ms/query and 640-677 MB index size. This is a
  speed/memory stress test only; synthetic-expanded rows are not biological
  retrieval accuracy.
- Exp 32 done: check. Closed the first eDNA evidence-decomposition gap.
  Geography/range-only arms already existed locally for RLS and OBIS priors,
  and `scripts/edna/build_merged_paper1_edna_source_tables.py` now imports
  them into the merged Paper 1 source tables. Added and ran
  `scripts/edna/eval_global_edna_sample_cooccurrence_prior_only.py`, which
  builds a same-sample community prior for each query while excluding that
  query's own sequence-derived labels. Output:
  `results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_sample_cooccurrence_prior_only_top10/`.
  Pure-prior ASV top10 metrics are now in
  `merged_global_edna_asv_metrics.csv`: RLS prior-only species/genus/family/order
  2.03/11.89/37.60/40.53; OBIS prior-only 2.18/8.58/27.73/28.65; same-sample
  co-occurrence-only 2.24/8.49/29.98/34.19. These are decomposition baselines,
  not final sequence assignment systems.
- Exp 33 done: check. Added and ran the local APPLES-like distance-placement
  diagnostic:
  `scripts/edna/eval_apples_like_distance_placement.py`. It uses prepared
  EPA-ng placement inputs, VSEARCH top-25 candidate neighborhoods, aligned
  query/reference COI p-distance, and the same Paper 1 rank/nearest-reference
  scoring layer. Output:
  `results/paper1_phylo_calibrated_assignment/source_tables/apples_like_distance_placement_summary.csv`.
  Nearest-reference match rates are Eval C 54.4%, seen-test 78.8%, and
  unseen-genera 22.1%. This is APPLES-like, not official APPLES.
- Exp 34 done: check. Added and ran
  `scripts/edna/build_placement_tree_error_tables.py`. Output:
  `results/paper1_phylo_calibrated_assignment/source_tables/placement_tree_error_summary.csv`.
  The matched nearest-reference diagnostic shows APPLES-like is stronger than
  current EPA-ng on this metric: Eval C 54.4% vs 22.7%, seen-test 78.8% vs
  26.3%, and unseen-genera 22.1% vs 7.8%. This is not exact Fernando PCP.
- Exp 35 done: check. Added and ran
  `scripts/edna/bootstrap_rank_no_call_policy.py` with 1000 bootstrap
  replicates at target precision 0.99. Output:
  `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/missing_reference_aware_policy_bootstrap.csv`.
  CNN seed1206 gives Eval C assigned precision 90.0% with bootstrap 95% CI
  89.5-90.5 and false species-call rate 1.61%; unseen-genera assigned
  precision 83.7% with bootstrap 95% CI 82.9-84.5 and false species-call rate
  0.066%.
- Exp 36 done: check. Added the testing/documentation layer:
  `TESTING.md`, `SOURCE_TABLES.md`, `CLAIM_BOUNDARIES.md`, and
  `EXPERIMENT_REGISTRY.csv`. Added the manuscript-facing end-to-end summary
  builder `scripts/edna/build_paper1_end_to_end_summary.py`, which writes
  `pipeline_end_to_end_summary.csv`. No GPU is needed for the current
  table/documentation pass; GPU is only needed for new neural training,
  compatible Mamba query-embedding export, pretrained encoder baselines, or
  strict tree-pruned retraining.
- Exp 37 done: check. Added and ran
  `scripts/edna/build_fernando_like_pcp_diagnostics.py`. It computes a
  Fernando-like sister-clade diagnostic from EPA-ng jplace outputs by comparing
  the placed descendant side against the nearest represented sister support in
  the full fish tree. Outputs:
  - `placement_pcp_like_per_query.csv`
  - `placement_pcp_like_summary.csv`
  - `placement_pcp_like_lwr_summary.csv`
  Current exact sister-clade rates are Eval C 7.1%, seen-test 4.7%, and
  unseen-genera 0.6%; any-overlap rates are 24.1%, 15.8%, and 15.3%. This is
  useful because it proves we have not matched Fernando PCP yet. It should be
  treated as a diagnostic bridge, not as an official PCP reproduction.
- Exp 38 done: check. Added and ran
  `scripts/edna/build_fernando_simulated_placement_pcp.py`. It grafts each
  query onto the EPA-ng top-LWR edge to build a simulated placement tree, then
  compares query sister support against the full fish tree. Outputs:
  - `placement_simulated_tree_pcp_per_query.csv`
  - `placement_simulated_tree_pcp_summary.csv`
  - `placement_simulated_tree_pcp_manifest.json`
  - simulated trees under
    `results/paper1_phylo_calibrated_assignment/fernando_simulated_placement_trees/`
  Species-representative simulated PCP-like exact/overlap rates are Eval C
  7.3/22.1%, seen-test 24.0/50.4%, and unseen-genera 0.2/45.8%. This is closer
  to Fernando's generated-tree sister comparison than placed-clade containment,
  but still not exact Fernando PCP because our current splits and query universe
  are not their matched backbone-completeness protocol.
- Exp 39 done: check. Added and ran
  `scripts/edna/build_fernando_completeness_sweeps.py`. It generated 30
  Fernando-style input packs under `data/phylo/fernando_completeness_sweeps/`:
  random and family-stratified sampling at 99/80/60/40/20% completeness with
  3 replicates each. Local smoke prep passed for `random_c99_rep01`, pruning
  the fish tree to 3801 backbone tips and writing placement inputs for 38
  held-out species. The generated input root is currently about 2 GB because
  each split carries sequence JSONs.
- Exp 40 done: check. Added the Vast/Linux runner
  `experiments/paper1_phylo_calibrated_assignment/runs/07_vast_fernando_completeness_sweeps.sh`
  and verified it with `bash -n`. The runner reads
  `data/phylo/fernando_completeness_sweeps/sweep_manifest.csv`, prepares each
  split, runs MAFFT reference/add-fragments, and runs EPA-ng with per-phase
  logs. This is ready for a Linux/Vast job when we choose to spend GPU/CPU time;
  it does not run official APPLES yet.
- Exp 41 done: check. Updated
  `scripts/edna/build_paper1_pipeline_benchmarks.py` so
  `pipeline_placement_benchmark.csv` now includes simulated-placement-tree PCP
  columns alongside placed-clade containment, tree-distance, APPLES-like, and
  edge-to-sister diagnostics. Regenerated
  `pipeline_placement_benchmark.csv` and `pipeline_end_to_end_summary.csv`.
  Claim boundary is now explicit: these are diagnostics until the matched
  Fernando completeness sweeps have been executed and scored.
- Exp 42 done: check. Added and ran
  `scripts/edna/build_controlled_vector_speed_benchmark.py`, a repeat-based
  local speed benchmark for exact vector and HNSW retrieval. Outputs:
  - `controlled_vector_speed_detail.csv`
  - `controlled_vector_speed_benchmark.csv`
  - `controlled_vector_speed_manifest.json`
  Current local CNN seed1206 Eval C median timings are exact vector 0.091
  ms/query, HNSW m16/ef50 0.043 ms/query, and HNSW m32/ef50 0.045 ms/query
  over 5 repeats. These are vector-retrieval-only timings and still need
  target-hardware replication before final speed claims.
- Exp 43 done: check. Added and ran
  `scripts/edna/build_strict_missing_reference_inputs.py`. It produced strict
  pruned input packs under `data/phylo/paper1_strict_missing_reference_inputs/`
  for Eval C and unseen-genera with species/genus/family hidden before
  candidate-tree construction and before training/reference sequence
  construction. Summary:
  - eval_c_hide_species: 531 hidden candidates, 11107 kept candidates,
    3839 kept train species.
  - eval_c_hide_genus: 5307 hidden candidates, 6331 kept candidates,
    1868 kept train species.
  - eval_c_hide_family: 9871 hidden candidates, 1767 kept candidates,
    543 kept train species.
  - unseen_genera_hide_species: 614 hidden candidates, 11024 kept candidates,
    3839 kept train species.
  - unseen_genera_hide_genus: 744 hidden candidates, 10894 kept candidates,
    3839 kept train species.
  - unseen_genera_hide_family: 6974 hidden candidates, 4664 kept candidates,
    1682 kept train species.
  A local dry-run of `train_fish_tree_encoder_benchmark.py` on
  `eval_c_hide_species` passed.
- Exp 44 done: check. Added the strict missing-reference Vast/Linux runner
  `experiments/paper1_phylo_calibrated_assignment/runs/08_vast_strict_missing_reference_cnn.sh`
  and verified it with `bash -n`. Added
  `scripts/edna/build_strict_missing_reference_summary.py`; it currently writes
  `strict_missing_reference_summary.csv` with 6 pending rows. This is ready for
  a GPU run when we decide to spend compute.
- Exp 45 done: check. Added the target-host controlled vector speed runner
  `experiments/paper1_phylo_calibrated_assignment/runs/09_vast_controlled_vector_speed.sh`
  and verified it with `bash -n`. This reruns the repeat-based vector timing on
  a Linux/Vast host and refreshes the pipeline ledgers, without needing GPU
  training.
- Exp 46 done: check. Added the executable COI pipeline:
  `scripts/edna/run_paper1_coi_pipeline.py`, summary builder
  `scripts/edna/build_paper1_pipeline_run_summary.py`, and runner
  `experiments/paper1_phylo_calibrated_assignment/runs/10_run_executable_coi_pipeline.sh`.
  Ran it locally for CNN seed1206 with target-0.99 thresholds on Eval C and
  unseen-genera. Outputs are under
  `results/paper1_phylo_calibrated_assignment/pipeline_runs/`, with source
  summary `pipeline_run_summary.csv`.
  - Eval C: 11594 queries, 0.097 ms/query vector search, 96.1% coverage,
    90.0% assigned precision, 1.61% false species-call rate.
  - unseen-genera: 9148 queries, 0.109 ms/query vector search, 93.7% coverage,
    83.7% assigned precision, 0.066% false species-call rate.
  This is now an actual executable vector-first/rank-adaptive pipeline, not only
  a benchmark ledger. It still lacks alignment reranking and strict
  tree-pruned retrained validation.
- Exp 47 done: check. Extended `scripts/edna/run_paper1_coi_pipeline.py` and
  `experiments/paper1_phylo_calibrated_assignment/runs/10_run_executable_coi_pipeline.sh`
  with optional HNSW retrieval and optional train-reference p-distance reranking
  over retrieved top-k candidates. Reran the executable pipeline and refreshed
  `pipeline_run_summary.csv`, `pipeline_component_status.csv`, and
  `pipeline_end_to_end_summary.csv`.
  Current rows:
  - Eval C exact calibrated: 0.108 ms/query, 96.1% coverage, 90.0% assigned
    precision, 1.61% false species-call rate.
  - Eval C HNSW: 0.038 ms/query, 96.4% coverage, 89.5% assigned precision,
    1.61% false species-call rate.
  - Eval C p-distance experimental: 0.184 ms/query candidate stage, 96.7%
    coverage, 92.0% assigned precision, 0.73% false species-call rate.
  - unseen-genera exact calibrated: 0.095 ms/query, 93.7% coverage, 83.7%
    assigned precision, 0.066% false species-call rate.
  - unseen-genera HNSW: 0.073 ms/query, 94.3% coverage, 82.8% assigned
    precision, 0.066% false species-call rate.
  - unseen-genera p-distance experimental: 0.186 ms/query candidate stage,
    93.8% coverage, 83.1% assigned precision, 0.219% false species-call rate.
  Interpretation: p-distance reranking is scientifically useful but not yet a
  final result. It improves the Eval C precision/false-species-call tradeoff
  but worsens unseen-genera false species calls, so reranked candidate order
  needs independent threshold calibration before manuscript claims.
- Exp 48 done: check. Added `scripts/edna/calibrate_paper1_pipeline_modes.py`
  and ran seen-test-derived calibration for exact-vector, HNSW, and p-distance
  executable modes. New outputs:
  - `pipeline_calibration/pipeline_mode_thresholds.csv`
  - `pipeline_calibration/pipeline_mode_policy_summary.csv`
  - `pipeline_calibration/pipeline_mode_calibration_manifest.json`
  Target-0.99 policy rows:
  - exact vector Eval C/unseen-genera remain 90.0% / 83.7% assigned precision
    with false species-call rates 1.61% / 0.066%.
  - HNSW Eval C/unseen-genera are 89.8% / 83.3% assigned precision with false
    species-call rates 1.61% / 0.066%.
  - p-distance reranked Eval C/unseen-genera are 93.0% / 83.9% assigned
    precision with 0.0% false species-call rate on both splits. The policy
    achieves this by making no species calls at target 0.99 and backing off to
    genus/family/order.
- Exp 49 done: check. Added and ran
  `scripts/edna/build_paper1_manuscript_assets.py`. New outputs:
  - `manuscript_assets/figure_plan.csv`
  - `manuscript_assets/table_plan.csv`
  - `manuscript_assets/claim_evidence_map.csv`
  - `manuscript_assets/pipeline_operating_points.csv`
  - `manuscript_assets/missing_results_checklist.csv`
  - `manuscript_assets/manuscript_asset_manifest.json`
  Added `MANUSCRIPT_ASSETS.md` and updated testing/source-table docs. These
  files are planning artifacts so we can move quickly into writing after strict
  validation and target-host speed results land.

## 2026-05-30

Decision: Paper 1 combines MarineMamba-Phylo and rank-adaptive calibrated
assignment.

Scientific claim boundary:

- The model is not discovering arbitrary unknown species.
- It is learning to map held-out barcode sequences into a fixed fish
  species-tree candidate space.
- The right output is calibrated rank assignment, not always species top-1.

Current state:

- Clean COI fish-tree split exists.
- Eval C and unseen-genera species are genuinely held out from training
  sequence references.
- Cosine512 seed repeats are complete and stable.
- Ledger regenerated after seed repeats.

Results so far:

- Eval C zero-shot/reference tree recovery:
  - seed1206: Pearson 0.914, Spearman 0.859
  - seed1207: Pearson 0.916, Spearman 0.862
  - seed1208: Pearson 0.921, Spearman 0.867
- Unseen-genera zero-shot/reference tree recovery:
  - seed1206: Pearson 0.859, Spearman 0.821
  - seed1207: Pearson 0.858, Spearman 0.820
  - seed1208: Pearson 0.860, Spearman 0.824
- Eval C top10 species/genus/family/order:
  - seed1206: 10.84 / 73.77 / 86.94 / 91.56
  - seed1207: 11.01 / 73.15 / 88.73 / 93.41
  - seed1208: 11.11 / 74.17 / 88.62 / 92.70

What this answers:

- A short COI barcode can learn tree-position structure for held-out fish
  species, not only memorize seen labels.
- Tree recovery can be evaluated directly with Pearson/Spearman correlations
  against the reference species tree.
- Open-candidate retrieval and tree-distance recovery are different from
  closed-reference classification.

How this is unique/useful:

- Most barcode models emphasize label accuracy. This track evaluates whether
  learned sequence embeddings preserve real species-tree geometry.
- It gives a biologically interpretable error measure: how far wrong on the
  tree, not only whether the species name is exact.
- It can become a model-agnostic benchmark. Mamba is not the only possible
  encoder; CNN, LSTM, Transformer, S5, and pretrained DNA encoders can all be
  tested against the same tree-space target.

Architecture stance:

- Paper 1 should not be framed as SSM-only.
- The method is sequence-to-tree-space learning plus calibrated assignment.
- Mamba/SSM is the current strongest trained encoder, but the contribution is
  stronger if other encoders are evaluated under the same objective.

Next actions:

1. Run or implement same-split BLAST/VSEARCH/k-mer candidate baselines.
2. Add negative controls:
   - shuffled species-to-tree labels,
   - random tree distances,
   - optionally random candidate tree embeddings.
3. Build nearest-reference-distance diagnostics.
4. Build COI rank-adaptive calibration/no-call.
5. Convert results into source tables and figures.

## 2026-05-30 Baseline Queue

Added and launched the Paper 1 baseline/control queue on the Vast RTX PRO 6000
instance.

New local scripts:

- `scripts/edna/eval_fish_tree_candidate_baselines.py`
- `scripts/edna/eval_fish_tree_prediction_negative_controls.py`
- `scripts/edna/build_fish_tree_reference_diagnostics.py`

New runner:

- `experiments/paper1_phylo_calibrated_assignment/runs/01_vast_baselines_controls.sh`

Remote output root:

- `results/paper1_phylo_calibrated_assignment/`

Queue scope:

- k-mer, BLAST, and VSEARCH baselines for Eval C, seen-test, and unseen-genera.
- shuffled-label and random-ranked negative controls for seed1206/1207/1208.
- nearest-reference and rank-coverage diagnostics for each split.

Important interpretation note:

- Sequence baselines can only rank species with reference sequences. In Eval C
  and unseen-genera, held-out query species do not have reference sequences, so
  species-level recovery is not a fair expectation for BLAST/VSEARCH/k-mer.
  Higher-rank recovery is still directly informative.

Early baseline signal:

- Eval C 6-mer/k-mer baseline completed first on Vast.
- It produced strong higher-rank retrieval:
  - genus top10: 94.1%
  - family top10: 97.5%
  - order top10: 98.5%
  - species top10: 0.0%
- Interpretation: k-mer/sequence similarity is very strong for higher-rank
  neighborhood recovery, but cannot recover held-out species labels that have
  no sequence reference. The neural tree-space model's species-level open-
  candidate recovery is therefore a different capability, while higher-rank
  claims require careful baseline comparison.

What remains for Paper 1:

1. Finish BLAST/VSEARCH/k-mer baselines for Eval C, seen-test, and unseen-
   genera.
2. Finish shuffled-label and random-ranked negative controls.
3. Finish nearest-reference-distance diagnostics.
4. Build rank-adaptive calibration/no-call:
   - species if confident,
   - genus if not,
   - family if not,
   - order if not,
   - no-call if not.
5. Add reference/missingness diagnostics:
   - nearest reference species distance,
   - whether true genus/family/order exists in reference,
   - candidate clade density,
   - taxonomic distance of errors.
6. Add architecture baselines after the current validation queue:
   - CNN,
   - biLSTM,
   - small Transformer,
   - S5 if setup is clean.
7. Convert outputs into figure source tables and plots.

## 2026-05-30 Baseline/Control Completion

Remote Paper 1 baseline/control queue completed on the Vast RTX PRO 6000
instance and was copied locally to:

- `results/remote_runs/2026-05-30/rtx_pro_6000/paper1_phylo_calibrated_assignment/`

Canonical ledger regenerated after copy:

- `results/summary/results_ledger.json`
- `results/summary/results_ledger.csv`

Baseline top1/top5/top10 species/genus/family/order:

- Eval C:
  - k-mer species: 0.00 / 0.00 / 0.00
  - k-mer genus/family/order top10: 94.07 / 97.52 / 98.46
  - BLAST species: 0.00 / 0.00 / 0.00
  - BLAST genus/family/order top10: 94.96 / 98.12 / 98.60
  - VSEARCH species: 0.00 / 0.00 / 0.00
  - VSEARCH genus/family/order top10: 95.57 / 98.06 / 98.72
- Seen-test:
  - k-mer species top10: 98.24
  - BLAST species top10: 98.28
  - VSEARCH species top10: 98.26
  - all three baselines reach at least 98.70 genus top10 and 99.35 family
    top10.
- Unseen-genera:
  - species and genus top10 are 0.00 for k-mer, BLAST, and VSEARCH because
    the true species and genus are not represented by references.
  - BLAST family/order top10: 82.96 / 92.36
  - VSEARCH family/order top10: 82.91 / 92.40
  - k-mer family/order top10: 78.92 / 89.72

Negative controls completed for seeds 1206, 1207, and 1208:

- shuffled-label and random-ranked controls stay near random at species/genus
  top10 across Eval C, seen-test, and unseen-genera.
- family/order negative-control scores can appear non-zero because broad
  clades occupy many candidates; these are useful null baselines for rank-aware
  calibration figures.

Interpretation:

- Classical sequence baselines are very strong for higher-rank neighborhood
  recovery when a nearby reference exists.
- They cannot recover held-out species labels that are absent from the
  reference candidate sequence set.
- The neural tree-space result is therefore most distinctive as open-candidate
  species-tree placement, not as a blanket replacement for BLAST/VSEARCH/k-mer
  at genus/family/order.

Next Paper 1 actions:

1. Build rank-adaptive calibration/no-call curves using neural seeds,
   BLAST/VSEARCH/k-mer, and negative controls.
2. Turn reference diagnostics into nearest-reference/missingness source tables.
3. Add model-family baselines only after calibration tables are stable.
4. Draft figure source tables for tree recovery, retrieval, baselines,
   controls, and no-call curves.

## 2026-05-30 Rank-Adaptive Calibration First Pass

Added:

- `scripts/edna/build_rank_adaptive_calibration.py`

Outputs:

- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/calibration_per_query.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/coverage_calibration_curves.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/target_precision_thresholds.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/rank_adaptive_policy_summary.csv`
- `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/rank_adaptive_calibration_manifest.json`

Scope:

- 36 prediction sets.
- 438,060 per-query rows.
- Neural cosine512 seeds 1206/1207/1208.
- k-mer, BLAST, VSEARCH baselines.
- shuffled-label and random-ranked negative controls.

Calibration features:

- top score,
- top1-top2 score margin,
- relative margin.

First-pass observations using the empirical 95% target and confidence margin:

- Eval C:
  - neural seeds assign about 22% of queries at roughly 92-94% observed
    precision, mostly genus/family/order calls;
  - BLAST assigns about 92% at roughly 94% observed precision, mostly
    family/order calls;
  - k-mer and VSEARCH assign about 66% and 59% respectively, mostly
    family/order calls.
- Seen-test:
  - all methods can assign nearly all queries;
  - sequence baselines are extremely strong because the species references are
    present.
- Unseen-genera:
  - top-1 margin calibration is too strict for neural seeds at a 95% target;
  - sequence baselines only support very small high-confidence family/order
    subsets.

Interpretation:

- The rank-adaptive framing is useful, but final Paper 1 calibration should not
  rely only on top-1 margin.
- Next calibration pass should include top-k evidence, tree-distance of the
  candidate set, nearest-reference diagnostics, and possibly split-calibrated
  thresholds.

## 2026-05-30 Encoder Benchmark Expansion

Decision:

- Paper 1 should not stop at MarineMamba or at Eval C retrieval.
- The method should become an encoder-agnostic barcode-to-tree benchmark.
- Mamba is one encoder candidate, not the definition of the paper.

Added:

- `experiments/paper1_phylo_calibrated_assignment/ENCODER_BENCHMARKS.md`
- `scripts/edna/train_fish_tree_encoder_benchmark.py`
- `experiments/paper1_phylo_calibrated_assignment/runs/02_vast_encoder_benchmarks.sh`

Implemented first neural encoder tier:

- CNN barcode encoder,
- biLSTM encoder,
- small Transformer encoder.

Fixed protocol:

- same clean COI fish-tree inputs,
- same candidate universe,
- same tree embedding NPZ,
- same Eval C / seen-test / unseen-genera splits,
- same zero-shot candidate metrics,
- same downstream calibration/no-call analysis.

Literature-driven encoder ladder:

1. classical sequence similarity:
   - k-mer,
   - BLAST,
   - VSEARCH.
2. small supervised neural encoders:
   - CNN,
   - biLSTM,
   - Transformer.
3. state-space/long-context encoders:
   - Mamba/BarcodeMamba,
   - S5/S4,
   - Caduceus-style reverse-complement-aware Mamba.
4. barcode-specific foundation encoders:
   - BarcodeBERT.
5. general DNA foundation encoders:
   - DNABERT-2,
   - Nucleotide Transformer,
   - HyenaDNA,
   - Caduceus.

Claim logic:

- If Mamba wins, architecture matters.
- If CNN/LSTM/Transformer match Mamba, tree-space supervision is the central
  method contribution.
- If all neural encoders beat classical baselines only on open-candidate
  species placement, that is still valuable because BLAST/VSEARCH/k-mer cannot
  retrieve species absent from the reference sequence database.
- If classical methods dominate higher ranks, the final calibrated system
  should combine neural and classical evidence rather than force one winner.

Launch:

- Copied the benchmark script and runner to the active Vast RTX PRO 6000 host.
- Launched queue:
  `results/paper1_phylo_calibrated_assignment/encoder_benchmarks/encoder_benchmark_queue.pid`.
- Remote PID: 83257.
- First phase: CNN train/eval_c.
- Monitor: `monitor-paper1-encoder-benchmarks`.

Overnight follow-up:

- Added generic tree-recovery evaluator for non-Mamba encoders:
  `scripts/edna/eval_fish_tree_encoder_tree_recovery.py`.
- Added Vast follow-up runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/03_vast_encoder_tree_recovery_followup.sh`.
- Launched follow-up PID 86477. It waits for the encoder benchmark queue to
  finish, then evaluates CNN/biLSTM/Transformer tree recovery on Eval C and
  unseen-genera.
- Updated monitor `monitor-paper1-encoder-benchmarks` to copy results only
  after both retrieval and tree-recovery outputs complete.

## 2026-05-30 Encoder Benchmark Results Copied

Copied completed Vast outputs locally to:

- `results/remote_runs/2026-05-30/rtx_pro_6000/paper1_encoder_benchmarks/`

Completed:

- CNN train/eval_c plus seen-test and unseen-genera predictions.
- biLSTM train/eval_c plus seen-test and unseen-genera predictions.
- small Transformer train/eval_c plus seen-test and unseen-genera predictions.
- Eval C and unseen-genera tree-recovery evaluations for all three encoders.

Retrieval top10 species/genus/family/order:

- Eval C:
  - CNN: 7.94 / 68.19 / 89.80 / 96.52
  - biLSTM: 8.10 / 50.21 / 73.29 / 80.84
  - Transformer: 4.27 / 36.91 / 60.36 / 70.44
  - Mamba cosine512 seed1206: 10.84 / 73.77 / 86.94 / 91.56
  - BLAST: 0.00 / 94.96 / 98.12 / 98.60
- Seen-test:
  - CNN: 58.85 / 71.83 / 91.67 / 98.80
  - biLSTM: 41.32 / 57.93 / 83.80 / 93.46
  - Transformer: 26.63 / 40.33 / 65.56 / 78.08
  - Mamba cosine512 seed1206: 77.32 / 85.42 / 96.01 / 98.95
  - BLAST: 98.28 / 98.78 / 99.38 / 99.70
- Unseen-genera:
  - CNN: 7.77 / 29.68 / 77.74 / 90.97
  - biLSTM: 4.47 / 22.62 / 46.69 / 62.22
  - Transformer: 3.83 / 16.64 / 45.53 / 60.73
  - Mamba cosine512 seed1206: 5.37 / 21.69 / 62.92 / 81.74
  - BLAST: 0.00 / 0.00 / 82.96 / 92.36

Tree recovery, zero-shot/reference Pearson/Spearman:

- Eval C:
  - CNN: 0.938 / 0.885
  - biLSTM: 0.867 / 0.822
  - Transformer: 0.824 / 0.797
  - Mamba cosine512 seed1206: 0.914 / 0.859
- Unseen-genera:
  - CNN: 0.925 / 0.875
  - biLSTM: 0.835 / 0.798
  - Transformer: 0.780 / 0.758
  - Mamba cosine512 seed1206: 0.859 / 0.821

Interpretation:

- This is a strong and useful result, but it changes the story.
- The tree-space objective is not Mamba-specific: a simple CNN recovered tree
  geometry extremely well and outperformed the current Mamba run on tree
  recovery correlations.
- Mamba still has the best Eval C and seen-test species top10 retrieval among
  the neural encoders.
- CNN is stronger on unseen-genera retrieval and tree recovery.
- Classical baselines remain dominant for higher-rank retrieval when a nearby
  reference exists, but they cannot retrieve held-out species absent from the
  reference sequence database.

Paper implication:

- Paper 1 should be framed as a barcode-to-tree-space and calibrated
  assignment benchmark, not as "Mamba beats all encoders."
- The strongest method contribution is the tree-space objective plus honest
  rank-aware evaluation.
- The next architecture decision should be based on this result:
  - run seed repeats for CNN if we want to claim CNN strength;
  - run contrastive/hierarchical CNN if we want a fairer Mamba-vs-CNN objective
    comparison;
  - only then decide whether S5/BarcodeBERT/DNABERT are worth the next GPU
    spend.

## 2026-05-31 Comparator Boundary Added

- Added `COMPARATOR_MATRIX.md` because the broad idea is not unique:
  phylogenetic placement already places query sequences on fixed trees, DEPP
  already learns single-gene-to-species-tree placement, and a fish COI backbone
  placement study already benchmarked EPA-ng/APPLES.
- Paper 1 should compare against:
  - BLAST/VSEARCH/k-mer as classical similarity baselines,
  - EPA-ng/pplacer/SEPP/APPLES-style placement as direct fixed-tree baselines,
  - DEPP/H-DEPP as the closest neural placement predecessors,
  - CNN topology inference and Phyloformer as related but non-identical neural
    phylogenetic reconstruction work,
  - tree/hyperbolic embeddings as representation-geometry context.
- The comparison is only apples-to-apples if all methods use the same clean COI
  splits, reference sequence set, candidate tree, taxonomy, and shared scoring
  adapter.
- Different outputs must not be collapsed dishonestly:
  - similarity methods produce ranked hits,
  - placement methods produce edge placements/confidence,
  - neural tree-space encoders produce candidate distances/scores.
- Required next implementation step: build a placement-baseline feasibility
  check and output harmonization plan before claiming novelty.

## 2026-05-31 DEPP / H-DEPP Technique Notes

- Added `DEPP_HDEPP_NOTES.md`.
- Main lesson:
  - DEPP is supervised metric learning from gene sequences into
    species-tree-distance space, not just a species lookup table.
  - H-DEPP keeps the same sequence-to-tree idea but changes the target geometry
    to hyperbolic space, which can represent branching tree structure with
    lower distortion.
- Paper 1 implication:
  - separate distance learning from placement/rank/no-call;
  - evaluate distance distortion explicitly, not only Pearson/Spearman;
  - treat Euclidean vs hyperbolic tree geometry as a possible ablation;
  - keep our distinct contribution in barcode assignment behavior:
    candidate ranking, taxonomic rank recovery, calibration, no-call, and
    reference-gap diagnostics.

## 2026-05-31 Literature Gap Audit

- Added `LITERATURE_GAPS.md`.
- Main correction:
  - fish COI placement onto a backbone tree is already directly studied using
    EPA-ng and APPLES, so Paper 1 cannot claim that as new.
- Strongest remaining gap:
  - unified benchmark across neural encoders, sequence similarity, and
    phylogenetic placement, all under the same clean missing-reference COI
    splits, scored by tree-distance and calibrated species/genus/family/order
    no-call behavior.
- Best novelty path:
  - not architecture superiority;
  - not forced species assignment;
  - rather: calibrated tree-aware biodiversity inference under missing
    references, with reference-gap diagnostics and rank-adaptive outputs.

## 2026-05-31 Classical Baseline Refresh

- Completed and copied BLAST/VSEARCH/k-mer baselines for:
  - Eval C,
  - seen-test,
  - unseen-genera.
- Fixed `scripts/edna/build_fish_tree_reference_diagnostics.py` so nearest
  reference binning handles missing/non-finite tree distances instead of
  crashing in `pandas.qcut`.
- Reran reference diagnostics cleanly after the patch.

Key top10 species/genus/family/order results:

- Eval C:
  - k-mer: 0.00 / 94.07 / 97.52 / 98.46
  - BLAST: 0.00 / 94.96 / 98.12 / 98.60
  - VSEARCH: 0.00 / 95.57 / 98.06 / 98.72
- seen-test:
  - k-mer: 98.24 / 98.70 / 99.35 / 99.67
  - BLAST: 98.28 / 98.78 / 99.38 / 99.70
  - VSEARCH: 98.26 / 98.73 / 99.37 / 99.71
- unseen-genera:
  - k-mer: 0.00 / 0.00 / 78.92 / 89.72
  - BLAST: 0.00 / 0.00 / 82.96 / 92.36
  - VSEARCH: 0.00 / 0.00 / 82.91 / 92.40

Interpretation:

- Classical methods are very strong for reference-backed species assignment and
  higher-rank recovery.
- They necessarily score 0.00 species top10 on Eval C and unseen-genera because
  the true held-out species are absent from the reference sequence database.
- They also score 0.00 genus top10 on unseen-genera because the true genera are
  absent by split design.
- This strengthens the Paper 1 framing: the contribution is not replacing
  BLAST/VSEARCH/k-mer everywhere, but defining when classical reference-backed
  evidence is enough and when tree-aware/rank-aware inference is needed.

## 2026-05-31 Local Source Tables And Geometry Diagnostics

Added:

- `scripts/edna/build_paper1_source_tables.py`

Generated:

- `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/tree_recovery_metrics.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/tree_distance_bin_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/tree_distance_sample_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/neighborhood_preservation.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/candidate_ablation_rank_backoff.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/reference_diagnostics_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/source_table_manifest.json`

What these tables add:

- a single retrieval-metric table across Mamba seeds/variants, CNN, biLSTM,
  Transformer, BLAST, VSEARCH, k-mer, and negative controls;
- a single tree-recovery table across Mamba and encoder baselines;
- sampled true-tree-distance versus learned-embedding-distance bins;
- neighborhood preservation/enrichment relative to the candidate pool;
- post-hoc candidate-ablation/rank-backoff summaries where true species,
  genus, or family candidates are hidden from the saved top-50 rankings.

Important new diagnostic signal:

- Candidate ablation confirms why rank-aware output matters. On Eval C, after
  hiding the true species from the top-50 candidate list:
  - Mamba seed1206 still recovers genus/family/order top10 at
    73.59 / 86.94 / 91.56;
  - CNN seed1206 recovers 67.77 / 89.79 / 96.52;
  - BLAST/VSEARCH/k-mer remain strongest for genus/family/order when sequence
    references support the clade.
- After hiding the true genus, all methods drop at family/order, and after
  hiding the true family, order recovery becomes much harder. This is exactly
  the missing-reference/rank-backoff regime Paper 1 should formalize.

Caveat:

- The original copied reference-diagnostic nearest-reference distance bins were
  all `missing`. This was fixed locally by normalizing tree labels with spaces
  to the underscore labels used by the clean input packs. Eval C and
  unseen-genera now have nearest-reference tree-distance bins; seen-test is
  zero-distance by design because query species are represented in the
  reference set.
- CNN/biLSTM/Transformer query embeddings and CNN repeat embeddings are now
  copied locally. Mamba query embeddings remain missing because `mamba-ssm`
  did not build cleanly on the current PyTorch 2.12 / CUDA 13 image.
- Strict candidate-ablation still requires either full candidate score matrices
  or a retrained/tree-pruned protocol; the current top-50 and embedding
  ablations are diagnostic, not final causal tests.

Rank-adaptive calibration refresh:

- Updated `scripts/edna/build_rank_adaptive_calibration.py` to include
  CNN/biLSTM/Transformer encoder-benchmark prediction sets.
- Regenerated:
  - `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/calibration_per_query.csv`
  - `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/coverage_calibration_curves.csv`
  - `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/target_precision_thresholds.csv`
  - `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/rank_adaptive_policy_summary.csv`
  - `results/paper1_phylo_calibrated_assignment/rank_adaptive_calibration/rank_adaptive_calibration_manifest.json`
- The refreshed calibration covers 45 prediction sets and 547,575 query rows.
- Caveat remains: these are same-split empirical diagnostics, not prospective
  operating thresholds.

Exp 45 done: check. Added species-disabled nested threshold-stability testing
for the full sequence+tree Eco-Phylo posterior.

- Added `scripts/edna/build_species_disabled_nested_calibration.py`.
- Generated:
  - `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_summary.csv`
  - `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_per_repeat.csv`
  - `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_thresholds.csv`
  - `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_full/eco_phylo_species_disabled_nested_calibration_assignments.csv.gz`
- Across 30 calibration resplits, the original held-out Global_eDNA groups at
  target-95 average 40.2% assignment at 94.3% accuracy; the 5th-95th percentile
  held-out accuracy range is 93.9-94.9%.
- Interpretation: the species-disabled genus -> family -> order threshold
  policy is reasonably stable as a thresholding layer. It is still not a full
  nested posterior retrain, and species remains disabled for eDNA claims.

Exp 46 done: check. Ran the true nested full sequence+tree Eco-Phylo posterior
on Vast.

- Remote output was copied into:
  `results/remote_runs/2026-06-02/rtx_pro_6000/eco_phylo_candidate_level_sequence_tree_evidence_nested_fit70_rep0/`.
- Local working output is:
  `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep0/`.
- Model fit split: 70% of calibration groups.
- Threshold split: remaining 30% of calibration groups.
- Evaluation split: original held-out Global_eDNA groups.
- Species still fails:
  - species target-95 assigns 2.3% at 1.3% held-out accuracy.
- Higher ranks transfer:
  - genus target-90: 21.2% assignment at 94.1% accuracy;
  - family target-95: 34.0% assignment at 95.4% accuracy;
  - order target-95: 34.4% assignment at 96.7% accuracy.
- Species-disabled rank-backoff:
  - target-90: 48.8% assignment at 85.9% accuracy;
  - target-95: 38.9% assignment at 93.4% accuracy.
- Interpretation: the true nested fit supports conservative higher-rank eDNA
  inference but does not support species-level eDNA calls or a guaranteed 95%
  mixed rank-backoff operating point.

Exp 47 done: check. Added literature-grounded method-angle docs and packaged
production pipeline v1.

- Added five method-angle docs under
  `experiments/research_program/method_angles/`:
  - fast vector-first barcode retrieval;
  - candidate-level Eco-Phylo posterior;
  - rank-adaptive no-call calibration;
  - multi-marker shared tree space;
  - hyperbolic tree geometry.
- Added `scripts/edna/run_paper1_production_v1.py` and
  `experiments/paper1_phylo_calibrated_assignment/runs/12_run_production_v1.sh`.
- Production v1 applies locked p-distance-rerank mode thresholds to existing
  CNN seed1206 pipeline outputs.
- Production v1 summary:
  - seen-test: 95.2% coverage, 94.9% assigned precision, 0.0% false species
    calls;
  - Eval C: 95.8% coverage, 93.0% assigned precision, 0.0% false species calls;
  - unseen-genera: 92.3% coverage, 83.9% assigned precision, 0.0% false
    species calls.
- Updated README, PIPELINE, CURRENT_RESULTS, SOURCE_TABLES, TESTING, and
  CLAIM_BOUNDARIES so stale nested-posterior and p-distance-experimental
  language does not hide the current production-v1 state.
- Caveat: production v1 is saved-embedding COI inference, not raw
  FASTA-to-final-call production yet.

Exp 48 done: check. Ran raw split-sequence production-v1 on Vast.

- Added `scripts/edna/run_paper1_raw_sequence_production_v1.py`.
- Added `scripts/edna/build_paper1_raw_sequence_production_summary.py`.
- Used Vast endpoint `ssh -p 23156 root@194.14.47.19` with `/venv/main`
  PyTorch 2.10 CUDA environment.
- Installed missing small dependencies into `/venv/main`: pandas,
  scikit-learn, hnswlib, dendropy, biopython.
- Ran CNN seed1206 from raw clean split `zero_shot_queries.csv` sequences
  through embedding export, exact vector retrieval, top-25 train-reference
  p-distance rerank, and locked production-v1 rank/no-call packaging.
- Outputs copied locally:
  - `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_raw_sequence_production_v1/`
  - `results/paper1_phylo_calibrated_assignment/raw_sequence_production_v1/`
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/raw_sequence_production_v1_summary.csv`.
- Timing and assignment summary:
  - seen-test: 15,763 queries, 49.80 seconds, 3.16 ms/query, 95.2% coverage,
    94.9% assigned precision, 0.0% false species calls;
  - Eval C: 11,594 queries, 46.77 seconds, 4.03 ms/query, 95.8% coverage,
    93.0% assigned precision, 0.0% false species calls;
  - unseen-genera: 9,148 queries, 47.12 seconds, 5.15 ms/query, 92.3%
    coverage, 83.9% assigned precision, 0.0% false species calls.
- Interpretation: production v1 now runs from raw split sequences, but still
  needs arbitrary FASTA parsing and a CLI/API wrapper before it should be
  called a finished user-facing tool.

Exp 49 done: check. Added the COI DL evidence model as an optional production
CLI decision layer.

- Added `scripts/edna/apply_paper1_coi_evidence_model.py`.
- Patched `scripts/edna/run_paper1_fasta_inference_v1.py` with
  `--decision-mode dl_mlp_species_disabled`.
- The default CLI mode remains `production_thresholds`; the DL mode is explicit
  and species-disabled.
- Local adapter smoke on
  `coi_cnn_seed1206_eval_c_target099_pdistance_experimental` reproduced the
  expected held-out fish result:
  - 11,594 queries;
  - 94.2% coverage;
  - 97.4% assigned precision if known;
  - 0 species calls.
- Vast full CLI known-label smoke:
  - output copied to
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_eval_c_known_dl_rerun/`;
  - 16 queries;
  - 100.0% coverage;
  - 100.0% assigned precision if known;
  - 0 species calls;
  - 14.08 seconds total.
- Vast full CLI unlabeled FASTA smoke:
  - output copied to
    `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_unlabeled_fasta_dl/`;
  - 8 queries;
  - 100.0% coverage;
  - precision unavailable by design because no known labels were supplied;
  - 0 species calls;
  - 17.16 seconds total.
- Interpretation: the DL rank/no-call model is now usable inside the pipeline.
  It improves precision at lower coverage in held-out evaluation, but it is not
  yet the default operating point until seed repeats and strict hidden-taxonomy
  tests are done.

Exp 50 done: check. Ran COI DL evidence model seed repeats and aggregated the
results into source tables.

- Trained MLP repeats with seeds 1207 and 1208 using the same settings as the
  seed1206 model:
  `--epochs 80 --patience 12 --batch-size 1024 --hidden-dim 64 --dropout 0.1 --n-bootstrap 1000 --cpu`.
- Added and ran
  `scripts/edna/build_paper1_dl_evidence_seed_summary.py`.
- Source tables:
  - `results/paper1_phylo_calibrated_assignment/source_tables/dl_evidence_seed_summary.csv`;
  - `results/paper1_phylo_calibrated_assignment/source_tables/dl_evidence_seed_bootstrap_summary.csv`;
  - `results/paper1_phylo_calibrated_assignment/source_tables/dl_evidence_seed_summary_manifest.json`.

- Species-disabled target-0.99 range across MLP seeds 1206/1207/1208:
  - held-out fish: 94.2-96.0% coverage, 97.1-97.4% assigned precision, 0.0%
    false species-call rate;
  - unseen-genera: 88.5-91.3% coverage, 92.9-93.5% assigned precision, 0.0%
    false species-call rate.
- Interpretation: the precision gain is stable across seeds. Coverage varies,
  but the species-disabled DL layer consistently avoids false species calls and
  is now a credible precision-first alternative to hand thresholds.

Exp 51 done: check. Applied the COI DL evidence model to strict
hidden-reference executable pipeline runs.

- Regenerated executable p-distance pipeline rows from strict CNN
  `query_embeddings.npz` for all six strict packs:
  - Eval C hide species/genus/family;
  - unseen-genera hide species/genus/family.
- Applied `scripts/edna/apply_paper1_coi_evidence_model.py` with
  species-disabled target-0.99 policy.
- Added and ran
  `scripts/edna/build_paper1_dl_strict_apply_summary.py`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/dl_evidence_strict_apply_summary.csv`.
- Results:
  - Eval C hide species: 58.0% coverage, 88.5% precision, 0 species calls.
  - Eval C hide genus: 57.9% coverage, 82.5% precision, 0 species calls.
  - Eval C hide family: 16.1% coverage, 81.5% precision, 0 species calls.
  - unseen-genera hide species: 50.6% coverage, 91.3% precision, 0 species calls.
  - unseen-genera hide genus: 52.9% coverage, 89.5% precision, 0 species calls.
  - unseen-genera hide family: 41.9% coverage, 72.7% precision, 0 species calls.
- Interpretation: the DL calibrator is conservative under strict missing
  references, but hidden-family stress exposes the need for a dedicated
  reference-gap detector and/or candidate-level reranker before using DL as the
  default production policy.

Exp 63 done: check. Trained the first diagnostic COI reference-gap detector.

- Added `scripts/edna/train_paper1_reference_gap_detector.py`.
- Added `scripts/edna/build_paper1_reference_gap_detector_summary.py`.
- Generated two detector variants:
  - `reference_gap_detector/coi_mlp_seed1206/`: target-0.90 with candidate-set
    counts included; diagnostic only because counts can encode strict-pack
    identity.
  - `reference_gap_detector/coi_mlp_seed1206_no_counts_target099/`: target-0.99
    without candidate-count features; preferred honest per-query diagnostic.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/reference_gap_detector_summary.csv`.
- Preferred no-counts target-0.99 normal supported false-gap flag rates:
  - held-out fish species/genus/family: 4.7 / 0.3 / 0.2%;
  - seen-test species/genus/family: 1.5 / 0.1 / 0.0%;
  - unseen-genera species/genus/family: 11.1 / 1.2 / 1.1%.
- Strict unseen-genera gap recall:
  - hidden species: species-gap recall 54.0%;
  - hidden genus: genus-gap recall 1.0%;
  - hidden family: family-gap recall 2.0%.
- Interpretation: this is not production-ready as a reason layer. It gives us
  a useful boundary: aggregate top-k evidence can partly detect missing species,
  but reliable genus/family gap diagnosis needs candidate-level tree,
  BLAST/VSEARCH identity, and reference-density features.

Exp 64 done: check. Trained the first diagnostic COI candidate-level reranker.

- Added `scripts/edna/train_paper1_candidate_reranker.py`.
- Added `scripts/edna/build_paper1_candidate_reranker_summary.py`.
- Trained a top-50 candidate MLP on seen-test candidate rows using vector
  score, p-distance, candidate rank, top-k taxonomic cluster, and reference
  availability features.
- Output root:
  `results/paper1_phylo_calibrated_assignment/candidate_reranker/coi_mlp_seed1206_top50/`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/candidate_reranker_summary.csv`.
- Candidate-ordering improvements over p-distance top-1:
  - held-out fish genus: 49.7% vs 38.4%, +11.3 percentage points;
  - held-out fish family: 87.3% vs 84.1%, +3.2 percentage points;
  - unseen-genera family: 71.3% vs 69.1%, +2.2 percentage points.
- Calibration transfer from seen-test target-0.99 is not production-ready:
  - held-out fish genus/family/order assigned precision:
    98.2 / 98.1 / 97.8%;
  - unseen-genera family/order assigned precision: 96.1 / 93.6%.
- Interpretation: candidate-level DL can improve ranking, but the first
  version is a development result. The next version needs BLAST/VSEARCH
  identity, explicit tree-neighborhood density, and stronger calibration before
  replacing the current production-v1 rank/no-call policy.

Exp 65 done: check. Completed and copied the retrieval-DL encoder sweep.

- Vast endpoint: `ssh -p 23156 root@194.14.47.19`.
- Runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/13_vast_retrieval_dl_sweep.sh`.
- Remote queue PID: `8985`, completed.
- Remote queue root:
  `/workspace/marinemamba/results/paper1_phylo_calibrated_assignment/retrieval_dl_sweep/`.
- Local copy:
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_retrieval_dl_sweep/`.
- Arms:
  - `coi_cnn_retrieval_contrastive_seed1301`;
  - `coi_cnn_retrieval_hybrid_seed1301`;
  - `coi_cnn_retrieval_hier_contrastive_seed1301`;
  - `coi_transformer_retrieval_hier_contrastive_seed1301`.
- Added `scripts/edna/build_paper1_retrieval_dl_sweep_summary.py`.
- Source tables:
  - `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_dl_sweep_summary.csv`;
  - `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_dl_sweep_tree_recovery.csv`;
  - `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_dl_sweep_training_history.csv`;
  - `results/paper1_phylo_calibrated_assignment/source_tables/retrieval_dl_sweep_manifest.json`.
- Best candidate-generator result:
  - CNN contrastive held-out top50 species/genus/family/order:
    61.2 / 96.2 / 97.4 / 98.2%;
  - CNN hybrid held-out top50 species/genus/family/order:
    60.8 / 95.6 / 97.4 / 98.0%;
  - unseen-genera top50 species/genus/family/order:
    CNN contrastive 34.1 / 62.9 / 86.1 / 90.5%;
    CNN hybrid 34.0 / 60.3 / 86.4 / 91.6%.
- Tree-geometry tradeoff:
  - CNN hierarchical contrastive improved held-out zero-shot/reference tree
    recovery to Pearson/Spearman 0.585 / 0.587;
  - Transformer hierarchical contrastive reached 0.630 / 0.575, but its
    retrieval recall was much weaker.
- Interpretation: CNN contrastive/hybrid are the practical retrieval-front-end
  candidates. Hierarchical/tree-shaped losses are useful probes for geometry,
  but they currently sacrifice too much candidate recall. This is
  model-development evidence, not a replacement for BLAST/VSEARCH or the
  current production rank/no-call policy.

Exp 66 done: check. Completed retrieval-DL pipeline diagnostics and
second-stage BLAST/VSEARCH-aware candidate reranker training on Vast.

- Vast endpoint: `ssh -p 23156 root@194.14.47.19`.
- Runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/14_vast_retrieval_dl_pipeline_reranker.sh`.
- Remote queue PID: `30799`.
- Remote queue root:
  `/workspace/marinemamba/results/paper1_phylo_calibrated_assignment/retrieval_dl_pipeline_reranker/`.
- Arms:
  - `coi_cnn_retrieval_contrastive_seed1301`;
  - `coi_cnn_retrieval_hybrid_seed1301`.
- Sequence:
  - generate p-distance-reranked pipeline rows for Eval C, seen-test, and
    unseen-genera from each retrieval-DL embedding set;
  - train a top-50 candidate MLP reranker for each arm;
  - include vector score, p-distance, taxonomic consensus, reference flags, and
    BLAST/VSEARCH top-50 candidate evidence.
- Script change:
  `scripts/edna/train_paper1_candidate_reranker.py` now accepts repeated
  `--run split=path` arguments and optional `--baseline-methods blast vsearch`
  evidence.
- Training status:
  - contrastive reranker: 1,825,250 candidate rows, 25 features, CUDA training
    completed with train/calibration loss 0.136 / 0.133 at epoch 80;
  - hybrid reranker: 1,825,250 candidate rows, 25 features, CUDA training
    completed with train/calibration loss 0.135 / 0.132 at epoch 80.
- Outputs copied locally:
  `results/paper1_phylo_calibrated_assignment/pipeline_runs/coi_cnn_retrieval_*_target099_pdistance_experimental/`
  and
  `results/paper1_phylo_calibrated_assignment/candidate_reranker/*_top50_blast_vsearch/`.
- Refreshed source tables:
  `candidate_reranker_summary.csv`, `pipeline_run_summary.csv`,
  `pipeline_coi_method_benchmark.csv`, `pipeline_end_to_end_summary.csv`, and
  manuscript asset inventories.
- Best transfer signal:
  - held-out fish: top-1 genus improves by about +33 to +34 pp over p-distance
    order; family improves by about +6 to +7 pp;
  - unseen-genera: family improves by +1.6 to +2.6 pp and order by +0.6 to
    +0.9 pp;
  - species remains unsupported in the missing-reference splits, as expected.
- Interpretation: BLAST/VSEARCH-aware candidate-level DL is a useful evidence
  fusion layer, but it is not yet the final rank/no-call policy. It needs
  tree-neighborhood features and independent calibration before it can replace
  the conservative production-v1 decision layer.

Exp 67 done: check. Completed retrieval-DL candidate reranker with explicit
tree-neighborhood evidence on Vast.

- Vast endpoint: `ssh -p 23156 root@194.14.47.19`.
- Runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/15_vast_retrieval_dl_tree_reranker.sh`.
- Remote queue PID: `32217`.
- Remote queue root:
  `/workspace/marinemamba/results/paper1_phylo_calibrated_assignment/retrieval_dl_tree_reranker/`.
- Arms:
  - `coi_cnn_retrieval_contrastive_seed1301_top50_blast_vsearch_tree10`;
  - `coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10`.
- Added features:
  inference-safe distances within the retrieved top-10 candidate neighborhood,
  top-k tree-distance spread, top-k genus/family/order diversity, and agreement
  with the top retrieved candidate. These do not use true-query-to-candidate
  tree distance.
- Training status:
  - contrastive tree10 reranker: 1,825,250 candidate rows, 40 features, CUDA
    training completed with train/calibration loss 0.105 / 0.103 at epoch 80;
  - hybrid tree10 reranker: 1,825,250 candidate rows, 40 features, CUDA
    training completed with train/calibration loss 0.105 / 0.104 at epoch 80.
- Outputs copied locally:
  `results/paper1_phylo_calibrated_assignment/candidate_reranker/*_top50_blast_vsearch_tree10/`
  and
  `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_retrieval_dl_tree_reranker/`.
- Refreshed source tables:
  `candidate_reranker_summary.csv`, `pipeline_coi_method_benchmark.csv`,
  `pipeline_end_to_end_summary.csv`, and manuscript asset inventories.
- Best transfer signal:
  - held-out fish genus top-1 reaches 83.7-84.0%, versus 48.2-49.1%
    p-distance order;
  - held-out fish family top-1 reaches 94.7-95.4%, versus 87.6-88.0%
    p-distance order;
  - unseen-genera family top-1 reaches 71.6-72.5%, versus 68.9-69.4%
    p-distance order.
- Interpretation: inference-safe tree-neighborhood features improve candidate
  ordering, especially genus/family, and lower calibration loss substantially.
  They still do not create a production-ready rank/no-call policy by
  themselves; next step is listwise/pairwise training and separate calibration.

Exp 68 done: check. Completed focused query-listwise tree10 candidate reranker
on Vast.

- Vast endpoint: `ssh -p 23156 root@194.14.47.19`.
- Runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/16_vast_retrieval_dl_listwise_tree_reranker.sh`.
- Remote queue PID: `32946` completed.
- Remote queue root:
  `/workspace/marinemamba/results/paper1_phylo_calibrated_assignment/retrieval_dl_listwise_tree_reranker/`.
- Arm:
  `coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10_listwise`.
- Method:
  query-listwise multi-positive softmax loss over each top-50 candidate list,
  using vector/p-distance/BLAST/VSEARCH/tree10 evidence. This optimizes
  within-query candidate ordering directly rather than independent candidate
  labels.
- Local outputs:
  `results/paper1_phylo_calibrated_assignment/candidate_reranker/coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10_listwise/`.
- Refreshed:
  `candidate_reranker_summary.csv`, `paper1_pipeline_benchmarks.csv`,
  `paper1_end_to_end_summary.csv`, and manuscript assets.
- Training:
  1,825,250 candidate rows, 11,822 train query groups, 3,941 calibration query
  groups, 40 features, query-listwise multi-positive softmax. Loss dropped from
  train/calib 3.396/2.784 to 2.338/2.354 across 80 epochs.
- Key comparison against pointwise hybrid tree10:
  held-out fish genus top-1 fell from 84.0% to 69.4%; held-out fish family
  fell from 95.4% to 88.0%; unseen-genera family fell from 72.5% to 65.0%;
  unseen-genera order fell from 85.8% to 83.2%.
- Interpretation:
  listwise training optimized within-query ranking directly but produced a
  weaker and less stable operating point than the pointwise tree10 reranker.
  Keep it as a negative model-development result. The next useful direction is
  independent calibration or pairwise training, not using this listwise model as
  the production reranker.

Exp 69 done: check. Added cross-split calibration-transfer diagnostics for
candidate rerankers.

- New script:
  `scripts/edna/build_paper1_candidate_reranker_calibration_transfer.py`.
- New source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/candidate_reranker_calibration_transfer.csv`.
- New manifest:
  `results/paper1_phylo_calibrated_assignment/source_tables/candidate_reranker_calibration_transfer_manifest.json`.
- Check:
  `python3 -m py_compile scripts/edna/build_paper1_candidate_reranker_calibration_transfer.py`
  passed, then the script completed with progress logging.
- Key result for the strongest pointwise hybrid tree10 reranker using
  seen-test-derived target-0.99 thresholds:
  - held-out fish transfer: genus precision 93.2% at 74.5% coverage, family
    96.5% at 98.7% coverage, order 97.0% at 100.0% coverage;
  - unseen-genera transfer: family precision 77.0% at 93.5% coverage, order
    85.9% at 99.8% coverage;
  - species/genus remain impossible in missing-reference splits by design.
- Interpretation:
  this confirms the DL reranker is a strong candidate-ordering and
  evidence-fusion layer, but seen-test target-0.99 thresholds do not transfer
  to harder missing-reference splits. Keep production rank/no-call conservative
  until we build a stricter independent calibration policy.

Exp 70 done: check. Completed focused query-pairwise tree10 candidate reranker
on Vast.

- Vast endpoint: `ssh -p 23156 root@194.14.47.19`.
- Runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/17_vast_retrieval_dl_pairwise_tree_reranker.sh`.
- Remote queue PID: `33400` completed.
- Remote queue root:
  `/workspace/marinemamba/results/paper1_phylo_calibrated_assignment/retrieval_dl_pairwise_tree_reranker/`.
- Arm:
  `coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10_pairwise`.
- Method:
  query-pairwise positive-vs-negative loss over each top-50 candidate list,
  using vector/p-distance/BLAST/VSEARCH/tree10 evidence. This rewards correct
  candidates scoring above incorrect candidates within the same query.
- Preflight:
  local compile, local runner syntax, remote compile, and remote runner syntax
  passed. The job started and began building candidate rows.
- Training:
  1,825,250 candidate rows, 11,822 train query groups, 3,941 calibration query
  groups, 40 features. Loss dropped from train/calib 0.4336/0.1814 to
  0.0511/0.0548 across 80 epochs.
- Local outputs:
  `results/paper1_phylo_calibrated_assignment/candidate_reranker/coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10_pairwise/`.
- Refreshed:
  `candidate_reranker_summary.csv`,
  `candidate_reranker_calibration_transfer.csv`,
  `paper1_pipeline_benchmarks.csv`, `paper1_end_to_end_summary.csv`, and
  manuscript assets.
- Key comparison against pointwise hybrid tree10:
  - held-out fish genus top-1: 84.1% vs 84.0%; transferred precision 95.9%
    at 43.0% coverage vs 93.2% at 74.5%;
  - held-out fish family top-1: 95.3% vs 95.4%; transferred precision 96.3%
    at 98.2% coverage vs 96.5% at 98.7%;
  - unseen-genera family top-1: 72.4% vs 72.5%; transferred precision 74.9%
    at 92.1% coverage vs 77.0% at 93.5%;
  - unseen-genera order top-1: 86.0% vs 85.8%; transferred precision 86.1%
    at 100.0% coverage vs 85.9% at 99.8%.
- Interpretation:
  pairwise training is not a clean win. It improves some rank-ordering or
  precision-transfer cells but loses coverage or precision elsewhere. The
  strongest current diagnostic reranker remains pointwise tree10, with pairwise
  as a useful alternative diagnostic. The next problem is calibration, not more
  candidate scoring alone.

Exp 71 done: check. Completed selected-candidate assignment calibrators for
pointwise and pairwise tree10 reranker outputs.

- New script:
  `scripts/edna/train_paper1_candidate_assignment_calibrator.py`.
- New source-table script:
  `scripts/edna/build_paper1_candidate_assignment_calibrator_summary.py`.
- Vast runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/18_vast_candidate_assignment_calibrators.sh`.
- Vast queue PID: `33858`, completed.
- Output roots:
  `results/paper1_phylo_calibrated_assignment/candidate_assignment_calibrator/coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10_assignment_calibrator/`
  and
  `results/paper1_phylo_calibrated_assignment/candidate_assignment_calibrator/coi_cnn_retrieval_hybrid_seed1301_top50_blast_vsearch_tree10_pairwise_assignment_calibrator/`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/candidate_assignment_calibrator_summary.csv`.
- Key results:
  - pointwise-selected calibrator: held-out fish genus 93.7% precision at
    62.3% coverage, unseen-genera family 75.9% precision at 95.2% coverage,
    unseen-genera order 85.8% precision at 99.9% coverage;
  - pairwise-selected calibrator: held-out fish genus 94.4% precision at 58.7%
    coverage, unseen-genera family 72.6% precision at 99.6% coverage,
    unseen-genera order 86.1% precision at 99.9% coverage.
- Interpretation:
  a second DL calibrator improves internal calibration but does not fix
  missing-reference transfer. It should not replace the conservative production
  rank/no-call policy.

Exp 72 done: check. Completed no-counts reference-gap detector follow-up.

- Runs:
  `reference_gap_detector/coi_mlp_seed1301_no_counts_target095/` and
  `reference_gap_detector/coi_mlp_seed1301_no_counts_target099/`.
- Source table refreshed:
  `results/paper1_phylo_calibrated_assignment/source_tables/reference_gap_detector_summary.csv`.
- Target-0.95 tradeoff:
  normal supported splits show higher false-gap flag rates, especially species
  gap flags on held-out fish and unseen-genera; hidden-reference recall improves
  substantially for species-level gaps and modestly for genus/family gaps.
- Target-0.99 tradeoff:
  similar to the previous no-counts run, conservative and high precision where
  it fires, but weak genus/family recall.
- Interpretation:
  reference-gap detection is promising as a warning signal for missing
  references, but it is not yet good enough to drive final rank/no-call by
  itself. It needs a better training design with realistic missing-reference
  positives and normal supported negatives.

Exp 73 done: check. Completed candidate-evidence reference-gap detector v2.

- Runs:
  `reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_target095/`
  and
  `reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_target099/`.
- Source table refreshed:
  `results/paper1_phylo_calibrated_assignment/source_tables/reference_gap_detector_summary.csv`.
- Feature design:
  83 inference-safe candidate-list features, including top-k p-distance traces,
  taxonomic diversity, nearest-reference density, and tree-neighborhood
  evidence. It avoids global candidate-count features and split labels.
- Target-0.95 strict unseen-genera gap recall:
  hidden species species-gap 95.4%; hidden genus genus-gap 30.0%;
  hide-family genus-gap 41.3%; hide-family family-gap 32.6%. All four cells
  had 100.0% precision because every strict-pack row is truly unsupported at
  that rank.
- Target-0.99 strict unseen-genera gap recall:
  hidden species species-gap 79.7%; hidden genus genus-gap 10.1%;
  hide-family genus-gap 19.0%; hide-family family-gap 18.9%.
- Normal supported warning rates remain the limiting factor:
  target-0.95 flags unseen-genera species/genus/family at 34.3 / 10.7 / 7.0%
  and held-out fish at 15.3 / 3.4 / 2.4%; target-0.99 lowers these but loses
  recall.
- Interpretation:
  candidate-level evidence is a real improvement for reference-gap diagnosis,
  especially for hidden genus/family cases, but the layer is still a warning
  and explanation signal rather than a production rank/no-call replacement.

Exp 74 done: check. Completed candidate-evidence reference-gap no-tree
ablation.

- Run:
  `reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_notree_target095/`.
- Feature design:
  77 candidate-list features, excluding the 6 tree-neighborhood features.
- Key comparison against tree-aware v2 at target-0.95:
  hidden species species-gap 95.9% no-tree vs 95.4% tree-aware; hidden genus
  genus-gap 30.8% vs 30.0%; hide-family genus-gap 38.8% vs 41.3%;
  hide-family family-gap 27.2% vs 32.6%.
- Interpretation:
  the strongest gain comes from candidate-level sequence/taxonomic evidence.
  Tree-neighborhood evidence adds value for the hardest hide-family case, but
  it is not yet a universally dominant feature family.

Exp 75 done: check. Built the production-v1 gap-warning overlay.

- Script:
  `scripts/edna/build_paper1_gap_warning_overlay.py`.
- Outputs:
  `results/paper1_phylo_calibrated_assignment/source_tables/gap_warning_overlay_summary.csv`,
  `gap_warning_overlay_examples.csv`, and
  `gap_warning_overlay_manifest.json`.
- What it tests:
  no new training; it joins production-v1 assignments with v2 reference-gap
  probabilities and asks whether warning scores should abstain from assigned
  ranks or explain why backing off to broader ranks is justified.
- Main read:
  hard rank-specific warning abstention barely changes precision because
  production-v1 already avoids most risky species/genus calls. With tree-aware
  target-0.95, assigned-rank warnings cover only 0.04% of held-out fish and
  0.34% of unseen-genera assignments, moving precision by only +0.02 and
  +0.12 percentage points.
- More useful read:
  softer more-specific gap warnings explain 12.6% of held-out broader-rank
  assignments and 29.1% of unseen-genera broader-rank assignments. They catch
  25.0% of wrong held-out assignments and 41.3% of wrong unseen-genera
  assignments under the any-warning view. This is useful for reason labels and
  curation prioritization, not yet for final abstention.

Exp 76 done: check. Trained the missing-reference-aware rank/no-call
calibrator.

- Script:
  `scripts/edna/train_paper1_missing_reference_aware_calibrator.py`.
- Summary collector:
  `scripts/edna/build_paper1_missing_reference_calibrator_summary.py`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/dl_evidence_rank_backoff/coi_mlp_seed1401_missing_reference_aware_v2_gap/`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/missing_reference_aware_calibrator_summary.csv`.
- Training design:
  train/calibrate on normal supported seen-test rows plus strict Eval C
  hidden-species/genus/family rows; evaluate on normal held-out fish,
  normal unseen-genera, and strict unseen-genera hidden-species/genus/family
  rows. Features include production-v1 evidence plus v2 reference-gap
  probabilities, but no split labels, roles, or candidate-count leakage.
- Loss:
  calibration loss dropped from 0.499 at epoch 1 to 0.245 at epoch 100.
- Best conservative read, species-disabled target-0.99:
  held-out fish 91.0% coverage, 98.3% assigned precision, 0 species calls;
  unseen-genera 79.7% coverage, 95.7% assigned precision, 0 species calls.
- Strict unseen-genera stress, species-disabled target-0.99:
  hidden species 37.3% coverage / 94.7% precision; hidden genus 37.9% /
  94.8%; hidden family 28.6% / 78.7%.
- Interpretation:
  this is the first calibrator trained with deployment-matched
  missing-reference positives. It improves precision and strict-family
  caution, but coverage drops versus production-v1 and hidden-family precision
  remains below a manuscript-grade target. Keep it as a precision-first
  model-development result for now.

Exp 77 done: check. Completed eDNA Eco-Phylo posterior nested stability repeats.

- Vast endpoint:
  `ssh -p 23156 root@194.14.47.19`.
- Queue log:
  `/workspace/marinemamba/results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/nested_repeat_queue/nested_repeats_1_2.log`.
- Outputs:
  `candidate_level_sequence_tree_evidence_nested_fit70_rep1/` and
  `candidate_level_sequence_tree_evidence_nested_fit70_rep2/`, copied locally
  and summarized with species-disabled rank-backoff tables.
- Result:
  rep0 target-95 assigns 38.9% at 93.4% accuracy; rep1 assigns 38.5% at
  95.4%; rep2 assigns 54.9% at 84.5%.
- Interpretation:
  the eDNA posterior is useful evidence for higher-rank assignment, but current
  target-95 mixed rank-backoff is unstable across nested refits and should not
  be a headline final operating point yet.

Exp 78 done: check. Ran first MarkerMirror / BarcodeBridge cross-marker probe.

- Script:
  `scripts/edna/train_marker_mirror_bridge.py`.
- Vast endpoint:
  `ssh -p 23156 root@194.14.47.19`.
- Output root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_multisource/`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`.
- Design:
  frozen Nucleotide Transformer v2-50M embeddings plus small learned projection
  heads aligning COI and 12S sequences for species with both markers. Species
  split is 674 train, 144 validation, 145 held-out test overlap species.
- Training:
  contrastive loss dropped from 4.578 at epoch 1 to 0.902 at epoch 80.
- Held-out cross-marker top-10 improvement, 12S query to COI species
  prototypes:
  frozen NT species/genus/family/order 0.7 / 3.5 / 24.2 / 43.0%;
  MarkerMirror projection 7.8 / 16.9 / 49.0 / 65.9%.
- Interpretation:
  this is the best new breakthrough prospect from the foundation-model lane.
  It proves the cross-marker objective creates real held-out signal that frozen
  embeddings do not have. It is not yet a finished tool because train retrieval
  is very high while held-out species retrieval remains modest. Next: hard
  negatives, genus/family-balanced batching, tree-distance-aware contrastive
  loss, then LoRA/backbone fine-tuning if the improved objective still helps.

Exp 79 done: check. Ran taxonomy-hard MarkerMirror follow-up.

- Script update:
  `scripts/edna/train_marker_mirror_bridge.py` now supports
  `--batch-strategy taxonomy_hard` and logs validation loss.
- Output root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_multisource_taxonomy_hard/`.
- Source table refreshed:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`
  now contains both the random-negative and taxonomy-hard bridge runs.
- Training:
  loss dropped from 4.573 to 0.519; validation loss rose from 4.573 to 5.133,
  so the objective overfits, but retrieval still improved.
- Held-out cross-marker top-10, taxonomy-hard run:
  species/genus/family/order 10.3 / 20.4 / 50.7 / 70.4%.
- Comparison to random-negative run:
  species/genus/family/order top-10 improved from 7.8 / 16.9 / 49.0 / 65.9%.
- Interpretation:
  close-relative negatives help. More epochs alone are not the answer. Next
  objective should use tree-distance-aware soft targets or supervised
  contrastive positives so close relatives are not treated as equally wrong as
  distant species.

Exp 80 done: check. Ran taxonomy-soft MarkerMirror and best-validation restore.

- Script update:
  `scripts/edna/train_marker_mirror_bridge.py` now supports
  `--loss-mode taxonomy_soft` and `--restore-best-val`.
- Output roots:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_multisource_taxonomy_soft/`
  and
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_multisource_taxonomy_soft_bestval/`.
- Source table refreshed:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`
  now contains random-negative, taxonomy-hard, taxonomy-soft, and taxonomy-soft
  best-val bridge runs.
- Taxonomy-soft final epoch, held-out top-10:
  species/genus/family/order 9.8 / 20.7 / 54.5 / 72.5%.
- Taxonomy-soft best-val restore, held-out top-10:
  species/genus/family/order 4.4 / 7.1 / 26.7 / 51.1%.
- Interpretation:
  taxonomy-soft rank targets are best for family/order retrieval and align with
  the rank-aware paper story. Best-validation restore is not helpful under the
  current loss; the validation objective and final retrieval metric are not yet
  aligned. Next model step should use actual tree-distance targets and/or
  supervised contrastive positives before any LoRA/backbone fine-tune.

Exp 81 done: check. Ran actual tree-distance-aware MarkerMirror projection.

- Script update:
  `scripts/edna/train_marker_mirror_bridge.py` now supports
  `--loss-mode tree_soft`, `--tree-file`, and `--tree-soft-scale`.
- Remote run:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/nt_v2_50m_multisource_tree_distance_scale25/`.
- Local copied root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_multisource_tree_distance_scale25/`.
- Source table refreshed:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`
  now contains five completed MarkerMirror variants.
- Broad auto-scale tree-soft run:
  stopped early because loss stayed essentially flat through epoch 20; estimated
  scale 212.56 made targets too diffuse.
- Scale-25 tree-soft run:
  all 963 overlap species were present in the fish tree; training loss dropped
  from 4.573 to 1.227 over 60 epochs, but validation loss overfit after epoch
  10.
- Held-out top-10:
  species/genus/family/order 8.4 / 18.3 / 48.3 / 68.1%.
- Interpretation:
  actual patristic tree-distance targets are not automatically better than
  taxonomy-soft rank targets. Taxonomy-soft remains the best family/order
  bridge result at 9.8 / 20.7 / 54.5 / 72.5 top-10. Next step should be
  retrieval-aligned checkpointing or a hybrid objective, not a heavier
  backbone fine-tune yet.

Exp 82 done: check. Ran retrieval-aligned MarkerMirror checkpointing.

- Script update:
  `scripts/edna/train_marker_mirror_bridge.py` now supports
  `--restore-best-retrieval`, `--retrieval-eval-every`,
  `--retrieval-selection-ranks`, and `--retrieval-selection-k`.
- Remote/local run:
  `nt_v2_50m_multisource_taxonomy_soft_retrieval_best`.
- Selection rule:
  restore the epoch with best validation top-10 mean over genus/family/order.
- Validation retrieval score:
  improved from 16.1 at epoch 1 to best 48.6 at epoch 110; restored epoch 110.
- Held-out top-10:
  species/genus/family/order 8.5 / 18.8 / 55.4 / 75.4%.
- Comparison:
  this is the best family/order bridge result so far, improving over
  taxonomy-soft final epoch family/order 54.5 / 72.5. It does not beat
  taxonomy-hard species/genus.
- Interpretation:
  checkpoint selection must use retrieval metrics, not contrastive loss. The
  bridge is strongest as high-rank 12S-to-COI/tree evidence. Next gated task:
  dependency check and a small backbone/LoRA fine-tune only if it uses the same
  held-out species split and retrieval-aligned checkpointing.

Exp 83 done: check. Ran the first LoRA MarkerMirror adapter feasibility run.

- Script added:
  `scripts/edna/train_marker_mirror_lora_bridge.py`.
- Remote/local run:
  `nt_v2_50m_multisource_lora_taxonomy_soft_retrieval_best`.
- Adapter:
  Nucleotide Transformer v2 50M with PEFT LoRA r=8 alpha=16 on attention
  `query,key,value` modules.
- Training:
  taxonomy-hard batches, taxonomy-soft rank targets, 24 epochs, retrieval
  checkpointing every 4 epochs, validation selection by genus/family/order
  top-10.
- Validation retrieval:
  best retrieval checkpoint was epoch 20 with score 24.7045; training loss
  dropped from 3.1867 to 2.5465 by epoch 24.
- Held-out top-10:
  species/genus/family/order 4.0 / 7.5 / 28.8 / 50.1%.
- Comparison:
  this is below the frozen projection-head bridge. Current best species/genus
  remains taxonomy-hard projection at 10.3 / 20.4. Current best family/order
  remains taxonomy-soft retrieval-best projection at 55.4 / 75.4.
- Interpretation:
  naive adapter fine-tuning is not automatically better than projection-head
  alignment. The next MarkerMirror step should change the objective or paired
  data, not simply run LoRA longer.

Exp 84 done: check. Ran multi-positive MarkerMirror projection training.

- Script update:
  `scripts/edna/train_marker_mirror_bridge.py` now supports
  `--sequences-per-species-per-batch` and same-species soft positives.
- Remote/local run:
  `nt_v2_50m_multisource_taxonomy_soft_multipositive_retrieval_best`.
- Training:
  frozen Nucleotide Transformer embeddings, taxonomy-hard batches,
  taxonomy-soft rank targets, two sequences per species per batch,
  retrieval-selected checkpointing.
- Validation retrieval:
  best checkpoint was epoch 110 with genus/family/order top-10 score 42.3759.
- Held-out top-10:
  species/genus/family/order 9.9 / 17.3 / 47.0 / 69.3%.
- Comparison:
  multi-positive improves held-out species over taxonomy-soft retrieval-best
  8.5 -> 9.9%, but it does not beat taxonomy-hard species/genus 10.3 / 20.4
  and it loses the strongest family/order bridge result 55.4 / 75.4.
- Interpretation:
  same-species multi-positive targets reduce false-negative pressure, but the
  current objective trades away high-rank transfer. Keep it as an objective
  diagnostic, not the best bridge.

Exp 85 done: check. Built the first 16S reference layer for 12S/16S MarkerMirror.

- Script added:
  `scripts/edna/build_16s_reference_from_ncbi.py`.
- Dry-run source query:
  explicit NCBI nuccore Actinopterygii mitochondrial 16S/rrnL query returned
  40,034 candidate records.
- Bounded fetch:
  5,000 accessions fetched, producing 4,673 usable records and 1,865 species.
- Output root:
  `data/edna/stalder_inputs/16s_multisource/`.
- Overlap:
  502 species overlap existing 12S; 607 overlap COI; 319 overlap 12S+COI.
- Taxonomy enrichment:
  `scripts/edna/enrich_16s_reference_taxonomy.py` filled family/order for
  1,133 of 1,865 16S species from existing 12S/COI candidate tables.
- Scope decision:
  near-term eDNA marker expansion is 12S+16S. CytB/18S/non-fish markers are
  out of scope unless 12S/16S coverage fails.

Exp 86 done: check. First 12S-to-16S MarkerMirror bridge completed and copied.

- Remote run:
  `nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best`.
- Copied root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/nt_v2_50m_12s_to_16s_taxonomy_soft_retrieval_best/`.
- Data:
  502 overlap species, split into 351 train / 75 validation / 76 held-out test.
- Objective:
  frozen Nucleotide Transformer embeddings, taxonomy-hard batches,
  taxonomy-soft rank targets, retrieval-selected checkpointing.
- Training:
  retrieval-selected checkpoint restored epoch 120 with validation
  genus/family/order top-10 score 62.5140.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`
  now has 612 rows across 9 MarkerMirror runs.
- Held-out 12S query to 16S prototype top-10:
  species/genus/family/order 33.9 / 45.4 / 67.1 / 73.4%.
- Frozen NT baseline on the same split:
  species/genus/family/order 18.4 / 23.0 / 34.9 / 49.7%.
- Interpretation:
  learned 12S->16S alignment is much cleaner at species/genus than the current
  12S->COI bridge. This supports the scope decision that 12S+16S is enough for
  the current eDNA marker expansion, with COI retained as a barcode/tree anchor
  and comparator.

Exp 87 done: check. Reverse 16S-to-12S MarkerMirror bridge completed and copied.

- Goal:
  determine whether the ribosomal bridge works both ways or whether the strong
  12S->16S result is direction-specific.
- Remote/local run:
  `nt_v2_50m_16s_to_12s_taxonomy_soft_retrieval_best`.
- Data:
  same 502 overlap species, split into 351 train / 75 validation / 76 held-out
  test.
- Training:
  best retrieval checkpoint was epoch 10 with validation genus/family/order
  top-10 score 70.8920.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_retrieval_metrics.csv`
  now has 684 rows across 10 MarkerMirror runs.
- Held-out 16S query to 12S prototype top-10:
  species/genus/family/order 44.4 / 54.8 / 70.4 / 76.3%.
- Frozen NT baseline on the same split:
  species/genus/family/order 11.9 / 17.8 / 32.6 / 48.2%.
- Interpretation:
  this is the strongest MarkerMirror result so far. It makes the 12S/16S
  bridge genuinely promising, but still as candidate evidence rather than a
  final calibrated assignment system. Next step: shared 12S/16S species-space
  prototype plus seed/curation checks.

Exp 88 done: check. First shared 12S/16S species-space MarkerMirror prototype
completed and copied.

- Script added:
  `scripts/edna/train_marker_mirror_shared_space.py`.
- Remote/local run:
  `nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_shared_retrieval_metrics.csv`
  with 144 rows.
- Data:
  same 502 overlap species, split into 351 train / 75 validation / 76 held-out
  test.
- Model:
  one shared projection head for both 12S and 16S on top of frozen Nucleotide
  Transformer embeddings.
- Training:
  taxonomy-hard batches, taxonomy-soft targets, retrieval-selected checkpoint.
  Best checkpoint was epoch 60 with combined validation genus/family/order
  top-10 score 81.7221 across both directions.
- Held-out shared 12S->16S top-10:
  species/genus/family/order 42.1 / 50.0 / 68.5 / 81.5%.
- Held-out shared 16S->12S top-10:
  species/genus/family/order 64.3 / 71.3 / 78.3 / 85.3%.
- Interpretation:
  this is now the strongest MarkerMirror result. It improves over separate
  directional bridges in both directions. It is still candidate retrieval, not
  final species identification, so the next steps are seed repeats and
  integration into candidate reranking/rank-no-call calibration.

Exp 89 done: check. Shared 12S/16S species-space seed repeats completed and
were copied locally.

- Remote/local runs:
  `nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best_seed1902`
  and
  `nt_v2_50m_12s_16s_shared_space_taxonomy_soft_retrieval_best_seed1903`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_shared_retrieval_metrics.csv`
  now has 432 rows across 3 shared-space runs.
- Held-out 12S->16S top-10 mean species/genus/family/order:
  43.4 / 50.7 / 68.1 / 77.6%.
- Held-out 12S->16S top-10 range:
  species 42.1-45.4, genus 49.0-53.0, family 65.6-70.2, order 72.9-81.5%.
- Held-out 16S->12S top-10 mean species/genus/family/order:
  66.4 / 73.9 / 81.9 / 86.4%.
- Held-out 16S->12S top-10 range:
  species 63.3-71.5, genus 71.3-78.5, family 78.3-86.8, order 84.2-89.6%.
- Interpretation:
  the 12S/16S shared-space signal is stable across seeds and remains the lead
  MarkerMirror result. This clears the bar for downstream candidate reranking
  and rank/no-call integration.

Exp 90 done: check. Tri-marker 12S/16S/COI shared-space prototype completed
and was copied/scored.

- Script:
  `scripts/edna/train_marker_mirror_triad_space.py`.
- Remote run:
  `nt_v2_50m_12s_16s_coi_triad_shared_space_taxonomy_soft_retrieval_best`.
- Pair overlaps:
  12S/16S total/train/val/test 502 / 364 / 66 / 72;
  12S/COI 963 / 669 / 164 / 130;
  16S/COI 607 / 424 / 95 / 88.
- Goal:
  test whether adding 16S to a shared marker space improves 12S->COI/tree
  candidate retrieval compared with the earlier direct 12S->COI bridge.
- Result:
  the triad improved every direction over frozen NT but did not beat the best
  direct 12S->COI bridge. Keep the shared 12S/16S space as the lead
  MarkerMirror result and use COI as a downstream barcode/tree anchor.

Exp 91 done: check. Shared 12S/16S seed-repeat full-reference candidate exports
completed for the original run and seed 1902.

- Remote queue:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_seed_repeats/`.
- Copied root:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/`.
- Local candidate tables:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1901/marker_mirror_candidate_rankings.csv.gz`
  and
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1902/marker_mirror_candidate_rankings.csv.gz`.
- Each export produced 294,700 candidate rows.

Exp 92 done: check. Integrated MarkerMirror evidence joins and rank/no-call
calibrators completed across seeds 1901, 1902, and 1903.

- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_evidence_join_seed_repeat_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_best_target099.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_integrated_rank_seed_repeat_target099_stability.csv`.
- Target-0.99 learned MarkerMirror stability:
  - 12S->16S: 51.0% mean coverage, 98.9% mean assigned precision, 0.11% mean
    false species-call rate, 99.8% mean species precision.
  - 16S->12S: 71.1% mean coverage, 98.7% mean assigned precision, 0.47% mean
    false species-call rate, 99.3% mean species precision.
- Caveat:
  current ambiguity features include exact-sequence ambiguity and top-k
  consensus, not yet a full near-exact 12S/16S resolvability map.

Exp 93 done: check. Built explicit 12S/16S MarkerMirror resolvability features.

- Script:
  `scripts/edna/build_marker_mirror_marker_resolvability.py`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/marker_resolvability/`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_marker_resolvability_by_species.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_marker_resolvability_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_marker_resolvability_backend.csv`.
- 0.99 proxy identity result:
  - 12S: 91.8% species-resolvable, 98.2% genus-or-better, 99.8%
    family-or-better.
  - 16S: 92.0% species-resolvable, 98.6% genus-or-better, 99.8%
    family-or-better.
- Caveat:
  0.99 rows use a rare-kmer prefix-identity proxy, not VSEARCH/edlib.

Exp 94 done: check. Wired resolvability features into the MarkerMirror
evidence compiler and reran seed1903 calibration.

- Compiler hook:
  `scripts/edna/build_marker_mirror_evidence_join.py --marker-resolvability-table`.
- Output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/candidate_rankings_shared_seed1903/evidence_join_resolvability/`.
- Source tables:
  `marker_mirror_evidence_join_resolvability_seed1903_summary.csv`;
  `marker_mirror_resolvability_calibrator_seed1903_summary.csv`;
  `marker_mirror_resolvability_calibrator_seed1903_best_target099.csv`;
  `marker_mirror_production_handoff_next_actions.csv`.
- Result:
  adding 0.99 resolvability features did not change the best seed1903
  target-0.99 rows. This is not a negative result: it means ambiguity is now
  explicit/auditable, while performance remains stable.

Exp 95 done: check. Added the MarkerMirror production-handoff research
candidate-generator.

- Script:
  `scripts/edna/run_marker_mirror_candidate_generator.py`.
- Smoke output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_smoke_12s_to_16s/`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_smoke_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_smoke_manifest_summary.csv`.
- Smoke test:
  one 12S query, 25 16S target species, top-5 candidates, CPU.
- Result:
  candidate rows, summary, and manifest were written successfully after fixing
  checkpoint-argument manifest serialization.
- Caveat:
  this is candidate generation only. The next step is full-reference GPU/cached
  inference and connection to the evidence compiler/rank-no-call layer.

Exp 96 done: check. Ran the MarkerMirror full-reference GPU candidate-generator
smoke on Vast.

- Vast endpoint:
  `ssh -p 23156 root@194.14.47.19`.
- Script:
  `scripts/edna/run_marker_mirror_candidate_generator.py`.
- Checkpoint:
  shared 12S/16S seed1903 MarkerMirror projection head.
- Output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`.
- Archived copy:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_fullref_smoke_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_fullref_smoke_manifest_summary.csv`.
- Run:
  32 held-out 12S queries, full 1,865-species 16S reference, top-50 candidates.
- Result:
  1,600 candidate rows. Known-target top-50 recovery was
  25.0 / 59.4 / 78.1 / 84.4% for species/genus/family/order.
- Caveat:
  this proves executable candidate generation against the full 16S reference.
  It is not a final assignment system until connected to the evidence
  compiler/rank-no-call layer.

Exp 97 done: check. Added and tested cache-backed MarkerMirror target-reference
embeddings.

- Code:
  `scripts/edna/run_marker_mirror_candidate_generator.py` now supports
  `--target-embedding-cache` and `--refresh-target-embedding-cache`.
- Cache:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/cache/marker_mirror_16s_nt_v2_50m_fullref_embeddings.npz`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_cache_smoke_summary.csv`.
- Result:
  the cache-write pass embedded the 2,971-row 16S reference and wrote the cache;
  the cache-read pass loaded it and produced an exactly identical 1,600-row
  candidate table.
- Caveat:
  this is implementation hardening. It improves repeat inference mechanics but
  does not change the biological candidate-generation metric.

Exp 98 done: check. Added the MarkerMirror candidate-generator evidence handoff.

- Script:
  `scripts/edna/build_marker_mirror_candidate_generator_handoff.py`.
- Input:
  cache-backed full-reference candidate-generator output from
  `production_handoff_fullref_cache_read_12s_to_16s`.
- Output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_cache_read_12s_to_16s/evidence_handoff/`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_evidence_handoff_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_evidence_handoff_manifest_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_evidence_handoff_feature_inventory.csv`.
- Result:
  1,600 evidence rows for 32 queries, 97 production numeric features, tree
  evidence enabled, 3,502 marker-resolvability rows available. Top-1 candidates
  had same-marker 12S reference evidence for 40.6% of queries.
- Caveat:
  labels are evaluation diagnostics only. This is rank/no-call-ready input, not
  final production assignment. The next step is applying/serializing the
  integrated MarkerMirror rank/no-call calibrator.

Exp 99 done: check. Ran full-query MarkerMirror handoff and integrated
rank/no-call apply diagnostics.

- Full candidate-generator run:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_all_queries_12s_to_16s/`.
- Query/reference setup:
  3,566 12S query rows, cached full 1,865-species 16S reference, top-50
  candidates.
- Candidate/evidence rows:
  178,300.
- Full-query known-target top-50 recovery:
  9.5 / 39.9 / 59.8 / 76.3% for species/genus/family/order.
- New apply script:
  `scripts/edna/apply_marker_mirror_integrated_rank_calibrator.py`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_handoff_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_evidence_handoff_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_rank_apply_summary.csv`.
- Rank/no-call apply diagnostics:
  - logistic target-0.99, species enabled:
    7.0% coverage, 85.5% assigned precision, 20 species calls, 0.56% false
    species-call rate;
  - logistic target-0.99, species disabled:
    6.5% coverage, 93.1% assigned precision, 0 false species calls;
  - HGB target-0.99, species disabled:
    4.7% coverage, 71.1% assigned precision.
- Caveat:
  this is an important pipeline milestone, but not a final production operating
  point. Calibration does not transfer to the full-query production-style table
  at the nominal 0.99 target. Species calls should remain disabled for
  MarkerMirror assignment until independently validated.

Exp 100 done: check. Added MarkerMirror calibration-transfer diagnostics.

- New script:
  `scripts/edna/build_marker_mirror_calibration_transfer_diagnostics.py`.
- Input:
  seed1903 controlled MarkerMirror evidence table, full-query production-style
  handoff evidence table, and the species-disabled logistic rank/no-call apply
  assignments.
- Output directory:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_all_queries_12s_to_16s/calibration_transfer_diagnostics/`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_calibration_transfer_cohort_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_calibration_transfer_handoff_strata.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_calibration_transfer_feature_drift.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_calibration_transfer_top_feature_drift.csv`.
- Key diagnosis:
  the controlled validation split has 100.0% query-species coverage in the 16S
  target reference and top-50 recovery of 47.6 / 60.8 / 74.3 / 81.1% for
  species/genus/family/order. The full production-style handoff has only 26.6%
  query-species coverage in the 16S target reference and top-50 recovery of
  9.5 / 39.9 / 59.8 / 76.3%.
- Stratified result:
  when the query species is present in the 16S target reference, top-50 species
  recovery is 35.8%; when absent, species recovery is 0.0%, while genus/family/
  order remain useful.
- Interpretation:
  MarkerMirror's current bottleneck is reference-aware calibration and
  rank/no-call policy. This strengthens the "deepest defensible rank" story and
  explains why species calls should remain disabled for production-style
  MarkerMirror assignment.

Exp 101 done: check. Added a reference-aware MarkerMirror policy repair
diagnostic.

- New script:
  `scripts/edna/build_marker_mirror_reference_aware_policy.py`.
- Input:
  full-query production-style handoff evidence table plus the species-disabled
  logistic rank/no-call assignments.
- Output directory:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_all_queries_12s_to_16s/reference_aware_policy/`.
- Source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_aware_policy_summary.csv`.
- Baseline:
  species-disabled logistic apply gives 6.48% coverage at 93.07% assigned
  precision and 0 false species calls.
- Best higher-coverage production-safe repair:
  top-1 MarkerMirror score >= 0.620484 gives 5.83% coverage at 95.67% assigned
  precision and 0 false species calls.
- Strict production-safe repair:
  top-1 MarkerMirror score >= 0.697663 gives 3.25% coverage at 100.00%
  assigned precision and 0 false species calls.
- Caveat:
  these gates were evaluated on the labelled handoff table. They show the
  direction for a reference-aware abstention policy but are not independent
  production thresholds yet.

Exp 102 done: check. Added independent validation for the reference-aware
MarkerMirror abstention gates.

- New script:
  `scripts/edna/build_marker_mirror_reference_aware_policy_validation.py`.
- Method:
  choose production-safe gates on calibration query species, then evaluate the
  locked gate on held-out query species across 50 repeats. Also run source-
  holdout checks when enough assignments exist.
- Output directory:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_all_queries_12s_to_16s/reference_aware_policy_validation/`.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_aware_policy_validation_summary.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_aware_policy_validation_per_split.csv`.
- Species-split target-0.95:
  mean held-out coverage 5.79%, mean assigned precision 94.39%, 5th-95th
  percentile precision 86.59-100.00%, target met in 48% of repeats.
- Species-split target-0.99:
  mean held-out coverage 4.13%, mean assigned precision 98.27%, 5th-95th
  percentile precision 90.97-100.00%, target met in 70% of repeats.
- Source-holdout:
  target-0.99 is reasonable for MitoHelper/rCRUX and exact on the tiny Mare-MAGE
  heldout, but target-0.95 is unstable on Mare-MAGE.
- Interpretation:
  reference-aware abstention is promising and moves in the right direction, but
  the current full-query MarkerMirror assignment policy is not a locked
  production threshold. The next scientific fix is either richer cohort-aware
  calibration or a union candidate generator that raises candidate support.

Exp 103 done: check. Added a MarkerMirror plus same-marker union
candidate-support audit.

- New script:
  `scripts/edna/build_marker_mirror_union_candidate_support.py`.
- Inputs:
  the full-query MarkerMirror 12S->16S evidence handoff table,
  `data/edna/stalder_inputs/multisource/zero_shot_queries.csv`, and the
  multisource 12S and 16S reference sequence tables.
- Output source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_candidate_support_summary.csv`;
  `marker_mirror_union_candidate_support_per_query.csv`;
  `marker_mirror_same_marker_kmer_candidates_top50.csv.gz`;
  `marker_mirror_union_candidate_support_manifest.json`.
- Result:
  MarkerMirror-only full-query top50 support is 9.5 / 39.9 / 59.8 / 76.3% for
  species/genus/family/order. Same-marker 12S k-mer top50 support is 0.0 /
  89.5 / 94.9 / 99.5%. The union top50 support is 9.5 / 91.7 / 95.1 / 99.6%.
- Interpretation:
  same-marker 12S evidence cannot restore species calls in this zero-shot table
  because the query species are absent from the current 12S reference sequence
  table, and only 26.6% are present in the 16S reference. It can provide strong
  genus/family/order support. This supports the production design: union
  candidate generation, then evidence compilation, then calibrated rank/no-call.
- Caveat:
  the same-marker arm is a TF-IDF character-kmer audit, not final BLAST/VSEARCH
  alignment evidence.

Exp 104 done: check. Converted the MarkerMirror union candidate audit into a
production-style union candidate table plus first rank/no-call diagnostics.

- New script:
  `scripts/edna/build_marker_mirror_union_rank_policy.py`.
- Production-style output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_candidate_rank_policy/marker_mirror_union_production_candidates.csv.gz`.
  This table has 355,231 candidate rows and does not require hidden labels.
- Source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_static_policy_summary.csv`;
  `marker_mirror_union_score_gate_validation_summary.csv`;
  `marker_mirror_union_score_gate_validation_per_split.csv`;
  `marker_mirror_union_rank_policy_manifest.json`.
- Static source-agreement result:
  top-1 MarkerMirror and same-marker agreement at family/order only assigns
  25.2% of queries at 98.4% assigned precision with 0 species calls.
- Score-gate validation:
  repeated species-split validation shows high-rank signal but not locked
  thresholds. Family target-0.95 averages 92.7% coverage at 94.8% precision.
  Order target-0.99 averages 68.5% coverage at 99.0% precision.
- Interpretation:
  the union path is now a real candidate/evidence handoff, not just a support
  audit. The best immediate result is conservative family/order calling from
  independent-source agreement. The next scientific step is a calibrated
  evidence compiler over the union features; species remains reference-limited.

Exp 105 done: check. Trained the first learned MarkerMirror union evidence
compiler.

- New script:
  `scripts/edna/train_marker_mirror_union_evidence_compiler.py`.
- Method:
  HGB classifiers over 102 production-available top-1 union features. Labels
  are used only for train/calibration/evaluation. Each repeat splits by query
  species into train, calibration, and evaluation cohorts.
- Output roots:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_evidence_compiler/`;
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_evidence_compiler_order_only/`.
- Source tables:
  `marker_mirror_union_evidence_compiler_summary.csv`;
  `marker_mirror_union_evidence_compiler_family_order_summary.csv`;
  `marker_mirror_union_evidence_compiler_order_summary.csv`;
  `marker_mirror_union_evidence_compiler_features.csv`.
- Mixed family/order result:
  target-0.99 averages 72.2% coverage at 97.9% precision and meets target in
  20% of species-split repeats.
- Order-only result:
  target-0.99 averages 67.4% coverage at 98.5% precision and meets target in
  44% of species-split repeats.
- Interpretation:
  this learned compiler does not beat the simpler diagnostics. It is a useful
  negative result: the current feature/model setup does not solve calibration
  transfer. The best clean union result remains family/order source agreement
  at 25.2% coverage and 98.4% precision; the high-coverage order signal remains
  the simple same-marker score gate.

Exp 106 done: check. Added MarkerMirror union reason codes and reference-
curation priorities.

- New script:
  `scripts/edna/build_marker_mirror_union_reason_codes.py`.
- Commands checked:
  `python3 -m py_compile scripts/edna/build_marker_mirror_union_reason_codes.py`;
  `python3 scripts/edna/build_marker_mirror_union_reason_codes.py --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_union_reason_codes.log`.
- Source tables:
  `marker_mirror_union_reason_code_summary.csv`;
  `marker_mirror_union_reason_code_by_source.csv`;
  `marker_mirror_union_reason_code_per_query.csv`;
  `marker_mirror_union_reference_curation_priorities.csv`;
  `marker_mirror_union_reason_code_manifest.json`.
- Result:
  2,249/3,566 full-query rows have genus-level union support without a species
  call. Conservative source agreement emits 621 family calls at 98.1%
  precision and 277 order calls at 99.3% precision, with species disabled.
- Reference curation:
  top priorities include `Epinephelus_coioides`, `Gadus_morhua`,
  `Pareiorhaphis_hystrix`, and `Acrossocheilus_paradoxus`, where high-rank
  support is strong but current marker-reference coverage blocks species-level
  calls. `Trichiurus_lepturus` is a different failure mode: it is present in
  16S but poorly recovered at species level, so it is a retrieval/model target.
- Caveat:
  current 12S reference-gap labels are current-table/split-design diagnostics,
  not final real-world absence statements. The same-marker arm remains a k-mer
  audit until BLAST/VSEARCH/edlib evidence is wired in.

Exp 107 done: check. Validated the same-marker 12S candidate pool with edlib
alignment scoring.

- New script:
  `scripts/edna/build_marker_mirror_same_marker_edlib_validation.py`.
- Remote:
  Vast `ssh -p 23156 root@194.14.47.19`; installed `edlib` with system
  Python; smoke tested 10 queries; full run scored 176,931 candidate rows.
- Copied outputs:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_edlib_validation/edlib_same_marker_validation/`.
- Source tables:
  `marker_mirror_same_marker_edlib_support_summary.csv`;
  `marker_mirror_same_marker_edlib_support_per_query.csv`;
  `marker_mirror_same_marker_edlib_candidates_top50.csv.gz`;
  `marker_mirror_same_marker_edlib_validation_manifest.json`.
- Result:
  edlib reranking preserves the high-rank signal. Top10 edlib-reranked support
  is 0.0 / 87.8 / 94.3 / 98.8% for species/genus/family/order, compared with
  0.0 / 86.9 / 93.9 / 98.7% for the original k-mer order.
- Caveat:
  this validates/reranks the existing k-mer top50 pool. It is not full all-vs-
  all BLAST/VSEARCH candidate generation.

Exp 108 done: check. Trained a list-level selective compiler over union
candidate lists.

- New script:
  `scripts/edna/train_marker_mirror_union_listwise_selective_compiler.py`.
- Method:
  HGB rank selectors over 223 production-available list features:
  MarkerMirror list concentration, same-marker k-mer list concentration,
  edlib-reranked list concentration, source agreement, score margins, and
  edlib identity features. Validation uses repeated query-species
  train/calibration/evaluation splits.
- Runs:
  family/order 50 repeats under
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler/`;
  order-only 50 repeats under
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/union_listwise_selective_compiler_order_only/`.
- Source tables:
  `marker_mirror_union_listwise_selective_family_order_summary.csv`;
  `marker_mirror_union_listwise_selective_order_summary.csv`;
  matching per-split, threshold, and feature tables.
- Result:
  family/order target-0.99 averages 83.8% coverage at 98.2% precision and
  meets target in 26% of repeats. Order-only target-0.99 averages 83.1%
  coverage at 98.8% precision and meets target in 56% of repeats.
- Interpretation:
  list-level evidence improves the high-coverage order diagnostic over the
  earlier top-1 HGB, but it still does not lock target-0.99. The production-safe
  layer remains conservative source agreement plus reason codes.

Exp 109 done: check. Ran VSEARCH same-marker 12S candidate generation and
refreshed MarkerMirror + VSEARCH union support.

- New scripts:
  `scripts/edna/build_marker_mirror_same_marker_vsearch_candidates.py`;
  `scripts/edna/build_marker_mirror_union_vsearch_candidate_support.py`.
- Remote:
  installed VSEARCH 2.27.0 on Vast `ssh -p 23156 root@194.14.47.19`; smoke
  tested 10 queries; full run used 64 threads over 3,566 query sequences and
  12,593 current 12S reference sequences.
- Copied outputs:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_vsearch_same_marker/vsearch_same_marker_full/`.
- Source tables:
  `marker_mirror_same_marker_vsearch_support_summary.csv`;
  `marker_mirror_same_marker_vsearch_support_per_query.csv`;
  `marker_mirror_same_marker_vsearch_candidates_top50.csv.gz`;
  `marker_mirror_union_vsearch_candidate_support_summary.csv`;
  `marker_mirror_union_vsearch_candidate_support_per_query.csv`;
  matching manifests.
- Result:
  same-marker VSEARCH top50 support is 0.0 / 90.4 / 94.9 / 99.4% for
  species/genus/family/order. MarkerMirror + VSEARCH top50 union support is
  9.5 / 91.8 / 95.1 / 99.6%.
- Interpretation:
  VSEARCH global alignment preserves the high-rank same-marker story and gives
  us a stronger claim-facing classical arm than the original k-mer audit. It is
  still not BLAST local alignment and cannot recover query species absent from
  the current 12S reference table.

Exp 110 done: check. Ran BLASTN same-marker 12S candidate generation and
refreshed MarkerMirror + BLASTN union support.

- New scripts:
  `scripts/edna/build_marker_mirror_same_marker_blast_candidates.py`;
  `scripts/edna/build_marker_mirror_union_blast_candidate_support.py`.
- Remote:
  installed NCBI BLAST+ 2.12.0 on Vast `ssh -p 23156 root@194.14.47.19`;
  smoke tested 10 queries; full run used 64 threads over 3,566 query sequences
  and 12,593 current 12S reference sequences.
- Copied outputs:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_blast_same_marker/blast_same_marker_full/`.
- Source tables:
  `marker_mirror_same_marker_blast_support_summary.csv`;
  `marker_mirror_same_marker_blast_support_per_query.csv`;
  `marker_mirror_same_marker_blast_candidates_top50.csv.gz`;
  `marker_mirror_union_blast_candidate_support_summary.csv`;
  `marker_mirror_union_blast_candidate_support_per_query.csv`;
  matching manifests.
- Result:
  same-marker BLASTN top50 support is 0.0 / 90.7 / 95.1 / 99.4% for
  species/genus/family/order. MarkerMirror + BLASTN top50 union support is
  9.5 / 92.1 / 95.3 / 99.7%.
- Interpretation:
  BLASTN local alignment independently confirms the high-rank same-marker
  story. Species remains unchanged because the current same-marker reference
  table lacks the held-out query species by split design.

Exp 111 done: check. Built BLAST/VSEARCH-backed calibration-transfer repair
diagnostics.

- New script:
  `scripts/edna/build_marker_mirror_blast_vsearch_calibration_repair.py`.
- Commands checked:
  `python3 -m py_compile scripts/edna/build_marker_mirror_blast_vsearch_calibration_repair.py`;
  `python3 scripts/edna/build_marker_mirror_blast_vsearch_calibration_repair.py --repeats 50 --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_blast_vsearch_calibration_repair.log`.
- Source tables:
  `marker_mirror_blast_vsearch_calibration_repair_features.csv`;
  `marker_mirror_blast_vsearch_calibration_repair_policy_rows.csv.gz`;
  `marker_mirror_blast_vsearch_calibration_repair_summary.csv`;
  `marker_mirror_blast_vsearch_calibration_repair_per_split.csv`;
  `marker_mirror_blast_vsearch_calibration_repair_thresholds.csv`;
  `marker_mirror_blast_vsearch_calibration_repair_manifest.json`.
- Result:
  all-source top1 order agreement gives the stable conservative target-0.99
  repair: 24.8% mean coverage, 99.6% mean precision, target met in 100% of 50
  query-species repeats. Higher-coverage order rows are promising but not
  locked: BLAST top10 source-stratified Wilson95 gives 69.0% coverage at 99.4%
  mean precision and target met in 82% of repeats.
- Interpretation:
  this repairs the conservative order-call layer but does not yet solve
  high-coverage rank/no-call calibration transfer.

Exp 112 done: check. Built explicit stable order policy assignment/reason-code
tables from the Exp 111 all-source agreement result.

- New script:
  `scripts/edna/build_marker_mirror_stable_order_policy.py`.
- Commands checked:
  `python3 -m py_compile scripts/edna/build_marker_mirror_stable_order_policy.py`;
  `python3 scripts/edna/build_marker_mirror_stable_order_policy.py`.
- Source tables:
  `marker_mirror_stable_order_policy_assignments.csv`;
  `marker_mirror_stable_order_policy_summary.csv`;
  `marker_mirror_stable_order_policy_by_source.csv`;
  `marker_mirror_stable_order_policy_reason_counts.csv`;
  `marker_mirror_stable_order_policy_manifest.json`.
- Result:
  unthresholded all-source top1 order agreement assigns 886/3,566 full-query
  12S rows, 24.8% coverage, at 99.7% precision with 0 false species calls. The
  conservative max-repeat target-0.99 threshold assigns 880/3,566 rows, 24.7%
  coverage, at 99.7% precision with 0 false species calls.
- Interpretation:
  this gives the current production-safe MarkerMirror handoff: order/no-call
  only, with reason codes. It does not solve species identification or the
  higher-coverage rank/no-call calibration problem.

Exp 113 done: check. Added the label-stripped production handoff payload for
the stable MarkerMirror/BLASTN/VSEARCH order policy.

- Updated script:
  `scripts/edna/build_marker_mirror_stable_order_policy.py`.
- New source table:
  `marker_mirror_stable_order_policy_production_assignments.csv`.
- Result:
  the production payload keeps query id, source, decision mode, assigned
  rank/label, confidence, threshold, top-1 order evidence from MarkerMirror,
  BLASTN, and VSEARCH, and a reason code. It excludes truth labels and
  correctness columns.
- Remaining gap:
  arbitrary 12S FASTA orchestration is not yet one command. The intended chain
  is MarkerMirror candidate generation -> BLASTN/VSEARCH same-marker candidate
  generation -> shared source feature table -> stable order/no-call policy.

Exp 114 done: check. Added and smoked the dependency-gated one-command
MarkerMirror 12S production-v1 wrapper.

- New script:
  `scripts/edna/run_marker_mirror_12s_production_v1.py`.
- Commands checked:
  `python3 -m py_compile scripts/edna/run_marker_mirror_12s_production_v1.py`;
  `python3 scripts/edna/run_marker_mirror_12s_production_v1.py --input data/edna/stalder_inputs/multisource/zero_shot_queries.csv --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke --limit 2 --dry-run`;
  `python3 scripts/edna/build_marker_mirror_same_marker_blast_candidates.py --query-table results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/input_queries/zero_shot_queries.csv --same-marker-reference-dir data/edna/stalder_inputs/multisource --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/blast_smoke --top-k 50 --threads 4 --log-file results/paper1_phylo_calibrated_assignment/logs/run_marker_mirror_12s_production_v1_blast_smoke.log`.
- Artifacts:
  `marker_mirror_12s_production_dependency_report.csv`;
  `marker_mirror_12s_production_plan.json`;
  `marker_mirror_12s_production_next_actions.csv`;
  `marker_mirror_12s_production_manifest.json`;
  `blast_smoke/marker_mirror_same_marker_blast_candidates_top50.csv.gz`.
- Result:
  the wrapper normalized 2 12S rows and wrote the full planned chain:
  MarkerMirror -> BLASTN -> VSEARCH -> feature table -> stable order/no-call.
  Local BLASTN and makeblastdb are available; VSEARCH is missing. BLASTN smoke
  completed and produced 100 top-50 candidate rows.
- Interpretation:
  the one-command 12S production path is now real but dependency-gated locally.
  Run on Vast or install VSEARCH locally to execute the full all-source
  order/no-call chain.

Exp 115 done: check. Ran the full one-command MarkerMirror 12S production-v1
wrapper on Vast for all current 12S query rows.

- Command:
  `python3 -u scripts/edna/run_marker_mirror_12s_production_v1.py --input data/edna/stalder_inputs/multisource/zero_shot_queries.csv --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/vast_full_all_queries_20260603 --device cuda --threads 32`.
- Copied local output:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/`.
- Stages completed:
  MarkerMirror 12S->16S candidate generation, BLASTN same-marker candidates,
  VSEARCH same-marker candidates, all-source feature table, stable order/no-call
  policy.
- Result:
  3,566 queries; 880 order calls; 2,686 no-calls. Labelled diagnostic precision
  is 99.7% with 0 false species calls for the conservative max-repeat
  threshold.
- Runtime:
  15.0 s MarkerMirror, 254.6 s BLASTN, 48.8 s VSEARCH, 1.6 s stable policy.
- Interpretation:
  this is the first end-to-end executable MarkerMirror 12S order/no-call
  research pipeline. It is not species identification and not field-eDNA
  validation.

Exp 116 done: check. Ran the MarkerMirror 12S production-v1 wrapper on an
unlabeled FASTA input and documented the CLI handoff.

- Input:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/unlabeled_fasta_smoke_input/unlabeled_12s_queries.fa`.
- Vast command:
  `python3 -u scripts/edna/run_marker_mirror_12s_production_v1.py --input results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/unlabeled_fasta_smoke_input/unlabeled_12s_queries.fa --output-dir results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604 --device cuda --threads 16`.
- Copied local output:
  `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/`.
- Result:
  2 FASTA records; 1 order call; 1 no-call. Diagnostic precision/correctness
  fields are blank by design because no truth labels were supplied.
- CLI usage doc:
  `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md`.
- Interpretation:
  this verifies the executable 12S order/no-call path can run without hidden
  labels and still emit assignments plus reason codes. It is not an accuracy
  estimate, species identification, or field-eDNA validation.

Exp 117 done: check. Added and ran the nested high-coverage order repair
diagnostic for BLAST/VSEARCH-backed MarkerMirror evidence.

- New script:
  `scripts/edna/build_marker_mirror_high_coverage_order_repair.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_high_coverage_order_repair.py`.
- Source tables:
  `marker_mirror_high_coverage_order_repair_summary.csv`;
  `marker_mirror_high_coverage_order_repair_per_split.csv`;
  `marker_mirror_high_coverage_order_repair_thresholds.csv`;
  `marker_mirror_high_coverage_order_repair_locked_thresholds.csv`;
  `marker_mirror_high_coverage_order_repair_assignments.csv`;
  `marker_mirror_high_coverage_order_repair_assignment_summary.csv`;
  `marker_mirror_high_coverage_order_repair_manifest.json`.
- Nested species-split result:
  BLASTN/VSEARCH top-10 order agreement with nested global Wilson95 locking
  averages 57.2% held-out coverage at 99.8% assigned precision, meets target
  0.99 in 100% of 50 outer repeats, and has a minimum repeat precision of
  99.3%.
- Full-table locked diagnostic:
  2,513/3,566 rows receive an order call, 70.5% coverage, with 99.8%
  labelled precision. The remaining 1,053 rows no-call.
- Interpretation:
  this is a major improvement over the conservative 24.7% order coverage, but
  it is still order-level only. Exp 118 wires it into the wrapper as explicitly
  labelled `high_coverage_order` mode; it remains diagnostic/research mode, not
  the default.

Exp 118 done: check. Wired the Exp 117 high-coverage order policy into the 12S
wrapper as an explicit decision mode and smoked it on Vast.

- Updated script:
  `scripts/edna/run_marker_mirror_12s_production_v1.py`.
- New CLI behavior:
  `--decision-mode stable_order` remains default;
  `--decision-mode high_coverage_order` applies BLASTN/VSEARCH top-10 order
  agreement with the Exp 117 nested global Wilson95 threshold.
- Vast smoke outputs copied locally:
  - `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_stable_labelled_20260604/`;
  - `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_high_coverage_labelled_20260604/`;
  - `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_stable_unlabeled_20260604/`;
  - `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_smoke_high_coverage_unlabeled_20260604/`.
- Labelled smoke result:
  stable mode assigned 1/4 order calls at 100% diagnostic precision;
  high-coverage mode assigned 3/4 order calls at 100% diagnostic precision.
- Unlabeled FASTA smoke result:
  stable mode assigned 1/2 order calls; high-coverage mode assigned 2/2 order
  calls. Precision/correctness fields are blank by design.
- Interpretation:
  the high-coverage order mode is now executable, but it is still order/no-call
  only and should be presented as a research diagnostic mode rather than a
  species identifier or deployed field-eDNA production tool.

Exp 119 done: check. Tested whether the high-coverage repair can safely move
from order to family or genus.

- Updated script:
  `scripts/edna/build_marker_mirror_high_coverage_order_repair.py` now accepts
  `--rank genus|family|order` and defaults to the prior order behavior.
- Commands:
  `python3 scripts/edna/build_marker_mirror_high_coverage_order_repair.py --rank family --output-prefix marker_mirror_high_coverage_family_repair --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_high_coverage_family_repair.log`;
  `python3 scripts/edna/build_marker_mirror_high_coverage_order_repair.py --rank genus --output-prefix marker_mirror_high_coverage_genus_repair --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_high_coverage_genus_repair.log`;
  then repeated both ranks across all eight BLAST/VSEARCH/MarkerMirror policy
  rows with `_all_policies` output prefixes.
- Source tables:
  `marker_mirror_high_coverage_rank_repair_comparison.csv`;
  `marker_mirror_high_coverage_family_repair_all_policies_summary.csv`;
  `marker_mirror_high_coverage_genus_repair_all_policies_summary.csv`.
- Result:
  no family or genus row met target-0.99 in 100% of 50 species-split repeats.
  Best family row was VSEARCH top-10 mode with nested global precision:
  35.5% mean coverage, 99.35% mean precision, target met in 94% of repeats.
  Best genus row was VSEARCH top-10 mode with nested global precision:
  7.8% mean coverage, 99.79% mean precision, target met in 98% of repeats.
- Interpretation:
  family/genus evidence is promising but not stable enough to enable. The
  12S wrapper should keep high-coverage mode order/no-call only.

Exp 120 done: check. Created the coauthor-facing MarkerMirror one-pager.

- New document:
  `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_COAUTHOR_ONE_PAGER.md`.
- Contents:
  one-sentence summary, pipeline, candidate-support table, stable and
  high-coverage order/no-call numbers, runtime, family/genus negative result,
  claim boundaries, and source-table pointers.
- Updated:
  `COAUTHOR_BRIEF.md` now points to the one-pager, and `SOURCE_TABLES.md`
  lists it as the concise MarkerMirror evidence package.

Exp 121 done: check. Tested hierarchical set-valued MarkerMirror candidate
output as a different family/genus strategy.

- New script:
  `scripts/edna/build_marker_mirror_hierarchical_candidate_sets.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_hierarchical_candidate_sets.py --log-file results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_hierarchical_candidate_sets.log`.
- Inputs:
  full copied 12S wrapper candidate lists from the Vast run:
  MarkerMirror 12S->16S, BLASTN same-marker top50, and VSEARCH same-marker
  top50.
- Source tables:
  `marker_mirror_hierarchical_candidate_sets_summary.csv`;
  `marker_mirror_hierarchical_candidate_sets_policy_grid_summary.csv`;
  `marker_mirror_hierarchical_candidate_sets_per_split.csv`;
  `marker_mirror_hierarchical_candidate_sets_assignments.csv.gz`;
  `marker_mirror_hierarchical_candidate_sets_manifest.json`.
- Result:
  set-valued output does not rescue family/genus at target 0.99. Best
  full-query family set coverage is 95.4% using all-source union top50, but the
  mean set size is 34.4 families. Best genus set coverage is 92.4% with mean set
  size 79.6 genera. The only stable target-0.99 set-valued rank remains order.
- Interpretation:
  the family/genus failure is evidence-level, not just a single-label threshold
  problem. The 12S MarkerMirror wrapper should stay order/no-call until new
  evidence is added.

Exp 122 done: check. Created the MarkerMirror manuscript-facing asset package.

- New script:
  `scripts/edna/build_marker_mirror_manuscript_assets.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_manuscript_assets.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_assets.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/`.
- Outputs:
  `marker_mirror_candidate_support_table.csv`;
  `marker_mirror_order_policy_table.csv`;
  `marker_mirror_rank_boundary_table.csv`;
  `marker_mirror_runtime_table.csv`;
  `marker_mirror_figure_plan.csv`;
  `marker_mirror_methods_blurb.md`;
  `marker_mirror_manuscript_asset_manifest.json`.
- Interpretation:
  these files are writing assets, not new benchmark rows. They package the
  MarkerMirror candidate-support, conservative/high-coverage order policy,
  family/genus/species claim boundary, runtime, figure plan, and methods text
  for manuscript drafting and coauthor review.

Exp 123 done: check. Rendered MarkerMirror manuscript figure drafts.

- New script:
  `scripts/edna/build_marker_mirror_manuscript_figures.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_manuscript_figures.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_figures.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/`.
- Outputs:
  `marker_mirror_candidate_support_bars.png` / `.pdf`;
  `marker_mirror_order_policy_tradeoff.png` / `.pdf`;
  `marker_mirror_rank_boundary.png` / `.pdf`;
  `marker_mirror_runtime_breakdown.png` / `.pdf`;
  `marker_mirror_slide_ready_summary.md`;
  `marker_mirror_manuscript_figure_manifest.json`.
- Sanity check:
  figure PNGs were rendered locally and the candidate-support plot was revised
  after visual inspection to avoid label overlap.
- Interpretation:
  these are draft manuscript/coauthor visuals rendered from Exp 122 source
  assets. They are not new metrics and do not change the claim boundary.

Exp 124 done: check. Created slide-ready MarkerMirror tables and a five-slide
outline.

- New script:
  `scripts/edna/build_marker_mirror_slide_tables.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_slide_tables.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_slide_tables.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/`.
- Outputs:
  `marker_mirror_candidate_support_slide_table.csv` / `.md`;
  `marker_mirror_order_policy_slide_table.csv` / `.md`;
  `marker_mirror_rank_boundary_slide_table.csv` / `.md`;
  `marker_mirror_runtime_slide_table.csv` / `.md`;
  `marker_mirror_slide_package_outline.md`;
  `marker_mirror_slide_tables_manifest.json`.
- Interpretation:
  these files package the MarkerMirror story into slide-ready tables and a
  concise five-slide outline for coauthor review or later deck assembly. They
  are not new metrics.

Exp 125 done: check. Created MarkerMirror manuscript captions and text
snippets.

- New script:
  `scripts/edna/build_marker_mirror_manuscript_text.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_manuscript_text.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_text.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`.
- Outputs:
  `marker_mirror_figure_captions.md`;
  `marker_mirror_results_paragraph.md`;
  `marker_mirror_methods_paragraph.md`;
  `marker_mirror_claim_boundary_box.md`;
  `marker_mirror_caption_inventory.csv`;
  `marker_mirror_manuscript_text_manifest.json`.
- Interpretation:
  these files package figure captions, results text, methods text, and claim
  boundary language for manuscript drafting. They are generated from existing
  MarkerMirror source tables and do not add new metrics.

Exp 126 done: check. Created a MarkerMirror manuscript section outline and
checklist.

- New script:
  `scripts/edna/build_marker_mirror_manuscript_section_outline.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_manuscript_section_outline.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_manuscript_section_outline.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`.
- Outputs:
  `marker_mirror_manuscript_section_outline.md`;
  `marker_mirror_manuscript_section_checklist.csv`;
  `marker_mirror_manuscript_section_manifest.json`.
- Interpretation:
  these files organize the Exp 122-125 MarkerMirror assets into a proposed
  Paper 1 section structure with figure/text placement and claim-boundary
  reminders. They are manuscript planning artifacts, not new metrics.

Exp 127 done: check. Audited genuinely new evidence sources for future
MarkerMirror family/genus work.

- New script:
  `scripts/edna/build_marker_mirror_next_evidence_audit.py`.
- Command:
  `python3 scripts/edna/build_marker_mirror_next_evidence_audit.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_next_evidence_audit.log`.
- Outputs:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_next_evidence_source_audit.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_by_lineage.csv`;
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_next_evidence_source_manifest.json`;
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_family_genus_next_evidence_plan.md`.
- Findings:
  P0 next evidence sources are lineage-specific reference coverage,
  alignment-backed marker-resolvability, and active reference-curation/value of
  information. Geography/range and co-occurrence are available for sample-aware
  eDNA mode, but not arbitrary single-query FASTA without site/sample metadata.
- Interpretation:
  this is a planning/source-table audit only. It does not enable family/genus
  calls and it prevents the next work from repeating threshold-only repairs.

Exp 128 done: check. Tested lineage/reference-coverage features as a new
MarkerMirror family/genus evidence source.

- New script:
  `scripts/edna/build_marker_mirror_reference_coverage_policy_diagnostic.py`.
- Commands:
  `python3 -m py_compile scripts/edna/build_marker_mirror_reference_coverage_policy_diagnostic.py`;
  `python3 scripts/edna/build_marker_mirror_reference_coverage_policy_diagnostic.py`;
  `/tmp` two-repeat smoke after filtering inactive rank-specific feature
  columns.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_reference_coverage_policy_diagnostic.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/reference_coverage_policy_diagnostic/`.
- Source-table outputs:
  `marker_mirror_reference_coverage_policy_diagnostic_summary.csv`;
  `marker_mirror_reference_coverage_policy_diagnostic_per_split.csv`;
  `marker_mirror_reference_coverage_policy_diagnostic_thresholds.csv`;
  `marker_mirror_reference_coverage_policy_diagnostic_features.csv`;
  `marker_mirror_reference_coverage_policy_diagnostic_lineage_features.csv`;
  `marker_mirror_reference_coverage_policy_diagnostic_manifest.json`.
- Result:
  over 50 species-split repeats, no family/genus target-0.99 policy row
  transferred cleanly. Best target-0.99 genus row averaged 17.7% coverage at
  97.6% precision and met target in 42% of repeats. Best target-0.99 family
  row averaged 87.4% coverage at 98.0% precision and met target in 10% of
  repeats.
- Interpretation:
  lineage/reference coverage is useful diagnostic evidence, but it is not
  sufficient on its own to enable family/genus calls. The MarkerMirror 12S
  wrapper remains order/no-call only. The next family/genus attempt needs
  alignment-backed marker-resolvability, active reference-curation evidence, or
  sample-aware geography/co-occurrence rather than another threshold wrapper.

Exp 129 done: check. Ran VSEARCH-backed marker-resolvability for 12S and 16S.

- Script:
  `scripts/edna/build_12s_near_exact_resolvability.py`.
- Vast endpoint:
  `ssh -p 23156 root@194.14.47.19`.
- Remote output:
  `/workspace/marinemamba/results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/vsearch_marker_resolvability_20260604/`.
- Local copied output:
  `results/remote_runs/2026-06-04/rtx_pro_6000/marker_mirror_vsearch_resolvability_20260604/`.
- Source-table outputs:
  `marker_mirror_vsearch_resolvability_summary.csv`;
  `marker_mirror_vsearch_resolvability_cluster_rank_counts.csv`;
  `marker_mirror_vsearch_resolvability_query_oracle_rates.csv`;
  `marker_mirror_vsearch_resolvability_12s_summary.csv`;
  `marker_mirror_vsearch_resolvability_16s_summary.csv`;
  `marker_mirror_vsearch_resolvability_manifest.json`.
- Result:
  at 0.99 VSEARCH identity, 12S query oracle support is 77.9% species,
  95.2% genus, 99.6% family, and 99.7% order, while only 19.6% of query
  clusters contain a current reference. At 0.95 identity, species support falls
  to 38.3%, but order support remains 98.2%.
- 16S caveat:
  the current `16s_multisource` table has no `zero_shot_queries.csv`, so 16S
  rows are reference-cluster summaries only. At 0.99 identity, 16S reference
  clusters are mostly species-level: 1876/1988 clusters.
- Interpretation:
  this replaces the rare-kmer marker-resolvability proxy with an
  alignment-backed VSEARCH diagnostic. It supports marker-ceiling/source-table
  hardening but does not enable family/genus/species calls without a separately
  validated rank/no-call policy.

Exp 130 done: check. Tested a VSEARCH-resolvability-aware MarkerMirror policy
diagnostic.

- New script:
  `scripts/edna/build_marker_mirror_vsearch_resolvability_policy_diagnostic.py`.
- Commands:
  `python3 -m py_compile scripts/edna/build_marker_mirror_vsearch_resolvability_policy_diagnostic.py`;
  `python3 scripts/edna/build_marker_mirror_vsearch_resolvability_policy_diagnostic.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_vsearch_resolvability_policy_diagnostic.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/vsearch_resolvability_policy_diagnostic/`.
- Source-table outputs:
  `marker_mirror_vsearch_resolvability_policy_diagnostic_summary.csv`;
  `marker_mirror_vsearch_resolvability_policy_diagnostic_per_split.csv`;
  `marker_mirror_vsearch_resolvability_policy_diagnostic_thresholds.csv`;
  `marker_mirror_vsearch_resolvability_policy_diagnostic_features.csv`;
  `marker_mirror_vsearch_resolvability_policy_diagnostic_query_features.csv`;
  `marker_mirror_vsearch_resolvability_policy_diagnostic_manifest.json`.
- Feature boundary:
  hidden Exp 129 oracle-support columns were not used as model features. The
  diagnostic uses production-available cluster features such as reference
  presence and reference/query record counts.
- Result:
  over 50 species-split repeats, no family/genus target-0.99 row transferred
  cleanly. Family target-0.99 averaged 57.3% coverage at 95.5% precision and
  met target in 44% of repeats. Genus target-0.99 averaged 11.5% coverage at
  87.8% precision and met target in 38% of repeats.
- Interpretation:
  VSEARCH marker-resolvability is useful marker-ceiling evidence, but adding
  production-available VSEARCH cluster features to a learned policy still does
  not justify family/genus output. The MarkerMirror 12S wrapper remains
  order/no-call only.

Exp 131 done: check. Built active reference-curation/value-of-information
tables for MarkerMirror.

- New script:
  `scripts/edna/build_marker_mirror_active_reference_value.py`.
- Commands:
  `python3 -m py_compile scripts/edna/build_marker_mirror_active_reference_value.py`;
  `python3 scripts/edna/build_marker_mirror_active_reference_value.py`.
- Log:
  `results/paper1_phylo_calibrated_assignment/logs/build_marker_mirror_active_reference_value.log`.
- Output root:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/active_reference_value/`.
- Source-table outputs:
  `marker_mirror_active_reference_value_species.csv`;
  `marker_mirror_active_reference_value_lineage.csv`;
  `marker_mirror_active_reference_value_actions.csv`;
  `marker_mirror_active_reference_value_manifest.json`.
- Result:
  the table ranks 795 species groups and 698 lineage/action rows by which
  reference or evidence addition would most plausibly improve MarkerMirror's
  current no-call/high-rank-only behavior. The largest action category is
  `add_12s_and_16s_species_reference_high_expected_value`: 532 species groups
  and 1,928 queries. The next largest evidence categories are
  `add_reference_then_validate_genus_family_not_species` with 147 species
  groups and 685 queries, and `add_12s_same_marker_reference_then_revalidate`
  with 89 species groups and 690 queries.
- Top active targets:
  `Trichiurus_lepturus` is the clearest retrieval/target-curation failure
  because the 16S target species exists but MarkerMirror retrieval is weak;
  `Epinephelus_coioides`, `Oryzias_latipes`, `Gadus_morhua`,
  `Acrossocheilus_paradoxus`, and `Pareiorhaphis_hystrix` are the next
  highest-value reference/evidence targets.
- Interpretation:
  this is not a new rank/no-call policy and does not enable family/genus/species.
  It is the active-curation layer: it tells us which missing marker references,
  target-marker curation fixes, or multi-marker/context additions are most
  likely to change the evidence base before another family/genus attempt.
  Labelled VSEARCH oracle columns are included only for benchmark curation
  triage and must not be used as production inference features.

Exp 132 done: check. Added a paper-level storyline for coauthor/manuscript
alignment.

- New doc:
  `experiments/paper1_phylo_calibrated_assignment/PAPER_STORYLINE.md`.
- Linked from:
  `COAUTHOR_BRIEF.md`, `MANUSCRIPT_ASSETS.md`, and
  `MERGED_MANUSCRIPT_OUTLINE.md`.
- Purpose:
  a concise narrative for the paper: COI establishes calibrated rank/no-call
  under missing references; MarkerMirror shows 12S/16S evidence integration;
  BLASTN/VSEARCH remain essential; order/no-call is the current defensible 12S
  output; active curation turns abstentions into reference/evidence priorities.
- Boundary:
  this is a writing artifact, not a new benchmark or policy. It keeps
  family/genus/species MarkerMirror output disabled.
