# Production Pipeline V1

Last updated: 2026-06-03

## What V1 Is

Production v1 is the current executable COI pipeline packaging layer. It can
start from saved query embeddings, clean split sequence tables, or arbitrary
FASTA/CSV specimen inputs, then applies locked mode-specific rank/no-call
thresholds.

```text
saved query embeddings
  -> vector candidate retrieval
  -> train-reference p-distance rerank over top-k
  -> top-k taxonomic consensus features
  -> seen-test-derived mode-specific thresholds
  -> species/genus/family/order/no-call output
```

The first saved-embedding production pass is complete, the raw split-sequence
timing wrapper runs from `zero_shot_queries.csv`, and the FASTA/CSV CLI smoke
test now runs arbitrary specimen-style inputs through final rank/no-call
output. This is still a research CLI, not a deployed API.

## MarkerMirror Handoff Status

MarkerMirror is not part of the COI production-v1 operating point. It is now a
separate executable 12S research wrapper with conservative order/no-call output:

```text
12S FASTA/CSV
  -> MarkerMirror 12S->16S candidates
  -> BLASTN same-marker 12S candidates
  -> VSEARCH same-marker 12S candidates
  -> shared feature table
  -> stable_order or high_coverage_order decision mode
  -> order/no-call plus reason code
```

Current controlled marker-reference result:

- three-seed target-0.99 stability:
  - 12S->16S: 51.0% mean coverage at 98.9% mean assigned precision;
  - 16S->12S: 71.1% mean coverage at 98.7% mean assigned precision.
- seed1903 resolvability-enhanced compiler:
  same best target-0.99 operating point as the previous seed1903 compiler.

Executable research handoff:

- script:
  `scripts/edna/run_marker_mirror_candidate_generator.py`;
- CPU smoke output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_smoke_12s_to_16s/`;
- full-reference GPU smoke output:
  `results/paper1_phylo_calibrated_assignment/marker_mirror_bridge/production_handoff_fullref_smoke_12s_to_16s/`;
- source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_smoke_summary.csv`
  and
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_smoke_manifest_summary.csv`;
- full-reference source tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_fullref_smoke_summary.csv`
  and
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_fullref_smoke_manifest_summary.csv`.
- cache source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_candidate_generator_cache_smoke_summary.csv`.

The smoke test used one 12S query, 25 16S target species, and top-5 candidates
on CPU. It verifies that specimen-style input can enter the MarkerMirror
candidate-generator path and produce auditable candidate/manifest files. It is
not an accuracy result.

The first full-reference GPU smoke used 32 held-out 12S queries, the full 1,865
species 16S reference, and top-50 candidates. It wrote 1,600 candidate rows. In
this harder specimen-style path, the known target appeared in the top-50 at
25.0 / 59.4 / 78.1 / 84.4% for species/genus/family/order. Treat this as
candidate-generation evidence only.

Cached target-reference embeddings now work. The first cache-write pass wrote a
2,971-row 16S foundation-embedding cache, and the second cache-read pass loaded
that cache, skipped target embedding, and produced an identical candidate table.

The first candidate-generator evidence handoff is also complete. It takes the
cache-backed candidate rows plus the original query table and emits
rank/no-call-ready evidence rows with same-marker sequence checks,
reference-availability flags, tree-neighborhood distances, and
marker-resolvability features. The full-reference smoke produced 1,600 evidence
rows with 97 production numeric features. Labels, where present, are evaluation
diagnostics only.

The full-query production-style run is now complete as a stricter test. It used
all 3,566 12S query rows, the cached full 1,865-species 16S reference, and
top-50 candidates. It wrote 178,300 candidate/evidence rows. Known-target top-50
recovery was 9.5 / 39.9 / 59.8 / 76.3% for species/genus/family/order. This is
lower than the overlap-only benchmark because it is the harder full-reference
handoff path.

The first integrated rank/no-call apply path also runs, but it is not yet a
production operating point. Logistic target-0.99 with species enabled made
species calls that failed transfer. The honest current read is species-disabled:
231 genus/family/order calls over 3,566 queries, 6.5% coverage, 93.1% assigned
precision, and 0 false species calls. This is useful but below the nominal 0.99
target, so keep it as a diagnostic until calibration transfer improves.

Calibration-transfer diagnostics now explain the gap: the controlled validation
split has 100.0% query-species coverage in the 16S target reference, while the
full production-style handoff has only 26.6%. A labelled-handoff
reference-aware policy sweep shows that production-safe abstention gates can
improve the species-disabled row: top-1 MarkerMirror score >= 0.620484 gives
5.83% coverage at 95.67% assigned precision, and score >= 0.697663 gives 3.25%
coverage at 100.00% assigned precision. These are diagnostic gates, not locked
independent thresholds.

The first independent validation pass is complete. Across 50 repeated
query-species splits, target-0.95 gates average 5.79% held-out coverage at
94.39% assigned precision and meet target in 48% of repeats. Target-0.99 gates
average 4.13% held-out coverage at 98.27% assigned precision and meet target in
70% of repeats. This is encouraging, but not yet enough to make MarkerMirror a
final production assignment mode.

Next implementation step:

- use the manuscript-facing MarkerMirror assets for figure/table drafting, or
  add genuinely new evidence before attempting family/genus again. Threshold
  and set-only repairs have already failed to unlock those ranks.

Claim boundary:

- controlled held-out marker-reference validation exists;
- field eDNA production validation does not yet exist;
- 0.99 marker-resolvability rows currently use a rare-kmer prefix-identity
  proxy, not alignment-backed clustering.

## Implementation

Scripts:

- `scripts/edna/run_paper1_coi_pipeline.py`
- `scripts/edna/calibrate_paper1_pipeline_modes.py`
- `scripts/edna/run_paper1_production_v1.py`
- `scripts/edna/run_paper1_raw_sequence_production_v1.py`
- `scripts/edna/run_paper1_fasta_inference_v1.py`
- `scripts/edna/build_paper1_raw_sequence_production_summary.py`
- `experiments/paper1_phylo_calibrated_assignment/runs/12_run_production_v1.sh`

Inputs:

- existing pipeline runs under
  `results/paper1_phylo_calibrated_assignment/pipeline_runs/`;
- locked thresholds from
  `results/paper1_phylo_calibrated_assignment/pipeline_calibration/pipeline_mode_thresholds.csv`.

Outputs:

- `results/paper1_phylo_calibrated_assignment/production_v1/production_v1_summary_all.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/raw_sequence_production_v1_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/production_reason_code_summary.csv`
- `results/paper1_phylo_calibrated_assignment/source_tables/production_reason_code_examples.csv`
- `results/paper1_phylo_calibrated_assignment/raw_sequence_production_v1/*/raw_sequence_production_v1_manifest.json`
- `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_eval_c_known/`
- `results/remote_runs/2026-06-02/rtx_pro_6000/paper1_production_v1_cli/smoke_unlabeled_fasta/`
- `results/paper1_phylo_calibrated_assignment/production_v1/*/production_v1_assignments.csv`
- `results/paper1_phylo_calibrated_assignment/production_v1/*/production_v1_manifest.json`

## Current Operating Point

Mode:

- encoder: CNN seed1206;
- retrieval: exact vector over saved embeddings;
- rerank: train-reference p-distance over retrieved top-k;
- assignment source: reranked candidate order;
- threshold target: 0.99, calibrated on seen-test rows.

Results:

| Split | Coverage | Assigned Precision | False Species-Call Rate | Species Calls |
|---|---:|---:|---:|---:|
| seen-test | 95.2% | 94.9% | 0.0% | 5 |
| held-out fish | 95.8% | 93.0% | 0.0% | 0 |
| unseen-genera | 92.3% | 83.9% | 0.0% | 0 |

Interpretation:

- the held-out operating point is conservative;
- it makes no species calls on the two missing-reference style splits;
- the value is useful genus/family/order assignment without hallucinated
  species labels.

## Raw Sequence Timing

Vast RTX PRO 6000 run using the CNN seed1206 checkpoint:

| Split | Queries | Total Seconds | ms/query | Embedding Seconds | Vector Seconds | Rerank Seconds |
|---|---:|---:|---:|---:|---:|---:|
| seen-test | 15,763 | 49.80 | 3.16 | 30.08 | 13.13 | 1.18 |
| held-out fish | 11,594 | 46.77 | 4.03 | 38.44 | 3.08 | 0.86 |
| unseen-genera | 9,148 | 47.12 | 5.15 | 33.36 | 8.41 | 0.69 |

These timings include raw split-sequence embedding export, exact vector search,
top-25 train-reference p-distance reranking, and locked production-v1
rank/no-call packaging. They do not include API upload overhead or external
deployment overhead.

## FASTA/CSV CLI

The first specimen-facing CLI is:

- `scripts/edna/run_paper1_fasta_inference_v1.py`
- detailed usage doc:
  `experiments/paper1_phylo_calibrated_assignment/PRODUCTION_CLI_V1.md`

It accepts:

- FASTA headers/sequences;
- CSV rows with a sequence column such as `nucleotides`, `sequence`, `seq`, or
  `barcode`;
- optional known taxonomy columns for evaluation.

Smoke tests on the Vast RTX PRO 6000:

| Input | Queries | Coverage | Precision If Known | Species Calls | Total Seconds |
|---|---:|---:|---:|---:|---:|
| CSV with known labels | 16 | 100.0% | 87.5% | 0 | 13.29 |
| FASTA without labels | 8 | 100.0% | unavailable | 0 | 13.07 |

These are smoke tests only. Small-batch timing is dominated by Python/Torch
startup. Use the full raw split-sequence timing table for throughput claims.

## Claim Boundary

Safe claim:

> A vector-first barcode pipeline with classical-light p-distance reranking and
> locked rank/no-call calibration can assign many held-out COI queries to
> genus/family/order while avoiding false species calls at the current
> conservative operating point.

New explanatory overlay:

> Existing production-v1 assignments can be translated into reason codes that
> distinguish broader-rank support, no-call, species not supported at the
> operating point, likely reference gaps, and top-k ambiguity. This is an
> explanation layer over the current operating point, not a new classifier.

Do not claim:

- deployment-grade API/web packaging is complete;
- the research CLI is a deployed product or API;
- species-level assignment is solved;
- p-distance reranking is universally superior;
- vector retrieval replaces BLAST.

## Next V1 Hardening

1. Wire reason codes into `inference_assignments.csv` from the FASTA/CSV CLI.
2. Improve MarkerMirror rank/no-call calibration transfer; current
   species-disabled full-query apply is 93.1% precision, below the 0.99 target.
3. Replace the MarkerMirror 0.99 resolvability proxy with VSEARCH/edlib if the
   near-exact rows become manuscript-facing.
4. Add a polished API wrapper around the working FASTA/CSV research CLI.
5. Add full-batch user-style FASTA timing, including file parsing and batching.
6. Add BLAST/VSEARCH optional rerank over the vector top-k candidate set.
7. Add bootstrap confidence intervals to production-v1 summary rows.
8. Add one external-style demo input with no internal split metadata.
9. Convert the MarkerMirror union-candidate audit into a selectable 12S/16S
   candidate source. First diagnostic complete: the production-style union
   candidate table has 355,231 rows, improves full-query top50 genus/family/
   order support from 39.9/59.8/76.3% to 91.7/95.1/99.6%, and the conservative
   family/order source-agreement policy assigns 25.2% of queries at 98.4%
   precision with 0 species calls. Next hardening step: replace the simple
   k-mer arm with BLAST/VSEARCH/edlib where needed and build a richer
   hierarchical selective/conformal evidence compiler. The first top-1 HGB
   compiler was diagnostic but not better than source agreement or simple
   order score gating.
10. Add the MarkerMirror union reason-code layer to candidate/evidence output.
    First diagnostic complete: 2,249/3,566 full-query rows have genus-level
    union support without a species call, 621 rows emit conservative family
    calls at 98.1% precision, and 277 rows emit conservative order calls at
    99.3% precision. The layer also writes reference-curation priorities, but
    current 12S reference-gap labels remain current-table/split-design
    diagnostics until the final production reference set is defined.
11. Promote same-marker evidence from k-mer audit toward alignment-backed
    evidence. First edlib validation complete: edlib reranking of the existing
    same-marker top50 pool preserves top10 genus/family/order support at
    87.8/94.3/98.8%. This is enough to reduce the "k-mer artifact" risk, but
    full all-vs-all BLAST/VSEARCH-style candidate generation is still separate.
12. Keep the list-level selective compiler as a diagnostic, not a production
    policy. It improves order-only target-0.99 coverage from the previous top-1
    HGB's 67.4% to 83.1%, with 98.8% mean precision, but target-0.99 is met in
    only 56% of species-split repeats. Production-safe output remains source
    agreement plus reason codes until calibration transfer is stronger.
13. Use BLASTN and VSEARCH as measured same-marker 12S classical candidate
    arms. The VSEARCH global run produced MarkerMirror + VSEARCH union top50
    support of 9.5/91.8/95.1/99.6% for species/genus/family/order. The BLASTN
    local run produced MarkerMirror + BLASTN union top50 support of
    9.5/92.1/95.3/99.7%. These replace the earlier k-mer-only audit for
    high-rank candidate-generation claims.
14. Use all-source top1 order agreement as the current conservative
    MarkerMirror production-safe repair candidate. In the 50-repeat
    BLAST/VSEARCH calibration-transfer diagnostic it averages 24.8% coverage at
    99.6% precision and meets target-0.99 in every repeat. Exp 117 adds a
    stricter nested repair for the higher-coverage BLAST/VSEARCH top10 order
    rows: BLASTN/VSEARCH top-10 order agreement with nested global Wilson95
    locking reaches 57.2% mean held-out coverage at 99.8% precision and meets
    target-0.99 in all 50 outer repeats. Exp 118 exposes this as explicit
    `high_coverage_order` mode; keep it labelled as an order-only diagnostic
    and not the default production-safe mode.
15. The stable order policy now has a production-style assignment table:
    `marker_mirror_stable_order_policy_assignments.csv`. The conservative
    max-repeat target-0.99 threshold assigns 880/3,566 full-query 12S rows
    (24.7% coverage) at 99.7% precision with 0 false species calls. This is the
    current safest CLI/rationale behavior: order call when MarkerMirror,
    BLASTN, and VSEARCH agree; otherwise no-call with a reason code.
16. A label-stripped handoff table now exists:
    `marker_mirror_stable_order_policy_production_assignments.csv`. This is the
    correct downstream CLI/API payload once MarkerMirror, BLASTN, and VSEARCH
    candidate features have been generated. The remaining gap is orchestration:
    arbitrary 12S FASTA -> MarkerMirror candidates -> BLASTN/VSEARCH candidates
    -> shared feature table -> this stable order/no-call policy.
17. The first one-command orchestration wrapper now exists:
    `scripts/edna/run_marker_mirror_12s_production_v1.py`. It normalizes
    12S FASTA/CSV input, plans MarkerMirror, BLASTN, VSEARCH, feature-table,
    and stable-policy stages, and writes a dependency report/manifest. Dry-run
    smoke output is under
    `results/paper1_phylo_calibrated_assignment/marker_mirror_12s_production_v1/dry_run_smoke/`.
    Local status: BLASTN and makeblastdb are available; VSEARCH is missing.
    A two-query BLASTN stage smoke completed locally and wrote 100 top-50
    candidate rows. Full local all-source execution needs local VSEARCH
    installation; Vast execution is recorded in the next item.
18. The full one-command MarkerMirror 12S wrapper has now run successfully on
    Vast for all 3,566 current 12S queries:
    `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_full_all_queries_20260603/`.
    It ran MarkerMirror, BLASTN, VSEARCH, feature-table construction, and the
    stable order/no-call policy end to end. Final output:
    880 order calls and 2,686 no-calls. Diagnostic labelled precision is 99.7%
    with 0 false species calls. Runtime components were about 15.0 s
    MarkerMirror, 254.6 s BLASTN, 48.8 s VSEARCH, and 1.6 s stable policy.
19. An unlabeled FASTA smoke has also completed on Vast:
    `results/remote_runs/2026-06-03/rtx_pro_6000/marker_mirror_12s_production_v1/vast_unlabeled_fasta_smoke_20260604/`.
    It used 2 FASTA records with taxonomy stripped, emitted 1 order call and
    1 no-call, and correctly left diagnostic precision/correctness fields blank
    because no truth labels were supplied.
20. CLI usage is summarized in
    `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md`.
21. Exp 117 produced the first stable high-coverage order/no-call repair
    candidate:
    `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_high_coverage_order_repair_summary.csv`.
    The locked full-table diagnostic assigns 2,513/3,566 rows, 70.5% coverage,
    at 99.8% labelled precision. This should not replace the conservative
    stable CLI behavior until it is intentionally wired as a high-coverage mode.
22. Exp 118 wires that policy as explicit
    `--decision-mode high_coverage_order` in
    `scripts/edna/run_marker_mirror_12s_production_v1.py`. The default remains
    `stable_order`. Vast smokes passed for labelled and unlabeled inputs in both
    modes; high-coverage remains order/no-call only and diagnostic.
23. Exp 119 tested whether the same nested repair could safely emit family or
    genus. It could not: no family/genus row met target-0.99 in all 50
    species-split repeats. The wrapper should continue to expose only
    order/no-call decisions for the high-coverage mode.
24. Exp 121 tested a set-valued family/genus alternative. It also failed to
    produce useful target-0.99 family/genus output: family peaks at 95.4%
    full-query set coverage with mean set size 34.4, and genus peaks at 92.4%
    with mean set size 79.6. This confirms that family/genus require new
    evidence, not only a different decision wrapper.
25. Exp 122 created a manuscript-facing MarkerMirror package:
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/`.
    It contains candidate-support, order-policy, rank-boundary, runtime,
    figure-plan, methods-blurb, and manifest files. These are writing assets
    built from existing source tables, not new benchmark metrics.
26. Exp 123 rendered draft MarkerMirror figures from the Exp 122 package:
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/`.
    The figure drafts cover candidate support, stable versus high-coverage
    order/no-call, rank boundaries, runtime, and a slide-ready summary. These
    are not new production metrics.
27. Exp 124 created slide-ready MarkerMirror tables and a five-slide outline:
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/`.
    This is for coauthor review/deck assembly and does not alter the production
    claim boundary.
28. Exp 125 created manuscript-facing MarkerMirror captions and text snippets:
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`.
    This includes figure captions, results/methods paragraphs, and a
    claim-boundary box for drafting. It does not alter the production claim
    boundary.
29. Exp 126 created a MarkerMirror manuscript section outline and checklist:
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_section_outline.md`
    and
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_section_checklist.csv`.
    This is a writing handoff that places MarkerMirror as a focused 12S/eDNA
    extension section; it does not alter the executable pipeline boundary.
30. Exp 127 created a next-evidence audit for future family/genus work:
    `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_next_evidence_source_audit.csv`,
    `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_by_lineage.csv`,
    and
    `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_family_genus_next_evidence_plan.md`.
    This explicitly says the next family/genus attempt needs new evidence
    such as reference coverage plus alignment-backed marker-resolvability; it
    does not enable family/genus in production.
31. Exp 128 tested the first new-evidence follow-up from Exp 127:
    lineage/reference-coverage features joined to BLASTN/VSEARCH/MarkerMirror
    policy rows in a nested species-split diagnostic. Output source tables are
    `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_*.csv/json`.
    No family/genus target-0.99 row transferred cleanly, so this remains a
    diagnostic negative result and does not change the executable wrapper:
    `stable_order` remains the default and `high_coverage_order` remains an
    explicit order-only research mode.
32. Exp 129 created VSEARCH-backed marker-resolvability source tables:
    `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_*.csv/json`.
    This replaces the rare-kmer proxy as an alignment-backed marker-ceiling
    diagnostic. It supports future policy work but does not change the
    executable wrapper; family/genus/species remain disabled.
33. Exp 130 tested that future policy work with production-available VSEARCH
    cluster features:
    `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_*.csv/json`.
    It did not produce stable target-0.99 family/genus transfer. The wrapper
    remains order/no-call only.
34. Exp 131 adds active reference-curation/value-of-information source tables:
    `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_*.csv/json`.
    This is not a new assignment mode. It ranks which missing 12S/16S
    references, target-marker curation fixes, or multi-marker/context additions
    would most likely change the evidence for future no-calls and high-rank-only
    cases. The executable wrapper still emits only `stable_order` or explicit
    `high_coverage_order` order/no-call outputs.
