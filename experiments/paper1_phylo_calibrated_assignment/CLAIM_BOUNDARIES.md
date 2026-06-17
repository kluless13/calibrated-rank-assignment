# Paper 1 Claim Boundaries

This file is a guardrail for writing. It separates what the current results
support from what they do not support yet.

## Supported Now

### Tree-Aware Barcode Inference Is Useful

Supported by:

- COI retrieval metrics;
- tree-recovery metrics;
- nearest-reference diagnostics;
- rank/no-call policy tables.

Safe claim:

> Barcode evidence can be represented and audited in species-tree space, and
> tree-aware diagnostics help decide whether a species, genus, family, order, or
> no-call output is justified.

### Rank/No-Call Beats Forced Species Calling Under Missing References

Supported by:

- `missing_reference_aware_policy_summary.csv`;
- `missing_reference_aware_policy_bootstrap.csv`;
- `pipeline_coi_method_benchmark.csv`.

Safe claim:

> A seen-test-derived missing-reference-aware policy can trade forced species
> calls for higher-rank assignments and lower false species-call rates on
> held-out species/genera.

Current candidate locked row (CNN seed1206, p-distance reranked, target 0.99 —
the headline operating point used throughout):

- Eval C: 95.8% coverage, 93.0% assigned precision, **0.0% false species calls**.
- Unseen-genera: 92.3% coverage, 83.9% assigned precision, **0.0% false species calls**.
- Prospective (species-disjoint calibration, 30 repeats): 0.923 coverage, 0.900
  precision, 0.0% false species — survives every repeat.

(The non-reranked exact-vector variant gives 96.1%/90.0% coverage/precision on
Eval C but does **not** reach 0% false species; the reranked row above is the
locked candidate.)

### Vector-First Retrieval Is Feasible

Supported by:

- `ann_vector_runtime_comparison.csv`;
- `ann_vector_recall_against_exact.csv`;
- `ann_vector_stress_runtime.csv`;
- `controlled_vector_speed_benchmark.csv`.

Safe claim:

> A vector index can retrieve candidate barcode neighbors at sub-millisecond
> local latency, making fast candidate generation plausible.

Do not claim final deployment speed until target-hardware timing is run.
Current controlled rows are repeat-based local timings, not deployment timing
on target hardware.

The executable COI pipeline now has HNSW rows, but those rows are approximate
candidate retrieval checks. Report exact-vector calibrated rows as the primary
pipeline operating point unless the HNSW recall/assignment tradeoff is
explicitly discussed.

### Classical-Light Reranking Is Calibrated But Still Needs Strict Validation

Supported by:

- `pipeline_run_summary.csv` p-distance experimental rows;
- `pipeline_calibration/pipeline_mode_thresholds.csv`;
- `pipeline_calibration/pipeline_mode_policy_summary.csv`;
- `production_v1/production_v1_summary_all.csv`;
- `pipeline_runs/*_pdistance_experimental/pipeline_manifest.json`.

Safe claim:

> Train-reference p-distance reranking can be inserted after vector retrieval
> without using held-out query species sequences. With seen-test-derived
> rerank-specific thresholds, it can reduce false species calls by backing off
> to higher taxonomic ranks.

Do not claim:

> p-distance reranking is final or universally superior.

The calibrated p-distance rows still need strict tree-pruned validation before
final missing-reference claims. Current target-0.99 calibrated p-distance rows
make zero species calls on Eval C and unseen-genera, so the claim is
rank-adaptive conservatism, not species-level superiority.

Production v1 now packages this conservative operating point as final
assignment/summary/manifest files for saved embeddings, clean split sequences,
and specimen-style FASTA/CSV smoke-test input. It remains a research CLI, not a
deployed API.

### MarkerMirror 12S/16S Evidence Compiler

Supported by:

- shared 12S/16S retrieval seed repeats;
- full-reference candidate exports for seeds 1901, 1902, and 1903;
- executable full-reference MarkerMirror candidate-generator smoke;
- candidate-generator evidence handoff table;
- integrated evidence joins and logistic/HGB calibrators;
- seed1903 resolvability-enhanced evidence join.

Safe claim:

> A learned 12S/16S shared marker space can generate cross-marker candidate
> lists, and an evidence compiler can convert those candidates into
> high-precision controlled species/rank calls with near-zero false species
> calls on held-out marker-reference tests.

Current controlled target-0.99 stability:

- 12S->16S: 51.0% mean coverage at 98.9% mean assigned precision;
- 16S->12S: 71.1% mean coverage at 98.7% mean assigned precision.

Claim boundary:

- this is controlled marker-reference validation, not real field eDNA
  production validation;
- the 0.99 resolvability rows currently use a rare-kmer prefix-identity proxy,
  not VSEARCH/edlib clustering;
- MarkerMirror now has a research FASTA/CSV candidate-generator script, but it
  is candidate-only. The full-reference GPU smoke proves candidate generation
  runs against the 16S reference, and the handoff table proves those candidates
  can be converted into evidence rows. The integrated calibrator apply path now
  runs, but calibration does not transfer at the nominal 0.99 target. Species
  calls must remain disabled in production-style MarkerMirror assignment until
  independently validated.
- The full-query transfer diagnostic explains why: the controlled validation
  split has 100.0% query-species coverage in the 16S target reference, while the
  full multisource handoff has only 26.6%. Full-query top-50 species recovery is
  9.5% overall and 35.8% only when the true species is present in the 16S
  reference. This supports a reference-aware genus/family/order/no-call claim,
  not a species-identification claim.
- A labelled-handoff reference-aware policy sweep shows that production-safe
  abstention gates can improve precision: top-1 MarkerMirror score >= 0.620484
  gives 5.83% coverage at 95.67% assigned precision with no species calls; a
  stricter score >= 0.697663 gives 3.25% coverage at 100.00% assigned
  precision. These are not independently calibrated production thresholds yet.
- Repeated species-split validation partially supports this direction but does
  not yet lock the threshold. Target-0.95 gates average 94.39% held-out
  precision at 5.79% coverage and meet target in 48% of repeats. Target-0.99
  gates average 98.27% held-out precision at 4.13% coverage and meet target in
  70% of repeats. Treat MarkerMirror assignment as promising reference-aware
  abstention, not a production guarantee.

### Ecology Alone Does Not Solve Species-Level eDNA Assignment

Supported by:

- pure RLS prior-only rows;
- pure OBIS prior-only rows;
- same-sample co-occurrence-only rows;
- sequence/tree and sequence+ecology rows.

Safe claim:

> Ecological and sample-context priors carry higher-rank signal, but they do not
> replace sequence evidence for species-level eDNA assignment.

## Partial / Diagnostic

### EPA-ng, APPLES, And Fernando-Style Placement

Supported by:

- EPA-ng jplace scoring;
- official APPLES 2.0.11 jplace scoring on matched completeness sweeps;
- APPLES-like local distance-placement scoring;
- nearest-reference tree-error summaries.
- Fernando-like edge-to-sister diagnostics;
- simulated-placement-tree PCP diagnostics.

Safe claim:

> We compare against likelihood placement and an explicitly labelled
> APPLES-like distance-placement diagnostic on the same clean splits, and
> against official APPLES/EPA-ng on Fernando-style matched completeness sweeps.

Also safe:

> We ran official APPLES 2.0.11 and EPA-ng on our own Fernando-style
> completeness-sweep matrix.

Do not claim:

> We reproduced exact Fernando PCP.

The final `fernando_completeness_final_30/` rows are full matched diagnostics
for our public setup: 30 sweeps, EPA-ng plus official APPLES, and both
rank/clade and simulated-placement-tree PCP-style summaries. They are closer to
Fernando than the earlier held-out-split placement diagnostics, but they are
still not an exact Fernando reproduction because the reference set, tree
extraction, backbone generation, and PCP implementation are not identical to
Fernando et al.'s workflow.

### eDNA Rank/No-Call

Supported only for conservative higher-rank calls under the species-disabled
Eco-Phylo posterior.

Safe claim:

> A candidate-level Eco-Phylo posterior can make conservative genus/family/order
> eDNA calls while disabling unsupported species calls.

Current operating point:

- species-disabled genus -> family -> order target-95 backoff assigns 40.3% of
  held-out Global_eDNA queries at 94.3% accuracy;
- 30 nested threshold resplits average 40.2% held-out assignment at 94.3%
  accuracy.
- true nested fit70 posterior repeats assign 38.9%, 38.5%, and 54.9% of
  held-out Global_eDNA queries at 93.4%, 95.4%, and 84.5% accuracy under the
  species-disabled target-95 backoff.

Do not claim species-level eDNA assignment. Do not describe the mixed
species-disabled target-95 policy as a guaranteed 95% held-out operating point;
under true nested refits it is unstable across calibration-group splits.

### MarkerMirror Union Candidate / Rank Policy

Supported as a diagnostic production-style candidate/evidence path.

Safe claim:

> A union 12S candidate path combining MarkerMirror cross-marker candidates with
> same-marker 12S k-mer candidates substantially improves genus/family/order
> support and can make conservative family/order calls when independent
> candidate sources agree.

Current diagnostic operating point:

- MarkerMirror-only full-query top50 support is 9.5 / 39.9 / 59.8 / 76.3% for
  species/genus/family/order.
- Union top50 support is 9.5 / 91.7 / 95.1 / 99.6%.
- Top-1 source agreement at family/order only assigns 25.2% of queries at
  98.4% assigned precision with 0 species calls.
- Repeated species-split same-marker score gates show high-rank signal but are
  not locked: family target-0.95 averages 92.7% coverage at 94.8% precision;
  order target-0.99 averages 68.5% coverage at 99.0% precision.
- The first learned HGB compiler over 102 union features is a negative
  diagnostic: order-only target-0.99 averages 67.4% coverage at 98.5%
  precision and target is met in 44% of species-split repeats. It does not beat
  the simple order score gate.
- The union reason-code layer separates failures into high-rank-only support,
  current marker-reference gaps, species candidates requiring calibration, and
  cross-marker retrieval misses. The largest bucket is high-rank union support
  to genus: 2,249/3,566 full-query rows. Conservative source-agreement rows
  remain clean: family calls are 98.1% precise and order calls are 99.3%
  precise with species disabled.
- Edlib validation over the existing same-marker 12S top50 candidate pool
  supports the same high-rank story. Edlib-reranked top10 support is 0.0 /
  87.8 / 94.3 / 98.8% for species/genus/family/order, close to or slightly
  better than the original k-mer order for top10. This validates the candidate
  pool but is not full all-vs-all BLAST/VSEARCH search.
- A list-level selective HGB compiler improves the high-coverage order
  diagnostic over the first top-1 HGB: order-only target-0.99 reaches 83.1%
  mean coverage at 98.8% mean precision, versus 67.4% coverage at 98.5% for
  the top-1 HGB. This is promising but still not production-locked because the
  99% target is met in only 56% of species-split repeats.
- VSEARCH global same-marker candidate generation now supports the high-rank
  union story with a classical alignment-based arm. Same-marker VSEARCH top50
  support is 0.0 / 90.4 / 94.9 / 99.4% for species/genus/family/order, and the
  MarkerMirror + VSEARCH union top50 support is 9.5 / 91.8 / 95.1 / 99.6%.
- BLASTN local same-marker candidate generation independently supports the
  same story. Same-marker BLASTN top50 support is 0.0 / 90.7 / 95.1 / 99.4%
  for species/genus/family/order, and MarkerMirror + BLASTN union top50 support
  is 9.5 / 92.1 / 95.3 / 99.7%.
- The first BLAST/VSEARCH-backed calibration-transfer repair finds one stable
  conservative operating point: all-source top1 order agreement averages 24.8%
  coverage at 99.6% precision and meets target-0.99 in 100% of 50
  query-species repeats. Higher-coverage order policies reach about 56.1-69.0%
  coverage at about 99.4-99.5% mean precision but meet target-0.99 in 82-94% of
  repeats, so they are not locked production thresholds yet.
- The stable all-source order policy has been applied as an explicit
  assignment/reason-code handoff. With the conservative max-repeat threshold it
  assigns 24.7% of full-query 12S rows at 99.7% precision and 0 false species
  calls. The label-stripped production assignment table is available, and the
  arbitrary-FASTA orchestration wrapper is now built but dependency-gated. This
  is an order-only policy, not a species identifier.
- The first arbitrary-input 12S orchestration wrapper is now built and dry-run
  smoked. It normalizes FASTA/CSV input and plans MarkerMirror, BLASTN,
  VSEARCH, feature-table, and stable-policy stages. Local execution is blocked
  at the full all-source stage because VSEARCH is not installed locally; BLASTN
  smoke succeeded.
- The same wrapper has completed a full 3,566-query run on Vast, where VSEARCH
  is available. The result reproduces the conservative stable policy as an
  executable chain: 880 order calls, 2,686 no-calls, 99.7% diagnostic precision,
  and 0 false species calls.
- An unlabeled FASTA smoke has completed on Vast. It emits assignments/reason
  codes while leaving precision/correctness blank when truth labels are absent.
  This checks production output behavior, not accuracy.
- A high-coverage BLAST/VSEARCH order repair diagnostic now exists. In nested
  species-split evaluation, BLASTN/VSEARCH top-10 order agreement with nested
  global Wilson95 locking reaches 57.2% mean coverage at 99.8% mean assigned
  precision and meets target-0.99 in all 50 repeats. This supports a stronger
  order/no-call research mode, not species identification.
- The high-coverage order repair is now exposed in the 12S wrapper as explicit
  `--decision-mode high_coverage_order`. The default remains conservative
  `stable_order`.
- Family/genus high-coverage repair has been tested and did not meet stable
  target-0.99 transfer across all 50 species-split repeats. Do not present
  family/genus as enabled production decisions.
- Set-valued family/genus output has also been tested. It still does not reach
  target-0.99 under current evidence without very large candidate sets. Do not
  frame the limitation as merely a single-label calibration problem.

Do not claim:

- union candidates solve species-level 12S/16S identification;
- same-marker k-mer is the final BLAST/VSEARCH/edlib comparator; VSEARCH and
  BLASTN candidate-generation tables are now available and should be used for
  claim-facing same-marker evidence.
- score-gated family/order policies are field-eDNA production thresholds.
- the first learned union evidence compiler fixes calibration transfer.
- current 12S reference-gap labels are final real-world absence statements;
  they are current-table/split-design diagnostics until the final production
  reference set is defined.
- the list-level selective compiler is a locked target-0.99 production policy.
- raw Exp 111 high-coverage BLAST/VSEARCH top10 order rows are locked
  production thresholds; the stricter Exp 117 nested repair is the stable
  high-coverage diagnostic to cite, and it is still not the default deployed
  behavior.
- the stable all-source order handoff supports genus, family, or species calls;
  the current locked behavior is order/no-call only.
- the one-command 12S wrapper has completed a full all-source production run
  locally; the full run completed on Vast, while local execution remains blocked
  by missing VSEARCH.
- the full Vast wrapper run is field-eDNA validation or a deployed production
  API. It is still a research pipeline run over the current labelled 12S table.
- the unlabeled FASTA smoke is an accuracy estimate; it only verifies behavior
  when no hidden labels are present.
- the high-coverage BLAST/VSEARCH order repair is species, genus, or family
  identification; it is order-level only.
- the high-coverage order repair is the default deployed production behavior;
  it is an explicit research mode, not the default and not a deployed API.

## Not Supported

- Mamba is the best architecture.
- Neural methods beat BLAST/VSEARCH/k-mer for exact species retrieval.
- The system discovers taxa absent from the candidate tree.
- Current placement tables are exact Fernando PCP.
- Local APPLES-like p-distance output is official APPLES.
- 12S sequence-only evidence solves species-level eDNA assignment.
- Local vector timing is a final production benchmark.
- Current p-distance rerank rows prove species-level assignment is solved.
- Production v1 is deployed API/web production.
- MarkerMirror is production-proven on field eDNA.
- MarkerMirror plus same-marker k-mer support solves species-level 12S/16S
  identification when the species is absent from the relevant reference tables.
- The same-marker k-mer union audit is a final BLAST/VSEARCH alignment
  comparator.
- Edlib reranking of the existing k-mer top50 pool is a complete replacement
  for full all-vs-all BLAST/VSEARCH search.
- VSEARCH global alignment and BLASTN local alignment are interchangeable;
  both are now measured and should be described as separate classical arms.
- The Exp 104 score gates are locked field-eDNA production thresholds.
- Exp 131 active reference-curation scores are production inference
  probabilities. They are heuristic value-of-information priorities for
  targeted reference/evidence improvement, not rank/no-call assignment scores.

## Next Claim-Strengthening Steps

1. Decide whether the Fernando-style sweep diagnostics are sufficient
   comparator context, or whether exact Fernando PCP implementation is required.
2. Add neural tree-space encoders to the same reduced-backbone completeness
   sweep matrix if we want direct neural-versus-placement comparison.
3. Target-hardware end-to-end speed benchmark including reranking/calibration.
4. Decide whether to improve the eDNA posterior calibration so mixed
   species-disabled target-95 reaches the target on held-out groups, or report
   the current 93.4% held-out operating point honestly.
5. Larger reference-scale stress test for vector-first retrieval.
6. Add raw FASTA input plus selected encoder export to production v1.
7. Use the completed MarkerMirror 12S wrapper plus Exp 123 figure drafts,
   Exp 125 manuscript text, and Exp 126 section outline for order/no-call
   manuscript review. Do not attempt family/genus again without adding
   genuinely new evidence such as reference coverage, marker-resolvability,
   geography, or co-occurrence priors.
8. Exp 127 narrows the next family/genus path: lineage-specific reference
   coverage plus alignment-backed marker-resolvability are the first
   production-style evidence sources to test. Geography and co-occurrence
   belong in sample-aware eDNA mode, not arbitrary single-query FASTA mode.
9. Exp 128 directly tested lineage/reference-coverage features in a nested
   species-split policy diagnostic. They improved the evidence inventory but
   did not produce stable target-0.99 family/genus transfer, so family/genus
   remain disabled. The next legitimate attempt needs alignment-backed
   marker-resolvability, active reference-curation evidence, or sample-aware
   ecological context.
10. Exp 129 replaces the earlier rare-kmer marker-resolvability proxy with
    VSEARCH-backed near-exact clustering. This strengthens the marker-ceiling
    evidence but does not itself enable family/genus/species. A separate
    species-disjoint rank/no-call policy would still be required before those
    ranks can be emitted.
11. Exp 130 tested that separate policy direction by joining production-
    available VSEARCH cluster features to MarkerMirror/BLASTN/VSEARCH policy
    rows. It still did not stabilize family/genus transfer, so family/genus
    remain disabled. Do not turn Exp 129 oracle-support columns into production
    features because they use hidden truth labels.
12. Exp 131 built the active reference-curation/value-of-information layer.
    It ranks missing/weak reference actions and identifies high-value model or
    target-curation failures such as `Trichiurus_lepturus`. This is a
    curation-planning artifact, not an enabled family/genus/species policy.
13. Build a list-level or hierarchical selective compiler. The first top-1 HGB
   compiler was useful as a negative result but did not improve the operating
   point.
14. If the listwise compiler remains central, improve calibration transfer
    before claiming it as an operating point: current order-only target-0.99
    mean precision is 98.8%, not 99% locked.
