# MarkerMirror Coauthor One-Pager

## One-Sentence Summary

MarkerMirror is a 12S evidence tool that combines learned cross-marker
retrieval with BLASTN/VSEARCH same-marker evidence, then makes conservative
order/no-call decisions instead of forcing unsupported species labels.

## Why This Matters

Short 12S fragments often cannot support species-level identification,
especially when the true species is missing from the reference table. The useful
question is therefore not "can we always name the species?" but:

> What is the deepest taxonomic claim supported by the molecular evidence?

MarkerMirror currently answers that at the order/no-call level for the present
12S benchmark, with explicit reason codes and strict abstention.

## Pipeline

```text
12S FASTA/CSV query
  -> MarkerMirror learned 12S->16S candidate retrieval
  -> BLASTN same-marker 12S candidate retrieval
  -> VSEARCH same-marker 12S candidate retrieval
  -> shared evidence table
  -> calibrated order/no-call policy
  -> assignment plus reason code
```

Current wrapper modes:

- `stable_order`: conservative default.
- `high_coverage_order`: explicit research/diagnostic mode.

## Headline Results

Candidate support on 3,566 full-query 12S sequences:

| Candidate source | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| MarkerMirror 12S->16S only | 9.5% | 39.9% | 59.8% | 76.3% |
| Same-marker 12S BLASTN | 0.0% | 90.7% | 95.1% | 99.4% |
| MarkerMirror + BLASTN union | 9.5% | 92.1% | 95.3% | 99.7% |
| Same-marker 12S VSEARCH | 0.0% | 90.4% | 94.9% | 99.4% |
| MarkerMirror + VSEARCH union | 9.5% | 91.8% | 95.1% | 99.6% |

Order/no-call policies:

| Mode | Calls | Coverage | Diagnostic precision | Notes |
|---|---:|---:|---:|---|
| `stable_order` | 880 / 3,566 | 24.7% | 99.7% | Default conservative mode. |
| `high_coverage_order` diagnostic | 2,513 / 3,566 | 70.5% | 99.8% | Explicit order-only research mode. |

Nested validation for the high-coverage order diagnostic:

- BLASTN/VSEARCH top-10 order agreement with nested global Wilson95 locking.
- 57.2% mean held-out coverage.
- 99.8% mean held-out precision.
- Target 99% precision met in 100% of 50 species-split repeats.
- Minimum repeat precision: 99.3%.

Runtime for the full labelled 12S wrapper on the Vast RTX host:

- MarkerMirror: 15.0 s.
- BLASTN: 254.6 s.
- VSEARCH: 48.8 s.
- Stable policy: 1.6 s.

## Important Negative Result

Family and genus were tested with the same nested repair framework and are not
stable enough to enable:

| Rank | Best mean coverage | Best mean precision | Target met rate | Decision |
|---|---:|---:|---:|---|
| Order | 57.2% | 99.8% | 100% | Enabled only as explicit diagnostic mode. |
| Family | 35.5% | 99.35% | 94% | Diagnostic only; not enabled. |
| Genus | 7.8% | 99.79% | 98% | Diagnostic only; not enabled. |

This is scientifically useful because it prevents overclaiming. The current
tool can make high-confidence order calls, but it should not claim reliable
family, genus, or species identification on this benchmark yet.

We also tested set-valued family/genus output. That did not solve the problem:
the best full-query family set coverage was 95.4% with a mean set size of 34.4
families, and the best genus set coverage was 92.4% with a mean set size of
79.6 genera. The limitation is not just single-label calibration; the current
evidence is not strong enough for useful target-99 family/genus output.

We then tested lineage/reference-coverage features as a genuinely new evidence
source. That also did not unlock family/genus: in 50 species-split repeats, the
target-99 family diagnostic averaged 87.4% coverage at 98.0% precision and met
target in only 10% of repeats; the target-99 genus diagnostic averaged 17.7%
coverage at 97.6% precision and met target in 42% of repeats. This keeps
family/genus disabled and points the next attempt toward alignment-backed
marker-resolvability or sample-aware ecological evidence.

We have now replaced the earlier rare-kmer marker-resolvability proxy with
VSEARCH clustering. At 99% identity, 12S query oracle support is 77.9% species,
95.2% genus, 99.6% family, and 99.7% order, but only 19.6% of query clusters
contain a current reference. This supports the high-rank marker-ceiling story,
but it is still not a validated family/genus decision policy.

We also tested the production-available VSEARCH cluster features inside a
learned policy. That still did not solve family/genus: target-99 family averaged
57.3% coverage at 95.5% precision, and target-99 genus averaged 11.5% coverage
at 87.8% precision. The next family/genus attempt needs a new information
source, not another threshold/model wrapper around the same 12S evidence.

We now have that next layer as a reference-curation output rather than a rank
policy. The active value-of-information table ranks 795 species groups by which
reference or evidence addition would most likely improve future no-calls. The
largest category is adding both 12S and 16S species references for 532 species
groups and 1,928 queries. This gives the system a useful "what data should we
collect next?" output while keeping family/genus/species disabled.

## What We Can Claim

- Combining learned marker-bridging with BLASTN/VSEARCH produces strong
  high-rank candidate support from 12S.
- Conservative evidence agreement can emit precise order-level calls while
  abstaining on unsupported cases.
- The wrapper runs on labelled and unlabeled FASTA/CSV inputs and emits
  production-style assignments plus reason codes.
- The system can now emit active reference-curation priorities for unsupported
  cases, instead of only returning a no-call.
- The contribution is rank-aware evidence compilation under missing or
  ambiguous molecular evidence, not "deep learning beats BLAST."

## What We Should Not Claim

- Species identification is solved.
- Family/genus calls are production-ready.
- MarkerMirror replaces BLASTN or VSEARCH.
- The current benchmark is field-eDNA validation.
- The high-coverage order mode is the default production policy.

## Files To Check

- CLI usage:
  `experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md`
- Current central results:
  `experiments/paper1_phylo_calibrated_assignment/CURRENT_RESULTS.md`
- Claim boundaries:
  `experiments/paper1_phylo_calibrated_assignment/CLAIM_BOUNDARIES.md`
- Candidate support:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_union_blast_candidate_support_summary.csv`
- Stable order policy:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_stable_order_policy_summary.csv`
- High-coverage order repair:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_high_coverage_order_repair_summary.csv`
- Rank repair comparison:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_high_coverage_rank_repair_comparison.csv`
- Manuscript-facing tables and figure plan:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/`
- Draft figures and slide-ready summary:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/`
- Slide-ready tables and outline:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/`
- Manuscript captions, results/methods paragraphs, and claim-boundary text:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/`
- Manuscript section outline and readiness checklist:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_section_outline.md`
  and
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_manuscript_section_checklist.csv`
- Family/genus next-evidence plan:
  `results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_family_genus_next_evidence_plan.md`
- Active reference-curation tables:
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_actions.csv`
  and
  `results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_species.csv`
