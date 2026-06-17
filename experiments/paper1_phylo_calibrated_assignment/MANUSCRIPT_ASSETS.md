# Manuscript Assets

This folder tracks writing-facing assets for Paper 1. These are not new
metrics; they are source inventories built from the existing source tables.

## Builder

```bash
python3 scripts/edna/build_paper1_manuscript_assets.py
```

MarkerMirror-specific manuscript package:

```bash
python3 scripts/edna/build_marker_mirror_manuscript_assets.py
```

Output root:

```text
results/paper1_phylo_calibrated_assignment/manuscript_assets/
```

## Outputs

- `figure_plan.csv`: proposed main figures, source tables, readiness, and
  dependencies.
- `table_plan.csv`: proposed main and supplementary tables.
- `claim_evidence_map.csv`: manuscript claims linked to current evidence and
  claim boundaries.
- `pipeline_operating_points.csv`: exact-vector, HNSW, p-distance, and
  species-disabled eDNA posterior operating points from raw, calibrated, and
  threshold-stability rows.
- `missing_results_checklist.csv`: remaining outputs that still block final
  claims.
- `manuscript_asset_manifest.json`: generation metadata.

## MarkerMirror Package

Exp 122 adds a writing-facing MarkerMirror package under:

```text
results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/
```

Files:

- `marker_mirror_candidate_support_table.csv`: MarkerMirror-only,
  same-marker BLASTN/VSEARCH, and union candidate-support rows.
- `marker_mirror_order_policy_table.csv`: conservative and high-coverage
  order/no-call policy rows.
- `marker_mirror_rank_boundary_table.csv`: why family/genus/species are not
  enabled under the current evidence.
- `marker_mirror_runtime_table.csv`: Vast full-wrapper stage timings.
- `marker_mirror_figure_plan.csv`: proposed MarkerMirror figure panels and
  source tables.
- `marker_mirror_methods_blurb.md`: short methods text for manuscript drafting.
- `marker_mirror_manuscript_asset_manifest.json`: generation metadata.

These files reorganize existing source-table results; they are not new
metrics. Use them to draft figures, tables, and methods text while keeping the
claim boundary that MarkerMirror currently supports order/no-call only.

Exp 123 renders draft figures from the Exp 122 package:

```bash
python3 scripts/edna/build_marker_mirror_manuscript_figures.py
```

Figure output root:

```text
results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/figures/
```

Files:

- `marker_mirror_candidate_support_bars.png` / `.pdf`
- `marker_mirror_order_policy_tradeoff.png` / `.pdf`
- `marker_mirror_rank_boundary.png` / `.pdf`
- `marker_mirror_runtime_breakdown.png` / `.pdf`
- `marker_mirror_slide_ready_summary.md`
- `marker_mirror_manuscript_figure_manifest.json`

These are figure drafts and slide-ready text, not final journal artwork.

Exp 124 creates slide-ready tables and a deck outline from the same package:

```bash
python3 scripts/edna/build_marker_mirror_slide_tables.py
```

Output root:

```text
results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/slide_tables/
```

Files:

- `marker_mirror_candidate_support_slide_table.csv` / `.md`
- `marker_mirror_order_policy_slide_table.csv` / `.md`
- `marker_mirror_rank_boundary_slide_table.csv` / `.md`
- `marker_mirror_runtime_slide_table.csv` / `.md`
- `marker_mirror_slide_package_outline.md`
- `marker_mirror_slide_tables_manifest.json`

These are intended for coauthor review or manual slide assembly; they are not a
new benchmark.

Exp 125 creates manuscript text snippets from the same source tables:

```bash
python3 scripts/edna/build_marker_mirror_manuscript_text.py
```

Output root:

```text
results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/
```

Files:

- `marker_mirror_figure_captions.md`
- `marker_mirror_results_paragraph.md`
- `marker_mirror_methods_paragraph.md`
- `marker_mirror_claim_boundary_box.md`
- `marker_mirror_caption_inventory.csv`
- `marker_mirror_manuscript_text_manifest.json`

These snippets are intended for manuscript drafting and coauthor review. They
repackage existing MarkerMirror metrics and do not add new results.

Exp 126 turns the MarkerMirror text, figures, and slide tables into a draft
manuscript section plan:

```bash
python3 scripts/edna/build_marker_mirror_manuscript_section_outline.py
```

Output root:

```text
results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/
```

Files:

- `marker_mirror_manuscript_section_outline.md`
- `marker_mirror_manuscript_section_checklist.csv`
- `marker_mirror_manuscript_section_manifest.json`

The outline proposes where MarkerMirror belongs in Paper 1, which figures and
text snippets to use, and what claim boundaries must remain visible. It is a
writing scaffold, not a new benchmark.

Exp 127 adds a family/genus next-evidence plan from existing source tables:

```bash
python3 scripts/edna/build_marker_mirror_next_evidence_audit.py
```

Key output:

```text
results/paper1_phylo_calibrated_assignment/manuscript_assets/marker_mirror/text/marker_mirror_family_genus_next_evidence_plan.md
```

The plan identifies lineage-specific reference coverage, alignment-backed
marker-resolvability, and active reference-curation value as the next
scientifically different evidence sources. It is a planning/source-table audit,
not an enabled family/genus result.

Exp 128 tests the first follow-up from that plan:

```bash
python3 scripts/edna/build_marker_mirror_reference_coverage_policy_diagnostic.py
```

Key outputs:

```text
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_summary.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_per_split.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_reference_coverage_policy_diagnostic_manifest.json
```

This is a manuscript-boundary result: lineage/reference-coverage features alone
do not produce stable family/genus transfer, so they should be discussed as a
negative diagnostic and motivation for marker-resolvability or sample-aware
evidence, not as an enabled rank/no-call result.

Exp 129 adds the VSEARCH-backed marker-resolvability replacement for the older
rare-kmer proxy:

```text
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_summary.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_query_oracle_rates.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_manifest.json
```

Use this as marker-ceiling evidence in the manuscript. At 0.99 identity, 12S
query oracle support is 77.9 / 95.2 / 99.6 / 99.7% for
species/genus/family/order, but only 19.6% of query clusters contain a current
reference. It should not be written as an enabled family/genus policy.

Exp 130 tests the production-available part of that VSEARCH evidence as policy
features:

```text
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_summary.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_vsearch_resolvability_policy_diagnostic_manifest.json
```

This is also a boundary result. It did not stabilize family/genus transfer, so
the manuscript should keep MarkerMirror's enabled output at order/no-call.

Exp 131 adds the active reference-curation/value-of-information package:

```text
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_species.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_lineage.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_actions.csv
results/paper1_phylo_calibrated_assignment/source_tables/marker_mirror_active_reference_value_manifest.json
```

Use this as the manuscript-facing answer to "what does the system do when it
cannot safely identify lower ranks?" It ranks reference-curation and evidence
collection actions. It should not be presented as a family/genus/species
accuracy result.

Exp 132 adds a coauthor-facing paper storyline:

```text
experiments/paper1_phylo_calibrated_assignment/PAPER_STORYLINE.md
```

Use it to align the manuscript narrative before drafting: COI establishes the
rank/no-call principle, MarkerMirror demonstrates the 12S/16S evidence
extension, and active curation explains what the system does after abstention.

## Current Use

Use these files when drafting the paper outline and figure list. Strict
missing-reference validation and Fernando-style EPA-ng/APPLES completeness
sweeps have now landed, so refresh these assets before treating any readiness
flags as final.

The current strongest writing-ready points are:

- classical sequence methods are very strong when close references exist;
- MarkerMirror + BLASTN/VSEARCH gives a strong 12S high-rank candidate set:
  BLASTN union top50 support is 9.5 / 92.1 / 95.3 / 99.7% for
  species/genus/family/order, and VSEARCH union top50 support is
  9.5 / 91.8 / 95.1 / 99.6%;
- vector-first retrieval is fast enough to be useful as a candidate generator;
- calibrated rank/no-call output can reduce false species calls by backing off
  to genus/family/order;
- p-distance reranking is useful only when calibrated separately;
- 12S/eDNA species-level claims need marker-resolvability and ecological
  evidence decomposition;
- the current full sequence+tree Eco-Phylo posterior supports conservative
  species-disabled genus/family/order eDNA calls, with target-95 assigning about
  40% of held-out queries at about 94% accuracy.

Do not treat `figure_plan.csv` or `claim_evidence_map.csv` as final manuscript
claims. They are planning artifacts and must be refreshed after each new
source-table update, especially final figure/table selection and target-host
end-to-end speed results.
