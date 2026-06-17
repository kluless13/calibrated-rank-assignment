# Calibrated Rank-Adaptive Biodiversity Inference from DNA Barcodes

*Returning the deepest defensible taxonomic rank — or abstaining — under missing
and ambiguous references.*

When most of life is undescribed and most eDNA reads come from under-referenced
taxa, a forced species label is often a lie with a confidence score. This
repository builds and evaluates an evidence-compiler pipeline that retrieves
candidates, checks them with classical and tree-aware evidence, recognises
novelty, and returns the **deepest defensible rank (species / genus / family /
order) or an explicit no-call** with a measured false-species-call rate — then
converts abstentions into reference-curation priorities.

> Developed under the legacy working name *MarineMamba*; the project and
> repository are now **calibrated-rank-assignment**, framed around the science
> (encoder-agnostic — the headline encoder is a CNN — and marker- and
> taxon-general), not a single architecture.

## Thesis

Biodiversity assignment from short DNA barcodes should be **calibrated,
rank-adaptive inference under missing references** — not forced species
classification. The scientific output is the deepest *defensible* claim plus an
honest "I don't know," not a top-1 species name.

## The pipeline

```text
barcode sequence
  -> fast vector candidate retrieval
  -> classical sequence checks (BLAST / VSEARCH / p-distance)
  -> tree-aware evidence + open-set novelty (DETECT)
  -> reference-gap + marker-resolvability diagnostics
  -> ecological prior (for eDNA; geography / co-occurrence)
  -> calibrated species / genus / family / order / no-call
  -> reason codes + active reference-curation priorities
```

Design principle: **evidence separation** — each stream (retrieval, sequence
similarity, tree/DETECT, marker ceiling, ecology) is measured independently
before being fused into one calibrated decision.

## Headline results (verified)

All on audited, leakage-free splits (species held out before training).

| Result | Number | Source |
|---|---|---|
| Rank/no-call operating point (Eval C) | 95.8% coverage, 93.0% precision, **0% false-species** | `pipeline_mode_policy_summary.csv` |
| Rank/no-call (unseen genera) | 92.3% / 83.9% / **0%** | `pipeline_mode_policy_summary.csv` |
| **Prospective species-disjoint calibration** | 0.923 cov, 0.900 prec, **0% false-species survives all 30 repeats** | `independent_calibration_split.json` |
| Tree-geometry placement (tree recovery) | Pearson **0.914** | `weight_frontier.json` |
| Place audit — k-mer baseline (same split) | 0.375 (learned 2.44× better) | `eval_c_kmer_tree_baseline.json` |
| Place audit — shuffled-tree negative control | 0.919 → **0.094** (signal is real) | `shuffled_tree_control.json` |
| Open-set novelty (DETECT) | AUROC 0.77 / **0.84** genus | `novelty_detection_rigorous.json` |
| Rediscovery head-to-head (species AMI) | VSEARCH 0.915 / cd-hit 0.886 / ours 0.874; **ours wins family 0.756** | `vsearch_delimitation.json`, … |
| Fast retrieval | 0.40 ms exact, 0.0048 ms HNSW / query | `controlled_vector_speed_benchmark.csv` |

Figures for these are tracked under
[`results/paper1_phylo_calibrated_assignment/manuscript_assets/experiment1/figures/`](results/paper1_phylo_calibrated_assignment/manuscript_assets/experiment1/figures/)
(place-audit controls, rediscovery head-to-head, tree-vs-species frontier,
prospective calibration). Regenerate with
[`scripts/figures/plot_experiment1_figures.py`](scripts/figures/plot_experiment1_figures.py).

## What is novel vs conceded (honest positioning)

**Novel / defensible:** the integration (one calibrated rank/no-call system over
multiple evidence streams under audited missing-reference splits); **open-set
novelty detection in a tree-geometry embedding (DETECT)**; missing-reference as a
first-class evaluation regime; active reference-curation; a measured
false-species-call rate.

**Conceded / supporting (cited, not claimed):** the tree-distance embedding
(Stalder 2025, DEPP 2023); vector barcode retrieval (TaxoTagger/LISA/BarcodeBERT);
classical methods (BLAST/VSEARCH — kept as strong baselines; they win at species
clustering); species rediscovery (classical wins — an honest benchmark, not a win).

See [NOVELTY_AND_PRIOR_ART.md](experiments/paper1_phylo_calibrated_assignment/NOVELTY_AND_PRIOR_ART.md)
for the full prior-art audit and concessions.

## Documentation

All paper docs live under
[`experiments/paper1_phylo_calibrated_assignment/`](experiments/paper1_phylo_calibrated_assignment/).

**Start here**

- [Plain-language overview](experiments/paper1_phylo_calibrated_assignment/COAUTHOR_PLAIN_LANGUAGE.md) — the whole project explained jargon-free (read this first)
- [Experiment-1 scope](experiments/paper1_phylo_calibrated_assignment/EXPERIMENT_1_PIPELINE_SCOPE.md) — authoritative scope, verified-vs-todo, decisions
- [Coauthor brief](experiments/paper1_phylo_calibrated_assignment/COAUTHOR_BRIEF.md) — full results brief and open questions
- [Novelty & prior art](experiments/paper1_phylo_calibrated_assignment/NOVELTY_AND_PRIOR_ART.md) — what's novel vs conceded, with citations
- [Claim boundaries](experiments/paper1_phylo_calibrated_assignment/CLAIM_BOUNDARIES.md) — what we do and do not claim
- [Paper storyline](experiments/paper1_phylo_calibrated_assignment/PAPER_STORYLINE.md) — the narrative arc

**Pipeline & results**

- [Pipeline architecture](experiments/paper1_phylo_calibrated_assignment/PIPELINE.md)
- [Current results](experiments/paper1_phylo_calibrated_assignment/CURRENT_RESULTS.md)
- [Rediscovery benchmark](experiments/paper1_phylo_calibrated_assignment/REDISCOVERY_BENCHMARK.md) — classical vs neural vs frontier
- [Comparator matrix](experiments/paper1_phylo_calibrated_assignment/COMPARATOR_MATRIX.md)
- [Encoder benchmarks](experiments/paper1_phylo_calibrated_assignment/ENCODER_BENCHMARKS.md)
- [Source-tables index](experiments/paper1_phylo_calibrated_assignment/SOURCE_TABLES.md)

**Manuscript**

- [Manuscript assets](experiments/paper1_phylo_calibrated_assignment/MANUSCRIPT_ASSETS.md)
- [Merged manuscript outline](experiments/paper1_phylo_calibrated_assignment/MERGED_MANUSCRIPT_OUTLINE.md)

**Positioning & prior art**

- [Literature review](docs/LITERATURE_REVIEW.md)
- [Fernando 2025 positioning](experiments/paper1_phylo_calibrated_assignment/FERNANDO_2025_POSITIONING.md)
- [DEPP / H-DEPP notes](experiments/paper1_phylo_calibrated_assignment/DEPP_HDEPP_NOTES.md)

**Production**

- [Production pipeline v1](experiments/paper1_phylo_calibrated_assignment/PRODUCTION_PIPELINE_V1.md)
- [Production CLI v1](experiments/paper1_phylo_calibrated_assignment/PRODUCTION_CLI_V1.md)

**12S / cross-marker (Experiment 2 track)**

- [MarkerMirror coauthor one-pager](experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_COAUTHOR_ONE_PAGER.md)
- [MarkerMirror prospect](experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_PROSPECT.md)
- [MarkerMirror 12S CLI](experiments/paper1_phylo_calibrated_assignment/MARKER_MIRROR_12S_CLI.md)

## Repository layout

- `scripts/` — COI data prep, baselines, curriculum models, phylogenetic runs, result summaries.
- `scripts/edna/` — 12S/eDNA ingestion, leakage-free split building, calibration, audit controls, head-to-head, and DNA-to-tree training.
- `scripts/figures/` — source-table and figure generation (`plot_experiment1_figures.py`).
- `experiments/` — paper/workstream folders, run wrappers, comparator notes, claim-boundary docs.
- `configs/runs/` — dated run ledgers (commands, inputs, outputs, status) that survive cleanup of `data/`/`results/`.
- `docs/` — audits, literature review, dataset and interpretation notes.
- `data/`, `results/` — local workspaces; large artifacts ignored. Curated small evidence tables/figures for Experiment 1 are tracked under `results/.../source_tables/` and `manuscript_assets/experiment1/`.

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exact commands and output paths live in the dated ledgers under `configs/runs/`.
Some workflows additionally require BLAST/VSEARCH, CUDA-enabled PyTorch, or remote
GPU instances; those are noted in the relevant script help text and ledgers.

## Status

The pipeline is **research-complete and de-risked**: every component runs and is
measured, and the two hard rigor blockers (prospective species-disjoint
calibration and the place-result audit controls) are cleared. Remaining work is
the **Experiment-1 manuscript draft** plus optional add-ons (conformal-prediction
layer, CLI polish). 12S/eDNA species-level is intentionally a reported boundary
(marker ceiling), and MarkerMirror cross-marker bridging is deferred to
Experiment 2.

## License

MIT
