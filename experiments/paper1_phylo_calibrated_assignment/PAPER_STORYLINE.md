# Paper 1 Storyline

> **NOVELTY CORRECTION (2026-06-17):** Novelty claims here are superseded by
> `NOVELTY_AND_PRIOR_ART.md` (corrected after a verified literature audit). The
> tree-distance embedding is **prior art** (Stalder 2025 — same fish tree; DEPP
> 2023); the k-mer baseline is a **replication** (kf2vec 2025); neural clustering
> of barcode embeddings **already exists** (BarcodeBERT 2026, DNABERT-S,
> BIN/ABGD/ASAP). Genuine novelty = the **open-set DETECT axis**, the **3-way
> integration on fish COI**, and the **marker-resolution boundary**.

## Core Thesis

Barcode and eDNA assignment should not be treated as forced species
classification. A defensible biodiversity system should ask:

> What is the deepest taxonomic rank supported by the available molecular,
> tree, reference, marker-resolvability, and context evidence?

The calibrated-rank-assignment pipeline (legacy codename *MarineMamba*) is the
broader evidence-aware barcode/eDNA pipeline. MarkerMirror is the 12S/16S
marker-bridging branch inside that pipeline.

## One-Paragraph Story

Short molecular markers often contain real biodiversity signal, but that signal
is uneven: the true species may be absent from the reference database, the
marker may not separate close species, and classical sequence similarity may
only justify a broader rank. We build a pipeline that retrieves candidate taxa
quickly, checks them with classical sequence methods, adds tree-aware and
marker-aware evidence, calibrates species/genus/family/order/no-call decisions,
and reports why a query was assigned or withheld. The contribution is not a
single neural model replacing BLAST. The contribution is an uncertainty-aware
evidence compiler that produces the deepest defensible taxonomic claim and
identifies which missing reference data would most improve unresolved cases.

## Narrative Arc

### 1. The Problem

Species labels are often overconfident when references are missing or markers
are ambiguous. The paper opens by arguing that a no-call or broader-rank call
can be more scientific than a forced species label.

### 2. The Pipeline

```text
query sequence
  -> fast candidate retrieval
  -> BLAST/VSEARCH/classical sequence checks
  -> tree-aware or marker-bridging evidence
  -> reference-gap and marker-resolvability diagnostics
  -> calibrated species/genus/family/order/no-call
  -> reason code and active-curation priority
```

The key design principle is evidence separation: candidate generation, sequence
similarity, tree signal, marker ceiling, and calibration are measured
separately before being joined into a final decision.

### 3. COI Establishes The Rank/No-Call Principle

COI is the information-rich marker used to show the main pipeline behavior:
candidate retrieval is fast, classical tools remain strong, tree-aware evidence
is useful, and missing-reference stress tests require rank backoff. The claim is
not just top-1 accuracy. The claim is controlled precision and honest
abstention when a species call is not supported.

### 4. MarkerMirror Shows Why Evidence Integration Matters

12S is harder than COI. MarkerMirror learns a 12S-to-16S bridge, then combines
that bridge with same-marker BLASTN and VSEARCH. On 3,566 12S queries:

| Candidate source | Species | Genus | Family | Order |
|---|---:|---:|---:|---:|
| MarkerMirror 12S->16S only | 9.5% | 39.9% | 59.8% | 76.3% |
| MarkerMirror + BLASTN union | 9.5% | 92.1% | 95.3% | 99.7% |
| MarkerMirror + VSEARCH union | 9.5% | 91.8% | 95.1% | 99.6% |

This is the cleanest MarkerMirror story: the learned bridge alone is not a
species identifier, but it contributes cross-marker evidence; classical
alignment contributes strong same-marker evidence; the union gives a strong
high-rank candidate set.

### 5. The Output Is Honest: Order/No-Call

MarkerMirror currently supports order/no-call, not family/genus/species.

| Mode | Output | Coverage | Precision | Boundary |
|---|---:|---:|---:|---|
| Conservative all-source order agreement | 880 / 3,566 calls | 24.7% | 99.7% | Default-safe research mode |
| High-coverage BLASTN/VSEARCH order diagnostic | 57.2% held-out coverage | 57.2% | 99.8% | Explicit order-only diagnostic |

Family/genus/species remain disabled because threshold repair, learned
policies, set-valued outputs, lineage/reference coverage, and VSEARCH-cluster
features did not transfer cleanly at target-0.99.

### 6. The Pipeline Explains What Evidence Is Missing

The active-curation layer turns unresolved cases into actionable reference
priorities. It ranks 795 species groups by value-of-information. The largest
action category is adding both 12S and 16S references for 532 species groups
covering 1,928 queries. Other cases need same-marker 12S references, target 16S
curation, or additional marker/context evidence.

This is a strong endpoint for the story: the system does not merely abstain; it
explains what data would most improve the claim.

## Suggested Paper Framing

Possible title:

> Evidence-aware barcode and eDNA assignment with calibrated rank backoff under
> missing molecular references

Possible subtitle:

> A calibrated-rank-assignment pipeline integrating learned retrieval, classical
> alignment, tree evidence, and active reference curation.

## Main Claim Set

Claim-ready:

- Fast vector retrieval is useful as a candidate generator, but speed alone is
  not the scientific contribution.
- Classical sequence tools remain essential and should be retained.
- Missing-reference tests show why forced species labels are unsafe.
- Calibrated rank/no-call inference is a better target than raw species top-1.
- MarkerMirror + BLASTN/VSEARCH gives strong high-rank 12S candidate support.
- MarkerMirror can make high-precision order/no-call decisions under the
  current benchmark and explain unresolved cases with active curation
  priorities.

Not claim-ready:

- Species-level eDNA identification is solved.
- MarkerMirror replaces BLASTN or VSEARCH.
- Family/genus MarkerMirror calls are production-ready.
- Current 12S tests are field-eDNA deployment validation.
- Current Fernando-style diagnostics are exact Fernando PCP.

## Figure Sequence

1. Pipeline schematic:
   sequence -> retrieval -> classical checks -> tree/marker/context evidence ->
   rank/no-call -> reason code/curation priority.
2. COI benchmark:
   rank-aware performance and missing-reference rank backoff.
3. Vector/candidate retrieval:
   speed and top-k candidate support.
4. MarkerMirror candidate support:
   MarkerMirror-only versus BLASTN/VSEARCH versus union.
5. MarkerMirror order/no-call:
   conservative and high-coverage operating points.
6. Active curation:
   top reference/evidence actions and example species groups.

## Manuscript Positioning

The paper should be written as an evidence-integration contribution. Neural
models are useful components, but the defensible novelty is the full inference
behavior:

- recover candidates quickly;
- test candidates with strong classical methods;
- represent tree/marker evidence when helpful;
- calibrate the taxonomic rank;
- abstain when the evidence is insufficient;
- report what reference data would improve the next attempt.

That story is stronger than claiming a new species classifier.
