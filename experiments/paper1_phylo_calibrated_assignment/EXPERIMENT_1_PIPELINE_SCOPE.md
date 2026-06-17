# Experiment 1 — Scope: the calibrated rank-adaptive assignment pipeline

Last updated: 2026-06-17. This is the authoritative scope for the first paper.
It supersedes the model-first framing we explored (place/detect/rediscover as a
standalone model paper). Where `PAPER_STORYLINE.md`, `MERGED_MANUSCRIPT_OUTLINE.md`,
or `COAUTHOR_BRIEF.md` differ on scope, this doc wins.

---

## Why the pivot (model → pipeline)

We spent significant effort testing whether the *model* could be the headline. The
measured verdict, after this session's experiments:

- **Placement** (tree-distance embedding) is **prior art** — Stalder 2025 (same
  fish tree) and DEPP 2023 own it.
- **Rediscovery** is **not a win** — real alignment tools (VSEARCH 0.915, cd-hit
  0.886) beat our embedding at species clustering; matching them requires becoming
  a prior-art-style species model that discards our tree geometry (strict tradeoff,
  `REDISCOVERY_BENCHMARK.md`).
- Only the **open-set DETECT axis** is cleanly novel as a model result.

A standalone model paper is therefore thin — one novel axis. But the **pipeline**
that wraps the model is a fuller, more differentiated, more *useful* contribution,
and it is the work that is genuinely ours. So the pipeline is Experiment 1; the
tree-geometry model + DETECT is the model component inside it.

---

## Thesis

> Biodiversity assignment from short DNA barcodes should be **calibrated,
> rank-adaptive inference under missing references** — not forced species
> classification. We present an evidence-compiler pipeline that retrieves
> candidates, checks them with classical and tree-aware evidence, recognises
> novelty, returns the **deepest defensible rank (or abstains)** with a measured
> false-species-call rate, explains *why*, and converts abstentions into
> **reference-curation priorities**.

The honest core: when 86% of animal life is undescribed and most eDNA reads come
from under-referenced taxa, a forced species label is often a lie with a
confidence score. The scientific output is the deepest *defensible* claim plus an
explicit "I don't know," not a top-1 species name.

---

## The pipeline

```text
barcode sequence
  -> fast vector candidate retrieval
  -> classical sequence checks (BLAST / VSEARCH / p-distance)
  -> tree-aware evidence + open-set novelty (DETECT)
  -> reference-gap + marker-resolvability diagnostics
  -> ecological prior (for eDNA; geography / co-occurrence)
  -> calibrated species / genus / family / order / no-call
  -> reason codes
  -> active reference-curation priorities
```

Design principle: **evidence separation** — each stream (retrieval, sequence
similarity, tree/DETECT, marker ceiling, ecology) is measured separately before
being fused into a calibrated decision.

---

## What is the contribution (novel vs conceded)

**Novel / defensible (lead with these):**
1. **The integration** — one calibrated rank/no-call system over multiple evidence
   streams, evaluated under audited missing-reference splits. Unclaimed as a unit.
2. **Open-set novelty detection in a tree-geometry embedding** (DETECT, AUROC 0.84
   genus / 0.68 species) — no prior barcode method does this.
3. **Missing-reference as a first-class evaluation regime** — strict hide-species/
   genus/family stress tests; rare in the neural barcode literature.
4. **Active reference-curation / value-of-information** — converting abstention
   into "which references to sequence next." Rare.
5. **Calibrated rank/no-call with a measured false-species-call rate** — the
   practical contribution that makes the tool defensible for biodiversity use.

**Conceded / supporting (cite, do not claim):**
- The tree-distance embedding (Stalder/DEPP).
- Vector barcode retrieval (TaxoTagger/LISA/BarcodeBERT).
- Classical methods (BLAST/VSEARCH) — kept as strong baselines; we do **not** claim
  to beat them, and at species clustering they win.
- Species rediscovery (classical wins — included only as an honest benchmark).

---

## Scope: what is IN Experiment 1, what is OUT

**IN — the focused paper:**
- The COI calibrated rank-adaptive pipeline (the strongest, most verified track):
  retrieval + classical + tree/DETECT evidence + rank/no-call + reason codes.
- Missing-reference stress tests (the regime that justifies abstention).
- Classical + placement comparators (BLAST/VSEARCH; EPA-ng/official APPLES,
  Fernando-style sweeps).
- The DETECT axis (open-set novelty) as the novel model result.
- Active reference-curation.
- The rediscovery benchmark (`REDISCOVERY_BENCHMARK.md`) as a supporting section.
- **12S/eDNA as the marker-ceiling stress test** — the honest point that 12S often
  cannot resolve species, which *motivates* rank-adaptive inference. Reported as a
  boundary (genus/family reachable, species marker-limited), not a species win.

**OUT — deferred to Experiment 2 (or a tightly-scoped extension subsection):**
- **MarkerMirror** cross-marker bridging (12S↔16S) — promising but lands at
  order/no-call; a second-paper centerpiece.
- The full learned **Eco-Phylo posterior** at scale and learned co-occurrence
  reranking — real but the weakest-finished part (species unsolved; nested
  stability not locked).
- The production API/web packaging (research CLI is enough for Experiment 1).

Rationale: the over-scope risk is real (COI + 12S + MarkerMirror + posterior +
curation is 5–6 papers). Experiment 1 is the *calibrated COI pipeline + DETECT +
curation*, with 12S as the marker-ceiling motivation. Everything heavier is
Experiment 2.

---

## Verified vs to-do

**Verified / measured (in hand):**
- Rank/no-call operating point: Eval C 95.8% coverage / 93.0% precision / **0%
  false species calls**; unseen-genera 92.3 / 83.9 / 0% (`pipeline_mode_policy_summary.csv`).
- **Prospective species-disjoint calibration (the submission blocker — now CLEARED):**
  fit per-rank thresholds on a calibration species set, apply to a *disjoint*
  evaluation set, 30 repeats → coverage 0.923, assigned precision 0.900, and the
  **0% false-species-call rate SURVIVES every repeat** (range 0–0). Mean ranks land
  at genus/family with explicit no-call, never a false species. The headline
  operating point transfers to species the thresholds never saw
  (`independent_calibration_split.json`).
- Strict missing-reference collapse (hidden rank → 0, broader survives).
- DETECT: AUROC 0.77 / 0.84 genus (`novelty_detection_rigorous.json`).
- Tree recovery 0.91 (place), now with **both place-audit controls in hand**:
  Eval C k-mer baseline 0.375 (learned 2.44× better, same split,
  `eval_c_kmer_tree_baseline.json`) and a **shuffled-tree negative control** —
  retraining on a permuted tree collapses true-tree recovery from 0.919 to 0.094
  (`shuffled_tree_control.json`). The 0.91 is genuine evolutionary signal, not a
  sequence-similarity artifact.
- Fernando-style EPA-ng + official APPLES (30 sweeps); vector retrieval speed.
- Active-curation map (532 species value-of-information).
- Rediscovery benchmark (classical vs neural vs frontier).

**To-do before submission (mostly CPU/cheap):**
1. ~~**Independent calibration split**~~ — **DONE** (the single most important rigor
   fix; 0% false-species survives prospective species-disjoint calibration, above).
2. ~~**Shuffled-tree control** + Eval C k-mer baseline~~ — **DONE** (place-result
   audit closed, above).
3. **Conformal prediction** over the taxonomy (optional but high-credibility —
   turns calibrated thresholds into coverage guarantees).
4. **Resolve the encoder identity** — is the place/DETECT model CNN or Mamba?
   Frame encoder-agnostic; name the paper around the science, not "MarineMamba."
5. Raw-FASTA production CLI polish (nice-to-have).

The two hard rigor blockers (1 + 2) are cleared. What remains is optional
(conformal), framing (encoder name), or polish (CLI) — none block a first draft.

---

## Claim boundaries (carry into the manuscript)

**Claim:** calibrated rank-adaptive assignment under missing references; open-set
novelty in a tree embedding; a measured false-species-call rate; active curation;
an honest benchmark of where learned representations sit vs classical tools.

**Do NOT claim:** that we invented tree embeddings (Stalder/DEPP), vector retrieval
(TaxoTagger/LISA), or rank abstention (PROTAX/IDTAXA); that we beat BLAST/VSEARCH
(we don't, at species); that rediscovery is a win; that 12S species-level eDNA is
solved; that Mamba is best; exact Fernando PCP reproduction.

---

## Suggested manuscript structure

1. **Problem** — forced species labels are unsafe under missing references; 86%
   undescribed; the deepest defensible rank (or no-call) is the right target.
2. **Pipeline** — the evidence-compiler architecture (figure).
3. **COI rank/no-call** — the operating point, missing-reference stress, classical
   + placement comparators. The core result.
4. **DETECT** — open-set novelty in the tree embedding (the novel model axis).
5. **Marker ceiling (12S)** — why rank-adaptive is necessary; resolvability limits.
6. **Active reference-curation** — abstention → sequencing priorities.
7. **Benchmark** — where learned representations sit (rediscovery + frontier);
   classical remains gold standard for fine clustering; honest positioning.
8. **Discussion** — conceded priorities, the marker boundary, future (Experiment 2:
   MarkerMirror, Eco-Phylo, conformal).

---

## Decisions made (2026-06-17, lead)

1. **12S/eDNA: in scope as a marker-ceiling stress test**, not a full MarkerMirror
   paper. It motivates rank-adaptivity (12S often can't resolve species) and is
   reported as a boundary. MarkerMirror + full Eco-Phylo posterior are Experiment 2.
2. **Independent calibration is a submission BLOCKER**, not revision-only — the
   prospective species-disjoint calibration must be in the first submission.
3. **Title: drop "MarineMamba" from the paper title**; keep it as the software/
   pipeline name only. Working title around the science, e.g. *"Calibrated
   rank-adaptive biodiversity inference from DNA barcodes under missing references."*
4. **Venue: Molecular Ecology Resources** if the pipeline/tool angle stays central
   (likely target); MEE if it drifts more general-methods.

## Remaining open for the coauthor

- Final manuscript framing/emphasis once they review the scope.
- Whether to add the optional conformal layer before or after submission.
