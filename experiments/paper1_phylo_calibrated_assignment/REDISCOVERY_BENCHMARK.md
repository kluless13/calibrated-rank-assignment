# Rediscovery Benchmark — where learned barcode representations sit

Last updated: 2026-06-17. Supporting section for the Experiment-1 (pipeline) paper.

This is an honest, measured benchmark of **unsupervised species rediscovery** from
barcode embeddings: cluster reads from species the model never saw, and check
whether clusters recover the true taxa. It compares classical alignment tools,
our tree-geometry embedding, multi-task variants, and a neural foundation model —
and characterizes the tree-vs-species tradeoff. The takeaway is a finding, not a
win. Neural representations *can* match — even slightly beat — VSEARCH at species
clustering (species-heavy models reach 0.92–0.93 vs VSEARCH 0.915), but **doing so
costs the tree-space signal needed for DETECT/placement**: the same models'
tree-recovery collapses to ~0.57–0.66. So the claim is not "neural cannot match
classical clustering" — it is that matching it sacrifices our distinctive
tree-geometry value. Classical identity-threshold OTU clustering remains the
gold standard for fine-scale species clustering specifically.

All numbers measured on the **531 held-out fish species** (11,594 Eval C reads,
audited leakage-free). AMI = adjusted mutual information (0 random, 1 perfect).

---

## 1. Head-to-head (species recovery)

| Method | Species AMI | Genus | Family |
|---|---|---|---|
| **VSEARCH** 2.27 (identity-threshold OTU clustering) | **0.915** | **0.876** | 0.720 |
| **cd-hit** 4.8.1 (identity-threshold OTU clustering) | 0.886 | 0.847 | 0.692 |
| our tree-geometry embedding (KMeans k=species) | 0.874 | 0.859 | **0.756** |
| BarcodeBERT (frozen COI foundation model) | 0.492 | — | — |
| ~~raw k-mer proxy~~ (flawed — retired) | 0.618 | — | — |

**Findings:**
- **Classical alignment clustering wins at species and genus.** VSEARCH (0.915) and
  cd-hit (0.886) beat our embedding (0.874) at the fine ranks — species delimitation
  is fundamentally "are two barcodes near-identical?", which alignment measures
  directly and precisely.
- **Our embedding wins at family** (0.756 vs 0.72/0.69) — it captures coarse tree
  structure that identity-threshold clustering fragments (the classical tools
  over-segment to 1200–1500 clusters for 531 true species).
- **Our embedding far exceeds the neural foundation model** (0.87 vs BarcodeBERT
  0.49) — but note BarcodeBERT is invertebrate-trained (out-of-domain on fish).
- The earlier k-mer "proxy" (0.618) badly understated classical methods and is
  retired; do not cite it.

Sources: `vsearch_delimitation.json`, `cdhit_delimitation.json`,
`barcodebert_rediscovery.json`, `multitask_rediscovery.json`.

---

## 2. Tree-vs-species frontier (the "best of both" question)

Can a single learned representation match VSEARCH at species clustering *and* keep
the tree geometry that powers open-set DETECT? We swept the loss weighting
(tree-distance vs species-contrastive) and measured both. Answer: **no — strict
tradeoff.**

| Weighting (tree:species) | Species AMI (blind) | Tree-recovery Pearson |
|---|---|---|
| tree-only (cosine) | 0.799 | **0.919** |
| 1.0 : 0.0 (pure tree) | 0.787 | 0.907 |
| 1.0 : 0.1 (tree-heavy) | 0.859 | 0.842 |
| 1.0 : 0.3 (tree-leaning) | 0.897 | 0.743 |
| 1.0 : 1.0 (balanced) | 0.918 | 0.663 |
| species-heavy hybrid | 0.928 | 0.590 |
| species-only contrastive | 0.923 | 0.574 |

The two metrics move in near-perfect opposition. **No weighting holds both** (target
was species ≥0.89 AND tree ≥0.85):
- Keep tree recovery ≥0.85 → species clustering caps at ~0.86 (below VSEARCH).
- Match/beat VSEARCH at species (≥0.915) → tree recovery collapses to ≤0.66.

A model that matches VSEARCH at species is just a species-contrastive model
(≈ BarcodeBERT, prior art) that has discarded the tree geometry. **The model
component is a Pareto frontier, not a do-everything representation.**

Source: `weight_frontier.json`, `frontier_embeddings/`.

---

## 3. What this means for the paper

- **Species rediscovery is not where we win.** Classical alignment clustering is the
  gold standard; matching it requires becoming a prior-art-style species model and
  abandoning our distinctive tree geometry. We report this honestly.
- **The model's distinctive value is the tree-geometry end** (tree recovery 0.92):
  higher-rank structure, evolutionary placement, and **open-set novelty detection
  (DETECT)** — the genuinely novel axis. That is the model component inside the
  pipeline.
- **The frontier itself is a clean finding** — the species-discrimination vs
  tree-distance tension in learned barcode embeddings, quantified — and makes a
  strong supporting figure.

---

## 4. Methods & honest caveats

- **Splits:** Eval C, 531 species held out before training (audited, zero overlap).
- **Classical tools:** VSEARCH 2.27 `--cluster_fast` and cd-hit-est 4.8.1, at 97/98/
  99% identity (best reported); apt-installed (a static VSEARCH binary SIGFPE'd on
  the Blackwell CPU; N-padding stripped before clustering).
- **BarcodeBERT:** frozen `bioscan-ml/BarcodeBERT`, mean-pooled; invertebrate-trained
  (out-of-domain on fish — a real caveat).
- **DNABERT-S:** could not run faithfully — its hardcoded Triton flash-attention
  kernel won't compile on the Blackwell GPU and asserts CUDA on CPU; patching would
  risk computing different embeddings. Cite its published AMI numbers instead.
- **Calibration caveat:** both the neural blind thresholds and VSEARCH's identity
  thresholds are best-of-sweep, not independently calibrated. For a *locked* claim
  the threshold must be chosen on species-disjoint calibration and shown to
  transfer; the comparison here is best-vs-best, which is fair but not prospective.
- **Multi-task models** reuse the retrieval-DL sweep CNNs and a fixed seed; the
  frontier sweep used 20 epochs per weighting (sufficient for the trend).
