# Novelty & Prior Art — Coauthor Brief

Last updated: 2026-06-17 (corrected after a verified online literature audit)

This is the canonical "what is new, what others have done, and why it matters"
document — the single source of truth for novelty claims. If any other doc in the
repo disagrees with this one, this one wins.

**Read this first — the honest headline.** A 2026 literature audit (every paper
verified against its primary source) found that **two of our three layers are NOT
individually novel, and one of our "hero" experiments is a replication, not an
innovation.** What *is* novel is narrower and we state it precisely below.
Overclaiming the embedding, the k-mer baseline, or "neural species delimitation"
will not survive review. The defensible novelty is the **open-set DETECT axis**,
the **three-way integration on fish COI**, and the **quantified marker-resolution
boundary** — with embedding priority conceded to Stalder/DEPP and clustering
priority to BarcodeBERT.

---

## 1. The unifying frame (accurate, keep)

We learn a **coordinate system for the tree of life**: a CNN turns a DNA barcode
(COI, ~660bp) into a point in 512-D space, trained so that *distance between
points ≈ distance on the Fish Tree of Life*. Placing, recognising, and
rediscovering the unknown are then all **geometry** in that space. The
contribution is not "we use geometry" — it is the specific *combination* of how
we read this particular geometry (below), not the geometry itself, which is prior
art.

## 2. The problem (accurate)

~8.7M eukaryotic species estimated, ~1.2M described → **~86% of animal life is
unnamed** (Mora et al. 2011, *PLOS Biology*). Most eDNA reads come from taxa
absent from reference databases. The honest gap we occupy: existing methods
(LCA, IDTAXA, PROTAX) already abstain at higher ranks, but the unassigned reads
— which carry real evolutionary signal — are *discarded* rather than *placed and
characterised*.

---

## 3. Per-layer novelty verdict (the corrected core)

### PLACE — tree-distance embedding → **NOT novel. Prior art. Concede.**
- **Stalder et al. 2025** (*PLOS Comp Biol* 21(12):e1013776, DOI
  10.1371/journal.pcbi.1013776) is the **nearest prior art** and a near-miss on
  our exact idea: it trains an embedding where cosine distance ≈ normalised
  phylogenetic distance, maps DNA into it with a neural encoder, uses the **same
  ray-finned-fish tree (31,516 species)**, and **holds out 431 fish species**
  (24% species / 56% genus / 81% family). Differences from our PLACE: **12S eDNA
  not COI**, 64-D not 512-D, separate optimisation rather than one CNN read three
  ways, and **closed-set** (no novelty rejection).
- **DEPP** (Jiang et al., *Syst Biol* 72(1):17, 2023, DOI 10.1093/sysbio/syac031)
  invented the training signal (CNN so distance ≈ √(patristic distance)) in 2023.
- **We must cite Stalder and DEPP prominently and NOT claim the embedding as
  novel.** Our PLACE deltas (fish COI vs 12S; unknown-vs-unknown r=0.86; a
  phylogenetic-leakage audit vs random holdout) are real but modest.

### The k-mer baseline (0.91 learned vs 0.37 raw) → **NOT a novel experiment.**
- **kf2vec** (Rachtman, Jiang & Mirarab, *Mol Ecol Resour* 2025, DOI
  10.1111/1755-0998.70055) **already ran the k-mer-similarity baseline ablation**
  for the same purpose. Our 0.91-vs-0.37 is valid *evidence* but is
  **confirmation of a known result, not an original design.** Do not headline it
  as a "hero." Frame: "consistent with kf2vec, on fish COI."

### DETECT — open-set novelty in a tree-geometry space → **NOVEL. Lead with this.**
- No barcode paper performs open-set novelty detection by **distance in a
  phylogeny-distance-trained embedding**. **Stalder is closed-set** (forces every
  read to a species, no rejection). **Fujisawa & Imai 2026** (*Ecol Evol*
  16(2):e73112, DOI 10.1002/ece3.73112) use classifier softmax/energy confidence,
  not a phylo-embedding. **Open-Insect** (NeurIPS 2025, arXiv:2503.01691) is
  moth *images*, not DNA. This is the cleanest white space; our graded result
  (AUROC 0.84 genus / 0.68 species) is a new framing.

### REDISCOVER — clustering to recover species → **NOT novel, and NOT superior. Measured head-to-head 2026-06-17.**

**Head-to-head run on Vast (species recovery, AMI on the 531 held-out fish):**

| Method | Species | Genus | Family |
|---|---|---|---|
| VSEARCH (identity-threshold OTU clustering) | **0.915** | **0.876** | 0.720 |
| cd-hit (identity-threshold OTU clustering) | 0.886 | 0.847 | 0.692 |
| OUR tree-geometry embedding | 0.874 | 0.859 | **0.756** |
| BarcodeBERT (frozen COI FM) | 0.492 | — | — |

**Verdict:** classical identity-threshold OTU clustering (VSEARCH/cd-hit; proxies for BIN/ABGD-style delimitation, which are distinct workflows)
**beats our embedding at species and genus**. Our embedding wins only at *family*
(it captures coarse tree structure that identity-threshold clustering fragments —
the classical tools over-segment to 1200–1500 clusters for 531 species). Among
neural representations ours is far ahead of BarcodeBERT (0.87 vs 0.49), but
classical methods win overall. **Do not claim rediscovery as a strength.** Frame
it honestly: complementary (sequence identity for fine ranks, tree geometry for
coarse), not superior. An earlier k-mer "proxy" (0.62) badly understated the
classical tools and must not be cited. Sources:
`vsearch_delimitation.json`, `cdhit_delimitation.json`, `barcodebert_rediscovery.json`.

**The "best of both" question is settled — it's a strict tradeoff (measured
2026-06-17).** A loss-weight frontier sweep (tree-distance vs species-contrastive)
shows species-clustering AMI and tree-recovery Pearson move in near-perfect
opposition: keep tree recovery ≥0.85 and species clustering caps at ~0.86; push
species clustering to match/beat VSEARCH (0.92) and tree recovery collapses to
≤0.66. **No single model holds both** — a model that matches VSEARCH at species is
just a species-contrastive model (≈ BarcodeBERT, prior art) that has discarded the
tree geometry that powers DETECT. So the model component is a Pareto frontier, not
a do-everything representation: use the **tree-geometry end for DETECT/placement**
(the novel axis), and treat species clustering as a benchmark we do not win.
Source: `weight_frontier.json`; full benchmark in `REDISCOVERY_BENCHMARK.md`.

(Prior-art context below remains; the measured result above supersedes any claim
that our embedding beats neural/classical delimitation.)
- **BarcodeBERT** (Millán Arias et al., *Bioinformatics Advances* 6(1):vbag054,
  2026, DOI 10.1093/bioadv/vbag054) and **DNABERT-S** (Zhou et al. 2024,
  arXiv:2402.08777) **already cluster learned barcode embeddings to recover
  species on held-out taxa, scored with AMI/ARI** (BarcodeBERT reconstructs BOLD
  BINs at ~80%). **BIOSCAN-5M** (NeurIPS 2024, arXiv:2406.12723) defines the
  AMI-scored zero-shot clustering benchmark.
- Classical species delimitation we must also cite: **BIN/RESL** (Ratnasingham &
  Hebert 2013, DOI 10.1371/journal.pone.0066213), **ABGD** (2012), **ASAP**
  (2021), **GMYC** (2006), **bPTP/mPTP** (2013/2017).
- Our **only** differentiator: the embedding is **phylogeny-distance-trained**,
  not MLM/contrastive. Narrow. If we call this "neural species delimitation," it
  already exists. Frame strictly as "rediscovery from a *tree-geometry* latent
  space," and report both numbers (AMI 0.87 with the count; 0.74 blind /
  under-segments).

### The INTEGRATION + the BOUNDARY → **Novel as a unit.**
- No single paper combines a learned tree-geometry barcode embedding + open-set
  DETECT + unsupervised REDISCOVER under an audited missing-reference split on
  fish COI. Stalder is the closest and covers at most 2 of the 3 axes, closed-set.
- The **quantified marker-resolution boundary** — genus-knowable, species-limited,
  measured consistently across place, detect, and rediscover — is a useful
  finding we have not seen stated this way.

---

## 4. The repositioned thesis (what the paper actually claims)

> We extend the phylogeny-distance barcode-embedding line (Stalder 2025, DEPP
> 2023) to fish COI, and add an **open-set novelty-detection axis that no prior
> barcode method has** — integrated with placement and rediscovery into one
> system, and evaluated under a phylogenetic-leakage audit. Across all three
> readings we quantify a consistent boundary: a short marker makes the **genus
> level knowable and the species level marker-limited**. We concede priority on
> the embedding (Stalder/DEPP), the similarity baseline (kf2vec), and neural
> clustering (BarcodeBERT); our contribution is the open-set axis, the
> integration, and the measured boundary.

That is honest, defensible, and survives review *because* it concedes priority.

---

## 5. Verified results (numbers + source files)

| Layer | Result | Source |
|---|---|---|
| PLACE | tree recovery unseen: 0.914 (vs ref) / 0.864 (vs unseen); raw 6-mer 0.37 | `..._tree_recovery_eval_c/...json`, `phylo_tree_distance_baselines_clean/...json` |
| PLACE | leakage audit empty; negative controls collapse (shuffle 0.0%, random 0.06%) | `fish_tree_clean_splits/manifest.json`, `negative_controls_seed1206_eval_c/` |
| DETECT | AUROC 0.77 overall / 0.84 genus / 0.68 species; bottom-25% → 90% novel | `novelty_detection_rigorous.json` |
| REDISCOVER | species AMI 0.87 (94% pure) with count; 0.74 blind (under-segments) | `rediscovery_clustering.json` |
| Boundary | forcing species unsafe; honest abstention 95.8/93.0/0% false species | `strict_rank_backoff_summary.csv`, `pipeline_mode_policy_summary.csv` |

We do **not** beat BLAST/VSEARCH on retrieval accuracy (≈95/98/99 genus/family/
order) and do not claim to.

---

## 6. Claim boundaries (corrected)

**We will claim:**
- An **open-set novelty axis** for barcodes built on a tree-geometry embedding —
  no prior barcode method does this (AUROC 0.84 genus).
- The **integration** of place + detect + rediscover on fish COI under a
  leakage audit, as one coordinate system.
- A **quantified marker-resolution boundary**: genus-knowable, species-limited,
  consistent across all three readings.

**We will NOT claim:**
- That we invented tree-distance barcode embeddings — **Stalder 2025 and DEPP
  2023 have priority** (Stalder on the same fish tree).
- That the k-mer baseline is a novel experiment — **kf2vec did it**; ours confirms.
- "Neural species delimitation" as novel — **BarcodeBERT / DNABERT-S** cluster
  barcode embeddings to recover species already; ours differs only by tree-training.
- That rediscovery beats prior art — **measured 2026-06-17, classical OTU clustering
  (VSEARCH 0.915, cd-hit 0.886) beat our embedding (0.874) at species**; we
  win only at family. Rediscovery is complementary, not superior.
- Perfect rediscovery — the 0.87 AMI used the true count; blind clustering
  under-segments. Report both.
- That we beat BLAST, that Mamba is best (it's a CNN), or exact Fernando PCP.

---

## 7. Must-cite list (verified DOIs, one-line differentiation)

1. **Stalder et al. 2025**, DOI 10.1371/journal.pcbi.1013776 — *nearest prior
   art*; phylo-distance fish embedding, same tree, held-out species; **12S,
   closed-set, no clustering**. We differ on COI + open-set DETECT + REDISCOVER.
2. **DEPP** — Jiang et al. 2023, DOI 10.1093/sysbio/syac031 — invented neural
   seq→√tree-distance; placement only, microbial.
3. **H-DEPP** (10.3390/biology11091256) / **C-DEPP** (10.1093/bioinformatics/btae361)
   — variants; placement only.
4. **kf2vec** — Rachtman et al. 2025, DOI 10.1111/1755-0998.70055 — already ran
   the k-mer baseline; frame ours as confirmation.
5. **Phyloformer** — Nesterenko et al. 2025, DOI 10.1093/molbev/msaf051 —
   MSA-based distance regression; different mechanism.
6. **BarcodeBERT** — Millán Arias et al. 2026, DOI 10.1093/bioadv/vbag054 —
   zero-shot clustering + AMI + BIN reconstruction on unseen species; differ
   (no tree geometry, insects).
7. **DNABERT-S** — Zhou et al. 2024, arXiv:2402.08777 — contrastive embeddings
   cluster unseen species (ARI); not COI, not phylogenetic.
8. **BIOSCAN-5M** — Gharaee et al. 2024, arXiv:2406.12723 — AMI-scored zero-shot
   clustering benchmark; insects, no phylo embedding.
9. **Fujisawa & Imai 2026** — DOI 10.1002/ece3.73112 — OOD on insect COI via
   classifier confidence; differ (phylo-embedding distance, not softmax/energy).
10. **Open-Insect** — Chen et al., NeurIPS 2025, arXiv:2503.01691 — open-set on
    moth images, not DNA.
11. **BIN/RESL** (10.1371/journal.pone.0066213), **ABGD**
    (10.1111/j.1365-294X.2011.05239.x), **ASAP** (10.1111/1755-0998.13281),
    **GMYC** (10.1080/10635150600852011), **bPTP** (10.1093/bioinformatics/btt499),
    **mPTP** (10.1093/bioinformatics/btx025) — classical delimitation baselines
    REDISCOVER must compare against.
12. **Fernando, Fu & Adamowicz 2025**, PMC11706799 — fish COI placement
    comparator (EPA-ng/APPLES); we run Fernando-*style* sweeps, not exact PCP.
13. **BarcodeMamba** — Gao & Taylor, arXiv:2412.11084 — namesake SSM barcode
    model; cite to avoid name confusion.
14. **Villon et al. 2026** — DOI 10.3390/biology15030285 (*Biology* 15(3):285) —
    *closest motivation-sharing work.* A **closed-set** position-aware CNN softmax
    classifier on 12S teleo amplicons that beats reference-based tools at
    genus/family when the query species is absent, but **cannot abstain, cannot
    detect out-of-class novelty, and reports no calibrated error rate** — a query
    from an unseen family is forced into a known class with a confident score. We
    reframe as calibrated, open-set rank/no-call inference. Must cite and contrast.
15. **Zito, Rigon & Dunson 2023** (BayesANT) — DOI 10.1111/2041-210X.14009
    (*Methods Ecol. Evol.* 14:529–542) — nearest **rank-adaptive / new-taxon
    discovery** prior work: Bayesian species-sampling priors that discover
    unobserved taxa at each rank. Non-DL; not integrated with tree-geometry
    placement, a measured false-species-call rate, or reference-curation.

---

## 8. Honest status

**Measured / verified in hand:** place (tree recovery 0.914; Eval C k-mer baseline
0.375 same-split; shuffled-tree negative control 0.919→0.094), detect (0.77/0.84
genus), rediscover (0.87 with count / 0.74 blind), prospective species-disjoint
calibration (0% false-species survives all 30 repeats), the boundary, abstention,
Fernando-style comparators.

**Still to do (optional only):** OpenMax open-set head — optional, would
strengthen DETECT but is not a blocker. Everything that was previously "to do"
is now **done**: the shuffled-tree control (0.919→0.094) and the Eval C k-mer
baseline (0.375 vs learned 0.914, on the Eval C split) close the place-result
audit; the REDISCOVER comparison vs VSEARCH/cd-hit/BarcodeBERT and the
tree-vs-species frontier are done (`REDISCOVERY_BENCHMARK.md`); **optional** exact
ABGD/ASAP runs only if reviewers demand the specific delimitation workflows.

**Standing:** CNN-not-Mamba naming resolved (repo renamed
`calibrated-rank-assignment`, dropping the architecture name); work is committed
and pushed.

**Audit caveats:** the agent could not rule out an obscure 2025–2026 preprint
doing the exact conjunction; a few differentiations are abstract-level, not
full-text-verified. Confirm the verified DOIs and the "(reported)" competitor
figures against source before the manuscript.

---

## 9. Questions for the coauthor

1. Given Stalder 2025 is on the *same fish tree*, is "extend to COI + add open-set
   + integrate" a strong enough wedge, or do we need a sharper differentiator?
2. Is the **open-set DETECT axis** strong enough to anchor the paper's novelty,
   with place/rediscover as supporting (priority conceded)?
3. Do we add the required head-to-head: REDISCOVER vs BIN/ABGD/ASAP/BarcodeBERT?
4. Venue, given the narrowed novelty: methods/ecology (MEE, Mol Ecol Resour)?
5. Naming: retitle around the science (e.g. "open-set novelty detection in a
   phylogenetic barcode embedding")?
