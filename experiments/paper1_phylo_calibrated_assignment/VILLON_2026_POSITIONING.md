# Positioning vs. Villon et al. (2026)

Standalone analysis of the closest motivation-sharing paper, for the coauthor and
for the manuscript's Related Work (§2). Every bibliographic and method detail here
was verified against the primary source (PMC12897059).

## Citation

> Villon, S., Mangeas, M., Berteaux-Lecellier, V., Vigliola, L. & Lecellier, G.
> (2026). Fine-Grained Assignment of Unknown Marine eDNA Sequences Using Neural
> Networks. *Biology (Basel)* 15(3), 285. https://doi.org/10.3390/biology15030285
> (PMID 41677756; PMCID PMC12897059; published 2026-02-05).

## What they did

- **Task:** assign short **12S TELEO** marine-fish eDNA amplicons to genus and
  family when the query *species* is absent from the training set.
- **Method:** a **closed-set**, position-aware CNN with a softmax head
  (five heads, each over a specific word; five convolutional layers per head).
- **Data:** 4,739 species–sequence pairs from NCBI/GenBank; a family dataset of
  **50 families** (21–425 sequences each) and a genus dataset of **17 genera**
  (23–53 sequences each) — two separate datasets, not one hierarchy.
- **Result:** **94.72% genus / 86.51% family** accuracy, beating reference-based
  tools when the query species is absent — genus: 94.7% vs OBITools 27.3%, Lolo
  19.4%, Kraken2 16.1%; family: 86.5% vs 73.1% / 65.4% / 41.3%.
- **Scope:** in-silico only (taxonomic absence enforced computationally); no mock
  communities or environmental samples; **no abstention, no calibration, no
  open-set / novelty capability.**

## Why it does not scoop Paper 1

It shares our *motivation* (reliable higher-rank assignment when the species is
missing) but not our *formulation*. The decisive differences:

| Dimension | Villon et al. 2026 | This paper |
|---|---|---|
| Class set | **Closed-set** (fixed families/genera) | **Open-set** (can recognise out-of-class) |
| Abstention | None — always emits a class | Explicit **no-call** at any rank |
| Calibration | None | Prospective, species-disjoint; **measured 0% false-species-call rate** |
| Novelty detection | None | DETECT (AUROC 0.84 genus) |
| Tree geometry | None | Tree-distance embedding (recovery 0.91, controlled) |
| Value from "unknown" | None | **Active reference-curation** (sequencing agenda) |
| Marker/taxon | 12S, fish | COI, fish (marker- and taxon-general framing) |

**The substantive point (use in §4.3 / §4.5):** because *every* test taxon's genus
and family in Villon et al. remain training classes, their "unknown sequence" test
is not truly open-set — a query from a genuinely unseen family is forced into a
known family with a confident softmax score. That is exactly the failure mode our
open-set detector (§4.3) and hidden-family stress test (§4.5) measure and prevent.

## Where it is cited

- **Introduction** — one sentence, as recognition that reliable higher-rank
  assignment under incomplete references is increasingly sought.
- **Related work (§2)** — the closed-set-vs-open-set contrast above.
- **§4.5** — the fixed-class-cannot-abstain contrast, measured against our
  rank-collapse result.
- **References** and the **must-cite list** in `NOVELTY_AND_PRIOR_ART.md` (#14).
