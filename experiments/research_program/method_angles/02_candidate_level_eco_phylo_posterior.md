# Candidate-Level Eco-Phylo Posterior

## Core Idea

Every candidate taxon should carry a feature vector:

```text
candidate = sequence score
          + p-distance / BLAST evidence
          + tree-neighborhood evidence
          + marker-resolvability evidence
          + geography/range prior
          + co-occurrence prior
          + reference support
```

The posterior ranks candidates and decides which taxonomic rank is justified.

## Literature Boundary

The closest prior work is TAXDNA, which combines 12S sequence evidence,
phylogenetic embeddings, and range/community co-occurrence for eDNA annotation:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013776

Phylogenetic placement tools also produce candidate placements with uncertainty:

- pplacer:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3098090/
- EPA-ng:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6368480/
- APPLES:
  https://pubmed.ncbi.nlm.nih.gov/31545363/

Set-based neural models are relevant because a candidate list is a set, not an
ordered sentence:

- Deep Sets:
  https://arxiv.org/abs/1703.06114
- Set Transformer:
  https://proceedings.mlr.press/v97/lee19d.html

## Our Gap

TAXDNA shows that sequence + phylogeny + co-occurrence can help. Our distinct
question is:

> Given an ambiguous molecular query and a candidate set, which evidence source
> justifies which taxonomic rank?

This means the posterior should be candidate-level and auditable. It should say
why a candidate was accepted or rejected.

## Experiments To Run

1. Candidate-level logistic/GBM posterior using current top-5 Global_eDNA table.
2. Add per-candidate BLAST/local-alignment identity rather than only method-arm
   scores.
3. Add explicit range/co-occurrence weights per candidate, not just arm labels.
4. Train a set-aware reranker that sees the whole candidate list.
5. Evaluate held-out sites with species disabled unless species thresholds
   transfer.

## Current Evidence

Candidate-level posterior work exists and is useful at higher ranks:

- full sequence+tree candidate posterior: species fails, genus/family/order
  carry signal;
- true nested fit confirms conservative higher-rank eDNA inference;
- species-disabled rank-backoff is the honest current eDNA operating point.

## Success Criterion

The method is successful if it improves held-out family/order/genus assignment
coverage at fixed precision while preserving low false species-call behavior.

Species-level eDNA is only allowed if an independently calibrated species
threshold transfers to held-out sites.

