# Multi-Marker Shared Tree Space

## Core Idea

COI is often richer for species-level fish barcoding. 12S/eDNA is common for
environmental monitoring but often less species-resolving. The shared object is
not the sequence alphabet. It is the species tree.

The method:

```text
COI encoder -> fish species-tree coordinates
12S encoder -> same fish species-tree coordinates
candidate inference -> shared rank/no-call policy
```

COI helps 12S by shaping the biological coordinate system, not by pretending
COI sequence patterns transfer directly to 12S.

## Literature Boundary

TAXDNA already maps 12S sequences into a phylogeny-aware embedding space and
uses co-occurrence/range context:
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013776

BarcodeBERT and DNABERT-S show that sequence encoders can learn useful barcode
or species-aware embeddings:

- https://arxiv.org/abs/2311.02401
- https://arxiv.org/abs/2402.08777

Classical DNA barcoding literature already recognizes that different markers
have different taxonomic resolution. Marker choice is not new.

## Our Gap

The missing bridge is a shared tree-space transfer experiment:

- COI-rich supervision learns a species-tree geometry;
- 12S/eDNA queries are mapped into that same coordinate system;
- the model reports the deepest defensible rank when 12S lacks species signal.

This is a way to use COI for 12S/eDNA without overclaiming direct sequence
transfer.

## Experiments To Run

1. Train/evaluate COI tree-space encoder on fish.
2. Train/evaluate 12S encoder into the same tree coordinates.
3. Compare 12S-only versus shared-tree 12S candidate retrieval.
4. Add marker-resolvability ceilings per candidate group.
5. Evaluate on Global_eDNA with species disabled until species thresholds
   transfer.

## Current Evidence

We already have:

- strong COI tree/rank-backoff results;
- 12S resolvability maps showing species-level ceilings;
- broad 12S SSM/CNN zero-shot evidence;
- candidate-level eDNA posterior features.

## Success Criterion

The method is successful if shared tree-space supervision improves held-out
12S/eDNA higher-rank inference and uncertainty calibration relative to a 12S
sequence-only baseline.

