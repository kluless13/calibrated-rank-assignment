# Fast Vector-First Barcode Retrieval

## Core Idea

Use neural or learned barcode embeddings as the first-pass candidate generator:

```text
query barcode -> embedding -> vector index -> top-k candidate taxa
```

Then keep classical checks for the small candidate set:

```text
top-k candidates -> p-distance / BLAST-like identity / tree checks -> rank/no-call
```

This is the "DL BLAST-like" part of the system, but the claim must be precise:
we are not inventing vector DNA search. We are using it as a fast candidate
layer inside uncertainty-aware biodiversity inference.

## Literature Boundary

Existing work already covers fast or learned DNA search:

- BarcodeBERT reports barcode embeddings for biodiversity analysis and
  BLAST-comparable species classification with a large speedup:
  https://arxiv.org/abs/2311.02401
- TaxoTagger is an open-source DNA barcode vector database / semantic search
  tool:
  https://mycoai.github.io/taxotagger/latest/
- LISA uses learned indexing to accelerate DNA sequence search:
  https://arxiv.org/abs/1910.04728
- DNABERT-S learns species-aware DNA embeddings:
  https://arxiv.org/abs/2402.08777

Therefore, "vector retrieval for DNA barcodes" is not enough as a novelty
claim.

## Our Gap

Most vector-search barcode work focuses on retrieval or classification. Our
pipeline uses retrieval as only the first stage, then asks:

- is the top hit biologically defensible?
- does the marker resolve species or only genus/family/order?
- is the nearest species missing from the reference?
- do tree distance, p-distance, and ecology agree?
- should the system abstain from species?

That turns speed into an inference layer rather than a standalone search demo.

## Experiments To Run

1. Benchmark exact vector, HNSW, BLAST, VSEARCH, and k-mer retrieval under the
   same query/reference splits.
2. Measure top-k recall at species/genus/family/order.
3. Measure latency per query and throughput for batch sizes relevant to eDNA.
4. Add "classical-light reranking" over vector top-k only.
5. Compare final rank/no-call behavior, not only top-1 species.

## Current Evidence

Current COI CNN seed1206 vector timing:

- exact vector retrieval: sub-millisecond per query;
- HNSW ANN retrieval: much faster still;
- p-distance rerank over top-k gives the current conservative production
  operating point.

Production v1 now packages the p-distance-reranked COI outputs using locked
mode-specific thresholds.

## Success Criterion

The useful claim is:

> A vector-first candidate layer can make barcode inference fast without giving
> up classical sequence checks, tree evidence, or calibrated uncertainty.

The claim is not:

> Neural vectors replace BLAST.

