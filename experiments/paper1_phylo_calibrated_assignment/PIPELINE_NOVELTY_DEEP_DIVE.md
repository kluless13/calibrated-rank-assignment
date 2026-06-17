# Pipeline Novelty Deep Dive

Last updated: 2026-06-02

> **NOVELTY CORRECTION (2026-06-17):** Superseded for novelty claims by
> `NOVELTY_AND_PRIOR_ART.md` (corrected after a verified literature audit).
> Concede priority on the tree-distance embedding (Stalder 2025 / DEPP 2023), the
> k-mer baseline (kf2vec 2025), and neural barcode clustering (BarcodeBERT 2026,
> DNABERT-S, BIN/ABGD/ASAP). Genuine novelty = open-set DETECT + 3-way integration
> on fish COI + the genus-knowable/species-limited boundary.

## Bottom Line

We are no longer in the "random experiments" phase for Paper 1. The current
work has a coherent direction:

> turn barcode/eDNA assignment into auditable evidence integration under marker
> ambiguity and missing references.

The novelty is not any single ingredient. The adjacent literature already owns
many ingredients:

- BLAST/VSEARCH/k-mer search owns mature sequence-similarity evidence.
- pplacer/EPA-ng/SEPP/APPLES own phylogenetic placement on fixed trees.
- Fernando et al. own a close fish COI backbone-completeness placement study.
- DEPP/H-DEPP/C-DEPP own neural single-gene-to-species-tree placement.
- PROTAX owns probabilistic taxonomic placement with missing/unknown branches.
- BarcodeBERT/BarcodeMamba/DNABERT-S/TaxoTagger own learned/vector barcode
  representation, retrieval, and classification.
- TAXDNA owns 12S sequence + fish tree + co-occurrence/range priors for eDNA.

What still looks under-built is a single benchmarked tool layer that:

1. accepts uncertain marker evidence;
2. retrieves candidates quickly;
3. keeps classical sequence evidence instead of throwing it away;
4. adds tree/placement/reference-gap evidence;
5. optionally adds ecology for eDNA;
6. outputs the deepest defensible rank with reason codes.

That is the missing edge we should own.

## Knowledge Graph View

Existing nodes:

```text
sequence similarity -> BLAST/VSEARCH/k-mer
sequence to tree edge -> pplacer/EPA-ng/SEPP/APPLES
single gene to species tree -> DEPP/H-DEPP/C-DEPP
barcode representation -> BarcodeBERT/BarcodeMamba/DNABERT-S/TaxoTagger
uncertain taxonomy -> PROTAX/IDTAXA/RDP-style classifiers
12S + ecology -> TAXDNA/eDNA metabarcoding pipelines
reference gaps -> reference-library audit literature
```

Missing edges we can build:

```text
all candidate generators -> one candidate evidence schema
candidate evidence -> calibrated species/genus/family/order/no-call
tree placement confidence -> rank decision and reason label
reference gaps -> warning plus active curation priority
12S marker ceiling -> species disabled unless evidence transfers
fast vector retrieval -> conservative biodiversity inference, not forced labels
```

## What We Should Learn From Each Literature Family

### BLAST / VSEARCH / k-mer / Kraken-Style Tools

What they teach:

- exact or local sequence evidence is very strong when close references exist;
- k-mer indexes are fast because they turn the query into many small exact
  evidence pieces;
- BLAST is a heuristic local-alignment engine, not "AI," but it is excellent
  at the problem it was built for;
- fast classifiers such as Kraken resolve conflicting k-mer evidence through a
  taxonomy tree.

Adaptation for us:

- keep BLAST/VSEARCH/p-distance as evidence features and rerankers;
- use vector retrieval as the fast candidate stage, then spend alignment/tree
  computation only on top-k candidates;
- add "conflicting evidence" reason labels when top-k candidates disagree
  across genus/family/order.

### pplacer / EPA-ng / SEPP / APPLES

What they teach:

- the right classical output is not a species label; it is a tree placement
  with confidence/edge information;
- EPA-ng/pplacer provide likelihood-weight style uncertainty;
- APPLES shows that distance-based placement is a useful scalable alternative;
- placement workflows require careful reference alignment/tree preparation.

Adaptation for us:

- score placement outputs through the same rank/no-call layer as neural and
  similarity methods;
- include Fernando-style backbone-completeness sweeps as comparator context;
- convert placement confidence into reason codes, not just a separate figure.

### Fernando et al. 2025

What they teach:

- COI can place missing fish species onto a backbone tree surprisingly well;
- family-stratified backbone sampling matters;
- EPA-ng beats APPLES on their setup;
- even high-confidence placements are not always correct.

Adaptation for us:

- do not claim "COI can place fish on a tree" as novel;
- use their protocol as a required comparator;
- extend beyond their question by asking: after placement/retrieval, what rank
  can we responsibly report to a biodiversity user?

### DEPP / H-DEPP / C-DEPP

What they teach:

- neural sequence-to-species-tree placement is already a known method family;
- the clean decomposition is sequence encoder -> query/reference distance
  matrix -> placement/update;
- hyperbolic geometry is already motivated for tree-like embeddings;
- scaling large trees requires tree-aware partitioning/ensembles.

Adaptation for us:

- treat our encoders as candidate evidence generators, not proof that neural
  tree placement is new;
- add a DEPP-style distance-matrix adapter if we want direct comparison;
- use hyperbolic/tree geometry only if it improves rank decisions or
  reference-gap diagnostics, not as architecture novelty by itself.

### PROTAX / IDTAXA / RDP-Style Classifiers

What they teach:

- taxonomic output should stop when confidence is insufficient;
- missing references and unknown branches can be explicit model outcomes;
- classifier confidence needs calibration, not raw softmax trust;
- reference mislabelling and taxonomy incompleteness are first-class problems.

Adaptation for us:

- make "unknown/unsupported species under this genus/family" a reason-code
  class;
- report calibration and risk-coverage curves by rank;
- train missing-reference-aware calibration on strict hidden species/genus/family
  packs, not only normal held-out rows.

### BarcodeBERT / BarcodeMamba / DNABERT-S / TaxoTagger

What they teach:

- learned barcode/vector retrieval is not new;
- strong barcode encoders can be fast and competitive for classification;
- domain-specific barcode pretraining matters;
- vector databases are already natural for barcode semantic search.

Adaptation for us:

- do not sell vector retrieval alone as novelty;
- benchmark speed, recall, and final calibrated rank accuracy separately;
- optionally import/fine-tune a barcode foundation model only as another
  candidate generator in our common evidence schema.

### TAXDNA And eDNA Priors

What they teach:

- 12S sequence-only species assignment is weak under open candidates;
- phylogeny and co-occurrence/range priors can help;
- real eDNA validation must be compared to traditional pipelines.

Adaptation for us:

- do not claim "sequence + tree + ecology improves eDNA" as new;
- make marker resolvability and rank/no-call the unique eDNA angle;
- keep species disabled unless nested held-out thresholds transfer.

## Ranked Novelty Bets

### 1. Evidence Compiler For Defensible Taxonomic Claims

Claim:

> Any barcode/eDNA method should be converted into the same auditable evidence
> table and judged by the deepest defensible rank, not only species top-1.

Why this is strongest:

- it bridges existing tools instead of competing where they are already strong;
- it fits our current COI and 12S/eDNA results;
- it is useful to real users because it explains why a species label is or is
  not justified.

Next implementation:

- add reason-code output to production v1;
- standardize candidate schemas across vector, BLAST/VSEARCH/k-mer, EPA-ng, and
  APPLES outputs;
- publish risk-coverage and false-species-call tables.

### 2. Missing-Reference Stress-Test Benchmark

Claim:

> Barcode tools should be stress-tested by hiding true species, genera, and
> families from the reference layer.

Why this matters:

- most real biodiversity databases are incomplete;
- current forced-label benchmarks hide the most important failure mode;
- we already have strict hidden-reference runs.

Next implementation:

- turn the strict packs into a clean benchmark table;
- add placement and vector-first methods to the same packs where feasible;
- report false species-call rate as a first-class metric.

### 3. Reference-Gap And Active-Curation Layer

Claim:

> The system should identify which missing references would most reduce
> uncertainty.

Why this can set us apart:

- reference-gap papers diagnose the problem, but fewer tools convert assignment
  uncertainty into sequencing priorities;
- our v2 gap detector already has signal, especially for hidden genus/family
  cases.

Next implementation:

- build clade-level active-curation tables:
  ambiguous queries, warning burden, likely missing rank, expected assignment
  gain if a reference is added;
- attach warnings to user-facing outputs as reason labels, not hard filters.

### 4. Vector-First But Classical-Light Inference

Claim:

> Vector retrieval is useful as a fast front-end only when followed by
> alignment/tree checks and calibrated rank/no-call decisions.

Why this matters:

- it avoids overclaiming against BarcodeBERT/TaxoTagger;
- it gives us a practical tool path;
- it lets us compare speed without pretending speed alone is novelty.

Next implementation:

- benchmark full end-to-end timing:
  embedding -> ANN top-k -> p-distance/BLAST rerank -> rank/no-call;
- compare against BLAST/VSEARCH/k-mer latency and final decision quality.

### 5. 12S/eDNA Reliability Extension

Claim:

> 12S species-level eDNA assignment is an evidence-integration problem under a
> marker information ceiling.

Why this matters:

- TAXDNA already covers sequence + phylogeny + ecology;
- our unique extension is deciding when species is not justified and reporting
  a defensible broader rank.

Next implementation:

- finish nested posterior repeats;
- keep species disabled unless thresholds transfer;
- add per-query evidence/reason labels:
  marker ambiguous, sequence weak, tree/neighborhood weak, ecology supports
  only broader rank.

## How Far Off We Are

We are past "just experiments" for the COI half:

- production v1 exists as a research CLI;
- strict missing-reference tests exist;
- comparator tables exist;
- DL layers have been trained and tested;
- vector speed is promising;
- Fernando-style and APPLES/EPA sweeps exist as comparator context.

We are not yet at a publishable tool contribution because the final system does
not yet fully expose the evidence-accounting layer. The missing product/method
pieces are:

1. reason-code output for every assignment;
2. one normalized candidate evidence schema across all methods;
3. active reference-curation tables;
4. final eDNA nested posterior stability;
5. a polished manuscript table that separates:
   sequence-only, vector-first, placement, reranked, rank/no-call, and ecology
   contributions.

Pragmatic estimate:

- one focused pass gets us to a coherent Paper 1 results package;
- one more pass gets us to a genuinely distinct tool/method prototype;
- major new DL architecture work should wait until the evidence compiler is
  stable, otherwise we risk training models without a sharper claim.

## Immediate Build Order

1. Production reason-code overlay.
2. Active reference-curation table from strict hidden-reference and v2 gap
   detector outputs.
3. Common candidate evidence adapter:
   vector/BLAST/VSEARCH/k-mer/EPA-ng/APPLES -> one table.
4. Finish/copy nested eDNA posterior repeats and update claim boundaries.
5. Only then train/import stronger encoders:
   BarcodeBERT, BarcodeMamba-compatible image, DNABERT-S, DEPP-style adapter.

## Sources Checked

- pplacer: https://pmc.ncbi.nlm.nih.gov/articles/PMC3098090/
- EPA-ng: https://pmc.ncbi.nlm.nih.gov/articles/PMC6368480/
- APPLES / APPLES-2: https://pmc.ncbi.nlm.nih.gov/articles/PMC7164367/ and
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9404983/
- Fernando et al. fish COI placement:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11706799/
- Fernando code/data repository:
  https://github.com/Thanu92/Realignment
- DEPP: https://pmc.ncbi.nlm.nih.gov/articles/PMC10198656/ and
  https://github.com/yueyujiang/DEPP
- H-DEPP: https://pmc.ncbi.nlm.nih.gov/articles/PMC9495508/ and
  https://github.com/yueyujiang/hdepp
- C-DEPP: https://pmc.ncbi.nlm.nih.gov/articles/PMC11193062/
- PROTAX: https://academic.oup.com/bioinformatics/article/32/19/2920/2196481
- IDTAXA: https://pmc.ncbi.nlm.nih.gov/articles/PMC6085705/
- RDP classifier: https://pmc.ncbi.nlm.nih.gov/articles/PMC1950982/
- PROTAX-GPU: https://pmc.ncbi.nlm.nih.gov/articles/PMC11070247/
- TAXDNA: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013776
- BarcodeBERT: https://arxiv.org/abs/2311.02401
- BarcodeMamba: https://arxiv.org/abs/2412.11084 and
  https://github.com/bioscan-ml/BarcodeMamba
- DNABERT-S: https://pmc.ncbi.nlm.nih.gov/articles/PMC10896361/ and
  https://github.com/MAGICS-LAB/DNABERT_S
- LISA learned sequence search: https://arxiv.org/abs/1910.04728
- LV Barcoding: https://arxiv.org/abs/1407.3348
- Reference database challenges:
  https://pubmed.ncbi.nlm.nih.gov/36478393/
- VSEARCH: https://pmc.ncbi.nlm.nih.gov/articles/5075697/
- Kraken: https://pmc.ncbi.nlm.nih.gov/articles/PMC4053813/
