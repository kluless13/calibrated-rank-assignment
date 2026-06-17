# Paper 1 Encoder Benchmark Ladder

Paper 1 should not stop at MarineMamba or at Eval C retrieval. The stable
question is:

> Which sequence encoders can map short COI barcodes into a real fish
> species-tree coordinate system, and which can produce calibrated
> species/genus/family/order/no-call assignments?

The evaluation stays fixed:

- same clean COI fish-tree inputs,
- same candidate species tree,
- same tree embeddings,
- same Eval C, seen-test, and unseen-genera splits,
- same retrieval metrics,
- same tree-recovery metrics,
- same calibration/no-call analysis,
- same BLAST/VSEARCH/k-mer baselines and negative controls.

## Encoder Families To Test

### 1. Classical Sequence Similarity

Completed for Paper 1 clean splits:

- k-mer cosine similarity,
- BLAST,
- VSEARCH.

Purpose:

- establish the classical floor/ceiling for reference-backed assignment;
- show what these methods cannot do when the true species or genus lacks a
  reference sequence.

### 2. Small Neural Encoders Trained From Scratch

Implemented and run for seed1206 because they are cheap, controllable, and do
not depend on large pretrained model downloads:

- CNN barcode encoder,
- biLSTM encoder,
- small Transformer encoder.

Script:

- `scripts/edna/train_fish_tree_encoder_benchmark.py`

Runner:

- `experiments/paper1_phylo_calibrated_assignment/runs/02_vast_encoder_benchmarks.sh`

Purpose:

- test whether the tree-space result is a Mamba-specific effect or a general
  property of barcode sequence encoders;
- identify whether recurrence, convolution, or attention is enough for COI
  barcode-to-tree learning.

Current read:

- CNN is the strongest current tree-geometry encoder.
- CNN seed repeats 1207 and 1208 are complete and stable.
- biLSTM/Transformer have reusable query embeddings copied locally for Eval C,
  seen-test, and unseen-genera, but they are not yet the lead architecture.

### 3. State-Space / Long-Context Encoders

Already tested:

- BarcodeMamba / Mamba-style SSM.

Future:

- S5/S4-style state-space encoder,
- Caduceus-style bidirectional reverse-complement-aware Mamba, if dependency
  setup is clean.

Purpose:

- test whether linear-time sequence models preserve barcode phylogenetic signal
  better than CNN/LSTM/Transformer baselines.

### 4. Barcode-Specific Foundation Models

Candidate:

- BarcodeBERT-style self-supervised barcode transformer.

Purpose:

- test whether barcode-domain pretraining helps tree-space placement more than
  training from scratch.

### 5. General DNA Foundation Models

Candidates:

- DNABERT-2,
- Nucleotide Transformer,
- HyenaDNA,
- Caduceus.

Purpose:

- test whether general genomic foundation models transfer to short barcode
  biodiversity inference;
- compare frozen embeddings, projection-only tuning, and full fine-tuning if
  resources allow.

## Literature Anchors

- BarcodeBERT shows that barcode-specific self-supervised Transformer
  pretraining can be strong for biodiversity identification, and explicitly
  compares against CNNs, general DNA foundation models, and BLAST:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC13008329/>
- BarcodeMamba motivates state-space models for barcode biodiversity analysis:
  <https://arxiv.org/abs/2412.11084>
- Mamba motivates selective state-space models as linear-time sequence models:
  <https://arxiv.org/abs/2312.00752>
- S5 is a simplified state-space layer family for long sequence modeling:
  <https://arxiv.org/abs/2208.04933>
- HyenaDNA targets long-range DNA sequence modeling with subquadratic scaling:
  <https://arxiv.org/abs/2306.15794>
- DNABERT-2 is a multi-species genome foundation model and benchmark:
  <https://arxiv.org/abs/2306.15006>
- Nucleotide Transformer benchmarks robust DNA foundation models:
  <https://www.nature.com/articles/s41592-024-02523-z>
- Caduceus adds bidirectional and reverse-complement-equivariant DNA modeling:
  <https://arxiv.org/abs/2403.03234>

## Build Order

Completed:

1. Small neural encoder matrix:
   - CNN,
   - biLSTM,
   - small Transformer.
2. Query-embedding exports for CNN/biLSTM/Transformer across Eval C,
   seen-test, and unseen-genera.
3. CNN seed repeats 1207 and 1208 with tree-recovery follow-ups.
4. First rank-adaptive calibration/no-call tables.

Next:

1. Add vector-first retrieval benchmarking from saved embeddings.
2. Add placement comparators to the same scoring layer.
3. Decide whether the best next model is:
   - S5/S4,
   - BarcodeBERT,
   - DNABERT-2/Nucleotide Transformer,
   - Caduceus/HyenaDNA.
4. Export Mamba query embeddings only on a compatible PyTorch/CUDA image.

## Claim Logic

- If Mamba wins: architecture matters.
- If CNN/LSTM/Transformer matches Mamba: the tree-space objective matters more
  than the architecture.
- If all neural encoders beat species-level classical baselines on held-out
  species: open-candidate tree-space learning is the core contribution.
- If classical methods dominate higher ranks: rank-adaptive systems should use
  neural and classical evidence together instead of forcing one winner.
