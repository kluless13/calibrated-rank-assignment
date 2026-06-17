# Encoder Benchmark Log

## 2026-05-30

Decision: do not pigeonhole the work as SSM-only.

The stronger framing is:

> a sequence encoder inside a phylogeny-aware, ecology-aware, calibrated
> biodiversity inference system.

The sequence encoder can be SSM/Mamba, CNN, LSTM, Transformer, S5, k-mer, or a
pretrained DNA model. The method contribution should survive encoder swaps.

Near-term plan:

1. Finish Paper 1 placement comparators and scoring first.
2. Use completed CNN/biLSTM/Transformer embeddings for vector-first retrieval
   and full-candidate ablation.
3. Treat S5/BarcodeBERT/DNABERT-style models as follow-ups only after the
   scoring layer is stable.
4. Report where architecture matters and where tree/ecology/calibration matter
   more than architecture.

What this answers:

- Is the performance gain coming from architecture, tree-space supervision,
  ecological context, calibration, or reference coverage?
- Do SSMs actually help, or would CNN/LSTM/Transformer/S5 do the same under a
  fair protocol?
- Is a simple k-mer/BLAST/VSEARCH baseline enough in some regimes?

How this is unique/useful:

- It prevents the project from being pigeonholed as a single-architecture paper.
- It makes the main contribution encoder-agnostic.
- It lets the paper say where deep learning is useful and where classical
  methods remain stronger.

Fair comparison contract:

- same train/test splits;
- same candidate species universe;
- same tree embeddings;
- same objective family where possible;
- same top-k species/genus/family/order metrics;
- same calibration/no-call protocol;
- same ecological posterior for the merged Paper 1 eDNA work package;
- same runtime and resource reporting.

Candidate encoder ladder:

1. Classical:
   - k-mer;
   - BLAST;
   - VSEARCH.
2. Lightweight neural:
   - CNN/TAXDNA-style: run, repeated, currently strongest tree-geometry
     baseline;
   - biLSTM: run, query embeddings copied;
   - small Transformer: run, query embeddings copied.
3. State-space:
   - Mamba/SSM: run for core tree-space metrics, query embeddings blocked on
     current CUDA/PyTorch image;
   - S5 if dependencies and implementation remain clean.
4. Pretrained DNA models:
   - only after the local benchmark interface is stable.

Outcome interpretation:

- If Mamba wins, that supports the architecture.
- If Transformer/LSTM/CNN wins, the broader method still matters.
- If architecture barely matters but tree/ecology/calibration matter, that is a
  strong scientific finding.
- If k-mer/BLAST wins in some regimes, report that honestly and use it to define
  where deep learning is unnecessary.
