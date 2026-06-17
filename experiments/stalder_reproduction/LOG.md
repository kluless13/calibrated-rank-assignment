# Stalder/TAXDNA Comparator Log

## 2026-05-30

Role in the research program:

- Support the merged Paper 1 eDNA work package by documenting exact TAXDNA
  reproduction status.
- Provide public Stalder-style reconstruction where exact official assets are
  unavailable.
- Keep official reproduction separate from our independent MarineMamba results.

Current boundary:

- Official pretrained assets are available.
- Exact retraining is blocked by missing processed `species_sequences.json` and
  co-occurrence JSONs.
- Public FISHGLOB co-occurrence reconstruction exists and can be used as a
  transparent comparator, but it is not the unreleased official TAXDNA input.

Next actions:

1. Continue treating exact TAXDNA reproduction as blocked until official data
   are retrieved from a live Renku session export or from the authors/SDSC.
2. Use public reconstruction only with explicit provenance.
3. Keep the Stalder track as a comparator and alignment layer for the merged
   Paper 1 eDNA work package.

What this answers:

- Which parts of the Stalder/TAXDNA system are publicly reproducible now?
- Which assets are official pretrained assets versus missing processed training
  inputs?
- Which public ecological sources can be reconstructed transparently for our
  own comparator experiments?

How this is unique/useful:

- It prevents us from overstating exact reproduction.
- It gives the merged Paper 1 eDNA work package a clean comparator boundary:
  - exact official inference if supported by released checkpoints;
  - public Stalder-style reconstruction where official processed JSONs are not
    available;
  - independent MarineMamba/Eco-Phylo results reported separately.

What remains:

1. Try live Renku session export or direct author/SDSC request for missing
   processed JSONs.
2. Run official TAXDNA inference-only baseline on Global_eDNA representatives
   if dependencies can be installed cleanly.
3. Keep public FISHGLOB/RLS/OBIS reconstructions documented with provenance.
