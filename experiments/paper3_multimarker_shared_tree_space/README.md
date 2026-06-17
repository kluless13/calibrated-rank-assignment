# Paper 3: Multi-Marker Shared Tree Space

Working title:

> Connecting COI and 12S biodiversity evidence through a shared fish-tree space.

## Core Question

Can COI and 12S help each other by mapping into the same species-tree coordinate
system?

## First-Principles Boundary

COI cannot transfer raw sequence information into 12S. The markers observe
different mitochondrial regions.

What can transfer is the latent biological object:

- species identity,
- taxonomy,
- phylogenetic neighborhood,
- candidate tree position,
- expected ambiguity among close relatives.

The tree is the bridge.

## Concept

Train marker-specific encoders into one shared species coordinate system:

```text
COI encoder -> shared fish-tree species space
12S encoder -> shared fish-tree species space
ecology     -> same candidate species space
```

The model does not pretend COI and 12S have the same sequence signal. It uses
both markers as different observations of the same species/tree structure.

## Why It Matters

COI is richer and more species-discriminative. 12S is common in fish eDNA but
often shorter and more ambiguous. A shared species-tree space is a principled
way to use COI's reference richness without making invalid sequence-transfer
claims.

## Gaps Addressed

- Multi-marker transfer in barcode/eDNA ML is underdeveloped.
- COI-rich reference knowledge is usually disconnected from 12S eDNA ambiguity.
- Current models often treat each marker as a separate closed problem.

## Existing Inputs

Likely inputs:

- clean COI fish-tree inputs from `data/phylo/fish_tree_clean_phylo_inputs/`
- 12S multisource inputs from `data/edna/stalder_inputs/`
- candidate fish tree and species taxonomy used by both tracks
- tree embeddings from `scripts/edna/learn_tree_embedding_npz.py`

## Remaining Work

1. Define a shared species universe between COI and 12S tracks.
2. Build one canonical species-tree embedding for that universe.
3. Train COI encoder and 12S encoder into the same target space.
4. Compare:
   - COI-only tree-space learning,
   - 12S-only tree-space learning,
   - shared-space training,
   - shared-space plus ecology.
5. Evaluate on:
   - COI held-out species,
   - 12S held-out species,
   - Global_eDNA,
   - rank-adaptive calibration.

This is not the immediate first paper, but it is the most natural long-term
extension.

