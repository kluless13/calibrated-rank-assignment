# Paper 3 Log

## 2026-05-30

Decision: Multi-marker shared tree space is the third paper/workstream, not the
first. It should begin after the COI tree-space and 12S Eco-Phylo systems are
stable.

Scientific boundary:

- COI does not teach 12S sequence motifs.
- COI can help define species/tree geometry.
- 12S can use that geometry for eDNA-relevant assignment and calibration.

What this answers:

- Can COI-rich reference structure help 12S/eDNA without pretending COI and 12S
  have the same nucleotide signal?
- Can two marker-specific encoders map into the same biological coordinate
  system?
- Does a shared species-tree space improve 12S calibration and higher-rank
  assignment?

How this is unique/useful:

- COI is species-rich and discriminative; 12S is eDNA-relevant but often short
  and ambiguous.
- A shared tree space lets COI contribute biological structure while 12S remains
  the actual eDNA observation.
- This gives a principled multi-marker transfer route: transfer through biology,
  not through raw sequence similarity.

What we already have:

- Clean COI fish-tree inputs and strong tree-space results.
- Multisource 12S/TAXDNA-style inputs.
- A shared candidate fish tree used across the main tracks.
- Tree embedding and zero-shot prediction utilities.

What remains:

1. Inventory overlap between COI candidate species and 12S candidate species.
2. Build a canonical shared species ID/taxonomy table.
3. Build a shared tree embedding manifest.
4. Train marker-specific encoders against the same species embeddings:
   - COI encoder -> shared tree space;
   - 12S encoder -> shared tree space.
5. Compare:
   - COI-only tree-space learning;
   - 12S-only tree-space learning;
   - shared-space multi-marker learning;
   - shared-space plus ecology.
6. Evaluate on:
   - COI held-out species;
   - 12S held-out species;
   - Global_eDNA;
   - rank-adaptive calibration.

Next actions:

1. Inventory overlap between COI candidate species and 12S candidate species.
2. Build a canonical shared species ID/taxonomy table.
3. Build a shared tree embedding manifest.
4. Prototype a dual-encoder training script only after the shared species table
   is audited.
