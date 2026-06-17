# DL 05: eDNA Eco-Phylo Posterior

## Question

Can candidate-level evidence integration make reliable 12S/eDNA calls when
sequence evidence alone is ambiguous?

## Boundary Against TAXDNA

TAXDNA already combines sequence, phylogeny, range/community context, and eDNA
validation. Our contribution cannot be the broad idea of sequence + tree +
ecology.

The useful angle is reliability:

- marker-resolvability ceilings;
- candidate-level evidence decomposition;
- held-out calibration;
- species-disabled rank/no-call when species thresholds do not transfer.

## Current Status

Done:

- candidate-level posterior table;
- direct 12S sequence evidence;
- inference-safe tree-neighborhood evidence;
- true nested fit70 posterior;
- species-disabled higher-rank backoff.

Current true nested species-disabled result:

- rep0 target-95 assigns 38.9% of held-out eDNA queries at 93.4% accuracy;
- rep1 target-95 assigns 38.5% at 95.4%;
- rep2 target-95 assigns 54.9% at 84.5%.

Current interpretation:

- output roots:
  `results/paper1_phylo_calibrated_assignment/eco_phylo_posterior/candidate_level_sequence_tree_evidence_nested_fit70_rep1/`
  and
  `candidate_level_sequence_tree_evidence_nested_fit70_rep2/`;
- the posterior is useful evidence for higher-rank eDNA assignment, but rep2
  shows the current mixed species-disabled target-95 policy is not stable
  enough to be a final operating point.

## Next Experiments

1. Improve calibration so mixed family/order policy reaches the target
   accuracy on held-out sites, not only calibration sites.
2. Add stronger candidate-level geography/range features.
3. Add co-occurrence uncertainty features rather than a single weight.
4. Compare BLAST-sequence-only, SSM/CNN-sequence-only, and posterior arms under
   identical rank/no-call output.
5. Keep species disabled unless species thresholds transfer cleanly.
