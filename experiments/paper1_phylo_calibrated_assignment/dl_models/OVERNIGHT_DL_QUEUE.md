# Overnight DL Queue

## Current State

As of 2026-06-02, the overnight DL queue has tested five follow-up layers
after the retrieval-DL encoder sweep:

1. pairwise tree-aware candidate reranking;
2. independent selected-candidate rank/no-call calibration;
3. no-counts reference-gap target sweep;
4. candidate-evidence reference-gap detector v2 with a no-tree ablation;
5. mixed missing-reference-aware rank/no-call calibration using v2 gap
   probabilities as soft evidence.

All five are useful, but none replaces the conservative production operating
point yet.

## What We Learned

The strongest current DL result is still candidate ordering, not final
assignment:

- pointwise tree10 reranker strongly improves held-out genus/family ordering;
- pairwise tree10 is mixed and does not clearly replace pointwise tree10;
- listwise tree10 is a negative result;
- selected-candidate calibrators do not fix missing-reference transfer;
- reference-gap detectors can identify many hidden species cases, but
  genus/family gap recall only becomes useful after candidate-level evidence;
- v2 reference-gap scores are promising diagnostics, but supported-case warning
  rates are still too high for default production behavior.
- mixed missing-reference-aware calibration gives a high-precision optional
  mode: species-disabled target-0.99 reaches 98.3% held-out precision and
  95.7% unseen-genera precision, with zero species calls, but lower coverage.

## Current Boundary

DL is valuable as evidence inside the pipeline:

```text
vector retrieval -> p-distance/BLAST/VSEARCH/tree evidence -> DL candidate
ordering -> conservative rank/no-call policy -> diagnostic reason labels
```

DL is not yet the final rank/no-call decision maker.

The missing piece is not another small classifier on the same data. The missing
piece is a better training design where calibration positives and negatives
match deployment:

- true reference present;
- species hidden;
- genus hidden;
- family hidden;
- ambiguous nearest-neighbor groups;
- normal supported negatives that are not synthetic leakage.

## Next High-Leverage DL Experiments

1. Use reranker score as soft evidence in production-v1.
   Do not replace p-distance thresholds directly. Add it as a second evidence
   term and ask whether precision/coverage improves without false species
   calls.

2. Add production reason labels.
   Connect marker ambiguity, low evidence, and reference-gap warnings to the
   FASTA/CSV output so the tool explains why it backs off.

3. Run seed repeats for the mixed missing-reference-aware calibrator only if it
   becomes a candidate manuscript operating point.

4. Extend this to 12S/eDNA only after COI calibration is honest.
   For eDNA, keep species disabled unless thresholds transfer. Use candidate
   sequence evidence, marker resolvability, tree-neighborhood evidence, and
   ecological priors.

## Stop Conditions

Do not promote a DL layer to production if:

- unseen-genera family/order precision stays below target;
- false species calls appear in missing-reference splits;
- the model relies on split-specific flags or candidate-count leakage;
- thresholds only work on the calibration split.

## Current Recommendation

Keep production-v1 as the manuscript operating point. Present DL reranking and
v2 reference-gap detection as promising evidence-fusion layers, and frame the
calibration failures as an important scientific result: missing-reference
biodiversity inference needs explicit deployment-matched calibration, not just
stronger candidate scorers or stronger warning detectors.
