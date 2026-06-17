# Rank-Adaptive No-Call Calibration

## Core Idea

The output should not always be a species. It should be:

```text
species -> genus -> family -> order -> no-call
```

depending on the evidence.

This is the scientific reliability layer. A correct family call is more useful
than a false species call.

## Literature Boundary

Rank-aware and ambiguity-aware taxonomic assignment already exists in several
forms:

- Lowest common ancestor and flexible ambiguous-read assignment:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3024944/
- BLAST/LCA-style taxonomic assignment and the problem of missing references:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3712219/
- Selective classification and reject-option learning provide the ML framework
  for abstaining when confidence is insufficient:
  https://proceedings.mlr.press/v97/geifman19a.html

Therefore, "abstain when uncertain" is not novel by itself.

## Our Gap

Our rank/no-call policy is tied to biological evidence:

- marker resolvability;
- candidate reference support;
- tree-distance behavior;
- consensus among top candidates;
- sequence and ecological agreement;
- strict missing-reference validation.

The contribution is making abstention taxonomic and evidence-specific, not a
generic confidence threshold.

## Experiments To Run

1. Calibrate thresholds on one split and evaluate on held-out species/genera.
2. Run strict missing-reference validation where the hidden species/genus/family
   is absent before training/reference construction.
3. Measure coverage, assigned precision, and false species-call rate.
4. Compare against BLAST/VSEARCH/k-mer, EPA-ng/APPLES, and vector-first modes.
5. Add bootstrap confidence intervals for final operating points.

## Current Evidence

Current conservative COI production operating point:

- held-out fish split: 95.8% coverage, 93.0% assigned precision, 0.0% false
  species calls;
- unseen-genera split: 92.3% coverage, 83.9% assigned precision, 0.0% false
  species calls.

This works because the system backs off to higher ranks instead of forcing a
species.

## Success Criterion

The paper-ready claim is:

> Under missing-reference uncertainty, calibrated rank backoff reduces false
> species calls while preserving useful genus/family/order information.

