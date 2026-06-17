# Fernando 2025 Positioning

## Why This Paper Matters

Fernando, Fu, and Adamowicz 2025 is the closest direct comparator for Paper 1:

- same broad biological domain: bony fish;
- same marker: COI;
- same backbone family: Fish Tree of Life / Rabosky-style actinopterygian tree;
- same broad question: can missing species be placed onto a fish backbone tree
  from COI barcode evidence?

Source:

- PubMed: https://pubmed.ncbi.nlm.nih.gov/39781258/
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11706799/
- Reported data/code: https://github.com/Thanu92/Realignment

## What Fernando 2025 Did

Their experiment:

- used 4520 bony fish species with 27-gene reference-tree evidence;
- used COI barcodes as query evidence for missing species;
- created reduced backbone trees at 20%, 40%, 60%, 80%, and 99%
  completeness;
- compared random, stratified, and biased species sampling;
- placed the missing species with EPA-ng and APPLES;
- scored percentage of correct placement (PCP), plus EPA-ng high-confidence
  placements using LWR > 0.9.

Their key results:

- EPA-ng placed roughly 70%-78% of missing species correctly across backbone
  completeness settings.
- At 99% completeness with stratified sampling, EPA-ng reached 78% PCP.
- Among high-confidence EPA-ng placements at that setting, PCP reached 87%.
- APPLES was lower, roughly 65%-69%.
- Biased sampling performed substantially worse.

## What We Must Not Claim

Do not claim:

- "COI barcodes can place fish onto a species tree" as a new finding.
- "Phylogenetic placement of fish COI onto a backbone tree" as our novelty.
- "EPA-ng/pplacer/APPLES are optional side baselines."
- "Pearson tree-recovery correlation alone is comparable to Fernando PCP."

Fernando already owns the classical-placement version of the fish COI backbone
question.

## How We Are Different

Our defensible distinction is not tree placement by itself. It is the integrated
benchmark and inference layer:

1. Neural barcode encoders are trained into a species-tree coordinate system.
2. The same held-out species/genera splits are evaluated across:
   - CNN / biLSTM / Transformer / Mamba where available,
   - BLAST / VSEARCH / k-mer,
   - EPA-ng / pplacer / APPLES-style placement.
3. Outputs are converted into species/genus/family/order/no-call behavior.
4. Missing-reference behavior is measured directly:
   - true species hidden,
   - true genus hidden,
   - true family hidden,
   - rank-backoff instead of forced species calls.
5. Tree geometry is audited, not only placement accuracy:
   - tree-distance recovery,
   - nearest-neighbor preservation,
   - residuals by rank/clade/reference density,
   - calibration/no-call curves.

The paper should therefore be framed as:

> a transparent barcode-to-tree benchmark and rank-adaptive inference framework
> for fish COI under missing-reference conditions.

Not:

> a new proof that COI can be placed onto a fish tree.

## Experiments Needed Because Of Fernando

### Required

1. Finish EPA-ng/pplacer on our clean held-out splits.
   - EPA-ng is complete for Eval C, seen-test, and unseen-genera.
   - pplacer is blocked until a valid refpkg/stats model is supplied.
2. Add APPLES or an APPLES-equivalent distance-placement baseline if feasible.
   - A labelled local APPLES-like p-distance baseline now exists.
   - Official APPLES 2.0.11 has now been run on the matched
     Fernando-style completeness sweeps.
3. Implement a placement-output scorer that reports Fernando-adjacent metrics:
   - PCP-like placement correctness,
   - LWR-binned EPA-ng accuracy,
   - nearest valid clade/rank,
   - tree-distance error,
   - species/genus/family/order/rank-backoff.
   - Current status: placed-clade containment, LWR bins, tree-distance error,
     rank-backoff, a Fernando-like edge-to-sister diagnostic, and a
     simulated-placement-tree PCP diagnostic exist.
   - Current EPA-ng edge-to-sister exact rates are Eval C 7.1%, seen-test
     4.7%, and unseen-genera 0.6%.
   - Current simulated-placement-tree species-representative exact/overlap
     rates are Eval C 7.3/22.1%, seen-test 24.0/50.4%, and unseen-genera
     0.2/45.8%.
   - The matched backbone-completeness protocol has now been run and scored
     separately under `fernando_completeness_final_30/`.
4. Report why our Pearson/Spearman tree-recovery metrics are complementary but
   not directly identical to PCP.

### Matched Completeness Sweep Status

Matched backbone-completeness sweeps are complete:

- 99%, 80%, 60%, 40%, 20%;
- random and family-stratified sampling;
- EPA-ng and official APPLES completed;
- same PCP-like score plus rank-adaptive outputs.

The completed classical comparator matrix covers EPA-ng and official APPLES.
Neural tree-space encoders have not yet been rerun inside every reduced-backbone
pack, so that remains the clean next extension if we want a direct
classical-versus-neural completeness-sweep comparison.

Current setup status:

- `scripts/edna/build_fernando_completeness_sweeps.py` generated 30 input packs:
  random and family-stratified sampling at 99/80/60/40/20% completeness with 3
  replicates each.
- Vast/Linux EPA-ng runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/07_vast_fernando_completeness_sweeps.sh`.
- Vast/Linux official APPLES runner:
  `experiments/paper1_phylo_calibrated_assignment/runs/11_vast_fernando_apples_sweeps.sh`.
- Final scored outputs:
  `results/paper1_phylo_calibrated_assignment/source_tables/fernando_completeness_final_30/`.
- The generated packs are large because each split currently carries its own
  sequence JSONs; do not increase replicate count without a storage decision.

Final public-setup diagnostic averages:

- APPLES placed-clade genus/family/order: 32.8 / 57.2 / 65.6%.
- EPA-ng placed-clade genus/family/order: 17.3 / 45.2 / 57.0%.
- APPLES sister-clade any-overlap/exact: 42.5 / 21.4%.
- EPA-ng sister-clade any-overlap/exact: 14.8 / 3.2%.
- Simulated-placement-tree represented sister-overlap:
  APPLES 18.0% sequence-level and 26.8% species-representative;
  EPA-ng 5.4% and 15.2%.

## Current Claim Boundary

Before the completeness sweeps, the safe claim was:

> CNN/Mamba-style barcode encoders can recover substantial fish species-tree
> geometry under held-out species and held-out genera, and the resulting
> candidate rankings support rank-aware missing-reference diagnostics.

After the Fernando-style placement experiments, the claim can become:

> compared with classical phylogenetic placement and similarity baselines under
> the same clean fish COI missing-reference splits, neural tree-space encoders
> provide a complementary route to calibrated rank-adaptive biodiversity
> inference.

Use careful wording:

- Say: "Fernando-style" or "Fernando-inspired matched completeness sweeps."
- Say: "official APPLES was run on our matched sweep matrix."
- Do not say: "exact Fernando reproduction" or "we reproduced Fernando PCP."
