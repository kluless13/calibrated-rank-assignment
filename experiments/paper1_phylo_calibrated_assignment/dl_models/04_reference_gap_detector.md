# DL 04: Reference-Gap Detector

## Question

Can a model detect when the current reference set does not support a species or
genus call?

## Motivation

Reference incompleteness is not a nuisance variable. It is central to real
biodiversity inference. The system should say why it backs off:

- species likely absent;
- genus likely absent;
- family/order only;
- marker ambiguous;
- weak or conflicting evidence.

## Relation To Prior Work

PROTAX-style methods explicitly model known species, species without reference
sequences, and unknown taxa. Fernando-style placement tests backbone
completeness. Our gap is to attach that logic to the learned/vector/tree
pipeline and output actionable diagnostics.

## Current Status

Done:

- strict hidden species/genus/family packs;
- strict CNN retrains;
- strict rank-backoff summaries.
- strict species-disabled DL calibrator application.
- first diagnostic reference-gap MLP:
  `scripts/edna/train_paper1_reference_gap_detector.py`.
- candidate-evidence reference-gap MLP v2:
  `scripts/edna/train_paper1_reference_gap_detector_v2.py`.
- source-table collector:
  `scripts/edna/build_paper1_reference_gap_detector_summary.py`.

Current strict DL stress result:

- hidden species: the DL layer still assigns about 50.6-58.0% of queries with
  88.5-91.3% precision and no species calls.
- hidden genus: assigns about 52.9-57.9% with 82.5-89.5% precision and no
  species calls.
- hidden family: drops to 16.1-41.9% coverage with 72.7-81.5% precision and no
  species calls.

Interpretation: the current rank/no-call MLP backs off conservatively, but it
does not yet explicitly diagnose why evidence is unsupported. Hidden-family
cases are the strongest signal that a separate reference-gap model is needed.

Not done:

- explanation labels connected to the production CLI.
- a production-quality reference-gap reason layer.

Detector outputs:

- primary source table:
  `results/paper1_phylo_calibrated_assignment/source_tables/reference_gap_detector_summary.csv`.
- honest per-query run:
  `results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1206_no_counts_target099/`.
- v2 tree-aware candidate-evidence runs:
  `results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_target095/`
  and
  `results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_target099/`.
- v2 no-tree ablation:
  `results/paper1_phylo_calibrated_assignment/reference_gap_detector/coi_mlp_seed1301_v2_candidate_evidence_notree_target095/`.
- caveat: a second count-feature run exists, but candidate-set size can encode
  synthetic strict-pack identity and should not be used as a headline result.

No-counts target-0.99 detector:

- normal supported false-gap flag rates:
  - held-out fish: species/genus/family 4.7 / 0.3 / 0.2%;
  - seen-test: species/genus/family 1.5 / 0.1 / 0.0%;
  - unseen-genera: species/genus/family 11.1 / 1.2 / 1.1%.
- strict unseen-genera gap recall:
  - hidden species: species-gap recall 54.0%;
  - hidden genus: genus-gap recall 1.0%;
  - hidden family: family-gap recall 2.0%.

Interpretation: species gaps are partly detectable but still noisy on normal
supported cases, while genus/family gaps are flagged conservatively with very
low recall. This is a useful diagnostic boundary: per-query top-k evidence is
not enough for a mature reference-gap detector. The next version needs
candidate-level tree-neighborhood features, BLAST/VSEARCH identity features,
and reference-library density features.

Seed1301 no-counts follow-up:

- target-0.95:
  - hidden unseen-genera species gaps: species-gap recall 92.8% at 100.0%
    precision;
  - hidden unseen-genera genus gaps: genus-gap recall 5.7% in hide-genus and
    23.5% in hide-family, both at 100.0% precision;
  - hidden unseen-genera family gaps: family-gap recall 7.0% in hide-family,
    still weak;
  - normal supported false-gap flags rise, especially species flags.
- target-0.99:
  - behaves similarly to the previous no-counts target-0.99 detector:
    conservative, high precision when it fires, weak genus/family recall.

Interpretation:

Lowering the precision target can recover more missing-reference cases, but the
normal supported false-gap rate also rises. This is useful as a tunable warning
signal, not a final reason layer. The scientific result is that missing species
are much easier to detect than missing genus/family from the current evidence
traces.

## Candidate-Evidence V2

The v2 detector changes the input representation. Instead of using only
aggregate production traces, it builds candidate-level evidence from the top-k
list:

- p-distance and score traces;
- candidate taxonomic diversity;
- same-as-top1 species/genus/family/order fractions;
- nearest-reference density features;
- optional tree-neighborhood evidence from the fish species tree.

It deliberately avoids global candidate-count features and split-specific
flags.

Target-0.95 strict unseen-genera gap recall:

| Detector | Hidden Species Species Gap | Hidden Genus Genus Gap | Hide-Family Genus Gap | Hide-Family Family Gap |
|---|---:|---:|---:|---:|
| no-counts v1 | 92.8% | 5.7% | 23.5% | 7.0% |
| v2 no-tree | 95.9% | 30.8% | 38.8% | 27.2% |
| v2 tree-aware | 95.4% | 30.0% | 41.3% | 32.6% |

Target-0.99 tree-aware v2 is more conservative:

- hidden species species-gap recall: 79.7%;
- hidden genus genus-gap recall: 10.1%;
- hide-family genus-gap recall: 19.0%;
- hide-family family-gap recall: 18.9%.

Supported-case warning rates are still the constraint. At target 0.95, the
tree-aware v2 detector flags normal supported unseen-genera species/genus/
family as gaps at 34.3 / 10.7 / 7.0%, and normal supported held-out fish at
15.3 / 3.4 / 2.4%. Target 0.99 lowers warnings but loses recall.

Interpretation:

Candidate-level evidence is a real improvement for hidden-reference diagnosis.
Tree-neighborhood features improve the hardest hide-family genus/family cells,
but sequence/taxonomic candidate-list evidence is the main gain. This remains a
diagnostic reason layer. It should feed a gap-aware rank/no-call policy, not
replace the production-v1 policy by itself.

## Next Experiments

1. Build a missing-reference-aware rank/no-call policy that consumes v2
   reference-gap scores as soft warnings.
2. Keep production species calls disabled unless the policy transfers under
   strict hidden-reference stress.
3. Add diagnostic reason labels to the production CLI output.
4. Use high-confidence gap predictions to prioritize reference-library
   curation.
