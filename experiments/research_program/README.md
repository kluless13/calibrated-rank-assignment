# MarineMamba Research Program

Core goal:

> Inference about biodiversity from imperfect molecular evidence.

This folder tracks the overall research direction. The implementation remains
in the specific experiment folders such as `experiments/fish_tree_clean/`,
`experiments/taxdna_ssm/`, and `experiments/stalder_reproduction/`.

## Scientific Spine

The work should answer a linked sequence of questions:

1. Can short barcode sequences learn biologically meaningful structure?
2. Can that structure be expressed as position in a real fish species tree?
3. Can predictions be calibrated to the deepest defensible rank rather than
   forced to species?
4. Can reference gaps, marker ambiguity, and taxonomy explain uncertainty?
5. Can ecological context improve real eDNA assignment without hiding bias?
6. Can COI and 12S be joined through a shared species-tree coordinate system?

Each answer should make the next question possible.

## Workstreams

### Paper 1: Uncertainty-Aware Barcode And eDNA Inference

Folder: `experiments/paper1_phylo_calibrated_assignment/`

Combines:

- COI barcode-to-tree retrieval,
- neural/classical/placement comparator benchmark,
- rank-adaptive calibrated assignment,
- reference/missingness diagnostics,
- vector-first learned retrieval as the fast-tool direction.
- 12S/eDNA resolvability and ecological-context stress tests.

Practical implementation currently lives mainly in:

- `experiments/fish_tree_clean/`
- `experiments/paper1_phylo_calibrated_assignment/`
- `scripts/edna/eval_phylo_checkpoint_tree_recovery.py`
- `scripts/edna/eval_zero_shot_candidate_predictions.py`
- `scripts/edna/train_fish_tree_encoder_benchmark.py`
- `scripts/edna/eval_fish_tree_candidate_baselines.py`
- `experiments/taxdna_ssm/`
- `experiments/paper2_eco_phylo_edna/`
- `scripts/edna/build_12s_near_exact_resolvability.py`
- `scripts/edna/eval_global_edna_*`

### Paper 1 Work Package: Eco-Phylo eDNA Posterior

Folder: `experiments/paper2_eco_phylo_edna/`

Status: merged into Paper 1.

Combines:

- 12S/eDNA sequence assignment
- fixed species-tree candidate space
- co-occurrence/range/geographic priors
- Global_eDNA validation

Practical implementation currently lives mainly in:

- `experiments/taxdna_ssm/`
- `experiments/stalder_reproduction/`
- `scripts/edna/eval_global_edna_*`
- `scripts/edna/train_npz_cooccurrence_model.py`

### Paper 3: Multi-Marker Shared Tree Space

Folder: `experiments/paper3_multimarker_shared_tree_space/`

Connects:

- COI-rich species discrimination
- 12S/eDNA marker ambiguity
- one shared species-tree coordinate system

This remains the second separate manuscript/workstream after the merged Paper 1
is stable.

### Comparator/Replication Track

Folder: `experiments/stalder_reproduction/`

This is not a paper on its own yet. It supports Paper 1 by documenting what is
and is not possible for exact TAXDNA reproduction, and by hosting public
Stalder-style reconstruction work.

### Encoder Benchmark Track

Folder: `experiments/encoder_benchmarks/`

This cross-cuts all papers. The research program should not depend on claiming
that one architecture is always best. Mamba/SSM, CNN, LSTM, Transformer, S5,
k-mer, BLAST, and VSEARCH should be treated as swappable sequence-evidence
modules inside the same tree/ecology/calibration system.

## Build Order

Completed or substantially advanced:

- Paper 1 BLAST/VSEARCH/k-mer baselines and negative controls.
- Paper 1 CNN/biLSTM/Transformer encoder matrix.
- CNN seed repeats.
- COI rank-adaptive calibration first pass.
- Reference/missingness diagnostics.
- 12S exact and near-exact resolvability maps.

Current build order:

1. Finish Paper 1 phylogenetic placement comparators:
   EPA-ng/pplacer now; APPLES or equivalent next.
2. Implement placement-output scoring and Fernando-style metrics.
3. Build vector-first retrieval benchmark from saved embeddings.
4. Tighten Paper 1 rank-adaptive/no-call calibration using independent
   thresholds.
5. Formalize Eco-Phylo Posterior for 12S/eDNA inside the merged Paper 1.
6. Start Multi-Marker Shared Tree Space as the next paper.
7. Treat hyperbolic geometry as an optional mathematical ablation unless the
   cosine/tree-space results and baselines justify a dedicated push.

## Paper Logic

Paper 1 now combines the original Paper 1 and Paper 2:

> Short marker sequences support different levels of biodiversity inference
> depending on marker resolution, reference completeness, tree position, and
> ecological context. A defensible molecular biodiversity system should combine
> fast candidate retrieval, tree-aware scoring, reference diagnostics,
> ecological priors where justified, and calibrated rank/no-call assignment.

Next separate paper:

> COI and 12S can be connected through a shared species-tree coordinate system,
> allowing marker-specific evidence to support a unified biodiversity inference
> framework.

Architecture principle:

> The method should be encoder-agnostic. If MarineMamba is strong, it should win
> under a fair shared protocol; if another encoder wins, the paper still has
> value because the main contribution is calibrated biodiversity inference from
> imperfect molecular evidence.
