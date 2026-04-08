# MarineMamba

Training objective determines what neural networks learn from marine DNA barcodes: hierarchical classification and evolutionary tree recovery from COI sequences.

## Key Findings

### 1. Curriculum Learning → Stalder-Inspired Evaluation

Multi-head coarse-to-fine training (order → family → genus → species) on a BarcodeMamba backbone, evaluated on unseen species within seen genera (a Stalder-inspired protocol — see *Evaluation Notes* below):

| Level | Stalder et al. (2025, 12S) | **Ours (COI, Marine 869K)** |
|-------|---------------------------|----------------------------|
| Genus | 50.7% | **85.0%** |
| Family | 80.5% | **93.1%** |
| Order | 86.7% | **96.7%** |

Source: `results/eval_c_stalder_protocol.json`. Stalder et al. used the 12S marker with a 2-layer CNN on ~10K sequences; we use COI with BarcodeMamba on 869K marine sequences. Markers, datasets, and splits differ, so this is a contextual comparison, not a head-to-head on identical data.

On the standard unseen-genera evaluation (Eval A) with Marine 869K, the curriculum model reaches **57.3% family / 80.4% order** on 557 held-out genera, and **98.6% genus / 99.4% family** on seen-genera / new-species splits (Eval B). Source: `results/multihead_hierarchical_results.json`.

### 2. Phylogenetic Embeddings → Tree of Life Recovery

Trained to match Fish Tree of Life evolutionary distances (Rabosky et al. 2018).

**Tree recovery** (Pearson correlation between embedding distances and real evolutionary distances):

| Setting | Pearson r |
|---------|-----------|
| Trained species (fish-only, dim=128) | **0.969** |
| **Unseen species (dim=384)** | **0.865** |

Source: `results/phylo_fish_only_results.json`, `results/tree_recovery_unseen_dim384.json`. The model places unseen fish species at the correct evolutionary distance from other species, spanning 4–386 million years of divergence. This extends Stalder et al.'s demonstration of phylogenetic-embedding transfer with strong generalisation in the COI fish setting.

On the same Stalder-inspired evaluation, the phylo model (dim=128) reaches 57.2% genus / 80.3% family / 90.7% order — comparable to Stalder on family, better on genus and order. Source: `results/phylo_fish_only_results.json`.

### 3. Foundation Models Underperform

To our knowledge, the first published evaluation of Evo 2 (7B, Nature 2026) on DNA barcodes. Frozen: 88.4% species on Marine 869K, below the 93.0% k-NN 6-mer baseline on BOLD_318K. With LoRA fine-tuning: 80.6% species. Source: `results/model_f_evo2_results.json`, `results/model_g_lora_results.json`.

## Data

### Marine 869K (Full Benchmark)
- **869,222 COI barcode sequences** from [BOLD v5 API](https://portal.boldsystems.org/api)
- **14 marine phyla**: fish, crabs, molluscs, corals, jellyfish, sea stars, worms, sponges, sharks
- **76,925 species** | 14,216 genera | 1,850 families
- 557 genera held out for zero-shot evaluation

### Fish Tree of Life Subset
- 5,888 fish species matched to [Fish Tree of Life](https://fishtreeoflife.org) (Rabosky et al. 2018)
- 140,112 training sequences with real evolutionary distances in millions of years

## Experiments

All results below are backed by JSON files in `results/`.

| Experiment | Script | Dataset | Key Result | JSON |
|-----------|--------|---------|------------|------|
| BLAST | `03_baselines.py` | BOLD_318K | 90.2% species | `baselines.json` |
| k-NN 6-mer (species) | `03_baselines.py` | BOLD_318K | 93.0% species | `baselines.json` |
| k-NN 6-mer hierarchical (unseen) | `03_baselines.py` | BOLD_318K | 56.0% family / 65.3% order | `hierarchical_eval.json` |
| BarcodeMamba Model C (transfer) | `04_barcodemamba_models.py --mode transfer` | BOLD_318K | 87.2% finetune / 91.7% linear probe | `model_c_results.json` |
| BarcodeMamba Model D (scratch) | `04_barcodemamba_models.py --mode scratch` | BOLD_318K | 86.8% finetune / 91.9% linear probe | `model_d_results.json` |
| BarcodeMamba Model E (adapt) | `04_barcodemamba_models.py --mode adapt` | BOLD_318K | 86.2% finetune / 91.6% linear probe | `model_e_results.json` |
| Standard SSM hierarchical (unseen genera) | `hierarchical_eval_ssm.py` | BOLD_318K | C: 37.5/44.6 · D: 47.0/57.2 · E: 48.4/57.7 (family/order) | `hierarchical_eval_ssm.json` |
| Evo 2 7B frozen | `05_evo2_embeddings.py` | Marine 869K | 88.4% species | `model_f_evo2_results.json` |
| Evo 2 7B + LoRA | `10_evo2_lora_curriculum.py` | Marine 869K | 80.6% species | `model_g_lora_results.json` |
| **Curriculum (pretrained backbone)** | `09_multihead_hierarchical.py --pretrain-ckpt` | **Marine 869K** | **57.3% family / 80.4% order Eval A; 98.6% Eval B genus; 94.5% test species** | `multihead_hierarchical_results.json` |
| Curriculum + 6-mer | `11_curriculum_6mer.py` | Marine 869K | 53.9% family / 76.9% order Eval A | `curriculum_6mer_results.json` |
| **Curriculum Eval C (Stalder-inspired)** | `eval_c_stalder_protocol.py` | Marine 869K | **85.0% genus / 93.1% family / 96.7% order** | `eval_c_stalder_protocol.json` |
| **Phylo fish-only (dim=128)** | `12_phylo_fish_only.py --embed-dim 128` | Fish Tree of Life | **r=0.969 trained, 95.1% species, 80.3% family Eval C** | `phylo_fish_only_results.json` |
| **Tree recovery unseen (dim=384)** | `tree_recovery_unseen.py --embed-dim 384` | Fish Tree of Life | **r=0.865 (unseen–unseen), r=0.922 (unseen–train)** | `tree_recovery_unseen_dim384.json` |
| Tree recovery unseen (dim=128) | `tree_recovery_unseen.py --embed-dim 128` | Fish Tree of Life | r=0.854 | `tree_recovery_unseen_dim128.json` |
| Tree recovery unseen (dim=64) | `tree_recovery_unseen.py --embed-dim 64` | Fish Tree of Life | r=0.804 | `tree_recovery_unseen_dim64.json` |

**Pending:** A same-dataset standard SSM baseline on Marine 869K is planned to strengthen the "training objective vs. architecture" comparison. Current standard-SSM hierarchical numbers (Models C/D/E) are on BOLD_318K, so the curriculum vs. standard-SSM comparison across datasets should be treated as contextual, not head-to-head, until that rerun lands.

### Evaluation Notes

- **Eval A** — unseen genera: 557 genera held out of training; k-NN genus accuracy is mathematically 0 by construction. Reports family and order generalisation.
- **Eval B** — seen genera, new species: measures within-genus generalisation on species not seen during training.
- **Eval C** — Stalder-inspired: unseen species held out from genera that *are* in training. Uses local `holdout_fraction=0.2` and `min_species_per_genus=3`, so it is Stalder-inspired, not an exact reproduction of Stalder et al.'s split.

## Quick Start

```bash
# Clone
git clone https://github.com/kluless13/marinemamba.git
cd marinemamba

# Fetch BOLD marine data (requires internet)
python3 scripts/fetch_bold_marine.py
# Writes data/raw/merged_marine_barcodes.csv

# Process into train/test/unseen splits
# NOTE: 02_clean_and_split.py currently reads data/raw/merged_barcodes.csv — rename
# or symlink the fetch output until the script is patched:
ln -sf merged_marine_barcodes.csv data/raw/merged_barcodes.csv
python3 scripts/02_clean_and_split.py

# Run curriculum learning (needs GPU)
python3 scripts/09_multihead_hierarchical.py --data-dir data/processed --output-dir results \
    --pretrain-ckpt checkpoints/model_e/lightning_logs/version_3/checkpoints/last.ckpt

# Run phylogenetic embeddings on fish subset (needs dendropy)
pip install dendropy
python3 scripts/12_phylo_fish_only.py --data-dir data/processed --output-dir results --embed-dim 128
```

## Architecture

Built on [BarcodeMamba](https://arxiv.org/abs/2412.11084) (Mamba-2 SSM). 2 layers, d_model=384, character-level tokenization.

| Model | Params | Training Objective | Best For |
|-------|--------|-------------------|----------|
| Curriculum model | ~17M (4.3M backbone + 4 heads) | Order→Family→Genus→Species staged | Classification of novel taxa |
| Phylo model | ~4.5M (4.3M backbone + projection) | Match Fish Tree of Life distances | Evolutionary placement |

## Novelty

All claims below are intentionally hedged with "to our knowledge" because exhaustive literature search across preprints and posters is infeasible. Each row states the concrete difference from the closest prior work.

| Claim | Positioning |
|-------|-------------|
| Strong tree-recovery generalisation on unseen COI species (r=0.865) | Stalder et al. (2025) demonstrated phylogenetic-embedding transfer to unseen species on 12S; DEPP (Jiang et al., 2022) discussed unseen-species generalisation as a failure mode for phylogenetic placement. We show strong generalisation in the COI fish setting with r=0.865 on species never seen during training. |
| Evo 2 on DNA barcode classification | To our knowledge, the first published evaluation. Evo 2's original paper (Brixi et al., Nature 2026) reports variant effect prediction and genome generation, not species identification from COI. |
| Taxonomic rank curriculum (staged head activation) | A distinct staged rank-by-rank training formulation. SnailBaLLsp (Ye et al., 2026) uses a hierarchy-aware decision chain at inference with data-modality staging during training; DNABERT-S (2024) uses a sample-difficulty curriculum; BarcodeMamba+ (2025) uses simultaneous multi-head outputs. Ours stages which classification head is active and how losses are weighted across training phases. |
| Phylogenetic distance training with SSM backbone on DNA barcodes | To our knowledge, a novel combination. Phyla (NeurIPS 2025) combined Mamba with phylogenetic loss on proteins; BarcodeMamba (Gao & Taylor, 2024) used an SSM on barcodes without a phylogenetic objective. |
| Marine-specific COI benchmark (869K, 14 phyla) | To our knowledge, the first large marine-specific COI benchmark spanning 14 marine phyla. DeepCOI (2025) introduced a larger cross-phyla COI benchmark with different taxon selection; ours is marine-specific with a 557-genus zero-shot holdout. |

## Key References

- BarcodeMamba (Gao & Taylor, 2024) — [arXiv:2412.11084](https://arxiv.org/abs/2412.11084)
- Stalder et al. (2025) — [PLOS Comp Bio](https://doi.org/10.1371/journal.pcbi.1013776)
- Ye et al. / SnailBaLLsp (2026) — [MEE](https://doi.org/10.1111/2041-210x.70264)
- DEPP (Jiang et al., 2022) — Systematic Biology
- DeepCOI (2025) — [Genome Biology](https://link.springer.com/article/10.1186/s13059-025-03861-7)
- Evo 2 (Brixi et al., 2026) — [Nature](https://doi.org/10.1038/s41586-026-10176-5)
- Fish Tree of Life (Rabosky et al., 2018) — [Nature](https://doi.org/10.1038/s41586-018-0273-1)

## Citation

```bibtex
@article{marinemamba2026,
  title={Training Objective Determines Hierarchical Classification and Evolutionary Recovery from Marine DNA Barcodes},
  author={TODO},
  year={2026}
}
```

## License

MIT
