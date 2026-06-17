#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS="results/remote_runs/2026-05-26/rtx_pro/global_edna_multisource_teleo_hier_strong_seed1207_predictions/zero_shot_candidate_predictions.csv"
INPUT_DIR="data/edna/real_edna_queries/global_tropical_multisource_teleo"
SAMPLE_MAP="${INPUT_DIR}/sample_query_map.csv"
OBIS_PRIOR="data/edna/raw/real_edna/global_obis_range_prior_site20_pad05/obis_site_prior_counts.csv"

python3 scripts/edna/eval_global_edna_occurrence_prior_rerank.py \
  --predictions "${PREDICTIONS}" \
  --sample-query-map "${SAMPLE_MAP}" \
  --input-dir "${INPUT_DIR}" \
  --occurrence-prior-counts "${OBIS_PRIOR}" \
  --output-dir results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_obis_site20_w005 \
  --site-column site20 \
  --prior-weight 0.05 \
  --output-top-k 50

python3 scripts/edna/eval_global_edna_occurrence_prior_only.py \
  --input-dir "${INPUT_DIR}" \
  --sample-query-map "${SAMPLE_MAP}" \
  --occurrence-prior-counts "${OBIS_PRIOR}" \
  --output-dir results/edna/global_tropical_validation/obis_prior_only_site20 \
  --site-column site20 \
  --top-k 50

python3 scripts/edna/eval_global_edna_combined_prior_rerank.py \
  --predictions "${PREDICTIONS}" \
  --sample-query-map "${SAMPLE_MAP}" \
  --input-dir "${INPUT_DIR}" \
  --occurrence-prior-counts "${OBIS_PRIOR}" \
  --output-dir results/edna/global_tropical_validation/multisource_teleo_hier_strong_seed1207_rls_obis_site20_w005 \
  --site-column site20 \
  --rls-weight 0.05 \
  --obis-weight 0.05 \
  --output-top-k 50

python3 scripts/edna/summarize_global_edna_benchmarks.py
python3 scripts/edna/build_global_edna_calibration_matrix.py
python3 scripts/summarize_results_ledger.py
python3 scripts/figures/build_source_tables.py
python3 scripts/figures/plot_source_tables.py
