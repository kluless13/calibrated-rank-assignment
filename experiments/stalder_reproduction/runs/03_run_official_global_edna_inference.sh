#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

python3 scripts/edna/run_official_taxdna_global_edna.py \
  --taxdna-repo data/edna/raw/stalder_taxdna \
  --input-dir data/edna/raw/real_edna/global_tropical_swarm_representatives \
  --output-dir results/edna/stalder_reproduction/official_taxdna_global_edna \
  "$@"
