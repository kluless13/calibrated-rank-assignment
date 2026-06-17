#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO="${STALDER_TAXDNA_REPO:-data/edna/raw/stalder_taxdna}"

python3 scripts/edna/inventory_stalder_assets.py \
  --repo "${REPO}" \
  --output data/edna/raw/stalder_taxdna_manifest.json

python3 scripts/edna/check_stalder_reproduction_assets.py \
  --repo "${REPO}" \
  --output results/edna/stalder_reproduction/asset_readiness.json
