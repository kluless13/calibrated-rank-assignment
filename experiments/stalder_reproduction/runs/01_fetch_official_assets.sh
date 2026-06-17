#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

REPO="${STALDER_TAXDNA_REPO:-data/edna/raw/stalder_taxdna}"
REMOTE_URL="${STALDER_TAXDNA_REMOTE:-https://gitlab.renkulab.io/dnai/TAXDNA.git}"

if [[ ! -d "${REPO}/.git" ]]; then
  mkdir -p "$(dirname "${REPO}")"
  git clone "${REMOTE_URL}" "${REPO}"
fi

if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
  cat >&2 <<'EOF'
git-lfs is required to fetch the official TAXDNA assets.

Install it first, then rerun this script:

  brew install git-lfs
  git lfs install

If Git LFS cannot access the assets, use the Renku project data browser/export:

  https://renkulab.io/v2/projects/sdsc/taxdna
EOF
  exit 2
fi

git -C "${REPO}" lfs install --local
git -C "${REPO}" lfs pull

python3 scripts/edna/inventory_stalder_assets.py \
  --repo "${REPO}" \
  --output data/edna/raw/stalder_taxdna_manifest.json

python3 scripts/edna/check_stalder_reproduction_assets.py \
  --repo "${REPO}" \
  --output results/edna/stalder_reproduction/asset_readiness.json
