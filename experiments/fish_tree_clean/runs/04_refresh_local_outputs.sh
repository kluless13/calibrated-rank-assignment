#!/usr/bin/env bash
set -euo pipefail

python3 scripts/summarize_results_ledger.py
python3 scripts/figures/build_source_tables.py
python3 scripts/figures/plot_source_tables.py
