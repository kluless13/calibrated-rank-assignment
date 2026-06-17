#!/usr/bin/env bash
set -euo pipefail

ROOT="results/edna/resolvability"

python3 scripts/edna/build_12s_resolvability_map.py \
  --input-dir data/edna/stalder_inputs/multisource \
  --output-dir "${ROOT}/multisource_exact_acgt" \
  --normalization acgt \
  --min-length 30

python3 scripts/edna/build_12s_resolvability_map.py \
  --input-dir data/edna/stalder_inputs/multisource_teleo \
  --output-dir "${ROOT}/multisource_teleo_exact_acgt" \
  --normalization acgt \
  --min-length 30

python3 scripts/edna/build_12s_resolvability_map.py \
  --input-dir data/edna/stalder_inputs/rcrux_cleaned \
  --output-dir "${ROOT}/rcrux_cleaned_exact_acgt" \
  --normalization acgt \
  --min-length 30

python3 scripts/edna/build_12s_resolvability_map.py \
  --input-dir data/edna/stalder_inputs/mitohelper_full_teleo \
  --output-dir "${ROOT}/mitohelper_full_teleo_exact_acgt" \
  --normalization acgt \
  --min-length 30

python3 - <<'PY'
import json
from pathlib import Path
import pandas as pd

roots = {
    "multisource": "results/edna/resolvability/multisource_exact_acgt",
    "multisource_teleo": "results/edna/resolvability/multisource_teleo_exact_acgt",
    "rcrux_cleaned": "results/edna/resolvability/rcrux_cleaned_exact_acgt",
    "mitohelper_full_teleo": "results/edna/resolvability/mitohelper_full_teleo_exact_acgt",
}
rows = []
for name, path in roots.items():
    data = json.loads((Path(path) / "resolvability_manifest.json").read_text())
    query = data.get("query_oracle_metrics") or {}
    rows.append({
        "dataset": name,
        "species_with_sequences": data["species_with_sequences"],
        "sequence_records_used": data["sequence_records_used"],
        "exact_cluster_count": data["exact_cluster_count"],
        "species_best_species_count": data["species_best_rank_counts"]["species"],
        "species_best_species_fraction": data["species_best_rank_counts"]["species"] / data["species_with_sequences"],
        "query_count": query.get("query_count"),
        "query_reference_exact_cluster_found_rate": query.get("reference_exact_cluster_found_rate"),
        "query_species_oracle_supported_rate": query.get("species_oracle_supported_rate"),
        "query_genus_oracle_supported_rate": query.get("genus_oracle_supported_rate"),
        "query_family_oracle_supported_rate": query.get("family_oracle_supported_rate"),
        "query_order_oracle_supported_rate": query.get("order_oracle_supported_rate"),
    })
out = Path("results/edna/resolvability/resolvability_overview.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out}")
PY

