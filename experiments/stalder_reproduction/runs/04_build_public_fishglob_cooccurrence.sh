#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

RAW_DIR="data/edna/raw/stalder_public_sources/fishglob"
RDATA="$RAW_DIR/FishGlob_public_clean.RData"
METADATA_RDATA="$RAW_DIR/FishGlob_public_metadata_clean.RData"
CSV_GZ="$RAW_DIR/FishGlob_public_clean_taxdna_columns.csv.gz"
OUTPUT_DIR="data/edna/cooccurrence_inputs/stalder_public"

mkdir -p "$RAW_DIR" "$OUTPUT_DIR"

if [[ ! -f "$RDATA" ]]; then
  curl -L --fail --retry 3 \
    -o "$RDATA" \
    "https://raw.githubusercontent.com/fishglob/FishGlob_data/main/outputs/Compiled_data/FishGlob_public_clean.RData"
fi

if [[ ! -f "$METADATA_RDATA" ]]; then
  curl -L --fail --retry 3 \
    -o "$METADATA_RDATA" \
    "https://raw.githubusercontent.com/fishglob/FishGlob_data/main/outputs/Compiled_data/FishGlob_public_metadata_clean.RData"
fi

if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript is required to convert the public FISHGLOB RData export." >&2
  exit 2
fi

if [[ ! -f "$CSV_GZ" ]]; then
  Rscript -e "
    infile <- '$RDATA'
    outfile <- '$CSV_GZ'
    e <- new.env()
    load(infile, envir=e)
    x <- get('data', envir=e)
    cols <- c(
      'survey', 'haul_id', 'year', 'latitude', 'longitude', 'accepted_name',
      'aphia_id', 'order', 'family', 'genus', 'num', 'num_cpue', 'num_cpua',
      'wgt', 'wgt_cpue', 'wgt_cpua', 'survey_unit'
    )
    cols <- cols[cols %in% names(x)]
    utils::write.csv(x[, cols], gzfile(outfile), row.names=FALSE, na='')
    cat(outfile, '\n')
    cat(nrow(x), 'rows\n')
    cat(paste(cols, collapse=','), '\n')
  "
fi

python scripts/edna/build_public_fishglob_taxdna_json.py \
  --fishglob-csv "$CSV_GZ" \
  --input-dir data/edna/real_edna_queries/global_tropical_multisource_teleo \
  --output-dir "$OUTPUT_DIR" \
  --min-species-per-group 2 \
  --encoding latin1
