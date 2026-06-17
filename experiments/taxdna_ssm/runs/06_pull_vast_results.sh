#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/Users/kluless/marinemamba}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/marinemamba}"
RUN_DATE="${RUN_DATE:-2026-05-26}"

H200_HOST="${H200_HOST:-93.91.156.89}"
H200_PORT="${H200_PORT:-50952}"
RTX_HOST="${RTX_HOST:-140.150.159.1}"
RTX_PORT="${RTX_PORT:-25513}"

copy_remote_dir() {
  local label="$1"
  local host="$2"
  local port="$3"
  local rel_dir="$4"
  local dest="${LOCAL_ROOT}/results/remote_runs/${RUN_DATE}/${label}"

  mkdir -p "${dest}"
  if ssh -p "${port}" "root@${host}" "test -d '${REMOTE_ROOT}/${rel_dir}'"; then
    scp -P "${port}" -r "root@${host}:${REMOTE_ROOT}/${rel_dir}" "${dest}/"
    echo "copied ${label}:${rel_dir} -> ${dest}"
  else
    echo "skip ${label}:${rel_dir}; remote directory not found"
  fi
}

copy_remote_dir h200 "${H200_HOST}" "${H200_PORT}" "results/edna/taxdna_ssm"
copy_remote_dir rtx_pro "${RTX_HOST}" "${RTX_PORT}" "results/edna/taxdna_ssm"
