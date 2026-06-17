#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/Users/kluless/marinemamba}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/marinemamba}"
RUN_DATE="${RUN_DATE:-2026-05-26}"

H200_HOST="${H200_HOST:-93.91.156.89}"
H200_PORT="${H200_PORT:-50952}"
RTX_HOST="${RTX_HOST:-140.150.159.1}"
RTX_PORT="${RTX_PORT:-25513}"

copy_pattern() {
  local label="$1"
  local host="$2"
  local port="$3"
  local pattern="$4"
  local dest="${LOCAL_ROOT}/results/remote_runs/${RUN_DATE}/${label}"

  mkdir -p "${dest}"
  ssh -p "${port}" "root@${host}" \
    "find '${REMOTE_ROOT}/results' -maxdepth 1 -type d -name '${pattern}' -print" |
    while IFS= read -r remote_dir; do
      if [[ -n "${remote_dir}" ]]; then
        scp -P "${port}" -r "root@${host}:${remote_dir}" "${dest}/"
        echo "copied ${label}:${remote_dir} -> ${dest}"
      fi
    done
}

copy_pattern h200 "${H200_HOST}" "${H200_PORT}" "coi_fish_tree_clean_phylo_mamba_cosine_dim384*"
copy_pattern rtx_pro "${RTX_HOST}" "${RTX_PORT}" "coi_fish_tree_clean_phylo_mamba_cosine_dim384*"
