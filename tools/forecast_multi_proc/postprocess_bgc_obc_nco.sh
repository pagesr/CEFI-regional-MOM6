#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SCRIPT="${SCRIPT_DIR}/../forecast_cgoa/postprocess_bgc_obc_nco.sh"

if [[ ! -x "$SOURCE_SCRIPT" ]]; then
  echo "ERROR: source script not found or not executable: $SOURCE_SCRIPT" >&2
  exit 1
fi

exec "$SOURCE_SCRIPT" "$@"
