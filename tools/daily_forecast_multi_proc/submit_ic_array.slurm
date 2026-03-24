#!/bin/bash
#SBATCH --job-name=cgoa_ic_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

: "${TASK_FILE:?TASK_FILE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${REPO_ROOT:?REPO_ROOT is required}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/nbhome/role.medgrp/.conda/envs/medpy311}"

HAVE_CONDA_SH=0
if [[ -f "/nbhome/role.medgrp/.conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source /nbhome/role.medgrp/.conda/etc/profile.d/conda.sh
  HAVE_CONDA_SH=1
elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.conda/etc/profile.d/conda.sh"
  HAVE_CONDA_SH=1
else
  echo "WARNING: could not find conda.sh; will try direct python path from CONDA_ENV_PATH." >&2
fi

cd "$REPO_ROOT"

if [[ "$HAVE_CONDA_SH" == "1" ]] && command -v conda >/dev/null 2>&1; then
  set +u
  conda activate "$CONDA_ENV_PATH"
  set -u
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$CONDA_ENV_PATH/bin/python" ]]; then
    PYTHON_BIN="$CONDA_ENV_PATH/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: no python executable found (python/python3)." >&2
    exit 127
  fi
fi

"$PYTHON_BIN" tools/daily_forecast_multi_proc/run_task_from_tsv.py \
  --stage ic \
  --task-file "$TASK_FILE" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-root "$OUTPUT_ROOT" \
  ${FORCE_FLAG:-}
