#!/bin/bash
#SBATCH --job-name=cgoa_phy_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -euo pipefail

: "${TASK_FILE:?TASK_FILE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${REPO_ROOT:?REPO_ROOT is required}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/nbhome/role.medgrp/.conda/envs/medpy311}"

if [[ -f "/nbhome/role.medgrp/.conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source /nbhome/role.medgrp/.conda/etc/profile.d/conda.sh
elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.conda/etc/profile.d/conda.sh"
else
  echo "ERROR: could not find conda.sh to activate env: $CONDA_ENV_PATH" >&2
  exit 127
fi

cd "$REPO_ROOT"

set +u
conda activate "$CONDA_ENV_PATH"
set -u

python tools/forecast_multi_proc/run_task_from_tsv.py \
  --stage phy \
  --task-file "$TASK_FILE" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-root "$OUTPUT_ROOT" \
  ${FORCE_FLAG:-}
