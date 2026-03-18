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

cd "$REPO_ROOT"

python tools/forecast_multi_proc/run_task_from_tsv.py \
  --stage phy \
  --task-file "$TASK_FILE" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --output-root "$OUTPUT_ROOT" \
  ${FORCE_FLAG:-}
