#!/bin/bash
#SBATCH --job-name=bgc_obc_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROC_DIR="${PROC_DIR:-${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}}"

TASK_LIST="${TASK_LIST:-$PROC_DIR/bgc_obc_postprocess_tasks.txt}"
REMOVE_ORIGINALS_ON_SUCCESS="${REMOVE_ORIGINALS_ON_SUCCESS:-1}"
REUSE_REGGRID_FILE="${REUSE_REGGRID_FILE:-}"

if [[ ! -f "$TASK_LIST" ]]; then
  echo "ERROR: TASK_LIST not found: $TASK_LIST" >&2
  exit 1
fi

cd "$PROC_DIR"

module load nco
module load miniforge

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /nbhome/role.medgrp/.conda/envs/medpy311
set -u

merge_script="${MERGE_SCRIPT:-$PROC_DIR/postprocess_bgc_obc_nco.sh}"
if [[ ! -x "$merge_script" ]]; then
  echo "ERROR: merge script missing or not executable: $merge_script" >&2
  exit 1
fi

processed_count=0

readarray -t tasks < <(awk 'NF && $1 !~ /^#/' "$TASK_LIST")
task_count="${#tasks[@]}"

if [[ "$task_count" -eq 0 ]]; then
  echo "ERROR: No runnable tasks found in $TASK_LIST" >&2
  exit 1
fi

process_task_line() {
  local task_line="$1"
  local output_dir year month ensemble final_out

  read -r output_dir year month ensemble final_out <<< "$task_line"

  month="$(printf '%02d' "$month")"
  ensemble="$(printf '%02d' "$ensemble")"

  merge_args=("$output_dir" "$year" "$month" "$ensemble")
  if [[ -n "${final_out:-}" ]]; then
    merge_args+=("$final_out")
    produced_file="$final_out"
  else
    produced_file="$output_dir/bgc_obc_${year}_${month}_e${ensemble}.nc"
  fi

  if [[ -n "$REUSE_REGGRID_FILE" ]]; then
    merge_args+=("$REUSE_REGGRID_FILE")
  fi

  "$merge_script" "${merge_args[@]}"

  if [[ ! -s "$produced_file" ]]; then
    echo "ERROR: merge did not produce a valid output file: $produced_file" >&2
    exit 1
  fi

  if [[ "$REMOVE_ORIGINALS_ON_SUCCESS" == "1" ]]; then
    shopt -s nullglob
    original_files=("$output_dir"/*_[0-9][0-9][0-9]_"$year".nc)
    shopt -u nullglob

    if [[ ${#original_files[@]} -gt 0 ]]; then
      rm -f "${original_files[@]}"
      echo "Removed ${#original_files[@]} original segment files from $output_dir"
    else
      echo "No original segment files to remove in $output_dir"
    fi
  fi
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  task_idx=$((SLURM_ARRAY_TASK_ID - 1))
  if (( task_idx < 0 || task_idx >= task_count )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID outside valid range 1-$task_count" >&2
    exit 1
  fi

  process_task_line "${tasks[$task_idx]}"
  processed_count=1
  echo "Done. Processed array task ${SLURM_ARRAY_TASK_ID}/${task_count}."
else
  for task_line in "${tasks[@]}"; do
    process_task_line "$task_line"
    processed_count=$((processed_count + 1))
  done
  echo "Done. Processed ${processed_count} merge task(s)."
fi

echo "Tip: submit one SLURM job per task line with:"
echo "  sbatch --array=1-${task_count} \"${BASH_SOURCE[0]}\""
