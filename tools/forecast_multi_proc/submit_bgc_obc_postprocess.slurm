#!/bin/bash
#SBATCH --job-name=bgc_obc_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

: "${TASK_LIST:?TASK_LIST is required}"

# Optional: keep originals even when merge succeeds.
REMOVE_ORIGINALS_ON_SUCCESS="${REMOVE_ORIGINALS_ON_SUCCESS:-1}"

# REPO_ROOT is optional. If not set, default to where sbatch was launched.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "ERROR: REPO_ROOT directory not found: $REPO_ROOT" >&2
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -f "$TASK_LIST" ]]; then
  echo "ERROR: TASK_LIST not found: $TASK_LIST" >&2
  exit 1
fi

module load nco

merge_script="${MERGE_SCRIPT:-$REPO_ROOT/tools/forecast_multi_proc/postprocess_bgc_obc_nco.sh}"
if [[ ! -x "$merge_script" ]]; then
  task_dir="$(cd "$(dirname "$TASK_LIST")" && pwd)"
  fallback_merge_script="$task_dir/postprocess_bgc_obc_nco.sh"
  if [[ -x "$fallback_merge_script" ]]; then
    merge_script="$fallback_merge_script"
  else
    echo "ERROR: merge script missing or not executable: $merge_script" >&2
    echo "       (also tried TASK_LIST sibling: $fallback_merge_script)" >&2
    exit 1
  fi
fi

processed_count=0

while read -r output_dir year month ensemble final_out; do
  # Skip blank lines and comments
  if [[ -z "${output_dir:-}" || "${output_dir:0:1}" == "#" ]]; then
    continue
  fi

  month="$(printf '%02d' "$month")"
  ensemble="$(printf '%02d' "$ensemble")"

  if [[ -n "${final_out:-}" ]]; then
    "$merge_script" "$output_dir" "$year" "$month" "$ensemble" "$final_out"
    produced_file="$final_out"
  else
    "$merge_script" "$output_dir" "$year" "$month" "$ensemble"
    produced_file="$output_dir/bgc_obc_${year}_${month}_e${ensemble}.nc"
  fi

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

  processed_count=$((processed_count + 1))
done < "$TASK_LIST"

echo "Done. Processed ${processed_count} merge task(s)."
