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
PROC_DIR="${PROC_DIR:-$SCRIPT_DIR}"

# Optional override; by default we use the task-list in this directory.
TASK_LIST="${TASK_LIST:-$PROC_DIR/bgc_obc_postprocess_tasks.txt}"

# Optional: keep originals even when merge succeeds.
REMOVE_ORIGINALS_ON_SUCCESS="${REMOVE_ORIGINALS_ON_SUCCESS:-1}"

# Optional: reuse an existing reggrid file during merge if supported by merge script.
# Leave empty to disable.
REUSE_REGGRID_FILE="${REUSE_REGGRID_FILE:-}"

if [[ ! -f "$TASK_LIST" ]]; then
  echo "ERROR: TASK_LIST not found: $TASK_LIST" >&2
  exit 1
fi

cd "$PROC_DIR"

module load nco

merge_script="${MERGE_SCRIPT:-$PROC_DIR/postprocess_bgc_obc_nco.sh}"
if [[ ! -x "$merge_script" ]]; then
  echo "ERROR: merge script missing or not executable: $merge_script" >&2
  exit 1
fi

processed_count=0

while read -r output_dir year month ensemble final_out; do
  # Skip blank lines and comments
  if [[ -z "${output_dir:-}" || "${output_dir:0:1}" == "#" ]]; then
    continue
  fi

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

  processed_count=$((processed_count + 1))
done < "$TASK_LIST"

echo "Done. Processed ${processed_count} merge task(s)."
