#!/usr/bin/env bash
# Merge BGC OBC segment files into one file per ensemble using NCO.
#
# Expected per-tracer inputs in OUTPUT_DIR (created by OBC_BGC.py):
#   <tracer>_<SEGMENT3>_<YEAR>.nc
# Example:
#   alk_001_2012.nc alk_002_2012.nc
#
# Output:
#   bgc_obc_<YEAR>_<MONTH>_e<ENSEMBLE>.nc

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  postprocess_bgc_obc_nco.sh <output_dir> <year> <month> <ensemble> [final_output]

Arguments:
  output_dir    Directory containing tracer segment files from OBC_BGC.py
  year          4-digit year (e.g., 2012)
  month         2-digit month (e.g., 01)
  ensemble      2-digit ensemble (e.g., 01)
  final_output  Optional full path for final merged file

Example:
  postprocess_bgc_obc_nco.sh outputs/2012/01/OBC/BGC/e01 2012 01 01
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

OUTPUT_DIR="$1"
YEAR="$2"
MONTH="$(printf '%02d' "$3")"
ENSEMBLE="$(printf '%02d' "$4")"
FINAL_OUT="${5:-${OUTPUT_DIR}/bgc_obc_${YEAR}_${MONTH}_e${ENSEMBLE}.nc}"

if ! command -v ncks >/dev/null 2>&1; then
  echo "ERROR: ncks not found in PATH. Load/install NCO first." >&2
  exit 127
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "ERROR: output_dir not found: $OUTPUT_DIR" >&2
  exit 1
fi

shopt -s nullglob
files=( "$OUTPUT_DIR"/*_[0-9][0-9][0-9]_"$YEAR".nc )
shopt -u nullglob
if [[ ${#files[@]} -eq 0 ]]; then
  echo "ERROR: no tracer segment files found in $OUTPUT_DIR for year $YEAR" >&2
  exit 1
fi

tmp_dir="$OUTPUT_DIR/.merge_tmp_${YEAR}_${MONTH}_e${ENSEMBLE}"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

declare -A tracer_seen=()
tracers=()

for f in "${files[@]}"; do
  b="$(basename "$f")"
  tracer="${b%_[0-9][0-9][0-9]_${YEAR}.nc}"
  if [[ -z "${tracer_seen[$tracer]+x}" ]]; then
    tracer_seen["$tracer"]=1
    tracers+=( "$tracer" )
  fi
done

IFS=$'\n' tracers=( $(printf '%s\n' "${tracers[@]}" | sort) )
unset IFS

merged_tracers=()
for tracer in "${tracers[@]}"; do
  shopt -s nullglob
  seg_files=( "$OUTPUT_DIR"/"${tracer}"_[0-9][0-9][0-9]_"$YEAR".nc )
  shopt -u nullglob
  if [[ ${#seg_files[@]} -eq 0 ]]; then
    continue
  fi

  IFS=$'\n' seg_files=( $(printf '%s\n' "${seg_files[@]}" | sort) )
  unset IFS

  tracer_out="$tmp_dir/${tracer}_${YEAR}_merged.nc"
  cp -f "${seg_files[0]}" "$tracer_out"
  for sf in "${seg_files[@]:1}"; do
    ncks -A "$sf" "$tracer_out"
  done
  merged_tracers+=( "$tracer_out" )
done

if [[ ${#merged_tracers[@]} -eq 0 ]]; then
  echo "ERROR: no merged tracer files produced." >&2
  exit 1
fi

cp -f "${merged_tracers[0]}" "$FINAL_OUT"
for tf in "${merged_tracers[@]:1}"; do
  ncks -A "$tf" "$FINAL_OUT"
done

echo "Wrote final merged BGC OBC file: $FINAL_OUT"
