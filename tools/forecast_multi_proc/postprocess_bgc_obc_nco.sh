#!/usr/bin/env bash
# Merge BGC OBC segment files into one file per ensemble using NCO.
#
# Expected per-tracer inputs in OUTPUT_DIR:
#   <tracer>_<SEGMENT3>_<YEAR>.nc
#
# Output:
#   bgc_obc_<YEAR>_<MONTH>_e<ENSEMBLE>.nc

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  postprocess_bgc_obc_nco.sh <output_dir> <year> <month> <ensemble> [final_output] [reggrid_file]
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 4 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi

OUTPUT_DIR="$1"
YEAR="$2"
MONTH="$(printf '%02d' "$3")"
ENSEMBLE="$(printf '%02d' "$4")"
FINAL_OUT="${5:-${OUTPUT_DIR}/bgc_obc_${YEAR}_${MONTH}_e${ENSEMBLE}.nc}"
REGGRID_FILE="${6:-}"

if ! command -v ncks >/dev/null 2>&1; then
  echo "ERROR: ncks not found in PATH. Load/install NCO first." >&2
  exit 127
fi

if ! command -v ncatted >/dev/null 2>&1; then
  echo "ERROR: ncatted not found in PATH. Load/install NCO first." >&2
  exit 127
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python not found in PATH." >&2
  exit 127
fi

if [[ -n "$REGGRID_FILE" && ! -f "$REGGRID_FILE" ]]; then
  echo "ERROR: reggrid_file not found: $REGGRID_FILE" >&2
  exit 1
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

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

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

echo "Found ${#tracers[@]} tracer(s) in $OUTPUT_DIR"

merged_tracers=()

###############################################################################
# Merge segments within each tracer
###############################################################################
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

  ncatted -O -h -a history,global,d,, "$tracer_out" >/dev/null 2>&1 || true

  echo "Merging tracer '$tracer' from ${#seg_files[@]} segment file(s)..."

  for sf in "${seg_files[@]:1}"; do
    ncatted -O -h -a history,global,d,, "$sf" >/dev/null 2>&1 || true
    ncks -A -h "$sf" "$tracer_out"
  done

  ncatted -O -h -a history,global,d,, "$tracer_out" >/dev/null 2>&1 || true
  merged_tracers+=( "$tracer_out" )
done

if [[ ${#merged_tracers[@]} -eq 0 ]]; then
  echo "ERROR: no merged tracer files produced." >&2
  exit 1
fi

###############################################################################
# Seed final file with first merged tracer file
###############################################################################
mkdir -p "$(dirname "$FINAL_OUT")"
cp -f "${merged_tracers[0]}" "$FINAL_OUT"
ncatted -O -h -a history,global,d,, "$FINAL_OUT" >/dev/null 2>&1 || true

echo "Building final file: $FINAL_OUT"

###############################################################################
# Append only true tracer data variables from each remaining merged tracer file
###############################################################################
for tf in "${merged_tracers[@]:1}"; do
  b="$(basename "$tf")"
  tracer="${b%_${YEAR}_merged.nc}"

  ncatted -O -h -a history,global,d,, "$tf" >/dev/null 2>&1 || true

  vars_to_append="$(
python - "$tf" "$tracer" <<'PY'
import sys
import xarray as xr

path = sys.argv[1]
tracer = sys.argv[2]

ds = xr.open_dataset(path, decode_times=False)

# Skip standard coordinates / metadata / bounds-like variables
skip_exact = {
    "time",
    "nv",
}
skip_prefixes = (
    "lat_",
    "lon_",
    "geolat_",
    "geolon_",
    "time_",
    "z_",
    "zl",
    "zi",
    "dz_",
    "depth",
    "ht",
    "wet",
    "mask",
)

vars_keep = []
for v in ds.data_vars:
    if v in skip_exact:
        continue
    if any(v.startswith(p) for p in skip_prefixes):
        continue
    # Keep variables matching the tracer name or beginning with tracer_
    if v == tracer or v.startswith(tracer + "_"):
        vars_keep.append(v)

ds.close()
print(",".join(vars_keep))
PY
)"

  if [[ -z "$vars_to_append" ]]; then
    echo "ERROR: no appendable variables found in $tf for tracer $tracer" >&2
    echo "Try: ncdump -h $tf | less" >&2
    exit 1
  fi

  echo "Appending from $(basename "$tf"): $vars_to_append"
  ncks -A -h -C -v "$vars_to_append" "$tf" "$FINAL_OUT"
done

###############################################################################
# Optional reggrid append
###############################################################################
if [[ -n "$REGGRID_FILE" ]]; then
  reggrid_tmp="$tmp_dir/reggrid_append.nc"
  cp -f "$REGGRID_FILE" "$reggrid_tmp"
  ncatted -O -h -a history,global,d,, "$reggrid_tmp" >/dev/null 2>&1 || true
  echo "Appending reggrid file: $REGGRID_FILE"
  ncks -A -h "$reggrid_tmp" "$FINAL_OUT"
fi

ncatted -O -h -a history,global,d,, "$FINAL_OUT" >/dev/null 2>&1 || true

if [[ ! -s "$FINAL_OUT" ]]; then
  echo "ERROR: final output file was not created correctly: $FINAL_OUT" >&2
  exit 1
fi

echo "Wrote final merged BGC OBC file: $FINAL_OUT"
