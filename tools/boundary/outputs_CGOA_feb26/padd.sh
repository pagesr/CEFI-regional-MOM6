#!/bin/bash
set -euo pipefail

module load nco/5.0.1 || true   # if needed on your system

# edit as needed
START=1993
END=2024
VARS=("so" "thetao" "uv" "zos")
SEGS=("001" "002")

OUTDIR="./padded"
TMPDIR="./_tmp_pad"
mkdir -p "$OUTDIR" "$TMPDIR"

for var in "${VARS[@]}"; do
  for seg in "${SEGS[@]}"; do
    for ((y=START; y<=END; y++)); do

      cur="${var}_${seg}_${y}.nc"
      [[ -f "$cur" ]] || { echo "Missing $cur, skipping"; continue; }

      parts=()

      # prepend last record from previous year (ORIGINAL file)
      prev=$((y-1))
      prevf="${var}_${seg}_${prev}.nc"
      if [[ $y -gt $START && -f "$prevf" ]]; then
        prev_last="${TMPDIR}/${var}_${seg}_${prev}_last.nc"
        ncks -O -d time,-1 "$prevf" "$prev_last"
        parts+=("$prev_last")
      fi

      # current year
      parts+=("$cur")

      # append first record from next year (ORIGINAL file)
      next=$((y+1))
      nextf="${var}_${seg}_${next}.nc"
      if [[ $y -lt $END && -f "$nextf" ]]; then
        next_first="${TMPDIR}/${var}_${seg}_${next}_first.nc"
        ncks -O -d time,0 "$nextf" "$next_first"
        parts+=("$next_first")
      fi

      out="${OUTDIR}/${var}_${seg}_${y}_pad.nc"
      echo "Writing $out  (parts: ${#parts[@]})"
      ncrcat -O "${parts[@]}" "$out"

      # optional: quick sanity check (prints first/last time)
      # ncks -H -C -v time "$out" | head -n 2
      # ncks -H -C -v time "$out" | tail -n 2

    done
  done
done

echo "Done. Padded files are in: $OUTDIR"
