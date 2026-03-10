#!/bin/bash
module load slurm

for YEAR in {1995..2024}; do
  sbatch --export=YEAR=$YEAR submit_bgc_boundary.sh
done
