#!/bin/bash
#BATCH --job-name=write_bgc   # Job name
#SBATCH --output=log_%A_%a.out # Standard output file (one per array task)
#SBATCH --error=log_%A_%a.err  # Standard error file (one per array task)
#SBATCH --constraint=bigmem
#SBATCH --partition=analysis
#SBATCH --ntasks=1             # Number of tasks (1 per job)
#SBATCH --cpus-per-task=4      # Number of CPUs per task
#SBATCH --time=24:00:00        # Maximum runtime (adjust as needed)

module load miniforge
conda activate /nbhome/role.medgrp/.conda/envs/medpy311
module load nco  # Load required modules

# Use the YEAR variable passed by the loop
echo "Processing year $YEAR"
./write_bgc_boundary_nutrients.py --config bgc_obc.yaml --year $YEAR
