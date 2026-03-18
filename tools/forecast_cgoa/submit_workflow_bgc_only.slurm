#!/bin/bash
#SBATCH --job-name=cgoa_ic_obc
#SBATCH --partition=analysis
#SBATCH --constraint=bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -eo pipefail

module load miniforge
module load nco

# Some site activate hooks (e.g., magics-activate.sh) reference MAGPLUS_HOME and
# fail under nounset if it is undefined. Keep activation robust.
export MAGPLUS_HOME="${MAGPLUS_HOME:-}"
set +u
conda activate /nbhome/role.medgrp/.conda/envs/medpy311
set -u

WORKFLOW_DIR=/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/forecast_cgoa
cd "${WORKFLOW_DIR}"
python generate_configs.py \
  --years 2012 \
  --months 01 \
  --ensembles 01 02 03 04 05 \
  --output-root outputs \
  --config-root generated_configs

for m in 01; do
  for e in 01 02 03 04 05; do
    python run_bgc_obc.py \
      --config "generated_configs/2012/${m}/e${e}/obc_bgc.yaml" \
      --year 2012 \
      --month "${m}" \
      --ensemble "${e}" \
      --output-root outputs \
      --force
  done
done
