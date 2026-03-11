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

python run_workflow.py \
  --years 2012 2013 2014 2015 2016 2017 2018 2019 \
  --months 01 04 07 10 \
  --ensembles 01 02 03 04 05
