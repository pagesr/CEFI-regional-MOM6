#!/bin/bash
#SBATCH --job-name=cgoa_ic_obc
#SBATCH --partition=analysis
#SBATCH --constraint=bigmem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

module load miniforge
module load nco
conda activate /nbhome/role.medgrp/.conda/envs/medpy311

WORKFLOW_DIR=/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/forecast_cgoa
cd "${WORKFLOW_DIR}"

python run_workflow.py \
  --years 2012 2013 2014 2015 2016 2017 2018 2019 \
  --months 01 04 07 10 \
  --ensembles 01 02 03 04 05
