#!/bin/bash -e
#SBATCH --job-name=nep2goa_bgc_ic
#SBATCH --time=24:00:00
#SBATCH --output=logs/CGOA_BDRY_I%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

module load miniforge
module load nco
conda activate /nbhome/role.medgrp/.conda/envs/medpy311

# --- user settings ---
SCRIPT="/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/initial/nep_to_goa_bgc_ic.py"
YAML="/work/Remi.Pages/IC-BC-GOA/CEFI-regional-MOM6/tools/initial/nep_to_goa_bgc_ic.yaml"

mkdir -p logs

echo "=================================================="
echo "Host: $(hostname)"
echo "PWD : $(pwd)"
echo "Date: $(date)"
echo "Python: $(which python)"
python --version
echo "SCRIPT: $SCRIPT"
echo "YAML  : $YAML"
echo "=================================================="

# (Optional) show first few YAML lines (helps debugging)
echo "---- YAML head ----"
head -n 60 "$YAML" || true
echo "-------------------"

# Run
python "$SCRIPT" --config_file "$YAML"

echo "=================================================="
echo "Done: $(date)"
echo "=================================================="
