#!/bin/bash -e
#SBATCH --time=24:00:00
#SBATCH --output=logs/glorys_coars%j.out
module load miniforge
conda activate boundary_remi 
module load gcp
module load fre/bronx-22
module load cdo

./write_bgc_initial_CGOA.py --config bgc_ic_CGOA.yaml
