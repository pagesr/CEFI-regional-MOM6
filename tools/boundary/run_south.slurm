#!/bin/bash -e
#SBATCH --time=24:00:00
#SBATCH --output=logs/bc_south%j.out

module load miniforge
conda activate /nbhome/role.medgrp/.conda/envs/medpy311
module load gcp
module load cdo
module load nco



python -u write_CGOA_boundary_south.py --config glorys_obc_CGOA_south.yaml
