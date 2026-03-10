#!/bin/bash -e
#SBATCH --time=24:00:00
#SBATCH --output=logs/padd%j.out

module load miniforge
conda activate /nbhome/role.medgrp/.conda/envs/medpy311
module load gcp
module load cdo
module load nco


./padd.sh
