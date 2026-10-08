#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=run_33
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=logs/other/%x-%j.out
#SBATCH --error=logs/other/%x-%j.err
#SBATCH --chdir=/home/maxbecht/Path-based-Earth-system-resilience-public

module load anaconda/2025
source activate fair

python si/33_coarse_grid.py
