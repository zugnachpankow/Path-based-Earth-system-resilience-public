#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=run_12
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=600G
#SBATCH --output=logs/other/%x-%j.out
#SBATCH --error=logs/other/%x-%j.err
#SBATCH --chdir=/home/maxbecht/Path-based-Earth-system-resilience-public

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "$SLURM_CPUS_PER_TASK cpus/task"
echo "------------------------------------------------------------"

module load anaconda/2025
source activate fair

python 12_finder.py