#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=run_14grid
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=113
#SBATCH --mem=250G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/other/%x-%j.out
#SBATCH --error=logs/other/%x-%j.err
#SBATCH --chdir=/home/maxbecht/Path-based-Earth-system-resilience-public

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID  |  $SLURM_CPUS_PER_TASK cpus/task"
echo "------------------------------------------------------------"
module load anaconda/2025
source activate fair
mkdir -p logs/other logs/tipping
python 14_confirmator.py grid
