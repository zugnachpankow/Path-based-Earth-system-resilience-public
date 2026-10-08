#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=run_25
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=logs/other/%x-%j.out
#SBATCH --error=logs/other/%x-%j.err
#SBATCH --chdir=/home/maxbecht/Path-based-Earth-system-resilience-public

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID  |  $SLURM_CPUS_PER_TASK cpus/task"
echo "------------------------------------------------------------"

module load anaconda/2025
source activate fair

# SSP resilience: full horizon (2300) vs cut (2109) vs old. Light compute; the 31G
# running-mean pickle load is the only heavy part -> single task, large mem.
python si/25_tables_si_horizon.py
