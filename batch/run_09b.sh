#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=run_09b
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=04:00:00
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

# hierarchical bootstrap CIs for the UpSet bars. Each tipping-tensor bootstrap is
# ~400s (4 per scenario x 4 scenarios ~= 2h) + a 31G running-mean pickle load, so:
# single task, large mem, few-hour walltime. Resumable — reruns skip finished CSVs.
python 09b_bootstrap_upset.py
