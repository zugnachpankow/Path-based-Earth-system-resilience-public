#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=run_26
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
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

# SI diagnostic: tipping-susceptibility PCA + RF quality. One RF fit per main scenario
# over the shared LHS + a few plots; light-moderate (tipping load dominates).
python si/26_figures_si_tipping_susc_quality.py
