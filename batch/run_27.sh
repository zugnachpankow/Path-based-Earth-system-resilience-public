#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=run_27
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
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

# SI diagnostic: GFP construction quality (PCA loadings/variance, GFP vs warming/ECS).
# over the shared LHS + a few plots; light-moderate (tipping load dominates).
python si/27_figures_si_feedback_quality.py
