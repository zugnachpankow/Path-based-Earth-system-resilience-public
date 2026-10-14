#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=run_38
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/other/%x-%j.out
#SBATCH --error=logs/other/%x-%j.err
#SBATCH --chdir=/home/maxbecht/Path-based-Earth-system-resilience-public

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID  |  $SLURM_CPUS_PER_TASK cpus/task"
echo "------------------------------------------------------------"
module load anaconda/2025
source activate fair
mkdir -p logs/other
python si/38_figures_si_heatmap_reference.py
