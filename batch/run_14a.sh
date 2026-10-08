#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=run_14a
#SBATCH --account=copan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=113
#SBATCH --mem=250G
#SBATCH --output=logs/other/%x-%j.out
#SBATCH --error=logs/other/%x-%j.err
#SBATCH --chdir=/home/maxbecht/Path-based-Earth-system-resilience-public

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID  |  $SLURM_CPUS_PER_TASK cpus/task"
echo "------------------------------------------------------------"
module load anaconda/2025
source activate fair
# only 2035.5 is shown in the calculator figure (panel b); other target years skipped
python 14_confirmator.py 0.01 2035.5
