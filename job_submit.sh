#!/bin/bash

#SBATCH --qos=medium
#SBATCH --job-name=run-tipping-scenario
#SBATCH --account=copan

#SBATCH --chdir=/home/maxbecht/Path-based-Earth-system-resilience-public
#SBATCH --output=logs/tipping/%x-%j.out
#SBATCH --error=logs/tipping/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
python /home/maxbecht/Path-based-Earth-system-resilience-public/src/run_tipping.py REMIND-MAgPIE_3.3-4.8___Current_Policies__start2050.5_r0.10