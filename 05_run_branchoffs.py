import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve data/ and output/ from the repo root, not the job's cwd
from monte_carlo_runner import monte_carlo_fair, aggregate_runs

fair_params_file = "data/raw/calibrated_constrained_parameters_calibration1.4.0.csv"
fair_species_configs_file = "data/raw/species_configs_properties_NGFS_calibration1.4.0.csv"
emissions_file = "data/processed/current_policy_branch_offs_2300.csv"
forcing_file = "data/processed/current_policy_branch_offs_2300_forcing.csv"
time = np.arange(1750, 2301, 1)

df_emissions = pd.read_csv(emissions_file)
scenarios = df_emissions["scenario"].unique().tolist()

run_files = monte_carlo_fair(
    time,
    scenarios,
    fair_params_file,
    fair_species_configs_file,
    emissions_file=emissions_file,
    forcing_file=forcing_file,
    n_runs=100,
    output_dir="output/runs_branchoffs"
)

print("Aggregating results...")
aggregate_runs(run_files, "output/aggregate_branchoffs.nc")
print("Aggregation completed. File written: output/aggregate_branchoffs.nc")
