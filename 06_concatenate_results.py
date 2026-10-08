import os
from os.path import join
import xarray as xr

base_path = "output"

ngfs_path = join(base_path, "runs_ngfs")
ngfs_run_files = sorted(
    os.path.join(ngfs_path, f) for f in os.listdir(ngfs_path)
)
ngfs_datasets = (xr.open_dataset(f) for f in ngfs_run_files)
ngfs_results = xr.concat(ngfs_datasets, dim="run")

ssp_path = join(base_path, "runs_ssp")
ssp_run_files = sorted(
    os.path.join(ssp_path, f) for f in os.listdir(ssp_path)
)
ssp_datasets = (xr.open_dataset(f) for f in ssp_run_files)
ssp_results = xr.concat(ssp_datasets, dim="run")

branch_off_path = join(base_path, "runs_branchoffs")
branch_off_run_files = sorted(
    os.path.join(branch_off_path, f) for f in os.listdir(branch_off_path)
)
branch_off_datasets = (xr.open_dataset(f) for f in branch_off_run_files)
branch_off_results = xr.concat(branch_off_datasets, dim="run")

ngfs_scenarios = ngfs_results.scenario.values.tolist()
ssp_scenarios = ssp_results.scenario.values.tolist()
branch_off_scenarios = branch_off_results.scenario.values.tolist()

# all scenarios
scenarios = ngfs_scenarios + ssp_scenarios + branch_off_scenarios

configs = ngfs_results.config.values.tolist()
runs = ngfs_results.run.values.tolist()

# temperature data
print("Concatenating temperature data from all scenarios...")
ngfs_temperature = ngfs_results["__xarray_dataarray_variable__"]  # shape: (run, timebounds, scenario, config)
ssp_temperature = ssp_results["__xarray_dataarray_variable__"]
branch_off_temperature = branch_off_results["__xarray_dataarray_variable__"]

temperature = xr.concat([ngfs_temperature, ssp_temperature, branch_off_temperature], dim="scenario")
print(f"Temperature data concatenated. New shape: {temperature.shape}")

# save new temperature dataset as netcdf
print("Saving temperature dataset as netcdf...")
save_path = "output"
output_path = join(save_path, "all_scenarios_temperature.nc")
temperature.to_netcdf(output_path)
print(f"Temperature dataset saved to {output_path}")