import os
import copy

import pandas as pd
import xarray as xr
from tqdm import trange

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

from rcmip import fill_from_rcmip_locally


def monte_carlo_fair(
    time,
    scenarios,
    params_file,
    species_configs_file,
    emissions_file=None,
    forcing_file=None,
    n_runs=100,
    seed_start=1234,
    output_dir="runs",
    stochastic=True
):
    """Run monte carlo FAIR with stochastic climate model.
    Parameters
    ----------
    time : array-like
        Array of years to run the model over.
    scenarios : list of str
        List of scenario names to run.
    params_file : str
        Path to CSV file containing FAIR model parameters for each configuration.
    species_configs_file : str
        Path to CSV file containing species configuration properties.
    emissions_file : str, optional
        Path to CSV file containing emissions data. If None, RCMIP data will be used
    forcing_file : str, optional
        Path to CSV file containing forcing data. If None, RCMIP data will be used
    n_runs : int, optional
        Number of Monte Carlo runs to perform. Default is 100.
    seed_start : int, optional
        Starting seed for random number generation. Default is 1234.
    output_dir : str
        Path to output directory for saving single runs.
    stochastic : bool, optional
        Whether to enable FAIR's stochastic climate variability. Default True.
        The calculator passes ``n_runs > 1`` so a single bisection run is
        deterministic (reproducible), matching the original calculator.
    Returns
    -------
    files : list of str
        Paths to the per-run NetCDF files written under ``output_dir``, one per run.
    """
    
    print("Running monte carlo FAIR with stochastic climate...")
    os.makedirs(output_dir, exist_ok=True)
    print("Setting up FAIR model...")
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(time[0], time[-1], 1)  # start, end, step
    f.define_scenarios(scenarios)
    fair_params_file = params_file
    df_configs = pd.read_csv(fair_params_file, index_col=0)
    configs = df_configs.index
    f.define_configs(configs)
    fair_species_configs_file = species_configs_file
    species, properties = read_properties(filename=fair_species_configs_file)
    f.define_species(species, properties)
    f.allocate()
    if (emissions_file is None) != (forcing_file is None):
        raise ValueError(
            "emissions_file and forcing_file must be provided together (or both "
            "left as None to fall back to RCMIP); got "
            f"emissions_file={emissions_file!r}, forcing_file={forcing_file!r}."
        )
    if emissions_file is not None:
        f.fill_from_csv(
            emissions_file=emissions_file,
            forcing_file=forcing_file
            )
    else:
        fill_from_rcmip_locally(f)
    f.forcing.sel(specie="Volcanic")
    fill(
        f.forcing,
        f.forcing.sel(specie="Volcanic") * df_configs["forcing_scale[Volcanic]"].values.squeeze(),
        specie="Volcanic",
    )
    fill(
        f.forcing,
        f.forcing.sel(specie="Solar") * df_configs["forcing_scale[Solar]"].values.squeeze(),
        specie="Solar",
    )
    f.fill_species_configs(fair_species_configs_file)
    f.override_defaults(fair_params_file)
    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.ocean_heat_content_change, 0)
    f_baseline = copy.deepcopy(f)
    print("Model set up. Starting Monte Carlo runs...")
    files = []
    for i in trange(n_runs, desc="Monte Carlo"):
        print(f"Run #{i} started.", flush=True)
        f = copy.deepcopy(f_baseline)
        seed = seed_start + i*99
        for config in configs:
            fill(f.climate_configs['stochastic_run'], stochastic, config=config)
            fill(f.climate_configs['use_seed'], True, config=config)
            fill(f.climate_configs['seed'], seed, config=config)

        f.run(progress=False)
        # save f.temperature
        result = f.temperature.sel(layer=0)
        out_file = os.path.join(output_dir, f"run_{i:04d}.nc")
        result.to_netcdf(out_file)
        files.append(out_file)
        print(f"Run #{i} completed.", flush=True)

    print(f"All runs completed. {len(files)} files written.")
    return files


def aggregate_runs(run_files, out_path):
    """Concatenate per-run temperature files along ``run`` and write ensemble stats.

    Each file holds one run's temperature saved from a DataArray, so it reopens as
    the variable ``__xarray_dataarray_variable__``; we read them with
    ``open_dataarray`` so the statistics stay DataArrays (writing a Dataset built
    from Datasets raises a TypeError). Writes mean, std and a set of percentiles as
    a Dataset to ``out_path``.
    """
    da = xr.concat([xr.open_dataarray(f) for f in run_files], dim="run")
    quantiles = {"p0": 0.0, "p5": 0.05, "p16": 0.16, "p84": 0.84, "p95": 0.95, "p100": 1.0}
    stats = {"mean": da.mean("run"), "std": da.std("run")}
    for name, q in quantiles.items():
        stats[name] = da.quantile(q, dim="run").drop_vars("quantile")
    xr.Dataset(stats).to_netcdf(out_path)
    return out_path