"""Per-scenario tipping runner (CLI).

Usage: python src/run_tipping.py <scenario>

Loads the concatenated temperature field, runs the pycascades tipping cascade for
one scenario (1000 LHS samples over all config/run), and writes the tipping
probabilities. Submitted per scenario by 07_submit_tipping.py.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from tipping import compute_tip_prob
from tipping_params import sample_lhs_params

# repo root (this file lives in <repo>/src/); all I/O is relative to <repo>/output,
# which is symlinked to scratch for now.
OUTPUT = Path(__file__).resolve().parents[1] / "output"

if __name__ == "__main__":
    # load nc file at this location in shape (run, timebounds, scenario, config)
    path = str(OUTPUT / "all_scenarios_temperature.nc")
    temperature = xr.open_dataset(path)

    # rename scenarios to match what i can pass via .sh - brackets are not possible
    temperature = temperature.assign_coords(
        scenario=[s.replace("(", "").replace(")", "") for s in temperature.scenario.values]
    )

    configs = temperature.config.values.tolist()
    runs = temperature.run.values.tolist()

    scenario = np.array(sys.argv[1:], dtype=str)[0]  # have to pass this

    n_samples = 1000
    params = sample_lhs_params(n_samples=n_samples)

    prob_any, prob_any_sample, prob_elements = compute_tip_prob(
        temperature["__xarray_dataarray_variable__"].sel(scenario=scenario),
        params, n_samples=n_samples, configs=configs, runs=runs,
    )

    ds_prob = xr.Dataset(
        {
            "prob_any_tipping": prob_any,
            "prob_any_tipping_sample": prob_any_sample,
            "prob_element_tipping": prob_elements
        }
    )

    # save (under output/tipping, symlinked to scratch for now)
    out_dir = OUTPUT / "tipping"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_prob.to_netcdf(str(out_dir / f"{scenario}_tipping_probabilities.nc"))
