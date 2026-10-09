"""Shared functions for the resilience post-processing.
"""
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import xarray as xr

from temperature import processed_temperature


def load_temperature(path):
    """Load the concatenated temperature DataArray, stripping ``()`` from scenarios."""
    da = xr.open_dataset(path)["__xarray_dataarray_variable__"]
    return da.assign_coords(
        scenario=da.scenario.str.replace(r"[\(\)]", "", regex=True)
    )


def load_tipping(tipping_dir):
    """Load per-scenario tipping-probability datasets into ``{scenario: ds}``."""
    files = [
        f for f in listdir(tipping_dir)
        if f.endswith(".nc") and isfile(join(tipping_dir, f))
    ]
    out = {}
    for f in files:
        scenario = f.split("_tipping_probabilities.nc")[0]
        out[scenario] = xr.open_dataset(join(tipping_dir, f))
    return out


def compute_running_means(temperature, scenarios, window=20, baseline=(1850, 1901)):
    """Per-member rebased, 20-yr centred running-mean GMST per scenario.

    Members are the (run, config) combinations. The baseline is each member's own weighted
    mean over ``baseline`` (1850..1901 inclusive, half weights on the two endpoints) — the
    rebase now replaces the earlier single ensemble-mean scalar baseline, and both the
    running mean and the rebase live in src/temperature.py (shared with the tipping forcing).
    Returns ``{scenario: DataArray(timebounds, member)}``.
    """
    running_means = {}
    for scenario in scenarios:
        data = temperature.sel(scenario=scenario).stack(member=("run", "config"))
        running_means[scenario] = processed_temperature(data, window=window, baseline=baseline)
    return running_means


def compute_resilience(rm, res_x, res_y, rate, res_end=None):
    """Binary climate resilience for one scenario's running-mean field.

    Resilient iff neither threshold is violated:
      - climate:   GMST > ``res_y`` at any time in [``res_x``, ``res_end``]
      - warming rate: 10-yr rate of change > ``rate`` over 2000..``res_x``
    ``res_end`` bounds the climate window's upper edge; ``None`` (default) scans to
    the end of the trajectory (the current behaviour). Set it (e.g. 2109/2110.5) to
    evaluate resilience over a fixed horizon consistent across scenario families —
    the "ever-exceeds" operator otherwise makes the resilient fraction depend on how
    long each family is run (SSPs to 2300 vs NGFS to ~2110).
    Returns a tidy DataFrame with (config, run) and a ``resilience_cr`` column.
    """
    climate_violation = (
        rm.sel(timebounds=slice(res_x, res_end)) > res_y
    ).any(dim="timebounds")
    rate_of_change = rm.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation = (
        rate_of_change.sel(timebounds=slice(2000, res_x)) > rate
    ).any(dim="timebounds")

    binary = (~(climate_violation | rate_violation)).unstack("member").astype(float)
    df = binary.to_dataframe(name="resilience_cr").reset_index()
    return df.drop(columns=["layer"], errors="ignore")


def crossed_bootstrap(T, B=1000, seed=0):
    """Multi-level crossed bootstrap resampling (config, run, sample) of a score tensor.

    Each of the three dimensions of ``T`` (shape ``(n_cfg, n_runs, n_samp)``) is
    resampled with replacement independently, and the draws are crossed (outer
    product) before taking the grand mean; returns the mean and 95% CI. The
    ensemble is a full crossed grid -- every config shares the same run and
    tipping-sample axes -- so the factors are resampled as crossed effects, not
    nested within one another.
    """
    rng = np.random.default_rng(seed)
    n_cfg, n_runs, n_samp = T.shape
    Q = np.empty(B, dtype=np.float32)
    for b in range(B):
        cfg_idx = rng.integers(0, n_cfg, size=n_cfg)
        run_idx = rng.integers(0, n_runs, size=n_runs)
        samp_idx = rng.integers(0, n_samp, size=n_samp)
        Q[b] = T[cfg_idx][:, run_idx][:, :, samp_idx].mean()
    return {
        "mean": float(Q.mean()),
        "ci_low": float(np.quantile(Q, 0.025)),
        "ci_high": float(np.quantile(Q, 0.975)),
    }
