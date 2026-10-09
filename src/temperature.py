"""Single temperature series used by BOTH the temperature criterion (src/resilience.py)
and the tipping forcing (src/run_tipping.py).

processed_temperature(da) = per-member 1850-1900 rebase -> 20-yr centred running mean.
The 20-yr running-mean code lives here only (moved out of src/resilience.py); the per-member
rebase replaces the single ensemble-mean scalar baseline that resilience.py used before.
"""
import numpy as np
import xarray as xr


def _baseline_weights(n):
    # half weights on the two endpoints (as fair-calibrate / the existing weights_51yr):
    # np.ones(52); [0] = 0.5; [-1] = 0.5  for the 1850-1901 (52-yr) baseline.
    w = np.ones(n); w[0] = w[-1] = 0.5
    return w


def rebase_per_member(da, baseline=(1850, 1901)):
    """Subtract each member's weighted 1850-1900 mean (half weights on the endpoints).

    Operates per member (every non-timebounds dim is a member dimension), replacing the
    single ensemble-mean scalar baseline. Shape is preserved.
    """
    base_slice = da.sel(timebounds=slice(*baseline))
    w = xr.DataArray(_baseline_weights(base_slice.timebounds.size),
                     dims="timebounds", coords={"timebounds": base_slice.timebounds})
    base = base_slice.weighted(w).mean("timebounds")
    return da - base


def running_mean(da, window=20):
    """20-yr centred running mean; NaN edges from the centred window are dropped."""
    return da.rolling(timebounds=window, center=True).mean().dropna("timebounds", how="all")


def processed_temperature(da, window=20, baseline=(1850, 1901)):
    """Per-member rebased, 20-yr centred running-mean temperature (shape preserved)."""
    return running_mean(rebase_per_member(da, baseline), window)


def last_valid_year(da):
    """Last timebound with a valid (non-NaN) value (per the processed series)."""
    other = [d for d in da.dims if d != "timebounds"]
    valid = da.notnull().any(other) if other else da.notnull()
    idx = np.where(valid.values)[0]
    return float(da.timebounds.values[idx[-1]]) if idx.size else float("nan")
