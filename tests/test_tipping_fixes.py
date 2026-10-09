"""Tests for the branch-`tau` tipping fixes (steps A-E). Run with pytest, or standalone."""
import os, sys, copy
import numpy as np
from pyDOE import lhs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tipping import Earth_System
from tipping_params import param_bounds, pf_bounds, all_bounds, sample_lhs_params, SEED


# ── STEP A: AMOC(THC) tipped stabilises GIS ──────────────────────────────────────────
def test_a_amoc_to_gis_stabilising():
    p = sample_lhs_params(n_samples=1000)
    worst = -np.inf
    for i in range(1000):
        pr = {k: float(p[k][i]) for k in param_bounds}
        pf = {k: float(p[k][i]) for k in pf_bounds}
        net = Earth_System(**pr, **pf).dynamic_earth_network(
            gmt_function=lambda t: 0.0, strength=float(p["strength"][i]), kk0=1, kk1=1)
        f_thc_tipped = np.asarray(net.f([-1.0, 1.0, -1.0, -1.0], 2100.0))[0]
        f_base = np.asarray(net.f([-1.0, -1.0, -1.0, -1.0], 2100.0))[0]
        worst = max(worst, f_thc_tipped - f_base)   # must be < 0 for every sample
    assert worst < 0.0, f"AMOC->GIS not stabilising for some sample (max df_GIS = {worst:.3e})"


# ── STEP B: WAIS lower bound 500; all other columns unchanged ────────────────────────
def test_b_wais_range_and_others_unchanged():
    new = sample_lhs_params(n_samples=1000)
    assert new["wais_time"].min() >= 500.0 and new["wais_time"].max() <= 13000.0
    # reproduce the sampler with the OLD wais bound (2000) and the SAME unit LHS matrix
    old_bounds = copy.deepcopy(all_bounds); old_bounds["wais_time"] = (2000, 13000)
    np.random.seed(SEED)
    unit = lhs(len(all_bounds), samples=1000)
    old = {k: lo + unit[:, i] * (hi - lo) for i, (k, (lo, hi)) in enumerate(old_bounds.items())}
    for k in all_bounds:
        if k == "wais_time":
            assert not np.allclose(new[k], old[k])      # this column changed
        else:
            assert np.allclose(new[k], old[k]), f"column {k} changed unexpectedly"


# ── STEP D: shared processed temperature (per-member rebase + 20-yr running mean) ────
import xarray as xr
from scipy.interpolate import interp1d
from temperature import processed_temperature, rebase_per_member, last_valid_year
from tipping_params import T_END
import resilience


def _synthetic_temperature(scenarios=("sA",), n_end=2110):
    tb = np.arange(1750, n_end + 1)
    base = np.where(tb < 1850, 0.0, (tb - 1850) / (n_end - 1850) * 2.0)
    arr = np.empty((len(scenarios), 2, 2, len(tb)))
    for s in range(len(scenarios)):
        for c in range(2):
            for r in range(2):
                arr[s, c, r] = base + 0.3 * c - 0.2 * r + 0.1 * s
    return xr.DataArray(arr, dims=("scenario", "config", "run", "timebounds"),
                        coords={"scenario": list(scenarios), "config": [0, 1], "run": [0, 1], "timebounds": tb})


def test_d1_rebase_zero_mean():
    da = _synthetic_temperature().isel(scenario=0)
    reb = rebase_per_member(da)
    w = np.ones(52); w[0] = w[-1] = 0.5
    bmean = np.average(reb.sel(timebounds=slice(1850, 1901)).values, axis=-1, weights=w)
    assert np.max(np.abs(bmean)) < 1e-10


def test_d2_criterion_and_forcing_same_series():
    da = _synthetic_temperature(("sA",))
    rm = resilience.compute_running_means(da, ["sA"])["sA"].unstack("member")        # criterion side
    proc = processed_temperature(da.isel(scenario=0))                                # forcing side
    a = rm.transpose("timebounds", "config", "run").values
    b = proc.transpose("timebounds", "config", "run").values
    assert np.allclose(a, b, equal_nan=True)


def test_d3_forcing_held_and_no_nan():
    proc = processed_temperature(_synthetic_temperature().isel(scenario=0)).sel(config=0, run=0)
    t = proc.timebounds.values; v = proc.values
    g = interp1d(t, v, kind="linear", bounds_error=False, fill_value=(v[0], v[-1]))
    lv = last_valid_year(processed_temperature(_synthetic_temperature().isel(scenario=0)))
    grid = np.linspace(0, T_END, 2000)
    assert not np.isnan(g(grid)).any()                       # no NaN reaches the solver
    assert abs(float(g(lv + 10000)) - float(g(lv))) < 1e-12  # held constant after last valid year


# ── STEP E: robustness (solver-failure -> NaN, attrs) + vendored cusp ─────────────────
def test_e_robustness_attrs_and_vendored_cusp():
    import tipping, tipping_element
    assert tipping.cusp is tipping_element.cusp          # cusp vendored, not from pycascades
    tb = np.arange(1750, 2111)
    base = np.where(tb < 1850, 0.0, (tb - 1850) / 260 * 2.5)
    da = xr.DataArray(np.broadcast_to(base, (1, 1, len(tb))).copy(),
                      dims=("config", "run", "timebounds"),
                      coords={"config": [0], "run": [0], "timebounds": tb})
    p = sample_lhs_params(n_samples=10)
    pa, pas, pe = tipping.compute_tip_prob(da, p, n_samples=10, configs=[0], runs=[0], t_end=50000)
    assert "n_solver_failures" in pa.attrs and "n_tip_final_disagreements" in pa.attrs
    assert pa.attrs["n_solver_failures"] == 0            # no failures on a clean path
    assert not np.isnan(pas.values).any()                # no spurious NaN when all solves succeed


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
