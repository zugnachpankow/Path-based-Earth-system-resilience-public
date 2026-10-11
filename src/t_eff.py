"""Effective tipping temperature (T_eff) per LHS tipping sample.

T_eff is the lowest constant GMT (held from t=0) at which ANY tipping element crosses
x>0 before T_END. The model path (Earth_System: tau conversion, signs, couplings,
initial state) is imported from src/tipping.py unchanged -- only the thin solve/
bisection wrapper lives here, so 07's compute_tip_prob is untouched.

Heavy (~10k ODE solves per 1000-sample set); parallel over samples via joblib,
respecting SLURM_CPUS_PER_TASK. Driven from 11; results -> output/feedback/t_eff_*.csv.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from tqdm import trange

from tipping import Earth_System                       # the exact model compute_tip_prob uses
from tipping_params import param_bounds, pf_bounds, T_END

n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))


def run_sample_tipping(params, i, gmt_function, t_end=T_END, t_eval=None, convert_tau=True):
    """Integrate one LHS sample ``i`` under ``gmt_function``; identical setup to
    compute_tip_prob.run_single_sample (same Earth_System, y0, solver settings).

    Returns ``(tipped[4], success)``; a failed solve -> ``(nan4, False)``.
    """
    if t_eval is None:
        n_eval = int(np.ceil((t_end - 0) / 50.0)) + 1   # output spacing <= 50 yr
        t_eval = np.linspace(0, t_end, n_eval)
    pr_i = {key: params[key][i] for key in param_bounds}
    pf_i = {key: params[key][i] for key in pf_bounds}
    sys = Earth_System(**pr_i, **pf_i, convert_tau=convert_tau)
    net = sys.dynamic_earth_network(
        gmt_function=gmt_function, strength=params["strength"][i], kk0=1, kk1=1
    )
    sol = solve_ivp(
        fun=lambda t, x: net.f(x, t),
        t_span=(0, t_end),
        y0=[-1, -1, -1, -1],
        t_eval=t_eval,
        atol=1e-3, rtol=1e-3, method="LSODA",
    )
    if (not sol.success) or sol.y.shape[1] == 0:
        return np.full(4, np.nan), False
    return np.any(sol.y > 0.0, axis=1).astype(float), True


def sample_tips_at(params, i, gmt, t_eval=None, convert_tau=True):
    """True/False whether sample ``i`` tips any element at constant ``gmt``; None on failure."""
    tipped, ok = run_sample_tipping(params, i, lambda t: gmt, t_eval=t_eval,
                                    convert_tau=convert_tau)
    return None if not ok else bool(np.any(tipped > 0.0))


def effective_tipping_temperature(params, i, lo=0.0, hi=6.0, tol=0.005,
                                  t_end=T_END, convert_tau=True):
    """Lowest constant GMT at which sample ``i`` tips, by bisection on [lo, hi].

    Returns ``(T_eff, flag)``: ``"ok"`` for a bracketed root (T_eff = hi of the final
    bracket), ``"capped"`` if even ``hi`` does not tip (T_eff = hi), ``"nan"`` on a
    solver failure.
    """
    n_eval = int(np.ceil((t_end - 0) / 50.0)) + 1
    t_eval = np.linspace(0, t_end, n_eval)
    tips = lambda g: sample_tips_at(params, i, g, t_eval=t_eval, convert_tau=convert_tau)

    hi_t = tips(hi)
    if hi_t is None:
        return np.nan, "nan"
    if not hi_t:
        return hi, "capped"
    lo_t = tips(lo)
    if lo_t is None:
        return np.nan, "nan"
    if lo_t:
        return lo, "ok"                      # tips even at the bottom of the bracket
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        m = tips(mid)
        if m is None:
            return np.nan, "nan"
        if m:
            hi = mid
        else:
            lo = mid
    return hi, "ok"


def compute_t_eff(params, n_samples, lo=0.0, hi=6.0, tol=0.005, t_end=T_END,
                  n_jobs=n_jobs, convert_tau=True):
    """T_eff for the first ``n_samples`` LHS samples (parallel over samples).

    Returns a DataFrame indexed by ``sample`` with columns ``T_eff``, ``flag`` and the
    sampled element thresholds (``limits_*``).
    """
    res = Parallel(n_jobs=n_jobs)(
        delayed(effective_tipping_temperature)(params, i, lo, hi, tol, t_end, convert_tau)
        for i in trange(n_samples, desc="T_eff")
    )
    df = pd.DataFrame(res, columns=["T_eff", "flag"])
    df.index.name = "sample"
    for k in ("limits_gis", "limits_thc", "limits_wais", "limits_amaz"):
        df[k] = np.asarray(params[k])[:n_samples]
    return df
