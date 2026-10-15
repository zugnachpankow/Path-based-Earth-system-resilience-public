"""Effective tipping temperature (T_eff) per LHS tipping sample.

T_eff is the lowest constant GMT (held from t=0) at which ANY tipping element crosses
x>0 before T_END. The model path (Earth_System: tau conversion, signs, couplings,
initial state) is imported from src/tipping.py so it matches compute_tip_prob exactly;
only the thin solve/bisection wrapper lives here.

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
from tipping_params import param_bounds, pf_bounds, T_END, all_bounds, sample_lhs_params

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


# ── T_eff under central literature values (Armstrong McKay et al. 2022) ────────────
# Thresholds + timescales are fixed to the central estimates; the interactions (pf_*)
# and the coupling strength keep their LHS values (same bounds and sign convention as
# the main LHS). The model path / bisection is unchanged (compute_t_eff above).
CENTRAL_LIT = {
    "limits_gis": 1.5, "limits_wais": 1.5, "limits_thc": 4.0, "limits_amaz": 3.5,   # [°C]
    "gis_time": 10000.0, "wais_time": 2000.0, "thc_time": 50.0, "amaz_time": 100.0,  # [yr]
}


def central_lit_params(n_samples=1000, seed=2468):
    """LHS params (new fixed seed) with the thresholds/timescales overwritten to the
    AM2022 central estimates; pf_* and strength keep their LHS draws. Asserts in bounds."""
    p = sample_lhs_params(n_samples=n_samples, seed=seed)
    for k, v in CENTRAL_LIT.items():
        lo, hi = all_bounds[k]
        assert lo <= v <= hi, f"central-lit {k}={v} outside tipping_params bounds [{lo}, {hi}]"
        p[k] = np.full(n_samples, float(v))
    return p


def compute_t_eff_central_lit(save_path, n_samples=1000, seed=2468, n_jobs=n_jobs):
    """Compute (and cache to ``save_path``) T_eff under the central-lit thresholds/timescales,
    plus a no-interaction (strength=0) reference. Prints the summary + monotonicity check."""
    p = central_lit_params(n_samples, seed)
    df = compute_t_eff(p, n_samples, n_jobs=n_jobs)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)

    # no-interaction reference: strength -> 0 so the couplings vanish (single run)
    p0 = {k: np.array([np.asarray(v)[0]]) for k, v in p.items()}
    p0["strength"] = np.array([0.0])
    noint, noint_flag = effective_tipping_temperature(p0, 0)

    te = df["T_eff"].values
    print(f"[central-lit] n={n_samples} seed={seed} -> {save_path}", flush=True)
    print(f"  median={np.nanmedian(te):.3f}  5-95%=[{np.nanpercentile(te, 5):.3f}, {np.nanpercentile(te, 95):.3f}]  "
          f"min={np.nanmin(te):.3f}  max={np.nanmax(te):.3f}", flush=True)
    print(f"  no-interaction (strength=0) T_eff = {noint:.3f} ({noint_flag})", flush=True)

    # monotonicity on 20 random samples (0.05 degC grid, 0..6)
    rng = np.random.default_rng(0)
    idx = rng.choice(n_samples, 20, replace=False)
    grid = np.arange(0.0, 6.0 + 1e-9, 0.05)
    n_eval = int(np.ceil(T_END / 50.0)) + 1
    t_eval = np.linspace(0, T_END, n_eval)
    nonmono = sum(
        np.any(np.diff([1 if sample_tips_at(p, int(i), g, t_eval=t_eval) else 0 for g in grid]) < 0)
        for i in idx
    )
    print(f"  monotonicity (20 samples, 0.05 grid): non-monotone = {int(nonmono)}", flush=True)
    return df
