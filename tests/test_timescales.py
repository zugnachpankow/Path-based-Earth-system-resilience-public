"""Timescale-conversion tests for the cusp tipping model (src/tipping.py).

The literature tau is the x=-1 -> x=+1 transition time of an uncoupled cusp at GMT 4.0 /
threshold 1.8 (pycascades timing.py). In dx/dt=(1/T)(-x^3+x+c) that takes I_CAL*T, so the ODE
time constant must be T=tau/I_CAL. These tests check the conversion, the old (unconverted)
2.627x-too-slow behaviour, and that the conversion is a pure clock rescaling (f_fixed = I_CAL*f_current).
"""
import os, sys
import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from tipping import Earth_System, I_CAL, ode_timescale
from tipping_params import param_bounds, pf_bounds, sample_lhs_params

GMT_CAL, TCRIT_CAL = 4.0, 1.8
NO_PF = dict(pf_wais_to_gis=0.0, pf_thc_to_gis=0.0, pf_gis_to_thc=0.0, pf_wais_to_thc=0.0,
             pf_gis_to_wais=0.0, pf_thc_to_wais=0.0, pf_thc_to_amaz=0.0)
TAUS = {0: ("GIS", [1000, 10000, 15000]), 1: ("AMOC", [15, 50, 300]),
        2: ("WAIS", [500, 2000, 13000]), 3: ("AMAZ", [50, 100, 200])}


def _crossing_time(elem, tau, convert_tau, gmt=GMT_CAL, thr=TCRIT_CAL, target=1.0):
    """Time for the active element to go from x=-1 to x=target (uncoupled, constant GMT)."""
    times = [1000.0, 1000.0, 1000.0, 1000.0]; lims = [99.0, 99.0, 99.0, 99.0]
    times[elem] = tau; lims[elem] = thr
    es = Earth_System(gis_time=times[0], thc_time=times[1], wais_time=times[2], amaz_time=times[3],
                      limits_gis=lims[0], limits_thc=lims[1], limits_wais=lims[2], limits_amaz=lims[3],
                      convert_tau=convert_tau, **NO_PF)
    net = es.dynamic_earth_network(gmt_function=lambda t: gmt, strength=0.0, kk0=1, kk1=1)
    ev = lambda t, x: x[elem] - target
    ev.terminal = True; ev.direction = 1
    sol = solve_ivp(lambda t, x: net.f(x, t), (0.0, 50.0 * tau), [-1.0, -1.0, -1.0, -1.0],
                    events=ev, method="LSODA", rtol=1e-8, atol=1e-10, dense_output=True)
    assert sol.t_events[0].size == 1, f"no crossing for elem {elem}, tau {tau}"
    return float(sol.t_events[0][0])


def test_1_I_CAL():
    assert abs(I_CAL - 2.6267) < 1e-4


@pytest.mark.parametrize("elem,tau", [(e, t) for e, (_, ts) in TAUS.items() for t in ts])
def test_2_fixed_crossing_equals_tau(elem, tau):
    tc = _crossing_time(elem, tau, convert_tau=True)
    assert abs(tc - tau) / tau < 0.005, f"{TAUS[elem][0]} tau={tau}: crossing {tc:.1f} != tau (0.5%)"


@pytest.mark.parametrize("elem,tau", [(e, t) for e, (_, ts) in TAUS.items() for t in ts])
def test_3_current_ratio_is_I_CAL(elem, tau):
    tc = _crossing_time(elem, tau, convert_tau=False)
    assert abs(tc / tau - I_CAL) / I_CAL < 0.005, f"{TAUS[elem][0]} tau={tau}: ratio {tc/tau:.4f} != {I_CAL:.4f}"


def test_4_uniform_clock():
    p = sample_lhs_params(n_samples=10)
    pr = {k: float(p[k][0]) for k in param_bounds}
    pf = {k: float(p[k][0]) for k in pf_bounds}
    strength = float(p["strength"][0])
    kw = dict(gmt_function=lambda t: 2.0, strength=strength, kk0=1, kk1=1)
    net_cur = Earth_System(**pr, **pf, convert_tau=False).dynamic_earth_network(**kw)
    net_fix = Earth_System(**pr, **pf, convert_tau=True).dynamic_earth_network(**kw)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(-1.5, 1.5, size=4)
        f_cur = np.asarray(net_cur.f(x, 2100.0)); f_fix = np.asarray(net_fix.f(x, 2100.0))
        assert np.allclose(f_fix, I_CAL * f_cur, rtol=1e-12, atol=0.0), \
            f"f_fixed != I_CAL*f_current: {f_fix} vs {I_CAL*f_cur}"
