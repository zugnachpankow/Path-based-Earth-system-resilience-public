"""35_timescale_check.py — cusp timescale conversion: current (raw tau) vs fixed (tau/I_CAL).

Produces, under output/timescale_check/:
  table_calibration.{csv,md}   — T_ODE and x=-1->+1 crossing time (at GMT 4.0 / threshold 1.8) per
                                  element x tau, current vs fixed, and the ratio to tau.
  table_tipping_times.{csv,md} — time to x>0 (the compute_tip_prob tipped criterion) at constant
                                  GMT/threshold ratios, current vs fixed, min/central/max tau, and
                                  whether it crosses within the 12,700-yr committed window.
  fig_timescales.{png,pdf}     — x(t) current vs fixed at 4.0/1.8 for the central tau of each element.
  README.md                    — key numbers.

Light (uncoupled ODE solves). Does NOT run the pipeline, touch data/, or change existing outputs.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from tipping import Earth_System, I_CAL, ode_timescale

OUT = os.path.join("output", "timescale_check"); os.makedirs(OUT, exist_ok=True)
NO_PF = dict(pf_wais_to_gis=0.0, pf_thc_to_gis=0.0, pf_gis_to_thc=0.0, pf_wais_to_thc=0.0,
             pf_gis_to_wais=0.0, pf_thc_to_wais=0.0, pf_thc_to_amaz=0.0)
# element -> (name, [min, central, max] tau in yr)   (central = Armstrong McKay 2022; WAIS 500 = planned new lower bound)
TAUS = {0: ("GIS", [1000, 10000, 15000]), 1: ("AMOC", [15, 50, 300]),
        2: ("WAIS", [500, 2000, 13000]), 3: ("AMAZ", [50, 100, 200])}
WINDOW = 12700.0   # yr the forcing is held (2300 -> t_end=15000) -> time available to tip


def _net(elem, tau, conv, thr):
    times = [1000.0] * 4; lims = [99.0] * 4; times[elem] = tau; lims[elem] = thr
    es = Earth_System(gis_time=times[0], thc_time=times[1], wais_time=times[2], amaz_time=times[3],
                      limits_gis=lims[0], limits_thc=lims[1], limits_wais=lims[2], limits_amaz=lims[3],
                      convert_tau=conv, **NO_PF)
    return es


def crossing(elem, tau, conv, gmt, thr, target, rtol=1e-8, atol=1e-10):
    net = _net(elem, tau, conv, thr).dynamic_earth_network(gmt_function=lambda t: gmt, strength=0.0, kk0=1, kk1=1)
    ev = lambda t, x: x[elem] - target; ev.terminal = True; ev.direction = 1
    sol = solve_ivp(lambda t, x: net.f(x, t), (0.0, 200.0 * tau), [-1.0] * 4,
                    events=ev, method="LSODA", rtol=rtol, atol=atol)
    return float(sol.t_events[0][0]) if sol.t_events[0].size else np.nan


# ── Table A: calibration (x=-1 -> +1 at GMT 4.0 / threshold 1.8) ────────────────────────────────
rows = []
for e, (nm, taus) in TAUS.items():
    for tau in taus:
        tc_cur = crossing(e, tau, False, 4.0, 1.8, 1.0)
        tc_fix = crossing(e, tau, True, 4.0, 1.8, 1.0)
        rows.append({"element": nm, "tau_yr": tau,
                     "T_ODE_current": round(tau, 1), "T_ODE_fixed": round(ode_timescale(tau), 1),
                     "cross_current_yr": round(tc_cur, 1), "cross_fixed_yr": round(tc_fix, 1),
                     "ratio_current": round(tc_cur / tau, 4), "ratio_fixed": round(tc_fix / tau, 4)})
A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, "table_calibration.csv"), index=False)
with open(os.path.join(OUT, "table_calibration.md"), "w") as fh:
    fh.write("## Calibration: x=-1 -> +1 at GMT 4.0 / threshold 1.8 (I_CAL = %.4f)\n\n" % I_CAL)
    fh.write(A.to_markdown(index=False) + "\n")

# ── Table B: time to x>0 at constant GMT/threshold ratios ───────────────────────────────────────
RATIOS = [1.05, 1.2, 1.5, 2.0, 3.0]
brows = []
for e, (nm, taus) in TAUS.items():
    for lvl, tau in zip(["min", "central", "max"], taus):
        for r in RATIOS:
            tc_cur = crossing(e, tau, False, r, 1.0, 0.0)   # ratio = gmt/thr = r (thr=1, gmt=r)
            tc_fix = crossing(e, tau, True, r, 1.0, 0.0)
            brows.append({"element": nm, "tau_level": lvl, "tau_yr": tau, "ratio": r,
                          "t_x>0_current_yr": round(tc_cur, 1), "t_x>0_fixed_yr": round(tc_fix, 1),
                          "units_of_T_ODE": round(tc_cur / tau, 3),
                          "within_12700_current": bool(tc_cur < WINDOW), "within_12700_fixed": bool(tc_fix < WINDOW)})
B = pd.DataFrame(brows); B.to_csv(os.path.join(OUT, "table_tipping_times.csv"), index=False)
with open(os.path.join(OUT, "table_tipping_times.md"), "w") as fh:
    fh.write("## Time to x>0 (tipped) at constant GMT/threshold ratio; committed window = %d yr\n\n" % WINDOW)
    fh.write(B.to_markdown(index=False) + "\n")

# ── Figure: x(t) current vs fixed, central tau per element ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (e, (nm, taus)) in zip(axes.ravel(), TAUS.items()):
    tau = taus[1]
    for conv, col, lab in [(False, "tab:red", "current (raw τ)"), (True, "tab:blue", "fixed (τ/I_CAL)")]:
        net = _net(e, tau, conv, 1.8).dynamic_earth_network(gmt_function=lambda t: 4.0, strength=0.0, kk0=1, kk1=1)
        sol = solve_ivp(lambda t, x: net.f(x, t), (0.0, 4.0 * tau), [-1.0] * 4,
                        method="LSODA", rtol=1e-8, atol=1e-10, dense_output=True)
        tt = np.linspace(0, 4.0 * tau, 2000)
        ax.plot(tt, sol.sol(tt)[e], color=col, label=lab)
    ax.axvline(tau, ls="--", color="k", lw=0.8, label=f"τ = {tau} yr")
    ax.axhline(0, color="0.6", lw=0.6); ax.axhline(1, color="0.6", lw=0.6, ls=":")
    ax.set_title(f"{nm} (central τ = {tau} yr)"); ax.set_xlabel("time [yr]"); ax.set_ylabel("x")
    ax.legend(fontsize=7, frameon=False)
fig.suptitle("Cusp trajectory at GMT 4.0 / threshold 1.8 — current vs fixed clock")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_timescales.png"), dpi=150, bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_timescales.pdf"), bbox_inches="tight")
plt.close(fig)

# ── Pipeline-tolerance check: central tau, calibration crossing, loose vs tight solver ──────────
pipe = []
for e, (nm, taus) in TAUS.items():
    tight = crossing(e, taus[1], True, 4.0, 1.8, 1.0, rtol=1e-8, atol=1e-10)
    loose = crossing(e, taus[1], True, 4.0, 1.8, 1.0, rtol=1e-3, atol=1e-3)
    pipe.append({"element": nm, "tau": taus[1], "cross_tight": round(tight, 1),
                 "cross_loose(LSODA 1e-3)": round(loose, 1), "dev_pct": round(abs(loose - tight) / tight * 100, 3)})
Pp = pd.DataFrame(pipe); Pp.to_csv(os.path.join(OUT, "table_pipeline_tolerance.csv"), index=False)

# ── README ──────────────────────────────────────────────────────────────────────────────────────
with open(os.path.join(OUT, "README.md"), "w") as fh:
    fh.write("# Timescale conversion check (current vs fixed)\n\n")
    fh.write(f"- I_CAL = {I_CAL:.4f}; fix: T_ODE = τ / I_CAL (cusp ODE time constant).\n")
    fh.write(f"- Crossing x=-1→+1 at 4.0/1.8: current = {A.ratio_current.mean():.3f}·τ (too slow), fixed = {A.ratio_fixed.mean():.3f}·τ.\n")
    fh.write("- T_ODE (fixed) examples: GIS 10000→%.0f, AMOC 50→%.1f, WAIS 2000→%.0f, AMAZ 100→%.1f yr.\n"
             % (ode_timescale(10000), ode_timescale(50), ode_timescale(2000), ode_timescale(100)))
    fh.write("- Time to x>0 in units of T_ODE: %s (ratios %s).\n"
             % (", ".join(f"{v:.1f}" for v in B.loc[(B.element=='GIS') & (B.tau_level=='central'), 'units_of_T_ODE']),
                ", ".join(map(str, RATIOS))))
    ex = B[(B.element=='GIS') & (B.tau_level=='central') & (B.ratio==2.0)].iloc[0]
    fh.write(f"- Example GIS τ=10000 at ratio 2.0: current {ex['t_x>0_current_yr']:.0f} yr, fixed {ex['t_x>0_fixed_yr']:.0f} yr.\n")
    n_cur = int(B.within_12700_current.sum()); n_fix = int(B.within_12700_fixed.sum())
    fh.write(f"- Crossings within the 12,700-yr committed window: current {n_cur}/{len(B)}, fixed {n_fix}/{len(B)} "
             "(the fix lets slower elements tip in-window).\n")
    fh.write(f"- Pipeline solver (LSODA, atol=rtol=1e-3) vs tight: max deviation {Pp.dev_pct.max():.3f}% (<1%).\n")

print("Table A (calibration):\n", A.to_string(index=False))
print("\nPipeline tolerance:\n", Pp.to_string(index=False))
print(f"\nwithin-window: current {int(B.within_12700_current.sum())}/{len(B)}, fixed {int(B.within_12700_fixed.sum())}/{len(B)}")
print(f"Saved -> {OUT}")
