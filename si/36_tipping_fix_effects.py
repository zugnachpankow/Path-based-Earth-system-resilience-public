"""36_tipping_fix_effects.py — cumulative effect of the branch-`tau` tipping fixes.

Real compute_tip_prob on a synthetic temperature (1 config, 1 run, 1000 LHS samples):
path 0 until 1850, 1.3 in 2020, 1.7 in 2050, T in 2100, then held. T in {1.0,1.2,1.4,1.6}.
Each fix is switched on via flags (LHS bounds + compute_tip_prob args), not by editing files:
  (a) as on main: old sign, convert_tau=False, WAIS 2000-13000, t_end 15000
  (b) + sign     (c) + tau conversion     (d) + WAIS 500     (e) + 50,000-yr horizon
Step D (forcing processing) is NOT in this table (the stylised path has no noise/baseline shift).

Output -> output/tipping_fix_check/ : table_effects.{csv,md}, README.md
"""
import os, sys
import numpy as np, pandas as pd
from pyDOE import lhs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
import xarray as xr
from tipping import compute_tip_prob, elements
from tipping_params import all_bounds, SEED

OUT = os.path.join("output", "tipping_fix_check"); os.makedirs(OUT, exist_ok=True)
TEMPS = [1.0, 1.2, 1.4, 1.6]
# (label, sign_new, convert_tau, wais_new, t_end) — cumulative
VARIANTS = [
    ("a_main",        False, False, False, 15000),
    ("b_sign",        True,  False, False, 15000),
    ("c_tau",         True,  True,  False, 15000),
    ("d_wais500",     True,  True,  True,  15000),
    ("e_horizon50k",  True,  True,  True,  50000),
]

np.random.seed(SEED)
unit = lhs(len(all_bounds), samples=1000)   # one unit LHS matrix shared by all variants


def make_params(sign_new, wais_new):
    b = dict(all_bounds)   # current (fixed) bounds
    b["pf_thc_to_gis"] = (0.1, 1.0) if sign_new else (-1.0, -0.1)
    b["wais_time"]     = (500, 13000) if wais_new else (2000, 13000)
    return {k: lo + unit[:, i] * (hi - lo) for i, (k, (lo, hi)) in enumerate(b.items())}


def stylised_da(T):
    tb = np.arange(1750, 2101)
    temp = np.interp(tb, [1850, 2020, 2050, 2100], [0.0, 1.3, 1.7, T], left=0.0)
    return xr.DataArray(temp[None, None, :], dims=("config", "run", "timebounds"),
                        coords={"config": [0], "run": [0], "timebounds": tb})


rows = []
for T in TEMPS:
    da = stylised_da(T)
    for lbl, sign_new, conv, wais_new, t_end in VARIANTS:
        params = make_params(sign_new, wais_new)
        pa, pas, pe = compute_tip_prob(da, params, n_samples=1000, configs=[0], runs=[0],
                                       t_end=t_end, convert_tau=conv)
        rows.append({"T": T, "variant": lbl, "P_any": round(float(pa.mean()), 4),
                     **{f"P_{e}": round(float(pe.sel(element=e).mean()), 4) for e in elements}})
        print(f"T={T} {lbl:14s} P_any={rows[-1]['P_any']:.4f}  "
              + " ".join(f"P_{e}={rows[-1]['P_'+e]:.3f}" for e in elements), flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "table_effects.csv"), index=False)
piv = df.pivot(index="variant", columns="T", values="P_any").reindex([v[0] for v in VARIANTS])
with open(os.path.join(OUT, "table_effects.md"), "w") as fh:
    fh.write("## Cumulative tipping-fix effects — P(any tipping), stylised path\n\n")
    fh.write(piv.to_markdown() + "\n\n")
    fh.write("Full P per element in table_effects.csv.\n")

with open(os.path.join(OUT, "README.md"), "w") as fh:
    fh.write("# Tipping-fix cumulative effects (stylised path)\n\n")
    for T in (1.2, 1.4):
        vals = ", ".join(f"{v}={df[(df.T == T) & (df.variant == v)]['P_any'].iloc[0]:.3f}"
                         for v, *_ in VARIANTS)
        fh.write(f"- P(any) at T={T}: {vals}\n")
    fh.write("- (b) sign fix lowers tipping (AMOC→GIS now stabilising); (c) tau conversion raises it "
             "(elements ~2.6x faster); (d) WAIS 500 adds a little; (e) 50k horizon raises it most.\n")
    fh.write("- Step D (forcing) is reported separately; not in this stylised table.\n")

print("\nP(any) pivot:\n", piv.to_string())
print(f"\nSaved -> {OUT}")
