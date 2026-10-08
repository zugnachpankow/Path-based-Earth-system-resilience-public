"""30_tables_si_mc_sensitivity.py — SI: Monte-Carlo / Sobol sensitivity results.

Tabulates the Sobol first-order (S1) and total-order (ST) indices with their 95%
bootstrap confidence intervals for the four criterion parameters (return year,
temperature threshold, warming-rate threshold, tipping on/off), for the three focal
SSPs, at the production rate range (0.25–0.55 °C per decade). This is the numeric
companion to the Sobol figure panel and the rate-cutoff SI figure (28).

Reads output/sensitivity/rate0.025_0.055/sobol_*.pkl (10). Light — runs on login.
Output -> output/tables/si_mc_sensitivity.{csv,md}
"""
import os
import sys
import pickle

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import get_scenario_type

OUT_DIR = os.path.join("tables"); os.makedirs(OUT_DIR, exist_ok=True)
SENS_DIR = "output/sensitivity/rate0.025_0.055"
FOCAL = ["ssp126", "ssp245", "ssp534-over"]
PARAMS = ["Return year", "Temperature threshold", "Rate threshold", "Tipping"]

rows = []
for sc in FOCAL:
    with open(os.path.join(SENS_DIR, f"sobol_{sc}.pkl"), "rb") as f:
        Si = pickle.load(f)["Si"]
    for i, p in enumerate(PARAMS):
        rows.append({
            "Scenario": get_scenario_type(sc), "Parameter": p,
            "S1": round(float(Si["S1"][i]), 4), "S1 95% CI": round(float(Si["S1_conf"][i]), 4),
            "ST": round(float(Si["ST"][i]), 4), "ST 95% CI": round(float(Si["ST_conf"][i]), 4),
        })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "si_mc_sensitivity.csv"), index=False)
with open(os.path.join(OUT_DIR, "si_mc_sensitivity.md"), "w") as fh:
    fh.write("Sobol sensitivity of path-based resilience to the four criterion parameters "
             "(Saltelli N=1024 + LHS Monte Carlo N=10000; rate range 0.25–0.55 °C per decade). "
             "S1 = first-order, ST = total-order; ± is the 95% bootstrap CI.\n\n")
    fh.write(df.to_markdown(index=False) + "\n")
print(df.to_string(index=False), flush=True)
print("\nWrote output/tables/si_mc_sensitivity.{csv,md}", flush=True)
