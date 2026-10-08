"""19_tables_si.py — supplementary tables.

Table SI-warming: per scenario, the ensemble warming summary from the 20-yr running
mean (baseline-corrected, °C):
  * median warming in 2100 (+ 5–95th percentile across the config×run ensemble),
  * peak warming of the median path up to 2100 and the year it occurs,
  * whether the peak is *before* 2100 (overshoot).

Peak is taken on the ENSEMBLE-MEDIAN trajectory (median over members at each year),
restricted to years ≤ 2100. Loads the 08 running-mean pickle -> submit via SLURM.

Outputs (output/tables/):
  si_warming_table.csv   (all numeric columns)
  si_warming_table.md    (formatted, for the SI)
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))

from figures_common import (
    FIG_DIR, load_running_means, build_naming, get_scenario_type, get_model,
)

OUT_DIR = os.path.join(os.path.dirname(FIG_DIR), "tables")   # output/tables
os.makedirs(OUT_DIR, exist_ok=True)

PEAK_HORIZON = 2100          # only look for a peak up to here
OVERSHOOT_EPS = 0.01         # °C the peak must exceed the 2100 value to count as overshoot

running_mean_temps = load_running_means()
# real scenarios only (SSPs + NGFS families); the 71 Current-Policies branch-offs are
# the ambition grid (Fig 3a) and would swamp the table.
scenarios = [s for s in running_mean_temps
             if "GCAM" not in s and "MESSAGE" not in s and "Current_Policies__start" not in s]
fancy_titles, _, _, _ = build_naming(scenarios)
print(f"Loaded running means for {len(scenarios)} scenarios (excl. GCAM/MESSAGE)", flush=True)

rows = []
for sc in scenarios:
    rm = running_mean_temps[sc]                      # dims: (timebounds, member)

    # median @2100 + ensemble spread
    at2100 = rm.sel(timebounds=PEAK_HORIZON, method="nearest")
    med_2100 = float(at2100.median("member"))
    p5_2100  = float(at2100.quantile(0.05, dim="member"))
    p95_2100 = float(at2100.quantile(0.95, dim="member"))

    # peak of the ensemble-median path up to 2100
    med_path = rm.median("member").sel(timebounds=slice(None, PEAK_HORIZON))
    peak_val = float(med_path.max())
    peak_year = float(med_path.idxmax("timebounds"))
    overshoot = (peak_year < PEAK_HORIZON) and (peak_val > med_2100 + OVERSHOOT_EPS)

    rows.append({
        "scenario": sc,
        "name": fancy_titles.get(sc, sc),
        "family": ("SSP" if get_model(sc) == "Other" and sc.startswith("ssp")
                   else ("Branch-off" if "Current_Policies__start" in sc else "NGFS")),
        "median_warming_2100_C": med_2100,
        "p5_2100_C": p5_2100,
        "p95_2100_C": p95_2100,
        "peak_warming_le2100_C": peak_val,
        "peak_year": peak_year,
        "peaks_before_2100": bool(overshoot),
    })

df = pd.DataFrame(rows).sort_values("median_warming_2100_C", ascending=False).reset_index(drop=True)

csv_path = os.path.join(OUT_DIR, "si_warming_table.csv")
df.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}  ({len(df)} scenarios)", flush=True)

# formatted markdown for the SI
md = df.copy()
md["Median warming 2100 [°C]"] = md.apply(
    lambda r: f"{r['median_warming_2100_C']:.2f} [{r['p5_2100_C']:.2f}–{r['p95_2100_C']:.2f}]", axis=1)
md["Peak warming ≤2100 [°C]"] = md.apply(
    lambda r: (f"{r['peak_warming_le2100_C']:.2f} ({int(round(r['peak_year']))})"
               if r["peaks_before_2100"] else "—"), axis=1)
md = md[["name", "family", "Median warming 2100 [°C]", "Peak warming ≤2100 [°C]"]]
md = md.rename(columns={"name": "Scenario", "family": "Family"})
md_path = os.path.join(OUT_DIR, "si_warming_table.md")
md.to_markdown(md_path, index=False)
print(f"Wrote {md_path}", flush=True)
print(md.to_string(index=False))
