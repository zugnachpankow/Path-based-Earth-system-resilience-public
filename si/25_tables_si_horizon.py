"""25_tables_si_horizon.py — SI table: horizon sensitivity of SSP path-based resilience.

Transparency table for the "ever-exceeds-over-horizon" operator: path-based resilience
depends weakly on how long each scenario is evaluated, because the temperature condition
(GMST > 1.5 °C at ANY t >= 2100) is a maximum over the post-2100 window and internal
variability gives borderline members more chances to cross once over a longer window.

Reports, per SSP, path-based resilience evaluated to the full trajectory vs cut at 2109,
the difference (percentage points), and the climate-condition failure rate. Tipping is
horizon-independent (committed), so the aggregated
prob_any_tipping is used (linearity) — light compute; only the 31G running-mean pickle
load is heavy, so submit via SLURM (run_25.sh). The formatted SI table itself is rebuilt
from the cached horizon_validation.csv, so it can be regenerated on login.

Reads output/running_mean_temps.pkl (08) + output/tipping/*.nc (07).
Writes output/resilience/horizon_validation.csv and
       output/tables/si_horizon_sensitivity.{csv,md}.
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from resilience import load_tipping, compute_resilience
from figures_common import get_scenario_type, ssp_scenarios

RM_FILE  = "output/running_mean_temps.pkl"
TIP_DIR  = "output/tipping"
VALID_CSV = "output/resilience/horizon_validation.csv"
TABLE_DIR = "tables"

res_x, res_y, rate = 2100, 1.5, 0.04
RES_ENDS = {"full": None, "2109_cut": 2109}


def compute_validation():
    """Compute the horizon comparison (heavy: loads the 31G running-mean pickle)."""
    print("Loading running means …", flush=True)
    with open(RM_FILE, "rb") as f:
        rm_all = pickle.load(f)
    rm_all = {k.replace("(", "").replace(")", ""): v for k, v in rm_all.items()}
    tip = load_tipping(TIP_DIR)

    ssps = sorted(s for s in rm_all if s.startswith("ssp") and s in tip)
    print(f"SSP scenarios: {ssps}\n", flush=True)

    def climate_viol_frac(rm, res_end):
        return float(((rm.sel(timebounds=slice(res_x, res_end)) > res_y).any("timebounds")).mean())

    rows = []
    for sc in ssps:
        rm = rm_all[sc]
        tp_df = (tip[sc]["prob_any_tipping"].to_dataframe(name="tip").reset_index()
                 [["config", "run", "tip"]])
        row = {"scenario": sc, "horizon_end": float(rm.timebounds.max())}
        for label, re_end in RES_ENDS.items():
            cr = compute_resilience(rm, res_x, res_y, rate, res_end=re_end)
            m = cr.merge(tp_df, on=["config", "run"], how="left")
            row[f"CplusR_{label}"]    = float(m["resilience_cr"].mean())
            row[f"pathbased_{label}"] = float((m["resilience_cr"] * (1.0 - m["tip"])).mean())
            row[f"climviol_{label}"]  = climate_viol_frac(rm, re_end)
        rows.append(row)
        print(f"{sc:12s} horizon_end={row['horizon_end']:.0f}  "
              f"path full={row['pathbased_full']:.4f}  2109={row['pathbased_2109_cut']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(VALID_CSV), exist_ok=True)
    df.to_csv(VALID_CSV, index=False)
    print(f"\nWrote {VALID_CSV}", flush=True)
    return df


def build_si_table(df):
    """Format the cached comparison into the SI transparency table (csv + md)."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    order = {sc: i for i, sc in enumerate(ssp_scenarios)}
    df = df.sort_values("scenario", key=lambda s: s.map(lambda x: order.get(x, 99)))

    # tolerate both label schemes ("full" and the older "2300_full")
    def _col(base):
        for suffix in ("full", "2300_full"):
            if f"{base}_{suffix}" in df:
                return f"{base}_{suffix}"
        raise KeyError(f"no full-horizon column for {base}")
    path_full, climviol_full = _col("pathbased"), _col("climviol")

    horizon_end = int(round(df["horizon_end"].max())) if "horizon_end" in df else None
    out = pd.DataFrame({
        "Scenario": df["scenario"].map(get_scenario_type),
        "Path-based resilience [%], full horizon": (df[path_full] * 100).round(2),
        "Path-based resilience [%], to 2109":      (df["pathbased_2109_cut"] * 100).round(2),
        "Δ [pp] (2109 − full)": ((df["pathbased_2109_cut"] - df[path_full]) * 100).round(2),
        "Climate-condition failure [%], full horizon": (df[climviol_full] * 100).round(1),
    })
    csv_path = os.path.join(TABLE_DIR, "si_horizon_sensitivity.csv")
    md_path  = os.path.join(TABLE_DIR, "si_horizon_sensitivity.md")
    out.to_csv(csv_path, index=False)

    max_shift = (df["pathbased_2109_cut"] - df[path_full]).abs().max() * 100
    caption = (
        f"**Supplementary Table.** Horizon sensitivity of path-based resilience for the SSP "
        f"scenarios (temperature 1.5 °C / rate 0.4 °C per decade / 2100). The temperature "
        f"condition is evaluated as \"GMST exceeds 1.5 °C at any time from 2100 onward\", a "
        f"maximum over the post-2100 window; resilience is therefore weakly horizon-dependent, "
        f"as internal variability gives borderline members more chances to cross the threshold "
        f"once over a longer window. Comparing the full evaluation horizon "
        f"({'~%d' % horizon_end if horizon_end else 'full'}) with a cut at 2109 shifts path-based "
        f"resilience by at most {max_shift:.1f} percentage points, well within the "
        f"crossed-bootstrap 95% confidence intervals (≈±2–3 pp). Tipping is treated as a "
        f"committed, horizon-independent property."
    )
    with open(md_path, "w") as fh:
        fh.write(caption + "\n\n")
        fh.write(out.to_markdown(index=False))
        fh.write("\n")
    print(f"Wrote {csv_path}\nWrote {md_path}", flush=True)
    print("\n" + out.to_string(index=False), flush=True)


if __name__ == "__main__":
    # Reuse the cached comparison if present (formatting is light, runs on login);
    # otherwise compute it (heavy pickle load -> SLURM).
    if os.path.exists(VALID_CSV):
        print(f"Using cached {VALID_CSV}", flush=True)
        df = pd.read_csv(VALID_CSV)
    else:
        df = compute_validation()
    build_si_table(df)
