"""29_tables_si_constraints.py — SI: resilience per scenario for the ALTERNATIVE
criterion sets (the main table 19 covers the standard 1.5 °C / 0.4 °C-per-decade / 2100).

Same content/format as 19 (path-based resilience + both 95% intervals: crossed
bootstrap from 09, and exact Clopper–Pearson with N = configs), for the two other
criterion sets produced by 09:
  * 1.5 °C, 0.3 °C per decade
  * 2.0 °C, 0.4 °C per decade
Real scenarios only (SSPs + NGFS families; the branch-offs are the Fig-3a grid).

Reads the 09 summaries — light, runs on login.
Output -> output/tables/si_resilience_constraints.{csv,md}
"""
import os
import sys

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import RESILIENCE_BASE, FAIR_PARAMS_FILE, build_naming, get_model

OUT_DIR = os.path.join("tables"); os.makedirs(OUT_DIR, exist_ok=True)
N_CP = len(pd.read_csv(FAIR_PARAMS_FILE, index_col=0))   # configs = effective independent n

# (res_x, res_y, rate_str, human label) — the two alternative criterion sets
CONDITIONS = [
    (2100, 1.5, "03", "1.5 °C, 0.3 °C per decade"),
    (2100, 2.0, "04", "2.0 °C, 0.4 °C per decade"),
]


def _family(sc):
    return "SSP" if sc.startswith("ssp") else "NGFS"


def _cp(p):
    lo, hi = proportion_confint(count=round(p * N_CP), nobs=N_CP, alpha=0.05, method="beta")
    return lo * 100.0, hi * 100.0


all_rows, md_blocks = [], []
for res_x, res_y, rate_str, label in CONDITIONS:
    summary = os.path.join(
        RESILIENCE_BASE, f"resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}",
        "summary", f"resilience_summary_time{res_x}_temp{res_y}_rate{rate_str}.csv",
    )
    if not os.path.exists(summary):
        print(f"  missing {summary} — skipping", flush=True); continue
    df = pd.read_csv(summary)
    df = df[~df["scenario"].str.contains("GCAM|MESSAGE|Current_Policies__start", regex=True)].copy()
    fancy, _, _, _ = build_naming(df["scenario"].tolist())
    df["name"] = df["scenario"].map(fancy)
    df["family"] = df["scenario"].map(_family)
    df["resilience_pct"] = df["resilience"] * 100.0
    df["boot"] = df.apply(lambda r: f"[{r['resilience_ci_low']*100:.1f}–{r['resilience_ci_high']*100:.1f}]", axis=1)
    df["cp"] = df["resilience"].map(lambda p: "[{:.1f}–{:.1f}]".format(*_cp(p)))
    df["criterion"] = label
    df = df.sort_values("resilience", ascending=False).reset_index(drop=True)
    all_rows.append(df[["criterion", "scenario", "name", "family", "resilience_pct",
                        "resilience_ci_low", "resilience_ci_high"]])

    md = df[["name", "family", "resilience_pct", "boot", "cp"]].copy()
    md["resilience_pct"] = md["resilience_pct"].map(lambda v: f"{v:.1f}")
    md = md.rename(columns={"name": "Scenario", "family": "Family",
                            "resilience_pct": "Path-based resilience [%]",
                            "boot": "95% CI (bootstrap)", "cp": "95% CI (Clopper–Pearson)"})
    md_blocks.append(f"### Criterion: {label}\n\n" + md.to_markdown(index=False) + "\n")
    print(f"\n=== {label} ===\n" + md.to_string(index=False), flush=True)

pd.concat(all_rows, ignore_index=True).to_csv(
    os.path.join(OUT_DIR, "si_resilience_constraints.csv"), index=False)
with open(os.path.join(OUT_DIR, "si_resilience_constraints.md"), "w") as fh:
    fh.write(f"Path-based resilience per scenario for the alternative criterion sets "
             f"(Clopper–Pearson N = {N_CP} configs). Standard 1.5 °C / 0.4 / 2100 is the "
             f"main table.\n\n" + "\n".join(md_blocks))
print(f"\nWrote output/tables/si_resilience_constraints.{{csv,md}}  [N_CP={N_CP}]", flush=True)
