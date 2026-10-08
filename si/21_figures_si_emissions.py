"""21_figures_si_emissions.py — SI emissions figure by family (CO2 FFI+AFOLU).

One 3-panel figure (no titles, GtCO2/yr, Nature style, shared scenario colours;
panels have independent y-limits, single y-label on the middle panel):
  (a) SSPs (from RCMIP)
  (b) REMIND NGFS families
  (c) Current-Policies branch-offs, coloured by reduction rate

Colours/linestyles reuse figures_common.composite_styles (same as the combined heatmap)
for the main scenarios, ssp_color for the other SSPs. Light — reads emission CSVs only.

Output -> output/figures/si/si_emissions_combined.{pdf,png}
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmcs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))

import matplotlib.transforms as mtrans
from string import ascii_lowercase
from figures_common import (
    TWO_COL, FIG_DIR, main_scenarios, get_scenario_type, build_naming,
    ssp_color, composite_styles,
)

OUT_DIR = os.path.join(FIG_DIR, "si")
os.makedirs(OUT_DIR, exist_ok=True)

NGFS_CSV   = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
BRANCH_CSV = "data/processed/current_policy_branch_offs_2300.csv"
RCMIP_CSV  = "data/raw/rcmip-emissions-annual-means-v5-1-0.csv"
X0, X1 = 2000, 2100
YLABEL = "CO$_2$ FFI+AFOLU [GtCO$_2$ yr$^{-1}$]"

_styles = composite_styles(main_scenarios)
FRAGMENTED_COLOR = "#8c510a"


def _style_for(sc):
    if sc in _styles:                    # main scenarios: exact heatmap colour + linestyle
        return _styles[sc]
    c = ssp_color(sc)                    # other SSPs (370/434/460/585) -> canonical colour
    if c is not None:
        return (c, "-")
    if "Fragmented" in sc:
        return (FRAGMENTED_COLOR, "-")
    return ("dimgray", "-")


def _year_cols(df):
    yc = [c for c in df.columns if c.replace(".", "", 1).isdigit()]
    return yc, np.array([float(c) for c in yc])


def _co2_series(df, scenario, year_cols, ffi, afolu, region="World"):
    m = (df["scenario"] == scenario) & (df["region"] == region) & (df["variable"].isin([ffi, afolu]))
    # min_count=1 keeps NaN where a year is unreported (RCMIP reports SSPs only every
    # ~10 yr after 2015) — otherwise the NaN gaps become 0 and the line looks pulsed.
    return df.loc[m, year_cols].sum(axis=0, min_count=1).values.astype(float) / 1000.0   # Mt -> Gt


# ── load all three sources ───────────────────────────────────────────────────────
ngfs = pd.read_csv(NGFS_CSV)
yc, yrs = _year_cols(ngfs)
mask = (yrs >= X0 - 1) & (yrs <= X1 + 1)

rc = pd.read_csv(RCMIP_CSV).rename(
    columns={"Scenario": "scenario", "Region": "region", "Variable": "variable"})
ssp_list = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp434", "ssp460", "ssp534-over", "ssp585"]
ycr = [c for c in rc.columns if c.isdigit()]
yrr = np.array([float(c) for c in ycr])
maskr = (yrr >= X0 - 1) & (yrr <= X1 + 1)
FFI = "Emissions|CO2|MAGICC Fossil and Industrial"
AFO = "Emissions|CO2|MAGICC AFOLU"

bo = pd.read_csv(BRANCH_CSV)
ycb, yrsb = _year_cols(bo)
maskb = (yrsb >= X0 - 1) & (yrsb <= X1 + 1)
bo_scen = [s for s in bo["scenario"].unique() if "start" in s]

# ── combined 3-panel figure ───────────────────────────────────────────────────────
fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(TWO_COL, TWO_COL * 0.36))
_lbl_off = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)

# (a) SSPs (RCMIP) — widest range, kept on its own y-axis
for sc in ssp_list:
    y = _co2_series(rc, sc, ycr, FFI, AFO)
    sel = maskr & np.isfinite(y)
    col, ls = _style_for(sc)
    axa.plot(yrr[sel], y[sel], color=col, linestyle=ls, lw=1.2, label=get_scenario_type(sc), zorder=3)
axa.legend(frameon=False, fontsize=4.5, ncol=1, loc="upper left")

# (b) NGFS families
scen = [s for s in ngfs["scenario"].unique()
        if "REMIND" in s and "start" not in s and "GCAM" not in s and "MESSAGE" not in s]
for sc in scen:
    y = _co2_series(ngfs, sc, yc, "CO2 FFI", "CO2 AFOLU")
    col, ls = _style_for(sc)
    axb.plot(yrs[mask], y[mask], color=col, linestyle=ls, lw=1.2, label=get_scenario_type(sc), zorder=3)
axb.legend(frameon=False, fontsize=4.5, ncol=1, loc="upper right")

# (c) branch-offs, coloured by reduction rate
rates = sorted({float(re.search(r"_r(\d+\.?\d*)", s).group(1)) for s in bo_scen
                if re.search(r"_r(\d+\.?\d*)", s)})
norm = plt.Normalize(min(rates), max(rates)); cmap = cmcs.batlow
for sc in bo_scen:
    m = re.search(r"_r(\d+\.?\d*)", sc)
    if not m:
        continue
    y = _co2_series(bo, sc, ycb, "CO2 FFI", "CO2 AFOLU")
    axc.plot(yrsb[maskb], y[maskb], color=cmap(norm(float(m.group(1)))), lw=1.0, alpha=0.85, zorder=2)
y_cp = _co2_series(ngfs, "REMIND-MAgPIE_3.3-4.8___Current_Policies", yc, "CO2 FFI", "CO2 AFOLU")
axc.plot(yrs[mask], y_cp[mask], color="black", lw=1.4, label="Current Policies (base)", zorder=4)
axc.legend(frameon=False, fontsize=4.5, loc="upper right")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = fig.colorbar(sm, ax=axc, fraction=0.046, pad=0.02)
cb.set_label("Reduction rate [fraction yr$^{-1}$]", fontsize=6); cb.ax.tick_params(labelsize=5)

_titles = ["SSPs", "NGFS families", "Current-Policies branch-offs"]
for k, ax in enumerate((axa, axb, axc)):
    ax.set_xlim(X0, X1); ax.set_xlabel("Year")
    ax.axhline(0, color="black", lw=0.4, zorder=1); ax.grid(True, alpha=0.3)
    ax.text(0.0, 1.03, ascii_lowercase[k], transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.text(0.0, 1.03, _titles[k], transform=ax.transAxes + _lbl_off, fontsize=6,
            fontstyle="italic", va="bottom", ha="left", color="#444444", clip_on=False)
axb.set_ylabel(YLABEL)   # single y-label on the middle panel (panels have independent y-lims)
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_emissions_combined" + ext), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved si_emissions_combined.{pdf,png}", flush=True)
