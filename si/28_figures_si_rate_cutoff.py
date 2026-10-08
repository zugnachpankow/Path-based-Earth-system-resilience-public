"""28_figures_si_rate_cutoff.py — SI: Sobol sensitivity vs the rate-threshold cutoff.

The warming-rate limit is not well constrained, and the resilience metric's sensitivity
to it depends on the chosen range. This compares the total-order Sobol index (ST) of the
four criterion parameters under the two rate-threshold ranges we ran at full resolution:
  * 0.25–0.55 °C/decade  (production; rate0.025_0.055)
  * 0.30–0.50 °C/decade  (narrower;   rate0.03_0.05)
for the three focal SSPs. The rate parameter's ST changes markedly with the cutoff (e.g.
SSP5-3.4-OS), which is the point; the other parameters are largely stable.

Reads output/sensitivity/rate*/sobol_*.pkl (10). Light — runs on login.
Output -> output/figures/si/si_rate_cutoff_sensitivity.{pdf,png}
"""
import os
import sys
import pickle
from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import cmcrameri.cm as cmcs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import TWO_COL, FIG_DIR, apply_style, get_scenario_type

apply_style()
OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)

RANGES = [("0.25–0.55 °C/dec", "rate0.025_0.055", cmcs.glasgow(0.30)),
          ("0.30–0.50 °C/dec", "rate0.03_0.05",   cmcs.glasgow(0.70))]
FOCAL = ["ssp126", "ssp245", "ssp534-over"]
PARAMS = ["Year", "Temp", "Rate", "Tipping"]

fig, axes = plt.subplots(1, len(FOCAL), figsize=(TWO_COL, TWO_COL * 0.34), sharey=True)
_lbl_off = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)
x = np.arange(len(PARAMS)); w = 0.38
for k, (ax, sc) in enumerate(zip(axes, FOCAL)):
    for i, (label, subdir, col) in enumerate(RANGES):
        with open(os.path.join("output/sensitivity", subdir, f"sobol_{sc}.pkl"), "rb") as f:
            Si = pickle.load(f)["Si"]
        ax.bar(x + (i - 0.5) * w, Si["ST"], w, yerr=Si["ST_conf"], capsize=1.5,
               color=col, label=label if k == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(PARAMS, fontsize=6)
    ax.axhline(0, color="k", lw=0.4); ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.text(0.0, 1.03, ascii_lowercase[k], transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.text(0.0, 1.03, get_scenario_type(sc), transform=ax.transAxes + _lbl_off,
            fontsize=6, fontstyle="italic", va="bottom", ha="left", color="#444444",
            clip_on=False)
axes[0].set_ylabel("Total-order Sobol index (ST)")
axes[0].legend(frameon=False, fontsize=5, loc="upper left", title="Rate threshold range",
               title_fontsize=5)
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_rate_cutoff_sensitivity" + ext), dpi=300, bbox_inches="tight")
print("Saved si_rate_cutoff_sensitivity.{pdf,png}", flush=True)
