"""24_figures_si_os_sensitivity.py — SI: MC sensitivity for the overshoot scenario.

Nature-style reproduction of three panels from the original MC sensitivity dashboard
(code/sensitivity/study.py), for SSP5-3.4-OS only:
  (a) MC distribution of path-based resilience, with vs without the tipping constraint
      (tail-risk view; + baselines),
  (b) resilience vs temperature threshold res_y, coloured by tipping / no-tipping
      (the "res_y cliff"),
  (c) conditional importance of the rate: Spearman(rate, resilience) within res_y
      quintiles, plotted against res_y.

Reads output/sensitivity/mc_ssp534-over.pkl — light, runs on login.
Output -> output/figures/si/si_os_mc_sensitivity.{pdf,png}
"""
import os
import sys
import pickle
from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import cmcrameri.cm as cmcs
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import TWO_COL, FIG_DIR, SENS_DIR_SOBOL

OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
SC = "ssp534-over"
# reuse the glasgow palette used elsewhere: light = no tipping, dark = with tipping
NOTIP_C, TIP_C, SPEAR_C = cmcs.glasgow(0.15), cmcs.glasgow(0.85), cmcs.glasgow(0.5)

with open(os.path.join(SENS_DIR_SOBOL, f"mc_{SC}.pkl"), "rb") as f:
    d = pickle.load(f)
pm = np.asarray(d["param_mc"])          # cols: res_x, res_y, rate, tipping
Y = np.asarray(d["Y_mc"]) * 100.0       # resilience [%]
base_notip = d.get("baseline_no_tip", np.nan) * 100.0
base_tip = d.get("baseline_tip", np.nan) * 100.0
mask_tip = pm[:, 3] > 0.5
mask_notip = ~mask_tip
print(f"{SC}: {len(Y)} MC draws, resilience {Y.min():.2f}-{Y.max():.2f}%", flush=True)


def conditional_spearman_rate(param_mc_sc, Y_sc, n_bins=5):
    """Spearman r(rate, resilience) conditioned on res_y quintile."""
    res_y_bins = np.percentile(param_mc_sc[:, 1], np.linspace(0, 100, n_bins + 1))
    centers, r_vals = [], []
    for lo, hi in zip(res_y_bins[:-1], res_y_bins[1:]):
        m = (param_mc_sc[:, 1] >= lo) & (param_mc_sc[:, 1] < hi)
        if m.sum() < 30:
            continue
        r, _ = spearmanr(param_mc_sc[m, 2], Y_sc[m])
        centers.append((lo + hi) / 2); r_vals.append(r)
    return np.array(centers), np.array(r_vals)


fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(TWO_COL, TWO_COL * 0.32))
_lbl = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)

# (a) MC distribution: with vs without tipping
axa.hist(Y[mask_notip], bins=50, density=True, color=NOTIP_C, alpha=0.55, label="Without tipping")
axa.hist(Y[mask_tip], bins=50, density=True, color=TIP_C, alpha=0.55, label="With tipping")
if np.isfinite(base_notip):
    axa.axvline(base_notip, color=NOTIP_C, lw=1.2, ls=":", label="Baseline, no tipping")
if np.isfinite(base_tip):
    axa.axvline(base_tip, color=TIP_C, lw=1.2, ls=":", label="Baseline, with tipping")
axa.set_xlabel("Path-based resilience [%]"); axa.set_ylabel("Density")
axa.legend(frameon=False, fontsize=5, loc="upper right")

# (b) resilience vs temperature threshold (res_y cliff)
axb.scatter(pm[mask_notip, 1], Y[mask_notip], c=[NOTIP_C], alpha=0.25, s=4, rasterized=True, label="No tipping")
axb.scatter(pm[mask_tip, 1], Y[mask_tip], c=[TIP_C], alpha=0.25, s=4, rasterized=True, label="With tipping")
axb.axvline(1.5, color="k", lw=0.9, ls="--", label="1.5 °C baseline")
axb.set_xlabel("Temperature threshold [°C]"); axb.set_ylabel("Path-based resilience [%]")
axb.legend(frameon=False, fontsize=5, loc="upper left")

# (c) conditional importance of rate
centers, r_vals = conditional_spearman_rate(pm, np.asarray(d["Y_mc"]))
axc.plot(centers, r_vals, "o-", color=SPEAR_C, lw=1.2, ms=4, label="Spearman r (rate vs resilience)")
axc.axhline(0, color="k", lw=0.5)
axc.set_xlabel("Temperature threshold [°C]")
axc.set_ylabel("Spearman r (rate vs resilience)")
axc.set_ylim(-0.1, 1.0)
axc.legend(frameon=False, fontsize=5, loc="upper right")

for k, ax in enumerate((axa, axb, axc)):
    ax.text(0.0, 1.02, ascii_lowercase[k], transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_os_mc_sensitivity" + ext), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved si_os_mc_sensitivity.{pdf,png}", flush=True)
