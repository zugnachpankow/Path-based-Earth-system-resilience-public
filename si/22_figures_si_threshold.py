"""22_figures_si_threshold.py — SI: resilience sensitivity to the criterion thresholds.

Partial-dependence of path-based resilience on each of the four criterion parameters
(res_x, res_y, rate, tipping), from the 10 MC pickles (10 000 LHS draws over all four).
2x2 panels; one line per focal SSP in its canonical colour; dashed reference line at
the standard criterion (1.5 °C / 0.4 °C-per-decade / 2100). Light — reads mc_*.pkl.

Output -> output/figures/si/si_threshold_sensitivity.{pdf,png}
"""
import os
import sys
import pickle
from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import TWO_COL, FIG_DIR, SENS_DIR_SOBOL, main_scenarios, get_scenario_type, composite_styles

OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
FOCAL = ["ssp126", "ssp245", "ssp534-over"]
_styles = composite_styles(main_scenarios)

# param columns of param_mc: 0=res_x, 1=res_y, 2=rate, 3=tipping
mc = {}
for sc in FOCAL:
    with open(os.path.join(SENS_DIR_SOBOL, f"mc_{sc}.pkl"), "rb") as f:
        d = pickle.load(f)
    mc[sc] = (np.asarray(d["param_mc"]), np.asarray(d["Y_mc"]))
print("loaded MC for", list(mc.keys()), flush=True)


def pd_line(x, y, nbins=12):
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    means = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(nbins)])
    return mids, means * 100.0


# panel spec: (col index in param_mc, x-transform, xlabel, reference value)
panels = [
    (1, lambda v: v,        "Temperature threshold [°C]", 1.5),
    (2, lambda v: v * 10,   "Rate threshold [°C per decade]", 0.4),
    (0, lambda v: v,        "Return year", 2100),
    (3, None,               "Tipping", None),
]

fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, TWO_COL * 0.7))
_lbl = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)
for k, (ax, (col, xf, xlab, ref)) in enumerate(zip(axes.ravel(), panels)):
    for sc in FOCAL:
        pm, Y = mc[sc]
        color, ls = _styles.get(sc, ("grey", "-"))
        if col == 3:  # tipping: off (<0.5) vs on (>0.5)
            off = Y[pm[:, 3] < 0.5].mean() * 100.0
            on  = Y[pm[:, 3] >= 0.5].mean() * 100.0
            ax.plot([0, 1], [off, on], marker="o", color=color, linestyle=ls, lw=1.2, ms=4,
                    label=get_scenario_type(sc))
            ax.set_xticks([0, 1]); ax.set_xticklabels(["off", "on"])
        else:
            xm, ym = pd_line(xf(pm[:, col]), Y)
            ax.plot(xm, ym, color=color, linestyle=ls, lw=1.2, label=get_scenario_type(sc))
    if ref is not None:
        ax.axvline(ref, color="grey", ls=":", lw=0.8, zorder=1)
    ax.set_xlabel(xlab)
    ax.grid(True, alpha=0.3)
    ax.text(0.0, 1.02, ascii_lowercase[k], transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
axes[0, 0].set_ylabel("Path-based resilience [%]")
axes[1, 0].set_ylabel("Path-based resilience [%]")
axes[0, 0].legend(frameon=False, fontsize=5, loc="upper left")
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_threshold_sensitivity" + ext), dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved si_threshold_sensitivity.{pdf,png}", flush=True)
