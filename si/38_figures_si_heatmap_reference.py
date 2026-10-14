"""38_figures_si_heatmap_reference.py — SI: the Fig. 3 combined figure (a: NZ2050 heatmap,
b: fragility) with reference marks for "where we likely are".

This is a self-contained COPY of 16's combined-figure drawing (not a refactor — the main
Fig. 3 in 16 is untouched) plus:
  * y (TCRE): AR6 likely range 1.0–2.3 K/1000 PgC as a light band on the left marginal, with
    the AR6 best estimate 1.65 as a dark dashed line on that marginal (IPCC AR6 WGI SPM
    D.1.1: 0.45 [0.27–0.63] °C per 1000 GtCO2, × 3.664).
  * x (T_eff): the 5–95 % of t_eff_central_lit as a light band on the bottom marginal, with
    its median as a dark dashed line on that marginal.
The reference marks live in the marginals only (not across the heatmap cells); their legend
sits below panel a, and panel b is identical to Fig. 3b.

Reads 16's combined_cache.pkl + tcre.csv + t_eff_lhs1000.csv + t_eff_central_lit.csv
(computed/cached here if missing — needs SLURM CPUs; the plotting is light).
Output -> figures/si/si_heatmap_reference_marks.{pdf,png}
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import (TWO_COL, FIG_DIR, RESILIENCE_BASE, apply_style,
                            composite_styles, main_scenarios, get_scenario_type)
from heatmap_common import N_DEC, build_axis, draw_heatmap
import t_eff

apply_style()
OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
FEEDBACK = os.path.join(os.path.dirname(RESILIENCE_BASE), "feedback")
CENTRAL_CSV = os.path.join(FEEDBACK, "t_eff_central_lit.csv")

# labels (copied from 16, unchanged)
Y_LABEL = "Climate and carbon sensitivity\nTCRE [K per 1000 PgC]"
X_LABEL = "Tipping susceptibility\nEffective tipping temperature [°C]"
FRAG_LABEL = "Earth system fragility\n(Rank composite)"
MARGINAL_LABEL = "Path-based\nresilience [%]"
MARK = dict(color="0.15", linestyle="--", linewidth=0.9, zorder=6)   # plain dark dashed, marginals only
BAND = dict(color="0.45", alpha=0.20, zorder=0)


def _legend_name(sc):
    return {"Current Policies": "Current\nPolicies", "Net Zero 2050": "Net Zero\n2050",
            "Below 2°C": "Below\n2 °C", "Low Demand": "Low\nDemand"}.get(get_scenario_type(sc),
                                                                          get_scenario_type(sc))

# ── data ──────────────────────────────────────────────────────────────────────────
cache = pickle.load(open(os.path.join(FIG_DIR, "heatmaps", "combined_cache.pkl"), "rb"))
nz = cache["nz2050_data"]; fragility_results = cache["fragility_results"]
tcre = pd.read_csv(os.path.join(FEEDBACK, "tcre.csv"), index_col="config")["TCRE"]
teff = pd.read_csv(os.path.join(FEEDBACK, "t_eff_lhs1000.csv"), index_col="sample")["T_eff"]
tcre_dec, tcre_rank, tcre_edges = build_axis(tcre)
teff_dec, teff_rank, teff_edges_asc = build_axis(teff)
teff_edges_desc = teff_edges_asc[::-1]
_styles = composite_styles(main_scenarios)

if not os.path.exists(CENTRAL_CSV):
    t_eff.compute_t_eff_central_lit(CENTRAL_CSV)
cl = pd.read_csv(CENTRAL_CSV, index_col="sample")["T_eff"]

AR6 = {"best": 0.45 * 3.664, "lo": 0.27 * 3.664, "hi": 0.63 * 3.664}   # K per 1000 PgC
cl_med, cl_5, cl_95 = float(np.nanmedian(cl)), float(np.nanpercentile(cl, 5)), float(np.nanpercentile(cl, 95))


# ── value -> decile-axis position:  pos = i + (v - e_i)/(e_{i+1} - e_i) ─────────────
def _pos(v, edges, name):
    e0, eN = edges[0], edges[-1]
    lo_edge, hi_edge = min(e0, eN), max(e0, eN)
    if v <= lo_edge:
        if v < lo_edge:
            print(f"[warn] {name}={v:.3f} below axis range [{lo_edge:.3f}, {hi_edge:.3f}] -> clipped", flush=True)
        return 0.0 if edges[0] < edges[-1] else float(N_DEC)
    if v >= hi_edge:
        if v > hi_edge:
            print(f"[warn] {name}={v:.3f} above axis range [{lo_edge:.3f}, {hi_edge:.3f}] -> clipped", flush=True)
        return float(N_DEC) if edges[0] < edges[-1] else 0.0
    for i in range(N_DEC):            # works for ascending or descending edges
        if (edges[i] - v) * (edges[i + 1] - v) <= 0:
            return i + (v - edges[i]) / (edges[i + 1] - edges[i])
    return float(N_DEC)


# ══════════════════════════════════════════════════════════════════════════════════
# combined figure (copied from 16) + reference marks
# ══════════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(TWO_COL, TWO_COL * 9 / 18))
sf_a, sf_b = fig.subfigures(1, 2, width_ratios=[1.3, 1.0], wspace=0.12)
gs_a = GridSpec(2, 3, width_ratios=[1.2, 4, 0.25], height_ratios=[4, 1.2], hspace=0.2, wspace=0.12, figure=sf_a)
ax_heat = sf_a.add_subplot(gs_a[0, 1]); ax_left = sf_a.add_subplot(gs_a[0, 0], sharey=ax_heat)
ax_bottom = sf_a.add_subplot(gs_a[1, 1], sharex=ax_heat); ax_cbar = sf_a.add_subplot(gs_a[0, 2])
draw_heatmap(ax_heat, ax_left, ax_bottom, ax_cbar, nz, sf_a, tcre_edges, teff_edges_desc, Y_LABEL, X_LABEL, cell_fs=5)
ax_heat.plot([0, N_DEC], [0, N_DEC], color="grey", linestyle="--", linewidth=2.0, alpha=0.8, zorder=2.5)
ax_left.legend(loc="upper left", frameon=False, fontsize=5)
_marg_leg = ax_bottom.legend(loc="upper right", frameon=False, fontsize=5)   # keep (re-added below)
sf_a.text(0.10, 0.19, MARGINAL_LABEL, ha="left", va="bottom", fontsize=6)
_lbl = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)
ax_left.text(0.0, 1.02, "a", transform=ax_left.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", clip_on=False)
ax_left.text(0.0, 1.02, "Net Zero 2050", transform=ax_left.transAxes + _lbl, fontsize=6, fontstyle="italic",
             va="bottom", ha="left", color="#444444", clip_on=False)

# ── reference marks on panel a ──
y_lo, y_hi, y_best = _pos(AR6["lo"], tcre_edges, "AR6 lo"), _pos(AR6["hi"], tcre_edges, "AR6 hi"), _pos(AR6["best"], tcre_edges, "AR6 best")
x_med, x_5, x_95 = _pos(cl_med, teff_edges_desc, "T_eff med"), _pos(cl_5, teff_edges_desc, "T_eff 5%"), _pos(cl_95, teff_edges_desc, "T_eff 95%")
ax_left.axhspan(y_lo, y_hi, **BAND)                                    # AR6 likely range band
ax_left.axhline(y_best, **MARK)                                       # AR6 best estimate line (marginal only)
ax_bottom.axvspan(min(x_5, x_95), max(x_5, x_95), **BAND)             # central-lit 5-95% band
ax_bottom.axvline(x_med, **MARK)                                      # central-lit median line (marginal only)
# reference-marks legend below panel a (under the x-axis label), ncol=2: col1 TCRE, col2 T_eff.
# matplotlib fills row-major, so order handles row0=[TCRE band, Teff band], row1=[TCRE line, Teff line].
_mk_line = lambda: Line2D([0], [0], color=MARK["color"], linestyle=MARK["linestyle"], linewidth=MARK["linewidth"])
_mk_band = lambda: Patch(facecolor=BAND["color"], alpha=BAND["alpha"])
_marks_leg = ax_bottom.legend(
    handles=[_mk_band(), _mk_band(), _mk_line(), _mk_line()],
    labels=["TCRE: AR6 likely range", "$T_\\mathrm{eff}$: central thresholds, 5–95 %",
            "TCRE: AR6 best estimate", "$T_\\mathrm{eff}$: central thresholds, median"],
    loc="upper center", bbox_to_anchor=(0.5, -1.15), ncol=2, frameon=False, fontsize=5,
    handlelength=1.8, columnspacing=1.4)
ax_bottom.add_artist(_marg_leg)                                       # restore the Mean/95% CI legend

# ── panel b (copied) + composite reference line ──
ax_b = sf_b.add_subplot(111)
for sc_name, res_b in fragility_results.items():
    if get_scenario_type(sc_name) == "Delayed Transition":
        continue
    col_b, ls_b = _styles.get(sc_name, ("grey", "-"))
    ax_b.plot(res_b["fragility_mid"], res_b["mean"] * 100, color=col_b, linewidth=1.5,
              linestyle=ls_b, label=_legend_name(sc_name), zorder=3)
ax_b.set_xlabel(FRAG_LABEL); ax_b.set_ylabel("Path-based resilience [%]")
ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 100); ax_b.grid(True, alpha=0.3)
ax_b.legend(frameon=False, fontsize=5, handlelength=3.75)
ax_b.text(0.0, 1.02, "b", transform=ax_b.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", clip_on=False)
ax_b.text(0.0, 1.02, "All scenarios", transform=ax_b.transAxes + _lbl, fontsize=6, fontstyle="italic",
          va="bottom", ha="left", color="#444444", clip_on=False)

for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_heatmap_reference_marks" + ext), dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/si_heatmap_reference_marks.{{pdf,png}}", flush=True)

# ── caption numbers ─────────────────────────────────────────────────────────────────
print("\n== caption numbers ==", flush=True)
print(f"TCRE decile positions: 1.0 -> {_pos(1.0, tcre_edges, '1.0'):.2f}, "
      f"1.65 -> {_pos(1.65, tcre_edges, '1.65'):.2f}, 2.3 -> {_pos(2.3, tcre_edges, '2.3'):.2f} (axis 0..{N_DEC})", flush=True)
print(f"central-lit T_eff positions: median({cl_med:.3f}) -> {x_med:.2f}, "
      f"5%({cl_5:.3f}) -> {x_5:.2f}, 95%({cl_95:.3f}) -> {x_95:.2f}", flush=True)
print(f"posterior TCRE median = {float(tcre.median()):.3f} K/1000 PgC", flush=True)
i_row = int(np.clip(_pos(AR6["best"], tcre_edges, "best"), 0, N_DEC - 1))
j_col = int(np.clip(x_med, 0, N_DEC - 1))
print(f"NZ2050 at central estimates: cell[row{i_row},col{j_col}] PERI = {nz['pivot'][i_row, j_col] * 100:.2f}%  "
      f"| row(TCRE 1.65) PERI = {nz['row_mean'][i_row] * 100:.2f}%  "
      f"| col(T_eff med) PERI = {nz['col_mean'][j_col] * 100:.2f}%", flush=True)
