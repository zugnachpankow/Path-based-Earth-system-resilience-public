"""37_figures_si_upset.py — SI: UpSet decomposition of path-based resilience for four
scenarios (2 × 2), from the 09b bootstrap output (no new computation).

Each panel = bars (top) + UpSet matrix (bottom), exactly as Fig. 2b (same bar order =
descending value, colours via combo_color, matrix dots/lines, CI whiskers, the all-three
"final" bar rightmost with xx-hatch/bold). Bars + CIs are read straight from
output/bootstrap_upset/upset_bootstrap_combined_B1000.csv (09b).

Panels: a Net Zero 2050, b SSP1-1.9, c SSP1-2.6, d SSP5-3.4-OS.
Output -> figures/si/si_upset_four_scenarios.{pdf,png}
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import cmcrameri.cm as cmcs
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import (
    TWO_COL, FIG_DIR, apply_style, get_scenario_type, combo_color, _key_to_criterion,
    RESILIENCE_BASE, res_x, res_y, rate_str,
)

apply_style()
OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
CSV = "output/bootstrap_upset/upset_bootstrap_combined_B1000.csv"

SCENARIOS = [  # (scenario key, panel letter, left-column?)
    ("REMIND-MAgPIE_3.3-4.8___Net_Zero_2050", "a", True),
    ("ssp119", "b", False),
    ("ssp126", "c", True),
    ("ssp534-over", "d", False),
]
CRITERIA = ["T", "R", "C"]                        # matrix rows y=0,1,2 (Tipping, Rate, Temperature)
ROW_LABELS = ["Tipping", "Rate", "Temperature"]
PAL = cmcs.glasgow_r(np.linspace(0.15, 0.85, 4))
HATCH = {"C": "//", "R": "\\\\", "T": "////"}

df = pd.read_csv(CSV)


def _panel_data(sc):
    """{key: point%}, {key: (ci_low%, ci_high%)} from the 09b CSV for one scenario."""
    sub = df[df["scenario"] == sc].set_index("criterion")
    combos = {k: float(sub.loc[c, "point_est"]) for k, c in _key_to_criterion.items()}
    cis = {k: (float(sub.loc[c, "ci_low"]), float(sub.loc[c, "ci_high"])) for k, c in _key_to_criterion.items()}
    return combos, cis


def _bar_label(v):
    """Value label (v in %): exactly 0 -> '0', 0<v<0.05% -> '<0.1', else one decimal."""
    if v == 0.0:
        return "0"
    if v < 0.05:
        return "<0.1"
    return f"{v:.1f}"


def draw_panel(ax_bars, ax_matrix, combos, cis, letter, scenario, show_ticklabels, show_rowlabels,
               lbl_off, legend=False):
    ordered = sorted(combos.items(), key=lambda x: -x[1])   # descending -> all-three rightmost
    keys = [k for k, _ in ordered]
    vals = np.array([v for _, v in ordered])
    n_combos = len(keys)

    # ── matrix (dots + connecting line) — only where an axis is given (lower row) ──
    # shares x with ax_bars (caller) so every combination sits exactly under its bar;
    # dots via scatter (round via marker size, no equal-aspect that would misalign cols).
    if ax_matrix is not None:
        ax_matrix.set_ylim(-0.5, len(CRITERIA) - 0.5)
        for xi, key in enumerate(keys):
            active = [ci for ci, c in enumerate(CRITERIA) if c in key]
            if len(active) > 1:
                ax_matrix.plot([xi, xi], [min(active), max(active)], color="black", linewidth=2.0,
                               solid_capstyle="round", zorder=1)
            ax_matrix.scatter([xi] * len(CRITERIA), range(len(CRITERIA)), s=55,
                              c=["black" if c in key else "#cccccc" for c in CRITERIA],
                              edgecolors="none", zorder=2)
        ax_matrix.set_yticks(range(len(CRITERIA)))
        ax_matrix.set_yticklabels(ROW_LABELS if show_rowlabels else [], fontsize=6)
        ax_matrix.set_xticks([]); ax_matrix.tick_params(axis="y", length=0)
        for sp in ax_matrix.spines.values():
            sp.set_visible(False)

    # ── bars + CI whiskers + value labels ──
    for xi, (key, val) in enumerate(zip(keys, vals)):
        is_full = set(key) == {"C", "R", "T"}
        hatch = "xx" if is_full else (HATCH[key[0]] if len(key) == 1 else None)
        ax_bars.bar(xi, val, color=combo_color(key, PAL), edgecolor="black",
                    linewidth=0.5, hatch=hatch, width=0.7, zorder=3)
        lo, hi = cis[key]
        ax_bars.errorbar(xi, val, yerr=[[max(val - lo, 0)], [max(hi - val, 0)]], fmt="none",
                         color="black", linewidth=0.8, capsize=2, zorder=4)
        ax_bars.text(xi, min(hi + 1.0, 99.0), _bar_label(val), ha="center", va="bottom",
                     fontsize=5, fontweight="bold" if is_full else "normal", clip_on=False)
    ax_bars.set_xlim(-0.5, n_combos - 0.5); ax_bars.set_ylim(0, 100); ax_bars.set_xticks([])
    if not show_ticklabels:
        ax_bars.tick_params(labelleft=False)
    ax_bars.yaxis.grid(True, linestyle="--", color="grey", alpha=0.4, linewidth=0.4)
    ax_bars.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_bars.spines[sp].set_visible(False)
    if legend:
        ax_bars.legend(handles=[Patch(facecolor=PAL[0], hatch="//", label="Temperature", lw=0.5),
                                Patch(facecolor=PAL[1], hatch="\\\\", label="Rate", lw=0.5),
                                Patch(facecolor=PAL[2], hatch="////", label="Tipping", lw=0.5),
                                Patch(facecolor=PAL[3], hatch="xx", label="All", lw=0.5)],
                       loc="upper right", frameon=False, fontsize=4.5, ncol=2, handlelength=1.5)
    # panel label: bold letter + italic grey scenario name (Fig. 3 style)
    ax_bars.text(0.0, 1.02, letter, transform=ax_bars.transAxes, fontsize=8, fontweight="bold",
                 va="bottom", ha="left", clip_on=False)
    ax_bars.text(0.0, 1.02, get_scenario_type(scenario), transform=ax_bars.transAxes + lbl_off,
                 fontsize=6, fontstyle="italic", va="bottom", ha="left", color="#444444", clip_on=False)


fig = plt.figure(figsize=(TWO_COL, TWO_COL * 0.72))
_lbl_off = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)
# top row = bars only, bottom row = bars + matrix; height_ratios keep bar heights equal
outer = GridSpec(2, 2, figure=fig, height_ratios=[4.0, 5.35], hspace=0.16, wspace=0.12)
panel_cache = {}
for idx, (sc, letter, left) in enumerate(SCENARIOS):
    r, c = divmod(idx, 2)
    combos, cis = _panel_data(sc)
    panel_cache[sc] = (combos, cis)
    if r == 0:                       # upper row: bars only, no UpSet matrix
        ax_bars = fig.add_subplot(outer[r, c])
        draw_panel(ax_bars, None, combos, cis, letter, sc, show_ticklabels=left,
                   show_rowlabels=False, lbl_off=_lbl_off, legend=(idx == 0))   # legend in panel a
    else:                            # lower row: bars + matrix (cols aligned under bars via sharex;
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[r, c], height_ratios=[4, 1.35], hspace=0.08)
        ax_bars = fig.add_subplot(inner[0])
        ax_matrix = fig.add_subplot(inner[1], sharex=ax_bars)       # share x -> each combo under its bar
        draw_panel(ax_bars, ax_matrix, combos, cis, letter, sc, show_ticklabels=left,
                   show_rowlabels=left, lbl_off=_lbl_off)            # row labels shared on the left (c)
fig.supylabel("Path-based resilience [%]", fontsize=7, x=0.03)   # one shared y-label

for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_upset_four_scenarios" + ext), dpi=300, bbox_inches="tight")
print(f"Saved {OUT_DIR}/si_upset_four_scenarios.{{pdf,png}}", flush=True)

# ── checks ──────────────────────────────────────────────────────────────────────
summary = pd.read_csv(f"{RESILIENCE_BASE}/resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}/"
                      f"summary/resilience_summary_time{res_x}_temp{res_y}_rate{rate_str}.csv").set_index("scenario")["resilience"]
print("\n== check: all-three (C+R+T) bar vs 09 summary PERI ==", flush=True)
for sc, _, _ in SCENARIOS:
    combos, _ = panel_cache[sc]
    full = combos[("C", "R", "T")]
    s09 = float(summary.get(sc, np.nan)) * 100
    print(f"  {get_scenario_type(sc):14s}: UpSet C+R+T={full:.2f}%  09 summary={s09:.2f}%  diff={full - s09:+.3f} pp", flush=True)

print("\n== for the text ==", flush=True)
for sc in ("ssp119", "ssp126"):
    combos, cis = panel_cache[sc]
    c = combos[("C",)]; lo, hi = cis[("C",)]
    print(f"  {get_scenario_type(sc)} temperature-only (C) = {c:.1f}% [{lo:.1f}, {hi:.1f}]", flush=True)

nz = "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050"
combos, cis = panel_cache[nz]
print(f"  Net Zero 2050 single-constraint:", flush=True)
for key, name in [(("C",), "Temperature"), (("R",), "Rate"), (("T",), "Tipping")]:
    v = combos[key]; lo, hi = cis[key]
    print(f"      {name:12s} = {v:.1f}% [{lo:.1f}, {hi:.1f}]", flush=True)
cT = combos[("C",)]
del_tip = cT - combos[("C", "T")]      # extra removal by tipping beyond temperature
del_rate = cT - combos[("C", "R")]     # extra removal by rate beyond temperature
print(f"      extra removal by TIPPING beyond temperature (C - C+T) = {del_tip:.1f} pp "
      f"[C {cT:.1f} ({cis[('C',)][0]:.1f},{cis[('C',)][1]:.1f}) ; C+T {combos[('C','T')]:.1f} "
      f"({cis[('C','T')][0]:.1f},{cis[('C','T')][1]:.1f})]", flush=True)
print(f"      extra removal by RATE    beyond temperature (C - C+R) = {del_rate:.1f} pp "
      f"[C {cT:.1f} ({cis[('C',)][0]:.1f},{cis[('C',)][1]:.1f}) ; C+R {combos[('C','R')]:.1f} "
      f"({cis[('C','R')][0]:.1f},{cis[('C','R')][1]:.1f})]", flush=True)
