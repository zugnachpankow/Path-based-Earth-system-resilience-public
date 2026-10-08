"""31_figures_si_demands.py — SI: path-based resilience under alternative safety demands.

Two-row version of the main-text C1 trajectory+UpSet figure for a representative scenario
(Net Zero 2050), under the two alternative criterion sets:
  (a) temperature 2.0 °C, rate 0.4 °C/decade, 2100
  (b) temperature 1.5 °C, rate 0.3 °C/decade, 2100
Each row: safe/unsafe running-mean trajectories (tipping-weighted line widths) + the UpSet
bars of the criterion combinations. Logic verbatim from 15_figures_main.py (C1 plain).

Reads output/running_mean_temps.pkl (08) + output/tipping/*.nc (07). Heavy (running-mean
pickle) -> submit via SLURM (run_31.sh).
Output -> output/figures/si/si_trajectories_demands.{pdf,png}
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import matplotlib.patheffects as pe
import cmcrameri.cm as cmcs
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import (
    TWO_COL, FIG_DIR, res_x, get_scenario_type, prob_to_lw, combo_color,
    _key_to_criterion, load_running_means, load_tipping,
)

OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
REP = "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050"
DEMANDS = [(2.0, 0.04, "2.0 °C / 0.4 °C per decade"),
           (1.5, 0.03, "1.5 °C / 0.3 °C per decade")]
MAX_TRAJ = 800

running_mean_temps = load_running_means()
_, tipping_combined = load_tipping()
rm_all = running_mean_temps[REP]
tip_prob = tipping_combined["prob_any_tipping"].sel(scenario=REP).fillna(0.0)
print(f"{REP}: rm {rm_all.sizes}", flush=True)

fig = plt.figure(figsize=(TWO_COL, TWO_COL * 0.86))
outer = GridSpec(2, 1, figure=fig, hspace=0.34)
_lbl = mtrans.ScaledTranslation(9 / 72, 0, fig.dpi_scale_trans)


def _row(row_i, res_y_c, rate_c, letter, subtitle):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row_i], width_ratios=[4, 1.4], wspace=0.38)
    ax_main = fig.add_subplot(gs[0])
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[2.5, 1], hspace=0.04)
    ax_bars, ax_matrix = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

    # ── criterion (verbatim from 15) ──
    climate_violation = (rm_all.sel(timebounds=slice(res_x, None)) > res_y_c).any("timebounds")
    roc = rm_all.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation = (roc.sel(timebounds=slice(2000, res_x)) > rate_c).any("timebounds")
    resilient = ~(climate_violation | rate_violation)
    filtered_temp = rm_all.where(resilient, drop=True)
    non_resilient_temp = rm_all.where(~resilient, drop=True)
    res_s = resilient.set_index(member=["config", "run"]).unstack("member")
    res_with_tip = res_s.astype(float) * (1.0 - tip_prob)
    no_climate = (~climate_violation).astype(float)
    no_rate = (~rate_violation).astype(float)
    no_tipping = (1.0 - tip_prob).stack(member=("run", "config"))
    combos = {
        ("C",): no_climate.mean().item(), ("R",): no_rate.mean().item(), ("T",): no_tipping.mean().item(),
        ("C", "R"): resilient.sum().item() / resilient.size,
        ("C", "T"): (no_climate * no_tipping).mean().item(),
        ("R", "T"): (no_rate * no_tipping).mean().item(),
        ("C", "R", "T"): res_with_tip.sum().item() / res_with_tip.size,
    }

    # ── trajectory panel ──
    times = non_resilient_temp.timebounds.values
    segs = [np.column_stack([times, non_resilient_temp.sel(member=m).values])
            for m in non_resilient_temp.member.values]
    ax_main.axhline(-100.0, lw=0.5, alpha=0.5, color="gray", label="Unsafe trajectories")
    if segs:
        ax_main.add_collection(LineCollection(segs, colors="gray", linewidths=0.3, alpha=0.15, rasterized=True))
    rng = np.random.default_rng(42)
    members = list(filtered_temp.member.values)
    sel = rng.permutation(len(members))[:MAX_TRAJ]
    cols = cmcs.batlow(np.linspace(0.2, 0.9, max(len(sel), 1))); rng.shuffle(cols)
    for k, i in enumerate(sel):
        m = members[i]; p = tip_prob.sel(config=m[1], run=m[0]).item()
        ax_main.plot(filtered_temp.timebounds, filtered_temp.sel(member=m),
                     color=cols[k], linewidth=(1.0 if k == 1 else prob_to_lw(p)),
                     alpha=(0.8 if k == 1 else 1.0), label="Safe trajectories" if k == 1 else None)
    if rm_all.member.size > 0:
        ax_main.plot(rm_all.timebounds, rm_all.quantile(0.5, dim="member"), color="black",
                     lw=1.6, ls="--", zorder=10, label="Ensemble median",
                     path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    t_line = np.array([1900, 2101]); T_line = res_y_c + rate_c * (t_line - 2030)
    ax_main.plot(t_line, T_line, color="black", ls=":", lw=0.8, zorder=5,
                 label=f"Rate condition\n<{rate_c*10:.1f} °C/decade")
    ax_main.axhline(0.0, color="black", lw=0.1)
    ax_main.errorbar(res_x, res_y_c / 2, yerr=res_y_c / 2, fmt="none", color="black",
                     lw=2, capsize=5, zorder=11, label=f"Temperature condition\n<{res_y_c} °C in {res_x}")
    ax_main.set_xlabel("Year"); ax_main.set_ylabel("Temperature anomaly [°C]")
    ax_main.set_xlim(1900, 2101); ax_main.set_ylim(-0.2, 4.0)
    ax_main.legend(frameon=False, loc="upper left", fontsize=5)
    ax_main.text(0.0, 1.02, letter, transform=ax_main.transAxes, fontsize=8, fontweight="bold",
                 va="bottom", ha="left", clip_on=False)
    ax_main.text(0.0, 1.02, f"{get_scenario_type(REP)} — {subtitle}", transform=ax_main.transAxes + _lbl,
                 fontsize=6, fontstyle="italic", va="bottom", ha="left", color="#444444", clip_on=False)

    # ── UpSet panel ──
    ordered = sorted(combos.items(), key=lambda x: -x[1])
    keys = [k for k, _ in ordered]; vals = np.array([v for _, v in ordered]) * 100
    criteria = ["T", "R", "C"]; n_crit, n_combos = 3, len(keys)
    pal = cmcs.glasgow_r(np.linspace(0.15, 0.85, 4))
    hatch_map = {"C": "//", "R": "\\\\", "T": "////"}
    ax_matrix.set_xlim(-0.5, n_combos - 0.5); ax_matrix.set_ylim(-0.5, n_crit - 0.5)
    for xi, key in enumerate(keys):
        active = [ci for ci, c in enumerate(criteria) if c in key]
        if len(active) > 1:
            ax_matrix.plot([xi, xi], [min(active), max(active)], color="black", lw=2.5,
                           solid_capstyle="round", zorder=1)
        for ci, c in enumerate(criteria):
            ax_matrix.add_patch(plt.Circle((xi, ci), 0.28, color="black" if c in key else "#cccccc",
                                           zorder=2, transform=ax_matrix.transData))
    ax_matrix.set_yticks(range(n_crit)); ax_matrix.set_yticklabels(["Tipping", "Rate", "Temp."], fontsize=6)
    ax_matrix.set_xticks([]); ax_matrix.tick_params(axis="y", length=0); ax_matrix.set_aspect("equal")
    for sp in ax_matrix.spines.values():
        sp.set_visible(False)
    for xi, (key, val) in enumerate(zip(keys, vals)):
        is_full = set(key) == {"C", "R", "T"}
        hatch = "xx" if is_full else (hatch_map[key[0]] if len(key) == 1 else None)
        ax_bars.bar(xi, val, color=combo_color(key, pal), edgecolor="black", linewidth=0.5,
                    hatch=hatch, width=0.7, zorder=3)
        ax_bars.text(xi, val + 0.8, f"{val:.0f}%", ha="center", va="bottom", fontsize=5,
                     fontweight="bold" if is_full else "normal")
    ax_bars.set_xlim(-0.5, n_combos - 0.5); ax_bars.set_ylim(0, 115); ax_bars.set_xticks([])
    ax_bars.set_ylabel("Path-based resilience [%]")
    ax_bars.yaxis.grid(True, linestyle="--", color="grey", alpha=0.4, linewidth=0.4)
    ax_bars.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax_bars.spines[sp].set_visible(False)
    ax_bars.legend(handles=[Patch(facecolor=pal[0], hatch="//", label="Temp.", lw=0.5),
                            Patch(facecolor=pal[1], hatch="\\\\", label="Rate", lw=0.5),
                            Patch(facecolor=pal[2], hatch="////", label="Tipping", lw=0.5),
                            Patch(facecolor=pal[3], hatch="xx", label="All", lw=0.5)],
                   loc="upper right", frameon=False, fontsize=5)


_row(0, *DEMANDS[0][:2], "a", DEMANDS[0][2])
_row(1, *DEMANDS[1][:2], "b", DEMANDS[1][2])
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_trajectories_demands" + ext), dpi=300, bbox_inches="tight")
print("Saved si_trajectories_demands.{pdf,png}", flush=True)
