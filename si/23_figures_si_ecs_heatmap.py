"""23_figures_si_ecs_heatmap.py — SI: resilience heatmap over ECS × tipping susceptibility.

The SI twin of the main GFP×tipping heatmap, with ECS on the y-axis. Ported verbatim
from the ECS-heatmap appendix of results_nature/plot.py: heatmap + ECS (left) and
tipping-susceptibility (bottom) marginals with Wilson CIs, the GTS ensemble-density
background, the ECS very-likely range shaded (2–5 °C) and the Myhre et al. constraint
line (2.9 °C) — both shown ONLY in the ECS marginal, not on the heatmap.

Computation mirrors 16 (per-scenario RF tipping susceptibility, resilience_cr from 09,
per-sample resilience binning); only the y-axis binning variable is ECS. Conventions:
°C, "Path-based resilience". Moderate — like 16.

Outputs -> output/figures/si/ecs/Resilience_heatmap_ECS_<scenario>.{pdf,png}
"""
import os
import sys
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import matplotlib.colors as mcolors
import cmcrameri.cm as cmcs
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import (
    TWO_COL, FIG_DIR, RESILIENCE_BASE, res_x, res_y, rate_str,
    main_scenarios, get_scenario_type, load_tipping, load_df_params,
)
from tipping_params import all_bounds, sample_lhs_params

OUT_DIR = os.path.join(FIG_DIR, "si", "ecs"); os.makedirs(OUT_DIR, exist_ok=True)
MYHRE_ECS = 2.9          # updated Myhre et al. constraint (was 2.5)
VERY_LIKELY = (2.0, 5.0)  # ECS very likely range [°C]
N_BINS = 10
colormap = cmcs.batlow_r
color_left, color_bottom = cmcs.glasgow(0.1), cmcs.glasgow(0.9)

# thin-support flag — identical to the main heatmap (16): all cells show the point
# estimate; cells with n_eff = min(configs in the row, samples in the column) < 20
# are hatched rather than blanked.
HATCH_MIN_N = 20
plt.rcParams["hatch.linewidth"] = 0.4


def _hatch_low_n(ax, x_edges, y_edges, i, j):
    """Overlay a light hatch on cell (i, j) to flag n_eff < HATCH_MIN_N (matches 16)."""
    ax.add_patch(Rectangle((x_edges[j], y_edges[i]),
                           x_edges[j + 1] - x_edges[j], y_edges[i + 1] - y_edges[i],
                           fill=False, hatch="\\\\", edgecolor="0.55", linewidth=0.0, zorder=2.3))

tipping_sample_dict, _ = load_tipping()
df_params_all = load_df_params()
data_path = os.path.join(RESILIENCE_BASE, f"resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}")

df_lhs = pd.DataFrame(sample_lhs_params(n_samples=1000))
X_scaled_all = StandardScaler().fit_transform(df_lhs[list(all_bounds.keys())])

for scenario_name in tqdm(main_scenarios, desc="ECS heatmaps"):
    if scenario_name not in tipping_sample_dict:
        continue
    csv_path = os.path.join(data_path, f"resilience_{scenario_name}_time{res_x}_temp{res_y}_rate{rate_str}.csv")
    if not os.path.exists(csv_path):
        continue
    safe_name = scenario_name.replace("/", "_")
    df_resilience_sc = pd.read_csv(csv_path)
    da = tipping_sample_dict[scenario_name]["prob_any_tipping_sample"]
    n_s, n_c, n_r = da.shape

    tipping_prob_sample = da.mean(dim=["config", "run"]).values
    rf_ecs = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)
    rf_ecs.fit(X_scaled_all, tipping_prob_sample)
    tipping_susc_rf_ecs = rf_ecs.predict(X_scaled_all)

    res_pivot = df_resilience_sc.pivot_table(index="config", columns="run", values="resilience_cr")
    resilience_cr = res_pivot.reindex(index=da.config.values, columns=da.run.values).values
    df_p = df_params_all.loc[df_params_all["scenario"] == scenario_name, ["config", "run", "ecs"]]
    ecs_pivot = df_p.pivot_table(index="config", columns="run", values="ecs")
    ecs_flat = ecs_pivot.reindex(index=da.config.values, columns=da.run.values).values.ravel()

    rb_flat_e = (resilience_cr[np.newaxis] * (1 - da.values)).reshape(n_s, -1)
    del resilience_cr, res_pivot, ecs_pivot, df_p

    ecs_bin_edges = np.linspace(ecs_flat.min(), ecs_flat.max(), N_BINS + 1)
    ts_bin_edges_e = np.linspace(0.0, 1.0, N_BINS + 1)
    ecs_bin_idx = np.searchsorted(ecs_bin_edges[1:-1], ecs_flat, side="left")
    ts_bin_idx_e = np.searchsorted(ts_bin_edges_e[1:-1], tipping_susc_rf_ecs, side="left")
    n_ecs_b, n_ts_e = N_BINS, N_BINS
    ecs_bin_mids = 0.5 * (ecs_bin_edges[:-1] + ecs_bin_edges[1:])
    ts_bin_mids_e = 0.5 * (ts_bin_edges_e[:-1] + ts_bin_edges_e[1:])

    ecs_per_config = ecs_flat[::n_r]
    ecs_bin_idx_config = np.searchsorted(ecs_bin_edges[1:-1], ecs_per_config, side="left")
    cr_per_ecs_full = np.array([(ecs_bin_idx == i).sum() for i in range(n_ecs_b)])
    cr_per_ecs = np.array([(ecs_bin_idx_config == i).sum() for i in range(n_ecs_b)])
    s_per_ts_e = np.array([(ts_bin_idx_e == j).sum() for j in range(n_ts_e)])
    counts_arr_e = cr_per_ecs_full[:, np.newaxis] * s_per_ts_e[np.newaxis, :]

    rb_by_ecs = np.full((n_ecs_b, n_s), np.nan)
    for i in range(n_ecs_b):
        m = ecs_bin_idx == i
        if m.any():
            rb_by_ecs[i] = rb_flat_e[:, m].mean(axis=1)
    pivot_ecs = np.full((n_ecs_b, n_ts_e), np.nan)
    for j in range(n_ts_e):
        m = ts_bin_idx_e == j
        if m.any():
            pivot_ecs[:, j] = rb_by_ecs[:, m].mean(axis=1)

    res_by_ecs_mean = np.array([rb_flat_e[:, ecs_bin_idx == i].mean() if (ecs_bin_idx == i).any() else np.nan for i in range(n_ecs_b)])
    res_by_ts_mean_e = np.array([rb_flat_e[ts_bin_idx_e == j].mean() if (ts_bin_idx_e == j).any() else np.nan for j in range(n_ts_e)])

    ci_ecs_lo, ci_ecs_hi = np.full(n_ecs_b, np.nan), np.full(n_ecs_b, np.nan)
    for _i in range(n_ecs_b):
        _n = int(cr_per_ecs[_i])
        if _n > 0 and not np.isnan(res_by_ecs_mean[_i]):
            _lo, _hi = proportion_confint(int(round(res_by_ecs_mean[_i] * _n)), _n, alpha=0.05, method="wilson")
            ci_ecs_lo[_i] = max((res_by_ecs_mean[_i] - _lo) * 100, 0.0)
            ci_ecs_hi[_i] = max((_hi - res_by_ecs_mean[_i]) * 100, 0.0)
    ci_ts_lo_e, ci_ts_hi_e = np.full(n_ts_e, np.nan), np.full(n_ts_e, np.nan)
    for _j in range(n_ts_e):
        _n = int(s_per_ts_e[_j])
        if _n > 0 and not np.isnan(res_by_ts_mean_e[_j]):
            _lo, _hi = proportion_confint(int(round(res_by_ts_mean_e[_j] * _n)), _n, alpha=0.05, method="wilson")
            ci_ts_lo_e[_j] = max((res_by_ts_mean_e[_j] - _lo) * 100, 0.0)
            ci_ts_hi_e[_j] = max((_hi - res_by_ts_mean_e[_j]) * 100, 0.0)

    Z_ecs = pivot_ecs   # show all point estimates; thin-support cells flagged by hatch (as in 16)
    X_e, Y_e = np.meshgrid(ts_bin_edges_e, ecs_bin_edges)

    fig_e = plt.figure(figsize=(TWO_COL, TWO_COL * 8 / 11))
    gs_e = GridSpec(2, 3, width_ratios=[1.2, 4, 0.25], height_ratios=[4, 1.2], hspace=0.18, wspace=0.1)
    ax_heat_e = fig_e.add_subplot(gs_e[0, 1])
    ax_left_e = fig_e.add_subplot(gs_e[0, 0], sharey=ax_heat_e)
    ax_bottom_e = fig_e.add_subplot(gs_e[1, 1], sharex=ax_heat_e)
    ax_cbar_e = fig_e.add_subplot(gs_e[0, 2])

    norm_e = mcolors.Normalize(vmin=0, vmax=100)
    pcm_e = ax_heat_e.pcolormesh(X_e, Y_e, np.where(np.isnan(Z_ecs), np.nan, Z_ecs * 100),
                                 cmap=colormap, vmin=0, vmax=100, shading="flat")
    fig_e.colorbar(pcm_e, cax=ax_cbar_e, label="Path-based\nresilience [%]")
    for i, y in enumerate(ecs_bin_mids):
        for j, x in enumerate(ts_bin_mids_e):
            val = pivot_ecs[i, j]
            if np.isfinite(val):
                r, g, b, _ = colormap(norm_e(val * 100))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax_heat_e.text(x, y, f"{100*val:.1f}", ha="center", va="center",
                               fontsize=6, color="black" if lum > 0.45 else "white")
                if min(int(cr_per_ecs[i]), int(s_per_ts_e[j])) < HATCH_MIN_N:
                    _hatch_low_n(ax_heat_e, ts_bin_edges_e, ecs_bin_edges, i, j)
    ax_heat_e.set_xlabel(""); ax_heat_e.set_ylabel(""); ax_heat_e.tick_params(labelleft=False)
    ax_heat_e.set_xlim(ts_bin_edges_e[0], ts_bin_edges_e[-1])
    ax_heat_e.set_ylim(ecs_bin_edges[0], ecs_bin_edges[-1])

    # ── ECS marginal (left): mean ± Wilson CI, very-likely shading + Myhre line ──
    ax_left_e.plot(res_by_ecs_mean * 100, ecs_bin_mids, color=color_left, linewidth=2, label="Mean")
    ax_left_e.errorbar(res_by_ecs_mean * 100, ecs_bin_mids, xerr=[ci_ecs_lo, ci_ecs_hi], fmt="none",
                       ecolor=color_left, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_left_e.axhspan(*VERY_LIKELY, color=color_left, alpha=0.10, zorder=0,
                      label=f"Very likely ({VERY_LIKELY[0]:.0f}–{VERY_LIKELY[1]:.0f} °C)")
    #ax_left_e.axhline(MYHRE_ECS, color="black", linewidth=1.0, linestyle="--", zorder=3,
    #                  label=f"Myhre et al.\n({MYHRE_ECS} °C)")
    ax_left_e.set_ylim(ax_heat_e.get_ylim())
    ax_left_e.set_ylabel("Binned ECS [°C]")
    ax_left_e.set_xlim(100, 0); ax_left_e.margins(y=0); ax_left_e.grid(True, alpha=0.3)
    ax_left_e.legend(loc="upper left", frameon=False, fontsize=5)

    # ── tipping-susceptibility marginal (bottom): mean ± Wilson CI + GTS density ──
    ax_bottom_e.plot(ts_bin_mids_e, res_by_ts_mean_e * 100, color=color_bottom, linewidth=2, label="Mean")
    ax_bottom_e.errorbar(ts_bin_mids_e, res_by_ts_mean_e * 100, yerr=[ci_ts_lo_e, ci_ts_hi_e], fmt="none",
                         ecolor=color_bottom, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_bottom_e.set_xlabel("Tipping susceptibility\n(Binned generalised tipping parameter)")
    ax_bottom_e.set_ylim(0, 100); ax_bottom_e.margins(x=0); ax_bottom_e.grid(True, alpha=0.3)
    _hist, _ = np.histogram(tipping_susc_rf_ecs, bins=ts_bin_edges_e)
    _ax_bhe = ax_bottom_e.twinx()
    _ax_bhe.bar(ts_bin_mids_e, _hist / _hist.sum(), width=np.diff(ts_bin_edges_e).mean() * 0.88,
                color=color_bottom, alpha=0.25, zorder=0)
    _ax_bhe.set_ylim(0, 1); _ax_bhe.set_yticks([]); _ax_bhe.tick_params(right=False, which="both")
    handles_be, _ = ax_bottom_e.get_legend_handles_labels()
    ax_bottom_e.legend(handles=handles_be + [Patch(facecolor=color_bottom, alpha=0.4, label="Sample\ndensity")],
                       loc="upper right", frameon=False, fontsize=5)

    ax_heat_e.set_aspect("auto")
    plt.setp(ax_heat_e.get_xticklabels(), visible=False)
    fig_e.text(0.16, 0.185, "Path-based\nresilience [%]", ha="left", va="bottom", fontsize=7)
    _lbl = mtrans.ScaledTranslation(9 / 72, 0, fig_e.dpi_scale_trans)
    ax_left_e.text(0.0, 1.02, get_scenario_type(scenario_name), transform=ax_left_e.transAxes + _lbl,
                   fontsize=6, fontstyle="italic", va="bottom", ha="left", color="#444444", clip_on=False)

    fig_e.savefig(os.path.join(OUT_DIR, f"Resilience_heatmap_ECS_{safe_name}.pdf"), bbox_inches="tight")
    fig_e.savefig(os.path.join(OUT_DIR, f"Resilience_heatmap_ECS_{safe_name}.png"), bbox_inches="tight")
    plt.close(fig_e)
    del rb_flat_e, rb_by_ecs, ecs_flat, df_resilience_sc, rf_ecs, da
    gc.collect()

print(f"Saved ECS heatmaps to {OUT_DIR}", flush=True)
