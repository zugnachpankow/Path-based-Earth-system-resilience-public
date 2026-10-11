"""heatmap_common.py — shared decile-heatmap machinery for Fig. 3 (16) and its SI twins (si/23).

A heatmap cell is the path-based resilience (PERI) averaged over (configs in a y-axis
decile) × (all runs) × (samples in a T_eff decile): resilience_cr × (1 − tipping_any).
The y-axis is any scenario-independent per-config quantity (TCRE in the main figure; ECS
or the 2-var PCA score in the SI); the x-axis is always the per-sample T_eff. Both axes
are binned into equal-count deciles whose physical edges label the ticks.
"""
import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import cmcrameri.cm as cmcs
from matplotlib.patches import Rectangle
from statsmodels.stats.proportion import proportion_confint

N_DEC = 10
N_FRAG = 10
HATCH_MIN_N = 20

colormap     = cmcs.batlow_r
color_left   = cmcs.glasgow(0.1)
color_bottom = cmcs.glasgow(0.9)

fragility_edges = np.linspace(0.0, 1.0, N_FRAG + 1)
fragility_mids  = 0.5 * (fragility_edges[:-1] + fragility_edges[1:])


def build_axis(series):
    """(decile idx per key, percentile rank per key, ascending physical decile edges)."""
    dec = pd.qcut(series, N_DEC, labels=False)
    rank = series.rank(pct=True)
    edges = np.quantile(series.values, np.linspace(0, 1, N_DEC + 1))
    return dec, rank, edges


def _wilson(mean, n):
    if n > 0 and np.isfinite(mean):
        lo, hi = proportion_confint(count=int(round(mean * n)), nobs=n, alpha=0.05, method="wilson")
        return max((mean - lo) * 100, 0.0), max((hi - mean) * 100, 0.0)
    return np.nan, np.nan


def compute_scenario(da, resilience_cr, y_dec_all, y_rank_all, teff_dec_all, teff_rank_all):
    """Per-scenario heatmap dict from a tipping DataArray (sample, config, run) and the
    aligned resilience_cr (n_c, n_r). ``y_*_all`` / ``teff_*_all`` are scenario-independent
    Series indexed by config / sample respectively.
    """
    n_s, n_c, n_r = da.shape
    rb = resilience_cr[np.newaxis] * (1.0 - da.values)       # (n_s, n_c, n_r) PERI per cell
    rb_flat = rb.reshape(n_s, n_c * n_r)                     # config outer, run inner

    y_dec_cfg  = y_dec_all.reindex(da.config.values).values
    y_rank_cfg = y_rank_all.reindex(da.config.values).values
    teff_dec_smp  = teff_dec_all.reindex(da.sample.values).values
    teff_rank_smp = teff_rank_all.reindex(da.sample.values).values

    y_dec_flat  = np.repeat(y_dec_cfg, n_r)
    y_rank_flat = np.repeat(y_rank_cfg, n_r)
    teff_col = (N_DEC - 1) - teff_dec_smp                    # x: high T_eff (0) → low (9)

    cfg_per_row = np.array([(y_dec_cfg == i).sum() for i in range(N_DEC)])
    smp_per_col = np.array([(teff_col == j).sum() for j in range(N_DEC)])

    rb_by_row = np.full((N_DEC, n_s), np.nan)
    for i in range(N_DEC):
        m = (y_dec_flat == i)
        if m.any():
            rb_by_row[i] = rb_flat[:, m].mean(axis=1)
    pivot = np.full((N_DEC, N_DEC), np.nan)
    for j in range(N_DEC):
        m = (teff_col == j)
        if m.any():
            pivot[:, j] = rb_by_row[:, m].mean(axis=1)

    row_mean = np.array([rb_flat[:, y_dec_flat == i].mean() if (y_dec_flat == i).any() else np.nan
                         for i in range(N_DEC)])
    col_mean = np.array([rb_flat[teff_col == j].mean() if (teff_col == j).any() else np.nan
                         for j in range(N_DEC)])
    row_ci = np.array([_wilson(row_mean[i], int(cfg_per_row[i])) for i in range(N_DEC)]).T
    col_ci = np.array([_wilson(col_mean[j], int(smp_per_col[j])) for j in range(N_DEC)]).T

    frag_sum = np.zeros(N_FRAG); frag_cnt = np.zeros(N_FRAG, dtype=np.int64)
    for s in range(n_s):
        frag = 0.5 * y_rank_flat + 0.5 * (1.0 - teff_rank_smp[s])
        b = np.clip(np.searchsorted(fragility_edges[1:-1], frag), 0, N_FRAG - 1)
        frag_sum += np.bincount(b, weights=rb_flat[s], minlength=N_FRAG)
        frag_cnt += np.bincount(b, minlength=N_FRAG)
    frag_mean = np.where(frag_cnt > 0, frag_sum / np.where(frag_cnt > 0, frag_cnt, 1), np.nan)

    peri_overall = float(np.nanmean(rb_flat))
    cell_counts = cfg_per_row[:, None] * smp_per_col[None, :]
    peri_cells = float(np.nansum(pivot * cell_counts) / cell_counts[np.isfinite(pivot)].sum())
    peri_frag = float(np.nansum(frag_mean * frag_cnt) / frag_cnt[np.isfinite(frag_mean)].sum())

    return dict(pivot=pivot, row_mean=row_mean, col_mean=col_mean, row_ci=row_ci, col_ci=col_ci,
                cfg_per_row=cfg_per_row, smp_per_col=smp_per_col, frag_mean=frag_mean, frag_cnt=frag_cnt,
                peri_overall=peri_overall, peri_cells=peri_cells, peri_frag=peri_frag)


def _hatch(ax, i, j):
    ax.add_patch(Rectangle((j, i), 1, 1, fill=False, hatch="\\\\",
                           edgecolor="0.55", linewidth=0.0, zorder=2.3))


def draw_heatmap(ax_heat, ax_left, ax_bottom, ax_cbar, d, fig,
                 y_edges, teff_edges_desc, y_label, x_label, cell_fs=6):
    """Heatmap + left (y) / bottom (T_eff) marginals from a compute_scenario() dict."""
    import matplotlib.pyplot as plt
    norm = mcolors.Normalize(vmin=0, vmax=100)
    Z = np.where(np.isfinite(d["pivot"]), d["pivot"] * 100, np.nan)
    X, Y = np.meshgrid(np.arange(N_DEC + 1), np.arange(N_DEC + 1))
    pcm = ax_heat.pcolormesh(X, Y, Z, cmap=colormap, vmin=0, vmax=100, shading="flat")
    if ax_cbar is not None:
        fig.colorbar(pcm, cax=ax_cbar, label="Average path-based\nresilience [%]")

    for i in range(N_DEC):
        for j in range(N_DEC):
            val = d["pivot"][i, j]
            if np.isfinite(val):
                if int(min(d["cfg_per_row"][i], d["smp_per_col"][j])) < HATCH_MIN_N:
                    _hatch(ax_heat, i, j)
                r, g, b, _ = colormap(norm(val * 100))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax_heat.text(j + 0.5, i + 0.5, f"{100.0 * val:.1f}", ha="center", va="center",
                             fontsize=cell_fs, color="black" if lum > 0.45 else "white")

    ax_heat.set_xlim(0, N_DEC); ax_heat.set_ylim(0, N_DEC); ax_heat.set_aspect("auto")
    ax_heat.set_xticks(np.arange(N_DEC + 1)); ax_heat.set_yticks(np.arange(N_DEC + 1))
    ax_heat.set_xticklabels([f"{v:.2f}" for v in teff_edges_desc], fontsize=5, rotation=90)
    ax_heat.tick_params(labelleft=False)

    yc = np.arange(N_DEC) + 0.5
    ax_left.plot(d["row_mean"] * 100, yc, color=color_left, linewidth=2, label="Mean")
    ax_left.errorbar(d["row_mean"] * 100, yc, xerr=d["row_ci"], fmt="none",
                     ecolor=color_left, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_left.set_ylim(0, N_DEC); ax_left.set_xlim(100, 0); ax_left.margins(y=0)
    ax_left.set_yticks(np.arange(N_DEC + 1)); ax_left.set_yticklabels([f"{v:.2f}" for v in y_edges], fontsize=5)
    ax_left.set_ylabel(y_label); ax_left.grid(True, alpha=0.3); ax_left.set_axisbelow(True)

    xc = np.arange(N_DEC) + 0.5
    ax_bottom.plot(xc, d["col_mean"] * 100, color=color_bottom, linewidth=2, label="Mean")
    ax_bottom.errorbar(xc, d["col_mean"] * 100, yerr=d["col_ci"], fmt="none",
                       ecolor=color_bottom, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_bottom.set_xlim(0, N_DEC); ax_bottom.set_ylim(0, 100); ax_bottom.margins(x=0)
    ax_bottom.set_xticks(np.arange(N_DEC + 1))
    ax_bottom.set_xticklabels([f"{v:.2f}" for v in teff_edges_desc], fontsize=5, rotation=90)
    ax_bottom.set_xlabel(x_label); ax_bottom.grid(True, alpha=0.3); ax_bottom.set_axisbelow(True)
    plt.setp(ax_heat.get_xticklabels(), visible=False)
    ax_heat.tick_params(axis="x", labelbottom=False)
