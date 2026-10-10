"""15_figures_heatmap.py — GFP × Tipping resilience heatmaps + fragility.

Ported from the canonical results_nature/plot.py heatmap loop (main-figure parts
only; the dev/diagnostic sub-figures — RF_diagnostic, plateau, GFP-drop, coverage —
are intentionally omitted). Consumes the cleaned pipeline's outputs:
  * tipping samples -> output/tipping/*.nc               (07)
  * GFP per config  -> output/feedback/df_params_all.pkl (11)
  * resilience CR   -> output/resilience/.../*.csv        (09)
  * LHS tipping pars-> src/tipping_params (seed 1234, n=1000; matches src/tipping.py)

Produces (under output/figures/heatmaps):
  * Resilience_heatmap_<scenario>.{pdf,png}   — per main scenario
  * resilience_fragility_diagonal_scenarios.{pdf,png}
  * heatmap_nz2050_fragility_combined.{pdf,png}   — the combined main figure
  * combined_cache.pkl, median_ci_table.md

The RF (per-scenario tipping susceptibility) and PCA tipping_susc are recomputed
inline, exactly as in plot.py. Heavy -> submit via SLURM.
"""
import os
import sys
import gc
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import cmcrameri.cm as cmcs
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from statsmodels.stats.proportion import proportion_confint

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from figures_common import (
    TWO_COL, FIG_DIR, RESILIENCE_BASE, res_x, res_y, rate, rate_str,
    main_scenarios, get_scenario_type, build_naming, composite_styles,
    load_tipping, load_df_params,
)
from tipping_params import all_bounds, sample_lhs_params
from pca_orient import orient_by_correlation

HEATMAP_DIR = os.path.join(FIG_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

# ── load pre-computed pipeline outputs ─────────────────────────────────────────
tipping_sample_dict, _ = load_tipping()
df_params_all = load_df_params()
fancy_titles, _colors, _linestyles, _color_map = build_naming(
    df_params_all["scenario"].unique().tolist()
)
data_path = os.path.join(
    RESILIENCE_BASE, f"resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}"
)
print(f"Loaded tipping ({len(tipping_sample_dict)}) + df_params_all {df_params_all.shape}", flush=True)

# ── LHS tipping params + PCA tipping susceptibility (verbatim) ──────────────────
params = sample_lhs_params(n_samples=1000)          # seed 1234 — matches src/tipping.py
df_lhs = pd.DataFrame(params)
tipping_columns = list(all_bounds.keys())           # 16 params, param->pf->strength order

scaler_tr = StandardScaler()
X_tr_scaled = scaler_tr.fit_transform(df_lhs[tipping_columns])
pca_tr = PCA(n_components=1)
tipping_susc_pc = pca_tr.fit_transform(X_tr_scaled).flatten()
# orient so tipping susceptibility correlates positively with the mean tipping
# probability per LHS sample (averaged over scenarios/config/run), deterministically
_tip_per_sample = np.nanmean(
    [tipping_sample_dict[sc]['prob_any_tipping_sample'].mean(dim=['config', 'run']).values
     for sc in sorted(tipping_sample_dict)],
    axis=0,
)
tipping_susc_pc = orient_by_correlation(tipping_susc_pc, _tip_per_sample)
tipping_susc_scaled = (tipping_susc_pc - tipping_susc_pc.min()) / (tipping_susc_pc.max() - tipping_susc_pc.min())
df_lhs["tipping_susc"] = tipping_susc_scaled
df_lhs_with_sample = df_lhs.reset_index().rename(columns={'index': 'sample'})

# Pre-scale LHS params once — reused for every RF fit
params_all = df_lhs[list(all_bounds.keys())]
scaler_rf = StandardScaler()
X_scaled_all = scaler_rf.fit_transform(params_all)

n_bins_gf = 10
n_bins_ts = 10
n_fragility_bins = 10

# All cells show the point-estimate resilience [%] (also the colour). A cell's effective
# sample size is the binding dimension, n_eff = min(configs in the row, samples in the
# column); cells with n_eff < HATCH_MIN_N are hatched to flag thin support, so an extreme
# like 0/100 off few configs reads as indicative rather than exact.
HATCH_MIN_N = 20
plt.rcParams["hatch.linewidth"] = 0.4   # thin hatch lines (less prominent)


def _cell_label(val):
    return f"{100.0 * val:.1f}"


def _hatch_low_n(ax, x_edges, y_edges, i, j):
    """Overlay a light hatch on cell (i, j) to flag n_eff < HATCH_MIN_N.
    Backslash direction is perpendicular to the fragility diagonal, to avoid clutter."""
    ax.add_patch(Rectangle((x_edges[j], y_edges[i]),
                           x_edges[j + 1] - x_edges[j], y_edges[i + 1] - y_edges[i],
                           fill=False, hatch="\\\\", edgecolor="0.55", linewidth=0.0, zorder=2.3))

colormap = cmcs.batlow_r
color_left   = cmcs.glasgow(0.1)
color_bottom = cmcs.glasgow(0.9)

fragility_results = {}
ci_table_rows = []
nz2050_data = {}

# ══════════════════════════════════════════════════════════════════════════════
# Heatmap loop (main-figure parts, verbatim)
# ══════════════════════════════════════════════════════════════════════════════
for scenario_name in tqdm(main_scenarios, desc="Heatmaps"):
    safe_name = scenario_name.replace("/", "_")
    if scenario_name not in tipping_sample_dict:
        print(f"  Skipping {scenario_name}: no tipping data", flush=True)
        continue

    csv_path = os.path.join(
        data_path, f"resilience_{scenario_name}_time{res_x}_temp{res_y}_rate{rate_str}.csv"
    )
    if not os.path.exists(csv_path):
        print(f"  Skipping {scenario_name}: no resilience CSV", flush=True)
        continue

    df_resilience_sc = pd.read_csv(csv_path)
    da = tipping_sample_dict[scenario_name]['prob_any_tipping_sample']

    # Per-scenario RF: mean tipping prob per sample ~ LHS params
    tipping_prob_sample = da.mean(dim=['config', 'run']).values  # (n_s,)
    rf_all = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)
    rf_all.fit(X_scaled_all, tipping_prob_sample)
    y_rf_all_pred = rf_all.predict(X_scaled_all)
    r2 = r2_score(tipping_prob_sample, y_rf_all_pred)
    print(f"  R² RF {scenario_name}: {r2:.3f}", flush=True)

    tipping_susc_rf = y_rf_all_pred
    df_lhs_with_sample["tipping_susc_rf"] = tipping_susc_rf

    n_s, n_c, n_r = da.shape

    # Align resilience_cr to da's (config, run) order → (n_c, n_r)
    res_pivot = df_resilience_sc.pivot_table(index='config', columns='run', values='resilience_cr')
    resilience_cr = res_pivot.reindex(index=da.config.values, columns=da.run.values).values
    del res_pivot

    # Align general_feedback to da's (config, run) order → (n_c*n_r,)
    df_p = df_params_all.loc[
        df_params_all['scenario'] == scenario_name, ['config', 'run', 'general_feedback']
    ]
    gf_pivot = df_p.pivot_table(index='config', columns='run', values='general_feedback')
    gf_flat = gf_pivot.reindex(index=da.config.values, columns=da.run.values).values.ravel()
    del df_p, gf_pivot

    # resilience_binary[s, c, r] = resilience_cr[c,r] * (1 - tipping_any[s,c,r])
    rb = resilience_cr[np.newaxis] * (1 - da.values)   # (n_s, n_c, n_r)
    rb_flat = rb.reshape(n_s, -1)                       # (n_s, n_c*n_r)
    del rb, resilience_cr

    # Binning — both axes fixed [0,1] with 0.1-width bins
    gf_norm = gf_flat  # general_feedback already [0,1] per-scenario from PCA step
    gf_bin_edges = np.linspace(0.0, 1.0, n_bins_gf + 1)
    ts_bin_edges = np.linspace(0.0, 1.0, n_bins_ts + 1)

    gf_bin_idx = np.searchsorted(gf_bin_edges[1:-1], gf_norm, side='left')
    gf_per_config = gf_flat[::n_r]  # one GFP value per config (n_c,)
    gf_bin_idx_config = np.searchsorted(gf_bin_edges[1:-1], gf_per_config, side='left')
    ts_bin_idx = np.searchsorted(ts_bin_edges[1:-1], tipping_susc_rf, side='left')

    n_gf = len(gf_bin_edges) - 1
    n_ts = len(ts_bin_edges) - 1
    gf_bin_mids = 0.5 * (gf_bin_edges[:-1] + gf_bin_edges[1:])
    ts_bin_mids = 0.5 * (ts_bin_edges[:-1] + ts_bin_edges[1:])

    cr_per_gf_full = np.array([(gf_bin_idx == i).sum() for i in range(n_gf)])
    s_per_ts       = np.array([(ts_bin_idx == j).sum() for j in range(n_ts)])
    counts_arr     = cr_per_gf_full[:, np.newaxis] * s_per_ts[np.newaxis, :]
    cr_per_gf      = np.array([(gf_bin_idx_config == i).sum() for i in range(n_gf)])

    # Pivot: mean resilience_binary per (gf_bin, ts_bin)
    rb_by_gf = np.full((n_gf, n_s), np.nan)
    for i in range(n_gf):
        mask = (gf_bin_idx == i)
        if mask.any():
            rb_by_gf[i] = rb_flat[:, mask].mean(axis=1)

    pivot_arr = np.full((n_gf, n_ts), np.nan)
    for j in range(n_ts):
        mask = (ts_bin_idx == j)
        if mask.any():
            pivot_arr[:, j] = rb_by_gf[:, mask].mean(axis=1)

    # Marginals
    res_by_gf_mean = np.array([rb_flat[:, gf_bin_idx == i].mean() if (gf_bin_idx == i).any() else np.nan for i in range(n_gf)])
    res_by_gf_std  = np.array([rb_flat[:, gf_bin_idx == i].std()  if (gf_bin_idx == i).any() else np.nan for i in range(n_gf)])
    res_by_ts_mean = np.array([rb_flat[ts_bin_idx == j].mean()    if (ts_bin_idx == j).any() else np.nan for j in range(n_ts)])
    res_by_ts_std  = np.array([rb_flat[ts_bin_idx == j].std()     if (ts_bin_idx == j).any() else np.nan for j in range(n_ts)])

    # Binomial CIs (heatmap cells, for CI table only)
    se      = np.sqrt(pivot_arr * (1 - pivot_arr) / np.where(counts_arr > 0, counts_arr, np.nan))
    ci_low  = pivot_arr - 1.96 * se
    ci_high = pivot_arr + 1.96 * se

    # Wilson CIs for marginals — corrected n_eff (no pseudoreplication)
    ci_gf_lo, ci_gf_hi = np.full(n_gf, np.nan), np.full(n_gf, np.nan)
    for _i in range(n_gf):
        _n = int(cr_per_gf[_i])
        if _n > 0 and not np.isnan(res_by_gf_mean[_i]):
            _lo, _hi = proportion_confint(count=int(round(res_by_gf_mean[_i] * _n)), nobs=_n,
                                          alpha=0.05, method='wilson')
            ci_gf_lo[_i] = max((res_by_gf_mean[_i] - _lo) * 100, 0.0)
            ci_gf_hi[_i] = max((_hi - res_by_gf_mean[_i]) * 100, 0.0)
    ci_ts_lo, ci_ts_hi = np.full(n_ts, np.nan), np.full(n_ts, np.nan)
    for _j in range(n_ts):
        _n = int(s_per_ts[_j])
        if _n > 0 and not np.isnan(res_by_ts_mean[_j]):
            _lo, _hi = proportion_confint(count=int(round(res_by_ts_mean[_j] * _n)), nobs=_n,
                                          alpha=0.05, method='wilson')
            ci_ts_lo[_j] = max((res_by_ts_mean[_j] - _lo) * 100, 0.0)
            ci_ts_hi[_j] = max((_hi - res_by_ts_mean[_j]) * 100, 0.0)

    # ── Heatmap figure (verbatim) ──────────────────────────────────────────────
    Z = np.where(counts_arr < 100, np.nan, pivot_arr)
    X, Y = np.meshgrid(ts_bin_edges, gf_bin_edges)

    fig = plt.figure(figsize=(TWO_COL, TWO_COL * 8 / 11))
    gs = GridSpec(nrows=2, ncols=3, width_ratios=[1.2, 4, 0.25], height_ratios=[4, 1.2], hspace=0.18, wspace=0.1)
    ax_heat   = fig.add_subplot(gs[0, 1])
    ax_left   = fig.add_subplot(gs[0, 0], sharey=ax_heat)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_heat)
    ax_cbar   = fig.add_subplot(gs[0, 2])

    norm_cm = mcolors.Normalize(vmin=0, vmax=100)
    Z_pct = np.where(np.isnan(Z), np.nan, Z * 100)
    ci_width = ci_high - ci_low
    median_ci_pct = np.nanmedian(ci_width) * 100
    pcm = ax_heat.pcolormesh(X, Y, Z_pct, cmap=colormap, vmin=0, vmax=100, shading="flat")
    fig.colorbar(pcm, cax=ax_cbar, label="Average path-based\nresilience [%]")

    for i, y in enumerate(gf_bin_mids):
        for j, x in enumerate(ts_bin_mids):
            val = pivot_arr[i, j]
            if np.isfinite(val):
                n_eff = int(min(cr_per_gf[i], s_per_ts[j]))
                if n_eff < HATCH_MIN_N:
                    _hatch_low_n(ax_heat, ts_bin_edges, gf_bin_edges, i, j)
                r, g, b, _ = colormap(norm_cm(val * 100))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax_heat.text(x, y, _cell_label(val), ha="center", va="center",
                             fontsize=6, color="black" if lum > 0.45 else "white")

    ax_heat.set_xlabel(""); ax_heat.set_ylabel("")
    ax_heat.tick_params(labelleft=False)
    ax_heat.set_xlim(ts_bin_edges[0], ts_bin_edges[-1])
    ax_heat.set_ylim(gf_bin_edges[0], gf_bin_edges[-1])

    ax_left.plot(res_by_gf_mean * 100, gf_bin_mids, color=color_left, linewidth=2, label="Mean")
    ax_left.errorbar(res_by_gf_mean * 100, gf_bin_mids, xerr=[ci_gf_lo, ci_gf_hi], fmt='none',
                     ecolor=color_left, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_left.set_ylim(ax_heat.get_ylim())
    ax_left.set_ylabel("Feedback intensity\n(Binned generalised climate feedback parameter)")
    ax_left.set_axisbelow(True)
    ax_left.grid(True, alpha=0.3)
    _gf_hist_counts, _ = np.histogram(gf_per_config, bins=gf_bin_edges)
    _gf_hist_norm = _gf_hist_counts / _gf_hist_counts.sum()
    _ax_lh = ax_left.twiny()
    _ax_lh.barh(gf_bin_mids, _gf_hist_norm, height=np.diff(gf_bin_edges).mean() * 0.88,
                color=color_left, alpha=0.35, zorder=1)
    _ax_lh.set_xlim(0, 1); _ax_lh.set_xticks([])
    _ax_lh.tick_params(top=False, labeltop=False, which='both'); _ax_lh.grid(False)
    _hist_handle_gf = Patch(facecolor=color_left, alpha=0.4, label="Config\ndensity")
    handles_l, _ = ax_left.get_legend_handles_labels()
    ax_left.legend(handles=handles_l + [_hist_handle_gf], loc="upper left", frameon=False)

    ax_bottom.set_axisbelow(True)
    ax_bottom.plot(ts_bin_mids, res_by_ts_mean * 100, color=color_bottom, linewidth=2, label="Mean")
    ax_bottom.errorbar(ts_bin_mids, res_by_ts_mean * 100, yerr=[ci_ts_lo, ci_ts_hi], fmt='none',
                       ecolor=color_bottom, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_bottom.set_xlabel("Tipping susceptibility\n(Binned generalised tipping parameter)")
    ax_bottom.grid(True, alpha=0.3)
    _ts_hist_counts, _ = np.histogram(tipping_susc_rf, bins=ts_bin_edges)
    _ts_hist_norm = _ts_hist_counts / _ts_hist_counts.sum()
    _ax_bh = ax_bottom.twinx()
    _ax_bh.bar(ts_bin_mids, _ts_hist_norm, width=np.diff(ts_bin_edges).mean() * 0.88,
               color=color_bottom, alpha=0.35, zorder=1)
    _ax_bh.set_ylim(0, 1); _ax_bh.set_yticks([])
    _ax_bh.tick_params(right=False, labelright=False, which='both'); _ax_bh.grid(False)
    _hist_handle_ts = Patch(facecolor=color_bottom, alpha=0.4, label="Sample density")
    handles_b, _ = ax_bottom.get_legend_handles_labels()
    ax_bottom.legend(handles=handles_b + [_hist_handle_ts], loc="upper right", frameon=False)

    ax_left.set_xlim(100, 0)
    ax_bottom.set_ylim(0, 100)
    ax_left.margins(y=0); ax_bottom.margins(x=0)
    ax_heat.set_aspect("auto")
    plt.setp(ax_heat.get_xticklabels(), visible=False)
    fig.text(0.16, 0.185, "Path-based\nresilience [%]", ha="left", va="bottom", fontsize=7)
    ax_heat.set_title(get_scenario_type(scenario_name), fontsize=7, pad=3)
    ci_table_rows.append({'scenario': get_scenario_type(scenario_name),
                          'median_95ci_width_pp': round(median_ci_pct, 2)})
    #"Cell values are the point-estimate resilience [%]; "
    #"cells with n = min(configs, samples) < 20 are hatched (thin support)."
    fig.savefig(f"{HEATMAP_DIR}/Resilience_heatmap_{safe_name}.pdf", bbox_inches='tight')
    fig.savefig(f"{HEATMAP_DIR}/Resilience_heatmap_{safe_name}.png", bbox_inches='tight')
    plt.close(fig)

    # ── fragility score (per-sample, verbatim) ─────────────────────────────────
    gf_norm_arr = gf_norm
    ts_norm_arr = (tipping_susc_rf - tipping_susc_rf.min()) / (tipping_susc_rf.max() - tipping_susc_rf.min())
    fragility_edges = np.linspace(0.0, 1.0, n_fragility_bins + 1)
    fragility_mids  = 0.5 * (fragility_edges[:-1] + fragility_edges[1:])
    rb_fragility_sum   = np.zeros(n_fragility_bins)
    rb_fragility_sq    = np.zeros(n_fragility_bins)
    rb_fragility_count = np.zeros(n_fragility_bins, dtype=np.int64)
    for s_idx in range(n_s):
        fragility_s = 0.5 * ts_norm_arr[s_idx] + 0.5 * gf_norm_arr
        bin_s = np.clip(np.searchsorted(fragility_edges[1:-1], fragility_s), 0, n_fragility_bins - 1)
        rb_s = rb_flat[s_idx]
        rb_fragility_sum   += np.bincount(bin_s, weights=rb_s,    minlength=n_fragility_bins)
        rb_fragility_sq    += np.bincount(bin_s, weights=rb_s**2, minlength=n_fragility_bins)
        rb_fragility_count += np.bincount(bin_s,                  minlength=n_fragility_bins)
    valid = rb_fragility_count > 0
    fragility_mean = np.where(valid, rb_fragility_sum / rb_fragility_count, np.nan)
    fragility_std  = np.where(valid, np.sqrt(np.maximum(rb_fragility_sq / rb_fragility_count - fragility_mean**2, 0)), np.nan)
    fragility_results[scenario_name] = pd.DataFrame({
        'fragility_mid': fragility_mids, 'mean': fragility_mean,
        'std': fragility_std, 'count': rb_fragility_count,
    })

    if scenario_name == 'REMIND-MAgPIE_3.3-4.8___Net_Zero_2050':
        nz2050_data = dict(
            pivot_arr=pivot_arr.copy(),
            gf_bin_mids=gf_bin_mids.copy(), ts_bin_mids=ts_bin_mids.copy(),
            gf_bin_edges=gf_bin_edges.copy(), ts_bin_edges=ts_bin_edges.copy(),
            res_by_gf_mean=res_by_gf_mean.copy(), res_by_gf_std=res_by_gf_std.copy(),
            res_by_ts_mean=res_by_ts_mean.copy(), res_by_ts_std=res_by_ts_std.copy(),
            ci_gf_lo=ci_gf_lo.copy(), ci_gf_hi=ci_gf_hi.copy(),
            ci_ts_lo=ci_ts_lo.copy(), ci_ts_hi=ci_ts_hi.copy(),
            Z=Z.copy(), gf_per_config=gf_per_config.copy(),
            tipping_susc_rf=tipping_susc_rf.copy(),
            cr_per_gf=cr_per_gf.copy(), s_per_ts=s_per_ts.copy(),
        )

    del rb_flat, rb_by_gf, gf_flat, gf_norm, gf_norm_arr, df_resilience_sc, rf_all
    gc.collect()

print(f"Heatmap loop complete. Processed {len(fragility_results)} scenarios.", flush=True)

with open(os.path.join(HEATMAP_DIR, "combined_cache.pkl"), "wb") as _f:
    pickle.dump({"nz2050_data": nz2050_data, "fragility_results": fragility_results}, _f)
if ci_table_rows:
    pd.DataFrame(ci_table_rows).to_markdown(os.path.join(HEATMAP_DIR, "median_ci_table.md"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Fragility diagonal across scenarios (verbatim)
# ══════════════════════════════════════════════════════════════════════════════
# shared (color, linestyle) per scenario — reused by the ECS histogram (20) etc.
_styles = composite_styles(main_scenarios)
fig, ax = plt.subplots(figsize=(TWO_COL, TWO_COL * 5 / 9))
all_means = []
for scenario_name, res in fragility_results.items():
    if get_scenario_type(scenario_name) == "Delayed Transition":
        continue
    m = res['mean'] * 100
    x = res['fragility_mid']
    col, ls = _styles.get(scenario_name, ("grey", "-"))
    ax.plot(x, m, color=col, linewidth=1.5, linestyle=ls,
            label=fancy_titles.get(scenario_name, scenario_name), zorder=3)
    all_means.append(m.values)
median_across = np.nanmedian(np.stack(all_means, axis=0), axis=0)
ax.plot(x, median_across, color='black', linewidth=1.5, linestyle='--',
        label='Median across scenarios', zorder=2)
ax.set_xlabel("Earth system fragility\n(Composite feedback and tipping score)")
ax.set_ylabel("Path-based resilience [%]")
ax.set_xlim(0, 1); ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3); ax.legend(frameon=False)
ax.annotate("heatmap diagonal", xy=(0.97, -0.17), xycoords='axes fraction',
            xytext=(0.03, -0.17), textcoords='axes fraction', fontsize=5,
            color='#555555', fontstyle='italic', ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7), annotation_clip=False)
fig.savefig(f"{HEATMAP_DIR}/resilience_fragility_diagonal_scenarios.pdf", bbox_inches='tight')
fig.savefig(f"{HEATMAP_DIR}/resilience_fragility_diagonal_scenarios.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved resilience_fragility_diagonal_scenarios.{pdf,png}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Combined figure: (a) NZ2050 heatmap + (b) fragility diagonal (verbatim)
# ══════════════════════════════════════════════════════════════════════════════
if nz2050_data:
    fig_comb = plt.figure(figsize=(TWO_COL, TWO_COL * 9 / 18))
    sf_a, sf_b = fig_comb.subfigures(1, 2, width_ratios=[1.3, 1.0], wspace=0.12)

    gs_a = GridSpec(nrows=2, ncols=3, width_ratios=[1.2, 4, 0.25],
                    height_ratios=[4, 1.2], hspace=0.18, wspace=0.1, figure=sf_a)
    ax_a_heat   = sf_a.add_subplot(gs_a[0, 1])
    ax_a_left   = sf_a.add_subplot(gs_a[0, 0], sharey=ax_a_heat)
    ax_a_bottom = sf_a.add_subplot(gs_a[1, 1], sharex=ax_a_heat)
    ax_a_cbar   = sf_a.add_subplot(gs_a[0, 2])

    d = nz2050_data
    norm_a  = mcolors.Normalize(vmin=0, vmax=100)
    Z_a_pct = np.where(np.isnan(d['Z']), np.nan, d['Z'] * 100)
    X_a, Y_a = np.meshgrid(d['ts_bin_edges'], d['gf_bin_edges'])
    pcm_a = ax_a_heat.pcolormesh(X_a, Y_a, Z_a_pct, cmap=colormap, vmin=0, vmax=100, shading="flat")
    sf_a.colorbar(pcm_a, cax=ax_a_cbar, label="Path-based resilience [%]")

    # fragility diagonal (low GFP+tipping -> high): grey dashed, behind the numbers
    ax_a_heat.plot([d['ts_bin_edges'][0], d['ts_bin_edges'][-1]],
                   [d['gf_bin_edges'][0], d['gf_bin_edges'][-1]],
                   color="grey", linestyle="--", linewidth=2.0, alpha=0.8, zorder=2.5)

    for i, y in enumerate(d['gf_bin_mids']):
        for j, x in enumerate(d['ts_bin_mids']):
            val = d['pivot_arr'][i, j]
            if np.isfinite(val):
                n_eff = int(min(d['cr_per_gf'][i], d['s_per_ts'][j]))
                if n_eff < HATCH_MIN_N:
                    _hatch_low_n(ax_a_heat, d['ts_bin_edges'], d['gf_bin_edges'], i, j)
                r, g, b, _ = colormap(norm_a(val * 100))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax_a_heat.text(x, y, _cell_label(val), ha="center", va="center",
                               fontsize=5, color="black" if lum > 0.45 else "white")

    ax_a_left.plot(d['res_by_gf_mean'] * 100, d['gf_bin_mids'], color=color_left, linewidth=2, label="Mean")
    ax_a_left.errorbar(d['res_by_gf_mean'] * 100, d['gf_bin_mids'], xerr=[d['ci_gf_lo'], d['ci_gf_hi']],
                       fmt='none', ecolor=color_left, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_a_left.set_xlim(100, 0); ax_a_left.margins(y=0)
    ax_a_left.set_ylim(ax_a_heat.get_ylim())
    ax_a_left.set_ylabel("Climate and carbon feedback sensitivity\n(Binned generalised feedback parameter)")
    ax_a_left.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a_left.yaxis.set_major_locator(ticker.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax_a_left.grid(True, alpha=0.3)
    _gf_hist_counts_a, _ = np.histogram(d['gf_per_config'], bins=d['gf_bin_edges'])
    _gf_hist_frac_a = _gf_hist_counts_a / _gf_hist_counts_a.sum()
    ax_a_left.set_axisbelow(True)
    _ax_lh_a = ax_a_left.twiny()
    _ax_lh_a.barh(d['gf_bin_mids'], _gf_hist_frac_a, height=np.diff(d['gf_bin_edges']).mean() * 0.88,
                  color=color_left, alpha=0.35, zorder=1)
    _ax_lh_a.set_xlim(0, 1); _ax_lh_a.set_xticks([])
    _ax_lh_a.tick_params(top=False, labeltop=False, which='both'); _ax_lh_a.grid(False)
    _hist_handle_gf_a = Patch(facecolor=color_left, alpha=0.4, label="Config\ndensity")
    handles_la, _ = ax_a_left.get_legend_handles_labels()
    ax_a_left.legend(handles=handles_la + [_hist_handle_gf_a], loc="upper left", frameon=False, fontsize=5)

    ax_a_bottom.plot(d['ts_bin_mids'], d['res_by_ts_mean'] * 100, color=color_bottom, linewidth=2, label="Mean")
    ax_a_bottom.errorbar(d['ts_bin_mids'], d['res_by_ts_mean'] * 100, yerr=[d['ci_ts_lo'], d['ci_ts_hi']],
                         fmt='none', ecolor=color_bottom, elinewidth=0.8, capsize=2, alpha=0.8, label="95% CI")
    ax_a_bottom.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a_bottom.xaxis.set_major_locator(ticker.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax_a_bottom.set_xlabel("Tipping susceptibility\n(Binned generalised tipping parameter)")
    ax_a_bottom.set_ylim(0, 100); ax_a_bottom.margins(x=0)
    ax_a_bottom.set_axisbelow(True); ax_a_bottom.grid(True, alpha=0.3)
    _ts_hist_counts_a, _ = np.histogram(d['tipping_susc_rf'], bins=d['ts_bin_edges'])
    _ts_hist_frac_a = _ts_hist_counts_a / _ts_hist_counts_a.sum()
    _ax_bh_a = ax_a_bottom.twinx()
    _ax_bh_a.bar(d['ts_bin_mids'], _ts_hist_frac_a, width=np.diff(d['ts_bin_edges']).mean() * 0.88,
                 color=color_bottom, alpha=0.5, zorder=1)
    _ax_bh_a.set_ylim(0, 1); _ax_bh_a.set_yticks([])
    _ax_bh_a.tick_params(right=False, labelright=False, which='both'); _ax_bh_a.grid(False)
    _hist_handle_ts_a = Patch(facecolor=color_bottom, alpha=0.4, label="Sample\ndensity")
    handles_ba, _ = ax_a_bottom.get_legend_handles_labels()
    ax_a_bottom.legend(handles=handles_ba + [_hist_handle_ts_a], loc="upper right", frameon=False, fontsize=5)

    ax_a_heat.set_xlim(d['ts_bin_edges'][0], d['ts_bin_edges'][-1])
    ax_a_heat.set_ylim(d['gf_bin_edges'][0], d['gf_bin_edges'][-1])
    ax_a_heat.set_aspect("auto"); ax_a_heat.tick_params(labelleft=False)
    plt.setp(ax_a_heat.get_xticklabels(), visible=False)
    sf_a.text(0.10, 0.19, "Path-based\nresilience [%]", ha="left", va="bottom", fontsize=6)

    _lbl_off_comb = mtrans.ScaledTranslation(9/72, 0, fig_comb.dpi_scale_trans)
    ax_a_left.text(0.0, 1.02, "a", transform=ax_a_left.transAxes, fontsize=8,
                   fontweight='bold', va='bottom', ha='left', clip_on=False)
    ax_a_left.text(0.0, 1.02, "Net zero 2050", transform=ax_a_left.transAxes + _lbl_off_comb,
                   fontsize=6, fontstyle='italic', va='bottom', ha='left', color='#444444', clip_on=False)

    ax_b = sf_b.add_subplot(111)
    for sc_name, res_b in fragility_results.items():
        if get_scenario_type(sc_name) == "Delayed Transition":
            continue
        m_b = res_b['mean'] * 100
        x_b = res_b['fragility_mid']
        col_b, ls_b = _styles.get(sc_name, ("grey", "-"))
        ax_b.plot(x_b, m_b, color=col_b, linewidth=1.5, linestyle=ls_b,
                  label=get_scenario_type(sc_name).replace(' ', '\n'), zorder=3)
    ax_b.set_xlabel("Fragility\n(Composite feedback and tipping score)")
    ax_b.set_ylabel("Path-based resilience [%]")
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 100)
    ax_b.grid(True, alpha=0.3); ax_b.legend(frameon=False, fontsize=5)
    # ── caption styled to match the heatmap diagonal: ── Heatmap diagonal ──▶ ──
    ax_b.annotate("", xy=(0.97, -0.17), xytext=(0.03, -0.17),
                  xycoords='axes fraction', textcoords='axes fraction',
                  arrowprops=dict(arrowstyle='->', color='grey', lw=2.0, linestyle='--'),
                  annotation_clip=False)
    ax_b.text(0.5, -0.17, "Heatmap diagonal", transform=ax_b.transAxes,
              fontsize=5, color='grey', fontstyle='italic', ha='center', va='center',
              bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'), clip_on=False)
    ax_b.text(0.0, 1.02, "b", transform=ax_b.transAxes, fontsize=8,
              fontweight='bold', va='bottom', ha='left', clip_on=False)
    ax_b.text(0.0, 1.02, "All scenarios", transform=ax_b.transAxes + _lbl_off_comb,
              fontsize=6, fontstyle='italic', va='bottom', ha='left', color='#444444', clip_on=False)
    #"Cell values are the point-estimate resilience [%]; "
    #"cells with n = min(configs, samples) < 20 are hatched (thin support)."
    fig_comb.savefig(f"{HEATMAP_DIR}/heatmap_nz2050_fragility_combined.pdf", bbox_inches='tight')
    fig_comb.savefig(f"{HEATMAP_DIR}/heatmap_nz2050_fragility_combined.png", dpi=150, bbox_inches='tight')
    plt.close(fig_comb)
    print("Saved heatmap_nz2050_fragility_combined.{pdf,png}", flush=True)

print(f"\nHeatmap figures written under {HEATMAP_DIR}", flush=True)
