"""14_figures_main.py — main-paper figures.

Ported from the canonical ERI/plots/results_nature/plot.py. Unlike the original
monolith, this consumes the cleaned pipeline's PRE-COMPUTED outputs instead of
recomputing them:
  * running means   -> output/running_mean_temps.pkl        (08)
  * tipping         -> output/tipping/*.nc                   (07)
  * GFP / ECS / T   -> output/feedback/df_params_all.pkl     (11)
  * Sobol           -> output/sensitivity/sobol_*.pkl        (10; optional)
  * resilience      -> output/resilience/.../summary/*.csv   (09)

Style, helpers and figure logic are kept verbatim; only paths and the (optional)
Sobol / bootstrap-CI inputs are adapted/guarded so the script runs before those
are available.

Figures produced:
  * C1 — trajectories + UpSet, per main scenario × 3 criterion sets (plain and
    ECS-coloured variant; the NZ2050 ECS variant also carries the 3-SSP Sobol panel c).
  (The GFP × tipping heatmap is a separate figure — see 16_figures_heatmap.py.)
"""
import os
import sys
import glob as _glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cmcrameri.cm as cmcs
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from figures_common import (
    TWO_COL, FIG_DIR, res_x, res_y, rate, main_scenarios,
    get_scenario_type, prob_to_lw, combo_color, _key_to_criterion,
    load_running_means, load_tipping, load_df_params, load_sobol,
)

os.makedirs(FIG_DIR, exist_ok=True)

# ── load pre-computed pipeline outputs ─────────────────────────────────────────
running_mean_temps = load_running_means()
print(f"Loaded running means: {len(running_mean_temps)} scenarios", flush=True)
tipping_sample_dict, tipping_combined = load_tipping()
print(f"Loaded tipping: {len(tipping_sample_dict)} scenarios", flush=True)

# GFP / ECS / T@2100 per (scenario, config, run) — for the ECS-coloured variant
df_params_all = load_df_params()
print(f"Loaded df_params_all: {df_params_all.shape}", flush=True)

# Sobol sensitivity (for the NZ2050 panel-c) — from 10 (guarded if not present)
_FOCAL_SC_SOBOL = ['ssp126', 'ssp245', 'ssp534-over']
_SSP_COL_SOBOL  = {'ssp126': '#003466', 'ssp245': '#f69320', 'ssp534-over': '#92397a'}
_SSP_LBL_SOBOL  = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp534-over': 'SSP5-3.4-OS (Overshoot)'}
_sobol_data_e = load_sobol(_FOCAL_SC_SOBOL)
print(f"Sobol available for: {list(_sobol_data_e.keys())}", flush=True)

# UpSet bootstrap CIs (optional — not produced by the cleaned pipeline yet)
_UPSET_BOOT_DIR = os.path.join(FIG_DIR, "..", "bootstrap_upset")
_upset_ci = {}
if os.path.isdir(_UPSET_BOOT_DIR):
    for _csv in sorted(_glob.glob(os.path.join(_UPSET_BOOT_DIR, "upset_bootstrap_*_B*.csv"))):
        try:
            _df_ci = pd.read_csv(_csv)
            if "scenario" not in _df_ci.columns:
                continue
            for _, _r in _df_ci.iterrows():
                _sc, _cr, _b = _r["scenario"], _r["criterion"], int(_r["B"])
                _upset_ci.setdefault(_sc, {})
                if _cr not in _upset_ci[_sc] or _b > _upset_ci[_sc][_cr][2]:
                    _upset_ci[_sc][_cr] = (float(_r["ci_low"]), float(_r["ci_high"]), _b)
        except Exception:
            pass
print(f"UpSet bootstrap CIs for {len(_upset_ci)} scenario(s)", flush=True)


# Cap on plotted resilient trajectories per panel — a random subsample keeps the
# lines (and their tipping-weighted thickness) distinguishable when a scenario has
# tens of thousands of resilient members. Raise/remove to show more.
MAX_TRAJ = 2500

# ══════════════════════════════════════════════════════════════════════════════
# C1 — trajectories + UpSet, per main scenario × 3 criterion sets
# (verbatim from plot.py; plain + ECS-coloured variant with the NZ2050 Sobol panel)
# ══════════════════════════════════════════════════════════════════════════════
for _res_y, _rate, _sub in [
    (res_y, rate, "trajectories"),
    (2.0,   rate, "trajectories/2deg"),
    (res_y, 0.03, "trajectories/0o3K"),
]:
    _traj_dir = os.path.join(FIG_DIR, _sub)
    os.makedirs(_traj_dir, exist_ok=True)
    # UpSet bootstrap CIs (09b) exist only for the standard criterion; don't attach
    # them to the alternative-demand panels (2.0 °C, 0.03) where the bar values differ.
    _is_std = (_res_y == res_y and _rate == rate)
    for scenario in tqdm(main_scenarios, desc=f"C1 {_res_y}°C / {_rate*10:.1f}K/dec"):
        if scenario not in running_mean_temps:
            print(f"  Skipping {scenario}: no running mean data", flush=True)
            continue
        if scenario not in tipping_combined.scenario.values:
            print(f"  Skipping {scenario}: not in tipping_combined", flush=True)
            continue

        safe_name = scenario.replace("/", "_")
        running_mean_temp = running_mean_temps[scenario]

        climate_violation_time = (running_mean_temp.sel(timebounds=slice(res_x, None)) > _res_y)
        rate_of_change = running_mean_temp.diff(dim="timebounds").rolling(timebounds=10, center=True).mean()
        rate_violation_time = (rate_of_change.sel(timebounds=slice(2000, res_x)) > _rate)
        climate_violation = climate_violation_time.any(dim="timebounds")
        rate_violation    = rate_violation_time.any(dim="timebounds")
        resilient         = ~(climate_violation | rate_violation)
        n_total           = resilient.size
        resilience_index  = resilient.sum().item() / n_total if n_total > 0 else 0
        filtered_temp     = running_mean_temp.where(resilient, drop=True)
        non_resilient_temp = running_mean_temp.where(~resilient, drop=True)

        res_s     = resilient.set_index(member=["config", "run"]).unstack("member")
        tip_prob  = tipping_combined["prob_any_tipping"].sel(scenario=scenario).fillna(0.0)
        res_with_tip = res_s.astype(float) * (1.0 - tip_prob)
        resilience_index_with_tipping = res_with_tip.sum().item() / res_with_tip.size

        no_climate = (~climate_violation).astype(float)
        no_rate    = (~rate_violation).astype(float)
        no_tipping = (1.0 - tip_prob).stack(member=("run", "config"))

        combos = {
            ("C",)       : no_climate.mean().item(),
            ("R",)       : no_rate.mean().item(),
            ("T",)       : no_tipping.mean().item(),
            ("C", "R")   : resilience_index,
            ("C", "T")   : (no_climate * no_tipping).mean().item(),
            ("R", "T")   : (no_rate    * no_tipping).mean().item(),
            ("C", "R", "T"): resilience_index_with_tipping,
        }

        times    = non_resilient_temp.timebounds.values
        segments = [
            np.column_stack([times, non_resilient_temp.sel(member=m).values])
            for m in non_resilient_temp.member.values
        ]

        # ── UpSet figure ────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(TWO_COL, TWO_COL * 6 / 14))
        outer   = GridSpec(1, 2, figure=fig, width_ratios=[4, 1.4], wspace=0.38)
        ax_main = fig.add_subplot(outer[0])
        inner   = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], height_ratios=[2.5, 1], hspace=0.04)
        ax_bars   = fig.add_subplot(inner[0])
        ax_matrix = fig.add_subplot(inner[1])

        ordered  = sorted(combos.items(), key=lambda x: -x[1])
        keys     = [k for k, _ in ordered]
        vals     = np.array([v for _, v in ordered]) * 100
        criteria = ["T", "R", "C"]
        n_crit   = 3
        n_combos = len(keys)
        pal      = cmcs.glasgow_r(np.linspace(0.15, 0.85, 4))
        hatch_map = {"C": "//", "R": "\\\\", "T": "////"}

        ax_matrix.set_xlim(-0.5, n_combos - 0.5)
        ax_matrix.set_ylim(-0.5, n_crit - 0.5)
        for xi, key in enumerate(keys):
            active = [ci for ci, c in enumerate(criteria) if c in key]
            if len(active) > 1:
                ax_matrix.plot([xi, xi], [min(active), max(active)],
                               color="black", linewidth=2.5, solid_capstyle="round", zorder=1)
            for ci, c in enumerate(criteria):
                ax_matrix.add_patch(plt.Circle(
                    (xi, ci), 0.28,
                    color="black" if c in key else "#cccccc",
                    zorder=2, transform=ax_matrix.transData))
        ax_matrix.set_yticks(range(n_crit))
        ax_matrix.set_yticklabels(["Tipping", "Rate", "Temp."], fontsize=6)
        ax_matrix.set_xticks([])
        ax_matrix.tick_params(axis='y', length=0)
        ax_matrix.set_aspect('equal')
        for sp in ax_matrix.spines.values():
            sp.set_visible(False)

        for xi, (key, val) in enumerate(zip(keys, vals)):
            is_full = set(key) == {"C", "R", "T"}
            hatch   = "xx" if is_full else (hatch_map[key[0]] if len(key) == 1 else None)
            weight  = "bold" if is_full else "normal"
            ax_bars.bar(xi, val, color=combo_color(key, pal), edgecolor="black",
                        linewidth=0.5, hatch=hatch, width=0.7, zorder=3)
            _crit = _key_to_criterion.get(key)
            _ci   = _upset_ci.get(scenario, {}).get(_crit) if (_crit and _is_std) else None
            if _ci:
                _lo_err = max(val - _ci[0], 0)
                _hi_err = max(_ci[1] - val, 0)
                ax_bars.errorbar(xi, val, yerr=[[_lo_err], [_hi_err]], fmt='none',
                                 color='black', linewidth=0.8, capsize=2, zorder=4)
                _text_y = _ci[1] + 0.8
            else:
                _text_y = val + 0.8
            ax_bars.text(xi, _text_y, f"{val:.0f}%", ha="center", va="bottom",
                         fontsize=5, fontweight=weight)

        ax_bars.set_xlim(-0.5, n_combos - 0.5)
        ax_bars.set_ylim(0, 115)
        ax_bars.set_xticks([])
        ax_bars.set_ylabel("Path-based resilience [%]")
        ax_bars.yaxis.grid(True, linestyle='--', color='grey', alpha=0.4, linewidth=0.4)
        ax_bars.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax_bars.spines[sp].set_visible(False)
        ax_bars.legend(handles=[
            Patch(facecolor=pal[0], hatch="//",   label="Temp.", lw=0.5),
            Patch(facecolor=pal[1], hatch="\\\\", label="Rate", lw=0.5),
            Patch(facecolor=pal[2], hatch="////", label="Tipping", lw=0.5),
            Patch(facecolor=pal[3], hatch="xx",   label="All", lw=0.5),
        ], loc="upper right", frameon=False, fontsize=5)

        # ── Trajectory panel ─────────────────────────────────────────────────────
        plt.sca(ax_main)
        ax_main.axhline(-100.0, linewidth=0.5, alpha=0.5, color="gray", label="Unsafe Trajectories")
        if len(segments) > 0:
            ax_main.add_collection(LineCollection(segments, colors="gray", linewidths=0.3,
                                                  alpha=0.15, rasterized=True))
        rng = np.random.default_rng(42)
        _members_p = list(filtered_temp.member.values)
        _sel_p = rng.permutation(len(_members_p))[:MAX_TRAJ]   # cap for readability
        traj_colors = cmcs.batlow(np.linspace(0.2, 0.9, max(len(_sel_p), 1)))
        rng.shuffle(traj_colors)
        for k, i in enumerate(_sel_p):
            member = _members_p[i]
            run, config = member
            p_tip = tip_prob.sel(config=config, run=run).item()
            lw = prob_to_lw(p_tip)
            if k == 1:
                ax_main.plot(filtered_temp.timebounds, filtered_temp.sel(member=member),
                             color=traj_colors[k], linewidth=1.0, alpha=0.8, label="Safe Trajectories")
            else:
                ax_main.plot(filtered_temp.timebounds, filtered_temp.sel(member=member),
                             color=traj_colors[k], linewidth=lw, alpha=1.0)
        if running_mean_temp.member.size > 0:
            median = running_mean_temp.quantile(0.5, dim="member")
            ax_main.plot(running_mean_temp.timebounds, median, color="black",
                         linewidth=1.6, linestyle="--", alpha=1.0, label="Ensemble Median",
                         zorder=10, path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

        t_line = np.array([1900, 2101])
        T_line = _res_y + _rate * (t_line - 2030)
        ax_main.plot(t_line, T_line, color="black", linestyle=":", linewidth=0.8,
                     alpha=0.9, zorder=5, label=f"Rate Condition\n<{(_rate*10):.1f}°C/Decade")
        ax_main.axhline(0.0, color="black", linewidth=0.1)
        ax_main.errorbar(res_x, _res_y/2, yerr=_res_y/2, fmt='none', color="black",
                         linewidth=2, capsize=5, zorder=11,
                         label=f"Temperature Condition\n<{_res_y}°C in {res_x}")
        ax_main.set_xlabel('Year')
        ax_main.set_ylabel('Temperature anomaly [°C]')
        ax_main.set_xlim(1900, 2101)
        ax_main.set_ylim(-0.2, 4.0)
        ax_main.legend(frameon=False, loc="upper left")

        _lbl_off = mtrans.ScaledTranslation(9/72, 0, fig.dpi_scale_trans)
        ax_main.text(0.0, 1.02, "a", transform=ax_main.transAxes,
                     fontsize=8, fontweight='bold', va='bottom', ha='left', clip_on=False)
        ax_main.text(0.0, 1.02, get_scenario_type(scenario), transform=ax_main.transAxes + _lbl_off,
                     fontsize=6, fontstyle='italic', va='bottom', ha='left', color='#444444', clip_on=False)
        ax_bars.text(0.0, 1.02, "b", transform=ax_bars.transAxes,
                     fontsize=8, fontweight='bold', va='bottom', ha='left', clip_on=False)
        ax_bars.text(0.0, 1.02, "Safety criteria", transform=ax_bars.transAxes + _lbl_off,
                     fontsize=6, fontstyle='italic', va='bottom', ha='left', color='#444444', clip_on=False)

        fig.tight_layout()
        fig.savefig(f"{_traj_dir}/{safe_name}_upset.pdf", bbox_inches='tight')
        fig.savefig(f"{_traj_dir}/{safe_name}_upset.png", bbox_inches='tight')
        plt.close(fig)

        # ── ECS-coloured trajectory version (+ Sobol panel-c for NZ2050) ──────────
        _ecs_per_cfg = (
            df_params_all.loc[df_params_all['scenario'] == scenario]
            .groupby('config')['ecs'].first()
        )
        if filtered_temp.member.size > 0 and not _ecs_per_cfg.isna().all():
            _ecs_vals = np.array([
                _ecs_per_cfg.get(cfg, np.nan)
                for (_, cfg) in filtered_temp.member.values
            ])
            _ecs_vmin = float(np.nanpercentile(_ecs_per_cfg.values, 2))
            _ecs_vmax = float(np.nanpercentile(_ecs_per_cfg.values, 98))
            _ecs_norm = mcolors.Normalize(vmin=_ecs_vmin, vmax=_ecs_vmax)
            _ecs_cmap = cmcs.batlow

            # Sobol panel only on the main criterion dir, for NZ2050, if 10 is done
            _add_sobol_e = (
                _sub == "trajectories" and "Net_Zero_2050" in scenario
                and all(_sc in _sobol_data_e for _sc in _FOCAL_SC_SOBOL)
            )
            if _add_sobol_e:
                fig_e = plt.figure(figsize=(TWO_COL, TWO_COL * 10 / 14))
                _outer_rows_e = GridSpec(2, 1, figure=fig_e, height_ratios=[1.0, 0.5], hspace=0.32)
                outer_e = GridSpecFromSubplotSpec(1, 2, subplot_spec=_outer_rows_e[0],
                                                  width_ratios=[4, 1.4], wspace=0.38)
                _gs_c_e = GridSpecFromSubplotSpec(1, 3, subplot_spec=_outer_rows_e[1], wspace=0.18)
            else:
                fig_e = plt.figure(figsize=(TWO_COL, TWO_COL * 6 / 14))
                outer_e = GridSpec(1, 2, figure=fig_e, width_ratios=[4, 1.4], wspace=0.38)
            ax_me     = fig_e.add_subplot(outer_e[0])
            inner_e   = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_e[1],
                                                height_ratios=[2.5, 1], hspace=0.04)
            ax_bars_e   = fig_e.add_subplot(inner_e[0])
            ax_matrix_e = fig_e.add_subplot(inner_e[1])

            ax_me.axhline(-100.0, linewidth=0.5, alpha=0.5, color="gray", label="Unsafe Trajectories")
            if len(segments) > 0:
                ax_me.add_collection(LineCollection(segments, colors="gray", linewidths=0.3,
                                                    alpha=0.15, rasterized=True))

            # Cap + semi-random-by-ECS draw order: subsample MAX_TRAJ, sort by ECS, split
            # into blocks of _BLOCK, shuffle within each block, and draw low-ECS blocks
            # first. This keeps the sparse high-ECS trajectories on top (co-author note)
            # and distinguishable, while the within-block shuffle + low alpha avoid a
            # single-colour blob.
            _TRAJ_ALPHA = 0.55
            _BLOCK = 50
            _members = list(filtered_temp.member.values)
            _rng = np.random.default_rng(42)
            # NB: don't reuse the outer loop's `_sub` (criterion-dir name) here.
            _ss = _rng.permutation(len(_members))[:MAX_TRAJ]
            _ss = _ss[np.argsort(np.nan_to_num(_ecs_vals[_ss], nan=-np.inf))]  # low -> high ECS
            _order = []
            for _b in range(0, len(_ss), _BLOCK):
                _blk = _ss[_b:_b + _BLOCK].copy()
                _rng.shuffle(_blk)
                _order.extend(_blk.tolist())
            for _draw_i, i in enumerate(_order):
                member = _members[i]
                _, cfg = member
                p_tip = tip_prob.sel(config=cfg, run=member[0]).item()
                lw    = prob_to_lw(p_tip)
                col   = _ecs_cmap(_ecs_norm(_ecs_vals[i])) if np.isfinite(_ecs_vals[i]) else 'gray'
                label = "Safe Trajectories" if _draw_i == 0 else None
                ax_me.plot(filtered_temp.timebounds, filtered_temp.sel(member=member),
                           color=col, linewidth=lw, alpha=_TRAJ_ALPHA, label=label, zorder=2)

            if running_mean_temp.member.size > 0:
                # median forced on top with a white halo so it reads over the trajectories
                ax_me.plot(running_mean_temp.timebounds, running_mean_temp.quantile(0.5, dim="member"),
                           color="black", linewidth=1.6, linestyle="--", alpha=1.0, zorder=10,
                           label="Ensemble Median",
                           path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])

            ax_me.plot(t_line, T_line, color="black", linestyle=":", linewidth=0.8, alpha=0.9,
                       zorder=5, label=f"Rate Condition\n<{(_rate*10):.1f}°C/Decade")
            ax_me.axhline(0.0, color="black", linewidth=0.1)
            ax_me.errorbar(res_x, _res_y / 2, yerr=_res_y / 2, fmt='none', color="black",
                           linewidth=2, capsize=5, zorder=11,
                           label=f"Temperature Condition\n<{_res_y}°C in {res_x}")

            _sm = plt.cm.ScalarMappable(cmap=_ecs_cmap, norm=_ecs_norm)
            _sm.set_array([])
            _cb = fig_e.colorbar(_sm, ax=ax_me, fraction=0.03, pad=0.02, label="ECS [°C]")
            _cb.ax.tick_params(labelsize=6)

            ax_me.set_xlabel('Year')
            ax_me.set_ylabel('Temperature anomaly [°C]')
            ax_me.set_xlim(1900, 2101)
            ax_me.set_ylim(-0.2, 4.0)
            ax_me.legend(frameon=False, loc="upper left")
            _lbl_off_e = mtrans.ScaledTranslation(9/72, 0, fig_e.dpi_scale_trans)
            ax_me.text(0.0, 1.02, "a", transform=ax_me.transAxes, fontsize=8,
                       fontweight='bold', va='bottom', ha='left', clip_on=False)
            ax_me.text(0.0, 1.02, get_scenario_type(scenario), transform=ax_me.transAxes + _lbl_off_e,
                       fontsize=6, fontstyle='italic', va='bottom', ha='left', color='#444444', clip_on=False)

            # UpSet panel — identical logic to main figure
            ax_matrix_e.set_xlim(-0.5, n_combos - 0.5)
            ax_matrix_e.set_ylim(-0.5, n_crit - 0.5)
            for xi, key in enumerate(keys):
                active = [ci for ci, c in enumerate(criteria) if c in key]
                if len(active) > 1:
                    ax_matrix_e.plot([xi, xi], [min(active), max(active)], color="black",
                                     linewidth=2.5, solid_capstyle="round", zorder=1)
                for ci, c in enumerate(criteria):
                    ax_matrix_e.add_patch(plt.Circle((xi, ci), 0.28,
                                          color="black" if c in key else "#cccccc",
                                          zorder=2, transform=ax_matrix_e.transData))
            ax_matrix_e.set_yticks(range(n_crit))
            ax_matrix_e.set_yticklabels(["Tipping", "Rate", "Temp."], fontsize=6)
            ax_matrix_e.set_xticks([])
            ax_matrix_e.tick_params(axis='y', length=0)
            ax_matrix_e.set_aspect('equal')
            for sp in ax_matrix_e.spines.values():
                sp.set_visible(False)

            for xi, (key, val) in enumerate(zip(keys, vals)):
                is_full = set(key) == {"C", "R", "T"}
                hatch_e = "xx" if is_full else (hatch_map[key[0]] if len(key) == 1 else None)
                weight_e = "bold" if is_full else "normal"
                ax_bars_e.bar(xi, val, color=combo_color(key, pal), edgecolor="black",
                              linewidth=0.5, hatch=hatch_e, width=0.7, zorder=3)
                _crit_e = _key_to_criterion.get(key)
                _ci_e   = _upset_ci.get(scenario, {}).get(_crit_e) if (_crit_e and _is_std) else None
                if _ci_e:
                    ax_bars_e.errorbar(xi, val, yerr=[[max(val - _ci_e[0], 0)], [max(_ci_e[1] - val, 0)]],
                                       fmt='none', color='black', linewidth=0.8, capsize=2, zorder=4)
                    _ty_e = _ci_e[1] + 0.8
                else:
                    _ty_e = val + 0.8
                ax_bars_e.text(xi, _ty_e, f"{val:.0f}%", ha="center", va="bottom",
                               fontsize=5, fontweight=weight_e)

            ax_bars_e.set_xlim(-0.5, n_combos - 0.5)
            ax_bars_e.set_ylim(0, 115)
            ax_bars_e.set_xticks([])
            ax_bars_e.set_ylabel("Path-based resilience [%]")
            ax_bars_e.yaxis.grid(True, linestyle='--', color='grey', alpha=0.4, linewidth=0.4)
            ax_bars_e.set_axisbelow(True)
            for sp in ["top", "right"]:
                ax_bars_e.spines[sp].set_visible(False)
            ax_bars_e.legend(handles=[
                Patch(facecolor=pal[0], hatch="//",   label="Temp.", lw=0.5),
                Patch(facecolor=pal[1], hatch="\\\\", label="Rate",  lw=0.5),
                Patch(facecolor=pal[2], hatch="////", label="Tipping", lw=0.5),
                Patch(facecolor=pal[3], hatch="xx",   label="All",   lw=0.5),
            ], loc="upper right", frameon=False, fontsize=5)
            ax_bars_e.text(0.0, 1.02, "b", transform=ax_bars_e.transAxes, fontsize=8,
                           fontweight='bold', va='bottom', ha='left', clip_on=False)
            ax_bars_e.text(0.0, 1.02, "Safety criteria", transform=ax_bars_e.transAxes + _lbl_off_e,
                           fontsize=6, fontstyle='italic', va='bottom', ha='left', color='#444444', clip_on=False)

            if _add_sobol_e:
                _param_lbl_e = ['Return\nyear', 'Temp', 'Rate', 'Tipping']
                _x_e = np.arange(4)
                _w_e = 0.32
                _ax_c_first_e = None
                for _i, _sc in enumerate(_FOCAL_SC_SOBOL):
                    _ax = fig_e.add_subplot(_gs_c_e[0, _i])
                    if _ax_c_first_e is None:
                        _ax_c_first_e = _ax
                    _Si  = _sobol_data_e[_sc]
                    _col = _SSP_COL_SOBOL[_sc]
                    _ax.bar(_x_e - _w_e/2, _Si['S1'], _w_e, yerr=_Si['S1_conf'], capsize=1.5,
                            color=_col, alpha=0.9, label='S1' if _i == 0 else None)
                    _ax.bar(_x_e + _w_e/2, _Si['ST'], _w_e, yerr=_Si['ST_conf'], capsize=1.5,
                            color=_col, alpha=0.45, label='ST' if _i == 0 else None)
                    _ax.set_xticks(_x_e)
                    _ax.set_xticklabels(_param_lbl_e, fontsize=6)  # match a/b tick labels
                    _ax.set_ylim(-0.05, 1.15)
                    _ax.axhline(0, color='k', lw=0.4)
                    _ax.set_title(_SSP_LBL_SOBOL[_sc], fontsize=6, pad=2.0)
                    _ax.grid(True, alpha=0.2, axis='y')
                    if _i == 0:
                        _ax.set_ylabel('Variance contribution\n(Sobol index)', fontsize=7)  # match a/b axis label
                        _ax.legend(fontsize=5, loc='upper right', frameon=False)
                    else:
                        _ax.tick_params(labelleft=False)
                    _ax.tick_params(labelsize=6)  # match a/b tick labels
                _ax_c_first_e.text(0.0, 1.02, "c", transform=_ax_c_first_e.transAxes, fontsize=8,
                                   fontweight='bold', va='bottom', ha='left', clip_on=False)
                _ax_c_first_e.text(0.0, 1.02, "Sensitivity",
                                   transform=_ax_c_first_e.transAxes + _lbl_off_e,
                                   fontsize=6, fontstyle='italic', va='bottom', ha='left',
                                   color='#444444', clip_on=False)

            fig_e.tight_layout()
            fig_e.savefig(f"{_traj_dir}/{safe_name}_upset_ecs.pdf", bbox_inches='tight')
            fig_e.savefig(f"{_traj_dir}/{safe_name}_upset_ecs.png", bbox_inches='tight')
            plt.close(fig_e)

print(f"\nC1 trajectories written under {FIG_DIR}/trajectories[/2deg,/0o3K]", flush=True)
