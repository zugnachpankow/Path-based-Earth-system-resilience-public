"""16_figures_heatmap.py — TCRE × T_eff resilience heatmaps + fragility (Fig. 3).

Replaces the old GFP × tipping-susceptibility heatmap. Both axes are now physical and
scenario-independent:
  * y — TCRE per config (output/feedback/tcre.csv, from 11); high TCRE = fragile.
  * x — effective tipping temperature T_eff per LHS sample
        (output/feedback/t_eff_lhs1000.csv, from 11); low T_eff = fragile.

Panel (a) heatmap: rows = TCRE deciles (low→high, bottom→top), cols = T_eff deciles
(high→low, left→right); tick labels are the physical decile edges. Cell = PERI averaged
over (configs in the row) × (all runs) × (samples in the col) = resilience_cr ×
(1 − tipping_any). Cells with n_eff = min(configs, samples) < HATCH_MIN_N are hatched.
The fragile corner is top-right and the resilience diagonal runs bottom-left → top-right.

Panel (b) fragility diagonal: rank composite = 0.5·TCRE_pctrank + 0.5·(1 − T_eff_pctrank)
in 10 equal-width [0, 1] bins (same bins every scenario; computed once).

Shared heatmap machinery lives in src/heatmap_common.py (reused by si/23). Heavy -> SLURM.
Produces (figures/heatmaps): Resilience_heatmap_<scenario>.{pdf,png},
resilience_fragility_diagonal_scenarios.{pdf,png}, heatmap_nz2050_fragility_combined.{pdf,png},
combined_cache.pkl, median_ci_table.md, heatmap_peri_crosscheck.csv.
"""
import os
import sys
import gc
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from figures_common import (
    TWO_COL, FIG_DIR, RESILIENCE_BASE, res_x, res_y, rate_str,
    main_scenarios, get_scenario_type, build_naming, composite_styles, load_tipping,
)
from heatmap_common import (
    N_DEC, build_axis, compute_scenario, draw_heatmap, _wilson, fragility_mids,
)

HEATMAP_DIR = os.path.join(FIG_DIR, "heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)

FEEDBACK_DIR = os.path.join(os.path.dirname(RESILIENCE_BASE), "feedback")
Y_LABEL = "TCRE (K per 1000 PgC)"
X_LABEL = "Effective tipping temperature (°C)"
FRAG_LABEL = "Earth-system fragility (rank composite)"

# ── scenario-independent axes (computed once) ───────────────────────────────────
tcre = pd.read_csv(os.path.join(FEEDBACK_DIR, "tcre.csv"), index_col="config")["TCRE"]
teff = pd.read_csv(os.path.join(FEEDBACK_DIR, "t_eff_lhs1000.csv"), index_col="sample")["T_eff"]
tcre_dec, tcre_rank, tcre_edges = build_axis(tcre)
teff_dec, teff_rank, teff_edges_asc = build_axis(teff)
teff_edges_desc = teff_edges_asc[::-1]      # x plotted high→low (left→right)

tipping_sample_dict, _ = load_tipping()
fancy_titles, _c, _l, _m = build_naming(main_scenarios)
data_path = os.path.join(RESILIENCE_BASE, f"resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}")
# 09 per-scenario PERI (the published value) — the cross-check reference below
_summary_csv = os.path.join(data_path, "summary",
                            f"resilience_summary_time{res_x}_temp{res_y}_rate{rate_str}.csv")
summary_peri = pd.read_csv(_summary_csv).set_index("scenario")["resilience"].to_dict()
print(f"Loaded tipping ({len(tipping_sample_dict)}) + TCRE ({len(tcre)}) + T_eff ({len(teff)})", flush=True)


def _scenario_dict(scenario_name):
    if scenario_name not in tipping_sample_dict:
        return None
    csv_path = os.path.join(data_path, f"resilience_{scenario_name}_time{res_x}_temp{res_y}_rate{rate_str}.csv")
    if not os.path.exists(csv_path):
        return None
    # the 09 CSV is per-(config, run, sample) (~84 M rows, several GB); resilience_cr is
    # constant across sample, so read only the 3 needed columns and dedup to (config, run).
    df_res = (pd.read_csv(csv_path, usecols=["config", "run", "resilience_cr"])
              .drop_duplicates(["config", "run"]))
    da = tipping_sample_dict[scenario_name]["prob_any_tipping_sample"]
    res_pivot = df_res.pivot_table(index="config", columns="run", values="resilience_cr")
    resilience_cr = res_pivot.reindex(index=da.config.values, columns=da.run.values).values
    return compute_scenario(da, resilience_cr, tcre_dec, tcre_rank, teff_dec, teff_rank)


# ══════════════════════════════════════════════════════════════════════════════
# per-scenario heatmaps + caches
# ══════════════════════════════════════════════════════════════════════════════
fragility_results = {}
ci_table_rows, crosscheck_rows = [], []
nz2050_data = None

for scenario_name in tqdm(main_scenarios, desc="Heatmaps"):
    d = _scenario_dict(scenario_name)
    if d is None:
        print(f"  Skipping {scenario_name}: no tipping/resilience data", flush=True)
        continue
    safe_name = scenario_name.replace("/", "_")

    fig = plt.figure(figsize=(TWO_COL, TWO_COL * 8 / 11))
    gs = GridSpec(2, 3, width_ratios=[1.2, 4, 0.25], height_ratios=[4, 1.2], hspace=0.2, wspace=0.12)
    ax_heat = fig.add_subplot(gs[0, 1]); ax_left = fig.add_subplot(gs[0, 0], sharey=ax_heat)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_heat); ax_cbar = fig.add_subplot(gs[0, 2])
    draw_heatmap(ax_heat, ax_left, ax_bottom, ax_cbar, d, fig, tcre_edges, teff_edges_desc, Y_LABEL, X_LABEL)
    ax_heat.set_title(get_scenario_type(scenario_name), fontsize=7, pad=3)
    ax_left.legend(loc="upper left", frameon=False); ax_bottom.legend(loc="upper right", frameon=False)
    fig.savefig(f"{HEATMAP_DIR}/Resilience_heatmap_{safe_name}.pdf", bbox_inches="tight")
    fig.savefig(f"{HEATMAP_DIR}/Resilience_heatmap_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fragility_results[scenario_name] = pd.DataFrame(
        {"fragility_mid": fragility_mids, "mean": d["frag_mean"], "count": d["frag_cnt"]})
    _ci = np.array([_wilson(d["pivot"][i, j], int(min(d["cfg_per_row"][i], d["smp_per_col"][j])))
                    for i in range(N_DEC) for j in range(N_DEC) if np.isfinite(d["pivot"][i, j])])
    ci_table_rows.append({"scenario": get_scenario_type(scenario_name),
                          "median_95ci_width_pp": round(float(np.nanmedian(_ci.sum(axis=1))), 2) if _ci.size else np.nan})
    # cross-check the count-weighted cell/bin means against the 09 summary PERI (the
    # published value), not against the mean of the same array.
    s09 = summary_peri.get(scenario_name, np.nan)
    crosscheck_rows.append({"scenario": get_scenario_type(scenario_name),
                            "peri_09_summary_pct": round(s09 * 100, 4),
                            "peri_cells_pct": round(d["peri_cells"] * 100, 4),
                            "peri_frag_pct": round(d["peri_frag"] * 100, 4),
                            "max_abs_diff_pp": round(max(abs(d["peri_cells"] - s09),
                                                         abs(d["peri_frag"] - s09)) * 100, 5)})
    if scenario_name == "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050":
        nz2050_data = d
    gc.collect()

print(f"Heatmap loop complete. Processed {len(fragility_results)} scenarios.", flush=True)

with open(os.path.join(HEATMAP_DIR, "combined_cache.pkl"), "wb") as _f:
    pickle.dump({"nz2050_data": nz2050_data, "fragility_results": fragility_results}, _f)
if ci_table_rows:
    pd.DataFrame(ci_table_rows).to_markdown(os.path.join(HEATMAP_DIR, "median_ci_table.md"), index=False)
if crosscheck_rows:
    cc = pd.DataFrame(crosscheck_rows)
    cc.to_csv(os.path.join(HEATMAP_DIR, "heatmap_peri_crosscheck.csv"), index=False)
    print(f"[cross-check] count-weighted cell/bin mean vs 09 PERI: max abs diff = "
          f"{cc['max_abs_diff_pp'].max():.5f} pp (expect ~0)", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# fragility diagonal across scenarios (standalone)
# ══════════════════════════════════════════════════════════════════════════════
_styles = composite_styles(main_scenarios)
fig, ax = plt.subplots(figsize=(TWO_COL, TWO_COL * 5 / 9))
all_means, x = [], fragility_mids
for scenario_name, res in fragility_results.items():
    if get_scenario_type(scenario_name) == "Delayed Transition":
        continue
    m = res["mean"] * 100
    col, ls = _styles.get(scenario_name, ("grey", "-"))
    ax.plot(x, m, color=col, linewidth=1.5, linestyle=ls,
            label=fancy_titles.get(scenario_name, scenario_name), zorder=3)
    all_means.append(m.values)
if all_means:
    ax.plot(x, np.nanmedian(np.stack(all_means, 0), 0), color="black", linewidth=1.5,
            linestyle="--", label="Median across scenarios", zorder=2)
ax.set_xlabel(FRAG_LABEL); ax.set_ylabel("Path-based resilience [%]")
ax.set_xlim(0, 1); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); ax.legend(frameon=False)
fig.savefig(f"{HEATMAP_DIR}/resilience_fragility_diagonal_scenarios.pdf", bbox_inches="tight")
fig.savefig(f"{HEATMAP_DIR}/resilience_fragility_diagonal_scenarios.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved resilience_fragility_diagonal_scenarios.{pdf,png}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# combined figure: (a) NZ2050 heatmap + (b) fragility diagonal
# ══════════════════════════════════════════════════════════════════════════════
if nz2050_data is not None:
    fig_comb = plt.figure(figsize=(TWO_COL, TWO_COL * 9 / 18))
    sf_a, sf_b = fig_comb.subfigures(1, 2, width_ratios=[1.3, 1.0], wspace=0.12)
    gs_a = GridSpec(2, 3, width_ratios=[1.2, 4, 0.25], height_ratios=[4, 1.2],
                    hspace=0.2, wspace=0.12, figure=sf_a)
    ax_a_heat = sf_a.add_subplot(gs_a[0, 1]); ax_a_left = sf_a.add_subplot(gs_a[0, 0], sharey=ax_a_heat)
    ax_a_bottom = sf_a.add_subplot(gs_a[1, 1], sharex=ax_a_heat); ax_a_cbar = sf_a.add_subplot(gs_a[0, 2])
    draw_heatmap(ax_a_heat, ax_a_left, ax_a_bottom, ax_a_cbar, nz2050_data, sf_a,
                 tcre_edges, teff_edges_desc, Y_LABEL, X_LABEL, cell_fs=5)
    ax_a_heat.plot([0, N_DEC], [0, N_DEC], color="grey", linestyle="--", linewidth=2.0, alpha=0.8, zorder=2.5)
    ax_a_left.legend(loc="upper left", frameon=False, fontsize=5)
    ax_a_bottom.legend(loc="upper right", frameon=False, fontsize=5)
    _lbl = mtrans.ScaledTranslation(9 / 72, 0, fig_comb.dpi_scale_trans)
    ax_a_left.text(0.0, 1.02, "a", transform=ax_a_left.transAxes, fontsize=8, fontweight="bold",
                   va="bottom", ha="left", clip_on=False)
    ax_a_left.text(0.0, 1.02, "Net zero 2050", transform=ax_a_left.transAxes + _lbl, fontsize=6,
                   fontstyle="italic", va="bottom", ha="left", color="#444444", clip_on=False)

    ax_b = sf_b.add_subplot(111)
    for sc_name, res_b in fragility_results.items():
        if get_scenario_type(sc_name) == "Delayed Transition":
            continue
        col_b, ls_b = _styles.get(sc_name, ("grey", "-"))
        ax_b.plot(res_b["fragility_mid"], res_b["mean"] * 100, color=col_b, linewidth=1.5,
                  linestyle=ls_b, label=get_scenario_type(sc_name).replace(" ", "\n"), zorder=3)
    ax_b.set_xlabel(FRAG_LABEL); ax_b.set_ylabel("Path-based resilience [%]")
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 100); ax_b.grid(True, alpha=0.3); ax_b.legend(frameon=False, fontsize=5)
    ax_b.text(0.0, 1.02, "b", transform=ax_b.transAxes, fontsize=8, fontweight="bold",
              va="bottom", ha="left", clip_on=False)
    ax_b.text(0.0, 1.02, "All scenarios", transform=ax_b.transAxes + _lbl, fontsize=6,
              fontstyle="italic", va="bottom", ha="left", color="#444444", clip_on=False)
    fig_comb.savefig(f"{HEATMAP_DIR}/heatmap_nz2050_fragility_combined.pdf", bbox_inches="tight")
    fig_comb.savefig(f"{HEATMAP_DIR}/heatmap_nz2050_fragility_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig_comb)
    print("Saved heatmap_nz2050_fragility_combined.{pdf,png}", flush=True)

print(f"\nHeatmap figures written under {HEATMAP_DIR}", flush=True)
