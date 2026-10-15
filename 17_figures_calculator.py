"""17_figures_calculator.py — calculator main figure (panels a, b, c).

Ported from ERI/plots/results_nature/calculator_combined.py.
  (a) ambition heatmap — resilience gain over the Current-Policies branch-off grid
      (start-year × decarbonisation rate), parsed from the 09 resilience summary.
  (b) emission trajectories — NDC baseline vs +1% and +10% candidates at 2035.5.
  (c) fragility grid — 3×3 heatmap of the required NDC cut over TCRE × T_eff terciles
      (rows TCRE low→high, cols T_eff high→low), for the +10 %pt target.

Consumes the cleaned pipeline's calculator outputs (14 confirmator gain + grid modes).
Light — reads a few CSVs; can run on the login node.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtrans
import cmcrameri.cm as cmcs
from matplotlib.colors import LinearSegmentedColormap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)

from figures_common import (apply_style, TWO_COL, FIG_DIR, res_x, res_y, rate, rate_str,
                            FAIR_PARAMS_FILE, time_to_net_zero_yr)

apply_style()

# Each panel-a cell is a full-ensemble per-scenario resilience estimate, so the effective
# independent n is the number of configs (like the main table). A cell whose UNROUNDED
# estimate falls beyond the one-sided 97.5% Clopper-Pearson bound carries no usable point
# estimate -> show that bound (two decimals) instead:
#   below U = 100*(1 - 0.025**(1/n)) ≈ 0.44% -> "<0.44"
#   above L = 100*0.025**(1/n)       ≈ 99.56% -> ">99.56"
N_CP = len(pd.read_csv(FAIR_PARAMS_FILE, index_col=0))


def _cell_label(val_pct, n=N_CP):
    if n >= 1:
        _lo = 100.0 * (1.0 - 0.025 ** (1.0 / n))   # one-sided 97.5% CP upper bound (k=0)
        _hi = 100.0 * 0.025 ** (1.0 / n)           # one-sided 97.5% CP lower bound (k=n)
        if val_pct > _hi:               # unrounded estimate above the upper-saturation bound
            return f">{_hi:.2f}"
        if val_pct < _lo:               # unrounded estimate below the lower-saturation bound
            return f"<{_lo:.2f}"
    return f"{val_pct:.1f}"

# ── paths (cleaned pipeline) ───────────────────────────────────────────────────
_RBASE = os.environ.get("PERI_RESILIENCE_BASE", "output/resilience")   # switchable to resilience_alt
SUMMARY_CSV = (
    f"{_RBASE}/resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}/"
    f"summary/resilience_summary_time{res_x}_temp{res_y}_rate{rate_str}.csv"
)
RESULTS_1PCT   = "output/calculator/confirmator/confirmed_results_gain0.01.csv"
RESULTS_10PCT  = "output/calculator/confirmator/confirmed_results_gain0.1.csv"
# require the CONFIRMED grid (14 grid mode); never silently fall back to the unconfirmed
# finder (13) output, which would mix confirmed and unconfirmed numbers.
GRID_CONFIRMED = "output/calculator/confirmator/confirmed_grid.csv"
if not os.path.exists(GRID_CONFIRMED):
    raise FileNotFoundError(
        f"Confirmed grid file missing: {GRID_CONFIRMED} (run 14 in grid mode). "
        "Refusing to fall back to the unconfirmed finder output."
    )
EMISSIONS_FILE = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
NDC_SCENARIO   = "REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_(NDCs)"
REDUCTION_COL  = "reduction_frac"   # confirmator column (== finder's converged_reduction_frac)

OUT_DIR = os.path.join(FIG_DIR, "calculator")
os.makedirs(OUT_DIR, exist_ok=True)


# ── data loaders  ────────────────────────
def _load_ambition_data(summary_csv):
    df = pd.read_csv(summary_csv)
    records = []
    for _, row in df.iterrows():
        m = re.search(r"start(\d+\.?\d*)_r(\d+\.?\d*)", row["scenario"])
        if not m:
            continue
        start_year = int(float(m.group(1)))
        rate_pct   = int(round(float(m.group(2)) * 100))
        records.append({
            "start_year": start_year,
            "rate":       rate_pct,
            "resilience": float(row["resilience"]) * 100,
        })
    df_bo = pd.DataFrame(records)
    df_bo["start_year"] = df_bo["start_year"].astype(int)

    # panel a shows pure path-based resilience per branch-off (not a gain above baseline)
    pivot_gain = (
        df_bo.pivot_table(index="start_year", columns="rate",
                          values="resilience", aggfunc="mean")
        .sort_index(ascending=False)
    )
    all_rates  = sorted(df_bo["rate"].unique())
    pivot_gain = pivot_gain.reindex(columns=all_rates)

    res_by_start = (
        df_bo.groupby("start_year")
             .agg(mean=("resilience", "mean"), std=("resilience", "std"))
             .reset_index().sort_values("start_year", ascending=False).reset_index(drop=True)
    )
    res_by_rate = (
        df_bo.groupby("rate")
             .agg(mean=("resilience", "mean"), std=("resilience", "std"))
             .reset_index().sort_values("rate").reset_index(drop=True)
    )
    y_labels = pivot_gain.index.tolist()
    x_labels = pivot_gain.columns.tolist()
    y_pos = np.arange(len(y_labels)) + 0.5
    x_pos = np.arange(len(x_labels)) + 0.5
    res_by_start["y"] = res_by_start["start_year"].map(dict(zip(y_labels, y_pos)))
    res_by_rate["x"]  = res_by_rate["rate"].map(dict(zip(x_labels, x_pos)))
    return pivot_gain, res_by_start, res_by_rate, y_labels, x_labels, y_pos, x_pos


def _co2_total(df_em, scenario, year_cols, region="World"):
    co2_vars = ("CO2 FFI", "CO2 AFOLU")
    mask = ((df_em["scenario"] == scenario) & (df_em["region"] == region) &
            df_em["variable"].isin(co2_vars))
    # cleaned NGFS emissions are in Mt CO2/yr -> convert to Gt for the GtCO2 axes
    return df_em.loc[mask, year_cols].sum(axis=0).values.astype(float) / 1000.0


_CONF_WS = "output/calculator/confirmator"


def _confirmator_candidate_co2(gain, disp_cols, ty=2035, region="World"):
    """CO2 (FFI+AFOLU) of the candidate that was ACTUALLY simulated, read from the
    confirmator's per-candidate emissions.csv -- not reconstructed. Returns Gt CO2/yr
    over ``disp_cols``.
    """
    g = f"{gain:g}"   # 0.01 -> "0.01", 0.10 -> "0.1" (matches 14's gain<...> dir)
    path = os.path.join(_CONF_WS, f"gain{g}", f"ty{int(ty)}", "emissions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"confirmator emissions not found: {path} "
                                f"(run 14 in gain mode for gain {g})")
    df = pd.read_csv(path)
    cand = [s for s in df["scenario"].unique() if str(s).startswith("NDC_calc__")]
    if len(cand) != 1:
        raise ValueError(f"expected exactly 1 candidate scenario in {path}, found {cand}")
    return _co2_total(df, cand[0], disp_cols, region)


def _write_ambition_stats(res_by_start, res_by_rate):
    """Linear fits of the panel-a marginals (resilience vs start-year and vs decarb rate),
    for the results text. Writes tables/calculator_ambition_stats.{csv,md} + prints."""
    s = linregress(res_by_start["start_year"], res_by_start["mean"])   # %pt per year
    r = linregress(res_by_rate["rate"], res_by_rate["mean"])           # %pt per %/yr
    gain_5yr_earlier = -s.slope * 5.0                                  # start earlier -> more resilient
    rows = [
        {"marginal": "resilience vs start-year", "slope_pp_per_unit": round(s.slope, 4),
         "unit": "%pt per year", "derived": f"{gain_5yr_earlier:.2f} %pt per 5 yr earlier",
         "r2": round(s.rvalue ** 2, 4)},
        {"marginal": "resilience vs decarb.-rate", "slope_pp_per_unit": round(r.slope, 4),
         "unit": "%pt per %/yr", "derived": f"{r.slope:.2f} %pt per +1 %/yr rate",
         "r2": round(r.rvalue ** 2, 4)},
    ]
    out_dir = os.path.join(os.path.dirname(FIG_DIR), "tables"); os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "calculator_ambition_stats.csv"), index=False)
    with open(os.path.join(out_dir, "calculator_ambition_stats.md"), "w") as fh:
        fh.write("Panel-a marginal linear fits (path-based resilience, %pt).\n\n")
        fh.write(df.to_markdown(index=False) + "\n")
    print(f"[panel a] resilience vs start-year: {s.slope:.3f} %pt/yr "
          f"(-> {gain_5yr_earlier:.2f} %pt per 5 yr earlier), R²={s.rvalue**2:.4f}", flush=True)
    print(f"[panel a] resilience vs decarb rate: {r.slope:.3f} %pt per %/yr, R²={r.rvalue**2:.4f}", flush=True)


def plot_calculator_combined(output_stem):
    (pivot_gain, res_by_start, res_by_rate,
     y_labels, x_labels, y_pos, x_pos) = _load_ambition_data(SUMMARY_CSV)
    _write_ambition_stats(res_by_start, res_by_rate)

    results_1c5   = pd.read_csv(RESULTS_1PCT)
    results_10pct = pd.read_csv(RESULTS_10PCT)
    grid_df       = pd.read_csv(GRID_CONFIRMED)

    df_em     = pd.read_csv(EMISSIONS_FILE)
    year_cols = [c for c in df_em.columns if c.replace(".", "", 1).isdigit()]
    years_arr = np.array([float(c) for c in year_cols])

    disp_mask  = (years_arr >= 2019.5) & (years_arr <= 2047.5)
    disp_cols  = [c for c, k in zip(year_cols, disp_mask) if k]
    disp_years = years_arr[disp_mask] - 0.5

    ndc_vals = _co2_total(df_em, NDC_SCENARIO, disp_cols)

    r2035_row = results_1c5[results_1c5["target_year"].round(1) == 2035.5].iloc[0]
    R_2035    = float(r2035_row[REDUCTION_COL])
    cand_vals = _confirmator_candidate_co2(0.01, disp_cols)

    r2035_10_row = results_10pct[results_10pct["target_year"].round(1) == 2035.5].iloc[0]
    R_2035_10    = float(r2035_10_row[REDUCTION_COL])
    cand10_vals  = _confirmator_candidate_co2(0.10, disp_cols)

    color_left   = cmcs.glasgow(0.1)
    color_bottom = cmcs.glasgow(0.9)
    traj_cols = {"NDC": "black", "T2035_1": cmcs.batlow(0.25), "T2035_10": cmcs.batlow(0.50)}

    fig = plt.figure(figsize=(TWO_COL, TWO_COL * 0.88))
    _lbl_off = mtrans.ScaledTranslation(9/72, 0, fig.dpi_scale_trans)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.80], hspace=0.30)

    gs_a = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0],
                                            width_ratios=[1.2, 4, 0.25], height_ratios=[4, 1.2],
                                            hspace=0.18, wspace=0.10)
    ax_heat   = fig.add_subplot(gs_a[0, 1])
    ax_left   = fig.add_subplot(gs_a[0, 0], sharey=ax_heat)
    ax_bottom = fig.add_subplot(gs_a[1, 1], sharex=ax_heat)
    ax_cbar   = fig.add_subplot(gs_a[0, 2])

    gs_bc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                             width_ratios=[3, 2], wspace=0.30)
    ax_b = fig.add_subplot(gs_bc[0, 0])
    ax_c = fig.add_subplot(gs_bc[0, 1])

    # ── panel (a): ambition heatmap ──────────────────────────────────────────
    Z = pivot_gain.values
    X, Y = np.meshgrid(np.arange(len(x_labels) + 1), np.arange(len(y_labels) + 1))
    pcm = ax_heat.pcolormesh(X, Y, Z, cmap=cmcs.batlow_r, shading="flat", vmin=0, vmax=100)
    fig.colorbar(pcm, cax=ax_cbar, label="Path-based resilience [%]")
    for i, yp in enumerate(y_pos):
        for j, xp in enumerate(x_pos):
            val = Z[i, j]
            if np.isfinite(val):
                ax_heat.text(xp, yp, _cell_label(val), ha="center", va="center",
                             fontsize=5, color="white" if val > 50 else "black")
    ax_heat.set_xlim(0, len(x_labels)); ax_heat.set_ylim(0, len(y_labels))
    ax_heat.invert_yaxis(); ax_heat.set_aspect("auto")
    ax_heat.set_xlabel(""); ax_heat.set_ylabel(""); ax_heat.tick_params(labelleft=False)
    ax_heat.set_yticks(y_pos); ax_heat.set_yticklabels(y_labels)
    plt.setp(ax_heat.get_xticklabels(), visible=False)

    ax_left.plot(res_by_start["mean"], res_by_start["y"], color=color_left, linewidth=1.5, label="Mean")
    ax_left.fill_betweenx(res_by_start["y"], res_by_start["mean"] - res_by_start["std"],
                          res_by_start["mean"] + res_by_start["std"], color=color_left, alpha=0.3, label="Std. dev.")
    ax_left.set_ylim(ax_heat.get_ylim()); ax_left.invert_xaxis()
    ax_left.set_ylabel("Decarbonisation start year")
    ax_left.set_yticks(y_pos); ax_left.set_yticklabels(y_labels)
    ax_left.set_xlim(100, 0); ax_left.set_xticks([0, 50, 100]); ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc="upper left", frameon=False)
    ax_left.text(0.0, 1.02, "a", transform=ax_left.transAxes, fontsize=8,
                 fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax_left.text(0.0, 1.02, "Path-based resilience\ncurrent-policies baseline",
                 transform=ax_left.transAxes + _lbl_off, fontsize=6, fontstyle="italic",
                 va="bottom", ha="left", color="#444444", clip_on=False)

    ax_bottom.plot(res_by_rate["x"], res_by_rate["mean"], color=color_bottom, linewidth=1.5, label="Mean")
    ax_bottom.fill_between(res_by_rate["x"], res_by_rate["mean"] - res_by_rate["std"],
                           res_by_rate["mean"] + res_by_rate["std"], color=color_bottom, alpha=0.3, label="Std. dev.")
    ax_bottom.set_xlim(ax_heat.get_xlim()); ax_bottom.set_xticks(x_pos); ax_bottom.set_xticklabels(x_labels)
    ax_bottom.set_xlabel("Decarbonisation rate [% yr⁻¹]"); ax_bottom.set_ylim(0, 100)
    # second x-axis (top of heatmap): time to net-zero CO2 = 1/rate [yr] (rate_pct/100)
    ax_top = ax_heat.twiny()
    ax_top.set_xlim(ax_heat.get_xlim()); ax_top.set_xticks(x_pos)
    ax_top.set_xticklabels([f"{time_to_net_zero_yr(xr)}" for xr in x_labels])
    ax_top.set_xlabel("Time to net-zero [yr]")
    ax_bottom.grid(True, alpha=0.3); ax_bottom.legend(loc="upper left", frameon=False)
    # ax_bottom shares x with the heatmap, so setting its tick labels above re-enabled
    # the heatmap's bottom labels; hide them again (rate labels belong on the marginal).
    ax_heat.tick_params(axis="x", labelbottom=False)

    # ── panel (b): emission trajectories ─────────────────────────────────────
    ax_b.plot(disp_years, ndc_vals, color=traj_cols["NDC"], lw=1.2, label="NDC baseline", zorder=3)
    ax_b.plot(disp_years, cand_vals, color=traj_cols["T2035_1"], lw=1.0, ls="-", label="+1 %pt target", zorder=2)
    ax_b.plot(disp_years, cand10_vals, color=traj_cols["T2035_10"], lw=1.0, ls="-", label="+10 %pt target", zorder=2)
    ty_idx    = int(np.argmin(np.abs(disp_years - 2035)))
    ndc_at, cand_at, cand10_at = ndc_vals[ty_idx], cand_vals[ty_idx], cand10_vals[ty_idx]
    ax_b.scatter(2035, cand_at, color=traj_cols["T2035_1"], s=12, zorder=4)
    ax_b.scatter(2035, cand10_at, color=traj_cols["T2035_10"], s=12, zorder=4)
    # annotations: all-emissions cut % (CO2 part) in 2035 vs. NDC, then the confirmed gain + 95% CI
    co2_cut = float(r2035_row["co2_GtCO2"]) / 1000.0
    # ci_low/ci_high are the PAIRED GAIN CI (confirmator resamples the difference
    # tensor), so they are the gain CI directly -- no baseline subtraction.
    _g1 = float(r2035_row["achieved_gain"]) * 100
    _lo1 = float(r2035_row["ci_low"]) * 100; _hi1 = float(r2035_row["ci_high"]) * 100
    ax_b.text(2034.5, cand_at,
              f"−{R_2035*100:.0f}% (−{co2_cut:.1f} GtCO₂) in 2035 vs. NDC\nfor +{_g1:.1f} %pt [{_lo1:.1f}–{_hi1:.1f}]",
              va="top", ha="right", fontsize=5.5, color=traj_cols["T2035_1"], linespacing=1.3)
    co2_cut_10 = float(r2035_10_row["co2_GtCO2"]) / 1000.0
    _g10 = float(r2035_10_row["achieved_gain"]) * 100
    _lo10 = float(r2035_10_row["ci_low"]) * 100; _hi10 = float(r2035_10_row["ci_high"]) * 100
    ax_b.text(2034.5, cand10_at,
              f"−{R_2035_10*100:.0f}% (−{co2_cut_10:.1f} GtCO₂) in 2035 vs. NDC\nfor +{_g10:.1f} %pt [{_lo10:.1f}–{_hi10:.1f}]",
              va="top", ha="right", fontsize=5.5, color=traj_cols["T2035_10"], linespacing=1.3)
    ax_b.set_xlim(disp_years[0] - 0.5, 2045.5)
    ax_b.set_xlabel("Year"); ax_b.set_ylabel("CO₂ FFI+AFOLU [GtCO₂ yr⁻¹]")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(frameon=False, loc="upper right", handlelength=1.5)
    ax_b.text(0.0, 1.02, "b", transform=ax_b.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax_b.text(0.0, 1.02, "Ambition required for ~1 %pt and ~10 %pt with NDC baseline",
              transform=ax_b.transAxes + _lbl_off, fontsize=6, fontstyle="italic",
              va="bottom", ha="left", color="#444444", clip_on=False)

    # ── panel (c): fragility grid — 3×3 TCRE × T_eff, required +10 %pt cut ─────
    # rows = TCRE terciles low→high (bottom→top), cols = T_eff terciles high→low
    # (left→right), so the fragile corner (high TCRE, low T_eff) is top-right. Cell
    # text: the required NDC-2035 cut (R% and GtCO2). Cells where the finder could not
    # reach +10 %pt even at a 99% cut ("max_cut_insufficient") or where +10 %pt is
    # already met at the NDC (R*=0) are grey-hatched with an explanatory label.
    TCRE_ORDER = ["low", "medium", "high"]   # y: bottom→top
    TEFF_ORDER = ["high", "medium", "low"]   # x: left→right (T_eff high→low)
    gi = grid_df.set_index(["tcre_tercile", "teff_tercile"])
    Rpct = np.full((3, 3), np.nan); co2 = np.full((3, 3), np.nan)
    status = np.empty((3, 3), dtype=object); gainpp = np.full((3, 3), np.nan)
    for i, trc in enumerate(TCRE_ORDER):
        for j, tef in enumerate(TEFF_ORDER):
            r = gi.loc[(trc, tef)]
            Rpct[i, j] = float(r["reduction_frac"]) * 100.0
            co2[i, j] = float(r["co2_GtCO2"]) / 1000.0
            status[i, j] = str(r["finder_status"])
            gainpp[i, j] = float(r["achieved_gain"]) * 100.0

    # physical tercile edges for the tick labels (shared across the grid)
    tcre_edges = [float(gi.loc[("low", "high")]["tcre_lo"]), float(gi.loc[("low", "high")]["tcre_hi"]),
                  float(gi.loc[("medium", "high")]["tcre_hi"]), float(gi.loc[("high", "high")]["tcre_hi"])]
    teff_edges = [float(gi.loc[("low", "high")]["teff_hi"]), float(gi.loc[("low", "high")]["teff_lo"]),
                  float(gi.loc[("low", "medium")]["teff_lo"]), float(gi.loc[("low", "low")]["teff_lo"])]

    reach = np.array([[status[i, j] == "ok" for j in range(3)] for i in range(3)])
    Zc = np.where(reach, Rpct, np.nan)
    Xc, Yc = np.meshgrid(np.arange(4), np.arange(4))
    pcm_c = ax_c.pcolormesh(Xc, Yc, Zc, cmap=cmcs.batlow, shading="flat", vmin=0, vmax=100)
    _cbc = fig.colorbar(pcm_c, ax=ax_c, fraction=0.046, pad=0.03)
    _cbc.set_label("NDC-2035 cut for +10 %pt [%]", fontsize=6)
    for i in range(3):
        for j in range(3):
            if reach[i, j]:
                # luminance rule on the (non-reversed) batlow colour at this value,
                # matching panel a — batlow is light at high values, so a plain
                # "white if >50" would put white text on light cells.
                r, g, b, _ = cmcs.batlow(Rpct[i, j] / 100.0)
                col = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.45 else "white"
                ax_c.text(j + 0.5, i + 0.5, f"−{Rpct[i, j]:.0f}%\n(−{co2[i, j]:.1f} Gt)",
                          ha="center", va="center", fontsize=5.5, color=col, linespacing=1.2)
            elif status[i, j] == "max_cut_insufficient":
                # not reachable even at a 99% cut -> hatch
                ax_c.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch="////",
                                             edgecolor="0.55", linewidth=0.0, zorder=2.3))
                ax_c.text(j + 0.5, i + 0.5, f"max +{gainpp[i, j]:.1f} pt\n(99% cut)",
                          ha="center", va="center", fontsize=5, color="0.3", linespacing=1.2)
            else:  # met_at_ndc -> light fill, no hatch
                ax_c.add_patch(plt.Rectangle((j, i), 1, 1, facecolor="0.9",
                                             edgecolor="none", zorder=2.2))
                ax_c.text(j + 0.5, i + 0.5, "met at\nNDC", ha="center", va="center",
                          fontsize=5, color="0.3", linespacing=1.2)
    ax_c.set_xlim(0, 3); ax_c.set_ylim(0, 3); ax_c.set_aspect("auto")
    ax_c.set_xticks(range(4)); ax_c.set_xticklabels([f"{v:.2f}" for v in teff_edges])
    ax_c.set_yticks(range(4)); ax_c.set_yticklabels([f"{v:.2f}" for v in tcre_edges])
    ax_c.set_xlabel("Effective tipping temperature (°C)")
    ax_c.set_ylabel("TCRE (K per 1000 PgC)")
    ax_c.text(0.0, 1.02, "c", transform=ax_c.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax_c.text(0.0, 1.02, "Ambition required vs. Earth-system fragility (+10 %pt)",
              transform=ax_c.transAxes + _lbl_off, fontsize=6, fontstyle="italic",
              va="bottom", ha="left", color="#444444", clip_on=False)

    for ext in (".pdf", ".svg", ".png"):
        fig.savefig(output_stem + ext, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_stem + ext}")
    plt.close(fig)


if __name__ == "__main__":
    plot_calculator_combined(os.path.join(OUT_DIR, "calculator_combined"))
