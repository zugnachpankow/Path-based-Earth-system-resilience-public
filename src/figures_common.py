"""figures_common.py — shared setup + loaders for the figure scripts (15/16/17).
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cmcrameri.cm as cmcs

from fair_config import FAIR_PARAMS


def time_to_net_zero_yr(rate_pct):
    """Years to net-zero CO2 at a constant ``rate_pct`` %/yr cut = round(100/rate).

    Single source of truth so the 17 heatmap axis and the si/32 table agree (e.g.
    8 %/yr -> 100/8 = 12.5 -> 13). Uses round-half-up (Python's round / "%.0f" use
    banker's rounding, which disagreed: 12 vs 13).
    """
    import math
    return int(math.floor(100.0 / rate_pct + 0.5))


# ── repo-relative paths (no os.chdir needed) ───────────────────────────────────
REPO   = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "output"
DATA   = REPO / "data"

TEMPERATURE_FILE   = str(OUTPUT / "all_scenarios_temperature.nc")
RUNNING_MEANS_FILE = str(OUTPUT / "running_mean_temps.pkl")
TIPPING_DIR        = str(OUTPUT / "tipping")
DF_PARAMS_FILE     = str(OUTPUT / "feedback" / "df_params_all.pkl")
FAIR_PARAMS_FILE   = FAIR_PARAMS   # single calibration (v1.4.0) via src/fair_config
# production rate range for the sensitivity (0.25–0.55 °C/decade); each range is saved
# in its own subdir by 10_sensitivity so cutoffs can be compared in an SI.
SOBOL_RATE_DIR     = "rate0.025_0.055"
SENS_DIR_SOBOL     = str(OUTPUT / "sensitivity" / SOBOL_RATE_DIR) + os.sep
RESILIENCE_BASE    = str(OUTPUT / "resilience")
FIG_DIR            = str(REPO / "figures")
TABLE_DIR          = str(REPO / "tables")

# ── Nature figure style (verbatim from plot.py) ────────────────────────────────
MM = 1 / 25.4
ONE_COL   = 89  * MM
MID_COL   = 136 * MM
TWO_COL   = 183 * MM
MAX_HEIGHT = 247 * MM


def apply_style():
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          7,
        'axes.labelsize':     7,
        'axes.titlesize':     7,
        'xtick.labelsize':    6,
        'ytick.labelsize':    6,
        'legend.fontsize':    6,
        'legend.title_fontsize': 6,
        'axes.linewidth':     0.5,
        'xtick.major.width':  0.5,
        'ytick.major.width':  0.5,
        'xtick.minor.width':  0.3,
        'ytick.minor.width':  0.3,
        'xtick.major.size':   2.5,
        'ytick.major.size':   2.5,
        'lines.linewidth':    1.0,
        'patch.linewidth':    0.5,
        'savefig.dpi':        300,
        'figure.dpi':         150,
        'pdf.fonttype':       42,
        'ps.fonttype':        42,
    })


apply_style()

# ── resilience criterion (global) ──────────────────────────────────────────────
res_x = 2100
res_y = 1.5
rate  = 0.04
rate_str = str(round(rate * 10, 4)).replace('.', '')  # 0.04 -> "04"

weights_51yr = np.ones(52); weights_51yr[0] = 0.5; weights_51yr[-1] = 0.5

# ── scenario lists ─────────────────────────────────────────────────────────────
main_scenarios = [
    'REMIND-MAgPIE_3.3-4.8___Delayed_transition',
    'REMIND-MAgPIE_3.3-4.8___Current_Policies',
    'REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_NDCs',
    'REMIND-MAgPIE_3.3-4.8___Net_Zero_2050',
    'REMIND-MAgPIE_3.3-4.8___Below_2C',
    'REMIND-MAgPIE_3.3-4.8___Low_demand',
    'ssp119', 'ssp126', 'ssp245', 'ssp534-over',
]
ssp_scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585', 'ssp434', 'ssp460', 'ssp534-over']
ngfs_scenarios = [
    'REMIND-MAgPIE_3.3-4.8___Delayed_transition',
    'REMIND-MAgPIE_3.3-4.8___Current_Policies',
    'REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_NDCs',
    'REMIND-MAgPIE_3.3-4.8___Net_Zero_2050',
    'REMIND-MAgPIE_3.3-4.8___Below_2C',
    'REMIND-MAgPIE_3.3-4.8___Low_demand',
]


# ── canonical SSP colours (IPCC-style; 126/245/534 match the Sobol panel) ──────
SSP_COLORS = {
    "ssp119": "#00a9cf",
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp434": "#2274ae",
    "ssp460": "#b0724e",
    "ssp534-over": "#92397a",
    "ssp585": "#980002",
}


def ssp_color(sc, default=None):
    """Return the canonical SSP colour if `sc` is an SSP, else `default`."""
    for key, col in SSP_COLORS.items():
        if key in sc:
            return col
    return default


linestyles_cycle = ['-', '--', '-.', ':']


def composite_styles(scenarios):
    """{scenario: (color, linestyle)} matching the combined-heatmap fragility diagonal:
    glasgow palette by index over `scenarios`, SSP scenarios overridden to their
    canonical colour, cycling linestyles. Pass the SAME ordered list everywhere
    (e.g. main_scenarios) so styles are consistent across figures."""
    cols = cmcs.glasgow(np.linspace(0.1, 0.9, len(scenarios)))
    return {sc: (ssp_color(sc, cols[i]), linestyles_cycle[i % len(linestyles_cycle)])
            for i, sc in enumerate(scenarios)}


# ── scenario naming / colours ──────────────────────────────────────────────────
def get_scenario_type(sc):
    if "Improved-COP30" in sc:
        return "COP30 NDCs"
    if "Nationally_Determined_Contributions_NDCs" in sc:
        return "NDCs"
    if "Current_Policies" in sc:
        return "Current Policies"
    if "Delayed_transition" in sc:
        return "Delayed Transition"
    if "Fragmented_World" in sc:
        return "Fragmented World"
    if "Net_Zero_2050" in sc:
        return "Net Zero 2050"
    if "Low_demand" in sc:
        return "Low Demand"
    if "Below_2C" in sc:
        return "Below 2°C"
    if "ssp119" in sc:
        return "SSP1-1.9"
    if "ssp126" in sc:
        return "SSP1-2.6"
    if "ssp245" in sc:
        return "SSP2-4.5"
    if "ssp370" in sc:
        return "SSP3-7.0"
    if "ssp585" in sc:
        return "SSP5-8.5"
    if "ssp434" in sc:
        return "SSP4-3.4"
    if "ssp460" in sc:
        return "SSP4-6.0"
    if "ssp534-over" in sc:
        return "SSP5-3.4-OS"
    return "Other"


def get_model(sc):
    if "GCAM" in sc:
        return "GCAM"
    if "MESSAGEix-GLOBIOM" in sc:
        return "MESSAGEix-GLOBIOM"
    if "REMIND-MAgPIE" in sc:
        return "REMIND-MAgPIE"
    return "Other"


def build_naming(scenarios):
    """Return (fancy_titles, colors, linestyles, color_map) for a scenario list."""
    fancy_titles = {}
    for sc in scenarios:
        _st = get_scenario_type(sc)
        _md = get_model(sc)
        fancy_titles[sc] = f"{_md if _md != 'Other' else ''} {_st if _st != 'Other' else ''}".strip()

    scenario_types = sorted({get_scenario_type(sc) for sc in scenarios})
    cmap = plt.colormaps["cmc.batlow"].resampled(len(scenario_types))
    color_map = {sc_type: cmap(i) for i, sc_type in enumerate(scenario_types)}
    linestyles_map = {"GCAM": "-", "MESSAGEix-GLOBIOM": "--", "REMIND-MAgPIE": ":", "Other": "-."}
    colors     = {sc: color_map[get_scenario_type(sc)] for sc in scenarios}
    linestyles = {sc: linestyles_map[get_model(sc)] for sc in scenarios}
    return fancy_titles, colors, linestyles, color_map


# ── UpSet / trajectory helpers ─────────────────────────────────────────────────
_key_to_criterion = {
    ("C",): "C", ("R",): "R", ("T",): "T",
    ("C", "R"): "C+R", ("C", "T"): "C+T", ("R", "T"): "R+T", ("C", "R", "T"): "C+R+T",
}


def prob_to_alpha(p, alpha_min=0.1, alpha_max=1.0, gamma=0.5):
    return alpha_min + (alpha_max - alpha_min) * (1 - p) ** gamma


def prob_to_lw(p, lw_min=0.3, lw_max=1.5, gamma=0.6):
    return lw_min + (lw_max - lw_min) * (1 - p) ** gamma


def combo_color(key, pal):
    crit_colors_local = {"C": pal[0], "R": pal[1], "T": pal[2]}
    if set(key) == {"C", "R", "T"}:
        return pal[3]
    return np.mean([crit_colors_local[k] for k in key], axis=0)


# ── loaders (pre-computed pipeline outputs) ────────────────────────────────────
def load_coords():
    """(scenarios, configs, runs) — reads coords only, not the 37 GB data."""
    da = xr.open_dataset(TEMPERATURE_FILE)["__xarray_dataarray_variable__"]
    da = da.assign_coords(scenario=da.scenario.str.replace(r"[\(\)]", "", regex=True))
    scenarios = da.scenario.values.tolist()
    configs   = da.config.values.tolist()
    runs      = da.run.values.tolist()
    da.close()
    return scenarios, configs, runs


def load_running_means():
    with open(RUNNING_MEANS_FILE, "rb") as f:
        rmt = pickle.load(f)
    return {k.replace("(", "").replace(")", ""): v for k, v in rmt.items()}


def load_tipping():
    """(tipping_sample_dict, tipping_combined) from output/tipping/*.nc."""
    d = {}
    for f in sorted(os.listdir(TIPPING_DIR)):
        if f.endswith("_tipping_probabilities.nc"):
            sc = f.split("_tipping_probabilities.nc")[0]
            d[sc] = xr.open_dataset(os.path.join(TIPPING_DIR, f))
    combined = xr.concat(
        [d[sc]["prob_any_tipping"] for sc in d],
        dim=pd.Index(list(d.keys()), name="scenario"),
    ).to_dataset(name="prob_any_tipping")
    return d, combined


def load_df_params():
    return pd.read_pickle(DF_PARAMS_FILE)


def load_df_configs():
    return pd.read_csv(FAIR_PARAMS_FILE, index_col=0)


def load_sobol(focal=('ssp126', 'ssp245', 'ssp534-over')):
    """{scenario: Si} for whichever sobol_*.pkl exist (10 may still be running)."""
    out = {}
    for sc in focal:
        p = os.path.join(SENS_DIR_SOBOL, f"sobol_{sc}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[sc] = pickle.load(f)['Si']
    return out
