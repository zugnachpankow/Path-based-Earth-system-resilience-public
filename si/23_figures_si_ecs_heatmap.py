"""23_figures_si_ecs_heatmap.py — SI resilience heatmaps over ECS × T_eff and PCA × T_eff.

The SI twins of the main TCRE × T_eff heatmap (16), with the y-axis swapped to:
  * ECS per config (output/feedback/tcre.csv column), and
  * the 2-variable climate-feedback PCA score (κ1, ε) per config (output/feedback/config_pca.csv).
The x-axis (T_eff deciles) and all binning/averaging are identical to 16 (shared via
src/heatmap_common.py). Conventions: physical decile edges on the ticks; cells with
n_eff = min(configs, samples) < 20 hatched.

Outputs -> figures/si/{ecs,pca}/Resilience_heatmap_<var>_<scenario>.{pdf,png}
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import (
    TWO_COL, FIG_DIR, RESILIENCE_BASE, res_x, res_y, rate_str,
    main_scenarios, get_scenario_type, load_tipping,
)
from heatmap_common import build_axis, compute_scenario, draw_heatmap

plt.rcParams["hatch.linewidth"] = 0.4
FEEDBACK_DIR = os.path.join(os.path.dirname(RESILIENCE_BASE), "feedback")
X_LABEL = "Effective tipping temperature (°C)"

# scenario-independent axes
teff = pd.read_csv(os.path.join(FEEDBACK_DIR, "t_eff_lhs1000.csv"), index_col="sample")["T_eff"]
teff_dec, teff_rank, teff_edges_asc = build_axis(teff)
teff_edges_desc = teff_edges_asc[::-1]

ecs = pd.read_csv(os.path.join(FEEDBACK_DIR, "tcre.csv"), index_col="config")["ECS"]
pca = pd.read_csv(os.path.join(FEEDBACK_DIR, "config_pca.csv"), index_col="config")["pca_score"]

VARIANTS = {
    "ecs": dict(series=ecs, label="ECS (°C)", subdir="ecs"),
    "pca": dict(series=pca, label="Climate-feedback PCA score (κ1, ε)", subdir="pca"),
}

tipping_sample_dict, _ = load_tipping()
data_path = os.path.join(RESILIENCE_BASE, f"resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}")
print(f"Loaded tipping ({len(tipping_sample_dict)}) + ECS ({len(ecs)}) + PCA ({len(pca)})", flush=True)

# one y-axis per variant; one output dir per variant
AXES = {key: build_axis(v["series"]) for key, v in VARIANTS.items()}
OUT_DIRS = {key: os.path.join(FIG_DIR, "si", v["subdir"]) for key, v in VARIANTS.items()}
for d_ in OUT_DIRS.values():
    os.makedirs(d_, exist_ok=True)

# loop scenarios OUTER so the multi-GB 09 CSV is read ONCE per scenario (3 cols,
# deduped to (config, run)) and reused for every y-variant.
for scenario_name in tqdm(main_scenarios, desc="ECS/PCA heatmaps"):
    if scenario_name not in tipping_sample_dict:
        continue
    csv_path = os.path.join(data_path, f"resilience_{scenario_name}_time{res_x}_temp{res_y}_rate{rate_str}.csv")
    if not os.path.exists(csv_path):
        continue
    df_res = (pd.read_csv(csv_path, usecols=["config", "run", "resilience_cr"])
              .drop_duplicates(["config", "run"]))
    da = tipping_sample_dict[scenario_name]["prob_any_tipping_sample"]
    res_pivot = df_res.pivot_table(index="config", columns="run", values="resilience_cr")
    resilience_cr = res_pivot.reindex(index=da.config.values, columns=da.run.values).values

    for key, v in VARIANTS.items():
        y_dec, y_rank, y_edges = AXES[key]
        d = compute_scenario(da, resilience_cr, y_dec, y_rank, teff_dec, teff_rank)
        fig = plt.figure(figsize=(TWO_COL, TWO_COL * 8 / 11))
        gs = GridSpec(2, 3, width_ratios=[1.2, 4, 0.25], height_ratios=[4, 1.2], hspace=0.2, wspace=0.12)
        ax_heat = fig.add_subplot(gs[0, 1]); ax_left = fig.add_subplot(gs[0, 0], sharey=ax_heat)
        ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_heat); ax_cbar = fig.add_subplot(gs[0, 2])
        draw_heatmap(ax_heat, ax_left, ax_bottom, ax_cbar, d, fig, y_edges, teff_edges_desc, v["label"], X_LABEL)
        ax_heat.set_title(get_scenario_type(scenario_name), fontsize=7, pad=3)
        ax_left.legend(loc="upper left", frameon=False); ax_bottom.legend(loc="upper right", frameon=False)
        safe = scenario_name.replace("/", "_")
        fig.savefig(f"{OUT_DIRS[key]}/Resilience_heatmap_{key}_{safe}.pdf", bbox_inches="tight")
        fig.savefig(f"{OUT_DIRS[key]}/Resilience_heatmap_{key}_{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

print("SI ECS/PCA heatmaps written.", flush=True)
