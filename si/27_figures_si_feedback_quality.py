"""27_figures_si_feedback_quality.py — SI: construction quality of the generalised
feedback parameter (GFP).

The GFP is defined as -PC1 of three standardised earth-system feedback parameters
— iirf_uptake[CO2] (carbon-cycle feedback), ocean_heat_transfer[0] and
deep_ocean_efficacy (climate feedback / transient ocean response) — rescaled to [0,1]
per scenario. This figure documents how that composite is built and how it relates to
warming and to ECS. It is a construction-quality diagnostic, not a claim that these are
the parameters with the largest marginal effect on T@2100.

Ported from the GFP-PCA / GFP↔ECS diagnostics of results_nature/plot.py.
Reads output/feedback/df_params_all.pkl (11). Pickle load dominates -> SLURM (run_27.sh).
Output -> output/figures/si/si_feedback_quality.{pdf,png}
"""
import os
import sys
from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmcs
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import TWO_COL, FIG_DIR, apply_style, get_scenario_type, load_df_params

apply_style()
OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
REP = "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050"
PARAMS = ["iirf_uptake", "ocean_heat_transfer_0", "deep_ocean_efficacy"]
LABELS = ["iirf_uptake[CO2]", "ocean_heat_transfer[0]", "deep_ocean_efficacy"]
_C = cmcs.glasgow(0.35)

dp = load_df_params()
sub = dp[dp["scenario"] == REP].copy()
print(f"{REP}: {len(sub)} rows", flush=True)

# ── GFP PCA on the three feedback parameters (per config; rep scenario) ─────────
X = StandardScaler().fit_transform(sub[PARAMS].values)
pca = PCA(n_components=3).fit(X)
pc1 = -pca.components_[0]                    # GFP = -PC1

fig, axes = plt.subplots(2, 2, figsize=(TWO_COL, TWO_COL * 0.72))
(axa, axb), (axc, axd) = axes

# (a) PC1 loadings
axa.bar(range(len(LABELS)), pc1, color=["steelblue" if v > 0 else "tomato" for v in pc1])
axa.axhline(0, color="k", lw=0.5)
axa.set_xticks(range(len(LABELS))); axa.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=5)
axa.set_ylabel("PC1 loading"); axa.set_title("GFP = −PC1 loadings", fontsize=7)

# (b) explained variance
axb.bar(range(1, 4), pca.explained_variance_ratio_ * 100, color=_C)
axb.set_xticks(range(1, 4)); axb.set_xticklabels([f"PC{i}" for i in range(1, 4)])
axb.set_ylabel("Explained variance [%]"); axb.set_title("PCA explained variance", fontsize=7)

# (c) GFP vs T@2100
axc.scatter(sub["general_feedback"], sub["temp"], s=4, alpha=0.15,
            color=cmcs.glasgow(0.6), rasterized=True)
_rp, _ = spearmanr(sub["general_feedback"], sub["temp"])
axc.set_xlabel("GFP (0–1)"); axc.set_ylabel("Temperature at 2100 [°C]")
axc.set_title(f"GFP vs warming — ρ={_rp:.2f}", fontsize=7)

# (d) GFP vs ECS, coloured by warming
m = sub["ecs"].notna()
sc = axd.scatter(sub.loc[m, "ecs"], sub.loc[m, "general_feedback"], c=sub.loc[m, "temp"],
                 cmap=cmcs.batlow, s=4, alpha=0.5, rasterized=True)
cb = fig.colorbar(sc, ax=axd, fraction=0.046, pad=0.02); cb.set_label("T@2100 [°C]", fontsize=6)
cb.ax.tick_params(labelsize=5)
_rpe, _ = pearsonr(sub.loc[m, "ecs"], sub.loc[m, "general_feedback"])
_rse, _ = spearmanr(sub.loc[m, "ecs"], sub.loc[m, "general_feedback"])
axd.set_xlabel("ECS [°C]"); axd.set_ylabel("GFP (0–1)")
axd.set_title("GFP vs ECS", fontsize=7)
axd.legend(frameon=False, fontsize=5, loc="upper right")
axd.text(0.97, 0.03, f"Pearson r = {_rpe:.2f}\nSpearman ρ = {_rse:.2f}",
         transform=axd.transAxes, fontsize=5.5, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))

for k, ax in enumerate(axes.ravel()):
    ax.grid(True, alpha=0.3)
    ax.text(0.0, 1.04, ascii_lowercase[k], transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_feedback_quality" + ext), dpi=300, bbox_inches="tight")
print("Saved si_feedback_quality.{pdf,png}", flush=True)
