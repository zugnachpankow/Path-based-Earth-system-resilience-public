"""27_figures_si_feedback_quality.py — SI: construction quality of the 2-variable
climate-feedback PCA score.

The SI fragility alternative to TCRE: standardised PC1 of the two climate-model
feedback parameters κ1 = ocean_heat_transfer[0] and ε = deep_ocean_efficacy
(iirf_uptake dropped), oriented so the score correlates positively with ECS. This
figure documents the construction: PC1 loadings, explained variance, and the score vs
ECS. It matches the per-config PCA computed in 11 (output/feedback/config_pca.csv).

Light — reads two CSVs. Output -> figures/si/si_feedback_quality.{pdf,png}
"""
import os
import sys
from string import ascii_lowercase

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmcs
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import TWO_COL, FIG_DIR, apply_style
from fair_config import FAIR_PARAMS, load_fair_params

apply_style()
OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
PARAMS = ["ocean_heat_transfer[0]", "deep_ocean_efficacy"]
LABELS = ["κ1 (ocean_heat_transfer[0])", "ε (deep_ocean_efficacy)"]
_C = cmcs.glasgow(0.35)

# per-config score + ECS from 11; loadings / explained variance recomputed identically
cp = pd.read_csv("output/feedback/config_pca.csv", index_col="config")
df_fair, _ = load_fair_params(FAIR_PARAMS)
configs = cp.index.tolist()
X = StandardScaler().fit_transform(df_fair.loc[configs, PARAMS].values)
pca = PCA(n_components=2).fit(X)
score = pca.transform(X)[:, 0]
loadings = pca.components_[0]
if np.corrcoef(score, cp["ECS"].values)[0, 1] < 0:    # orient +corr with ECS (matches 11)
    loadings = -loadings
print(f"loadings κ1={loadings[0]:+.3f} ε={loadings[1]:+.3f}  "
      f"explained var PC1={pca.explained_variance_ratio_[0]:.3f}", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, TWO_COL * 0.34))
axa, axb, axc = axes

# (a) PC1 loadings
axa.bar(range(len(LABELS)), loadings, color=["steelblue" if v > 0 else "tomato" for v in loadings])
axa.axhline(0, color="k", lw=0.5)
axa.set_xticks(range(len(LABELS))); axa.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=5)
axa.set_ylabel("PC1 loading"); axa.set_title("PCA score = PC1 loadings", fontsize=7)

# (b) explained variance
axb.bar(range(1, 3), pca.explained_variance_ratio_ * 100, color=_C)
axb.set_xticks(range(1, 3)); axb.set_xticklabels(["PC1", "PC2"])
axb.set_ylabel("Explained variance [%]"); axb.set_title("PCA explained variance", fontsize=7)

# (c) score vs ECS
axc.scatter(cp["ECS"], cp["pca_score"], s=5, alpha=0.4, color=cmcs.glasgow(0.6), rasterized=True)
_rpe, _ = pearsonr(cp["ECS"], cp["pca_score"])
_rse, _ = spearmanr(cp["ECS"], cp["pca_score"])
axc.set_xlabel("ECS (°C)"); axc.set_ylabel("PCA score (κ1, ε)")
axc.set_title("PCA score vs ECS", fontsize=7)
axc.text(0.97, 0.03, f"Pearson r = {_rpe:.2f}\nSpearman ρ = {_rse:.2f}",
         transform=axc.transAxes, fontsize=5.5, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))

for k, ax in enumerate(axes):
    ax.grid(True, alpha=0.3)
    ax.text(0.0, 1.04, ascii_lowercase[k], transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_feedback_quality" + ext), dpi=300, bbox_inches="tight")
print("Saved si_feedback_quality.{pdf,png}", flush=True)
