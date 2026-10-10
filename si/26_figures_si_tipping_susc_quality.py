"""26_figures_si_tipping_susc_quality.py — SI: quality of the tipping-susceptibility
PCA + random-forest regressor (the heatmap x-axis).

The heatmap (16) bins tipping samples by a susceptibility score built two ways:
  * PCA  — PC1 of the 16 LHS pycascades parameters (scenario-independent),
  * RF   — per-scenario random forest of the mean tipping probability per sample on the
           same 16 parameters (this is what the heatmap actually uses).
This figure validates both: PCA explained variance + PC1 loadings; RF R² across the main
scenarios; and for a representative scenario the RF importances, predicted-vs-actual, and
PCA-vs-RF agreement.

Ported from the diagnostic blocks of results_nature/plot.py (RF_diagnostic / plateau).
Reads output/tipping/*.nc (07) + the shared LHS (src/tipping_params). Moderate (one RF
fit per main scenario) -> submit via SLURM (run_26.sh).
Output -> output/figures/si/si_tipping_susc_quality.{pdf,png}
"""
import os
import sys
from string import ascii_lowercase

import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmcs
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
os.chdir(os.path.join(_HERE, ".."))
from figures_common import TWO_COL, FIG_DIR, apply_style, main_scenarios, get_scenario_type, load_tipping
from tipping_params import all_bounds, sample_lhs_params

apply_style()
OUT_DIR = os.path.join(FIG_DIR, "si"); os.makedirs(OUT_DIR, exist_ok=True)
REP = "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050"   # representative scenario for the detail panels
_C = cmcs.glasgow(0.35)

# ── shared LHS + PCA susceptibility (verbatim from 16) ─────────────────────────
params = sample_lhs_params(n_samples=1000)
feat = list(all_bounds.keys())
X = StandardScaler().fit_transform(np.column_stack([params[k] for k in feat]))
pca = PCA(n_components=3).fit(X)
_raw_score = pca.transform(X)[:, 0]

# ── per-scenario RF (R² across scenarios; detail for REP) ──────────────────────
tip, _ = load_tipping()
# orient PC1 so susceptibility correlates positively with the mean tipping
# probability per sample (same rule as 11/16); flip the loadings to match.
from pca_orient import orient_by_correlation
_tip_ref = np.nanmean([tip[sc]["prob_any_tipping_sample"].mean(dim=["config", "run"]).values
                       for sc in main_scenarios if sc in tip], axis=0)
susc_pca = orient_by_correlation(_raw_score, _tip_ref)
pc1_loadings = (pca.components_[0] if np.array_equal(susc_pca, _raw_score)
                else -pca.components_[0])
susc_pca = (susc_pca - susc_pca.min()) / (susc_pca.max() - susc_pca.min())
r2_by_sc, rep = {}, {}
for sc in main_scenarios:
    if sc not in tip:
        continue
    y = tip[sc]["prob_any_tipping_sample"].mean(dim=["config", "run"]).values
    rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1).fit(X, y)
    y_pred = rf.predict(X)
    r2_by_sc[sc] = r2_score(y, y_pred)
    print(f"  R2 RF {sc}: {r2_by_sc[sc]:.3f}", flush=True)
    if sc == REP:
        rep = {"y": y, "y_pred": y_pred, "imp": rf.feature_importances_}

# ── figure: 2x3 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(TWO_COL, TWO_COL * 0.62))
(axa, axb, axc), (axd, axe, axf) = axes

# (a) PCA explained variance
axa.bar(range(1, 4), pca.explained_variance_ratio_ * 100, color=_C)
axa.set_xticks(range(1, 4)); axa.set_xticklabels([f"PC{i}" for i in range(1, 4)])
axa.set_ylabel("Explained variance [%]"); axa.set_title("Tipping-parameter PCA", fontsize=7)

# (b) PC1 loadings
order = np.argsort(np.abs(pc1_loadings))[::-1]
axb.bar(range(len(feat)), pc1_loadings[order], color=_C)
axb.set_xticks(range(len(feat)))
axb.set_xticklabels([feat[i] for i in order], rotation=90, fontsize=3.5)
axb.axhline(0, color="k", lw=0.5); axb.set_ylabel("PC1 loading"); axb.set_title("PC1 loadings", fontsize=7)

# (c) RF R² across scenarios
scs = [s for s in main_scenarios if s in r2_by_sc]
axc.bar(range(len(scs)), [r2_by_sc[s] for s in scs], color=_C)
axc.set_xticks(range(len(scs)))
axc.set_xticklabels([get_scenario_type(s) for s in scs], rotation=90, fontsize=4)
axc.set_ylabel("RF R²"); axc.set_ylim(0, 1); axc.set_title("RF fit across scenarios", fontsize=7)

# (d) RF feature importances (representative)
imp = rep["imp"]; io = np.argsort(imp)[::-1]
axd.bar(range(len(feat)), imp[io], color=_C)
axd.set_xticks(range(len(feat)))
axd.set_xticklabels([feat[i] for i in io], rotation=90, fontsize=3.5)
axd.set_ylabel("RF importance"); axd.set_title(f"RF importances — {get_scenario_type(REP)}", fontsize=7)

# (e) predicted vs actual (representative)
axe.scatter(rep["y"], rep["y_pred"], s=6, alpha=0.4, color=cmcs.glasgow(0.7), rasterized=True)
lims = [min(rep["y"].min(), rep["y_pred"].min()), max(rep["y"].max(), rep["y_pred"].max())]
axe.plot(lims, lims, "k--", lw=0.8)
axe.set_xlabel("Actual tipping prob. / sample"); axe.set_ylabel("RF predicted")
axe.set_title(f"Predicted vs actual — R²={r2_by_sc[REP]:.3f}", fontsize=7)

# (f) PCA vs RF susceptibility (representative)
susc_rf = rep["y_pred"]
rho, _ = spearmanr(susc_pca, susc_rf)
axf.scatter(susc_pca, susc_rf, s=6, alpha=0.4, color=cmcs.glasgow(0.5), rasterized=True)
axf.set_xlabel("PCA susceptibility (0–1)"); axf.set_ylabel("RF susceptibility")
axf.set_title(f"PCA vs RF — ρ={rho:.3f}", fontsize=7)

for k, ax in enumerate(axes.ravel()):
    ax.grid(True, alpha=0.3)
    ax.text(0.0, 1.06, ascii_lowercase[k], transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", clip_on=False)
fig.tight_layout()
for ext in (".pdf", ".png"):
    fig.savefig(os.path.join(OUT_DIR, "si_tipping_susc_quality" + ext), dpi=300, bbox_inches="tight")
print("Saved si_tipping_susc_quality.{pdf,png}", flush=True)
