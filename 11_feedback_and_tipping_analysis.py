"""Feedback & tipping analysis.

Constructs the generalised climate feedback (GFP) + ECS per config, the GFP<->ECS
correlation check, tipping susceptibilities and a random-forest test.
"""
import os
import sys
import gc
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from pyDOE import lhs

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve output/ + data/ from the repo root
from resilience import load_tipping

PARAMS_FILE = "data/raw/calibrated_constrained_parameters_calibration1.4.1.csv"
TEMPERATURE_FILE = "output/all_scenarios_temperature.nc"
RUNNING_MEANS_FILE = "output/running_mean_temps.pkl"
TIPPING_DIR = "output/tipping"
SAVE_DIR = "output/feedback"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- scenarios / configs / runs from the concatenated set ---
temperature = xr.open_dataset(TEMPERATURE_FILE)["__xarray_dataarray_variable__"]
temperature = temperature.assign_coords(
    scenario=temperature.scenario.str.replace(r"[\(\)]", "", regex=True)
)
scenarios = temperature.scenario.values.tolist()
configs = temperature.config.values.tolist()
runs = temperature.run.values.tolist()
del temperature
gc.collect()

# GCAM / MESSAGE scenarios are not used
scenarios_no_gcam_message = [s for s in scenarios if "GCAM" not in s and "MESSAGE" not in s]

# --- running means (08 cache) ---
with open(RUNNING_MEANS_FILE, "rb") as f:
    running_mean_temps = pickle.load(f)
running_mean_temps = {
    k.replace("(", "").replace(")", ""): v for k, v in running_mean_temps.items()
}
tipping_sample_dict = load_tipping(TIPPING_DIR)

#### PARAMS  (verbatim from results_nature/plot.py)
df_configs = pd.read_csv(PARAMS_FILE, index_col=0)

df_params_all = pd.MultiIndex.from_product(
    [scenarios_no_gcam_message, configs, runs],   # include runs!
    names=["scenario", "config", "run"]
).to_frame(index=False)
df_params_all['config_id'] = df_params_all['config']
df_params_all['iirf_uptake'] = df_params_all['config'].map(df_configs['iirf_uptake[CO2]'])
df_params_all['ocean_heat_transfer_0'] = df_params_all['config'].map(df_configs['ocean_heat_transfer[0]'])
df_params_all['deep_ocean_efficacy'] = df_params_all['config'].map(df_configs['deep_ocean_efficacy'])
# ECS = forcing_4co2 / ocean_heat_transfer[0] / 2
ecs_per_config = df_configs['forcing_4co2'] / df_configs['ocean_heat_transfer[0]'] / 2
df_params_all['ecs'] = df_params_all['config'].map(ecs_per_config)
df_params_all['time'] = 2100
temps_list = []
for scenario in scenarios_no_gcam_message:
    temp_da = running_mean_temps[scenario].sel(timebounds=2100)
    temp_ordered = temp_da.to_series().sort_index(level=['config', 'run']).values
    temps_list.extend(temp_ordered)
df_params_all['temp'] = temps_list
print(df_params_all.head(), flush=True)
print(f"df_params_all shape: {df_params_all.shape}", flush=True)

param_columns = ['iirf_uptake', 'ocean_heat_transfer_0', 'deep_ocean_efficacy']

df_scenario = df_params_all.groupby("scenario")

for scenario in scenarios_no_gcam_message:
    df_s = df_scenario.get_group(scenario)
    X_params = df_s[param_columns]
    # Standardize
    scaler_pca = StandardScaler()
    X_scaled_pca = scaler_pca.fit_transform(X_params)
    # PCA with 1 component
    pca = PCA(n_components=1)
    general_feedback_pc = pca.fit_transform(X_scaled_pca).flatten()

    general_feedback_pc = -general_feedback_pc

    # Rescale to 0–1 within this scenario
    general_feedback_pc_scaled = (general_feedback_pc - general_feedback_pc.min()) / (general_feedback_pc.max() - general_feedback_pc.min())

    # write back using index
    df_params_all.loc[df_s.index, "general_feedback"] = general_feedback_pc_scaled

print("Computed general_feedback PCA for all scenarios", flush=True)

# --- GFP <-> ECS correlation check (verbatim numerical part) ---
rep_sc = scenarios_no_gcam_message[0]
df_rep_gfp = df_params_all[df_params_all['scenario'] == rep_sc].copy()

r_pearson, p_pearson   = pearsonr(df_rep_gfp['ecs'].dropna(), df_rep_gfp.loc[df_rep_gfp['ecs'].notna(), 'general_feedback'])
r_spearman_ge, p_spear = spearmanr(df_rep_gfp['ecs'].dropna(), df_rep_gfp.loc[df_rep_gfp['ecs'].notna(), 'general_feedback'])
print(f"\nGFP vs ECS ({rep_sc}): Pearson r={r_pearson:.3f} (p={p_pearson:.2e}), "
      f"Spearman rho={r_spearman_ge:.3f} (p={p_spear:.2e})")
del df_rep_gfp

# --- save the GFP/ECS table for the consumers (heatmaps, correlation plots) ---
df_params_all.to_pickle(os.path.join(SAVE_DIR, "df_params_all.pkl"))
print(f"Wrote {os.path.join(SAVE_DIR, 'df_params_all.pkl')}")

# ═══════════════ Tipping susceptibility (LHS + PCA + RF) — verbatim ══════════
# Define parameter bounds (identical to src/tipping.py, so the LHS samples align
# with the tipping output's `sample` dimension)
param_bounds = {
    "gis_time":   (1000, 15000),
    "thc_time":   (15, 300),
    "wais_time":  (2000, 13000),
    "amaz_time":  (50, 200),
    "limits_gis": (0.8, 3.0),
    "limits_thc": (1.4, 8.0),
    "limits_wais": (1.0, 3.0),
    "limits_amaz": (2.0, 6.0),
}
pf_bounds = {
    "pf_wais_to_gis":  (0.1, 0.2),
    "pf_thc_to_gis":   (-1.0, -0.1),
    "pf_gis_to_thc":   (0.1, 1.0),
    "pf_wais_to_thc":  (-0.3, 0.3),
    "pf_gis_to_wais":  (0.1, 1.0),
    "pf_thc_to_wais":  (0.1, 0.15),
    "pf_thc_to_amaz":  (-0.4, 0.4)
}
strength_bounds = {
    "strength": (0.1, 1.0)
}
all_bounds = {}
all_bounds.update(param_bounds)
all_bounds.update(pf_bounds)
all_bounds.update(strength_bounds)

# Generate LHS samples
np.random.seed(1234)
n_samples = 1000
n_dim = len(all_bounds)

lhs_unit = lhs(n_dim, samples=n_samples)
params = {}
for i, (key, (low, high)) in enumerate(all_bounds.items()):
    params[key] = low + lhs_unit[:, i] * (high - low)

df_lhs = pd.DataFrame(params)

# Calculate PCA-based tipping susceptibility
tipping_columns = [
    "gis_time", "thc_time", "wais_time", "amaz_time",
    "limits_gis", "limits_thc", "limits_wais", "limits_amaz",
    "pf_wais_to_gis", "pf_thc_to_gis", "pf_gis_to_thc", "pf_wais_to_thc",
    "pf_gis_to_wais", "pf_thc_to_wais", "pf_thc_to_amaz", "strength"
]

scaler_tr = StandardScaler()
X_tr_scaled = scaler_tr.fit_transform(df_lhs[tipping_columns])
pca_tr = PCA(n_components=1)
tipping_susc_pc = pca_tr.fit_transform(X_tr_scaled).flatten()
tipping_susc_pc = -tipping_susc_pc
tipping_susc_scaled = (tipping_susc_pc - tipping_susc_pc.min()) / (tipping_susc_pc.max() - tipping_susc_pc.min())
df_lhs["tipping_susc"] = tipping_susc_scaled

# Pre-scale LHS params once — reused for every RF fit
params_all = df_lhs[list(all_bounds.keys())]
scaler_rf = StandardScaler()
X_scaled_all = scaler_rf.fit_transform(params_all)

df_lhs.to_pickle(os.path.join(SAVE_DIR, "df_lhs.pkl"))
print(f"Wrote {os.path.join(SAVE_DIR, 'df_lhs.pkl')}")

# --- Per-scenario RF tipping susceptibility + standard test (verbatim) ---
feature_names = list(all_bounds.keys())
rf_rows = []
rf_susc = {}  # {scenario: RF-predicted tipping susceptibility per sample}
for scenario_name in sorted(tipping_sample_dict):
    # Load sample-level tipping data
    da = tipping_sample_dict[scenario_name]['prob_any_tipping_sample']

    # Mean tipping prob per sample (no DataFrame needed for RF training)
    tipping_prob_sample = da.mean(dim=['config', 'run']).values  # shape (1000,)

    rf_all = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1)
    rf_all.fit(X_scaled_all, tipping_prob_sample)
    y_rf_all_pred = rf_all.predict(X_scaled_all)
    r2 = r2_score(tipping_prob_sample, y_rf_all_pred)
    print(f"  R² RF {scenario_name}: {r2:.3f}", flush=True)

    rf_susc[scenario_name] = y_rf_all_pred
    rf_rows.append({
        "scenario": scenario_name,
        "r2": r2,
        **{f"imp_{f}": imp for f, imp in zip(feature_names, rf_all.feature_importances_)},
    })

pd.DataFrame(rf_rows).to_csv(os.path.join(SAVE_DIR, "rf_tipping_test.csv"), index=False)
with open(os.path.join(SAVE_DIR, "rf_tipping_susc.pkl"), "wb") as f:
    pickle.dump(rf_susc, f)
print(f"Wrote {os.path.join(SAVE_DIR, 'rf_tipping_test.csv')} and rf_tipping_susc.pkl")
