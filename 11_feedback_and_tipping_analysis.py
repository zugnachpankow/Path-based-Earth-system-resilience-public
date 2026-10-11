"""Scenario-independent Earth-system axes: TCRE, effective tipping temperature, SI PCA.

Replaces the old GFP / RF-tipping-susceptibility analysis. Writes three physical,
scenario-independent quantities consumed by the heatmaps (16) and the calculator
figures (17), plus one SI diagnostic:

  * output/feedback/tcre.csv          — per config: TCR, cumulative C at 2xCO2, TCRE, ECS
                                        (FAIR 2.2.3, v1.4.0 params, concentration-driven
                                         1pctCO2, 140 yr). y-axis of Fig 3 (high TCRE = fragile).
  * output/feedback/t_eff_lhs1000.csv — per LHS tipping sample (seed 1234, n=1000): the
  * output/feedback/t_eff_finder.csv    effective tipping temperature T_eff and sampled
                                        thresholds (finder set: separate n=100, seed 4321).
                                        x-axis of Fig 3 (low T_eff = fragile).
  * output/feedback/config_pca.csv    — SI only: per-config standardised PC1 of the two
                                        climate-feedback parameters kappa1 (ocean_heat_transfer[0])
                                        and epsilon (deep_ocean_efficacy), oriented +corr with ECS.

The RF tipping test, the 16-parameter tipping PCA and the (scenario, config, run) GFP
table are all removed from the main path. T_eff is heavy (~10k ODE solves per set) and
parallel over SLURM_CPUS_PER_TASK -> submit via SLURM.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xarray as xr

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve output/ + data/ from the repo root

from fair_config import FAIR_PARAMS, SPECIES_CONFIGS, load_fair_params, fair_override_file
from tipping_params import sample_lhs_params, SEED
import t_eff

TEMPERATURE_FILE = "output/all_scenarios_temperature.nc"
RUNNING_MEANS_FILE = "output/running_mean_temps.pkl"
SAVE_DIR = "output/feedback"
os.makedirs(SAVE_DIR, exist_ok=True)

FINDER_LHS_SEED = 4321      # finder's separate 100-point LHS ("F3"); matches 13
N_FINDER = 100

# config set / run / scenario coords of the run-output ensemble (guard: TCRE/PCA
# must be on the same configs; the slim df_params_all is broadcast over these)
_temp = xr.open_dataset(TEMPERATURE_FILE)["__xarray_dataarray_variable__"]
_temp = _temp.assign_coords(scenario=_temp.scenario.str.replace(r"[\(\)]", "", regex=True))
ensemble_configs = set(_temp.config.values.tolist())
ensemble_runs = _temp.run.values.tolist()
ensemble_scenarios = _temp.scenario.values.tolist()
del _temp
# GCAM / MESSAGE scenarios are not carried into the feedback table
scenarios_slim = [s for s in ensemble_scenarios if "GCAM" not in s and "MESSAGE" not in s]

df_fair, _ = load_fair_params(FAIR_PARAMS)
configs = df_fair.index.tolist()
if set(configs) != ensemble_configs:
    raise ValueError(
        f"config mismatch: {FAIR_PARAMS} has {len(configs)} configs, "
        f"{TEMPERATURE_FILE} has {len(ensemble_configs)}; sets differ."
    )

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — TCRE per config (concentration-driven 1pctCO2, FAIR 2.2.3 / v1.4.0)
# ══════════════════════════════════════════════════════════════════════════════
# 1pctCO2: CO2 concentration grows 1%/yr from its v1.4.0 baseline; CH4 and N2O are
# held at baseline; no natural forcing, no other species, deterministic run. TCR is
# the warming at yr70 (2xCO2); cumulative emitted C is diagnosed by FAIR's carbon
# cycle. TCRE = TCR / cumC(2x) * 1000  (K per 1000 PgC). ECS = F_4xCO2 / kappa1 / 2.
f = FAIR(ch4_method="Thornhill2021")
f.define_time(0, 140, 1)
f.define_scenarios(["1pct"])
f.define_configs(configs)
_, props = read_properties(filename=SPECIES_CONFIGS)
sp3 = ["CO2", "CH4", "N2O"]
f.define_species(sp3, {k: {**props[k], "input_mode": "concentration"} for k in sp3})
f.allocate()
f.fill_species_configs(SPECIES_CONFIGS)
with fair_override_file(df_fair) as ov:
    f.override_defaults(ov)

bc = f.species_configs["baseline_concentration"]
t = np.arange(0, 141)
fill(f.concentration, bc.sel(specie="CO2").values[None, :] * (1.01 ** t[:, None])[:, None, :], specie="CO2")
fill(f.concentration, bc.sel(specie="CH4").values[None, None, :] * np.ones((141, 1, 1)), specie="CH4")
fill(f.concentration, bc.sel(specie="N2O").values[None, None, :] * np.ones((141, 1, 1)), specie="N2O")
fill(f.climate_configs["stochastic_run"], False)
for a in ["forcing", "temperature", "cumulative_emissions", "airborne_emissions", "ocean_heat_content_change"]:
    initialise(getattr(f, a), 0)
f.run(progress=False)

TCR = f.temperature.sel(layer=0).values[70, 0, :]                       # K at yr70 (2xCO2)
cumC = f.cumulative_emissions.sel(specie="CO2").values[70, 0, :] / 3.664  # GtCO2 -> PgC
TCRE = TCR / cumC * 1000.0                                               # K / 1000 PgC
ECS = (df_fair["forcing_4co2"] / df_fair["ocean_heat_transfer[0]"] / 2).values

df_tcre = pd.DataFrame({"config": configs, "TCR": TCR, "cumC_2x": cumC,
                        "TCRE": TCRE, "ECS": ECS}).set_index("config")
df_tcre.to_csv(os.path.join(SAVE_DIR, "tcre.csv"))

print(f"\n[TCRE] n={len(df_tcre)} configs  (config set == ensemble: {set(configs) == ensemble_configs})")
print(f"[TCRE] median={np.median(TCRE):.3f}  5-95%=[{np.percentile(TCRE, 5):.3f}, {np.percentile(TCRE, 95):.3f}]  "
      f"(AR6: 1.65 [1.0, 2.3])")
print(f"[TCRE] TCR median={np.median(TCR):.3f}  corr(TCRE,TCR)={np.corrcoef(TCRE, TCR)[0, 1]:.3f}  "
      f"corr(TCRE,ECS)={np.corrcoef(TCRE, ECS)[0, 1]:.3f}")
print(f"Wrote {os.path.join(SAVE_DIR, 'tcre.csv')}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# SI — per-config PCA of the two climate-feedback parameters (kappa1, epsilon)
# ══════════════════════════════════════════════════════════════════════════════
# Only the two climate-model feedback parameters (iirf_uptake dropped): kappa1 =
# ocean_heat_transfer[0], epsilon = deep_ocean_efficacy. Standardised PC1, oriented
# so it correlates positively with ECS. SI alternative fragility axis to TCRE.
pca_cols = ["ocean_heat_transfer[0]", "deep_ocean_efficacy"]
X = StandardScaler().fit_transform(df_fair.loc[configs, pca_cols].values)
pca = PCA(n_components=1)
score = pca.fit_transform(X).flatten()
if np.corrcoef(score, ECS)[0, 1] < 0:          # deterministic sign (+corr with ECS)
    score = -score
    loadings = -pca.components_[0]
else:
    loadings = pca.components_[0]
pd.DataFrame({"config": configs, "pca_score": score, "ECS": ECS}).set_index("config").to_csv(
    os.path.join(SAVE_DIR, "config_pca.csv"))
print(f"\n[SI PCA] loadings: kappa1={loadings[0]:+.3f}, epsilon={loadings[1]:+.3f}  "
      f"explained var={pca.explained_variance_ratio_[0]:.3f}  corr(score,ECS)={np.corrcoef(score, ECS)[0, 1]:.3f}")
print(f"Wrote {os.path.join(SAVE_DIR, 'config_pca.csv')}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Slim per-(scenario, config, run) table for the ECS figures (15, si/20, si/23)
# ══════════════════════════════════════════════════════════════════════════════
# Only ECS (per config, the SAME value as tcre.csv — computed once and broadcast to
# all runs, so the figures and the new TCRE axis cannot disagree) and the 2100
# running-mean temperature (merged on the (config, run) LABELS, not by position).
# The old GFP / parameter columns are gone; GFP-specific figures use config_pca.csv.
ecs_per_config = pd.Series(ECS, index=configs)
with open(RUNNING_MEANS_FILE, "rb") as fh:
    running_mean_temps = pickle.load(fh)
running_mean_temps = {k.replace("(", "").replace(")", ""): v for k, v in running_mean_temps.items()}

_frames = []
for sc in scenarios_slim:
    base = pd.MultiIndex.from_product([configs, ensemble_runs], names=["config", "run"]).to_frame(index=False)
    base["scenario"] = sc
    base["ecs"] = base["config"].map(ecs_per_config)
    temp = (running_mean_temps[sc].sel(timebounds=2100).to_series().rename("temp").reset_index())
    base = base.merge(temp[["config", "run", "temp"]], on=["config", "run"], how="left")
    assert base["temp"].notna().all(), f"temp merge left unmatched (config,run) rows for {sc}"
    assert len(base) == len(configs) * len(ensemble_runs), (
        f"{sc}: {len(base)} rows != {len(configs)}x{len(ensemble_runs)}")
    _frames.append(base[["scenario", "config", "run", "ecs", "temp"]])
df_params_all = pd.concat(_frames, ignore_index=True)
df_params_all.to_pickle(os.path.join(SAVE_DIR, "df_params_all.pkl"))
print(f"\n[df_params_all] {df_params_all.shape}  ({len(scenarios_slim)} scenarios x "
      f"{len(configs)} configs x {len(ensemble_runs)} runs); columns={list(df_params_all.columns)}")
print(f"Wrote {os.path.join(SAVE_DIR, 'df_params_all.pkl')}", flush=True)
del running_mean_temps, _frames, df_params_all   # free the running-means before the heavy T_eff step

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — effective tipping temperature T_eff per LHS tipping sample
# ══════════════════════════════════════════════════════════════════════════════
# Lowest constant GMT (held from t=0) at which any element tips before T_END, by
# bisection on [0, 6] degC. Same model path as compute_tip_prob (Earth_System,
# imported in src/t_eff). Two sample sets: the 1000-sample analysis LHS (seed 1234)
# and the finder's separate 100-sample LHS (seed 4321).
p1000 = sample_lhs_params(n_samples=1000, seed=SEED)
df_te = t_eff.compute_t_eff(p1000, 1000)
df_te.to_csv(os.path.join(SAVE_DIR, "t_eff_lhs1000.csv"))

p100 = sample_lhs_params(n_samples=N_FINDER, seed=FINDER_LHS_SEED)
df_tef = t_eff.compute_t_eff(p100, N_FINDER)
df_tef.to_csv(os.path.join(SAVE_DIR, "t_eff_finder.csv"))

te = df_te["T_eff"].values
minthr = df_te[["limits_gis", "limits_thc", "limits_wais", "limits_amaz"]].min(axis=1).values
ok = ~np.isnan(te)
print(f"\n[T_eff 1000] median={np.nanmedian(te):.3f}  5-95%=[{np.nanpercentile(te, 5):.3f}, {np.nanpercentile(te, 95):.3f}]  "
      f"capped={(df_te['flag'] == 'capped').sum()}  nan={(df_te['flag'] == 'nan').sum()}")
print(f"[T_eff 1000] corr(T_eff, min sampled threshold) = {np.corrcoef(te[ok], minthr[ok])[0, 1]:.3f}")
print(f"[T_eff finder] median={np.nanmedian(df_tef['T_eff']):.3f}  "
      f"capped={(df_tef['flag'] == 'capped').sum()}  nan={(df_tef['flag'] == 'nan').sum()}")
print(f"Wrote {os.path.join(SAVE_DIR, 't_eff_lhs1000.csv')} and t_eff_finder.csv", flush=True)
