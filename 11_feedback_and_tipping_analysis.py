"""Scenario-independent Earth-system axes: TCRE, effective tipping temperature, SI PCA.

Writes three physical, scenario-independent quantities consumed by the heatmaps (16)
and the calculator figures (17), plus one SI diagnostic:

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

T_eff is heavy (~10k ODE solves per set) and parallel over SLURM_CPUS_PER_TASK ->
submit via SLURM.
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

FINDER_LHS_SEED = 4321      # finder's separate 100-point LHS; matches 13
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
# Part 1 — TCRE per config (concentration-driven 1pctCO2) — replicates fair-calibrate
#          v1.4.0 parallel_1pct.py exactly, so our per-config TCRE == theirs.
# ══════════════════════════════════════════════════════════════════════════════
# CO2/CH4/N2O only, concentration-driven, 1850..1990. CO2 grows 1%/yr from the FIXED
# 1850 value (NOT each config's own baseline); CH4/N2O held at their 1850 values; the
# same three 1850 values are used as baseline_concentration AND as the forcing
# reference for every config. Only the 16 climate / carbon-cycle / forcing-scale
# columns that parallel_1pct overrides are taken per config from the v1.4.0 calibration
# (not override_defaults on the whole file, which would reset the fixed 1850 baseline).
# TCR = T at index 70 (1920, 2xCO2); TCRE = TCR / cumC_2x * 1000 (K/1000 PgC);
# AF70 = airborne fraction at index 70; T_1000GtC = warming at 1000 GtC (3664 GtCO2).
CO2_1850, CH4_1850, N2O_1850 = 284.3169988, 808.2490285, 273.021047
GTCO2_PER_GTC = 44.009 / 12.011         # CO2/C molar-mass ratio (exact; matches fair-calibrate)
PARALLEL_COLS = [
    "ocean_heat_capacity[0]", "ocean_heat_capacity[1]", "ocean_heat_capacity[2]",
    "ocean_heat_transfer[0]", "ocean_heat_transfer[1]", "ocean_heat_transfer[2]",
    "deep_ocean_efficacy", "gamma_autocorrelation", "forcing_4co2",
    "iirf_0[CO2]", "iirf_airborne[CO2]", "iirf_uptake[CO2]", "iirf_temperature[CO2]",
    "forcing_scale[CO2]", "forcing_scale[CH4]", "forcing_scale[N2O]",
]

f = FAIR(ch4_method="Thornhill2021")
f.define_time(1850, 1990, 1)
f.define_scenarios(["1pctCO2"])
f.define_configs(configs)
sp3 = ["CO2", "CH4", "N2O"]
_, props = read_properties(filename=SPECIES_CONFIGS)
props = {k: {**props[k], "input_mode": "concentration", "greenhouse_gas": True,
             "aerosol_chemistry_from_emissions": False,
             "aerosol_chemistry_from_concentration": False} for k in sp3}
f.define_species(sp3, props)
f.allocate()
f.fill_species_configs(SPECIES_CONFIGS)
with fair_override_file(df_fair[PARALLEL_COLS]) as ov:   # only parallel_1pct's 16 columns
    f.override_defaults(ov)

t = np.arange(0, 141)
fill(f.concentration, (CO2_1850 * 1.01 ** t)[:, None, None], specie="CO2")
fill(f.concentration, CH4_1850, specie="CH4")
fill(f.concentration, N2O_1850, specie="N2O")
for sp, val in [("CO2", CO2_1850), ("CH4", CH4_1850), ("N2O", N2O_1850)]:
    fill(f.species_configs["baseline_concentration"], val, specie=sp)
    fill(f.species_configs["forcing_reference_concentration"], val, specie=sp)
fill(f.climate_configs["stochastic_run"], False)
initialise(f.concentration, f.species_configs["baseline_concentration"])
for a in ["forcing", "temperature", "cumulative_emissions", "airborne_emissions"]:
    initialise(getattr(f, a), 0)
f.run(progress=False)

TCR = f.temperature.sel(layer=0).values[70, 0, :]                        # K at index 70 (1920)
cum_co2 = f.cumulative_emissions.sel(specie="CO2").values[:, 0, :]        # GtCO2 (timebounds, config)
air_co2 = f.airborne_emissions.sel(specie="CO2").values[:, 0, :]
cumC = cum_co2[70] / GTCO2_PER_GTC                                       # GtCO2 -> PgC at index 70
TCRE = TCR / cumC * 1000.0                                               # K / 1000 PgC
AF70 = air_co2[70] / cum_co2[70]                                         # airborne fraction at index 70
temp_ts = f.temperature.sel(layer=0).values[:, 0, :]                     # (timebounds, config)
T_1000GtC = np.array([np.interp(1000.0, cum_co2[:, c] / GTCO2_PER_GTC, temp_ts[:, c])
                      for c in range(len(configs))])                     # warming at 1000 GtC
ECS = (df_fair["forcing_4co2"] / df_fair["ocean_heat_transfer[0]"] / 2).values

df_tcre = pd.DataFrame({"config": configs, "TCR": TCR, "cumC_2x": cumC, "TCRE": TCRE,
                        "AF70": AF70, "T_1000GtC": T_1000GtC, "ECS": ECS}).set_index("config")
df_tcre.to_csv(os.path.join(SAVE_DIR, "tcre.csv"))

print(f"\n[TCRE] n={len(df_tcre)} configs  (config set == ensemble: {set(configs) == ensemble_configs})")
print(f"[TCRE] median={np.median(TCRE):.3f}  5-95%=[{np.percentile(TCRE, 5):.3f}, {np.percentile(TCRE, 95):.3f}]")
print(f"[TCRE] T70   5/50/95 = {np.percentile(TCR, [5, 50, 95])}")
print(f"[TCRE] T1000 5/50/95 = {np.percentile(T_1000GtC, [5, 50, 95])}")
print(f"[TCRE] AF70  16/50/84 = {np.percentile(AF70, [16, 50, 84])}")
print(f"[TCRE] TCRE  16/50/84 = {np.percentile(TCRE, [16, 50, 84])}")
print(f"[TCRE] corr(TCRE,TCR)={np.corrcoef(TCRE, TCR)[0, 1]:.3f}  corr(TCRE,ECS)={np.corrcoef(TCRE, ECS)[0, 1]:.3f}")
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
# Only ECS (per config, the same value as tcre.csv — computed once and broadcast to
# all runs, so the figures and the TCRE axis agree) and the 2100 running-mean
# temperature (merged on the (config, run) labels, not by position).
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
