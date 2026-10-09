"""calculator.py — shared library for an X%-resilience-gain calculator.
"""
import copy
import glob
import multiprocessing
import os

import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from tqdm import tqdm, trange

from monte_carlo_runner import monte_carlo_fair
from fair_config import load_fair_params
from tipping import compute_tip_prob
from tipping_params import sample_lhs_params

# ── resilience defaults ──────────────────────────
DEFAULT_TEMP_THRESHOLD = 1.5    # °C by year 2100
DEFAULT_RATE_THRESHOLD = 0.04   # °C/decade, 2000–2100
DEFAULT_YEAR_TARGET    = 2100

# fair time axis: extended to 2110 so the 20-yr rolling mean is valid at 2100.
FAIR_TIME = np.arange(1750, 2111, 1)

# resolution knobs (calculator's originals; single-stage finder uses BISECT, the
# confirmator uses FINAL)
N_FAIR_RUNS_BISECT       = 1     # single deterministic run during bisection
N_FAIR_RUNS_FINAL        = 100   # full ensemble for the confirmatory run
N_TIPPING_SAMPLES_BISECT = 100   # fast samples during bisection
N_TIPPING_SAMPLES_FINAL  = 1000  # full samples for the final run

n_jobs = multiprocessing.cpu_count()


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_candidate_scenario(df_emissions, years, year_cols,
                             base_scenario_name, target_year,
                             reduction_frac, candidate_name,
                             ramp_start_year=2025.5, baseline_map=None):
    """
    Create a new emissions scenario that applies an additional fractional reduction
    to base_scenario_name at target_year, carrying the absolute offset forward.

    ``baseline_map`` {variable: natural background emission} clamps each species at its
    natural floor (FaIR ``baseline_emissions``), matching the Current-Policies branch-off
    grid (02). If None, species clamp at 0 (legacy behaviour). The floor never exceeds the
    ramp-start value, so species already below their background are not raised.

    Parameters
    ----------
    df_emissions      : DataFrame [scenario, region, variable, unit, *year_cols]
    years             : float array matching year_cols
    year_cols         : list of column name strings
    base_scenario_name: exact scenario string in df_emissions
    target_year       : year at which full reduction is reached (e.g. 2035.5)
    reduction_frac    : additional fraction to cut (e.g. 0.10 for 10%)
    candidate_name    : name to assign to new scenario rows
    ramp_start_year   : where ramp begins (default 2025.5)

    Returns
    -------
    DataFrame with only candidate_name rows (same columns as df_emissions)
    """
    i_base   = np.argmin(np.abs(years - ramp_start_year))
    i_target = np.argmin(np.abs(years - target_year))

    df_base = df_emissions[df_emissions["scenario"] == base_scenario_name].copy()
    if df_base.empty:
        raise ValueError(f"Scenario '{base_scenario_name}' not found in emissions.")

    new_rows = []
    for _, row in df_base.iterrows():
        new_row  = row.copy()
        vals     = new_row[year_cols].astype(float).values.copy()
        v_base       = vals[i_base]
        v_target_old = vals[i_target]
        v_target_new = v_target_old * (1.0 - reduction_frac)

        # linear ramp from ramp_start_year to target_year
        ramp_mask = (years >= ramp_start_year) & (years <= target_year)
        vals[ramp_mask] = np.maximum(
            np.linspace(v_base, v_target_new, ramp_mask.sum()), 0.0
        )

        # constant absolute offset for all years after target_year
        difference = v_target_old - v_target_new
        vals[years > target_year] -= np.abs(difference)

        # clamp at the species' natural background (or 0 if none / no map). Never above
        # the ramp-start value, so a species already below its background isn't raised.
        flr = 0.0 if baseline_map is None else float(baseline_map.get(str(row["variable"]).strip(), 0.0))
        vals = np.maximum(vals, min(flr, v_base))

        new_row[year_cols] = vals
        new_row["scenario"] = candidate_name
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)[df_emissions.columns].reset_index(drop=True)


def _extend_for_rolling_mean(df, year_cols, extensions_length=2,
                              trend_start=2090.5, trend_end=2100.5):
    """
    Extend emissions DataFrame by `extensions_length` extra year columns beyond
    trend_end, using a linear extrapolation fitted on [trend_start, trend_end].
    """
    years_arr = np.array([float(c) for c in year_cols])
    idx_start = int(np.argmin(np.abs(years_arr - trend_start)))
    idx_end   = int(np.argmin(np.abs(years_arr - trend_end)))
    last_year = float(year_cols[idx_end])
    extra_years = [last_year + 5 * (i + 1) for i in range(extensions_length)]
    extra_cols  = [f"{y:.1f}" for y in extra_years]

    new_rows = []
    for _, row in df.iterrows():
        vals = row[year_cols].astype(float).values
        x    = years_arr[idx_start:idx_end + 1]
        y    = vals[idx_start:idx_end + 1]
        y_valid = y[np.isfinite(y)]
        if len(y_valid) < 2:
            y_extra = np.full(len(extra_years), y_valid[-1] if len(y_valid) else np.nan)
        else:
            a, b = np.polyfit(x, y, 1)
            y_extra = a * np.array(extra_years) + b
        y_extra = np.maximum(y_extra, 0.0)  # no negative emissions in extension
        new_row = row.copy()
        for col, val in zip(extra_cols, y_extra):
            new_row[col] = val
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)


def write_mini_csvs(df_ndc_rows, df_candidate_rows, emissions_path, forcing_path,
                    df_forcing_template, year_cols, extend=True):
    """
    Write a 2-scenario emissions CSV (extended to ~2110 for 20-yr rolling mean)
    and matching forcing CSV for fair.
    Copies Volcanic/Solar rows from df_forcing_template for any missing scenarios.

    extend : if True (calculator default), linearly extrapolate two extra year
             columns past 2100.5 so the 20-yr rolling mean is valid at 2100.
             The cleaned pipeline's NGFS emissions are ALREADY extended to 2110.5
             (real NGFS extension used by 04/08), so 12/13 pass extend=False to
             avoid duplicating those columns / re-extrapolating.
    """
    os.makedirs(os.path.dirname(emissions_path), exist_ok=True)
    if extend:
        df_ndc_rows  = _extend_for_rolling_mean(df_ndc_rows, year_cols)
        df_candidate_rows = _extend_for_rolling_mean(df_candidate_rows, year_cols)
    df_mini = pd.concat([df_ndc_rows, df_candidate_rows], ignore_index=True)
    df_mini.to_csv(emissions_path, index=False)

    all_scenarios = df_mini["scenario"].unique()
    template_v = df_forcing_template[df_forcing_template["Variable"] == "Volcanic"].iloc[0:1]
    template_s = df_forcing_template[df_forcing_template["Variable"] == "Solar"].iloc[0:1]

    new_forcing_rows = []
    for sc in all_scenarios:
        for var, tmpl in [("Volcanic", template_v), ("Solar", template_s)]:
            if not ((df_forcing_template["Scenario"] == sc) &
                    (df_forcing_template["Variable"] == var)).any():
                new_forcing_rows.append(tmpl.assign(Scenario=sc))

    existing = df_forcing_template[df_forcing_template["Scenario"].isin(all_scenarios)]
    df_forcing_out = pd.concat(
        [existing] + new_forcing_rows, ignore_index=True
    ).drop_duplicates()
    df_forcing_out.to_csv(forcing_path, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE AGGREGATION 
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_temperature_from_runs(run_dir, scenario_name, max_runs=None):
    """
    Load fair run_*.nc files and return raw temperature and 20-yr running mean
    for the given scenario.

    max_runs : if set, only load the first max_runs files (used to cap stale
               directories that contain more files than the current N_FAIR_RUNS_BISECT).

    Returns
    -------
    raw_temp : DataArray (run, timebounds, config) — raw fair output for tipping
    run_mean : DataArray (timebounds, member)      — baseline-corrected running mean
                                                      for resilience criteria
    configs  : list of config labels
    runs     : list of run integer indices
    """
    run_files = sorted(glob.glob(os.path.join(run_dir, "run_*.nc")))
    if not run_files:
        raise FileNotFoundError(f"No run_*.nc files in {run_dir}")
    if max_runs is not None:
        run_files = run_files[:max_runs]

    das = [xr.open_dataset(f)["__xarray_dataarray_variable__"] for f in run_files]
    temp_all = xr.concat(das, dim=pd.RangeIndex(len(das), name="run"))

    # try selecting by scenario name with and without parentheses
    scenario_clean = scenario_name.replace("(", "").replace(")", "")
    try:
        raw_temp = temp_all.sel(scenario=scenario_name)
    except KeyError:
        raw_temp = temp_all.sel(scenario=scenario_clean)

    configs = raw_temp.config.values.tolist()
    runs    = raw_temp.run.values.tolist()

    # 20-yr running mean with 1850–1901 baseline (matches process_data.py)
    weights_51yr      = np.ones(52)
    weights_51yr[0]   = 0.5
    weights_51yr[-1]  = 0.5
    data     = raw_temp.stack(member=("run", "config"))
    baseline = np.average(
        data.sel(timebounds=slice(1850, 1901)).mean("member"),
        weights=weights_51yr,
    )
    run_mean = (data - baseline).rolling(timebounds=20, center=True).mean()
    run_mean = run_mean.dropna(dim="timebounds", how="all")

    return raw_temp, run_mean, configs, runs


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def fast_full_crossed_bootstrap(T, B=1000, seed=0):
    """
    Multi-level crossed bootstrap for resilience tensor T of shape (n_cfg, n_runs, n_samp).
    Returns dict with mean, ci_low (2.5%), ci_high (97.5%), samples.
    """
    rng = np.random.default_rng(seed)
    n_cfg, n_runs, n_samp = T.shape
    Q = np.empty(B, dtype=np.float32)
    for b in trange(B, desc="Bootstrap"):
        ci = rng.integers(0, n_cfg,  size=n_cfg)
        ri = rng.integers(0, n_runs, size=n_runs)
        si = rng.integers(0, n_samp, size=n_samp)
        Q[b] = T[ci][:, ri][:, :, si].mean()
    return {
        "mean":    float(Q.mean()),
        "ci_low":  float(np.quantile(Q, 0.025)),
        "ci_high": float(np.quantile(Q, 0.975)),
        "samples": Q,
    }


def bernoulli_ci(p, n, z=1.96):
    """
    Normal-approximation (Wald) CI for a proportion p with effective sample size n.
    Used for quick per-iteration CI monitoring during bisection.
    Returns (ci_low, ci_high).
    """
    se = np.sqrt(p * (1.0 - p) / max(n, 1))
    return max(0.0, p - z * se), min(1.0, p + z * se)


# ══════════════════════════════════════════════════════════════════════════════
# RESILIENCE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_resilience_scalar(run_mean_temp, prob_any_tipping,
                               temp_threshold=DEFAULT_TEMP_THRESHOLD,
                               rate_threshold=DEFAULT_RATE_THRESHOLD,
                               year_target=DEFAULT_YEAR_TARGET,
                               config_subset=None):
    """
    Compute mean resilience = P(climate OK) × P(no tipping).

    Parameters
    ----------
    run_mean_temp    : DataArray (timebounds, member) — 20-yr running mean, baseline-corrected
    prob_any_tipping : DataArray (config, run) — aggregated mean tipping probability
    temp_threshold   : °C limit at year_target
    rate_threshold   : °C/decade rate limit over 2000–year_target
    year_target      : year at which temperature must be below threshold
    config_subset    : list of config values to include (None = all)

    Returns
    -------
    resilience_mean : float
    """
    rm = run_mean_temp

    climate_violation = (rm.sel(timebounds=slice(year_target, None)) > temp_threshold).any("timebounds")
    rate_of_change    = rm.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation    = (rate_of_change.sel(timebounds=slice(2000, year_target)) > rate_threshold).any("timebounds")

    binary_resilient = ~(climate_violation | rate_violation)
    binary_resilient = binary_resilient.unstack("member").astype(float)  # (run, config)

    # xarray aligns by dimension name: (run, config) × (config, run)
    resilience_binary = binary_resilient * (1.0 - prob_any_tipping)

    if config_subset is not None:
        resilience_binary = resilience_binary.sel(config=config_subset)

    return float(resilience_binary.mean())


def compute_resilience_with_bootstrap(run_mean_temp, prob_any_tipping_sample,
                                       temp_threshold=DEFAULT_TEMP_THRESHOLD,
                                       rate_threshold=DEFAULT_RATE_THRESHOLD,
                                       year_target=DEFAULT_YEAR_TARGET,
                                       config_subset=None,
                                       n_bootstrap=1000):
    """
    Full resilience computation with crossed bootstrap CI.
    Used once after bisection converges, with the 1000-sample final tipping run.

    Parameters
    ----------
    prob_any_tipping_sample : DataArray (sample, config, run) — per-sample binary tipping

    Returns
    -------
    dict with mean, ci_low, ci_high, ci_half_width
    """
    rm = run_mean_temp

    climate_violation = (rm.sel(timebounds=slice(year_target, None)) > temp_threshold).any("timebounds")
    rate_of_change    = rm.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation    = (rate_of_change.sel(timebounds=slice(2000, year_target)) > rate_threshold).any("timebounds")

    binary_resilient = ~(climate_violation | rate_violation)
    binary_resilient = binary_resilient.unstack("member").astype(float)  # (run, config)

    # broadcast: (run, config) × (sample, config, run) to (sample, config, run)
    resilience_binary = binary_resilient * (1.0 - prob_any_tipping_sample)

    if config_subset is not None:
        resilience_binary = resilience_binary.sel(config=config_subset)

    # (config, run, sample) tensor for crossed bootstrap
    T = resilience_binary.transpose("config", "run", "sample").values.astype(np.float32)
    ci = fast_full_crossed_bootstrap(T, B=n_bootstrap)
    ci["ci_half_width"] = (ci["ci_high"] - ci["ci_low"]) / 2.0
    return ci


# ══════════════════════════════════════════════════════════════════════════════
# CO2 / NDC REDUCTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_co2_reduction_GtCO2(df_base, df_candidate, target_year,
                                 co2_vars=("CO2 FFI", "CO2 AFOLU")):
    """
    CO₂ (FFI + AFOLU) reduction at target_year (positive = candidate emits less).
    NOTE ON UNITS: despite the name, this returns the reduction in the emissions CSV's
    native units, which are **Mt CO₂/yr** — divide by 1000 for GtCO₂/yr (done downstream
    in 17_figures_calculator.py / 32_tables_fig4.py; the published GtCO₂ numbers are
    therefore correct). Does NOT apply GWP conversions for other species, pure CO2 only.
    """
    year_col = str(float(target_year))
    if year_col not in df_base.columns:
        year_col = str(int(target_year))

    region = "World"
    total  = 0.0
    for var in co2_vars:
        bv = df_base.loc[(df_base["region"] == region) & (df_base["variable"] == var), year_col]
        cv = df_candidate.loc[(df_candidate["region"] == region) & (df_candidate["variable"] == var), year_col]
        if not bv.empty and not cv.empty:
            total += float(bv.values[0]) - float(cv.values[0])
    return total


def compute_reduction_vs_2030ndc(df_ndc, df_candidate, target_year,
                                 ref_year=2030.5,
                                 co2_vars=("CO2 FFI", "CO2 AFOLU")):
    """
    Compare candidate scenario at target_year against NDC at ref_year (default 2030.5).
    Positive values mean the candidate emits LESS than the 2030 NDC baseline.
    Returns (abs_GtCO2, frac, ndc_ref_GtCO2).
    """
    ref_col = str(float(ref_year))
    if ref_col not in df_ndc.columns:
        ref_col = str(int(ref_year))
    tgt_col = str(float(target_year))
    if tgt_col not in df_candidate.columns:
        tgt_col = str(int(target_year))

    region = "World"
    ndc_ref  = 0.0
    cand_tgt = 0.0
    for var in co2_vars:
        nv = df_ndc.loc[(df_ndc["region"] == region) & (df_ndc["variable"] == var), ref_col]
        cv = df_candidate.loc[(df_candidate["region"] == region) & (df_candidate["variable"] == var), tgt_col]
        if not nv.empty:
            ndc_ref  += float(nv.values[0])
        if not cv.empty:
            cand_tgt += float(cv.values[0])

    abs_GtCO2 = ndc_ref - cand_tgt
    frac      = abs_GtCO2 / ndc_ref if ndc_ref != 0 else float("nan")
    return abs_GtCO2, frac, ndc_ref


def compute_fragility_masks(configs, params_file, n_terciles=3):
    """
    Config-level generalised-feedback (GFP) score per config, via PCA on
    [iirf_uptake[CO2], ocean_heat_transfer[0], deep_ocean_efficacy] (climate-model
    component only; no tipping).
    """
    df_configs, _ = load_fair_params(params_file)
    param_cols = ["iirf_uptake[CO2]", "ocean_heat_transfer[0]", "deep_ocean_efficacy"]
    df_sub     = df_configs.loc[configs, param_cols].copy()

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    X_scaled = StandardScaler().fit_transform(df_sub.values)
    gf_raw   = PCA(n_components=1).fit_transform(X_scaled).flatten()
    gf_raw   = -gf_raw
    gf_norm  = (gf_raw - gf_raw.min()) / (gf_raw.max() - gf_raw.min())
    gf_scores = pd.Series(gf_norm, index=df_sub.index)

    labels     = ["low", "medium", "high"]
    boundaries = np.linspace(0, 1, n_terciles + 1)
    masks = {}
    for i, label in enumerate(labels[:n_terciles]):
        lo_b, hi_b = boundaries[i], boundaries[i + 1]
        if i == n_terciles - 1:
            m = (gf_scores >= lo_b) & (gf_scores <= hi_b)
        else:
            m = (gf_scores >= lo_b) & (gf_scores < hi_b)
        masks[label] = gf_scores[m].index.tolist()
    return masks, gf_scores


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE EVALUATION  
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_candidate(reduction_frac, df_emissions, years, year_cols,
                       base_scenario_name, target_year, df_forcing_template,
                       params_file, species_configs_file, iter_dir,
                       n_runs, n_samples,
                       fair_time=FAIR_TIME,
                       temp_threshold=DEFAULT_TEMP_THRESHOLD,
                       rate_threshold=DEFAULT_RATE_THRESHOLD,
                       year_target=DEFAULT_YEAR_TARGET,
                       config_subset=None,
                       candidate_name=None,
                       lhs_params=None,
                       return_bootstrap=False,
                       n_bootstrap=1000,
                       extend=True):
    """
    Evaluate one candidate reduction

    Parameters
    ----------
    reduction_frac    : fraction to additionally cut at target_year
    df_emissions      : full emissions DataFrame (must contain base_scenario_name)
    years, year_cols  : float array + matching column-name list
    base_scenario_name: the reference (e.g. the NDC) scenario, parentheses kept
    target_year       : year at which the reduction is fully applied
    df_forcing_template : volcanic/solar forcing template DataFrame
    params_file       : FAIR calibrated-parameters CSV
    species_configs_file : FAIR species-configs CSV
    iter_dir          : working directory for this evaluation's CSVs / runs
    n_runs            : FAIR runs (1 = deterministic during finding, 100 = final)
    n_samples         : LHS tipping samples (100 fast, 1000 final)
    config_subset     : optional list of configs to restrict tipping+resilience to
    candidate_name    : scenario name for the candidate (auto if None)
    lhs_params        : pre-sampled tipping params (sample_lhs_params result); if
                        None, sampled here at the shared seed. A slice of the first
                        n_samples is used, matching the calculator's "generate max,
                        use a slice" behaviour.
    return_bootstrap  : if True, also run the 1000-sample tipping + crossed
                        bootstrap CI (final/confirmator mode)

    Returns
    -------
    dict: reduction_frac, candidate_name, achieved_resilience, and (if
    return_bootstrap) the bootstrap CI dict under "bootstrap".
    """
    if candidate_name is None:
        candidate_name = f"calc__ty{int(target_year)}_r{reduction_frac:.6f}"

    os.makedirs(iter_dir, exist_ok=True)
    emissions_path = os.path.join(iter_dir, "emissions.csv")
    forcing_path   = os.path.join(iter_dir, "forcing.csv")
    run_dir        = os.path.join(iter_dir, "runs")

    df_base = df_emissions[df_emissions["scenario"] == base_scenario_name].copy()

    # ── 1. build candidate scenario ──────────────────────────────────────────
    # natural-background floor per species (FaIR baseline_emissions), matching the
    # Current-Policies branch-off grid (02) so panels a and b/c clamp consistently.
    _sp = pd.read_csv(species_configs_file)
    baseline_map = dict(zip(_sp["name"], _sp["baseline_emissions"].fillna(0.0)))
    df_candidate = build_candidate_scenario(
        df_emissions, years, year_cols,
        base_scenario_name, target_year, reduction_frac, candidate_name,
        baseline_map=baseline_map,
    )

    # ── 2. write mini CSVs ───────────────────────────────────────────────────
    write_mini_csvs(df_base, df_candidate, emissions_path, forcing_path,
                    df_forcing_template, year_cols, extend=extend)

    # ── 3. run fair (deterministic single run when n_runs == 1) ──────────────
    monte_carlo_fair(
        fair_time,
        [base_scenario_name, candidate_name],
        params_file, species_configs_file,
        emissions_file=emissions_path,
        forcing_file=forcing_path,
        n_runs=n_runs,
        output_dir=run_dir,
        stochastic=n_runs > 1,
    )

    # ── 4. aggregate temperature ─────────────────────────────────────────────
    raw_temp, run_mean, configs, runs = aggregate_temperature_from_runs(
        run_dir, candidate_name, max_runs=n_runs
    )

    # ── 5. tipping probabilities (shared cascade) ────────────────────────────
    tip_configs = configs if config_subset is None else [c for c in configs if c in set(config_subset)]
    if lhs_params is None:
        lhs_params = sample_lhs_params(n_samples=N_TIPPING_SAMPLES_FINAL)

    prob_any, prob_any_sample, _ = compute_tip_prob(
        raw_temp, lhs_params, n_samples=n_samples, configs=tip_configs, runs=runs,
    )

    # ── 6. resilience ────────────────────────────────────────────────────────
    achieved_resilience = compute_resilience_scalar(
        run_mean, prob_any,
        temp_threshold, rate_threshold, year_target,
        config_subset=config_subset,
    )

    # emissions-side reductions for the Fig-3 panel-b annotations (df_base = the
    # NDC baseline; cheap, no FAIR/tipping)
    co2_GtCO2 = compute_co2_reduction_GtCO2(df_base, df_candidate, target_year)
    vs30_abs, vs30_frac, ndc_2030 = compute_reduction_vs_2030ndc(df_base, df_candidate, target_year)

    result = {
        "reduction_frac": reduction_frac,
        "candidate_name": candidate_name,
        "target_year": target_year,
        "achieved_resilience": achieved_resilience,
        "co2_GtCO2": co2_GtCO2,
        "reduction_vs_ndc2030_GtCO2": vs30_abs,
        "reduction_vs_ndc2030_frac": vs30_frac,
        "ndc_2030_GtCO2": ndc_2030,
        "n_runs": n_runs,
        "n_samples": n_samples,
    }

    if return_bootstrap:
        result["bootstrap"] = compute_resilience_with_bootstrap(
            run_mean, prob_any_sample,
            temp_threshold, rate_threshold, year_target,
            config_subset=config_subset, n_bootstrap=n_bootstrap,
        )

    return result
