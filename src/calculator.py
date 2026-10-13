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
from fair_config import load_fair_params, FAIR_PARAMS
from emission_floor import build_floor_map
from scenario_ops import build_candidate_scenario   # shared with 17 (single source)
from temperature import processed_temperature
from tipping import compute_tip_prob
from tipping_params import sample_lhs_params

# ── resilience defaults ──────────────────────────
DEFAULT_TEMP_THRESHOLD = 1.5    # °C by year 2100
DEFAULT_RATE_THRESHOLD = 0.04   # °C/yr: 10-yr mean of annual differences of the 20-yr running mean (~30-yr kernel), [2000, 2096]
DEFAULT_YEAR_TARGET    = 2100

# fair time axis: must match the production NGFS runs (04_run_NGFS.py uses
# np.arange(1750, 2110)) so calculator candidates and the baseline ensemble share
# the same timebounds -> identical running-mean edges. The emissions are extended
# to 2110.5 so the 20-yr centred mean is still valid at 2100.
FAIR_TIME = np.arange(1750, 2110, 1)

# resolution knobs (calculator's originals; single-stage finder uses BISECT, the
# confirmator uses FINAL)
N_FAIR_RUNS_BISECT       = 1     # single deterministic run during bisection
N_FAIR_RUNS_FINAL        = 100   # full ensemble for the confirmatory run
N_TIPPING_SAMPLES_BISECT = 100   # fast samples during bisection
N_TIPPING_SAMPLES_FINAL  = 1000  # full samples for the final run

# Respect the SLURM allocation (cpus-per-task) rather than every core on the node;
# fall back to the process's CPU affinity when not under SLURM.
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def write_mini_csvs(df_ndc_rows, df_candidate_rows, emissions_path, forcing_path,
                    df_forcing_template, year_cols):
    """
    Write a 2-scenario emissions CSV and matching forcing CSV for fair.
    Copies Volcanic/Solar rows from df_forcing_template for any missing scenarios.

    The cleaned pipeline's NGFS emissions are ALREADY extended to 2110.5 (the real
    NGFS extension used by 04/08), so no extra extrapolation is done here.
    """
    os.makedirs(os.path.dirname(emissions_path), exist_ok=True)
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

    # per-member 1850–1901 rebase -> 20-yr centred running mean, via the single
    # shared implementation in src/temperature.py (same as resilience.py and the
    # tipping forcing), replacing the old ensemble-mean scalar baseline so the
    # calculator's resilience matches the main pipeline member-for-member.
    run_mean = processed_temperature(raw_temp.stack(member=("run", "config")))

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
        # NaN-aware: failed tipping solves are NaN (never fillna(0)); drop them here
        Q[b] = np.nanmean(T[ci][:, ri][:, :, si])
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
    rate_threshold   : °C/yr rate limit (10-yr mean of annual diffs of the 20-yr running
                       mean, ~30-yr kernel) over 2000–year_target
    year_target      : year at which temperature must be below threshold
    config_subset    : list of config values to include (None = all)

    Returns
    -------
    resilience_mean : float
    """
    rm = run_mean_temp

    climate_violation = (rm.sel(timebounds=slice(year_target, None)) > temp_threshold).any("timebounds")
    rate_of_change    = rm.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation    = (rate_of_change.sel(timebounds=slice(2000, 2096)) > rate_threshold).any("timebounds")

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
    resilience_binary = resilience_tensor(
        run_mean_temp, prob_any_tipping_sample,
        temp_threshold, rate_threshold, year_target, config_subset=config_subset,
    )
    T = resilience_binary.values.astype(np.float32)
    ci = fast_full_crossed_bootstrap(T, B=n_bootstrap)
    ci["ci_half_width"] = (ci["ci_high"] - ci["ci_low"]) / 2.0
    return ci


def resilience_tensor(run_mean_temp, prob_any_tipping_sample,
                      temp_threshold=DEFAULT_TEMP_THRESHOLD,
                      rate_threshold=DEFAULT_RATE_THRESHOLD,
                      year_target=DEFAULT_YEAR_TARGET,
                      config_subset=None):
    """Per-(config, run, sample) resilience field: P(climate OK) x (1 - tip) per sample.

    Shared by the single-scenario bootstrap and the paired candidate-minus-NDC gain
    bootstrap. Returns a DataArray transposed to (config, run, sample).
    """
    rm = run_mean_temp
    climate_violation = (rm.sel(timebounds=slice(year_target, None)) > temp_threshold).any("timebounds")
    rate_of_change    = rm.diff("timebounds").rolling(timebounds=10, center=True).mean()
    rate_violation    = (rate_of_change.sel(timebounds=slice(2000, 2096)) > rate_threshold).any("timebounds")

    binary_resilient = ~(climate_violation | rate_violation)
    binary_resilient = binary_resilient.unstack("member").astype(float)  # (run, config)

    resilience_binary = binary_resilient * (1.0 - prob_any_tipping_sample)
    if config_subset is not None:
        resilience_binary = resilience_binary.sel(config=config_subset)
    return resilience_binary.transpose("config", "run", "sample")


def cell_vs_full_gain_bootstrap(T_cell, T_full, cell_config_pos, cell_sample_pos, B=1000, seed=0):
    """Crossed-bootstrap CI for an UNPAIRED gain = PERI(cell candidate) − PERI(full NDC).

    The gain is unpaired (candidate cell vs full-ensemble NDC — different scenarios), but
    configs, runs and LHS samples are the SAME physical units in both tensors: the 841
    calibration configs, the per-run FAIR seeds (seed_start + i*99, fixed across runs), and
    the seed-1234 1000-sample LHS of which the cell is a T_eff/TCRE subset. So each
    bootstrap draw must resample (config, run, sample) ONCE and apply the same indices to
    both means — resampling the two independently would ignore that shared variation and
    inflate the difference CI.

    The cell tensor (``T_cell``, over a subset of configs/samples) is laid onto the full
    (config, run, sample) grid at ``cell_config_pos`` × all runs × ``cell_sample_pos`` and
    padded with NaN elsewhere, so one shared index set drives both and the cell mean
    (NaN-aware) averages only the drawn units that fall in the cell.
    """
    rng = np.random.default_rng(seed)
    Tf = T_full.transpose("config", "run", "sample").values.astype(np.float32)
    Tc = T_cell.transpose("config", "run", "sample").values.astype(np.float32)
    nc_f, nr, ns_f = Tf.shape
    cell_config_pos = np.asarray(cell_config_pos)
    cell_sample_pos = np.asarray(cell_sample_pos)
    assert Tc.shape == (len(cell_config_pos), nr, len(cell_sample_pos)), (
        f"cell tensor {Tc.shape} inconsistent with positions "
        f"({len(cell_config_pos)}, {nr}, {len(cell_sample_pos)})")

    Tc_full = np.full((nc_f, nr, ns_f), np.nan, dtype=np.float32)   # cell on the full grid
    Tc_full[np.ix_(cell_config_pos, np.arange(nr), cell_sample_pos)] = Tc

    Q = np.empty(B, dtype=np.float32)
    for b in trange(B, desc="Bootstrap(cell-full)"):
        ci = rng.integers(0, nc_f, nc_f); ri = rng.integers(0, nr, nr); si = rng.integers(0, ns_f, ns_f)
        full = np.nanmean(Tf[ci][:, ri][:, :, si])
        cell = np.nanmean(Tc_full[ci][:, ri][:, :, si])            # NaNs outside the cell ignored
        Q[b] = cell - full
    return {
        "mean": float(Q.mean()),
        "ci_low": float(np.quantile(Q, 0.025)),
        "ci_high": float(np.quantile(Q, 0.975)),
        "ci_half_width": float((np.quantile(Q, 0.975) - np.quantile(Q, 0.025)) / 2.0),
        "samples": Q,
    }


def compute_gain_bootstrap(rm_cand, pas_cand, rm_base, pas_base,
                           temp_threshold=DEFAULT_TEMP_THRESHOLD,
                           rate_threshold=DEFAULT_RATE_THRESHOLD,
                           year_target=DEFAULT_YEAR_TARGET,
                           config_subset=None, n_bootstrap=1000):
    """Crossed-bootstrap CI for the PAIRED gain = resilience(candidate) - resilience(NDC).

    Both resilience tensors come from the same FAIR run (same seeds, configs) and the
    same LHS tipping samples, so the difference is taken cell-by-cell on the shared
    (config, run, sample) grid before resampling.
    """
    T_cand = resilience_tensor(rm_cand, pas_cand, temp_threshold, rate_threshold,
                               year_target, config_subset=config_subset)
    T_base = resilience_tensor(rm_base, pas_base, temp_threshold, rate_threshold,
                               year_target, config_subset=config_subset)
    D = (T_cand - T_base).transpose("config", "run", "sample").values.astype(np.float32)
    ci = fast_full_crossed_bootstrap(D, B=n_bootstrap)
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


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def assert_configs_match_ensemble(temperature_file="output/all_scenarios_temperature.nc",
                                  params_file=FAIR_PARAMS):
    """Raise unless the parameter file and the concatenated ensemble share configs.

    The calculator drives FAIR from ``params_file`` and compares its resilience to
    the main-pipeline ensemble (same configs). If the two are from different
    calibrations the config sets diverge and every per-config comparison is silently
    misaligned; this guard makes that a hard error.
    """
    df, _ = load_fair_params(params_file)
    cfg = xr.open_dataset(temperature_file).config.values
    if set(df.index) != set(cfg):
        raise ValueError(
            f"config mismatch: {params_file} has {len(df.index)} configs, "
            f"{temperature_file} has {len(cfg)}; sets differ -- parameter file and "
            "ensemble are from different calibrations."
        )


def gain_brackets_target(gain_lo, gain_hi, target_gain):
    """Whether a monotone-increasing ``gain(R)`` has its ``gain == target_gain`` root
    inside the bracket, i.e. ``gain(lo) <= target <= gain(hi)``.

    The finder's bisection assumes more reduction yields more resilience gain and that
    the seeded bracket straddles the target. If it does not (both endpoints above or
    both below), plain bisection converges silently to a bracket edge; callers use this
    to detect that and widen the bracket.
    """
    return gain_lo <= target_gain <= gain_hi


def evaluate_candidate(reduction_frac, df_emissions, years, year_cols,
                       base_scenario_name, target_year, df_forcing_template,
                       params_file, species_configs_file, iter_dir,
                       n_runs, n_samples,
                       fair_time=FAIR_TIME,
                       temp_threshold=DEFAULT_TEMP_THRESHOLD,
                       rate_threshold=DEFAULT_RATE_THRESHOLD,
                       year_target=DEFAULT_YEAR_TARGET,
                       config_subset=None,
                       sample_subset=None,
                       run_base=True,
                       candidate_name=None,
                       lhs_params=None,
                       return_bootstrap=False,
                       paired_gain_bootstrap=True,
                       return_tensors=False,
                       n_bootstrap=1000):
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
    sample_subset     : optional index array into the LHS tipping samples (e.g. a
                        T_eff tercile). When given, lhs_params is sliced to those
                        indices and the tipping run uses n_samples = len(sample_subset);
                        None leaves the tipping sampling unchanged.
    run_base          : if False, skip the NDC base aggregation + its tipping run (and
                        ndc_resilience / achieved_gain / T_ndc / paired bootstrap).
                        grid/tcre modes pass False (they need only the candidate); gain
                        mode keeps True (the paired NDC gain).
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
    # natural-background floor per species (FaIR baseline_emissions, converted to
    # each variable's CSV unit), matching the Current-Policies branch-off grid (02)
    # so panels a and b/c clamp consistently.
    baseline_map = build_floor_map(df_emissions, species_file=species_configs_file)
    df_candidate = build_candidate_scenario(
        df_emissions, years, year_cols,
        base_scenario_name, target_year, reduction_frac, candidate_name,
        baseline_map=baseline_map,
    )

    # ── 2. write mini CSVs ───────────────────────────────────────────────────
    write_mini_csvs(df_base, df_candidate, emissions_path, forcing_path,
                    df_forcing_template, year_cols)

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

    # ── 4. aggregate temperature: candidate AND the NDC base, from the SAME runs
    raw_cand, rm_cand, configs, runs = aggregate_temperature_from_runs(
        run_dir, candidate_name, max_runs=n_runs
    )
    # The NDC base is only needed for the PAIRED gain (gain mode); grid/tcre use the
    # candidate alone (gain vs the externally-built full NDC). run_base=False skips the
    # base aggregation + its tipping run, halving the per-call tipping cost.
    raw_base = rm_base = None
    if run_base:
        raw_base, rm_base, _, _ = aggregate_temperature_from_runs(
            run_dir, base_scenario_name, max_runs=n_runs
        )

    # ── 5. tipping probabilities, driven by the PROCESSED temperature exactly as
    #       src/run_tipping.py (per-member 1850-1900 rebase + 20-yr running mean,
    #       then held from the last valid year to T_END inside compute_tip_prob) ─
    tip_configs = configs if config_subset is None else [c for c in configs if c in set(config_subset)]
    if lhs_params is None:
        lhs_params = sample_lhs_params(n_samples=N_TIPPING_SAMPLES_FINAL)

    # restrict to a subset of LHS samples (e.g. a T_eff tercile) by slicing the
    # params dict; the tipping run then has n_tip = len(sample_subset) samples.
    n_tip = n_samples
    if sample_subset is not None:
        sub = np.asarray(sample_subset)
        lhs_params = {k: np.asarray(v)[sub] for k, v in lhs_params.items()}
        n_tip = len(sub)

    pa_cand, pas_cand, _ = compute_tip_prob(
        processed_temperature(raw_cand), lhs_params,
        n_samples=n_tip, configs=tip_configs, runs=runs,
    )
    pa_base = pas_base = None
    if run_base:
        pa_base, pas_base, _ = compute_tip_prob(
            processed_temperature(raw_base), lhs_params,
            n_samples=n_tip, configs=tip_configs, runs=runs,
        )

    # ── 6. resilience + PAIRED gain (candidate - NDC from the same run/seeds/LHS) ─
    achieved_resilience = compute_resilience_scalar(
        rm_cand, pa_cand, temp_threshold, rate_threshold, year_target,
        config_subset=config_subset,
    )
    if run_base:
        ndc_resilience = compute_resilience_scalar(
            rm_base, pa_base, temp_threshold, rate_threshold, year_target,
            config_subset=config_subset,
        )
        achieved_gain = achieved_resilience - ndc_resilience
    else:
        ndc_resilience = np.nan      # base not run (grid/tcre use the external full NDC)
        achieved_gain = np.nan

    # emissions-side reductions for the Fig-3 panel-b annotations (df_base = the
    # NDC baseline; cheap, no FAIR/tipping)
    co2_GtCO2 = compute_co2_reduction_GtCO2(df_base, df_candidate, target_year)
    vs30_abs, vs30_frac, ndc_2030 = compute_reduction_vs_2030ndc(df_base, df_candidate, target_year)

    result = {
        "reduction_frac": reduction_frac,
        "candidate_name": candidate_name,
        "target_year": target_year,
        "achieved_resilience": achieved_resilience,
        "ndc_resilience": ndc_resilience,       # cross-check vs the 09 summary value
        "achieved_gain": achieved_gain,         # paired difference, same run
        "co2_GtCO2": co2_GtCO2,
        "reduction_vs_ndc2030_GtCO2": vs30_abs,
        "reduction_vs_ndc2030_frac": vs30_frac,
        "ndc_2030_GtCO2": ndc_2030,
        "n_runs": n_runs,
        "n_samples": n_tip,
    }

    if return_tensors:
        # per-(config, run, sample) resilience field for the candidate (and the NDC,
        # only when run_base); grid/tcre bootstrap the candidate cell vs an external T_full.
        result["T_cand"] = resilience_tensor(
            rm_cand, pas_cand, temp_threshold, rate_threshold, year_target,
            config_subset=config_subset)
        if run_base:
            result["T_ndc"] = resilience_tensor(
                rm_base, pas_base, temp_threshold, rate_threshold, year_target,
                config_subset=config_subset)

    if return_bootstrap:
        if paired_gain_bootstrap:
            if not run_base:
                raise ValueError("paired_gain_bootstrap requires run_base=True (needs the NDC run)")
            # gain mode: CI of the PAIRED candidate-minus-NDC difference (same run)
            result["bootstrap"] = compute_gain_bootstrap(
                rm_cand, pas_cand, rm_base, pas_base,
                temp_threshold, rate_threshold, year_target,
                config_subset=config_subset, n_bootstrap=n_bootstrap,
            )
        else:
            # gfp mode: CI of the CANDIDATE's resilience (the caller subtracts the
            # full-ensemble 09 baseline point estimate to get the gain CI)
            result["bootstrap"] = compute_resilience_with_bootstrap(
                rm_cand, pas_cand,
                temp_threshold, rate_threshold, year_target,
                config_subset=config_subset, n_bootstrap=n_bootstrap,
            )

    return result
