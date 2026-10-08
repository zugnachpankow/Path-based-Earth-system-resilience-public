"""12_finder.py — minimum additional NDC reduction for a target resilience gain.

Simplified single-stage bisection (drops the old calculator's Stage1/Stage2 split
and multiple epsilons). Seeded from the previously converged reduction fractions so
it only has to refine, at reduced resolution: 1 deterministic FAIR run + 100 tipping
samples per evaluation. The canonical NDC baseline resilience is read from the 09
resilience summary, so the gain is defined identically to the main pipeline.

Publishable finder. Heavy -> submit via SLURM (do not run on the login node).

Output: output/calculator/finder/finder_results_gain<G>.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve data/ and output/ from the repo root, not the job's cwd

from calculator import (
    evaluate_candidate,
    N_FAIR_RUNS_BISECT,
    N_TIPPING_SAMPLES_BISECT,
    N_TIPPING_SAMPLES_FINAL,
)
from tipping_params import sample_lhs_params

# ── configuration ───────────────────────────────────────────────────────────────
# positional gain: `python 12_finder.py 0.01` (default) or `... 0.10`
TARGET_GAIN     = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
TEMP_THRESHOLD  = 1.5
RATE_THRESHOLD  = 0.04
YEAR_TARGET     = 2100
TARGET_YEARS    = [2025.5, 2030.5, 2035.5, 2040.5]

BASE_SCENARIO       = "REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_(NDCs)"
BASE_SCENARIO_CLEAN = BASE_SCENARIO.replace("(", "").replace(")", "")

EMISSIONS_FILE = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
FORCING_FILE   = "data/processed/NGFS_forcing.csv"
PARAMS_FILE    = "data/raw/calibrated_constrained_parameters_calibration1.4.1.csv"
SPECIES_FILE   = "data/raw/species_configs_properties_NGFS.csv"

# Old converged reductions, per gain — used to seed the bracket so the finder only
# has to refine (fast). We have both +1% and +10% from the previous calculator runs.
SEEDS_BY_GAIN = {
    0.01: "data/reference/finder_seeds_gain0.01.csv",
    0.10: "data/reference/finder_seeds_gain0.10.csv",
}
SEEDS_CSV = SEEDS_BY_GAIN.get(round(TARGET_GAIN, 2))
WORKSPACE = "output/calculator/finder"

# bisection controls (single stage). Bracket is centred on the seed but widened,
# because the cleaned baseline differs from the one the seeds were tuned to.
BRACKET  = 0.06     # ± around the seed reduction fraction
TOL      = 0.002    # stop when the reduction-fraction interval is narrower than this
MAX_ITER = 10
# 5 stochastic realizations (was 1 deterministic): averages internal variability so
# the resilience estimate isn't quantized at ~1/n_config or biased vs the 100-run
# baseline. Every run already spans all 841 configs (the full ECS distribution), so
# n_runs only samples weather noise. Kept at 100 tipping samples per the cost tradeoff.
N_FINDER_RUNS    = 10  # higher-res search so confirmed gains land near the +1/+10% target
                       # (n_runs=3 systematically oversized R -> confirmed overshoot; lower if finder too slow)
N_FINDER_SAMPLES = N_TIPPING_SAMPLES_BISECT  # 100 tipping samples during finding

# ── canonical NDC baseline resilience (from 09) ───────────────────────────────────
rate_str = f"{int(round(RATE_THRESHOLD * 100)):02d}"  # 0.04 -> "04"
summary_csv = (
    f"output/resilience/resilience_conditions_time{YEAR_TARGET}"
    f"_temp{TEMP_THRESHOLD}_rate{rate_str}/summary/"
    f"resilience_summary_time{YEAR_TARGET}_temp{TEMP_THRESHOLD}_rate{rate_str}.csv"
)
summary = pd.read_csv(summary_csv)
row = summary[summary["scenario"] == BASE_SCENARIO_CLEAN]
if row.empty:
    raise ValueError(f"NDC baseline not found in {summary_csv}")
baseline_resilience = float(row["resilience"].values[0])
target_resilience   = baseline_resilience + TARGET_GAIN
print(f"NDC baseline resilience = {baseline_resilience:.6f}  "
      f"-> target = {target_resilience:.6f} (+{TARGET_GAIN})")

# ── seeds (previously converged reduction fractions) ──────────────────────────────
seeds = {}
if SEEDS_CSV and os.path.exists(SEEDS_CSV):
    sdf = pd.read_csv(SEEDS_CSV)
    seeds = dict(zip(sdf["target_year"].astype(float), sdf["converged_reduction_frac"].astype(float)))
    print(f"Loaded seeds for gain {TARGET_GAIN}: {seeds}")
else:
    print(f"[warn] seed file {SEEDS_CSV} not found — using 0.15 bracket centre.")

# ── emissions / forcing ───────────────────────────────────────────────────────────
df_emissions = pd.read_csv(EMISSIONS_FILE)
year_cols = [c for c in df_emissions.columns if c.replace(".", "", 1).isdigit()]
years = np.array([float(c) for c in year_cols])
df_forcing = pd.read_csv(FORCING_FILE)

# LHS tipping params drawn once at full size; evaluate_candidate uses the first
# N_FINDER_SAMPLES (matches the calculator's "generate max, use a slice").
lhs_params = sample_lhs_params(n_samples=N_TIPPING_SAMPLES_FINAL)

# ── bisection per target year ─────────────────────────────────────────────────────
# Seed the bracket from the old converged reductions wherever we have them (+1% and
# +10%) so the finder only refines. Fall back to the full uncapped bracket [0, 0.99]
# only for target years with no seed.
results = []
for ty in TARGET_YEARS:
    if ty in seeds:
        seed = seeds[ty]
        lo, hi = max(0.0, seed - BRACKET), seed + BRACKET
    else:
        seed = float("nan")
        lo, hi = 0.0, 0.99
    print(f"\n══ target_year={ty}  seed={seed:.4f}  bracket=[{lo:.4f}, {hi:.4f}] ══")

    achieved_resilience = np.nan
    achieved_gain = np.nan
    history = []
    it = 0
    for it in range(MAX_ITER):
        mid = 0.5 * (lo + hi)
        iter_dir = os.path.join(WORKSPACE, f"gain{TARGET_GAIN}", f"ty{int(ty)}", f"iter_{it:02d}")
        res = evaluate_candidate(
            mid, df_emissions, years, year_cols,
            BASE_SCENARIO, ty, df_forcing,
            PARAMS_FILE, SPECIES_FILE, iter_dir,
            n_runs=N_FINDER_RUNS, n_samples=N_FINDER_SAMPLES,
            temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD,
            year_target=YEAR_TARGET, lhs_params=lhs_params, extend=False,
            candidate_name=f"NDC_calc__ty{int(ty)}_r{mid:.6f}",
        )
        achieved_resilience = res["achieved_resilience"]
        achieved_gain = achieved_resilience - baseline_resilience

        if achieved_gain < TARGET_GAIN:
            lo = mid
        else:
            hi = mid

        history.append({"iteration": it, "R": mid,
                        "achieved_resilience": achieved_resilience,
                        "achieved_gain": achieved_gain, "lo": lo, "hi": hi})
        print(f"  iter {it:02d}: R={mid:.6f}  resilience={achieved_resilience:.6f}  "
              f"gain={achieved_gain:.6f}  target={TARGET_GAIN}  [lo={lo:.4f}, hi={hi:.4f}]")

        if hi - lo < TOL:
            break

    converged = 0.5 * (lo + hi)
    results.append({
        "target_year": ty,
        "converged_reduction_frac": converged,
        "achieved_resilience": achieved_resilience,
        "achieved_gain": achieved_gain,
        "baseline_resilience": baseline_resilience,
        "co2_GtCO2": res["co2_GtCO2"],
        "reduction_vs_ndc2030_GtCO2": res["reduction_vs_ndc2030_GtCO2"],
        "reduction_vs_ndc2030_frac": res["reduction_vs_ndc2030_frac"],
        "ndc_2030_GtCO2": res["ndc_2030_GtCO2"],
        "target_gain": TARGET_GAIN,
        "temp_threshold": TEMP_THRESHOLD,
        "rate_threshold": RATE_THRESHOLD,
        "iterations": it + 1,
        "n_runs": N_FINDER_RUNS,
        "n_samples": N_FINDER_SAMPLES,
    })

    # incremental save — rewrite the combined CSV after each target year so a
    # mid-run failure doesn't lose completed years.
    os.makedirs(WORKSPACE, exist_ok=True)
    out_csv = os.path.join(WORKSPACE, f"finder_results_gain{TARGET_GAIN}.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)

print(f"\nFinder complete. Results -> {out_csv}")
