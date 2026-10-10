"""13_gfp_percentile.py — required NDC reduction vs Earth-system fragility (GFP).

Ported from ERI/code/improved_X%_calculator/gfp_percentile_bisection.py.

For 5 GFP percentiles (10/30/50/70/90th) it takes a ±GFP_HALFWIDTH window of configs in
normalised GFP space and, for each gain target (+1%, +10%), bisects the additional
NDC reduction at TARGET_YEAR=2035.5 (no cap, bisect_hi=0.99) needed to reach the
target gain over that config subset. Feeds panel (c) of the calculator figure.

GFP scores are the config-level PCA feedback score (climate-model component only),
via calculator.compute_fragility_masks. Resilience over the subset is compared to
the FULL NDC baseline (from 09) — same reference as the original.

Reduced resolution during search (like the finder): N_RUNS FAIR runs + N_SAMPLES
tipping samples. Heavy -> SLURM.

Output: output/calculator/gfp_percentile/gfp_percentile_results.csv
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)

from calculator import (
    evaluate_candidate, compute_fragility_masks, assert_configs_match_ensemble,
    gain_brackets_target,
    N_FAIR_RUNS_BISECT, N_TIPPING_SAMPLES_BISECT, N_TIPPING_SAMPLES_FINAL,
)
from tipping_params import sample_lhs_params
from fair_config import FAIR_PARAMS, SPECIES_CONFIGS_NGFS

# ── configuration ───────────────────────────────────────────────────────────────
TARGET_YEAR   = 2035.5
GAIN_LEVELS   = [0.01, 0.10]
PERCENTILES   = [10, 30, 50, 70, 90]
GFP_HALFWIDTH         = 0.10            # ± half-window in normalised GFP space [0,1]
TEMP_THRESHOLD = 1.5
RATE_THRESHOLD = 0.04
YEAR_TARGET    = 2100

BASE_SCENARIO       = "REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_(NDCs)"
BASE_SCENARIO_CLEAN = BASE_SCENARIO.replace("(", "").replace(")", "")

EMISSIONS_FILE = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
FORCING_FILE   = "data/processed/NGFS_forcing.csv"
PARAMS_FILE    = FAIR_PARAMS            # single calibration (v1.4.0) via src/fair_config
SPECIES_FILE   = SPECIES_CONFIGS_NGFS
WORKSPACE      = "output/calculator/gfp_percentile"

# single-stage bisection (no cap)
BISECT_LO = 0.0
BISECT_HI = 0.99
TOL       = 0.004
MAX_ITER  = 12
BRACKET   = 0.06                      # ± around an old seed (when available)
N_RUNS    = 10                        # higher-res search so confirmed gains land near target (was 3; overshot)
N_SAMPLES = N_TIPPING_SAMPLES_BISECT  # 100

# old converged reductions per (percentile, gain) — seed the bracket so we only refine
OLD_GFP_CSV = "data/reference/gfp_percentile_seeds.csv"
gfp_seeds = {}
if os.path.exists(OLD_GFP_CSV):
    _sdf = pd.read_csv(OLD_GFP_CSV)
    gfp_seeds = {(int(r["percentile"]), round(float(r["gain_target"]), 2)): float(r["reduction_frac"])
                 for _, r in _sdf.iterrows()}
    print(f"Loaded {len(gfp_seeds)} GFP-percentile seeds", flush=True)

os.makedirs(WORKSPACE, exist_ok=True)
OUTPUT_CSV = os.path.join(WORKSPACE, "gfp_percentile_results.csv")

# ── canonical NDC baseline resilience (from 09) ───────────────────────────────────
rate_str = f"{int(round(RATE_THRESHOLD * 100)):02d}"
summary_csv = (
    f"output/resilience/resilience_conditions_time{YEAR_TARGET}"
    f"_temp{TEMP_THRESHOLD}_rate{rate_str}/summary/"
    f"resilience_summary_time{YEAR_TARGET}_temp{TEMP_THRESHOLD}_rate{rate_str}.csv"
)
summary = pd.read_csv(summary_csv)
baseline_resilience = float(summary.loc[summary["scenario"] == BASE_SCENARIO_CLEAN, "resilience"].values[0])
print(f"NDC baseline resilience (full) = {baseline_resilience:.6f}", flush=True)

# ── emissions / forcing / GFP scores ──────────────────────────────────────────────
df_emissions = pd.read_csv(EMISSIONS_FILE)
year_cols = [c for c in df_emissions.columns if c.replace(".", "", 1).isdigit()]
years = np.array([float(c) for c in year_cols])
df_forcing = pd.read_csv(FORCING_FILE)

df_configs = pd.read_csv(PARAMS_FILE, index_col=0)
all_configs = df_configs.index.tolist()
assert_configs_match_ensemble()   # params file must match the ensemble's config set
_, gf_scores = compute_fragility_masks(all_configs, PARAMS_FILE)
print(f"GFP scores for {len(gf_scores)} configs (range [{gf_scores.min():.3f}, {gf_scores.max():.3f}])", flush=True)

lhs_params = sample_lhs_params(n_samples=N_TIPPING_SAMPLES_FINAL)

# ── resume ──────────────────────────────────────────────────────────────────────
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done    = set(zip(done_df["percentile"], done_df["gain_target"]))
    records = done_df.to_dict("records")
    print(f"[cache] {len(records)} existing results", flush=True)
else:
    done, records = set(), []


def _bisect_subset(cfg_subset, gain, tag, lo, hi):
    """Single-stage bisection of the reduction over a config subset, given a bracket."""
    def _eval(R, label):
        iter_dir = os.path.join(WORKSPACE, tag, label)
        return evaluate_candidate(
            R, df_emissions, years, year_cols,
            BASE_SCENARIO, TARGET_YEAR, df_forcing,
            PARAMS_FILE, SPECIES_FILE, iter_dir,
            n_runs=N_RUNS, n_samples=N_SAMPLES,
            temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD,
            year_target=YEAR_TARGET, config_subset=cfg_subset,
            lhs_params=lhs_params, extend=False,
            candidate_name=f"NDC_calc__{tag}_r{R:.6f}",
        )

    # bracket-sign check (ported from 12): the monotone gain(R) must straddle the
    # target gain across [lo, hi]; else widen to [0, 0.99]; else raise.
    g_lo = _eval(lo, "bracket_lo")["achieved_gain"]
    g_hi = _eval(hi, "bracket_hi")["achieved_gain"]
    if not gain_brackets_target(g_lo, g_hi, gain):
        print(f"    [warn] bracket [{lo:.4f},{hi:.4f}] gains ({g_lo:.4f},{g_hi:.4f}) do not "
              f"straddle target {gain}; widening to [0, 0.99]", flush=True)
        lo, hi = 0.0, 0.99
        g_lo = _eval(lo, "bracket_lo_wide")["achieved_gain"]
        g_hi = _eval(hi, "bracket_hi_wide")["achieved_gain"]
        if not gain_brackets_target(g_lo, g_hi, gain):
            raise RuntimeError(
                f"target gain {gain} not bracketed on [0, 0.99] for {tag} "
                f"(gain(0)={g_lo:.4f}, gain(0.99)={g_hi:.4f})"
            )

    res = None
    for it in range(MAX_ITER):
        mid = 0.5 * (lo + hi)
        res = _eval(mid, f"iter_{it:02d}")
        gain_now = res["achieved_gain"]   # paired candidate - NDC from the same run
        if gain_now < gain:
            lo = mid
        else:
            hi = mid
        print(f"    iter {it:02d}: R={mid:.4f} gain={gain_now:.5f} [lo={lo:.4f}, hi={hi:.4f}]", flush=True)
        if hi - lo < TOL:
            break

    converged = 0.5 * (lo + hi)
    res = _eval(converged, "converged")   # report at the converged R, not the last step
    return converged, res


# ── main loop ─────────────────────────────────────────────────────────────────────
for pct in PERCENTILES:
    gf_target = float(np.percentile(gf_scores.values, pct))
    mask = (gf_scores >= gf_target - GFP_HALFWIDTH) & (gf_scores <= gf_target + GFP_HALFWIDTH)
    cfg_subset = gf_scores[mask].index.tolist()
    print(f"\n── {pct:2d}th pct | GFP={gf_target:.3f} | {len(cfg_subset)} configs ──", flush=True)

    for gain in GAIN_LEVELS:
        if (pct, gain) in done:
            print(f"  [cache] pct={pct} gain={gain} — skip", flush=True)
            continue
        tag = f"pct{pct:02d}_gain{int(gain*100):02d}_ty{int(TARGET_YEAR)}"
        # seed the bracket from the old GFP-percentile result when available
        _seed = gfp_seeds.get((pct, round(gain, 2)))
        if _seed is not None:
            lo, hi = max(BISECT_LO, _seed - BRACKET), min(BISECT_HI, _seed + BRACKET)
            print(f"  seed R={_seed:.4f} -> bracket [{lo:.4f}, {hi:.4f}]", flush=True)
        else:
            lo, hi = BISECT_LO, BISECT_HI
        red, res = _bisect_subset(cfg_subset, gain, tag, lo, hi)
        records.append({
            "percentile": pct,
            "gf_value": gf_target,
            "n_configs": len(cfg_subset),
            "gain_target": gain,
            "reduction_frac": red,
            "achieved_gain": res["achieved_gain"] if res else np.nan,       # paired
            "ndc_resilience": res["ndc_resilience"] if res else np.nan,     # cross-check
            "co2_GtCO2": res["co2_GtCO2"] if res else np.nan,
        })
        done.add((pct, gain))
        pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)  # incremental
        print(f"  → pct={pct} gain={gain}: R={red*100:.1f}%", flush=True)

df_res = pd.DataFrame(records).sort_values(["gain_target", "percentile"])
df_res.to_csv(OUTPUT_CSV, index=False)
print(f"\nGFP-percentile complete -> {OUTPUT_CSV}")
print(df_res.to_string(index=False))
