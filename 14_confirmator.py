"""14_confirmator.py — full-resolution confirmation of ALL finder reductions.

Confirms the finder outputs at full resolution — 100 FAIR Monte Carlo runs +
1000-sample tipping + crossed bootstrap CI — the publishable numbers.

Modes (positional arg):
    python 14_confirmator.py 0.01          # confirm 12_finder +1% (per target year)
    python 14_confirmator.py 0.10          # confirm 12_finder +10%
    python 14_confirmator.py 0.10 2035.5   # just one target year
    python 14_confirmator.py gfp           # confirm 13_gfp_percentile (per pct × gain)

Gain modes are resumable per target year (per-year CSVs). GFP mode is resumable per
(percentile, gain). Heavy -> submit via SLURM.

Outputs (output/calculator/confirmator/):
    confirmed_results_gain<G>.csv          # gain modes
    confirmed_gfp_percentile.csv           # gfp mode
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)

from calculator import (
    evaluate_candidate, compute_fragility_masks, assert_configs_match_ensemble,
    N_FAIR_RUNS_FINAL, N_TIPPING_SAMPLES_FINAL,
)
from tipping_params import sample_lhs_params
from fair_config import FAIR_PARAMS, SPECIES_CONFIGS_NGFS

# ── mode (positional) ─────────────────────────────────────────────────────────────
_mode = sys.argv[1] if len(sys.argv) > 1 else "0.01"
GFP_MODE = (_mode == "gfp")
TARGET_GAIN = 0.01 if GFP_MODE else float(_mode)
_REQUESTED_YEARS = [float(a) for a in sys.argv[2:]]  # gain mode only

# ── configuration (keep in sync with 12/13) ───────────────────────────────────────
TEMP_THRESHOLD  = 1.5
RATE_THRESHOLD  = 0.04
YEAR_TARGET     = 2100

BASE_SCENARIO       = "REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_(NDCs)"
BASE_SCENARIO_CLEAN = BASE_SCENARIO.replace("(", "").replace(")", "")

EMISSIONS_FILE = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
FORCING_FILE   = "data/processed/NGFS_forcing.csv"
PARAMS_FILE    = FAIR_PARAMS            # single calibration (v1.4.0) via src/fair_config
SPECIES_FILE   = SPECIES_CONFIGS_NGFS
WORKSPACE      = "output/calculator/confirmator"

# GFP-mode settings (must match 13_gfp_percentile.py)
GFP_FINDER_CSV = "output/calculator/gfp_percentile/gfp_percentile_results.csv"
GFP_TARGET_YEAR = 2035.5
GFP_HALFWIDTH       = 0.10

N_BOOTSTRAP = 1000
os.makedirs(WORKSPACE, exist_ok=True)

# ── canonical NDC baseline resilience (from 09) ───────────────────────────────────
rate_str = f"{int(round(RATE_THRESHOLD * 100)):02d}"
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
print(f"NDC baseline resilience = {baseline_resilience:.6f}  |  mode={_mode}", flush=True)

# ── emissions / forcing / full-res LHS ─────────────────────────────────────────────
df_emissions = pd.read_csv(EMISSIONS_FILE)
year_cols = [c for c in df_emissions.columns if c.replace(".", "", 1).isdigit()]
years = np.array([float(c) for c in year_cols])
df_forcing = pd.read_csv(FORCING_FILE)
assert_configs_match_ensemble()   # params file must match the ensemble's config set
lhs_params = sample_lhs_params(n_samples=N_TIPPING_SAMPLES_FINAL)


# ══════════════════════════════════════════════════════════════════════════════════
# GFP mode — confirm each (percentile, gain) reduction over its config subset
# ══════════════════════════════════════════════════════════════════════════════════
if GFP_MODE:
    if not os.path.exists(GFP_FINDER_CSV):
        raise FileNotFoundError(f"GFP finder output not found: {GFP_FINDER_CSV} (run 13 first).")
    gfp = pd.read_csv(GFP_FINDER_CSV)

    df_configs = pd.read_csv(PARAMS_FILE, index_col=0)
    _, gf_scores = compute_fragility_masks(df_configs.index.tolist(), PARAMS_FILE)

    out_csv = os.path.join(WORKSPACE, "confirmed_gfp_percentile.csv")
    done = set()
    records = []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        records = prev.to_dict("records")
        done = set(zip(prev["percentile"], prev["gain_target"]))

    for _, r in gfp.sort_values(["gain_target", "percentile"]).iterrows():
        pct = int(r["percentile"]); gain = round(float(r["gain_target"]), 2)
        red = float(r["reduction_frac"])
        if (pct, gain) in done:
            print(f"[cache] pct={pct} gain={gain} — skip", flush=True)
            continue
        gf_target = float(np.percentile(gf_scores.values, pct))
        mask = (gf_scores >= gf_target - GFP_HALFWIDTH) & (gf_scores <= gf_target + GFP_HALFWIDTH)
        cfg_subset = gf_scores[mask].index.tolist()
        print(f"\n══ confirm pct={pct} gain={gain} R={red:.4f} | {len(cfg_subset)} configs "
              f"({N_FAIR_RUNS_FINAL} runs × {N_TIPPING_SAMPLES_FINAL} samples) ══", flush=True)
        iter_dir = os.path.join(WORKSPACE, "gfp", f"pct{pct:02d}_gain{int(gain*100):02d}")
        res = evaluate_candidate(
            red, df_emissions, years, year_cols,
            BASE_SCENARIO, GFP_TARGET_YEAR, df_forcing,
            PARAMS_FILE, SPECIES_FILE, iter_dir,
            n_runs=N_FAIR_RUNS_FINAL, n_samples=N_TIPPING_SAMPLES_FINAL,
            temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD,
            year_target=YEAR_TARGET, config_subset=cfg_subset,
            lhs_params=lhs_params, extend=False,
            return_bootstrap=True, n_bootstrap=N_BOOTSTRAP,
            candidate_name=f"NDC_calc__gfp_pct{pct:02d}_g{int(gain*100):02d}_r{red:.6f}",
        )
        boot = res["bootstrap"]   # paired candidate - NDC gain bootstrap (over subset)
        records.append({
            "percentile": pct, "gf_value": gf_target, "n_configs": len(cfg_subset),
            "gain_target": gain, "reduction_frac": red,
            "achieved_resilience": res["achieved_resilience"],
            "ndc_resilience": res["ndc_resilience"],
            "achieved_gain": res["achieved_gain"],
            "baseline_resilience": baseline_resilience,
            "bootstrap_gain": boot["mean"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "ci_half_width": boot["ci_half_width"],
            "co2_GtCO2": res["co2_GtCO2"],
            "n_runs": N_FAIR_RUNS_FINAL, "n_samples": N_TIPPING_SAMPLES_FINAL,
        })
        done.add((pct, gain))
        pd.DataFrame(records).to_csv(out_csv, index=False)  # incremental
        print(f"  confirmed pct={pct} gain={gain}: gain={records[-1]['achieved_gain']:.5f}", flush=True)

    print(f"\nGFP confirmation complete -> {out_csv}")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════════
# Gain mode — confirm the main finder reductions per target year
# ══════════════════════════════════════════════════════════════════════════════════
FINDER_CSV = f"output/calculator/finder/finder_results_gain{TARGET_GAIN}.csv"
if not os.path.exists(FINDER_CSV):
    raise FileNotFoundError(f"Finder output not found: {FINDER_CSV} (run 12 for gain {TARGET_GAIN} first).")
fdf = pd.read_csv(FINDER_CSV)
reductions = dict(zip(fdf["target_year"].astype(float),
                      fdf["converged_reduction_frac"].astype(float)))
print(f"Confirming finder reductions from {FINDER_CSV}: {reductions}", flush=True)

if _REQUESTED_YEARS:
    _requested = {round(float(a), 1) for a in _REQUESTED_YEARS}
    reductions = {ty: r for ty, r in reductions.items() if round(ty, 1) in _requested}
    if not reductions:
        raise ValueError(f"None of {sorted(_requested)} found among finder target years.")


def _peryear_csv(ty):
    return os.path.join(WORKSPACE, f"confirmed_results_gain{TARGET_GAIN}_ty{int(ty)}.csv")


for ty, red in sorted(reductions.items()):
    peryear_csv = _peryear_csv(ty)
    if os.path.exists(peryear_csv):
        print(f"[cache] target_year={ty} already done ({peryear_csv}) — skipping.", flush=True)
        continue

    print(f"\n══ confirming target_year={ty}  reduction_frac={red:.6f} "
          f"({N_FAIR_RUNS_FINAL} runs × {N_TIPPING_SAMPLES_FINAL} samples) ══", flush=True)
    iter_dir = os.path.join(WORKSPACE, f"gain{TARGET_GAIN}", f"ty{int(ty)}")
    res = evaluate_candidate(
        red, df_emissions, years, year_cols,
        BASE_SCENARIO, ty, df_forcing,
        PARAMS_FILE, SPECIES_FILE, iter_dir,
        n_runs=N_FAIR_RUNS_FINAL, n_samples=N_TIPPING_SAMPLES_FINAL,
        temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD,
        year_target=YEAR_TARGET, lhs_params=lhs_params, extend=False,
        return_bootstrap=True, n_bootstrap=N_BOOTSTRAP,
        candidate_name=f"NDC_calc__ty{int(ty)}_r{red:.6f}",
    )
    boot = res["bootstrap"]           # paired candidate - NDC gain bootstrap
    achieved_gain = res["achieved_gain"]
    # cross-check: this gain-mode run uses the full ensemble (config_subset=None), so
    # the calculator's NDC resilience must reproduce the 09 summary value exactly.
    if abs(res["ndc_resilience"] - baseline_resilience) > 1e-9:
        raise AssertionError(
            f"calculator NDC resilience {res['ndc_resilience']:.12f} != 09 baseline "
            f"{baseline_resilience:.12f} (diff {res['ndc_resilience'] - baseline_resilience:.2e}); "
            "the calculator and the main pipeline disagree for the NDC scenario."
        )
    print(f"  resilience={res['achieved_resilience']:.6f}  gain={achieved_gain:.6f}  "
          f"gain bootstrap mean={boot['mean']:.6f} [{boot['ci_low']:.6f}, {boot['ci_high']:.6f}]", flush=True)

    row = {
        "target_year": ty, "reduction_frac": red,
        "achieved_resilience": res["achieved_resilience"],
        "ndc_resilience": res["ndc_resilience"],
        "achieved_gain": achieved_gain, "baseline_resilience": baseline_resilience,
        "co2_GtCO2": res["co2_GtCO2"],
        "reduction_vs_ndc2030_GtCO2": res["reduction_vs_ndc2030_GtCO2"],
        "reduction_vs_ndc2030_frac": res["reduction_vs_ndc2030_frac"],
        "ndc_2030_GtCO2": res["ndc_2030_GtCO2"],
        # bootstrap is now over the PAIRED gain; ci_* are the gain CI directly
        "bootstrap_gain": boot["mean"], "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"], "ci_half_width": boot["ci_half_width"],
        "target_gain": TARGET_GAIN, "temp_threshold": TEMP_THRESHOLD, "rate_threshold": RATE_THRESHOLD,
        "n_runs": N_FAIR_RUNS_FINAL, "n_samples": N_TIPPING_SAMPLES_FINAL,
    }
    pd.DataFrame([row]).to_csv(peryear_csv, index=False)
    print(f"  wrote {peryear_csv}", flush=True)

# merge per-year CSVs into the combined result
_peryear_files = sorted(glob.glob(os.path.join(WORKSPACE, f"confirmed_results_gain{TARGET_GAIN}_ty*.csv")))
if _peryear_files:
    combined = pd.concat([pd.read_csv(f) for f in _peryear_files], ignore_index=True).sort_values("target_year")
    out_csv = os.path.join(WORKSPACE, f"confirmed_results_gain{TARGET_GAIN}.csv")
    combined.to_csv(out_csv, index=False)
    print(f"\nConfirmator merged {len(_peryear_files)} target year(s) -> {out_csv}")
