"""14_confirmator.py — full-resolution confirmation of the finder reductions.

Confirms finder outputs at full resolution — 100 FAIR Monte Carlo runs + 1000-sample
tipping + crossed bootstrap CI — the publishable numbers.

Modes (positional arg):
    python 14_confirmator.py 0.01          # confirm 12_finder +1pp (per target year)
    python 14_confirmator.py 0.10          # confirm 12_finder +10pp
    python 14_confirmator.py 0.10 2035.5   # just one target year
    python 14_confirmator.py grid          # confirm 13_fragility_finder grid (Fig 4c 3×3)
    python 14_confirmator.py tcre          # confirm 13_fragility_finder tcre windows (SI)

Fragility modes (grid/tcre) confirm each cell/window at its finder R* over the cell's
configs (TCRE tercile/window, from 841) × the cell's T_eff tercile of the 1000-sample
set (grid) or all samples (tcre). The gain is vs the 09 full-ensemble NDC PERI and its
CI is an UNPAIRED crossed bootstrap of (cell PERI − full NDC PERI). Cells whose finder
status is "max_cut_insufficient" are confirmed at R=0.99 and additionally report the
minimum CO2 FFI / AFOLU over 2035–2100 (to confirm the emission floor binds).

Gain modes are resumable per target year; fragility modes per cell/window. Heavy -> SLURM.

Outputs (output/calculator/confirmator/):
    confirmed_results_gain<G>.csv          # gain modes
    confirmed_grid.csv                     # grid mode
    confirmed_tcre_windows.csv             # tcre mode
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
    evaluate_candidate, assert_configs_match_ensemble, cell_vs_full_gain_bootstrap,
    N_FAIR_RUNS_FINAL, N_TIPPING_SAMPLES_FINAL,
)
from tipping_params import sample_lhs_params
from fair_config import FAIR_PARAMS, SPECIES_CONFIGS_NGFS

# ── mode (positional) ─────────────────────────────────────────────────────────────
_mode = sys.argv[1] if len(sys.argv) > 1 else "0.01"
FRAGILITY_MODE = _mode in ("grid", "tcre")
TARGET_GAIN = None if FRAGILITY_MODE else float(_mode)
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

# fragility-mode settings (must match 13_fragility_finder.py)
FRAG_TARGET_YEAR = 2035.5
GRID_FINDER_CSV  = "output/calculator/fragility_finder/grid_results.csv"
TCRE_FINDER_CSV  = "output/calculator/fragility_finder/tcre_window_results.csv"
TERCILE_Q = [100.0 / 3.0, 200.0 / 3.0]
MAX_CUT_R = 0.99

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
# Fragility modes (grid / tcre) — confirm 13's cells/windows at full resolution
# ══════════════════════════════════════════════════════════════════════════════════
if FRAGILITY_MODE:
    tcre = pd.read_csv("output/feedback/tcre.csv", index_col="config")["TCRE"]
    teff_1000 = pd.read_csv("output/feedback/t_eff_lhs1000.csv", index_col="sample")["T_eff"]
    tcre_bounds = np.percentile(tcre.values, TERCILE_Q)
    teff_bounds = np.nanpercentile(teff_1000.values, TERCILE_Q)
    teff_tier_1000 = np.digitize(teff_1000.values, teff_bounds)   # per 1000-sample
    TERCILES = ["low", "medium", "high"]

    co2_year_cols = [c for c in year_cols if 2035.0 <= float(c) <= 2100.0]

    def _min_floor_co2(iter_dir, candidate_name):
        """Min CO2 FFI and CO2 AFOLU (World) over 2035–2100 for the candidate (floor check)."""
        em = pd.read_csv(os.path.join(iter_dir, "emissions.csv"))
        cand = em[(em["scenario"] == candidate_name) & (em["region"] == "World")]
        out = {}
        for var in ("CO2 FFI", "CO2 AFOLU"):
            v = cand[cand["variable"] == var][co2_year_cols]
            out[var] = float(v.values.min()) if not v.empty else np.nan
        return out["CO2 FFI"], out["CO2 AFOLU"]

    # full-ensemble NDC resilience tensor at full resolution (R=0, no subsets) — the
    # reference for every cell's unpaired gain bootstrap; also cross-checks the 09 value.
    base = evaluate_candidate(
        0.0, df_emissions, years, year_cols, BASE_SCENARIO, FRAG_TARGET_YEAR, df_forcing,
        PARAMS_FILE, SPECIES_FILE, os.path.join(WORKSPACE, _mode, "baseline_full"),
        n_runs=N_FAIR_RUNS_FINAL, n_samples=N_TIPPING_SAMPLES_FINAL,
        temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD, year_target=YEAR_TARGET,
        lhs_params=lhs_params, return_tensors=True, candidate_name="NDC_calc__baseline_full",
    )
    T_full = base["T_ndc"]
    full_cfg_pos = {c: k for k, c in enumerate(T_full.config.values)}   # config id -> full-grid row
    n_samp_full = T_full.sizes["sample"]
    _ndc_diff = base["ndc_resilience"] - baseline_resilience
    print(f"NDC cross-check: calculator {base['ndc_resilience']:.9f} vs 09 "
          f"{baseline_resilience:.9f} (diff {_ndc_diff:+.2e})", flush=True)
    if abs(_ndc_diff) > 1e-6:
        raise AssertionError("calculator NDC resilience != 09 baseline (see cross-check above).")

    if _mode == "grid":
        if not os.path.exists(GRID_FINDER_CSV):
            raise FileNotFoundError(f"grid finder output not found: {GRID_FINDER_CSV} (run 13 grid first).")
        fin = pd.read_csv(GRID_FINDER_CSV)
        out_csv = os.path.join(WORKSPACE, "confirmed_grid.csv")
        done, records = set(), []
        if os.path.exists(out_csv):
            prev = pd.read_csv(out_csv); records = prev.to_dict("records")
            done = set(zip(prev["tcre_tercile"], prev["teff_tercile"]))

        for _, r in fin.iterrows():
            ci = TERCILES.index(r["tcre_tercile"]); sj = TERCILES.index(r["teff_tercile"])
            if (r["tcre_tercile"], r["teff_tercile"]) in done:
                print(f"[cache] {r['tcre_tercile']}×{r['teff_tercile']} — skip", flush=True)
                continue
            R = MAX_CUT_R if r["status"] == "max_cut_insufficient" else float(r["reduction_frac"])
            cfg_subset = tcre.index[np.digitize(tcre.values, tcre_bounds) == ci].tolist()
            samp_subset = np.nonzero(teff_tier_1000 == sj)[0]
            tag = f"tcre{ci}_teff{sj}"
            iter_dir = os.path.join(WORKSPACE, "grid", tag)
            cand_name = f"NDC_calc__grid_{tag}_r{R:.6f}"
            print(f"\n══ confirm TCRE {r['tcre_tercile']} × T_eff {r['teff_tercile']} "
                  f"R={R:.4f} status={r['status']} | {len(cfg_subset)} cfg × {len(samp_subset)} samp ══", flush=True)
            res = evaluate_candidate(
                R, df_emissions, years, year_cols, BASE_SCENARIO, FRAG_TARGET_YEAR, df_forcing,
                PARAMS_FILE, SPECIES_FILE, iter_dir,
                n_runs=N_FAIR_RUNS_FINAL, n_samples=N_TIPPING_SAMPLES_FINAL,
                temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD, year_target=YEAR_TARGET,
                config_subset=cfg_subset, sample_subset=samp_subset, lhs_params=lhs_params,
                return_tensors=True, candidate_name=cand_name,
            )
            # shared-unit positions on the full grid: cell configs (by id) × its T_eff tercile samples
            cfg_pos = np.array([full_cfg_pos[c] for c in res["T_cand"].config.values])
            boot = cell_vs_full_gain_bootstrap(res["T_cand"], T_full, cfg_pos, samp_subset, B=N_BOOTSTRAP)
            gain = res["achieved_resilience"] - baseline_resilience
            rec = {
                "tcre_tercile": r["tcre_tercile"], "teff_tercile": r["teff_tercile"],
                "tcre_lo": r["tcre_lo"], "tcre_hi": r["tcre_hi"],
                "teff_lo": r["teff_lo"], "teff_hi": r["teff_hi"],
                "n_configs": len(cfg_subset), "n_samples": int(len(samp_subset)),
                "finder_status": r["status"], "reduction_frac": R,
                "achieved_resilience": res["achieved_resilience"], "achieved_gain": gain,
                "baseline_resilience": baseline_resilience,
                "bootstrap_gain": boot["mean"], "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"], "ci_half_width": boot["ci_half_width"],
                "co2_GtCO2": res["co2_GtCO2"],
                "n_runs": N_FAIR_RUNS_FINAL, "n_samples_full": res["n_samples"],
            }
            if r["status"] == "max_cut_insufficient":
                ffi, afolu = _min_floor_co2(iter_dir, cand_name)
                rec["min_co2_ffi_2035_2100"] = ffi
                rec["min_co2_afolu_2035_2100"] = afolu
                print(f"  R=0.99 floor check: min CO2 FFI={ffi:.1f}, AFOLU={afolu:.1f} (MtCO2/yr)", flush=True)
            records.append(rec)
            done.add((r["tcre_tercile"], r["teff_tercile"]))
            pd.DataFrame(records).to_csv(out_csv, index=False)
            print(f"  gain={gain:.5f} [{boot['ci_low']:.5f}, {boot['ci_high']:.5f}]", flush=True)
        print(f"\nGrid confirmation complete -> {out_csv}")
        sys.exit(0)

    else:  # tcre windows
        if not os.path.exists(TCRE_FINDER_CSV):
            raise FileNotFoundError(f"tcre finder output not found: {TCRE_FINDER_CSV} (run 13 tcre first).")
        fin = pd.read_csv(TCRE_FINDER_CSV)
        tcre_rank = tcre.rank(pct=True) * 100.0
        TCRE_HALFWIDTH = 10.0
        out_csv = os.path.join(WORKSPACE, "confirmed_tcre_windows.csv")
        done, records = set(), []
        if os.path.exists(out_csv):
            prev = pd.read_csv(out_csv); records = prev.to_dict("records")
            done = set(zip(prev["percentile"], prev["gain_target"]))

        for _, r in fin.sort_values(["gain_target", "percentile"]).iterrows():
            pct = int(r["percentile"]); gain_t = round(float(r["gain_target"]), 2)
            if (pct, gain_t) in done:
                print(f"[cache] pct={pct} gain={gain_t} — skip", flush=True)
                continue
            R = MAX_CUT_R if r["status"] == "max_cut_insufficient" else float(r["reduction_frac"])
            mask = (tcre_rank >= pct - TCRE_HALFWIDTH) & (tcre_rank <= pct + TCRE_HALFWIDTH)
            cfg_subset = tcre.index[mask].tolist()
            tag = f"pct{pct:02d}_gain{int(gain_t*100):02d}"
            iter_dir = os.path.join(WORKSPACE, "tcre", tag)
            cand_name = f"NDC_calc__tcre_{tag}_r{R:.6f}"
            print(f"\n══ confirm TCRE pct={pct} gain={gain_t} R={R:.4f} status={r['status']} | "
                  f"{len(cfg_subset)} configs ══", flush=True)
            res = evaluate_candidate(
                R, df_emissions, years, year_cols, BASE_SCENARIO, FRAG_TARGET_YEAR, df_forcing,
                PARAMS_FILE, SPECIES_FILE, iter_dir,
                n_runs=N_FAIR_RUNS_FINAL, n_samples=N_TIPPING_SAMPLES_FINAL,
                temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD, year_target=YEAR_TARGET,
                config_subset=cfg_subset, lhs_params=lhs_params,
                return_tensors=True, candidate_name=cand_name,
            )
            # tcre windows use all samples; cell is a config subset over the full 1000 samples
            cfg_pos = np.array([full_cfg_pos[c] for c in res["T_cand"].config.values])
            boot = cell_vs_full_gain_bootstrap(res["T_cand"], T_full, cfg_pos,
                                               np.arange(n_samp_full), B=N_BOOTSTRAP)
            gain = res["achieved_resilience"] - baseline_resilience
            rec = {
                "percentile": pct, "tcre_value": r["tcre_value"], "n_configs": len(cfg_subset),
                "gain_target": gain_t, "finder_status": r["status"], "reduction_frac": R,
                "achieved_resilience": res["achieved_resilience"], "achieved_gain": gain,
                "baseline_resilience": baseline_resilience,
                "bootstrap_gain": boot["mean"], "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"], "ci_half_width": boot["ci_half_width"],
                "co2_GtCO2": res["co2_GtCO2"],
                "n_runs": N_FAIR_RUNS_FINAL, "n_samples": res["n_samples"],
            }
            records.append(rec)
            done.add((pct, gain_t))
            pd.DataFrame(records).to_csv(out_csv, index=False)
            print(f"  gain={gain:.5f} [{boot['ci_low']:.5f}, {boot['ci_high']:.5f}]", flush=True)
        print(f"\nTCRE-window confirmation complete -> {out_csv}")
        sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════════
# Gain mode — confirm the main finder (12) reductions per target year
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
        year_target=YEAR_TARGET, lhs_params=lhs_params,
        return_bootstrap=True, n_bootstrap=N_BOOTSTRAP,
        candidate_name=f"NDC_calc__ty{int(ty)}_r{red:.6f}",
    )
    boot = res["bootstrap"]           # paired candidate - NDC gain bootstrap
    achieved_gain = res["achieved_gain"]
    # cross-check: this gain-mode run uses the full ensemble (config_subset=None), so
    # the calculator's NDC resilience must reproduce the 09 summary value.
    _ndc_diff = res["ndc_resilience"] - baseline_resilience
    print(f"  NDC cross-check: calculator {res['ndc_resilience']:.9f} vs 09 "
          f"{baseline_resilience:.9f} (diff {_ndc_diff:+.2e})", flush=True)
    if abs(_ndc_diff) > 1e-6:
        raise AssertionError(
            f"calculator NDC resilience {res['ndc_resilience']:.9f} != 09 baseline "
            f"{baseline_resilience:.9f} (diff {_ndc_diff:.2e}); "
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
