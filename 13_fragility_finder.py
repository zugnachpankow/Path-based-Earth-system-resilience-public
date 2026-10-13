"""13_fragility_finder.py — required NDC cut vs Earth-system fragility (TCRE × T_eff).

Replaces the old GFP-percentile finder. Two modes (positional arg):

    python 13_fragility_finder.py grid     # Fig 4c: 3×3 grid of TCRE × T_eff terciles, gain +10pp
    python 13_fragility_finder.py tcre      # SI: 1D TCRE percentile windows, gains +1pp / +10pp

For each cell/window it bisects the additional NDC reduction at TARGET_YEAR=2035.5 needed
to reach the target resilience gain, where

    gain(R) = PERI_cell(candidate at R) − PERI_full(NDC)          (NOT paired)

and the full-ensemble NDC baseline is computed ONCE inside this script at the search
resolution (so 13 does not depend on 09). Cells are defined scenario-independently:
  * TCRE terciles  — from the 841-config tcre.csv (high TCRE = fragile).
  * T_eff terciles — bounds from the 1000-sample t_eff_lhs1000.csv, applied to the
                     finder's own 100-sample LHS (seed 4321) -> sample subsets.

Bracket [0, 0.99] with endpoint checks:
  * gain(0) ≥ target           -> met at NDC, R*=0, no bisection.
  * gain(0.99) < target        -> status "max_cut_insufficient", report gain at 0.99.
  * else                        -> bisect to TOL=0.004 / MAX_ITER=12.

Reduced search resolution (N_RUNS FAIR runs + 100 tipping samples). Heavy -> SLURM.

Output: output/calculator/fragility_finder/{grid_results.csv, tcre_window_results.csv}
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)

from calculator import evaluate_candidate, assert_configs_match_ensemble
from tipping_params import sample_lhs_params
from fair_config import FAIR_PARAMS, SPECIES_CONFIGS_NGFS

# ── mode ──────────────────────────────────────────────────────────────────────
MODE = sys.argv[1] if len(sys.argv) > 1 else "grid"
if MODE not in ("grid", "tcre"):
    raise SystemExit("usage: python 13_fragility_finder.py [grid|tcre]")

# ── configuration ───────────────────────────────────────────────────────────────
TARGET_YEAR    = 2035.5
TEMP_THRESHOLD = 1.5
RATE_THRESHOLD = 0.04
YEAR_TARGET    = 2100

GRID_GAIN      = 0.10                         # single gain target for the 3×3 grid
TCRE_GAINS     = [0.01, 0.10]                 # SI 1D windows
TCRE_PCTS      = [10, 30, 50, 70, 90]         # SI window centres (percentile of TCRE)
TCRE_HALFWIDTH = 10.0                         # ± window half-width in TCRE-percentile points

BASE_SCENARIO       = "REMIND-MAgPIE_3.3-4.8___Nationally_Determined_Contributions_(NDCs)"

EMISSIONS_FILE = "data/processed/NGFS_historic_merged_extended_to_2110.csv"
FORCING_FILE   = "data/processed/NGFS_forcing.csv"
PARAMS_FILE    = FAIR_PARAMS
SPECIES_FILE   = SPECIES_CONFIGS_NGFS
WORKSPACE      = "output/calculator/fragility_finder"

BISECT_LO, BISECT_HI = 0.0, 0.99
TOL       = 0.004
MAX_ITER  = 12
N_RUNS    = 10          # higher-res search so confirmed gains land near target
N_SAMPLES = 100         # finder tipping samples

FINDER_LHS_SEED = 4321  # finder's own 100-point LHS ("F3"); matches 11 t_eff_finder
TERCILE_Q = [100.0 / 3.0, 200.0 / 3.0]

os.makedirs(WORKSPACE, exist_ok=True)

# ── scenario-independent fragility axes (from 11) ───────────────────────────────
tcre = pd.read_csv("output/feedback/tcre.csv", index_col="config")["TCRE"]
teff_1000 = pd.read_csv("output/feedback/t_eff_lhs1000.csv", index_col="sample")["T_eff"]
teff_finder = pd.read_csv("output/feedback/t_eff_finder.csv", index_col="sample")["T_eff"]

assert_configs_match_ensemble()
df_configs = pd.read_csv(PARAMS_FILE, index_col=0)
assert set(tcre.index) == set(df_configs.index), "tcre.csv configs != parameter-file configs"

# TCRE terciles over the 841 configs (low/medium/high; high = fragile)
tcre_bounds = np.percentile(tcre.values, TERCILE_Q)
# T_eff terciles: bounds from the 1000-sample set, applied to the finder's 100 samples
teff_bounds = np.nanpercentile(teff_1000.values, TERCILE_Q)
TERCILES = ["low", "medium", "high"]


def _tercile_idx(values, bounds):
    """0/1/2 tercile index for each value given the two interior bounds."""
    return np.digitize(values, bounds)   # <b0 ->0, [b0,b1)->1, >=b1 ->2


tcre_tier = _tercile_idx(tcre.values, tcre_bounds)              # per config
teff_tier = _tercile_idx(teff_finder.values, teff_bounds)       # per finder sample

# ── emissions / forcing / finder LHS ────────────────────────────────────────────
df_emissions = pd.read_csv(EMISSIONS_FILE)
year_cols = [c for c in df_emissions.columns if c.replace(".", "", 1).isdigit()]
years = np.array([float(c) for c in year_cols])
df_forcing = pd.read_csv(FORCING_FILE)
lhs_params = sample_lhs_params(n_samples=N_SAMPLES, seed=FINDER_LHS_SEED)


def _eval(R, label, config_subset, sample_subset, run_base=True):
    iter_dir = os.path.join(WORKSPACE, MODE, label)
    return evaluate_candidate(
        R, df_emissions, years, year_cols,
        BASE_SCENARIO, TARGET_YEAR, df_forcing,
        PARAMS_FILE, SPECIES_FILE, iter_dir,
        n_runs=N_RUNS, n_samples=N_SAMPLES,
        temp_threshold=TEMP_THRESHOLD, rate_threshold=RATE_THRESHOLD,
        year_target=YEAR_TARGET,
        config_subset=config_subset, sample_subset=sample_subset, run_base=run_base,
        lhs_params=lhs_params,
        candidate_name=f"NDC_calc__{MODE}_{label}_r{R:.6f}",
    )


# full-ensemble NDC baseline PERI at the SEARCH resolution — computed once (no subsets)
_base = _eval(0.0, "baseline_full", None, None)
baseline_full = _base["ndc_resilience"]
print(f"[baseline] full-ensemble NDC PERI @ search res (N_RUNS={N_RUNS}, "
      f"N_SAMPLES={N_SAMPLES}) = {baseline_full:.6f}", flush=True)


def find_reduction(gain, tag, config_subset, sample_subset):
    """Bisect R so gain(R) = PERI_cell(candidate) − baseline_full reaches `gain`.

    Returns a dict with status in {met_at_ndc, max_cut_insufficient, ok}, the reduction
    fraction and the achieved gain / co2 at that R (reported at the converged R).
    """
    r0 = _eval(0.0, f"{tag}_r0", config_subset, sample_subset, run_base=False)
    g0 = r0["achieved_resilience"] - baseline_full
    if g0 >= gain:
        print(f"    met at NDC: gain(0)={g0:.5f} ≥ {gain}; R*=0", flush=True)
        return dict(status="met_at_ndc", reduction_frac=0.0, achieved_gain=g0, res=r0)

    rhi = _eval(BISECT_HI, f"{tag}_rhi", config_subset, sample_subset, run_base=False)
    ghi = rhi["achieved_resilience"] - baseline_full
    if ghi < gain:
        print(f"    not reachable: gain(0.99)={ghi:.5f} < {gain}; status=max_cut_insufficient", flush=True)
        return dict(status="max_cut_insufficient", reduction_frac=BISECT_HI, achieved_gain=ghi, res=rhi)

    lo, hi = BISECT_LO, BISECT_HI
    for it in range(MAX_ITER):
        mid = 0.5 * (lo + hi)
        g = _eval(mid, f"{tag}_it{it:02d}", config_subset, sample_subset, run_base=False)["achieved_resilience"] - baseline_full
        if g < gain:
            lo = mid
        else:
            hi = mid
        print(f"    iter {it:02d}: R={mid:.4f} gain={g:.5f} [lo={lo:.4f}, hi={hi:.4f}]", flush=True)
        if hi - lo < TOL:
            break
    conv = 0.5 * (lo + hi)
    res = _eval(conv, f"{tag}_converged", config_subset, sample_subset, run_base=False)
    return dict(status="ok", reduction_frac=conv,
                achieved_gain=res["achieved_resilience"] - baseline_full, res=res)


# ══════════════════════════════════════════════════════════════════════════════
# GRID mode — 3×3 TCRE × T_eff terciles, gain +10pp
# ══════════════════════════════════════════════════════════════════════════════
if MODE == "grid":
    out_csv = os.path.join(WORKSPACE, "grid_results.csv")
    done, records = set(), []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        records = prev.to_dict("records")
        done = set(zip(prev["tcre_tercile"], prev["teff_tercile"]))

    tcre_edges = [tcre.min()] + list(tcre_bounds) + [tcre.max()]
    teff_edges = [np.nanmin(teff_1000)] + list(teff_bounds) + [np.nanmax(teff_1000)]
    for ci in range(3):
        cfg_subset = tcre.index[tcre_tier == ci].tolist()
        for sj in range(3):
            if (TERCILES[ci], TERCILES[sj]) in done:
                print(f"[cache] TCRE={TERCILES[ci]} T_eff={TERCILES[sj]} — skip", flush=True)
                continue
            samp_subset = np.nonzero(teff_tier == sj)[0]
            tag = f"tcre{ci}_teff{sj}"
            print(f"\n── TCRE {TERCILES[ci]} × T_eff {TERCILES[sj]} | "
                  f"{len(cfg_subset)} configs × {len(samp_subset)} samples ──", flush=True)
            r = find_reduction(GRID_GAIN, tag, cfg_subset, samp_subset)
            records.append({
                "tcre_tercile": TERCILES[ci], "teff_tercile": TERCILES[sj],
                "tcre_lo": tcre_edges[ci], "tcre_hi": tcre_edges[ci + 1],
                "teff_lo": teff_edges[sj], "teff_hi": teff_edges[sj + 1],
                "n_configs": len(cfg_subset), "n_samples": int(len(samp_subset)),
                "gain_target": GRID_GAIN, "status": r["status"],
                "reduction_frac": r["reduction_frac"], "achieved_gain": r["achieved_gain"],
                "co2_GtCO2": r["res"]["co2_GtCO2"], "baseline_full": baseline_full,
            })
            done.add((TERCILES[ci], TERCILES[sj]))
            pd.DataFrame(records).to_csv(out_csv, index=False)
            print(f"  → status={r['status']} R={r['reduction_frac']*100:.1f}% "
                  f"gain={r['achieved_gain']:.4f}", flush=True)
    print(f"\nGrid finder complete -> {out_csv}")
    print(pd.DataFrame(records).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# TCRE mode — SI 1D percentile windows (all samples), gains +1pp / +10pp
# ══════════════════════════════════════════════════════════════════════════════
else:
    out_csv = os.path.join(WORKSPACE, "tcre_window_results.csv")
    done, records = set(), []
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        records = prev.to_dict("records")
        done = set(zip(prev["percentile"], prev["gain_target"]))

    tcre_rank = tcre.rank(pct=True) * 100.0        # percentile rank per config
    for pct in TCRE_PCTS:
        mask = (tcre_rank >= pct - TCRE_HALFWIDTH) & (tcre_rank <= pct + TCRE_HALFWIDTH)
        cfg_subset = tcre.index[mask].tolist()
        tcre_val = float(np.percentile(tcre.values, pct))
        for gain in TCRE_GAINS:
            if (pct, gain) in done:
                print(f"[cache] pct={pct} gain={gain} — skip", flush=True)
                continue
            tag = f"pct{pct:02d}_gain{int(gain*100):02d}"
            print(f"\n── TCRE {pct}th pct (={tcre_val:.3f}) ± {TCRE_HALFWIDTH}pp | "
                  f"{len(cfg_subset)} configs | gain {gain} ──", flush=True)
            r = find_reduction(gain, tag, cfg_subset, None)
            records.append({
                "percentile": pct, "tcre_value": tcre_val, "n_configs": len(cfg_subset),
                "gain_target": gain, "status": r["status"],
                "reduction_frac": r["reduction_frac"], "achieved_gain": r["achieved_gain"],
                "co2_GtCO2": r["res"]["co2_GtCO2"], "baseline_full": baseline_full,
            })
            done.add((pct, gain))
            pd.DataFrame(records).to_csv(out_csv, index=False)
            print(f"  → status={r['status']} R={r['reduction_frac']*100:.1f}% "
                  f"gain={r['achieved_gain']:.4f}", flush=True)
    print(f"\nTCRE-window finder complete -> {out_csv}")
    print(pd.DataFrame(records).to_string(index=False))
