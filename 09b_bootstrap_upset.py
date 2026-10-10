"""09b_bootstrap_upset.py — crossed bootstrap CIs for the UpSet bars.

Faithful port of ERI/code/bootstrap_upset_bars/bootstrap_upset_bars.py. Produces the
95% crossed-bootstrap CIs for all seven UpSet-bar combinations (C, R, T, C+R,
C+T, R+T, C+R+T) that 15_figures_main.py draws as error bars on panel b. Only the
I/O paths are adapted to the cleaned pipeline; the bootstrap and criterion logic are
kept verbatim.

The 7 criteria and their tensors:
  C       no climate violation        (n_cfg, n_runs, 1)
  R       no rate violation           (n_cfg, n_runs, 1)
  T       no tipping                  (n_cfg, n_runs, n_samp)
  C+R     climate AND rate            (n_cfg, n_runs, 1)
  C+T     climate AND no tipping      (n_cfg, n_runs, n_samp)
  R+T     rate AND no tipping         (n_cfg, n_runs, n_samp)
  C+R+T   all three (= resilience)    (n_cfg, n_runs, n_samp)

For criteria without a sample dimension (C, R, C+R) the tensor is broadcast to
(n_cfg, n_runs, 1) so the same bootstrap function handles all seven.

Reads output/running_mean_temps.pkl (08) + output/tipping/*.nc (07);
writes output/bootstrap_upset/upset_bootstrap_*.csv. Heavy — submit via SLURM.
"""
import os
import sys
import time
import pickle

import numpy as np
import pandas as pd
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve output/ from the repo root, not the job's cwd
from resilience import BOOTSTRAP_SEED   # shared bootstrap seed (same as 09)

RUNNING_MEAN_PATH = "output/running_mean_temps.pkl"
TIPPING_DIR       = "output/tipping"
OUT_DIR           = "output/bootstrap_upset"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Resilience conditions (must match 09_process_resilience.py / figures) ──────
res_x = 2100
res_y = 1.5
rate  = 0.04   # K/yr; 0.4 K/decade

# ── Scenarios and bootstrap iterations ────────────────────────────────────────
SCENARIOS = [
    "REMIND-MAgPIE_3.3-4.8___Net_Zero_2050",
    "ssp119",
    "ssp126",
    "ssp534-over",
]
B = 1000


# ── Bootstrap ──────────────────────────────────────────────────────────────────
def crossed_bootstrap(T: np.ndarray, B: int = 1000, seed: int = 0) -> dict:
    """
    Full crossed bootstrap across all three dimensions.

    T : float32 array, shape (n_cfg, n_runs, n_samp).
        For quantities without a sample dimension, pass shape (n_cfg, n_runs, 1).
    Returns dict with mean, ci_low, ci_high (95 %).
    """
    rng = np.random.default_rng(seed)
    n_cfg, n_runs, n_samp = T.shape

    # Pre-draw all index arrays — avoids per-iteration overhead
    cfg_idx_all  = rng.integers(0, n_cfg,  size=(B, n_cfg))
    run_idx_all  = rng.integers(0, n_runs, size=(B, n_runs))
    samp_idx_all = rng.integers(0, n_samp, size=(B, n_samp))

    Q = np.empty(B, dtype=np.float64)
    for b in range(B):
        # np.ix_ builds an open mesh — one temporary array instead of three
        Q[b] = T[np.ix_(cfg_idx_all[b], run_idx_all[b], samp_idx_all[b])].mean()

    return {
        "mean":    float(Q.mean()),
        "ci_low":  float(np.quantile(Q, 0.025)),
        "ci_high": float(np.quantile(Q, 0.975)),
    }


# ── Load data (once, shared across scenarios) ──────────────────────────────────
print("Loading running mean temperatures …", flush=True)
with open(RUNNING_MEAN_PATH, "rb") as fh:
    running_mean_temps = pickle.load(fh)
running_mean_temps = {
    k.replace("(", "").replace(")", ""): v
    for k, v in running_mean_temps.items()
}

print("Loading tipping data …", flush=True)
# In the cleaned pipeline the per-sample (prob_any_tipping_sample) and aggregated
# (prob_any_tipping) probabilities live in the SAME file per scenario.
tipping_dict = {}
for fname in sorted(os.listdir(TIPPING_DIR)):
    if fname.endswith("_tipping_probabilities.nc"):
        sc = fname.split("_tipping_probabilities.nc")[0]
        tipping_dict[sc] = xr.open_dataset(os.path.join(TIPPING_DIR, fname))

# ── Per-scenario loop ──────────────────────────────────────────────────────────
all_rows = []

for SCENARIO in SCENARIOS:
    print(f"\n{'='*60}", flush=True)
    print(f"Scenario: {SCENARIO}", flush=True)
    print(f"{'='*60}", flush=True)

    if SCENARIO not in running_mean_temps or SCENARIO not in tipping_dict:
        missing = [n for n, d in (("running mean temps", running_mean_temps),
                                  ("tipping", tipping_dict)) if SCENARIO not in d]
        print(f"  SKIP — missing: {missing}", flush=True)
        continue

    # Resumable: if this scenario's CSV already exists, reuse it (each T-tensor takes
    # ~400s, so a timeout mustn't force recomputing already-finished scenarios).
    out_file = os.path.join(OUT_DIR, f"upset_bootstrap_{SCENARIO.replace('/', '_')}_B{B}.csv")
    if os.path.exists(out_file):
        all_rows.extend(pd.read_csv(out_file).to_dict("records"))
        print(f"  SKIP — already done ({out_file})", flush=True)
        continue

    # ── Tipping source consistency check ──────────────────────────────────────
    # T in the bootstrap uses (1 - prob_any_tipping_sample), which integrates over
    # samples. The figures use (1 - prob_any_tipping), the pre-averaged version.
    # By linearity these are equal; any discrepancy flags an alignment problem.
    da     = tipping_dict[SCENARIO]["prob_any_tipping_sample"]   # (samp, cfg, run)
    ds_agg = tipping_dict[SCENARIO]

    t_from_samples = float(da.mean(dim="sample").values.mean())  # mean over (cfg, run) after collapsing sample
    t_from_agg     = float(ds_agg["prob_any_tipping"].values.mean())
    abs_diff = abs(t_from_samples - t_from_agg)
    status   = "OK" if abs_diff < 1e-4 else "WARNING — large discrepancy"
    print(
        f"\n  [T source check]  sample-collapsed: {t_from_samples:.4f}  "
        f"aggregated: {t_from_agg:.4f}  |diff|: {abs_diff:.2e}  {status}",
        flush=True,
    )

    # ── Build tensors ─────────────────────────────────────────────────────────
    rm = running_mean_temps[SCENARIO]

    climate_viol = (rm.sel(timebounds=slice(res_x, None)) > res_y).any(dim="timebounds")
    rate_viol    = (rm.diff(dim="timebounds")
                      .rolling(timebounds=10, center=True).mean()
                      .sel(timebounds=slice(2000, res_x)) > rate
                   ).any(dim="timebounds")

    no_climate_da    = (~climate_viol).astype(float).set_index(member=["config", "run"]).unstack("member")
    no_rate_da       = (~rate_viol).astype(float).set_index(member=["config", "run"]).unstack("member")
    resilience_cr_da = no_climate_da * no_rate_da

    cfg_coords = da.config.values
    run_coords = da.run.values
    n_cfg  = len(cfg_coords)
    n_runs = len(run_coords)
    n_samp = da.sizes["sample"]

    no_climate_2d    = no_climate_da.sel(config=cfg_coords, run=run_coords).values
    no_rate_2d       = no_rate_da.sel(config=cfg_coords, run=run_coords).values
    resilience_cr_2d = resilience_cr_da.sel(config=cfg_coords, run=run_coords).values

    # (n_cfg, n_runs, n_samp) — sample axis moved from front to back
    no_tipping_3d = (1.0 - da.transpose("config", "run", "sample").values).astype(np.float32)

    bytes_3d = no_tipping_3d.nbytes
    print(
        f"\n  Dimensions: n_cfg={n_cfg}, n_runs={n_runs}, n_samp={n_samp}"
        f"  |  3-D tensor: {bytes_3d / 1e6:.0f} MB",
        flush=True,
    )

    tensors = {
        "C":     no_climate_2d[:, :, np.newaxis].astype(np.float32),
        "R":     no_rate_2d[:, :, np.newaxis].astype(np.float32),
        "T":     no_tipping_3d,
        "C+R":   resilience_cr_2d[:, :, np.newaxis].astype(np.float32),
        "C+T":   (no_climate_2d[:, :, np.newaxis] * no_tipping_3d).astype(np.float32),
        "R+T":   (no_rate_2d[:, :, np.newaxis] * no_tipping_3d).astype(np.float32),
        "C+R+T": (resilience_cr_2d[:, :, np.newaxis] * no_tipping_3d).astype(np.float32),
    }

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    print(f"\n  Running bootstrap (B={B}) …\n", flush=True)
    rows = []
    for name, T in tensors.items():
        point_est = float(T.mean())
        t0 = time.perf_counter()
        result = crossed_bootstrap(T, B=B, seed=BOOTSTRAP_SEED)
        elapsed = time.perf_counter() - t0

        row = {
            "scenario":  SCENARIO,
            "criterion": name,
            "point_est": round(point_est * 100, 2),
            "boot_mean": round(result["mean"] * 100, 2),
            "ci_low":    round(result["ci_low"] * 100, 2),
            "ci_high":   round(result["ci_high"] * 100, 2),
            "ci_width":  round((result["ci_high"] - result["ci_low"]) * 100, 2),
            "runtime_s": round(elapsed, 1),
            "n_cfg": n_cfg, "n_runs": n_runs, "n_samp": int(T.shape[2]), "B": B,
        }
        rows.append(row)
        all_rows.append(row)
        print(
            f"    {name:5s}  {point_est*100:5.1f}%  "
            f"[{result['ci_low']*100:.1f}, {result['ci_high']*100:.1f}]  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    # ── Save per-scenario CSV ──────────────────────────────────────────────────
    out_file = os.path.join(OUT_DIR, f"upset_bootstrap_{SCENARIO.replace('/', '_')}_B{B}.csv")
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"\n  Saved → {out_file}", flush=True)

# ── Save combined CSV ──────────────────────────────────────────────────────────
combined_file = os.path.join(OUT_DIR, f"upset_bootstrap_combined_B{B}.csv")
df_all = pd.DataFrame(all_rows)
df_all.to_csv(combined_file, index=False)
print(f"\nSaved combined → {combined_file}", flush=True)
print(df_all.to_string(index=False))
