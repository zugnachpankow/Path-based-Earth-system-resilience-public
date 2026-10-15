"""09_process_resilience_fast.py — parallel, slim drop-in for 09_process_resilience.py.

Same CONDITIONS, same scenario list (running means minus GCAM/MESSAGE), same solver-report
check, same output file names/formats — but writes to BASE_PATH = output/resilience_alt and:

  * runs the (condition, scenario) tasks in parallel (fork Pool, n = SLURM_CPUS_PER_TASK,
    OMP_NUM_THREADS=1). The parent loads the 31 GB running means ONCE; each worker opens its
    own scenario tipping .nc by path (no shared netCDF handles across fork) and is passed only
    names/scalars.
  * builds T = resilience_cr x (1 - prob_any_tipping_sample) IN MEMORY, ordered exactly as 09's
    pivot_table (config, run, sample all ascending), float64 for the point estimate and
    .astype(float32) for crossed_bootstrap(T, B=1000, seed=0) -> bit-identical to 09.
  * point estimate = T.sum() / T.size (float64).
  * writes a SLIM per-scenario CSV with only (config, run, resilience_cr) under the same file
    name (the slim CSV avoids the full per-(config,run,sample) explosion; downstream
    16/si23 only use resilience_cr per (config, run)).
  * writes the stats and summary CSVs with exactly 09's columns, row order and solver merge.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy; avoid BLAS oversubscription
import sys
import pickle
from multiprocessing import get_context
from os.path import join

import numpy as np
import pandas as pd
import xarray as xr
from climateforcing.utils import mkdir_p

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)
from resilience import load_tipping, compute_resilience, crossed_bootstrap  # noqa: E402

RUNNING_MEANS_FILE = "output/running_mean_temps.pkl"
TIPPING_DIR = "output/tipping"
BASE_PATH = "output/resilience_alt"                 # NEXT TO the running 09's output/resilience
N_BOOTSTRAP = 1000
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))

CONDITIONS = [
    dict(res_x=2100, res_y=1.5, rate=0.04),
    dict(res_x=2100, res_y=1.5, rate=0.03),
    dict(res_x=2100, res_y=2.0, rate=0.04),
]


def _rate_str(rate):
    return str(round(rate * 10, 4)).replace(".", "")   # 0.04 -> "04"


def _tag(name, res_x, res_y, rate):
    return f"{name}_time{res_x}_temp{res_y}_rate{_rate_str(rate)}"


def _cond_path(res_x, res_y, rate):
    return join(BASE_PATH, f"resilience_conditions_time{res_x}_temp{res_y}_rate{_rate_str(rate)}")


# running_mean_temps is a module global set in __main__ BEFORE the Pool forks, so workers
# inherit it copy-on-write (read-only) without it being pickled per task.
running_mean_temps = None


def _process_one(task):
    """One (condition, scenario): slim CSV + the stats/summary row. Runs in a worker."""
    res_x, res_y, rate, scenario = task
    cond_path = _cond_path(res_x, res_y, rate)
    slim_csv = join(cond_path, f"resilience_{_tag(scenario, res_x, res_y, rate)}.csv")

    # resilience_cr per (config, run) — same function/inputs as 09
    df_res = compute_resilience(running_mean_temps[scenario], res_x, res_y, rate)

    # T[config, run, sample] exactly as 09's pivot_table(index=[config,run], columns=sample)
    # .reshape(...): all three axes sorted ascending. xarray aligns by label; sortby fixes order.
    ds = xr.open_dataset(join(TIPPING_DIR, f"{scenario}_tipping_probabilities.nc"))
    try:
        pas = ds["prob_any_tipping_sample"]
        rc = df_res.set_index(["config", "run"])["resilience_cr"].to_xarray()
        T_da = (rc * (1.0 - pas)).transpose("config", "run", "sample").sortby(["config", "run", "sample"])
        T = T_da.values                                     # float64
    finally:
        ds.close()

    point = float(T.sum() / T.size)                         # point estimate (float64)
    stat = crossed_bootstrap(T.astype(np.float32), B=N_BOOTSTRAP)   # seed=0, bit-identical to 09

    # slim per-scenario CSV: only (config, run, resilience_cr)
    slim = df_res[["config", "run", "resilience_cr"]].sort_values(["config", "run"]).reset_index(drop=True)
    slim.to_csv(slim_csv, index=False)

    return {
        "res_x": res_x, "res_y": res_y, "rate": rate, "scenario": scenario,
        "resilience": point,
        "resilience_mean": stat["mean"],
        "resilience_ci_low": stat["ci_low"],
        "resilience_ci_high": stat["ci_high"],
    }


def main():
    global running_mean_temps
    with open(RUNNING_MEANS_FILE, "rb") as f:
        running_mean_temps = pickle.load(f)
    running_mean_temps = {k.replace("(", "").replace(")", ""): v for k, v in running_mean_temps.items()}

    # solver report check: refuse to process if any tipping solve failed
    SOLVER_REPORT = join(TIPPING_DIR, "solver_report.csv")
    solver_report_df = None
    if os.path.exists(SOLVER_REPORT):
        _raw = pd.read_csv(SOLVER_REPORT)
        _raw = _raw[_raw["scenario"] != "scenario"].copy()
        for _c in ("n_solver_failures", "n_tip_final_disagreements"):
            _raw[_c] = pd.to_numeric(_raw[_c], errors="coerce").fillna(0).astype(int)
        solver_report_df = _raw.groupby("scenario", as_index=False).max()
        bad = solver_report_df[solver_report_df["n_solver_failures"] > 0]
        if not bad.empty:
            raise RuntimeError(
                f"failed tipping solves in {SOLVER_REPORT}; refusing to process. "
                f"Offending scenarios:\n{bad.to_string(index=False)}"
            )
        print(f"Solver report OK: 0 failures across {len(solver_report_df)} scenarios.", flush=True)
    else:
        print(f"[warn] no solver report at {SOLVER_REPORT} (run src/run_tipping.py via 07).", flush=True)

    scenarios = [s for s in running_mean_temps if "GCAM" not in s and "MESSAGE" not in s]
    for cond in CONDITIONS:
        mkdir_p(_cond_path(**cond))

    # flat (condition, scenario) task list; parallel over all of them
    tasks = [(c["res_x"], c["res_y"], c["rate"], s) for c in CONDITIONS for s in scenarios]
    print(f"Processing {len(tasks)} (condition x scenario) tasks on {N_JOBS} workers …", flush=True)
    with get_context("fork").Pool(processes=N_JOBS) as pool:
        results = pool.map(_process_one, tasks)

    # index results and assemble per-condition stats + summary IN scenarios order (== 09)
    by_key = {(r["res_x"], r["res_y"], r["rate"], r["scenario"]): r for r in results}
    for cond in CONDITIONS:
        res_x, res_y, rate = cond["res_x"], cond["res_y"], cond["rate"]
        cond_path = _cond_path(res_x, res_y, rate)
        ordered = [by_key[(res_x, res_y, rate, s)] for s in scenarios]

        stats_path = join(cond_path, "stats"); mkdir_p(stats_path)
        stats_df = pd.DataFrame([{
            "scenario": r["scenario"], "resilience_mean": r["resilience_mean"],
            "resilience_ci_low": r["resilience_ci_low"], "resilience_ci_high": r["resilience_ci_high"],
        } for r in ordered])
        stats_df.to_csv(join(stats_path, f"resilience_stats_time{res_x}_temp{res_y}_rate{_rate_str(rate)}.csv"),
                        index=False)

        summary_path = join(cond_path, "summary"); mkdir_p(summary_path)
        summary_df = pd.DataFrame([{
            "scenario": r["scenario"], "resilience": r["resilience"],
            "resilience_ci_low": r["resilience_ci_low"], "resilience_ci_high": r["resilience_ci_high"],
        } for r in ordered])
        # attach per-scenario solver counts
        if solver_report_df is not None:
            clean = solver_report_df.copy()
            clean["scenario"] = clean["scenario"].str.replace(r"[\(\)]", "", regex=True)
            summary_df["_sc"] = summary_df["scenario"].str.replace(r"[\(\)]", "", regex=True)
            summary_df = summary_df.merge(
                clean.rename(columns={"scenario": "_sc"}), on="_sc", how="left"
            ).drop(columns="_sc")
        summary_file = join(summary_path, f"resilience_summary_time{res_x}_temp{res_y}_rate{_rate_str(rate)}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Wrote summary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
