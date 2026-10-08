"""Resilience processing for the paper's resilience-criterion sets.

Reads the cached running means (08_preprocess.py) + the tipping probabilities,
then for each condition in CONDITIONS computes, per scenario:
  - the binary climate resilience (climate + warming-rate thresholds),
  - its combination with the tipping probability,
  - a crossed-bootstrap mean + 95% CI,
  - a per-scenario summary table.
"""
import os
import sys
import re
import pickle
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
from climateforcing.utils import mkdir_p
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve output/ from the repo root, not the job's cwd
from resilience import load_tipping, compute_resilience, crossed_bootstrap

RUNNING_MEANS_FILE = "output/running_mean_temps.pkl"
TIPPING_DIR = "output/tipping"
BASE_PATH = "output/resilience"
N_BOOTSTRAP = 1000

# the resilience-criterion sets to produce (res_x fixed at 2100)
CONDITIONS = [
    dict(res_x=2100, res_y=1.5, rate=0.04),
    dict(res_x=2100, res_y=1.5, rate=0.03),
    dict(res_x=2100, res_y=2.0, rate=0.04),
]

# --- load cached preprocessing + tipping once ---
with open(RUNNING_MEANS_FILE, "rb") as f:
    running_mean_temps = pickle.load(f)
running_mean_temps = {
    k.replace("(", "").replace(")", ""): v for k, v in running_mean_temps.items()
}
tipping_sample_dict = load_tipping(TIPPING_DIR)

# GCAM / MESSAGE scenarios are not used
scenarios = [s for s in running_mean_temps if "GCAM" not in s and "MESSAGE" not in s]


def process_condition(res_x, res_y, rate):
    rate_str = str(round(rate * 10, 4)).replace(".", "")  # e.g. 0.04 -> "04"
    condition_folder = f"resilience_conditions_time{res_x}_temp{res_y}_rate{rate_str}"
    condition_path = join(BASE_PATH, condition_folder)
    mkdir_p(condition_path)
    print(f"\n=== {condition_folder} (temp>{res_y} K, rate>{rate} K/yr) ===")

    def tag(name):
        return f"{name}_time{res_x}_temp{res_y}_rate{rate_str}"

    # --- resilience + tipping (resume: skip scenarios already written) ---
    existing = {
        re.sub(r"^resilience_", "", f).rsplit(f"_time{res_x}", 1)[0]
        for f in listdir(condition_path)
        if f.endswith(".csv") and isfile(join(condition_path, f))
    }
    to_calculate = [s for s in scenarios if s not in existing]

    records = []
    for scenario in tqdm(to_calculate, desc="Resilience"):
        df_res = compute_resilience(running_mean_temps[scenario], res_x, res_y, rate)
        df_res["scenario"] = scenario
        records.append(df_res)

    if records:
        df_resilience = pd.concat(records, ignore_index=True)
        for scenario in tqdm(to_calculate, desc="Resilience x tipping"):
            df_tip = (
                tipping_sample_dict[scenario]["prob_any_tipping_sample"]
                .to_dataframe(name="tipping_any")
                .reset_index()
            )
            df_tip["scenario"] = scenario
            df_merged = df_tip.merge(df_resilience, on=["scenario", "config", "run"], how="left")
            df_merged["resilience_binary"] = df_merged["resilience_cr"] * (1 - df_merged["tipping_any"])
            df_merged.to_csv(join(condition_path, f"resilience_{tag(scenario)}.csv"), index=False)

    # --- bootstrap stats (resume via stats file) ---
    stats_path = join(condition_path, "stats")
    mkdir_p(stats_path)
    stats_file = join(stats_path, f"resilience_stats_time{res_x}_temp{res_y}_rate{rate_str}.csv")
    stats_df = pd.read_csv(stats_file) if os.path.exists(stats_file) else pd.DataFrame()
    done = stats_df["scenario"].tolist() if not stats_df.empty else []

    new_stats = []
    for scenario in tqdm([s for s in scenarios if s not in done], desc="Bootstrap"):
        df = pd.read_csv(join(condition_path, f"resilience_{tag(scenario)}.csv"))
        db = df[["config", "run", "sample", "resilience_binary"]]
        T = (
            db.pivot_table(index=["config", "run"], columns="sample", values="resilience_binary")
            .values.reshape(db["config"].nunique(), db["run"].nunique(), db["sample"].nunique())
            .astype(np.float32)
        )
        stat = crossed_bootstrap(T, B=N_BOOTSTRAP)
        new_stats.append({
            "scenario": scenario,
            "resilience_mean": stat["mean"],
            "resilience_ci_low": stat["ci_low"],
            "resilience_ci_high": stat["ci_high"],
        })
    if new_stats:
        stats_df = pd.concat([stats_df, pd.DataFrame(new_stats)], ignore_index=True)
        stats_df.to_csv(stats_file, index=False)

    # --- per-scenario summary table (mean resilience + CI) ---
    summary_path = join(condition_path, "summary")
    mkdir_p(summary_path)
    summary_file = join(summary_path, f"resilience_summary_time{res_x}_temp{res_y}_rate{rate_str}.csv")
    rows = []
    for scenario in scenarios:
        df = pd.read_csv(join(condition_path, f"resilience_{tag(scenario)}.csv"))
        res = (
            df.groupby("scenario")["resilience_binary"].mean()
            .reset_index().rename(columns={"resilience_binary": "resilience"})
        )
        res = res.merge(
            stats_df[["scenario", "resilience_ci_low", "resilience_ci_high"]],
            on="scenario", how="left",
        )
        rows.append(res)
    pd.concat(rows, ignore_index=True).to_csv(summary_file, index=False)
    print(f"Wrote summary: {summary_file}")


for condition in CONDITIONS:
    process_condition(**condition)
