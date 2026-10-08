"""Run-once preprocessing for the resilience post-processing.

Computes the expensive, criterion-independent artifact (the 20-yr running-mean
GMST per scenario) from the concatenated temperature output, and caches it so
the per-condition step (09_process_resilience.py) can be run repeatedly and
cheaply for each (res_y, rate) set.
"""
import os
import sys
import pickle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
os.chdir(_HERE)  # resolve output/ from the repo root, not the job's cwd
from resilience import load_temperature, compute_running_means

TEMPERATURE_FILE = "output/all_scenarios_temperature.nc"  # written by 06_concatenate
RUNNING_MEANS_FILE = "output/running_mean_temps.pkl"

temperature = load_temperature(TEMPERATURE_FILE)
scenarios = temperature.scenario.values.tolist()

print(f"Computing 20-yr running means for {len(scenarios)} scenarios...")
running_means = compute_running_means(temperature, scenarios)

with open(RUNNING_MEANS_FILE, "wb") as f:
    pickle.dump(running_means, f)
print(f"Wrote {RUNNING_MEANS_FILE}")
