import os

import xarray as xr

repo = os.path.dirname(os.path.abspath(__file__))

temperature = xr.open_dataset(os.path.join(repo, "output", "all_scenarios_temperature.nc"))
scenarios = [
    s.replace("(", "").replace(")", "")
    for s in temperature.scenario.values.tolist()
    if "GCAM" not in s and "MESSAGE" not in s
]

# check if these scenarios have already been run (resume support). Results live
# under <repo>/output/tipping (output/ is symlinked to scratch for now).
results_dir = os.path.join(repo, "output", "tipping")
existing_outputs = []
if os.path.isdir(results_dir):
    existing_outputs = os.listdir(results_dir)
existing_set = set(existing_outputs)
scenarios_to_run = [
    scenario for scenario in scenarios
    if f"{scenario}_tipping_probabilities.nc" not in existing_set   # exact match, not substring
]

print(f"Scenarios to run: {scenarios_to_run}", flush=True)

scenarios = scenarios_to_run

for scenario in scenarios:
    #iniate job script
    with open("job_submit.sh", "w+") as fh:
        fh.writelines("#!/bin/bash\n\n")

        #specifications of the job that should be submitted
        fh.writelines("#SBATCH --qos=medium\n")
        fh.writelines("#SBATCH --job-name=run-tipping-scenario\n")
        fh.writelines("#SBATCH --account=copan\n\n")

        fh.writelines(f"#SBATCH --chdir={repo}\n")
        fh.writelines("#SBATCH --output=logs/tipping/%x-%j.out\n")
        fh.writelines("#SBATCH --error=logs/tipping/%x-%j.err\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --cpus-per-task=32\n")
        fh.writelines("#SBATCH --mem=128G\n")

        #fh.writelines("module load anaconda/2025\n")
        #fh.writelines("source activate fair\n\n")

        #job to be submitted
        fh.writelines(f"python {os.path.join(repo, 'src', 'run_tipping.py')} {str(scenario)}")
        fh.close()

    os.system(f"sbatch job_submit.sh")
