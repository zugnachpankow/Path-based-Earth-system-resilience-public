# Path-based Earth-System Resilience

Code base for the forthcoming publication "Ambitious emission reductions maintain Earth resilience, conditional on climate-carbon feedbacks and tipping points".

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-blue)](https://doi.org/PENDING)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)

<!-- FILL IN: preprint/journal link, Zenodo software + dataset DOIs, author list -->

---

## Table of Contents
1. [Background](#background)
2. [Features](#features)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data](#data)
7. [License](#license)
8. [Citation](#citation)
9. [Contact](#contact)

---

## Background
This project quantifies the **path-based resilience** given a climate mitigation scenario. It is estimated as the share of modelled trajectories that stay within a safe operating state. For each scenario we run a large Monte-Carlo ensemble of the [fair](https://docs.fairmodel.net/) simple climate model (climate-parameter configurations × internal-variability realisations) and a tipping-cascade sampler [pycascades](https://github.com/pik-copan/pycascades), and evaluate three joint conditions:

- **Temperature** — the 20-yr running-mean GMST never exceeds a threshold (default 1.5 °C) from the return year (2100) onward;
- **Rate** — the warming rate never exceeds a threshold (default 0.4 °C per decade);
- **Tipping** — weighted by the committed probability of triggering a tipping cascade (GIS / THC / WAIS / AMAZ).

The pipeline also provides a Sobol/Monte-Carlo sensitivity analysis of the criterion parameters, a generalised feedback parameter (GFP) view, and an "ambition calculator" translating a target resilience gain into required emission cuts.

![Emission "ambition calculator": required additional NDC reduction for a target path-based resilience gain.](figures/example/calculator_combined.png)

## Features
- Path-based resilience indicator combining temperature, warming-rate, and tipping conditions
- Large fair (2.2.3) Monte-Carlo ensemble over calibrated-constrained climate configurations
- Pycascades tipping-cascade probabilities (cusp elements) via Latin-hypercube sampling
- Sobol + Latin-hypercube sensitivity analysis of the resilience criterion
- Diagnostics of resilience depending on Earth system feedbacks and tipping susceptibility
- Emission "ambition calculator" (required NDC reduction for a target resilience gain)
- Publication-quality figure and table scripts
- Designed to run on a workstation or an HPC/SLURM cluster

---

## Repository layout
The numbered scripts are the pipeline, run in order. Main-paper scripts live at the repo
root; supplementary scripts and the SLURM batch scripts live in subfolders.

| Stage | Scripts | What it does |
|---|---|---|
| Data prep | `00` | reproduce the NGFS emissions CSV from the licensed IAM workbook (see [Data](#data)) |
| Inputs & runs | `01`–`06` | clean/extend NGFS emissions, build branch-offs, run the fair ensemble (SSPs, NGFS, branch-offs), concatenate |
| Tipping | `07` | submit the pycascade tipping-cascade sampler (per scenario) |
| Preprocess | `08` | 20-yr running-mean temperatures |
| Resilience | `09`, `09b` | path-based resilience + bootstrap CIs; UpSet-bar CIs |
| Sensitivity | `10` | Sobol + Monte-Carlo over the criterion parameters |
| Feedback/tipping | `11` | GFP, ECS, random-forest tipping susceptibility |
| Calculator | `12`–`14` | resilience "ambition calculator" |
| Main figures | `15`–`17` | trajectories + UpSet; GFP×tipping heatmap; calculator figure |
| Main table | `18` | path-based resilience per scenario (bootstrap + Clopper–Pearson CIs) |
| Supplement | `si/` | SI figures and tables |
| Batch scripts | `batch/` | example SLURM submission scripts (`run_*.sh`) |

> **The `batch/run_*.sh` SLURM scripts are cluster-specific examples.** Adapt the `--account`, `--chdir` and `module`/`conda` lines for your HPC.

---

## Installation

### Requirements
- Python `3.14` (other versions may work; exact reproducibility is only guaranteed with this one)
- The dependencies pinned in [`environment.yml`](environment.yml) / [`requirements.txt`](requirements.txt) (fair 2.2.3, numpy, pandas, xarray, scipy, matplotlib, netCDF4, statsmodels, scikit-learn, SALib, pyDOE, cmcrameri, climateforcing)

### Clone
```bash
git clone https://github.com/zugnachpankow/Path-based-Earth-system-resilience-public.git
cd Path-based-Earth-system-resilience
```

### Environment (conda recommended)
```bash
conda env create -f environment.yml
conda activate fair
```

---

## Usage
Run the numbered scripts in order from the repo root. Each script reads the pipeline's pre-computed outputs and writes to `output/` (see below). The fair ensemble, pycascades tipping sampler and confirmator are heavy and are intended for the cluster (`batch/run_*.sh`); the figure/table scripts are light and run on a workstation.

### Reproducing the figures/tables
1. Obtain the input data (see [Data](#data)): download NGFS + RCMIP, then run `00_clean_NGFS_IAM.py`.
2. Regenerate intermediate output by running `01`–`14`.
3. Run the figure/table scripts (`15`–`18` and `si/`).

---

## Data

This repository ships **only** the small, openly-licensed **fair calibration** inputs. RCMIP and (lincensed) NGFS data must be downloaded and placed in the right folder as indicated below. All pipeline intermediates and results are regenerated and are not tracked.

### Included — `data/raw/` (fair 1.4.1 calibration, openly licensed)
`calibrated_constrained_parameters_calibration1.4.1.csv`,
`species_configs_properties_calibration1.4.1.csv`, `species_configs_properties_NGFS.csv`,
`fair_units_reference.csv`, `fair_variables_reference.csv`, `volcanic_solar.csv`.
Source: Smith, C. (2024). *fair calibration data* (v1.4.1). Zenodo. https://doi.org/10.5281/zenodo.10566813

### Add

**1. NGFS scenario emissions.**
Download the IAM-output workbook from the NGFS data portal
(https://data.ece.iiasa.ac.at/ngfs), save it as `data/NGFS_raw/IAM_data.xlsx`
(sheet `data`), then run:
```bash
python 00_clean_NGFS_IAM.py
```
This reproduces `data/raw/NGFS_cleaned_IAM_all.csv`, the input to `01_clean_and_extend_NGFS.py`.
Source: Richters, O., Kriegler, E., Bertram, C., et al. *NGFS Climate Scenarios Data Set* (5 Nov 2024). https://doi.org/10.5281/zenodo.13989530

> **Licensed data must never be committed.** `data/NGFS_raw/`, `data/raw/NGFS_cleaned_IAM_all.csv`
> and `data/processed/` are git-ignored for this reason — do not force-add them.

**2. RCMIP concentrations / emissions / forcing — open, not shipped (size).**
Download from Zenodo and copy the three `rcmip-*.csv` files into `data/raw/`.
Source: Nicholls, Z., & Lewis, J. (2021). *RCMIP protocol* (v5.1.0). Zenodo. https://doi.org/10.5281/zenodo.4589756

### Intermediate + output data
`data/processed/` (NGFS-derived intermediates) and `output/` (all results) are regenerated by the pipeline and are git-ignored.

---

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) — see [`LICENSE`](LICENSE).

## Citation
If you use this code, please cite:
- Software: <!-- FILL IN: Zenodo software DOI -->
- The accompanying publication: TBD

## Contact
- Max Bechthold — <maxbecht@pik-potsdam.de>

## Paper co-authors
- John M. Anderies
- Jobst Heitzig
- Johan Rockström
- Chris Smith
- Ricarda Winkelmann
- Nico Wunderling
- Jonathan Donges