import os, sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from emission_floor import build_floor_map

"""This script builds the branch off scenarios from the NGFS current policy scenario."""

df = pd.read_csv("data/processed/NGFS_historic_merged.csv")

# filter to current policies + model subset
mask_current_policy = df["scenario"].str.contains("Current_Policies")
df_current_policy = df[mask_current_policy].copy()

# just keep REMIND
mask_model = df_current_policy["scenario"].str.contains("REMIND")   # it gets too large otherwise
df_current = df_current_policy[mask_model].copy()

start_interp = 2020.5
final_year = 2300.5
interp_years = np.arange(start_interp, final_year + 0.1, 1.0)   # 2020.5, 2021.5, ..., 2300.5
interp_cols = [str(y) for y in interp_years]

# identify non-year columns and existing year columns
non_year_cols = [c for c in df_current.columns if not c.replace('.', '', 1).isdigit()]

# create the full target column order: keep non-year cols then the new annual years
all_year_cols = [str(y) for y in interp_years]
target_cols = non_year_cols + all_year_cols

# reindex to the annual grid: keeps the original 5-yearly values, adds NaNs for the new years in between
df_interp = df_current.reindex(columns=target_cols).copy()

# interpolate across columns (axis=1). pandas .interpolate does NOT extrapolate: it
# linearly fills interior gaps, and limit_direction='both' back-/forward-FILLS the
# edges (so the last native value, 2100.5, is held constant out to 2300.5 -- not a
# linear trend). The explicit ffill/bfill below is a belt-and-suspenders no-op.
df_interp[all_year_cols] = df_interp[all_year_cols].astype(float).interpolate(axis=1, limit_direction='both')

# if there are still NaNs (unlikely), fill forward/backward as last resort:
df_interp[all_year_cols] = df_interp[all_year_cols].ffill(axis=1).bfill(axis=1)

# recompute year_cols to the new annual grid
year_cols = all_year_cols
years = np.array([float(c) for c in year_cols])

def apply_absolute_reduction(
    row,
    start_year,
    reduction_rate,
    year_cols,
    baseline=0.0,
):
    """
    Absolute case: subtract a constant yearly amount, but never below ``baseline``.

    absolute_amount = reduction_rate * value(start_year), then
    value[t] = max(value[t-1] - absolute_amount, baseline).

    The floor is the species' natural background (fair ``baseline_emissions``). A
    species whose start value is already at or below that background -- including
    net-negative emitters such as CO2 AFOLU -- is held constant from the start year:
    there is nothing left to reduce, and the old code (abs_amount = rate*start_value,
    which is < 0 for a negative start) made the value RISE every year instead.
    """
    out = row.copy()
    years_float = np.array([float(y) for y in year_cols])
    values = row[year_cols].astype(float).values

    # locate start index
    start_idx_arr = np.where(years_float >= start_year)[0]
    if len(start_idx_arr) == 0:
        start_idx = len(values) - 1
    else:
        start_idx = start_idx_arr[0]

    start_value = values[start_idx]

    if start_value <= baseline:
        # at/below the natural floor already -> hold constant from the start year
        values[start_idx + 1:] = start_value
    else:
        abs_amount = start_value * reduction_rate  # constant amount per year
        for i in range(start_idx + 1, len(values)):
            values[i] = max(values[i - 1] - abs_amount, baseline)

    out[year_cols] = values
    if np.isnan(values).any():
        raise ValueError("NaNs produced in apply_absolute_reduction")
    return out


#  apply for multiple start years and rates
reduction_rates = np.arange(0.01, 0.105, 0.01)    # 0% .. 10% inclusive
starting_years = np.arange(2020.5, 2055.5, 5.0)    # 2020.5 .. 2100.5 inclusive

# natural background floor per species (fair baseline_emissions, converted to
# each variable's CSV unit). reductions clamp here, not at 0 (see
# apply_absolute_reduction); 17 of the 50 emitted species have a non-zero natural
# source, the rest floor at 0.
baseline_map = build_floor_map(df_interp)

all_results = []

for r in reduction_rates:
    for sy in starting_years:
        df_abs = df_interp.apply(
            lambda row: apply_absolute_reduction(
                row, sy, r, year_cols,
                baseline=baseline_map.get(row["variable"], 0.0)
            ),
            axis=1
        )
        df_abs = df_abs.copy()
        df_abs["base_scenario"] = df_abs["scenario"]
        rate_tag = f"{r:.2f}"
        df_abs["scenario"] += f"__start{sy}_r{rate_tag}"
        all_results.append(df_abs)

df_scenarios = pd.concat(all_results, ignore_index=True, sort=False)

# ensure consistent column ordering (non-year cols then year cols)
non_year_cols_after = [c for c in df_scenarios.columns if not c.replace('.', '', 1).isdigit()]
df_scenarios = df_scenarios[non_year_cols_after + year_cols]

# --- pre-2020 Current Policy ---
mask_cp = df["scenario"].str.contains("Current_Policies")
df_cp_pre2020 = df.loc[mask_cp].copy()

pre2020_cols = [c for c in df.columns if c.replace('.', '', 1).isdigit() and float(c) <= 2019.5]

# --- prepare the post-2020 branch-offs ---
df_generated = df_scenarios.copy()

# --- replicate pre2020 for each generated scenario ---
# pre-2020 values depend only on region/variable/unit (shared history), so attach them in one merge
pre2020_lookup = (
    df_cp_pre2020[['region', 'variable', 'unit'] + pre2020_cols]
    .drop_duplicates(['region', 'variable', 'unit'])
)
df_final = df_generated.merge(
    pre2020_lookup, on=['region', 'variable', 'unit'], how='left', validate='many_to_one'
)

# reorder columns
df_final = df_final[['scenario', 'region', 'variable', 'unit'] + pre2020_cols + year_cols]

# save
df_final.to_csv("data/processed/current_policy_branch_offs_2300.csv", index=False)

# also need the corresponding forcing files
# Load existing volcanic and solar forcings
df_vs = pd.read_csv("data/raw/volcanic_solar.csv")

# all scenarios to include
all_scenarios = df_scenarios["scenario"].unique()

# template rows for Volcanic and Solar
template_volcanic = df_vs[df_vs["Variable"] == "Volcanic"].iloc[0:1]
template_solar = df_vs[df_vs["Variable"] == "Solar"].iloc[0:1]

# prepare list of new rows
new_rows = []

for sc in all_scenarios:
    for var, template in [("Volcanic", template_volcanic), ("Solar", template_solar)]:
        # Check if this scenario already has this variable
        if not ((df_vs["Scenario"] == sc) & (df_vs["Variable"] == var)).any():
            new_rows.append(template.assign(Scenario=sc))

# Combine original df with new rows
df_vs_expanded = pd.concat([df_vs] + new_rows, ignore_index=True)
df_vs_expanded = df_vs_expanded.drop_duplicates()

# Save
df_vs_expanded.to_csv("data/processed/current_policy_branch_offs_2300_forcing.csv", index=False)