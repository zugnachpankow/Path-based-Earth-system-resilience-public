"""Emission-scenario construction shared by the calculator and the figures.

Kept light (numpy + pandas only, no FAIR/pycascades) so figure scripts can build
the *same* candidate the calculator simulates -- one rule, one place. Reimplementing
this inline is exactly what produced the old Fig. 4b straight-line bug.
"""
import numpy as np
import pandas as pd


def build_candidate_scenario(df_emissions, years, year_cols,
                             base_scenario_name, target_year,
                             reduction_frac, candidate_name,
                             ramp_start_year=2025.5, baseline_map=None):
    """
    Create a new emissions scenario that applies an additional fractional reduction
    to base_scenario_name at target_year, carrying the absolute offset forward.

    ``baseline_map`` {variable: natural background emission} clamps each species at its
    natural floor (FaIR ``baseline_emissions``), matching the Current-Policies branch-off
    grid (02). If None, species clamp at 0 (legacy behaviour). The reduction only ever
    lowers emissions and never raises a species already below its background; at
    ``reduction_frac == 0`` the candidate reproduces the base exactly.

    Parameters
    ----------
    df_emissions      : DataFrame [scenario, region, variable, unit, *year_cols]
    years             : float array matching year_cols
    year_cols         : list of column name strings
    base_scenario_name: exact scenario string in df_emissions
    target_year       : year at which full reduction is reached (e.g. 2035.5)
    reduction_frac    : additional fraction to cut (e.g. 0.10 for 10%)
    candidate_name    : name to assign to new scenario rows
    ramp_start_year   : where ramp begins (default 2025.5)

    Returns
    -------
    DataFrame with only candidate_name rows (same columns as df_emissions)
    """
    i_target = np.argmin(np.abs(years - target_year))

    df_base = df_emissions[df_emissions["scenario"] == base_scenario_name].copy()
    if df_base.empty:
        raise ValueError(f"Scenario '{base_scenario_name}' not found in emissions.")

    new_rows = []
    for _, row in df_base.iterrows():
        new_row  = row.copy()
        vals     = new_row[year_cols].astype(float).values.copy()
        v_target_old = vals[i_target]

        # Apply the reduction as a ramped ABSOLUTE OFFSET subtracted from the base
        # trajectory, preserving the base's shape. The old code overwrote the ramp
        # window with a straight line (v_base -> v_target_new), which linearised the
        # interior (e.g. 2030) and did NOT reproduce the base at reduction_frac == 0.
        # With an offset, R == 0 gives offset == 0 everywhere, so candidate == base.
        difference = v_target_old * reduction_frac          # absolute cut reached at target
        offset = np.zeros_like(vals)
        ramp_mask = (years >= ramp_start_year) & (years <= target_year)
        offset[ramp_mask] = np.linspace(0.0, difference, ramp_mask.sum())
        offset[years > target_year] = difference
        reduced = vals - offset

        # clamp at the species' natural background; a reduction only lowers emissions
        # and never raises a species already below background. min(base, max(reduced,
        # floor)) also keeps R == 0 identically equal to the base.
        flr = 0.0 if baseline_map is None else float(baseline_map.get(str(row["variable"]).strip(), 0.0))
        vals = np.minimum(vals, np.maximum(reduced, flr))

        new_row[year_cols] = vals
        new_row["scenario"] = candidate_name
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)[df_emissions.columns].reset_index(drop=True)
