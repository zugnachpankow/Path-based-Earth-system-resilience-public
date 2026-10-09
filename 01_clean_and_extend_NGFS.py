import pandas as pd
import re
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from rcmip import clean_rcmip_historical_for_fair
from emission_floor import build_floor_map

"""This script cleans and extends the NGFS scenarios for use in FAIR."""

# before loading need to clean the ° by hand, because cannot be read otherwise, also cleaned and shortened variable names
df_to_clean = pd.read_csv("data/raw/NGFS_cleaned_IAM_all.csv")

# put headers into emission file format
def transform_header(col):
    """
    Transform column name:
    - make lowercase
    - if it's a year integer, convert '2000' --> '2000.5'
    """
    col_lower = col.lower()
    # Check if the column is a pure integer (e.g. '2000', '2035', etc.)
    if re.fullmatch(r"\d{3,4}", col_lower):  # 3–4 digits
        return f"{col_lower}.5"
    return col_lower


# apply transformation to all columns
df_to_clean.columns = [transform_header(c) for c in df_to_clean.columns]

# replace whitespaces
df_to_clean[["model", "scenario"]] = (
    df_to_clean[["model", "scenario"]]
    .replace(r"\s+", "_", regex=True)
)

# merge model and scenario
df_to_clean["scenario"] = df_to_clean["model"] + "___" + df_to_clean["scenario"]
df_to_clean = df_to_clean.drop(columns=["model"])

# load list of variables that need to be in fair and then check which ones do not match
fair_variables = pd.read_csv("data/raw/fair_variables_reference.csv")

fair_vars = set(fair_variables["variable"].unique())
ngfs_vars = set(df_to_clean["variable"].unique())

ngfs_not_in_fair = ngfs_vars - fair_vars   # in ngfs but NOT in fair
print(list(ngfs_not_in_fair))

# we see that Basically all variables that include CFC, HCFC, HFC, Halon need to be renamed such that CFC-, HCFC-, HFC-, Halon- is added in the string. Also cC4F8 needs to be c-C4F8 and FFI and AFOLU need to become CO2 FFI and CO2 AFOLU.
def rename_variable(var):
    v = var.strip()

    for prefix in ["CFC", "HCFC", "HFC", "Halon"]:
        if v.startswith(prefix) and not v.startswith(prefix + "-"):
            # turn e.g. CFC11 → CFC-11
            return re.sub(rf"^{prefix}(.*)", rf"{prefix}-\1", v)

    return v

# apply variable renaming to dataframe
df_to_clean["variable"] = df_to_clean["variable"].astype(str).apply(rename_variable)

# clean single ones
rename_map = {"FFI": "CO2 FFI", "AFOLU": "CO2 AFOLU", "cC4F8": "c-C4F8", "HFC-43-10": "HFC-4310mee"}
df_to_clean["variable"] = df_to_clean["variable"].replace(rename_map)

# lets check again
fair_vars = set(fair_variables["variable"].unique())
ngfs_vars = set(df_to_clean["variable"].unique())

ngfs_not_in_fair = ngfs_vars - fair_vars   # in ngfs but NOT in fair
print(list(ngfs_not_in_fair))

# find: need to delete 'Halon-1202' and 'HFC-245ca' from NGFS
df_to_clean = df_to_clean[df_to_clean["variable"] != "Halon-1202"]
df_to_clean = df_to_clean[df_to_clean["variable"] != "HFC-245ca"]

# lets check again
fair_vars = set(fair_variables["variable"].unique())
ngfs_vars = set(df_to_clean["variable"].unique())

ngfs_not_in_fair = ngfs_vars - fair_vars   # in ngfs but NOT in fair
print(list(ngfs_not_in_fair))

# do the same for the units
fair_units = pd.read_csv("data/raw/fair_units_reference.csv")

fair_un = set(fair_units["unit"].unique())
ngfs_un = set(df_to_clean["unit"].unique())

ngfs_not_in_fair = ngfs_un - fair_un   # in ngfs but NOT in fair
print(list(ngfs_not_in_fair))

# we will take care of CO2 and N20 later
def rename_unit(var):
    v = var.strip()

    if v == "kt HFC43-10/yr":
        return "kt HFC4310mee/yr"

    return v

df_to_clean["unit"] = df_to_clean["unit"].astype(str).apply(rename_unit)

# historical period from RCMIP (scenario "historical", annual up to 2014.5), so NGFS
# shares the exact same history as the SSP study (SSP/study.py, via fill_from_rcmip_locally).
# RCMIP historical ends at 2014.5; NGFS futures start at 2020.5, and the 2015-2019 gap is
# filled by the downstream annual interpolation (2_build) / FAIR ingestion.
df1 = clean_rcmip_historical_for_fair(species=fair_variables["variable"].tolist())
df2 = df_to_clean

fair_not_in_ngfs = fair_vars - ngfs_vars
print(list(fair_not_in_ngfs))

# find: need to delete 'HFC-245fa' from fair
df1 = df1[df1["variable"] != "HFC-245fa"]

# unique scenarios to extend
scenarios = df2["scenario"].unique().tolist()

# identify which columns are years
year_cols_df1 = [c for c in df1.columns if c.replace('.', '', 1).isdigit()]
year_cols_df2 = [c for c in df2.columns if c.replace('.', '', 1).isdigit()]

# replicate df1 for each scenario
df1_expanded = pd.concat(
    [df1.assign(scenario=sc) for sc in scenarios],
    ignore_index=True
)

# harmonize units
# make dictionary of variable -> unit from df1
unit_map = df1.groupby("variable")["unit"].first().to_dict()

# define some basic conversions
unit_conversion = {
    ("Mt CO2/yr", "Gt CO2/yr"): 1/1000,
    ("Gt CO2/yr", "Mt CO2/yr"): 1000,
    ("kt CO2/yr", "Mt CO2/yr"): 1/1000,
    ("Mt CH4/yr", "Gt CH4/yr"): 1/1000,
    ("Gt CH4/yr", "Mt CH4/yr"): 1000,
    ("kt SO2/yr", "Mt SO2/yr"): 1/1000,
    ("Mt SO2/yr", "kt SO2/yr"): 1000,
    ("kt N2O/yr", "Mt N2O/yr"): 1/1000,
    ("Mt N2O/yr", "kt N2O/yr"): 1000,
    # RCMIP labels NOx "Mt NOx/yr"; NGFS uses the NO2-mass basis "Mt NO2/yr". fair
    # treats them as identical mass (compound_convert['NO2']['NOx'] == 1.0), so this
    # is a pure relabel to make the merge keys line up.
    ("Mt NO2/yr", "Mt NOx/yr"): 1,
}

# apply conversions
for var, target_unit in unit_map.items():
    mask = df2["variable"] == var
    if mask.any():
        current_unit = df2.loc[mask, "unit"].iloc[0]
        if current_unit != target_unit:
            factor = unit_conversion.get((current_unit, target_unit))
            if factor:
                df2.loc[mask, year_cols_df2] *= factor
                df2.loc[mask, "unit"] = target_unit
                print(f"Converted {current_unit} → {target_unit} for {var}")
            else:
                print(f"No conversion rule from {current_unit} → {target_unit} for {var}")

df2.to_csv("data/processed/NGFS_only_cleaned.csv", index=False)

# check the history block (df1) units against fair after replicating per scenario
ext_units = set(df1_expanded["unit"].unique())
ext_not_in_fair = ext_units - fair_un   # reuse fair_un from the variable-unit check above
print(list(ext_not_in_fair))

# drop all years > 2014.5 (RCMIP historical ends here; NGFS futures resume at 2020.5)
df1_trimmed = df1_expanded.loc[:, ["scenario", "region", "variable", "unit"] +
                               [c for c in year_cols_df1 if float(c) <= 2014.5]]

# append df2 year columns 
# df2’s timeline from 2020.5 onward
df2_years = [c for c in year_cols_df2 if float(c) >= 2020.5]

# merge by scenario, region, variable, unit
merge_keys = ["scenario", "region", "variable", "unit"]

# ensure df2 has only the needed columns
df2_trimmed = df2[merge_keys + df2_years]

# align
df_final = pd.merge(
    df1_trimmed,
    df2_trimmed,
    how="left",
    on=merge_keys,
    validate="one_to_one"
)

assert not df_final.duplicated(merge_keys).any()
check = df_final[df_final[df2_years].isna().any(axis=1)]
assert check.empty, check[merge_keys].drop_duplicates()

print(df_final.dtypes)

df_final = df_final.reset_index(drop=True)
df_final.to_csv("data/processed/NGFS_historic_merged.csv", index=False)

# need to rebuild volcanic and solar forcing files for NGFS as well, using original from calibrated constrained ensemble

# Load existing volcanic and solar forcings
df_vs = pd.read_csv("data/raw/volcanic_solar.csv")

# template rows for Volcanic and Solar
template_volcanic = df_vs[df_vs["Variable"] == "Volcanic"].iloc[0:1]
template_solar = df_vs[df_vs["Variable"] == "Solar"].iloc[0:1]

# prepare list of new rows
new_rows = []

for sc in scenarios:
    for var, template in [("Volcanic", template_volcanic), ("Solar", template_solar)]:
        # Check if this scenario already has this variable
        if not ((df_vs["Scenario"] == sc) & (df_vs["Variable"] == var)).any():
            new_rows.append(template.assign(Scenario=sc))

# combine original df with new rows
df_vs_expanded = pd.concat([df_vs] + new_rows, ignore_index=True)
df_vs_expanded = df_vs_expanded.drop_duplicates()

# save
df_vs_expanded.to_csv("data/processed/NGFS_forcing.csv", index=False)

# extend everything for 2 steps by extrapolation such that we can look at the 20 year running mean in 2100
# NOTE (methods caveat): the extrapolation lets the centred 20-yr running mean reach 2100;
# with the ~5-yr NGFS spacing that window effectively rests on only ~3 points near the end.
def expand_for_running_mean(
    df,
    years,
    year_cols,
    interpolation_range_start=2090.5,
    interpolation_range_end=2100.5,
    extensions_length=2,
    floor_map=None,
):
    # find positions
    idx_start = np.argmin(np.abs(years - interpolation_range_start))
    idx_end   = np.argmin(np.abs(years - interpolation_range_end))

    # build new columns
    last_year = float(year_cols[idx_end])
    extra_years = [last_year + 5*(i + 1) for i in range(extensions_length)]
    extra_cols = [f"{y:.1f}" for y in extra_years]

    new_rows = []

    for _, row in df.iterrows():
        base_vals = row[year_cols].astype(float).values # get values

        # slice for trend fitting
        x = years[idx_start:idx_end+1] # decade's years [2090.5, 2100.5]
        y = base_vals[idx_start:idx_end+1] # decade's values

        # fit degree-1 polynomial (linear)
        a, b = np.polyfit(x, y, 1) 
        
        # extrapolate
        x_extra = np.array(extra_years)
        y_extra = a * x_extra + b

        # clamp the extrapolation tail at the species' natural floor so a declining
        # linear trend cannot run below background (never clamp above the last real
        # value, so already-sub-floor trajectories like net-negative CO2 AFOLU hold).
        if floor_map is not None:
            floor_eff = min(floor_map.get(row["variable"], 0.0), base_vals[idx_end])
            y_extra = np.maximum(y_extra, floor_eff)

        # build new row
        new_row = row.copy()
        for col, val in zip(extra_cols, y_extra):
            new_row[col] = val

        new_rows.append(new_row)

    return pd.DataFrame(new_rows)

# get the years and year columns from df_final
year_cols = [c for c in df_final.columns if c.replace('.', '', 1).isdigit()]
years = np.array([float(c) for c in year_cols])

df_extended = expand_for_running_mean(
    df_final,
    years,
    year_cols,
    interpolation_range_start=2090.5,
    interpolation_range_end=2100.5,
    extensions_length=2,
    floor_map=build_floor_map(df_final),
)

df_extended.to_csv("data/processed/NGFS_historic_merged_extended_to_2110.csv", index=False)