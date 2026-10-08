import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from fair.interface import fill
from fair.exceptions import MissingDataError
from fair.structure.units import (
    compound_convert,
    desired_concentration_units,
    desired_emissions_units,
    mixing_ratio_convert,
    prefix_convert,
    time_convert,
)

# RCMIP source files live in the repo's data/raw (downloaded manually on HPC,
# see the data availability statement in the README). Resolved relative to this
# file so callers work regardless of their working directory.
_DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RCMIP_EMISSIONS = str(_DATA_RAW / "rcmip-emissions-annual-means-v5-1-0.csv")
RCMIP_CONCENTRATIONS = str(_DATA_RAW / "rcmip-concentrations-annual-means-v5-1-0.csv")
RCMIP_FORCING = str(_DATA_RAW / "rcmip-radiative-forcing-annual-means-v5-1-0.csv")

def build_species_map(species):
    """Map fair species names to their RCMIP variable-name suffixes.

    Single source of truth shared by every RCMIP reader in this module, so the
    SSP runs and the NGFS harmonisation resolve names identically. The returned
    map only contains keys present in ``species``; specials that aren't part of
    the requested set are pruned (reproducing the original inline behaviour).
    """
    m = {specie: specie.replace("-", "") for specie in species}
    m["CO2 FFI"] = "CO2|MAGICC Fossil and Industrial"
    m["CO2 AFOLU"] = "CO2|MAGICC AFOLU"
    m["NOx aviation"] = "NOx|MAGICC Fossil and Industrial|Aircraft"
    m["Aerosol-radiation interactions"] = "Aerosols-radiation interactions"
    m["Aerosol-cloud interactions"] = "Aerosols-radiation interactions"
    m["Contrails"] = "Contrails and Contrail-induced Cirrus"
    m["Light absorbing particles on snow and ice"] = "BC on Snow"
    m["Stratospheric water vapour"] = "CH4 Oxidation Stratospheric H2O"
    m["Land use"] = "Albedo Change"
    return {specie: name for specie, name in m.items() if specie in species}


def clean_rcmip_historical_for_fair(
    species,
    emissions_file=RCMIP_EMISSIONS,
    region="World",
    last_year=2014,
):
    """Return RCMIP historical emissions as a wide, merge-ready history block.

    Mirrors the per-specie RCMIP lookup of ``fill_from_rcmip_locally``, but
    instead of interpolating onto a fair time grid and filling an xarray, it
    collects one row per specie into a dataframe with columns
    ``["scenario", "region", "variable", "unit"]`` + annual ``"YYYY.5"`` up to
    ``last_year``. Values and units are RCMIP's own (unchanged) -- the unit
    harmonisation against NGFS happens in 1_clean. It deliberately does NOT
    extrapolate past ``last_year``: that is the point of the 2014.5 splice onto
    the NGFS futures.
    """
    # lookup converting fair default names to RCMIP names
    species_to_rcmip = build_species_map(species)

    df_emis = pd.read_csv(emissions_file)

    scenario = "historical"

    # RCMIP stores integer year columns; keep history up to last_year and label
    # them midyear ("YYYY.5") to match the emissions-file / extension convention.
    hist_years = [c for c in df_emis.columns if str(c).isdigit() and int(c) <= last_year]
    out_years = [f"{int(c)}.5" for c in hist_years]

    records = []
    for specie, specie_rcmip_name in species_to_rcmip.items():
        # same selection idiom as fill_from_rcmip_locally
        mask = (
            (df_emis["Scenario"] == scenario)
            & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
            & (df_emis["Region"] == region)
        )

        # throw error if data missing or ambiguous (guards against the endswith
        # map catching a sub-sector or missing a series)
        if mask.sum() != 1:
            raise MissingDataError(
                f"expected exactly 1 value for scenario={scenario}, variable "
                f"name ending with |{specie_rcmip_name}, region={region} in the "
                f"RCMIP emissions database, found {int(mask.sum())}."
            )

        # grab raw historical emissions; interpolate fills any interior gap but
        # does not extrapolate beyond the observed range.
        emis_in = df_emis.loc[mask, hist_years].interpolate(axis=1).values.squeeze()

        # parse unit from input file (kept as-is; 1_clean harmonises vs NGFS)
        unit = df_emis.loc[mask, "Unit"].values[0]

        record = {"scenario": scenario, "region": region, "variable": specie, "unit": unit}
        record.update(dict(zip(out_years, emis_in)))
        records.append(record)

    return pd.DataFrame(records)


def fill_from_rcmip_locally(
    fair_instance,
    emissions_file=RCMIP_EMISSIONS,
    concentration_file=RCMIP_CONCENTRATIONS,
    forcing_file=RCMIP_FORCING,
):
    """
    Fill emissions, concentrations and/or forcing from RCMIP scenarios.

    This is a copy from the FAIR class, adjusted to work on an HPC where downloading might not be possible. 
    It uses local files instead that need to be downloaded manually, see data availability statement in the README.
    """
    # lookup converting fair default names to RCMIP names
    species_to_rcmip = build_species_map(fair_instance.species)

    df_emis = pd.read_csv(emissions_file)
    df_conc = pd.read_csv(concentration_file)
    df_forc = pd.read_csv(forcing_file)

    for scenario in fair_instance.scenarios:
        for specie, specie_rcmip_name in species_to_rcmip.items():
            if fair_instance.properties_df.loc[specie, "input_mode"] == "emissions":
                # grab raw emissions from dataframe
                emis_in = (
                    df_emis.loc[
                        (df_emis["Scenario"] == scenario)
                        & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_emis["Region"] == "World"),
                        "1750":"2500",
                    ]
                    .interpolate(axis=1)
                    .values.squeeze()
                )

                # throw error if data missing
                if emis_in.shape[0] == 0:
                    raise MissingDataError(
                        f"I can't find a value for scenario={scenario}, variable "
                        f"name ending with {specie_rcmip_name} in the RCMIP "
                        f"emissions database."
                    )

                # avoid NaNs from outside the interpolation range being mixed into
                # the results
                notnan = np.nonzero(~np.isnan(emis_in))

                # RCMIP are "annual averages"; for emissions this is basically
                # the emissions over the year, for concentrations and forcing
                # it would be midyear values. In every case, we can assume
                # midyear values and interpolate to our time grid.
                rcmip_index = np.arange(1750.5, 2501.5)
                interpolator = interp1d(
                    rcmip_index[notnan],
                    emis_in[notnan],
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                emis = interpolator(fair_instance.timepoints)

                # we won't throw an error if the time is out of range for RCMIP,
                # but we will fill with NaN to allow a user to manually specify
                # pre- and post- emissions.
                emis[fair_instance.timepoints < 1750] = np.nan
                emis[fair_instance.timepoints > 2501] = np.nan

                # parse and possibly convert unit in input file to what fair wants
                unit = df_emis.loc[
                    (df_emis["Scenario"] == scenario)
                    & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
                    & (df_emis["Region"] == "World"),
                    "Unit",
                ].values[0]
                emis = emis * (
                    prefix_convert[unit.split()[0]][
                        desired_emissions_units[specie].split()[0]
                    ]
                    * compound_convert[unit.split()[1].split("/")[0]][
                        desired_emissions_units[specie].split()[1].split("/")[0]
                    ]
                    * time_convert[unit.split()[1].split("/")[1]][
                        desired_emissions_units[specie].split()[1].split("/")[1]
                    ]
                )  # * fair_instance.timestep

                # fill fair xarray
                fill(fair_instance.emissions, emis[:, None], specie=specie, scenario=scenario)

            if fair_instance.properties_df.loc[specie, "input_mode"] == "concentration":
                # grab raw concentration from dataframe
                conc_in = (
                    df_conc.loc[
                        (df_conc["Scenario"] == scenario)
                        & (df_conc["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_conc["Region"] == "World"),
                        "1700":"2500",
                    ]
                    .interpolate(axis=1)
                    .values.squeeze()
                )

                # throw error if data missing
                if conc_in.shape[0] == 0:
                    raise MissingDataError(
                        f"I can't find a value for scenario={scenario}, variable "
                        f"name ending with {specie_rcmip_name} in the RCMIP "
                        f"concentration database."
                    )

                # avoid nans from outside the interpolation range being mixed into
                # the results
                notnan = np.nonzero(~np.isnan(conc_in))

                # interpolate: this time to timebounds
                rcmip_index = np.arange(1700.5, 2501.5)
                interpolator = interp1d(
                    rcmip_index[notnan],
                    conc_in[notnan],
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                conc = interpolator(fair_instance.timebounds)

                # strip out pre- and post-
                conc[fair_instance.timebounds < 1700] = np.nan
                conc[fair_instance.timebounds > 2501] = np.nan

                # Parse and possibly convert unit in input file to what fair wants
                unit = df_conc.loc[
                    (df_conc["Scenario"] == scenario)
                    & (df_conc["Variable"].str.endswith("|" + specie_rcmip_name))
                    & (df_conc["Region"] == "World"),
                    "Unit",
                ].values[0]
                conc = conc * (
                    mixing_ratio_convert[unit][desired_concentration_units[specie]]
                )

                # fill fair xarray
                fill(
                    fair_instance.concentration,
                    conc[:, None],
                    specie=specie,
                    scenario=scenario,
                )

            if fair_instance.properties_df.loc[specie, "input_mode"] == "forcing":
                # grab raw concentration from dataframe
                forc_in = (
                    df_forc.loc[
                        (df_forc["Scenario"] == scenario)
                        & (df_forc["Variable"].str.endswith("|" + specie_rcmip_name))
                        & (df_forc["Region"] == "World"),
                        "1750":"2500",
                    ]
                    .interpolate(axis=1)
                    .values.squeeze()
                )

                # throw error if data missing
                if forc_in.shape[0] == 0:
                    raise MissingDataError(
                        f"I can't find a value for scenario={scenario}, variable "
                        f"name ending with {specie_rcmip_name} in the RCMIP "
                        f"radiative forcing database."
                    )

                # avoid nans from outside the interpolation range being mixed into
                # the results
                notnan = np.nonzero(~np.isnan(forc_in))

                # interpolate: this time to timebounds
                rcmip_index = np.arange(1750.5, 2501.5)
                interpolator = interp1d(
                    rcmip_index[notnan],
                    forc_in[notnan],
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                forc = interpolator(fair_instance.timebounds)

                # strip out pre- and post-
                forc[fair_instance.timebounds < 1750] = np.nan
                forc[fair_instance.timebounds > 2501] = np.nan

                # Forcing so far is always W m-2, but perhaps this will change.

                # fill fair xarray
                fill(fair_instance.forcing, forc[:, None], specie=specie, scenario=scenario)
