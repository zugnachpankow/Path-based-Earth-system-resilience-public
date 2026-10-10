"""Build the FAIR calibration v1.4.0 input files from the fair-calibrate v1.4.0
source files.

Reads the Zenodo-10566646 posterior parameters (``calibrated_constrained_parameters.csv``)
and the v1.4.0 methane-lifetime calibration (``CH4_lifetime.csv``) and emits the
three files the pipeline consumes:

  data/raw/calibrated_constrained_parameters_calibration1.4.0.csv
  data/raw/species_configs_properties_calibration1.4.0.csv
  data/raw/species_configs_properties_NGFS_calibration1.4.0.csv

The parameter file is the per-config override table in fair's own column naming
(as read by ``FAIR.override_defaults``); the species files are the fair 2.2.3
default ``species_configs_properties`` with the v1.4.0 scalar overrides applied.

Column renaming and override values mirror the reference script
``input/fair-2.1.3/v1.4/AR6-updated_no-contrails/constraining/
05_constrained-ssp-projections.py`` of fair-calibrate at tag v1.4.0, so that a
run driven by these files reproduces the fair-calibrate v1.4.0 projections.

Faithful port: logic/values are kept verbatim from the reference; nothing here
re-derives or "improves" the calibration.
"""
import os

import fair
import pandas as pd

DATA_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

SRC_PARAMS = os.path.join(DATA_RAW, "calibrated_constrained_parameters.csv")
SRC_CH4 = os.path.join(DATA_RAW, "CH4_lifetime.csv")

OUT_PARAMS = os.path.join(DATA_RAW, "calibrated_constrained_parameters_calibration1.4.0.csv")
OUT_SPECIES = os.path.join(DATA_RAW, "species_configs_properties_calibration1.4.0.csv")
OUT_SPECIES_NGFS = os.path.join(DATA_RAW, "species_configs_properties_NGFS_calibration1.4.0.csv")

EESC = "Equivalent effective stratospheric chlorine"

# The 40 minor greenhouse gases that all share the single ``fscale_minorGHG``
# scaling in the source file (order matches the fair-calibrate column layout).
MINOR_GASES = [
    "CFC-11", "CFC-12", "CFC-113", "CFC-114", "CFC-115", "HCFC-22", "HCFC-141b",
    "HCFC-142b", "CCl4", "CHCl3", "CH2Cl2", "CH3Cl", "CH3CCl3", "CH3Br",
    "Halon-1211", "Halon-1301", "Halon-2402", "CF4", "C2F6", "C3F8", "c-C4F8",
    "C4F10", "C5F12", "C6F14", "C7F16", "C8F18", "NF3", "SF6", "SO2F2",
    "HFC-125", "HFC-134a", "HFC-143a", "HFC-152a", "HFC-227ea", "HFC-32",
    "HFC-365mfc", "HFC-4310mee", "HFC-23", "HFC-236fa", "HFC-245fa",
]

# fair-format parameter columns in output order, EXCLUDING forcing_scale[Solar]
# (v1.4.0 scales solar in the runner via the fscale_solar_* columns, appended after
# these two extras). Hard-coded so 00b needs only the v1.4.0 source files -- no 1.4.1
# template. This is the exact schema the pipeline's load_fair_params() expects.
FAIR_PARAM_COLUMNS = (
    ["gamma_autocorrelation"]
    + [f"ocean_heat_capacity[{i}]" for i in range(3)]
    + [f"ocean_heat_transfer[{i}]" for i in range(3)]
    + ["deep_ocean_efficacy", "sigma_eta", "sigma_xi", "forcing_4co2"]
    + ["iirf_0[CO2]", "iirf_uptake[CO2]", "iirf_temperature[CO2]", "iirf_airborne[CO2]"]
    + [f"erfari_radiative_efficiency[{s}]"
       for s in ["BC", "OC", "Sulfur", "NOx", "VOC", "NH3", "CH4", "N2O", EESC]]
    + [f"aci_shape[{s}]" for s in ["Sulfur", "BC", "OC"]]
    + ["aci_scale"]
    + [f"ozone_radiative_efficiency[{s}]"
       for s in ["CH4", "N2O", EESC, "CO", "VOC", "NOx"]]
    + ["forcing_scale[CH4]", "forcing_scale[N2O]"]
    + [f"forcing_scale[{g}]" for g in MINOR_GASES]
    + ["forcing_scale[Stratospheric water vapour]", "forcing_scale[Land use]",
       "forcing_scale[Volcanic]",
       "forcing_scale[Light absorbing particles on snow and ice]",
       "forcing_scale[CO2]"]
    + ["baseline_concentration[CO2]", "seed", "stochastic_run", "use_seed"]
)

# non-fair columns consumed directly by the runner (src/fair_config) for solar scaling
SOLAR_EXTRA_COLUMNS = ["fscale_solar_amplitude", "fscale_solar_trend"]

# species dropped from the fair default (no contrails / no aviation NOx set)
DROP_SPECIES = ["Halon-1202", "NOx aviation", "Contrails"]


def _fair_default_species_file():
    return os.path.join(
        os.path.dirname(fair.__file__),
        "defaults", "data", "ar6", "species_configs_properties.csv",
    )


def _source_column(fair_col):
    """Map one fair parameter-file column name to its source (``clim_*`` etc.) name.

    Returns the source column name, or raises if the fair column is not one this
    builder knows how to fill (guards against silent schema drift).
    """
    # bracketed columns: name[index]
    if "[" in fair_col:
        name, index = fair_col.split("[")[0], fair_col.split("[")[1][:-1]
    else:
        name, index = fair_col, None

    simple = {
        "gamma_autocorrelation": "clim_gamma",
        "deep_ocean_efficacy": "clim_epsilon",
        "sigma_eta": "clim_sigma_eta",
        "sigma_xi": "clim_sigma_xi",
        "forcing_4co2": "clim_F_4xCO2",
    }
    if name in simple and index is None:
        return simple[name]

    if name == "ocean_heat_capacity":
        return f"clim_c{int(index) + 1}"
    if name == "ocean_heat_transfer":
        return f"clim_kappa{int(index) + 1}"

    if name == "iirf_0" and index == "CO2":
        return "cc_r0"
    if name == "iirf_uptake" and index == "CO2":
        return "cc_rU"
    if name == "iirf_temperature" and index == "CO2":
        return "cc_rT"
    if name == "iirf_airborne" and index == "CO2":
        return "cc_rA"

    if name == "erfari_radiative_efficiency":
        return f"ari_{index}"
    if name == "ozone_radiative_efficiency":
        return f"o3_{index}"

    if name == "aci_shape":
        return {"Sulfur": "aci_shape_so2", "BC": "aci_shape_bc", "OC": "aci_shape_oc"}[index]
    if name == "aci_scale":
        return "aci_beta"

    if name == "baseline_concentration" and index == "CO2":
        return "cc_co2_concentration_1750"

    if name == "forcing_scale":
        named = {
            "CH4": "fscale_CH4",
            "N2O": "fscale_N2O",
            "CO2": "fscale_CO2",
            "Stratospheric water vapour": "fscale_Stratospheric water vapour",
            "Land use": "fscale_Land use",
            "Volcanic": "fscale_Volcanic",
            "Light absorbing particles on snow and ice":
                "fscale_Light absorbing particles on snow and ice",
        }
        if index in named:
            return named[index]
        if index in MINOR_GASES:
            return "fscale_minorGHG"

    raise KeyError(f"no source mapping for fair parameter column {fair_col!r}")


def build_parameters():
    """Write the fair-format per-config parameter table for calibration v1.4.0."""
    src = pd.read_csv(SRC_PARAMS, index_col=0)

    out = pd.DataFrame(index=src.index)
    for col in FAIR_PARAM_COLUMNS:
        if col == "seed":
            out[col] = src["seed"].values
        elif col in ("stochastic_run", "use_seed"):
            out[col] = True
        else:
            out[col] = src[_source_column(col)].values

    for col in SOLAR_EXTRA_COLUMNS:
        out[col] = src[col].values

    out.to_csv(OUT_PARAMS)
    return out


def build_species():
    """Write the species_configs files (full + NGFS) for calibration v1.4.0."""
    df_ch4 = pd.read_csv(SRC_CH4, index_col=0)
    best = df_ch4.loc["historical_best"]

    sp = pd.read_csv(_fair_default_species_file())
    sp = sp[~sp["name"].isin(DROP_SPECIES)].reset_index(drop=True)
    # columns overridden with floats below are int64 in the default; cast up-front
    # so assignment doesn't trip pandas' incompatible-dtype warning.
    sp["forcing_efficacy"] = sp["forcing_efficacy"].astype(float)

    def setval(specie, col, value):
        sp.loc[sp["name"] == specie, col] = value

    # methane lifetime baseline (all four gas-partition boxes) and sensitivities
    for i in range(4):
        setval("CH4", f"unperturbed_lifetime{i}", best["base"])
    setval("CH4", "ch4_lifetime_chemical_sensitivity", best["CH4"])
    setval("N2O", "ch4_lifetime_chemical_sensitivity", best["N2O"])
    setval("VOC", "ch4_lifetime_chemical_sensitivity", best["VOC"])
    setval("NOx", "ch4_lifetime_chemical_sensitivity", best["NOx"])
    setval(EESC, "ch4_lifetime_chemical_sensitivity", best["HC"])
    setval("CH4", "lifetime_temperature_sensitivity", best["temp"])

    # emissions adjustments for N2O, CH4 and NOx (fair-calibrate v1.4.0 values)
    setval("CH4", "baseline_emissions", 19.019783117809567)
    setval("N2O", "baseline_emissions", 0.08602230754)
    setval("NOx", "baseline_emissions", 19.423526730206152)

    # tuned-down volcanic efficacy
    setval("Volcanic", "forcing_efficacy", 0.6)

    # h2o_stratospheric_factor intentionally left at the fair 2.2.3 default.

    sp.to_csv(OUT_SPECIES, index=False)

    # NGFS variant: same, minus HFC-245fa (not represented in the NGFS emissions)
    sp_ngfs = sp[sp["name"] != "HFC-245fa"].reset_index(drop=True)
    sp_ngfs.to_csv(OUT_SPECIES_NGFS, index=False)

    return sp, sp_ngfs


if __name__ == "__main__":
    params = build_parameters()
    species, species_ngfs = build_species()
    print(f"wrote {OUT_PARAMS}  ({params.shape[0]} configs x {params.shape[1]} cols)")
    print(f"wrote {OUT_SPECIES}  ({len(species)} species)")
    print(f"wrote {OUT_SPECIES_NGFS}  ({len(species_ngfs)} species)")
