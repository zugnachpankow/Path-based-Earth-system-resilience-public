"""STEP 2 tests: the v1.4.0 runner forcing/parameter wiring.

Run standalone (pytest is not installed in the HPC env):

    python dev/tests/test_fair_v140_runner.py

The headline test reproduces the fair-calibrate v1.4.0 reference projection
(05_constrained-ssp-projections.py) for ssp245: the *raw source* parameter file
driven through the 05 script's verbatim ``fill`` calls must agree, to 1e-6 K over
1750-2300, with our *built* files driven through the runner helpers
(``src/fair_config.py``). This jointly validates STEP 1 (the build) and STEP 2
(the runner's natural-forcing + override wiring).

RCMIP source files are excluded from the public repo; set ERI_DATA_RAW to a dir
that contains them (defaults to this repo's data/raw). Tests skip with a clear
message if the files are missing.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
DATA_RAW = os.path.join(REPO, "data", "raw")
# RCMIP location (overridable); defaults to the repo's data/raw.
ERI_DATA_RAW = os.environ.get("ERI_DATA_RAW", DATA_RAW)

PARAMS = os.path.join(DATA_RAW, "calibrated_constrained_parameters_calibration1.4.0.csv")
SPECIES = os.path.join(DATA_RAW, "species_configs_properties_calibration1.4.0.csv")
SRC_PARAMS = os.path.join(DATA_RAW, "calibrated_constrained_parameters.csv")
SRC_CH4 = os.path.join(DATA_RAW, "CH4_lifetime.csv")
SOLAR_ERF = os.path.join(DATA_RAW, "solar_erf_timebounds.csv")
VOLCANIC_ERF = os.path.join(DATA_RAW, "volcanic_ERF_1750-2101_timebounds.csv")
VOLCANIC_SOLAR = os.path.join(DATA_RAW, "volcanic_solar.csv")

EESC = "Equivalent effective stratospheric chlorine"
MINOR_GASES = [
    "CFC-11", "CFC-12", "CFC-113", "CFC-114", "CFC-115", "HCFC-22", "HCFC-141b",
    "HCFC-142b", "CCl4", "CHCl3", "CH2Cl2", "CH3Cl", "CH3CCl3", "CH3Br",
    "Halon-1211", "Halon-1301", "Halon-2402", "CF4", "C2F6", "C3F8", "c-C4F8",
    "C4F10", "C5F12", "C6F14", "C7F16", "C8F18", "NF3", "SF6", "SO2F2",
    "HFC-125", "HFC-134a", "HFC-143a", "HFC-152a", "HFC-227ea", "HFC-23",
    "HFC-236fa", "HFC-245fa", "HFC-32", "HFC-365mfc", "HFC-4310mee",
]

RCMIP_FILES = {
    "emissions_file": os.path.join(ERI_DATA_RAW, "rcmip-emissions-annual-means-v5-1-0.csv"),
    "concentration_file": os.path.join(ERI_DATA_RAW, "rcmip-concentrations-annual-means-v5-1-0.csv"),
    "forcing_file": os.path.join(ERI_DATA_RAW, "rcmip-radiative-forcing-annual-means-v5-1-0.csv"),
}


def _rcmip_available():
    return all(os.path.exists(p) for p in RCMIP_FILES.values())


# --------------------------------------------------------------------------- #
# unit tests on the fair_config helpers
# --------------------------------------------------------------------------- #

def test_natural_forcing_matches_fair_calibrate():
    """volcanic_solar.csv (pipeline natural forcing) == fair-calibrate files."""
    from fair_config import natural_forcing
    vs = pd.read_csv(VOLCANIC_SOLAR)
    yearcols = [c for c in vs.columns if c.isdigit()]
    years = np.array([int(c) for c in yearcols])
    solar = pd.read_csv(SOLAR_ERF, index_col="year")["erf"]
    volc = pd.read_csv(VOLCANIC_ERF, index_col="timebounds")["erf"]

    for var, series in [("Solar", solar), ("Volcanic", volc)]:
        row = vs[vs["Variable"] == var].iloc[0]
        vals = pd.Series(row[yearcols].astype(float).values, index=years)
        common = [y for y in years if y in series.index]
        d = np.max(np.abs(vals.loc[common].values - series.loc[common].values))
        assert d < 3e-8, f"{var} natural forcing differs by {d:.2e} (> 3e-8)"

    # natural_forcing returns the fair-calibrate values on the given timebounds
    tb = np.arange(1750, 2301)
    vol_a, sol_a = natural_forcing(tb)
    assert np.allclose(sol_a, solar.loc[1750:2300].values, atol=0, rtol=0)
    assert np.allclose(vol_a[:352], volc.loc[1750:2101].values, atol=0, rtol=0)
    assert np.all(vol_a[352:] == 0.0), "volcanic forcing must be zero after 2101"
    print("PASS test_natural_forcing_matches_fair_calibrate "
          "(volcanic_solar.csv == fair-calibrate within 3e-8; helper exact)")


def test_solar_trend_shape():
    from fair_config import solar_trend_shape
    tb = np.arange(1750, 2301)
    shape = solar_trend_shape(tb)
    assert shape[0] == 0.0                       # 1750
    assert np.isclose(shape[2020 - 1750], 1.0)   # 2020
    assert np.all(shape[2020 - 1750:] == 1.0)    # flat at 1 after 2020
    # linear over the ramp (matches 05's linspace(0,1,271))
    assert np.allclose(shape[:271], np.linspace(0, 1, 271))
    print("PASS test_solar_trend_shape (0 at 1750, 1 at 2020, flat after; linear ramp)")


def test_load_fair_params():
    from fair_config import load_fair_params
    df_fair, solar = load_fair_params(PARAMS)
    assert "fscale_solar_amplitude" not in df_fair.columns
    assert "fscale_solar_trend" not in df_fair.columns
    assert list(solar.columns) == ["fscale_solar_amplitude", "fscale_solar_trend"]
    assert len(df_fair) == 841 and len(solar) == 841
    print("PASS test_load_fair_params (solar columns split off; df_fair override-safe)")


# --------------------------------------------------------------------------- #
# reproduction of the fair-calibrate v1.4.0 reference (05 script) for ssp245
# --------------------------------------------------------------------------- #

def _base_fair(configs):
    from fair import FAIR
    from fair.io import read_properties
    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2300, 1)
    f.define_scenarios(["ssp245"])
    f.define_configs(configs)
    species, properties = read_properties()
    for drop in ["Halon-1202", "NOx aviation", "Contrails"]:
        species.remove(drop)
    f.define_species(species, properties)
    f.allocate()
    return f


def _build_reference(configs):
    """FAIR set up by the 05 script's verbatim fill calls, from the raw source."""
    from fair.interface import fill, initialise
    from rcmip import fill_from_rcmip_locally
    src = pd.read_csv(SRC_PARAMS, index_col=0).loc[configs]
    meth = pd.read_csv(SRC_CH4, index_col=0).loc["historical_best"]

    f = _base_fair(configs)
    fill_from_rcmip_locally(f, **RCMIP_FILES)

    # natural forcing (05 script, verbatim construction)
    n = len(f.timebounds)
    solar = pd.read_csv(SOLAR_ERF, index_col="year")["erf"]
    volc = pd.read_csv(VOLCANIC_ERF, index_col="timebounds")["erf"]
    volcanic_forcing = np.zeros(n)
    volcanic_forcing[:352] = volc.loc[1750:2101].values
    solar_forcing = solar.loc[1750:2300].values
    trend_shape = np.ones(n)
    trend_shape[:271] = np.linspace(0, 1, 271)
    fill(f.forcing,
         volcanic_forcing[:, None, None] * src["fscale_Volcanic"].values.squeeze(),
         specie="Volcanic")
    fill(f.forcing,
         solar_forcing[:, None, None] * src["fscale_solar_amplitude"].values.squeeze()
         + trend_shape[:, None, None] * src["fscale_solar_trend"].values.squeeze(),
         specie="Solar")

    # climate response
    fill(f.climate_configs["ocean_heat_capacity"], src.loc[:, "clim_c1":"clim_c3"].values)
    fill(f.climate_configs["ocean_heat_transfer"], src.loc[:, "clim_kappa1":"clim_kappa3"].values)
    fill(f.climate_configs["deep_ocean_efficacy"], src["clim_epsilon"].values.squeeze())
    fill(f.climate_configs["gamma_autocorrelation"], src["clim_gamma"].values.squeeze())
    fill(f.climate_configs["sigma_eta"], src["clim_sigma_eta"].values.squeeze())
    fill(f.climate_configs["sigma_xi"], src["clim_sigma_xi"].values.squeeze())
    fill(f.climate_configs["seed"], src["seed"])
    fill(f.climate_configs["stochastic_run"], False)   # stochastic OFF
    fill(f.climate_configs["use_seed"], False)
    fill(f.climate_configs["forcing_4co2"], src["clim_F_4xCO2"])

    f.fill_species_configs()

    # carbon cycle
    fill(f.species_configs["iirf_0"], src["cc_r0"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_airborne"], src["cc_rA"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_uptake"], src["cc_rU"].values.squeeze(), specie="CO2")
    fill(f.species_configs["iirf_temperature"], src["cc_rT"].values.squeeze(), specie="CO2")

    # aerosol-cloud
    fill(f.species_configs["aci_scale"], src["aci_beta"].values.squeeze())
    fill(f.species_configs["aci_shape"], src["aci_shape_so2"].values.squeeze(), specie="Sulfur")
    fill(f.species_configs["aci_shape"], src["aci_shape_bc"].values.squeeze(), specie="BC")
    fill(f.species_configs["aci_shape"], src["aci_shape_oc"].values.squeeze(), specie="OC")

    # methane lifetime
    fill(f.species_configs["unperturbed_lifetime"], meth["base"], specie="CH4")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"], meth["CH4"], specie="CH4")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"], meth["N2O"], specie="N2O")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"], meth["VOC"], specie="VOC")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"], meth["NOx"], specie="NOx")
    fill(f.species_configs["ch4_lifetime_chemical_sensitivity"], meth["HC"], specie=EESC)
    fill(f.species_configs["lifetime_temperature_sensitivity"], meth["temp"])

    # baseline emissions
    fill(f.species_configs["baseline_emissions"], 19.019783117809567, specie="CH4")
    fill(f.species_configs["baseline_emissions"], 0.08602230754, specie="N2O")
    fill(f.species_configs["baseline_emissions"], 19.423526730206152, specie="NOx")

    # aerosol-radiation
    for specie in ["BC", "CH4", "N2O", "NH3", "NOx", "OC", "Sulfur", "VOC", EESC]:
        fill(f.species_configs["erfari_radiative_efficiency"], src[f"ari_{specie}"], specie=specie)

    # forcing scaling
    for specie in ["CO2", "CH4", "N2O", "Stratospheric water vapour",
                   "Light absorbing particles on snow and ice", "Land use"]:
        fill(f.species_configs["forcing_scale"], src[f"fscale_{specie}"].values.squeeze(), specie=specie)
    for specie in MINOR_GASES:
        fill(f.species_configs["forcing_scale"], src["fscale_minorGHG"].values.squeeze(), specie=specie)

    # ozone
    for specie in ["CH4", "N2O", EESC, "CO", "VOC", "NOx"]:
        fill(f.species_configs["ozone_radiative_efficiency"], src[f"o3_{specie}"], specie=specie)

    # volcanic efficacy + CO2 initial condition
    fill(f.species_configs["forcing_efficacy"], 0.6, specie="Volcanic")
    fill(f.species_configs["baseline_concentration"],
         src["cc_co2_concentration_1750"].values.squeeze(), specie="CO2")

    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    f.run(progress=False)
    return f


def _build_runner(configs):
    """FAIR set up exactly as monte_carlo_runner does, from the built files."""
    from fair.interface import fill, initialise
    from rcmip import fill_from_rcmip_locally
    from fair_config import (load_fair_params, fair_override_file,
                             natural_forcing, solar_trend_shape)

    df_fair, solar = load_fair_params(PARAMS)
    df_fair = df_fair.loc[configs]
    solar = solar.loc[configs]

    f = _base_fair(configs)
    fill_from_rcmip_locally(f, **RCMIP_FILES)

    volcanic, solar_erf = natural_forcing(f.timebounds)
    trend = solar_trend_shape(f.timebounds)
    fill(f.forcing,
         volcanic[:, None, None] * df_fair["forcing_scale[Volcanic]"].values.squeeze(),
         specie="Volcanic")
    fill(f.forcing,
         solar_erf[:, None, None] * solar["fscale_solar_amplitude"].values.squeeze()
         + trend[:, None, None] * solar["fscale_solar_trend"].values.squeeze(),
         specie="Solar")

    f.fill_species_configs(SPECIES)
    with fair_override_file(df_fair) as ov:
        f.override_defaults(ov)

    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)

    # deterministic run (stochastic off), as the runner does for a single run
    for config in configs:
        fill(f.climate_configs["stochastic_run"], False, config=config)
        fill(f.climate_configs["use_seed"], False, config=config)
    f.run(progress=False)
    return f


def test_runner_reproduces_05_ssp245():
    if not _rcmip_available():
        print("SKIP test_runner_reproduces_05_ssp245 (RCMIP files not found; "
              "set ERI_DATA_RAW)")
        return
    configs = list(pd.read_csv(PARAMS, index_col=0).index[:20])
    f_ref = _build_reference(configs)
    f_run = _build_runner(configs)
    t_ref = f_ref.temperature.sel(layer=0).values
    t_run = f_run.temperature.sel(layer=0).values
    assert np.isfinite(t_ref).all() and np.isfinite(t_run).all()
    dmax = float(np.max(np.abs(t_ref - t_run)))
    assert dmax < 1e-6, f"runner vs 05 reference differ by {dmax:.2e} K (> 1e-6)"
    print(f"PASS test_runner_reproduces_05_ssp245 "
          f"(20 configs, ssp245 1750-2300, max|dT|={dmax:.2e} K < 1e-6)")


if __name__ == "__main__":
    fns = [
        test_natural_forcing_matches_fair_calibrate,
        test_solar_trend_shape,
        test_load_fair_params,
        test_runner_reproduces_05_ssp245,
    ]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
