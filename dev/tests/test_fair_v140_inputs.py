"""STEP 1 tests: the FAIR calibration v1.4.0 input files.

Run standalone (pytest is not installed in the HPC env):

    python tests/test_fair_v140_inputs.py

RCMIP source files are excluded from the public repo; for the FAIR spot-check and
the 1750-baseline comparison they are read from the ERI-cleaned working copy via a
path override (nothing is written into this repo's data/).
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
DATA_RAW = os.path.join(REPO, "data", "raw")

PARAMS = os.path.join(DATA_RAW, "calibrated_constrained_parameters_calibration1.4.0.csv")
SPECIES = os.path.join(DATA_RAW, "species_configs_properties_calibration1.4.0.csv")
SPECIES_NGFS = os.path.join(DATA_RAW, "species_configs_properties_NGFS_calibration1.4.0.csv")
SRC_PARAMS = os.path.join(DATA_RAW, "calibrated_constrained_parameters.csv")
SPECIES_141 = os.path.join(DATA_RAW, "species_configs_properties_calibration1.4.1.csv")

# RCMIP source files are excluded from the public repo; set ERI_DATA_RAW to a dir
# that contains them (defaults to this repo's data/raw). Tests skip if missing.
ERI_RAW = os.environ.get("ERI_DATA_RAW", DATA_RAW)


def test_params_shape_index_nonan():
    src = pd.read_csv(SRC_PARAMS, index_col=0)
    out = pd.read_csv(PARAMS, index_col=0)
    assert out.shape[0] == 841, out.shape
    assert list(out.index) == list(src.index), "index must match the source configs"
    assert not out.isnull().any().any(), "no NaNs allowed in the parameter table"
    # schema: 1.4.1 columns minus forcing_scale[Solar], plus the two solar extras
    ref = list(pd.read_csv(
        os.path.join(DATA_RAW, "calibrated_constrained_parameters_calibration1.4.1.csv"),
        index_col=0, nrows=0).columns)
    expected = [c for c in ref if c != "forcing_scale[Solar]"] + [
        "fscale_solar_amplitude", "fscale_solar_trend"]
    assert list(out.columns) == expected, "unexpected column layout"
    print("PASS test_params_shape_index_nonan "
          f"(841 x {out.shape[1]}, index matches source, no NaN)")


def test_params_rename_values():
    """Spot-check the rename map carried the source values through verbatim."""
    src = pd.read_csv(SRC_PARAMS, index_col=0)
    out = pd.read_csv(PARAMS, index_col=0)
    checks = {
        "gamma_autocorrelation": "clim_gamma",
        "ocean_heat_capacity[0]": "clim_c1",
        "ocean_heat_transfer[2]": "clim_kappa3",
        "deep_ocean_efficacy": "clim_epsilon",
        "forcing_4co2": "clim_F_4xCO2",
        "iirf_uptake[CO2]": "cc_rU",
        "iirf_airborne[CO2]": "cc_rA",
        "erfari_radiative_efficiency[NOx]": "ari_NOx",
        "aci_shape[Sulfur]": "aci_shape_so2",
        "aci_scale": "aci_beta",
        "ozone_radiative_efficiency[NOx]": "o3_NOx",
        "forcing_scale[Volcanic]": "fscale_Volcanic",
        "forcing_scale[CO2]": "fscale_CO2",
        "baseline_concentration[CO2]": "cc_co2_concentration_1750",
        "fscale_solar_amplitude": "fscale_solar_amplitude",
        "fscale_solar_trend": "fscale_solar_trend",
        "seed": "seed",
    }
    for fair_col, src_col in checks.items():
        assert np.allclose(out[fair_col].values, src[src_col].values), fair_col
    # all 40 minor gases carry the single fscale_minorGHG value
    for gas in ["CFC-11", "HFC-245fa", "SF6", "Halon-1301"]:
        assert np.allclose(out[f"forcing_scale[{gas}]"].values,
                           src["fscale_minorGHG"].values), gas
    assert out["stochastic_run"].all() and out["use_seed"].all()
    print("PASS test_params_rename_values (rename map + minorGHG broadcast verbatim)")


def test_species_names_match_141():
    sp = pd.read_csv(SPECIES)
    sp_ngfs = pd.read_csv(SPECIES_NGFS)
    s141 = pd.read_csv(SPECIES_141)
    assert list(sp["name"]) == list(s141["name"]), "full species list must match 1.4.1"
    assert len(sp) == 61 and len(sp_ngfs) == 60
    assert set(sp["name"]) - set(sp_ngfs["name"]) == {"HFC-245fa"}
    for drop in ["Halon-1202", "NOx aviation", "Contrails"]:
        assert drop not in set(sp["name"])
    print("PASS test_species_names_match_141 (61 full / 60 NGFS, HFC-245fa only diff)")


def test_species_overrides():
    sp = pd.read_csv(SPECIES).set_index("name")
    ch4 = pd.read_csv(os.path.join(DATA_RAW, "CH4_lifetime.csv"), index_col=0).loc["historical_best"]
    for i in range(4):
        assert np.isclose(sp.loc["CH4", f"unperturbed_lifetime{i}"], ch4["base"])
    assert np.isclose(sp.loc["CH4", "ch4_lifetime_chemical_sensitivity"], ch4["CH4"])
    assert np.isclose(sp.loc["N2O", "ch4_lifetime_chemical_sensitivity"], ch4["N2O"])
    assert np.isclose(sp.loc["VOC", "ch4_lifetime_chemical_sensitivity"], ch4["VOC"])
    assert np.isclose(sp.loc["NOx", "ch4_lifetime_chemical_sensitivity"], ch4["NOx"])
    assert np.isclose(
        sp.loc["Equivalent effective stratospheric chlorine", "ch4_lifetime_chemical_sensitivity"],
        ch4["HC"])
    assert np.isclose(sp.loc["CH4", "lifetime_temperature_sensitivity"], ch4["temp"])
    assert np.isclose(sp.loc["CH4", "baseline_emissions"], 19.019783117809567)
    assert np.isclose(sp.loc["N2O", "baseline_emissions"], 0.08602230754)
    assert np.isclose(sp.loc["NOx", "baseline_emissions"], 19.423526730206152)
    assert np.isclose(sp.loc["Volcanic", "forcing_efficacy"], 0.6)
    # h2o_stratospheric_factor kept at fair 2.2.3 default
    assert np.isclose(sp.loc["CH4", "h2o_stratospheric_factor"], 4.39704431e-05)
    print("PASS test_species_overrides (CH4 lifetime/chem, baselines, Volcanic 0.6)")


def test_baselines_vs_rcmip_1750():
    """CH4/N2O baseline_emissions equal RCMIP 1750 ssp245 World (native units)."""
    rcmip = os.path.join(ERI_RAW, "rcmip-emissions-annual-means-v5-1-0.csv")
    if not os.path.exists(rcmip):
        print("SKIP test_baselines_vs_rcmip_1750 (RCMIP emissions not found)")
        return
    df = pd.read_csv(rcmip)
    sel = df[(df["Scenario"] == "ssp245") & (df["Region"] == "World")]
    sp = pd.read_csv(SPECIES).set_index("name")

    def rcmip_1750(suffix, unit_scale=1.0):
        row = sel[sel["Variable"] == f"Emissions|{suffix}"]
        return float(row["1750"].values[0]) * unit_scale

    # CH4, N2O reported in Mt and kt respectively; fair baseline_emissions are in
    # Mt CH4 and Mt N2O -> N2O needs kt->Mt (/1000).
    ch4_1750 = rcmip_1750("CH4")
    n2o_1750 = rcmip_1750("N2O", 1e-3)
    assert np.isclose(sp.loc["CH4", "baseline_emissions"], ch4_1750, rtol=1e-6), \
        (sp.loc["CH4", "baseline_emissions"], ch4_1750)
    assert np.isclose(sp.loc["N2O", "baseline_emissions"], n2o_1750, rtol=1e-6), \
        (sp.loc["N2O", "baseline_emissions"], n2o_1750)
    print(f"PASS test_baselines_vs_rcmip_1750 (CH4 {ch4_1750:.6f} Mt, N2O {n2o_1750:.8f} Mt; "
          "NOx baseline uses the STEP-3 sector splice, checked there)")


def test_fair_spotcheck_3configs():
    """Load the new files into FAIR for 3 configs and confirm a finite run."""
    rcmip_files = {
        "emissions_file": os.path.join(ERI_RAW, "rcmip-emissions-annual-means-v5-1-0.csv"),
        "concentration_file": os.path.join(ERI_RAW, "rcmip-concentrations-annual-means-v5-1-0.csv"),
        "forcing_file": os.path.join(ERI_RAW, "rcmip-radiative-forcing-annual-means-v5-1-0.csv"),
    }
    if not all(os.path.exists(p) for p in rcmip_files.values()):
        print("SKIP test_fair_spotcheck_3configs (RCMIP files not found)")
        return
    from rcmip import fill_from_rcmip_locally
    from fair import FAIR
    from fair.interface import initialise
    from fair.io import read_properties

    configs = list(pd.read_csv(PARAMS, index_col=0).index[:3])

    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2300, 1)
    f.define_scenarios(["ssp245"])
    f.define_configs(configs)
    species, properties = read_properties(filename=SPECIES)
    f.define_species(species, properties)
    f.allocate()
    fill_from_rcmip_locally(f, **rcmip_files)
    f.fill_species_configs(SPECIES)

    # override_defaults only understands fair config columns; drop the two extra
    # solar columns (consumed by the runner, not fair) for this load.
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    df = pd.read_csv(PARAMS, index_col=0).loc[configs].drop(
        columns=["fscale_solar_amplitude", "fscale_solar_trend"])
    df.to_csv(tmp.name)
    f.override_defaults(tmp.name)
    os.unlink(tmp.name)

    initialise(f.concentration, f.species_configs["baseline_concentration"])
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.cumulative_emissions, 0)
    initialise(f.airborne_emissions, 0)
    f.run(progress=False)

    temp = f.temperature.sel(layer=0).values
    assert np.isfinite(temp).all(), "FAIR produced non-finite temperatures"
    # sanity: 2100 ssp245 warming above 1995-2014 is positive and plausible
    t2100 = f.temperature.sel(layer=0)[350, 0, :].values.mean()
    print(f"PASS test_fair_spotcheck_3configs (3 configs, finite 1750-2300; "
          f"mean T[2100]={t2100:.3f} K abs)")


if __name__ == "__main__":
    fns = [
        test_params_shape_index_nonan,
        test_params_rename_values,
        test_species_names_match_141,
        test_species_overrides,
        test_baselines_vs_rcmip_1750,
        test_fair_spotcheck_3configs,
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
