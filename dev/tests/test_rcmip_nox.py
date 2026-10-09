"""STEP 3 tests: the RCMIP NOx sector reconstruction (src/rcmip.py).

Run standalone (pytest is not installed in the HPC env):

    python dev/tests/test_rcmip_nox.py

RCMIP is excluded from the public repo; set ERI_DATA_RAW to a dir that contains
the rcmip-*.csv files (defaults to this repo's data/raw). Tests skip if missing.
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
DATA_RAW = os.path.join(REPO, "data", "raw")
ERI_DATA_RAW = os.environ.get("ERI_DATA_RAW", DATA_RAW)
RCMIP_EMISSIONS = os.path.join(ERI_DATA_RAW, "rcmip-emissions-annual-means-v5-1-0.csv")

# fair-calibrate v1.4.0 reference values (reconstructed total NOx, Mt NO2/yr)
NOX_1750_REF = 19.423526730206152
NOX_SSP245_2014_REF = 162.66804709882882


def _skip_if_missing():
    if not os.path.exists(RCMIP_EMISSIONS):
        print("SKIP (RCMIP emissions not found; set ERI_DATA_RAW)")
        return True
    return False


def test_nox_1750_reconstructed():
    if _skip_if_missing():
        return
    from rcmip import reconstruct_nox_emissions
    df = pd.read_csv(RCMIP_EMISSIONS)
    val = float(reconstruct_nox_emissions(df, "historical", "World", ["1750"])[0])
    assert np.isclose(val, NOX_1750_REF, rtol=1e-9), (val, NOX_1750_REF)
    # the aggregate series is the wrong (low) value the fix replaces
    agg = float(df[(df["Scenario"] == "historical") & (df["Region"] == "World")
                   & (df["Variable"] == "Emissions|NOx")]["1750"].values[0])
    assert abs(agg - 12.7352) < 1e-3 and val > agg
    print(f"PASS test_nox_1750_reconstructed (1750 NOx = {val:.6f} Mt NO2/yr; "
          f"aggregate {agg:.4f} corrected)")


def test_nox_ssp245_2014_reference():
    if _skip_if_missing():
        return
    from rcmip import reconstruct_nox_emissions
    df = pd.read_csv(RCMIP_EMISSIONS)
    val = float(reconstruct_nox_emissions(df, "ssp245", "World", ["2014"])[0])
    assert np.isclose(val, NOX_SSP245_2014_REF, rtol=1e-9), (val, NOX_SSP245_2014_REF)
    print(f"PASS test_nox_ssp245_2014_reference (ssp245 2014 NOx = {val:.6f} Mt NO2/yr)")


def test_clean_historical_uses_reconstruction():
    if _skip_if_missing():
        return
    from rcmip import clean_rcmip_historical_for_fair
    df = clean_rcmip_historical_for_fair(species=["NOx"], emissions_file=RCMIP_EMISSIONS)
    row = df[df["variable"] == "NOx"].iloc[0]
    assert np.isclose(float(row["1750.5"]), NOX_1750_REF, rtol=1e-9)
    assert np.isclose(float(row["2014.5"]), NOX_SSP245_2014_REF, rtol=1e-9)
    print("PASS test_clean_historical_uses_reconstruction "
          f"(NGFS historical NOx 1750.5={row['1750.5']:.4f}, 2014.5={row['2014.5']:.4f})")


def test_fill_from_rcmip_nox_endtoend():
    """fill_from_rcmip_locally drives the reconstructed NOx into FAIR emissions."""
    if _skip_if_missing():
        return
    from fair import FAIR
    from fair.io import read_properties
    import rcmip
    from rcmip import fill_from_rcmip_locally

    f = FAIR(ch4_method="Thornhill2021")
    f.define_time(1750, 2300, 1)
    f.define_scenarios(["ssp245"])
    f.define_configs(["x"])
    species, properties = read_properties()
    for drop in ["Halon-1202", "NOx aviation", "Contrails"]:
        species.remove(drop)
    f.define_species(species, properties)
    f.allocate()
    paths = dict(
        emissions_file=RCMIP_EMISSIONS,
        concentration_file=os.path.join(ERI_DATA_RAW, "rcmip-concentrations-annual-means-v5-1-0.csv"),
        forcing_file=os.path.join(ERI_DATA_RAW, "rcmip-radiative-forcing-annual-means-v5-1-0.csv"),
    )
    fill_from_rcmip_locally(f, **paths)
    # emissions are on timepoints (midyear); NOx unit is Mt NOx == Mt NO2 (factor 1)
    nox = f.emissions.sel(specie="NOx", scenario="ssp245").values.squeeze()
    tp = f.timepoints
    nox_1750 = float(nox[np.argmin(np.abs(tp - 1750.5))])
    nox_2014 = float(nox[np.argmin(np.abs(tp - 2014.5))])
    assert np.isclose(nox_1750, NOX_1750_REF, rtol=1e-6), (nox_1750, NOX_1750_REF)
    assert np.isclose(nox_2014, NOX_SSP245_2014_REF, rtol=1e-6), (nox_2014, NOX_SSP245_2014_REF)
    print(f"PASS test_fill_from_rcmip_nox_endtoend (FAIR NOx[1750.5]={nox_1750:.4f}, "
          f"[2014.5]={nox_2014:.4f} Mt NO2/yr)")


if __name__ == "__main__":
    fns = [
        test_nox_1750_reconstructed,
        test_nox_ssp245_2014_reference,
        test_clean_historical_uses_reconstruction,
        test_fill_from_rcmip_nox_endtoend,
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
