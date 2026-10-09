"""Shared FAIR calibration v1.4.0 configuration: file paths, parameter loading
and the natural (solar + volcanic) forcing used by every scenario family.

Single source of truth so the Monte-Carlo runner and the calculator build FAIR
identically. The natural forcing and the per-config solar/volcanic scaling follow
the fair-calibrate v1.4.0 reference projection script
``.../constraining/05_constrained-ssp-projections.py`` verbatim.
"""
import os
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd

_DATA_RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "raw")

# calibration v1.4.0 input files (built by 00b_build_fair_v140_inputs.py)
FAIR_PARAMS = os.path.join(_DATA_RAW, "calibrated_constrained_parameters_calibration1.4.0.csv")
SPECIES_CONFIGS = os.path.join(_DATA_RAW, "species_configs_properties_calibration1.4.0.csv")
SPECIES_CONFIGS_NGFS = os.path.join(_DATA_RAW, "species_configs_properties_NGFS_calibration1.4.0.csv")

# natural-forcing source files (fair-calibrate v1.4.0 data/forcing)
SOLAR_ERF = os.path.join(_DATA_RAW, "solar_erf_timebounds.csv")
VOLCANIC_ERF = os.path.join(_DATA_RAW, "volcanic_ERF_1750-2101_timebounds.csv")

# extra, non-fair columns carried in the parameter file (consumed here, not by
# FAIR.override_defaults, which only understands fair config names).
SOLAR_SCALE_COLS = ["fscale_solar_amplitude", "fscale_solar_trend"]

# the solar trend ramps linearly from 0 to 1 over 1750..2020, then stays at 1.
_TREND_START, _TREND_END = 1750, 2020


def load_fair_params(path=FAIR_PARAMS):
    """Read a calibration parameter file, split off the solar-scale columns.

    Returns ``(df_fair, solar)`` where ``df_fair`` carries only the fair config
    columns (safe to hand to ``FAIR.override_defaults``) and ``solar`` is a
    DataFrame with ``fscale_solar_amplitude``/``fscale_solar_trend`` indexed by
    config, or ``None`` for a legacy file (e.g. 1.4.1) that lacks them.
    """
    df = pd.read_csv(path, index_col=0)
    present = [c for c in SOLAR_SCALE_COLS if c in df.columns]
    solar = df[present].copy() if len(present) == len(SOLAR_SCALE_COLS) else None
    df_fair = df.drop(columns=present)
    return df_fair, solar


@contextmanager
def fair_override_file(df_fair):
    """Write ``df_fair`` to a temporary CSV for ``FAIR.override_defaults``.

    ``override_defaults`` takes a file path and raises on any column it does not
    recognise, so the (already solar-stripped) frame is round-tripped through a
    short-lived temp file that is removed on exit.
    """
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    tmp.close()
    try:
        df_fair.to_csv(tmp.name)
        yield tmp.name
    finally:
        os.unlink(tmp.name)


def natural_forcing(timebounds, solar_file=SOLAR_ERF, volcanic_file=VOLCANIC_ERF):
    """Solar and volcanic ERF time series aligned to ``timebounds`` (years).

    Mirrors the reference 05 script: volcanic forcing is taken from the 1750..2101
    file and is zero outside that range; solar forcing is taken from the
    1750..2300 file. Returns ``(volcanic, solar)`` 1-D arrays, same length as
    ``timebounds``. Raises if a requested solar year falls outside the file.
    """
    years = np.asarray(timebounds).astype(int)
    solar = pd.read_csv(solar_file, index_col="year")["erf"]
    volc = pd.read_csv(volcanic_file, index_col="timebounds")["erf"]

    solar_out = solar.reindex(years).values
    if np.isnan(solar_out).any():
        missing = years[np.isnan(solar_out)]
        raise ValueError(f"solar ERF undefined for years {missing[:5]}... "
                         f"(file covers {int(solar.index.min())}..{int(solar.index.max())})")

    volc_out = np.zeros(len(years))
    in_range = (years >= int(volc.index.min())) & (years <= int(volc.index.max()))
    volc_out[in_range] = volc.reindex(years[in_range]).values
    return volc_out, solar_out


def solar_trend_shape(timebounds):
    """Linear 0->1 ramp over 1750..2020, flat at 1 afterwards (reference 05)."""
    years = np.asarray(timebounds).astype(float)
    shape = np.ones(len(years))
    ramp = (years >= _TREND_START) & (years <= _TREND_END)
    shape[ramp] = (years[ramp] - _TREND_START) / (_TREND_END - _TREND_START)
    shape[years < _TREND_START] = 0.0
    return shape
