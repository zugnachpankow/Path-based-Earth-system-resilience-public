"""Natural-background emission floor per species, in the pipeline CSV units.

Emission reductions (the Current-Policies branch-offs in 02, the ambition
calculator, and the 01 extrapolation tail) must not drive a species below its
natural background, i.e. fair's ``baseline_emissions``. fair stores those in its
own emission units (e.g. ``Mt N2O/yr``), but the pipeline emissions CSVs carry
the reporting units of their source (e.g. ``kt N2O/yr``). The floor is therefore
converted to each variable's CSV unit before it is compared against CSV values
(N2O: 0.0860 Mt/yr -> 86.02 kt/yr; NOx: Mt NO2 == Mt NOx, factor 1).
"""
import pandas as pd

from fair.structure.units import (
    compound_convert,
    desired_emissions_units,
    prefix_convert,
    time_convert,
)
from fair_config import SPECIES_CONFIGS_NGFS


def _emissions_unit_factor(src_unit, tgt_unit):
    """Factor converting an emission rate from ``src_unit`` to ``tgt_unit``.

    Units look like ``"Mt N2O/yr"``. Raises ``KeyError`` on an unknown prefix,
    compound or time token (fail loud rather than silently skip).
    """
    src_prefix, src_rest = src_unit.split()
    tgt_prefix, tgt_rest = tgt_unit.split()
    src_compound, src_time = src_rest.split("/")
    tgt_compound, tgt_time = tgt_rest.split("/")
    return (
        prefix_convert[src_prefix][tgt_prefix]
        * compound_convert[src_compound][tgt_compound]
        * time_convert[src_time][tgt_time]
    )


def build_floor_map(df_emissions, species_file=SPECIES_CONFIGS_NGFS):
    """``{variable: baseline_emissions in that variable's CSV unit}``.

    ``df_emissions`` must have ``variable`` and ``unit`` columns. Variables with
    no fair ``baseline_emissions`` (or absent from the species file) floor at 0.
    """
    sp = pd.read_csv(species_file)
    base = dict(zip(sp["name"], sp["baseline_emissions"].fillna(0.0)))
    units = (
        df_emissions.drop_duplicates("variable")
        .set_index("variable")["unit"]
        .to_dict()
    )

    floors = {}
    for variable, csv_unit in units.items():
        value = float(base.get(variable, 0.0))
        if value != 0.0 and variable in desired_emissions_units:
            value *= _emissions_unit_factor(desired_emissions_units[variable], csv_unit)
        floors[variable] = value
    return floors
