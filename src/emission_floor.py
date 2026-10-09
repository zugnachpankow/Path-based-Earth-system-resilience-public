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

from fair.structure.units import desired_emissions_units
from fair_config import SPECIES_CONFIGS_NGFS
from units import emissions_unit_factor


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
            value *= emissions_unit_factor(desired_emissions_units[variable], csv_unit)
        floors[variable] = value
    return floors
