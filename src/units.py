"""Emission-unit conversion built on fair's own unit tables.

Replaces the hand-written ``unit_conversion`` dict in 01 (which silently left a
pair unconverted whenever it was missing, via ``if factor:``). ``convert_emissions``
resolves the factor from ``fair.structure.units`` and *raises* on an unknown
prefix/compound/time token, so an unhandled unit can never slip through silently.

The NOx relabel (``Mt NO2/yr`` == ``Mt NOx/yr``) is handled natively: fair's
``compound_convert['NO2']['NOx'] == 1.0``.
"""
from fair.structure.units import compound_convert, prefix_convert, time_convert


def emissions_unit_factor(from_unit, to_unit):
    """Multiplicative factor converting an emission rate ``from_unit`` -> ``to_unit``.

    Units are ``"<prefix> <compound>/<time>"`` (e.g. ``"Mt N2O/yr"``). Raises
    ``ValueError`` if a unit can't be parsed or a token is unknown to fair.
    """
    if from_unit == to_unit:
        return 1.0
    try:
        from_prefix, from_rest = from_unit.split()
        to_prefix, to_rest = to_unit.split()
        from_compound, from_time = from_rest.split("/")
        to_compound, to_time = to_rest.split("/")
    except ValueError as exc:
        raise ValueError(
            f"cannot parse emission unit {from_unit!r} or {to_unit!r} "
            "(expected '<prefix> <compound>/<time>')"
        ) from exc
    try:
        return (
            prefix_convert[from_prefix][to_prefix]
            * compound_convert[from_compound][to_compound]
            * time_convert[from_time][to_time]
        )
    except KeyError as exc:
        raise ValueError(
            f"no fair emission-unit conversion {from_unit!r} -> {to_unit!r} "
            f"(unknown token {exc})"
        ) from exc


def convert_emissions(values, from_unit, to_unit):
    """Return ``values`` scaled from ``from_unit`` to ``to_unit`` (see factor rules)."""
    return values * emissions_unit_factor(from_unit, to_unit)
