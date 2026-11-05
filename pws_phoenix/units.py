"""Physical units: the closed vocabulary behind ``DataArrayMeta.units``.

A declaration's ``units`` is a free string in PRINCIPLE but drawn from the
closed ``VOCABULARY`` below in PRACTICE: the keys are the pywatershed-
natural spellings that appear in declarations (``"inches"``, ``"cfs"``,
``"acre-inches"``); the values are the pint expressions that give each
its dimension and conversion factor. Keeping pint behind a table does
three things: typos (``"inch"``) fail the vocabulary test instead of
silently parsing; spellings pint would misread (``acre-inches`` is a
subtraction to pint) never reach its parser; and quantities pint cannot
express (an exponent-dependent power-law coefficient) still get a
checkable NAME -- their value is ``None`` and they match by string only.

Stage 1 (declarations + this vocabulary + ``tests/test_units.py``) has
no runtime behavior. Stage 2 adds the build-time checks (supplied
``attrs["units"]`` vs the declaration; Map source vs target; budget
terms) and the optional in-place conversion of resident arrays, all
through :func:`registry` -- pint is imported lazily and only then.

pint is OPTIONAL at runtime (``environment.yaml`` lists it; a serial
env without it runs every model with the units checks off).

Conventions encoded in the table:

- ``fraction`` = a proportion (``"-"``, ``"decimal fraction"``);
  ``none`` = dimensionless but not a proportion (float-stored flags such
  as ``pptmix``/``transp_on``). Both are pint ``dimensionless``.
- Temperature DIFFERENCES use pint's delta units (``1/delta_degF``,
  ``cal/(cm^2 delta_degC)``); absolute temperatures use ``degF``/``degC``.
- Rate coefficients keep their physical ``1/day`` (``gwflow_coef`` etc.)
  even though the daily kernels apply them as per-step fractions: the
  kernel-level units check (stage 3) carries the ``dt = 1 day``
  convention, not the declaration.
- The ``io ...`` keys are PLACEHOLDERS resolved at FlowGraph composition
  (``flow_graph.make_flow_graph``) via :data:`IO_PLACEHOLDERS`; they
  must never survive on a composed class.
"""

from __future__ import annotations

from typing import Any

# declaration string -> pint expression (None = not expressible in pint;
# such a unit matches by exact string only)
VOCABULARY: dict[str, str | None] = {
    # -- length / area / volume (imperial, the PRMS working system) --
    "inches": "inch",
    "feet": "foot",
    "acres": "acre",
    "acre-inches": "acre * inch",
    "cubic feet": "foot ** 3",
    "Mcf": "1e6 * foot ** 3",
    "cubic feet/inch": "foot ** 3 / inch",
    # -- SI (channel geometry, STARFIT) --
    "m": "meter",
    "m^2": "meter ** 2",
    "m/s": "meter / second",
    "cubic meters": "meter ** 3",
    "MCM": "1e6 * meter ** 3",
    "MCM/cms": "1e6 * meter ** 3 / (meter ** 3 / second)",
    # -- flow --
    "cfs": "foot ** 3 / second",
    "cms": "meter ** 3 / second",
    # -- energy --
    "cal/cm^2": "calorie / centimeter ** 2",
    "cal/(cm^2 delta_degC)": "calorie / (centimeter ** 2 * delta_degC)",
    # -- temperature (absolute vs difference) --
    "degF": "degF",
    "degC": "degC",
    "1/delta_degF": "1 / delta_degF",
    "dday/delta_degF": "1 / delta_degF",
    # -- rates and reciprocals --
    "1/day": "1 / day",
    "inches/day": "inch / day",
    "1/inch": "1 / inch",
    "seconds/m^(1/3)": "second / meter ** (1 / 3)",
    # -- time --
    "days": "day",
    "hours": "hour",
    "seconds": "second",
    # -- dimensionless --
    "fraction": "dimensionless",
    "none": "dimensionless",
    "percent": "percent",
    "degrees": "degree",
    # -- named but not expressible in pint --
    "dday": None,  # PRMS degree-day index (dimensionless by use)
    "day of year": None,  # a calendar ordinal stored as float
    "m per cms^exp": None,  # power-law coefficient, exponent-dependent
    # -- FlowGraph placeholders, resolved at composition --
    "io flow units": None,
    "io flow units x s": None,
    "io storage units": None,
}

# placeholder -> (units in a cfs graph, units in a cms graph)
IO_PLACEHOLDERS: dict[str, tuple[str, str]] = {
    "io flow units": ("cfs", "cms"),
    "io flow units x s": ("cubic feet", "cubic meters"),
    "io storage units": ("Mcf", "MCM"),
}

_registry: Any = None


def registry() -> Any:
    """The project pint ``UnitRegistry`` (built once, pint imported here
    and only here). ``autoconvert_offset_to_baseunit`` lets absolute
    temperatures convert (degF -> degC) without an explicit delta."""
    global _registry
    if _registry is None:
        import pint

        _registry = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
    return _registry


def quantity(units: str) -> Any:
    """The pint Quantity (magnitude 1 in the expression's own scale) for
    a vocabulary key. Raises KeyError for a string outside the
    vocabulary and ValueError for a key with no pint expression."""
    expr = VOCABULARY[units]
    if expr is None:
        raise ValueError(f"units {units!r} have no pint expression")
    return registry()(expr)
