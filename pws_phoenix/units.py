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

Runtime behavior (the Model boundary; ``control["units"]``):

- ``None`` (default): nothing is checked, pint is never imported.
- ``"check"``: at BUILD, every supplied parameter and input's
  ``attrs["units"]`` (or its ``control["units_patch"]`` entry) must be
  a vocabulary key naming the declared quantity (:func:`same`); a Map's
  source and target declarations must agree. Nothing runs per step.
- ``"convert"``: as "check", but a RESIDENT array (a parameter, an
  in-memory input) supplied in different-but-compatible units is
  converted IN PLACE, once, by the affine map :func:`affine` returns
  (``x_declared = scale * x_supplied + offset``; the offset serves
  degF -> degC). A streamed/lazy input that would need converting
  raises -- conversion is never a per-step edit; pre-process the file.
- the PRMS boundary is separate: ``prms_translate.prms_units``
  translates PRMS's unit words to vocabulary keys and
  ``package_parameters`` judges them against the declarations.

Budget terms are compared by string in ``process.py``: with a closed
canonical vocabulary that IS the exact check. pint is OPTIONAL at
runtime (``environment.yaml`` lists it; a serial env without it runs
every model with ``control["units"]`` = None).

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

import math
from typing import Any

import numpy as np

MODES = (None, "check", "convert")

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
    "km": "kilometer",
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
    "delta_degC": "delta_degC",
    "1/delta_degF": "1 / delta_degF",
    "dday/delta_degF": "1 / delta_degF",
    # -- rates and reciprocals --
    "1/day": "1 / day",
    "inches/day": "inch / day",
    "1/inch": "1 / inch",
    "1/(inch day)": "1 / (inch * day)",
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
    "radians": "radian",
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


def _convert_value(value: float, supplied: str, declared: str) -> float:
    """`value` in `supplied` units expressed in `declared` units (both
    vocabulary keys with pint expressions; a key's expression may carry
    a magnitude, e.g. MCM = 1e6 m^3)."""
    qs, qd = quantity(supplied), quantity(declared)
    as_supplied = registry().Quantity(value * qs.magnitude, qs.units)
    return float(as_supplied.to(qd.units).magnitude / qd.magnitude)


def affine(supplied: str, declared: str) -> tuple[float, float]:
    """(scale, offset) with ``x_declared = scale * x_supplied + offset``.
    Raises ValueError when the two keys have different dimensions or
    either has no pint expression."""
    import pint

    # two-point evaluation; the wide span keeps the roundoff of pint's
    # offset-unit path (through kelvin) out of the scale (5/9 exact)
    span = 1.0e3
    try:
        offset = _convert_value(0.0, supplied, declared)
        scale = (_convert_value(span, supplied, declared) - offset) / span
    except pint.DimensionalityError as err:
        raise ValueError(
            f"units {supplied!r} cannot be converted to {declared!r}: {err}"
        ) from None
    return scale, offset


def same(declared: str, supplied: str) -> bool:
    """Do two vocabulary keys name the same unit? Equal strings always
    do (incl. keys without a pint expression); otherwise both need an
    expression and must convert with scale 1 and offset 0 -- so
    ``fraction``/``none`` agree, ``inches``/``feet`` do not. A key with
    no expression never equals a different key."""
    if declared == supplied:
        return True
    if VOCABULARY[declared] is None or VOCABULARY[supplied] is None:
        return False
    qd, qs = quantity(declared), quantity(supplied)
    if qd.dimensionality != qs.dimensionality:
        return False
    scale, offset = affine(supplied, declared)
    return math.isclose(scale, 1.0, rel_tol=1e-12) and abs(offset) < 1e-12


def reconcile(
    name: str,
    da: Any,
    declared: str | None,
    mode: str | None,
    patch: dict[str, str],
    *,
    convertible: bool,
    context: str,
) -> None:
    """The Model-boundary check for ONE supplied array (an xr.DataArray
    whose ``.values`` is the working buffer). No-op for mode None or a
    unit-less (int/bool) declaration. Otherwise the supplied units
    (``patch[name]`` wins over ``da.attrs["units"]``) must be a
    vocabulary key and :func:`same` as `declared`, or -- mode "convert"
    and `convertible` -- the array is converted in place (the write
    flag is restored afterwards) and its attrs updated. Everything
    else raises ValueError naming `context` and `name`."""
    conv = conversion(name, da.attrs, declared, mode, patch, context=context)
    if conv is None:
        return
    scale, offset = conv
    supplied = patch.get(name, da.attrs.get("units"))
    where = f"{context}: '{name}' (declared {declared!r})"
    if not convertible:
        raise ValueError(
            f"{where}: supplied in {supplied!r}, which would need "
            "converting, but the array is streamed/lazy -- conversion is "
            "never a per-step edit; pre-process the file to "
            f"{declared!r} (scale {scale!r}, offset {offset!r})."
        )
    apply_affine(da.values, scale, offset, where=where)
    da.attrs["units"] = declared


def conversion(
    name: str,
    attrs: Any,
    declared: str | None,
    mode: str | None,
    patch: dict[str, str],
    *,
    context: str,
) -> tuple[float, float] | None:
    """The JUDGING half of :func:`reconcile`, for a supplied array known
    only by its ``attrs`` (so also for data that is COPIED into a model
    buffer rather than being one: supplied initial state, restart
    files). Returns None when nothing is to be done (mode None, a
    unit-less declaration, or supplied units that are :func:`same` as
    `declared`); returns ``(scale, offset)`` (see :func:`affine`) under
    mode "convert" for compatible units; raises ValueError otherwise
    (no units, a string outside the vocabulary, a mismatch under
    "check", incompatible dimensions)."""
    if mode is None or declared is None:
        return None
    supplied = patch.get(name, attrs.get("units"))
    where = f"{context}: '{name}' (declared {declared!r})"
    if supplied is None:
        raise ValueError(
            f"{where} has no units -- set attrs['units'] on the supplied "
            f"array or control['units_patch'][{name!r}]."
        )
    if supplied not in VOCABULARY:
        raise ValueError(
            f"{where}: supplied units {supplied!r} are not in the "
            "pws_phoenix vocabulary (pws_phoenix/units.py) -- spell them "
            "as a vocabulary key via control['units_patch'] or add the key."
        )
    if same(declared, supplied):
        return None
    if mode == "check":
        raise ValueError(
            f"{where}: supplied units are {supplied!r}; control['units'] "
            "= 'convert' converts resident arrays in place."
        )
    return affine(supplied, declared)  # raises if incompatible


def apply_affine(
    values: np.ndarray, scale: float, offset: float, *, where: str
) -> None:
    """``values = scale * values + offset`` IN PLACE (no temporary; the
    write flag is restored afterwards)."""
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(
            f"{where}: cannot convert a {values.dtype} array in place."
        )
    writeable = values.flags.writeable
    values.flags.writeable = True
    try:
        values *= scale
        if offset != 0.0:
            values += offset
    finally:
        values.flags.writeable = writeable
