"""Harness for the stage-3 units tests (tests/test_kernel_units.py):
run a kernel's UNDECORATED body (the numba Dispatcher's ``py_func``)
on pint Quantity arrays built from the declarations and compare it,
magnitude for magnitude, to the compiled kernel.

Kernel argument names == declared field names, so the declarations
drive everything. A field without units, or of integer dtype, is
passed as a plain array (an integer field may declare units, e.g.
the channel's ``tsi`` in hours, but the kernel uses it as an integer:
``(ihr + 1) % tsi[jseg]``). A kernel argument that is not a field
takes its units from ``NON_FIELD_UNITS``; one that receives a derived
field of another name is mapped in ``ARG_FIELDS``.

``func=`` runs a plain-numpy function instead of the kernel (an
init-time derivation such as ``muskingum_mann_coefficients``): its
argument names must also be ``cls``'s field names, and the plain-float
run of the same function stands in for the compiled run. ``patches``
swaps module attributes for the duration of
the Quantity run: implicit-unit module floats (``_NEARZERO``,
``_ZERO_INCHES``) become Quantities, and helper njit Dispatchers
become their ``py_func`` (a Dispatcher cannot take a Quantity).

COVERAGE: pint only checks the operations that execute, so
:func:`assert_kernel_dimensionally_consistent` also requires every
line of the kernel body -- and of every patched helper -- to run in
the Quantity run (``sys.settrace``; executable lines from the code
object's line table). A line that cannot run on any input carries the
marker ``# units-test: unreachable (<reason>)`` in the kernel source,
for review by eye. Statement coverage, not branch coverage: a skipped
``if`` body is required, the paths between statements are not.
"""

import contextlib
import dis
import inspect
import sys
import types
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pint

from pws_phoenix import units as units_mod
from pws_phoenix.process import _resolved_fields

# parameters declared as rates (1/day) that the daily kernels apply
# once per step; the Quantity run multiplies them by 1 day (JLM's
# convention, Nov 5, 2026)
PER_DAY = frozenset(
    {
        "gwflow_coef",
        "gwsink_coef",
        # soilzone: the 1/day routing rates, the per-day cap
        # soil2gw_max (inches/day) and the 1/(inch day) coefficients
        "fastcoef_lin",
        "slowcoef_lin",
        "ssr2gw_rate",
        "soil2gw_max",
        "fastcoef_sq",
        "slowcoef_sq",
    }
)

# kernel arguments that receive a DERIVED field of another name:
# argument name -> field name. Upstream's underscore names mark a
# derived copy beside the raw parameter (``_sat_threshold`` = the
# ``sat_threshold`` parameter zeroed for lake/inactive HRUs), so the
# bare argument name is often ALSO a field -- an explicit table, never
# a fallback rule.
ARG_FIELDS = {
    "sat_threshold": "_sat_threshold",
    "pref_flow_den": "_pref_flow_den",
    "pref_flow_flag": "_pref_flow_flag",
    "soil2gw_flag": "_soil2gw_flag",
}

# kernel arguments that are not fields: argument name -> vocabulary key
NON_FIELD_UNITS = {"s_per_time": "seconds"}  # the channel's dt

Patches = dict[tuple[Any, str], Any]

UNREACHABLE_MARKER = "# units-test: unreachable"


def quantify(values: np.ndarray, key: str) -> pint.Quantity:
    """`values` (magnitudes in vocabulary key `key`) as a Quantity array."""
    unit = units_mod.quantity(key)
    return units_mod.registry().Quantity(
        np.asarray(values, dtype=np.float64) * unit.magnitude, unit.units
    )


def magnitudes(qarr: pint.Quantity, key: str) -> np.ndarray:
    """The Quantity array's magnitudes expressed in vocabulary key `key`."""
    unit = units_mod.quantity(key)
    return qarr.to(unit.units).magnitude / unit.magnitude


def inches(value: float) -> pint.Quantity:
    """A scalar Quantity in inches (for `patches`)."""
    return value * units_mod.quantity("inches")


def _body(cls: type, func: Callable | None) -> Callable:
    """The Python body run on Quantities: `func`, or the kernel's
    py_func."""
    return func if func is not None else cls._calculate.py_func


def kernel_arg_names(cls: type, func: Callable | None = None) -> list[str]:
    """The kernel's (or `func`'s) positional parameters == declared
    field names."""
    return list(inspect.signature(_body(cls, func)).parameters)


def kernel_units(
    cls: type,
    declared: dict[str, str] | None = None,
    func: Callable | None = None,
) -> dict[str, str | None]:
    """Kernel argument -> vocabulary key (None = passed plain: no units,
    or an integer field): the declaration of the argument's field (its
    own name, or `ARG_FIELDS`; `NON_FIELD_UNITS` for a non-field
    argument), or the `declared` override (for the tests that
    prove a wrong declaration is caught)."""
    fields = _resolved_fields(cls)
    out: dict[str, str | None] = {}
    for name in kernel_arg_names(cls, func):
        field = ARG_FIELDS.get(name, name)
        if field in fields:
            meta = fields[field]
            key = None if np.issubdtype(meta.dtype, np.integer) else meta.units
        else:
            key = NON_FIELD_UNITS[name]
        out[name] = (declared or {}).get(name, key)
    return out


@contextlib.contextmanager
def patched(patches: Patches | None) -> Iterator[None]:
    """setattr every ``(module, name): value`` in `patches`; restore on
    exit."""
    saved = {target: getattr(*target) for target in patches or {}}
    try:
        for (module, name), value in (patches or {}).items():
            setattr(module, name, value)
        yield
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)


def run_compiled(
    cls: type, data: dict[str, np.ndarray], func: Callable | None = None
) -> dict:
    """The compiled kernel (or plain `func`) on copies of `data`;
    returns the arrays."""
    arrays = {name: data[name].copy() for name in kernel_arg_names(cls, func)}
    (func if func is not None else cls._calculate)(*arrays.values())
    return arrays


def executable_lines(code: types.CodeType) -> set[int]:
    """Lines of `code` that carry bytecode, minus the ``def`` line and
    the lines marked unreachable in the source."""
    lines = {ln for _, ln in dis.findlinestarts(code) if ln is not None}
    lines.discard(code.co_firstlineno)
    source, first = inspect.getsourcelines(code)
    marked = {
        first + offset
        for offset, text in enumerate(source)
        if UNREACHABLE_MARKER in text
    }
    return lines - marked


@contextlib.contextmanager
def traced_lines(
    codes: list[types.CodeType],
) -> Iterator[dict[types.CodeType, set[int]]]:
    """Record the lines executed in each of `codes` (other frames are
    not traced); the previous trace function is restored on exit."""
    fired: dict[types.CodeType, set[int]] = {cc: set() for cc in codes}

    def tracer(frame, event, arg):
        if frame.f_code in fired:
            if event == "line":
                fired[frame.f_code].add(frame.f_lineno)
            return tracer
        return None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        yield fired
    finally:
        sys.settrace(previous)


def _traced_codes(
    cls: type, patches: Patches | None, func: Callable | None
) -> list[types.CodeType]:
    """The kernel body (or `func`) plus every patched-in plain function
    (helpers)."""
    codes = [_body(cls, func).__code__]
    for value in (patches or {}).values():
        if isinstance(value, types.FunctionType):
            codes.append(value.__code__)
    return codes


def run_quantified(
    cls: type,
    data: dict[str, np.ndarray],
    *,
    declared: dict[str, str] | None = None,
    per_day: frozenset[str] = PER_DAY,
    patches: Patches | None = None,
    fired: dict[types.CodeType, set[int]] | None = None,
    func: Callable | None = None,
) -> dict:
    """The kernel body (`py_func`) on Quantity arrays in the units
    `kernel_units` gives (unit-less fields plain); `per_day` parameters
    are multiplied by 1 day first; `patches` applied during the call.
    If `fired` is given, the executed lines of the kernel and patched
    helpers are added to it. Returns the arrays."""
    one_day = units_mod.quantity("days")
    arrays: dict[str, Any] = {}
    for name, key in kernel_units(cls, declared, func).items():
        if key is None:
            arrays[name] = data[name].copy()
            continue
        qarr = quantify(data[name].copy(), key)
        if name in per_day:
            qarr = qarr * one_day
        arrays[name] = qarr
    codes = _traced_codes(cls, patches, func)
    # a stripped unit (numpy downcasting a Quantity) is a line pint no
    # longer checks: make it an error
    with patched(patches), traced_lines(codes) as got:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pint.errors.UnitStrippedWarning)
            _body(cls, func)(*arrays.values())
    if fired is not None:
        for code, lines in got.items():
            fired.setdefault(code, set()).update(lines)
    return arrays


def assert_all_lines_ran(fired: dict[types.CodeType, set[int]]) -> None:
    """Every executable, unmarked line of each traced code object ran."""
    missing = {}
    for code, lines in fired.items():
        never = sorted(executable_lines(code) - lines)
        if never:
            source, first = inspect.getsourcelines(code)
            missing[code.co_name] = [
                f"{ln}: {source[ln - first].strip()}" for ln in never
            ]
    assert not missing, (
        "lines never executed under pint (extend the fixture, or mark a "
        f"line that cannot run '{UNREACHABLE_MARKER} (<reason>)'): "
        f"{missing}"
    )


def assert_kernel_dimensionally_consistent(
    cls: type,
    data: dict[str, np.ndarray],
    *,
    declared: dict[str, str] | None = None,
    patches: Patches | None = None,
    require_all_lines: bool = True,
    func: Callable | None = None,
) -> dict:
    """Quantity run == compiled run, element for element and bit for
    bit, each array read in the units it was BUILT in (inputs come
    back unchanged, outputs are the check); and (`require_all_lines`)
    every unmarked line of the kernel and patched helpers ran. Returns
    the compiled arrays (for branch asserts)."""
    compiled = run_compiled(cls, data, func)
    fired: dict[types.CodeType, set[int]] = {}
    quantified = run_quantified(
        cls, data, declared=declared, patches=patches, fired=fired, func=func
    )
    for name, key in kernel_units(cls, declared, func).items():
        if name in PER_DAY:
            continue  # carries the 1 day factor by construction
        got = (
            quantified[name]
            if key is None
            else magnitudes(quantified[name], key)
        )
        np.testing.assert_array_equal(got, compiled[name], err_msg=name)
    if require_all_lines:
        assert_all_lines_ran(fired)
    return compiled
