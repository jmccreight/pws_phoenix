"""Harness for the stage-3 units tests (tests/test_kernel_units.py):
run a kernel's UNDECORATED body (the numba Dispatcher's ``py_func``)
on pint Quantity arrays built from the declarations and compare it,
magnitude for magnitude, to the compiled kernel.

Kernel argument names == declared field names, so the declarations
drive everything. A field without units (int/bool) is passed as a
plain array. ``patches`` swaps module attributes for the duration of
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
from collections.abc import Iterator
from typing import Any

import numpy as np
import pint

from pws_phoenix import units as units_mod
from pws_phoenix.process import _resolved_fields

# parameters declared as rates (1/day) that the daily kernels apply
# once per step; the Quantity run multiplies them by 1 day (JLM's
# convention, Nov 5, 2026)
PER_DAY = frozenset({"gwflow_coef", "gwsink_coef"})

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


def kernel_arg_names(cls: type) -> list[str]:
    """The kernel's positional parameters == declared field names."""
    return list(inspect.signature(cls._calculate.py_func).parameters)


def kernel_units(
    cls: type, declared: dict[str, str] | None = None
) -> dict[str, str | None]:
    """Kernel argument -> vocabulary key (None for a unit-less int/bool
    field): the declaration, or the `declared` override (for the tests
    that prove a wrong declaration is caught)."""
    fields = _resolved_fields(cls)
    return {
        name: (declared or {}).get(name, fields[name].units)
        for name in kernel_arg_names(cls)
    }


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


def run_compiled(cls: type, data: dict[str, np.ndarray]) -> dict:
    """The compiled kernel on copies of `data`; returns the arrays."""
    arrays = {name: data[name].copy() for name in kernel_arg_names(cls)}
    cls._calculate(*arrays.values())
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


def _traced_codes(cls: type, patches: Patches | None) -> list[types.CodeType]:
    """The kernel body plus every patched-in plain function (helpers)."""
    codes = [cls._calculate.py_func.__code__]
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
) -> dict:
    """The kernel body (`py_func`) on Quantity arrays in the units
    `kernel_units` gives (unit-less fields plain); `per_day` parameters
    are multiplied by 1 day first; `patches` applied during the call.
    If `fired` is given, the executed lines of the kernel and patched
    helpers are added to it. Returns the arrays."""
    one_day = units_mod.quantity("days")
    arrays: dict[str, Any] = {}
    for name, key in kernel_units(cls, declared).items():
        if key is None:
            arrays[name] = data[name].copy()
            continue
        qarr = quantify(data[name].copy(), key)
        if name in per_day:
            qarr = qarr * one_day
        arrays[name] = qarr
    with patched(patches), traced_lines(_traced_codes(cls, patches)) as got:
        cls._calculate.py_func(*arrays.values())
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
) -> dict:
    """Quantity run == compiled run, element for element and bit for
    bit, each array read in the units it was BUILT in (inputs come
    back unchanged, outputs are the check); and (`require_all_lines`)
    every unmarked line of the kernel and patched helpers ran. Returns
    the compiled arrays (for branch asserts)."""
    compiled = run_compiled(cls, data)
    fired: dict[types.CodeType, set[int]] = {}
    quantified = run_quantified(
        cls, data, declared=declared, patches=patches, fired=fired
    )
    for name, key in kernel_units(cls, declared).items():
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
