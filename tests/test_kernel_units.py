"""Test a kernel's dimensional consistency.

The compiled kernels see plain float arrays; their units live only in
the declarations. This test runs the UNDECORATED kernel body (the
numba Dispatcher's ``py_func``) on pint Quantity arrays built from the
declarations, so every ``+``, ``-``, comparison and element assignment
inside the kernel is checked by pint:

- an incompatible operation (``inches + acre-inches``) raises
  ``DimensionalityError``;
- a COMPATIBLE but wrong-scale result (``feet`` assigned into an
  ``inches`` array) is silently converted by pint on assignment -- so
  the magnitudes are also compared, bit for bit, to the compiled run.

Convention for ``1/day`` coefficients (JLM, Nov 5, 2026): declarations
keep the physical rate; the daily kernels apply it once per step, so
the test multiplies those parameters by ``1 day`` before the call
(``PER_DAY``). Pilot kernel: the groundwater family (no implicit-unit
literals). Synthetic data, CI-able, no numba compile of anything new
(the compiled run uses the kernel as the model does).
"""

import inspect

import numpy as np
import pytest

pint = pytest.importorskip("pint")

from pws_phoenix import units as units_mod  # noqa: E402
from pws_phoenix.hydrology.prms_groundwater import (  # noqa: E402
    PRMSGroundwater,
    PRMSGroundwaterNoDprst,
)
from pws_phoenix.process import _resolved_fields  # noqa: E402

# parameters declared as rates that the kernels apply per daily step
PER_DAY = frozenset({"gwflow_coef", "gwsink_coef"})


# ----------------------------------------------------------------------
# harness (test-side; promote to a shared module with the second kernel)
# ----------------------------------------------------------------------


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


def kernel_arg_names(cls: type) -> list[str]:
    """The kernel's positional parameters == declared field names."""
    return list(inspect.signature(cls._calculate.py_func).parameters)


def run_compiled(cls: type, data: dict[str, np.ndarray]) -> dict:
    """The compiled kernel on copies of `data`; returns the arrays."""
    arrays = {name: data[name].copy() for name in kernel_arg_names(cls)}
    cls._calculate(*arrays.values())
    return arrays


def kernel_units(cls: type, declared: dict[str, str] | None = None) -> dict:
    """Kernel argument -> vocabulary key: the declaration, or the
    `declared` override (for the tests that prove a wrong declaration
    is caught)."""
    fields = _resolved_fields(cls)
    return {
        name: (declared or {}).get(name, fields[name].units)
        for name in kernel_arg_names(cls)
    }


def run_quantified(
    cls: type,
    data: dict[str, np.ndarray],
    *,
    declared: dict[str, str] | None = None,
    per_day: frozenset[str] = PER_DAY,
) -> dict:
    """The kernel body (`py_func`) on Quantity arrays in the units
    `kernel_units` gives; `per_day` parameters are multiplied by 1 day
    first. Returns the Quantity arrays."""
    one_day = units_mod.quantity("days")
    arrays = {}
    for name, key in kernel_units(cls, declared).items():
        qarr = quantify(data[name].copy(), key)
        if name in per_day:
            qarr = qarr * one_day
        arrays[name] = qarr
    cls._calculate.py_func(*arrays.values())
    return arrays


def assert_kernel_dimensionally_consistent(
    cls: type,
    data: dict[str, np.ndarray],
    *,
    declared: dict[str, str] | None = None,
) -> None:
    """Quantity run == compiled run, magnitude for magnitude, each array
    read in the units it was BUILT in (inputs come back unchanged,
    outputs are the check)."""
    compiled = run_compiled(cls, data)
    quantified = run_quantified(cls, data, declared=declared)
    for name, key in kernel_units(cls, declared).items():
        if name in PER_DAY:
            continue  # carries the 1 day factor by construction
        np.testing.assert_array_equal(
            magnitudes(quantified[name], key), compiled[name], err_msg=name
        )


# ----------------------------------------------------------------------
# groundwater family
# ----------------------------------------------------------------------


@pytest.fixture
def gw_data() -> dict[str, np.ndarray]:
    """Three hrus, one step: the last hru's gwsink_coef trips the
    `sink > stor` clamp; hru 1 starts empty."""
    hru_area = np.array([10.0, 250.0, 1000.0])
    return {
        "hru_area": hru_area,
        "hru_in_to_cf": hru_area * 43560.0 / 12.0,
        "gwflow_coef": np.array([0.01, 0.05, 0.2]),
        "gwsink_coef": np.array([0.0, 0.01, 5.0]),
        "soil_to_gw": np.array([0.1, 0.0, 0.3]),
        "ssr_to_gw": np.array([0.05, 0.2, 0.0]),
        "dprst_seep_hru": np.array([0.0, 0.02, 0.1]),
        "gwres_stor": np.array([1.0, 0.0, 2.5]),
        "gwres_stor_old": np.array([1.0, 0.0, 2.5]),
        "gwres_flow": np.zeros(3),
        "gwres_sink": np.zeros(3),
        "gwres_stor_change": np.zeros(3),
        "gwres_flow_vol": np.zeros(3),
    }


@pytest.mark.parametrize("cls", [PRMSGroundwaterNoDprst, PRMSGroundwater])
def test_groundwater_kernel_units(cls, gw_data):
    assert_kernel_dimensionally_consistent(cls, gw_data)
    # the clamp branch was exercised
    out = run_compiled(cls, gw_data)
    assert out["gwres_stor"][2] == 0.0 and out["gwres_sink"][2] > 0.0


def test_per_day_convention_is_load_bearing(gw_data):
    """Without the 1 day factor, flow = stor * gwflow_coef is a volume
    per day and `stor - flow` is dimensionally inconsistent."""
    with pytest.raises(pint.DimensionalityError):
        run_quantified(PRMSGroundwater, gw_data, per_day=frozenset())


def test_incompatible_declaration_is_caught(gw_data):
    """gwres_flow_vol declared in inches: the kernel assigns cubic feet
    into it -> DimensionalityError."""
    with pytest.raises(pint.DimensionalityError):
        run_quantified(
            PRMSGroundwater, gw_data, declared={"gwres_flow_vol": "inches"}
        )


def test_compatible_wrong_declaration_is_caught(gw_data):
    """gwres_flow_vol declared in acre-inches: pint converts the kernel's
    cubic feet silently on assignment -- the magnitude comparison with
    the compiled run is what catches it."""
    with pytest.raises(AssertionError, match="gwres_flow_vol"):
        assert_kernel_dimensionally_consistent(
            PRMSGroundwater,
            gw_data,
            declared={"gwres_flow_vol": "acre-inches"},
        )
