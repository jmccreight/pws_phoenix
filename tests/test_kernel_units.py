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
literals); then PRMSCanopy (implicit-unit module floats + an njit
helper, patched via the harness); PRMSChannel (a non-field kernel
argument, dt = s_per_time in seconds; ts in hours used as a sample
count). Harness: tests/kernel_units.py.
Synthetic data, CI-able.
"""

import numba
import numpy as np
import pytest

pint = pytest.importorskip("pint")

from kernel_units import (  # noqa: E402
    assert_kernel_dimensionally_consistent,
    inches,
    run_compiled,
    run_quantified,
)
from pws_phoenix.hydrology import prms_canopy, prms_channel  # noqa: E402
from pws_phoenix.hydrology.prms_canopy import PRMSCanopy  # noqa: E402
from pws_phoenix.hydrology.prms_channel import PRMSChannel  # noqa: E402
from pws_phoenix.units import quantity  # noqa: E402
from pws_phoenix.hydrology.prms_groundwater import (  # noqa: E402
    PRMSGroundwater,
    PRMSGroundwaterNoDprst,
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


# ----------------------------------------------------------------------
# canopy
# ----------------------------------------------------------------------

CANOPY_PATCHES = {
    (prms_canopy, "_NEARZERO"): inches(prms_canopy._NEARZERO),
    (prms_canopy, "_DNEARZERO"): inches(prms_canopy._DNEARZERO),
    (prms_canopy, "_ZERO_INCHES"): inches(prms_canopy._ZERO_INCHES),
    (prms_canopy, "intercept"): prms_canopy.intercept.py_func,
}


@pytest.fixture
def canopy_data() -> dict[str, np.ndarray]:
    """Fourteen hrus, one step, together running every kernel line
    (the harness enforces it):

    0 bare soil holding water (extra_water)
    1 summer -> winter changeover, then rain-form evaporation
    2 winter -> summer changeover (shrinking store), trees intercept rain
    3 grasses intercept rain (no snowpack, no snow)
    4 grasses with a snowpack: no interception
    5 trees intercept trace snow -> it becomes rain, pptmix zeroed
    6 snow-form sublimation (snowfall below _NEARZERO; store over snow_intcp)
    7 the potet cap (potet_sublim * cov > 1: not physical, reaches it)
    8 summer -> winter onto zero winter cover: store dropped
    9 summer -> winter with covden_win > covden_sum: store rescaled
    10 winter -> summer onto zero summer cover: store dropped
    11 snow-form sublimation of the whole store
    12 rain-form evaporation of part of the store
    13 the potet cap on zero cover (potet < 0: not physical, reaches it)
    """
    nhru = 14

    def full(value):
        return np.full(nhru, value)

    def at(default, **by_hru):
        arr = full(default)
        for key, value in by_hru.items():
            arr[int(key[1:])] = value
        return arr

    hru_rain = at(0.0, h0=0.2, h2=0.3, h3=0.2, h4=0.1, h5=0.05)
    hru_snow = at(0.0, h5=1.0e-7, h6=5.0e-7, h7=5.0e-7, h11=5.0e-7)
    zeros = full(0.0)
    return {
        # in/out state + outputs
        "net_ppt": zeros.copy(),
        "net_rain": zeros.copy(),
        "net_snow": zeros.copy(),
        "intcp_changeover": zeros.copy(),
        "intcp_evap": zeros.copy(),
        "intcp_form": np.zeros(nhru, dtype=np.int64),
        "intcp_stor": at(
            0.0,
            h0=0.1,
            h1=0.05,
            h2=0.04,
            h6=0.2,
            h7=0.2,
            h8=0.04,
            h9=0.04,
            h10=0.04,
            h11=0.01,
            h12=0.2,
            h13=0.05,
        ),
        "intcp_transp_on": at(1, h2=0, h10=0).astype(np.int64),
        "hru_intcpevap": zeros.copy(),
        "hru_intcpstor": zeros.copy(),
        "hru_intcpstor_change": zeros.copy(),
        "pptmix": at(0.0, h5=1.0, h6=1.0),
        "hru_intcpstor_old": zeros.copy(),
        # inputs
        "pk_ice_prev": at(0.0, h4=1.0),
        "freeh2o_prev": zeros.copy(),
        "transp_on": at(1.0, h1=0.0, h8=0.0, h9=0.0),
        "hru_ppt": hru_rain + hru_snow,
        "hru_rain": hru_rain,
        "hru_snow": hru_snow,
        "potet": at(0.1, h13=-0.1),
        # parameters
        "cov_type": at(3, h0=0, h3=1, h4=1).astype(np.int64),
        "covden_sum": at(0.8, h9=0.5, h10=0.0, h13=0.0),
        "covden_win": at(0.3, h8=0.0, h9=0.9),
        "srain_intcp": full(0.1),
        "wrain_intcp": full(0.05),
        "snow_intcp": at(0.1, h7=0.5),
        "potet_sublim": at(0.5, h7=1.5),
    }


def test_canopy_kernel_units(canopy_data):
    out = assert_kernel_dimensionally_consistent(
        PRMSCanopy, canopy_data, patches=CANOPY_PATCHES
    )
    # each case did what the fixture says (the harness already required
    # every line to run; these pin WHICH hru ran it)
    assert out["intcp_changeover"][0] == 0.1  # extra_water
    assert out["intcp_changeover"][1] > 0.0
    assert out["intcp_evap"][1] == 0.05  # store evaporated whole
    assert out["intcp_stor"][2] == 0.1  # intercepted up to srain_intcp
    assert out["intcp_stor"][3] == 0.1  # grasses intercept (summer cap)
    assert out["net_rain"][4] == 0.1  # grasses under a snowpack
    assert out["pptmix"][5] == 0.0 and out["net_snow"][5] == 0.0
    assert out["intcp_form"][6] == 1 and out["intcp_evap"][6] == 0.05
    assert out["hru_intcpevap"][7] == pytest.approx(0.1)  # capped at potet
    assert out["intcp_stor"][8] == 0.0
    assert out["intcp_changeover"][9] == 0.0
    assert out["intcp_evap"][9] == pytest.approx(0.04 * 0.5 / 0.9)
    assert out["intcp_stor"][10] == 0.0
    assert out["intcp_changeover"][10] == pytest.approx(0.04 * 0.3)
    assert out["intcp_form"][11] == 1 and out["intcp_stor"][11] == 0.0
    assert out["intcp_evap"][11] == pytest.approx(0.01 + 5.0e-7)
    assert out["intcp_stor"][12] == pytest.approx(0.1)
    assert out["intcp_evap"][12] == 0.1
    assert out["intcp_evap"][13] == 0.0
    assert out["intcp_stor"][13] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "dropped, error, match",
    [
        # a bare float compared with an inches Quantity
        ("_NEARZERO", ValueError, "Cannot compare"),
        ("_DNEARZERO", ValueError, "Cannot compare"),
        # a bare 0.0 scaled by cov is dimensionless, then compared with
        # potet (inches)
        ("_ZERO_INCHES", pint.DimensionalityError, "dimensionless"),
        # the compiled helper cannot type a Quantity
        ("intercept", numba.core.errors.TypingError, "nopython"),
    ],
)
def test_canopy_patches_are_load_bearing(canopy_data, dropped, error, match):
    """Each patch is needed: without it the Quantity run fails."""
    patches = {kk: vv for kk, vv in CANOPY_PATCHES.items() if kk[1] != dropped}
    assert len(patches) == len(CANOPY_PATCHES) - 1
    with pytest.raises(error, match=match):
        run_quantified(PRMSCanopy, canopy_data, patches=patches)


# ----------------------------------------------------------------------
# channel
# ----------------------------------------------------------------------

CHANNEL_PATCHES = {
    (prms_channel, "_ROUTING_STEP_HOURS"): (
        prms_channel._ROUTING_STEP_HOURS * quantity("hours")
    ),
    (prms_channel, "_ZERO_CFS"): prms_channel._ZERO_CFS * quantity("cfs"),
    (prms_channel, "_ZERO_CUBIC_FEET"): (
        prms_channel._ZERO_CUBIC_FEET * quantity("cubic feet")
    ),
}


@pytest.fixture
def channel_data() -> dict[str, np.ndarray]:
    """Five segments, one day, every kernel line run (the harness
    enforces it)::

        0 --> 2 --> 4 --> outlet
        1 --'      ^
        3 ---------'

    tsi: 0 within-hour (-1: outflow = inflow), 1 hourly, 2 every two
    hours (the off hours skip routing), 3 every four, 4 every six.
    """
    nseg = 5
    s_per_time = np.float64(86400.0)
    lateral_cfs = np.array([1.0, 2.0, 0.5, 3.0, 0.25])
    tsi = np.array([-1, 1, 2, 4, 6], dtype=np.int64)
    zeros = np.zeros(nseg)
    return {
        # outputs, state and work buffers
        "seg_lateral_inflow": zeros.copy(),
        "seg_upstream_inflow": zeros.copy(),
        "seg_inflow": zeros.copy(),
        "seg_outflow": zeros.copy(),
        "seg_stor_change": zeros.copy(),
        "channel_outflow_vol": zeros.copy(),
        "inflow_ts": zeros.copy(),
        "outflow_ts": np.array([1.0, 2.0, 3.0, 3.0, 6.0]),
        "inflow_ts_prev": np.array([1.0, 2.0, 3.0, 3.0, 6.0]),
        "seg_current_sum": zeros.copy(),
        # Map-fed inputs [cubic feet]
        "seg_sroff_vol": 0.5 * lateral_cfs * s_per_time,
        "seg_ssres_flow_vol": 0.3 * lateral_cfs * s_per_time,
        "seg_gwres_flow_vol": 0.2 * lateral_cfs * s_per_time,
        # topology + coefficients (set directly; c0 + c1 + c2 = 1)
        "segment_order": np.array([0, 1, 3, 2, 4], dtype=np.int64),
        "tosegment0": np.array([2, 2, 4, 4, -1], dtype=np.int64),
        "ts": tsi.clip(1).astype(np.float64),
        "tsi": tsi,
        "c0": np.array([0.0, 0.25, 0.2, 0.1, 0.05]),
        "c1": np.array([1.0, 0.5, 0.5, 0.6, 0.55]),
        "c2": np.array([0.0, 0.25, 0.3, 0.3, 0.4]),
        "s_per_time": s_per_time,
    }


def test_channel_kernel_units(channel_data):
    out = assert_kernel_dimensionally_consistent(
        PRMSChannel, channel_data, patches=CHANNEL_PATCHES
    )
    # within-hour headwater: outflow = inflow, hour by hour
    assert out["seg_outflow"][0] == out["seg_inflow"][0]
    # only the outlet sends volume out of the domain
    assert out["channel_outflow_vol"][4] > 0.0
    assert np.all(out["channel_outflow_vol"][:4] == 0.0)
    # seg 2 is fed by 0 and 1
    assert out["seg_upstream_inflow"][2] > 0.0


@pytest.mark.parametrize(
    "dropped, error, match",
    [
        # inflow_ts / ts is cfs / hour, assigned into a cfs array
        ("_ROUTING_STEP_HOURS", pint.DimensionalityError, "/ hour / second"),
        # a bare 0.0 assigned into a cfs / cubic feet array
        (
            "_ZERO_CFS",
            pint.DimensionalityError,
            "from '0.0' to 'foot \\*\\* 3 /",
        ),
        (
            "_ZERO_CUBIC_FEET",
            pint.DimensionalityError,
            "from '0.0' to 'foot \\*\\* 3'",
        ),
    ],
)
def test_channel_patches_are_load_bearing(channel_data, dropped, error, match):
    """Each patch is needed: without it the Quantity run fails."""
    patches = {
        kk: vv for kk, vv in CHANNEL_PATCHES.items() if kk[1] != dropped
    }
    assert len(patches) == len(CHANNEL_PATCHES) - 1
    with pytest.raises(error, match=match):
        run_quantified(PRMSChannel, channel_data, patches=patches)
