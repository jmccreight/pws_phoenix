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
count), and its init derivation muskingum_mann_coefficients (plain
numpy, run through the harness's ``func=``); the PRMSSoilzone family
(four helpers, derived fields under other argument names via
``ARG_FIELDS``, the 1/(inch day) ``*coef_sq``); the PRMSRunoff family
(five helpers; zeros and near-zeros in inches, acres and acre-inches;
a bool kernel option run both ways, coverage accumulated). Harness:
tests/kernel_units.py.
Synthetic data, CI-able.
"""

import numba
import numpy as np
import pytest

pint = pytest.importorskip("pint")

from kernel_units import (  # noqa: E402
    PER_DAY,
    assert_kernel_dimensionally_consistent,
    inches,
    run_compiled,
    run_quantified,
)
from pws_phoenix.hydrology import (  # noqa: E402
    prms_canopy,
    prms_channel,
    prms_runoff,
    prms_soilzone,
)
from pws_phoenix.hydrology.prms_canopy import PRMSCanopy  # noqa: E402
from pws_phoenix.hydrology.prms_channel import PRMSChannel  # noqa: E402
from pws_phoenix.hydrology.prms_runoff import (  # noqa: E402
    PRMSRunoff,
    PRMSRunoffNoDprst,
)
from pws_phoenix.hydrology.prms_soilzone import (  # noqa: E402
    PRMSSoilzone,
    PRMSSoilzoneNoDprst,
)
from pws_phoenix.units import quantity, registry  # noqa: E402
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


# ----------------------------------------------------------------------
# channel init: muskingum_mann_coefficients (plain numpy, not a kernel;
# it writes the ts/tsi/c0/c1/c2 the kernel reads)
# ----------------------------------------------------------------------

MUSKINGUM_PATCH_UNITS = {
    "_SECONDS_PER_MINUTE": "second / minute",
    "_MINUTES_PER_HOUR": "minute / hour",
    "_KCOEF_MIN_HOURS": "hour",
    "_KCOEF_MAX_HOURS": "hour",
    "_ONE_HOUR": "hour",
    "_DD_NEARZERO_HOURS": "hour",
    "_DD_FLOOR_HOURS": "hour",
}
MUSKINGUM_PATCHES = {
    (prms_channel, name): getattr(prms_channel, name) * registry()(expr)
    for name, expr in MUSKINGUM_PATCH_UNITS.items()
}


@pytest.fixture
def muskingum_data() -> dict[str, np.ndarray]:
    """Eleven segments: one per rung of the sub-timestep ladder (kcoef
    targets in hours; the first clamped up to 0.01, the ninth down to
    24), then a lake (kcoef forced to 24) and a flat segment (velocity
    0, kcoef stays 24). Segment 1 (x_coef 0.6) has c2 < 0 (short travel
    time), segment 7 (x_coef 0.4) c0 < 0 (long)."""
    kcoef_hours = np.array(
        [0.001, 1.1, 2.5, 3.5, 5.0, 7.0, 10.0, 20.0, 30.0, 5.0, 5.0]
    )
    nseg = kcoef_hours.size
    lake, flat = 9, 10
    # Manning velocity with these mann_n/slope/depth: 0.5 m/s
    velocity_m_per_hour = (1.0 / 0.04) * np.sqrt(0.0004) * 1.0 * 3600.0
    x_coef = np.full(nseg, 0.2)
    x_coef[1] = 0.6
    x_coef[7] = 0.4
    seg_slope = np.full(nseg, 0.0004)
    seg_slope[flat] = 0.0
    segment_type = np.ones(nseg, dtype=np.int64)
    segment_type[lake] = 2
    return {
        # outputs
        "ts": np.zeros(nseg),
        "tsi": np.zeros(nseg, dtype=np.int64),
        "c0": np.zeros(nseg),
        "c1": np.zeros(nseg),
        "c2": np.zeros(nseg),
        # parameters
        "mann_n": np.full(nseg, 0.04),
        "seg_slope": seg_slope,
        "seg_depth": np.ones(nseg),
        "seg_length": kcoef_hours * velocity_m_per_hour,
        "segment_type": segment_type,
        "x_coef": x_coef,
    }


def test_muskingum_coefficients_units(muskingum_data):
    out = assert_kernel_dimensionally_consistent(
        PRMSChannel,
        muskingum_data,
        patches=MUSKINGUM_PATCHES,
        func=prms_channel.muskingum_mann_coefficients,
    )
    np.testing.assert_array_equal(
        out["tsi"], [-1, 1, 2, 3, 4, 6, 8, 12, 24, 24, 24]
    )
    np.testing.assert_array_equal(
        out["ts"], [1, 1, 2, 3, 4, 6, 8, 12, 24, 24, 24]
    )
    assert out["c2"][1] == 0.0  # short travel time
    assert out["c0"][7] == 0.0 and out["c2"][7] > 0.0  # long
    np.testing.assert_allclose(out["c0"] + out["c1"] + out["c2"], 1.0)


@pytest.mark.parametrize(
    "dropped, error, match",
    [
        # m/s * 60 * 60 stays m/s under pint: kcoef comes out in seconds
        # and pint converts it SILENTLY into the hours array -- only the
        # magnitude comparison sees it (ts lands on the wrong rung)
        ("_SECONDS_PER_MINUTE", AssertionError, "ts"),
        ("_MINUTES_PER_HOUR", AssertionError, "ts"),
        ("_KCOEF_MIN_HOURS", ValueError, "Cannot compare"),
        # plain ones * 24.0: numpy strips the units on kcoef[wh_moving] = ...
        ("_KCOEF_MAX_HOURS", pint.errors.UnitStrippedWarning, "stripped"),
        ("_ONE_HOUR", pint.DimensionalityError, "from '1.0' to 'hour'"),
        ("_DD_NEARZERO_HOURS", ValueError, "Cannot compare"),
        ("_DD_FLOOR_HOURS", pint.DimensionalityError, "to 'hour'"),
    ],
)
def test_muskingum_patches_are_load_bearing(
    muskingum_data, dropped, error, match
):
    """Each patch is needed: without it the check fails."""
    patches = {
        kk: vv for kk, vv in MUSKINGUM_PATCHES.items() if kk[1] != dropped
    }
    assert len(patches) == len(MUSKINGUM_PATCHES) - 1
    with pytest.raises(error, match=match):
        assert_kernel_dimensionally_consistent(
            PRMSChannel,
            muskingum_data,
            patches=patches,
            func=prms_channel.muskingum_mann_coefficients,
        )


# ----------------------------------------------------------------------
# soilzone
# ----------------------------------------------------------------------

SOILZONE_PATCHES = {
    (prms_soilzone, "_NEARZERO"): inches(prms_soilzone._NEARZERO),
    (prms_soilzone, "_ZERO_INCHES"): inches(prms_soilzone._ZERO_INCHES),
    (
        prms_soilzone,
        "compute_soilmoist",
    ): prms_soilzone.compute_soilmoist.py_func,
    (
        prms_soilzone,
        "compute_interflow",
    ): prms_soilzone.compute_interflow.py_func,
    (prms_soilzone, "compute_gwflow"): prms_soilzone.compute_gwflow.py_func,
    (prms_soilzone, "compute_szactet"): prms_soilzone.compute_szactet.py_func,
}


@pytest.fixture
def soilzone_data() -> dict[str, np.ndarray]:
    """Twenty-four hrus, one step, together running every line of both
    soilzone kernels and the four helpers (the harness enforces it).
    Default: a LAND loam hru under trees, transpiring, no infiltration,
    ssr2gw_exp = 1 (the case where ssr2gw_rate's 1/day is exact).

    0 infiltration absorbed by the capillary reservoir (no excess)
    1 capillary excess, part to groundwater (soil2gw_flag)
    2 soil_moist above soil_moist_max: all infiltration rejected (not
      physical, reaches it)
    3 preferential flow: pfr overflows to dunnian, gravity excess to
      pfr overflows too, fast interflow
    4 dry: no infiltration, no soil moisture, soil_lower_max 0
    5 swale ponding above sat_threshold, swale ET
    6 interflow, coef_lin = 0 and ssres_in <= 0
    7 interflow, coef_sq = 0; negative result clamped to 0
    8 interflow, coef_lin = coef_sq = 0 with ssres_in > 0: none
    9 interflow above storage, clamped to storage
    10 interflow with c3 = sqrt(negative) = nan (slow_stor above
       pref_flow_thrsh at the start: not physical, reaches it)
    11 ET: avail_potet below _NEARZERO
    12 ET: no transpiration, snow covered -> default (none)
    13 ET: no transpiration, half snow cover -> evaporation only
    14 ET: bare (cov_type 0), snow covered -> default (none)
    15 ET: bare, snow free -> evaporation only
    16 sand, pcts and pctr below 0.25
    17 loam, pcts and pctr below 0.5
    18 clay, pcts and pctr between 1/3 and 2/3
    19 clay, pcts and pctr at most 1/3
    20 potet_rechr above soil_rechr
    21 soil_rechr above soil_moist, evaporation only (not physical,
       reaches it)
    22 potet_lower above soil_moist
    23 potet_lower taken from soil_moist
    """
    nhru = 24

    def full(value, dtype=np.float64):
        return np.full(nhru, value, dtype=dtype)

    def at(default, dtype=np.float64, **by_hru):
        arr = full(default, dtype)
        for key, value in by_hru.items():
            arr[int(key[1:])] = value
        return arr

    zeros = full(0.0)
    return {
        # in/out state + outputs
        "cap_infil_tot": zeros.copy(),
        "cap_waterin": zeros.copy(),
        "dunnian_flow": zeros.copy(),
        "hru_actet": zeros.copy(),
        "perv_actet": zeros.copy(),
        "perv_actet_hru": zeros.copy(),
        "potet_lower": zeros.copy(),
        "potet_rechr": zeros.copy(),
        "pref_flow": zeros.copy(),
        "pref_flow_in": zeros.copy(),
        "pref_flow_infil": zeros.copy(),
        "pref_flow_stor": at(0.0, h3=0.45),
        "pref_flow_stor_change": zeros.copy(),
        "recharge": zeros.copy(),
        "slow_flow": zeros.copy(),
        "slow_stor": at(0.5, h3=1.2, h5=2.5, h7=2.0, h8=0.1, h9=0.1, h10=2.0),
        "slow_stor_change": zeros.copy(),
        "soil_lower": zeros.copy(),
        "soil_lower_change": zeros.copy(),
        "soil_lower_change_hru": zeros.copy(),
        "soil_lower_ratio": zeros.copy(),
        "soil_moist": at(
            2.0,
            h1=3.95,
            h2=4.5,
            h4=0.0,
            h8=3.95,
            h9=3.95,
            h16=0.8,
            h17=1.0,
            h19=1.0,
            h21=0.05,
            h22=0.09,
            h23=3.0,
        ),
        "soil_moist_tot": zeros.copy(),
        "soil_rechr": at(
            1.0,
            h4=0.0,
            h16=0.3,
            h17=0.5,
            h18=0.75,
            h19=0.3,
            h20=0.015,
            h22=0.005,
            h23=0.3,
        ),
        "soil_rechr_change": zeros.copy(),
        "soil_rechr_change_hru": zeros.copy(),
        "soil_to_gw": zeros.copy(),
        "soil_to_ssr": zeros.copy(),
        "ssr_to_gw": zeros.copy(),
        "ssres_flow": zeros.copy(),
        "ssres_flow_vol": zeros.copy(),
        "ssres_in": zeros.copy(),
        "ssres_stor": zeros.copy(),
        "swale_actet": zeros.copy(),
        "unused_potet": zeros.copy(),
        # mutable inputs
        "sroff": full(0.05),
        "sroff_vol": zeros.copy(),
        # prior state
        "pref_flow_stor_prev": zeros.copy(),
        "soil_lower_prev": full(1.0),
        "soil_rechr_prev": full(1.0),
        "slow_stor_prev": full(0.5),
        # inputs
        "dprst_evap_hru": full(0.01),
        "dprst_seep_hru": full(0.005),
        "hru_impervevap": full(0.01),
        "hru_intcpevap": full(0.01),
        "infil_hru": at(0.0, h0=0.1, h1=0.8, h2=0.05, h3=0.5, h8=0.8, h9=0.8),
        "potet": at(0.2, h5=0.5, h11=0.0),
        "transp_on": at(1.0, h12=0.0, h13=0.0),
        "snow_evap": zeros.copy(),
        "snowcov_area": at(0.0, h12=1.0, h13=0.5, h14=1.0),
        # parameters + derived
        "hru_type": at(1, np.int64, h5=3),
        "hru_in_to_cf": full(3000.0),
        "cov_type": at(3, np.int64, h14=0, h15=0, h21=0),
        "fastcoef_lin": full(0.1),
        "fastcoef_sq": full(0.8),
        "pref_flow_infil_frac": at(0.0, h3=0.2, h4=0.1),
        "slowcoef_lin": at(0.015, h6=0.0, h7=0.5, h8=0.0),
        "slowcoef_sq": at(0.1, h7=0.0, h8=0.0, h9=10.0),
        "soil2gw_max": at(0.0, h1=0.1),
        "soil_moist_max": at(4.0, h22=0.1),
        "soil_type": at(2, np.int64, h16=1, h18=3, h19=3, h22=3),
        "ssr2gw_exp": full(1.0),
        "ssr2gw_rate": full(0.1),
        "hru_frac_perv": full(0.8),
        "soil_rechr_max": at(1.5, h4=4.0, h20=0.02, h22=0.05),
        "soil_lower_max": at(2.5, h4=0.0, h20=3.98, h22=0.05),
        "sat_threshold": full(2.0),
        "pref_flow_den": at(0.0, h3=0.1),
        "pref_flow_thrsh": at(1.0, h7=0.5),
        "pref_flow_max": full(0.5),
        "pref_flow_flag": full(False, np.bool_),
        "soil2gw_flag": at(False, np.bool_, h1=True),
    }


# hru 10 takes np.sqrt of a negative on purpose (c3 = nan)
SQRT_NAN = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in sqrt:RuntimeWarning"
)
SOILZONE_CLASSES = [PRMSSoilzoneNoDprst, PRMSSoilzone]


@SQRT_NAN
@pytest.mark.parametrize("cls", SOILZONE_CLASSES)
def test_soilzone_kernel_units(cls, soilzone_data):
    out = assert_kernel_dimensionally_consistent(
        cls, soilzone_data, patches=SOILZONE_PATCHES
    )
    # each case did what the fixture says (the harness already required
    # every line to run; these pin WHICH hru ran it). avail = potet -
    # the evaporation seed of hru_actet (NoDprst: no dprst_evap_hru)
    avail = 0.2 - 0.02 - (0.01 if cls is PRMSSoilzone else 0.0)
    assert out["soil_to_ssr"][0] == 0.0  # absorbed
    assert out["soil_to_gw"][1] == 0.1  # soil2gw_max x 1 day
    assert out["cap_waterin"][2] == 0.0  # infiltration rejected
    assert out["pref_flow"][3] > 0.0
    assert out["dunnian_flow"][3] == pytest.approx(0.25)  # pfr + gvr
    assert out["perv_actet"][4] == 0.0
    assert out["swale_actet"][5] > 0.0
    assert 0.0 < out["slow_flow"][6] < out["slow_flow"][0]
    assert out["slow_flow"][7] == 0.0  # negative, clamped
    assert out["slow_flow"][8] == 0.0  # no interflow
    assert out["slow_flow"][9] == pytest.approx(0.86)  # = storage
    assert out["slow_stor"][9] == 0.0
    assert out["slow_flow"][10] == 0.0  # nan path, then clamped
    assert out["perv_actet"][11] == 0.0
    assert out["perv_actet"][12] == 0.0
    assert out["perv_actet"][13] == pytest.approx(0.5 * avail)  # x snow_free
    assert out["perv_actet"][14] == 0.0
    assert out["perv_actet"][15] == pytest.approx(avail)
    assert out["potet_lower"][16] == pytest.approx(0.5 * 0.2 * avail)
    assert out["potet_lower"][17] == pytest.approx(0.25 * avail)
    assert out["potet_lower"][18] == pytest.approx(0.5 * avail)
    assert out["potet_lower"][19] == pytest.approx(0.5 * 0.25 * avail)
    assert out["potet_rechr"][20] == 0.015  # all of soil_rechr
    assert out["potet_rechr"][21] == 0.05  # all of soil_moist
    assert out["soil_rechr"][21] == 0.0  # clamped to soil_moist
    assert out["perv_actet"][22] == 0.09  # all of soil_moist
    assert out["perv_actet"][23] == pytest.approx(avail)  # potet_lower


@SQRT_NAN
@pytest.mark.parametrize(
    "dropped, error, match",
    [
        # a bare float compared with an inches Quantity
        ("_NEARZERO", ValueError, "Cannot compare"),
        # a bare 0.0 assigned into an inches array
        ("_ZERO_INCHES", pint.DimensionalityError, "from '0.0' to 'inch'"),
        # a compiled helper cannot type a Quantity
        ("compute_soilmoist", numba.core.errors.TypingError, "nopython"),
        ("compute_interflow", numba.core.errors.TypingError, "nopython"),
        ("compute_gwflow", numba.core.errors.TypingError, "nopython"),
        ("compute_szactet", numba.core.errors.TypingError, "nopython"),
    ],
)
def test_soilzone_patches_are_load_bearing(
    soilzone_data, dropped, error, match
):
    """Each patch is needed: without it the Quantity run fails."""
    patches = {
        kk: vv for kk, vv in SOILZONE_PATCHES.items() if kk[1] != dropped
    }
    assert len(patches) == len(SOILZONE_PATCHES) - 1
    with pytest.raises(error, match=match):
        run_quantified(PRMSSoilzone, soilzone_data, patches=patches)


@SQRT_NAN
@pytest.mark.parametrize(
    "name",
    [
        "fastcoef_lin",
        "slowcoef_lin",
        "ssr2gw_rate",
        "soil2gw_max",
        "fastcoef_sq",
        "slowcoef_sq",
    ],
)
def test_soilzone_per_day_is_load_bearing(soilzone_data, name):
    """Each soilzone rate needs its 1 day: without it the run fails."""
    with pytest.raises(pint.DimensionalityError, match="day"):
        run_quantified(
            PRMSSoilzone,
            soilzone_data,
            patches=SOILZONE_PATCHES,
            per_day=PER_DAY - {name},
        )


@SQRT_NAN
@pytest.mark.parametrize("name", ["fastcoef_sq", "slowcoef_sq"])
def test_coef_sq_dimensionless_declaration_is_caught(soilzone_data, name):
    """The upstream declaration (dimensionless; PRMS and pywatershed)
    fails: compute_interflow adds 1.0 to coef_sq * storage."""
    with pytest.raises(pint.DimensionalityError, match="inch"):
        run_quantified(
            PRMSSoilzone,
            soilzone_data,
            declared={name: "fraction"},
            patches=SOILZONE_PATCHES,
            per_day=PER_DAY - {name},
        )


@SQRT_NAN
def test_ssr2gw_rate_is_exact_only_for_unit_exponent(soilzone_data):
    """ssr2gw_rate is truly inch^(1 - ssr2gw_exp)/day; its 1/day
    declaration is exact for ssr2gw_exp == 1 only (the fixture's
    value)."""
    data = dict(soilzone_data)
    data["ssr2gw_exp"] = np.full_like(data["ssr2gw_exp"], 0.5)
    with pytest.raises(pint.DimensionalityError, match=r"inch \*\* 0.5"):
        run_quantified(PRMSSoilzone, data, patches=SOILZONE_PATCHES)


# ----------------------------------------------------------------------
# runoff
# ----------------------------------------------------------------------


def acres(value: float) -> pint.Quantity:
    return value * quantity("acres")


def acre_inches(value: float) -> pint.Quantity:
    return value * quantity("acre-inches")


RUNOFF_PATCHES = {
    (prms_runoff, "_NEARZERO_INCHES"): inches(prms_runoff._NEARZERO_INCHES),
    (prms_runoff, "_NEARZERO_ACRES"): acres(prms_runoff._NEARZERO_ACRES),
    (prms_runoff, "_DNEARZERO"): inches(prms_runoff._DNEARZERO),
    (prms_runoff, "_ZERO_INCHES"): inches(prms_runoff._ZERO_INCHES),
    (prms_runoff, "_ZERO_ACRES"): acres(prms_runoff._ZERO_ACRES),
    (prms_runoff, "_ZERO_ACRE_INCHES"): acre_inches(
        prms_runoff._ZERO_ACRE_INCHES
    ),
    (prms_runoff, "_SMIDX_MAX_INCHES"): inches(prms_runoff._SMIDX_MAX_INCHES),
    (prms_runoff, "perv_comp"): prms_runoff.perv_comp.py_func,
    (prms_runoff, "check_capacity"): prms_runoff.check_capacity.py_func,
    (prms_runoff, "imperv_et"): prms_runoff.imperv_et.py_func,
    (prms_runoff, "compute_infil"): prms_runoff.compute_infil.py_func,
    (prms_runoff, "dprst_comp"): prms_runoff.dprst_comp.py_func,
}


@pytest.fixture
def runoff_data() -> dict[str, np.ndarray]:
    """Twenty hrus, one step (run with intcp_changeover_in_net_rain
    False and True), together running every line of both runoff kernels
    and the five helpers (the harness enforces it). Default: a LAND hru
    of 100 acres, 10% impervious, 5 acres of depressions (60% open),
    half full; dry.

    0 dry: depression evaporation, seepage only
    1 rain on a snow-free hru: pervious, impervious and depression
      runoff
    2 canopy changeover water; potet below the impervious store
    3 rain/snow mix with no antecedent pack onto a snowpack: capacity
      check of the infiltration
    4 snowmelt from a pack
    5 snowmelt over capacity: excess above snowinfil_max runs off
    6 snowmelt that depleted the pack, with rain
    7 snowmelt that depleted the pack, no rain
    8 smidx above 25 inches (not physical, reaches it)
    9 contributing fraction above carea_max (large smidx_coef)
    10 negative avail_et under full snow cover (hru_intcpevap > potet:
       not physical, reaches it)
    11 a lake hru
    12 sro_to_dprst_* > 1: srp and sri driven negative, clamped (not
       physical, reaches it)
    13 nearly empty depressions: no area
    14 depressions over their maximum volume: full area, spill
    15 negative va_*_exp: area above its maximum, clamped (not physical,
       reaches it)
    16 depression evaporation limited by the unsatisfied ET (10 acres)
    17 seepage rates above 1/day: volumes driven negative, clamped (not
       physical, reaches it)
    18 outflow coefficient above 1/day with no threshold: open volume
       driven negative, clamped (not physical, reaches it)
    19 empty depressions: the area stays the bare-zero initial value
       (_ZERO_ACRES -- statement coverage alone would not need this hru)
    """
    nhru = 20

    def full(value, dtype=np.float64):
        return np.full(nhru, value, dtype=dtype)

    def at(default, dtype=np.float64, **by_hru):
        arr = full(default, dtype)
        for key, value in by_hru.items():
            arr[int(key[1:])] = value
        return arr

    zeros = full(0.0)
    rain = dict(h1=0.5, h8=0.5, h9=0.5, h12=0.5, h16=0.5)
    return {
        # in/out state + outputs
        "contrib_fraction": zeros.copy(),
        "infil": zeros.copy(),
        "infil_hru": zeros.copy(),
        "sroff": zeros.copy(),
        "sroff_vol": zeros.copy(),
        "hru_sroffp": zeros.copy(),
        "hru_sroffi": zeros.copy(),
        "imperv_stor": at(0.0, h10=0.02),
        "imperv_evap": zeros.copy(),
        "hru_impervevap": zeros.copy(),
        "hru_impervstor": zeros.copy(),
        "hru_impervstor_change": zeros.copy(),
        "dprst_vol_open": at(6.0, h13=1.0e-8, h14=15.0, h19=0.0),
        "dprst_vol_clos": at(4.0, h13=1.0e-8, h14=10.0, h19=0.0),
        "dprst_vol_open_frac": zeros.copy(),
        "dprst_vol_clos_frac": zeros.copy(),
        "dprst_vol_frac": zeros.copy(),
        "dprst_area_open": zeros.copy(),
        "dprst_area_clos": full(1.0),
        "dprst_sroff_hru": zeros.copy(),
        "dprst_seep_hru": zeros.copy(),
        "dprst_evap_hru": zeros.copy(),
        "dprst_insroff_hru": zeros.copy(),
        "dprst_stor_hru": zeros.copy(),
        "dprst_stor_hru_change": zeros.copy(),
        "dprst_in": zeros.copy(),
        # prior state
        "hru_impervstor_old": zeros.copy(),
        "dprst_stor_hru_old": zeros.copy(),
        # inputs
        "soil_lower_prev": at(1.0, h5=3.0, h8=30.0),
        "soil_rechr_prev": at(0.5, h5=0.9, h8=0.0),
        "net_rain": at(0.0, h3=0.2, h6=0.2, **rain),
        "net_ppt": at(0.0, h3=0.3, h6=0.2, **rain),
        "net_snow": at(0.0, h3=0.1),
        "potet": at(0.2, h2=0.01, h10=0.0),
        "snowmelt": at(0.0, h4=0.3, h5=1.0, h6=0.3, h7=0.3),
        "snow_evap": zeros.copy(),
        "pkwater_equiv": at(0.0, h3=0.5, h4=1.0, h5=1.0),
        "pptmix_nopack": at(0.0, h3=1.0),
        "snowcov_area": at(0.0, h10=1.0),
        "through_rain": at(0.0, h3=0.2, **rain),
        "hru_intcpevap": at(0.01, h10=0.01, h16=0.18),
        "intcp_changeover": at(0.0, h2=0.3),
        # parameters + derived
        "hru_type": at(1, np.int64, h11=2),
        "hru_area": at(100.0, h16=10.0),
        "hru_in_to_cf": full(3000.0),
        "hru_percent_imperv": full(0.1),
        "imperv_stor_max": full(0.05),
        "carea_max": full(0.6),
        "smidx_coef": at(0.005, h9=1.0),
        "smidx_exp": full(0.3),
        "soil_moist_max": full(4.0),
        "snowinfil_max": at(2.0, h5=0.5),
        "dprst_et_coef": full(1.0),
        "dprst_flow_coef": at(0.05, h18=3.0),
        "dprst_frac_open": full(0.6),
        "dprst_seep_rate_clos": at(0.02, h17=2.0),
        "dprst_seep_rate_open": at(0.02, h17=2.0),
        "sro_to_dprst_imperv": at(0.2, h12=2.0),
        "sro_to_dprst_perv": at(0.2, h12=2.0),
        "va_open_exp": at(0.5, h15=-0.5),
        "va_clos_exp": at(0.5, h15=-0.5),
        "hru_perv": at(85.0, h16=4.0),
        "hru_frac_perv": at(0.85, h16=0.4),
        "hru_imperv": at(10.0, h16=1.0),
        "dprst_area_max": full(5.0),
        "dprst_area_open_max": full(3.0),
        "dprst_area_clos_max": full(2.0),
        "dprst_frac_clos": full(0.4),
        "dprst_vol_open_max": full(12.0),
        "dprst_vol_clos_max": full(8.0),
        "dprst_vol_thres_open": at(6.0, h18=0.0),
        # option (set per run)
        "intcp_changeover_in_net_rain": np.bool_(False),
    }


RUNOFF_CLASSES = [PRMSRunoffNoDprst, PRMSRunoff]


def runoff_patches(cls) -> dict:
    """RUNOFF_PATCHES, less dprst_comp for the NoDprst kernel (a
    patched helper is coverage-traced, so it must be one the kernel
    calls)."""
    if cls is PRMSRunoff:
        return RUNOFF_PATCHES
    return {
        kk: vv for kk, vv in RUNOFF_PATCHES.items() if kk[1] != "dprst_comp"
    }


def runoff_both_options(cls, data) -> tuple[dict, dict]:
    """The kernel under intcp_changeover_in_net_rain False and True,
    line coverage accumulated over both; returns the compiled arrays
    of each."""
    fired: dict = {}
    out = []
    for option in (False, True):
        out.append(
            assert_kernel_dimensionally_consistent(
                cls,
                data | {"intcp_changeover_in_net_rain": np.bool_(option)},
                patches=runoff_patches(cls),
                fired=fired,
                require_all_lines=option,
            )
        )
    return out[0], out[1]


@pytest.mark.parametrize("cls", RUNOFF_CLASSES)
def test_runoff_kernel_units(cls, runoff_data):
    out, out_in_net_rain = runoff_both_options(cls, runoff_data)
    # each case did what the fixture says (the harness already required
    # every line to run; these pin WHICH hru ran it)
    assert out["infil"][0] == 0.0  # dry
    assert out["hru_sroffp"][1] > 0.0 and out["hru_sroffi"][1] > 0.0
    assert out["infil"][2] > 0.0  # changeover water infiltrates ...
    assert out_in_net_rain["infil"][2] == 0.0  # ... unless in net_rain
    assert out["imperv_stor"][2] == 0.05  # filled to imperv_stor_max
    assert out["hru_impervevap"][2] == 0.0  # avail_et exhausted
    # smidx on the rain part first, then the capacity check
    assert out["contrib_fraction"][3] > 0.0
    assert out["infil"][3] > 0.0
    assert out["infil"][4] == pytest.approx(0.3)  # all melt infiltrates
    # over capacity: snowinfil_max x 1 day + capacity (4 - 3.9)
    assert out["infil"][5] == pytest.approx(0.5 + 0.1)
    assert out["contrib_fraction"][6] > 0.0  # smidx on melt + rain
    assert out["contrib_fraction"][7] == 0.0  # capacity check
    assert out["contrib_fraction"][8] == 0.6  # smidx > 25: carea_max
    assert out["contrib_fraction"][9] == 0.6  # clamped to carea_max
    assert out["hru_impervevap"][10] == 0.0  # negative, clamped
    assert out["sroff"][11] == 0.0  # lake
    if cls is not PRMSRunoff:
        return
    assert out["dprst_in"][1] > 0.0
    assert out["hru_sroffp"][12] == 0.0 and out["hru_sroffi"][12] == 0.0
    assert out["dprst_area_open"][13] == 0.0
    assert out["dprst_area_open"][14] == 3.0  # full
    assert out["dprst_sroff_hru"][14] > 0.0  # spill
    assert out["dprst_area_open"][15] == 3.0  # clamped
    assert out["dprst_evap_hru"][16] == pytest.approx(0.2 - 0.18)
    assert out["dprst_vol_open"][17] == 0.0
    assert out["dprst_vol_clos"][17] == 0.0
    assert out["dprst_vol_open"][18] == 0.0
    assert out["dprst_area_open"][19] == 0.0


@pytest.mark.parametrize(
    "dropped, error, match",
    [
        # a bare float compared with a Quantity
        ("_NEARZERO_INCHES", ValueError, "Cannot compare"),
        ("_NEARZERO_ACRES", ValueError, "Cannot compare"),
        ("_DNEARZERO", ValueError, "Cannot compare"),
        ("_SMIDX_MAX_INCHES", ValueError, "Cannot compare"),
        # a bare 0.0 assigned into a dimensional array (acre-inches:
        # dprst_in's bare start divided by hru_area is 1/acre)
        ("_ZERO_INCHES", pint.DimensionalityError, "from '0.0' to 'inch'"),
        ("_ZERO_ACRES", pint.DimensionalityError, "from '0.0' to 'acre'"),
        (
            "_ZERO_ACRE_INCHES",
            pint.DimensionalityError,
            "from '0.0 / acre' to 'inch'",
        ),
        # a compiled helper cannot type a Quantity
        ("perv_comp", numba.core.errors.TypingError, "nopython"),
        ("check_capacity", numba.core.errors.TypingError, "nopython"),
        ("imperv_et", numba.core.errors.TypingError, "nopython"),
        ("compute_infil", numba.core.errors.TypingError, "nopython"),
        ("dprst_comp", numba.core.errors.TypingError, "nopython"),
    ],
)
def test_runoff_patches_are_load_bearing(runoff_data, dropped, error, match):
    """Each patch is needed: without it the Quantity run fails."""
    patches = {kk: vv for kk, vv in RUNOFF_PATCHES.items() if kk[1] != dropped}
    assert len(patches) == len(RUNOFF_PATCHES) - 1
    with pytest.raises(error, match=match):
        run_quantified(PRMSRunoff, runoff_data, patches=patches)


@pytest.mark.parametrize(
    "name",
    [
        "snowinfil_max",
        "dprst_flow_coef",
        "dprst_seep_rate_open",
        "dprst_seep_rate_clos",
    ],
)
def test_runoff_per_day_is_load_bearing(runoff_data, name):
    """Each runoff rate needs its 1 day: without it the run fails."""
    with pytest.raises(pint.DimensionalityError, match="day"):
        run_quantified(
            PRMSRunoff,
            runoff_data,
            patches=RUNOFF_PATCHES,
            per_day=PER_DAY - {name},
        )
