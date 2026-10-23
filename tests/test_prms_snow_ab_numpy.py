"""A/B parity: the ported PRMSSnow vs pywatershed's OWN numpy path,
in memory, at EXACT equality (atol = rtol = 0).

The generated answer files come from pywatershed's numba path compiled
with fastmath=True, whose ulp-level drift flips pack-survival knife
edges (see test_prms_snow.py). This test removes the answers from the
equation entirely: it runs pywatershed's PRMSSnow with
calc_method="numpy" (strict IEEE, the same arithmetic contract as this
port) side by side with our Model for the first N_STEPS days (a full
snow season) and requires the compared states to be BIT-IDENTICAL --
except on the hrus touched by the port's ONE deliberate departure
from PRMS/pywatershed (Oct 2026, found by the mass budget):
calc_ppt_to_pack 1.1.3 ACCUMULATES freeh2o when rain exceeds a cold
pack's heat deficit (`freeh2o = freeh2o + net_rain - pndz`) where
PRMS overwrites it (`freeh2o = net_rain - pndz`), dropping any trace
of free water the cold pack held. The affected hrus are identified
FROM THE DATA (a rain day on a pack holding free water with a heat
deficit) and EXCLUDED: a 1e-7-inch trace flips pack-survival and
albedo-reset knife edges (tcal by O(100), albedo by 0.4, a day's rain
between through_rain and snowmelt), so pywatershed is no reference
there -- their mass is what test_prms_snow.py's budget assert covers.
Every other hru must still match bit for bit, and the affected count
is pinned small (9 of 765 on drb).

This is the same A/B pattern as test_starfit_daily_parity.py: requires
pywatershed importable (the pwpx env carries its import chain) and the
generated drb_2yr input files; skips with a reason otherwise. N_STEPS
is limited because pywatershed's numpy path is a pure-python HRU loop
(slow); 120 days covers Jan-Apr 1979 including the pack build-up and
melt-out.
"""

import pathlib as pl
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr
from pws_phoenix.discretization import Discretization
from pws_phoenix.hydrology.prms_snow import PRMSSnow
from pws_phoenix.model import Model

MPIX_ROOT = pl.Path(__file__).parents[2]
# pywatershed is used from the mpix-root repo clone (parity-test
# convention; see test_starfit_daily_parity.py)
sys.path.insert(0, str(MPIX_ROOT / "pywatershed"))
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

N_STEPS = 120
# states compared bit-identically (a spread over the whole kernel:
# mass, energy, density, SCA, albedo, season flags, through_rain)
CHECK = (
    "pkwater_equiv",
    "pk_ice",
    "freeh2o",
    "pk_def",
    "pk_temp",
    "pk_den",
    "pk_depth",
    "snowcov_area",
    "snowmelt",
    "snow_evap",
    "albedo",
    "tcal",
    "iso",
    "through_rain",
)
# captured from OUR run to locate the departure (see module docstring)
TRIGGER_NAMES = ("freeh2o_prev", "net_rain")
# hrus the departure can touch within N_STEPS (observed on drb: 9);
# the departure must stay a rare trace effect
MAX_AFFECTED_HRUS = 12

_needed = [
    DOMAIN_DIR / "parameters_PRMSSnow.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "nhm.control",
    GEN_DIR / "soltab_horad_potsw.nc",
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pywatershed = pytest.importorskip(
    "pywatershed", reason="pywatershed not importable"
)
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def pyws_states():
    """Run pywatershed PRMSSnow (calc_method='numpy') for N_STEPS."""
    from pywatershed.base.control import Control
    from pywatershed.hydrology.prms_snow import PRMSSnow as PywsSnow
    from pywatershed.parameters import Parameters, PrmsParameters

    control = Control.load_prms(
        DOMAIN_DIR / "nhm.control", warn_unused_options=False
    )
    control.options["verbosity"] = 0
    dis = Parameters.from_netcdf(
        DOMAIN_DIR / "parameters_dis_hru.nc", encoding=False
    )
    params = PrmsParameters.from_netcdf(DOMAIN_DIR / "parameters_PRMSSnow.nc")
    input_variables = {
        key: GEN_DIR / f"{key}.nc" for key in PywsSnow.get_inputs()
    }
    pyws = PywsSnow(
        control=control,
        discretization=dis,
        parameters=params,
        **input_variables,
        calc_method="numpy",
    )
    nhru = params.dims["nhru"]
    states = {nn: np.zeros((N_STEPS, nhru)) for nn in CHECK}
    for istep in range(N_STEPS):
        control.advance()
        pyws.advance()
        pyws.calculate(1.0)
        for nn in CHECK:
            states[nn][istep, :] = getattr(pyws, nn)
    pyws.finalize()
    return states


@pytest.fixture(scope="module")
def our_states():
    """Run the ported PRMSSnow for N_STEPS, capturing states."""
    params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSnow.nc")
    soltab = xr.load_dataarray(GEN_DIR / "soltab_horad_potsw.nc").rename(
        {"doy": "ndoy", "nhm_id": "nhru"}
    )
    params = xr.merge([params, soltab.to_dataset(name="soltab_horad_potsw")])
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
        for nn in PRMSSnow.get_inputs()
    }
    model = Model(
        {
            "prms_snow": {
                "class": PRMSSnow,
                "discretization": "nhru",
                "parameters": params,
                **forcings,
            }
        },
        {
            "output_var_names": ["pkwater_equiv"],
            "output_serial_zarr": (pl.Path(tempfile.mkdtemp()) / "ab.zarr"),
            "time_chunk_size": 61,
        },
        discretizations={
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            )
        },
    )
    proc = model.model_dict["prms_snow"]
    nhru = proc["pkwater_equiv"].values.shape[0]
    states = {nn: np.zeros((N_STEPS, nhru)) for nn in CHECK + TRIGGER_NAMES}
    orig_calculate = proc.calculate

    def capturing(dt, time):
        orig_calculate(dt, time)
        tt = time.current_index
        for nn in CHECK + TRIGGER_NAMES:
            states[nn][tt, :] = proc[nn].values

    proc.calculate = capturing
    with model:
        model.run(np.float64(1.0), np.int32(N_STEPS))
    return states


class TestPRMSSnowABNumpy:
    def test_bit_identical_except_departure(self, pyws_states, our_states):
        """Every compared state matches pywatershed-numpy EXACTLY on
        every hru the freeh2o departure cannot touch (rain on a cold
        pack holding free water); the touched hrus are few."""
        # start-of-step cold pack: pk_def at the END of the prior step
        pk_def_prev = np.zeros_like(our_states["pk_def"])
        pk_def_prev[1:] = our_states["pk_def"][:-1]
        trigger = (
            (our_states["freeh2o_prev"] > 0.0)
            & (pk_def_prev > 0.0)
            & (our_states["net_rain"] > 0.0)
        )
        affected = trigger.any(axis=0)
        assert affected.sum() <= MAX_AFFECTED_HRUS, (
            f"{affected.sum()} departure-affected hrus: "
            f"{np.nonzero(affected)[0].tolist()}"
        )
        for nn in CHECK:
            np.testing.assert_array_equal(
                our_states[nn][:, ~affected],
                pyws_states[nn][:, ~affected],
                err_msg=f"variable '{nn}' differs from pywatershed numpy "
                "on an hru the departure cannot touch",
            )
