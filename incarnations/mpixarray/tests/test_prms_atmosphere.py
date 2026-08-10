"""Serial regression: the ported PRMSAtmosphere vs pywatershed answers.

Runs the pywatershed drb_2yr domain (765 HRUs x 731 daily steps)
through the serial Model, feeding the CBH inputs (prcp/tmax/tmin --
float32 files, widened exactly to f64) and the GENERATED soltab tables
(isolating this test from the solar-geometry factory, which has its
own test), and compares output variables against pywatershed's answers
at its OWN autotest tolerance (rtol = atol = 1e-5; see
pywatershed/autotest/test_prms_atmosphere.py -- its own comment asks
"why is this relatively low accuracy?"; the port is per-step where
upstream is all-time vectorized, so start there and note the observed
level).

tmax_sum and transp_check (state) have no answer files; pptmix is
compared as-is (the file may carry canopy's in-place edits, which do
not fire on drb -- see test_prms_canopy.py).

Requires GENERATED pywatershed test data; skips with a reason if
absent. The pywatershed repo is expected at the mpix meta-repo root.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from atmosphere.prms_atmosphere import PRMSAtmosphere
from discretization import Discretization
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

CBH_NAMES = ("prcp", "tmax", "tmin")
ANSWER_NAMES = (
    "tmaxf",
    "tminf",
    "tmaxc",
    "tminc",
    "tavgc",
    "prmx",
    "hru_ppt",
    "hru_rain",
    "hru_snow",
    "pptmix",
    "swrad",
    "orad_hru",
    "potet",
    "transp_on",
)
# pywatershed's own atmosphere autotest comparison standard
RTOL = ATOL = 1.0e-5

_needed = (
    [
        DOMAIN_DIR / "parameters_PRMSAtmosphere.nc",
        DOMAIN_DIR / "parameters_dis_hru.nc",
        GEN_DIR / "soltab_potsw.nc",
        GEN_DIR / "soltab_horad_potsw.nc",
    ]
    + [DOMAIN_DIR / f"{nn}.nc" for nn in CBH_NAMES]
    + [GEN_DIR / f"{nn}.nc" for nn in ANSWER_NAMES]
)
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def parameters():
    """Process parameters + the generated soltab tables (dims renamed
    to the declared (ndoy, nhru)); hru_slope/hru_lat arrive via the
    dis."""
    params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSAtmosphere.nc")
    soltabs = [
        xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        .rename({"doy": "ndoy", "nhm_id": "nhru"})
        .to_dataset(name=nn)
        for nn in ("soltab_potsw", "soltab_horad_potsw")
    ]
    return xr.merge([params, *soltabs])


@pytest.fixture(scope="module")
def answers():
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in ANSWER_NAMES}


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_atmosphere_output")
    # CBH files: (time, nhm_id) float32 -> the grid dim, f64 (exact)
    forcings = {
        nn: xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
        .rename({"nhm_id": "nhru"})
        .astype(np.float64)
        for nn in CBH_NAMES
    }
    process_dict = {
        "prms_atmosphere": {
            "class": PRMSAtmosphere,
            "discretization": "nhru",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        # ccov_hru has no upstream answer file (see its pin test)
        "output_var_names": list(ANSWER_NAMES) + ["ccov_hru"],
        "output_serial_zarr": out_dir / "prms_atmosphere.zarr",
        "time_chunk_size": 61,
    }
    discretizations = {
        "nhru": Discretization(
            ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
        ),
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(np.float64(1.0), np.int32(model.ntime))
    return {"model": model, "control": control}


class TestPRMSAtmosphere:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(self, model_run, answers):
        """Every output variable matches pywatershed over the full run."""
        output_ds = xr.open_zarr(
            model_run["control"]["output_serial_zarr"], consolidated=False
        )
        for nn in ANSWER_NAMES:
            np.testing.assert_allclose(
                output_ds[nn].values,
                answers[nn].values,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_ccov_hru_pin(self, model_run, parameters):
        """ccov_hru matches the verbatim upstream block applied to the
        model's OWN swrad, exactly. No upstream HRU-level answer file
        exists (upstream computes cloud cover inline in stream temp's
        hru->segment aggregation; it was relocated here so Maps never
        originate variables); the segment-level seg_ccov parity rides
        the stream-temp aggregation/chain tests."""
        output_ds = xr.open_zarr(
            model_run["control"]["output_serial_zarr"], consolidated=False
        )
        swrad = output_ds["swrad"].values
        actual = output_ds["ccov_hru"].values
        soltab = parameters["soltab_potsw"].values
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        cossl = np.cos(np.arctan(dis_hru["hru_slope"].values))
        times = output_ds["time"].values.astype("datetime64[D]")
        doys = (
            times - times.astype("datetime64[Y]").astype("datetime64[D]")
        ).astype(int) + 1
        potsw = soltab[doys - 1, :]
        expected = np.where(
            potsw <= 10.0,
            1.0 - swrad / 10.0 * cossl,
            1.0 - swrad / potsw * cossl,
        )
        expected = np.where(expected < 1.0e-6, 0.0, np.minimum(expected, 1.0))
        np.testing.assert_array_equal(actual, expected)

    def test_final_state(self, model_run, answers):
        """Final in-memory state matches the last answer timestep."""
        proc = model_run["model"].model_dict["prms_atmosphere"]
        for nn in ("hru_ppt", "swrad", "potet", "transp_on"):
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' final state differs",
            )
