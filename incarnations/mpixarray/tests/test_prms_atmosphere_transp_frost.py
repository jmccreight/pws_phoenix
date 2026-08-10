"""Serial regression: PRMSAtmosphereTranspFrost vs pywatershed.

Runs the pywatershed ucb_2yr domain through the serial Model with the
FROST-WINDOW transpiration leaf (transp_on = spring_frost <= jsol <=
fall_frost, jsol = solar day of year) and compares against the
nhm_transp_frost answers at pywatershed's own atmosphere standard
(rtol = atol = 1e-5); transp_on is compared EXACTLY (a 0/1 window on
integer-valued solar days).

spring_frost / fall_frost live in transp_frost.param -- a PARTIAL
PRMS parameter file (the control's param_file lists it alongside
myparam.param), parsed here directly (PrmsParameters.load expects the
full-file header; there is no parameters_PRMSAtmosphereTranspFrost.nc
in ucb_2yr).

Requires the ucb_2yr domain with GENERATED nhm_transp_frost answers
(output_transp_frost/) and the pywatershed repo importable at the
mpix root; skips with a clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from atmosphere.prms_atmosphere import PRMSAtmosphereTranspFrost
from discretization import Discretization
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
PYWS_ROOT = MPIX_ROOT / "pywatershed"
DOMAIN_DIR = PYWS_ROOT / "test_data" / "ucb_2yr"
GEN_DIR = DOMAIN_DIR / "output_transp_frost"

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
EXACT_NAMES = ("transp_on",)

_needed = (
    [
        DOMAIN_DIR / "parameters_PRMSAtmosphere.nc",
        DOMAIN_DIR / "parameters_dis_hru.nc",
        DOMAIN_DIR / "transp_frost.param",
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
        "pywatershed ucb_2yr nhm_transp_frost data not generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


def _read_partial_param(path: pl.Path, name: str) -> np.ndarray:
    """Read one parameter from a PRMS partial parameter file
    (#### / name / ndims / dim names / size / type / values)."""
    lines = path.read_text().splitlines()
    ii = lines.index(name)
    ndims = int(lines[ii + 1])
    size = int(lines[ii + 2 + ndims])
    vals = lines[ii + 4 + ndims : ii + 4 + ndims + size]
    return np.array([float(vv) for vv in vals], dtype=np.float64)


@pytest.fixture(scope="module")
def parameters():
    """Process parameters + generated soltabs + the frost-window
    parameters from the partial PRMS parameter file."""
    params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSAtmosphere.nc")
    soltabs = [
        xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        .rename({"doy": "ndoy", "nhm_id": "nhru"})
        .to_dataset(name=nn)
        for nn in ("soltab_potsw", "soltab_horad_potsw")
    ]
    frost_ds = xr.Dataset(
        {
            nn: (
                ("nhru",),
                _read_partial_param(
                    DOMAIN_DIR / "transp_frost.param", nn
                ),
            )
            for nn in ("spring_frost", "fall_frost")
        }
    )
    return xr.merge([params, *soltabs, frost_ds])


@pytest.fixture(scope="module")
def answers():
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in ANSWER_NAMES}


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_atmos_transp_frost_output")
    # CBH files: (time, nhm_id) float32 -> the grid dim, f64 (exact)
    forcings = {
        nn: xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
        .rename({"nhm_id": "nhru"})
        .astype(np.float64)
        for nn in CBH_NAMES
    }
    process_dict = {
        "prms_atmosphere": {
            "class": PRMSAtmosphereTranspFrost,
            "discretization": "nhru",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_atmos_transp_frost.zarr",
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


class TestPRMSAtmosphereTranspFrost:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(self, model_run, answers):
        """Every output variable matches pywatershed over the full run;
        the frost-window transp_on matches EXACTLY."""
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in ANSWER_NAMES:
            if nn in EXACT_NAMES:
                np.testing.assert_array_equal(
                    output_ds[nn].values,
                    answers[nn].values,
                    err_msg=f"variable '{nn}' differs from pywatershed",
                )
            else:
                np.testing.assert_allclose(
                    output_ds[nn].values,
                    answers[nn].values,
                    rtol=RTOL,
                    atol=ATOL,
                    err_msg=f"variable '{nn}' differs from pywatershed",
                )

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
