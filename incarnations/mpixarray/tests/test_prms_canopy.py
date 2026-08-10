"""Serial regression: the ported PRMSCanopy vs pywatershed answers.

Runs the pywatershed drb_2yr domain (765 HRUs x 731 daily steps)
through the serial Model, feeding the 8 inputs from pywatershed's
generated files (pptmix is a MUTABLE input -- canopy zeroes it in
place but never reads it, so the post-edit answer file is an exact
input), and compares output variables against pywatershed's answers
at its OWN autotest tolerance (rtol = atol = 1e-12; see
pywatershed/autotest/test_prms_canopy.py).

intcp_transp_on has no generated answer file (excluded); pptmix is
compared as an output (input + recomputed edits).

Requires GENERATED pywatershed test data; skips with a reason if
absent. The pywatershed repo is expected at the mpix meta-repo root.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_canopy import PRMSCanopy
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_NAMES = (
    "pk_ice_prev",
    "freeh2o_prev",
    "transp_on",
    "hru_ppt",
    "hru_rain",
    "hru_snow",
    "potet",
    # mutable input (zeroed in place where trace snow becomes rain)
    "pptmix",
)
ANSWER_NAMES = (
    "net_ppt",
    "net_rain",
    "net_snow",
    "intcp_changeover",
    "intcp_evap",
    "intcp_form",
    "intcp_stor",
    "hru_intcpevap",
    "hru_intcpstor",
    "hru_intcpstor_change",
    "hru_intcpstor_old",
)
# pywatershed's own canopy autotest comparison standard
RTOL = ATOL = 1.0e-12

_needed = [
    DOMAIN_DIR / "parameters_PRMSCanopy.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in INPUT_NAMES + ANSWER_NAMES]
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
    return xr.load_dataset(DOMAIN_DIR / "parameters_PRMSCanopy.nc")


@pytest.fixture(scope="module")
def answers():
    names = ANSWER_NAMES + ("pptmix",)
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in names}


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_canopy_output")
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        .rename({"nhm_id": "nhru"})
        for nn in INPUT_NAMES
    }
    process_dict = {
        "prms_canopy": {
            "class": PRMSCanopy,
            "discretization": "nhru",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_canopy.zarr",
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


class TestPRMSCanopy:
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

    def test_final_state(self, model_run, answers):
        """Final in-memory state matches the last answer timestep --
        including the MUTABLE pptmix (input + recomputed edits)."""
        proc = model_run["model"].model_dict["prms_canopy"]
        for nn in ("intcp_stor", "hru_intcpstor", "net_rain", "pptmix"):
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' final state differs",
            )
