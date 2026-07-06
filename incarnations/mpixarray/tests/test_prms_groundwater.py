"""Serial regression: the ported PRMSGroundwater vs pywatershed answers.

Runs the pywatershed drb_2yr domain (765 HRUs x 731 daily steps,
1979-1980) through the serial Model and compares every output variable
against pywatershed's generated answer files at pywatershed's OWN
autotest tolerance (rtol = atol = 1e-13; see
pywatershed/autotest/test_prms_groundwater.py).

Requires GENERATED pywatershed test data (the autotest data-generation
workflow populates test_data/drb_2yr/output/); skips with a clear
reason if absent. The pywatershed repo is expected at the mpix
meta-repo root (a sibling of pws_phoenix).
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_groundwater import PRMSGroundwater
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_NAMES = ("soil_to_gw", "ssr_to_gw", "dprst_seep_hru")
ANSWER_NAMES = (
    "gwres_stor",
    "gwres_flow",
    "gwres_sink",
    "gwres_stor_change",
    "gwres_flow_vol",
)
# pywatershed's own autotest comparison standard
RTOL = ATOL = 1.0e-13

_needed = [
    DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
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
    """PROCESS parameters only -- the dis_hru variables (hru_area,
    hru_in_to_cf) arrive via the Discretization (dis-first sourcing)."""
    return xr.open_dataset(DOMAIN_DIR / "parameters_PRMSGroundwater.nc")


@pytest.fixture(scope="module")
def answers():
    return {nn: xr.open_dataarray(GEN_DIR / f"{nn}.nc") for nn in ANSWER_NAMES}


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_gw_output")
    # pywatershed output files put forcings on the "nhm_id" dim; the
    # parameter files use "nhru" -- unify on the grid dim (assembly
    # rejects an input whose spatial dim is not the grid dim)
    forcings = {
        nn: xr.open_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
        for nn in INPUT_NAMES
    }
    process_dict = {
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
            "parameters": parameters,
            "gwstor_init": parameters["gwstor_init"],
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_groundwater.zarr",
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


class TestPRMSGroundwater:
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
        """Final in-memory state matches the last answer timestep."""
        proc = model_run["model"].model_dict["prms_groundwater"]
        for nn in ("gwres_stor", "gwres_flow"):
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
            )
