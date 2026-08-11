"""Serial regression: the ported PRMSRunoff vs pywatershed answers.

Runs the pywatershed drb_2yr domain (765 HRUs x 731 daily steps,
1979-1980) through the serial Model, feeding the 14 inputs from
pywatershed's generated files, and compares output variables against
pywatershed's answers at its OWN autotest tolerance (rtol = atol =
1e-10; see pywatershed/autotest/test_prms_runoff.py -- runoff is a
branchy threshold process, upstream does not hold it to the 1e-13 of
groundwater/channel).

Static (init-only) quantities -- the parameter_internal geometry plus
dprst_area_clos (never written back by the kernel; see the module
docstring) -- are validated once against the answer files' first
timestep rather than streamed.

Requires GENERATED pywatershed test data (test_data/drb_2yr/output/);
skips with a clear reason if absent. The pywatershed repo is expected
at the mpix meta-repo root.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_runoff import PRMSRunoff
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_NAMES = (
    "soil_lower_prev",
    "soil_rechr_prev",
    "net_rain",
    "net_ppt",
    "net_snow",
    "potet",
    "snowmelt",
    "snow_evap",
    "pkwater_equiv",
    "pptmix_nopack",
    "snowcov_area",
    "through_rain",
    "hru_intcpevap",
    "intcp_changeover",
)
# time-varying answers (pywatershed compares all variables; dprst_in
# has no answer file and dprst_vol_thres_open is excluded by upstream's
# own autotest -- it is parameter_internal here)
ANSWER_NAMES = (
    "contrib_fraction",
    "infil",
    "infil_hru",
    "sroff",
    "sroff_vol",
    "hru_sroffp",
    "hru_sroffi",
    "imperv_stor",
    "imperv_evap",
    "hru_impervevap",
    "hru_impervstor",
    "hru_impervstor_change",
    "dprst_vol_open",
    "dprst_vol_clos",
    "dprst_vol_open_frac",
    "dprst_vol_clos_frac",
    "dprst_vol_frac",
    "dprst_area_open",
    "dprst_sroff_hru",
    "dprst_seep_hru",
    "dprst_evap_hru",
    "dprst_insroff_hru",
    "dprst_stor_hru",
    "dprst_stor_hru_change",
)
# static after initialize(): validated once vs the answers' step 0
STATIC_ANSWER_NAMES = (
    "dprst_area_clos",
    "dprst_area_open_max",
    "dprst_area_clos_max",
)
# pywatershed's own autotest comparison standard for runoff
RTOL = ATOL = 1.0e-10

_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoff.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in INPUT_NAMES + ANSWER_NAMES + STATIC_ANSWER_NAMES
]
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
    """PROCESS parameters only -- the dis_hru variables (hru_type,
    hru_area, hru_in_to_cf) arrive via the Discretization."""
    return xr.load_dataset(DOMAIN_DIR / "parameters_PRMSRunoff.nc")


@pytest.fixture(scope="module")
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        for nn in ANSWER_NAMES + STATIC_ANSWER_NAMES
    }


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_runoff_output")
    # pywatershed output files put forcings on the "nhm_id" dim; the
    # parameter files use "nhru" -- unify on the grid dim
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
        for nn in INPUT_NAMES
    }
    process_dict = {
        "prms_runoff": {
            "class": PRMSRunoff,
            "discretization": "nhru",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_runoff.zarr",
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


class TestPRMSRunoff:
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

    def test_static_init_quantities(self, model_run, answers):
        """Init-computed statics match the answers' first timestep."""
        proc = model_run["model"].model_dict["prms_runoff"]
        for nn in STATIC_ANSWER_NAMES:
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[0, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"static quantity '{nn}' differs from pywatershed",
            )

    def test_final_state(self, model_run, answers):
        """Final in-memory state matches the last answer timestep."""
        proc = model_run["model"].model_dict["prms_runoff"]
        for nn in ("sroff", "sroff_vol", "dprst_vol_open", "imperv_stor"):
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
            )
