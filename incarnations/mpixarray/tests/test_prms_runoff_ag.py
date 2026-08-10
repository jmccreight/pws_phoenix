"""Serial regression: the ported PRMSRunoffAg vs pywatershed answers.

Runs the pywatershed fgr_ag_2yr domain (612 HRUs, 2 years daily,
spinup configuration: static ag_frac, dprst ACTIVE) through the serial
Model, feeding the 16 disk inputs from pywatershed's generated files
plus a constant-in-time ag_frac built from ag_frac_static.nc, and
compares output variables against pywatershed's answers at its OWN ag
autotest standard (rtol = atol = 1e-5 with the sroff_vol /
dprst_vol_open exceptions -- see pywatershed/autotest/
test_prms_runoff_ag.py; the answers are converted GSFLOW Fortran
output, partly single precision).

Requires the fgr_ag_2yr domain (symlinked into test_data/ from
pywatershed_addtl_domains) with GENERATED answers (output_spinup/);
skips with a clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_runoff import PRMSRunoffAg
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "fgr_ag_2yr"
GEN_DIR = DOMAIN_DIR / "output_spinup"

DISK_INPUT_NAMES = (
    "soil_lower_prev",
    "soil_rechr_prev",
    "ag_soil_moist_prev",
    "ag_soil_rechr_prev",
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
# every variable with an answer file (no hru_sroff_ag / infil_perv_hru
# files are generated upstream; sroff_vol excluded like upstream's own
# test: single-precision Fortran errors scale with hru_area, rely on
# sroff)
ANSWER_NAMES = (
    "contrib_fraction",
    "infil",
    "infil_ag",
    "infil_hru",
    "sroff",
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
# static after init (dprst_area_clos: never written back -- see the
# prms_runoff module docstring)
STATIC_ANSWER_NAMES = (
    "dprst_area_clos",
    "dprst_area_open_max",
    "dprst_area_clos_max",
)
# pywatershed's own ag autotest standard (Fortran answers)
RTOL = ATOL = 1.0e-5
PER_VAR_TOL = {
    "dprst_vol_open": (3.0e-4, 3.0e-4),
}

_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoffAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "ag_frac_static.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in DISK_INPUT_NAMES + ANSWER_NAMES + STATIC_ANSWER_NAMES
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed fgr_ag_2yr test data not present/generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def parameters():
    return xr.load_dataset(DOMAIN_DIR / "parameters_PRMSRunoffAg.nc")


@pytest.fixture(scope="module")
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        for nn in ANSWER_NAMES + STATIC_ANSWER_NAMES
    }


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_runoff_ag_output")
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_id": "nhru"}
        )
        for nn in DISK_INPUT_NAMES
    }
    # spinup: STATIC ag_frac (no dyn_ag_frac_flag in its control) --
    # constant in time from the domain's 1-D ag_frac_static.nc
    template = forcings["potet"]
    ag_static = xr.load_dataarray(DOMAIN_DIR / "ag_frac_static.nc")
    ag_frac = xr.DataArray(
        np.tile(ag_static.values, (template.sizes["time"], 1)),
        dims=("time", "nhru"),
        coords={"time": template["time"], "nhru": template["nhru"]},
        name="ag_frac",
    )
    process_dict = {
        "prms_runoff_ag": {
            "class": PRMSRunoffAg,
            "discretization": "nhru",
            "parameters": parameters,
            "ag_frac": ag_frac,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_runoff_ag.zarr",
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


class TestPRMSRunoffAg:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(self, model_run, answers):
        """Every output variable matches pywatershed over the full run."""
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in ANSWER_NAMES:
            rtol, atol = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            np.testing.assert_allclose(
                output_ds[nn].values,
                answers[nn].values,
                rtol=rtol,
                atol=atol,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_static_init_quantities(self, model_run, answers):
        """Init-computed statics match the answers' first timestep."""
        proc = model_run["model"].model_dict["prms_runoff_ag"]
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
        proc = model_run["model"].model_dict["prms_runoff_ag"]
        for nn in ("sroff", "dprst_vol_open", "infil_ag"):
            rtol, atol = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=rtol,
                atol=atol,
            )
