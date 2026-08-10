"""Serial regression: the NoDprst classes vs pywatershed answers.

Standalone parity for PRMSRunoffNoDprst / PRMSSoilzoneNoDprst /
PRMSGroundwaterNoDprst against pywatershed's nhm_no_dprst drb_2yr
simulation (nhm_no_dprst_{control,model}.yaml; dprst_flag: false),
whose generated answers land in test_data/drb_2yr/output_no_dprst/.
Inputs are fed from that same directory (the no-dprst chain's own
outputs); tolerances are each process's own standard (runoff 1e-10,
soilzone 1e-10 observed, groundwater 1e-13), matching the full-class
standalone tests.

pywatershed's yaml drives the NoDprst classes with the FULL parameter
files (extra dprst params ignored by its get_parameters()); here the
reduced parameters_PRMS*NoDprst.nc files are used -- identical values
for every declared parameter.

SKIPS until the no-dprst answers are generated (pywatershed test-data
generation for the nhm_no_dprst simulation). The additive-structure
consistency is already pinned bit-for-bit without external answers in
test_prms_no_dprst.py.
"""

import pathlib as pl
import sys
from typing import Any

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_groundwater import PRMSGroundwaterNoDprst
from hydrology.prms_runoff import PRMSRunoffNoDprst
from hydrology.prms_soilzone import PRMSSoilzoneNoDprst
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output_no_dprst"

CASES: dict[str, dict[str, Any]] = {
    "runoff": {
        "class": PRMSRunoffNoDprst,
        "param_file": "parameters_PRMSRunoffNoDprst.nc",
        "inputs": (
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
        ),
        "answers": (
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
        ),
        "statics": (),
        "tol": 1.0e-10,
    },
    "soilzone": {
        "class": PRMSSoilzoneNoDprst,
        "param_file": "parameters_PRMSSoilzoneNoDprst.nc",
        "inputs": (
            "hru_impervevap",
            "hru_intcpevap",
            "infil_hru",
            "potet",
            "transp_on",
            "snow_evap",
            "snowcov_area",
            # mutable inputs (dunnian added in place; zero on drb)
            "sroff",
            "sroff_vol",
        ),
        "answers": (
            "cap_infil_tot",
            "cap_waterin",
            "dunnian_flow",
            "hru_actet",
            "perv_actet",
            "potet_lower",
            "potet_rechr",
            "pref_flow",
            "pref_flow_in",
            "pref_flow_infil",
            "pref_flow_stor",
            "pref_flow_stor_change",
            "pref_flow_stor_prev",
            "recharge",
            "slow_flow",
            "slow_stor",
            "slow_stor_change",
            "slow_stor_prev",
            "soil_lower",
            "soil_lower_change",
            "soil_lower_prev",
            "soil_lower_ratio",
            "soil_moist",
            "soil_moist_tot",
            "soil_rechr",
            "soil_rechr_change",
            "soil_rechr_prev",
            "soil_to_gw",
            "soil_to_ssr",
            "ssr_to_gw",
            "ssres_flow",
            "ssres_flow_vol",
            "ssres_in",
            "ssres_stor",
            "swale_actet",
            "unused_potet",
        ),
        "statics": ("pref_flow_thrsh", "pref_flow_max"),
        "tol": 1.0e-10,
    },
    "groundwater": {
        "class": PRMSGroundwaterNoDprst,
        "param_file": "parameters_PRMSGroundwaterNoDprst.nc",
        "inputs": ("soil_to_gw", "ssr_to_gw"),
        "answers": (
            "gwres_stor",
            "gwres_flow",
            "gwres_sink",
            "gwres_stor_change",
            "gwres_flow_vol",
        ),
        "statics": (),
        "tol": 1.0e-13,
    },
}

_needed = [DOMAIN_DIR / cc["param_file"] for cc in CASES.values()] + [
    GEN_DIR / f"{nn}.nc"
    for cc in CASES.values()
    for nn in cc["inputs"] + cc["answers"] + cc["statics"]
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr nhm_no_dprst answers not generated "
        "(test_data/drb_2yr/output_no_dprst/); missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case_run(request, tmp_path_factory):
    """Build + run + finalize one NoDprst process Model."""
    case = CASES[request.param]
    out_dir = tmp_path_factory.mktemp(f"no_dprst_parity_{request.param}")
    parameters = xr.load_dataset(DOMAIN_DIR / case["param_file"])
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_id": "nhru"}
        )
        for nn in case["inputs"]
    }
    process_dict = {
        "proc": {
            "class": case["class"],
            "discretization": "nhru",
            "parameters": parameters,
            **forcings,
        },
    }
    if request.param == "groundwater":
        process_dict["proc"]["gwstor_init"] = parameters["gwstor_init"]
    control = {
        "output_var_names": list(case["answers"]),
        "output_serial_zarr": out_dir / "proc.zarr",
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
    answers = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        for nn in case["answers"] + case["statics"]
    }
    return {"case": case, "model": model, "control": control,
            "answers": answers}


class TestNoDprstParity:
    def test_all_variables_all_timesteps(self, case_run):
        """Every output variable matches pywatershed over the full run."""
        case = case_run["case"]
        output_ds = xr.load_dataset(
            case_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in case["answers"]:
            np.testing.assert_allclose(
                output_ds[nn].values,
                case_run["answers"][nn].values,
                rtol=case["tol"],
                atol=case["tol"],
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_static_init_quantities(self, case_run):
        """Init-computed statics match the answers' first timestep."""
        case = case_run["case"]
        proc = case_run["model"].model_dict["proc"]
        for nn in case["statics"]:
            np.testing.assert_allclose(
                proc[nn].values,
                case_run["answers"][nn].values[0, :],
                rtol=case["tol"],
                atol=case["tol"],
                err_msg=f"static quantity '{nn}' differs from pywatershed",
            )
