"""Serial regression: the ported PRMSSoilzoneAg vs pywatershed answers.

Runs the pywatershed fgr_ag_2yr domain (612 HRUs, 2 years daily,
spinup configuration: static ag_frac, dprst ACTIVE, NO obs-ET
iteration) through the serial Model and compares against pywatershed's
answers at its OWN ag autotest standard (rtol = atol = 1e-5 plus its
per-variable exception dict -- the answers are converted GSFLOW
Fortran output, partly single precision; see
pywatershed/autotest/test_prms_soilzone_ag.py).

Masking mirrors upstream's EFFECTIVE behavior: the listed
pervious-zone variables are compared only on NON-ag HRUs (upstream's
mask_dict; note its dangling-else makes the ag-HRU mask inert, so
everything else -- ag_* included -- is compared on all HRUs).

sroff / sroff_vol are MUTABLE inputs (dunnian added in place) fed from
disk -- comparing them here would be tautological.

Requires the fgr_ag_2yr domain with GENERATED answers (output_spinup/);
skips with a clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_soilzone_ag import PRMSSoilzoneAg
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "fgr_ag_2yr"
GEN_DIR = DOMAIN_DIR / "output_spinup"

DISK_INPUT_NAMES = (
    "dprst_evap_hru",
    "dprst_seep_hru",
    "hru_impervevap",
    "hru_intcpevap",
    "infil",
    "infil_ag",
    "potet",
    "transp_on",
    "snow_evap",
    "snowcov_area",
    # mutable inputs (dunnian added in place)
    "sroff",
    "sroff_vol",
)
ANSWER_NAMES = (
    "ag_actet",
    "ag_hortonian",
    "ag_potet_lower",
    "ag_potet_rechr",
    "ag_soil_lower",
    "ag_soil_moist",
    "ag_soil_moist_change",
    "ag_soil_moist_prev",
    "ag_soil_rechr",
    "ag_soil_rechr_change",
    "ag_soil_rechr_prev",
    "ag_soil_saturated",
    "ag_soil_to_gvr",
    "ag_soil_to_gw",
    "hru_ag_actet",
    "unused_ag_et",
    "cap_infil_tot",
    "cap_waterin",
    "dunnian_flow",
    "hru_actet",
    "perv_actet",
    "perv_soil_to_gvr",
    "perv_soil_to_gw",
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
    "soil_saturated",
    "soil_to_gw",
    "soil_to_ssr",
    "ssr_to_gw",
    "ssres_flow",
    "ssres_flow_vol",
    "ssres_in",
    "ssres_stor",
    "swale_actet",
    "unused_potet",
)
# statics (parameter_internal here; upstream variables never written)
STATIC_ANSWER_NAMES = ("pref_flow_thrsh", "pref_flow_max")
# compared only on NON-ag HRUs (upstream mask_dict)
NOT_AG_ONLY_NAMES = (
    "soil_moist",
    "soil_rechr",
    "soil_lower",
    "soil_moist_tot",
    "soil_rechr_change",
    "soil_lower_change",
    "perv_actet",
    "potet_rechr",
    "potet_lower",
    "cap_infil_tot",
    "cap_waterin",
)
# pywatershed's own ag autotest standard + its exception dict
RTOL = ATOL = 1.0e-5
PER_VAR_TOL = {  # (rtol, atol)
    "ssres_flow_vol": (1.0e-2, 2.0),
    "soil_lower_ratio": (1.0e-5, 1.0e-4),
    "slow_flow": (1.0e-5, 2.0e-5),
    "ssres_flow": (1.0e-5, 2.0e-5),
    "slow_stor": (1.0e-5, 1.0e-4),
    "slow_stor_prev": (1.0e-5, 1.0e-4),
    "ssres_stor": (1.0e-5, 1.0e-4),
    "soil_moist_tot": (1.0e-5, 1.0e-4),
    "recharge": (1.0e-5, 2.0e-5),
    "slow_stor_change": (1.0e-5, 2.0e-5),
    "ssr_to_gw": (1.0e-5, 2.0e-5),
    "soil_lower_change": (1.0e-5, 2.0e-5),
    "soil_to_gw": (1.0e-5, 2.0e-5),
    "perv_soil_to_gw": (1.0e-5, 2.0e-5),
    "ag_soil_moist_change": (1.0e-5, 2.0e-5),
}

_needed = [
    DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc",
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
    return xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc")


@pytest.fixture(scope="module")
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        for nn in ANSWER_NAMES + STATIC_ANSWER_NAMES
    }


@pytest.fixture(scope="module")
def not_ag_idx():
    ag_static = xr.load_dataarray(DOMAIN_DIR / "ag_frac_static.nc")
    return np.where(ag_static.values <= 0.0)[0]


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_soilzone_ag_output")
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
        for nn in DISK_INPUT_NAMES
    }
    # spinup: STATIC ag_frac, constant in time
    template = forcings["potet"]
    ag_static = xr.load_dataarray(DOMAIN_DIR / "ag_frac_static.nc")
    ag_frac = xr.DataArray(
        np.tile(ag_static.values, (template.sizes["time"], 1)),
        dims=("time", "nhru"),
        coords={"time": template["time"], "nhru": template["nhru"]},
        name="ag_frac",
    )
    process_dict = {
        "prms_soilzone_ag": {
            "class": PRMSSoilzoneAg,
            "discretization": "nhru",
            "parameters": parameters,
            "ag_frac": ag_frac,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_soilzone_ag.zarr",
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


class TestPRMSSoilzoneAg:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(self, model_run, answers, not_ag_idx):
        """Every output variable matches pywatershed over the full run."""
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in ANSWER_NAMES:
            rtol, atol = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            actual = output_ds[nn].values
            desired = answers[nn].values
            if nn in NOT_AG_ONLY_NAMES:
                actual = actual[:, not_ag_idx]
                desired = desired[:, not_ag_idx]
            np.testing.assert_allclose(
                actual,
                desired,
                rtol=rtol,
                atol=atol,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_static_init_quantities(self, model_run, answers):
        """Init-computed statics match the answers' first timestep."""
        proc = model_run["model"].model_dict["prms_soilzone_ag"]
        for nn in STATIC_ANSWER_NAMES:
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[0, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"static quantity '{nn}' differs from pywatershed",
            )

    def test_final_state(self, model_run, answers, not_ag_idx):
        """Final in-memory state matches the last answer timestep."""
        proc = model_run["model"].model_dict["prms_soilzone_ag"]
        for nn in ("ag_soil_moist", "slow_stor", "ssres_flow", "soil_to_gw"):
            rtol, atol = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            actual = proc[nn].values
            desired = answers[nn].values[-1, :]
            if nn in NOT_AG_ONLY_NAMES:
                actual = actual[not_ag_idx]
                desired = desired[not_ag_idx]
            np.testing.assert_allclose(
                actual,
                desired,
                rtol=rtol,
                atol=atol,
                err_msg=f"final '{nn}' differs",
            )
