"""Serial regression: the ported PRMSSoilzoneAgObsET vs pywatershed.

Runs the fgr_ag_2yr ANALYSIS configuration (obs-AET iteration ON,
DYNAMIC ag_frac from dyn_ag_frac.param) through the serial Model and
compares against pywatershed's generated answers at its OWN ag
autotest standard (1e-5 + the per-variable exception dict; GSFLOW
Fortran answers). This exercises, on top of the PRMSSoilzoneAg core:
the It0 iteration loop with irrigation additions, the AET_external
validation, and the ag_frac-change storage redistribution in
_update_areas (annual Jan-1 changes in the dynamic parameter file).

ag_frac is built test-side by FORWARD-FILL of the PRMS dynamic
parameter file onto the forcing time axis (pywatershed's
AdapterDynamicParameter semantics: most recent file date <= current
date), using pywatershed's own reader.

Requires the fgr_ag_2yr domain with GENERATED answers
(output_analysis/) and the pywatershed repo importable at the mpix
root; skips with a clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_soilzone_ag import PRMSSoilzoneAgObsET
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
PYWS_ROOT = MPIX_ROOT / "pywatershed"
DOMAIN_DIR = PYWS_ROOT / "test_data" / "fgr_ag_2yr"
GEN_DIR = DOMAIN_DIR / "output_analysis"

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
    # mutable inputs (dunnian added in place; zero on fgr)
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
    "ag_soil_moist_prev",
    "ag_soil_rechr",
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
    "slow_stor_prev",
    "soil_lower",
    "soil_lower_prev",
    "soil_lower_ratio",
    "soil_moist",
    "soil_moist_tot",
    "soil_rechr",
    "soil_rechr_prev",
    # NOT compared (upstream's own exclusion): the five
    # redistribution-corrected change vars (ag_soil_moist_change,
    # ag_soil_rechr_change, slow_stor_change, soil_lower_change,
    # soil_rechr_change) -- deliberately differ from Fortran
    # postprocessing on dynamic-ag_frac change dates; the mass budget
    # validates them upstream. The static-ag spinup test compares
    # them (redistributions are zero there).
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
    # obs-ET iteration outputs
    "AET_external",
    "ag_irrigation_add",
    "ag_irrigation_add_vol",
    "ag_soilwater_deficit",
)
STATIC_ANSWER_NAMES = ("pref_flow_thrsh", "pref_flow_max")
NOT_AG_ONLY_NAMES = (
    "soil_moist",
    "soil_rechr",
    "soil_lower",
    "soil_moist_tot",
    "perv_actet",
    "potet_rechr",
    "potet_lower",
    "cap_infil_tot",
    "cap_waterin",
)
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
    DOMAIN_DIR / "dyn_ag_frac.param",
    DOMAIN_DIR / "aet_observed.nc",
    PYWS_ROOT / "pywatershed" / "utils" / "prms_dyn_param.py",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in DISK_INPUT_NAMES + ANSWER_NAMES + STATIC_ANSWER_NAMES
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed fgr_ag_2yr analysis data not present/generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


def _dynamic_ag_frac(template: xr.DataArray) -> xr.DataArray:
    """Forward-fill the PRMS dynamic parameter file onto the forcing
    time axis (AdapterDynamicParameter semantics)."""
    sys.path.insert(0, str(PYWS_ROOT))
    from pywatershed.utils.prms_dyn_param import PrmsDynamicParameter

    dp = PrmsDynamicParameter.load(
        DOMAIN_DIR / "dyn_ag_frac.param", dtype="float"
    )
    file_dates = np.array(
        [
            np.datetime64(f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}")
            for yy, mm, dd in dp.dates
        ]
    )
    times = template["time"].values.astype("datetime64[D]")
    idx = np.searchsorted(file_dates, times, side="right") - 1
    idx = np.clip(idx, 0, len(file_dates) - 1)
    return xr.DataArray(
        dp.data[idx, :],
        dims=("time", "nhru"),
        coords={"time": template["time"], "nhru": template["nhru"]},
        name="ag_frac",
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
def ag_frac_da(model_run):
    return model_run["ag_frac"]


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_soilzone_ag_obs_output")
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_id": "nhru"}
        )
        for nn in DISK_INPUT_NAMES
    }
    template = forcings["potet"]
    ag_frac = _dynamic_ag_frac(template)
    aet_observed = (
        xr.load_dataarray(DOMAIN_DIR / "aet_observed.nc")
        .sel(time=template["time"])
        .assign_coords(nhru=template["nhru"])
    )
    process_dict = {
        "prms_soilzone_ag_obs": {
            "class": PRMSSoilzoneAgObsET,
            "discretization": "nhru",
            "parameters": parameters,
            "ag_frac": ag_frac,
            "aet_observed": aet_observed,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_soilzone_ag_obs.zarr",
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
    return {"model": model, "control": control, "ag_frac": ag_frac}


class TestPRMSSoilzoneAgObsET:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(
        self, model_run, answers, ag_frac_da
    ):
        """Every output variable matches pywatershed over the full run.

        The not-ag mask is per-TIMESTEP here (dynamic ag_frac)."""
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        not_ag = ag_frac_da.values <= 0.0  # (time, nhru) mask
        for nn in ANSWER_NAMES:
            rtol, atol = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            actual = output_ds[nn].values
            desired = answers[nn].values
            if nn in NOT_AG_ONLY_NAMES:
                actual = np.where(not_ag, actual, 0.0)
                desired = np.where(not_ag, desired, 0.0)
            np.testing.assert_allclose(
                actual,
                desired,
                rtol=rtol,
                atol=atol,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_static_init_quantities(self, model_run, answers):
        proc = model_run["model"].model_dict["prms_soilzone_ag_obs"]
        for nn in STATIC_ANSWER_NAMES:
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[0, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"static quantity '{nn}' differs from pywatershed",
            )

    def test_iteration_ran(self, model_run):
        """The obs-AET iteration actually iterated at least once."""
        proc = model_run["model"].model_dict["prms_soilzone_ag_obs"]
        assert (proc["ag_irrigation_add"].values >= 0.0).all()
        assert proc["iter_count"].values.max() >= 1
