"""Serial regression: the ported PRMSSoilzone vs pywatershed answers.

Runs the pywatershed drb_2yr domain (765 HRUs x 731 daily steps)
through the serial Model, feeding the 11 inputs from pywatershed's
generated files, and compares output variables against pywatershed's
answers at 1e-10 (observed; upstream's own autotest standard is a
much looser 5e-6 -- see the RTOL comment below and
pywatershed/autotest/test_prms_soilzone.py).

sroff / sroff_vol are MUTABLE inputs (dunnian flow added in place);
comparing them here would be tautological (upstream's own comment) --
and on drb dunnian is identically zero (sat_threshold >= 999).

Static (init-only) quantities -- pref_flow_thrsh / pref_flow_max
(parameter_derived here, upstream "variables" its kernel never
writes) -- are validated once against the answers' first timestep.

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
from hydrology.prms_soilzone import PRMSSoilzone
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_NAMES = (
    "dprst_evap_hru",
    "dprst_seep_hru",
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
)
# every variable with an answer file; excluded: the three *_hru with
# no generated files (upstream also excludes), sroff/sroff_vol
# (tautology, see docstring)
ANSWER_NAMES = (
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
)
# static after initialize(): validated once vs the answers' step 0
STATIC_ANSWER_NAMES = ("pref_flow_thrsh", "pref_flow_max")
# pywatershed's own soilzone autotest standard is 5e-6 (its fastmath
# numba path vs its numpy path only agree to that); against the
# generated answers drb holds 1e-10 -- pin the tighter observed level
# (relax toward 5e-6 if another platform's libm ulps ever bite)
RTOL = ATOL = 1.0e-10

_needed = [
    DOMAIN_DIR / "parameters_PRMSSoilzone.nc",
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
    return xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSoilzone.nc")


@pytest.fixture(scope="module")
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        for nn in ANSWER_NAMES + STATIC_ANSWER_NAMES
    }


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_soilzone_output")
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        .rename({"nhm_id": "nhru"})
        for nn in INPUT_NAMES
    }
    process_dict = {
        "prms_soilzone": {
            "class": PRMSSoilzone,
            "discretization": "nhru",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_soilzone.zarr",
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


class TestPRMSSoilzone:
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
        proc = model_run["model"].model_dict["prms_soilzone"]
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
        proc = model_run["model"].model_dict["prms_soilzone"]
        for nn in ("soil_moist", "slow_stor", "ssres_flow", "soil_to_gw"):
            np.testing.assert_allclose(
                proc[nn].values,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
            )

    def test_dunnian_zero_on_drb(self, model_run):
        """drb: sat_threshold >= 999 -> dunnian identically zero (the
        precondition for runoff's sroff being unmodified here)."""
        proc = model_run["model"].model_dict["prms_soilzone"]
        assert (proc["dunnian_flow"].values == 0.0).all()
