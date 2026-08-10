"""Serial regression: LIVE PRMSRunoffAg -> PRMSSoilzoneAg chain.

The two ag processes coupled on one hru grid (fgr_ag_2yr spinup
configuration), mirroring pywatershed's own runoff+soilzone ag model
test: runoff_ag produces infil / infil_ag / dprst_evap_hru /
dprst_seep_hru / hru_impervevap consumed by soilzone_ag via structural
sharing; soilzone_ag's soil_lower_prev / soil_rechr_prev /
ag_soil_moist_prev / ag_soil_rechr_prev feed BACK to runoff_ag
(prior-step-correct: all advance() hooks run before any calculate());
sroff / sroff_vol are the mutable chain (dunnian added in place -- zero
on fgr). Only the atmosphere/canopy/snow products, transp_on, and
ag_frac come from disk.

Both processes' outputs are validated against the spinup answers at
upstream's ag standard (1e-5 + per-variable exceptions; sroff_vol
excluded as upstream does).
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_runoff import PRMSRunoffAg
from hydrology.prms_soilzone_ag import PRMSSoilzoneAg
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "fgr_ag_2yr"
GEN_DIR = DOMAIN_DIR / "output_spinup"

# runoff_ag disk inputs; its other 4 declared inputs (soil_lower_prev,
# soil_rechr_prev, ag_soil_moist_prev, ag_soil_rechr_prev) are LIVE
# soilzone_ag back-edges
RUNOFF_DISK_INPUTS = (
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
# soilzone_ag's ONLY disk input; everything else is live from
# runoff_ag (dprst_evap_hru, dprst_seep_hru, hru_impervevap, infil,
# infil_ag, sroff, sroff_vol) or shared with runoff_ag's feed
SOILZONE_DISK_INPUTS = ("transp_on",)

RUNOFF_ANSWER_NAMES = (
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
SOILZONE_ANSWER_NAMES = (
    "ag_actet",
    "ag_soil_lower",
    "ag_soil_moist",
    "ag_soil_moist_change",
    "ag_soil_moist_prev",
    "ag_soil_rechr",
    "ag_soil_rechr_change",
    "ag_soil_rechr_prev",
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
    "pref_flow_stor",
    "recharge",
    "slow_flow",
    "slow_stor",
    "slow_stor_change",
    "soil_lower",
    "soil_lower_ratio",
    "soil_moist",
    "soil_moist_tot",
    "soil_rechr",
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
RTOL = ATOL = 1.0e-5
PER_VAR_TOL = {  # (rtol, atol)
    "dprst_vol_open": (3.0e-4, 3.0e-4),
    "ssres_flow_vol": (1.0e-2, 2.0),
    "soil_lower_ratio": (1.0e-5, 1.0e-4),
    "slow_flow": (1.0e-5, 2.0e-5),
    "ssres_flow": (1.0e-5, 2.0e-5),
    "slow_stor": (1.0e-5, 1.0e-4),
    "ssres_stor": (1.0e-5, 1.0e-4),
    "soil_moist_tot": (1.0e-5, 1.0e-4),
    "recharge": (1.0e-5, 2.0e-5),
    "slow_stor_change": (1.0e-5, 2.0e-5),
    "ssr_to_gw": (1.0e-5, 2.0e-5),
    "soil_to_gw": (1.0e-5, 2.0e-5),
    "perv_soil_to_gw": (1.0e-5, 2.0e-5),
    "ag_soil_moist_change": (1.0e-5, 2.0e-5),
}

_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoffAg.nc",
    DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "ag_frac_static.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in RUNOFF_DISK_INPUTS
    + SOILZONE_DISK_INPUTS
    + RUNOFF_ANSWER_NAMES
    + SOILZONE_ANSWER_NAMES
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
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc")
        for nn in set(RUNOFF_ANSWER_NAMES + SOILZONE_ANSWER_NAMES)
    }


@pytest.fixture(scope="module")
def not_ag_idx():
    ag_static = xr.load_dataarray(DOMAIN_DIR / "ag_frac_static.nc")
    return np.where(ag_static.values <= 0.0)[0]


@pytest.fixture(scope="module")
def model_run(tmp_path_factory):
    """Build + run + finalize the two-process Model once."""
    out_dir = tmp_path_factory.mktemp("runoff_soilzone_ag_output")
    runoff_forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_id": "nhru"}
        )
        for nn in RUNOFF_DISK_INPUTS
    }
    soilzone_forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_id": "nhru"}
        )
        for nn in SOILZONE_DISK_INPUTS
    }
    template = runoff_forcings["potet"]
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
            "parameters": xr.load_dataset(
                DOMAIN_DIR / "parameters_PRMSRunoffAg.nc"
            ),
            "ag_frac": ag_frac,
            **runoff_forcings,
        },
        "prms_soilzone_ag": {
            "class": PRMSSoilzoneAg,
            "discretization": "nhru",
            "parameters": xr.load_dataset(
                DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc"
            ),
            **soilzone_forcings,
        },
    }
    control = {
        "output_var_names": sorted(
            set(RUNOFF_ANSWER_NAMES + SOILZONE_ANSWER_NAMES)
        ),
        "output_serial_zarr": out_dir / "runoff_soilzone_ag.zarr",
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


class TestRunoffSoilzoneAgChain:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(
        self, model_run, answers, not_ag_idx
    ):
        """Both processes' outputs match pywatershed over the full run."""
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in sorted(set(RUNOFF_ANSWER_NAMES + SOILZONE_ANSWER_NAMES)):
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

    def test_dunnian_zero_on_fgr(self, model_run):
        """fgr: sat_threshold >= 999 -> dunnian identically zero (the
        precondition for runoff's sroff being unmodified here)."""
        proc = model_run["model"].model_dict["prms_soilzone_ag"]
        assert (proc["dunnian_flow"].values == 0.0).all()
