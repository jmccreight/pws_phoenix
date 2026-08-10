"""Perfect-restart tests for the Ag family (fgr_ag_2yr data).

Extends the R2 recipe (test_restart_processes.perfect_restart) to the
three ag processes. The ag-specific wrinkle is the PER-STEP AREAS:
under time-varying ag_frac both processes read the PREVIOUS step's
areas at step start (runoff-ag's kernel before _post_areas;
soilzone-ag's _update_areas old_ag_frac = ag_area/harea) and the
istep0 area blocks only run at time zero -- so hru_perv/hru_frac_perv/
ag_area (runoff-ag) and hru_area_perv/ag_area (soilzone-ag) carry
restart=True as prognostic markers even though they are not storages.
The storages mirror plain soilzone's set (incl. the derived
soil_lower, and ag_soil_lower for symmetry). ObsET adds NO restart
state: its It0 buffers are per-step scratch (overwritten from current
values at each step start before any read).

Coverage: the live RunoffAg -> SoilzoneAg chain (static tiled ag_frac,
the spinup configuration) is the definitive test; the ObsET standalone
runs the DYNAMIC ag_frac + observed-AET iteration configuration.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
sys.path.append(str(pl.Path(__file__).parent))
from discretization import Discretization
from hydrology.prms_runoff import PRMSRunoff, PRMSRunoffAg
from hydrology.prms_soilzone_ag import (
    PRMSSoilzoneAg,
    PRMSSoilzoneAgObsET,
)
from model import Model
from test_restart_processes import perfect_restart

MPIX_ROOT = pl.Path(__file__).parents[4]
PYWS_ROOT = MPIX_ROOT / "pywatershed"
DOMAIN_DIR = PYWS_ROOT / "test_data" / "fgr_ag_2yr"
GEN_SPINUP = DOMAIN_DIR / "output_spinup"
GEN_ANALYSIS = DOMAIN_DIR / "output_analysis"

# the live-chain configuration (test_prms_runoff_soilzone_ag.py)
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
SOILZONE_DISK_INPUTS = ("transp_on",)

# the ObsET standalone configuration (test_prms_soilzone_ag_obs_et.py)
OBS_DISK_INPUTS = (
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
    "sroff",
    "sroff_vol",
)

_chain_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoffAg.nc",
    DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "ag_frac_static.nc",
] + [
    GEN_SPINUP / f"{nn}.nc"
    for nn in RUNOFF_DISK_INPUTS + SOILZONE_DISK_INPUTS
]
_chain_missing = [str(ff) for ff in _chain_needed if not ff.exists()]
chain_skipif = pytest.mark.skipif(
    bool(_chain_missing),
    reason=(
        "pywatershed fgr_ag_2yr spinup data not present/generated; "
        "missing: " + ", ".join(_chain_missing[:3])
    ),
)

_obs_needed = [
    DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "dyn_ag_frac.param",
    DOMAIN_DIR / "aet_observed.nc",
    PYWS_ROOT / "pywatershed" / "utils" / "prms_dyn_param.py",
] + [GEN_ANALYSIS / f"{nn}.nc" for nn in OBS_DISK_INPUTS]
_obs_missing = [str(ff) for ff in _obs_needed if not ff.exists()]
obs_skipif = pytest.mark.skipif(
    bool(_obs_missing),
    reason=(
        "pywatershed fgr_ag_2yr analysis data not present/generated; "
        "missing: " + ", ".join(_obs_missing[:3])
    ),
)


def _load_forcings(names, gen_dir):
    return {
        nn: xr.load_dataarray(gen_dir / f"{nn}.nc").rename(
            {"nhm_id": "nhru"}
        )
        for nn in names
    }


def _static_ag_frac(template):
    """The spinup configuration: the static ag_frac time-tiled."""
    ag_static = xr.load_dataarray(DOMAIN_DIR / "ag_frac_static.nc")
    return xr.DataArray(
        np.tile(ag_static.values, (template.sizes["time"], 1)),
        dims=("time", "nhru"),
        coords={"time": template["time"], "nhru": template["nhru"]},
        name="ag_frac",
    )


def _dynamic_ag_frac(template):
    """Forward-fill the PRMS dynamic parameter file onto the forcing
    time axis (as in test_prms_soilzone_ag_obs_et.py)."""
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


def test_flags_derivation_pin():
    """The ag flag sets, exactly (data-free)."""
    assert set(PRMSRunoffAg.get_restart_variables()) == set(
        PRMSRunoff.get_restart_variables()
    ) | {"hru_perv", "hru_frac_perv", "ag_area"}
    assert set(PRMSSoilzoneAg.get_restart_variables()) == {
        "soil_moist",
        "soil_rechr",
        "soil_lower",
        "slow_stor",
        "pref_flow_stor",
        "ag_soil_moist",
        "ag_soil_rechr",
        "ag_soil_lower",
        "ag_area",
        "hru_area_perv",
    }
    # ObsET adds iteration diagnostics + It0 scratch only -- nothing
    # prognostic beyond its base
    assert set(PRMSSoilzoneAgObsET.get_restart_variables()) == set(
        PRMSSoilzoneAg.get_restart_variables()
    )


@chain_skipif
def test_runoff_soilzone_ag_chain(tmp_path):
    """Live RunoffAg -> SoilzoneAg chain (spinup config), 90 days."""
    runoff_forcings = _load_forcings(RUNOFF_DISK_INPUTS, GEN_SPINUP)
    soilzone_forcings = _load_forcings(SOILZONE_DISK_INPUTS, GEN_SPINUP)
    ag_frac = _static_ag_frac(runoff_forcings["potet"])
    runoff_params = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSRunoffAg.nc"
    )
    soilzone_params = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc"
    )

    def make_model(control):
        process_dict = {
            "prms_runoff_ag": {
                "class": PRMSRunoffAg,
                "discretization": "nhru",
                "parameters": runoff_params,
                "ag_frac": ag_frac,
                **runoff_forcings,
            },
            "prms_soilzone_ag": {
                "class": PRMSSoilzoneAg,
                "discretization": "nhru",
                "parameters": soilzone_params,
                **soilzone_forcings,
            },
        }
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(
            process_dict, dict(control), discretizations=discretizations
        )

    perfect_restart(make_model, np.float64(1.0), 90, tmp_path)


@obs_skipif
def test_soilzone_ag_obs_et(tmp_path):
    """ObsET standalone: DYNAMIC ag_frac + AET iteration, 120 days."""
    forcings = _load_forcings(OBS_DISK_INPUTS, GEN_ANALYSIS)
    template = forcings["potet"]
    ag_frac = _dynamic_ag_frac(template)
    aet_observed = (
        xr.load_dataarray(DOMAIN_DIR / "aet_observed.nc")
        .sel(time=template["time"])
        .assign_coords(nhru=template["nhru"])
    )
    parameters = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSSoilzoneAg.nc"
    )

    def make_model(control):
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
        discretizations = {
            "nhru": Discretization(
                ["nhru"], parameters=DOMAIN_DIR / "parameters_dis_hru.nc"
            ),
        }
        return Model(
            process_dict, dict(control), discretizations=discretizations
        )

    perfect_restart(make_model, np.float64(1.0), 120, tmp_path)
