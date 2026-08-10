"""Internal consistency: NoDprst classes vs full classes with dprst off.

The KEY no-external-answers test of the additive variant structure
(PORTS.md "How variants are done here"): for each of PRMSRunoff /
PRMSSoilzone / PRMSGroundwater, the NoDprst base class must agree
EXACTLY (bit-for-bit, assert_array_equal) with the full class run with
depression storage disabled by data:

- full runoff / soilzone: ``dprst_frac = 0`` everywhere (all dprst
  geometry collapses to zero; every dprst block is guarded by
  ``dprst_area_max > 0`` / ``dprst_frac > 0``);
- full soilzone / groundwater: zero ``dprst_evap_hru`` /
  ``dprst_seep_hru`` input arrays (their kernel terms add +0.0 --
  value-identical in IEEE).

This pins the independent NoDprst kernels/inits to the validated full
ones without needing generated no-dprst answers (those land in
test_prms_no_dprst_parity.py). Forcings are the same drb_2yr generated
files the standalone parity tests use; the NoDprst runs read the
actual parameters_PRMS*NoDprst.nc files.

Requires GENERATED pywatershed test data (test_data/drb_2yr/output/);
skips with a clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_groundwater import (
    PRMSGroundwater,
    PRMSGroundwaterNoDprst,
)
from hydrology.prms_runoff import PRMSRunoff, PRMSRunoffNoDprst
from hydrology.prms_soilzone import PRMSSoilzone, PRMSSoilzoneNoDprst
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

RUNOFF_INPUT_NAMES = (
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
# the NoDprst (= shared) variables; hru_impervstor_old is bookkeeping
RUNOFF_COMPARE_NAMES = (
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
)
RUNOFF_DERIVED_NAMES = ("hru_perv", "hru_frac_perv", "hru_imperv")

SOILZONE_INPUT_NAMES = (
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
SOILZONE_DPRST_INPUT_NAMES = ("dprst_evap_hru", "dprst_seep_hru")
# all shared variables except the mutable inputs (fed from disk each
# step -- comparing them is a tautology) and the *_prev bookkeeping
SOILZONE_COMPARE_NAMES = (
    "cap_infil_tot",
    "cap_waterin",
    "dunnian_flow",
    "hru_actet",
    "perv_actet",
    "perv_actet_hru",
    "potet_lower",
    "potet_rechr",
    "pref_flow",
    "pref_flow_in",
    "pref_flow_infil",
    "pref_flow_stor",
    "pref_flow_stor_change",
    "recharge",
    "slow_flow",
    "slow_stor",
    "slow_stor_change",
    "soil_lower",
    "soil_lower_change",
    "soil_lower_change_hru",
    "soil_lower_ratio",
    "soil_moist",
    "soil_moist_tot",
    "soil_rechr",
    "soil_rechr_change",
    "soil_rechr_change_hru",
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
SOILZONE_DERIVED_NAMES = ("hru_frac_perv",)

GW_INPUT_NAMES = ("soil_to_gw", "ssr_to_gw")
GW_COMPARE_NAMES = (
    "gwres_stor",
    "gwres_flow",
    "gwres_sink",
    "gwres_stor_change",
    "gwres_flow_vol",
)

_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoff.nc",
    DOMAIN_DIR / "parameters_PRMSRunoffNoDprst.nc",
    DOMAIN_DIR / "parameters_PRMSSoilzone.nc",
    DOMAIN_DIR / "parameters_PRMSSoilzoneNoDprst.nc",
    DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
    DOMAIN_DIR / "parameters_PRMSGroundwaterNoDprst.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in set(
        RUNOFF_INPUT_NAMES
        + SOILZONE_INPUT_NAMES
        + SOILZONE_DPRST_INPUT_NAMES
        + GW_INPUT_NAMES
    )
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


def _load_forcings(names):
    return {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_id": "nhru"}
        )
        for nn in names
    }


def _zero_forcings(names, template):
    """Zero (time, nhru) input arrays shaped like a real forcing."""
    out = {}
    for nn in names:
        zda = template * 0.0
        zda.name = nn
        out[nn] = zda
    return out


def _run_one(proc_name, proc_class, parameters, forcings, out_names, out_dir):
    process_dict = {
        proc_name: {
            "class": proc_class,
            "discretization": "nhru",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(out_names),
        "output_serial_zarr": out_dir / f"{proc_class.__name__}.zarr",
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
    output = xr.load_dataset(
        control["output_serial_zarr"], engine="zarr", consolidated=False
    )
    return {"model": model, "output": output}


def _assert_identical(pair, proc_name, compare_names, derived_names):
    full, nodprst = pair
    for nn in compare_names:
        np.testing.assert_array_equal(
            nodprst["output"][nn].values,
            full["output"][nn].values,
            err_msg=f"variable '{nn}' differs (NoDprst vs full-dprst-off)",
        )
    proc_full = full["model"].model_dict[proc_name]
    proc_nodprst = nodprst["model"].model_dict[proc_name]
    for nn in derived_names:
        np.testing.assert_array_equal(
            proc_nodprst[nn].values,
            proc_full[nn].values,
            err_msg=f"derived '{nn}' differs (NoDprst vs full-dprst-off)",
        )


# ----------------------------------------------------------------------
# PRMSRunoffNoDprst vs PRMSRunoff(dprst_frac = 0)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def runoff_pair(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("no_dprst_runoff")
    forcings = _load_forcings(RUNOFF_INPUT_NAMES)

    params_off = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSRunoff.nc")
    params_off["dprst_frac"].values[:] = 0.0
    full = _run_one(
        "prms_runoff",
        PRMSRunoff,
        params_off,
        forcings,
        RUNOFF_COMPARE_NAMES,
        out_dir,
    )

    params_nodprst = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSRunoffNoDprst.nc"
    )
    nodprst = _run_one(
        "prms_runoff",
        PRMSRunoffNoDprst,
        params_nodprst,
        forcings,
        RUNOFF_COMPARE_NAMES,
        out_dir,
    )
    return full, nodprst


class TestRunoffNoDprst:
    def test_identical(self, runoff_pair):
        _assert_identical(
            runoff_pair,
            "prms_runoff",
            RUNOFF_COMPARE_NAMES,
            RUNOFF_DERIVED_NAMES,
        )


# ----------------------------------------------------------------------
# PRMSSoilzoneNoDprst vs PRMSSoilzone(dprst_frac = 0, zero dprst inputs)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def soilzone_pair(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("no_dprst_soilzone")
    forcings = _load_forcings(SOILZONE_INPUT_NAMES)

    params_off = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSoilzone.nc")
    params_off["dprst_frac"].values[:] = 0.0
    forcings_full = dict(forcings) | _zero_forcings(
        SOILZONE_DPRST_INPUT_NAMES, forcings["potet"]
    )
    full = _run_one(
        "prms_soilzone",
        PRMSSoilzone,
        params_off,
        forcings_full,
        SOILZONE_COMPARE_NAMES,
        out_dir,
    )

    params_nodprst = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSSoilzoneNoDprst.nc"
    )
    nodprst = _run_one(
        "prms_soilzone",
        PRMSSoilzoneNoDprst,
        params_nodprst,
        forcings,
        SOILZONE_COMPARE_NAMES,
        out_dir,
    )
    return full, nodprst


class TestSoilzoneNoDprst:
    def test_identical(self, soilzone_pair):
        _assert_identical(
            soilzone_pair,
            "prms_soilzone",
            SOILZONE_COMPARE_NAMES,
            SOILZONE_DERIVED_NAMES,
        )


# ----------------------------------------------------------------------
# PRMSGroundwaterNoDprst vs PRMSGroundwater(zero dprst_seep_hru)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def groundwater_pair(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("no_dprst_gw")
    forcings = _load_forcings(GW_INPUT_NAMES)

    params_full = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSGroundwater.nc"
    )
    forcings_full = dict(forcings) | _zero_forcings(
        ("dprst_seep_hru",), forcings["soil_to_gw"]
    )
    full = _run_one(
        "prms_groundwater",
        PRMSGroundwater,
        params_full,
        forcings_full,
        GW_COMPARE_NAMES,
        out_dir,
    )

    params_nodprst = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSGroundwaterNoDprst.nc"
    )
    nodprst = _run_one(
        "prms_groundwater",
        PRMSGroundwaterNoDprst,
        params_nodprst,
        forcings,
        GW_COMPARE_NAMES,
        out_dir,
    )
    return full, nodprst


class TestGroundwaterNoDprst:
    def test_identical(self, groundwater_pair):
        _assert_identical(
            groundwater_pair,
            "prms_groundwater",
            GW_COMPARE_NAMES,
            (),
        )
