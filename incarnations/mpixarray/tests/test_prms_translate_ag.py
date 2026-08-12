"""prms_translate: the agricultural family + transp_frost + dynamic
parameters, from the raw legacy controls (fgr_ag_2yr, ucb_2yr).

What is pinned here:

- class RESOLUTION for the ag/frost shapes (data-free);
- the pyPRMS metadata injections (readers): the 13 GSFLOW ag
  parameters, the ag control entries, the OpenET actet CBH, the PRMS
  multiple-parameter-file feature (full + partial files);
- kit assembly against pywatershed's converted files as ORACLES
  (parameters_PRMSSoilzoneAg.nc, aet_observed.nc);
- transp_on over the FULL window, EXACTLY, against the GSFLOW Fortran
  answers -- both leaves (static frost params in the spinup shape;
  dynamic frost INPUTS in the analysis shape);
- one-liner smoke runs of all three configurations.

Deliberately NOT a hard test: the full fgr chain vs the Fortran
answers below the snow line (atmosphere holds 1e-5 and transp_on is
exact, but snow knife-edges put ~2-3% of hru-days outside 1e-5 on
downstream vars -- the drb fastmath story with a Fortran reference;
the ag process physics is already pinned at 1e-5 by the disk-driven
tests in test_prms_*_ag*.py).
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
pytest.importorskip("pyPRMS", reason="pyPRMS not installed")

from atmosphere.prms_atmosphere import (  # noqa: E402
    PRMSAtmosphereTranspFrost,
    PRMSAtmosphereTranspFrostDyn,
)
from hydrology.prms_runoff import PRMSRunoffAg  # noqa: E402
from hydrology.prms_soilzone_ag import (  # noqa: E402
    PRMSSoilzoneAg,
    PRMSSoilzoneAgObsET,
)
from model import Model  # noqa: E402
from prms_translate import (  # noqa: E402
    assemble_from_control,
    load_parameters,
    model_from_control,
    resolve_classes,
)
from prms_translate.dyn_param import (  # noqa: E402
    forward_fill,
    load_dynamic_parameter,
)

MPIX_ROOT = pl.Path(__file__).parents[4]
PYWS_ROOT = MPIX_ROOT / "pywatershed"
FGR_DIR = PYWS_ROOT / "test_data" / "fgr_ag_2yr"
UCB_DIR = PYWS_ROOT / "test_data" / "ucb_2yr"

# ctl.modules as pyPRMS normalizes the fgr controls
AG_MODULES = {
    "temp_module": "temperature_hru",
    "precip_module": "precipitation_hru",
    "et_module": "potet_jh",
    "solrad_module": "ddsolrad",
    "srunoff_module": "srunoff_smidx",
    "strmflow_module": "muskingum_mann",
    "transp_module": "transp_frost",
    "basin_module": "basin",
    "intcp_module": "intcp",
    "obs_module": "obs",
    "snow_module": "snowcomp",
    "gw_module": "gwflow",
    "soilzone_module": "soilzone_ag",
}

_fgr_needed = [
    FGR_DIR / "spinup.control",
    FGR_DIR / "analysis.control",
    FGR_DIR / "myparam.param",
    FGR_DIR / "prcp.cbh",
    FGR_DIR / "actet_openet.cbh",
    FGR_DIR / "dyn_ag_frac.param",
    FGR_DIR / "spring_frost.dyn",
    FGR_DIR / "fall_frost.dyn",
    FGR_DIR / "parameters_PRMSSoilzoneAg.nc",
    FGR_DIR / "aet_observed.nc",
    FGR_DIR / "output_spinup" / "transp_on.nc",
    FGR_DIR / "output_analysis" / "transp_on.nc",
]
_fgr_missing = [str(ff) for ff in _fgr_needed if not ff.exists()]
fgr_skipif = pytest.mark.skipif(
    bool(_fgr_missing),
    reason="fgr_ag_2yr legacy/answer files missing: "
    + ", ".join(_fgr_missing[:3]),
)

_ucb_needed = [
    UCB_DIR / "nhm_transp_frost.control",
    UCB_DIR / "myparam.param",
    UCB_DIR / "transp_frost.param",
    UCB_DIR / "prcp.cbh",
    UCB_DIR / "output_transp_frost" / "transp_on.nc",
]
_ucb_missing = [str(ff) for ff in _ucb_needed if not ff.exists()]
ucb_skipif = pytest.mark.skipif(
    bool(_ucb_missing),
    reason="ucb_2yr transp_frost files missing: "
    + ", ".join(_ucb_missing[:3]),
)

DT = np.float64(60.0 * 60.0 * 24.0)


# =====================================================================
# resolution pins (data-free)
# =====================================================================


def test_resolve_ag_spinup_shape():
    """soilzone_ag + transp_frost, no ag flags: the spinup shape."""
    classes = resolve_classes(AG_MODULES, {"dprst_flag": 1})
    assert classes["prms_atmosphere"] is PRMSAtmosphereTranspFrost
    assert classes["prms_runoff"] is PRMSRunoffAg
    assert classes["prms_soilzone"] is PRMSSoilzoneAg


def test_resolve_ag_analysis_shape():
    """iter_aet + dynamic ag_frac + dynamic frost: the analysis
    shape."""
    classes = resolve_classes(
        AG_MODULES,
        {
            "dprst_flag": 1,
            "iter_aet_flag": 1,
            "dyn_ag_frac_flag": 1,
            "dyn_fallfrost_flag": 1,
            "dyn_springfrost_flag": 1,
        },
    )
    assert classes["prms_atmosphere"] is PRMSAtmosphereTranspFrostDyn
    assert classes["prms_runoff"] is PRMSRunoffAg
    assert classes["prms_soilzone"] is PRMSSoilzoneAgObsET


def test_resolve_ag_raises():
    """The loud stops: ag without dprst, mixed frost dyn flags,
    untranslated dynamic parameters, ag flags on a non-ag soilzone."""
    with pytest.raises(NotImplementedError, match="dprst_flag=0"):
        resolve_classes(AG_MODULES, {"dprst_flag": 0})
    with pytest.raises(NotImplementedError, match="must match"):
        resolve_classes(
            AG_MODULES, {"dprst_flag": 1, "dyn_springfrost_flag": 1}
        )
    with pytest.raises(NotImplementedError, match="dyn_imperv_flag"):
        resolve_classes(AG_MODULES, {"dprst_flag": 1, "dyn_imperv_flag": 1})
    non_ag = AG_MODULES | {"soilzone_module": "soilzone"}
    with pytest.raises(NotImplementedError, match="non-ag"):
        resolve_classes(non_ag, {"dprst_flag": 1, "iter_aet_flag": 1})


# =====================================================================
# readers: metadata injections + the multi-parameter-file feature
# =====================================================================


@fgr_skipif
def test_fgr_ag_parameters_load():
    """The 13 GSFLOW ag parameters survive the pyPRMS parse (they are
    silently SKIPPED without the metadata injection)."""
    params = load_parameters(FGR_DIR / "myparam.param")
    expected = {
        "ag_cov_type",
        "ag_covden_sum",
        "ag_covden_win",
        "ag_frac",
        "ag_soil2gw_max",
        "ag_soil_moist_init_frac",
        "ag_soil_moist_max",
        "ag_soil_rechr_init_frac",
        "ag_soil_rechr_max_frac",
        "ag_soil_type",
        "ag_soilwater_deficit_min",
        "max_soilzone_ag_iter",
        "soilzone_aet_converge",
    }
    assert expected <= set(params.data_vars)
    assert params["ag_cov_type"].dtype == np.int64
    assert params["ag_frac"].dtype == np.float64
    assert params["max_soilzone_ag_iter"].dims == ("one",)
    assert int(params["max_soilzone_ag_iter"].values[0]) == 100
    assert float(params["soilzone_aet_converge"].values[0]) == 0.01


@ucb_skipif
def test_multi_parameter_file():
    """The PRMS multiple-parameter-file feature: a PARTIAL second file
    (transp_frost.param) merges over the full first one; values pinned
    against a direct parse of the partial file's text."""
    merged = load_parameters(
        [UCB_DIR / "myparam.param", UCB_DIR / "transp_frost.param"]
    )
    for name in ("spring_frost", "fall_frost"):
        assert merged[name].dims == ("nhru",)
        # direct text parse of the partial file's block
        lines = (UCB_DIR / "transp_frost.param").read_text().splitlines()
        ii = lines.index(name)
        ndims = int(lines[ii + 1])
        size = int(lines[ii + 2 + ndims])
        vals = np.array(
            lines[ii + 4 + ndims : ii + 4 + ndims + size], dtype=np.int64
        )
        np.testing.assert_array_equal(merged[name].values, vals)
    # everything from the full file is still there
    assert "soil_moist_max" in merged

    with pytest.raises(ValueError, match="must be a FULL"):
        load_parameters([UCB_DIR / "transp_frost.param"])


# =====================================================================
# kit assembly vs pywatershed's converted files (the oracles)
# =====================================================================


@fgr_skipif
def test_spinup_kit_vs_oracles():
    """The spinup kit: ag parameters packaged bitwise-equal to
    pywatershed's converted parameter file; static ag_frac served as a
    time-constant input equal to the parameter."""
    kit = assemble_from_control(FGR_DIR / "spinup.control")
    assert kit.process_dict["prms_soilzone"]["class"] is PRMSSoilzoneAg

    oracle = xr.load_dataset(FGR_DIR / "parameters_PRMSSoilzoneAg.nc")
    packaged = kit.process_dict["prms_soilzone"]["parameters"]
    for name in (
        "ag_cov_type",
        "ag_soil_moist_max",
        "ag_soil_moist_init_frac",
        "ag_soil_rechr_init_frac",
        "ag_soil_rechr_max_frac",
        "ag_soil_type",
    ):
        np.testing.assert_array_equal(
            packaged[name].values,
            oracle[name].values,
            err_msg=f"{name} differs from the pywatershed oracle",
        )

    af = kit.process_dict["prms_runoff"]["ag_frac"]
    static = xr.load_dataset(FGR_DIR / "ag_frac_static.nc")
    static_vals = static[list(static.data_vars)[0]].values
    assert af.dims == ("time", "nhru")
    assert (af.values == af.values[0, :]).all()  # time-constant
    np.testing.assert_array_equal(af.values[0, :], static_vals)


@fgr_skipif
def test_analysis_kit_vs_oracles():
    """The analysis kit: aet_observed decoded from the OpenET CBH
    bitwise-equal to pywatershed's converted file; dynamic ag_frac and
    frost dates forward-filled and actually time-varying."""
    kit = assemble_from_control(FGR_DIR / "analysis.control")
    assert kit.process_dict["prms_soilzone"]["class"] is PRMSSoilzoneAgObsET
    assert (
        kit.process_dict["prms_atmosphere"]["class"]
        is PRMSAtmosphereTranspFrostDyn
    )

    aet = kit.process_dict["prms_soilzone"]["aet_observed"]
    oracle = xr.load_dataarray(FGR_DIR / "aet_observed.nc")
    np.testing.assert_array_equal(aet.values, oracle.values)

    times = aet["time"].values
    for name, dyn_file, slot in (
        ("ag_frac", "dyn_ag_frac.param", "prms_runoff"),
        ("spring_frost", "spring_frost.dyn", "prms_atmosphere"),
        ("fall_frost", "fall_frost.dyn", "prms_atmosphere"),
    ):
        served = kit.process_dict[slot][name]
        expected = forward_fill(
            load_dynamic_parameter(FGR_DIR / dyn_file), times
        )
        np.testing.assert_array_equal(served.values, expected.values)
        assert (served.values != served.values[0, :]).any(), (
            f"{name} should vary in time in the analysis shape"
        )


# =====================================================================
# transp_on: FULL window, EXACT, vs the GSFLOW Fortran answers
# =====================================================================


@fgr_skipif
@pytest.mark.parametrize("config", ["spinup", "analysis"])
def test_transp_on_exact_full_window(config, tmp_path):
    """Both frost leaves against the Fortran answers over the full 2
    years, exactly (transp_on is a 0/1 window on integer solar days):
    static frost parameters (spinup) and dynamic frost inputs
    (analysis). Atmosphere-only model -- the leaf under test."""
    kit = assemble_from_control(FGR_DIR / f"{config}.control")
    process_dict = {"prms_atmosphere": kit.process_dict["prms_atmosphere"]}
    control = {
        "output_var_names": ["transp_on"],
        "output_serial_zarr": tmp_path / f"transp_{config}.zarr",
        "time_chunk_size": 123,
    }
    discretizations = {"nhru": kit.discretizations["nhru"]}
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(DT, np.int32(model.ntime))
    out = xr.load_dataset(
        control["output_serial_zarr"], engine="zarr", consolidated=False
    )
    answers = xr.load_dataarray(FGR_DIR / f"output_{config}" / "transp_on.nc")
    np.testing.assert_array_equal(out["transp_on"].values, answers.values)


# =====================================================================
# one-liner smoke runs (construction IS the validation + a short run)
# =====================================================================


@fgr_skipif
@pytest.mark.parametrize("config", ["spinup", "analysis"])
def test_fgr_one_liner_runs(config):
    model = model_from_control(FGR_DIR / f"{config}.control")
    model.run(DT, np.int32(3))
    soilzone = model.model_dict["prms_soilzone"]
    assert np.isfinite(soilzone["ag_soil_moist"].values).all()
    assert np.isfinite(soilzone["hru_actet"].values).all()
    if config == "analysis":
        assert (soilzone["iter_count"].values >= 1).all()


@ucb_skipif
def test_ucb_one_liner_runs():
    """The multi-param_file control end-to-end: build, run 5 days,
    transp_on exact vs the generated answers."""
    model = model_from_control(UCB_DIR / "nhm_transp_frost.control")
    model.run(DT, np.int32(5))
    answers = xr.load_dataarray(
        UCB_DIR / "output_transp_frost" / "transp_on.nc"
    )
    np.testing.assert_array_equal(
        model.model_dict["prms_atmosphere"]["transp_on"].values,
        answers.values[4, :],
    )
