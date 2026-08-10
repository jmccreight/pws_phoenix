"""prms_translate tests: class resolution pins (data-free) + decode
parity against pywatershed's converted files (drb/fgr generated data).

The parity direction matters: prms_translate must reproduce, FROM THE
LEGACY ASCII FILES ALONE via pyPRMS, exactly what pywatershed's own
converters produced (float64 parameter parse, CBH values, dynamic-
parameter forward-fill) -- pywatershed appears here ONLY as the test
oracle, never as a prms_translate dependency.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
pytest.importorskip("pyPRMS", reason="pyPRMS not installed")

from prms_translate import (  # noqa: E402
    assemble_from_control,
    from_control,
    load_cbh,
    load_control,
    load_dynamic_parameter,
    load_parameters,
    model_from_control,
    package_parameters,
    resolve_classes,
    volume_map_weights,
)
from prms_translate.dyn_param import forward_fill  # noqa: E402

MPIX_ROOT = pl.Path(__file__).parents[4]
PYWS_ROOT = MPIX_ROOT / "pywatershed"
DRB_DIR = PYWS_ROOT / "test_data" / "drb_2yr"
FGR_DIR = PYWS_ROOT / "test_data" / "fgr_ag_2yr"

# ctl.modules as pyPRMS normalizes the drb NHM control
NHM_MODULES = {
    "temp_module": "temperature_hru",
    "precip_module": "precipitation_hru",
    "et_module": "potet_jh",
    "solrad_module": "ddsolrad",
    "srunoff_module": "srunoff_smidx",
    "strmflow_module": "muskingum_mann",
    "transp_module": "transp_tindex",
    "basin_module": "basin",
    "intcp_module": "intcp",
    "obs_module": "obs",
    "snow_module": "snowcomp",
    "gw_module": "gwflow",
    "soilzone_module": "soilzone",
}

_drb_needed = [
    DRB_DIR / "nhm_stream_temp.control",
    DRB_DIR / "myparam.param",
    DRB_DIR / "prcp.cbh",
    DRB_DIR / "rhavg.cbh",
    DRB_DIR / "parameters_PRMSGroundwater.nc",
    DRB_DIR / "parameters_PRMSSnow.nc",
    DRB_DIR / "parameters_PRMSChannel.nc",
    DRB_DIR / "prcp.nc",
    DRB_DIR / "cbh.nc",
]
_drb_missing = [str(ff) for ff in _drb_needed if not ff.exists()]
drb_skipif = pytest.mark.skipif(
    bool(_drb_missing),
    reason="drb_2yr legacy/converted files missing: "
    + ", ".join(_drb_missing[:3]),
)

_fgr_needed = [
    FGR_DIR / "dyn_ag_frac.param",
    PYWS_ROOT / "pywatershed" / "utils" / "prms_dyn_param.py",
]
_fgr_missing = [str(ff) for ff in _fgr_needed if not ff.exists()]
fgr_skipif = pytest.mark.skipif(
    bool(_fgr_missing),
    reason="fgr dynamic-parameter files missing: "
    + ", ".join(_fgr_missing),
)


# ----------------------------------------------------------------------
# class resolution (data-free)
# ----------------------------------------------------------------------


def test_resolve_nhm_dprst():
    from atmosphere.prms_atmosphere import PRMSAtmosphere
    from hydrology.prms_channel import PRMSChannel
    from hydrology.prms_groundwater import PRMSGroundwater
    from hydrology.prms_runoff import PRMSRunoff
    from hydrology.prms_soilzone import PRMSSoilzone

    classes = resolve_classes(NHM_MODULES, {"dprst_flag": 1})
    assert list(classes) == [
        "prms_atmosphere",
        "prms_canopy",
        "prms_snow",
        "prms_runoff",
        "prms_soilzone",
        "prms_groundwater",
        "prms_channel",
    ]
    assert classes["prms_atmosphere"] is PRMSAtmosphere
    assert classes["prms_runoff"] is PRMSRunoff
    assert classes["prms_soilzone"] is PRMSSoilzone
    assert classes["prms_groundwater"] is PRMSGroundwater
    assert classes["prms_channel"] is PRMSChannel


def test_resolve_no_dprst():
    from hydrology.prms_groundwater import PRMSGroundwaterNoDprst
    from hydrology.prms_runoff import PRMSRunoffNoDprst
    from hydrology.prms_soilzone import PRMSSoilzoneNoDprst

    classes = resolve_classes(NHM_MODULES, {"dprst_flag": 0})
    assert classes["prms_runoff"] is PRMSRunoffNoDprst
    assert classes["prms_soilzone"] is PRMSSoilzoneNoDprst
    assert classes["prms_groundwater"] is PRMSGroundwaterNoDprst


def test_resolve_stream_temp_leaves():
    from hydrology.prms_stream_temp import (
        PRMSStreamTemp,
        PRMSStreamTempConstantShade,
        PRMSStreamTempSegHumidity,
    )

    base = {"dprst_flag": 1, "stream_temp_flag": 1}
    classes = resolve_classes(NHM_MODULES, base)
    assert list(classes)[-2:] == [
        "prms_hydraulic_geometry",
        "prms_stream_temp",
    ]
    assert classes["prms_stream_temp"] is PRMSStreamTemp
    classes = resolve_classes(
        NHM_MODULES, base | {"stream_temp_shade_flag": 1}
    )
    assert classes["prms_stream_temp"] is PRMSStreamTempConstantShade
    classes = resolve_classes(
        NHM_MODULES, base | {"strmtemp_humidity_flag": 1}
    )
    assert classes["prms_stream_temp"] is PRMSStreamTempSegHumidity
    with pytest.raises(NotImplementedError, match="combined"):
        resolve_classes(
            NHM_MODULES,
            base
            | {"stream_temp_shade_flag": 1, "strmtemp_humidity_flag": 1},
        )


@pytest.mark.parametrize(
    "slot, value",
    [
        ("srunoff_module", "srunoff_carea"),
        ("soilzone_module", "soilzone_ag"),
        ("transp_module", "transp_frost"),
        ("strmflow_module", "strmflow_in_out"),
        ("et_module", "potet_pt"),
    ],
)
def test_resolve_unported_raises(slot, value):
    modules = NHM_MODULES | {slot: value}
    with pytest.raises(NotImplementedError, match=value):
        resolve_classes(modules, {"dprst_flag": 1})


# ----------------------------------------------------------------------
# control / parameters / cbh decode parity (drb legacy files)
# ----------------------------------------------------------------------


@drb_skipif
def test_from_control():
    path = DRB_DIR / "nhm_stream_temp.control"
    cfg = from_control(load_control(path), path)
    assert cfg.start_time == np.datetime64("1979-01-01")
    assert cfg.end_time == np.datetime64("1980-12-31")
    assert len(cfg.classes) == 9
    assert sorted(cfg.cbh_paths) == [
        "humidity_hru",
        "prcp",
        "tmax",
        "tmin",
    ]
    for pp in cfg.cbh_paths.values():
        assert pp.exists()
    assert cfg.dynamic_parameters == []


@drb_skipif
def test_parameters_exact_vs_pywatershed():
    """The float64 parse from the ASCII == pywatershed's converters,
    bitwise; monthly params on (nmonths, nhru); ints int64."""
    ds = load_parameters(DRB_DIR / "myparam.param")
    gw = xr.load_dataset(DRB_DIR / "parameters_PRMSGroundwater.nc")
    snow = xr.load_dataset(DRB_DIR / "parameters_PRMSSnow.nc")
    chan = xr.load_dataset(DRB_DIR / "parameters_PRMSChannel.nc")

    assert ds["gwflow_coef"].dtype == np.float64
    np.testing.assert_array_equal(
        ds["gwflow_coef"].values, gw["gwflow_coef"].values
    )
    assert ds["tmax_allsnow"].dims == ("nmonths", "nhru")
    np.testing.assert_array_equal(
        ds["tmax_allsnow"].values, snow["tmax_allsnow"].values
    )
    assert ds["hru_segment"].dtype == np.int64
    np.testing.assert_array_equal(
        ds["hru_segment"].values, chan["hru_segment"].values
    )


@drb_skipif
def test_cbh_exact_vs_pywatershed():
    prcp = load_cbh(DRB_DIR / "prcp.cbh")
    assert prcp.name == "prcp"
    assert prcp.dtype == np.float64
    assert prcp.dims == ("time", "nhru")
    theirs = xr.load_dataset(DRB_DIR / "prcp.nc")["prcp"]
    np.testing.assert_array_equal(
        prcp.values, theirs.values.astype(np.float64)
    )
    # pyPRMS itself maps rhavg -> humidity_hru (kept: the Map name)
    humid = load_cbh(
        DRB_DIR / "rhavg.cbh",
        start_time=np.datetime64("1979-01-01"),
        end_time=np.datetime64("1979-01-10"),
    )
    assert humid.name == "humidity_hru"
    assert humid.sizes["time"] == 10
    theirs = xr.load_dataset(DRB_DIR / "cbh.nc")["rhavg"]
    np.testing.assert_array_equal(
        humid.values,
        theirs.isel(time=slice(0, 10)).values.astype(np.float64),
    )


# ----------------------------------------------------------------------
# contract-driven parameter packaging
# ----------------------------------------------------------------------


@drb_skipif
def test_package_parameters():
    """The split is complete for the full stream-temp model, at the
    declared dims, with the derived and computed parameters routed."""
    from atmosphere.prms_solar_geometry import compute_soltabs

    params = load_parameters(DRB_DIR / "myparam.param")
    classes = resolve_classes(
        NHM_MODULES, {"dprst_flag": 1, "stream_temp_flag": 1}
    )
    soltabs = compute_soltabs(
        params[["hru_slope", "hru_aspect", "hru_lat"]], hru_dim="nhru"
    )
    packaged = package_parameters(
        params,
        classes,
        extra={str(nn): soltabs[nn] for nn in soltabs.data_vars},
    )
    assert sorted(packaged) == sorted(classes)
    # every declared parameter present (minus the dis-owned one)
    from process import _dict_of_kind

    for slot, cls in classes.items():
        declared = set(_dict_of_kind(cls, "parameter")) - {"segment_order"}
        assert set(packaged[slot].data_vars) == declared, slot
    # dim conventions from the positional-rename rule
    snow = packaged["prms_snow"]
    assert snow["albset_rna"].dims == ("scalar",)
    assert snow["tmax_allsnow"].dims == ("nmonth", "nhru")
    assert snow["soltab_horad_potsw"].dims == ("ndoy", "nhru")
    # PRMS dimensions the snow densities on nhru; uniform -> collapsed
    # to the declared scalar, matching pywatershed's separated file
    pws_snow = xr.load_dataset(DRB_DIR / "parameters_PRMSSnow.nc")
    assert snow["den_init"].dims == ("scalar",)
    np.testing.assert_array_equal(
        snow["den_init"].values, pws_snow["den_init"].values
    )
    stp = packaged["prms_stream_temp"]
    assert stp["hru_segment"].dims == ("nhru",)
    assert stp["seg_length"].dims == ("nsegment",)
    # the derived parameter, exact vs pywatershed's dis file
    dis_hru = xr.load_dataset(DRB_DIR / "parameters_dis_hru.nc")
    np.testing.assert_array_equal(
        packaged["prms_runoff"]["hru_in_to_cf"].values,
        dis_hru["hru_in_to_cf"].values,
    )
    # the volume-map weights match the established 0/1 construction
    weights = volume_map_weights(params)
    chan = xr.load_dataset(DRB_DIR / "parameters_PRMSChannel.nc")
    hru_segment = chan["hru_segment"].values
    expected = np.zeros((params.sizes["nsegment"], hru_segment.shape[0]))
    for ii in range(hru_segment.shape[0]):
        if hru_segment[ii] > 0:
            expected[hru_segment[ii] - 1, ii] = 1.0
    np.testing.assert_array_equal(weights, expected)


@drb_skipif
def test_package_parameters_missing_raises():
    params = load_parameters(DRB_DIR / "myparam.param")
    classes = resolve_classes(NHM_MODULES, {"dprst_flag": 1})
    with pytest.raises(KeyError, match="soltab_potsw"):
        package_parameters(params, classes)  # soltabs not supplied


# ----------------------------------------------------------------------
# the assembly kit + the one-liner
# ----------------------------------------------------------------------


@drb_skipif
def test_assemble_stream_temp_kit():
    """The full stream-temp kit: shapes, order, and the carrier."""
    kit = assemble_from_control(DRB_DIR / "nhm_stream_temp.control")
    slots = list(kit.process_dict)
    assert len(slots) == 10  # 9 resolved + the humidity carrier
    assert slots.index("humidity_carrier") == slots.index(
        "prms_channel"
    ) - 1  # just before the first segment-grid process
    assert len(kit.maps) == 13  # 3 volumes + 10 aggregations
    assert sorted(kit.discretizations) == ["nhru", "nsegment"]
    for slot in ("prms_atmosphere", "prms_stream_temp"):
        assert "parameters" in kit.process_dict[slot]
    # external inputs landed on their consumers
    assert "prcp" in kit.process_dict["prms_atmosphere"]
    assert "humidity_hru" in kit.process_dict["humidity_carrier"]


@drb_skipif
def test_assemble_no_stream_temp_kit():
    """nhm.control (no stream temp): no carrier, volumes only."""
    kit = assemble_from_control(DRB_DIR / "nhm.control")
    assert len(kit.process_dict) == 7
    assert "humidity_carrier" not in kit.process_dict
    assert sorted(kit.maps) == ["gw_vol", "sroff_vol", "ssres_vol"]
    # the kit constructs (construction IS the assembly validation)
    model = kit.model()
    assert model.ntime == 731


@drb_skipif
def test_model_from_control_runs():
    """The one-liner builds the complete stream-temp model and a
    short run produces the expected finite field."""
    model = model_from_control(DRB_DIR / "nhm_stream_temp.control")
    model.run(np.float64(86400.0), np.int32(3))
    seg_tave = model.model_dict["prms_stream_temp"]["seg_tave_water"]
    finite = np.isfinite(seg_tave.values)
    # 455 of 456: drb's single never-has-flow segment is NaN
    assert finite.sum() == 455


@drb_skipif
def test_output_var_names_translation(tmp_path):
    """PRMS *OutVar_names translate (filtered, gated on a supplied
    store); the control file's output PATHS never do."""
    path = DRB_DIR / "nhm_stream_temp.control"
    cfg = from_control(load_control(path), path)
    assert "seg_outflow" in cfg.output_var_names
    assert "gwres_flow" in cfg.output_var_names

    # no store supplied -> no injection (Model would require both)
    kit = assemble_from_control(path)
    assert "output_var_names" not in kit.control
    assert kit.dropped_output_var_names == []

    # store supplied -> filtered names injected; PRMS-only names drop
    kit = assemble_from_control(
        path,
        control={
            "output_serial_zarr": tmp_path / "run.zarr",
            "time_chunk_size": 61,
        },
    )
    kept = kit.control["output_var_names"]
    assert "seg_outflow" in kept
    assert set(kept).isdisjoint(kit.dropped_output_var_names)
    assert "hru_storage" in kit.dropped_output_var_names  # PRMS-only
    assert set(kept) | set(kit.dropped_output_var_names) == set(
        cfg.output_var_names
    )

    # an explicit caller list wins verbatim
    kit = assemble_from_control(
        path,
        control={
            "output_serial_zarr": tmp_path / "run2.zarr",
            "output_var_names": ["seg_outflow"],
            "time_chunk_size": 61,
        },
    )
    assert kit.control["output_var_names"] == ["seg_outflow"]

    # and the translated request actually writes
    store = tmp_path / "run3.zarr"
    with model_from_control(
        path,
        control={"output_serial_zarr": store, "time_chunk_size": 61},
    ) as model:
        model.run(np.float64(86400.0), np.int32(2))
    with xr.open_zarr(store, consolidated=False) as ds_out:
        assert "seg_outflow" in ds_out
        assert ds_out.sizes["time"] == 2


_seg_humid_control = DRB_DIR / "nhm_stream_temp_seg_humid_matrix.control"


@pytest.mark.skipif(
    not _seg_humid_control.exists(),
    reason="seg-humid control file not present",
)
def test_assemble_unwired_leaf_raises():
    with pytest.raises(NotImplementedError, match="SegHumidity"):
        assemble_from_control(_seg_humid_control)


# ----------------------------------------------------------------------
# dynamic-parameter reader vs pywatershed's (the rewritten capability)
# ----------------------------------------------------------------------


@fgr_skipif
def test_dyn_param_matches_pywatershed():
    dyn = load_dynamic_parameter(FGR_DIR / "dyn_ag_frac.param")
    assert dyn.dtype == np.float64
    assert dyn.dims == ("time", "nhru")

    sys.path.insert(0, str(PYWS_ROOT))
    from pywatershed.utils.prms_dyn_param import PrmsDynamicParameter

    dp = PrmsDynamicParameter.load(
        FGR_DIR / "dyn_ag_frac.param", dtype="float"
    )
    theirs_dates = np.array(
        [
            np.datetime64(f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}")
            for yy, mm, dd in dp.dates
        ]
    )
    np.testing.assert_array_equal(dyn["time"].values, theirs_dates)
    np.testing.assert_array_equal(dyn.values, dp.data)

    # forward-fill parity with the established test recipe
    times = np.arange(
        np.datetime64("1999-10-01"), np.datetime64("2001-10-01")
    )
    filled = forward_fill(dyn, times)
    idx = np.searchsorted(theirs_dates, times, side="right") - 1
    idx = np.clip(idx, 0, len(theirs_dates) - 1)
    np.testing.assert_array_equal(filled.values, dp.data[idx, :])
