"""Behavioral pin: PRMSStreamTempConstantShade (no external answers).

pywatershed generates drb answers only for the DYNAMIC-shade
configurations (stream_temp_shade_flag = 0), so the constant-shade
leaf (flag = 1; upstream PRMSStreamShadeConstant) has no Fortran
reference. This test pins the leaf's own semantics over a real run
(the nhm_stream_temp inputs + synthetic per-segment segshade
parameters):

- seg_shade == segshade_sum on summer days (doy 121-273) and
  segshade_win otherwise, EXACTLY, every day/segment -- the season
  boundary and the per-segment indexing;
- the vegetation shade index scratch (_seg_svi) is never written
  (stays at its initialize() zeros; upstream returns svi = 0.0);
- the never-has-flow structure (NaN columns of seg_tave_water,
  computed in the shared base initialize) is unchanged from the
  dynamic run's, and constant in time.

The temperature physics itself is validated on the dynamic leaves
(test_prms_stream_temp*.py); this leaf changes ONLY the shade source.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_stream_temp import PRMSStreamTempConstantShade
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output_stream_temp"

INPUT_NAMES = (
    "seg_outflow",
    "seg_lateral_inflow",
    "seg_flow_width",
    "seg_tave_air",
    "seg_humid",
    "seg_ccov",
    "seg_melt",
    "seg_rain",
    "seg_potet",
    "seginc_swrad",
    "seginc_sroff",
    "seginc_ssflow",
    "seginc_gwflow",
)
OUT_NAMES = ("seg_shade", "seg_tave_water")

_needed = [
    DOMAIN_DIR / "parameters_PRMSStreamTemp.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    GEN_DIR / "seg_tave_water.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in INPUT_NAMES]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr nhm_stream_temp data not generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def segshade():
    """Synthetic per-segment seasonal shade fractions (values VARY by
    segment so the selection pins per-segment indexing, not just the
    seasonal switch)."""
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    nsegment = dis_seg.sizes["nsegment"]
    return {
        "segshade_sum": np.linspace(0.05, 0.45, nsegment),
        "segshade_win": np.linspace(0.02, 0.20, nsegment),
    }


@pytest.fixture(scope="module")
def model_run(segshade, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stream_temp_const_shade_output")
    st = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc")
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
    parameters = xr.merge(
        [st, dis_seg[["lat_temp_adj"]], dis_hru[["hru_area"]]],
        compat="no_conflicts",
    )
    for name, values in segshade.items():
        parameters[name] = (("nsegment",), values)

    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_seg": "nsegment"}
        )
        for nn in INPUT_NAMES
    }
    process_dict = {
        "prms_stream_temp": {
            "class": PRMSStreamTempConstantShade,
            "discretization": "nsegment",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(OUT_NAMES),
        "output_serial_zarr": out_dir / "stream_temp_const_shade.zarr",
        "time_chunk_size": 61,
    }
    discretizations = {
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(np.float64(1.0), np.int32(model.ntime))
    output_ds = xr.load_dataset(
        control["output_serial_zarr"], engine="zarr", consolidated=False
    )
    return {"model": model, "output_ds": output_ds}


class TestPRMSStreamTempConstantShade:
    def test_seasonal_shade_selection_exact(self, model_run, segshade):
        output_ds = model_run["output_ds"]
        times = output_ds["time"].values.astype("datetime64[D]")
        doys = (
            times - times.astype("datetime64[Y]").astype("datetime64[D]")
        ).astype(int) + 1
        summer = (doys >= 121) & (doys <= 273)
        expected = np.where(
            summer[:, None],
            segshade["segshade_sum"][None, :],
            segshade["segshade_win"][None, :],
        )
        np.testing.assert_array_equal(output_ds["seg_shade"].values, expected)

    def test_svi_never_written(self, model_run):
        proc = model_run["model"].model_dict["prms_stream_temp"]
        assert (proc._seg_svi == 0.0).all()

    def test_never_flow_structure_unchanged(self, model_run):
        """The NaN (never-has-flow) column set comes from the shared
        base initialize -- must match the dynamic answers' constant
        sentinel columns and be constant in time here too."""
        actual = model_run["output_ds"]["seg_tave_water"].values
        nan_cols = np.isnan(actual).all(axis=0)
        assert (np.isnan(actual) == nan_cols[None, :]).all()
        dynamic = xr.load_dataarray(GEN_DIR / "seg_tave_water.nc").values
        # Fortran writes numeric sentinels at never-flow segments; the
        # sentinel there is constant over the whole run
        dyn_never_flow = (dynamic == dynamic[0, :]).all(axis=0) & (
            dynamic[0, :] <= -98.0
        )
        np.testing.assert_array_equal(nan_cols, dyn_never_flow)
