"""Serial regression: the ported PRMSStreamTemp vs pywatershed.

Runs the drb_2yr nhm_stream_temp configuration through the serial
Model: the segment-grid temperature physics with dynamic shade, fed
the routing/hydraulic-geometry inputs AND the HRU-derived aggregates
(seg_tave_air ... seginc_*) from the generated answers (the
aggregation itself is the chain stage -- see the module docstring).
Compared at pywatershed's OWN stream-temp family standard,
rtol = atol = 5e-3 (its comment: iteration-loop and trig noise "just
above 32-bit precision" vs the Fortran answers; errors don't grow
with time).

seg_tave_water sentinel handling: our port keeps upstream python's
NaN for never-has-flow segments; the Fortran answers carry numeric
sentinels there. Comparison masks to segments where OUR value is
finite and asserts the never-flow set is constant over the run.

Requires drb_2yr with GENERATED nhm_stream_temp answers; skips with a
clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_stream_temp import PRMSStreamTemp
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
ANSWER_NAMES = (
    "seg_tave_water",
    "seg_tave_upstream",
    "seg_tave_gw",
    "seg_tave_ss",
    "seg_tave_lat",
    "seg_shade",
)
# pywatershed's own stream-temp family standard (see module docstring)
RTOL = ATOL = 5.0e-3

_needed = [
    DOMAIN_DIR / "parameters_PRMSStreamTemp.nc",
    DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in INPUT_NAMES + ANSWER_NAMES]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr nhm_stream_temp data not generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def parameters():
    """Process + dynamic-shade parameters, hru_area (never-flow walk),
    and lat_temp_adj (2-D, supplied here rather than via the dis)."""
    st = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc")
    shade = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
    )
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
    return xr.merge(
        [
            st,
            shade,
            dis_seg[["lat_temp_adj"]],
            dis_hru[["hru_area"]],
        ],
        compat="no_conflicts",
    )


@pytest.fixture(scope="module")
def answers():
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in ANSWER_NAMES}


@pytest.fixture(scope="module")
def model_run(parameters, tmp_path_factory):
    """Build + run + finalize the Model once for the module."""
    out_dir = tmp_path_factory.mktemp("prms_stream_temp_output")
    forcings = {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename(
            {"nhm_seg": "nsegment"}
        )
        for nn in INPUT_NAMES
    }
    process_dict = {
        "prms_stream_temp": {
            "class": PRMSStreamTemp,
            "discretization": "nsegment",
            "parameters": parameters,
            **forcings,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / "prms_stream_temp.zarr",
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
    return {"model": model, "control": control}


class TestPRMSStreamTemp:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(self, model_run, answers):
        """Every output variable matches pywatershed over the full run
        (masked to finite values of ours for seg_tave_water)."""
        output_ds = xr.load_dataset(
            model_run["control"]["output_serial_zarr"],
            engine="zarr",
            consolidated=False,
        )
        for nn in ANSWER_NAMES:
            actual = output_ds[nn].values
            desired = answers[nn].values
            finite = np.isfinite(actual)
            if nn == "seg_tave_water":
                # never-flow segments: ours NaN by design (upstream
                # python), Fortran writes numeric sentinels -- the NaN
                # column set must be constant in time
                nan_cols = np.isnan(actual).all(axis=0)
                assert (np.isnan(actual) == nan_cols[None, :]).all()
            np.testing.assert_allclose(
                actual[finite],
                desired[finite],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_final_state(self, model_run, answers):
        proc = model_run["model"].model_dict["prms_stream_temp"]
        for nn in ("seg_tave_water", "seg_tave_gw", "seg_shade"):
            actual = proc[nn].values
            desired = answers[nn].values[-1, :]
            finite = np.isfinite(actual)
            np.testing.assert_allclose(
                actual[finite],
                desired[finite],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"final '{nn}' differs",
            )
