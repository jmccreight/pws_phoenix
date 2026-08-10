"""Serial regression: hydraulic geometry vs pywatershed answers.

Runs the drb_2yr nhm_stream_temp configuration's seg_outflow through
PRMSHydraulicGeometryWidthOnly (the class the NHM parameter files
support: width parameters only, strmflow_character depth defaults)
and compares all five variables against the generated answers at
pywatershed's own 1e-5 standard. Note upstream's own test effectively
validates only seg_res_time (skip_missing_ans + the PRMS/pywatershed
naming split); the converted answers here carry the pywatershed names
(seg_flow_*), so all five are validated.

Also pins the declaration-override seam: PRMSHydraulicGeometryFull
fed depth_alpha = 0.27 / depth_m = 0.39 as explicit parameters must
be bit-identical to WidthOnly's built-in defaults.

Requires drb_2yr with GENERATED nhm_stream_temp answers
(output_stream_temp/); skips with a clear reason if absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometryFull,
    PRMSHydraulicGeometryWidthOnly,
)
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output_stream_temp"

ANSWER_NAMES = (
    "seg_flow_width",
    "seg_flow_depth",
    "seg_flow_area",
    "seg_flow_velocity",
    "seg_res_time",
)
# pywatershed's own standard for the stream-temp family
RTOL = ATOL = 1.0e-5

_needed = [
    DOMAIN_DIR / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    GEN_DIR / "seg_outflow.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in ANSWER_NAMES]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr nhm_stream_temp data not generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in ANSWER_NAMES
    }


def _run_one(proc_class, parameters, out_dir):
    seg_outflow = xr.load_dataarray(GEN_DIR / "seg_outflow.nc").rename(
        {"nhm_seg": "nsegment"}
    )
    process_dict = {
        "hydraulic_geometry": {
            "class": proc_class,
            "discretization": "nsegment",
            "parameters": parameters,
            "seg_outflow": seg_outflow,
        },
    }
    control = {
        "output_var_names": list(ANSWER_NAMES),
        "output_serial_zarr": out_dir / f"{proc_class.__name__}.zarr",
        "time_chunk_size": 61,
    }
    discretizations = {
        "nsegment": Discretization(
            ["nsegment"], parameters=DOMAIN_DIR / "parameters_dis_seg.nc"
        ),
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(np.float64(1.0), np.int32(model.ntime))
    output = xr.load_dataset(
        control["output_serial_zarr"], engine="zarr", consolidated=False
    )
    return output


@pytest.fixture(scope="module")
def width_only_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("hydraulic_geometry_output")
    parameters = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSHydraulicGeometryWidthOnly.nc"
    )
    return _run_one(PRMSHydraulicGeometryWidthOnly, parameters, out_dir)


class TestPRMSHydraulicGeometry:
    # ============ TESTS ============

    def test_all_variables_all_timesteps(self, width_only_run, answers):
        """Every variable matches pywatershed over the full run."""
        for nn in ANSWER_NAMES:
            np.testing.assert_allclose(
                width_only_run[nn].values,
                answers[nn].values,
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    def test_full_with_default_depth_params_identical(
        self, width_only_run, tmp_path_factory
    ):
        """The declaration-override seam: Full fed the default depth
        values as explicit parameters == WidthOnly, bit for bit."""
        out_dir = tmp_path_factory.mktemp("hydraulic_geometry_full")
        parameters = xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSHydraulicGeometryWidthOnly.nc"
        )
        nseg = parameters.sizes["nsegment"]
        parameters["depth_alpha"] = (
            ("nsegment",),
            np.full(nseg, 0.27, dtype=np.float64),
        )
        parameters["depth_m"] = (
            ("nsegment",),
            np.full(nseg, 0.39, dtype=np.float64),
        )
        full_run = _run_one(PRMSHydraulicGeometryFull, parameters, out_dir)
        for nn in ANSWER_NAMES:
            np.testing.assert_array_equal(
                full_run[nn].values,
                width_only_run[nn].values,
                err_msg=f"variable '{nn}' differs (Full vs WidthOnly)",
            )
