"""MPI regression: the stream-temperature stack under ModelMPI.

The realistic MPI deployment shape for the stream-temp arc (segment
grids run REPLICATED -- the Step B pattern; there is no distributed
stream-temp code): a distributed "nhru" grid (PRMSGroundwater, the
proven standalone distributed process, streaming gwres_flow) alongside
the replicated serial "nsegment" grid running
PRMSHydraulicGeometryWidthOnly -> PRMSStreamTemp with **live**
seg_flow_width by structural sharing -- the first live pairing of
those two processes (the serial test feeds width from disk). The
grids are deliberately unconnected: the hru->segment aggregation is
the pending chain-stage design (see PORTS.md); stream temp's
HRU-derived aggregates come from the generated answers via serial
Input objects, exercising disk-fed serial-grid inputs under MPI.

Validation (rank 0, from disk after finalize): gwres_flow at
pywatershed's 1e-13; seg_flow_width / seg_res_time at the hydraulic
geometry standard 1e-5; the stream-temp variables at the family
standard 5e-3 (live width's 1e-5 noise is far inside it), with the
seg_tave_water never-flow NaN-column handling of the serial test.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_stream_temp_mpi.py -v
"""

import pathlib as pl
import shutil
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_groundwater import PRMSGroundwater
from hydrology.prms_hydraulic_geometry import PRMSHydraulicGeometryWidthOnly
from hydrology.prms_stream_temp import PRMSStreamTemp
from model import ModelMPI

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR_HRU = DOMAIN_DIR / "output"
GEN_DIR_SEG = DOMAIN_DIR / "output_stream_temp"

GW_INPUT_NAMES = ("soil_to_gw", "ssr_to_gw", "dprst_seep_hru")
# stream-temp disk inputs (seg_flow_width deliberately absent: LIVE)
ST_INPUT_NAMES = (
    "seg_outflow",
    "seg_lateral_inflow",
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
HG_ANSWER_NAMES = ("seg_flow_width", "seg_res_time")
ST_ANSWER_NAMES = (
    "seg_tave_water",
    "seg_tave_upstream",
    "seg_tave_gw",
    "seg_tave_ss",
    "seg_tave_lat",
    "seg_shade",
)
RTOL_GW = ATOL_GW = 1.0e-13
RTOL_HG = ATOL_HG = 1.0e-5
RTOL_ST = ATOL_ST = 5.0e-3

_needed = (
    [
        DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
        DOMAIN_DIR / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
        DOMAIN_DIR / "parameters_PRMSStreamTemp.nc",
        DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc",
        DOMAIN_DIR / "parameters_dis_hru.nc",
        DOMAIN_DIR / "parameters_dis_seg.nc",
    ]
    + [GEN_DIR_HRU / f"{nn}.nc" for nn in GW_INPUT_NAMES + ("gwres_flow",)]
    + [
        GEN_DIR_SEG / f"{nn}.nc"
        for nn in ST_INPUT_NAMES + HG_ANSWER_NAMES + ST_ANSWER_NAMES
    ]
)
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr (incl. nhm_stream_temp) data not "
        "generated; missing: " + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def mpi_paths():
    """Rank 0 assembles the distributed grid's ONE combined input file
    (groundwater params + dis vars + forcings) and broadcasts the temp
    dir; the serial zarr path lives in the same shared dir."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "prms_stream_temp_mpi_data"
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    serial_zarr = data_dir / "segment_output.zarr"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        proc_params = xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSGroundwater.nc"
        )
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        forcings = [
            xr.load_dataarray(GEN_DIR_HRU / f"{nn}.nc").rename(
                {"nhm_id": "nhru"}
            )
            for nn in GW_INPUT_NAMES
        ]
        combined = xr.merge(
            [
                proc_params[["gwflow_coef", "gwsink_coef", "gwstor_init"]],
                dis_hru[["hru_area", "hru_in_to_cf"]],
                *forcings,
            ],
            compat="no_conflicts",
        )
        combined = combined.assign_coords(
            nhru=np.arange(combined.sizes["nhru"])
        )
        combined.to_netcdf(input_file)
    comm.Barrier()
    yield {
        "input_file": input_file,
        "output_file": output_file,
        "serial_zarr": serial_zarr,
    }

    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def answers():
    """Ground truth straight from the answer files -- collective-free,
    every rank reads identically."""
    out = {"gwres_flow": xr.load_dataarray(GEN_DIR_HRU / "gwres_flow.nc")}
    for nn in HG_ANSWER_NAMES + ST_ANSWER_NAMES:
        out[nn] = xr.load_dataarray(GEN_DIR_SEG / f"{nn}.nc")
    return out


@pytest.fixture(scope="module")
def mpi_run(mpi_paths):
    """Build + run + finalize ModelMPI ONCE; every collective lives
    here."""
    comm = MPI.COMM_WORLD
    st_params = xr.merge(
        [
            xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc"),
            xr.load_dataset(
                DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
            ),
            xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")[
                ["lat_temp_adj"]
            ],
            xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")[
                ["hru_area"]
            ],
        ],
        compat="no_conflicts",
    )
    seg_forcings = {
        nn: xr.load_dataarray(GEN_DIR_SEG / f"{nn}.nc").rename(
            {"nhm_seg": "nsegment"}
        )
        for nn in ST_INPUT_NAMES
    }
    # seg_outflow is shared by both segment processes: fed ONCE, in the
    # first (hydraulic geometry) entry, shared structurally
    process_dict = {
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
        },
        "prms_hydraulic_geometry": {
            "class": PRMSHydraulicGeometryWidthOnly,
            "discretization": "nsegment",
            "parameters": DOMAIN_DIR
            / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
            "seg_outflow": seg_forcings.pop("seg_outflow"),
        },
        "prms_stream_temp": {
            "class": PRMSStreamTemp,
            "discretization": "nsegment",
            "parameters": st_params,
            **seg_forcings,
        },
    }
    control = {
        "input_file": mpi_paths["input_file"],
        "output_parallel_netcdf": mpi_paths["output_file"],
        "output_var_names": ["gwres_flow"]
        + list(HG_ANSWER_NAMES + ST_ANSWER_NAMES),
        "output_serial_zarr": mpi_paths["serial_zarr"],
        "time_chunk_size": 61,
        "mpi_grid": "nhru",
    }
    discretizations = {
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }
    model = ModelMPI(process_dict, control, discretizations=discretizations)
    model.run(np.float64(1.0))
    model.finalize()
    comm.Barrier()  # outputs fully flushed before reads
    return dict(mpi_paths)


@pytest.mark.mpi(min_size=2)
class TestPRMSStreamTempMPI:
    def test_streamed_gwres_flow_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_file"]) as ds_out:
            flow_out = ds_out["gwres_flow_out"].values
        np.testing.assert_allclose(
            flow_out,
            answers["gwres_flow"].values,
            rtol=RTOL_GW,
            atol=ATOL_GW,
        )

    def test_segment_grid_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        output_ds = xr.load_dataset(
            mpi_run["serial_zarr"], engine="zarr", consolidated=False
        )
        for nn in HG_ANSWER_NAMES + ST_ANSWER_NAMES:
            actual = output_ds[nn].values
            desired = answers[nn].values
            finite = np.isfinite(actual)
            if nn == "seg_tave_water":
                nan_cols = np.isnan(actual).all(axis=0)
                assert (np.isnan(actual) == nan_cols[None, :]).all()
            if nn in HG_ANSWER_NAMES:
                rtol, atol = RTOL_HG, ATOL_HG
            else:
                rtol, atol = RTOL_ST, ATOL_ST
            np.testing.assert_allclose(
                actual[finite],
                desired[finite],
                rtol=rtol,
                atol=atol,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )
