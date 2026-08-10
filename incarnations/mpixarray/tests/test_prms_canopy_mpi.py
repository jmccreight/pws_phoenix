"""MPI regression: the ported PRMSCanopy via ModelMPI vs pywatershed.

Same drb_2yr domain and answers as the serial test
(test_prms_canopy.py), distributed over the "nhru" grid with the
mpixarray streaming pipeline. Rank 0 assembles the ONE combined input
file (the 7 params + the 8 forcings incl. the MUTABLE pptmix);
`net_rain` streams to parallel NetCDF and is validated globally over
all timesteps; a representative set of the remaining variables is
validated from final state, gathered rank-ordered.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_canopy_mpi.py -v

rtol = atol = 1e-12 matches pywatershed's own canopy autotest standard.
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
from hydrology.prms_canopy import PRMSCanopy
from model import ModelMPI

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_NAMES = (
    "pk_ice_prev",
    "freeh2o_prev",
    "transp_on",
    "hru_ppt",
    "hru_rain",
    "hru_snow",
    "potet",
    "pptmix",
)
STREAMED_NAME = "net_rain"  # the one to_netcdf var (mpixarray limit)
FINAL_STATE_NAMES = (
    "net_ppt",
    "net_snow",
    "intcp_stor",
    "intcp_evap",
    "hru_intcpevap",
    "hru_intcpstor",
    "intcp_changeover",
)
# pywatershed's own canopy autotest comparison standard
RTOL = ATOL = 1.0e-12

_needed = [
    DOMAIN_DIR / "parameters_PRMSCanopy.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in INPUT_NAMES + FINAL_STATE_NAMES + (STREAMED_NAME,)
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
def mpi_paths():
    """Rank 0 assembles the ONE combined input file and broadcasts the
    temp dir; rank 0 cleans up afterward (no barrier in teardown)."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "prms_canopy_mpi_data"
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        proc_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSCanopy.nc")
        forcings = [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
            for nn in INPUT_NAMES
        ]
        combined = xr.merge(
            [proc_params, *forcings],  # exactly the 7 declared params
            compat="no_conflicts",
        )
        combined = combined.assign_coords(
            nhru=np.arange(combined.sizes["nhru"])
        )
        combined.to_netcdf(input_file)
    comm.Barrier()
    yield {"input_file": input_file, "output_file": output_file}

    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def answers():
    """Ground truth read straight from pywatershed's answer files --
    collective-free, every rank reads identically."""
    names = FINAL_STATE_NAMES + (STREAMED_NAME,)
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in names}


@pytest.fixture(scope="module")
def mpi_run(mpi_paths):
    """Build + run + finalize ModelMPI ONCE; every collective lives here."""
    comm = MPI.COMM_WORLD
    process_dict = {
        "prms_canopy": {
            "class": PRMSCanopy,
            "discretization": "nhru",
        },
    }
    control = {
        "input_file": mpi_paths["input_file"],
        "output_parallel_netcdf": mpi_paths["output_file"],
        "output_var_names": [STREAMED_NAME],
        "mpi_grid": "nhru",
    }
    model = ModelMPI(process_dict, control)
    model.run(np.float64(1.0))

    final = {
        nn: np.concatenate(
            comm.allgather(model._ds_mpi_stream[nn].values.copy())
        )
        for nn in FINAL_STATE_NAMES
    }
    model.finalize()
    comm.Barrier()  # output file fully flushed before reads
    return {"output_file": mpi_paths["output_file"], "final": final}


@pytest.mark.mpi(min_size=2)
class TestPRMSCanopyMPI:
    # -- streamed net_rain over all timesteps (global, rank 0) --
    def test_streamed_net_rain_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_file"]) as ds_out:
            rain_out = ds_out[f"{STREAMED_NAME}_out"].values
        np.testing.assert_allclose(
            rain_out,
            answers[STREAMED_NAME].values,
            rtol=RTOL,
            atol=ATOL,
        )

    # -- final state of everything else, gathered globally --
    def test_final_state_gathered(self, mpi_run, answers):
        for nn, vals in mpi_run["final"].items():
            np.testing.assert_allclose(
                vals,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' final state differs",
            )
