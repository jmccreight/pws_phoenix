"""MPI regression: the ported PRMSRunoff via ModelMPI vs pywatershed.

Same drb_2yr domain and answers as the serial test
(test_prms_runoff.py), distributed over the "nhru" grid with the
mpixarray streaming pipeline. Rank 0 assembles the ONE combined input
file (params + dis vars + the 14 forcings); `sroff` streams to
parallel NetCDF and is validated globally over all timesteps; a
representative set of the remaining variables is validated from final
state, gathered rank-ordered. PRMSRunoff's initialize()
(basin_init/dprst_init) is purely per-element, so each rank derives
its own rows' geometry -- a first: the other MPI-tested processes have
no init hook (groundwater) or run replicated (channel).

drb_2yr has 765 HRUs -- uneven over 4 ranks (192/191/191/191), the
deliberate uneven-decomposition probe.

All collective MPI ops live in module-scoped fixtures; test methods
are pure asserts (see test_up_low_regression_mpi.py for the pattern).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_runoff_mpi.py -v

rtol = atol = 1e-10 matches pywatershed's own runoff autotest standard
(see the serial test docstring).
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
from hydrology.prms_runoff import PRMSRunoff
from model import ModelMPI

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_NAMES = (
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
DIS_NAMES = ("hru_type", "hru_area", "hru_in_to_cf")
STREAMED_NAME = "sroff"  # the one to_netcdf var (mpixarray limit)
FINAL_STATE_NAMES = (
    "sroff_vol",
    "infil",
    "contrib_fraction",
    "imperv_stor",
    "hru_impervstor",
    "dprst_vol_open",
    "dprst_stor_hru",
    "dprst_seep_hru",
)
# pywatershed's own runoff autotest comparison standard
RTOL = ATOL = 1.0e-10

_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoff.nc",
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
    """Rank 0 assembles the ONE combined input file (all 20 process
    params + dis vars + forcings) and broadcasts the temp dir; rank 0
    cleans up afterward (no barrier in teardown -- see
    test_up_low_regression_mpi.py)."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "prms_runoff_mpi_data"
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        proc_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSRunoff.nc")
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        # pywatershed output files put forcings on the "nhm_id" dim;
        # the parameter files use "nhru" -- unify on the grid dim
        forcings = [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
            for nn in INPUT_NAMES
        ]
        combined = xr.merge(
            [
                proc_params,  # exactly the 20 declared process params
                dis_hru[list(DIS_NAMES)],
                *forcings,
            ],
            compat="no_conflicts",
        )
        # mpixarray parallelize/set_streaming need real dim-coordinates
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
        "prms_runoff": {
            "class": PRMSRunoff,
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

    # Final state of the non-streamed vars, gathered globally (single
    # scheme -> rank-ordered contiguous blocks; sizes may be uneven).
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
class TestPRMSRunoffMPI:
    # -- streamed sroff over all timesteps (global, rank 0) --
    def test_streamed_sroff_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_file"]) as ds_out:
            sroff_out = ds_out[f"{STREAMED_NAME}_out"].values
        np.testing.assert_allclose(
            sroff_out,
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
