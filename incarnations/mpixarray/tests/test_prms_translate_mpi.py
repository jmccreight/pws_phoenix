"""Legacy PRMS -> PARALLEL run (prms_translate.assemble_mpi).

The machinery: write_mpi_input_file (rank-0 prep from the serial kit)
+ mpi_model_from_control (the SPMD one-liner) build the complete NHM
+ stream-temp model distributed over the hru grid, replicated on the
segment grid. Validation vs the SERIAL one-liner on the same window:
hru-local physics (sroff) must be BIT-identical per rank (no cross-hru
communication touches it); the segment grid rides the MapMPI
Allreduce, whose summation order differs from the serial matmul, so
its comparison carries a tiny float tolerance.

All collective MPI ops live in the module-scoped fixtures; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_translate_mpi.py -v
"""

import pathlib as pl
import shutil
import sys
import tempfile

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, str(pl.Path(__file__).parent.parent))
pytest.importorskip("pyPRMS", reason="pyPRMS not installed")

from prms_translate import (  # noqa: E402
    assemble_from_control,
    model_from_control,
    mpi_model_from_control,
    write_mpi_input_file,
)

MPIX_ROOT = pl.Path(__file__).parents[4]
DRB_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
CONTROL_FILE = DRB_DIR / "nhm_stream_temp.control"

N_DAYS = 5
DT = np.float64(86400.0)

_needed = [
    CONTROL_FILE,
    DRB_DIR / "myparam.param",
    DRB_DIR / "prcp.cbh",
    DRB_DIR / "rhavg.cbh",
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason="drb_2yr legacy files missing: " + ", ".join(_missing[:3]),
)


@pytest.fixture(scope="module")
def mpi_run():
    """Prep (rank 0) + serial reference (rank 0, broadcast) + the
    parallel run (all ranks); every collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp)
    input_file = data_dir / "mpi_input.nc"

    serial_sroff = serial_seg_tave = None
    if comm.rank == 0:
        kit = assemble_from_control(CONTROL_FILE)
        write_mpi_input_file(kit, input_file, n_days=N_DAYS)
        with model_from_control(CONTROL_FILE) as serial_model:
            serial_model.run(DT, np.int32(N_DAYS))
            serial_sroff = serial_model.model_dict["prms_runoff"][
                "sroff"
            ].values.copy()
            serial_seg_tave = serial_model.model_dict["prms_stream_temp"][
                "seg_tave_water"
            ].values.copy()
    serial_sroff = comm.bcast(serial_sroff, root=0)
    serial_seg_tave = comm.bcast(serial_seg_tave, root=0)
    comm.Barrier()

    model = mpi_model_from_control(
        CONTROL_FILE,
        control={
            "input_file": input_file,
            "output_parallel_netcdf": data_dir / "mpi_output.nc",
            "output_var_names": ["sroff"],
            "mpi_grid": "nhru",
        },
    )
    model.run(DT)
    aa, bb = model._decomp_slice
    local_sroff = model._ds_mpi_stream["sroff"].values.copy()
    seg_tave = (
        model.discretizations["nsegment"]
        .dataset["seg_tave_water"]
        .values.copy()
    )
    gathered = comm.allgather(seg_tave)
    replicated_identical = all(
        np.array_equal(gathered[0], gg, equal_nan=True)
        for gg in gathered
    )  # equal_nan: the never-flow segment is NaN on every rank
    model.finalize()
    comm.Barrier()

    yield {
        "local_sroff": local_sroff,
        "serial_sroff_block": serial_sroff[aa:bb],
        "seg_tave": seg_tave,
        "serial_seg_tave": serial_seg_tave,
        "replicated_identical": replicated_identical,
    }
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestLegacyToParallel:
    def test_hru_local_bit_identical(self, mpi_run):
        """No cross-hru comm touches sroff: exact per-rank block."""
        np.testing.assert_array_equal(
            mpi_run["local_sroff"], mpi_run["serial_sroff_block"]
        )

    def test_segment_grid_matches_serial(self, mpi_run):
        """MapMPI Allreduce order != serial matmul order: tiny float
        tolerance on the segment grid."""
        finite = np.isfinite(mpi_run["serial_seg_tave"])
        assert (np.isfinite(mpi_run["seg_tave"]) == finite).all()
        np.testing.assert_allclose(
            mpi_run["seg_tave"][finite],
            mpi_run["serial_seg_tave"][finite],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_replicated_identical_across_ranks(self, mpi_run):
        assert mpi_run["replicated_identical"]
