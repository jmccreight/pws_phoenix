"""ModelMPI window depth: control["mpi_window_steps"].

mpixarray's time window can be deeper than one step. ModelMPI.run
then reads the streamed inputs once per window, runs the days of the
window in an inner loop, and writes the streamed outputs once per
window. The answers must not depend on the depth, bit for bit: the
same kernels see the same rows in the same order.

The complete NHM + stream-temperature model on drb_2yr, distributed
over the hru grid, is run three times over the whole record: depth 1
(the reference), depth 7 (it does not divide the record, so the last
window is shallower) and depth None (one window = the whole record).
Compared with depth 1: the final values of every variable on every
rank (both grids), every step of two streamed variables in the NetCDF
file, and every step of one segment variable in the rank-0 zarr
store.

A depth below 1, above the record length, or not an integer raises
at build, on every rank.

All collective MPI ops live in the module-scoped fixture; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_window_depth_mpi.py -v
"""

import pathlib as pl
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI

from conftest import PWS_TEST_DATA
from pws_phoenix.prms_translate import (
    assemble_from_control,
    mpi_model_from_control,
    write_mpi_input_file,
)

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
CONTROL_FILE = DRB_DIR / "nhm_stream_temp.control"

DT = np.float64(86400.0)

REFERENCE_DEPTH = 1
# 7 does not divide the record (asserted below); None = the whole record
TESTED_DEPTHS = [7, None]
STREAMED_NAMES = ["sroff", "gwres_flow"]
SERIAL_NAME = "seg_outflow"
BAD_DEPTH_TAGS = ["zero", "too_deep", "not_integer"]

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


def _control(data_dir: pl.Path, tag: str, depth: Any) -> dict[str, Any]:
    return {
        "input_file": data_dir / "mpi_input.nc",
        "output_parallel_netcdf": data_dir / f"output_{tag}.nc",
        "output_serial_zarr": data_dir / f"output_{tag}.zarr",
        "output_var_names": STREAMED_NAMES + [SERIAL_NAME],
        "mpi_grid": "nhru",
        "mpi_window_steps": depth,
    }


@pytest.fixture(scope="module")
def window_runs():
    """Prep (rank 0), one run per depth (all ranks), then the builds
    that must raise (all ranks); every collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp)

    if comm.rank == 0:
        kit = assemble_from_control(CONTROL_FILE)
        # n_days=None: the whole record
        write_mpi_input_file(kit, data_dir / "mpi_input.nc")
    comm.Barrier()

    runs: dict[Any, dict[str, Any]] = {}
    for depth in [REFERENCE_DEPTH] + TESTED_DEPTHS:
        control = _control(data_dir, f"depth_{depth}", depth)
        model = mpi_model_from_control(CONTROL_FILE, control=control)
        model.run(DT)
        # this rank's final values of every variable, both grids
        state: dict[str, np.ndarray] = {}
        for proc_name, proc in model.model_dict.items():
            grid = model._proc_grid[proc_name]
            dataset = model.discretizations[grid].dataset
            for name in type(proc).get_var_names(include_budget=True):
                state[f"{grid}/{name}"] = dataset[name].values.copy()
        runs[depth] = {
            "state": state,
            "window_steps": model._window_steps,
            "n_time": model._ntime,
            "output_file": control["output_parallel_netcdf"],
            "output_zarr": control["output_serial_zarr"],
        }
        model.finalize()
        comm.Barrier()

    # Builds that must raise (ModelMPI.__init__ closes the input file
    # they opened: tests/test_bad_build_mpi.py).
    n_time = runs[REFERENCE_DEPTH]["n_time"]
    bad_depths = dict(zip(BAD_DEPTH_TAGS, [0, n_time + 1, 2.5]))
    errors: dict[str, str] = {}
    for tag, bad in bad_depths.items():
        try:
            mpi_model_from_control(
                CONTROL_FILE, control=_control(data_dir, tag, bad)
            )
        except ValueError as err:
            errors[tag] = str(err)
    comm.Barrier()

    yield {"runs": runs, "n_time": n_time, "errors": errors}
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestWindowDepth:
    def test_depths_resolved(self, window_runs):
        runs = window_runs["runs"]
        n_time = window_runs["n_time"]
        assert runs[REFERENCE_DEPTH]["window_steps"] == 1
        assert runs[7]["window_steps"] == 7
        assert runs[None]["window_steps"] == n_time
        # depth 7 must leave a shallower last window
        assert n_time % 7 != 0

    @pytest.mark.parametrize("depth", TESTED_DEPTHS)
    def test_final_state_bit_identical(self, window_runs, depth):
        """Every variable of both grids, this rank's values."""
        reference = window_runs["runs"][REFERENCE_DEPTH]["state"]
        state = window_runs["runs"][depth]["state"]
        assert state.keys() == reference.keys()
        for name, values in reference.items():
            np.testing.assert_array_equal(state[name], values, err_msg=name)

    @pytest.mark.parametrize("depth", TESTED_DEPTHS)
    def test_streamed_output_bit_identical(self, window_runs, depth):
        """Every step in the NetCDF file: one write per window lands
        each row where the per-step writes of depth 1 do."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = window_runs["runs"]
        ds_ref = xr.load_dataset(runs[REFERENCE_DEPTH]["output_file"])
        ds_out = xr.load_dataset(runs[depth]["output_file"])
        assert ds_out.sizes["time"] == window_runs["n_time"]
        np.testing.assert_array_equal(
            ds_out["time"].values, ds_ref["time"].values
        )
        for name in STREAMED_NAMES:
            buf = f"{name}_out"
            np.testing.assert_array_equal(
                ds_out[buf].values, ds_ref[buf].values, err_msg=name
            )

    @pytest.mark.parametrize("depth", TESTED_DEPTHS)
    def test_serial_zarr_bit_identical(self, window_runs, depth):
        """Every step of a segment variable in the rank-0 zarr store
        (collected per day inside the window)."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = window_runs["runs"]
        with (
            xr.open_zarr(
                runs[REFERENCE_DEPTH]["output_zarr"], consolidated=False
            ) as ds_ref,
            xr.open_zarr(
                runs[depth]["output_zarr"], consolidated=False
            ) as ds_out,
        ):
            assert ds_out.sizes["time"] == window_runs["n_time"]
            np.testing.assert_array_equal(
                ds_out[SERIAL_NAME].values, ds_ref[SERIAL_NAME].values
            )

    @pytest.mark.parametrize("tag", BAD_DEPTH_TAGS)
    def test_bad_depth_raises(self, window_runs, tag):
        assert "mpi_window_steps" in window_runs["errors"].get(tag, "")
