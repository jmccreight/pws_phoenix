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

The order of an input's two dims must not change the answers either:
depths 7 and None are run again from the same input written on
(nhru, time) (write_mpi_input_file(transpose=True)) and compared with
the same reference, and the model is built (not run) from that file
at depth 1.

Each build warns once, on rank 0 only, when the streamed inputs are
stored in the slower order for its depth (data_io.slow_order_note):
the whole record from the (time, nhru) file always does.

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
import warnings
from typing import Any

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI

from conftest import PWS_TEST_DATA
from pws_phoenix.data_io import READ_PIECES_WARN_RATIO, read_pieces
from pws_phoenix.prms_translate import (
    assemble_from_control,
    mpi_model_from_control,
    write_mpi_input_file,
)

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
CONTROL_FILE = DRB_DIR / "nhm_stream_temp.control"

DT = np.float64(86400.0)

# a run is named by (the order of the streamed inputs' dims in the
# input file, the window depth)
TIME_FIRST = "time_first"  # (time, nhru)
GRID_FIRST = "grid_first"  # (nhru, time)
INPUT_FILES = {
    TIME_FIRST: "mpi_input.nc",
    GRID_FIRST: "mpi_input_transposed.nc",
}
REFERENCE = (TIME_FIRST, 1)
# 7 does not divide the record (asserted below); None = the whole record
TESTED = [
    (TIME_FIRST, 7),
    (TIME_FIRST, None),
    (GRID_FIRST, 7),
    (GRID_FIRST, None),
]
# built, not run
BUILT_ONLY = (GRID_FIRST, 1)
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


def _control(
    data_dir: pl.Path, tag: str, depth: Any, order: str = TIME_FIRST
) -> dict[str, Any]:
    return {
        "input_file": data_dir / INPUT_FILES[order],
        "output_parallel_netcdf": data_dir / f"output_{tag}.nc",
        "output_serial_zarr": data_dir / f"output_{tag}.zarr",
        "output_var_names": STREAMED_NAMES + [SERIAL_NAME],
        "mpi_grid": "nhru",
        "mpi_window_steps": depth,
    }


def _build(data_dir: pl.Path, order: str, depth: Any) -> tuple[Any, dict]:
    """Build the model of one run (collective). Returns it with its
    control and what the build showed: the slow-order warnings raised
    on this rank, rank 0's block, the size of the grid and the order
    each streamed input was found in."""
    comm = MPI.COMM_WORLD
    control = _control(data_dir, f"{order}_depth_{depth}", depth, order)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = mpi_model_from_control(CONTROL_FILE, control=control)
    start, stop = model._decomp_slice
    return model, {
        "control": control,
        "n_slow_order_warnings": sum(
            "slower order" in str(ww.message) for ww in caught
        ),
        "n_block_rank_0": comm.bcast(stop - start, root=0),
        "n_grid": comm.allreduce(stop - start, op=MPI.SUM),
        "stream_time_first": dict(model._stream_time_first),
        "window_steps": model._window_steps,
        "n_time": model._ntime,
    }


@pytest.fixture(scope="module")
def window_runs():
    """Prep (rank 0), one run per order and depth (all ranks), one
    build without a run, then the builds that must raise (all ranks);
    every collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp)

    if comm.rank == 0:
        kit = assemble_from_control(CONTROL_FILE)
        # n_days=None: the whole record
        write_mpi_input_file(kit, data_dir / INPUT_FILES[TIME_FIRST])
        write_mpi_input_file(
            kit, data_dir / INPUT_FILES[GRID_FIRST], transpose=True
        )
    comm.Barrier()

    runs: dict[Any, dict[str, Any]] = {}
    for order, depth in [REFERENCE] + TESTED:
        model, built = _build(data_dir, order, depth)
        control = built["control"]
        model.run(DT)
        # this rank's final values of every variable, both grids
        state: dict[str, np.ndarray] = {}
        for proc_name, proc in model.model_dict.items():
            grid = model._proc_grid[proc_name]
            dataset = model.discretizations[grid].dataset
            for name in type(proc).get_var_names(include_budget=True):
                state[f"{grid}/{name}"] = dataset[name].values.copy()
        runs[(order, depth)] = built | {
            "state": state,
            "output_file": control["output_parallel_netcdf"],
            "output_zarr": control["output_serial_zarr"],
        }
        model.finalize()
        comm.Barrier()

    model, built = _build(data_dir, *BUILT_ONLY)
    runs[BUILT_ONLY] = built
    model.finalize()
    comm.Barrier()

    # Builds that must raise (ModelMPI.__init__ closes the input file
    # they opened: tests/test_bad_build_mpi.py).
    n_time = runs[REFERENCE]["n_time"]
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
        assert runs[REFERENCE]["window_steps"] == 1
        for order in (TIME_FIRST, GRID_FIRST):
            assert runs[(order, 7)]["window_steps"] == 7
            assert runs[(order, None)]["window_steps"] == n_time
        assert runs[BUILT_ONLY]["window_steps"] == 1
        # depth 7 must leave a shallower last window
        assert n_time % 7 != 0

    @pytest.mark.parametrize(
        ("order", "depth"), [REFERENCE] + TESTED + [BUILT_ONLY]
    )
    def test_input_order_found(self, window_runs, order, depth):
        """Every streamed input is taken in the order its file has."""
        found = window_runs["runs"][(order, depth)]["stream_time_first"]
        assert found
        assert set(found.values()) == {order == TIME_FIRST}

    @pytest.mark.parametrize(
        ("order", "depth"), [REFERENCE] + TESTED + [BUILT_ONLY]
    )
    def test_slow_order_warning(self, window_runs, order, depth):
        """One warning on rank 0, none elsewhere, and only when the
        file's order takes READ_PIECES_WARN_RATIO times the pieces of
        the other order for rank 0's block and this depth."""
        run = window_runs["runs"][(order, depth)]
        n_time = window_runs["n_time"]
        pieces = {
            time_first: read_pieces(
                run["n_block_rank_0"],
                run["n_grid"],
                run["window_steps"],
                n_time,
                time_first,
            )
            for time_first in (True, False)
        }
        time_first = order == TIME_FIRST
        slow = (
            pieces[time_first]
            >= READ_PIECES_WARN_RATIO * pieces[not time_first]
        )
        expected = 1 if (slow and MPI.COMM_WORLD.rank == 0) else 0
        assert run["n_slow_order_warnings"] == expected
        if (order, depth) == (TIME_FIRST, None):
            # the whole record from the (time, nhru) file: n_time
            # pieces against 1
            assert n_time >= READ_PIECES_WARN_RATIO
            assert slow

    @pytest.mark.parametrize(("order", "depth"), TESTED)
    def test_final_state_bit_identical(self, window_runs, order, depth):
        """Every variable of both grids, this rank's values."""
        reference = window_runs["runs"][REFERENCE]["state"]
        state = window_runs["runs"][(order, depth)]["state"]
        assert state.keys() == reference.keys()
        for name, values in reference.items():
            np.testing.assert_array_equal(state[name], values, err_msg=name)

    @pytest.mark.parametrize(("order", "depth"), TESTED)
    def test_streamed_output_bit_identical(self, window_runs, order, depth):
        """Every step in the NetCDF file: one write per window lands
        each row where the per-step writes of depth 1 do."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = window_runs["runs"]
        ds_ref = xr.load_dataset(runs[REFERENCE]["output_file"])
        ds_out = xr.load_dataset(runs[(order, depth)]["output_file"])
        assert ds_out.sizes["time"] == window_runs["n_time"]
        np.testing.assert_array_equal(
            ds_out["time"].values, ds_ref["time"].values
        )
        for name in STREAMED_NAMES:
            buf = f"{name}_out"
            np.testing.assert_array_equal(
                ds_out[buf].values, ds_ref[buf].values, err_msg=name
            )

    @pytest.mark.parametrize(("order", "depth"), TESTED)
    def test_serial_zarr_bit_identical(self, window_runs, order, depth):
        """Every step of a segment variable in the rank-0 zarr store
        (collected per day inside the window)."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = window_runs["runs"]
        with (
            xr.open_zarr(
                runs[REFERENCE]["output_zarr"], consolidated=False
            ) as ds_ref,
            xr.open_zarr(
                runs[(order, depth)]["output_zarr"], consolidated=False
            ) as ds_out,
        ):
            assert ds_out.sizes["time"] == window_runs["n_time"]
            np.testing.assert_array_equal(
                ds_out[SERIAL_NAME].values, ds_ref[SERIAL_NAME].values
            )

    @pytest.mark.parametrize("tag", BAD_DEPTH_TAGS)
    def test_bad_depth_raises(self, window_runs, tag):
        assert "mpi_window_steps" in window_runs["errors"].get(tag, "")
