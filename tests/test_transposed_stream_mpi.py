"""mpixarray streams a (nhru, time) file as it streams a (time, nhru)
one, and hands each window back in the FILE's order.

This pins the mpixarray behavior ModelMPI relies on to take a streamed
input in either order of its two dims. The mpixarray calls are the
ones ModelMPI._build and ModelMPI.run make (pws_phoenix/model.py), in
their order, on a toy file written in both orders and streamed at
three window depths. ModelMPI itself on a transposed file is in
tests/test_window_depth_mpi.py.

The toy file: N_TIME days by N_HRU hrus, float32, value
1000 * day + hru, so every value names its own day and hru; plus one
(nhru,) parameter.

Per order and depth, on every rank:

- `step.mpi.src[name]` has the file's dims, and its values are this
  rank's block of the window's days: (days, block) from the
  (time, nhru) file, (block, days) from the (nhru, time) file;
- the declared (nhru,) buffer of the same name and the streamed
  (time, nhru) output buffer are the same for both files;
- the output file holds the input's values on (time, nhru).

No rank raises alone: a rank that catches an exception records its
text and the test reports it. If mpixarray raises on some ranks only,
in the middle of a collective call, the run hangs.

All collective MPI ops live in the module-scoped fixture; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_transposed_stream_mpi.py -v
"""

import pathlib as pl
import shutil
import tempfile
from typing import Any

import mpix
import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI
from mpixarray import open_dataset

from pws_phoenix.discretization import Discretization

N_TIME = 7
N_HRU = 10
GRID = "nhru"
DEPTHS = [1, 3, N_TIME]
# name of the order -> the dims of the forcing variable in the file
ORDERS = {
    "time_first": ("time", GRID),
    "hru_first": (GRID, "time"),
}
CASES = [(order, depth) for order in ORDERS for depth in DEPTHS]


def _expected(day0: int, day1: int, hru0: int, hru1: int) -> np.ndarray:
    """The (day, hru) values of days [day0, day1) and hrus
    [hru0, hru1)."""
    days = np.arange(day0, day1, dtype=np.float64)
    hrus = np.arange(hru0, hru1, dtype=np.float64)
    return 1000.0 * days[:, None] + hrus[None, :]


def _write_input(path: pl.Path, dims: tuple[str, str]) -> None:
    forcing = xr.DataArray(
        _expected(0, N_TIME, 0, N_HRU).astype(np.float32),
        dims=("time", GRID),
        coords={
            "time": np.datetime64("2000-01-01", "ns")
            + np.arange(N_TIME) * np.timedelta64(1, "D"),
            GRID: np.arange(N_HRU),
        },
    ).transpose(*dims)
    param = xr.DataArray(np.arange(N_HRU, dtype=np.float64), dims=(GRID,))
    xr.Dataset({"forcing": forcing, "param": param}).to_netcdf(path)


def _stream(path: pl.Path, out_path: pl.Path, depth: int) -> dict[str, Any]:
    """Stream `path` in windows of `depth` days. Returns what this
    rank saw; `problems` is empty when all was as expected."""
    result: dict[str, Any] = {
        "src_dims": [],
        "src_shapes": [],
        "window_days": [],
        "buffer_dims": None,
        "out_dims": None,
        "n_block": None,
        "problems": [],
    }
    problems: list[str] = result["problems"]
    opened: list[Any] = []
    try:
        comm = mpix.world_comm()
        ds_input = open_dataset(str(path), comm=comm)
        opened.append(ds_input)
        disc = Discretization([GRID], comm=comm)
        ds_mpi = disc.decompose(ds_input)
        comm = disc.comm
        stream = ds_mpi.mpi.as_iterable("time", chunk_size=depth, comm=comm)
        opened.append(stream)
        for name in list(stream.data_vars):
            if "time" not in stream[name].dims:
                stream[name] = stream[name].load()
        stream.mpi.open_writer(str(out_path), comm=comm)
        # the (nhru,) buffer has the file variable's name, as
        # ModelMPI's one-step input buffer has
        for name, dims, fill, to_file in (
            ("forcing", (GRID,), np.nan, False),
            ("forcing_out", ("time", GRID), 0.0, True),
        ):
            stream.mpi[name] = {
                "dims": dims,
                "dtype": np.float64,
                "fill_value": np.float64(fill),
                "to_file": to_file,
                "comm": comm,
            }
        stream.mpi.create(verbose=False)

        sizes = comm.allgather(int(stream.sizes[GRID]))
        hru0 = int(sum(sizes[: comm.rank]))
        hru1 = hru0 + int(sizes[comm.rank])
        result["n_block"] = hru1 - hru0

        day0 = 0
        for step in stream.mpi.iterate(comm=comm, progress=False):
            src = step.mpi.src["forcing"]
            window = src.values
            n_days = int(step.sizes["time"])
            values = step["forcing"].values
            out = step["forcing_out"].values
            result["src_dims"].append(tuple(src.dims))
            result["src_shapes"].append(window.shape)
            result["window_days"].append(n_days)
            result["buffer_dims"] = tuple(step["forcing"].dims)
            result["out_dims"] = tuple(step["forcing_out"].dims)
            want = _expected(day0, day0 + n_days, hru0, hru1)
            # a day of the window by the file's order, as ModelMPI.run
            # takes it
            time_first = src.dims[0] == "time"
            for dd in range(n_days):
                if time_first:
                    values[:] = window[dd, :]
                else:
                    values[:] = window[:, dd]
                if not np.array_equal(values, want[dd]):
                    problems.append(
                        f"day {day0 + dd}: the window's values are not "
                        "this block's"
                    )
                out[dd, :] = values
            step["forcing_out"].mpi.write()
            day0 += n_days
        if day0 != N_TIME:
            problems.append(f"the stream served {day0} of {N_TIME} days")
    except Exception as err:
        problems.append(f"raised {type(err).__name__}: {err}")
    # as ModelMPI._release: the stream (with its writer), then the input
    for dataset in reversed(opened):
        try:
            dataset.mpi.finalize()
        except Exception as err:
            problems.append(f"finalize raised {type(err).__name__}: {err}")
    return result


@pytest.fixture(scope="module")
def streamed():
    """Both files written (rank 0), then one stream per order and
    depth (all ranks); every collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    work = pl.Path(tmp)
    if comm.rank == 0:
        for order, dims in ORDERS.items():
            _write_input(work / f"input_{order}.nc", dims)
    comm.Barrier()

    results: dict[tuple[str, int], dict[str, Any]] = {}
    for order, depth in CASES:
        out_path = work / f"output_{order}_depth{depth}.nc"
        result = _stream(work / f"input_{order}.nc", out_path, depth)
        comm.Barrier()  # the output file is complete before any read
        result["output_file"] = out_path
        results[(order, depth)] = result

    yield results
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestTransposedStream:
    @pytest.mark.parametrize(("order", "depth"), CASES)
    def test_no_problems(self, streamed, order, depth):
        """Nothing raised, every day was served, and every day taken
        out of a window by the file's order held this block's values."""
        assert streamed[(order, depth)]["problems"] == []

    @pytest.mark.parametrize(("order", "depth"), CASES)
    def test_window_in_the_file_order(self, streamed, order, depth):
        result = streamed[(order, depth)]
        n_block = result["n_block"]
        # the last window is shallower when the depth does not divide
        n_full, n_last = divmod(N_TIME, depth)
        days = [depth] * n_full + ([n_last] if n_last else [])
        assert result["window_days"] == days
        dims = ORDERS[order]
        assert result["src_dims"] == [dims] * len(days)
        if order == "time_first":
            shapes = [(nn, n_block) for nn in days]
        else:
            shapes = [(n_block, nn) for nn in days]
        assert result["src_shapes"] == shapes

    @pytest.mark.parametrize(("order", "depth"), CASES)
    def test_buffers_do_not_depend_on_the_order(self, streamed, order, depth):
        result = streamed[(order, depth)]
        assert result["buffer_dims"] == (GRID,)
        assert result["out_dims"] == ("time", GRID)

    @pytest.mark.parametrize(("order", "depth"), CASES)
    def test_output_file(self, streamed, order, depth):
        if MPI.COMM_WORLD.rank != 0:
            return
        out = xr.load_dataset(streamed[(order, depth)]["output_file"])
        assert out["forcing_out"].dims == ("time", GRID)
        np.testing.assert_array_equal(
            out["forcing_out"].values, _expected(0, N_TIME, 0, N_HRU)
        )
