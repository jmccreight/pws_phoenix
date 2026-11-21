"""A time-varying input on ("time", <grid>) or on (<grid>, "time").

The order is taken from the dim names. The serial toy model gives the
same answers from either, bit for bit, and copies nothing more. An
input whose stored order makes the model's reads at least
data_io.READ_PIECES_WARN_RATIO times as many pieces as the other order
is named in ONE UserWarning at build (machine_probes/README.md "The
order of the dims decides the read").

data_io.Input alone is in tests/test_input.py; ModelMPI is in
tests/test_window_depth_mpi.py and tests/test_transposed_stream_mpi.py.
"""

import warnings
from collections.abc import Hashable
from typing import Any

import numpy as np
import pytest
import xarray as xr

from pws_phoenix.data_io import (
    READ_PIECES_WARN_RATIO,
    read_pieces,
    slow_order_note,
    warn_slow_order,
)
from pws_phoenix.model import Model
from pws_phoenix.processes_concrete import Lower, Upper

PARAM_NAMES = ["param_up_0", "param_up_1", "param_low_0", "param_shared_name"]
FORCING_NAMES = ["forcing_up", "forcing_low"]
COLLECT_NAMES = ["flow", "storage"]
DT = np.float64(1.0)
WINDOWS = [(None, None), ("2000-03-01", "2000-03-31")]
SOURCES = ["memory", "file", "file_load_all", "combined_file"]

# CONUS: 114,958 hrus, 17,167 days, 899 hrus per rank at 128 ranks
N_GRID = 114_958
N_TIME = 17_167
N_BLOCK = 899
# the table of machine_probes/README.md "The order of the dims decides
# the read": (n_block, n_days) -> pieces on (time, hru), on (hru, time)
README_TABLE = {
    "serial_one_day": ((N_GRID, 1), 1, N_GRID),
    "serial_load_all": ((N_GRID, N_TIME), 1, 1),
    "mpi_window_1": ((N_BLOCK, 1), 1, N_BLOCK),
    "mpi_window_365": ((N_BLOCK, 365), 365, N_BLOCK),
    "mpi_whole_record": ((N_BLOCK, N_TIME), N_TIME, 1),
}


@pytest.mark.parametrize("row", list(README_TABLE))
def test_read_pieces_readme_table(row):
    (n_block, n_days), time_first, grid_first = README_TABLE[row]
    assert read_pieces(n_block, N_GRID, n_days, N_TIME, True) == time_first
    assert read_pieces(n_block, N_GRID, n_days, N_TIME, False) == grid_first


# ---------------------------------------------------------------------------
# the serial toy model
# ---------------------------------------------------------------------------


def _process_dict(toy, forcing_up, forcing_low):
    return {
        "upper": {
            "class": Upper,
            "forcing_up": forcing_up,
            "flow": toy["flow_initial"],
            "parameters": toy[PARAM_NAMES],
        },
        "lower": {
            "class": Lower,
            "forcing_low": forcing_low,
            "storage": toy["storage_initial"],
            "parameters": toy[PARAM_NAMES],
        },
    }


def _control(start, end, load_all=False):
    control = {"collect_var_names": COLLECT_NAMES, "load_all": load_all}
    if start is not None:
        control["start_time"] = start
    if end is not None:
        control["end_time"] = end
    return control


def _run(model):
    model.run(DT, np.int32(model.ntime))
    return (
        {nn: model.results.values[nn].copy() for nn in COLLECT_NAMES},
        model.times.values.copy(),
    )


def _forcings(toy, source, dims, tmp_path):
    """The two forcings on `dims`, as `source` says: arrays in memory,
    one file each, or one file holding both."""
    in_memory = {
        name: toy[name].transpose(*dims).copy() for name in FORCING_NAMES
    }
    if source == "memory":
        return in_memory
    tmp_path.mkdir(exist_ok=True)
    if source == "combined_file":
        path = tmp_path / "forcing.nc"
        xr.Dataset(in_memory).to_netcdf(path)
        return {name: path for name in FORCING_NAMES}
    paths = {}
    for name, data in in_memory.items():
        paths[name] = tmp_path / f"{name}.nc"
        data.to_netcdf(paths[name])
    return paths


@pytest.mark.parametrize(("start", "end"), WINDOWS)
@pytest.mark.parametrize("source", SOURCES)
def test_toy_model_either_order(
    dimensions, make_toy_input, tmp_path, source, start, end
):
    """(space, time) forcings give the answers of (time, space)
    forcings, bit for bit: in memory (the caller's array is the one
    the model holds, or a view of it under a time window), from a file
    read one day per step, from a file loaded whole, and from one file
    that holds both."""
    load_all = source == "file_load_all"
    results = {}
    for dims in (("time", "space"), ("space", "time")):
        toy = make_toy_input(dimensions)
        forcings = _forcings(toy, source, dims, tmp_path / "_".join(dims))
        process_dict = _process_dict(
            toy, forcings["forcing_up"], forcings["forcing_low"]
        )
        with Model(process_dict, _control(start, end, load_all)) as model:
            for name in FORCING_NAMES:
                data = model.inputs_dict[name].data
                assert data.dims == dims
                if source == "memory":
                    given = forcings[name].values
                    if start is None:
                        assert data.values is given
                    else:
                        assert np.shares_memory(data.values, given)
                else:
                    assert data.variable._in_memory == load_all
            results[dims] = _run(model)
            if source in ("file", "combined_file"):
                for name in FORCING_NAMES:
                    data = model.inputs_dict[name].data
                    assert not data.variable._in_memory
    expected, expected_times = results[("time", "space")]
    got, got_times = results[("space", "time")]
    np.testing.assert_array_equal(got_times, expected_times)
    if start is not None:
        assert got_times[0] == np.datetime64(start)
        assert got_times[-1] == np.datetime64(end)
    for name in COLLECT_NAMES:
        assert got[name].shape[0] == got_times.size
        np.testing.assert_array_equal(got[name], expected[name])


# ---------------------------------------------------------------------------
# the warning
# ---------------------------------------------------------------------------

N_WIDE = READ_PIECES_WARN_RATIO  # the fewest grid elements that warn


def _stored(tmp_path, dims, shape, chunks=None):
    """A variable "prcp" of zeros on `dims`, written to a file and
    opened lazily; the caller closes it."""
    path = tmp_path / "prcp.nc"
    encoding: dict[Hashable, dict[str, Any]] = {}
    if chunks is not None:
        encoding["prcp"] = {"chunksizes": chunks}
    xr.DataArray(
        np.zeros(shape, dtype=np.float32), dims=dims, name="prcp"
    ).to_netcdf(path, encoding=encoding)
    return xr.open_dataset(path)


@pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
def test_stored_without_chunks_is_seen(tmp_path, engine):
    """What slow_order_note relies on: both of xarray's netCDF readers
    report a variable stored without chunks as
    encoding["chunksizes"] = None, a chunked one as a tuple, and a
    selection keeps the encoding."""
    path = tmp_path / "prcp.nc"
    data = xr.DataArray(np.zeros((6, 8)), dims=("time", "nhru"), name="prcp")
    data.to_netcdf(path)
    with xr.open_dataset(path, engine=engine) as ds:
        assert "chunksizes" in ds["prcp"].encoding
        assert ds["prcp"].encoding["chunksizes"] is None
        assert ds["prcp"].isel(time=slice(1, 3)).encoding["chunksizes"] is None
    chunked = tmp_path / "prcp_chunked.nc"
    data.to_netcdf(chunked, encoding={"prcp": {"chunksizes": (6, 4)}})
    with xr.open_dataset(chunked, engine=engine) as ds:
        assert tuple(ds["prcp"].encoding["chunksizes"]) == (6, 4)
    assert "chunksizes" not in data.encoding


# (dims, shape as (time, grid), n_days, n_block) -> a note or not
NOTE_CASES = {
    # serial, one day of the whole grid per step
    "serial_grid_first": (("nhru", "time"), (6, N_WIDE), 1, None, True),
    "serial_grid_first_narrow": (
        ("nhru", "time"),
        (6, N_WIDE - 1),
        1,
        None,
        False,
    ),
    "serial_time_first": (("time", "nhru"), (6, N_WIDE), 1, None, False),
    # serial, the whole record in one read
    "serial_load_all": (("nhru", "time"), (6, N_WIDE), 6, None, False),
    # an MPI rank: a block of 100 of 400 hrus
    "mpi_whole_record_time_first": (
        ("time", "nhru"),
        (N_WIDE, 400),
        N_WIDE,
        100,
        True,
    ),
    "mpi_whole_record_grid_first": (
        ("nhru", "time"),
        (N_WIDE, 400),
        N_WIDE,
        100,
        False,
    ),
    "mpi_window_1_time_first": (("time", "nhru"), (6, 400), 1, 100, False),
    "mpi_window_1_grid_first": (("nhru", "time"), (6, 400), 1, 100, True),
    # neither order is 100 times the other
    "mpi_window_3": (("time", "nhru"), (6, 400), 3, 100, False),
    "mpi_window_3_grid_first": (("nhru", "time"), (6, 400), 3, 100, False),
}


@pytest.mark.parametrize("case", list(NOTE_CASES))
def test_slow_order_note(tmp_path, case):
    dims, (n_time, n_grid), n_days, n_block, noted = NOTE_CASES[case]
    shape = (n_time, n_grid) if dims[0] == "time" else (n_grid, n_time)
    with _stored(tmp_path, dims, shape) as ds:
        note = slow_order_note("prcp", ds["prcp"], n_days, n_block)
        in_memory = ds["prcp"].load()
    if noted:
        assert note is not None
        assert note.startswith(f"'prcp' on {dims}")
        assert str(dims[::-1]) in note
    else:
        assert note is None
    # the same array from no file has no stored order
    in_memory.encoding = {}
    assert slow_order_note("prcp", in_memory, n_days, n_block) is None


def test_chunked_variable_is_not_judged(tmp_path):
    """Chunks put a block's bytes together in either order."""
    with _stored(
        tmp_path, ("nhru", "time"), (N_WIDE, 6), chunks=(N_WIDE, 6)
    ) as ds:
        assert slow_order_note("prcp", ds["prcp"], 1) is None
    with _stored(tmp_path, ("nhru", "time"), (N_WIDE, 6)) as ds:
        assert slow_order_note("prcp", ds["prcp"], 1) is not None


def test_warn_slow_order_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_slow_order([])
        assert not caught
        warn_slow_order(["line one", "line two"])
    assert len(caught) == 1
    assert caught[0].category is UserWarning
    message = str(caught[0].message)
    assert "machine_probes/README.md" in message
    assert "line one" in message and "line two" in message


def _slow_order_warnings(caught):
    return [ww for ww in caught if "slower order" in str(ww.message)]


@pytest.mark.parametrize(
    ("dims", "source", "n_warnings"),
    [
        (("space", "time"), "file", 1),
        (("space", "time"), "combined_file", 1),
        (("space", "time"), "file_load_all", 0),
        (("space", "time"), "memory", 0),
        (("time", "space"), "file", 0),
    ],
)
def test_model_warns_once(
    dimensions, make_toy_input, tmp_path, dims, source, n_warnings
):
    """A serial build warns ONCE, naming both forcings, when they are
    read from a (space, time) file one day per step and the grid has
    READ_PIECES_WARN_RATIO elements; and not otherwise."""
    wide = dict(dimensions, n_space=N_WIDE, space=np.arange(N_WIDE))
    toy = make_toy_input(wide)
    forcings = _forcings(toy, source, dims, tmp_path)
    process_dict = _process_dict(
        toy, forcings["forcing_up"], forcings["forcing_low"]
    )
    control = _control(None, None, load_all=(source == "file_load_all"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Model(process_dict, control) as model:
            model.run(DT, np.int32(3))
    found = _slow_order_warnings(caught)
    assert len(found) == n_warnings
    if n_warnings:
        message = str(found[0].message)
        for name in FORCING_NAMES:
            assert f"'{name}' on {dims}" in message


def test_narrow_grid_does_not_warn(dimensions, make_toy_input, tmp_path):
    """The toy grid of 20 elements is below the ratio."""
    assert dimensions["n_space"] < READ_PIECES_WARN_RATIO
    toy = make_toy_input(dimensions)
    forcings = _forcings(toy, "file", ("space", "time"), tmp_path)
    process_dict = _process_dict(
        toy, forcings["forcing_up"], forcings["forcing_low"]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Model(process_dict, _control(None, None)):
            pass
    assert not _slow_order_warnings(caught)
