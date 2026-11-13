"""units mode at the DISTRIBUTED grid's boundary (ModelMPI._build):
static params in the input_file are checked/converted in place per
rank; a streamed input in the wrong units raises on every rank. Two-grid
toy (distributed hru -> MapMPI -> replicated segment), 'convert' mode.

Run: mpirun -n 4 pytest --with-mpi tests/test_units_check_mpi.py
"""

import pathlib as pl
import shutil
import tempfile

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI
from pws_phoenix.maps import MapMPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.processes_concrete import Lower, Upper

UNITS = {
    "param_up_0": "fraction",
    "param_up_1": "fraction",
    "param_shared_name": "fraction",
    "param_low_0": "fraction",
    "forcing_up": "inches",
    "forcing_low": "inches",
}


@pytest.fixture(scope="module")
def two_grid_toy(dimensions, make_two_grid_toy):
    return make_two_grid_toy(dimensions)


@pytest.fixture(scope="module")
def answers(
    dimensions, two_grid_toy, two_grid_weights, compute_two_grid_answers
):
    return compute_two_grid_answers(two_grid_toy, two_grid_weights, dimensions)


@pytest.fixture(scope="module")
def data_dir():
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    yield pl.Path(tmp)
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_input_file(path, toy, *, param_up_0_percent, forcing_in_feet):
    """Rank 0 writes the distributed grid's input file with units attrs;
    optionally param_up_0 x100 as 'percent' (resident: convertible) and
    forcing_up /12 as 'feet' (streamed: never convertible)."""
    up = toy["up_params"]
    param_up_0 = up["param_up_0"]
    forcing_up = toy["forcing_up"]
    if param_up_0_percent:
        param_up_0 = param_up_0 * 100.0
    if forcing_in_feet:
        forcing_up = forcing_up / 12.0
    ds = xr.Dataset(
        data_vars=dict(
            forcing_up=forcing_up,
            param_up_0=param_up_0,
            param_up_1=up["param_up_1"],
            param_shared_name=up["param_shared_name"],
        )
    )
    for name, key in UNITS.items():
        if name in ds:
            ds[name].attrs["units"] = key
    if param_up_0_percent:
        ds["param_up_0"].attrs["units"] = "percent"
    if forcing_in_feet:
        ds["forcing_up"].attrs["units"] = "feet"
    ds.to_netcdf(path)


def _stamped_lower(toy):
    low = toy["low_params"].copy()
    for name in low.data_vars:
        low[name].attrs["units"] = UNITS[str(name)]
    forcing_low = toy["forcing_low"].copy(deep=False)
    forcing_low.attrs["units"] = "inches"
    return {
        "parameters": low,
        "forcing_low": forcing_low,
        "storage": toy["low_storage_initial"].assign_attrs(units="inches"),
    }


def _build(
    data_dir, toy, weights, tag, mode="convert", flow_in_feet=False, file=None
):
    # initial state: full extent, under the variable's own name, with
    # its units (optionally /12 as 'feet': converted in the model's copy)
    flow = toy["up_flow_initial"].assign_attrs(units="inches")
    if flow_in_feet:
        flow = (flow / 12.0).assign_attrs(units="feet")
    process_dict = {
        "upper": {
            "class": Upper,
            "discretization": "hru",
            "flow": flow,
        },
        "lower": {
            "class": Lower,
            "discretization": "segment",
            **_stamped_lower(toy),
        },
    }
    maps = {
        "hru_to_seg": MapMPI(
            weights=weights, grid={"hru": "segment"}, variable={"flow": "flow"}
        )
    }
    control = {
        "input_file": data_dir / f"hru_input_{file or tag}.nc",
        "output_parallel_netcdf": data_dir / f"hru_output_{tag}.nc",
        "output_serial_zarr": data_dir / f"segment_output_{tag}.zarr",
        "output_var_names": ["flow", "storage"],
        "time_chunk_size": 10,
        "mpi_grid": "hru",
        "units": mode,
    }
    return ModelMPI(process_dict, control, maps=maps), control


@pytest.mark.mpi(min_size=2)
def test_convert_static_parameter(
    data_dir, two_grid_toy, two_grid_weights, answers
):
    """param_up_0 supplied x100 as 'percent': converted in place on each
    rank's decomposed slice -> the declared-units answers."""
    comm = MPI.COMM_WORLD
    path = data_dir / "hru_input_percent.nc"
    if comm.rank == 0:
        _write_input_file(
            path, two_grid_toy, param_up_0_percent=True, forcing_in_feet=False
        )
    comm.Barrier()
    model, control = _build(
        data_dir, two_grid_toy, two_grid_weights, "percent"
    )
    model.run(np.float64(1.0))
    storage_final = model.discretizations["segment"].dataset["storage"].values
    np.testing.assert_allclose(
        storage_final, answers["storage"][-1], rtol=1e-12
    )
    model.finalize()
    comm.Barrier()
    if comm.rank == 0:
        with xr.load_dataset(control["output_parallel_netcdf"]) as ds_out:
            flow_out = ds_out["flow_out"].values
        np.testing.assert_allclose(flow_out, answers["flow"], rtol=1e-12)


@pytest.mark.mpi(min_size=2)
def test_streamed_input_mismatch_raises_everywhere(
    data_dir, two_grid_toy, two_grid_weights
):
    """forcing_up streamed as 'feet': not convertible -> every rank
    raises at build (same metadata on every rank, so no rank is left
    waiting in a collective)."""
    comm = MPI.COMM_WORLD
    path = data_dir / "hru_input_feet.nc"
    if comm.rank == 0:
        _write_input_file(
            path, two_grid_toy, param_up_0_percent=False, forcing_in_feet=True
        )
    comm.Barrier()
    with pytest.raises(ValueError, match="streamed input.*'forcing_up'"):
        _build(data_dir, two_grid_toy, two_grid_weights, "feet")
    raised_everywhere = comm.allgather(True)
    assert all(raised_everywhere)
    # ... and 'check' mode names the mismatch the same way
    with pytest.raises(ValueError, match="'forcing_up'.*'feet'"):
        _build(data_dir, two_grid_toy, two_grid_weights, "feet", mode="check")


@pytest.mark.mpi(min_size=2)
def test_supplied_state_units(
    data_dir, two_grid_toy, two_grid_weights, answers
):
    """Distributed-grid initial state supplied in 'feet': 'check' raises
    on every rank (the metadata is identical on every rank); 'convert'
    converts each rank's block of the model's copy -> the declared-units
    answers."""
    comm = MPI.COMM_WORLD
    path = data_dir / "hru_input_state.nc"
    if comm.rank == 0:
        _write_input_file(
            path, two_grid_toy, param_up_0_percent=False, forcing_in_feet=False
        )
    comm.Barrier()
    with pytest.raises(ValueError, match="initial state: 'flow'.*'feet'"):
        _build(
            data_dir,
            two_grid_toy,
            two_grid_weights,
            "state_check",
            mode="check",
            flow_in_feet=True,
            file="state",
        )
    raised_everywhere = comm.allgather(True)
    assert all(raised_everywhere)
    model, control = _build(
        data_dir,
        two_grid_toy,
        two_grid_weights,
        "state_convert",
        flow_in_feet=True,
        file="state",
    )
    model.run(np.float64(1.0))
    storage_final = model.discretizations["segment"].dataset["storage"].values
    np.testing.assert_allclose(
        storage_final, answers["storage"][-1], rtol=1e-12
    )
    model.finalize()
    comm.Barrier()
    if comm.rank == 0:
        with xr.load_dataset(control["output_parallel_netcdf"]) as ds_out:
            flow_out = ds_out["flow_out"].values
        np.testing.assert_allclose(flow_out, answers["flow"], rtol=1e-12)
