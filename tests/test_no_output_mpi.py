"""ModelMPI output is optional, and a store exists only when a name
is routed to it.

control["output_var_names"] is routed by owning grid: distributed-grid
names stream through the mpixarray NetCDF writer
(control["output_parallel_netcdf"]), serial-grid names go to the
rank-0 zarr Output (control["output_serial_zarr"]). With no names
(the key absent, or an empty list) the run writes nothing: no writer
is opened, no file is created and no output buffer is declared. With
names on one grid only, only that grid's store is created.

The reference of every run is the same model writing both outputs:
the final values of every variable on every rank are compared bit
for bit, and so is whatever a run does write.

Two configurations:

- the two-grid TOY (Upper distributed on "hru" -> MapMPI -> Lower
  replicated on "segment"): no output keys; an empty list beside
  both stores; the serial-grid name only; the distributed-grid name
  only. Builds that must raise, on every rank: a distributed-grid
  name without output_parallel_netcdf (raised after the stream
  exists, so the build releases a stream that has no writer), and a
  store without output_var_names.
- the NHM on drb_2yr through mpi_model_from_control, over two months:
  no output keys (the control file's names are then left out), and
  output_serial_zarr only (its segment names are injected, its hru
  names left out).

All collective MPI ops live in the module-scoped fixtures; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_no_output_mpi.py -v
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
from pws_phoenix.maps import MapMPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.prms_translate import (
    assemble_from_control,
    mpi_model_from_control,
    write_mpi_input_file,
)
from pws_phoenix.processes_concrete import Lower, Upper

NETCDF_NAME = "out.nc"
ZARR_NAME = "out.zarr"

# ---- toy ----
TOY_DT = np.float64(1.0)
TOY_REFERENCE = "both"
# run -> (output_var_names or None for no key, NetCDF store given,
#         zarr store given)
TOY_RUNS: dict[str, tuple[list[str] | None, bool, bool]] = {
    "both": (["flow", "storage"], True, True),
    "no_keys": (None, False, False),
    "empty_list": ([], True, True),
    "serial_name_only": (["storage"], False, True),
    "stream_name_only": (["flow"], True, False),
}
# what each run leaves in its own directory
TOY_FILES = {
    "both": [NETCDF_NAME, ZARR_NAME],
    "no_keys": [],
    "empty_list": [],
    "serial_name_only": [ZARR_NAME],
    "stream_name_only": [NETCDF_NAME],
}
TOY_BAD: dict[str, tuple[list[str] | None, bool, bool]] = {
    "stream_name_no_netcdf": (["flow", "storage"], False, True),
    "netcdf_no_names": (None, True, False),
    "zarr_no_names": (None, False, True),
}
TOY_BAD_MATCH = {
    "stream_name_no_netcdf": "control['output_parallel_netcdf']",
    "netcdf_no_names": "['output_parallel_netcdf'] and no output_var_names",
    "zarr_no_names": "['output_serial_zarr'] and no output_var_names",
}

# ---- drb_2yr: its record is 1979-01-01 .. 1980-12-31 ----
DRB_DIR = PWS_TEST_DATA / "drb_2yr"
DRB_CONTROL_FILE = DRB_DIR / "nhm.control"
DRB_DT = np.float64(86400.0)
DRB_WINDOW = {
    "start_time": "1979-03-01",
    "end_time": "1979-04-30",
    "mpi_window_steps": 7,
}
DRB_N_DAYS = 61
DRB_REFERENCE = "both"
DRB_STREAMED_NAMES = ["sroff", "gwres_flow"]
DRB_SERIAL_NAME = "seg_outflow"
# run -> (output_var_names or None for no key, NetCDF store given,
#         zarr store given)
DRB_RUNS: dict[str, tuple[list[str] | None, bool, bool]] = {
    "both": (DRB_STREAMED_NAMES + [DRB_SERIAL_NAME], True, True),
    "no_keys": (None, False, False),
    "zarr_only": (None, False, True),
}
DRB_FILES = {
    "both": [NETCDF_NAME, ZARR_NAME],
    "no_keys": [],
    "zarr_only": [ZARR_NAME],
}

_drb_needed = [
    DRB_CONTROL_FILE,
    DRB_DIR / "myparam.param",
    DRB_DIR / "prcp.cbh",
]
_drb_missing = [str(ff) for ff in _drb_needed if not ff.exists()]
drb_skipif = pytest.mark.skipif(
    bool(_drb_missing),
    reason="drb_2yr legacy files missing: " + ", ".join(_drb_missing[:3]),
)


@pytest.fixture(scope="module")
def tmp_dir():
    """One shared temp dir: rank 0 creates it, broadcasts it and
    removes it."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    yield pl.Path(tmp)
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_dir(data_dir: pl.Path, tag: str) -> pl.Path:
    """A directory of its own for one build (rank 0 makes it), so
    what the build leaves on disk can be listed."""
    comm = MPI.COMM_WORLD
    run_dir = data_dir / tag
    if comm.rank == 0:
        run_dir.mkdir(parents=True)
    comm.Barrier()
    return run_dir


def _output_control(
    run_dir: pl.Path, spec: tuple[list[str] | None, bool, bool]
) -> dict[str, Any]:
    names, netcdf, zarr = spec
    control: dict[str, Any] = {}
    if names is not None:
        control["output_var_names"] = list(names)
    if netcdf:
        control["output_parallel_netcdf"] = run_dir / NETCDF_NAME
    if zarr:
        control["output_serial_zarr"] = run_dir / ZARR_NAME
    return control


def _files_left(run_dir: pl.Path) -> list[str] | None:
    """Rank 0's listing of a build's directory, on every rank."""
    comm = MPI.COMM_WORLD
    comm.Barrier()  # the outputs are complete
    listing = None
    if comm.rank == 0:
        listing = sorted(pp.name for pp in run_dir.iterdir())
    return comm.bcast(listing, root=0)


def _final_state(model: ModelMPI) -> dict[str, np.ndarray]:
    """This rank's values of every variable of every grid."""
    state: dict[str, np.ndarray] = {}
    for proc_name, proc in model.model_dict.items():
        grid = model._proc_grid[proc_name]
        dataset = model.discretizations[grid].dataset
        for name in type(proc).get_var_names(include_budget=True):
            state[f"{grid}/{name}"] = dataset[name].values.copy()
    return state


def _run(model: ModelMPI, dt: np.float64, run_dir: pl.Path) -> dict[str, Any]:
    """Run and finalize a built model; what the tests read of it."""
    comm = MPI.COMM_WORLD
    result: dict[str, Any] = {
        "run_dir": run_dir,
        "writer_open": model._ds_mpi_stream.mpi.has_writer,
        "output_map": dict(model._output_map),
        # the zarr Output lives on rank 0 only
        "has_zarr_output": comm.bcast(model.output is not None, root=0),
        "stream_dims": {
            str(name): tuple(model._ds_mpi_stream[name].dims)
            for name in model._ds_mpi_stream.data_vars
        },
    }
    model.run(dt)
    result["state"] = _final_state(model)
    model.finalize()
    result["files"] = _files_left(run_dir)
    return result


def _assert_states_equal(state: dict, reference: dict) -> None:
    assert state.keys() == reference.keys()
    for name, values in reference.items():
        np.testing.assert_array_equal(state[name], values, err_msg=name)


# ----------------------------------------------------------------------
# two-grid toy: distributed Upper -> MapMPI -> replicated Lower
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_runs(tmp_dir, dimensions, make_two_grid_toy, two_grid_weights):
    """Every toy build and run; every collective lives here."""
    comm = MPI.COMM_WORLD
    toy = make_two_grid_toy(dimensions)
    data_dir = tmp_dir / "no_output_toy"
    input_file = data_dir / "hru_input.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            data_vars=dict(
                forcing_up=toy["forcing_up"],
                param_up_0=toy["up_params"]["param_up_0"],
                param_up_1=toy["up_params"]["param_up_1"],
                param_shared_name=toy["up_params"]["param_shared_name"],
            ),
        ).to_netcdf(input_file)
    comm.Barrier()

    def build(output_control: dict[str, Any]) -> ModelMPI:
        process_dict = {
            "upper": {
                "class": Upper,
                "discretization": "hru",
                "flow": toy["up_flow_initial"],
            },
            "lower": {
                "class": Lower,
                "discretization": "segment",
                "parameters": toy["low_params"],
                "forcing_low": toy["forcing_low"],
                "storage": toy["low_storage_initial"],
            },
        }
        maps = {
            "hru_to_seg": MapMPI(
                weights=two_grid_weights,
                grid={"hru": "segment"},
                variable={"flow": "flow"},
            )
        }
        control = {
            "input_file": input_file,
            "time_chunk_size": 10,
            "mpi_grid": "hru",
            **output_control,
        }
        return ModelMPI(process_dict, control, maps=maps)

    runs: dict[str, dict[str, Any]] = {}
    for tag, spec in TOY_RUNS.items():
        run_dir = _run_dir(data_dir, tag)
        runs[tag] = _run(
            build(_output_control(run_dir, spec)), TOY_DT, run_dir
        )

    # Builds that must raise. ModelMPI.__init__ releases what the
    # build opened (tests/test_bad_build_mpi.py); here the stream it
    # releases has no writer.
    errors: dict[str, str] = {}
    bad_files: dict[str, list[str] | None] = {}
    for tag, spec in TOY_BAD.items():
        run_dir = _run_dir(data_dir, tag)
        try:
            build(_output_control(run_dir, spec))
        except ValueError as err:
            errors[tag] = str(err)
        bad_files[tag] = _files_left(run_dir)

    return {"runs": runs, "errors": errors, "bad_files": bad_files}


@pytest.mark.mpi(min_size=2)
class TestNoOutputTwoGridToy:
    @pytest.mark.parametrize("tag", list(TOY_RUNS))
    def test_stores_follow_the_names(self, toy_runs, tag):
        """A store exists when a name is routed to it, and only
        then: the writer, its buffers, the rank-0 Output and the
        files left on disk."""
        run = toy_runs["runs"][tag]
        names = TOY_RUNS[tag][0] or []
        streamed = "flow" in names
        assert run["writer_open"] == streamed
        assert run["output_map"] == ({"flow": "flow_out"} if streamed else {})
        assert run["has_zarr_output"] == ("storage" in names)
        assert run["files"] == TOY_FILES[tag]
        # no (time, hru) output buffer without a streamed name: the
        # stream dataset then has no variable on time at all (the
        # streamed input is shadowed by its (hru,) one-step buffer)
        on_time = sorted(
            name for name, dims in run["stream_dims"].items() if "time" in dims
        )
        assert on_time == (["flow_out"] if streamed else [])

    @pytest.mark.parametrize("tag", list(TOY_RUNS))
    def test_final_state_bit_identical(self, toy_runs, tag):
        """Every variable of both grids, this rank's values."""
        _assert_states_equal(
            toy_runs["runs"][tag]["state"],
            toy_runs["runs"][TOY_REFERENCE]["state"],
        )

    def test_streamed_output_bit_identical(self, toy_runs):
        """The NetCDF file written without a zarr store."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = toy_runs["runs"]
        ds_ref = xr.load_dataset(runs[TOY_REFERENCE]["run_dir"] / NETCDF_NAME)
        ds_out = xr.load_dataset(
            runs["stream_name_only"]["run_dir"] / NETCDF_NAME
        )
        np.testing.assert_array_equal(
            ds_out["time"].values, ds_ref["time"].values
        )
        np.testing.assert_array_equal(
            ds_out["flow_out"].values, ds_ref["flow_out"].values
        )

    def test_serial_zarr_bit_identical(self, toy_runs):
        """The zarr store written without a NetCDF writer."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = toy_runs["runs"]
        with (
            xr.open_zarr(
                runs[TOY_REFERENCE]["run_dir"] / ZARR_NAME, consolidated=False
            ) as ds_ref,
            xr.open_zarr(
                runs["serial_name_only"]["run_dir"] / ZARR_NAME,
                consolidated=False,
            ) as ds_out,
        ):
            np.testing.assert_array_equal(
                ds_out["time"].values, ds_ref["time"].values
            )
            np.testing.assert_array_equal(
                ds_out["storage"].values, ds_ref["storage"].values
            )

    @pytest.mark.parametrize("tag", list(TOY_BAD))
    def test_bad_output_request_raises(self, toy_runs, tag):
        assert TOY_BAD_MATCH[tag] in toy_runs["errors"].get(tag, "")
        # and leaves nothing behind
        assert toy_runs["bad_files"][tag] == []


# ----------------------------------------------------------------------
# the NHM on drb_2yr
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def drb_runs(tmp_dir):
    """Prep (rank 0), then every run (all ranks); every collective
    lives here."""
    comm = MPI.COMM_WORLD
    data_dir = tmp_dir / "no_output_drb"
    input_file = data_dir / "mpi_input.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        write_mpi_input_file(
            assemble_from_control(DRB_CONTROL_FILE), input_file
        )
    comm.Barrier()

    runs: dict[str, dict[str, Any]] = {}
    for tag, spec in DRB_RUNS.items():
        run_dir = _run_dir(data_dir, tag)
        control = {
            "input_file": input_file,
            "mpi_grid": "nhru",
            **DRB_WINDOW,
            **_output_control(run_dir, spec),
        }
        model = mpi_model_from_control(DRB_CONTROL_FILE, control=control)
        ntime = int(model.ntime)
        runs[tag] = _run(model, DRB_DT, run_dir) | {"ntime": ntime}
    return runs


@drb_skipif
@pytest.mark.mpi(min_size=2)
class TestNoOutputDrb:
    @pytest.mark.parametrize("tag", list(DRB_RUNS))
    def test_stores_follow_the_names(self, drb_runs, tag):
        run = drb_runs[tag]
        assert run["ntime"] == DRB_N_DAYS
        assert run["files"] == DRB_FILES[tag]
        assert run["writer_open"] == (NETCDF_NAME in DRB_FILES[tag])
        assert run["has_zarr_output"] == (ZARR_NAME in DRB_FILES[tag])
        if tag != DRB_REFERENCE:
            # no (time, nhru) output buffer
            assert run["output_map"] == {}

    @pytest.mark.parametrize("tag", list(DRB_RUNS))
    def test_final_state_bit_identical(self, drb_runs, tag):
        """Every variable of both grids, this rank's values."""
        _assert_states_equal(
            drb_runs[tag]["state"], drb_runs[DRB_REFERENCE]["state"]
        )

    def test_zarr_only_holds_the_control_files_segment_names(self, drb_runs):
        """With output_serial_zarr alone, the control file's segment
        names are written and its hru names are left out."""
        if MPI.COMM_WORLD.rank != 0:
            return
        with (
            xr.open_zarr(
                drb_runs[DRB_REFERENCE]["run_dir"] / ZARR_NAME,
                consolidated=False,
            ) as ds_ref,
            xr.open_zarr(
                drb_runs["zarr_only"]["run_dir"] / ZARR_NAME,
                consolidated=False,
            ) as ds_out,
        ):
            assert ds_out.sizes["time"] == DRB_N_DAYS
            assert DRB_SERIAL_NAME in ds_out.data_vars
            for name in ds_out.data_vars:
                assert "nhru" not in ds_out[name].dims, name
            np.testing.assert_array_equal(
                ds_out[DRB_SERIAL_NAME].values,
                ds_ref[DRB_SERIAL_NAME].values,
            )
