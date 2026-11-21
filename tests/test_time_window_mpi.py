"""ModelMPI's time window: control["start_time"] / control["end_time"].

The model's time axis is the window, as in the serial Model: the
opened input file is sliced to the two bounds before it is decomposed
and streamed (nothing is read), every serial-grid input is sliced the
same way, and the clock, the restart index and both outputs are on
the window.

The reference of every run is the same model with NO window in its
control, on inputs sliced to the window before the model sees them
(an input file written with only the window's times). Compared bit
for bit: the final values of every variable on every rank, every step
and the time coordinate of the streamed NetCDF file, and every step
of a serial-grid variable in the rank-0 zarr store.

Two configurations:

- the two-grid TOY (Upper distributed on "hru" -> MapMPI -> Lower
  replicated on "segment", which has a time-varying input of its
  own): a window with both bounds, with a start only, with an end
  only, and with both bounds at a window depth of 7 steps
  (control["mpi_window_steps"]). A warm start inside a window: a run
  that ends on the restart day (by end_time) writes the restart, and
  a run over the whole window that reads it must end where the run
  without a restart ends. Builds that must raise, on every rank: a
  bound that is not a time of the file, a start after the end, and a
  restart whose state time is outside the window.
- the NHM on drb_2yr through mpi_model_from_control: two months that
  do not begin on the file's first day, at a window depth of 7.

All collective MPI ops live in the module-scoped fixtures; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_time_window_mpi.py -v
"""

import pathlib as pl
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI

from conftest import PWS_TEST_DATA, cold_start_state
from pws_phoenix.maps import MapMPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.prms_translate import (
    assemble_from_control,
    mpi_model_from_control,
    write_mpi_input_file,
)
from pws_phoenix.processes_concrete import Lower, Upper

# ---- toy: its time axis is 2000-01-01 .. 2000-12-30 ----
TOY_DT = np.float64(1.0)
# name -> (start_time, end_time, mpi_window_steps)
TOY_WINDOWS: dict[str, tuple[Any, Any, int]] = {
    "both": ("2000-03-01", "2000-04-30", 1),
    "start_only": (np.datetime64("2000-11-01"), None, 1),
    "end_only": (None, np.datetime64("2000-02-29T00", "ns"), 1),
    "both_depth_7": ("2000-03-01", "2000-04-30", 7),
}
# first and last day and the number of days of each window
TOY_EXPECTED = {
    "both": ("2000-03-01", "2000-04-30", 61),
    "start_only": ("2000-11-01", "2000-12-30", 60),
    "end_only": ("2000-01-01", "2000-02-29", 60),
    "both_depth_7": ("2000-03-01", "2000-04-30", 61),
}
# the restart day inside the window "both", and the index in that
# window of the day after it
TOY_RESTART_DAY = "2000-03-31"
TOY_RESUME_INDEX = 31
TOY_BAD_TAGS = [
    "start_not_in_file",
    "end_not_in_file",
    "start_after_end",
    "restart_outside_window",
]
TOY_BAD_MATCH = {
    "start_not_in_file": "control['start_time'] = 1999-12-31",
    "end_not_in_file": "control['end_time'] = 2001-06-01",
    "start_after_end": "is after",
    "restart_outside_window": "not found on this model's time axis",
}

# ---- drb_2yr: its record is 1979-01-01 .. 1980-12-31 ----
DRB_DIR = PWS_TEST_DATA / "drb_2yr"
DRB_CONTROL_FILE = DRB_DIR / "nhm.control"
DRB_DT = np.float64(86400.0)
DRB_START = "1979-03-01"
DRB_END = "1979-04-30"
DRB_N_DAYS = 61
DRB_DEPTH = 7
DRB_STREAMED_NAMES = ["sroff", "gwres_flow"]
DRB_SERIAL_NAME = "seg_outflow"

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


def _window_control(start: Any, end: Any) -> dict[str, Any]:
    control: dict[str, Any] = {}
    if start is not None:
        control["start_time"] = start
    if end is not None:
        control["end_time"] = end
    return control


def _final_state(model: ModelMPI) -> dict[str, np.ndarray]:
    """This rank's values of every variable of every grid."""
    state: dict[str, np.ndarray] = {}
    for proc_name, proc in model.model_dict.items():
        grid = model._proc_grid[proc_name]
        dataset = model.discretizations[grid].dataset
        for name in type(proc).get_var_names(include_budget=True):
            state[f"{grid}/{name}"] = dataset[name].values.copy()
    return state


def _time_facts(model: ModelMPI) -> dict[str, Any]:
    return {
        "ntime": int(model.ntime),
        "first": model.times.values[0],
        "last": model.times.values[-1],
        "start_index": model._start_index,
        "window_steps": model._window_steps,
    }


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
    data_dir = tmp_dir / "time_window_toy"
    full_file = data_dir / "hru_input.nc"

    def sliced_file(name: str) -> pl.Path:
        return data_dir / f"hru_input_{name}.nc"

    def hru_dataset(forcing_up: xr.DataArray) -> xr.Dataset:
        return xr.Dataset(
            data_vars=dict(
                forcing_up=forcing_up,
                param_up_0=toy["up_params"]["param_up_0"],
                param_up_1=toy["up_params"]["param_up_1"],
                param_shared_name=toy["up_params"]["param_shared_name"],
            ),
        )

    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        hru_dataset(toy["forcing_up"]).to_netcdf(full_file)
        for name, (start, end, _) in TOY_WINDOWS.items():
            hru_dataset(
                toy["forcing_up"].sel(time=slice(start, end))
            ).to_netcdf(sliced_file(name))
    comm.Barrier()

    def build(
        tag: str,
        input_file: pl.Path,
        forcing_low: xr.DataArray,
        extra_control: dict[str, Any],
    ) -> ModelMPI:
        process_dict = {
            "upper": {
                "class": Upper,
                "discretization": "hru",
                **cold_start_state(extra_control, flow=toy["up_flow_initial"]),
            },
            "lower": {
                "class": Lower,
                "discretization": "segment",
                "parameters": toy["low_params"],
                "forcing_low": forcing_low,
                **cold_start_state(
                    extra_control, storage=toy["low_storage_initial"]
                ),
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
            "output_parallel_netcdf": data_dir / f"out_{tag}.nc",
            "output_serial_zarr": data_dir / f"out_{tag}.zarr",
            "output_var_names": ["flow", "storage"],
            "time_chunk_size": 10,
            "mpi_grid": "hru",
            **extra_control,
        }
        return ModelMPI(process_dict, control, maps=maps)

    def run(model: ModelMPI, tag: str) -> dict[str, Any]:
        facts = _time_facts(model)
        model.run(TOY_DT)
        result = facts | {
            "state": _final_state(model),
            "output_file": data_dir / f"out_{tag}.nc",
            "output_zarr": data_dir / f"out_{tag}.zarr",
        }
        model.finalize()
        comm.Barrier()  # the outputs are complete before a rank-0 read
        return result

    # the window in the control, on the whole file and the whole
    # serial-grid input
    windowed: dict[str, dict[str, Any]] = {}
    # no window in the control, on inputs sliced beforehand (depth 1)
    reference: dict[str, dict[str, Any]] = {}
    for name, (start, end, depth) in TOY_WINDOWS.items():
        tag = f"windowed_{name}"
        model = build(
            tag,
            full_file,
            toy["forcing_low"],
            _window_control(start, end) | {"mpi_window_steps": depth},
        )
        windowed[name] = run(model, tag)
        tag = f"reference_{name}"
        model = build(
            tag,
            sliced_file(name),
            toy["forcing_low"].sel(time=slice(start, end)),
            {},
        )
        reference[name] = run(model, tag)

    # a warm start inside the window "both"
    start, end, _ = TOY_WINDOWS["both"]
    restart_dir = data_dir / "restart"
    model = build(
        "to_restart_day",
        full_file,
        toy["forcing_low"],
        _window_control(start, TOY_RESTART_DAY),
    )
    model.run(TOY_DT)
    model.write_restart(restart_dir)  # ends on a Barrier
    model.finalize()
    comm.Barrier()
    tag = "from_restart_day"
    model = build(
        tag,
        full_file,
        toy["forcing_low"],
        _window_control(start, end) | {"restart_read": restart_dir},
    )
    warm = run(model, tag)

    # Builds that must raise (ModelMPI.__init__ closes the input file
    # they opened: tests/test_bad_build_mpi.py).
    bad_controls = {
        "start_not_in_file": _window_control("1999-12-31", None),
        "end_not_in_file": _window_control(None, "2001-06-01"),
        "start_after_end": _window_control("2000-03-31", "2000-03-01"),
        # the restart's state time is 2000-03-31
        "restart_outside_window": _window_control("2000-04-10", None)
        | {"restart_read": restart_dir},
    }
    assert list(bad_controls) == TOY_BAD_TAGS
    errors: dict[str, str] = {}
    for tag, extra_control in bad_controls.items():
        try:
            build(tag, full_file, toy["forcing_low"], extra_control)
        except ValueError as err:
            errors[tag] = str(err)
    comm.Barrier()

    return {
        "windowed": windowed,
        "reference": reference,
        "warm": warm,
        "errors": errors,
    }


@pytest.mark.mpi(min_size=2)
class TestTimeWindowTwoGridToy:
    @pytest.mark.parametrize("name", list(TOY_WINDOWS))
    def test_time_axis_is_the_window(self, toy_runs, name):
        run = toy_runs["windowed"][name]
        first, last, n_days = TOY_EXPECTED[name]
        assert run["ntime"] == n_days
        assert run["first"] == np.datetime64(first)
        assert run["last"] == np.datetime64(last)
        assert run["start_index"] == 0
        assert run["window_steps"] == TOY_WINDOWS[name][2]
        reference = toy_runs["reference"][name]
        assert reference["ntime"] == n_days
        if name == "both_depth_7":
            # depth 7 must leave a shallower last window
            assert n_days % 7 != 0

    @pytest.mark.parametrize("name", list(TOY_WINDOWS))
    def test_final_state_bit_identical(self, toy_runs, name):
        """Every variable of both grids, this rank's values."""
        _assert_states_equal(
            toy_runs["windowed"][name]["state"],
            toy_runs["reference"][name]["state"],
        )

    @pytest.mark.parametrize("name", list(TOY_WINDOWS))
    def test_streamed_output_bit_identical(self, toy_runs, name):
        """Every step in the NetCDF file, and its time coordinate."""
        if MPI.COMM_WORLD.rank != 0:
            return
        first, last, n_days = TOY_EXPECTED[name]
        ds_ref = xr.load_dataset(toy_runs["reference"][name]["output_file"])
        ds_out = xr.load_dataset(toy_runs["windowed"][name]["output_file"])
        assert ds_out.sizes["time"] == n_days
        assert ds_out["time"].values[0] == np.datetime64(first)
        assert ds_out["time"].values[-1] == np.datetime64(last)
        np.testing.assert_array_equal(
            ds_out["time"].values, ds_ref["time"].values
        )
        np.testing.assert_array_equal(
            ds_out["flow_out"].values, ds_ref["flow_out"].values
        )

    @pytest.mark.parametrize("name", list(TOY_WINDOWS))
    def test_serial_zarr_bit_identical(self, toy_runs, name):
        """Every step of the segment variable in the rank-0 zarr
        store, and its time coordinate."""
        if MPI.COMM_WORLD.rank != 0:
            return
        first, last, n_days = TOY_EXPECTED[name]
        with (
            xr.open_zarr(
                toy_runs["reference"][name]["output_zarr"],
                consolidated=False,
            ) as ds_ref,
            xr.open_zarr(
                toy_runs["windowed"][name]["output_zarr"],
                consolidated=False,
            ) as ds_out,
        ):
            assert ds_out.sizes["time"] == n_days
            assert ds_out["time"].values[0] == np.datetime64(first)
            assert ds_out["time"].values[-1] == np.datetime64(last)
            np.testing.assert_array_equal(
                ds_out["time"].values, ds_ref["time"].values
            )
            np.testing.assert_array_equal(
                ds_out["storage"].values, ds_ref["storage"].values
            )

    def test_warm_start_index_is_in_the_window(self, toy_runs):
        """The restart index counts from the window's first day, and
        the time axis is still the whole window."""
        warm = toy_runs["warm"]
        first, last, n_days = TOY_EXPECTED["both"]
        assert warm["start_index"] == TOY_RESUME_INDEX
        assert warm["ntime"] == n_days
        assert warm["first"] == np.datetime64(first)
        assert warm["last"] == np.datetime64(last)

    def test_warm_start_final_state_bit_identical(self, toy_runs):
        _assert_states_equal(
            toy_runs["warm"]["state"], toy_runs["windowed"]["both"]["state"]
        )

    def test_warm_start_streamed_output(self, toy_runs):
        """The warm run's file holds the days after the restart day,
        equal to the same days of the run without a restart."""
        if MPI.COMM_WORLD.rank != 0:
            return
        n_days = TOY_EXPECTED["both"][2]
        ds_all = xr.load_dataset(toy_runs["windowed"]["both"]["output_file"])
        ds_warm = xr.load_dataset(toy_runs["warm"]["output_file"])
        assert ds_warm.sizes["time"] == n_days - TOY_RESUME_INDEX
        assert ds_warm["time"].values[0] == np.datetime64(
            TOY_RESTART_DAY
        ) + np.timedelta64(1, "D")
        np.testing.assert_array_equal(
            ds_warm["time"].values, ds_all["time"].values[TOY_RESUME_INDEX:]
        )
        np.testing.assert_array_equal(
            ds_warm["flow_out"].values,
            ds_all["flow_out"].values[TOY_RESUME_INDEX:],
        )

    @pytest.mark.parametrize("tag", TOY_BAD_TAGS)
    def test_bad_window_raises(self, toy_runs, tag):
        assert TOY_BAD_MATCH[tag] in toy_runs["errors"].get(tag, "")


# ----------------------------------------------------------------------
# the NHM on drb_2yr
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def drb_runs(tmp_dir):
    """Prep (rank 0), then the windowed run and its reference (all
    ranks); every collective lives here."""
    comm = MPI.COMM_WORLD
    data_dir = tmp_dir / "time_window_drb"
    full_file = data_dir / "mpi_input.nc"
    sliced_file = data_dir / "mpi_input_sliced.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        kit = assemble_from_control(DRB_CONTROL_FILE)
        write_mpi_input_file(kit, full_file)
        xr.load_dataset(full_file).sel(
            time=slice(DRB_START, DRB_END)
        ).to_netcdf(sliced_file)
    comm.Barrier()

    def run(
        tag: str, input_file: pl.Path, extra_control: dict[str, Any]
    ) -> dict[str, Any]:
        control = {
            "input_file": input_file,
            "output_parallel_netcdf": data_dir / f"output_{tag}.nc",
            "output_serial_zarr": data_dir / f"output_{tag}.zarr",
            "output_var_names": DRB_STREAMED_NAMES + [DRB_SERIAL_NAME],
            "mpi_grid": "nhru",
            **extra_control,
        }
        model = mpi_model_from_control(DRB_CONTROL_FILE, control=control)
        result = _time_facts(model) | {
            # the reference is only a reference when no serial-grid
            # input rides outside the input file at full extent
            "n_serial_inputs": len(model.inputs_dict),
            "output_file": control["output_parallel_netcdf"],
            "output_zarr": control["output_serial_zarr"],
        }
        model.run(DRB_DT)
        result["state"] = _final_state(model)
        model.finalize()
        comm.Barrier()
        return result

    windowed = run(
        "windowed",
        full_file,
        _window_control(DRB_START, DRB_END) | {"mpi_window_steps": DRB_DEPTH},
    )
    reference = run("reference", sliced_file, {})
    return {"windowed": windowed, "reference": reference}


@drb_skipif
@pytest.mark.mpi(min_size=2)
class TestTimeWindowDrb:
    def test_time_axis_is_the_window(self, drb_runs):
        for run in drb_runs.values():
            assert run["ntime"] == DRB_N_DAYS
            assert run["first"] == np.datetime64(DRB_START)
            assert run["last"] == np.datetime64(DRB_END)
            assert run["start_index"] == 0
            assert run["n_serial_inputs"] == 0
        assert drb_runs["windowed"]["window_steps"] == DRB_DEPTH
        assert drb_runs["reference"]["window_steps"] == 1
        assert DRB_N_DAYS % DRB_DEPTH != 0

    def test_final_state_bit_identical(self, drb_runs):
        """Every variable of both grids, this rank's values."""
        _assert_states_equal(
            drb_runs["windowed"]["state"], drb_runs["reference"]["state"]
        )

    def test_streamed_output_bit_identical(self, drb_runs):
        if MPI.COMM_WORLD.rank != 0:
            return
        ds_ref = xr.load_dataset(drb_runs["reference"]["output_file"])
        ds_out = xr.load_dataset(drb_runs["windowed"]["output_file"])
        assert ds_out.sizes["time"] == DRB_N_DAYS
        np.testing.assert_array_equal(
            ds_out["time"].values, ds_ref["time"].values
        )
        for name in DRB_STREAMED_NAMES:
            buf = f"{name}_out"
            np.testing.assert_array_equal(
                ds_out[buf].values, ds_ref[buf].values, err_msg=name
            )

    def test_serial_zarr_bit_identical(self, drb_runs):
        if MPI.COMM_WORLD.rank != 0:
            return
        with (
            xr.open_zarr(
                drb_runs["reference"]["output_zarr"], consolidated=False
            ) as ds_ref,
            xr.open_zarr(
                drb_runs["windowed"]["output_zarr"], consolidated=False
            ) as ds_out,
        ):
            assert ds_out.sizes["time"] == DRB_N_DAYS
            np.testing.assert_array_equal(
                ds_out["time"].values, ds_ref["time"].values
            )
            np.testing.assert_array_equal(
                ds_out[DRB_SERIAL_NAME].values,
                ds_ref[DRB_SERIAL_NAME].values,
            )
