"""A serial Model build that raises releases what it opened.

A constructor that raises returns no object, so neither finalize()
nor a `with` block can close its files: Model.__init__ does it
(Model._release), as ModelMPI does (test_bad_build_mpi.py). Two
raises are provoked on the Upper/Lower toy model, whose forcings are
read from one file:

- an unknown output variable, raised by the zarr Output before its
  store exists: the original error must reach the caller (finalizing
  an Output with no store would raise a different one);
- an unknown collect variable, raised after the store exists: the
  Output is finalized, and a second model on the SAME store runs.

The model object of a build that raises is reached through a subclass
that records `self` when it releases.

A file is judged closed by xarray's `_close` attribute, which
`close()` sets to None. The probe is checked on a model whose files
are open.
"""

import pathlib as pl
from typing import Any

import numpy as np
import pytest
import xarray as xr
from pws_phoenix.model import Model
from pws_phoenix.processes_concrete import Lower, Upper

PARAM_NAMES = ["param_up_0", "param_up_1", "param_low_0", "param_shared_name"]
TIME_CHUNK_SIZE = 61


class _RecordingModel(Model):
    """Keeps the object of a build that raises."""

    released: list[Model] = []

    def _release(self) -> None:
        _RecordingModel.released.append(self)
        super()._release()


def _process_dict(toy_ds: xr.Dataset, forcing_file: pl.Path) -> dict:
    # the same path twice: the file is opened once, as a Dataset
    return {
        "upper": {
            "class": Upper,
            "forcing_up": forcing_file,
            "flow": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "lower": {
            "class": Lower,
            "forcing_low": forcing_file,
            "storage": toy_ds["storage_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
    }


def _assert_released(model: Model) -> None:
    assert model._finalized
    assert model.inputs_dict  # the build got as far as the inputs
    assert all(inp._closed for inp in model.inputs_dict.values())
    assert model._opened_files
    assert all(ff._close is None for ff in model._opened_files)


@pytest.fixture
def toy(dimensions, make_toy_input, tmp_path) -> dict[str, Any]:
    toy_ds = make_toy_input(dimensions)
    forcing_file = tmp_path / "forcing.nc"
    toy_ds[["forcing_up", "forcing_low"]].to_netcdf(forcing_file)
    return {
        "toy_ds": toy_ds,
        "forcing_file": forcing_file,
        "store": tmp_path / "out.zarr",
    }


def test_unknown_output_name_raises_its_own_error(toy):
    control = {
        "output_var_names": ["not_a_variable"],
        "output_serial_zarr": toy["store"],
        "time_chunk_size": TIME_CHUNK_SIZE,
    }
    n_released = len(_RecordingModel.released)
    with pytest.raises(ValueError, match="not found in any process"):
        _RecordingModel(
            _process_dict(toy["toy_ds"], toy["forcing_file"]), control
        )
    assert len(_RecordingModel.released) == n_released + 1
    model = _RecordingModel.released[-1]
    _assert_released(model)
    # raised before the store was created: there is none to finalize
    assert model.output is None
    assert not toy["store"].exists()


def test_late_raise_finalizes_output(toy, dimensions):
    control = {
        "output_var_names": ["flow"],
        "output_serial_zarr": toy["store"],
        "time_chunk_size": TIME_CHUNK_SIZE,
    }
    n_released = len(_RecordingModel.released)
    with pytest.raises(ValueError, match="not found in any process"):
        _RecordingModel(
            _process_dict(toy["toy_ds"], toy["forcing_file"]),
            control | {"collect_var_names": ["not_a_variable"]},
        )
    assert len(_RecordingModel.released) == n_released + 1
    model = _RecordingModel.released[-1]
    _assert_released(model)
    assert model.output is not None
    assert model.output._finalized
    # the store the bad build created holds no time steps
    out = xr.open_zarr(toy["store"], consolidated=False)
    assert out["flow"].shape[0] == 0

    # the same store, built again and run
    model = Model(
        _process_dict(toy["toy_ds"], toy["forcing_file"]),
        control | {"collect_var_names": ["flow"]},
    )
    # what _assert_released reads, on a model whose files are open
    assert model._opened_files
    assert all(ff._close is not None for ff in model._opened_files)
    model.run(np.float64(1.0), np.int32(dimensions["n_time"]))
    model.finalize()
    assert model.results is not None
    out = xr.open_zarr(toy["store"], consolidated=False)
    np.testing.assert_array_equal(
        out["flow"].values, model.results.values["flow"]
    )
