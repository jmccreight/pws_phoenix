"""The serial Model's time window (``control["start_time"]`` /
``control["end_time"]``) and inputs taken from one combined file, on
the Upper/Lower toy model."""

import numpy as np
import pytest

from pws_phoenix.model import Model, ModelMPI
from pws_phoenix.processes_concrete import Lower, Upper

PARAM_NAMES = ["param_up_0", "param_up_1", "param_low_0", "param_shared_name"]
FORCING_NAMES = ["forcing_up", "forcing_low"]
COLLECT_NAMES = ["flow", "storage"]
DT = np.float64(1.0)

# the toy time axis is 2000-01-01 .. 2000-12-30
WINDOWS = [
    (None, None),
    ("2000-03-01", None),
    (None, "2000-03-31"),
    ("2000-03-01", "2000-03-31"),
    (np.datetime64("2000-03-01"), np.datetime64("2000-03-31T00", "ns")),
]


@pytest.fixture
def toy_ds(dimensions, make_toy_input):
    """A fresh in-memory copy of the toy input per test."""
    return make_toy_input(dimensions)


def _process_dict(toy_ds, forcing_up, forcing_low):
    return {
        "upper": {
            "class": Upper,
            "forcing_up": forcing_up,
            "flow": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "lower": {
            "class": Lower,
            "forcing_low": forcing_low,
            "storage": toy_ds["storage_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
    }


def _control(start, end):
    control = {"collect_var_names": COLLECT_NAMES}
    if start is not None:
        control["start_time"] = start
    if end is not None:
        control["end_time"] = end
    return control


def _run(model):
    """Run every step; the collected results and the time axis."""
    model.run(DT, np.int32(model.ntime))
    return (
        {nn: model.results.values[nn].copy() for nn in COLLECT_NAMES},
        model.times.values.copy(),
    )


def _reference(dimensions, make_toy_input, start, end):
    """The run on inputs sliced to the window BEFORE the Model sees
    them (no window in the control)."""
    toy = make_toy_input(dimensions)
    sliced = toy[FORCING_NAMES].sel(time=slice(start, end))
    process_dict = _process_dict(
        toy, sliced["forcing_up"], sliced["forcing_low"]
    )
    with Model(process_dict, _control(None, None)) as model:
        return _run(model)


@pytest.mark.parametrize(("start", "end"), WINDOWS)
def test_window_in_memory(dimensions, make_toy_input, toy_ds, start, end):
    """The window gives the answers of pre-sliced inputs, and never
    copies an in-memory input: no window keeps the caller's array
    itself, a window is a view of it."""
    expected, expected_times = _reference(
        dimensions, make_toy_input, start, end
    )
    process_dict = _process_dict(
        toy_ds, toy_ds["forcing_up"], toy_ds["forcing_low"]
    )
    with Model(process_dict, _control(start, end)) as model:
        for name in FORCING_NAMES:
            held = model.inputs_dict[name].data.values
            if start is None and end is None:
                assert held is toy_ds[name].values
            else:
                assert held is not toy_ds[name].values
                assert np.shares_memory(held, toy_ds[name].values)
        results, times = _run(model)
    np.testing.assert_array_equal(times, expected_times)
    if start is not None:
        assert times[0] == np.datetime64("2000-03-01")
    if end is not None:
        assert times[-1] == np.datetime64("2000-03-31")
    for name in COLLECT_NAMES:
        assert results[name].shape[0] == times.size
        np.testing.assert_array_equal(results[name], expected[name])


@pytest.mark.parametrize(
    ("start", "end"), [(None, None), ("2000-03-01", "2000-03-31")]
)
def test_combined_file(
    dimensions, make_toy_input, toy_ds, tmp_path, start, end
):
    """Two inputs given as the path of ONE file that holds both: each
    is the variable of its own name, the file is opened once, and the
    inputs stay on disk (one time row read per step), with or without
    a window."""
    expected, expected_times = _reference(
        dimensions, make_toy_input, start, end
    )
    path = tmp_path / "forcing.nc"
    toy_ds[FORCING_NAMES].to_netcdf(path)
    process_dict = _process_dict(toy_ds, path, path)
    with Model(process_dict, _control(start, end)) as model:
        assert len(model._opened_files) == 1
        results, times = _run(model)
        for name in FORCING_NAMES:
            assert not model.inputs_dict[name].data.variable._in_memory
    np.testing.assert_array_equal(times, expected_times)
    for name in COLLECT_NAMES:
        np.testing.assert_array_equal(results[name], expected[name])


def test_combined_file_missing_variable(toy_ds, tmp_path):
    path = tmp_path / "forcing.nc"
    toy_ds[["forcing_up", "param_up_0"]].to_netcdf(path)
    process_dict = _process_dict(toy_ds, path, path)
    with pytest.raises(KeyError, match="input 'forcing_low'.*no variable"):
        Model(process_dict, {})


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        ("1999-12-31", None, r"control\['start_time'\] = 1999-12-31"),
        (None, "2001-06-01", r"control\['end_time'\] = 2001-06-01"),
        ("2000-03-31", "2000-03-01", "is after"),
    ],
)
def test_window_raises(toy_ds, start, end, match):
    process_dict = _process_dict(
        toy_ds, toy_ds["forcing_up"], toy_ds["forcing_low"]
    )
    with pytest.raises(ValueError, match=match):
        Model(process_dict, _control(start, end))


@pytest.mark.parametrize("key", ["start_time", "end_time"])
def test_window_not_implemented_under_mpi(key):
    """ModelMPI raises on either key before it touches MPI or a file."""
    with pytest.raises(NotImplementedError, match="start_time"):
        ModelMPI({"upper": {"class": Upper}}, {key: "2000-03-01"})
