"""A forgotten Model.finalize() must not lose zarr output silently.

The toy regression (test_up_low_regression.py: 365 steps, chunk 10)
already round-trips a partial tail chunk THROUGH finalize(). This file
covers the omission: the trailing partial buffer is never written, the
store reads back the fill value there, and Output.__del__ warns.
"""

import gc
import warnings

import numpy as np
import xarray as xr
from pws_phoenix.model import Model
from pws_phoenix.processes_concrete import Lower, Upper

PARAM_NAMES = ["param_up_0", "param_up_1", "param_low_0", "param_shared_name"]
TIME_CHUNK_SIZE = 61


def _process_dict(toy_ds):
    return {
        "upper": {
            "class": Upper,
            "forcing_up": toy_ds["forcing_up"],
            "flow_initial": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "lower": {
            "class": Lower,
            "forcing_low": toy_ds["forcing_low"],
            "storage_initial": toy_ds["storage_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
    }


def test_unfinalized_output_warns(dimensions, make_toy_input, tmp_path):
    n_time = dimensions["n_time"]
    tail = n_time % TIME_CHUNK_SIZE
    assert tail > 0, "test needs a partial trailing chunk"
    store = tmp_path / "out.zarr"
    control = {
        "output_var_names": ["flow"],
        "output_serial_zarr": store,
        "time_chunk_size": TIME_CHUNK_SIZE,
    }
    model = Model(_process_dict(make_toy_input(dimensions)), control)
    model.run(np.float64(1.0), np.int32(n_time))
    flow_last = model.model_dict["upper"]["flow"].values.copy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        del model
        gc.collect()
    msgs = [str(ww.message) for ww in caught if ww.category is RuntimeWarning]
    assert any(f"{tail} buffered time step(s)" in mm for mm in msgs), msgs

    # What the omission costs: the tail reads back as zarr's fill, not data.
    out = xr.open_zarr(store, consolidated=False)["flow"].values
    assert out.shape[0] == n_time
    assert not np.allclose(out[-1], flow_last)


def test_finalized_output_is_silent(dimensions, make_toy_input, tmp_path):
    control = {
        "output_var_names": ["flow"],
        "output_serial_zarr": tmp_path / "out.zarr",
        "time_chunk_size": TIME_CHUNK_SIZE,
    }
    with Model(_process_dict(make_toy_input(dimensions)), control) as model:
        model.run(np.float64(1.0), np.int32(dimensions["n_time"]))
    flow_last = model.model_dict["upper"]["flow"].values.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        del model
        gc.collect()
    out = xr.open_zarr(control["output_serial_zarr"], consolidated=False)
    np.testing.assert_array_equal(out["flow"].values[-1], flow_last)
