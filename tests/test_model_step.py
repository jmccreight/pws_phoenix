"""Model.step(): one timestep at a time, and run() as a loop of them.

On the Upper/Lower toys (no external data):

- n calls of ``step(dt)`` leave the same state as one ``run(dt, n)``;
- ``run`` CONTINUES from the last step taken (``run(k)`` then
  ``run(n - k)`` == ``run(n)``), it does not restart at time zero;
- stepping past the inputs' time axis raises;
- ``ModelMPI.step`` raises NotImplementedError (mpixarray owns the
  streaming time loop).
"""

import numpy as np
import pytest
from pws_phoenix.model import Model, ModelMPI
from pws_phoenix.processes_concrete import Lower, Upper

PARAM_NAMES = [
    "param_up_0",
    "param_up_1",
    "param_low_0",
    "param_shared_name",
]
STATE_NAMES = ("flow", "flow_previous", "storage", "storage_previous")
DT = np.float64(1.0)


def _make_model(toy_ds):
    process_dict = {
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
    return Model(process_dict, {})


def _state(model):
    grid_ds = model.discretizations["space"].dataset
    return {nn: grid_ds[nn].values.copy() for nn in STATE_NAMES}


@pytest.fixture(scope="module")
def toy_ds(dimensions, make_toy_input):
    return make_toy_input(dimensions)


def test_steps_equal_run(toy_ds, dimensions):
    ntime = dimensions["n_time"]
    with _make_model(toy_ds) as by_run:
        by_run.run(DT, np.int32(ntime))
        expected = _state(by_run)
        assert int(by_run.current_time_index[0]) == ntime - 1
    with _make_model(toy_ds) as by_step:
        for tt in range(ntime):
            by_step.step(DT)
            assert int(by_step.current_time_index[0]) == tt
            assert by_step.time.current == by_step.times.values[tt]
        got = _state(by_step)
    for nn in STATE_NAMES:
        np.testing.assert_array_equal(got[nn], expected[nn], err_msg=nn)


def test_run_continues(toy_ds, dimensions):
    ntime = dimensions["n_time"]
    kk = ntime // 2
    with _make_model(toy_ds) as whole:
        whole.run(DT, np.int32(ntime))
        expected = _state(whole)
    with _make_model(toy_ds) as pieces:
        pieces.run(DT, np.int32(kk))
        assert int(pieces.current_time_index[0]) == kk - 1
        pieces.run(DT, np.int32(ntime - kk))
        got = _state(pieces)
    for nn in STATE_NAMES:
        np.testing.assert_array_equal(got[nn], expected[nn], err_msg=nn)


def test_step_past_end_raises(toy_ds, dimensions):
    ntime = dimensions["n_time"]
    with _make_model(toy_ds) as model:
        model.run(DT, np.int32(ntime))
        with pytest.raises(ValueError, match="past the end"):
            model.step(DT)
        model.finalize()
        with pytest.raises(RuntimeError, match="finalized"):
            model.step(DT)


def test_mpi_step_not_implemented():
    with pytest.raises(NotImplementedError, match="mpixarray"):
        ModelMPI.step(None, DT)  # type: ignore[arg-type]
