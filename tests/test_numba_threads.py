"""control["numba_n_threads"]: the Model holds numba at that count.

The main kernel of every hru-grid process is compiled threaded
(parallel=True, a numba.prange over the hrus), always. The Model sets
the number of threads they run on (Model._set_numba_threads): 1 unless
control["numba_n_threads"] says more, at build and again at the top of
run(). numba's maximum for the process (numba.config.NUMBA_NUM_THREADS)
is fixed when numba is imported, so these tests take it as they find
it: on a machine where it is 1, "several threads" is 1 thread and the
comparisons still hold.

The model is the drb_2yr NHM configuration built from its legacy
control file: all six threaded hru kernels plus the channel.
"""

import numba
import numpy as np
import pytest
from conftest import PWS_TEST_DATA
from pws_phoenix.model import Model
from pws_phoenix.prms_translate import model_from_control

CONTROL_FILE = PWS_TEST_DATA / "drb_2yr" / "nhm.control"
N_MAX = int(numba.config.NUMBA_NUM_THREADS)
N_SEVERAL = min(3, N_MAX)
N_STEPS = 30
DT = np.float64(86400.0)

pytestmark = pytest.mark.skipif(
    not CONTROL_FILE.exists(),
    reason=f"drb_2yr legacy files missing: {CONTROL_FILE}",
)


@pytest.fixture(autouse=True)
def one_thread_afterwards():
    """Leave numba at 1 thread for the tests that follow in this
    process (conftest.pytest_sessionstart sets the same)."""
    yield
    numba.set_num_threads(1)


def _numba_messages(records) -> list[str]:
    return [
        str(ww.message)
        for ww in records
        if "numba_n_threads" in str(ww.message)
    ]


def test_default_holds_one_thread(monkeypatch, recwarn):
    """No control["numba_n_threads"]: the build sets numba to 1
    thread, whatever it was, and says nothing when NUMBA_NUM_THREADS
    is not set."""
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    numba.set_num_threads(N_SEVERAL)
    model = model_from_control(CONTROL_FILE, control={})
    assert numba.get_num_threads() == 1
    assert _numba_messages(recwarn.list) == []
    model.finalize()


def test_several_threads_bit_identical():
    """A run on several threads equals the run on 1 thread bit for
    bit, in every variable of every process (nothing crosses hrus, so
    there is no reduction and no float reordering). Each run() sets
    its own model's count: the two models are built first, so the
    second build leaves numba at the other model's count."""
    model_one = model_from_control(CONTROL_FILE, control={"budget": "error"})
    model_several = model_from_control(
        CONTROL_FILE,
        control={"budget": "error", "numba_n_threads": N_SEVERAL},
    )
    assert numba.get_num_threads() == N_SEVERAL

    model_one.run(DT, np.int32(N_STEPS))
    assert numba.get_num_threads() == 1
    model_several.run(DT, np.int32(N_STEPS))
    assert numba.get_num_threads() == N_SEVERAL

    n_compared = 0
    for proc_name, proc_one in model_one.model_dict.items():
        proc_several = model_several.model_dict[proc_name]
        for name in proc_one.get_variables(include_budget=True):
            np.testing.assert_array_equal(
                proc_several._refs[name],
                proc_one._refs[name],
                err_msg=f"{proc_name}.{name}: {N_SEVERAL} threads vs 1",
            )
            n_compared += 1
    # the six hru processes and the channel declare well over this
    assert n_compared > 100
    model_one.finalize()
    model_several.finalize()


@pytest.mark.parametrize("n_threads", [0, N_MAX + 1])
def test_count_outside_range_raises(n_threads):
    """Below 1 or above numba's maximum for the process raises at the
    build, naming both. (The check comes before any process is
    touched: an empty process_dict reaches it.)"""
    with pytest.raises(ValueError, match="numba_n_threads") as err:
        Model({}, {"numba_n_threads": n_threads})
    assert str(N_MAX) in str(err.value)


def test_environment_variable_without_control_warns(monkeypatch, recwarn):
    """NUMBA_NUM_THREADS set above 1 and no control["numba_n_threads"]:
    the run is on 1 thread and the build says so. With a maximum of 1
    there is nothing to say."""
    monkeypatch.setenv("NUMBA_NUM_THREADS", str(N_MAX))
    model = model_from_control(CONTROL_FILE, control={})
    messages = _numba_messages(recwarn.list)
    if N_MAX > 1:
        assert len(messages) == 1
        assert f"NUMBA_NUM_THREADS={N_MAX}" in messages[0]
    else:
        assert messages == []
    assert numba.get_num_threads() == 1
    model.finalize()


def test_control_given_does_not_warn(monkeypatch, recwarn):
    """With control["numba_n_threads"] given, NUMBA_NUM_THREADS is
    only the maximum: no warning."""
    monkeypatch.setenv("NUMBA_NUM_THREADS", str(N_MAX))
    model = model_from_control(
        CONTROL_FILE, control={"numba_n_threads": N_SEVERAL}
    )
    assert _numba_messages(recwarn.list) == []
    assert numba.get_num_threads() == N_SEVERAL
    model.finalize()
