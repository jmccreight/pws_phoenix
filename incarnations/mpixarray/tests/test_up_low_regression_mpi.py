"""
test_up_low_regression_mpi.py
=============================
pytest-mpi MPI streaming regression for the Upper/Lower toy model via ModelMPI.

Shares `dimensions`, `make_toy_input`, and `compute_answers` with the serial
regression (conftest.py). The whole collective pipeline -- write the ONE
combined input on rank 0, build ModelMPI, run the streaming loop, finalize,
gather results -- lives in module-scoped fixtures, where ALL collective MPI ops
happen. The test_* methods are pure (collective-free) asserts, so a failing rank
can never interrupt a collective and hang the others.

Phase 1: one space-decomposed dataset streamed over time. `flow` is streamed to
disk and validated over all timesteps (global); `storage_previous` (not streamed
-- the one-`to_netcdf`-var limit) is validated from final state, gathered
globally. `param_up_1` is in the file but dropped by the streaming path
(ModelMPI warns -- asserted explicitly).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_up_low_regression_mpi.py -v

Prerequisites: pytest-mpi installed; run under mpirun with >= 2 ranks.
"""

import pathlib as pl
import shutil
import sys
import tempfile
import warnings

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(pl.Path(__file__).parent.parent))
from model import ModelMPI
from processes_concrete import Lower, Upper


@pytest.fixture(scope="module")
def mpi_paths(dimensions, make_toy_input):
    """Write the unified toy input to ONE combined file on rank 0; broadcast the
    temp dir. Yields the input/output paths; rank 0 cleans up afterward."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert (
        tmp is not None
    )  # set on rank 0, broadcast to all -> narrows str|None -> str
    data_dir = pl.Path(tmp) / "toy_model_mpi_data"
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        make_toy_input(dimensions).to_netcdf(input_file)
    comm.Barrier()
    yield {"input_file": input_file, "output_file": output_file}

    # --- Teardown (an important pytest + MPI detail) ---
    # A yield-fixture is a generator: pytest runs it to the `yield` for setup,
    # hands the dict to the tests, then at the END OF THIS FIXTURE'S SCOPE
    # resumes the generator past the `yield` (calls next() again) to run the
    # code below. Scope is "module", so this fires after the LAST test in this
    # file. It runs even if tests failed, provided setup reached the `yield`.
    #
    # Under MPI this teardown runs in EVERY rank's pytest process, but only
    # rank 0 owns the temp dir, so only rank 0 removes it. There is NO barrier
    # here on purpose: a rank that failed a test mid-module still reaches its
    # own teardown, so no rank hangs waiting on a collective.
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def answers(dimensions, make_toy_input, compute_answers):
    """Vectorized ground truth. The toy data is deterministic, so every rank
    recomputes identical answers from an in-memory copy (no file read needed)."""
    ds = make_toy_input(dimensions)
    return compute_answers(
        ds["forcing_0"].values,
        ds["flow_initial"].values,
        ds["storage_initial"].values,
        dimensions["n_time"],
    )


@pytest.fixture(scope="module")
def mpi_run(mpi_paths):
    """Build + run + finalize ModelMPI ONCE. Every collective MPI op lives here,
    so the test_* methods are pure asserts."""
    comm = MPI.COMM_WORLD
    process_dict = {"upper": {"class": Upper}, "lower": {"class": Lower}}
    control = {
        "input_file": mpi_paths["input_file"],
        "output_file": mpi_paths["output_file"],
        "output_var_names": [
            "flow"
        ],  # one streamed output (see ModelMPI note)
    }

    # ModelMPI warns that the time-varying param_up_1 is dropped -- capture it so
    # the drop is asserted explicitly (and warning-as-error configs don't fail).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ModelMPI(process_dict, control)
    param_up_1_warned = any("param_up_1" in str(w.message) for w in caught)

    model.run(np.float64(1.0))

    upper = model.model_dict["upper"]
    lower = model.model_dict["lower"]
    # storage_previous isn't streamed, so gather its final state globally as the
    # Lower-process check (single scheme -> rank-ordered contiguous blocks).
    storage_prev_final = np.concatenate(
        comm.allgather(model._ds_mpi["storage_previous"].values.copy())
    )
    result = {
        "output_file": mpi_paths["output_file"],
        "param_up_1_warned": param_up_1_warned,
        "storage_prev_final": storage_prev_final,
        # the shared buffer/ref checks need to happen before model.finalize()
        # because that deletes/closees ds_mpi.
        "shared_param_common": (
            upper._obj["param_common"].values
            is lower._obj["param_common"].values
        ),
        "shared_forcing_common": (
            upper._obj["forcing_common"].values
            is lower._obj["forcing_common"].values
        ),
        "shared_flow": upper._obj["flow"].values is lower._obj["flow"].values,
    }
    model.finalize()
    comm.Barrier()  # ensure the output file is fully flushed before reads
    return result


@pytest.mark.mpi(min_size=2)
class TestRegressionMPI:
    """MPI streaming regression for the Upper/Lower toy model via ModelMPI."""

    # -- structural buffer sharing (one ds_mpi) --
    # Asserts happen here but the boolean was evaluated before model.finalize
    def test_shared_param_common(self, mpi_run):
        assert mpi_run["shared_param_common"]

    def test_shared_forcing_common(self, mpi_run):
        assert mpi_run["shared_forcing_common"]

    def test_shared_flow_upper_lower(self, mpi_run):
        assert mpi_run["shared_flow"]

    # -- intentional difference vs serial: time-varying param dropped + warned --
    def test_param_up_1_dropped_warns(self, mpi_run):
        assert mpi_run["param_up_1_warned"]

    # -- streamed flow over all timesteps (global), validated on rank 0 --
    def test_streamed_flow_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.open_dataset(mpi_run["output_file"]) as ds_out:
            flow_out = ds_out["flow_out"].values  # (n_time, n_space) global
        np.testing.assert_allclose(
            flow_out, answers["expected_flow"], rtol=1e-12
        )

    # -- Lower process: final storage_previous, gathered globally --
    def test_storage_previous_final(self, mpi_run, answers):
        np.testing.assert_allclose(
            mpi_run["storage_prev_final"],
            answers["expected_storage_prev"][-1, :],
            rtol=1e-12,
        )
