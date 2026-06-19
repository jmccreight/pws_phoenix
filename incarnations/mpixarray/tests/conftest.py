"""Shared fixtures for the incarnations/mpixarray regression tests.

The serial (test_up_low_regression.py) and MPI
(test_up_low_regression_mpi.py) regressions build the SAME toy
Upper/Lower model from ONE input dataset (`make_toy_input`) and validate against
the SAME vectorized answers (`compute_answers`). Differences between the two are
intentional and explicit:

  - serial brings the input into memory (or round-trips it through per-input
    files); MPI writes ONE combined file and streams it.
  - `time` and `space` are real dim-coordinates (required by mpixarray's
    `set_streaming`/`parallelize`; the serial path reads them the same way).
  - serial uses `param_up_1` (a time-varying parameter); the MPI streaming path
    drops it and warns (see ModelMPI) -- a Phase 2 design question.
  - output backend differs (serial zarr store vs MPI streamed NetCDF).
"""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def dimensions():
    """Toy model dimensions shared by both regressions."""
    n_years = 1
    n_space = 20
    start_year = 2000
    start_time = np.datetime64(f"{start_year}-01-01")
    end_time = (
        np.datetime64(f"{start_year + n_years}-01-01") - np.timedelta64(1, "D")
    )
    time = np.arange(start_time, end_time, dtype="datetime64[D]")
    space = np.arange(n_space)
    return {
        "n_years": n_years,
        "n_space": n_space,
        "n_time": len(time),
        "time": time,
        "space": space,
    }


@pytest.fixture(scope="session")
def make_toy_input():
    """Factory returning the unified toy input Dataset.

    `time`/`space` are dim-coordinates. Vars: forcing_0, forcing_common
    (time, space); param_up_0, param_up_1 (time-varying), param_low_0,
    param_common, flow_initial, storage_initial (space). Deterministic per seed.
    """

    def _make(dimensions: dict, seed: int = 42) -> xr.Dataset:
        n_years = dimensions["n_years"]
        n_time = dimensions["n_time"]
        n_space = dimensions["n_space"]
        rng = np.random.default_rng(seed)
        sin_data = np.sin(
            np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)
        )
        shifts = rng.uniform(10, 100, n_space)
        forcing_0 = sin_data[:, np.newaxis] + shifts[np.newaxis, :]
        return xr.Dataset(
            data_vars=dict(
                forcing_0=(["time", "space"], forcing_0),
                forcing_common=(["time", "space"], np.ones((n_time, n_space))),
                param_up_0=(["space"], rng.uniform(0.1, 1, n_space)),
                param_up_1=(
                    ["time", "space"],
                    rng.uniform(0.1, 1, (n_time, n_space)),
                ),
                param_low_0=(["space"], rng.uniform(0.17, 0.23, n_space)),
                param_common=(["space"], np.zeros(n_space)),
                flow_initial=(["space"], rng.uniform(100, 1000, n_space)),
                storage_initial=(["space"], rng.uniform(100, 500, n_space)),
            ),
            coords=dict(
                time=("time", dimensions["time"]),
                space=("space", dimensions["space"]),
            ),
        )

    return _make


@pytest.fixture(scope="session")
def compute_answers():
    """Factory for the vectorized ground-truth Upper/Lower solution."""

    def _compute(forcing_0, flow_initial, storage_initial, n_time) -> dict:
        forcing_0 = np.asarray(forcing_0)
        flow_initial = np.asarray(flow_initial)
        storage_initial = np.asarray(storage_initial)
        n_space = forcing_0.shape[1]

        expected_flow = np.zeros((n_time, n_space))
        expected_flow_prev = np.zeros((n_time, n_space))
        for tt in range(n_time):
            expected_flow_prev[tt, :] = (
                flow_initial if tt == 0 else expected_flow[tt - 1, :]
            )
            expected_flow[tt, :] = (
                expected_flow_prev[tt, :] * 0.95 + forcing_0[tt, :]
            )

        expected_storage = np.zeros((n_time, n_space))
        expected_storage_prev = np.zeros((n_time, n_space))
        for tt in range(n_time):
            expected_storage_prev[tt, :] = (
                storage_initial if tt == 0 else expected_storage[tt - 1, :]
            )
            expected_storage[tt, :] = (
                expected_storage_prev[tt, :] * 0.95
                + expected_flow[tt, :] * 0.12
            )

        return {
            "expected_flow": expected_flow,
            "expected_flow_prev": expected_flow_prev,
            "expected_storage": expected_storage,
            "expected_storage_prev": expected_storage_prev,
        }

    return _compute
