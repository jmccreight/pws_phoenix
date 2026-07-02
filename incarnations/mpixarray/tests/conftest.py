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
  - both serial and MPI now use `param_up_1`, a cyclic-monthly time-varying
    parameter `(month, space)`, indexed each step via `time.month` (it stays
    resident through `set_streaming`, since `month` is not the streaming dim).
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
    end_time = np.datetime64(f"{start_year + n_years}-01-01") - np.timedelta64(
        1, "D"
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

    `time`/`space`/`month` are dim-coordinates. Vars: forcing_up, forcing_low
    (time, space; independent forcings for Upper/Lower); param_up_1 (month,
    space, cyclic-monthly); param_up_0, param_low_0, param_shared_name,
    flow_initial, storage_initial (space). Deterministic per seed.
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
        forcing_up = sin_data[:, np.newaxis] + shifts[np.newaxis, :]
        # Lower's own forcing -- deterministic and unrelated to forcing_up (cos
        # + a fixed per-space shift), so it doesn't perturb the rng draws.
        cos_data = np.cos(
            np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)
        )
        shifts_low = np.linspace(1.0, 5.0, n_space)
        forcing_low = cos_data[:, np.newaxis] + shifts_low[np.newaxis, :]
        return xr.Dataset(
            data_vars=dict(
                forcing_up=(["time", "space"], forcing_up),
                forcing_low=(["time", "space"], forcing_low),
                param_up_0=(["space"], rng.uniform(0.1, 1, n_space)),
                param_up_1=(
                    ["month", "space"],
                    rng.uniform(0.1, 1, (12, n_space)),
                ),
                param_low_0=(["space"], rng.uniform(0.17, 0.23, n_space)),
                param_shared_name=(["space"], np.zeros(n_space)),
                flow_initial=(["space"], rng.uniform(100, 1000, n_space)),
                storage_initial=(["space"], rng.uniform(100, 500, n_space)),
            ),
            coords=dict(
                time=("time", dimensions["time"]),
                space=("space", dimensions["space"]),
                month=("month", np.arange(1, 13)),
            ),
        )

    return _make


# ---------------------------------------------------------------------------
# Two-grid toy (Upper on "hru" -> Map -> Lower on "segment"): shared by the
# serial (test_two_grid.py) and MPI Step B (test_two_grid_mpi.py) tests.
# Deterministic per seed, so every MPI rank rebuilds identical data (the
# replication of the serial "segment" grid relies on this).
# ---------------------------------------------------------------------------

# N_HRU divides evenly over 2 and 4 ranks (uneven decomposition is a
# separate concern from the Step B cross-grid comm under test).
N_HRU = 8
N_SEG = 4


@pytest.fixture(scope="session")
def two_grid_weights():
    """(N_SEG, N_HRU): each segment aggregates two HRUs by averaging."""
    ww = np.zeros((N_SEG, N_HRU))
    for seg in range(N_SEG):
        ww[seg, 2 * seg] = 0.5
        ww[seg, 2 * seg + 1] = 0.5
    return ww


@pytest.fixture(scope="session")
def make_two_grid_toy():
    """Factory for the two-grid toy inputs: Upper's on dim "hru" (N_HRU),
    Lower's on "segment" (N_SEG) -- each grid's key is its real dim;
    separate datasets. `flow` is NOT given to Lower (the Map feeds it).
    `hru` and `time` are real dim-coordinates (mpixarray's `parallelize`/
    `set_streaming` need them on the distributed grid)."""

    def _make(dimensions: dict, seed: int = 11) -> dict:
        n_time = dimensions["n_time"]
        rng = np.random.default_rng(seed)
        sin = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / n_time))
        pu1 = rng.uniform(0.1, 1, (12, N_HRU))
        hru_coord = np.arange(N_HRU)

        # -- grid "hru" (Upper) --
        up_params = xr.Dataset(
            dict(
                param_up_0=(["hru"], rng.uniform(0.1, 1, N_HRU)),
                param_up_1=(["month", "hru"], pu1),
                param_shared_name=(["hru"], np.zeros(N_HRU)),
            ),
            coords=dict(
                month=("month", np.arange(1, 13)),
                hru=("hru", hru_coord),
            ),
        )
        forcing_up = xr.DataArray(
            sin[:, None] + rng.uniform(10, 100, N_HRU)[None, :],
            dims=["time", "hru"],
            coords={"time": dimensions["time"], "hru": hru_coord},
        )
        up_flow_initial = xr.DataArray(
            rng.uniform(100, 1000, N_HRU),
            dims=["hru"],
            coords={"hru": hru_coord},
        )

        # -- grid "segment" (Lower) --
        low_params = xr.Dataset(
            dict(
                param_low_0=(["segment"], rng.uniform(0.17, 0.23, N_SEG)),
                param_shared_name=(["segment"], np.zeros(N_SEG)),
            )
        )
        forcing_low = xr.DataArray(
            rng.uniform(1, 3, (n_time, N_SEG)),
            dims=["time", "segment"],
            coords={"time": dimensions["time"]},
        )
        low_storage_initial = xr.DataArray(
            rng.uniform(100, 500, N_SEG), dims=["segment"]
        )
        return dict(
            up_params=up_params,
            forcing_up=forcing_up,
            up_flow_initial=up_flow_initial,
            low_params=low_params,
            forcing_low=forcing_low,
            low_storage_initial=low_storage_initial,
        )

    return _make


@pytest.fixture(scope="session")
def compute_two_grid_answers():
    """Factory for the two-grid ground truth: Upper.flow (hru) ->
    W @ flow -> Lower.storage (segment)."""

    def _compute(toy: dict, weights, dimensions: dict) -> dict:
        n_time = dimensions["n_time"]
        time = np.asarray(dimensions["time"])
        months = time.astype("datetime64[M]").astype(int) % 12
        f0 = toy["forcing_up"].values
        pu1 = toy["up_params"]["param_up_1"].values
        flow_init = toy["up_flow_initial"].values
        fl = toy["forcing_low"].values
        stor_init = toy["low_storage_initial"].values

        flow = np.zeros((n_time, N_HRU))
        flow_prev = np.zeros((n_time, N_HRU))
        for tt in range(n_time):
            flow_prev[tt] = flow_init if tt == 0 else flow[tt - 1]
            flow[tt] = flow_prev[tt] * 0.95 + f0[tt] * pu1[months[tt]]

        storage = np.zeros((n_time, N_SEG))
        storage_prev = np.zeros((n_time, N_SEG))
        for tt in range(n_time):
            flow_seg = weights @ flow[tt]
            storage_prev[tt] = stor_init if tt == 0 else storage[tt - 1]
            storage[tt] = (
                storage_prev[tt] * 0.95 + flow_seg * 0.12 + fl[tt] * 0.10
            )
        return dict(flow=flow, storage=storage)

    return _compute


@pytest.fixture(scope="session")
def compute_answers():
    """Factory for the vectorized ground-truth Upper/Lower solution."""

    def _compute(
        forcing_up,
        flow_initial,
        storage_initial,
        n_time,
        param_up_1,
        time,
        forcing_low,
    ) -> dict:
        forcing_up = np.asarray(forcing_up)
        flow_initial = np.asarray(flow_initial)
        storage_initial = np.asarray(storage_initial)
        param_up_1 = np.asarray(param_up_1)
        forcing_low = np.asarray(forcing_low)
        n_space = forcing_up.shape[1]

        # day -> month index (0-11), matching Time.month - 1
        months = np.asarray(time).astype("datetime64[M]").astype(int) % 12

        expected_flow = np.zeros((n_time, n_space))
        expected_flow_prev = np.zeros((n_time, n_space))
        for tt in range(n_time):
            expected_flow_prev[tt, :] = (
                flow_initial if tt == 0 else expected_flow[tt - 1, :]
            )
            expected_flow[tt, :] = (
                expected_flow_prev[tt, :] * 0.95
                + forcing_up[tt, :] * param_up_1[months[tt], :]
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
                + forcing_low[tt, :] * 0.10
            )

        return {
            "expected_flow": expected_flow,
            "expected_flow_prev": expected_flow_prev,
            "expected_storage": expected_storage,
            "expected_storage_prev": expected_storage_prev,
        }

    return _compute
