"""Serial two-grid regression: Upper on grid1 ("hru") -> Map -> Lower on grid2
("segment").

The first multi-grid toy (Step A). Two grids of different sizes (both on dim
"space" -- separate datasets, so no conflict), each hosting one process, coupled
by a dense-weight Map that aggregates Upper's `flow` (hru) into Lower's `flow`
input (segment). Serial, no MPI. Validates: process->grid co-registration, the
grid-aware Model build (per-grid datasets), and the Map wiring + application.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from map import Map
from model import Model
from processes_concrete import Lower, Upper

N_HRU = 6
N_SEG = 3


class TestTwoGrid:
    """Serial Upper(hru) -> Map -> Lower(segment) regression."""

    @pytest.fixture
    def weights(self):
        """(N_SEG, N_HRU): each segment aggregates two HRUs by averaging."""
        w = np.zeros((N_SEG, N_HRU))
        for seg in range(N_SEG):
            w[seg, 2 * seg] = 0.5
            w[seg, 2 * seg + 1] = 0.5
        return w

    @pytest.fixture
    def toy(self, dimensions):
        """Upper's inputs on hru (N_HRU), Lower's on segment (N_SEG); both use
        dim "space" -- the grids differ by size + dict key. `flow` is NOT given
        to Lower (the Map feeds it)."""
        n_time = dimensions["n_time"]
        rng = np.random.default_rng(11)
        sin = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / n_time))
        pu1 = rng.uniform(0.1, 1, (12, N_HRU))

        # -- grid1 "hru" (Upper) --
        up_params = xr.Dataset(
            dict(
                param_up_0=(["space"], rng.uniform(0.1, 1, N_HRU)),
                param_up_1=(["month", "space"], pu1),
                param_common=(["space"], np.zeros(N_HRU)),
            ),
            coords=dict(month=("month", np.arange(1, 13))),
        )
        up_forcing_0 = xr.DataArray(
            sin[:, None] + rng.uniform(10, 100, N_HRU)[None, :],
            dims=["time", "space"],
            coords={"time": dimensions["time"]},
        )
        up_flow_initial = xr.DataArray(
            rng.uniform(100, 1000, N_HRU), dims=["space"]
        )

        # -- grid2 "segment" (Lower) --
        low_params = xr.Dataset(
            dict(
                param_low_0=(["space"], rng.uniform(0.17, 0.23, N_SEG)),
                param_common=(["space"], np.zeros(N_SEG)),
            )
        )
        low_forcing_low = xr.DataArray(
            rng.uniform(1, 3, (n_time, N_SEG)),
            dims=["time", "space"],
            coords={"time": dimensions["time"]},
        )
        low_storage_initial = xr.DataArray(
            rng.uniform(100, 500, N_SEG), dims=["space"]
        )
        return dict(
            up_params=up_params,
            up_forcing_0=up_forcing_0,
            up_flow_initial=up_flow_initial,
            low_params=low_params,
            low_forcing_low=low_forcing_low,
            low_storage_initial=low_storage_initial,
        )

    @pytest.fixture
    def answers(self, toy, weights, dimensions):
        """Two-grid ground truth: Upper.flow (hru) -> W @ flow -> Lower.storage
        (segment)."""
        n_time = dimensions["n_time"]
        time = np.asarray(dimensions["time"])
        months = time.astype("datetime64[M]").astype(int) % 12
        f0 = toy["up_forcing_0"].values
        pu1 = toy["up_params"]["param_up_1"].values
        flow_init = toy["up_flow_initial"].values
        fl = toy["low_forcing_low"].values
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

    def test_two_grid_map(self, toy, weights, dimensions, answers):
        process_dict = {
            "upper": {
                "class": Upper,
                "discretization": "grid1",
                "parameters": toy["up_params"],
                "forcing_0": toy["up_forcing_0"],
                "flow_initial": toy["up_flow_initial"],
            },
            "lower": {
                "class": Lower,
                "discretization": "grid2",
                "parameters": toy["low_params"],
                "forcing_low": toy["low_forcing_low"],
                "storage_initial": toy["low_storage_initial"],
                # NOTE: "flow" is NOT provided -- the Map feeds it.
            },
        }
        maps = {"g1_to_g2": Map("grid1", "flow", "grid2", "flow", weights)}

        dt = np.float64(1.0)
        with Model(process_dict, {}, maps=maps) as model:
            model.run(dt, np.int32(dimensions["n_time"]))

        # -- two grids exist --
        assert set(model.discretizations) == {"grid1", "grid2"}

        # -- Upper's flow on hru --
        np.testing.assert_allclose(
            model.model_dict["upper"]["flow"].values,
            answers["flow"][-1],
            rtol=1e-12,
        )
        # -- Lower's storage on segment --
        np.testing.assert_allclose(
            model.model_dict["lower"]["storage"].values,
            answers["storage"][-1],
            rtol=1e-12,
        )
        # -- Map coupling: Lower's flow == W @ Upper.flow (last step) --
        np.testing.assert_allclose(
            model.model_dict["lower"]["flow"].values,
            weights @ answers["flow"][-1],
            rtol=1e-12,
        )
