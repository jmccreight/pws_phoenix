"""Serial two-grid regression: Upper on grid "hru" -> Map -> Lower on grid
"segment".

The first multi-grid toy (Step A). Two grids of different sizes, each on
its own real dim (the grid key: "hru" / "segment") in a separate dataset,
each hosting one process, coupled by a dense-weight Map that aggregates
Upper's `flow` (hru) into Lower's `flow` input (segment). Serial, no MPI.
Validates: process->grid co-registration, the grid-aware Model build
(per-grid datasets), and the Map wiring + application.
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
        """Upper's inputs on dim "hru" (N_HRU), Lower's on "segment" (N_SEG) --
        each grid's key is its real dim; separate datasets. `flow` is NOT given
        to Lower (the Map feeds it)."""
        n_time = dimensions["n_time"]
        rng = np.random.default_rng(11)
        sin = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / n_time))
        pu1 = rng.uniform(0.1, 1, (12, N_HRU))

        # -- grid "hru" (Upper) --
        up_params = xr.Dataset(
            dict(
                param_up_0=(["hru"], rng.uniform(0.1, 1, N_HRU)),
                param_up_1=(["month", "hru"], pu1),
                param_shared_name=(["hru"], np.zeros(N_HRU)),
            ),
            coords=dict(month=("month", np.arange(1, 13))),
        )
        forcing_up = xr.DataArray(
            sin[:, None] + rng.uniform(10, 100, N_HRU)[None, :],
            dims=["time", "hru"],
            coords={"time": dimensions["time"]},
        )
        up_flow_initial = xr.DataArray(
            rng.uniform(100, 1000, N_HRU), dims=["hru"]
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

    @pytest.fixture
    def answers(self, toy, weights, dimensions):
        """Two-grid ground truth: Upper.flow (hru) -> W @ flow -> Lower.storage
        (segment)."""
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

    @pytest.fixture
    def process_dict(self, toy):
        return {
            "upper": {
                "class": Upper,
                "discretization": "hru",
                "parameters": toy["up_params"],
                "forcing_up": toy["forcing_up"],
                "flow_initial": toy["up_flow_initial"],
            },
            "lower": {
                "class": Lower,
                "discretization": "segment",
                "parameters": toy["low_params"],
                "forcing_low": toy["forcing_low"],
                "storage_initial": toy["low_storage_initial"],
                # NOTE: "flow" is NOT provided -- the Map feeds it.
            },
        }

    def test_two_grid_map(self, process_dict, weights, dimensions, answers):
        maps = {
            "hru_to_seg": Map(
                weights=weights,
                grid={"hru": "segment"},
                variable={"flow": "flow"},
            )
        }

        dt = np.float64(1.0)
        with Model(process_dict, {}, maps=maps) as model:
            model.run(dt, np.int32(dimensions["n_time"]))

        # -- two grids exist --
        assert set(model.discretizations) == {"hru", "segment"}

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

    def test_unresolved_input_raises(self, process_dict):
        """Without the Map, Lower's cross-grid `flow` input is unresolved:
        Model construction fails fast (assembly-time validation) instead of
        KeyError-ing mid-run."""
        with pytest.raises(ValueError, match="unresolved input"):
            Model(process_dict, {})

    def test_consumer_before_producer_raises(self, process_dict, weights):
        """Ordering the consumer (Lower) before the mapped variable's writer
        (Upper) would silently carry last step's flow across the boundary:
        assembly raises instead (one-pass order validation)."""
        reversed_dict = {
            "lower": process_dict["lower"],
            "upper": process_dict["upper"],
        }
        maps = {
            "hru_to_seg": Map(
                weights=weights,
                grid={"hru": "segment"},
                variable={"flow": "flow"},
            )
        }
        with pytest.raises(ValueError, match="writer"):
            Model(reversed_dict, {}, maps=maps)

    def test_map_applies_once_per_step(
        self, process_dict, weights, dimensions
    ):
        """A second consumer of the mapped variable does not re-apply the
        Map: it is applied exactly once per timestep, before its first
        consumer; later consumers re-read the same target buffer."""

        class CountingMap(Map):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.n_applies = 0

            def apply(self, source_ds):
                self.n_applies += 1
                super().apply(source_ds)

        # A 2nd process on "segment" consuming `flow` (same class: its state
        # vars are structurally shared with "lower", which is fine here --
        # only the apply count is under test).
        process_dict["lower2"] = dict(process_dict["lower"])
        counting_map = CountingMap(
            weights=weights,
            grid={"hru": "segment"},
            variable={"flow": "flow"},
        )
        maps = {"hru_to_seg": counting_map}

        n_time = dimensions["n_time"]
        with Model(process_dict, {}, maps=maps) as model:
            assert model._proc_maps["lower"] == [counting_map]
            assert model._proc_maps["lower2"] == []
            model.run(np.float64(1.0), np.int32(n_time))

        assert counting_map.n_applies == n_time
