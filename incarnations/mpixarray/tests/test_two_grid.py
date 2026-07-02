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

sys.path.append(str(pl.Path(__file__).parent.parent))
from map import Map
from model import Model
from processes_concrete import Lower, Upper


class TestTwoGrid:
    """Serial Upper(hru) -> Map -> Lower(segment) regression.

    The toy data, weights, and ground truth live in conftest.py (shared
    with the MPI Step B regression, test_two_grid_mpi.py)."""

    @pytest.fixture
    def weights(self, two_grid_weights):
        return two_grid_weights

    @pytest.fixture
    def toy(self, dimensions, make_two_grid_toy):
        return make_two_grid_toy(dimensions)

    @pytest.fixture
    def answers(self, toy, weights, dimensions, compute_two_grid_answers):
        return compute_two_grid_answers(toy, weights, dimensions)

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
