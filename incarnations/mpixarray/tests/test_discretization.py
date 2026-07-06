"""Part-(a) framework tests: dis-owned parameters, topological_order,
the per-process initialize() hook, and kind="parameter_derived".

Self-contained (tiny in-memory toy; no generated pywatershed data).
networkx (used lazily by topological_order) is a DECLARED dependency
(environment.yaml) -- a missing install should fail loudly, not skip.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from model import Model
from process import DataArrayMeta, Process

N_XY = 4


class DerivedToy(Process):
    """Minimal process exercising dis-sourced parameters and
    parameter_derived + initialize()."""

    base = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="dis-owned parameter (sourced from the dis)",
    )
    forcing_toy = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="toy forcing",
    )
    doubled = DataArrayMeta(
        kind="parameter_derived",
        dims=("space",),
        dtype=np.float64,
        description="computed by initialize(): 2 * base",
    )
    counts = DataArrayMeta(
        kind="parameter_derived",
        dims=("space",),
        dtype=np.int64,
        description="computed by initialize(): int64 exercise",
    )
    total = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="doubled + forcing",
    )

    def initialize(self) -> None:
        self._obj["doubled"].values[:] = 2.0 * self._obj["base"].values
        self._obj["counts"].values[:] = np.arange(
            self._obj["base"].values.shape[0], dtype=np.int64
        )

    def advance(self) -> None:
        pass

    def calculate(self, dt: np.float64, time) -> None:
        self._obj["total"].values[:] = (
            self._obj["doubled"].values + self._obj["forcing_toy"].values
        )


@pytest.fixture
def dis_parameters():
    return xr.Dataset(
        {"base": ("xy", np.arange(1.0, N_XY + 1))},
        coords={"xy": np.arange(N_XY)},
    )


@pytest.fixture
def forcing():
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    return xr.DataArray(
        np.ones((2, N_XY)),
        dims=("time", "xy"),
        coords={"time": times, "xy": np.arange(N_XY)},
        name="forcing_toy",
    )


@pytest.fixture
def model(dis_parameters, forcing):
    process_dict = {
        "toy": {
            "class": DerivedToy,
            "discretization": "xy",
            "forcing_toy": forcing,
            # NO "parameters": base must arrive via the dis
        },
    }
    discretizations = {
        "xy": Discretization(["xy"], parameters=dis_parameters),
    }
    with Model(process_dict, {}, discretizations=discretizations) as mm:
        mm.run(np.float64(1.0), np.int32(2))
    return mm


class TestDisParametersAndInitialize:
    def test_dis_sourced_parameter_zero_copy(self, model, dis_parameters):
        """base came from the dis (no process 'parameters' supplied), by
        reference, and the read-only flag reaches the dis's array."""
        proc = model.model_dict["toy"]
        assert proc["base"].values is dis_parameters["base"].values
        assert not dis_parameters["base"].values.flags.writeable

    def test_derived_computed_and_frozen(self, model):
        proc = model.model_dict["toy"]
        np.testing.assert_array_equal(
            proc["doubled"].values, 2.0 * proc["base"].values
        )
        np.testing.assert_array_equal(
            proc["counts"].values, np.arange(N_XY, dtype=np.int64)
        )
        assert proc["counts"].values.dtype == np.int64
        for name in ("doubled", "counts"):
            with pytest.raises(ValueError):
                proc[name].values[:] = 0

    def test_calculate_used_derived(self, model):
        proc = model.model_dict["toy"]
        np.testing.assert_array_equal(
            proc["total"].values, proc["doubled"].values + 1.0
        )

    def test_unknown_grid_discretization_raises(self, dis_parameters, forcing):
        process_dict = {
            "toy": {
                "class": DerivedToy,
                "discretization": "xy",
                "forcing_toy": forcing,
            },
        }
        discretizations = {
            "not_a_grid": Discretization(["not_a_grid"]),
        }
        with pytest.raises(ValueError, match="no process's home grid"):
            Model(process_dict, {}, discretizations=discretizations)


class TestTopologicalOrder:
    """tosegment (1-based, 0 = outlet): 0->2, 1->2, 2->4, 3->4;
    4 = outlet receiving flow; 5 = ISOLATED outlet (no edges at all,
    prepended by the pywatershed-replicating construction)."""

    @pytest.fixture
    def dis_seg(self):
        tosegment = np.array([3, 3, 5, 5, 0, 0], dtype=np.int64)
        return Discretization(
            ["nsegment"],
            parameters=xr.Dataset({"tosegment": ("nsegment", tosegment)}),
        )

    def test_valid_order(self, dis_seg):
        order = dis_seg.topological_order()
        to_seg = dis_seg.parameters["tosegment"].values - 1
        assert sorted(order) == list(range(6))
        position = {seg: ii for ii, seg in enumerate(order)}
        for iseg in range(6):
            if to_seg[iseg] >= 0:
                assert position[iseg] < position[to_seg[iseg]]

    def test_isolated_segment_prepended(self, dis_seg):
        assert dis_seg.topological_order()[0] == 5

    def test_cached(self, dis_seg):
        assert dis_seg.topological_order() is dis_seg.topological_order()

    def test_missing_variable_raises(self):
        dis = Discretization(["nsegment"], parameters=xr.Dataset())
        with pytest.raises(ValueError, match="not a variable"):
            dis.topological_order()
        with pytest.raises(ValueError, match="not a variable"):
            Discretization(["nsegment"]).topological_order()
