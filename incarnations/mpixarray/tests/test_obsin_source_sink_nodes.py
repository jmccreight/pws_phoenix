"""ObsIn + SourceSink node types vs hand-computed answers (synthetic).

Small self-contained graphs (isolated outlet nodes, constant or
per-day-varying inputs) with answers computed by hand from the
pywatershed node definitions -- every branch of each node's logic is
exercised by a dedicated node. No external data (unlike the drb
FlowGraph tests, these always run, e.g. in CI). The drb-equivalence
checks (a neutral obsin / source_sink insertion reproduces the
pass-through-insertion answers) live in test_flow_graph.py.

Branches covered:
- obsin: obs >= 0 -> outflow = obs, sink_source = obs - inflow;
  obs < 0 -> pass-through (outflow = inflow, sink_source = 0);
  per-day switching between the two (the input advances in lockstep).
- source_sink: (a) source >= 0 always applied; (b) sink skipped when
  inflow < flow_min; (c) sink LIMITED to hold outflow at flow_min;
  (d) sink fully applied when the result stays >= flow_min.
  node_sink_source reports the APPLIED source/sink.

All values are substep-invariant by construction (verified with
n_substeps=4, so the substep loop + accumulator means are exercised,
not bypassed).
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from flow_graph import make_flow_graph
from hydrology.obsin_flow_node import ObsInFlowNode
from hydrology.source_sink_flow_node import SourceSinkFlowNode
from model import Model

S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)
N_SUBSTEPS = 4
TIMES = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-05"))
N_TIME = TIMES.shape[0]
OUTPUT_VARS = ["node_outflows", "node_sink_source"]
# hand-computed answers: exact arithmetic expected (tiny tolerance
# for the accumulate-then-divide substep means)
RTOL = ATOL = 1.0e-14


def run_graph(node_type_class, n_nodes, out_dir, **process_entries):
    """One-type graph of isolated outlet nodes; returns the output ds
    (time, nnodes) plus the model for final-state checks."""
    graph_class = make_flow_graph(
        (node_type_class,),
        class_name=f"{node_type_class.__name__}TestGraph",
        n_substeps=N_SUBSTEPS,
    )
    discretizations = {
        "nnodes": Discretization(
            ["nnodes"],
            parameters=xr.Dataset(
                {
                    "to_graph_index": (
                        "nnodes",
                        np.full(n_nodes, -1, dtype=np.int64),
                    ),
                    "node_type": (
                        "nnodes",
                        np.zeros(n_nodes, dtype=np.int64),
                    ),
                }
            ),
            topo_order={"node_order": "to_graph_index"},
            topo_one_based=False,
        ),
    }
    process_dict = {
        "flow_graph": {
            "class": graph_class,
            "discretization": "nnodes",
            **process_entries,
        },
    }
    control = {
        "output_var_names": OUTPUT_VARS,
        "output_serial_zarr": out_dir / "graph.zarr",
        "time_chunk_size": 2,
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    output_ds = xr.open_zarr(control["output_serial_zarr"], consolidated=False)
    return output_ds, model


def time_nnodes_da(name, vals):
    return xr.DataArray(
        np.asarray(vals, dtype=np.float64),
        dims=("time", "nnodes"),
        coords={"time": TIMES},
        name=name,
    )


def lateral_vol_inputs(inflow_cfs):
    """The three graph volume inputs from a (time, nnodes) inflow
    [cfs]: all of it via sroff, the other two zero."""
    inflow = np.asarray(inflow_cfs, dtype=np.float64)
    return {
        "node_sroff_vol": time_nnodes_da(
            "node_sroff_vol", inflow * float(S_PER_TIME)
        ),
        "node_ssres_flow_vol": time_nnodes_da(
            "node_ssres_flow_vol", np.zeros_like(inflow)
        ),
        "node_gwres_flow_vol": time_nnodes_da(
            "node_gwres_flow_vol", np.zeros_like(inflow)
        ),
    }


class TestObsInFlowNode:
    # 2 nodes, inflow 10 cfs everywhere; node 0 switches obs sign by
    # day (obs advances in lockstep), node 1 is always pass-through
    inflow = np.full((N_TIME, 2), 10.0)
    obs = np.array(
        [
            [25.0, -1.0],
            [-1.0, -1.0],
            [30.0, -1.0],
            [-1.0, -1.0],
        ]
    )
    # obs >= 0: outflow = obs, sink_source = obs - inflow
    # obs < 0: outflow = inflow, sink_source = 0
    expect_outflows = np.array(
        [
            [25.0, 10.0],
            [10.0, 10.0],
            [30.0, 10.0],
            [10.0, 10.0],
        ]
    )
    expect_sink_source = np.array(
        [
            [15.0, 0.0],
            [0.0, 0.0],
            [20.0, 0.0],
            [0.0, 0.0],
        ]
    )

    # classmethod: instance-method class-scoped fixtures are
    # deprecated (PytestRemovedIn10Warning)
    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("obsin_nodes")
        return run_graph(
            ObsInFlowNode,
            2,
            out_dir,
            node_obs_flow=time_nnodes_da("node_obs_flow", cls.obs),
            **lateral_vol_inputs(cls.inflow),
        )

    def test_outflows(self, run):
        output_ds, _ = run
        np.testing.assert_allclose(
            output_ds["node_outflows"].values,
            self.expect_outflows,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_sink_source(self, run):
        output_ds, _ = run
        np.testing.assert_allclose(
            output_ds["node_sink_source"].values,
            self.expect_sink_source,
            rtol=RTOL,
            atol=ATOL,
        )


class TestSourceSinkFlowNode:
    # 4 nodes = the 4 branches, inflow 10 cfs, constant in time:
    #   (a) source +3, min 5        -> out 13, applied +3
    #   (b) sink -2, min 20 (in<min) -> out 10, applied 0
    #   (c) sink -8, min 5 (limited) -> out 5, applied -5
    #   (d) sink -3, min 5 (full)    -> out 7, applied -3
    inflow = np.full((N_TIME, 4), 10.0)
    flow_min = np.array([5.0, 20.0, 5.0, 5.0])
    source_sink = np.tile([3.0, -2.0, -8.0, -3.0], (N_TIME, 1))
    expect_outflows = np.tile([13.0, 10.0, 5.0, 7.0], (N_TIME, 1))
    expect_sink_source = np.tile([3.0, 0.0, -5.0, -3.0], (N_TIME, 1))

    # classmethod: instance-method class-scoped fixtures are
    # deprecated (PytestRemovedIn10Warning)
    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("source_sink_nodes")
        return run_graph(
            SourceSinkFlowNode,
            4,
            out_dir,
            parameters=xr.Dataset({"flow_min": ("nnodes", cls.flow_min)}),
            node_source_sink=time_nnodes_da(
                "node_source_sink", cls.source_sink
            ),
            **lateral_vol_inputs(cls.inflow),
        )

    def test_outflows(self, run):
        output_ds, _ = run
        np.testing.assert_allclose(
            output_ds["node_outflows"].values,
            self.expect_outflows,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_sink_source(self, run):
        output_ds, _ = run
        np.testing.assert_allclose(
            output_ds["node_sink_source"].values,
            self.expect_sink_source,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_storages_undefined(self, run):
        _, model = run
        proc = model.model_dict["flow_graph"]
        assert np.isnan(proc["node_storages"].values).all()
