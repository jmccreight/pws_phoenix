"""FlowGraph regression vs pywatershed answers (drb_2yr, serial).

Four scenarios (module-scoped, parametrized):
- pure_channel: a channel-only graph (456 nodes = the drb segments),
  composed WITHOUT any second type.
- pass_through_insert: pywatershed's own FlowGraph doctest scenario --
  one pass-through node inserted above nhm_seg 1829 (457 nodes);
  non-inserted nodes must still match.
- source_sink_insert: the same splice with a NEUTRAL source_sink
  node (all requests zero -> outflow = inflow) -- reproduces the
  pass-through behavior exactly, so the same answers apply.
- obsin_insert: a NEUTRAL obsin as a zero-inflow HEADWATER feeding
  nhm_seg 1829's node (not intercepting): with negative obs it emits
  exactly zero, so the answers hold. An INTERCEPTING obsin cannot be
  neutral: pywatershed's node (ported verbatim) latches the FIRST
  substep's inflow as its outflow for the whole day when obs < 0,
  which differs from pass-through under sub-hourly-varying muskingum
  inflows (found when this test's first version asserted otherwise).
Branch coverage for the two new types lives in
test_obsin_source_sink_nodes.py (synthetic, hand-computed).

All validate node_outflows against the seg_outflow answers at
rtol = atol = 1e-10 -- pywatershed's OWN standard for its scalar-node
muskingum vs its array muskingum (its doctest asserts abs < 1e-10).

Single-grid on purpose: the hru->node aggregation is PRE-COMPUTED here
(volumes @ weights.T per timestep -- identical math and float order to
a per-step Map apply); Map/MapMPI wiring is already proven by the
PRMSChannel submodel tests, so this test isolates the graph machinery.

Requires GENERATED pywatershed test data; skips with a reason if
absent.
"""

import pathlib as pl
import sys
from typing import Any

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from flow_graph import make_flow_graph
from hydrology.obsin_flow_node import ObsInFlowNode
from hydrology.pass_through_flow_node import PassThroughFlowNode
from hydrology.prms_channel_flow_node import PRMSChannelFlowNode
from hydrology.source_sink_flow_node import SourceSinkFlowNode
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_VOL_NAMES = ("sroff_vol", "ssres_flow_vol", "gwres_flow_vol")
NODE_INPUT_NAMES = {nn: f"node_{nn}" for nn in INPUT_VOL_NAMES}
DIS_FLOAT_VARS = ("seg_length", "seg_slope", "seg_depth")
NHM_SEG_INSERT_ABOVE = 1829
# pywatershed's own scalar-node-vs-array-muskingum standard
RTOL = ATOL = 1.0e-10
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = [
    DOMAIN_DIR / "parameters_PRMSChannel.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    GEN_DIR / "seg_outflow.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in INPUT_VOL_NAMES]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def channel_params_ds():
    return xr.open_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")


@pytest.fixture(scope="module")
def dis_seg_ds():
    return xr.open_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")


@pytest.fixture(scope="module")
def weights(channel_params_ds):
    """0/1 hru->segment aggregation weights from hru_segment."""
    hru_segment = channel_params_ds["hru_segment"].values
    n_seg = channel_params_ds.sizes["nsegment"]
    ww = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            ww[hru_segment[ihru] - 1, ihru] = 1.0
    return ww


@pytest.fixture(scope="module")
def answers():
    return xr.open_dataarray(GEN_DIR / "seg_outflow.nc")


# the insertion scenarios: type to splice in + its NEUTRAL (=
# pass-through-equivalent) extra data, built once times are known
# values are duck-typed node-type classes (see the make_flow_graph
# contract), hence Any
INSERT_TYPES: dict[str, Any] = {
    "pass_through_insert": PassThroughFlowNode,
    "obsin_insert": ObsInFlowNode,
    "source_sink_insert": SourceSinkFlowNode,
}


@pytest.fixture(
    scope="module",
    params=["pure_channel", *INSERT_TYPES.keys()],
)
def graph_run(
    request,
    dis_seg_ds,
    channel_params_ds,
    weights,
    tmp_path_factory,
):
    insert_class = INSERT_TYPES.get(request.param)
    insert = insert_class is not None
    out_dir = tmp_path_factory.mktemp(f"flow_graph_{request.param}")
    n_seg = dis_seg_ds.sizes["nsegment"]
    n_nodes = n_seg + 1 if insert else n_seg

    def pad(vals, fill):
        if not insert:
            return vals
        tail = np.array([fill], dtype=vals.dtype)
        return np.concatenate([vals, tail])

    # -- topology + composition --
    to_graph_index = np.zeros(n_nodes, dtype=np.int64)
    to_graph_index[:n_seg] = (
        dis_seg_ds["tosegment"].values.astype(np.int64) - 1
    )
    if insert_class is not None:
        graph_class = make_flow_graph(
            (PRMSChannelFlowNode, insert_class),
            class_name=f"Drb{insert_class.__name__}Graph",
        )
        node_type = np.full(
            n_nodes,
            graph_class.node_type_code("prms_channel"),
            dtype=np.int64,
        )
        node_type[-1] = graph_class.node_type_code(insert_class.type_name)
        wh_above = int(
            np.where(dis_seg_ds["nhm_seg"].values == NHM_SEG_INSERT_ABOVE)[0][
                0
            ]
        )
        if insert_class is ObsInFlowNode:
            # headwater insertion: the new node feeds nhm_seg 1829's
            # node but intercepts nothing (zero inflow) -- with
            # negative obs it emits exactly zero (see module
            # docstring: an INTERCEPTING obsin cannot be neutral)
            to_graph_index[-1] = wh_above
        else:
            # pywatershed doctest splice: the new node goes ABOVE
            # nhm_seg 1829 -- its upstream nodes now flow into the
            # new node, the new node flows into it
            wh_below = np.where(to_graph_index[:n_seg] == wh_above)[0]
            to_graph_index[-1] = wh_above
            to_graph_index[wh_below] = n_seg
    else:
        graph_class = make_flow_graph(
            (PRMSChannelFlowNode,), class_name="DrbChannelFlowGraph"
        )
        node_type = np.full(
            n_nodes,
            graph_class.node_type_code("prms_channel"),
            dtype=np.int64,
        )

    # -- the nnodes dis: topology + composition + padded dis_seg vars --
    graph_dis_vars = {
        "to_graph_index": ("nnodes", to_graph_index),
        "node_type": ("nnodes", node_type),
        "segment_type": (
            "nnodes",
            pad(dis_seg_ds["segment_type"].values, 0),
        ),
    }
    for vv in DIS_FLOAT_VARS:
        graph_dis_vars[vv] = ("nnodes", pad(dis_seg_ds[vv].values, np.nan))
    discretizations = {
        "nnodes": Discretization(
            ["nnodes"],
            parameters=xr.Dataset(graph_dis_vars),
            topo_order={"node_order": "to_graph_index"},
            topo_one_based=False,
        ),
    }

    # -- process parameters (padded) --
    graph_params = xr.Dataset(
        {
            vv: ("nnodes", pad(channel_params_ds[vv].values, np.nan))
            for vv in ("mann_n", "x_coef")
        }
    )
    if insert_class is SourceSinkFlowNode:
        # flow_min = 0: with zero requests, the source branch always
        # applies (outflow = inflow + 0); channel rows never read it
        graph_params["flow_min"] = ("nnodes", np.zeros(n_nodes))

    # -- inputs: hru volumes PRE-AGGREGATED to nodes (see docstring);
    # inserted node = zero column (no lateral inflow) --
    def node_input(name):
        hru_da = xr.open_dataarray(GEN_DIR / f"{name}.nc")
        node_vals = hru_da.values @ weights.T  # (time, n_seg)
        if insert:
            zero_col = np.zeros((node_vals.shape[0], 1))
            node_vals = np.concatenate([node_vals, zero_col], axis=1)
        return xr.DataArray(
            node_vals,
            dims=("time", "nnodes"),
            coords={"time": hru_da["time"].values},
            name=NODE_INPUT_NAMES[name],
        )

    node_inputs = {
        NODE_INPUT_NAMES[nn]: node_input(nn) for nn in INPUT_VOL_NAMES
    }

    # neutral per-step data for the inserted type (all nodes, all
    # times; non-member rows are never read)
    def flat_input(name, fill):
        times = node_inputs["node_sroff_vol"]["time"].values
        return xr.DataArray(
            np.full((times.shape[0], n_nodes), fill),
            dims=("time", "nnodes"),
            coords={"time": times},
            name=name,
        )

    extra_inputs = {}
    if insert_class is ObsInFlowNode:
        # negative observation -> the inserted node passes through
        extra_inputs["node_obs_flow"] = flat_input("node_obs_flow", -1.0)
    if insert_class is SourceSinkFlowNode:
        extra_inputs["node_source_sink"] = flat_input("node_source_sink", 0.0)

    process_dict = {
        "flow_graph": {
            "class": graph_class,
            "discretization": "nnodes",
            "parameters": graph_params,
            **node_inputs,
            **extra_inputs,
        },
    }
    control = {
        "output_var_names": ["node_outflows"],
        "output_serial_zarr": out_dir / "flow_graph.zarr",
        "time_chunk_size": 61,
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    return {
        "model": model,
        "control": control,
        "class": graph_class,
        "n_seg": n_seg,
        "insert": insert,
        "insert_class": insert_class,
    }


class TestFlowGraph:
    def test_node_outflows_all_timesteps(self, graph_run, answers):
        """Channel nodes match the PRMSChannel answers over the full
        run (inserted pass-through excluded -- it has no answer)."""
        output_ds = xr.open_zarr(
            graph_run["control"]["output_serial_zarr"], consolidated=False
        )
        n_seg = graph_run["n_seg"]
        np.testing.assert_allclose(
            output_ds["node_outflows"].values[:, :n_seg],
            answers.values,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_inserted_node_is_transparent(self, graph_run):
        """Cheap structural checks on the inserted node: intercepting
        types received flow, the obsin headwater emitted exactly zero,
        and the sink/source-tracking types applied no source or sink
        (their data are NEUTRAL by construction)."""
        if not graph_run["insert"]:
            pytest.skip("pure-channel scenario has no inserted node")
        proc = graph_run["model"].model_dict["flow_graph"]
        if graph_run["insert_class"] is ObsInFlowNode:
            assert proc["node_outflows"].values[-1] == 0.0
        else:
            assert proc["node_outflows"].values[-1] > 0.0
        if graph_run["insert_class"] in (
            ObsInFlowNode,
            SourceSinkFlowNode,
        ):
            assert proc["node_sink_source"].values[-1] == 0.0

    def test_derived_parameters_frozen(self, graph_run):
        proc = graph_run["model"].model_dict["flow_graph"]
        for nn in ("ts", "tsi", "c0", "c1", "c2"):
            with pytest.raises(ValueError):
                proc[nn].values[:] = 0

    def test_node_type_introspection(self, graph_run):
        """Codes are internal; names are on the class AND stamped into
        the node_type variable's attrs (self-describing dataset)."""
        graph_class = graph_run["class"]
        proc = graph_run["model"].model_dict["flow_graph"]
        names = proc["node_type"].attrs["node_type_names"]
        assert names[graph_class.node_type_code("prms_channel")] == (
            "prms_channel"
        )
        assert graph_class.node_type_names[0] == "prms_channel"

    def test_incomplete_node_type_raises(self):
        """A type missing the njit contract (prepare/substep/finalize)
        is rejected with a clear error -- registry dispatch means ANY
        conforming type composes, so the gate is the contract, not a
        name allow-list."""

        class BogusFlowNode:
            type_name = "bogus"
            fields: dict = {}

        with pytest.raises(ValueError, match="contract attribute"):
            make_flow_graph((BogusFlowNode,))
