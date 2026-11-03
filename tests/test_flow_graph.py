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

from typing import Any

import numba
import numpy as np
import pytest
import xarray as xr
from conftest import PYWS_INPUT_VOL_NAMES, pyws_domain_files
from pws_phoenix.discretization import Discretization
from pws_phoenix.flow_graph import make_flow_graph
from pws_phoenix.hydrology.obsin_flow_node import ObsInFlowNode
from pws_phoenix.hydrology.pass_through_flow_node import PassThroughFlowNode
from pws_phoenix.hydrology.prms_channel_flow_node import PRMSChannelFlowNode
from pws_phoenix.hydrology.source_sink_flow_node import SourceSinkFlowNode
from pws_phoenix.model import Model

DOMAIN = "drb_2yr"
NODE_INPUT_NAMES = {nn: f"node_{nn}" for nn in PYWS_INPUT_VOL_NAMES}
DIS_FLOAT_VARS = ("seg_length", "seg_slope", "seg_depth")
NHM_SEG_INSERT_ABOVE = 1829
# pywatershed's own scalar-node-vs-array-muskingum standard
RTOL = ATOL = 1.0e-10
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_missing = [str(ff) for ff in pyws_domain_files(DOMAIN) if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        f"pywatershed {DOMAIN} test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


# the insertion scenarios: type to splice in + its NEUTRAL (=
# pass-through-equivalent) extra data, built once times are known
# values are duck-typed node-type classes (see the make_flow_graph
# contract), hence Any
INSERT_TYPES: dict[str, Any] = {
    "pass_through_insert": PassThroughFlowNode,
    "obsin_insert": ObsInFlowNode,
    "source_sink_insert": SourceSinkFlowNode,
}


def build_graph_run(drb, insert_class, out_dir, class_name):
    """Assemble one scenario (pure channel when insert_class is None,
    else the splice/headwater insertion of that type): returns
    (process_dict, control, discretizations, info) for a Model. Split
    from the graph_run fixture so a test can compose the same graph
    with a PATCHED node type (test_budget_detects_last_substep_harvest)."""
    dis_seg_ds = drb["dis_seg_ds"]
    channel_params_ds = drb["channel_params_ds"]
    insert = insert_class is not None
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
            (PRMSChannelFlowNode, insert_class), class_name=class_name
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
            (PRMSChannelFlowNode,), class_name=class_name
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

    # -- inputs: hru volumes PRE-AGGREGATED to nodes (conftest
    # factory; inserted node = zero column, no lateral inflow) --
    node_inputs = {
        NODE_INPUT_NAMES[nn]: drb["node_vol_input"](
            nn, NODE_INPUT_NAMES[nn], 1 if insert else 0
        )
        for nn in PYWS_INPUT_VOL_NAMES
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
        "budget": "error",
    }
    info = {
        "control": control,
        "class": graph_class,
        "n_seg": n_seg,
        "insert": insert,
        "insert_class": insert_class,
        "answers": drb["seg_outflow"],
    }
    return process_dict, control, discretizations, info


@pytest.fixture(
    scope="module",
    params=["pure_channel", *INSERT_TYPES.keys()],
)
def graph_run(
    request,
    pyws_domain,
    tmp_path_factory,
):
    insert_class = INSERT_TYPES.get(request.param)
    class_name = (
        "DrbChannelFlowGraph"
        if insert_class is None
        else f"Drb{insert_class.__name__}Graph"
    )
    process_dict, control, discretizations, info = build_graph_run(
        pyws_domain(DOMAIN),
        insert_class,
        tmp_path_factory.mktemp(f"flow_graph_{request.param}"),
        class_name,
    )
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    return {"model": model, **info}


class TestFlowGraph:
    def test_node_outflows_all_timesteps(self, graph_run):
        """Channel nodes match the PRMSChannel answers over the full
        run (inserted pass-through excluded -- it has no answer)."""
        output_ds = xr.open_zarr(
            graph_run["control"]["output_serial_zarr"], consolidated=False
        )
        n_seg = graph_run["n_seg"]
        np.testing.assert_allclose(
            output_ds["node_outflows"].values[:, :n_seg],
            graph_run["answers"].values,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_inserted_node_is_transparent(self, graph_run):
        """Cheap structural checks on the inserted node: intercepting
        types received flow, the obsin headwater emitted exactly zero,
        and the sink/source-tracking types applied no source or sink
        (their data are NEUTRAL by construction). The pure-channel
        scenario has no inserted node: its graph is exactly the
        segment network, no padded row."""
        proc = graph_run["model"].model_dict["flow_graph"]
        if not graph_run["insert"]:
            assert proc["node_outflows"].values.shape == (graph_run["n_seg"],)
            return
        if graph_run["insert_class"] is ObsInFlowNode:
            assert proc["node_outflows"].values[-1] == 0.0
        else:
            assert proc["node_outflows"].values[-1] > 0.0
        if graph_run["insert_class"] in (
            ObsInFlowNode,
            SourceSinkFlowNode,
        ):
            assert proc["node_sink_source"].values[-1] == 0.0

    def test_internal_parameters_frozen(self, graph_run):
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


@numba.njit
def _finalize_last_substep(inode, n_sub, state):
    """pywatershed's source_sink harvest (departure #2 REVERTED): the
    LAST substep's outflow beside substep-MEAN inputs."""
    state.node_outflows[inode] = state.node_outflow_substep[inode]
    state.node_storage_changes[inode] = 0.0
    state.node_storages[inode] = np.nan
    state.node_sink_source[inode] = state.source_sink_sum[inode] / n_sub


def test_budget_detects_last_substep_harvest(
    pyws_domain, tmp_path, monkeypatch
):
    """The budget must still catch what it caught: with pywatershed's
    harvest restored on the spliced source_sink node (below varying
    muskingum flow, so last != mean: 4.5 cfs on day 1), error mode
    raises. Patched BEFORE composition -- make_flow_graph binds each
    type's finalize into the kernel at factory time, so the patch is
    order-independent. (The snow freeh2o departure is NOT reverted
    this way: PRMSSnow._calculate binds calc_ppt_to_pack at its first
    compile, so a patch is inert after any earlier snow test.)"""
    monkeypatch.setattr(SourceSinkFlowNode, "finalize", _finalize_last_substep)
    process_dict, control, discretizations, info = build_graph_run(
        pyws_domain(DOMAIN),
        SourceSinkFlowNode,
        tmp_path,
        class_name="DrbSourceSinkRevertedGraph",
    )
    # the one imbalanced location is the inserted node (index n_seg)
    at_inserted = rf"1 location\(s\).*first indices \[{info['n_seg']}\]"
    with pytest.raises(ValueError, match="budget imbalanced"):
        with pytest.warns(UserWarning, match=at_inserted):
            with Model(
                process_dict, control, discretizations=discretizations
            ) as model:
                model.run(S_PER_TIME, np.int32(model.ntime))
