"""
flow_graph_build.py
===================
Splice new flow nodes into a PRMS channel network and hand back
everything a Model needs for the composed FlowGraph: the class, the
``nnodes`` Discretization and the union parameter dataset.

This is the pws_phoenix counterpart of pywatershed's
``_build_flow_graph_inputs`` (hydrology/prms_channel_flow_graph.py) --
the graph-building arithmetic that lived test-side in
tests/test_mixed_channel_starfit.py and tests/test_flow_graph.py until
it stabilized (CLAUDE.md "FlowGraph"). The splice SEMANTICS are
pywatershed's, verbatim:

- the ``nsegment`` channel rows come first; new node ``ii`` is graph
  row ``n_seg + ii`` (collated order);
- ``flows_to`` an ``nhm_seg`` = INTERCEPTION: the target segment's
  CHANNEL upstreams are redirected into the new node and the new node
  flows into the target. Two new nodes may not intercept the same
  segment (chain them in series instead);
- ``flows_to`` another ``NewNode`` = IN SERIES, placed upstream of it:
  everything flowing into that node is redirected into this one. In-
  series nodes resolve after all interceptions, iteratively, in any
  collated order; an unresolvable chain raises.

Padding: every ``(nsegment,)`` variable of the segment dis and of the
channel parameters is extended to ``nnodes`` with nan (floats) or 0
(ints); a new node's own parameters fill its row, every other declared
parameter of its type stays nan. A node type's ``initialize_type``
owns the "nan where it matters" check (STARFIT raises on nan
``Obs_MEANFLOW_CUMECS`` -- that data-prep stays the caller's, as
decided for the STARFIT port).

Initial state: a new node's ``initial_state`` (restart variables of
its type, by name -- e.g. STARFIT's required ``lake_storage_sub``)
becomes one ``(nnodes,)`` array per name in
``FlowGraphKit.initial_state``, nan off the supplying nodes' rows,
ready to put in the graph's process_dict entry.

The ``nnodes`` dimension carries a COORDINATE: ``nhm_seg`` for the
channel rows, the caller's ``node_id`` for new nodes (pywatershed's
``node_maker_id``), so an output row is selectable by id
(``ds.sel(nnodes=999)``). The coordinate rides on the dis parameters
only -- the process parameters and inputs stay coordinate-free so the
Model's ``grid_ds[name] = source[name]`` assignments never realign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from pws_phoenix.discretization import Discretization
from pws_phoenix.flow_graph import FlowGraphBase, make_flow_graph
from pws_phoenix.hydrology.prms_channel_flow_node import PRMSChannelFlowNode
from pws_phoenix.process import _resolved_fields

# the segment dis variables that are TOPOLOGY, not per-node properties
# (rebuilt for the graph rather than padded)
_SEG_TOPOLOGY_VARS = ("tosegment", "segment_order")


@dataclass(eq=False)
class NewNode:
    """One node to splice into the channel network.

    Args:
        node_type: the flow-node TYPE (PassThroughFlowNode,
            StarfitFlowNode, ...) -- a class with the FlowGraph node
            contract, NOT PRMSChannelFlowNode (channel rows come from
            the network).
        flows_to: an ``nhm_seg`` (int: intercept above that segment)
            or another ``NewNode`` (in series, upstream of it).
        node_id: the value of the ``nnodes`` coordinate for this row
            (pywatershed's ``node_maker_id``); must not collide with
            an ``nhm_seg`` or another new node's id.
        parameters: this node's row of its type's ``kind="parameter"``
            fields, by name. Unknown names raise.
        initial_state: this node's row of its type's ``restart=True``
            variables, by name (starting values only -- e.g.
            ``{"lake_storage_sub": ...}`` [MCM] for the STARFIT
            family, which has no default). Unknown names raise.
    """

    # duck-typed node-type classes (the make_flow_graph contract)
    node_type: Any
    flows_to: int | NewNode
    node_id: int
    parameters: dict[str, float] = field(default_factory=dict)
    initial_state: dict[str, float] = field(default_factory=dict)


@dataclass
class FlowGraphKit:
    """What ``splice_flow_graph`` returns: the composed class, the
    ``nnodes`` grid, the union parameters, the new nodes' initial
    state, and the bookkeeping to pad hru->segment Map weights and
    locate rows."""

    graph_class: type[FlowGraphBase]
    discretization: Discretization
    parameters: xr.Dataset
    n_seg: int
    n_nodes: int
    # graph row of each NewNode, in collated order
    new_node_rows: list[int]
    # the new nodes' supplied initial state: name -> (nnodes,) array,
    # nan off the supplying nodes' rows (put these in the graph's
    # process_dict entry)
    initial_state: dict[str, xr.DataArray] = field(default_factory=dict)

    def pad_weights(self, weights: np.ndarray) -> np.ndarray:
        """``(n_seg, n_hru)`` hru->segment weights -> ``(n_nodes,
        n_hru)``: zero rows for the new nodes (no lateral inflow --
        pywatershed's own limitation, stated in its helper docstring)."""
        if weights.shape[0] != self.n_seg:
            raise ValueError(
                f"weights have {weights.shape[0]} rows; the channel "
                f"network has {self.n_seg} segments"
            )
        padded = np.zeros(
            (self.n_nodes, weights.shape[1]), dtype=weights.dtype
        )
        padded[: self.n_seg, :] = weights
        return padded


def _pad(values: np.ndarray, n_new: int) -> np.ndarray:
    """Extend a (n_seg,) array by n_new rows: nan for floats, 0 else."""
    fill = np.nan if np.issubdtype(values.dtype, np.floating) else 0
    tail = np.full(n_new, fill, dtype=values.dtype)
    return np.concatenate([values, tail])


def _resolve_topology(
    tosegment0: np.ndarray,
    nhm_seg: np.ndarray,
    new_nodes: list[NewNode],
) -> np.ndarray:
    """to_graph_index (0-based, -1 = outlet) for channel rows + new
    nodes; pywatershed's _build_flow_graph_inputs, step for step."""
    n_seg = tosegment0.shape[0]
    n_nodes = n_seg + len(new_nodes)
    to_graph_index = np.full(n_nodes, -1, dtype=np.int64)
    to_graph_index[:n_seg] = tosegment0

    # -- interceptions first --
    intercepted: set[int] = set()
    placed: dict[int, int] = {}  # id(NewNode) -> graph row
    for ii, node in enumerate(new_nodes):
        if isinstance(node.flows_to, NewNode):
            continue
        if node.flows_to in intercepted:
            raise ValueError(
                f"two new nodes intercept nhm_seg {node.flows_to}; "
                "chain them in series (flows_to=<the other NewNode>)"
            )
        intercepted.add(node.flows_to)
        wh_target = np.where(nhm_seg == node.flows_to)[0]
        if wh_target.shape[0] != 1:
            raise ValueError(
                f"nhm_seg {node.flows_to} not found exactly once in the "
                "segment dis"
            )
        target = int(wh_target[0])
        # only CHANNEL upstreams are redirected (pywatershed matches
        # on the channel node maker's rows)
        wh_ups = np.where(to_graph_index[:n_seg] == target)[0]
        to_graph_index[wh_ups] = n_seg + ii
        to_graph_index[n_seg + ii] = target
        placed[id(node)] = n_seg + ii

    # -- in-series nodes: resolve iteratively (collated order free) --
    pending = [
        (ii, node)
        for ii, node in enumerate(new_nodes)
        if isinstance(node.flows_to, NewNode)
    ]
    for _ in range(len(pending)):
        still = []
        for ii, node in pending:
            assert isinstance(node.flows_to, NewNode)
            if id(node.flows_to) not in placed:
                still.append((ii, node))
                continue
            target = placed[id(node.flows_to)]
            wh_ups = np.where(to_graph_index == target)[0]
            to_graph_index[wh_ups] = n_seg + ii
            to_graph_index[n_seg + ii] = target
            placed[id(node)] = n_seg + ii
        pending = still
        if not pending:
            break
    if pending:
        raise ValueError(
            "Unable to connect some new nodes in-series: "
            f"{[node.node_id for _, node in pending]} flow to a NewNode "
            "that is not in new_nodes"
        )
    return to_graph_index


def _units_attrs(units: str | None) -> dict[str, str]:
    return {} if units is None else {"units": units}


def _carried_units(var: xr.DataArray) -> dict[str, str]:
    """A padded array keeps the units its source was SUPPLIED in, so
    control["units"] judges the caller's data, not a label written
    here."""
    return _units_attrs(var.attrs.get("units"))


def splice_flow_graph(
    dis_seg: xr.Dataset,
    channel_params: xr.Dataset,
    new_nodes: list[NewNode],
    *,
    seg_dim: str = "nsegment",
    seg_id: str = "nhm_seg",
    n_substeps: int = 24,
    io_in_cfs: bool = True,
    class_name: str = "FlowGraph",
) -> FlowGraphKit:
    """Compose a channel-based FlowGraph with ``new_nodes`` spliced in.

    Args:
        dis_seg: the segment discretization dataset -- ``tosegment``
            (1-based, 0 = outlet), ``nhm_seg``, and the per-segment dis
            variables the channel node declares (seg_length, seg_slope,
            seg_depth, segment_type). Everything on ``seg_dim`` is
            padded onto the graph dis.
        channel_params: the PRMSChannel parameters (mann_n, x_coef,
            ...); everything on ``seg_dim`` is padded onto the graph
            parameters. Variables already on ``dis_seg`` are skipped
            (dis-first sourcing).
        new_nodes: the nodes to splice, collated order = graph row
            order after the channel rows. Empty = the pure channel
            graph.
        n_substeps, io_in_cfs, class_name: passed to make_flow_graph.
    """
    n_seg = dis_seg.sizes[seg_dim]
    n_new = len(new_nodes)
    n_nodes = n_seg + n_new
    nhm_seg = dis_seg[seg_id].values
    tosegment0 = dis_seg["tosegment"].values.astype(np.int64) - 1

    # -- composition: channel + each new type once, first-appearance --
    extra_types: list[Any] = []
    for node in new_nodes:
        if node.node_type is PRMSChannelFlowNode:
            raise ValueError(
                "new nodes cannot be PRMSChannelFlowNode (channel rows "
                "come from the network)"
            )
        if node.node_type not in extra_types:
            extra_types.append(node.node_type)
    graph_class = make_flow_graph(
        (PRMSChannelFlowNode, *extra_types),
        class_name=class_name,
        n_substeps=n_substeps,
        io_in_cfs=io_in_cfs,
    )

    # -- topology + node types + ids --
    to_graph_index = _resolve_topology(tosegment0, nhm_seg, new_nodes)
    node_type = np.full(
        n_nodes, graph_class.node_type_code("prms_channel"), dtype=np.int64
    )
    node_id = np.concatenate(
        [
            nhm_seg.astype(np.int64),
            np.array([node.node_id for node in new_nodes], dtype=np.int64),
        ]
    )
    ids, counts = np.unique(node_id, return_counts=True)
    if ids.shape[0] != n_nodes:
        raise ValueError(
            "node ids must be unique across nhm_seg and new nodes; "
            f"duplicates: {ids[counts > 1].tolist()}"
        )
    for ii, node in enumerate(new_nodes):
        node_type[n_seg + ii] = graph_class.node_type_code(
            node.node_type.type_name
        )

    # -- the nnodes dis: topology + composition + padded seg dis vars --
    dis_vars: dict[str, tuple] = {
        "to_graph_index": ("nnodes", to_graph_index),
        "node_type": ("nnodes", node_type),
    }
    for key, var in dis_seg.data_vars.items():
        name = str(key)
        if name in _SEG_TOPOLOGY_VARS or name == seg_id:
            continue
        if var.dims != (seg_dim,):
            continue
        dis_vars[name] = (
            "nnodes",
            _pad(var.values, n_new),
            _carried_units(var),
        )
    discretization = Discretization(
        ["nnodes"],
        parameters=xr.Dataset(dis_vars, coords={"nnodes": node_id}),
        topo_order={"node_order": "to_graph_index"},
        topo_one_based=False,
    )

    # -- union parameters: channel values padded; new rows filled --
    declared: dict[str, Any] = {}
    for tt in (PRMSChannelFlowNode, *extra_types):
        for name, meta in tt.fields.items():
            if meta.kind == "parameter":
                declared.setdefault(name, meta.dtype)
    # Units (control["units"] reads attrs): padded channel values carry
    # the units they were supplied in; a parameter filled here is
    # stamped with the DECLARED units -- NewNode.parameters values are
    # bare floats given in declared units by contract, so for those the
    # label is a statement, not a check.
    graph_fields = _resolved_fields(graph_class)
    params: dict[str, tuple] = {}
    for name, dtype in declared.items():
        if name in dis_vars:
            continue  # dis-owned (sourced dis-first by the Model)
        if name in channel_params and channel_params[name].dims == (seg_dim,):
            params[name] = (
                "nnodes",
                _pad(channel_params[name].values, n_new),
                _carried_units(channel_params[name]),
            )
        else:
            fill = np.nan if np.issubdtype(dtype, np.floating) else 0
            params[name] = (
                "nnodes",
                np.full(n_nodes, fill, dtype=dtype),
                _units_attrs(graph_fields[name].units),
            )
    for ii, node in enumerate(new_nodes):
        own = {
            name
            for name, meta in node.node_type.fields.items()
            if meta.kind == "parameter"
        }
        unknown = set(node.parameters) - own
        if unknown:
            raise ValueError(
                f"node {node.node_id} ({node.node_type.type_name}): "
                f"unknown parameters {sorted(unknown)}; declared: "
                f"{sorted(own)}"
            )
        for name, value in node.parameters.items():
            params[name][1][n_seg + ii] = value

    # -- the new nodes' initial state: one (nnodes,) array per name,
    # nan off the supplying nodes' rows --
    state: dict[str, np.ndarray] = {}
    for ii, node in enumerate(new_nodes):
        own_state = {
            name
            for name, meta in node.node_type.fields.items()
            if meta.restart
        }
        unknown = set(node.initial_state) - own_state
        if unknown:
            raise ValueError(
                f"node {node.node_id} ({node.node_type.type_name}): "
                f"unknown initial state {sorted(unknown)}; restart "
                f"variables: {sorted(own_state)}"
            )
        for name, value in node.initial_state.items():
            if name not in state:
                state[name] = np.full(n_nodes, np.nan, dtype=np.float64)
            state[name][n_seg + ii] = value

    return FlowGraphKit(
        graph_class=graph_class,
        discretization=discretization,
        parameters=xr.Dataset(params),
        n_seg=n_seg,
        n_nodes=n_nodes,
        new_node_rows=[n_seg + ii for ii in range(n_new)],
        # NewNode.initial_state is given in the declared units (the
        # contract of that argument), so the declaration is stamped:
        # control["units"] judges supplied state by its attrs
        initial_state={
            name: xr.DataArray(
                values,
                dims="nnodes",
                attrs=_units_attrs(graph_fields[name].units),
            )
            for name, values in state.items()
        },
    )
