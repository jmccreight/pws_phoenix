"""
flow_graph.py
=============
FlowGraph: heterogeneous flow-node TYPES composed on one DAG (e.g.
insert a reservoir or pass-through into a Muskingum network). Ported
from pywatershed base/flow_graph.py -- a REDESIGN, not a verbatim port
(design record: pws_phoenix/CLAUDE.md "FlowGraph port: agreed design").

pywatershed uses one Python object per node (scalar properties,
polymorphic dispatch) -- structurally incompatible with numba and the
memory prime directive. Here node types dissolve into DATA:

- ``make_flow_graph(node_types=...)`` composes a Process CLASS whose
  declarations are the UNION of its node types' fields (all state as
  (nnodes,) arrays; a field not applicable to a node's type holds
  nan/pad and is never read by other type branches).
- ``node_type`` int codes are INTERNAL (code = position in
  ``node_types``); builders speak names (``node_type_code(name)``), the
  {code: name} map is on the class and stamped into the ``node_type``
  variable's attrs (self-describing datasets).
- pywatershed's FlowNodeMakers dissolve: data-prep -> each type's
  ``initialize_type`` (run from the composed class's initialize());
  instantiation -> nothing.
- Compute is ONE in-place njit kernel walking ``node_order`` x
  n_substeps with a per-type switch. STAGE 1: the switch is
  hand-coded for the two known types (prms_channel, pass_through);
  registry dispatch (literal_unroll / first-class function types +
  the closure-binding signature contract) is the recorded evolution,
  forced when the first new type (STARFIT) arrives.

The graph is a Process on its own ``nnodes`` grid: serial in serial
runs, REPLICATED + Map-fed under MPI (the Step B pattern) -- no
FlowGraph-specific MPI code. Topology (``to_graph_index``: 0-based,
-1 = outlet -- the native convention) and ``node_order``
(``Discretization(topo_order=..., topo_one_based=False)``) are
dis-owned.

NOT ported (see CLAUDE.md): Budget/sink_source machinery (sink_source
is still HARVESTED per node type -- reservoirs source/sink mass -- but
no budget consumes it yet), plot()/pyvis, initialize_netcdf,
InflowExchange, type_check_nodes, allow_disconnected_nodes knob.
"""

from abc import ABCMeta
from typing import cast

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process


class FlowGraphBase(Process):
    """Abstract intermediate for composed FlowGraph classes: carries
    the name<->code introspection (codes are INTERNAL; builders speak
    names). Never instantiated directly -- use make_flow_graph()."""

    _node_type_codes: dict[str, int] = {}
    node_type_names: dict[int, str] = {}

    @classmethod
    def node_type_code(cls, name: str) -> int:
        return cls._node_type_codes[name]


# Stage-1 kernel: hand-coded switch for these type names only.
_KERNEL_TYPE_NAMES = ("prms_channel", "pass_through")

# stand-in arrays for a composition lacking a type (its code is -1, so
# its branch -- and these arrays -- are never touched)
_UNUSED = np.zeros(1, dtype=np.float64)
_UNUSED_I64 = np.zeros(1, dtype=np.int64)


_GRAPH_FIELDS = {
    # -- topology + composition (dis-owned / config) --
    "to_graph_index": DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Downstream node (0-based; -1 = graph outlet) -- "
        "native FlowGraph convention",
    ),
    "node_order": DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Upstream-to-downstream ordering -- DIS-derived: "
        "Discretization(topo_order={'node_order': 'to_graph_index'}, "
        "topo_one_based=False)",
    ),
    "node_type": DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Node type code (INTERNAL; names in this "
        "variable's 'node_type_names' attr and on the class)",
    ),
    # -- inputs (Map-fed or pre-aggregated volumes on nnodes) --
    "node_sroff_vol": DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff volume at nodes [cf]",
    ),
    "node_ssres_flow_vol": DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Interflow volume at nodes [cf]",
    ),
    "node_gwres_flow_vol": DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater discharge volume at nodes [cf]",
    ),
    # -- graph variables (pywatershed FlowGraph names) --
    "node_lateral_inflow": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Lateral inflow to each node [cfs]",
    ),
    "node_upstream_inflows": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Upstream inflow, daily mean [cfs]",
    ),
    "node_outflows": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Node outflow, daily mean [cfs]",
    ),
    "node_storages": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Node storage (nan where undefined by the type)",
    ),
    "node_storage_changes": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Node storage change [cfs] (node semantics: "
        "flow-rate units)",
    ),
    "node_sink_source": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Node sink/source (reservoirs; harvested but no "
        "Budget consumer yet)",
    ),
    "outflows": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Outflow leaving the graph (outlet nodes; else 0)",
    ),
    # -- kernel work buffers --
    "node_upstream_inflow_sub": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Sub-timestep upstream inflow (kernel work buffer)",
    ),
    "node_upstream_inflow_acc": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Upstream inflow accumulator (kernel work buffer)",
    ),
    "node_outflow_substep": DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Sub-timestep outflow per node (kernel work buffer)",
    ),
}


@numba.njit
def _calculate_flow_graph(
    node_lateral_inflow: np.ndarray,
    node_upstream_inflow_sub: np.ndarray,
    node_upstream_inflow_acc: np.ndarray,
    node_outflow_substep: np.ndarray,
    node_upstream_inflows: np.ndarray,
    node_outflows: np.ndarray,
    node_storages: np.ndarray,
    node_storage_changes: np.ndarray,
    node_sink_source: np.ndarray,
    outflows: np.ndarray,
    seg_inflow: np.ndarray,
    seg_outflow: np.ndarray,
    inflow_ts: np.ndarray,
    outflow_ts: np.ndarray,
    inflow_ts_prev: np.ndarray,
    accum_inflow: np.ndarray,
    node_sroff_vol: np.ndarray,
    node_ssres_flow_vol: np.ndarray,
    node_gwres_flow_vol: np.ndarray,
    node_order: np.ndarray,
    to_graph_index: np.ndarray,
    node_type: np.ndarray,
    ts: np.ndarray,
    tsi: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    code_channel: np.int64,
    code_pass_through: np.int64,
    s_per_time: np.float64,
    n_substeps: np.int64,
) -> None:
    n_nodes = node_outflows.shape[0]

    # lateral volumes -> cfs (map-then-sum, as in PRMSChannel)
    for jj in range(n_nodes):
        node_lateral_inflow[jj] = (
            node_sroff_vol[jj]
            + node_ssres_flow_vol[jj]
            + node_gwres_flow_vol[jj]
        ) / s_per_time

    # prepare_timestep, per type (pywatershed node semantics)
    for jj in range(n_nodes):
        if node_type[jj] == code_channel:
            seg_inflow[jj] = 0.0
            seg_outflow[jj] = 0.0
            inflow_ts[jj] = 0.0
        elif node_type[jj] == code_pass_through:
            accum_inflow[jj] = 0.0

    node_upstream_inflow_acc[:] = 0.0

    for istep in range(n_substeps):
        node_upstream_inflow_sub[:] = 0.0

        for pos in range(n_nodes):
            inode = node_order[pos]
            inflow_up = node_upstream_inflow_sub[inode]
            inflow_lat = node_lateral_inflow[inode]

            # ---- the Stage-1 hand-coded type switch ----
            if node_type[inode] == code_channel:
                # pywatershed _calculate_subtimestep_numpy, verbatim,
                # scalars at [inode]
                seg_current_inflow = inflow_lat + inflow_up
                seg_inflow[inode] += seg_current_inflow
                inflow_ts[inode] += seg_current_inflow

                remainder = (istep + 1) % tsi[inode]
                if remainder == 0:
                    # node routed on the current hour
                    inflow_ts[inode] /= ts[inode]
                    if tsi[inode] > 0:
                        # Muskingum routing equation
                        outflow_ts[inode] = (
                            inflow_ts[inode] * c0[inode]
                            + inflow_ts_prev[inode] * c1[inode]
                            + outflow_ts[inode] * c2[inode]
                        )
                    else:
                        outflow_ts[inode] = inflow_ts[inode]
                    inflow_ts_prev[inode] = inflow_ts[inode]
                    inflow_ts[inode] = 0.0

                seg_outflow[inode] += outflow_ts[inode]
                node_outflow_substep[inode] = outflow_ts[inode]

            elif node_type[inode] == code_pass_through:
                flow = inflow_up + inflow_lat
                accum_inflow[inode] += flow
                node_outflow_substep[inode] = flow

            # route this node's substep outflow downstream
            to_node = to_graph_index[inode]
            if to_node >= 0:
                node_upstream_inflow_sub[to_node] += node_outflow_substep[
                    inode
                ]

        for jj in range(n_nodes):
            node_upstream_inflow_acc[jj] += node_upstream_inflow_sub[jj]

    # finalize_timestep + harvest, per type
    for jj in range(n_nodes):
        node_upstream_inflows[jj] = node_upstream_inflow_acc[jj] / n_substeps
        if node_type[jj] == code_channel:
            seg_outflow[jj] /= n_substeps
            seg_inflow[jj] /= n_substeps
            node_outflows[jj] = seg_outflow[jj]
            node_storage_changes[jj] = seg_inflow[jj] - seg_outflow[jj]
            node_storages[jj] = np.nan
            node_sink_source[jj] = 0.0
        elif node_type[jj] == code_pass_through:
            node_outflows[jj] = accum_inflow[jj] / n_substeps
            node_storage_changes[jj] = 0.0
            node_storages[jj] = np.nan
            node_sink_source[jj] = 0.0
        if to_graph_index[jj] < 0:
            outflows[jj] = node_outflows[jj]
        else:
            outflows[jj] = 0.0


def make_flow_graph(
    node_types: tuple, class_name: str = "FlowGraph"
) -> type[FlowGraphBase]:
    """Compose a FlowGraph Process CLASS from node types.

    Args:
        node_types: tuple of node-type classes (e.g.
            PRMSChannelFlowNode, PassThroughFlowNode). Type codes =
            position in this tuple. Stage 1: only the types named in
            _KERNEL_TYPE_NAMES are supported (hand-coded kernel
            switch).
        class_name: name of the composed class. Pass a DISTINCT name
            per composition if resolving classes by name
            (Process._registry) -- same-named compositions overwrite.

    The composed class declares the UNION of _GRAPH_FIELDS and each
    type's fields; a same-named field shared by two types must be THE
    SAME DataArrayMeta declaration.
    """
    type_names = [tt.type_name for tt in node_types]
    unknown = [nn for nn in type_names if nn not in _KERNEL_TYPE_NAMES]
    if unknown:
        raise ValueError(
            f"node types {unknown} not supported by the Stage-1 "
            f"hand-coded kernel switch (supported: "
            f"{list(_KERNEL_TYPE_NAMES)}). Registry dispatch is the "
            "recorded evolution -- see CLAUDE.md."
        )
    if len(set(type_names)) != len(type_names):
        raise ValueError(f"duplicate node type names: {type_names}")

    class_attrs: dict = {}
    for name, meta in _GRAPH_FIELDS.items():
        class_attrs[name] = meta
    for tt_class in node_types:
        for name, meta in tt_class.fields.items():
            if name in class_attrs:
                if class_attrs[name] is not meta:
                    raise ValueError(
                        f"field '{name}' is declared differently by "
                        "two node types (a shared field must be the "
                        "same DataArrayMeta object)."
                    )
                continue
            class_attrs[name] = meta

    code_of = {nn: ii for ii, nn in enumerate(type_names)}
    node_type_names = {ii: nn for ii, nn in enumerate(type_names)}
    code_channel = np.int64(code_of.get("prms_channel", -1))
    code_pass_through = np.int64(code_of.get("pass_through", -1))
    has_channel = code_channel >= 0
    has_pass_through = code_pass_through >= 0

    def _values(self, name: str, present: bool) -> np.ndarray:
        if not present:
            return _UNUSED_I64 if name == "tsi" else _UNUSED
        return self._obj[name].values

    def initialize(self) -> None:
        for tt_class in node_types:
            tt_class.initialize_type(self._obj)
        # self-describing dataset: names by code ride on the variable
        self._obj["node_type"].attrs["node_type_names"] = list(type_names)

    def advance(self) -> None:
        for tt_class in node_types:
            tt_class.advance_type(self._obj)

    def calculate(self, dt: np.float64, time: Time) -> None:
        # dt is SECONDS (s_per_time); 86400.0 for daily PRMS
        obj = self._obj
        _calculate_flow_graph(
            obj["node_lateral_inflow"].values,
            obj["node_upstream_inflow_sub"].values,
            obj["node_upstream_inflow_acc"].values,
            obj["node_outflow_substep"].values,
            obj["node_upstream_inflows"].values,
            obj["node_outflows"].values,
            obj["node_storages"].values,
            obj["node_storage_changes"].values,
            obj["node_sink_source"].values,
            obj["outflows"].values,
            self._values("seg_inflow", has_channel),
            self._values("seg_outflow", has_channel),
            self._values("inflow_ts", has_channel),
            self._values("outflow_ts", has_channel),
            self._values("inflow_ts_prev", has_channel),
            self._values("accum_inflow", has_pass_through),
            obj["node_sroff_vol"].values,
            obj["node_ssres_flow_vol"].values,
            obj["node_gwres_flow_vol"].values,
            obj["node_order"].values,
            obj["to_graph_index"].values,
            obj["node_type"].values,
            self._values("ts", has_channel),
            self._values("tsi", has_channel),
            self._values("c0", has_channel),
            self._values("c1", has_channel),
            self._values("c2", has_channel),
            code_channel,
            code_pass_through,
            dt,
            np.int64(24),
        )

    class_attrs["_node_types"] = tuple(node_types)
    class_attrs["_node_type_codes"] = code_of
    class_attrs["node_type_names"] = node_type_names
    class_attrs["_values"] = _values
    class_attrs["initialize"] = initialize
    class_attrs["advance"] = advance
    class_attrs["calculate"] = calculate
    class_attrs["__doc__"] = (
        f"FlowGraph composed of node types {type_names} "
        "(see flow_graph.make_flow_graph)."
    )

    # Process is an ABC (ABCMeta metaclass); build the concrete
    # composed class via ABCMeta directly (a bare type() call raises a
    # metaclass conflict). node_type_code() is inherited from
    # FlowGraphBase (reads _node_type_codes). cast: ABCMeta(...) types
    # as ABCMeta, but constructs a FlowGraphBase subclass.
    return cast(
        "type[FlowGraphBase]",
        ABCMeta(class_name, (FlowGraphBase,), class_attrs),
    )
