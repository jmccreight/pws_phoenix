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
  (nnodes,) arrays; a field applies to one type, holding nan/pad on the
  others' rows, never read by another type's branch).
- ``node_type`` int codes are INTERNAL (code = position in
  ``node_types``); builders speak names (``node_type_code(name)``), the
  {code: name} map is on the class and stamped into the ``node_type``
  variable's attrs (self-describing datasets).
- pywatershed's FlowNodeMakers dissolve: data-prep -> each type's
  ``initialize_type`` (run from the composed class's initialize());
  instantiation -> nothing.

**Registry dispatch (Stage 2 Round A).** Compute is ONE in-place njit
kernel walking ``node_order`` x n_substeps. Each node type contributes
three njit functions with UNIFORM signatures -- ``prepare(inode,
state)``, ``substep(istep, inode, state)``, ``finalize(inode, n_sub,
state)`` -- where ``state`` is the composition's graph-state NAMEDTUPLE
(all union arrays). The kernel dispatches each by node-type code via
``numba.literal_unroll`` over the registered function tuples: a
COMPILER-generated switch, one branch per type, so ADDING A TYPE NEEDS
NO KERNEL EDIT. Writable state is passed as an ARGUMENT (the namedtuple)
-- captured arrays are readonly under njit; the closure-binding
alternative is dead (dispatch-spike finding, CLAUDE.md). The three njit
methods mirror pywatershed's FlowNode contract (prepare_timestep /
calculate_subtimestep / finalize_timestep + property harvest folded
into finalize).

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
from collections import namedtuple
from typing import cast

import numba
import numpy as np
from numba import literal_unroll

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


def _build_graph_kernel(prepare_fns, substep_fns, finalize_fns):
    """Build the njit graph kernel for one composition.

    Closes over the three per-type njit function tuples (indexed by
    node-type code) and dispatches each via ``numba.literal_unroll`` --
    a COMPILER-generated switch, one branch per registered type, so
    adding a type needs no kernel edit. ``state`` is the composition's
    graph-state namedtuple (all union arrays); node-type functions read
    their own inflows and write their own outputs through it. The
    graph-level work (lateral map-then-sum, downstream routing, outlet
    collection) stays in the kernel; only per-type physics is dispatched.
    """

    @numba.njit
    def kernel(state, s_per_time, n_substeps):
        node_order = state.node_order
        node_type = state.node_type
        to_graph_index = state.to_graph_index
        n_nodes = node_order.shape[0]

        # lateral volumes -> cfs (map-then-sum, as in PRMSChannel)
        for jj in range(n_nodes):
            state.node_lateral_inflow[jj] = (
                state.node_sroff_vol[jj]
                + state.node_ssres_flow_vol[jj]
                + state.node_gwres_flow_vol[jj]
            ) / s_per_time

        # prepare_timestep, per type (dispatch)
        for jj in range(n_nodes):
            code = node_type[jj]
            ii = 0
            for fn in literal_unroll(prepare_fns):
                if ii == code:
                    fn(jj, state)
                ii += 1

        state.node_upstream_inflow_acc[:] = 0.0

        for istep in range(n_substeps):
            state.node_upstream_inflow_sub[:] = 0.0

            for pos in range(n_nodes):
                inode = node_order[pos]
                code = node_type[inode]
                ii = 0
                for fn in literal_unroll(substep_fns):
                    if ii == code:
                        fn(istep, inode, state)
                    ii += 1
                # route this node's substep outflow downstream
                to_node = to_graph_index[inode]
                if to_node >= 0:
                    state.node_upstream_inflow_sub[to_node] += (
                        state.node_outflow_substep[inode]
                    )

            for jj in range(n_nodes):
                state.node_upstream_inflow_acc[jj] += (
                    state.node_upstream_inflow_sub[jj]
                )

        # finalize_timestep + harvest, per type (dispatch)
        for jj in range(n_nodes):
            state.node_upstream_inflows[jj] = (
                state.node_upstream_inflow_acc[jj] / n_substeps
            )
            code = node_type[jj]
            ii = 0
            for fn in literal_unroll(finalize_fns):
                if ii == code:
                    fn(jj, n_substeps, state)
                ii += 1
            if to_graph_index[jj] < 0:
                state.outflows[jj] = state.node_outflows[jj]
            else:
                state.outflows[jj] = 0.0

    return kernel


def make_flow_graph(
    node_types: tuple, class_name: str = "FlowGraph"
) -> type[FlowGraphBase]:
    """Compose a FlowGraph Process CLASS from node types.

    Args:
        node_types: tuple of node-type classes (e.g.
            PRMSChannelFlowNode, PassThroughFlowNode). Type codes =
            position in this tuple. Each must provide: ``type_name``,
            ``fields``, ``initialize_type``/``advance_type`` (numpy),
            and the njit contract ``prepare``/``substep``/``finalize``.
        class_name: name of the composed class. Pass a DISTINCT name
            per composition if resolving classes by name
            (Process._registry) -- same-named compositions overwrite.

    The composed class declares the UNION of _GRAPH_FIELDS and each
    type's fields; a same-named field shared by two types must be THE
    SAME DataArrayMeta declaration. Any node type with the contract
    composes -- no kernel edit (registry dispatch, see module docstring).
    """
    # every node type must supply the full contract (fields + the numpy
    # build/advance hooks + the njit prepare/substep/finalize trio)
    required = (
        "type_name",
        "fields",
        "initialize_type",
        "advance_type",
        "prepare",
        "substep",
        "finalize",
    )
    for tt in node_types:
        missing = [aa for aa in required if not hasattr(tt, aa)]
        if missing:
            raise ValueError(
                f"node type {getattr(tt, 'type_name', tt)!r} is missing "
                f"required contract attribute(s) {missing} (needs: "
                "type_name, fields, initialize_type/advance_type, and the "
                "njit prepare/substep/finalize)."
            )

    type_names = [tt.type_name for tt in node_types]
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

    # every field name so far is a state array; the graph-state
    # namedtuple carries the union (built once per composition)
    field_names = list(class_attrs.keys())
    # mypy wants a literal field list; ours is composed at runtime
    graph_state_type = namedtuple(  # type: ignore[misc]
        "GraphState", field_names
    )

    code_of = {nn: ii for ii, nn in enumerate(type_names)}
    node_type_names = {ii: nn for ii, nn in enumerate(type_names)}

    # per-type njit contract, ordered by type code (= node_types order)
    prepare_fns = tuple(tt.prepare for tt in node_types)
    substep_fns = tuple(tt.substep for tt in node_types)
    finalize_fns = tuple(tt.finalize for tt in node_types)
    kernel = _build_graph_kernel(prepare_fns, substep_fns, finalize_fns)

    def initialize(self) -> None:
        for tt_class in node_types:
            tt_class.initialize_type(self._obj)
        # self-describing dataset: names by code ride on the variable
        self._obj["node_type"].attrs["node_type_names"] = list(type_names)
        # cache the graph-state namedtuple ONCE -- references, NOT
        # copies (per the memory prime directive); the serial-grid
        # dataset buffers are stable, and Input buffers update in place.
        self._graph_state = graph_state_type(
            **{nm: self._obj[nm].values for nm in field_names}
        )

    def advance(self) -> None:
        for tt_class in node_types:
            tt_class.advance_type(self._obj)

    def calculate(self, dt: np.float64, time: Time) -> None:
        # dt is SECONDS (s_per_time); 86400.0 for daily PRMS
        kernel(self._graph_state, dt, np.int64(24))

    class_attrs["_node_types"] = tuple(node_types)
    class_attrs["_node_type_codes"] = code_of
    class_attrs["node_type_names"] = node_type_names
    class_attrs["_field_names"] = field_names
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
