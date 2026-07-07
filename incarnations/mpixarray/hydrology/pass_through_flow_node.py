"""
hydrology/pass_through_flow_node.py
===================================
PassThroughFlowNode: a node TYPE for the FlowGraph -- gives what it
takes, stores nothing. Ported from pywatershed
hydrology/pass_through_flow_node.py, re-expressed as DATA (see
flow_graph.py): no per-node objects. The type contributes field
declarations + the njit node contract and the numpy build/advance
hooks:

  - fields                     -- its DataArrayMeta declarations
  - initialize_type(dataset)   -- numpy, once at build
  - advance_type(dataset)      -- numpy, once per day
  - prepare(inode, state)      -- njit, zero this node's substep scratch
  - substep(istep, inode, state)  -- njit, the sub-timestep physics
  - finalize(inode, n_sub, state) -- njit, daily-mean harvest

The njit trio takes the graph-state NAMEDTUPLE `state` (all union
arrays); a node reads its own inflows and writes its own outputs
through it (writable because state is an ARGUMENT -- captured arrays
are readonly under njit; see the FlowGraph dispatch-spike finding in
CLAUDE.md). literal_unroll in the kernel dispatches by node-type code.
"""

import numba
import numpy as np

from process import DataArrayMeta


@numba.njit
def _prepare(inode, state):
    state.accum_inflow[inode] = 0.0


@numba.njit
def _substep(istep, inode, state, tctx, n_sub):
    # tctx (time context) unused: pass-through has no seasonal physics
    flow = (
        state.node_upstream_inflow_sub[inode]
        + state.node_lateral_inflow[inode]
    )
    state.accum_inflow[inode] += flow
    state.node_outflow_substep[inode] = flow


@numba.njit
def _finalize(inode, n_sub, state):
    state.node_outflows[inode] = state.accum_inflow[inode] / n_sub
    state.node_storage_changes[inode] = 0.0
    state.node_storages[inode] = np.nan
    state.node_sink_source[inode] = 0.0


class PassThroughFlowNode:
    """Node type: outflow_substep = upstream + lateral; daily outflow =
    accumulated inflow / n_substeps; no storage, no sink/source."""

    type_name = "pass_through"

    fields = {
        "accum_inflow": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Pass-through inflow accumulator [cfs] "
            "(kernel work buffer)",
        ),
    }

    # njit node contract (Dispatchers; dispatched by the graph kernel)
    prepare = _prepare
    substep = _substep
    finalize = _finalize

    @staticmethod
    def initialize_type(dataset, n_substeps, io_in_cfs) -> None:
        # n_substeps unused (no substep-length physics); io_in_cfs
        # unused (pass-through is unit-agnostic)
        dataset["accum_inflow"].values[:] = 0.0

    @staticmethod
    def advance_type(dataset) -> None:
        pass
