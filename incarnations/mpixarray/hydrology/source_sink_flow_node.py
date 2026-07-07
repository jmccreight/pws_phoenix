"""
hydrology/source_sink_flow_node.py
==================================
SourceSinkFlowNode: a node TYPE for the FlowGraph that adds or removes
a requested flow, honoring a minimum-flow parameter for sinks. Ported
from pywatershed hydrology/source_sink_flow_node.py, re-expressed as
DATA (see flow_graph.py). Sign convention (from the node's
perspective): sources positive, sinks negative [io flow units]. The APPLIED
source/sink is tracked in `node_sink_source` (a sink may be reduced or
zeroed by the min-flow rule) -- with obsin, a motivating consumer for
the deferred Budget design.

Port notes:
- pywatershed's per-node pandas Series looked up by date in
  prepare_timestep becomes the `node_source_sink` INPUT
  (kind="input", model-time axis), served in lockstep by the
  framework. The `missing_data_as_zero` option is NOT ported: inputs
  are on model time by construction, so gap-filling (with zero or
  otherwise) is a data-prep concern.
- The node's `_seg_outflow` scalar is recomputed every substep, so it
  lives directly in the graph's `node_outflow_substep` work buffer
  (only this node's row, per the own-rows rule). The only state is
  the applied-sink/source accumulator.
- pywatershed's per-substep running mean `_sink_source_sum/(isubstep
  + 1)` is only harvested after the LAST substep, so it is computed
  once in `finalize` (identical value, fewer writes).
"""

import numba
import numpy as np

from process import DataArrayMeta


@numba.njit
def _prepare(inode, state):
    # pywatershed reads its series by date here (+ missing-data
    # handling); the framework has already served the current slice
    state.source_sink_sum[inode] = 0.0


@numba.njit
def _substep(istep, inode, state, tctx, n_sub):
    # tctx (time context) unused: source/sink has no seasonal physics.
    # pywatershed calculate_subtimestep, verbatim branch structure.
    inflow = (
        state.node_upstream_inflow_sub[inode]
        + state.node_lateral_inflow[inode]
    )
    source_sink = state.node_source_sink[inode]
    min_flow = state.flow_min[inode]

    if source_sink >= 0.0:
        # a source is always applied
        outflow = inflow + source_sink
    elif inflow < min_flow:
        # (source_sink < 0) -- sink not applied when inflow < min_flow
        outflow = inflow
        source_sink = 0.0
    else:
        # (source_sink < 0) and (inflow >= min_flow)
        if (inflow + source_sink) < min_flow:
            # difference order is for negative sign convention
            source_sink = min_flow - inflow
            outflow = min_flow
        else:
            outflow = inflow + source_sink

    state.node_outflow_substep[inode] = outflow
    state.source_sink_sum[inode] += source_sink


@numba.njit
def _finalize(inode, n_sub, state):
    # harvest: outflow = the LAST substep's outflow (not a mean --
    # pywatershed node semantics); sink_source = substep mean of the
    # APPLIED source/sink
    state.node_outflows[inode] = state.node_outflow_substep[inode]
    state.node_storage_changes[inode] = 0.0
    state.node_storages[inode] = np.nan
    state.node_sink_source[inode] = state.source_sink_sum[inode] / n_sub


class SourceSinkFlowNode:
    """Node type: outflow = inflow + requested source/sink; sinks are
    limited (or skipped) so outflow does not drop below flow_min."""

    type_name = "source_sink"

    fields = {
        "flow_min": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Minimum flow below which sinks are not "
            "applied (and outflow is not drawn below) [io flow units]",
        ),
        "node_source_sink": DataArrayMeta(
            kind="input",
            dims=("space",),
            dtype=np.float64,
            description="Requested source (+) / sink (-) flow at "
            "nodes [io flow units] (node-perspective sign convention)",
        ),
        "source_sink_sum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Applied source/sink substep accumulator [io flow units]",
        ),
    }

    # njit node contract (Dispatchers; dispatched by the graph kernel)
    prepare = _prepare
    substep = _substep
    finalize = _finalize

    @staticmethod
    def initialize_type(dataset, n_substeps, io_in_cfs) -> None:
        # n_substeps unused (no substep-length physics); io_in_cfs
        # unused (requests, flow_min, and flows share the graph's
        # units, whatever they are)
        dataset["source_sink_sum"].values[:] = 0.0

    @staticmethod
    def advance_type(dataset) -> None:
        pass
