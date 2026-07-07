"""
hydrology/obsin_flow_node.py
============================
ObsInFlowNode: a node TYPE for the FlowGraph that takes inflows but
returns observed/specified flows (PRMS obsin/obsout functionality, as
an INSERTED node rather than an edit of an existing one). Ported from
pywatershed hydrology/obsin_flow_node.py, re-expressed as DATA (see
flow_graph.py). NOT mass conservative: tracks `node_sink_source`
(negative = incoming flow discarded, positive = flow created) -- a
motivating consumer for the deferred Budget design.

Port notes:
- pywatershed's per-node pandas Series looked up by date in
  prepare_timestep becomes the `node_obs_flow` INPUT (kind="input",
  model-time axis): the framework serves the current slice in
  lockstep, so `prepare` just reads it. Negative observations mean
  "pass inflows through" (resolved in `substep`, verbatim).
- The node's `_seg_outflow` scalar IS the substep outflow, so it lives
  directly in the graph's `node_outflow_substep` work buffer (set from
  the obs in `prepare`, read/updated across substeps -- only this
  node's row, per the own-rows rule). No extra field needed; the only
  state is the sink/source accumulator.
- pywatershed's per-substep running mean `_sink_source_sum/(isubstep
  + 1)` is only harvested after the LAST substep, so it is computed
  once in `finalize` (identical value, fewer writes).
- Missing dates raise in pywatershed; here the Input machinery
  guarantees lockstep coverage -- gaps are a data-prep concern.
"""

import numba
import numpy as np

from process import DataArrayMeta


@numba.njit
def _prepare(inode, state):
    # today's observation becomes the node outflow (negative -> pass
    # through, resolved in substep); pywatershed reads its series by
    # date here, the framework has already served the current slice
    state.node_outflow_substep[inode] = state.node_obs_flow[inode]
    state.obsin_sink_source_sum[inode] = 0.0


@numba.njit
def _substep(istep, inode, state, tctx, n_sub):
    # tctx (time context) unused: obsin has no seasonal physics.
    # pywatershed calculate_subtimestep, verbatim: a non-negative
    # outflow (the obs, or a prior substep's pass-through) sources/
    # sinks the difference; a negative one passes inflow through.
    inflow = (
        state.node_upstream_inflow_sub[inode]
        + state.node_lateral_inflow[inode]
    )
    if state.node_outflow_substep[inode] >= 0.0:
        state.obsin_sink_source_sum[inode] += (
            state.node_outflow_substep[inode] - inflow
        )
    else:
        state.node_outflow_substep[inode] = inflow
        # pywatershed adds zero to the sum here (no-op)


@numba.njit
def _finalize(inode, n_sub, state):
    # harvest: outflow = the LAST substep's outflow (not a mean --
    # pywatershed node semantics); sink_source = substep mean
    state.node_outflows[inode] = state.node_outflow_substep[inode]
    state.node_storage_changes[inode] = 0.0
    state.node_storages[inode] = np.nan
    state.node_sink_source[inode] = state.obsin_sink_source_sum[inode] / n_sub


class ObsInFlowNode:
    """Node type: outflow = observed/specified flow (negative obs =
    pass through); NOT mass conservative (tracks node_sink_source)."""

    type_name = "obsin"

    fields = {
        "node_obs_flow": DataArrayMeta(
            kind="input",
            dims=("space",),
            dtype=np.float64,
            description="Observed/specified flow at nodes [io flow units]; "
            "negative -> pass inflows through",
        ),
        "obsin_sink_source_sum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Obsin sink/source substep accumulator [io flow units]",
        ),
    }

    # njit node contract (Dispatchers; dispatched by the graph kernel)
    prepare = _prepare
    substep = _substep
    finalize = _finalize

    @staticmethod
    def initialize_type(dataset, n_substeps, io_in_cfs) -> None:
        # n_substeps unused (no substep-length physics); io_in_cfs
        # unused (obs + flows share the graph's units, whatever they
        # are)
        dataset["obsin_sink_source_sum"].values[:] = 0.0

    @staticmethod
    def advance_type(dataset) -> None:
        pass
