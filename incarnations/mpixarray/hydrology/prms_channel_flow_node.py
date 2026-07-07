"""
hydrology/prms_channel_flow_node.py
===================================
PRMSChannelFlowNode: the Muskingum-Mann node TYPE for the FlowGraph.
Ported from pywatershed hydrology/prms_channel_flow_graph.py
(PRMSChannelFlowNode + PRMSChannelFlowNodeMaker), re-expressed as DATA
(see flow_graph.py): the maker's coefficient derivation becomes
initialize_type (SHARING muskingum_mann_coefficients with PRMSChannel
-- one derivation, two consumers); the per-node scalar state becomes
(nnodes,) arrays; the node's `_calculate_subtimestep` numerics become
the njit `substep` (pywatershed _calculate_subtimestep_numpy verbatim,
scalars at [inode]), dispatched by the graph kernel via literal_unroll.

Numerics notes (pywatershed node semantics, kept verbatim):
- routing state starts at ZERO (the node maker passes no
  segment_flow_init -- unlike array PRMSChannel).
- storage_change = seg_inflow - seg_outflow in FLOW-RATE units [cfs]
  (the array PRMSChannel multiplies by s_per_time; the node does not).
- initialize_type computes coefficients over the FULL union arrays:
  rows belonging to other node types hold nan (or pad values) and fall
  through the Kcoef ladder to harmless defaults -- they are never read
  by other type branches. RULE for future types: a type must only
  write its OWN rows of any field it shares with another type.
"""

import numba
import numpy as np

from hydrology.prms_channel import muskingum_mann_coefficients
from process import DataArrayMeta


@numba.njit
def _prepare(inode, state):
    state.seg_inflow[inode] = 0.0
    state.seg_outflow[inode] = 0.0
    state.inflow_ts[inode] = 0.0


@numba.njit
def _substep(istep, inode, state):
    # pywatershed _calculate_subtimestep_numpy, verbatim, scalars at
    # [inode]; inflows read live from the graph state
    inflow_up = state.node_upstream_inflow_sub[inode]
    inflow_lat = state.node_lateral_inflow[inode]
    seg_current_inflow = inflow_lat + inflow_up
    state.seg_inflow[inode] += seg_current_inflow
    state.inflow_ts[inode] += seg_current_inflow

    remainder = (istep + 1) % state.tsi[inode]
    if remainder == 0:
        # node routed on the current hour
        state.inflow_ts[inode] /= state.ts[inode]
        if state.tsi[inode] > 0:
            # Muskingum routing equation
            state.outflow_ts[inode] = (
                state.inflow_ts[inode] * state.c0[inode]
                + state.inflow_ts_prev[inode] * state.c1[inode]
                + state.outflow_ts[inode] * state.c2[inode]
            )
        else:
            state.outflow_ts[inode] = state.inflow_ts[inode]
        state.inflow_ts_prev[inode] = state.inflow_ts[inode]
        state.inflow_ts[inode] = 0.0

    state.seg_outflow[inode] += state.outflow_ts[inode]
    state.node_outflow_substep[inode] = state.outflow_ts[inode]


@numba.njit
def _finalize(inode, n_sub, state):
    state.seg_outflow[inode] /= n_sub
    state.seg_inflow[inode] /= n_sub
    state.node_outflows[inode] = state.seg_outflow[inode]
    state.node_storage_changes[inode] = (
        state.seg_inflow[inode] - state.seg_outflow[inode]
    )
    state.node_storages[inode] = np.nan
    state.node_sink_source[inode] = 0.0


class PRMSChannelFlowNode:
    """Node type: Muskingum-Mann routing (PRMS >= 5.2.1)."""

    type_name = "prms_channel"

    fields = {
        # -- dis_seg variables (grid-owned) --
        "seg_length": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Segment length [m]",
        ),
        "seg_slope": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Segment slope [-]",
        ),
        "seg_depth": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Segment bank-full depth [m]",
        ),
        "segment_type": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.int64,
            description="Segment type (SegmentType enum; LAKE = 2)",
        ),
        # -- process parameters --
        "mann_n": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Manning's n roughness [-]",
        ),
        "x_coef": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Muskingum storage weighting factor [-]",
        ),
        # -- derived parameters (initialize_type, then frozen) --
        "ts": DataArrayMeta(
            kind="parameter_derived",
            dims=("space",),
            dtype=np.float64,
            description="Routing sub-timestep [h] (float)",
        ),
        "tsi": DataArrayMeta(
            kind="parameter_derived",
            dims=("space",),
            dtype=np.int64,
            description="Routing sub-timestep [h] (int; -1 = within-hour)",
        ),
        "c0": DataArrayMeta(
            kind="parameter_derived",
            dims=("space",),
            dtype=np.float64,
            description="Muskingum c0 coefficient",
        ),
        "c1": DataArrayMeta(
            kind="parameter_derived",
            dims=("space",),
            dtype=np.float64,
            description="Muskingum c1 coefficient",
        ),
        "c2": DataArrayMeta(
            kind="parameter_derived",
            dims=("space",),
            dtype=np.float64,
            description="Muskingum c2 coefficient",
        ),
        # -- per-node routing state (pywatershed node scalars ->
        # (nnodes,) arrays; zero-initialized, node semantics) --
        "seg_inflow": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Node inflow accumulator -> daily mean [cfs]",
        ),
        "seg_outflow": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Node outflow accumulator -> daily mean [cfs]",
        ),
        "inflow_ts": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Sub-timestep inflow accumulator (routing state)",
        ),
        "outflow_ts": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Sub-timestep outflow (routing state)",
        ),
        "inflow_ts_prev": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Previous sub-timestep inflow (routing state)",
        ),
    }

    # njit node contract (Dispatchers; dispatched by the graph kernel)
    prepare = _prepare
    substep = _substep
    finalize = _finalize

    @staticmethod
    def initialize_type(dataset) -> None:
        for name in (
            "seg_inflow",
            "seg_outflow",
            "inflow_ts",
            "outflow_ts",
            "inflow_ts_prev",
        ):
            dataset[name].values[:] = 0.0
        muskingum_mann_coefficients(
            dataset["ts"].values,
            dataset["tsi"].values,
            dataset["c0"].values,
            dataset["c1"].values,
            dataset["c2"].values,
            dataset["mann_n"].values,
            dataset["seg_slope"].values,
            dataset["seg_depth"].values,
            dataset["seg_length"].values,
            dataset["segment_type"].values,
            dataset["x_coef"].values,
        )

    @staticmethod
    def advance_type(dataset) -> None:
        # pywatershed node advance: inflow_ts_prev = seg_inflow (the
        # finalized daily mean). Full-array write is safe: this type
        # is the sole owner of both fields.
        dataset["inflow_ts_prev"].values[:] = dataset["seg_inflow"].values
