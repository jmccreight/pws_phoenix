"""
hydrology/starfit_source_sink_flow_node.py
==========================================
StarfitSourceSinkFlowNode: a STARFIT reservoir node whose sources and
sinks interact with STORAGE (diversions applied to the lake before the
release calculation), honoring a minimum-storage parameter for sinks.
Ported from pywatershed hydrology/starfit_source_sink_flow_node.py
(McCreight/Engott/Knowles), re-expressed as DATA (see flow_graph.py).

pywatershed implements this by SUBCLASSING StarfitFlowNode and
overriding four seams; here the same seams are the shared njit helpers
exported by hydrology/starfit_flow_node.py, composed with the
source/sink calculations:

  substep = pre_release_calculations
          -> _source_sink_calculations   (this module: diversion vs
             storage, min-storage rule; sets
             lake_storage_after_source_sink + lake_sink_source_sub)
          -> istarf_release(storage = lake_storage_after_source_sink)
          -> post_release_calculations(sink_source_sub =
             lake_sink_source_sub)      (the storage-change override:
             change = (inflow - release + sink_source) * m3ps_to_MCM)

Port scope follows the STARFIT family (see starfit_flow_node.py):
hourly path; internal cms/MCM with io-unit factors at the boundary
(graph io_in_cfs; identity factors in a cms graph) -- here that adds
the request conversion at read and the applied-diversion mean leaving
in io units; no Budget (pywatershed `_negative_sink_source` exists
only as a Budget term -- NOT ported), no NOR-midpoint init /
start-end window. The `node_source_sink` request INPUT is shared with
SourceSinkFlowNode (same DataArrayMeta object -- one array serves
both types in a mixed graph); `missing_data_as_zero` = data-prep, as
there. QUIRK kept verbatim: `source_sink_storage_min` is ALWAYS
internal MCM, even in a cfs graph (pywatershed never converts it).

Naming: pywatershed's `_source_sink` (applied, per substep) /
`_sink_source` (running mean) / `_sink_source_sum` carry its own
"very confusing" comment suggesting `_sink_source_sub`; adopted here
with the family convention: `lake_sink_source_sub` / `lake_sink_source`
/ `lake_sink_source_accum` [m^3/s]. Sign: sources positive, sinks
negative (node perspective).
"""

import numba
import numpy as np

from hydrology.source_sink_flow_node import SourceSinkFlowNode
from hydrology.starfit_flow_node import (
    StarfitFlowNode,
    initialize_starfit_type,
    istarf_release,
    post_release_calculations,
    pre_release_calculations,
    starfit_advance_type,
    starfit_finalize,
    starfit_prepare,
)
from process import DataArrayMeta


@numba.njit
def _prepare(inode, state):
    # base STARFIT prepare + the sink/source accumulator (pywatershed
    # _prepare_timestep_hourly + _prepare_timestep_source_sink; the
    # by-date data read is the framework's lockstep input serve)
    starfit_prepare(inode, state)
    state.lake_sink_source_accum[inode] = 0.0


@numba.njit
def _source_sink_calculations(istep, inode, state):
    # pywatershed _source_sink_calculations, verbatim, scalars at
    # [inode]; flow_to_vol_conversion = m3ps_to_MCM (the hourly-path
    # basis). The requested FLOW (io units; pywatershed converts it in
    # prepare -- the input buffer is read-only here, so convert at
    # read; io_to_cms = 1.0 in a cms graph) becomes a VOLUME against
    # storage.
    conv = state.m3ps_to_MCM[inode]
    source_sink_vol = (
        state.node_source_sink[inode] * state.io_to_cms[inode] * conv
    )
    min_storage = state.source_sink_storage_min[inode]
    storage = state.lake_storage_sub[inode]  # MCM (scalar copy)

    if source_sink_vol >= 0.0:
        # a source is always applied
        storage += source_sink_vol
    elif storage < min_storage:
        # (vol < 0) -- sink not applied when storage < min_storage;
        # storage stays the same
        source_sink_vol = 0.0
    else:
        # (vol < 0) and (storage >= min_storage)
        if (storage + source_sink_vol) < min_storage:
            # if the sink is too much, reduce it, down to min storage
            # (the difference order gives the negative sign)
            source_sink_vol = min_storage - storage
            storage = min_storage
        else:
            storage += source_sink_vol

    state.lake_storage_after_source_sink[inode] = storage  # MCM
    # back to flow units (verbatim multiply-by-reciprocal form)
    state.lake_sink_source_sub[inode] = source_sink_vol * (1.0 / conv)

    state.lake_sink_source_accum[inode] += state.lake_sink_source_sub[inode]
    state.lake_sink_source[inode] = state.lake_sink_source_accum[inode] / (
        istep + 1
    )


@numba.njit
def _substep(istep, inode, state, tctx, n_sub):
    # the pywatershed hourly path with the storage diversion between
    # pre-release and the release calculation (see module docstring)
    pre_release_calculations(inode, state)
    _source_sink_calculations(istep, inode, state)
    release, availability_status = istarf_release(
        inode,
        state,
        tctx,
        state.lake_inflow_sub[inode],
        state.lake_storage_after_source_sink[inode],
    )
    state.lake_release_sub[inode] = release  # m^3/day
    state.lake_availability_status_sub[inode] = availability_status
    post_release_calculations(
        istep, inode, state, state.lake_sink_source_sub[inode]
    )
    # pywatershed's post-release override tail: the running-mean
    # diversion leaves in io units (recomputed fresh from the cms
    # accumulator each substep, so no double conversion; identity in
    # a cms graph)
    state.lake_sink_source[inode] = (
        state.lake_sink_source[inode] * state.cms_to_io[inode]
    )


@numba.njit
def _finalize(inode, n_sub, state):
    # base harvest, then override node_sink_source with the applied
    # diversion running mean (pywatershed's sink_source property)
    starfit_finalize(inode, n_sub, state)
    state.node_sink_source[inode] = state.lake_sink_source[inode]


class StarfitSourceSinkFlowNode:
    """Node type: STARFIT reservoir with storage-interacting sources
    and sinks (hourly path, graph io units via io_in_cfs; see module
    docstring)."""

    type_name = "starfit_source_sink"

    # per-step time scalars the substep reads (see make_flow_graph)
    time_context = ("epiweek",)

    fields = {
        # the full STARFIT family declarations (SAME DataArrayMeta
        # objects -- required for composing both types in one graph)
        **StarfitFlowNode.fields,
        # the request input, shared with SourceSinkFlowNode (same
        # object; here it diverts STORAGE rather than flow)
        "node_source_sink": SourceSinkFlowNode.fields["node_source_sink"],
        # -- source/sink additions --
        "source_sink_storage_min": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Minimum storage below which sinks are not "
            "applied (and storage is not drawn below) [MCM] -- "
            "ALWAYS internal MCM, even in a cfs graph (pywatershed-"
            "verbatim: its __init__ converts initial_storage but "
            "never this)",
        ),
        "lake_storage_after_source_sink": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage after the source/sink "
            "diversion, fed to the release calculation [MCM]",
        ),
        "lake_sink_source_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Applied source/sink, current substep "
            "[m^3/s] (pywatershed _source_sink)",
        ),
        "lake_sink_source": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Applied source/sink, running daily mean "
            "[m^3/s] (pywatershed _sink_source)",
        ),
        "lake_sink_source_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Applied source/sink substep accumulator "
            "[m^3/s] (pywatershed _sink_source_sum)",
        ),
    }

    # njit node contract (Dispatchers; dispatched by the graph kernel)
    prepare = _prepare
    substep = _substep
    finalize = _finalize

    @staticmethod
    def initialize_type(dataset, n_substeps, io_in_cfs) -> None:
        initialize_starfit_type(
            dataset,
            n_substeps,
            io_in_cfs,
            StarfitSourceSinkFlowNode.type_name,
        )

    @staticmethod
    def advance_type(dataset) -> None:
        starfit_advance_type(dataset)
