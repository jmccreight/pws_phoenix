"""
hydrology/starfit_daily_flow_node.py
====================================
StarfitDailyFlowNode: STARFIT computing DAILY physics inside a
sub-daily (n_substeps >= 2, typically 24) graph. Ported from
pywatershed hydrology/starfit.py compute_daily=True
(_prepare_timestep_daily + _calc_subtimestep_daily), re-expressed as
DATA (see flow_graph.py), composing the family's shared helpers
(istarf_release etc. from starfit_flow_node.py).

"Fake daily" vs THIS daily mode -- read this before comparing numbers:

- The offline STARFIT reference (test_data/starfit/
  starfit_mean_output_1995-2001.nc) is a DAILY formulation: each day's
  release is computed from that SAME day's inflow and the current
  storage (concurrent).
- The HOURLY node is validated against that reference via the "fake
  daily" trick (pywatershed's own autotests, and ours): run the hourly
  physics with ONE substep of 24 hours (our n_substeps=1 graph =
  pywatershed nhrs_substep=24, range(1)). With one substep per day,
  "this substep's inflow" IS the day's inflow and the release is
  computed from current storage -- exactly the concurrent daily
  formulation, so it matches the reference (at 1e-7).
- THIS type is different BY CONSTRUCTION: it lives in a graph that
  substeps through the day (upstream muskingum needs 24), where
  today's MEAN inflow is unknowable until the day ends. So at the END
  of each day it computes the NEXT day's constant release from
  TODAY's mean inflow and updated storage, and applies that constant
  through tomorrow's substeps -- the applied flows LAG the inflow
  signal by one day (a forecast structure; the run's first day is
  seeded from the first substep's inflow). Daily-mode output can
  therefore NEVER match the concurrent daily reference at a tight
  tolerance -- do not "fix" it to do so.

Validation status: pywatershed has NO value-level validation of daily
mode -- its node autotest only runs hourly, and its mixed-graph
autotest (test_starfit_flow_graph.py) pastes the graph's own outputs
in as the expected values for the new nodes. Our validation is
therefore an A/B PARITY test against pywatershed's own
compute_daily=True node driven identically
(tests/test_starfit_daily_parity.py) -- it validates the PORT, not
the physics.

Port notes:
- n_substeps >= 2 is REQUIRED (initialize_type raises): in a
  1-substep graph the first day's if/elif/else never reaches the
  last-substep output bookkeeping (a pywatershed latent edge -- its
  daily mode hardcodes nsubsteps=24); use the hourly StarfitFlowNode
  for daily-reference work.
- m3ps_to_MCM here is the FULL-DAY basis (nhrs_substep = 24)
  regardless of the graph n_substeps -- per-type rows of the shared
  field (initialize_starfit_type(compute_daily=True)).
- `time_context = ("epiweek", "itime_step")`: the first-day special
  case needs the model timestep index.
- Spill semantics differ from hourly: storage is NOT capped when
  spill is computed ("spill doesn't affect the storage until the next
  timestep" -- pywatershed comment, kept verbatim).
- pywatershed allocates lake_{release,spill}_sub_next and
  lake_availability_status_next for compute_daily but never uses
  them -- NOT ported; only lake_outflow_sub_next is real state.
- NOT ported (as for the family): Budget, NOR-midpoint init,
  start/end window; no source/sink variant (pywatershed's combined
  node hardcodes compute_daily=False).
"""

import numba
import numpy as np

from hydrology.starfit_flow_node import (
    StarfitFlowNode,
    initialize_starfit_type,
    istarf_release,
    starfit_advance_type,
    starfit_finalize,
)
from process import DataArrayMeta


@numba.njit
def _prepare(inode, state):
    # pywatershed _prepare_timestep_daily: zero the inflow
    # accumulator; the marching storages return to internal units
    # (they LEAVE finalize in io units each day -- identity in a cms
    # graph; nan through the first day's prepare, harmlessly)
    io_to_cms = state.io_to_cms[inode]
    state.lake_inflow_accum[inode] = 0.0
    state.lake_storage[inode] = state.lake_storage[inode] * io_to_cms
    state.lake_storage_old[inode] = state.lake_storage_old[inode] * io_to_cms


@numba.njit
def _substep(istep, inode, state, tctx, n_sub):
    # pywatershed _calc_subtimestep_daily, verbatim, scalars at
    # [inode] (its hardcoded nsubsteps=24 -> n_sub, the graph value).
    # The routed outflow is a CONSTANT rate through the day, computed
    # at the END of the previous day; storage updates once, on the
    # last substep. lake_storage (not lake_storage_sub) is the
    # marching storage in this mode.
    io_to_cms = state.io_to_cms[inode]
    cms_to_io = state.cms_to_io[inode]
    m3ps_to_mcm = state.m3ps_to_MCM[inode]

    # accumulate inflows
    state.lake_inflow_sub[inode] = (
        state.node_upstream_inflow_sub[inode]
        + state.node_lateral_inflow[inode]
    ) * io_to_cms
    state.lake_inflow_accum[inode] += state.lake_inflow_sub[inode]

    if tctx.itime_step == 0 and istep == 0:
        # first substep of the RUN: the very first inflow stands in
        # for the nonexistent previous day's mean, so an average
        # outflow exists for the first timestep
        state.lake_inflow[inode] = state.lake_inflow_accum[inode]
        state.lake_storage[inode] = state.lake_storage_sub[inode]
        state.lake_storage_old[inode] = state.lake_storage_sub[inode]
        state.lake_storage_change[inode] = 0.0
    elif istep < (n_sub - 1):
        if istep == 0:
            # already in io units
            state.lake_outflow_sub[inode] = state.lake_outflow_sub_next[inode]
        state.node_outflow_substep[inode] = state.lake_outflow_sub[inode]
        return
    else:
        if tctx.itime_step == 0:
            # the end of the first timestep doesn't pass through
            # advance()
            state.lake_storage_old[inode] = state.lake_storage[inode]
        state.lake_inflow[inode] = state.lake_inflow_accum[inode] / (istep + 1)
        # the day's constant rates return to cms for output/storage
        # bookkeeping (identity in a cms graph)
        state.lake_outflow_sub[inode] = (
            state.lake_outflow_sub[inode] * io_to_cms
        )
        state.lake_release_sub[inode] = (
            state.lake_release_sub[inode] * io_to_cms
        )
        state.lake_spill_sub[inode] = state.lake_spill_sub[inode] * io_to_cms
        state.lake_outflow[inode] = state.lake_outflow_sub[inode]
        state.lake_release[inode] = state.lake_release_sub[inode]
        state.lake_spill[inode] = state.lake_spill_sub[inode]

        # calculate storage
        state.lake_storage_change_flow_units[inode] = (
            state.lake_inflow[inode] - state.lake_outflow[inode]
        )
        state.lake_storage_change[inode] = (
            state.lake_storage_change_flow_units[inode] * m3ps_to_mcm
        )
        state.lake_storage[inode] += state.lake_storage_change[inode]

    # -- the tail runs on the run's first substep and on each day's
    # last substep: compute the NEXT day's constant rates --

    # spill: storage is NOT capped ("spill doesn't affect the storage
    # until the next timestep")
    state.lake_spill_sub[inode] = 0.0
    if state.lake_storage[inode] > state.GRanD_CAP_MCM[inode]:
        state.lake_spill_sub[inode] = (
            state.lake_storage[inode] - state.GRanD_CAP_MCM[inode]
        ) * (1.0 / m3ps_to_mcm)

    # the (avg) release for the next timestep, from the day-mean
    # inflow and the marching storage
    release, availability_status = istarf_release(
        inode,
        state,
        tctx,
        state.lake_inflow[inode],
        state.lake_storage[inode],
    )
    state.lake_availability_status[inode] = availability_status
    # m^3/day -> MCM (m3ps_to_MCM / 24 / 60 / 60 == 1e-6 at the
    # daily basis)
    release = release * (m3ps_to_mcm / 24 / 60 / 60)
    # release never exceeds the day's storage
    if (state.lake_storage[inode] - release) < 0.0:
        release = state.lake_storage[inode]
    release = release * (1.0 / m3ps_to_mcm)  # MCM -> m^3/s
    state.lake_release_sub[inode] = release

    state.lake_outflow_sub_next[inode] = (
        state.lake_release_sub[inode] + state.lake_spill_sub[inode]
    )

    # next-day rates leave in io units (identity in a cms graph)
    state.lake_outflow_sub[inode] = state.lake_outflow_sub[inode] * cms_to_io
    state.lake_outflow_sub_next[inode] = (
        state.lake_outflow_sub_next[inode] * cms_to_io
    )
    state.lake_release_sub[inode] = state.lake_release_sub[inode] * cms_to_io
    state.lake_spill_sub[inode] = state.lake_spill_sub[inode] * cms_to_io

    if tctx.itime_step == 0 and istep == 0:
        # the first day has no previous-day rate: apply immediately
        state.lake_outflow_sub[inode] = state.lake_outflow_sub_next[inode]

    state.node_outflow_substep[inode] = state.lake_outflow_sub[inode]


class StarfitDailyFlowNode:
    """Node type: STARFIT with DAILY physics in a sub-daily graph --
    constant outflow through each day, computed at the previous day's
    end (one-day lag; see module docstring)."""

    type_name = "starfit_daily"

    # per-step time scalars the substep reads (see make_flow_graph)
    time_context = ("epiweek", "itime_step")

    fields = {
        # the full STARFIT family declarations (SAME DataArrayMeta
        # objects -- required for composing family types in one
        # graph). The hourly-only accumulators ride along unused
        # (union-of-fields design).
        **StarfitFlowNode.fields,
        # restart=True: computed at each day's end, read at the NEXT
        # day's first substep (the one-day lag) -- prognostic
        "lake_outflow_sub_next": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Next day's constant outflow rate, computed "
            "at the current day's end [io flow units]",
            restart=True,
        ),
    }

    # njit node contract (Dispatchers; dispatched by the graph kernel)
    prepare = _prepare
    substep = _substep
    finalize = starfit_finalize

    @staticmethod
    def initialize_type(dataset, n_substeps, io_in_cfs) -> None:
        if n_substeps < 2:
            raise ValueError(
                "starfit_daily requires a sub-daily graph "
                f"(n_substeps >= 2, got {n_substeps}): in a 1-substep "
                "graph the first day never reaches the last-substep "
                "output bookkeeping. For daily-reference work use the "
                "hourly StarfitFlowNode with n_substeps=1 (the 'fake "
                "daily' configuration; see module docstring)."
            )
        initialize_starfit_type(
            dataset,
            n_substeps,
            io_in_cfs,
            StarfitDailyFlowNode.type_name,
            compute_daily=True,
        )

    @staticmethod
    def advance_type(dataset) -> None:
        starfit_advance_type(dataset)
