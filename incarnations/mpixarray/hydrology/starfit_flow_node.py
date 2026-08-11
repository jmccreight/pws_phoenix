"""
hydrology/starfit_flow_node.py
==============================
StarfitFlowNode: the STARFIT reservoir node TYPE for the FlowGraph
(Storage Targets And Release Function Inference Tool; Turner et al.
2021). Ported from pywatershed hydrology/starfit.py (StarfitFlowNode +
Starfit._calc_istarf_release), re-expressed as DATA (see
flow_graph.py): the per-node scalar state becomes (nnodes,) arrays;
the node's hourly-path numerics become the njit `substep` (verbatim,
scalars at [inode]), dispatched by the graph kernel via
literal_unroll. First real new type through the registry dispatch --
no kernel edit.

Port scope (agreed with JLM, July 2026):
- The HOURLY path (pywatershed compute_daily=False). Substep length =
  24/n_substeps hours (pywatershed's nhrs_substep); a STARFIT-only
  graph uses n_substeps=1 (nhrs_substep=24) -- the "FAKE DAILY"
  configuration, which reproduces the CONCURRENT daily reference
  formulation exactly (see the trick note in
  tests/test_starfit_flow_node.py). The distinct compute_daily mode
  (daily physics inside a sub-daily graph, one-day lag) is
  hydrology/starfit_daily_flow_node.py.
- Computes internally in cms/MCM (as pywatershed does) and converts
  at its IO boundary per the GRAPH-level `io_in_cfs` (see
  make_flow_graph): in a cfs graph, inflows/initial storage convert
  in and the harvested flows/storages convert out (storages become
  millions of cubic feet); in a cms graph the factors are 1.0 --
  multiplying by 1.0 is an IEEE identity, so that path is
  byte-identical to the pre-conversion (Round B validated) kernel.
  The factors ride as per-node parameter_internal broadcasts
  (io_to_cms / cms_to_io) so the njit code reads them from state,
  branch-free.
- `time_context = ("epiweek",)`: the seasonal release harmonics read
  the CDC epiweek from tctx (folded into 52 here, as pywatershed does
  at its call site).

Numerics notes (pywatershed semantics, kept verbatim):
- The weekly volumes are written `7.0 * flow * 24.0 * 60.0 * 60.0` --
  ORDER-SENSITIVE vs the original R code (pywatershed's own comment);
  do not refactor into a constant.
- Unit constants IN the kernel: 1.0e6 (MCM -> m^3), `/ 24 / 60 / 60`
  (m^3/day -> m^3/s), and `m3ps_to_MCM` = nhrs_substep*3600/1e6
  (= 0.0864 at nhrs_substep=24), computed in initialize_type from the
  graph n_substeps and broadcast per node so the njit substep can read
  it from state.
- Data-prep moved OUT of the node (parameters freeze at assembly,
  before the initialize hooks): nan `Obs_MEANFLOW_CUMECS` must be
  filled with `inflow_mean` upstream (pywatershed does this in node
  __init__); nan `initial_storage` (pywatershed's NOR-midpoint
  fallback + start/end active-window gating) is NOT ported.
  initialize_type RAISES on either, scoped to this type's own rows.

NOT ported (recorded): compute_daily (pywatershed itself flags it for
deletion), Budget/mass_budget (sink_source is harvested as zero --
STARFIT conserves through storage), grand_id sanity check,
start_time/end_time window gating.

The njit numerics are SHARED HELPERS (pre_release_calculations,
istarf_release, post_release_calculations, starfit_prepare,
starfit_finalize; numpy: initialize_starfit_type) mirroring
pywatershed's inheritance seams, so StarfitSourceSinkFlowNode
composes them (see that module) -- the release takes storage as an
ARG and the post-release takes the applied storage-diversion flow
(0.0 here; adding 0.0 is an IEEE identity, byte-equivalent).
"""

import numba
import numpy as np

from process import DataArrayMeta

# weekly release/NOR harmonics over the 52-epiweek year (pywatershed
# hydrology/starfit.py module constant `omega`)
_OMEGA = 1.0 / 52.0

# unit constants, verbatim from pywatershed/constants.py (note its
# cm_to_cf == cms_to_cfs and cf_to_cm == cfs_to_cms -- one pair
# serves flows AND storages)
cms_to_cfs = 35.314666721489
cfs_to_cms = 1.0 / cms_to_cfs


@numba.njit
def starfit_prepare(inode, state):
    # pywatershed _prepare_timestep_hourly: zero the running-mean
    # accumulators (cms-native: no unit conversions here)
    state.lake_inflow_accum[inode] = 0.0
    state.lake_outflow_accum[inode] = 0.0
    state.lake_storage_accum[inode] = 0.0
    state.lake_storage_change_accum[inode] = 0.0
    state.lake_release_accum[inode] = 0.0
    state.lake_spill_accum[inode] = 0.0
    state.lake_availability_status_accum[inode] = 0.0


@numba.njit
def pre_release_calculations(inode, state):
    # pywatershed _pre_release_calculations_hourly (shared with
    # StarfitSourceSinkFlowNode). Graph flows enter in io units ->
    # cms (io_to_cms = 1.0 in a cms graph -- identity).
    state.lake_inflow_sub[inode] = (
        state.node_upstream_inflow_sub[inode]
        + state.node_lateral_inflow[inode]
    ) * state.io_to_cms[inode]
    state.lake_storage_old_sub[inode] = state.lake_storage_sub[inode]


@numba.njit
def istarf_release(inode, state, tctx, lake_inflow, lake_storage):
    # pywatershed Starfit._calc_istarf_release, verbatim, scalars at
    # [inode]; np.where -> if/else. `lake_inflow` [m^3/s] and
    # `lake_storage` [MCM] are ARGS: the hourly nodes pass
    # lake_inflow_sub + lake_storage_sub (source/sink:
    # lake_storage_after_source_sink); the daily node passes the
    # day-mean lake_inflow + its marching lake_storage. Returns
    # (release [m^3/day], availability_status).
    # epiweek 53 folds into 52 (pywatershed does this at its call site)
    epiweek = min(tctx.epiweek, 52)
    # MCM to m^3
    storage = lake_storage * 1.0e6
    capacity = state.GRanD_CAP_MCM[inode] * 1.0e6

    max_normal = min(
        state.NORhi_max[inode],
        max(
            state.NORhi_min[inode],
            state.NORhi_mu[inode]
            + state.NORhi_alpha[inode] * np.sin(2.0 * np.pi * _OMEGA * epiweek)
            + state.NORhi_beta[inode] * np.cos(2.0 * np.pi * _OMEGA * epiweek),
        ),
    )
    min_normal = min(
        state.NORlo_max[inode],
        max(
            state.NORlo_min[inode],
            state.NORlo_mu[inode]
            + state.NORlo_alpha[inode] * np.sin(2.0 * np.pi * _OMEGA * epiweek)
            + state.NORlo_beta[inode] * np.cos(2.0 * np.pi * _OMEGA * epiweek),
        ),
    )

    # ORDER-SENSITIVE (see module docstring): keep the literal
    # 7.0 * flow * 24.0 * 60.0 * 60.0 form
    forecasted_weekly_volume = 7.0 * lake_inflow * 24.0 * 60.0 * 60.0
    mean_weekly_volume = (
        7.0 * state.Obs_MEANFLOW_CUMECS[inode] * 24.0 * 60.0 * 60.0
    )

    standardized_inflow = (forecasted_weekly_volume / mean_weekly_volume) - 1.0

    standardized_weekly_release = (
        state.Release_alpha1[inode] * np.sin(2.0 * np.pi * _OMEGA * epiweek)
        + state.Release_alpha2[inode] * np.sin(4.0 * np.pi * _OMEGA * epiweek)
        + state.Release_beta1[inode] * np.cos(2.0 * np.pi * _OMEGA * epiweek)
        + state.Release_beta2[inode] * np.cos(4.0 * np.pi * _OMEGA * epiweek)
    )

    # m3/week to m3/day
    release_min_vol = mean_weekly_volume * (1 + state.Release_min[inode]) / 7.0
    release_max_vol = mean_weekly_volume * (1 + state.Release_max[inode]) / 7.0

    availability_status = (100.0 * storage / capacity - min_normal) / (
        max_normal - min_normal
    )

    # m3/week to m3/day
    release = (
        mean_weekly_volume
        * (
            1
            + (
                standardized_weekly_release
                + state.Release_c[inode]
                + state.Release_p1[inode] * availability_status
                + state.Release_p2[inode] * standardized_inflow
            )
        )
    ) / 7.0

    # m3/week to m3/day
    release_above_normal = (
        storage - (capacity * max_normal / 100.0) + forecasted_weekly_volume
    ) / 7.0
    release_below_normal = (
        storage - (capacity * min_normal / 100.0) + forecasted_weekly_volume
    ) / 7.0

    if availability_status > 1.0:
        release = release_above_normal
    if availability_status < 0.0:
        release = release_below_normal
    if release < release_min_vol:
        release = release_min_vol
    if release > release_max_vol:
        release = release_max_vol

    return release, availability_status


@numba.njit
def post_release_calculations(istep, inode, state, sink_source_sub):
    # pywatershed _post_release_calculations_hourly (shared).
    # `sink_source_sub` [m^3/s] = the APPLIED storage diversion in the
    # storage change: 0.0 for the plain node (adding 0.0 is an IEEE
    # identity -- byte-equivalent to the validated Round B form); the
    # source/sink node passes lake_sink_source_sub (its
    # _calc_storage_change_sub_hourly override).
    m3ps_to_mcm = state.m3ps_to_MCM[inode]
    mcm_to_m3ps = 1.0 / m3ps_to_mcm

    state.lake_release_sub[inode] = (
        state.lake_release_sub[inode] / 24 / 60 / 60
    )  # m^3/s
    state.lake_storage_change_sub[inode] = (
        state.lake_inflow_sub[inode]
        - state.lake_release_sub[inode]
        + sink_source_sub
    ) * m3ps_to_mcm  # MCM

    # can't release more than storage + inflow (deadpool = zero
    # storage; pywatershed's max(x, x * 0.0) form kept -- it
    # propagates nan where max(x, 0.0) would not)
    if (
        state.lake_storage_sub[inode] + state.lake_storage_change_sub[inode]
    ) < 0.0:
        potential_release = (
            state.lake_release_sub[inode]
            + (
                state.lake_storage_sub[inode]
                + state.lake_storage_change_sub[inode]
            )
            * mcm_to_m3ps
        )
        state.lake_release_sub[inode] = max(
            potential_release, potential_release * 0.0
        )  # m^3/s
        state.lake_storage_change_sub[inode] = (
            state.lake_inflow_sub[inode]
            - state.lake_release_sub[inode]
            + sink_source_sub
        ) * m3ps_to_mcm

    state.lake_storage_sub[inode] = max(
        state.lake_storage_sub[inode] + state.lake_storage_change_sub[inode],
        0.0,
    )  # MCM

    state.lake_spill_sub[inode] = np.nan
    if not np.isnan(state.lake_storage_sub[inode]):
        state.lake_spill_sub[inode] = 0.0
    if state.lake_storage_sub[inode] > state.GRanD_CAP_MCM[inode]:
        state.lake_spill_sub[inode] = (
            state.lake_storage_sub[inode] - state.GRanD_CAP_MCM[inode]
        ) * mcm_to_m3ps
        state.lake_storage_sub[inode] = state.GRanD_CAP_MCM[inode]

    state.lake_storage_change_sub[inode] = (
        state.lake_storage_sub[inode] - state.lake_storage_old_sub[inode]
    )

    # subtimestep -> timestep running means (flows m^3/s, storages MCM)
    nsub = istep + 1
    state.lake_inflow_accum[inode] += state.lake_inflow_sub[inode]
    state.lake_inflow[inode] = state.lake_inflow_accum[inode] / nsub

    state.lake_outflow_sub[inode] = (
        state.lake_release_sub[inode] + state.lake_spill_sub[inode]
    )
    state.lake_outflow_accum[inode] += state.lake_outflow_sub[inode]
    state.lake_outflow[inode] = state.lake_outflow_accum[inode] / nsub

    state.lake_release_accum[inode] += state.lake_release_sub[inode]
    state.lake_release[inode] = state.lake_release_accum[inode] / nsub

    state.lake_spill_accum[inode] += state.lake_spill_sub[inode]
    state.lake_spill[inode] = state.lake_spill_accum[inode] / nsub

    state.lake_availability_status_accum[inode] += (
        state.lake_availability_status_sub[inode]
    )
    state.lake_availability_status[inode] = (
        state.lake_availability_status_accum[inode] / nsub
    )

    state.lake_storage_accum[inode] += state.lake_storage_sub[inode]
    state.lake_storage[inode] = state.lake_storage_accum[inode] / nsub

    state.lake_storage_change_accum[inode] += state.lake_storage_change_sub[
        inode
    ]
    state.lake_storage_change[inode] = (
        state.lake_storage_change_accum[inode] / nsub
    )
    state.lake_storage_change_flow_units[inode] = (
        state.lake_storage_change[inode] * mcm_to_m3ps
    )

    # the substep outflow leaves in io units (pywatershed converts
    # lake_outflow_sub itself -- safe: it is recomputed from release +
    # spill next substep, and its accumulation already happened); the
    # graph kernel routes it downstream
    state.lake_outflow_sub[inode] = (
        state.lake_outflow_sub[inode] * state.cms_to_io[inode]
    )
    state.node_outflow_substep[inode] = state.lake_outflow_sub[inode]


@numba.njit
def _substep(istep, inode, state, tctx, n_sub):
    # the pywatershed hourly path: pre-release -> release (from the
    # marching storage) -> post-release (no storage diversion)
    pre_release_calculations(inode, state)
    release, availability_status = istarf_release(
        inode,
        state,
        tctx,
        state.lake_inflow_sub[inode],
        state.lake_storage_sub[inode],
    )
    state.lake_release_sub[inode] = release  # m^3/day
    state.lake_availability_status_sub[inode] = availability_status
    post_release_calculations(istep, inode, state, 0.0)


@numba.njit
def starfit_finalize(inode, n_sub, state):
    # running means are already final (computed per substep).
    # pywatershed finalize_timestep io units conversions (identity in
    # a cms graph), THEN harvest the FlowNode properties into the
    # graph arrays. lake_storage_old *= factor is pywatershed-verbatim
    # (it double-converts a value already in io units -- harmless:
    # storage_old is rewritten from storage in the next advance, and
    # the advance-computed lake_storage_change it corrupts is
    # transient, overwritten by the substep running means).
    # (shared: StarfitSourceSinkFlowNode overwrites node_sink_source)
    cms_to_io = state.cms_to_io[inode]
    state.lake_inflow[inode] = state.lake_inflow[inode] * cms_to_io
    state.lake_release[inode] = state.lake_release[inode] * cms_to_io
    state.lake_spill[inode] = state.lake_spill[inode] * cms_to_io
    state.lake_outflow[inode] = state.lake_outflow[inode] * cms_to_io
    state.lake_storage[inode] = state.lake_storage[inode] * cms_to_io
    state.lake_storage_old[inode] = state.lake_storage_old[inode] * cms_to_io
    state.lake_storage_change_flow_units[inode] = (
        state.lake_storage_change_flow_units[inode] * cms_to_io
    )

    state.node_outflows[inode] = state.lake_outflow[inode]
    state.node_storage_changes[inode] = state.lake_storage_change_flow_units[
        inode
    ]
    state.node_storages[inode] = state.lake_storage[inode]
    state.node_sink_source[inode] = 0.0


def initialize_starfit_type(
    dataset, n_substeps, io_in_cfs, type_name, compute_daily=False
) -> None:
    """Shared numpy build hook for the STARFIT family: data checks,
    the m3ps_to_MCM and io-unit factors, and the initial marching
    storage -- all scoped to `type_name`'s OWN rows (family types can
    hold DIFFERENT m3ps_to_MCM values in one graph: the daily node's
    basis is the full day regardless of the graph n_substeps)."""
    # scope to the type's rows (other types' rows hold nan pad);
    # the code<->name map is stamped on node_type before the hooks
    # run (see flow_graph.initialize)
    names = list(dataset["node_type"].attrs["node_type_names"])
    code = names.index(type_name)
    mine = dataset["node_type"].values == code

    obs = dataset["Obs_MEANFLOW_CUMECS"].values
    if np.isnan(obs[mine]).any():
        raise ValueError(
            f"{type_name}: nan Obs_MEANFLOW_CUMECS -- parameters "
            "freeze at assembly, so pywatershed's in-node fallback is "
            "data-prep here: fill nans with inflow_mean when building "
            "the parameter dataset."
        )
    init_stor = dataset["initial_storage"].values
    if np.isnan(init_stor[mine]).any():
        raise NotImplementedError(
            f"{type_name}: nan initial_storage -- pywatershed's "
            "NOR-midpoint fallback (+ start_time window) is not "
            "ported; supply initial_storage [MCM]."
        )

    # flow -> volume basis (pywatershed nhrs_substep * 3600 / 1e6),
    # per-node so the njit code reads it from state: hourly types use
    # the substep length (24/n_substeps hours); the DAILY type uses
    # the full day (nhrs_substep = 24) regardless of graph n_substeps
    # -- own rows only, the values differ across family types
    nhrs = 24.0 if compute_daily else 24.0 / n_substeps
    dataset["m3ps_to_MCM"].values[mine] = nhrs * 60.0 * 60.0 / 1.0e6

    # io-unit factors (see module docstring): pywatershed constants
    # in a cfs graph, 1.0 (identity) in a cms graph; per-node so the
    # njit code reads them from state, branch-free
    dataset["io_to_cms"].values[mine] = cfs_to_cms if io_in_cfs else 1.0
    dataset["cms_to_io"].values[mine] = cms_to_cfs if io_in_cfs else 1.0

    # pywatershed node __init__: the marching storage state starts
    # at initial_storage (supplied in io units; cf_to_cm when cfs --
    # identity when cms). Other state stays nan (pywatershed nan1d)
    # until first computed.
    dataset["lake_storage_sub"].values[mine] = (
        init_stor[mine] * dataset["io_to_cms"].values[mine]
    )


def starfit_advance_type(dataset) -> None:
    """Shared numpy advance for the STARFIT family: storage_change =
    storage - storage_old (daily, MCM); storage_old = storage.
    In-place, full-array: the fields are owned by the family, and a
    both-types graph applies this twice -- safe, because the
    storage_old march is idempotent and lake_storage_change is
    transient (overwritten by the substep running means)."""
    change = dataset["lake_storage_change"].values
    change[:] = dataset["lake_storage"].values
    change -= dataset["lake_storage_old"].values
    dataset["lake_storage_old"].values[:] = dataset["lake_storage"].values


class StarfitFlowNode:
    """Node type: STARFIT reservoir release/spill (hourly path, graph
    io units via io_in_cfs; see module docstring for port scope)."""

    type_name = "starfit"

    # per-step time scalars the substep reads (see make_flow_graph)
    time_context = ("epiweek",)

    fields = {
        # -- parameters (starfit_original_parameters.nc names,
        # verbatim; NOR bounds in % capacity, Release_* standardized,
        # capacities/storages MCM, flows m^3/s) --
        "initial_storage": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Initial lake storage [MCM] (nan = pywatershed"
            " NOR-midpoint fallback, NOT ported -- raises)",
        ),
        "GRanD_CAP_MCM": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="GRanD reservoir capacity [MCM]",
        ),
        "Obs_MEANFLOW_CUMECS": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Observed mean inflow [m^3/s] (nan -> "
            "inflow_mean fallback is DATA-PREP here -- raises)",
        ),
        "NORhi_min": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Upper NOR bound, harmonic min [% capacity]",
        ),
        "NORhi_max": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Upper NOR bound, harmonic max [% capacity]",
        ),
        "NORhi_alpha": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Upper NOR harmonic sine coefficient [-]",
        ),
        "NORhi_beta": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Upper NOR harmonic cosine coefficient [-]",
        ),
        "NORhi_mu": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Upper NOR harmonic mean [% capacity]",
        ),
        "NORlo_min": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Lower NOR bound, harmonic min [% capacity]",
        ),
        "NORlo_max": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Lower NOR bound, harmonic max [% capacity]",
        ),
        "NORlo_alpha": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Lower NOR harmonic sine coefficient [-]",
        ),
        "NORlo_beta": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Lower NOR harmonic cosine coefficient [-]",
        ),
        "NORlo_mu": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Lower NOR harmonic mean [% capacity]",
        ),
        "Release_min": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Minimum release (standardized) [-]",
        ),
        "Release_max": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Maximum release (standardized) [-]",
        ),
        "Release_alpha1": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Release harmonic sine coefficient 1 [-]",
        ),
        "Release_alpha2": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Release harmonic sine coefficient 2 [-]",
        ),
        "Release_beta1": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Release harmonic cosine coefficient 1 [-]",
        ),
        "Release_beta2": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Release harmonic cosine coefficient 2 [-]",
        ),
        "Release_c": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Release constant [-]",
        ),
        "Release_p1": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Release availability-status coefficient [-]",
        ),
        "Release_p2": DataArrayMeta(
            kind="parameter",
            dims=("space",),
            dtype=np.float64,
            description="Release standardized-inflow coefficient [-]",
        ),
        # -- derived parameters (initialize_type, then frozen) --
        "m3ps_to_MCM": DataArrayMeta(
            kind="parameter_internal",
            dims=("space",),
            dtype=np.float64,
            description="m^3/s -> MCM per substep "
            "(nhrs_substep*3600/1e6; graph n_substeps, broadcast)",
        ),
        "io_to_cms": DataArrayMeta(
            kind="parameter_internal",
            dims=("space",),
            dtype=np.float64,
            description="Graph io units -> cms/MCM factor "
            "(cfs_to_cms in a cfs graph, 1.0 [identity] in a cms "
            "graph; graph io_in_cfs, broadcast)",
        ),
        "cms_to_io": DataArrayMeta(
            kind="parameter_internal",
            dims=("space",),
            dtype=np.float64,
            description="cms/MCM -> graph io units factor "
            "(cms_to_cfs in a cfs graph, 1.0 [identity] in a cms "
            "graph; graph io_in_cfs, broadcast)",
        ),
        # -- per-node state (pywatershed node scalars -> (nnodes,)
        # arrays; nan-initialized like pywatershed's nan1d) --
        "lake_inflow": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake inflow, running daily mean [m^3/s]",
        ),
        "lake_inflow_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake inflow, current substep [m^3/s]",
        ),
        "lake_inflow_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake inflow substep accumulator [m^3/s]",
        ),
        "lake_outflow": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake outflow (release + spill), running "
            "daily mean [m^3/s]",
        ),
        "lake_outflow_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake outflow, current substep [m^3/s]",
        ),
        "lake_outflow_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake outflow substep accumulator [m^3/s]",
        ),
        "lake_release": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake release, running daily mean [m^3/s]",
        ),
        "lake_release_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake release, current substep [m^3/s]",
        ),
        "lake_release_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake release substep accumulator [m^3/s]",
        ),
        "lake_spill": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake spill, running daily mean [m^3/s]",
        ),
        "lake_spill_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake spill, current substep [m^3/s]",
        ),
        "lake_spill_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake spill substep accumulator [m^3/s]",
        ),
        "lake_availability_status": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Position in NOR, running daily mean [-]",
        ),
        "lake_availability_status_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Position in NOR, current substep [-]",
        ),
        "lake_availability_status_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Position-in-NOR substep accumulator [-]",
        ),
        # restart=True on the two storages: lake_storage_sub is the
        # hourly marching state; lake_storage is the daily marching
        # state (daily node) AND is read by starfit_advance_type
        # (storage_old march). Everything else is zeroed in prepare,
        # regenerated by advance, or recomputed within the day.
        "lake_storage": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage, running daily mean [MCM]",
            restart=True,
        ),
        "lake_storage_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage, current substep [MCM] (the "
            "marching state; = initial_storage at init)",
            restart=True,
        ),
        "lake_storage_old": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage at previous timestep [MCM]",
        ),
        "lake_storage_old_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage at previous substep [MCM]",
        ),
        "lake_storage_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage substep accumulator [MCM]",
        ),
        "lake_storage_change": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage change, running daily mean [MCM]",
        ),
        "lake_storage_change_sub": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage change, current substep [MCM]",
        ),
        "lake_storage_change_accum": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage-change substep accumulator [MCM]",
        ),
        "lake_storage_change_flow_units": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Lake storage change in flow units [m^3/s]",
        ),
    }

    # njit node contract (Dispatchers; dispatched by the graph kernel)
    prepare = starfit_prepare
    substep = _substep
    finalize = starfit_finalize

    @staticmethod
    def initialize_type(dataset, n_substeps, io_in_cfs) -> None:
        initialize_starfit_type(
            dataset, n_substeps, io_in_cfs, StarfitFlowNode.type_name
        )

    @staticmethod
    def advance_type(dataset) -> None:
        starfit_advance_type(dataset)
