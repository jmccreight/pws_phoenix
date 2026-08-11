"""
hydrology/prms_channel.py
=========================
PRMSChannel: muskingum_mann channel routing, ported from pywatershed
(pywatershed/hydrology/prms_channel.py; PRMS 5.2.1 physics, PRMS-IV
documentation: Markstrom et al. 2015, USGS TM 6-B7; muskingum_mann per
PRMS >= 5.2.1).

Second REAL process port (July 2026) -- the serial-segment-grid half of
the Step B shape. What pywatershed's channel does INTERNALLY is here
SEPARATED along framework seams (design decisions w/ JLM):

- **Topology -> the Discretization.** ``segment_order`` is dis-owned:
  computed at dis construction (``topo_order={"segment_order":
  "tosegment"}``) and received here BY DECLARATION like any other dis
  variable. ``tosegment``/``seg_length``/``seg_slope``/``seg_depth``/
  ``segment_type`` are dis_seg variables (separate_nhm_params.py).
- **The hru->segment aggregation -> explicit Maps.** pywatershed's
  per-HRU ``hru_segment`` loop becomes three Maps (one per flux, same
  0/1 weights by reference); ``hru_segment`` itself IS the weights and
  leaves the process entirely. The mapped inputs get NEW names
  (``seg_sroff_vol``, ``seg_ssres_flow_vol``, ``seg_gwres_flow_vol`` --
  the first deliberate departure from names-verbatim: these
  aggregated-to-segment quantities do not exist in pywatershed).
  Summing them into ``seg_lateral_inflow`` is channel physics and
  happens first thing in the kernel. Cross-dis fluxes are VOLUMES
  (cubic feet) -- dis-relative units (inches) never cross a Map.
- **Muskingum coefficients -> parameter_internal.** ``c0/c1/c2/ts/tsi``
  (+ 0-based ``tosegment0``) are computed in ``initialize()`` from
  process params (``mann_n``, ``x_coef``) + dis variables, then frozen.

dt is SECONDS (s_per_time = dt = 86400.0 for daily PRMS).

Deliberately NOT ported (each documented in place or in CLAUDE.md):
- Budget / ConservativeProcess (backlogged); adapters; restart;
  calc_method switch; verbose.
- ``hru_area`` (declared upstream, never used in its kernel),
  ``tosegment_nhm``, ``obsin_segment``/``obsout_segment`` (unimplemented
  todos upstream).
- The ``channel_*_vol`` per-HRU diagnostics + the ``hru_segment < 1``
  mass-discard bookkeeping (deferred with the Budget discussion; the
  discard itself IS the zero weight columns).
- The in-place ``seg_slope`` clamp/edit: pywatershed computes velocity
  BEFORE clamping, so the edit affects nothing downstream in channel
  init -- we do NOT edit the read-only dis variable. A deliberate
  design pass on init-time parameter editing is pending (CLAUDE.md).
"""

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process

# pywatershed constants.SegmentType.LAKE
_SEGMENT_TYPE_LAKE = 2


def muskingum_mann_coefficients(
    ts: np.ndarray,
    tsi: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    mann_n: np.ndarray,
    seg_slope: np.ndarray,
    seg_depth: np.ndarray,
    seg_length: np.ndarray,
    segment_type: np.ndarray,
    x_coef: np.ndarray,
) -> None:
    """Muskingum-Mann routing coefficients, written IN PLACE (out
    buffers first), replicating pywatershed
    PRMSChannel._initialize_channel_data numerics exactly.

    Shared by PRMSChannel.initialize() and the FlowGraph channel node
    type -- ONE derivation, two consumers. Init-time temporaries only
    (velocity, Kcoef, masks), never per-step.
    """
    n_seg = ts.shape[0]

    velocity = (
        ((1.0 / mann_n) * np.sqrt(seg_slope) * seg_depth ** (2.0 / 3.0))
        * 60.0
        * 60.0
    )
    # pywatershed clamps too-flat seg_slope IN PLACE here -- AFTER
    # velocity, so the edit affects nothing downstream in channel
    # init. We deliberately do NOT edit the read-only dis variable
    # (init-time parameter editing gets its own design pass).

    # Kcoef = 24 default; Manning travel time where velocity > 0;
    # lakes forced to 24; clamped to [0.01, 24]
    kcoef = np.full(n_seg, 24.0, dtype=np.float64)
    wh_moving = velocity > 0.0
    kcoef[wh_moving] = seg_length[wh_moving] / velocity[wh_moving]
    kcoef = np.where(segment_type == _SEGMENT_TYPE_LAKE, 24.0, kcoef)
    kcoef = np.where(kcoef < 0.01, 0.01, kcoef)
    kcoef = np.where(kcoef > 24.0, 24.0, kcoef)

    ts[:] = 1.0
    tsi[:] = 1
    # sub-timestep ladder (even divisors of 24 h) -- verbatim
    for iseg in range(n_seg):
        kk = kcoef[iseg]
        if kk < 1.0:
            tsi[iseg] = -1
        elif kk < 2.0:
            ts[iseg] = 1.0
            tsi[iseg] = 1
        elif kk < 3.0:
            ts[iseg] = 2.0
            tsi[iseg] = 2
        elif kk < 4.0:
            ts[iseg] = 3.0
            tsi[iseg] = 3
        elif kk < 6.0:
            ts[iseg] = 4.0
            tsi[iseg] = 4
        elif kk < 8.0:
            ts[iseg] = 6.0
            tsi[iseg] = 6
        elif kk < 12.0:
            ts[iseg] = 8.0
            tsi[iseg] = 8
        elif kk < 24.0:
            ts[iseg] = 12.0
            tsi[iseg] = 12
        else:
            ts[iseg] = 24.0
            tsi[iseg] = 24

    dd = kcoef - (kcoef * x_coef) + (0.5 * ts)
    dd = np.where(np.abs(dd) < 1e-6, 0.0001, dd)
    c0[:] = (-(kcoef * x_coef) + (0.5 * ts)) / dd
    c1[:] = ((kcoef * x_coef) + (0.5 * ts)) / dd
    c2[:] = (kcoef - (kcoef * x_coef) - (0.5 * ts)) / dd

    # short travel time
    wh_short = c2 < 0.0
    c1[wh_short] += c2[wh_short]
    c2[wh_short] = 0.0
    # long travel time
    wh_long = c0 < 0.0
    c1[wh_long] += c0[wh_long]
    c0[wh_long] = 0.0


class PRMSChannel(Process):
    """Muskingum-Mann channel routing on the segment grid.

    Sequential by nature (downstream accumulation in segment order over
    24 subhourly steps) -- the reason the segment grid stays
    serial/replicated under MPI.
    """

    # ------------------------------------------------------------------
    # Field declarations
    # ------------------------------------------------------------------

    # -- dis_seg variables (grid-owned; via Discretization(parameters=)) --
    seg_length = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Segment length [m]",
    )
    seg_slope = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Segment slope [-]",
    )
    seg_depth = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Segment bank-full depth [m]",
    )
    segment_type = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Segment type (SegmentType enum; LAKE = 2)",
    )
    tosegment = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Downstream segment (1-based; 0 = domain outlet)",
    )
    segment_order = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        derivation="Discretization(topo_order={'segment_order': 'tosegment'})",
        description=(
            "Upstream-to-downstream ordering (0-based) -- DIS-derived: "
            "Discretization(topo_order={'segment_order': 'tosegment'})"
        ),
    )

    # -- process parameters --
    mann_n = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Manning's n roughness [-]",
    )
    x_coef = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Muskingum storage weighting factor [-]",
    )

    # -- derived parameters (computed by initialize(), then frozen) --
    tosegment0 = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.int64,
        description="Downstream segment (0-based; negative = outlet)",
    )
    ts = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Routing sub-timestep [h] (float)",
    )
    tsi = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.int64,
        description="Routing sub-timestep [h] (int; -1 = within-hour)",
    )
    c0 = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Muskingum c0 coefficient",
    )
    c1 = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Muskingum c1 coefficient",
    )
    c2 = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Muskingum c2 coefficient",
    )

    # -- inputs (Map-fed: aggregated hru VOLUMES on the segment grid) --
    seg_sroff_vol = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff volume mapped to segments [cf]",
    )
    seg_ssres_flow_vol = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Interflow volume mapped to segments [cf]",
    )
    seg_gwres_flow_vol = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater discharge volume mapped to segments [cf]",
    )

    # -- variables --
    seg_lateral_inflow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Lateral inflow to each segment [cfs]",
    )
    seg_upstream_inflow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Upstream inflow, daily mean [cfs]",
    )
    seg_inflow = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Segment inflow, daily mean [cfs]",
    )
    seg_outflow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Segment outflow, daily mean [cfs]",
        initial="segment_flow_init",
    )
    inflow_ts_prev = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Previous sub-timestep inflow (routing state)",
    )
    outflow_ts = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Sub-timestep outflow (routing state; instantaneous)",
    )
    seg_stor_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Channel storage change over the timestep [cf]",
    )
    channel_outflow_vol = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Outflow volume leaving the domain (outlets) [cf]",
    )
    # kernel work buffers, declared so the kernel allocates NOTHING
    inflow_ts = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Sub-timestep inflow accumulator (kernel work buffer)",
    )
    seg_current_sum = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Upstream-inflow accumulator (kernel work buffer)",
    )

    # ------------------------------------------------------------------
    # Initialization (derived parameters + initial values)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Compute Muskingum coefficients and initial routing state,
        replicating pywatershed _initialize_channel_data numerics (and
        order) exactly. Temporaries here are INIT-time, not per-step."""
        obj = self._obj

        # -- init values: allocation fills nan; pywatershed zero-inits
        # these (channel_outflow_vol is nan upstream but overwritten
        # every step; zeroed here for tidiness) --
        for name in (
            "seg_lateral_inflow",
            "seg_upstream_inflow",
            "seg_inflow",
            "outflow_ts",
            "seg_stor_change",
            "channel_outflow_vol",
            "inflow_ts",
            "seg_current_sum",
        ):
            obj[name].values[:] = 0.0
        # pywatershed order: inflow_ts_prev from seg_inflow (still zero)
        # BEFORE the seg_inflow propagation below
        obj["inflow_ts_prev"].values[:] = obj["seg_inflow"].values

        # -- 0-based connectivity for the kernel --
        obj["tosegment0"].values[:] = obj["tosegment"].values - 1
        tosegment0 = obj["tosegment0"].values

        # -- initial seg_inflow propagation (non-restart). NOTE
        # assignment, not +=: with multiple upstreams the LAST one in
        # index order wins -- pywatershed verbatim. seg_outflow arrived
        # via initial="segment_flow_init". --
        seg_inflow = obj["seg_inflow"].values
        seg_outflow = obj["seg_outflow"].values
        for iseg in range(tosegment0.shape[0]):
            jseg = tosegment0[iseg]
            if jseg < 0:
                continue
            seg_inflow[jseg] = seg_outflow[iseg]

        # -- Muskingum coefficients (module function above; shared with
        # the FlowGraph channel node type) --
        muskingum_mann_coefficients(
            obj["ts"].values,
            obj["tsi"].values,
            obj["c0"].values,
            obj["c1"].values,
            obj["c2"].values,
            obj["mann_n"].values,
            obj["seg_slope"].values,
            obj["seg_depth"].values,
            obj["seg_length"].values,
            obj["segment_type"].values,
            obj["x_coef"].values,
        )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        # pywatershed verbatim (incl. its "seems strange" comment that
        # an instantaneous value is set to a daily-averaged one)
        self._obj["inflow_ts_prev"].values[:] = self._obj["seg_inflow"].values

    @staticmethod
    @numba.njit
    def _calculate(
        seg_lateral_inflow: np.ndarray,
        seg_upstream_inflow: np.ndarray,
        seg_inflow: np.ndarray,
        seg_outflow: np.ndarray,
        seg_stor_change: np.ndarray,
        channel_outflow_vol: np.ndarray,
        inflow_ts: np.ndarray,
        outflow_ts: np.ndarray,
        inflow_ts_prev: np.ndarray,
        seg_current_sum: np.ndarray,
        seg_sroff_vol: np.ndarray,
        seg_ssres_flow_vol: np.ndarray,
        seg_gwres_flow_vol: np.ndarray,
        segment_order: np.ndarray,
        tosegment0: np.ndarray,
        ts: np.ndarray,
        tsi: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
        s_per_time: np.float64,
    ) -> None:
        n_seg = seg_outflow.shape[0]

        # lateral inflow: mapped volumes summed -> cfs. (Float-order
        # note: pywatershed sums (a+b+c) per HRU then aggregates; we
        # aggregate each flux then sum -- same math, different addition
        # order; all-positive so no cancellation.)
        for jj in range(n_seg):
            seg_lateral_inflow[jj] = (
                seg_sroff_vol[jj]
                + seg_ssres_flow_vol[jj]
                + seg_gwres_flow_vol[jj]
            ) / s_per_time

        # muskingum_mann routing -- pywatershed _muskingum_mann_numpy,
        # rewritten in place (its per-day allocations become zeroed
        # buffers)
        seg_inflow[:] = 0.0
        seg_outflow[:] = 0.0
        inflow_ts[:] = 0.0
        seg_current_sum[:] = 0.0

        for ihr in range(24):
            seg_upstream_inflow[:] = 0.0

            for jseg in segment_order:
                # current inflow: upstream avg + lateral
                seg_current_inflow = (
                    seg_lateral_inflow[jseg] + seg_upstream_inflow[jseg]
                )
                seg_inflow[jseg] += seg_current_inflow
                inflow_ts[jseg] += seg_current_inflow
                seg_current_sum[jseg] += seg_upstream_inflow[jseg]

                remainder = (ihr + 1) % tsi[jseg]
                if remainder == 0:
                    # segment routed on the current hour
                    inflow_ts[jseg] /= ts[jseg]
                    if tsi[jseg] > 0:
                        # Muskingum routing equation
                        outflow_ts[jseg] = (
                            inflow_ts[jseg] * c0[jseg]
                            + inflow_ts_prev[jseg] * c1[jseg]
                            + outflow_ts[jseg] * c2[jseg]
                        )
                    else:
                        # travel time <= 1 hour: outflow = inflow
                        outflow_ts[jseg] = inflow_ts[jseg]
                    inflow_ts_prev[jseg] = inflow_ts[jseg]
                    inflow_ts[jseg] = 0.0

                # daily mean outflow accumulates hourly values
                seg_outflow[jseg] += outflow_ts[jseg]
                to_seg = tosegment0[jseg]
                if to_seg >= 0:
                    seg_upstream_inflow[to_seg] += outflow_ts[jseg]

        for jj in range(n_seg):
            seg_outflow[jj] /= 24.0
            seg_inflow[jj] /= 24.0
            seg_upstream_inflow[jj] = seg_current_sum[jj] / 24.0
            seg_stor_change[jj] = (
                seg_inflow[jj] - seg_outflow[jj]
            ) * s_per_time
            # domain outlets only (inline tosegment0 < 0; no mask array)
            if tosegment0[jj] < 0:
                channel_outflow_vol[jj] = seg_outflow[jj] * s_per_time
            else:
                channel_outflow_vol[jj] = 0.0

    def calculate(self, dt: np.float64, time: Time) -> None:
        # dt is SECONDS (s_per_time); 86400.0 for daily PRMS
        self._calculate(
            self._obj["seg_lateral_inflow"].values,
            self._obj["seg_upstream_inflow"].values,
            self._obj["seg_inflow"].values,
            self._obj["seg_outflow"].values,
            self._obj["seg_stor_change"].values,
            self._obj["channel_outflow_vol"].values,
            self._obj["inflow_ts"].values,
            self._obj["outflow_ts"].values,
            self._obj["inflow_ts_prev"].values,
            self._obj["seg_current_sum"].values,
            self._obj["seg_sroff_vol"].values,
            self._obj["seg_ssres_flow_vol"].values,
            self._obj["seg_gwres_flow_vol"].values,
            self._obj["segment_order"].values,
            self._obj["tosegment0"].values,
            self._obj["ts"].values,
            self._obj["tsi"].values,
            self._obj["c0"].values,
            self._obj["c1"].values,
            self._obj["c2"].values,
            dt,
        )
