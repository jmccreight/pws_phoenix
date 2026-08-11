"""
hydrology/prms_stream_temp.py
=============================
PRMSStreamTemp: PRMS/SNTemp daily mean stream temperature, ported
from pywatershed (pywatershed/hydrology/prms_stream_temp.py;
PRMS 5.2.1.1 stream_temp.f90; Markstrom 2012 OFR 2012-1116, Sanders
et al. 2017 TM 6-D4). Stage 3 of the stream-temperature arc.

SEGMENT-GRID process: energy-balance water temperature per segment in
topological order (upstream flow-weighted mixing via upstream_idx),
with gw/subsurface temperatures as running averages of segment air
temperature (circular "silo" buffers over gw_tau/ss_tau days),
lateral inflow temperature mixing (_lat_inflow + monthly
lat_temp_adj), shade via the ``_compute_shade`` hook on the abstract
``PRMSStreamTempBase`` (leaves: ``PRMSStreamTemp`` = DYNAMIC
prms_stream_shade._shday per segment per day;
``PRMSStreamTempConstantShade`` = seasonal segshade_sum/segshade_win
constants -- disjoint parameter sets, hence a base family like
PRMSAtmosphereBase), and the equilibrium-temperature solver
(_equilb/_teak1) + downstream averaging (_twavg).

THE structural departure (design in memory/pws-phoenix-status; the
chain stage will complete it): upstream is secretly a TWO-GRID
process -- it takes HRU inputs and aggregates them to segments
in-process (stream_temp.f90/routing.f90 logic incl. the seg_close
fallback machinery for segments without HRUs; drb has 40 such
segments). Here the HRU-derived quantities are segment-grid INPUTS
(seg_tave_air, seg_humid, seg_ccov, seg_melt, seg_rain, seg_potet,
seginc_swrad, seginc_sroff, seginc_ssflow, seginc_gwflow) -- exactly
the quantities pywatershed generates answers for -- which also
collapses upstream's humidity-source variants
(PRMSStreamTempHumidityCBH vs PRMSStreamTemp seg_humidity parameter):
seg_humid is an input regardless of how a model produces it. The
hru->segment aggregation port (and the humidity leaves) is the chain
stage.

Deliberately NOT ported: the energy-flux tracking variables
(heat_* / longwave_* / evaporative_cooling / convective_exchange --
upstream EXCLUDES them from its own comparison; the kernel is called
with track_energy_fluxes=False and its own dummy arrays, upstream's
off-path); seg_flow_depth/area/velocity (declared upstream, never
used -- only seg_flow_width is consumed); stream_tave_init (declared,
never read); seg_close / hru->seg aggregation (chain stage); Budget;
restart; calc_method; verbose.

Conventions/seams:
- ``segment_order`` = the dis topo-order seam
  (Discretization(topo_order={"segment_order": "tosegment"})), as
  PRMSChannel. Any valid upstream-before-downstream order is
  equivalent for the mixing.
- ``_seg_slope`` / ``seg_length_km`` are derived copies (upstream
  edits the parameters in place: slope clamp to 1e-7, m -> km).
- never-has-flow segments (no HRU area anywhere upstream, walked via
  tosegment) get seg_tave_water = NaN once at initialize; the
  ``hru_segment``/``hru_area`` ('nhru',)-dim parameters exist ONLY
  for that walk (foreign named dim, like nmonth/ndoy).
- gw/ss silos ((nsegment, 366) circular buffers, init -99.9),
  indices/sums, svi and the shade scratch are PYTHON-ATTR state
  (upstream privates, not variables).
- tolerance: the stream-temp family standard is 5e-3 (upstream's own
  comment: iteration-loop and trig noise "just above 32-bit
  precision" vs the Fortran answers).

The njit kernels and helpers below (_teak1, _equilb, _twavg,
_lat_inflow, _compute_mixed_inlet_temp, and the three loop kernels)
are extracted VERBATIM from upstream (staticmethods dedented; the
runtime nb.njit(...) wrapping becomes decorators).
"""

import numba as nb
import numpy as np

from globals import Time
from hydrology.prms_stream_shade import _shday
from process import DataArrayMeta, Process

# pywatershed constants (constants.py + prms_stream_temp.py)
NEARZERO = 1.0e-6
PI = np.pi
CFS_TO_CMS = 1.0 / 35.314666721489
NOFLOW_TEMP = -98.9
DAYS_YR = 365.25
MAX_DAYS_PER_YEAR = 366
ZERO_C = 273.15
TOLRN = 1.0e-4
AKZ = 1.65
A = 5.40e-8
MPS_CONVERT = 2.93981481e-07


# ----------------------------------------------------------------------
# Module helpers -- upstream verbatim (appended by extraction)
# ----------------------------------------------------------------------
@nb.jit(nopython=True)
def _teak1(a_coef, b_coef, c_coef, d_coef, teq, maxiter_sntemp):
    """Solve for equilibrium temperature using Newton-Raphson iteration.

    This is the teak1 function from PRMS.

    Args:
        a_coef: Coefficient (immutable)
        b_coef: Coefficient (immutable)
        c_coef: Coefficient (immutable)
        d_coef: Coefficient (immutable)
        teq: Initial guess for equilibrium temperature (immutable)
        maxiter_sntemp: Maximum iterations (immutable)

    Returns:
        teq: Equilibrium temperature (degC)
        ak1c: First-order thermal exchange coefficient
    """
    # Local variables
    fte = 99999.0
    delte = 99999.0
    kount = 0

    # Begin Newton iteration solution for TE
    while kount < maxiter_sntemp:
        if np.abs(fte) < TOLRN:
            break
        if abs(delte) < TOLRN:
            break
        teabs = teq + ZERO_C
        fte = (
            (a_coef * (teabs**4.0))
            + (b_coef * teq)
            - (c_coef * (teq**2.0))
            - d_coef
        )
        fpte = (4.0 * a_coef * (teabs**3.0)) + b_coef - (2.0 * c_coef * teq)
        delte = fte / fpte
        teq = teq - delte
        kount += 1

    # Determine 1st thermal exchange coefficient
    ak1c = (
        (4.0 * a_coef * ((teq + ZERO_C) ** 3.0))
        + b_coef
        - (2.0 * c_coef * teq)
    )

    return teq, ak1c


@nb.jit(nopython=True)
def _equilb(
    t_o,
    svi,
    seg_inflow,
    seginc_swrad,
    seg_humid,
    seg_elev,
    seg_potet,
    seg_shade,
    seg_ccov,
    seg_flow_width,
    seg_slope,
    seg_tave_gw,
    albedo,
    maxiter_sntemp,
):
    """Compute equilibrium temperature using full energy balance.

    This is the equilb function from PRMS.

    Args:
        t_o: Initial temperature (immutable, degC)
        svi: Vegetation shade index (immutable)
        seg_inflow: Segment inflow (immutable, CFS)
        seginc_swrad: Incident shortwave radiation (immutable)
        seg_humid: Segment humidity (immutable)
        seg_elev: Segment elevation (immutable)
        seg_potet: Segment potential ET (immutable)
        seg_shade: Segment shade fraction (immutable)
        seg_ccov: Cloud cover fraction (immutable)
        seg_flow_width: Flow width (immutable)
        seg_slope: Segment slope (immutable)
        seg_tave_gw: Groundwater temperature (immutable)
        albedo: Albedo (immutable)
        maxiter_sntemp: Maximum iterations (immutable)

    Returns:
        te: Equilibrium temperature (degC)
        ak1: First-order thermal exchange coefficient
        ak2: Second-order thermal exchange coefficient
        hs: Net shortwave solar radiation (W/m²)
        ha: Atmospheric longwave radiation (W/m²)
        hf: Friction heating (W/m²)
        hv: Vegetation longwave radiation (W/m²)
        evap: Evaporation rate (m/s)
        t_abs4: Temperature to 4th power (K^4)
    """
    # Local Variables
    taabs = float(t_o + ZERO_C)

    vp_sat = 6.108 * np.exp(17.26939 * t_o / (t_o + 237.3))

    # Convert units and set up parameters
    q_init = max(seg_inflow * CFS_TO_CMS, NEARZERO)

    sw_power = 11.63 / 24.0 * float(seginc_swrad)

    # If humidity is 1.0, there is a divide by zero below
    foo = min(seg_humid, 0.99)

    # Compute atmospheric pressure based on segment elevation
    press = 1013.0 - (0.1055 * seg_elev)

    bow_coeff = (0.00061 * press) / (vp_sat * (1.0 - foo))
    evap = float(seg_potet * MPS_CONVERT)

    # Heat flux components
    # Ha: atmospheric-emitted longwave radiation
    # Note: Fortran uses Seg_humid directly, not foo (clamped for bow_coeff)
    ha = (
        (3.354939e-8 + 2.74995e-9 * np.sqrt(seg_humid * vp_sat))
        * (1.0 - seg_shade)
        * (1.0 + (0.17 * (seg_ccov**2)))
    ) * (taabs**4)

    # Hf: heat dissipated from potential energy by friction
    hf = 9805.0 * (q_init / seg_flow_width) * seg_slope

    # Hs: net flux from shortwave solar radiation
    hs = (1.0 - seg_shade) * sw_power * (1.0 - albedo)

    # Hv: longwave radiation emitted by riparian vegetation
    hv = 5.24e-8 * svi * (taabs**4)

    # Determine equilibrium coefficients
    del_ht = 2.36e06
    ltnt_ht = 2495.0e06

    b = bow_coeff * evap * (ltnt_ht + (del_ht * t_o)) + AKZ - (del_ht * evap)
    c = bow_coeff * del_ht * evap
    d = (ha + hv + hf + hs) + (
        ltnt_ht * evap * ((bow_coeff * t_o) - 1.0) + (seg_tave_gw * AKZ)
    )

    # Determine equilibrium temperature & 1st order thermal exchange coef
    ted = t_o
    ted, ak1d = _teak1(A, b, c, d, ted, maxiter_sntemp)

    # Determine 2nd order thermal exchange coefficient
    hnet = (A * ((t_o + ZERO_C) ** 4)) + (b * t_o) - (c * (t_o**2.0)) - d
    delt = t_o - ted

    if abs(delt) < NEARZERO:
        ak2d = 0.0
    else:
        ak2d = ((delt * ak1d) - hnet) / (delt**2)

    return ted, ak1d, ak2d, hs, ha, hf, hv, evap, taabs**4


@nb.jit(nopython=True)
def _twavg(
    qup,
    t0,
    qlat,
    tl_avg,
    te,
    ak1,
    ak2,
    seg_flow_width,
    seg_length,
    atmos_exchange_factor=1.0,
):
    """Compute average water temperature with lateral inflows.

    This is the twavg function from PRMS.

    Args:
        qup: Upstream flow (immutable, cfs)
        t0: Inlet temperature (immutable, degC)
        qlat: Lateral flow (immutable, cms)
        tl_avg: Lateral flow temperature (immutable, degC)
        te: Equilibrium temperature (immutable, degC)
        ak1: First-order thermal exchange coefficient (immutable)
        ak2: Second-order thermal exchange coefficient (immutable)
        seg_flow_width: Flow width (immutable)
        seg_length: Segment length (immutable)
        atmos_exchange_factor: Factor to amplify atmospheric exchange effects
            (immutable, default=1.0)

    Returns:
        tw: Average water temperature (degC)
    """
    # Determine equation parameters
    q_init = float(qup * CFS_TO_CMS)
    ql = float(qlat)
    width = seg_flow_width
    length = seg_length

    # Local Variables
    tep = 0.0
    b = 0.0
    r = 0.0
    rexp = 0.0
    tw = 0.0
    delt = 0.0
    denom = 0.0

    if ql <= NEARZERO:
        # Zero lateral flow
        tep = te
        b = (ak1 * atmos_exchange_factor * width) / 4182.0e03
        rexp = -1.0 * (b * length) / q_init
        r = np.exp(rexp)

    elif ql < 0.0:
        # Losing stream (should not happen in PRMS)
        tep = te
        b = (ql / length) + ((ak1 * atmos_exchange_factor * width) / 4182.0e03)
        rexp = (ql - (b * length)) / ql
        r = 1.0 + (ql / q_init)
        r = r**rexp

    elif ql > NEARZERO and q_init <= NEARZERO:
        tep = te
        b = (ak1 * atmos_exchange_factor * width) / 4182.0e03
        rexp = -1.0 * (b * length) / ql
        r = np.exp(rexp)

    else:
        b = (ql / length) + ((ak1 * atmos_exchange_factor * width) / 4182.0e03)
        tep = (
            ((ql / length) * tl_avg)
            + (((ak1 * atmos_exchange_factor * width) / (4182.0e03)) * te)
        ) / b

        if ql > 0.0:
            rexp = -b / (ql / length)
        else:
            rexp = 0.0

        if q_init < NEARZERO:
            r = 2.0
        else:
            r = 1.0 + (ql / q_init)
        r = r**rexp

    # Determine water temperature
    delt = tep - t0
    denom = 1.0 + (ak2 / ak1) * delt * (1.0 - r)

    if np.abs(denom) < NEARZERO:
        denom = np.sign(denom) * NEARZERO if denom != 0.0 else NEARZERO

    tw = tep - (delt * r / denom)
    if tw < 0.0:
        tw = 0.0

    return tw


@nb.jit(nopython=True)
def _lat_inflow(
    seg_lateral_inflow,
    seginc_sroff,
    seginc_ssflow,
    seginc_gwflow,
    melt_temp,
    tave_gw,
    tave_air,
    tave_ss,
    melt,
    rain,
):
    """Compute lateral inflow temperature from components.

    This is the lat_inflow function from PRMS.

    Args:
        seg_lateral_inflow: Total lateral inflow to segment (immutable, cfs)
        seginc_sroff: Surface runoff component (immutable, cfs)
        seginc_ssflow: Subsurface flow component (immutable, cfs)
        seginc_gwflow: Groundwater flow component (immutable, cfs)
        melt_temp: Snowmelt temperature (immutable, degC)
        tave_gw: Groundwater temperature (immutable, degC)
        tave_air: Air temperature (immutable, degC)
        tave_ss: Subsurface temperature (immutable, degC)
        melt: Snowmelt (immutable, inches)
        rain: Rainfall (immutable, inches)

    Returns:
        tl_avg: Weighted average lateral inflow temperature (degC)
        qlat: Lateral inflow (cms)
    """
    weight_roff = 0.0
    weight_ss = 0.0
    weight_gw = 0.0
    melt_wt = 0.0
    rain_wt = 0.0
    troff = 0.0
    tss = 0.0

    qlat = seg_lateral_inflow * CFS_TO_CMS
    tl_avg = 0.0

    if qlat > 0.0:
        weight_roff = float((seginc_sroff * CFS_TO_CMS) / qlat)
        weight_ss = float((seginc_ssflow * CFS_TO_CMS) / qlat)
        weight_gw = float((seginc_gwflow * CFS_TO_CMS) / qlat)
    else:
        weight_roff = 0.0
        weight_ss = 0.0
        weight_gw = 0.0

    if melt > 0.0:
        melt_wt = melt / (melt + rain)
        if melt_wt < 0.0:
            melt_wt = 0.0
        if melt_wt > 1.0:
            melt_wt = 1.0
        rain_wt = 1.0 - melt_wt
        if rain == 0.0:
            troff = melt_temp
            tss = melt_temp
        else:
            troff = melt_temp * melt_wt + tave_air * rain_wt
            tss = melt_temp * melt_wt + tave_ss * rain_wt
    else:
        troff = tave_air
        tss = tave_ss

    if weight_roff == 0.0 and weight_ss == 0.0 and weight_gw == 0.0:
        tl_avg = np.nan
        qlat = np.nan
    else:
        tl_avg = weight_roff * troff + weight_ss * tss + weight_gw * tave_gw

    return tl_avg, qlat


@nb.jit(nopython=True)
def _compute_mixed_inlet_temp(
    upstream_flow, lateral_flow, seg_tave_upstream, seg_tave_lat
):
    """Compute mixed inlet temperature from upstream and lateral sources.

    Args:
        upstream_flow: Flow from upstream segments (immutable, cfs)
        lateral_flow: Lateral inflow (immutable, cfs)
        seg_tave_upstream: Upstream temperature (immutable, degC)
        seg_tave_lat: Lateral flow temperature (immutable, degC)

    Returns:
        Mixed inlet temperature (degC)
    """
    upstream_ready = upstream_flow > 0.0 and not np.isnan(seg_tave_upstream)
    lateral_ready = lateral_flow > 0.0 and not np.isnan(seg_tave_lat)

    if not upstream_ready and not lateral_ready:
        return np.nan
    elif upstream_ready and not lateral_ready:
        return seg_tave_upstream
    elif lateral_ready and not upstream_ready:
        return seg_tave_lat
    else:
        # Both sources present - compute weighted average
        return (
            seg_tave_upstream * upstream_flow + seg_tave_lat * lateral_flow
        ) / (upstream_flow + lateral_flow)


# ----------------------------------------------------------------------
# Loop kernels -- upstream staticmethods verbatim (dedented; njit'd)
# ----------------------------------------------------------------------


@nb.jit(nopython=True)
def _update_running_avg_temp(
    segment_order,
    seg_tave_water,
    seginc_swrad,
    seg_tave_air,
    gw_tau,
    gw_index,
    gw_silo,
    gw_sum,
    seg_tave_gw,
    ss_tau,
    ss_index,
    ss_silo,
    ss_sum,
    seg_tave_ss,
):
    """Update running average temperatures for groundwater and subsurface.

    This function can be optionally JIT-compiled with numba for
    performance.

    Args:
        segment_order: Order to process segments (immutable)
        seg_tave_water: Water temperature array (immutable, for skip
            checks)
        seginc_swrad: Solar radiation array (immutable, for skip checks)
        seg_tave_air: Air temperature array (immutable)
        gw_tau: Groundwater tau values (immutable)
        gw_index: Groundwater circular buffer indices (MUTATED)
        gw_silo: Groundwater circular buffer (MUTATED)
        gw_sum: Groundwater running sums (MUTATED)
        seg_tave_gw: Groundwater temperatures (MUTATED - output)
        ss_tau: Subsurface tau values (immutable)
        ss_index: Subsurface circular buffer indices (MUTATED)
        ss_silo: Subsurface circular buffer (MUTATED)
        ss_sum: Subsurface running sums (MUTATED)
        seg_tave_ss: Subsurface temperatures (MUTATED - output)
    """
    for jj in segment_order:
        # Skip if marked as permanently invalid (no HRUs
        # upstream/downstream)
        if seginc_swrad[jj] < -99.0:
            continue

        # Update groundwater running average
        idx_gw = gw_index[jj]
        tau_gw = int(gw_tau[jj])

        # Add new air temperature to silo
        gw_silo[jj, idx_gw] = seg_tave_air[jj]

        # Recompute sum and count valid entries (matches Fortran)
        # This handles spin-up period correctly
        at_sum = 0.0
        at_cnt = 0
        for j in range(tau_gw):
            if gw_silo[jj, j] > -98.0:
                at_sum += gw_silo[jj, j]
                at_cnt += 1

        # Compute average as sum / count
        if at_cnt > 0:
            seg_tave_gw[jj] = at_sum / at_cnt
        else:
            seg_tave_gw[jj] = 0.0

        # Update index
        if idx_gw < tau_gw - 1:
            gw_index[jj] = idx_gw + 1
        else:
            gw_index[jj] = 0

        # Update subsurface running average
        idx_ss = ss_index[jj]
        tau_ss = int(ss_tau[jj])

        # Add new air temperature to silo
        ss_silo[jj, idx_ss] = seg_tave_air[jj]

        # Recompute sum and count valid entries (matches Fortran)
        # This handles spin-up period correctly
        at_sum = 0.0
        at_cnt = 0
        for j in range(tau_ss):
            if ss_silo[jj, j] > -98.0:
                at_sum += ss_silo[jj, j]
                at_cnt += 1

        # Compute average as sum / count
        if at_cnt > 0:
            seg_tave_ss[jj] = at_sum / at_cnt
        else:
            seg_tave_ss[jj] = 0.0

        # Update index
        if idx_ss < tau_ss - 1:
            ss_index[jj] = idx_ss + 1
        else:
            ss_index[jj] = 0


@nb.jit(nopython=True)
def _compute_lateral_temp(
    segment_order,
    seg_tave_water,
    seginc_swrad,
    seg_lateral_inflow,
    seginc_sroff,
    seginc_ssflow,
    seginc_gwflow,
    seg_melt,
    seg_rain,
    seg_tave_gw,
    seg_tave_air,
    seg_tave_ss,
    melt_temp,
    lat_temp_adj,
    nowmonth,
    seg_tave_lat,
):
    """Compute lateral flow temperatures for all segments.

    This function can be optionally JIT-compiled with numba for
    performance.

    Args:
        segment_order: Order to process segments (immutable)
        seg_tave_water: Water temperature array (immutable, for skip
            checks)
        seginc_swrad: Solar radiation array (immutable, for skip checks)
        seg_lateral_inflow: Lateral inflow array (immutable)
        seginc_sroff: Surface runoff array (immutable)
        seginc_ssflow: Subsurface flow array (immutable)
        seginc_gwflow: Groundwater flow array (immutable)
        seg_melt: Snowmelt array (immutable)
        seg_rain: Rainfall array (immutable)
        seg_tave_gw: Groundwater temperature array (immutable)
        seg_tave_air: Air temperature array (immutable)
        seg_tave_ss: Subsurface temperature array (immutable)
        melt_temp: Melt temperature constant (immutable)
        lat_temp_adj: Monthly lateral temperature adjustment array
            (immutable)
        nowmonth: Current month (immutable, 1-based)
        seg_tave_lat: Lateral temperature array (MUTATED - output)
    """
    for jj in segment_order:
        # Skip if marked as never having flow (NaN = never has flow)
        if np.isnan(seg_tave_water[jj]):
            continue
        # Skip if marked as permanently invalid
        if seginc_swrad[jj] < -99.0:
            continue

        # Get segment values
        sroff = seginc_sroff[jj]
        ssflow = seginc_ssflow[jj]
        gwflow = seginc_gwflow[jj]
        melt = seg_melt[jj]
        rain = seg_rain[jj]
        tave_gw = seg_tave_gw[jj]
        tave_air = seg_tave_air[jj]
        tave_ss = seg_tave_ss[jj]

        # Use lat_inflow function for detailed lateral temperature
        # calculation
        tl_avg, qlat = _lat_inflow(
            seg_lateral_inflow[jj],
            sroff,
            ssflow,
            gwflow,
            melt_temp,
            tave_gw,
            tave_air,
            tave_ss,
            melt,
            rain,
        )

        # Apply monthly adjustment
        if not np.isnan(tl_avg):
            tl_avg += lat_temp_adj[nowmonth - 1, jj]

        # Ensure non-negative (also converts NaN to 0.0 to match Fortran)
        if np.isnan(tl_avg) or tl_avg < 0.0:
            tl_avg = 0.0

        seg_tave_lat[jj] = tl_avg


@nb.jit(nopython=True)
def _compute_water_temp(
    segment_order,
    seg_tave_water,
    seginc_swrad,
    seg_outflow,
    upstream_count,
    upstream_idx,
    seg_lateral_inflow,
    seg_tave_upstream,
    seg_svi_all,
    seg_inflow,
    seg_tave_lat,
    seginc_swrad_data,
    seg_humid,
    seg_elev,
    seg_potet,
    seg_shade,
    seg_ccov,
    seg_flow_width,
    seg_slope,
    seg_tave_gw,
    seg_length,
    albedo,
    maxiter_sntemp,
    track_energy_fluxes,
    hs_terms,
    ha_terms,
    hf_terms,
    hv_terms,
    evap_terms,
    t_abs4_terms,
    upstream_flows,
    lateral_flows,
    atmos_exchange_factor=1.0,
):
    """Compute water temperature for all segments.

    This function can be optionally JIT-compiled with numba for
    performance.

    Args:
        segment_order: Order to process segments (immutable)
        seg_tave_water: Water temperature array (MUTATED - input/output)
        seginc_swrad: Solar radiation array (immutable, for skip checks)
        seg_outflow: Segment outflow array (immutable)
        upstream_count: Number of upstream segments for each segment
            (immutable)
        upstream_idx: Indices of upstream segments (immutable)
        seg_lateral_inflow: Lateral inflow array (immutable)
        seg_tave_upstream: Upstream temperature array (MUTATED - output)
        seg_svi_all: Pre-computed vegetation shade index array (immutable)
        seg_inflow: Segment inflow array (MUTATED - output)
        seg_tave_lat: Lateral temperature array (immutable)
        seginc_swrad_data: Solar radiation data for energy balance
            (immutable)
        seg_humid: Humidity array (immutable)
        seg_elev: Elevation array (immutable)
        seg_potet: Potential ET array (immutable)
        seg_shade: Shade fraction array (immutable)
        seg_ccov: Cloud cover array (immutable)
        seg_flow_width: Flow width array (immutable)
        seg_slope: Slope array (immutable)
        seg_tave_gw: Groundwater temperature array (immutable)
        seg_length: Segment length array (immutable)
        albedo: Albedo value (immutable)
        maxiter_sntemp: Maximum iterations for temperature solver
            (immutable)
        track_energy_fluxes: Whether to track energy flux components
            (immutable)
        hs_terms: Solar radiation terms (MUTATED if tracking - output)
        ha_terms: Atmospheric longwave terms (MUTATED if tracking - output)
        hf_terms: Friction heat terms (MUTATED if tracking - output)
        hv_terms: Vegetation longwave terms (MUTATED if tracking - output)
        evap_terms: Evaporation terms (MUTATED if tracking - output)
        t_abs4_terms: Temperature^4 terms (MUTATED if tracking - output)
        upstream_flows: Upstream flow values (MUTATED if tracking - output)
        lateral_flows: Lateral flow values (MUTATED if tracking - output)
        atmos_exchange_factor: Factor to amplify atmospheric exchange
            (immutable, default=1.0)
    """
    for jj in segment_order:
        # Skip segments marked as never having flow (NaN = never has flow)
        if np.isnan(seg_tave_water[jj]):
            continue

        # Compute upstream temperature and flow in one pass
        # (consolidates _compute_upstream_temp and _compute_inflow)
        flow_sum = 0.0
        temp_sum = 0.0
        upstream_flow = 0.0

        for kk in range(upstream_count[jj]):
            up_idx = upstream_idx[jj, kk]
            if (
                not np.isnan(seg_tave_water[up_idx])
                and seg_tave_water[up_idx] > NOFLOW_TEMP
            ):
                flow = seg_outflow[up_idx]
                temp_sum += seg_tave_water[up_idx] * flow
                flow_sum += flow
                upstream_flow += flow

        if flow_sum > 0.0:
            seg_tave_upstream[jj] = temp_sum / flow_sum
        else:
            seg_tave_upstream[jj] = NOFLOW_TEMP

        # Get svi from pre-computed array
        svi = seg_svi_all[jj]

        # Compute total inflow
        seg_inflow[jj] = upstream_flow + seg_lateral_inflow[jj]

        # Now compute water temperature (logic from _compute_water_temp)

        # Skip if marked as permanently invalid
        if seginc_swrad[jj] < -99.0:
            seg_tave_water[jj] = np.nan
            continue

        # Check for no-flow conditions
        if seg_outflow[jj] <= 0.0:
            seg_tave_water[jj] = NOFLOW_TEMP
            continue

        lateral_flow = seg_lateral_inflow[jj]
        qlat = lateral_flow * CFS_TO_CMS

        # Match Fortran: check qlat in CMS (not lateral_flow in CFS)
        if upstream_flow * CFS_TO_CMS <= NEARZERO and qlat <= NEARZERO:
            seg_tave_water[jj] = NOFLOW_TEMP
            continue

        # Compute mixed inlet temperature
        t_in = _compute_mixed_inlet_temp(
            upstream_flow,
            lateral_flow,
            seg_tave_upstream[jj],
            seg_tave_lat[jj],
        )

        if np.isnan(t_in):
            seg_tave_water[jj] = NOFLOW_TEMP
            continue

        # Compute equilibrium temperature using full energy balance
        result = _equilb(
            t_in,
            svi,
            seg_inflow[jj],
            seginc_swrad_data[jj],
            seg_humid[jj],
            seg_elev[jj],
            seg_potet[jj],
            seg_shade[jj],
            seg_ccov[jj],
            seg_flow_width[jj],
            seg_slope[jj],
            seg_tave_gw[jj],
            albedo,
            maxiter_sntemp,
        )

        te = result[0]
        ak1 = result[1]
        ak2 = result[2]

        # Store energy flux terms if tracking
        if track_energy_fluxes:
            hs_terms[jj] = result[3]
            ha_terms[jj] = result[4]
            hf_terms[jj] = result[5]
            hv_terms[jj] = result[6]
            evap_terms[jj] = result[7]
            t_abs4_terms[jj] = result[8]
            upstream_flows[jj] = upstream_flow
            lateral_flows[jj] = lateral_flow

        # Compute final temperature using twavg function
        qup = upstream_flow
        tl_avg = seg_tave_lat[jj]

        seg_tave_water[jj] = _twavg(
            qup,
            t_in,
            qlat,
            tl_avg,
            te,
            ak1,
            ak2,
            seg_flow_width[jj],
            seg_length[jj],
            atmos_exchange_factor,
        )


@nb.jit(nopython=True)
def _reset_water_temp(
    segment_order: np.ndarray, seg_tave_water: np.ndarray
) -> None:
    """Per-step reset of valid segments to -99.9 (never-flow NaNs
    kept) -- upstream's python loop before _compute_water_temp."""
    for jj in segment_order:
        if not np.isnan(seg_tave_water[jj]):
            seg_tave_water[jj] = -99.9


@nb.jit(nopython=True)
def _shade_all(
    # written in place (allocation-free _shday_vectorized)
    seg_shade: np.ndarray,
    seg_svi: np.ndarray,
    # per-day scalars
    declination: float,
    summer_flag: int,
    # arrays
    seg_lat_rad: np.ndarray,
    seg_flow_width: np.ndarray,
    azrh: np.ndarray,
    alte: np.ndarray,
    altw: np.ndarray,
    vce: np.ndarray,
    voe: np.ndarray,
    vhe: np.ndarray,
    vdemx: np.ndarray,
    vdemn: np.ndarray,
    vcw: np.ndarray,
    vow: np.ndarray,
    vhw: np.ndarray,
    vdwmx: np.ndarray,
    vdwmn: np.ndarray,
    maxiter_sntemp: int,
) -> None:
    for ss in range(seg_shade.shape[0]):
        seg_shade[ss], seg_svi[ss] = _shday(
            seg_lat_rad[ss],
            declination,
            seg_flow_width[ss],
            azrh[ss],
            alte[ss],
            altw[ss],
            vce[ss],
            voe[ss],
            vhe[ss],
            vdemx[ss],
            vdemn[ss],
            summer_flag,
            vcw[ss],
            vow[ss],
            vhw[ss],
            vdwmx[ss],
            vdwmn[ss],
            maxiter_sntemp,
        )


def _meta(
    kind,
    description,
    dtype=np.float64,
    dims=("space",),
    restart=False,
    derivation=None,
):
    return DataArrayMeta(
        kind=kind,
        dims=dims,
        dtype=dtype,
        description=description,
        restart=restart,
        derivation=derivation,
    )


class PRMSStreamTempBase(Process):
    """PRMS/SNTemp daily mean stream temperature on the segment grid:
    the family CORE, shade-source-agnostic (abstract). HRU-derived
    meteorology/flow aggregates are segment inputs (see module
    docstring). Temperatures degC; flows cfs at the interface (cms
    internally where upstream converts).

    Shade is upstream's composed strategy objects
    (PRMSStreamShadeDynamic / PRMSStreamShadeConstant); here the seam
    is the ``_compute_shade(declination, summer_flag)`` hook -- leaves
    ADD their shade-source parameters and implement it, writing
    ``seg_shade`` (and the ``_seg_svi`` scratch) in place. Per the
    variants stance (PORTS.md "How variants are done here"): the
    dynamic and constant parameter sets are disjoint, so neither leaf
    may extend the other.
    """

    # -- dis_seg variables (grid-owned; dis-first sourcing) --
    segment_order = _meta(
        "parameter",
        "Upstream-to-downstream ordering (0-based)",
        np.int64,
        derivation=(
            "Discretization(topo_order={'segment_order': 'tosegment'})"
        ),
    )
    tosegment = _meta(
        "parameter",
        "Downstream segment index (1-based; 0 = outlet)",
        np.int64,
    )
    seg_length = _meta("parameter", "Segment length [m]")
    seg_slope = _meta("parameter", "Segment slope [-]")
    seg_lat = _meta("parameter", "Segment latitude [degrees N]")
    seg_elev = _meta("parameter", "Segment elevation [m]")
    lat_temp_adj = _meta(
        "parameter",
        "Monthly lateral temperature adjustment [degC]",
        dims=("nmonth", "space"),
    )

    # -- process parameters --
    gw_tau = _meta("parameter", "Groundwater temperature averaging [days]")
    ss_tau = _meta("parameter", "Subsurface temperature averaging [days]")
    albedo = _meta("parameter", "Water surface albedo [-]", dims=("scalar",))
    melt_temp = _meta(
        "parameter", "Temperature of snowmelt [degC]", dims=("scalar",)
    )
    maxiter_sntemp = _meta(
        "parameter",
        "Maximum solver iterations",
        np.int64,
        dims=("scalar",),
    )
    # ('nhru',)-dim: ONLY for the never-has-flow walk at initialize
    hru_segment = _meta(
        "parameter",
        "HRU -> segment assignment (1-based; 0 = none) -- never-flow "
        "walk only in this class",
        np.int64,
        dims=("nhru",),
    )
    hru_area = _meta(
        "parameter",
        "HRU area [acres] -- never-flow walk only in this class",
        dims=("nhru",),
    )

    # -- derived parameters (upstream edits/conversions of params) --
    _seg_slope = _meta(
        "parameter_internal", "seg_slope clamped to >= 1e-7 (upstream edit)"
    )
    seg_length_km = _meta(
        "parameter_internal", "Segment length [km] (m / 1000)"
    )
    _seg_lat_rad = _meta("parameter_internal", "Segment latitude [radians]")

    # -- inputs: routing (PRMSChannel) + hydraulic geometry --
    seg_outflow = _meta("input", "Streamflow leaving each segment [cfs]")
    seg_lateral_inflow = _meta("input", "Lateral inflow to each segment [cfs]")
    seg_flow_width = _meta("input", "Flow width [m] (PRMSHydraulicGeometry*)")

    # -- inputs: HRU-derived aggregates (see module docstring) --
    seg_tave_air = _meta("input", "Segment air temperature [degC]")
    seg_humid = _meta("input", "Segment relative humidity [fraction]")
    seg_ccov = _meta("input", "Segment cloud cover [fraction]")
    seg_melt = _meta("input", "Segment snowmelt [inches]")
    seg_rain = _meta("input", "Segment rainfall [inches]")
    seg_potet = _meta("input", "Segment potential ET [inches]")
    seginc_swrad = _meta(
        "input",
        "Segment shortwave [cal/cm^2] (-99.9 = no contributing HRUs "
        "anywhere: the permanent-skip marker)",
    )
    seginc_sroff = _meta("input", "Segment surface runoff [cfs]")
    seginc_ssflow = _meta("input", "Segment subsurface flow [cfs]")
    seginc_gwflow = _meta("input", "Segment groundwater flow [cfs]")

    # -- variables --
    seg_tave_water = _meta(
        "variable",
        "Water temperature [degC] (-99.9 invalid; -98.9 NOFLOW; NaN "
        "never-has-flow)",
        restart=True,
    )
    seg_tave_upstream = _meta(
        "variable", "Flow-weighted upstream water temperature [degC]"
    )
    seg_tave_gw = _meta(
        "variable", "Groundwater temperature [degC] (gw_tau-day average)"
    )
    seg_tave_ss = _meta(
        "variable", "Subsurface temperature [degC] (ss_tau-day average)"
    )
    seg_tave_lat = _meta("variable", "Lateral inflow temperature [degC]")
    seg_shade = _meta("variable", "Shade fraction [-] (dynamic shday)")

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        obj = self._obj
        nsegment = obj["seg_length"].values.shape[0]

        for name in (
            "seg_tave_gw",
            "seg_tave_ss",
            "seg_tave_lat",
            "seg_shade",
        ):
            obj[name].values[:] = 0.0
        obj["seg_tave_water"].values[:] = -99.9
        obj["seg_tave_upstream"].values[:] = np.nan

        # seg_length validation + m -> km (upstream edits the param)
        seg_length = obj["seg_length"].values
        if (seg_length < NEARZERO).any():
            raise ValueError(
                "seg_length too small for segments "
                f"{np.where(seg_length < NEARZERO)[0]}"
            )
        obj["seg_length_km"].values[:] = seg_length / 1000.0

        # seg_slope clamp (upstream edits the param; derived copy here)
        obj["_seg_slope"].values[:] = np.where(
            obj["seg_slope"].values < 0.0000001,
            0.0000001,
            obj["seg_slope"].values,
        )

        obj["_seg_lat_rad"].values[:] = obj["seg_lat"].values * (np.pi / 180.0)

        # scalar parameters
        self._albedo = float(obj["albedo"].values[0])
        self._melt_temp = float(obj["melt_temp"].values[0])
        self._maxiter = int(obj["maxiter_sntemp"].values[0])

        # segment HRU areas (never-flow walk only in this class)
        hru_segment = obj["hru_segment"].values
        hru_area = obj["hru_area"].values
        segment_hruarea = np.zeros(nsegment, dtype=np.float64)
        for jhru in range(hru_segment.shape[0]):
            seg_idx = hru_segment[jhru]
            if seg_idx > 0:
                segment_hruarea[seg_idx - 1] += hru_area[jhru]
        self._segment_hruarea = segment_hruarea

        # upstream segment info (tosegment 1-based scan; max 10)
        tosegment = obj["tosegment"].values
        self._upstream_count = np.zeros(nsegment, dtype=np.int32)
        self._upstream_idx = np.zeros((nsegment, 10), dtype=np.int32)
        for jseg in range(nsegment):
            count = 0
            for iseg in range(nsegment):
                if tosegment[iseg] > 0 and tosegment[iseg] == jseg + 1:
                    self._upstream_idx[jseg, count] = iseg
                    count += 1
            self._upstream_count[jseg] = count

        # never-has-flow marking (upstream init, verbatim walk): a
        # segment without HRUs anywhere upstream can never have flow
        seg_tave_water = obj["seg_tave_water"].values
        for iseg in range(nsegment):
            if segment_hruarea[iseg] <= NEARZERO:
                has_upstream_hrus = False
                this_seg = iseg
                visited = set()
                while this_seg not in visited:
                    visited.add(this_seg)
                    found_upstream = False
                    for jseg in range(nsegment):
                        if (
                            tosegment[jseg] > 0
                            and tosegment[jseg] == this_seg + 1
                        ):
                            if segment_hruarea[jseg] > NEARZERO:
                                has_upstream_hrus = True
                                break
                            this_seg = jseg
                            found_upstream = True
                            break
                    if has_upstream_hrus or not found_upstream:
                        break
                if not has_upstream_hrus:
                    seg_tave_water[iseg] = np.nan

        # solar declination table (upstream _precompute_solar_geometry)
        k = np.arange(MAX_DAYS_PER_YEAR) + 1.0
        self._declination = 0.40928 * np.cos(
            ((2.0 * PI) / DAYS_YR) * (172.0 - k)
        )

        # circular-buffer state + scratch (upstream privates)
        self._gw_silo = np.full(
            (nsegment, MAX_DAYS_PER_YEAR), -99.9, dtype=np.float64
        )
        self._ss_silo = np.full(
            (nsegment, MAX_DAYS_PER_YEAR), -99.9, dtype=np.float64
        )
        self._gw_index = np.zeros(nsegment, dtype=np.int32)
        self._ss_index = np.zeros(nsegment, dtype=np.int32)
        self._gw_sum = np.zeros(nsegment, dtype=np.float64)
        self._ss_sum = np.zeros(nsegment, dtype=np.float64)
        self._seg_inflow = np.zeros(nsegment, dtype=np.float64)
        self._seg_svi = np.zeros(nsegment, dtype=np.float64)
        self._dummy1 = np.zeros(1, dtype=np.float64)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        pass  # silos/indices persist; no *_prev variables

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        doy = time.doy  # 1-based
        declination = float(self._declination[doy - 1])
        summer_flag = 1 if 121 <= doy <= 273 else 0

        _update_running_avg_temp(
            obj["segment_order"].values,
            obj["seg_tave_water"].values,
            obj["seginc_swrad"].values,
            obj["seg_tave_air"].values,
            obj["gw_tau"].values,
            self._gw_index,
            self._gw_silo,
            self._gw_sum,
            obj["seg_tave_gw"].values,
            obj["ss_tau"].values,
            self._ss_index,
            self._ss_silo,
            self._ss_sum,
            obj["seg_tave_ss"].values,
        )

        _compute_lateral_temp(
            obj["segment_order"].values,
            obj["seg_tave_water"].values,
            obj["seginc_swrad"].values,
            obj["seg_lateral_inflow"].values,
            obj["seginc_sroff"].values,
            obj["seginc_ssflow"].values,
            obj["seginc_gwflow"].values,
            obj["seg_melt"].values,
            obj["seg_rain"].values,
            obj["seg_tave_gw"].values,
            obj["seg_tave_air"].values,
            obj["seg_tave_ss"].values,
            self._melt_temp,
            obj["lat_temp_adj"].values,
            time.month,
            obj["seg_tave_lat"].values,
        )

        # (upstream: zero each step; skipped segments keep 0.0)
        obj["seg_tave_upstream"].values[:] = 0.0

        self._compute_shade(declination, summer_flag)

        _reset_water_temp(
            obj["segment_order"].values, obj["seg_tave_water"].values
        )

        _compute_water_temp(
            obj["segment_order"].values,
            obj["seg_tave_water"].values,
            obj["seginc_swrad"].values,
            obj["seg_outflow"].values,
            self._upstream_count,
            self._upstream_idx,
            obj["seg_lateral_inflow"].values,
            obj["seg_tave_upstream"].values,
            self._seg_svi,
            self._seg_inflow,
            obj["seg_tave_lat"].values,
            obj["seginc_swrad"].values,
            obj["seg_humid"].values,
            obj["seg_elev"].values,
            obj["seg_potet"].values,
            obj["seg_shade"].values,
            obj["seg_ccov"].values,
            obj["seg_flow_width"].values,
            obj["_seg_slope"].values,
            obj["seg_tave_gw"].values,
            obj["seg_length_km"].values,
            self._albedo,
            self._maxiter,
            False,  # track_energy_fluxes (upstream's own off-path)
            self._dummy1,
            self._dummy1,
            self._dummy1,
            self._dummy1,
            self._dummy1,
            self._dummy1,
            self._dummy1,
            self._dummy1,
            1.0,  # atmos_exchange_factor (upstream default)
        )

    def _compute_shade(self, declination: float, summer_flag: int) -> None:
        """Write seg_shade (and the _seg_svi scratch) in place for the
        current day. Leaves supply the shade source."""
        raise NotImplementedError(
            "PRMSStreamTempBase is abstract: use PRMSStreamTemp "
            "(dynamic shade) or PRMSStreamTempConstantShade."
        )

    # -- restart hooks: the gw/ss running-average silos and their
    # circular-buffer indices are EVOLVING python-attr state (the
    # statics -- topology, declination table -- are rebuilt by
    # initialize(); the other attrs are per-step scratch) --

    def get_restart_state(self) -> dict[str, np.ndarray]:
        return {
            "gw_silo": self._gw_silo,
            "ss_silo": self._ss_silo,
            "gw_index": self._gw_index,
            "ss_index": self._ss_index,
        }

    def set_restart_state(self, state: dict[str, np.ndarray]) -> None:
        self._gw_silo[:] = state["gw_silo"]
        self._ss_silo[:] = state["ss_silo"]
        self._gw_index[:] = state["gw_index"]
        self._ss_index[:] = state["ss_index"]


class PRMSStreamTemp(PRMSStreamTempBase):
    """The core with DYNAMIC shade (stream_temp_shade_flag = 0, the
    drb configuration): per-day per-segment shday
    (prms_stream_shade._shday) from the vegetation/topography
    parameters."""

    # -- dynamic shade parameters (PRMS stream_temp shday) --
    azrh = _meta("parameter", "Stream azimuth angle [radians]")
    alte = _meta("parameter", "East bank topographic altitude [radians]")
    altw = _meta("parameter", "West bank topographic altitude [radians]")
    vce = _meta("parameter", "East bank vegetation crown width [m]")
    voe = _meta("parameter", "East bank vegetation offset [m]")
    vhe = _meta("parameter", "East bank vegetation height [m]")
    vdemx = _meta("parameter", "Max east bank vegetation density [-]")
    vdemn = _meta("parameter", "Min east bank vegetation density [-]")
    vcw = _meta("parameter", "West bank vegetation crown width [m]")
    vow = _meta("parameter", "West bank vegetation offset [m]")
    vhw = _meta("parameter", "West bank vegetation height [m]")
    vdwmx = _meta("parameter", "Max west bank vegetation density [-]")
    vdwmn = _meta("parameter", "Min west bank vegetation density [-]")

    def _compute_shade(self, declination: float, summer_flag: int) -> None:
        obj = self._obj
        _shade_all(
            obj["seg_shade"].values,
            self._seg_svi,
            declination,
            summer_flag,
            obj["_seg_lat_rad"].values,
            obj["seg_flow_width"].values,
            obj["azrh"].values,
            obj["alte"].values,
            obj["altw"].values,
            obj["vce"].values,
            obj["voe"].values,
            obj["vhe"].values,
            obj["vdemx"].values,
            obj["vdemn"].values,
            obj["vcw"].values,
            obj["vow"].values,
            obj["vhw"].values,
            obj["vdwmx"].values,
            obj["vdwmn"].values,
            self._maxiter,
        )


class PRMSStreamTempConstantShade(PRMSStreamTempBase):
    """The core with CONSTANT seasonal shade (stream_temp_shade_flag =
    1; upstream PRMSStreamShadeConstant, verbatim semantics): seg_shade
    = segshade_sum on summer days (doy 121-273) and segshade_win
    otherwise; the vegetation shade index svi is 0.0 (so the hv
    longwave term in _equilb vanishes) -- _seg_svi stays at its
    initialize() zeros and is never written."""

    segshade_sum = _meta(
        "parameter", "Total summer vegetation shade fraction [-]"
    )
    segshade_win = _meta(
        "parameter", "Total winter vegetation shade fraction [-]"
    )

    def _compute_shade(self, declination: float, summer_flag: int) -> None:
        obj = self._obj
        if summer_flag == 1:
            obj["seg_shade"].values[:] = obj["segshade_sum"].values
        else:
            obj["seg_shade"].values[:] = obj["segshade_win"].values


class PRMSStreamTempSegHumidity(PRMSStreamTemp):
    """PRMSStreamTemp with humidity from the monthly ``seg_humidity``
    PARAMETER instead of an input (upstream PRMSStreamTemp,
    strmtemp_humidity_flag = 1; the drb seg_humid_matrix / _scalar
    configurations): ``seg_humid`` is overridden input -> variable and
    assigned from the parameter each step before the kernels (exactly
    where upstream's aggregation sets it)."""

    # declaration OVERRIDE: supplied input -> computed variable
    seg_humid = _meta(
        "variable",
        "Segment relative humidity [fraction] (= seg_humidity[month])",
    )

    seg_humidity = _meta(
        "parameter",
        "Monthly segment relative humidity [fraction]",
        dims=("nmonth", "space"),
    )

    def initialize(self) -> None:
        super().initialize()
        self._obj["seg_humid"].values[:] = 0.0

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        # upstream flag==1 path: Seg_humid(i) = Seg_humidity(i, Nowmonth)
        obj["seg_humid"].values[:] = obj["seg_humidity"].values[
            time.month - 1, :
        ]
        super().calculate(dt, time)


# ----------------------------------------------------------------------
# hru -> segment aggregation -- upstream verbatim (the chain seam;
# also the A/B pin target). Includes the seg_close fallback
# machinery for segments without HRUs.
# ----------------------------------------------------------------------


@nb.jit(nopython=True)
def _compute_seg_humid_cbh_numba(
    nhru,
    nsegment,
    hru_segment,
    hru_area,
    humidity_hru,
    segment_hruarea,
    segment_order,
    seg_close,
    seg_humid,
):
    """Compute seg_humid from CBH humidity_hru input.

    Implements the flag==0 path from PRMS 5.2.1.1 stream_temp.f90:
    - Reset seg_humid to 0 (ELSE branch, reinstated fix)
    - Accumulate area-weighted humidity from HRUs, converting percent->fraction
    - Normalise by segment HRU area in a single segment_order pass, copying
      from seg_close for segments with no HRUs (matching PRMS single-pass).

    Args:
        nhru: Number of HRUs (immutable)
        nsegment: Number of segments (immutable)
        hru_segment: HRU to segment mapping (immutable, 1-based)
        hru_area: HRU areas in acres (immutable)
        humidity_hru: HRU relative humidity in percent 0-100 (immutable)
        segment_hruarea: Total HRU area per segment (immutable)
        segment_order: Order to process segments (immutable)
        seg_close: Closest segment with HRUs for each segment (immutable)
        seg_humid: Segment humidity in decimal fraction (MUTATED - output)
    """
    # Reset before accumulation — matches corrected PRMS 5.2.1.1 ELSE branch.
    # See humidity_bug_prms_fix.md "Secondary Bug" for full rationale.
    seg_humid[:] = 0.0

    # Accumulate area-weighted humidity from HRUs.
    # *0.01 applied during accumulation (percent -> decimal fraction) so units
    # are consistent throughout (stream_temp.f90 line 807):
    #   Seg_humid(i) = Seg_humid(i) + Humidity_hru(j) * 0.01 * harea
    for j in range(nhru):
        seg_idx = hru_segment[j]
        if seg_idx > 0:
            i = seg_idx - 1
            seg_humid[i] += humidity_hru[j] * 0.01 * hru_area[j]

    # Normalise in a single segment_order pass (matches PRMS single-pass).
    # For no-HRU segments, copy from seg_close in the same pass so that the
    # ordering matches PRMS exactly (stream_temp.f90 lines 829-844).
    for jj in range(nsegment):
        i = segment_order[jj]
        if segment_hruarea[i] > NEARZERO:
            seg_humid[i] /= segment_hruarea[i]
        else:
            seg_humid[i] = seg_humid[seg_close[i]]


@nb.jit(nopython=True)
def _compute_segment_aggregates_numba(
    nhru,
    nsegment,
    hru_segment,
    hru_area,
    sroff,
    ssres_flow,
    gwres_flow,
    swrad,
    segment_hruarea,
    segment_up,
    tosegment,
    seginc_sroff,
    seginc_ssflow,
    seginc_gwflow,
    seginc_swrad,
    # Meteorological aggregation inputs (excluding humidity)
    tavgc,
    snowmelt,
    hru_rain,
    soltab_potsw,
    hru_cossl,
    segment_order,
    seg_close,
    # Output arrays for meteorological variables (excluding seg_humid)
    seg_tave_air,
    seg_melt,
    seg_rain,
    seg_ccov,
):
    """Compute segment aggregate variables from HRU inputs.

    Humidity (seg_humid) is handled separately by _compute_seg_humid_cbh_numba
    (flag==0) or set directly from a parameter (flag==1).

    Args:
        nhru: Number of HRUs (immutable)
        nsegment: Number of segments (immutable)
        hru_segment: HRU to segment mapping (immutable, 1-based)
        hru_area: HRU areas (immutable)
        sroff: Surface runoff from HRUs (immutable)
        ssres_flow: Subsurface flow from HRUs (immutable)
        gwres_flow: Groundwater flow from HRUs (immutable)
        swrad: Solar radiation from HRUs (immutable)
        segment_hruarea: Total HRU area per segment (immutable)
        segment_up: Upstream segment indices (immutable, 0-based)
        tosegment: Downstream segment indices (immutable, 1-based)
        seginc_sroff: Segment surface runoff (MUTATED - output)
        seginc_ssflow: Segment subsurface flow (MUTATED - output)
        seginc_gwflow: Segment groundwater flow (MUTATED - output)
        seginc_swrad: Segment solar radiation (MUTATED - output)
        tavgc: HRU average temperature in Celsius (immutable)
        snowmelt: HRU snowmelt (immutable)
        hru_rain: HRU rainfall (immutable)
        soltab_potsw: Potential shortwave radiation for current day (immutable)
        hru_cossl: Cosine of HRU slope (immutable)
        segment_order: Order to process segments (immutable)
        seg_close: Closest segment with HRUs for each segment (immutable)
        seg_tave_air: Segment air temperature (MUTATED - output)
        seg_melt: Segment snowmelt (MUTATED - output)
        seg_rain: Segment rainfall (MUTATED - output)
        seg_ccov: Segment cloud cover (MUTATED - output)
    """
    # Initialize segment aggregate variables
    seginc_sroff[:] = 0.0
    seginc_ssflow[:] = 0.0
    seginc_gwflow[:] = 0.0
    seginc_swrad[:] = 0.0

    # Initialize meteorological segment variables
    seg_tave_air[:] = 0.0
    seg_melt[:] = 0.0
    seg_rain[:] = 0.0
    seg_ccov[:] = 0.0

    # Constants (from PRMS_SET_TIME)
    # Cfs_conv converts acre-inches/day to cfs
    # FT2_PER_ACRE / INCHES_PER_FOOT / SECS_PER_DAY
    # = 43560 / 12 / 86400
    cfs_conv = 43560.0 / 12.0 / 86400.0

    # Aggregate HRU values to segments
    for j in range(nhru):
        seg_idx = hru_segment[j]

        # Check if HRU contributes to a segment (seg_idx > 0)
        # hru_segment is 1-based, so valid segments are > 0
        if seg_idx > 0:
            # Convert to 0-based index
            i = seg_idx - 1
            harea = hru_area[j]

            # Convert from inches to cfs (area * inches/day * cfs_conv)
            tocfs = harea * cfs_conv

            # Accumulate flow components (converted to cfs)
            seginc_sroff[i] += sroff[j] * tocfs
            seginc_ssflow[i] += ssres_flow[j] * tocfs
            seginc_gwflow[i] += gwres_flow[j] * tocfs

            # Accumulate area-weighted radiation
            seginc_swrad[i] += swrad[j] * harea

            # Compute cloud cover for this HRU (stream_temp.f90 line 760-778)
            # ccov = 1.0 - (swrad / soltab_potsw * hru_cossl)
            potsw = soltab_potsw[j]
            if potsw <= 10.0:
                ccov = 1.0 - (swrad[j] / 10.0 * hru_cossl[j])
            else:
                ccov = 1.0 - (swrad[j] / potsw * hru_cossl[j])

            # Clamp ccov to [0, 1]
            if ccov < NEARZERO:
                ccov = 0.0
            elif ccov > 1.0:
                ccov = 1.0

            # Accumulate area-weighted meteorological variables
            seg_tave_air[i] += tavgc[j] * harea
            seg_ccov[i] += ccov * harea
            seg_melt[i] += snowmelt[j] * harea
            seg_rain[i] += hru_rain[j] * harea

    # Process seginc_swrad in numerical order first (matches original logic)
    # Divide radiation by segment HRU area to get averages
    # Process in numerical order to match routing.f90 (line 741-810)
    for i in range(nsegment):
        if segment_hruarea[i] > NEARZERO:
            seginc_swrad[i] /= segment_hruarea[i]

        else:
            # Segment has no HRUs - search upstream then downstream
            # (matches routing.f90 line 746-805)
            # Search upstream first (routing.f90 line 749-772)
            this_seg = i
            found = False
            max_iter = nsegment  # Prevent infinite loops
            iter_count = 0

            while not found and iter_count < max_iter:
                iter_count += 1

                # segment_up contains 0-based indices where 0 means
                # "no upstream". However, if segment 0 is upstream, we can't
                # distinguish it from "no upstream". The original code assumes
                # segment 0 is never upstream of anything
                upstream_seg = segment_up[this_seg]

                # Check if headwater (no upstream)
                if upstream_seg == 0 and this_seg != 0:
                    # No upstream segment exists
                    break
                elif upstream_seg == 0 and this_seg == 0:
                    # this_seg is segment 0, which has no upstream
                    break

                # Move to upstream segment (already 0-based)
                this_seg = upstream_seg

                if segment_hruarea[this_seg] > NEARZERO:
                    # Found segment with HRUs - compute average from
                    # accumulated value
                    seginc_swrad[i] = (
                        seginc_swrad[this_seg] / segment_hruarea[this_seg]
                    )
                    found = True
                    break

            # If not found upstream, search downstream
            # (routing.f90 line 776-800)
            if not found:
                this_seg = i
                iter_count = 0

                while not found and iter_count < max_iter:
                    iter_count += 1

                    # tosegment is 1-based, 0 means no downstream
                    downstream_seg = tosegment[this_seg]

                    # Check if terminal segment (no downstream)
                    if downstream_seg == 0:
                        break

                    # Move to downstream segment (convert 1-based to 0-based)
                    this_seg = downstream_seg - 1

                    if segment_hruarea[this_seg] > NEARZERO:
                        # Found segment with HRUs - compute average from
                        # accumulated value
                        seginc_swrad[i] = (
                            seginc_swrad[this_seg] / segment_hruarea[this_seg]
                        )
                        found = True
                        break

            # If still not found, set to invalid marker
            # (routing.f90 line 803-805)
            if not found:
                seginc_swrad[i] = -99.9

    # Process meteorological variables in single pass (matches Fortran).
    # Process in segment_order - for segments without HRUs, copy from
    # seg_close. Note: seg_close may point to upstream/downstream segments
    # not yet processed in some edge cases, but this matches the Fortran
    # single-pass behavior (stream_temp.f90 lines 820-844).
    for jj in range(nsegment):
        i = segment_order[jj]

        if segment_hruarea[i] > NEARZERO:
            # Segment has HRUs - compute area-weighted averages
            seg_tave_air[i] /= segment_hruarea[i]
            seg_ccov[i] /= segment_hruarea[i]
            seg_melt[i] /= segment_hruarea[i]
            seg_rain[i] /= segment_hruarea[i]
        else:
            # Segment has no HRUs - use values from seg_close
            # (stream_temp.f90 lines 832-844)
            close_seg = seg_close[i]
            seg_tave_air[i] = seg_tave_air[close_seg]
            seg_ccov[i] = seg_ccov[close_seg]
            seg_melt[i] = seg_melt[close_seg]
            seg_rain[i] = seg_rain[close_seg]


@nb.jit(nopython=True)
def _compute_seg_potet_numba(
    nhru,
    nsegment,
    hru_segment,
    hru_area,
    potet,
    segment_order,
    segment_hruarea,
    seg_close,
    seg_potet,
):
    """Compute seg_potet using stream_temp.f90 logic.

    This numba-optimized function replaces _compute_seg_potet.

    Args:
        nhru: Number of HRUs (immutable)
        nsegment: Number of segments (immutable)
        hru_segment: HRU to segment mapping (immutable, 1-based)
        hru_area: HRU areas (immutable)
        potet: Potential ET from HRUs (immutable)
        segment_order: Order to process segments (immutable)
        segment_hruarea: Total HRU area per segment (immutable)
        seg_close: Closest segment with HRUs for each segment (immutable)
        seg_potet: Segment potential ET (MUTATED - output)
    """
    # Initialize
    seg_potet[:] = 0.0

    # Accumulate from HRUs
    for j in range(nhru):
        seg_idx = hru_segment[j]
        if seg_idx > 0:
            i = seg_idx - 1
            seg_potet[i] += potet[j] * hru_area[j]

    # Process in segment_order
    for jj in range(nsegment):
        i = segment_order[jj]

        if segment_hruarea[i] > NEARZERO:
            seg_potet[i] /= segment_hruarea[i]

        else:
            # Segment has no HRUs - use seg_close
            close_seg = seg_close[i]
            seg_potet[i] = seg_potet[close_seg]


def resolve_aggregation_topology(
    hru_segment: np.ndarray,
    hru_area: np.ndarray,
    tosegment: np.ndarray,
    segment_order: np.ndarray,
    seg_close_param: np.ndarray,
) -> dict:
    """The static topology the hru->segment aggregation needs
    (upstream _initialize_stream_temp, verbatim logic): per-segment
    contributing HRU area; segment_up (the LAST 1-based-scan upstream
    segment); and the resolved seg_close (parameter -1 = auto:
    segment_up, with the previous-segment-in-route-order fallback for
    no-HRU segments without an upstream -- NOTE that fallback depends
    on the specific segment_order, so a no-HRU headwater segment can
    differ between orderings; drb has exactly one such segment).

    Init-time numpy staging; returns plain arrays for the njit
    aggregation kernels.
    """
    nsegment = tosegment.shape[0]

    segment_hruarea = np.zeros(nsegment, dtype=np.float64)
    for jhru in range(hru_segment.shape[0]):
        seg_idx = hru_segment[jhru]
        if seg_idx > 0:
            segment_hruarea[seg_idx - 1] += hru_area[jhru]

    segment_up = np.zeros(nsegment, dtype=np.int32)
    for jseg in range(nsegment):
        toseg = tosegment[jseg]
        if toseg > 0:
            segment_up[toseg - 1] = jseg

    if seg_close_param[0] == -1:
        seg_close = np.copy(segment_up)
        for jj in range(nsegment):
            iseg = segment_order[jj]
            if segment_hruarea[iseg] <= NEARZERO:
                if segment_up[iseg] == 0:
                    if jj > 0:
                        seg_close[iseg] = segment_order[jj - 1]
                    else:
                        raise ValueError(
                            "Cannot set associated segment for segment "
                            f"without associated HRU for segment {iseg}."
                            " Must specify seg_close for this case."
                        )
    else:
        errors = []
        for iseg in range(nsegment):
            if segment_hruarea[iseg] <= NEARZERO:
                if seg_close_param[iseg] == 0:
                    errors.append(
                        f"segment {iseg} does not have associated HRUs "
                        "but seg_close is 0"
                    )
        if errors:
            raise ValueError("\n".join(errors))
        seg_close = np.asarray(seg_close_param, dtype=np.int32)

    return {
        "segment_hruarea": segment_hruarea,
        "segment_up": segment_up,
        "seg_close": seg_close.astype(np.int32),
    }


# The ten aggregation Maps: target (segment) variable ->
# (source hru variable, key into derive_aggregation_weights()'s
# result). ccov_hru is PRMSAtmosphere's relocated cloud-cover variable
# (Maps never ORIGINATE variables -- the chain-stage design decision);
# humidity_hru is the CBH forcing (percent; the 0.01 percent->fraction
# factor is folded into the "humid" weights), so the CORE
# PRMSStreamTemp + this Map IS the strmtemp_humidity_flag=0 (CBH)
# configuration -- no separate leaf.
AGGREGATION_MAP_SPEC = {
    "seginc_sroff": ("sroff", "flow"),
    "seginc_ssflow": ("ssres_flow", "flow"),
    "seginc_gwflow": ("gwres_flow", "flow"),
    "seginc_swrad": ("swrad", "swrad"),
    "seg_tave_air": ("tavgc", "met"),
    "seg_ccov": ("ccov_hru", "met"),
    "seg_melt": ("snowmelt", "met"),
    "seg_rain": ("hru_rain", "met"),
    "seg_potet": ("potet", "met"),
    "seg_humid": ("humidity_hru", "humid"),
}


def derive_aggregation_weights(
    hru_segment: np.ndarray,
    hru_area: np.ndarray,
    tosegment: np.ndarray,
    segment_order: np.ndarray,
    seg_close_param: np.ndarray,
) -> dict:
    """The static (nsegment, nhru) hru->segment aggregation weight
    matrices, derived by PROBING the verbatim kernels with basis
    vectors -- the validated kernels stay the single source of truth
    (nothing re-implemented by hand; in particular the order-dependent
    seginc_swrad fallback rows, where the numerical-order
    normalization pass can read an already-normalized value, are
    captured exactly as the kernel computes them).

    Valid because, with cloud cover relocated to PRMSAtmosphere
    (ccov_hru), every aggregate is exactly LINEAR in its hru input
    with coefficients fixed by static topology/areas: plain weighted
    sums (flow: area * cfs_conv), area-averages with seg_close copy
    rows (met), the percent->fraction factor (humid = 0.01 * met), and
    the swrad search fallback. Returns
    {"flow", "swrad", "met", "humid"} -> dense weights for the Map
    machinery (see AGGREGATION_MAP_SPEC for the variable wiring).

    Raises NotImplementedError if any zero-input probe is nonzero
    (the -99.9 seginc_swrad marker: a segment with no contributing
    HRUs anywhere upstream or downstream) -- that needs an affine Map,
    unimplemented until a domain needs it (drb has none). Derived at
    build; a file-backed option belongs to the future PRMS
    pre-processing suite (see PORTS.md backlog).

    Cost: nhru + 1 kernel calls, one dense matrix per family --
    one-time build allocations (memory directive: justified).
    """
    nhru = hru_segment.shape[0]
    nsegment = tosegment.shape[0]
    topo = resolve_aggregation_topology(
        hru_segment, hru_area, tosegment, segment_order, seg_close_param
    )

    probe = np.zeros(nhru, dtype=np.float64)
    zeros_hru = np.zeros(nhru, dtype=np.float64)
    ones_hru = np.ones(nhru, dtype=np.float64)  # soltab/cossl stand-ins
    out_flow = np.zeros(nsegment, dtype=np.float64)
    out_unused1 = np.zeros(nsegment, dtype=np.float64)
    out_unused2 = np.zeros(nsegment, dtype=np.float64)
    out_swrad = np.zeros(nsegment, dtype=np.float64)
    out_tave = np.zeros(nsegment, dtype=np.float64)
    out_melt = np.zeros(nsegment, dtype=np.float64)
    out_rain = np.zeros(nsegment, dtype=np.float64)
    out_ccov = np.zeros(nsegment, dtype=np.float64)
    out_met = np.zeros(nsegment, dtype=np.float64)
    out_humid = np.zeros(nsegment, dtype=np.float64)

    def _probe_kernels(hru_values: np.ndarray) -> None:
        # flow + swrad share one combined-kernel call (independent
        # outputs); the in-kernel ccov path is fed neutral stand-ins
        # and its outputs ignored (ccov aggregates via "met")
        _compute_segment_aggregates_numba(
            nhru,
            nsegment,
            hru_segment,
            hru_area,
            hru_values,  # sroff
            zeros_hru,
            zeros_hru,
            hru_values,  # swrad
            topo["segment_hruarea"],
            topo["segment_up"],
            tosegment,
            out_flow,
            out_unused1,
            out_unused2,
            out_swrad,
            zeros_hru,  # tavgc
            zeros_hru,  # snowmelt
            zeros_hru,  # hru_rain
            ones_hru,  # soltab_potsw row (ccov path only)
            ones_hru,  # hru_cossl (ccov path only)
            segment_order,
            topo["seg_close"],
            out_tave,
            out_melt,
            out_rain,
            out_ccov,
        )
        _compute_seg_potet_numba(
            nhru,
            nsegment,
            hru_segment,
            hru_area,
            hru_values,
            segment_order,
            topo["segment_hruarea"],
            topo["seg_close"],
            out_met,
        )
        _compute_seg_humid_cbh_numba(
            nhru,
            nsegment,
            hru_segment,
            hru_area,
            hru_values,
            topo["segment_hruarea"],
            segment_order,
            topo["seg_close"],
            out_humid,
        )

    # zero-input probe: every family's constant must be zero (a
    # nonzero seginc_swrad entry is the -99.9 marker)
    _probe_kernels(probe)
    for name, out in (
        ("flow", out_flow),
        ("swrad", out_swrad),
        ("met", out_met),
        ("humid", out_humid),
    ):
        if np.any(out != 0.0):
            raise NotImplementedError(
                f"aggregation family '{name}' has nonzero zero-input "
                f"constants at segment(s) {np.where(out != 0.0)[0]} "
                "(no contributing HRUs anywhere upstream or "
                "downstream -> the -99.9 marker); an affine Map is "
                "not implemented. Handle these segments explicitly."
            )

    weights = {
        kk: np.zeros((nsegment, nhru), dtype=np.float64)
        for kk in ("flow", "swrad", "met", "humid")
    }
    for jj in range(nhru):
        probe[jj] = 1.0
        _probe_kernels(probe)
        weights["flow"][:, jj] = out_flow
        weights["swrad"][:, jj] = out_swrad
        weights["met"][:, jj] = out_met
        weights["humid"][:, jj] = out_humid
        probe[jj] = 0.0

    return weights
