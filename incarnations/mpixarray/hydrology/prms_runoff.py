"""
hydrology/prms_runoff.py
========================
PRMSRunoffNoDprst + PRMSRunoff: PRMS surface runoff, ported from
pywatershed (pywatershed/hydrology/prms_runoff.py and
prms_runoff_no_dprst.py; PRMS 5.2.1 physics, PRMS-IV documentation:
Markstrom et al. 2015, USGS TM 6-B7).

Third REAL process port (July 2026) -- the process that produces
``sroff_vol``, replacing the disk-fed carrier flux in the gw->channel
submodel. Ported: field declarations (names verbatim) and the numerics
of ``_calculate_numpy`` + its helper functions, rewritten to this
framework's in-place, out-first, zero-per-step-allocation kernel
convention. The helpers (``compute_infil``/``perv_comp``/
``check_capacity``/``dprst_comp``/``imperv_et``) stay separate njit
functions with pywatershed's exact signatures and bodies (scalar
in/out; keyword calls verbatim) -- per-element operation order is
identical, so results match pywatershed's answers.

**Variant structure (ADDITIVE -- see PORTS.md "How variants are done
here")**: pywatershed derives PRMSRunoffNoDprst FROM PRMSRunoff by
subtraction (re-declared interface, double-run basin_init, ~25 per-step
zero arrays fed to the shared kernel). Here the hierarchy points the
right way: ``PRMSRunoffNoDprst`` is the minimal core (pervious/
impervious runoff + infiltration partitioning, its own kernel over the
shared njit helpers) and ``PRMSRunoff`` EXTENDS it: dprst parameters/
derived/variables ADDED, ``initialize``/``advance``/kernel overridden.
The core never touches dprst.

Initialization -> framework seams (the Stage-2 mechanisms):

- **basin_init/dprst_init -> ``initialize()`` + ``parameter_internal``.**
  pywatershed computes static geometry (``hru_perv``/``hru_frac_perv``/
  ``hru_imperv``/``dprst_area_max``/``dprst_area_open_max``/
  ``dprst_area_clos_max``/``dprst_frac_clos``/``dprst_vol_open_max``/
  ``dprst_vol_clos_max``/``dprst_vol_thres_open``) at construction;
  here they are ``parameter_internal``: written once in ``initialize()``,
  then frozen. ``dprst_vol_thres_open`` is nominally a pywatershed
  "variable" but is never written by its kernel (its own autotest
  excludes it from comparison) -- derived here.
- **dprst initial state** (``dprst_vol_open``/``dprst_vol_clos``/
  ``dprst_area_open``/``dprst_area_clos``/``dprst_stor_hru`` + olds/
  fracs) is computed in ``initialize()`` (dprst_init verbatim, the
  no-restart path).

Deliberately NOT ported (conventions in pws_phoenix/CLAUDE.md):

- Budget / ConservativeProcess (backlogged); adapters; restart;
  calc_method switch (numba is THE path); verbose;
  ``imbalance_behavior``.
- The ``_dprst_clos_flag == OFF`` parameter-zeroing hack in dprst_init
  (hit by nhm domains: ``dprst_frac_open == 1`` everywhere, drb
  included) -- provably a no-op, see the comment in ``initialize()``.
- Hardwired-off upstream paths, kept hardwired: cascades, frozen
  ground, glaciers, water use.
- ``intcp_changeover_in_net_rain`` (GSFLOW/PRMS-6 accounting): fixed
  False (pywatershed default; module constant, compile-time).

Kernel quirk preserved deliberately: ``dprst_comp`` recomputes a LOCAL
``dprst_area_clos`` (used for closed-depression evap) but pywatershed
never assigns it back -- the ``dprst_area_clos`` array keeps its
dprst_init value for the whole run. Its answer file confirms (static
time series); do not "fix" this here.

Parameter provenance (utils/separate_nhm_params.py): ``hru_type``,
``hru_area``, ``hru_in_to_cf`` are DIS_HRU variables (dis-first
sourcing via parameters_dis_hru.nc); the 7 core process parameters
live in parameters_PRMSRunoffNoDprst.nc and the full 20 in
parameters_PRMSRunoff.nc.
"""

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process

# pywatershed constants (constants.py)
_NEARZERO = 1.0e-6  # nearzero
# dnearzero = epsilon64: pywatershed HARDCODES 2.23e-16 (slightly
# above np.finfo(float64).eps) -- verbatim, threshold branches differ
_DNEARZERO = 2.23e-16
_LAND = 1  # HruType.LAND.value
# GSFLOW 4.2.0 / PRMS 6 accounting; pywatershed default False. A
# RUNTIME argument of compute_infil / compute_infil_ag (as upstream):
# the non-ag classes pass this False constant; PRMSRunoffAg passes its
# class attribute (True -- fgr_ag_2yr answers are GSFLOW).
_INTCP_CHANGEOVER_IN_NET_RAIN = False


# ----------------------------------------------------------------------
# Kernel helper functions -- pywatershed staticmethods verbatim
# (scalar in/out; separate njit functions exactly as upstream njits
# them; called with keyword arguments below, also verbatim)
# ----------------------------------------------------------------------


@numba.njit
def perv_comp(
    soil_moist_prev,
    carea_max,
    smidx_coef,
    smidx_exp,
    pptp,
    ptc,
    infil,
    srp,
):
    """Pervious area computations (smidx module)."""
    smidx = soil_moist_prev + 0.5 * ptc
    if smidx > 25.0:
        ca_fraction = carea_max
    else:
        ca_fraction = smidx_coef * 10.0 ** (smidx_exp * smidx)

    if ca_fraction > carea_max:
        ca_fraction = carea_max

    srpp = ca_fraction * pptp
    infil = infil - srpp
    srp = srp + srpp

    return infil, srp, ca_fraction


@numba.njit
def check_capacity(soil_moist_prev, soil_moist_max, snowinfil_max, infil, srp):
    """Fill soil to soil_moist_max; if more than capacity restrict
    infiltration by snowinfil_max, with excess added to runoff."""
    capacity = soil_moist_max - soil_moist_prev
    excess = infil - capacity
    if excess > snowinfil_max:
        srp = srp + excess - snowinfil_max
        infil = snowinfil_max + capacity

    return infil, srp


@numba.njit
def imperv_et(imperv_stor, potet, imperv_evap, sca, avail_et, imperv_frac):
    if sca < 1.0:
        if potet < imperv_stor:
            imperv_evap = potet * (1.0 - sca)
        else:
            imperv_evap = imperv_stor * (1.0 - sca)
        if imperv_evap * imperv_frac > avail_et:
            imperv_evap = avail_et / imperv_frac
        imperv_stor = imperv_stor - imperv_evap
    return imperv_stor, imperv_evap


@numba.njit
def compute_infil(
    contrib_fraction,
    soil_moist_prev,
    soil_moist_max,
    carea_max,
    smidx_coef,
    smidx_exp,
    pptmix_nopack,
    net_rain,
    net_ppt,
    imperv_stor,
    imperv_stor_max,
    snowmelt,
    snowinfil_max,
    net_snow,
    pkwater_equiv,
    infil,
    hru_type,
    intcp_changeover,
    hruarea_imperv,
    sri,
    srp,
    through_rain,
    intcp_changeover_in_net_rain,
):
    isglacier = False  # hardwired upstream
    hru_flag = 0
    if hru_type == _LAND or isglacier:
        hru_flag = 1

    avail_water = 0.0

    # compute runoff from canopy changeover water
    if intcp_changeover > 0.0 and not intcp_changeover_in_net_rain:
        avail_water = avail_water + intcp_changeover
        infil = infil + intcp_changeover
        if hru_flag == 1:
            infil, srp, contrib_fraction = perv_comp(
                soil_moist_prev,
                carea_max,
                smidx_coef,
                smidx_exp,
                intcp_changeover,
                intcp_changeover,
                infil,
                srp,
            )

    # if rain/snow event with no antecedent snowpack, compute the
    # runoff from the rain first, then proceed with snowmelt
    cond2 = pptmix_nopack != 0
    if cond2:
        avail_water = avail_water + through_rain
        infil = infil + through_rain
        if hru_flag == 1:
            infil, srp, contrib_fraction = perv_comp(
                soil_moist_prev,
                carea_max,
                smidx_coef,
                smidx_exp,
                through_rain,
                through_rain,
                infil,
                srp,
            )

    # If precipitation on snowpack, all water available to the surface
    # is considered to be snowmelt; use the snowmelt infiltration
    # procedure. If no snowpack and no precip, check for melt from
    # last of snowpack. If rain/snow mix with no antecedent snowpack,
    # compute the snowmelt portion of runoff.
    cond4 = pkwater_equiv < _DNEARZERO
    cond6 = net_snow < _NEARZERO

    if snowmelt > 0.0:
        avail_water = avail_water + snowmelt
        infil = infil + snowmelt
        if hru_flag == 1:
            # rain-on-snow check: the presence-of-rain test depends on
            # whether intcp_changeover is in net_rain (GSFLOW/PRMS-6)
            # or not (pywatershed / PRMS < 6)
            if intcp_changeover_in_net_rain:
                check_condition = not (net_ppt - net_snow > 0.0)
            else:
                check_condition = net_rain < _NEARZERO
            if (pkwater_equiv > 0.0) or check_condition:
                # pervious area computations
                infil, srp = check_capacity(
                    soil_moist_prev,
                    soil_moist_max,
                    snowinfil_max,
                    infil,
                    srp,
                )
            else:
                # snowmelt occurred and depleted the snowpack
                infil, srp, contrib_fraction = perv_comp(
                    soil_moist_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    snowmelt,
                    net_ppt,
                    infil,
                    srp,
                )

    elif cond4:
        # No snowmelt and no snowpack, but if there was net snow the
        # snowpack was small and lost to sublimation.
        if cond6 and through_rain > 0.0:
            avail_water = avail_water + through_rain
            infil = infil + through_rain
            if hru_flag == 1:
                infil, srp, contrib_fraction = perv_comp(
                    soil_moist_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    through_rain,
                    through_rain,
                    infil,
                    srp,
                )

    # Snowpack exists; check whether infil exceeds the maximum daily
    # snowmelt infiltration rate (infil from rain/snow mix on a
    # snowfree surface).
    elif infil > 0.0:
        if hru_flag == 1:
            infil, srp = check_capacity(
                soil_moist_prev,
                soil_moist_max,
                snowinfil_max,
                infil,
                srp,
            )

    if hruarea_imperv > 0.0:
        imperv_stor = imperv_stor + avail_water
        if hru_flag == 1:
            if imperv_stor > imperv_stor_max:
                sri = imperv_stor - imperv_stor_max
                imperv_stor = imperv_stor_max

    return sri, srp, imperv_stor, infil, contrib_fraction


@numba.njit
def dprst_comp(
    dprst_vol_clos,
    dprst_area_clos_max,
    dprst_area_clos,
    dprst_vol_open_max,
    dprst_vol_open,
    dprst_area_open_max,
    dprst_sroff_hru,
    sro_to_dprst_perv,
    sro_to_dprst_imperv,
    dprst_evap_hru,
    pptmix_nopack,
    snowmelt,
    pkwater_equiv,
    net_snow,
    hru_area,
    dprst_insroff_hru,
    dprst_frac_open,
    dprst_frac_clos,
    va_open_exp,
    dprst_vol_clos_max,
    dprst_vol_clos_frac,
    va_clos_exp,
    potet,
    snowcov_area,
    dprst_et_coef,
    dprst_seep_rate_open,
    dprst_vol_thres_open,
    dprst_flow_coef,
    dprst_seep_rate_clos,
    avail_et,
    net_rain,
    dprst_in,
    srp,
    sri,
    imperv_frac,
    perv_frac,
):
    # cascades and dprst water use are hardwired OFF upstream
    inflow = 0.0
    if pptmix_nopack != 0:
        inflow = inflow + net_rain

    # If precipitation on snowpack, all water available to the surface
    # is considered snowmelt. If no snowpack and no precip, check for
    # melt from last of snowpack. If rain/snow mix with no antecedent
    # snowpack, compute the snowmelt portion of runoff.
    if snowmelt > 0.0:
        inflow = inflow + snowmelt
    # No snowmelt, but a snowpack may exist; if no snowpack, check for
    # rain on a snowfree HRU.
    elif pkwater_equiv < _DNEARZERO:
        # If no snowmelt and no snowpack but there was net snow, the
        # snowpack was small and lost to sublimation.
        if net_snow < _NEARZERO and net_rain > 0.0:
            inflow = inflow + net_rain

    dprst_in = 0.0
    if dprst_area_open_max > 0.0:
        dprst_in = inflow * dprst_area_open_max
        dprst_vol_open = dprst_vol_open + dprst_in

    if dprst_area_clos_max > 0.0:
        tmp1 = inflow * dprst_area_clos_max
        dprst_vol_clos = dprst_vol_clos + tmp1
        dprst_in = dprst_in + tmp1
    dprst_in = dprst_in / hru_area

    # add any pervious surface runoff fraction to depressions
    dprst_srp = 0.0
    dprst_sri = 0.0
    if srp > 0.0:
        tmp = srp * perv_frac * sro_to_dprst_perv * hru_area
        if dprst_area_open_max > 0.0:
            dprst_srp_open = tmp * dprst_frac_open
            dprst_srp = dprst_srp_open / hru_area
            dprst_vol_open = dprst_vol_open + dprst_srp_open
        if dprst_area_clos_max > 0.0:
            dprst_srp_clos = tmp * dprst_frac_clos
            dprst_srp = dprst_srp + dprst_srp_clos / hru_area
            dprst_vol_clos = dprst_vol_clos + dprst_srp_clos
        srp = srp - dprst_srp / perv_frac
        if srp < 0.0:
            srp = 0.0

    if sri > 0.0:
        tmp = sri * imperv_frac * sro_to_dprst_imperv * hru_area
        if dprst_area_open_max > 0.0:
            dprst_sri_open = tmp * dprst_frac_open
            dprst_sri = dprst_sri_open / hru_area
            dprst_vol_open = dprst_vol_open + dprst_sri_open
        if dprst_area_clos_max > 0.0:
            dprst_sri_clos = tmp * dprst_frac_clos
            dprst_sri = dprst_sri + dprst_sri_clos / hru_area
            dprst_vol_clos = dprst_vol_clos + dprst_sri_clos
        sri = sri - dprst_sri / imperv_frac
        if sri < 0.0:
            sri = 0.0

    dprst_insroff_hru = dprst_srp + dprst_sri

    dprst_area_open = 0.0
    if dprst_vol_open > 0.0:
        open_vol_r = dprst_vol_open / dprst_vol_open_max
        if open_vol_r < _NEARZERO:
            frac_op_ar = 0.0
        elif open_vol_r > 1.0:
            frac_op_ar = 1.0
        else:
            frac_op_ar = np.exp(va_open_exp * np.log(open_vol_r))
        dprst_area_open = dprst_area_open_max * frac_op_ar
        if dprst_area_open > dprst_area_open_max:
            dprst_area_open = dprst_area_open_max

    # NOTE: this recomputed dprst_area_clos is LOCAL (used for evap
    # below); pywatershed never assigns it back to the array
    if dprst_area_clos_max > 0.0:
        dprst_area_clos = 0.0
        if dprst_vol_clos > 0.0:
            clos_vol_r = dprst_vol_clos / dprst_vol_clos_max
            if clos_vol_r < _NEARZERO:
                frac_cl_ar = 0.0
            elif clos_vol_r > 1.0:
                frac_cl_ar = 1.0
            else:
                frac_cl_ar = np.exp(va_clos_exp * np.log(clos_vol_r))
            dprst_area_clos = dprst_area_clos_max * frac_cl_ar
            if dprst_area_clos > dprst_area_clos_max:
                dprst_area_clos = dprst_area_clos_max

    # evaporate water from depressions based on snowcov_area
    # dprst_evap_open & dprst_evap_clos = inches-acres on the HRU
    unsatisfied_et = avail_et
    dprst_avail_et = potet * (1.0 - snowcov_area) * dprst_et_coef
    dprst_evap_hru = 0.0
    if dprst_avail_et > 0.0:
        dprst_evap_open = 0.0
        dprst_evap_clos = 0.0
        if dprst_area_open > 0.0:
            dprst_evap_open = min(
                dprst_area_open * dprst_avail_et, dprst_vol_open
            )
            if dprst_evap_open / hru_area > unsatisfied_et:
                dprst_evap_open = unsatisfied_et * hru_area
            if dprst_evap_open > dprst_vol_open:
                dprst_evap_open = dprst_vol_open
            unsatisfied_et = unsatisfied_et - dprst_evap_open / hru_area
            dprst_vol_open = dprst_vol_open - dprst_evap_open

        if dprst_area_clos > 0.0:
            dprst_evap_clos = min(
                dprst_area_clos * dprst_avail_et, dprst_vol_clos
            )
            if dprst_evap_clos / hru_area > unsatisfied_et:
                dprst_evap_clos = unsatisfied_et * hru_area
            if dprst_evap_clos > dprst_vol_clos:
                dprst_evap_clos = dprst_vol_clos
            dprst_vol_clos = dprst_vol_clos - dprst_evap_clos

        dprst_evap_hru = (dprst_evap_open + dprst_evap_clos) / hru_area

    # compute seepage
    dprst_seep_hru = 0.0
    if dprst_vol_open > 0.0:
        seep_open = dprst_vol_open * dprst_seep_rate_open
        dprst_vol_open = dprst_vol_open - seep_open
        if dprst_vol_open < 0.0:
            seep_open = seep_open + dprst_vol_open
            dprst_vol_open = 0.0
        dprst_seep_hru = seep_open / hru_area

    # compute open surface runoff
    dprst_sroff_hru = 0.0
    if dprst_vol_open > 0.0:
        dprst_sroff_hru = max(0.0, dprst_vol_open - dprst_vol_open_max)
        dprst_sroff_hru = dprst_sroff_hru + max(
            0.0,
            (dprst_vol_open - dprst_sroff_hru - dprst_vol_thres_open)
            * dprst_flow_coef,
        )
        dprst_vol_open = dprst_vol_open - dprst_sroff_hru
        dprst_sroff_hru = dprst_sroff_hru / hru_area
        if dprst_vol_open < 0.0:
            dprst_vol_open = 0.0

    if dprst_area_clos_max > 0.0:
        if dprst_area_clos > _NEARZERO:
            seep_clos = dprst_vol_clos * dprst_seep_rate_clos
            dprst_vol_clos = dprst_vol_clos - seep_clos
            if dprst_vol_clos < 0.0:
                seep_clos = seep_clos + dprst_vol_clos
                dprst_vol_clos = 0.0
            dprst_seep_hru = dprst_seep_hru + seep_clos / hru_area

    avail_et = avail_et - dprst_evap_hru
    # upstream leaves these two CONDITIONALLY defined (numba
    # zero-inits undefined-on-path locals; numpy mode would NameError
    # on a dprst HRU with vol_open_max == 0) -- explicit zeros here,
    # behavior-identical wherever upstream is well-defined
    dprst_vol_open_frac = 0.0
    dprst_vol_frac = 0.0
    if dprst_vol_open_max > 0.0:
        dprst_vol_open_frac = dprst_vol_open / dprst_vol_open_max
    if dprst_vol_clos_max > 0.0:
        dprst_vol_clos_frac = dprst_vol_clos / dprst_vol_clos_max
    if dprst_vol_open_max + dprst_vol_clos_max > 0.0:
        dprst_vol_frac = (dprst_vol_open + dprst_vol_clos) / (
            dprst_vol_open_max + dprst_vol_clos_max
        )
    dprst_stor_hru = (dprst_vol_open + dprst_vol_clos) / hru_area

    return (
        dprst_in,
        dprst_vol_open,
        dprst_area_open,
        avail_et,
        dprst_vol_clos,
        dprst_sroff_hru,
        srp,
        sri,
        dprst_evap_hru,
        dprst_seep_hru,
        dprst_insroff_hru,
        dprst_vol_open_frac,
        dprst_vol_clos_frac,
        dprst_vol_frac,
        dprst_stor_hru,
    )


# ----------------------------------------------------------------------
# Agricultural (ag) kernel helpers -- pywatershed PRMSRunoffAg
# staticmethods verbatim (prms_runoff_ag.py; scalar in/out; function
# args and the intcp flag dropped per port convention)
# ----------------------------------------------------------------------


@numba.njit
def ag_comp(
    ag_soil_moist_prev,
    ag_soil_rechr_prev,
    carea_max,
    smidx_coef,
    smidx_exp,
    pptp,
    ptc,
    infil_ag,
    sroff_ag,
):
    """Agricultural area runoff computations (perv_comp on ag soil
    moisture; smidx module hardwired upstream, carea unimplemented).
    ag_soil_rechr_prev is in upstream's signature but unused."""
    smidx = ag_soil_moist_prev + 0.5 * ptc
    if smidx > 25.0:
        ca_fraction = carea_max
    else:
        ca_fraction = smidx_coef * 10.0 ** (smidx_exp * smidx)

    if ca_fraction > carea_max:
        ca_fraction = carea_max

    srpp = ca_fraction * pptp
    infil_ag = infil_ag - srpp
    sroff_ag = sroff_ag + srpp

    return infil_ag, sroff_ag


@numba.njit
def check_capacity_ag(
    ag_soil_moist_prev,
    ag_soil_moist_max,
    snowinfil_max,
    infil_ag,
    sroff_ag,
):
    """Fill agricultural soil to ag_soil_moist_max; if more than
    capacity restrict infiltration by snowinfil_max, excess to runoff."""
    capacity = ag_soil_moist_max - ag_soil_moist_prev
    excess = infil_ag - capacity
    if excess > snowinfil_max:
        sroff_ag = sroff_ag + excess - snowinfil_max
        infil_ag = snowinfil_max + capacity

    return infil_ag, sroff_ag


@numba.njit
def compute_infil_ag(
    ag_soil_moist_prev,
    ag_soil_rechr_prev,
    ag_soil_moist_max,
    ag_soil_rechr_max,
    carea_max,
    smidx_coef,
    smidx_exp,
    snowinfil_max,
    pptmix_nopack,
    net_rain,
    net_ppt,
    snowmelt,
    net_snow,
    pkwater_equiv,
    hru_type,
    intcp_changeover,
    through_rain,
    intcp_changeover_in_net_rain,
):
    """Agricultural infiltration and runoff for a single HRU
    (upstream _compute_infil_ag; ag_soil_rechr_max is in its
    signature but unused; note its rain-on-snow check is HARDCODED
    net_rain < nearzero -- upstream's flag does not reach it)."""
    infil_ag = 0.0
    sroff_ag = 0.0

    # Process intcp_changeover
    if intcp_changeover > 0.0 and not intcp_changeover_in_net_rain:
        infil_ag = infil_ag + intcp_changeover
        if hru_type == _LAND:
            infil_ag, sroff_ag = ag_comp(
                ag_soil_moist_prev,
                ag_soil_rechr_prev,
                carea_max,
                smidx_coef,
                smidx_exp,
                intcp_changeover,
                intcp_changeover,
                infil_ag,
                sroff_ag,
            )

    # Process pptmix_nopack
    if pptmix_nopack != 0:
        infil_ag = infil_ag + through_rain
        if hru_type == _LAND:
            infil_ag, sroff_ag = ag_comp(
                ag_soil_moist_prev,
                ag_soil_rechr_prev,
                carea_max,
                smidx_coef,
                smidx_exp,
                through_rain,
                through_rain,
                infil_ag,
                sroff_ag,
            )

    # Process snowmelt
    if snowmelt > 0.0:
        infil_ag = infil_ag + snowmelt
        if hru_type == _LAND:
            if (pkwater_equiv > 0.0) or (net_rain < _NEARZERO):
                # Check capacity
                infil_ag, sroff_ag = check_capacity_ag(
                    ag_soil_moist_prev,
                    ag_soil_moist_max,
                    snowinfil_max,
                    infil_ag,
                    sroff_ag,
                )
            else:
                # Snowmelt occurred and depleted the snowpack
                infil_ag, sroff_ag = ag_comp(
                    ag_soil_moist_prev,
                    ag_soil_rechr_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    snowmelt,
                    net_ppt,
                    infil_ag,
                    sroff_ag,
                )

    elif pkwater_equiv < _DNEARZERO:
        # No snowpack
        if net_snow < _NEARZERO and through_rain > 0.0:
            infil_ag = infil_ag + through_rain
            if hru_type == _LAND:
                infil_ag, sroff_ag = ag_comp(
                    ag_soil_moist_prev,
                    ag_soil_rechr_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    through_rain,
                    through_rain,
                    infil_ag,
                    sroff_ag,
                )

    elif infil_ag > 0.0:
        # Snowpack exists, check capacity
        if hru_type == _LAND:
            infil_ag, sroff_ag = check_capacity_ag(
                ag_soil_moist_prev,
                ag_soil_moist_max,
                snowinfil_max,
                infil_ag,
                sroff_ag,
            )

    return infil_ag, sroff_ag


@numba.njit
def dprst_comp_ag(
    dprst_vol_clos,
    dprst_area_clos_max,
    dprst_area_clos,
    dprst_vol_open_max,
    dprst_vol_open,
    dprst_area_open_max,
    dprst_sroff_hru,
    sro_to_dprst_perv,
    sro_to_dprst_imperv,
    dprst_evap_hru,
    through_rain,
    snowmelt,
    hru_area,
    dprst_insroff_hru,
    dprst_frac_open,
    dprst_frac_clos,
    va_open_exp,
    dprst_vol_clos_max,
    dprst_vol_clos_frac,
    va_clos_exp,
    potet,
    snowcov_area,
    dprst_et_coef,
    dprst_seep_rate_open,
    dprst_vol_thres_open,
    dprst_flow_coef,
    dprst_seep_rate_clos,
    avail_et,
    dprst_in,
    srp,
    sri,
    sroff_ag,
    imperv_frac,
    perv_frac,
    ag_frac,
):
    """Depression storage with agricultural runoff routing (upstream
    dprst_comp_ag; srunoff.f90 lines 1687-1700 for the ag part).

    NOT the same code as dprst_comp with ag added -- upstream rewrote
    it with real differences, all preserved verbatim here:
    - inflow = through_rain + snowmelt UNCONDITIONALLY (no
      pptmix_nopack / no-snowpack rain logic, no net_rain/availh2o);
    - avail_et is NOT reduced by dprst_evap_hru before return;
    - closed-depression seepage is guarded by dprst_vol_clos > 0
      (not dprst_area_clos > nearzero);
    - the duplicated dprst_evap_open min() line is upstream's.
    """
    inflow = through_rain + snowmelt

    dprst_in = 0.0
    if dprst_area_open_max > 0.0:
        dprst_in = inflow * dprst_area_open_max
        dprst_vol_open = dprst_vol_open + dprst_in

    if dprst_area_clos_max > 0.0:
        tmp1 = inflow * dprst_area_clos_max
        dprst_vol_clos = dprst_vol_clos + tmp1
        dprst_in = dprst_in + tmp1
    dprst_in = dprst_in / hru_area

    # Add pervious surface runoff fraction to depressions
    dprst_srp = 0.0
    dprst_sri = 0.0
    dprst_sra = 0.0

    if srp > 0.0:
        tmp = srp * perv_frac * sro_to_dprst_perv * hru_area
        if dprst_area_open_max > 0.0:
            dprst_srp_open = tmp * dprst_frac_open
            dprst_srp = dprst_srp_open / hru_area
            dprst_vol_open = dprst_vol_open + dprst_srp_open
        if dprst_area_clos_max > 0.0:
            dprst_srp_clos = tmp * dprst_frac_clos
            dprst_srp = dprst_srp + dprst_srp_clos / hru_area
            dprst_vol_clos = dprst_vol_clos + dprst_srp_clos
        srp = srp - dprst_srp / perv_frac
        if srp < 0.0:
            srp = 0.0

    if sri > 0.0:
        tmp = sri * imperv_frac * sro_to_dprst_imperv * hru_area
        if dprst_area_open_max > 0.0:
            dprst_sri_open = tmp * dprst_frac_open
            dprst_sri = dprst_sri_open / hru_area
            dprst_vol_open = dprst_vol_open + dprst_sri_open
        if dprst_area_clos_max > 0.0:
            dprst_sri_clos = tmp * dprst_frac_clos
            dprst_sri = dprst_sri + dprst_sri_clos / hru_area
            dprst_vol_clos = dprst_vol_clos + dprst_sri_clos
        sri = sri - dprst_sri / imperv_frac
        if sri < 0.0:
            sri = 0.0

    # Add pervious and impervious contributions first
    dprst_insroff_hru = dprst_srp + dprst_sri

    # Add agricultural surface runoff fraction to depressions
    if sroff_ag > 0.0:
        tmp = sroff_ag * ag_frac * sro_to_dprst_perv * hru_area
        if dprst_area_open_max > 0.0:
            dprst_sra_open = tmp * dprst_frac_open
            dprst_sra = dprst_sra_open / hru_area
            dprst_vol_open = dprst_vol_open + dprst_sra_open
        if dprst_area_clos_max > 0.0:
            dprst_sra_clos = tmp * dprst_frac_clos
            dprst_sra = dprst_sra + dprst_sra_clos / hru_area
            dprst_vol_clos = dprst_vol_clos + dprst_sra_clos
        sroff_ag = sroff_ag - dprst_sra / ag_frac
        if sroff_ag < 0.0:
            sroff_ag = 0.0
        dprst_insroff_hru = dprst_insroff_hru + dprst_sra

    dprst_area_open = 0.0
    if dprst_vol_open > 0.0:
        open_vol_r = dprst_vol_open / dprst_vol_open_max
        if open_vol_r < _NEARZERO:
            frac_op_ar = 0.0
        elif open_vol_r > 1.0:
            frac_op_ar = 1.0
        else:
            frac_op_ar = np.exp(va_open_exp * np.log(open_vol_r))
        dprst_area_open = dprst_area_open_max * frac_op_ar
        if dprst_area_open > dprst_area_open_max:
            dprst_area_open = dprst_area_open_max

    if dprst_area_clos_max > 0.0:
        dprst_area_clos = 0.0
        if dprst_vol_clos > 0.0:
            clos_vol_r = dprst_vol_clos / dprst_vol_clos_max
            if clos_vol_r < _NEARZERO:
                frac_cl_ar = 0.0
            elif clos_vol_r > 1.0:
                frac_cl_ar = 1.0
            else:
                frac_cl_ar = np.exp(va_clos_exp * np.log(clos_vol_r))
            dprst_area_clos = dprst_area_clos_max * frac_cl_ar
            if dprst_area_clos > dprst_area_clos_max:
                dprst_area_clos = dprst_area_clos_max

    # Evaporate water from depressions
    unsatisfied_et = avail_et
    dprst_avail_et = potet * (1.0 - snowcov_area) * dprst_et_coef
    dprst_evap_hru = 0.0
    if dprst_avail_et > 0.0:
        dprst_evap_open = 0.0
        dprst_evap_clos = 0.0
        if dprst_area_open > 0.0:
            dprst_evap_open = min(
                dprst_area_open * dprst_avail_et, dprst_vol_open
            )
            dprst_evap_open = min(
                dprst_area_open * dprst_avail_et, dprst_vol_open
            )
            if dprst_evap_open / hru_area > unsatisfied_et:
                dprst_evap_open = unsatisfied_et * hru_area
            if dprst_evap_open > dprst_vol_open:
                dprst_evap_open = dprst_vol_open
            unsatisfied_et = unsatisfied_et - dprst_evap_open / hru_area
            dprst_vol_open = dprst_vol_open - dprst_evap_open

        if dprst_area_clos > 0.0:
            dprst_evap_clos = min(
                dprst_area_clos * dprst_avail_et, dprst_vol_clos
            )
            if dprst_evap_clos / hru_area > unsatisfied_et:
                dprst_evap_clos = unsatisfied_et * hru_area
            if dprst_evap_clos > dprst_vol_clos:
                dprst_evap_clos = dprst_vol_clos
            dprst_vol_clos = dprst_vol_clos - dprst_evap_clos

        dprst_evap_hru = (dprst_evap_open + dprst_evap_clos) / hru_area

    # Compute seepage
    dprst_seep_hru = 0.0
    if dprst_vol_open > 0.0:
        seep_open = dprst_vol_open * dprst_seep_rate_open
        dprst_vol_open = dprst_vol_open - seep_open
        if dprst_vol_open < 0.0:
            seep_open = seep_open + dprst_vol_open
            dprst_vol_open = 0.0
        dprst_seep_hru = seep_open / hru_area

    if dprst_area_clos_max > 0.0:
        if dprst_vol_clos > 0.0:
            seep_clos = dprst_vol_clos * dprst_seep_rate_clos
            dprst_vol_clos = dprst_vol_clos - seep_clos
            if dprst_vol_clos < 0.0:
                seep_clos = seep_clos + dprst_vol_clos
                dprst_vol_clos = 0.0
            dprst_seep_hru = dprst_seep_hru + seep_clos / hru_area

    # Compute open surface runoff
    dprst_sroff_hru = 0.0
    if dprst_vol_open > 0.0:
        dprst_sroff_hru = max(0.0, dprst_vol_open - dprst_vol_open_max)
        dprst_sroff_hru = dprst_sroff_hru + max(
            0.0,
            (dprst_vol_open - dprst_sroff_hru - dprst_vol_thres_open)
            * dprst_flow_coef,
        )
        dprst_vol_open = dprst_vol_open - dprst_sroff_hru
        dprst_sroff_hru = dprst_sroff_hru / hru_area

    # Update fractions
    dprst_stor_hru = (dprst_vol_open + dprst_vol_clos) / hru_area
    if dprst_vol_open_max > 0.0:
        dprst_vol_open_frac = dprst_vol_open / dprst_vol_open_max
    else:
        dprst_vol_open_frac = 0.0

    if dprst_vol_clos_max > 0.0:
        dprst_vol_clos_frac = dprst_vol_clos / dprst_vol_clos_max
    else:
        dprst_vol_clos_frac = 0.0

    if (dprst_vol_open_max + dprst_vol_clos_max) > 0.0:
        dprst_vol_frac = (dprst_vol_open + dprst_vol_clos) / (
            dprst_vol_open_max + dprst_vol_clos_max
        )
    else:
        dprst_vol_frac = 0.0

    return (
        dprst_in,
        dprst_vol_open,
        dprst_area_open,
        avail_et,
        dprst_vol_clos,
        dprst_sroff_hru,
        srp,
        sri,
        sroff_ag,
        dprst_evap_hru,
        dprst_seep_hru,
        dprst_insroff_hru,
        dprst_vol_open_frac,
        dprst_vol_clos_frac,
        dprst_vol_frac,
        dprst_stor_hru,
    )


class PRMSRunoffNoDprst(Process):
    """PRMS surface runoff without depression storage: pervious/
    impervious runoff and infiltration partitioning per HRU.

    The minimal core of the runoff family; PRMSRunoff adds the
    surface-depression (dprst) storage physics.

    Storage and fluxes are in inches over the HRU (PRMS convention);
    sroff_vol is cubic feet (via hru_in_to_cf).
    """

    # ------------------------------------------------------------------
    # Field declarations (names verbatim from pywatershed)
    # ------------------------------------------------------------------

    # -- dis_hru variables (grid-owned; dis-first sourcing) --
    hru_type = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="HRU type (HruType enum; LAND = 1, LAKE = 2)",
    )
    hru_area = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="HRU area [acres]",
    )
    hru_in_to_cf = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Conversion of inches over the HRU to cubic feet",
        derivation="hru_area * 43560.0 / 12.0",
    )

    # -- process parameters --
    hru_percent_imperv = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of HRU area that is impervious [-]",
    )
    imperv_stor_max = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum impervious area retention storage [inches]",
    )
    carea_max = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum possible area contributing to surface runoff [-]",
    )
    smidx_coef = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Coefficient in contributing-area computation [-]",
    )
    smidx_exp = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Exponent in contributing-area computation [1/inch]",
    )
    soil_moist_max = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum capillary-reservoir water capacity [inches]",
    )
    snowinfil_max = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum snow infiltration per day [inches/day]",
    )

    # -- derived parameters (initialize(); basin_init) --
    hru_perv = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Pervious HRU area [acres]",
    )
    hru_frac_perv = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Pervious fraction of HRU area [-]",
    )
    hru_imperv = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Impervious HRU area [acres]",
    )

    # -- inputs --
    soil_lower_prev = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Previous lower-reservoir soil storage [inches]",
    )
    soil_rechr_prev = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Previous recharge-reservoir soil storage [inches]",
    )
    net_rain = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Rain through the canopy [inches]",
    )
    net_ppt = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Precipitation through the canopy [inches]",
    )
    net_snow = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snow through the canopy [inches]",
    )
    potet = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Potential evapotranspiration [inches]",
    )
    snowmelt = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snowmelt from the snowpack [inches]",
    )
    snow_evap = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Evaporation and sublimation from the snowpack [inches]",
    )
    pkwater_equiv = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snowpack water equivalent [inches]",
    )
    pptmix_nopack = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Mixed rain/snow event with no snowpack (0/1 flag)",
    )
    snowcov_area = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snow-covered area fraction [-]",
    )
    through_rain = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Rain that passes through snow [inches]",
    )
    hru_intcpevap = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="HRU area-weighted canopy evaporation [inches]",
    )
    intcp_changeover = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Canopy throughfall from winter->summer density change",
    )

    # -- variables --
    contrib_fraction = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Contributing area fraction of the pervious area [-]",
    )
    infil = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Infiltration to the capillary/preferential zones "
        "[inches over pervious area]",
    )
    infil_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Infiltration [inches over the HRU]",
    )
    sroff = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff to the stream network [inches]",
    )
    sroff_vol = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff volume [cubic feet]",
    )
    hru_sroffp = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pervious surface runoff [inches over the HRU]",
    )
    hru_sroffi = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Impervious surface runoff [inches over the HRU]",
    )
    imperv_stor = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Impervious area retention storage [inches over imperv]",
    )
    imperv_evap = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Impervious area evaporation [inches over imperv]",
    )
    hru_impervevap = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Impervious area evaporation [inches over the HRU]",
    )
    hru_impervstor = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Impervious retention storage [inches over the HRU]",
    )
    hru_impervstor_old = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Impervious retention storage, previous timestep",
    )
    hru_impervstor_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Impervious retention storage change [inches]",
    )

    # ------------------------------------------------------------------
    # Initialization (basin_init, no-dprst path)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """basin_init with the dprst branch OFF (upstream shared
        basin_init under _dprst_flag == False)."""
        obj = self._obj
        nhru = obj["hru_area"].values.shape[0]

        # -- zero-init all variables (pywatershed get_init_values) --
        for name in (
            "contrib_fraction",
            "infil",
            "infil_hru",
            "sroff",
            "sroff_vol",
            "hru_sroffp",
            "hru_sroffi",
            "imperv_stor",
            "imperv_evap",
            "hru_impervevap",
            "hru_impervstor",
            "hru_impervstor_old",
            "hru_impervstor_change",
        ):
            obj[name].values[:] = 0.0
        for name in (
            "hru_perv",
            "hru_frac_perv",
            "hru_imperv",
        ):
            obj[name].values[:] = 0.0

        hru_area = obj["hru_area"].values
        hru_percent_imperv = obj["hru_percent_imperv"].values
        hru_perv = obj["hru_perv"].values
        hru_frac_perv = obj["hru_frac_perv"].values
        hru_imperv = obj["hru_imperv"].values

        # -- basin_init (prms basin.f90/basinit subset) --
        for ii in range(nhru):
            harea = hru_area[ii]
            perv_area = harea
            if hru_percent_imperv[ii] > 0.0:
                hru_imperv[ii] = hru_percent_imperv[ii] * harea
                perv_area = perv_area - hru_imperv[ii]

            hru_perv[ii] = perv_area
            hru_frac_perv[ii] = perv_area / harea

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        obj = self._obj
        obj["hru_impervstor_old"].values[:] = obj["hru_impervstor"].values

    @staticmethod
    @numba.njit
    def _calculate(
        # in/out state + outputs (written in place)
        contrib_fraction: np.ndarray,
        infil: np.ndarray,
        infil_hru: np.ndarray,
        sroff: np.ndarray,
        sroff_vol: np.ndarray,
        hru_sroffp: np.ndarray,
        hru_sroffi: np.ndarray,
        imperv_stor: np.ndarray,
        imperv_evap: np.ndarray,
        hru_impervevap: np.ndarray,
        hru_impervstor: np.ndarray,
        hru_impervstor_change: np.ndarray,
        # prior state (read-only here; advance() maintains)
        hru_impervstor_old: np.ndarray,
        # inputs
        soil_lower_prev: np.ndarray,
        soil_rechr_prev: np.ndarray,
        net_rain: np.ndarray,
        net_ppt: np.ndarray,
        net_snow: np.ndarray,
        potet: np.ndarray,
        snowmelt: np.ndarray,
        snow_evap: np.ndarray,
        pkwater_equiv: np.ndarray,
        pptmix_nopack: np.ndarray,
        snowcov_area: np.ndarray,
        through_rain: np.ndarray,
        hru_intcpevap: np.ndarray,
        intcp_changeover: np.ndarray,
        # parameters + derived
        hru_type: np.ndarray,
        hru_area: np.ndarray,
        hru_in_to_cf: np.ndarray,
        hru_percent_imperv: np.ndarray,
        imperv_stor_max: np.ndarray,
        carea_max: np.ndarray,
        smidx_coef: np.ndarray,
        smidx_exp: np.ndarray,
        soil_moist_max: np.ndarray,
        snowinfil_max: np.ndarray,
        hru_perv: np.ndarray,
        hru_frac_perv: np.ndarray,
        hru_imperv: np.ndarray,
        # options
        intcp_changeover_in_net_rain: bool,
    ) -> None:
        # the PRMSRunoff kernel with the dprst block deleted (upstream
        # shared kernel under dprst_flag == OFF); everything else
        # identical, same shared njit helpers
        nhru = sroff.shape[0]
        for ii in range(nhru):
            # pywatershed allocates soil_moist_prev = lower + rechr as
            # an array pre-loop; scalar per element, same op order
            soil_moist_prev = soil_lower_prev[ii] + soil_rechr_prev[ii]

            runoff = 0.0
            hruarea = hru_area[ii]
            perv_area = hru_perv[ii]
            perv_frac = hru_frac_perv[ii]
            srp = 0.0
            sri = 0.0
            hru_sroffp[ii] = 0.0
            contrib_fraction[ii] = 0.0
            infil[ii] = 0.0
            hruarea_imperv = hru_imperv[ii]
            imperv_frac = 0.0
            if hruarea_imperv > 0.0:
                imperv_frac = hru_percent_imperv[ii]
                hru_sroffi[ii] = 0.0
                imperv_evap[ii] = 0.0
                hru_impervevap[ii] = 0.0

            avail_et = potet[ii] - snow_evap[ii] - hru_intcpevap[ii]

            (
                sri,
                srp,
                imperv_stor[ii],
                infil[ii],
                contrib_fraction[ii],
            ) = compute_infil(
                contrib_fraction=contrib_fraction[ii],
                soil_moist_prev=soil_moist_prev,
                soil_moist_max=soil_moist_max[ii],
                carea_max=carea_max[ii],
                smidx_coef=smidx_coef[ii],
                smidx_exp=smidx_exp[ii],
                pptmix_nopack=pptmix_nopack[ii],
                net_rain=net_rain[ii],
                net_ppt=net_ppt[ii],
                imperv_stor=imperv_stor[ii],
                imperv_stor_max=imperv_stor_max[ii],
                snowmelt=snowmelt[ii],
                snowinfil_max=snowinfil_max[ii],
                net_snow=net_snow[ii],
                pkwater_equiv=pkwater_equiv[ii],
                infil=infil[ii],
                hru_type=hru_type[ii],
                intcp_changeover=intcp_changeover[ii],
                hruarea_imperv=hruarea_imperv,
                sri=sri,
                srp=srp,
                through_rain=through_rain[ii],
                intcp_changeover_in_net_rain=intcp_changeover_in_net_rain,
            )

            # runoff for pervious and impervious areas
            srunoff = 0.0
            if hru_type[ii] == _LAND:
                runoff = runoff + srp * perv_area + sri * hruarea_imperv
                srunoff = runoff / hruarea
                hru_sroffp[ii] = srp * perv_frac

            # evaporation from impervious area
            if hruarea_imperv > 0.0:
                if imperv_stor[ii] > 0.0:
                    imperv_stor[ii], imperv_evap[ii] = imperv_et(
                        imperv_stor[ii],
                        potet[ii],
                        imperv_evap[ii],
                        snowcov_area[ii],
                        avail_et,
                        imperv_frac,
                    )
                    hru_impervevap[ii] = imperv_evap[ii] * imperv_frac
                    avail_et = avail_et - hru_impervevap[ii]
                    if avail_et < 0.0:
                        hru_impervevap[ii] = hru_impervevap[ii] + avail_et
                        if hru_impervevap[ii] < 0.0:
                            hru_impervevap[ii] = 0.0
                        imperv_evap[ii] = imperv_evap[ii] / imperv_frac
                        imperv_stor[ii] = (
                            imperv_stor[ii] - avail_et / imperv_frac
                        )
                        avail_et = 0.0

                    hru_impervstor[ii] = imperv_stor[ii] * imperv_frac

                hru_sroffi[ii] = sri * imperv_frac

            sroff[ii] = srunoff

            # pywatershed post-kernel array lines, folded per element
            infil_hru[ii] = infil[ii] * hru_frac_perv[ii]
            hru_impervstor_change[ii] = (
                hru_impervstor[ii] - hru_impervstor_old[ii]
            )
            sroff_vol[ii] = sroff[ii] * hru_in_to_cf[ii]

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate(
            obj["contrib_fraction"].values,
            obj["infil"].values,
            obj["infil_hru"].values,
            obj["sroff"].values,
            obj["sroff_vol"].values,
            obj["hru_sroffp"].values,
            obj["hru_sroffi"].values,
            obj["imperv_stor"].values,
            obj["imperv_evap"].values,
            obj["hru_impervevap"].values,
            obj["hru_impervstor"].values,
            obj["hru_impervstor_change"].values,
            obj["hru_impervstor_old"].values,
            obj["soil_lower_prev"].values,
            obj["soil_rechr_prev"].values,
            obj["net_rain"].values,
            obj["net_ppt"].values,
            obj["net_snow"].values,
            obj["potet"].values,
            obj["snowmelt"].values,
            obj["snow_evap"].values,
            obj["pkwater_equiv"].values,
            obj["pptmix_nopack"].values,
            obj["snowcov_area"].values,
            obj["through_rain"].values,
            obj["hru_intcpevap"].values,
            obj["intcp_changeover"].values,
            obj["hru_type"].values,
            obj["hru_area"].values,
            obj["hru_in_to_cf"].values,
            obj["hru_percent_imperv"].values,
            obj["imperv_stor_max"].values,
            obj["carea_max"].values,
            obj["smidx_coef"].values,
            obj["smidx_exp"].values,
            obj["soil_moist_max"].values,
            obj["snowinfil_max"].values,
            obj["hru_perv"].values,
            obj["hru_frac_perv"].values,
            obj["hru_imperv"].values,
            _INTCP_CHANGEOVER_IN_NET_RAIN,
        )


class PRMSRunoff(PRMSRunoffNoDprst):
    """PRMS surface runoff: the NoDprst core PLUS surface-depression
    (dprst) storage per HRU.

    Storage and fluxes are in inches over the HRU (PRMS convention);
    dprst volumes are acre-inches; sroff_vol is cubic feet (via
    hru_in_to_cf).
    """

    # ------------------------------------------------------------------
    # Field declarations ADDED to the NoDprst core (names verbatim)
    # ------------------------------------------------------------------

    # -- process parameters --
    dprst_depth_avg = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Average depth of surface depressions [inches]",
    )
    dprst_et_coef = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of unsatisfied PET to apply to dprst [-]",
    )
    dprst_flow_coef = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Coefficient in linear flow routing from open dprst [1/d]",
    )
    dprst_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of HRU area that has surface depressions [-]",
    )
    dprst_frac_init = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Initial fraction of maximum dprst storage volume [-]",
    )
    dprst_frac_open = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of open (spilling) surface depressions [-]",
    )
    dprst_seep_rate_clos = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Closed-depression seepage rate [fraction/day]",
    )
    dprst_seep_rate_open = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Open-depression seepage rate [fraction/day]",
    )
    sro_to_dprst_imperv = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of impervious runoff routed to dprst [-]",
    )
    sro_to_dprst_perv = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of pervious runoff routed to dprst [-]",
    )
    va_open_exp = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Open-depression volume-area exponent [-]",
    )
    va_clos_exp = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Closed-depression volume-area exponent [-]",
    )
    op_flow_thres = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Open-depression flow threshold (frac of max volume) [-]",
    )

    # -- derived parameters (initialize(); basin_init/dprst_init) --
    dprst_area_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Maximum surface-depression area [acres]",
    )
    dprst_area_open_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Maximum open surface-depression area [acres]",
    )
    dprst_area_clos_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Maximum closed surface-depression area [acres]",
    )
    dprst_frac_clos = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of closed surface depressions [-]",
    )
    dprst_vol_open_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Maximum open surface-depression volume [acre-inches]",
    )
    dprst_vol_clos_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Maximum closed surface-depression volume [acre-inches]",
    )
    dprst_vol_thres_open = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description=(
            "Open-depression volume above which flow occurs [acre-inches] "
            "(pywatershed 'variable', never written by its kernel)"
        ),
    )

    # -- variables --
    dprst_vol_open = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Open surface-depression storage volume [acre-inches]",
    )
    dprst_vol_clos = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Closed surface-depression storage volume [acre-inches]",
    )
    dprst_vol_open_frac = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Open depression volume fraction of maximum [-]",
    )
    dprst_vol_clos_frac = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Closed depression volume fraction of maximum [-]",
    )
    dprst_vol_frac = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Depression volume fraction of maximum (open+closed) [-]",
    )
    dprst_area_open = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Open surface-depression area [acres]",
    )
    dprst_area_clos = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description=(
            "Closed surface-depression area [acres] (static after init: "
            "pywatershed never writes it back -- see module docstring)"
        ),
    )
    dprst_sroff_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Open-depression surface runoff [inches over the HRU]",
    )
    dprst_seep_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Depression seepage to groundwater [inches over the HRU]",
    )
    dprst_evap_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Depression evaporation [inches over the HRU]",
    )
    dprst_insroff_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Runoff captured by depressions [inches over the HRU]",
    )
    dprst_stor_hru = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Depression storage [inches over the HRU]",
    )
    dprst_stor_hru_old = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Depression storage, previous timestep",
    )
    dprst_stor_hru_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Depression storage change [inches over the HRU]",
    )
    dprst_in = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="New water to depressions [inches over the HRU]",
    )

    # ------------------------------------------------------------------
    # Initialization (basin_init + dprst_init, no-restart path)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """basin_init + dprst_init verbatim (init-time loops; perf
        irrelevant). Derived geometry -> parameter_internal (frozen
        after this); dprst initial state -> variables."""
        obj = self._obj
        nhru = obj["hru_area"].values.shape[0]

        # -- zero-init all variables (pywatershed get_init_values) --
        for name in (
            "contrib_fraction",
            "infil",
            "infil_hru",
            "sroff",
            "sroff_vol",
            "hru_sroffp",
            "hru_sroffi",
            "imperv_stor",
            "imperv_evap",
            "hru_impervevap",
            "hru_impervstor",
            "hru_impervstor_old",
            "hru_impervstor_change",
            "dprst_vol_frac",
            "dprst_vol_clos",
            "dprst_vol_open",
            "dprst_vol_clos_frac",
            "dprst_vol_open_frac",
            "dprst_area_clos",
            "dprst_area_open",
            "dprst_sroff_hru",
            "dprst_seep_hru",
            "dprst_evap_hru",
            "dprst_insroff_hru",
            "dprst_stor_hru",
            "dprst_stor_hru_old",
            "dprst_stor_hru_change",
            "dprst_in",
        ):
            obj[name].values[:] = 0.0
        for name in (
            "hru_perv",
            "hru_frac_perv",
            "hru_imperv",
            "dprst_area_max",
            "dprst_area_open_max",
            "dprst_area_clos_max",
            "dprst_frac_clos",
            "dprst_vol_open_max",
            "dprst_vol_clos_max",
            "dprst_vol_thres_open",
        ):
            obj[name].values[:] = 0.0

        hru_area = obj["hru_area"].values
        hru_percent_imperv = obj["hru_percent_imperv"].values
        dprst_frac = obj["dprst_frac"].values
        dprst_frac_open = obj["dprst_frac_open"].values
        dprst_frac_init = obj["dprst_frac_init"].values
        dprst_depth_avg = obj["dprst_depth_avg"].values
        op_flow_thres = obj["op_flow_thres"].values
        va_open_exp = obj["va_open_exp"].values
        va_clos_exp = obj["va_clos_exp"].values

        hru_perv = obj["hru_perv"].values
        hru_frac_perv = obj["hru_frac_perv"].values
        hru_imperv = obj["hru_imperv"].values
        dprst_area_max = obj["dprst_area_max"].values
        dprst_area_open_max = obj["dprst_area_open_max"].values
        dprst_area_clos_max = obj["dprst_area_clos_max"].values
        dprst_frac_clos = obj["dprst_frac_clos"].values
        dprst_vol_open_max = obj["dprst_vol_open_max"].values
        dprst_vol_clos_max = obj["dprst_vol_clos_max"].values
        dprst_vol_thres_open = obj["dprst_vol_thres_open"].values
        dprst_vol_open = obj["dprst_vol_open"].values
        dprst_vol_clos = obj["dprst_vol_clos"].values
        dprst_area_open = obj["dprst_area_open"].values
        dprst_area_clos = obj["dprst_area_clos"].values
        dprst_stor_hru = obj["dprst_stor_hru"].values
        dprst_stor_hru_old = obj["dprst_stor_hru_old"].values
        dprst_vol_frac = obj["dprst_vol_frac"].values

        # -- basin_init (prms basin.f90/basinit subset) --
        dprst_clos_flag = False
        for ii in range(nhru):
            harea = hru_area[ii]
            perv_area = harea
            if hru_percent_imperv[ii] > 0.0:
                hru_imperv[ii] = hru_percent_imperv[ii] * harea
                perv_area = perv_area - hru_imperv[ii]

            dprst_area_max[ii] = dprst_frac[ii] * harea
            if dprst_area_max[ii] > 0.0:
                dprst_area_open_max[ii] = (
                    dprst_area_max[ii] * dprst_frac_open[ii]
                )
                dprst_frac_clos[ii] = 1.0 - dprst_frac_open[ii]
                dprst_area_clos_max[ii] = (
                    dprst_area_max[ii] - dprst_area_open_max[ii]
                )
                if dprst_area_clos_max[ii] > 0.0:
                    dprst_clos_flag = True
                perv_area = perv_area - dprst_area_max[ii]

            hru_perv[ii] = perv_area
            hru_frac_perv[ii] = perv_area / harea

        # -- dprst_init --
        # When no HRU has closed depressions (nhm domains:
        # dprst_frac_open == 1 everywhere), pywatershed zeroes
        # dprst_seep_rate_clos and va_clos_exp ("BAD practice of
        # editing parameters", its words). NOT ported: every use of
        # both parameters is guarded by dprst_area_clos_max > 0 or
        # dprst_vol_clos > 0, which never hold when the flag is off --
        # the edit is provably a no-op (same reasoning as the channel's
        # unported seg_slope clamp).
        del dprst_clos_flag

        for ii in range(nhru):
            if dprst_frac[ii] > 0.0:
                if dprst_depth_avg[ii] == 0.0:
                    raise ValueError(
                        f"dprst_frac > 0 and dprst_depth_avg == 0 for HRU {ii}"
                    )

                # open and closed volumes (acre-inches); upstream's
                # tautological flag reassign-then-test dance collapses
                # to unconditional computation
                dprst_vol_clos_max[ii] = (
                    dprst_area_clos_max[ii] * dprst_depth_avg[ii]
                )
                dprst_vol_open_max[ii] = (
                    dprst_area_open_max[ii] * dprst_depth_avg[ii]
                )
                # initial storage volumes (no-restart path)
                dprst_vol_open[ii] = (
                    dprst_frac_init[ii] * dprst_vol_open_max[ii]
                )
                dprst_vol_clos[ii] = (
                    dprst_frac_init[ii] * dprst_vol_clos_max[ii]
                )

                # threshold volume: fraction of max open storage above
                # which flow occurs
                dprst_vol_thres_open[ii] = (
                    op_flow_thres[ii] * dprst_vol_open_max[ii]
                )
                if dprst_vol_open[ii] > 0.0:
                    open_vol_r = dprst_vol_open[ii] / dprst_vol_open_max[ii]
                    if open_vol_r < _NEARZERO:
                        frac_op_ar = 0.0
                    elif open_vol_r > 1.0:
                        frac_op_ar = 1.0
                    else:
                        frac_op_ar = np.exp(
                            va_open_exp[ii] * np.log(open_vol_r)
                        )
                    dprst_area_open[ii] = dprst_area_open_max[ii] * frac_op_ar
                    if dprst_area_open[ii] > dprst_area_open_max[ii]:
                        dprst_area_open[ii] = dprst_area_open_max[ii]

                if dprst_vol_clos[ii] > 0.0:
                    clos_vol_r = dprst_vol_clos[ii] / dprst_vol_clos_max[ii]
                    if clos_vol_r < _NEARZERO:
                        frac_cl_ar = 0.0
                    elif clos_vol_r > 1.0:
                        frac_cl_ar = 1.0
                    else:
                        frac_cl_ar = np.exp(
                            va_clos_exp[ii] * np.log(clos_vol_r)
                        )
                    dprst_area_clos[ii] = dprst_area_clos_max[ii] * frac_cl_ar
                    if dprst_area_clos[ii] > dprst_area_clos_max[ii]:
                        dprst_area_clos[ii] = dprst_area_clos_max[ii]

                dprst_stor_hru[ii] = (
                    dprst_vol_open[ii] + dprst_vol_clos[ii]
                ) / hru_area[ii]
                dprst_stor_hru_old[ii] = dprst_stor_hru[ii]

                if dprst_vol_open_max[ii] + dprst_vol_clos_max[ii] > 0.0:
                    dprst_vol_frac[ii] = (
                        dprst_vol_open[ii] + dprst_vol_clos[ii]
                    ) / (dprst_vol_open_max[ii] + dprst_vol_clos_max[ii])

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        obj = self._obj
        obj["hru_impervstor_old"].values[:] = obj["hru_impervstor"].values
        obj["dprst_stor_hru_old"].values[:] = obj["dprst_stor_hru"].values

    @staticmethod
    @numba.njit
    def _calculate(
        # in/out state + outputs (written in place)
        contrib_fraction: np.ndarray,
        infil: np.ndarray,
        infil_hru: np.ndarray,
        sroff: np.ndarray,
        sroff_vol: np.ndarray,
        hru_sroffp: np.ndarray,
        hru_sroffi: np.ndarray,
        imperv_stor: np.ndarray,
        imperv_evap: np.ndarray,
        hru_impervevap: np.ndarray,
        hru_impervstor: np.ndarray,
        hru_impervstor_change: np.ndarray,
        dprst_vol_open: np.ndarray,
        dprst_vol_clos: np.ndarray,
        dprst_vol_open_frac: np.ndarray,
        dprst_vol_clos_frac: np.ndarray,
        dprst_vol_frac: np.ndarray,
        dprst_area_open: np.ndarray,
        dprst_area_clos: np.ndarray,
        dprst_sroff_hru: np.ndarray,
        dprst_seep_hru: np.ndarray,
        dprst_evap_hru: np.ndarray,
        dprst_insroff_hru: np.ndarray,
        dprst_stor_hru: np.ndarray,
        dprst_stor_hru_change: np.ndarray,
        dprst_in: np.ndarray,
        # prior state (read-only here; advance() maintains)
        hru_impervstor_old: np.ndarray,
        dprst_stor_hru_old: np.ndarray,
        # inputs
        soil_lower_prev: np.ndarray,
        soil_rechr_prev: np.ndarray,
        net_rain: np.ndarray,
        net_ppt: np.ndarray,
        net_snow: np.ndarray,
        potet: np.ndarray,
        snowmelt: np.ndarray,
        snow_evap: np.ndarray,
        pkwater_equiv: np.ndarray,
        pptmix_nopack: np.ndarray,
        snowcov_area: np.ndarray,
        through_rain: np.ndarray,
        hru_intcpevap: np.ndarray,
        intcp_changeover: np.ndarray,
        # parameters + derived
        hru_type: np.ndarray,
        hru_area: np.ndarray,
        hru_in_to_cf: np.ndarray,
        hru_percent_imperv: np.ndarray,
        imperv_stor_max: np.ndarray,
        carea_max: np.ndarray,
        smidx_coef: np.ndarray,
        smidx_exp: np.ndarray,
        soil_moist_max: np.ndarray,
        snowinfil_max: np.ndarray,
        dprst_et_coef: np.ndarray,
        dprst_flow_coef: np.ndarray,
        dprst_frac_open: np.ndarray,
        dprst_seep_rate_clos: np.ndarray,
        dprst_seep_rate_open: np.ndarray,
        sro_to_dprst_imperv: np.ndarray,
        sro_to_dprst_perv: np.ndarray,
        va_open_exp: np.ndarray,
        va_clos_exp: np.ndarray,
        hru_perv: np.ndarray,
        hru_frac_perv: np.ndarray,
        hru_imperv: np.ndarray,
        dprst_area_max: np.ndarray,
        dprst_area_open_max: np.ndarray,
        dprst_area_clos_max: np.ndarray,
        dprst_frac_clos: np.ndarray,
        dprst_vol_open_max: np.ndarray,
        dprst_vol_clos_max: np.ndarray,
        dprst_vol_thres_open: np.ndarray,
        # options
        intcp_changeover_in_net_rain: bool,
    ) -> None:
        nhru = sroff.shape[0]
        for ii in range(nhru):
            # pywatershed allocates soil_moist_prev = lower + rechr as
            # an array pre-loop; scalar per element, same op order
            soil_moist_prev = soil_lower_prev[ii] + soil_rechr_prev[ii]

            runoff = 0.0
            hruarea = hru_area[ii]
            perv_area = hru_perv[ii]
            perv_frac = hru_frac_perv[ii]
            srp = 0.0
            sri = 0.0
            hru_sroffp[ii] = 0.0
            contrib_fraction[ii] = 0.0
            infil[ii] = 0.0
            hruarea_imperv = hru_imperv[ii]
            imperv_frac = 0.0
            if hruarea_imperv > 0.0:
                imperv_frac = hru_percent_imperv[ii]
                hru_sroffi[ii] = 0.0
                imperv_evap[ii] = 0.0
                hru_impervevap[ii] = 0.0

            avail_et = potet[ii] - snow_evap[ii] - hru_intcpevap[ii]
            availh2o = intcp_changeover[ii] + net_rain[ii]

            (
                sri,
                srp,
                imperv_stor[ii],
                infil[ii],
                contrib_fraction[ii],
            ) = compute_infil(
                contrib_fraction=contrib_fraction[ii],
                soil_moist_prev=soil_moist_prev,
                soil_moist_max=soil_moist_max[ii],
                carea_max=carea_max[ii],
                smidx_coef=smidx_coef[ii],
                smidx_exp=smidx_exp[ii],
                pptmix_nopack=pptmix_nopack[ii],
                net_rain=net_rain[ii],
                net_ppt=net_ppt[ii],
                imperv_stor=imperv_stor[ii],
                imperv_stor_max=imperv_stor_max[ii],
                snowmelt=snowmelt[ii],
                snowinfil_max=snowinfil_max[ii],
                net_snow=net_snow[ii],
                pkwater_equiv=pkwater_equiv[ii],
                infil=infil[ii],
                hru_type=hru_type[ii],
                intcp_changeover=intcp_changeover[ii],
                hruarea_imperv=hruarea_imperv,
                sri=sri,
                srp=srp,
                through_rain=through_rain[ii],
                intcp_changeover_in_net_rain=intcp_changeover_in_net_rain,
            )

            # dprst (frozen ground hardwired OFF upstream)
            dprst_in[ii] = 0.0
            dprst_chk = False
            if dprst_area_max[ii] > 0.0:
                dprst_chk = True
                (
                    dprst_in[ii],
                    dprst_vol_open[ii],
                    dprst_area_open[ii],
                    avail_et,
                    dprst_vol_clos[ii],
                    dprst_sroff_hru[ii],
                    srp,
                    sri,
                    dprst_evap_hru[ii],
                    dprst_seep_hru[ii],
                    dprst_insroff_hru[ii],
                    dprst_vol_open_frac[ii],
                    dprst_vol_clos_frac[ii],
                    dprst_vol_frac[ii],
                    dprst_stor_hru[ii],
                ) = dprst_comp(
                    dprst_vol_clos=dprst_vol_clos[ii],
                    dprst_area_clos_max=dprst_area_clos_max[ii],
                    dprst_area_clos=dprst_area_clos[ii],
                    dprst_vol_open_max=dprst_vol_open_max[ii],
                    dprst_vol_open=dprst_vol_open[ii],
                    dprst_area_open_max=dprst_area_open_max[ii],
                    dprst_sroff_hru=dprst_sroff_hru[ii],
                    sro_to_dprst_perv=sro_to_dprst_perv[ii],
                    sro_to_dprst_imperv=sro_to_dprst_imperv[ii],
                    dprst_evap_hru=dprst_evap_hru[ii],
                    pptmix_nopack=pptmix_nopack[ii],
                    snowmelt=snowmelt[ii],
                    pkwater_equiv=pkwater_equiv[ii],
                    net_snow=net_snow[ii],
                    hru_area=hru_area[ii],
                    dprst_insroff_hru=dprst_insroff_hru[ii],
                    dprst_frac_open=dprst_frac_open[ii],
                    dprst_frac_clos=dprst_frac_clos[ii],
                    va_open_exp=va_open_exp[ii],
                    dprst_vol_clos_max=dprst_vol_clos_max[ii],
                    dprst_vol_clos_frac=dprst_vol_clos_frac[ii],
                    va_clos_exp=va_clos_exp[ii],
                    potet=potet[ii],
                    snowcov_area=snowcov_area[ii],
                    dprst_et_coef=dprst_et_coef[ii],
                    dprst_seep_rate_open=dprst_seep_rate_open[ii],
                    dprst_vol_thres_open=dprst_vol_thres_open[ii],
                    dprst_flow_coef=dprst_flow_coef[ii],
                    dprst_seep_rate_clos=dprst_seep_rate_clos[ii],
                    avail_et=avail_et,
                    net_rain=availh2o,
                    dprst_in=dprst_in[ii],
                    srp=srp,
                    sri=sri,
                    imperv_frac=imperv_frac,
                    perv_frac=perv_frac,
                )
                runoff = runoff + dprst_sroff_hru[ii] * hruarea

            # runoff for pervious, impervious, and depression areas
            srunoff = 0.0
            if hru_type[ii] == _LAND:
                runoff = runoff + srp * perv_area + sri * hruarea_imperv
                srunoff = runoff / hruarea
                hru_sroffp[ii] = srp * perv_frac

            # evaporation from impervious area
            if hruarea_imperv > 0.0:
                if imperv_stor[ii] > 0.0:
                    imperv_stor[ii], imperv_evap[ii] = imperv_et(
                        imperv_stor[ii],
                        potet[ii],
                        imperv_evap[ii],
                        snowcov_area[ii],
                        avail_et,
                        imperv_frac,
                    )
                    hru_impervevap[ii] = imperv_evap[ii] * imperv_frac
                    avail_et = avail_et - hru_impervevap[ii]
                    if avail_et < 0.0:
                        hru_impervevap[ii] = hru_impervevap[ii] + avail_et
                        if hru_impervevap[ii] < 0.0:
                            hru_impervevap[ii] = 0.0
                        imperv_evap[ii] = imperv_evap[ii] / imperv_frac
                        imperv_stor[ii] = (
                            imperv_stor[ii] - avail_et / imperv_frac
                        )
                        avail_et = 0.0

                    hru_impervstor[ii] = imperv_stor[ii] * imperv_frac

                hru_sroffi[ii] = sri * imperv_frac

            if dprst_chk:
                dprst_stor_hru[ii] = (
                    dprst_vol_open[ii] + dprst_vol_clos[ii]
                ) / hruarea

            sroff[ii] = srunoff

            # pywatershed post-kernel array lines, folded per element
            infil_hru[ii] = infil[ii] * hru_frac_perv[ii]
            hru_impervstor_change[ii] = (
                hru_impervstor[ii] - hru_impervstor_old[ii]
            )
            dprst_stor_hru_change[ii] = (
                dprst_stor_hru[ii] - dprst_stor_hru_old[ii]
            )
            sroff_vol[ii] = sroff[ii] * hru_in_to_cf[ii]

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate(
            obj["contrib_fraction"].values,
            obj["infil"].values,
            obj["infil_hru"].values,
            obj["sroff"].values,
            obj["sroff_vol"].values,
            obj["hru_sroffp"].values,
            obj["hru_sroffi"].values,
            obj["imperv_stor"].values,
            obj["imperv_evap"].values,
            obj["hru_impervevap"].values,
            obj["hru_impervstor"].values,
            obj["hru_impervstor_change"].values,
            obj["dprst_vol_open"].values,
            obj["dprst_vol_clos"].values,
            obj["dprst_vol_open_frac"].values,
            obj["dprst_vol_clos_frac"].values,
            obj["dprst_vol_frac"].values,
            obj["dprst_area_open"].values,
            obj["dprst_area_clos"].values,
            obj["dprst_sroff_hru"].values,
            obj["dprst_seep_hru"].values,
            obj["dprst_evap_hru"].values,
            obj["dprst_insroff_hru"].values,
            obj["dprst_stor_hru"].values,
            obj["dprst_stor_hru_change"].values,
            obj["dprst_in"].values,
            obj["hru_impervstor_old"].values,
            obj["dprst_stor_hru_old"].values,
            obj["soil_lower_prev"].values,
            obj["soil_rechr_prev"].values,
            obj["net_rain"].values,
            obj["net_ppt"].values,
            obj["net_snow"].values,
            obj["potet"].values,
            obj["snowmelt"].values,
            obj["snow_evap"].values,
            obj["pkwater_equiv"].values,
            obj["pptmix_nopack"].values,
            obj["snowcov_area"].values,
            obj["through_rain"].values,
            obj["hru_intcpevap"].values,
            obj["intcp_changeover"].values,
            obj["hru_type"].values,
            obj["hru_area"].values,
            obj["hru_in_to_cf"].values,
            obj["hru_percent_imperv"].values,
            obj["imperv_stor_max"].values,
            obj["carea_max"].values,
            obj["smidx_coef"].values,
            obj["smidx_exp"].values,
            obj["soil_moist_max"].values,
            obj["snowinfil_max"].values,
            obj["dprst_et_coef"].values,
            obj["dprst_flow_coef"].values,
            obj["dprst_frac_open"].values,
            obj["dprst_seep_rate_clos"].values,
            obj["dprst_seep_rate_open"].values,
            obj["sro_to_dprst_imperv"].values,
            obj["sro_to_dprst_perv"].values,
            obj["va_open_exp"].values,
            obj["va_clos_exp"].values,
            obj["hru_perv"].values,
            obj["hru_frac_perv"].values,
            obj["hru_imperv"].values,
            obj["dprst_area_max"].values,
            obj["dprst_area_open_max"].values,
            obj["dprst_area_clos_max"].values,
            obj["dprst_frac_clos"].values,
            obj["dprst_vol_open_max"].values,
            obj["dprst_vol_clos_max"].values,
            obj["dprst_vol_thres_open"].values,
            _INTCP_CHANGEOVER_IN_NET_RAIN,
        )


class PRMSRunoffAg(PRMSRunoff):
    """PRMS surface runoff with agriculture: the full (dprst) runoff
    PLUS an agricultural area per HRU with its own infiltration/runoff
    partitioning (pywatershed prms_runoff_ag.py; GSFLOW physics).

    ADDITIVE extension (upstream's interface is a strict superset of
    PRMSRunoff's): ag declarations added; ``initialize``/kernel/
    ``calculate`` overridden. ``ag_frac`` is a TIME-VARYING input
    (PRMS dynamic parameter), so ``hru_perv``/``hru_frac_perv`` are
    redeclared kind="variable" (declaration override; frozen in the
    bases) and, with ``ag_area``, are recomputed AFTER the kernel each
    step -- the kernel deliberately sees the PREVIOUS step's areas
    while its ``ag_frac[ii]`` scalars are current (upstream ordering,
    preserved). At time zero the upstream itime_step==0 block (ag area
    carved out of basin_init's pervious area) runs in ``calculate``,
    when ``ag_frac`` has been fed.

    Deliberately NOT ported: ``sat_threshold`` (declared upstream,
    never used by its kernel); ``intcp_changeover_budget`` (Budget
    machinery only); the unused ``_calculate_infil_ag`` method (dead
    alternate path); the dprst_flag switch (this port is dprst-ACTIVE,
    like PRMSRunoff -- fgr_ag_2yr runs dprst on).
    """

    # GSFLOW accounting: intcp_changeover IS in net_rain (upstream
    # derives this from the control's executable_desc; the fgr_ag_2yr
    # answers are GSFLOW). A PRMS-mode ag domain would flip this.
    _intcp_changeover_in_net_rain = True

    # ------------------------------------------------------------------
    # Declaration OVERRIDES: frozen geometry -> per-step variables
    # (dynamic ag_frac; enabled by the resolved-field MRO walk)
    # ------------------------------------------------------------------
    # restart=True on all three per-step areas: the kernel reads the
    # PREVIOUS step's areas (_post_areas updates them AFTER the
    # kernel) and the istep0 carve-out only runs at time zero, so a
    # restart must restore them (prognostic markers, not storages)
    hru_perv = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pervious HRU area [acres] (per-step: ag area and "
        "dynamic ag_frac carve it out)",
        restart=True,
    )
    hru_frac_perv = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pervious fraction of HRU area [-] (per-step under "
        "dynamic ag_frac)",
        restart=True,
    )

    # -- process parameters (ADDED) --
    ag_soil_moist_max = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum ag capillary-reservoir water capacity [inches]",
    )
    ag_soil_rechr_max_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Ag recharge-zone maximum as fraction of "
        "ag_soil_moist_max [-]",
    )

    # -- derived parameters (ADDED) --
    ag_soil_rechr_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Ag recharge-zone maximum storage [inches] "
        "(in upstream kernel signatures but unused)",
    )

    # -- inputs (ADDED) --
    ag_soil_moist_prev = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Previous TOTAL ag soil storage [inches] "
        "(soilzone-ag back-edge)",
    )
    ag_soil_rechr_prev = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Previous ag recharge-zone storage [inches] "
        "(soilzone-ag back-edge)",
    )
    ag_frac = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of HRU area that is agricultural [-] "
        "(TIME-VARYING: PRMS dynamic parameter)",
    )

    # -- variables (ADDED) --
    ag_area = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Agricultural HRU area [acres] (per-step under "
        "dynamic ag_frac)",
        restart=True,
    )
    infil_ag = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Ag infiltration [inches over ag area]",
    )
    infil_ag_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Ag infiltration [inches over the HRU]",
    )
    infil_perv_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pervious-only infiltration [inches over the HRU]",
    )
    hru_sroff_ag = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Agricultural surface runoff [inches over the HRU]",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """PRMSRunoff.initialize (basin_init + dprst_init) plus the ag
        additions. The ag-area carve-out of the pervious area happens
        at time zero in calculate() (upstream's itime_step==0 block):
        ag_frac is an input, not yet fed here."""
        super().initialize()
        obj = self._obj
        for name in (
            "ag_area",
            "infil_ag",
            "infil_ag_hru",
            "infil_perv_hru",
            "hru_sroff_ag",
        ):
            obj[name].values[:] = 0.0
        obj["ag_soil_rechr_max"].values[:] = (
            obj["ag_soil_moist_max"].values
            * obj["ag_soil_rechr_max_frac"].values
        )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @staticmethod
    @numba.njit
    def _calculate(
        # in/out state + outputs (written in place)
        contrib_fraction: np.ndarray,
        infil: np.ndarray,
        infil_ag: np.ndarray,
        sroff: np.ndarray,
        hru_sroffp: np.ndarray,
        hru_sroffi: np.ndarray,
        hru_sroff_ag: np.ndarray,
        imperv_stor: np.ndarray,
        imperv_evap: np.ndarray,
        hru_impervevap: np.ndarray,
        hru_impervstor: np.ndarray,
        dprst_vol_open: np.ndarray,
        dprst_vol_clos: np.ndarray,
        dprst_vol_open_frac: np.ndarray,
        dprst_vol_clos_frac: np.ndarray,
        dprst_vol_frac: np.ndarray,
        dprst_area_open: np.ndarray,
        dprst_area_clos: np.ndarray,
        dprst_sroff_hru: np.ndarray,
        dprst_seep_hru: np.ndarray,
        dprst_evap_hru: np.ndarray,
        dprst_insroff_hru: np.ndarray,
        dprst_stor_hru: np.ndarray,
        dprst_in: np.ndarray,
        # inputs
        soil_lower_prev: np.ndarray,
        soil_rechr_prev: np.ndarray,
        ag_soil_moist_prev: np.ndarray,
        ag_soil_rechr_prev: np.ndarray,
        net_rain: np.ndarray,
        net_ppt: np.ndarray,
        net_snow: np.ndarray,
        potet: np.ndarray,
        snowmelt: np.ndarray,
        snow_evap: np.ndarray,
        pkwater_equiv: np.ndarray,
        pptmix_nopack: np.ndarray,
        snowcov_area: np.ndarray,
        through_rain: np.ndarray,
        hru_intcpevap: np.ndarray,
        intcp_changeover: np.ndarray,
        ag_frac: np.ndarray,
        # parameters + derived (areas are the PREVIOUS step's)
        hru_type: np.ndarray,
        hru_area: np.ndarray,
        hru_percent_imperv: np.ndarray,
        imperv_stor_max: np.ndarray,
        carea_max: np.ndarray,
        smidx_coef: np.ndarray,
        smidx_exp: np.ndarray,
        soil_moist_max: np.ndarray,
        snowinfil_max: np.ndarray,
        ag_soil_moist_max: np.ndarray,
        ag_soil_rechr_max: np.ndarray,
        dprst_et_coef: np.ndarray,
        dprst_flow_coef: np.ndarray,
        dprst_frac_open: np.ndarray,
        dprst_seep_rate_clos: np.ndarray,
        dprst_seep_rate_open: np.ndarray,
        sro_to_dprst_imperv: np.ndarray,
        sro_to_dprst_perv: np.ndarray,
        va_open_exp: np.ndarray,
        va_clos_exp: np.ndarray,
        hru_perv: np.ndarray,
        hru_frac_perv: np.ndarray,
        hru_imperv: np.ndarray,
        ag_area: np.ndarray,
        dprst_area_max: np.ndarray,
        dprst_area_open_max: np.ndarray,
        dprst_area_clos_max: np.ndarray,
        dprst_frac_clos: np.ndarray,
        dprst_vol_open_max: np.ndarray,
        dprst_vol_clos_max: np.ndarray,
        dprst_vol_thres_open: np.ndarray,
        # options
        intcp_changeover_in_net_rain: bool,
    ) -> None:
        # upstream _calculate_numpy_ag verbatim per element; the
        # post-kernel array lines live in _post_areas (they must see
        # the UPDATED areas)
        nhru = sroff.shape[0]
        for ii in range(nhru):
            soil_moist_prev = soil_lower_prev[ii] + soil_rechr_prev[ii]

            runoff = 0.0
            hruarea = hru_area[ii]
            perv_area = hru_perv[ii]
            perv_frac = hru_frac_perv[ii]
            srp = 0.0
            sri = 0.0
            sroff_ag = 0.0
            hru_sroffp[ii] = 0.0
            hru_sroff_ag[ii] = 0.0
            contrib_fraction[ii] = 0.0
            infil[ii] = 0.0
            infil_ag[ii] = 0.0
            hruarea_imperv = hru_imperv[ii]
            imperv_frac = 0.0
            if hruarea_imperv > 0.0:
                imperv_frac = hru_percent_imperv[ii]
                hru_sroffi[ii] = 0.0
                imperv_evap[ii] = 0.0
                hru_impervevap[ii] = 0.0

            avail_et = potet[ii] - snow_evap[ii] - hru_intcpevap[ii]

            # Calculate pervious infiltration
            (
                sri,
                srp,
                imperv_stor[ii],
                infil[ii],
                contrib_fraction[ii],
            ) = compute_infil(
                contrib_fraction=contrib_fraction[ii],
                soil_moist_prev=soil_moist_prev,
                soil_moist_max=soil_moist_max[ii],
                carea_max=carea_max[ii],
                smidx_coef=smidx_coef[ii],
                smidx_exp=smidx_exp[ii],
                pptmix_nopack=pptmix_nopack[ii],
                net_rain=net_rain[ii],
                net_ppt=net_ppt[ii],
                imperv_stor=imperv_stor[ii],
                imperv_stor_max=imperv_stor_max[ii],
                snowmelt=snowmelt[ii],
                snowinfil_max=snowinfil_max[ii],
                net_snow=net_snow[ii],
                pkwater_equiv=pkwater_equiv[ii],
                infil=infil[ii],
                hru_type=hru_type[ii],
                intcp_changeover=intcp_changeover[ii],
                hruarea_imperv=hruarea_imperv,
                sri=sri,
                srp=srp,
                through_rain=through_rain[ii],
                intcp_changeover_in_net_rain=intcp_changeover_in_net_rain,
            )

            # Calculate agricultural infiltration (if ag area exists)
            if ag_area[ii] > 0.0:
                infil_ag[ii], sroff_ag = compute_infil_ag(
                    ag_soil_moist_prev=ag_soil_moist_prev[ii],
                    ag_soil_rechr_prev=ag_soil_rechr_prev[ii],
                    ag_soil_moist_max=ag_soil_moist_max[ii],
                    ag_soil_rechr_max=ag_soil_rechr_max[ii],
                    carea_max=carea_max[ii],
                    smidx_coef=smidx_coef[ii],
                    smidx_exp=smidx_exp[ii],
                    snowinfil_max=snowinfil_max[ii],
                    pptmix_nopack=pptmix_nopack[ii],
                    net_rain=net_rain[ii],
                    net_ppt=net_ppt[ii],
                    snowmelt=snowmelt[ii],
                    net_snow=net_snow[ii],
                    pkwater_equiv=pkwater_equiv[ii],
                    hru_type=hru_type[ii],
                    intcp_changeover=intcp_changeover[ii],
                    through_rain=through_rain[ii],
                    intcp_changeover_in_net_rain=(
                        intcp_changeover_in_net_rain
                    ),
                )

            # dprst incl. agricultural runoff routing (frozen ground
            # hardwired OFF upstream)
            dprst_in[ii] = 0.0
            dprst_chk = False
            if dprst_area_max[ii] > 0.0:
                dprst_chk = True
                (
                    dprst_in[ii],
                    dprst_vol_open[ii],
                    dprst_area_open[ii],
                    avail_et,
                    dprst_vol_clos[ii],
                    dprst_sroff_hru[ii],
                    srp,
                    sri,
                    sroff_ag,
                    dprst_evap_hru[ii],
                    dprst_seep_hru[ii],
                    dprst_insroff_hru[ii],
                    dprst_vol_open_frac[ii],
                    dprst_vol_clos_frac[ii],
                    dprst_vol_frac[ii],
                    dprst_stor_hru[ii],
                ) = dprst_comp_ag(
                    dprst_vol_clos=dprst_vol_clos[ii],
                    dprst_area_clos_max=dprst_area_clos_max[ii],
                    dprst_area_clos=dprst_area_clos[ii],
                    dprst_vol_open_max=dprst_vol_open_max[ii],
                    dprst_vol_open=dprst_vol_open[ii],
                    dprst_area_open_max=dprst_area_open_max[ii],
                    dprst_sroff_hru=dprst_sroff_hru[ii],
                    sro_to_dprst_perv=sro_to_dprst_perv[ii],
                    sro_to_dprst_imperv=sro_to_dprst_imperv[ii],
                    dprst_evap_hru=dprst_evap_hru[ii],
                    through_rain=through_rain[ii],
                    snowmelt=snowmelt[ii],
                    hru_area=hru_area[ii],
                    dprst_insroff_hru=dprst_insroff_hru[ii],
                    dprst_frac_open=dprst_frac_open[ii],
                    dprst_frac_clos=dprst_frac_clos[ii],
                    va_open_exp=va_open_exp[ii],
                    dprst_vol_clos_max=dprst_vol_clos_max[ii],
                    dprst_vol_clos_frac=dprst_vol_clos_frac[ii],
                    va_clos_exp=va_clos_exp[ii],
                    potet=potet[ii],
                    snowcov_area=snowcov_area[ii],
                    dprst_et_coef=dprst_et_coef[ii],
                    dprst_seep_rate_open=dprst_seep_rate_open[ii],
                    dprst_vol_thres_open=dprst_vol_thres_open[ii],
                    dprst_flow_coef=dprst_flow_coef[ii],
                    dprst_seep_rate_clos=dprst_seep_rate_clos[ii],
                    avail_et=avail_et,
                    dprst_in=dprst_in[ii],
                    srp=srp,
                    sri=sri,
                    sroff_ag=sroff_ag,
                    imperv_frac=imperv_frac,
                    perv_frac=perv_frac,
                    ag_frac=ag_frac[ii],
                )
                runoff = runoff + dprst_sroff_hru[ii] * hruarea

            # runoff for pervious, impervious, and agricultural areas
            srunoff = 0.0
            if hru_type[ii] == _LAND:
                runoff = runoff + srp * perv_area + sri * hruarea_imperv
                if ag_area[ii] > 0.0:
                    runoff = runoff + sroff_ag * ag_area[ii]
                srunoff = runoff / hruarea
                hru_sroffp[ii] = srp * perv_frac
                hru_sroff_ag[ii] = sroff_ag * ag_frac[ii]

            # evaporation from impervious area
            if hruarea_imperv > 0.0:
                if imperv_stor[ii] > 0.0:
                    imperv_stor[ii], imperv_evap[ii] = imperv_et(
                        imperv_stor[ii],
                        potet[ii],
                        imperv_evap[ii],
                        snowcov_area[ii],
                        avail_et,
                        imperv_frac,
                    )
                    hru_impervevap[ii] = imperv_evap[ii] * imperv_frac
                    avail_et = avail_et - hru_impervevap[ii]
                    if avail_et < 0.0:
                        hru_impervevap[ii] = hru_impervevap[ii] + avail_et
                        if hru_impervevap[ii] < 0.0:
                            hru_impervevap[ii] = 0.0
                        imperv_evap[ii] = imperv_evap[ii] / imperv_frac
                        imperv_stor[ii] = (
                            imperv_stor[ii] - avail_et / imperv_frac
                        )
                        avail_et = 0.0

                    hru_impervstor[ii] = imperv_stor[ii] * imperv_frac

                hru_sroffi[ii] = sri * imperv_frac

            if dprst_chk:
                dprst_stor_hru[ii] = (
                    dprst_vol_open[ii] + dprst_vol_clos[ii]
                ) / hruarea

            sroff[ii] = srunoff

    @staticmethod
    @numba.njit
    def _post_areas(
        # updated in place
        ag_area: np.ndarray,
        hru_perv: np.ndarray,
        hru_frac_perv: np.ndarray,
        infil_perv_hru: np.ndarray,
        infil_ag_hru: np.ndarray,
        infil_hru: np.ndarray,
        hru_impervstor_change: np.ndarray,
        dprst_stor_hru_change: np.ndarray,
        sroff_vol: np.ndarray,
        # read-only
        infil: np.ndarray,
        infil_ag: np.ndarray,
        ag_frac: np.ndarray,
        hru_area: np.ndarray,
        hru_imperv: np.ndarray,
        dprst_area_max: np.ndarray,
        hru_impervstor: np.ndarray,
        hru_impervstor_old: np.ndarray,
        dprst_stor_hru: np.ndarray,
        dprst_stor_hru_old: np.ndarray,
        sroff: np.ndarray,
        hru_in_to_cf: np.ndarray,
    ) -> None:
        """Upstream's post-kernel sequence, folded per element:
        _update_ag_areas (full recompute from the CURRENT ag_frac;
        dprst always active in this port) THEN the infil components --
        infil computed with the OLD pervious fraction is multiplied by
        the NEW one (upstream ordering quirk, preserved) -- and the
        change/volume lines."""
        for ii in range(hru_area.shape[0]):
            new_ag_area = ag_frac[ii] * hru_area[ii]
            new_hru_perv = hru_area[ii] - hru_imperv[ii] - new_ag_area
            new_hru_perv = new_hru_perv - dprst_area_max[ii]
            ag_area[ii] = new_ag_area
            hru_perv[ii] = new_hru_perv
            hru_frac_perv[ii] = hru_perv[ii] / hru_area[ii]

            infil_perv_hru[ii] = infil[ii] * hru_frac_perv[ii]
            infil_ag_hru[ii] = infil_ag[ii] * ag_frac[ii]
            infil_hru[ii] = infil_perv_hru[ii] + infil_ag_hru[ii]

            hru_impervstor_change[ii] = (
                hru_impervstor[ii] - hru_impervstor_old[ii]
            )
            dprst_stor_hru_change[ii] = (
                dprst_stor_hru[ii] - dprst_stor_hru_old[ii]
            )
            sroff_vol[ii] = sroff[ii] * hru_in_to_cf[ii]

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj

        if time.current_index == 0:
            # upstream itime_step==0 block: carve the ag area out of
            # basin_init's pervious area (ag_frac now fed; once-only
            # numpy staging)
            np.multiply(
                obj["ag_frac"].values,
                obj["hru_area"].values,
                out=obj["ag_area"].values,
            )
            obj["hru_perv"].values[:] -= obj["ag_area"].values
            np.divide(
                obj["hru_perv"].values,
                obj["hru_area"].values,
                out=obj["hru_frac_perv"].values,
            )

        self._calculate(
            obj["contrib_fraction"].values,
            obj["infil"].values,
            obj["infil_ag"].values,
            obj["sroff"].values,
            obj["hru_sroffp"].values,
            obj["hru_sroffi"].values,
            obj["hru_sroff_ag"].values,
            obj["imperv_stor"].values,
            obj["imperv_evap"].values,
            obj["hru_impervevap"].values,
            obj["hru_impervstor"].values,
            obj["dprst_vol_open"].values,
            obj["dprst_vol_clos"].values,
            obj["dprst_vol_open_frac"].values,
            obj["dprst_vol_clos_frac"].values,
            obj["dprst_vol_frac"].values,
            obj["dprst_area_open"].values,
            obj["dprst_area_clos"].values,
            obj["dprst_sroff_hru"].values,
            obj["dprst_seep_hru"].values,
            obj["dprst_evap_hru"].values,
            obj["dprst_insroff_hru"].values,
            obj["dprst_stor_hru"].values,
            obj["dprst_in"].values,
            obj["soil_lower_prev"].values,
            obj["soil_rechr_prev"].values,
            obj["ag_soil_moist_prev"].values,
            obj["ag_soil_rechr_prev"].values,
            obj["net_rain"].values,
            obj["net_ppt"].values,
            obj["net_snow"].values,
            obj["potet"].values,
            obj["snowmelt"].values,
            obj["snow_evap"].values,
            obj["pkwater_equiv"].values,
            obj["pptmix_nopack"].values,
            obj["snowcov_area"].values,
            obj["through_rain"].values,
            obj["hru_intcpevap"].values,
            obj["intcp_changeover"].values,
            obj["ag_frac"].values,
            obj["hru_type"].values,
            obj["hru_area"].values,
            obj["hru_percent_imperv"].values,
            obj["imperv_stor_max"].values,
            obj["carea_max"].values,
            obj["smidx_coef"].values,
            obj["smidx_exp"].values,
            obj["soil_moist_max"].values,
            obj["snowinfil_max"].values,
            obj["ag_soil_moist_max"].values,
            obj["ag_soil_rechr_max"].values,
            obj["dprst_et_coef"].values,
            obj["dprst_flow_coef"].values,
            obj["dprst_frac_open"].values,
            obj["dprst_seep_rate_clos"].values,
            obj["dprst_seep_rate_open"].values,
            obj["sro_to_dprst_imperv"].values,
            obj["sro_to_dprst_perv"].values,
            obj["va_open_exp"].values,
            obj["va_clos_exp"].values,
            obj["hru_perv"].values,
            obj["hru_frac_perv"].values,
            obj["hru_imperv"].values,
            obj["ag_area"].values,
            obj["dprst_area_max"].values,
            obj["dprst_area_open_max"].values,
            obj["dprst_area_clos_max"].values,
            obj["dprst_frac_clos"].values,
            obj["dprst_vol_open_max"].values,
            obj["dprst_vol_clos_max"].values,
            obj["dprst_vol_thres_open"].values,
            self._intcp_changeover_in_net_rain,
        )

        self._post_areas(
            obj["ag_area"].values,
            obj["hru_perv"].values,
            obj["hru_frac_perv"].values,
            obj["infil_perv_hru"].values,
            obj["infil_ag_hru"].values,
            obj["infil_hru"].values,
            obj["hru_impervstor_change"].values,
            obj["dprst_stor_hru_change"].values,
            obj["sroff_vol"].values,
            obj["infil"].values,
            obj["infil_ag"].values,
            obj["ag_frac"].values,
            obj["hru_area"].values,
            obj["hru_imperv"].values,
            obj["dprst_area_max"].values,
            obj["hru_impervstor"].values,
            obj["hru_impervstor_old"].values,
            obj["dprst_stor_hru"].values,
            obj["dprst_stor_hru_old"].values,
            obj["sroff"].values,
            obj["hru_in_to_cf"].values,
        )
