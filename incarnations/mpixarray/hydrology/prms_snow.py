"""
hydrology/prms_snow.py
======================
PRMSSnow: the PRMS snowpack, ported from pywatershed
(pywatershed/hydrology/prms_snow.py; PRMS 5.2.1 physics, PRMS-IV
documentation: Markstrom et al. 2015, USGS TM 6-B7).

Sixth REAL process port (July 2026) -- the largest and LOOSEST:
pywatershed's own snow autotest compares ONLY five variables (iso,
pkwater_equiv, snow_evap, tcal, through_rain) at rtol=atol=1e-3; every
other variable is commented out of its comparison list. Snow is deeply
branchy accumulated state -- threshold flips propagate -- so parity
expectations follow upstream's own.

Ported: field declarations (names verbatim) and the numerics of
``_calculate_numpy`` + its per-element helpers (``_calc_*`` -> module
``calc_*`` njit functions called directly, not passed as arguments),
rewritten to the in-place, out-first, zero-per-step-allocation
convention. Upstream's pre-loop array staging (canopy_covden, newsnow,
trd, the diagnostic resets) and post-loop staging (freeh2o/pk_ice
changes, the through_rain np.where chain) fold per element; the
post-loop block becomes a SECOND element loop because the main loop
``continue``s (LAKE / no-snow HRUs) skip to it.

Structural decisions:

- **soltab_horad_potsw is a (ndoy, space) PARAMETER**, not an input:
  it is a static table (pywatershed's netcdf reader indexes it by
  ``datetime_doy(current_time) - 1``); the kernel indexes
  ``[current_doy - 1, jj]``. Callers merge the (renamed) solar-geometry
  table into the parameters dataset.
- **(nmonth, space) parameters** (``tmax_allsnow`` -> derived
  ``tmax_allsnow_c``, ``cecn_coef``, ``tstorm_mo``) are indexed by
  ``current_month - 1`` in the kernel.
- **Scalar parameters** (dims ("scalar",) in the NHM files):
  ``albset_*``, ``den_init``, ``den_max``, ``settle_const`` are passed
  to the kernel as extracted floats (with ``deninv``/``denmaxinv``
  computed at call time) -- per-HRU variants of den_max/settle_const
  (upstream broadcasts) would surface as a dims mismatch, loudly.
- Time context: ``current_dowy``/``current_doy``/``current_month``
  from the Time clock (1-based, matching pywatershed time_utils).

Deliberately NOT ported: Budget/ConservativeProcess; adapters;
restart; calc_method; verbose (incl. all its dead debug prints);
``imbalance_behavior``; the ``doy`` parameter (declared upstream,
never used by its kernel); ``set_snow_zero`` (dead code);
``fastmath=True`` (strict IEEE here); the nonzero ``snowpack_init``
initialization block -- upstream's implementation of it is buggy
(transposed snarea_curve_2d indexing + a where-tuple indexing error)
and drb/nhm domains ship ``snowpack_init == 0`` everywhere, so
``initialize()`` raises NotImplementedError if any value is nonzero
rather than replicate the bug blind.

Parameter provenance: ``hru_type`` is DIS_HRU (dis-first);
``cov_type``/``covden_win``/``covden_sum``/``potet_sublim`` are shared
with the canopy/soilzone files (identical NHM values); the rest live
in parameters_PRMSSnow.nc.
"""

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process

# pywatershed constants
_NEARZERO = 1.0e-6  # nearzero
# dnearzero = epsilon64: pywatershed HARDCODES 2.23e-16
_DNEARZERO = 2.23e-16
_CLOSEZERO = 1.20e-07  # closezero = epsilon32 (hardcoded upstream)
_INCH2CM = 2.54
_LAKE = 2  # HruType.LAKE
_ONETHIRD = 1.0 / 3.0
_MAXALB = 15

# Albedo decay curves -- constants used like variables in PRMS6
# ("should probably be in the parameters", upstream's words)
_ACUM_INIT = np.array(
    [0.80, 0.77, 0.75, 0.72, 0.70, 0.69, 0.68, 0.67,
     0.66, 0.65, 0.64, 0.63, 0.62, 0.61, 0.60]
)  # fmt: skip
_AMLT_INIT = np.array(
    [0.72, 0.65, 0.60, 0.58, 0.56, 0.54, 0.52, 0.50,
     0.48, 0.46, 0.44, 0.43, 0.42, 0.41, 0.40]
)  # fmt: skip


# ----------------------------------------------------------------------
# Kernel helper functions -- pywatershed _calc_* staticmethods verbatim
# (scalar in/out; called directly, not passed as function arguments;
# verbose args and dead debug prints dropped)
# ----------------------------------------------------------------------


@numba.njit
def calc_sca_deplcrv(snarea_curve, frac_swe):
    """Interpolate along snow covered area depletion curve."""
    if frac_swe > 1.0:
        res = snarea_curve[-1]
    else:
        # indices (as integers) of the depletion curve that bracket the
        # given frac_swe (next highest and next lowest)
        idx = int(10.0 * (frac_swe + 0.2))  # [index]
        jdx = idx - 1  # [index]
        if idx > 11:
            idx = 11
        # fraction of the distance (from the next lowest) the given
        # frac_swe is between the next highest and lowest curve values
        dify = (frac_swe * 10.0) - float(jdx - 1)  # [fraction]
        # difference in snow covered area represented by next highest
        # and lowest curve values
        difx = snarea_curve[idx - 1] - snarea_curve[jdx - 1]
        res = snarea_curve[jdx - 1] + dify * difx

    return res


@numba.njit
def calc_calin(
    cal,
    den_max,
    denmaxinv,
    freeh2o,
    freeh2o_cap,
    iasw,
    pk_def,
    pk_den,
    pk_depth,
    pk_ice,
    pk_temp,
    pkwater_equiv,
    pss,
    pst,
    snowcov_area,
    snowmelt,
):
    """Compute changes in snowpack when a net gain in heat energy has
    occurred."""
    # difference between the incoming calories and the calories needed
    # to bring the pack to isothermal at 0 (heat deficit)
    dif = cal - pk_def  # [cal/cm^2]

    if dif < 0.0:
        # (1) Not enough heat to overcome heat deficit...
        pk_def = pk_def - cal  # [cal/cm^2]
        pk_temp = -1 * pk_def / (pkwater_equiv * 1.27)  # [degrees C]

    elif dif > 0.0:
        # (3) More than enough heat to overcome heat deficit (melt
        # ice)... 203.2 cal/(in cm^2) = latent heat of fusion 80 cal/cm3
        # * INCH2CM
        pmlt = dif / 203.2  # [inches]
        # potential snowmelt re-normalized to HRU area
        apmlt = pmlt * snowcov_area  # [inches]
        pk_def = 0.0  # [cal/cm^2]
        pk_temp = 0.0  # [degrees C]

        # pack ice re-normalized to the snowcovered area
        if snowcov_area > 0.0:
            apk_ice = pk_ice / snowcov_area  # [inches]
        else:
            apk_ice = 0.0

        if pmlt > apk_ice:
            # (3.1) Heat applied to snow covered area is sufficient to
            # melt all the ice in that snow pack: all pack water
            # equivalent becomes meltwater
            snowmelt = snowmelt + pkwater_equiv  # [inches]
            pkwater_equiv = 0.0  # [inches]
            iasw = False  # [flag]
            # set all snowpack states to 0 (snowcov_area unchanged)
            pk_def = 0.0
            pk_temp = 0.0
            pk_ice = 0.0
            freeh2o = 0.0
            pk_depth = 0.0
            pss = 0.0
            pst = 0.0
            pk_den = 0.0

        else:
            # (3.2) Heat only melts part of the ice in the snow pack
            pk_ice = pk_ice - apmlt  # [inches]
            freeh2o = freeh2o + apmlt  # [inches]
            # capacity of the snowpack to hold free water
            pwcap = freeh2o_cap * pk_ice  # [inches]
            # free water in excess of the capacity
            dif_water = freeh2o - pwcap  # [inches]

            if dif_water > 0.0:
                if dif_water > pkwater_equiv:
                    dif_water = pkwater_equiv

                pkwater_equiv = pkwater_equiv - dif_water  # [inches]
                freeh2o = pwcap  # [inches]
                if pk_den > 0.0:
                    pk_depth = pkwater_equiv / pk_den  # [inches]
                else:
                    # (mixed event on no existing snowpack: no density
                    # calculated yet)
                    pk_den = den_max
                    pk_depth = pkwater_equiv * denmaxinv  # [inches]

                snowmelt = snowmelt + dif_water  # [inches]
                pss = pkwater_equiv  # [inches]

    else:
        # (2) Just enough heat to overcome heat deficit: the pack is
        # "ripe"
        pk_temp = 0.0  # [degrees C]
        pk_def = 0.0  # [cal/cm^2]

    if not (pkwater_equiv > 0.0):
        pk_den = 0.0

    return (
        freeh2o,
        iasw,
        pk_def,
        pk_den,
        pk_ice,
        pk_depth,
        pk_temp,
        pss,
        pst,
        snowmelt,
        pkwater_equiv,
    )


@numba.njit
def calc_caloss(
    cal,
    freeh2o,
    pk_def,
    pk_ice,
    pk_temp,
    pkwater_equiv,
):
    """Compute change in snowpack when a net loss in heat energy has
    occurred."""
    if freeh2o < _CLOSEZERO:
        # (1) No free water exists in pack: heat deficit increases
        pk_def = pk_def - cal  # [cal/cm^2]

    else:
        # (2) Free water exists in pack
        # total heat per area released by free water freezing
        calnd = freeh2o * 203.2  # [cal/cm^2]
        # heat in free water vs heat absorbable by new snow (cal is
        # negative)
        dif = cal + calnd  # [cal/cm^2]

        if dif > 0.0:
            # only part of free water freezes
            pk_ice = pk_ice - (cal / 203.2)  # [inches]
            freeh2o = freeh2o + (cal / 203.2)  # [inches]
            return (
                freeh2o,
                pk_def,
                pk_ice,
                pk_temp,
                pkwater_equiv,
            )

        else:
            # all free water freezes; remaining absorbable heat becomes
            # the new pack heat deficit
            if dif < 0.0:
                pk_def = -dif  # [cal/cm^2]
            pk_ice = pk_ice + freeh2o  # [inches]
            freeh2o = 0.0  # [inches]

    if pkwater_equiv > 0.0:
        pk_temp = -1 * pk_def / (pkwater_equiv * 1.27)  # [degrees C]
    elif pkwater_equiv < 0.0:
        pkwater_equiv = 0.0

    return (
        freeh2o,
        pk_def,
        pk_ice,
        pk_temp,
        pkwater_equiv,
    )


@numba.njit
def calc_ppt_to_pack(
    den_max,
    denmaxinv,
    freeh2o,
    freeh2o_cap,
    iasw,
    net_ppt,
    net_rain,
    net_snow,
    pk_def,
    pk_den,
    pk_depth,
    pk_ice,
    pk_precip,
    pk_temp,
    pkwater_equiv,
    pptmix,
    pptmix_nopack,
    pss,
    pst,
    snowcov_area,
    snowmelt,
    tavgc,
    tmax_allsnow_c_current,
    tmaxc,
    tminc,
):
    """Add rain and/or snow to snowpack."""
    ppt_through = not (
        (pkwater_equiv > 0.0 and net_ppt > 0.0) or net_snow > 0.0
    )
    if ppt_through:
        return (
            freeh2o,
            iasw,
            pk_def,
            pk_den,
            pk_depth,
            pk_ice,
            pk_precip,
            pk_temp,
            pkwater_equiv,
            pptmix_nopack,
            pss,
            pst,
            snowmelt,
        )

    tsnow = tavgc  # [degrees C]
    train = tavgc  # placeholder; set in both branches below

    if pptmix == 1:
        # (1) If precipitation is mixed... rain temperature is halfway
        # between the max temperature and the allsnow temperature
        train = (tmaxc + tmax_allsnow_c_current) * 0.5  # [degrees C]

        if pkwater_equiv > 0.0:
            # snow temperature: halfway between tmin and allsnow max
            tsnow = (tminc + tmax_allsnow_c_current) * 0.5  # [degrees C]
        elif pkwater_equiv < 0.0:
            # no existing snowpack: ignore negative snowpack
            pkwater_equiv = 0.0

    else:
        # (2) All snow or all rain: rain temperature is the average
        # temperature
        train = tavgc  # [degrees C]
        if train < _CLOSEZERO:
            # near freezing: halfway between tmax and allsnow max
            train = (tmaxc + tmax_allsnow_c_current) * 0.5  # [degrees C]

    if train < 0.0:
        train = 0.0  # [degrees C]
    if tsnow > 0.0:
        tsnow = 0.0  # [degrees C]

    # If snowpack already exists, add rain first, then add snow (in a
    # mixed event the rain comes first, turning to snow as temperature
    # drops).
    if pkwater_equiv > 0.0:
        # (1) net rain on an existing snowpack
        if net_rain > 0.0:
            pkwater_equiv = pkwater_equiv + net_rain  # [inches]
            pk_precip = pk_precip + net_rain  # [inches]

            if pk_def > 0.0:
                # (1.1) snowpack is colder than freezing...
                # calories given up per inch of rain cooling to 0 degC
                # and freezing (80 cal/cm^3 latent + specific heat)
                caln = (80.000 + train) * _INCH2CM  # [cal/(in cm^2)]
                # rain needed to bring the snowpack to isothermal at 0
                pndz = pk_def / caln  # [inches]

                if abs(net_rain - pndz) < _CLOSEZERO:
                    # (1.1.1) exactly enough rain
                    pk_def = 0.0  # [cal/cm^2]
                    pk_temp = 0.0  # [degrees C]
                    pk_ice = pk_ice + net_rain  # [inches]

                elif net_rain < pndz:
                    # (1.1.2) not sufficient: deficit decreases, all
                    # rain freezes (1.27 = specific heat of ice * cm/in)
                    pk_def = pk_def - (caln * net_rain)
                    pk_temp = -1 * pk_def / (pkwater_equiv * 1.27)
                    pk_ice = pk_ice + net_rain

                else:
                    # (1.1.3) rain in excess of the isothermal amount
                    pk_def = 0.0
                    pk_temp = 0.0
                    pk_ice = pk_ice + pndz
                    # the rest becomes free water (no prior freeh2o:
                    # the pack had a heat deficit)
                    freeh2o = net_rain - pndz
                    # excess heat from the extra rain
                    calpr = train * (net_rain - pndz) * _INCH2CM  # [cal/cm^2]
                    (
                        freeh2o,
                        iasw,
                        pk_def,
                        pk_den,
                        pk_ice,
                        pk_depth,
                        pk_temp,
                        pss,
                        pst,
                        snowmelt,
                        pkwater_equiv,
                    ) = calc_calin(
                        cal=calpr,
                        den_max=den_max,
                        denmaxinv=denmaxinv,
                        freeh2o=freeh2o,
                        freeh2o_cap=freeh2o_cap,
                        iasw=iasw,
                        pk_def=pk_def,
                        pk_den=pk_den,
                        pk_depth=pk_depth,
                        pk_ice=pk_ice,
                        pk_temp=pk_temp,
                        pkwater_equiv=pkwater_equiv,
                        pss=pss,
                        pst=pst,
                        snowcov_area=snowcov_area,
                        snowmelt=snowmelt,
                    )

            else:
                # (1.2) rain on an isothermal snowpack: all net rain
                # becomes free water
                freeh2o = freeh2o + net_rain
                calpr = train * net_rain * _INCH2CM  # [cal/cm^2]
                (
                    freeh2o,
                    iasw,
                    pk_def,
                    pk_den,
                    pk_ice,
                    pk_depth,
                    pk_temp,
                    pss,
                    pst,
                    snowmelt,
                    pkwater_equiv,
                ) = calc_calin(
                    cal=calpr,
                    den_max=den_max,
                    denmaxinv=denmaxinv,
                    freeh2o=freeh2o,
                    freeh2o_cap=freeh2o_cap,
                    iasw=iasw,
                    pk_def=pk_def,
                    pk_den=pk_den,
                    pk_depth=pk_depth,
                    pk_ice=pk_ice,
                    pk_temp=pk_temp,
                    pkwater_equiv=pkwater_equiv,
                    pss=pss,
                    pst=pst,
                    snowcov_area=snowcov_area,
                    snowmelt=snowmelt,
                )

    elif net_rain > 0.0:
        # (2) net rain but no snowpack: flag a mix on no snowpack
        pptmix_nopack = 1.0  # [flag]

    # Net snow (with or without a pack)...
    if net_snow > 0.0:
        pkwater_equiv = pkwater_equiv + net_snow
        pk_precip = pk_precip + net_snow
        pk_ice = pk_ice + net_snow

        if tsnow >= 0.0:
            # (1) new snow at least 0 degC: heat content unchanged,
            # temperature "spreads out" over more snow
            pk_temp = -1 * pk_def / (pkwater_equiv * 1.27)  # [degrees C]

        else:
            # (2) new snow colder than 0 degC: heat absorbed warming
            # the new snow to 0 (negative of its heat deficit)
            calps = tsnow * net_snow * 1.27  # [cal/cm^2]

            if freeh2o > 0.0:
                # (2.1) free water in the pack: some will freeze
                (
                    freeh2o,
                    pk_def,
                    pk_ice,
                    pk_temp,
                    pkwater_equiv,
                ) = calc_caloss(
                    cal=calps,
                    freeh2o=freeh2o,
                    pk_def=pk_def,
                    pk_ice=pk_ice,
                    pk_temp=pk_temp,
                    pkwater_equiv=pkwater_equiv,
                )
            else:
                # (2.2) no free water: heat deficit increases
                pk_def = pk_def - calps  # [cal/cm^2]
                pk_temp = -1 * pk_def / (pkwater_equiv * 1.27)

    return (
        freeh2o,
        iasw,
        pk_def,
        pk_den,
        pk_depth,
        pk_ice,
        pk_precip,
        pk_temp,
        pkwater_equiv,
        pptmix_nopack,
        pss,
        pst,
        snowmelt,
    )


@numba.njit
def calc_snowcov(
    ai,
    frac_swe,
    iasw,
    net_snow,
    newsnow,
    pksv,
    pkwater_equiv,
    pst,
    scrv,
    snarea_curve,
    snarea_thresh,
    snowcov_area,
    snowcov_areasv,
):
    """Compute snow-covered area."""
    snowcov_area_ante = snowcov_area

    # reset snowcover area to the maximum
    snowcov_area = snarea_curve[11 - 1]  # [fraction of area]

    # track the maximum pack water equivalent for the current pack
    if pkwater_equiv > pst:
        pst = pkwater_equiv  # [inches]

    # ai = maximum packwater equivalent, capped at the complete-cover
    # threshold
    ai = pst  # [inches]
    if ai > snarea_thresh:
        ai = snarea_thresh  # [inches]

    if ai > _DNEARZERO:
        frac_swe = pkwater_equiv / ai  # [fraction]
        frac_swe = min(1.0, frac_swe)
    else:
        frac_swe = 0.0

    # Three curve conditions: (A) accumulating at maximum, (B)
    # depleting on the curve, (C) new snow on a depleting pack
    # (interpolated between 100% and the pre-new-snow area; 1/4 of the
    # new snow melts before cover drops below 100%).
    if pkwater_equiv >= ai:
        # (1) at the maximum: stay on the curve
        iasw = False

    else:
        # (2) below the maximum
        if newsnow:
            # (2.1) new snow...
            if iasw:
                # (2.1.1) already interpolating: track pack + 3/4 of
                # the new snow
                scrv = scrv + (0.75 * net_snow)  # [inches]
            else:
                # (2.1.2) currently on the curve: switch to
                # interpolation, saving the pre-new-snow state
                iasw = True  # [flag]
                snowcov_areasv = snowcov_area_ante  # [fraction]
                pksv = pkwater_equiv - net_snow  # [inches]
                scrv = pkwater_equiv - (0.25 * net_snow)  # [inches]

            # new snow always -> 100% cover (already set above)
            return (
                ai,
                frac_swe,
                iasw,
                pksv,
                pst,
                scrv,
                snowcov_area,
                snowcov_areasv,
            )

        elif iasw:
            # (2.2) no new snow, interpolating from a previous one...
            if pkwater_equiv > scrv:
                # first 1/4 of previous new snow not melted: still 100%
                return (
                    ai,
                    frac_swe,
                    iasw,
                    pksv,
                    pst,
                    scrv,
                    snowcov_area,
                    snowcov_areasv,
                )

            if pkwater_equiv >= pksv:
                # (2.2.1) new snow not melted back to original area:
                # interpolate between 100% and the pre-new-snow area
                difx = snowcov_area - snowcov_areasv
                dify = scrv - pksv  # [inches] (3/4 of previous new snow)
                fracy = 0.0  # [fraction]
                if dify > 0.0:
                    fracy = (pkwater_equiv - pksv) / dify  # [fraction]
                snowcov_area = snowcov_areasv + fracy * difx
                return (
                    ai,
                    frac_swe,
                    iasw,
                    pksv,
                    pst,
                    scrv,
                    snowcov_area,
                    snowcov_areasv,
                )

            else:
                # (2.2.2) back to the pre-new-snow water equivalent:
                # back on the curve
                iasw = False  # [flag]

        # adjust snow covered area along the depletion curve
        snowcov_area = calc_sca_deplcrv(snarea_curve, frac_swe)

    return (
        ai,
        frac_swe,
        iasw,
        pksv,
        pst,
        scrv,
        snowcov_area,
        snowcov_areasv,
    )


@numba.njit
def calc_snalbedo(
    albedo,
    albset_rna,
    albset_rnm,
    albset_sna,
    albset_snm,
    int_alb,
    iso,
    lst,
    net_snow,
    newsnow,
    pptmix,
    prmx,
    salb,
    slst,
    snsv,
):
    """Compute snowpack albedo."""
    # Albedo resets to a new (high) value with new snow above a
    # threshold, then decays with days since last snow; the decay curve
    # differs between accumulation and melt season.
    if not newsnow:
        # (1) no new snow: check for previous shallow new snow (lst)
        if lst:
            # set the albedo curve back three days
            slst = salb - 3.0  # [days]
            if slst < 1.0:
                slst = 1.0  # [days]

            if iso != 2:
                # not in melt season (unreachable upstream: lst is only
                # set during melt season; kept verbatim)
                if slst > 5.0:
                    slst = 5.0  # [days]

            lst = False  # [flag]
            snsv = 0.0  # [inches]

    elif iso == 2:
        # (2) new snow during the melt season
        if prmx < albset_rnm:
            # rain fraction does not prevent an albedo reset
            if net_snow > albset_snm:
                # (2.1) enough new snow to reset the albedo
                slst = 0.0  # [days]
                lst = False  # [flag]
                snsv = 0.0  # [inches]

            else:
                # (2.2) not enough new snow this time period alone
                snsv = snsv + net_snow  # [inches]

                if snsv > albset_snm:
                    # (2.2.1) accumulated shallow snow resets albedo
                    slst = 0.0  # [days]
                    lst = False  # [flag]
                    snsv = 0.0  # [inches]

                else:
                    # (2.2.2) not enough accumulated shallow snow
                    if not lst:
                        salb = slst  # [days]

                    slst = 0.0  # [days]
                    lst = True  # [flag]

    else:
        # (3) new snow during the accumulation season
        if pptmix == 0:
            # (3.1) snow-only event: always reset
            slst = 0.0  # [days]
            lst = False  # [flag]

        elif prmx >= albset_rna:
            # (3.2) mixed with too much rain: no reset, keep decaying
            lst = False  # [flag]

        elif net_snow >= albset_sna:
            # (3.3) mixed with enough snow: reset
            slst = 0.0  # [days]
            lst = False  # [flag]

        else:
            # (3.4) mixed, not enough snow: set the curve back 3 days
            slst = slst - 3.0  # [days]
            if slst < 0.0:
                slst = 0.0  # [days]
            if slst > 5.0:
                slst = 5.0  # [days]
            lst = False  # [flag]

        snsv = 0.0  # [inches]

    # days (or effective days) since last snowfall
    ll = int(slst + 0.5)  # [days]
    slst = slst + 1.0  # [days]

    # ****** compute albedo
    if ll > 0:
        # (1) more than 0 days since the last new snow
        if int_alb == 2:
            # (1.1) melt season curve
            if ll > _MAXALB:
                ll = _MAXALB  # [days]
            albedo = _AMLT_INIT[ll - 1]  # [fraction of radiation]

        elif ll <= _MAXALB:
            # (1.2) accumulation season curve, within the curve
            albedo = _ACUM_INIT[ll - 1]  # [fraction of radiation]

        else:
            # (1.3) accumulation curve exhausted: switch to the melt
            # curve at 12 days previous
            ll = ll - 12  # [days]
            if ll > _MAXALB:
                ll = _MAXALB  # [days]
            albedo = _AMLT_INIT[ll - 1]  # [fraction of radiation]

    elif iso == 2:
        # (2) new snow reset during melt season
        albedo = 0.72  # [fraction of radiation]
        int_alb = 2  # [flag]

    else:
        # (3) new snow reset during accumulation season
        albedo = 0.91  # [fraction of radiation]
        int_alb = 1  # [flag]

    return (
        albedo,
        int_alb,
        lst,
        salb,
        slst,
        snsv,
    )


@numba.njit
def calc_snowbal(
    niteda,
    cec,
    cst,
    esv,
    sw,
    temp,
    trd,
    canopy_covden,
    den_max,
    denmaxinv,
    emis_noppt,
    freeh2o,
    freeh2o_cap,
    hru_ppt,
    iasw,
    pk_def,
    pk_den,
    pk_depth,
    pk_ice,
    pk_temp,
    pkwater_equiv,
    pss,
    pst,
    snowcov_area,
    snowmelt,
    tcal,
    tstorm_mo,
):
    """Snowpack energy balance: 1st call is for the night period, 2nd
    call for the day period."""
    # potential long wave energy from air (black-body); Stefan
    # Boltzmann/2 = 0.585e-7 (half-day)
    air = 0.585e-7 * ((temp + 273.16) ** 4.0)  # [cal/cm^2]
    emis = esv  # [fraction of radiation]

    # snow surface temperature / longwave FROM the pack (cannot exceed
    # 0 degC)
    if temp < 0.0:
        ts = temp  # [degrees C]
        sno = air  # [cal/cm^2]
    else:
        ts = 0.0  # [degrees C]
        sno = 325.7  # [cal/cm^2]

    if hru_ppt > 0.0:
        # convective-thunderstorm precipitation resets the emissivity
        if tstorm_mo == 1:
            if niteda == 1:
                # (1) night
                emis = 0.85  # [fraction of radiation]
                if trd > _ONETHIRD:
                    emis = emis_noppt  # [fraction of radiation]
            else:
                # (2) day
                if trd > _ONETHIRD:
                    emis = 1.29 - (0.882 * trd)  # [fraction]
                if trd >= 0.5:
                    emis = 0.95 - (0.2 * trd)  # [fraction]

    # net incoming long wave: sky (uncovered) + canopy (perfect
    # blackbody) portions
    sky = (1.0 - canopy_covden) * ((emis * air) - sno)  # [cal/cm^2]
    can = canopy_covden * (air - sno)  # [cal/cm^2]

    # condensation/convection energy only when air above 0 degC with
    # precipitation
    cecsub = 0.0  # [cal/cm^2]
    if (temp > 0.0) and (hru_ppt > 0.0):
        cecsub = cec * temp  # [cal/cm^2]

    # total energy potentially available from atmosphere
    cal = sky + can + cecsub + sw  # [cal/cm^2]

    # surface at 0 degC with net incoming energy: apply directly
    if (ts >= 0.0) and (cal > 0.0):
        (
            freeh2o,
            iasw,
            pk_def,
            pk_den,
            pk_ice,
            pk_depth,
            pk_temp,
            pss,
            pst,
            snowmelt,
            pkwater_equiv,
        ) = calc_calin(
            cal=cal,
            den_max=den_max,
            denmaxinv=denmaxinv,
            freeh2o=freeh2o,
            freeh2o_cap=freeh2o_cap,
            iasw=iasw,
            pk_def=pk_def,
            pk_den=pk_den,
            pk_depth=pk_depth,
            pk_ice=pk_ice,
            pk_temp=pk_temp,
            pkwater_equiv=pkwater_equiv,
            pss=pss,
            pst=pst,
            snowcov_area=snowcov_area,
            snowmelt=snowmelt,
        )
        return (
            cal,
            freeh2o,
            iasw,
            pk_def,
            pk_den,
            pk_ice,
            pk_depth,
            pk_temp,
            pss,
            pst,
            snowmelt,
            pkwater_equiv,
        )

    # conductive heat flux between the deeper pack and its surface
    qcond = cst * (ts - pk_temp)  # [cal/cm^2]

    if qcond < 0.0:
        # (1) heat conducted from the snowpack to the surface
        if pk_temp < 0.0:
            pk_def = pk_def - qcond  # [cal/cm^2]
            pk_temp = -1 * pk_def / (pkwater_equiv * 1.27)  # [degC]
        else:
            (
                freeh2o,
                pk_def,
                pk_ice,
                pk_temp,
                pkwater_equiv,
            ) = calc_caloss(
                cal=qcond,
                freeh2o=freeh2o,
                pk_def=pk_def,
                pk_ice=pk_ice,
                pk_temp=pk_temp,
                pkwater_equiv=pkwater_equiv,
            )

    elif qcond < _CLOSEZERO:
        # (2) no heat conduction
        if pk_temp >= 0.0:
            # (unreachable upstream in current form; kept verbatim)
            if cal > 0.0:
                (
                    freeh2o,
                    iasw,
                    pk_def,
                    pk_den,
                    pk_ice,
                    pk_depth,
                    pk_temp,
                    pss,
                    pst,
                    snowmelt,
                    pkwater_equiv,
                ) = calc_calin(
                    cal=cal,
                    den_max=den_max,
                    denmaxinv=denmaxinv,
                    freeh2o=freeh2o,
                    freeh2o_cap=freeh2o_cap,
                    iasw=iasw,
                    pk_def=pk_def,
                    pk_den=pk_den,
                    pk_depth=pk_depth,
                    pk_ice=pk_ice,
                    pk_temp=pk_temp,
                    pkwater_equiv=pkwater_equiv,
                    pss=pss,
                    pst=pst,
                    snowcov_area=snowcov_area,
                    snowmelt=snowmelt,
                )

    elif ts >= 0.0:
        # (3) conduction into the pack, surface at 0 degC (cal <= 0
        # here)
        pk_defsub = pk_def - qcond
        if pk_defsub < 0.0:
            # deficit overcome: isothermal at 0 degC
            pk_def = 0.0  # [cal/cm^2]
            pk_temp = 0.0  # [degrees C]
        else:
            pk_def = pk_defsub  # [cal/cm^2]
            pk_temp = -pk_defsub / (pkwater_equiv * 1.27)  # [degC]

    else:
        # (4) conduction into the pack, surface below 0 degC
        pkt = -ts * (pkwater_equiv * 1.27)  # [cal/cm^2]
        pks = pk_def - pkt  # [cal/cm^2]
        pk_defsub = pks - qcond  # [cal/cm^2]

        if pk_defsub < 0.0:
            # (4.1) enough heat to bring the pack to the surface temp
            pk_def = pkt  # [cal/cm^2]
            pk_temp = ts  # [degrees C]
        else:
            # (4.2) not enough (equivalent to pk_def = pk_def - qcond)
            pk_def = pk_defsub + pkt  # [cal/cm^2]
            pk_temp = -1 * pk_def / (pkwater_equiv * 1.27)  # [degC]

    return (
        cal,
        freeh2o,
        iasw,
        pk_def,
        pk_den,
        pk_ice,
        pk_depth,
        pk_temp,
        pss,
        pst,
        snowmelt,
        pkwater_equiv,
    )


@numba.njit
def calc_step_4(
    trd,
    canopy_covden,
    albedo,
    cecn_coef,  # current-month, current-hru scalar
    cov_type,
    deninv,
    den_max,
    denmaxinv,
    emis_noppt,
    freeh2o,
    freeh2o_cap,
    hru_ppt,
    iasw,
    iso,
    lso,
    mso,
    net_snow,
    pk_def,
    pk_den,
    pk_depth,
    pk_ice,
    pk_temp,
    pkwater_equiv,
    pss,
    pst,
    rad_trncf,
    settle_const,
    snowcov_area,
    snowmelt,
    swrad,
    tavgc,
    tcal,
    tmaxc,
    tminc,
    tstorm_mo,  # current-month, current-hru scalar
):
    """Snowpack radiation fluxes / energy balance (PRMS "step 4")."""
    # emissivity of the air: no-precipitation value, reset to 1 with
    # any precipitation
    emis = emis_noppt  # [fraction of radiation]
    if hru_ppt > 0.0:
        emis = 1.0  # [fraction of radiation]
    esv = emis  # [fraction of radiation]

    # convection-condensation for a half-day interval; halved again for
    # trees
    cec = cecn_coef * 0.5  # [cal/(cm^2 degC)]
    if cov_type > 2:
        cec = cec * 0.5  # [cal/(cm^2 degC)]

    # forced spring melt bookkeeping: between melt-look and melt-force
    # days, melt season starts after the pack is isothermal at 0 degC
    # for more than 4 days
    if iso == 1:
        if mso == 2:
            if pk_temp >= 0.0:
                lso = lso + 1  # [days]
                if lso > 4:
                    iso = 2  # [flag]
                    lso = 0  # [days]
            else:
                lso = 0  # [days]

    # ---- night period energy balance ----
    niteda = 1  # [flag]
    temp = (tminc + tavgc) * 0.5
    swn = 0.0  # defined when pack exists below
    cst = 0.0

    if pkwater_equiv > 0.0:
        # incoming shortwave adjusted by albedo and winter-canopy
        # transmission
        swn = swrad * (1.0 - albedo) * rad_trncf  # [cal/cm^2]

        # new snow depth (Riley et al. 1973)
        pss = pss + net_snow  # [inches]
        dpt_before_settle = pk_depth + net_snow * deninv
        dpt1 = dpt_before_settle + settle_const * (
            (pss * denmaxinv) - dpt_before_settle
        )
        pk_depth = dpt1  # [inches]

        # snowpack density
        if dpt1 > 0.0:
            pk_den = pkwater_equiv / dpt1
        else:
            pk_den = 0.0  # [inch water equiv / inch depth]

        # effective conductivity term: (0.0077 * den^2)/(den * 0.5)
        effk = 0.0154 * pk_den  # [unitless]
        # 13751 = seconds in 12 hours over pi -> conductive heat
        # exchange per cm snow per cm^2 per degC for a half day
        cst = pk_den * (np.sqrt(effk * 13751.0))  # [cal/(cm^2 degC)]

        sw = 0.0  # no shortwave at night [cal/cm^2]

        (
            tcal,
            freeh2o,
            iasw,
            pk_def,
            pk_den,
            pk_ice,
            pk_depth,
            pk_temp,
            pss,
            pst,
            snowmelt,
            pkwater_equiv,
        ) = calc_snowbal(
            niteda=niteda,
            cec=cec,
            cst=cst,
            esv=esv,
            sw=sw,
            temp=temp,
            trd=trd,
            canopy_covden=canopy_covden,
            den_max=den_max,
            denmaxinv=denmaxinv,
            emis_noppt=emis_noppt,
            freeh2o=freeh2o,
            freeh2o_cap=freeh2o_cap,
            hru_ppt=hru_ppt,
            iasw=iasw,
            pk_def=pk_def,
            pk_den=pk_den,
            pk_depth=pk_depth,
            pk_ice=pk_ice,
            pk_temp=pk_temp,
            pkwater_equiv=pkwater_equiv,
            pss=pss,
            pst=pst,
            snowcov_area=snowcov_area,
            snowmelt=snowmelt,
            tcal=tcal,
            tstorm_mo=tstorm_mo,
        )

    # ---- day period energy balance (if the pack still exists) ----
    niteda = 2  # [flag]
    temp = (tmaxc + tavgc) * 0.5  # [degrees C]

    if pkwater_equiv > 0.0:
        sw = swn  # [cal/cm^2]
        (
            cals,
            freeh2o,
            iasw,
            pk_def,
            pk_den,
            pk_ice,
            pk_depth,
            pk_temp,
            pss,
            pst,
            snowmelt,
            pkwater_equiv,
        ) = calc_snowbal(
            niteda=niteda,
            cec=cec,
            cst=cst,
            esv=esv,
            sw=sw,
            temp=temp,
            trd=trd,
            canopy_covden=canopy_covden,
            den_max=den_max,
            denmaxinv=denmaxinv,
            emis_noppt=emis_noppt,
            freeh2o=freeh2o,
            freeh2o_cap=freeh2o_cap,
            hru_ppt=hru_ppt,
            iasw=iasw,
            pk_def=pk_def,
            pk_den=pk_den,
            pk_depth=pk_depth,
            pk_ice=pk_ice,
            pk_temp=pk_temp,
            pkwater_equiv=pkwater_equiv,
            pss=pss,
            pst=pst,
            snowcov_area=snowcov_area,
            snowmelt=snowmelt,
            tcal=tcal,
            tstorm_mo=tstorm_mo,
        )
        # total heat flux from both night and day periods
        tcal = tcal + cals  # [cal/cm^2]

    return (
        freeh2o,
        iasw,
        iso,
        lso,
        mso,
        pk_def,
        pk_den,
        pk_depth,
        pk_ice,
        pk_temp,
        pkwater_equiv,
        pss,
        tcal,
        snowmelt,
    )


@numba.njit
def calc_snowevap(
    freeh2o,
    hru_intcpevap,
    pk_def,
    pk_ice,
    pk_temp,
    pkwater_equiv,
    potet,
    potet_sublim,
    snow_evap,
    snowcov_area,
):
    """Snowpack loss to evaporation/sublimation."""
    # evaporation affecting the snowpack: potential minus canopy
    # interception evaporation
    ez = potet_sublim * potet * snowcov_area - hru_intcpevap  # [inches]

    if ez < _CLOSEZERO:
        # (1) no potential for evaporation
        snow_evap = 0.0  # [inches]

    elif ez >= pkwater_equiv:
        # (2) enough to entirely deplete the snowpack
        snow_evap = pkwater_equiv  # [inches]
        pkwater_equiv = 0.0  # [inches]
        pk_ice = 0.0  # [inches]
        freeh2o = 0.0  # [inches]
        pk_def = 0.0  # [cal/cm^2]
        pk_temp = 0.0  # [degrees C]

    else:
        # (3) partial depletion: sublimation removes ice
        pk_ice = pk_ice - ez
        if pk_ice < 0.0:
            # all ice removed -> no heat deficit (mass balance fix in
            # pywatershed's PRMS 5.2.1)
            freeh2o = freeh2o + pk_ice
            pk_ice = 0.0
            pk_def = 0.0
            pk_temp = 0.0
        else:
            # heat deficit removed by the sublimating ice (only
            # non-zero when pack temperature < 0 degC)
            cal = pk_temp * ez * 1.27
            pk_def = pk_def + cal

        pkwater_equiv = pkwater_equiv - ez
        snow_evap = ez

    if snow_evap < 0.0:
        pkwater_equiv = pkwater_equiv - snow_evap
        if pkwater_equiv < 0.0:
            pkwater_equiv = 0.0
        snow_evap = 0.0

    avail_et = potet - hru_intcpevap - snow_evap
    if avail_et < 0.0:
        snow_evap = snow_evap + avail_et
        pkwater_equiv = pkwater_equiv - avail_et

        if snow_evap < 0.0:
            pkwater_equiv = pkwater_equiv - snow_evap
            if pkwater_equiv < 0.0:
                pkwater_equiv = 0.0
            snow_evap = 0.0

    return (
        freeh2o,
        pk_def,
        pk_ice,
        pk_temp,
        pkwater_equiv,
        snow_evap,
    )


class PRMSSnow(Process):
    """PRMS snowpack: accumulation, snow-covered-area depletion,
    albedo decay, a two-period (night/day) energy balance, melt and
    sublimation per HRU.

    Storage and fluxes are in inches; energies in cal/cm^2 (Langleys).
    """

    # ------------------------------------------------------------------
    # Field declarations (names verbatim from pywatershed)
    # ------------------------------------------------------------------

    # -- dis_hru variables (grid-owned; dis-first sourcing) --
    hru_type = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="HRU type (INACTIVE=0, LAND=1, LAKE=2, SWALE=3)",
    )

    # -- process parameters (per HRU) --
    cov_type = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Cover type (0=bare, 1=grasses, 2=shrubs, 3=trees, "
        "4=coniferous)",
    )
    covden_win = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Winter vegetation cover density [-]",
    )
    covden_sum = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Summer vegetation cover density [-]",
    )
    emis_noppt = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Air emissivity when there is no precipitation [-]",
    )
    freeh2o_cap = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Free-water holding capacity as fraction of pack ice [-]",
    )
    hru_deplcrv = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Index (1-based) of the HRU's snow depletion curve",
    )
    melt_force = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Julian day to force snowpack to spring conditions",
    )
    melt_look = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Julian day to start looking for spring conditions",
    )
    potet_sublim = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of PET sublimated from the snow surface [-]",
    )
    rad_trncf = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Solar radiation transmission through winter canopy [-]",
    )
    snarea_thresh = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum SWE below which the depletion curve applies "
        "[inches]",
    )
    snowpack_init = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Initial snowpack water equivalent [inches] "
        "(must be zero -- see module docstring)",
    )
    # -- process parameters (monthly x HRU) --
    tmax_allsnow = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Maximum temperature for all-snow precipitation [degF]",
    )
    cecn_coef = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Convection-condensation energy coefficient "
        "[cal/(cm^2 degC)]",
    )
    tstorm_mo = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.int64,
        description="Convective-thunderstorm month flag (0/1)",
    )
    # -- process parameters (other dims) --
    snarea_curve = DataArrayMeta(
        kind="parameter",
        dims=("ndeplval",),
        dtype=np.float64,
        description="Snow-area depletion curves, flat (n_curves x 11); "
        "the kernel slices rows by hru_deplcrv",
    )
    # scalar parameters (dims ('scalar',) in the NHM files); extracted
    # as floats at calculate() time
    albset_rna = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Albedo-reset rain fraction threshold, accumulation [-]",
    )
    albset_rnm = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Albedo-reset rain fraction threshold, melt [-]",
    )
    albset_sna = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Albedo-reset snow threshold, accumulation [inches]",
    )
    albset_snm = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Albedo-reset snow threshold, melt [inches]",
    )
    den_init = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Initial (new) snow density [-]",
    )
    den_max = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Maximum snow density [-]",
    )
    settle_const = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Snowpack settlement-rate constant [1/day]",
    )
    # -- static solar table (solar-geometry product; see docstring) --
    soltab_horad_potsw = DataArrayMeta(
        kind="parameter",
        dims=("ndoy", "space"),
        dtype=np.float64,
        description="Potential shortwave on a horizontal plane per Julian "
        "day [cal/cm^2] -- static table indexed by current_doy",
        derivation="compute_soltabs(hru_slope, hru_aspect, hru_lat)",
    )

    # (tmax_allsnow_c: upstream converts the whole (nmonth, nhru)
    # array once at init; here the F->C conversion happens per element
    # in the kernel -- IEEE-identical, and it avoids a multi-dim
    # derived buffer, which mpixarray cannot declare: its buffer
    # creation decomposes EVERY declared dim, splitting nmonth across
    # ranks.)

    # -- inputs --
    hru_ppt = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Precipitation on the HRU [inches]",
    )
    hru_intcpevap = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="HRU area-weighted canopy evaporation [inches]",
    )
    net_ppt = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Precipitation through the canopy [inches]",
    )
    net_rain = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Rain through the canopy [inches]",
    )
    net_snow = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snow through the canopy [inches]",
    )
    orad_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Measured/estimated shortwave on a horizontal plane "
        "[cal/cm^2]",
    )
    potet = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Potential evapotranspiration [inches]",
    )
    pptmix = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Rain/snow mix flag (0/1)",
    )
    prmx = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of rain in a mixed event [-]",
    )
    swrad = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Shortwave radiation on the HRU [cal/cm^2]",
    )
    tavgc = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Average air temperature [degC]",
    )
    tmaxc = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Maximum air temperature [degC]",
    )
    tminc = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Minimum air temperature [degC]",
    )
    transp_on = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Transpiration occurring (0/1 flag)",
    )

    # -- variables --
    ai = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Maximum SWE for the current pack, capped [inches]",
    )
    albedo = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snow surface albedo [-]",
    )
    frac_swe = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="SWE as a fraction of the pack maximum (ai) [-]",
    )
    freeh2o = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Free liquid water in the snowpack [inches]",
    )
    freeh2o_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Free-water change over the timestep [inches]",
    )
    freeh2o_prev = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Free water, previous timestep (PRMSCanopy input)",
    )
    iasw = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.bool_,
        description="Interpolating SCA between 100% and the curve (flag)",
    )
    int_alb = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.int64,
        description="Albedo curve in use (1=accumulation, 2=melt)",
    )
    iso = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.int64,
        description="Melt-season state (1=before, 2=melt season)",
    )
    lso = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.int64,
        description="Days the pack has been isothermal at 0 degC [days]",
    )
    lst = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.bool_,
        description="Shallow new snow insufficient to reset albedo (flag)",
    )
    mso = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.int64,
        description="Melt-look state (1=before, 2=watching)",
    )
    newsnow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.bool_,
        description="New snow fell this timestep (flag)",
    )
    pk_def = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snowpack heat deficit [cal/cm^2]",
    )
    pk_den = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snowpack density [-]",
    )
    pk_depth = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snowpack depth [inches]",
    )
    pk_ice = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snowpack ice content [inches]",
    )
    pk_ice_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pack-ice change over the timestep [inches]",
    )
    pk_ice_prev = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pack ice, previous timestep (PRMSCanopy input)",
    )
    pk_precip = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Precipitation added to the snowpack [inches]",
    )
    pk_temp = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snowpack temperature [degC]",
    )
    pksv = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="SWE before the last new snow (SCA interpolation state)",
    )
    pkwater_equiv = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snowpack water equivalent [inches]",
    )
    pptmix_nopack = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Mixed event with no antecedent snowpack (0/1 "
        "flag; float64 -- PRMSRunoff consumes it as a float input on "
        "the shared grid dataset)",
    )
    pss = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Previous pack SWE plus new snow [inches]",
    )
    pst = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Maximum SWE for the current pack [inches]",
    )
    salb = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Days since last albedo-resetting snow (saved) [days]",
    )
    scrv = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Pack SWE plus 3/4 of new snow (SCA interpolation "
        "state) [inches]",
    )
    slst = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Days since last albedo-resetting snowfall [days]",
    )
    snow_evap = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Snowpack evaporation/sublimation [inches]",
    )
    snowcov_area = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Snow-covered area fraction [-]",
    )
    snowcov_areasv = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="SCA before the last new snow (interpolation state)",
    )
    snowmelt = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Snowmelt from the snowpack [inches]",
    )
    snsv = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Accumulated shallow new snow (albedo state) [inches]",
    )
    tcal = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Net snowpack energy balance [cal/cm^2]",
    )
    through_rain = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Rain that passes through snow [inches] "
        "(PRMSRunoff input)",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        obj = self._obj

        # zero-init all float variables; flags to their upstream inits
        for name in (
            "ai",
            "albedo",
            "frac_swe",
            "freeh2o",
            "freeh2o_change",
            "freeh2o_prev",
            "pk_def",
            "pk_den",
            "pk_depth",
            "pk_ice",
            "pk_ice_change",
            "pk_ice_prev",
            "pk_precip",
            "pk_temp",
            "pksv",
            "pkwater_equiv",
            "pss",
            "pst",
            "salb",
            "scrv",
            "slst",
            "snow_evap",
            "snowcov_area",
            "snowcov_areasv",
            "snowmelt",
            "snsv",
            "tcal",
            "through_rain",
        ):
            obj[name].values[:] = 0.0
        obj["iasw"].values[:] = False
        obj["lst"].values[:] = False
        obj["newsnow"].values[:] = False
        obj["pptmix_nopack"].values[:] = 0.0
        obj["int_alb"].values[:] = 1
        obj["iso"].values[:] = 1
        obj["mso"].values[:] = 1
        obj["lso"].values[:] = 0

        # snarea_curve sanity: flat length must be n_curves * 11
        if obj["snarea_curve"].values.shape[0] % 11 != 0:
            raise ValueError("snarea_curve length must be a multiple of 11")

        # nonzero snowpack_init requires upstream's (buggy) init block
        # -- not ported; see module docstring
        if (obj["snowpack_init"].values > 0.0).any():
            raise NotImplementedError(
                "PRMSSnow: nonzero snowpack_init is not ported (upstream "
                "initialization block is faulty; nhm domains use zero)"
            )
        # pkwater_equiv/pss/pst = snowpack_init = 0 (already zeroed)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        obj = self._obj
        obj["freeh2o_prev"].values[:] = obj["freeh2o"].values
        obj["pk_ice_prev"].values[:] = obj["pk_ice"].values

    @staticmethod
    @numba.njit
    def _calculate(
        # in/out state + outputs (written in place)
        ai: np.ndarray,
        albedo: np.ndarray,
        frac_swe: np.ndarray,
        freeh2o: np.ndarray,
        freeh2o_change: np.ndarray,
        iasw: np.ndarray,
        int_alb: np.ndarray,
        iso: np.ndarray,
        lso: np.ndarray,
        lst: np.ndarray,
        mso: np.ndarray,
        newsnow: np.ndarray,
        pk_def: np.ndarray,
        pk_den: np.ndarray,
        pk_depth: np.ndarray,
        pk_ice: np.ndarray,
        pk_ice_change: np.ndarray,
        pk_precip: np.ndarray,
        pk_temp: np.ndarray,
        pksv: np.ndarray,
        pkwater_equiv: np.ndarray,
        pptmix_nopack: np.ndarray,
        pss: np.ndarray,
        pst: np.ndarray,
        salb: np.ndarray,
        scrv: np.ndarray,
        slst: np.ndarray,
        snow_evap: np.ndarray,
        snowcov_area: np.ndarray,
        snowcov_areasv: np.ndarray,
        snowmelt: np.ndarray,
        snsv: np.ndarray,
        tcal: np.ndarray,
        through_rain: np.ndarray,
        # prior state (read-only here; advance() maintains)
        freeh2o_prev: np.ndarray,
        pk_ice_prev: np.ndarray,
        # inputs
        hru_ppt: np.ndarray,
        hru_intcpevap: np.ndarray,
        net_ppt: np.ndarray,
        net_rain: np.ndarray,
        net_snow: np.ndarray,
        orad_hru: np.ndarray,
        potet: np.ndarray,
        pptmix: np.ndarray,
        prmx: np.ndarray,
        swrad: np.ndarray,
        tavgc: np.ndarray,
        tmaxc: np.ndarray,
        tminc: np.ndarray,
        transp_on: np.ndarray,
        # parameters + derived
        hru_type: np.ndarray,
        cov_type: np.ndarray,
        covden_win: np.ndarray,
        covden_sum: np.ndarray,
        emis_noppt: np.ndarray,
        freeh2o_cap: np.ndarray,
        hru_deplcrv: np.ndarray,
        melt_force: np.ndarray,
        melt_look: np.ndarray,
        potet_sublim: np.ndarray,
        rad_trncf: np.ndarray,
        snarea_curve: np.ndarray,
        snarea_thresh: np.ndarray,
        tmax_allsnow: np.ndarray,
        cecn_coef: np.ndarray,
        tstorm_mo: np.ndarray,
        soltab_horad_potsw: np.ndarray,
        # scalar parameters + time context
        albset_rna: np.float64,
        albset_rnm: np.float64,
        albset_sna: np.float64,
        albset_snm: np.float64,
        den_max: np.float64,
        deninv: np.float64,
        denmaxinv: np.float64,
        settle_const: np.float64,
        current_dowy: np.int64,
        current_doy: np.int64,
        current_month: np.int64,
    ) -> None:
        nhru = pkwater_equiv.shape[0]
        month_ind = current_month - 1

        for jj in range(nhru):
            # upstream pre-loop array staging, per element (applies to
            # ALL HRUs, before the lake/no-snow continues)
            newsnow[jj] = net_snow[jj] > 0.0
            frac_swe[jj] = 0.0
            pk_precip[jj] = 0.0  # [inches]
            snowmelt[jj] = 0.0  # [inches]
            snow_evap[jj] = 0.0  # [inches]
            tcal[jj] = 0.0
            ai[jj] = 0.0

            if hru_type[jj] == _LAKE:
                continue

            # first day of the water year: reset seasonal state
            if current_dowy == 1:
                pss[jj] = 0.0  # [inches]
                iso[jj] = 1  # [flag]
                mso[jj] = 1  # [flag]
                lso[jj] = 0  # [counter]

            # default assumption
            pptmix_nopack[jj] = 0.0

            # forced melt / melt-look season flags
            if current_doy == melt_force[jj]:
                iso[jj] = 2  # [flag]
            if current_doy == melt_look[jj]:
                mso[jj] = 2  # [flag]

            if (pkwater_equiv[jj] < _DNEARZERO) and (not newsnow[jj]):
                # no snowpack and no new snow: reset and skip
                snowcov_area[jj] = 0.0
                continue

            if newsnow[jj] and (pkwater_equiv[jj] < _DNEARZERO):
                snowcov_area[jj] = 1.0

            # seasonal canopy cover density (upstream pre-loop array)
            if transp_on[jj] != 0.0:
                canopy_covden = covden_sum[jj]
            else:
                canopy_covden = covden_win[jj]

            # STEP 1: precipitation effects on pack water/heat content
            (
                freeh2o[jj],
                iasw[jj],
                pk_def[jj],
                pk_den[jj],
                pk_depth[jj],
                pk_ice[jj],
                pk_precip[jj],
                pk_temp[jj],
                pkwater_equiv[jj],
                pptmix_nopack[jj],
                pss[jj],
                pst[jj],
                snowmelt[jj],
            ) = calc_ppt_to_pack(
                den_max=den_max,
                denmaxinv=denmaxinv,
                freeh2o=freeh2o[jj],
                freeh2o_cap=freeh2o_cap[jj],
                iasw=iasw[jj],
                net_ppt=net_ppt[jj],
                net_rain=net_rain[jj],
                net_snow=net_snow[jj],
                pk_def=pk_def[jj],
                pk_den=pk_den[jj],
                pk_depth=pk_depth[jj],
                pk_ice=pk_ice[jj],
                pk_precip=pk_precip[jj],
                pk_temp=pk_temp[jj],
                pkwater_equiv=pkwater_equiv[jj],
                pptmix=pptmix[jj],
                pptmix_nopack=pptmix_nopack[jj],
                pss=pss[jj],
                pst=pst[jj],
                snowcov_area=snowcov_area[jj],
                snowmelt=snowmelt[jj],
                tavgc=tavgc[jj],
                tmax_allsnow_c_current=(
                    (tmax_allsnow[month_ind, jj] - 32.0) / 1.8
                ),
                tmaxc=tmaxc[jj],
                tminc=tminc[jj],
            )

            if pkwater_equiv[jj] > 0.0:
                # STEP 2: snow covered area from the depletion curve
                crv0 = (hru_deplcrv[jj] - 1) * 11
                (
                    ai[jj],
                    frac_swe[jj],
                    iasw[jj],
                    pksv[jj],
                    pst[jj],
                    scrv[jj],
                    snowcov_area[jj],
                    snowcov_areasv[jj],
                ) = calc_snowcov(
                    ai=ai[jj],
                    frac_swe=frac_swe[jj],
                    iasw=iasw[jj],
                    net_snow=net_snow[jj],
                    newsnow=newsnow[jj],
                    pksv=pksv[jj],
                    pkwater_equiv=pkwater_equiv[jj],
                    pst=pst[jj],
                    scrv=scrv[jj],
                    snarea_curve=snarea_curve[crv0 : crv0 + 11],
                    snarea_thresh=snarea_thresh[jj],
                    snowcov_area=snowcov_area[jj],
                    snowcov_areasv=snowcov_areasv[jj],
                )

                # STEP 3: albedo
                (
                    albedo[jj],
                    int_alb[jj],
                    lst[jj],
                    salb[jj],
                    slst[jj],
                    snsv[jj],
                ) = calc_snalbedo(
                    albedo=albedo[jj],
                    albset_rna=albset_rna,
                    albset_rnm=albset_rnm,
                    albset_sna=albset_sna,
                    albset_snm=albset_snm,
                    int_alb=int_alb[jj],
                    iso=iso[jj],
                    lst=lst[jj],
                    net_snow=net_snow[jj],
                    newsnow=newsnow[jj],
                    pptmix=pptmix[jj],
                    prmx=prmx[jj],
                    salb=salb[jj],
                    slst=slst[jj],
                    snsv=snsv[jj],
                )

            if pkwater_equiv[jj] > 0.0:
                # STEP 4: radiation fluxes and energy balance
                trd = orad_hru[jj] / soltab_horad_potsw[current_doy - 1, jj]
                (
                    freeh2o[jj],
                    iasw[jj],
                    iso[jj],
                    lso[jj],
                    mso[jj],
                    pk_def[jj],
                    pk_den[jj],
                    pk_depth[jj],
                    pk_ice[jj],
                    pk_temp[jj],
                    pkwater_equiv[jj],
                    pss[jj],
                    tcal[jj],
                    snowmelt[jj],
                ) = calc_step_4(
                    trd,
                    canopy_covden=canopy_covden,
                    albedo=albedo[jj],
                    cecn_coef=cecn_coef[month_ind, jj],
                    cov_type=cov_type[jj],
                    deninv=deninv,
                    den_max=den_max,
                    denmaxinv=denmaxinv,
                    emis_noppt=emis_noppt[jj],
                    freeh2o=freeh2o[jj],
                    freeh2o_cap=freeh2o_cap[jj],
                    hru_ppt=hru_ppt[jj],
                    iasw=iasw[jj],
                    iso=iso[jj],
                    lso=lso[jj],
                    mso=mso[jj],
                    net_snow=net_snow[jj],
                    pk_def=pk_def[jj],
                    pk_den=pk_den[jj],
                    pk_depth=pk_depth[jj],
                    pk_ice=pk_ice[jj],
                    pk_temp=pk_temp[jj],
                    pkwater_equiv=pkwater_equiv[jj],
                    pss=pss[jj],
                    pst=pst[jj],
                    rad_trncf=rad_trncf[jj],
                    settle_const=settle_const,
                    snowcov_area=snowcov_area[jj],
                    snowmelt=snowmelt[jj],
                    swrad=swrad[jj],
                    tavgc=tavgc[jj],
                    tcal=tcal[jj],
                    tmaxc=tmaxc[jj],
                    tminc=tminc[jj],
                    tstorm_mo=tstorm_mo[month_ind, jj],
                )

                # STEP 5: snowpack loss to evaporation
                if pkwater_equiv[jj] > 0.0:
                    # snow can evaporate when transpiration is off, or
                    # on with bare-soil/grass cover
                    if (transp_on[jj] == 0.0) or (
                        transp_on[jj] != 0.0 and cov_type[jj] < 2
                    ):
                        (
                            freeh2o[jj],
                            pk_def[jj],
                            pk_ice[jj],
                            pk_temp[jj],
                            pkwater_equiv[jj],
                            snow_evap[jj],
                        ) = calc_snowevap(
                            freeh2o=freeh2o[jj],
                            hru_intcpevap=hru_intcpevap[jj],
                            pk_def=pk_def[jj],
                            pk_ice=pk_ice[jj],
                            pk_temp=pk_temp[jj],
                            pkwater_equiv=pkwater_equiv[jj],
                            potet=potet[jj],
                            potet_sublim=potet_sublim[jj],
                            snow_evap=snow_evap[jj],
                            snowcov_area=snowcov_area[jj],
                        )

                elif pkwater_equiv[jj] < 0.0:
                    # ignore negative values
                    pkwater_equiv[jj] = 0.0

                # CLEAN-UP: final pack states
                if pkwater_equiv[jj] > 0.0:
                    if pk_den[jj] > 0.0:
                        pk_depth[jj] = pkwater_equiv[jj] / pk_den[jj]
                    else:
                        pk_den[jj] = den_max
                        pk_depth[jj] = pkwater_equiv[jj] * denmaxinv

                    pss[jj] = pkwater_equiv[jj]

                    # during melt with insufficient albedo-reset snow,
                    # reduce cumulative new snow by the melt
                    if lst[jj]:
                        snsv[jj] = snsv[jj] - snowmelt[jj]
                        if snsv[jj] < 0.0:
                            snsv[jj] = 0.0

            # LAST check: clear all state if the pack is gone
            if pkwater_equiv[jj] <= _DNEARZERO:
                pkwater_equiv[jj] = 0.0
                pk_depth[jj] = 0.0
                pss[jj] = 0.0
                snsv[jj] = 0.0
                lst[jj] = False
                pst[jj] = 0.0
                iasw[jj] = False
                albedo[jj] = 0.0
                pk_den[jj] = 0.0
                snowcov_area[jj] = 0.0
                pk_def[jj] = 0.0
                pk_temp[jj] = 0.0
                pk_ice[jj] = 0.0
                freeh2o[jj] = 0.0
                snowcov_areasv[jj] = 0.0  # (not in original PRMS)
                ai[jj] = 0.0
                frac_swe[jj] = 0.0
                scrv[jj] = 0.0
                pksv[jj] = 0.0

        # upstream post-loop array staging, per element (a SECOND loop:
        # the continues above skip to here; applies to ALL HRUs)
        for jj in range(nhru):
            freeh2o_change[jj] = freeh2o[jj] - freeh2o_prev[jj]
            pk_ice_change[jj] = pk_ice[jj] - pk_ice_prev[jj]

            cond1 = net_ppt[jj] > 0.0
            cond2 = pptmix_nopack[jj] != 0.0
            cond3 = snowmelt[jj] < _NEARZERO
            cond4 = pkwater_equiv[jj] < _DNEARZERO
            cond5 = snow_evap[jj] < _NEARZERO
            cond6 = net_snow[jj] < _NEARZERO
            cond7 = snow_evap[jj] > (
                -1 * (pk_ice_change[jj] + freeh2o_change[jj])
            )

            # np.where chain, reverse order from the PRMS ifs
            through_rain[jj] = 0.0
            if cond1 and cond3 and cond4 and cond6:
                through_rain[jj] = net_rain[jj]
            if cond1 and cond3 and cond4 and cond5:
                through_rain[jj] = net_ppt[jj]
            if cond1 and cond2:
                through_rain[jj] = net_rain[jj]
            # not in PRMS, needed for mass balance: rain on snow (no
            # new snow) where snow_evap consumes the pack this timestep
            if cond1 and cond6 and cond7:
                through_rain[jj] = 0.0

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        # scalar ('scalar',)-dim parameters, extracted once per step
        # (cheap); deninv/denmaxinv derived from them (upstream
        # _set_initial_conditions)
        den_init = np.float64(obj["den_init"].values[0])
        den_max = np.float64(obj["den_max"].values[0])
        self._calculate(
            obj["ai"].values,
            obj["albedo"].values,
            obj["frac_swe"].values,
            obj["freeh2o"].values,
            obj["freeh2o_change"].values,
            obj["iasw"].values,
            obj["int_alb"].values,
            obj["iso"].values,
            obj["lso"].values,
            obj["lst"].values,
            obj["mso"].values,
            obj["newsnow"].values,
            obj["pk_def"].values,
            obj["pk_den"].values,
            obj["pk_depth"].values,
            obj["pk_ice"].values,
            obj["pk_ice_change"].values,
            obj["pk_precip"].values,
            obj["pk_temp"].values,
            obj["pksv"].values,
            obj["pkwater_equiv"].values,
            obj["pptmix_nopack"].values,
            obj["pss"].values,
            obj["pst"].values,
            obj["salb"].values,
            obj["scrv"].values,
            obj["slst"].values,
            obj["snow_evap"].values,
            obj["snowcov_area"].values,
            obj["snowcov_areasv"].values,
            obj["snowmelt"].values,
            obj["snsv"].values,
            obj["tcal"].values,
            obj["through_rain"].values,
            obj["freeh2o_prev"].values,
            obj["pk_ice_prev"].values,
            obj["hru_ppt"].values,
            obj["hru_intcpevap"].values,
            obj["net_ppt"].values,
            obj["net_rain"].values,
            obj["net_snow"].values,
            obj["orad_hru"].values,
            obj["potet"].values,
            obj["pptmix"].values,
            obj["prmx"].values,
            obj["swrad"].values,
            obj["tavgc"].values,
            obj["tmaxc"].values,
            obj["tminc"].values,
            obj["transp_on"].values,
            obj["hru_type"].values,
            obj["cov_type"].values,
            obj["covden_win"].values,
            obj["covden_sum"].values,
            obj["emis_noppt"].values,
            obj["freeh2o_cap"].values,
            obj["hru_deplcrv"].values,
            obj["melt_force"].values,
            obj["melt_look"].values,
            obj["potet_sublim"].values,
            obj["rad_trncf"].values,
            obj["snarea_curve"].values,
            obj["snarea_thresh"].values,
            obj["tmax_allsnow"].values,
            obj["cecn_coef"].values,
            obj["tstorm_mo"].values,
            obj["soltab_horad_potsw"].values,
            np.float64(obj["albset_rna"].values[0]),
            np.float64(obj["albset_rnm"].values[0]),
            np.float64(obj["albset_sna"].values[0]),
            np.float64(obj["albset_snm"].values[0]),
            den_max,
            np.float64(1.0 / den_init),
            np.float64(1.0 / den_max),
            np.float64(obj["settle_const"].values[0]),
            np.int64(time.dowy),
            np.int64(time.doy),
            np.int64(time.month),
        )
