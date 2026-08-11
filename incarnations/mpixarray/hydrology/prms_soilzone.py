"""
hydrology/prms_soilzone.py
==========================
PRMSSoilzoneNoDprst + PRMSSoilzone: the PRMS soil zone, ported from
pywatershed (pywatershed/hydrology/prms_soilzone.py and
prms_soilzone_no_dprst.py; PRMS 5.2.1 physics, PRMS-IV documentation:
Markstrom et al. 2015, USGS TM 6-B7).

Fourth REAL process port (July 2026) -- the process that produces
``ssres_flow_vol`` (replacing the submodel carrier) plus groundwater's
``soil_to_gw``/``ssr_to_gw`` and runoff's ``soil_lower_prev``/
``soil_rechr_prev``. Ported: field declarations (names verbatim,
including pywatershed's underscore-private derived arrays) and the
numerics of ``_calculate_numpy`` + its helpers
(``compute_soilmoist``/``compute_interflow``/``compute_gwflow``/
``compute_szactet``, upstream leading underscores dropped), rewritten
to the in-place, out-first, zero-per-step-allocation convention.
Upstream's pre-/post-loop ARRAY expressions (hru_actet seed,
_snow_free, soil_moist_tot, recharge, the *_change and *_hru lines,
ssres_flow_vol / sroff_vol) are folded per element -- same op order
per element.

**Variant structure (ADDITIVE -- see PORTS.md "How variants are done
here")**: pywatershed derives PRMSSoilzoneNoDprst FROM PRMSSoilzone by
subtraction (re-declared interface, per-step zero arrays fed to the
shared kernel). Here the hierarchy points the right way:
``PRMSSoilzoneNoDprst`` is the minimal core (its own kernel over the
shared njit helpers; ``initialize()`` lives here with ONE overridable
hook, ``_set_hru_frac_perv``) and ``PRMSSoilzone`` EXTENDS it:
``dprst_frac`` + the two dprst inputs ADDED, the hook (dprst area
removed from the pervious fraction) and the kernel (dprst_evap_hru in
the hru_actet seed; dprst_seep_hru in recharge) overridden. The core
never touches dprst.

**Mutable inputs**: ``sroff`` and ``sroff_vol`` -- pywatershed's
soilzone ADDS dunnian_flow to runoff's sroff in place (its own
"WARNING" comment) and recomputes sroff_vol; declared
``kind="mutable_input"``. On nhm domains dunnian is identically zero
(sat_threshold >= 999 -- the reason pywatershed's own runoff autotest
skips otherwise), so runoff's parity is unaffected there.

Initialization -> framework seams:

- ``_initialize_soilzone_data`` -> ``initialize()`` +
  ``parameter_internal``: ``hru_frac_perv`` (variant hook -- see above;
  the full class matches PRMSRunoff's values), ``soil_rechr_max`` (+
  its two clamps, legal here: derived, pre-freeze), ``soil_lower_max``,
  ``_sat_threshold`` (zeroed for INACTIVE|LAKE), ``_pref_flow_den``
  (zeroed for non-LAND), ``pref_flow_thrsh``/``pref_flow_max`` (by
  hru_type; upstream "variables" never written by its kernel),
  ``_pref_flow_flag``/``_soil2gw_flag``.
- initial state (soil_moist/soil_rechr from init_frac params + the
  upstream value clamps, ssres_stor -> slow_stor/pref_flow_stor split,
  soil_lower, soil_lower_ratio, soil_moist_tot) is computed in
  ``initialize()`` in upstream's exact order (soil_rechr seeds from
  the PRE-clamp soil_rechr_max, then clamps apply).

Deliberately NOT ported (conventions in pws_phoenix/CLAUDE.md):

- Budget / ConservativeProcess; adapters; restart; calc_method;
  verbose; ``imbalance_behavior``; the Ag/ObsET variants (iter_aet).
- ``adjust_parameters``: the ONE edit of a true parameter
  (soil_moist_max < 1e-5 -> 1e-5) raises NotImplementedError if a
  domain would need it (frozen parameters; nhm domains do not).
  Derived/initial-value adjustments ARE applied (silently -- no warn
  machinery).
- ``soil_zone_max`` and ``_swale_limit``: computed upstream, never
  read by its kernel (soil_zone_max has no PRMS output either).
- ``fastmath=True``: upstream njits soilzone with fastmath; we keep
  strict IEEE like every other port. Differences land well inside
  upstream's own autotest tolerance (5e-6 -- soilzone's standard,
  NOT the 1e-13/1e-10 of the other ports).
- ``gwin`` (GSFLOW upslope inflow): hardwired zero upstream, kept as
  the scalar zero in ``ssres_in``.

Parameter provenance: ``hru_type``/``hru_area``/``hru_in_to_cf`` are
DIS_HRU variables; the 18 core process parameters live in
parameters_PRMSSoilzoneNoDprst.nc and the full 19 in
parameters_PRMSSoilzone.nc (``dprst_frac``/``hru_percent_imperv``/
``soil_moist_max`` are shared with PRMSRunoff's file -- same NHM
values, one shared field per grid).
"""

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process

# pywatershed constants (constants.py)
_NEARZERO = 1.0e-6  # nearzero
# HruType
_INACTIVE = 0
_LAND = 1
_LAKE = 2
_SWALE = 3
# SoilType
_SAND = 1
_LOAM = 2
_CLAY = 3
# ETType
_ET_DEFAULT = 1
_EVAP_ONLY = 2
_EVAP_PLUS_TRANSP = 3

_ONETHIRD = 1.0 / 3.0
_TWOTHIRDS = 2.0 / 3.0


# ----------------------------------------------------------------------
# Kernel helper functions -- pywatershed staticmethods verbatim
# (upstream leading underscores dropped; scalar in/out; ETType
# membership tests written as == comparisons on the int values)
# ----------------------------------------------------------------------


@numba.njit
def compute_soilmoist(
    soil2gw_flag,
    perv_frac,
    soil_moist_max,
    soil_rechr_max,
    soil2gw_max,
    infil,
    soil_moist,
    soil_rechr,
    soil_to_gw,
    soil_to_ssr,
):
    # PRMSIV Step 4 (eqn 1-125)
    soil_rechr = np.minimum(soil_rechr + infil, soil_rechr_max)

    # PRMSIV Step 5 (eqn 1-126)
    excess = soil_moist + infil
    soil_moist = np.minimum(excess, soil_moist_max)

    # PRMSIV Step 6 (eqns 1-128, 1-129)
    excess = (excess - soil_moist_max) * perv_frac

    if excess > 0.0:
        if soil2gw_flag:
            # PRMSIV eqn 1-130
            soil_to_gw = np.minimum(soil2gw_max, excess)
            # PRMSIV eqn 1-131 (start); this "excess" is gvr_maxin
            excess = excess - soil_to_gw

        if excess > (infil * perv_frac):
            infil = 0.0
        else:
            infil = infil - (excess / perv_frac)

        # PRMSIV eqn 1-131 (finish)
        soil_to_ssr = np.maximum(0.0, excess)

    return (
        infil,
        soil_moist,
        soil_rechr,
        soil_to_gw,
        soil_to_ssr,
    )


@numba.njit
def compute_interflow(coef_lin, coef_sq, ssres_in, storage, inter_flow):
    # inter_flow is in inches for the timestep
    if (coef_lin <= 0.0) and (ssres_in <= 0.0):
        c1 = coef_sq * storage
        inter_flow = storage * (c1 / (1.0 + c1))

    elif (coef_lin > 0.0) and (coef_sq <= 0.0):
        c2 = 1.0 - np.exp(-coef_lin)
        inter_flow = ssres_in * (1.0 - c2 / coef_lin) + storage * c2

    elif coef_sq > 0.0:
        c3 = np.sqrt(coef_lin**2.0 + 4.0 * coef_sq * ssres_in)
        sos = storage - ((c3 - coef_lin) / (2.0 * coef_sq))
        if c3 == 0.0:
            raise RuntimeError(
                "ERROR, in compute_interflow sos=0, "
                "please contact code developers"
            )

        c1 = coef_sq * sos / c3
        c2 = 1.0 - np.exp(-c3)

        if (1.0 + c1 * c2) > 0.0:
            inter_flow = ssres_in + ((sos * (1.0 + c1) * c2) / (1.0 + c1 * c2))
        else:
            inter_flow = ssres_in

    else:
        inter_flow = 0.0

    if inter_flow < 0.0:
        inter_flow = 0.0
    elif inter_flow > storage:
        inter_flow = storage

    storage = storage - inter_flow
    return storage, inter_flow


@numba.njit
def compute_gwflow(ssr2gw_rate, ssr2gw_exp, slow_stor):
    # Compute flow to groundwater
    ssr_to_gw = max(0.0, ssr2gw_rate * slow_stor**ssr2gw_exp)
    ssr_to_gw = min(ssr_to_gw, slow_stor)
    slow_stor = slow_stor - ssr_to_gw
    return ssr_to_gw, slow_stor


@numba.njit
def compute_szactet(
    transp_on,
    cov_type,
    soil_type,
    soil_moist_max,
    soil_rechr_max,
    snow_free,
    soil_moist,
    soil_rechr,
    avail_potet,
    potet_rechr,
    potet_lower,
):
    # Determine type of evapotranspiration:
    #   1 - default, 2 - evaporation only, 3 - transpiration + evap
    if avail_potet < _NEARZERO:
        et_type = _ET_DEFAULT
        avail_potet = 0.0
    elif transp_on == 0.0:
        if snow_free < 0.01:
            et_type = _ET_DEFAULT
        else:
            et_type = _EVAP_ONLY
    elif cov_type > 0:
        et_type = _EVAP_PLUS_TRANSP
    elif snow_free < 0.01:
        et_type = _ET_DEFAULT
    else:
        et_type = _EVAP_ONLY

    if (et_type == _EVAP_ONLY) or (et_type == _EVAP_PLUS_TRANSP):
        pcts = soil_moist / soil_moist_max
        pctr = soil_rechr / soil_rechr_max
        potet_lower = avail_potet
        potet_rechr = avail_potet

        if soil_type == _SAND:
            if pcts < 0.25:
                potet_lower = 0.5 * pcts * avail_potet
            if pctr < 0.25:
                potet_rechr = 0.5 * pctr * avail_potet
        elif soil_type == _LOAM:
            if pcts < 0.5:
                potet_lower = pcts * avail_potet
            if pctr < 0.5:
                potet_rechr = pctr * avail_potet
        elif soil_type == _CLAY:
            if (pcts < _TWOTHIRDS) and (pcts > _ONETHIRD):
                potet_lower = pcts * avail_potet
            elif pcts <= _ONETHIRD:
                potet_lower = 0.5 * pcts * avail_potet
            if (pctr < _TWOTHIRDS) and (pctr > _ONETHIRD):
                potet_rechr = pctr * avail_potet
            elif pctr <= _ONETHIRD:
                potet_rechr = 0.5 * pctr * avail_potet

        # ****** Soil moisture accounting
        if et_type == _EVAP_ONLY:
            potet_rechr = potet_rechr * snow_free

        if potet_rechr > soil_rechr:
            potet_rechr = soil_rechr
            soil_rechr = 0.0
        else:
            soil_rechr = soil_rechr - potet_rechr

        if (et_type == _EVAP_ONLY) or (potet_rechr >= potet_lower):
            if potet_rechr > soil_moist:
                potet_rechr = soil_moist
                soil_moist = 0.0
            else:
                soil_moist = soil_moist - potet_rechr
            et = potet_rechr
        elif potet_lower > soil_moist:
            et = soil_moist
            soil_moist = 0.0
        else:
            soil_moist = soil_moist - potet_lower
            et = potet_lower

        if soil_rechr > soil_moist:
            soil_rechr = soil_moist
    else:
        et = 0.0

    return (
        soil_moist,
        soil_rechr,
        avail_potet,
        potet_rechr,
        potet_lower,
        et,  # -> perv_actet
    )


class PRMSSoilzoneNoDprst(Process):
    """PRMS soil zone without depression storage: capillary
    (rechr/lower), gravity (slow) and preferential-flow reservoirs per
    HRU; produces interflow (ssres_flow), recharge to groundwater
    (soil_to_gw, ssr_to_gw), dunnian runoff (added IN PLACE to sroff),
    and pervious actual ET.

    The minimal core of the soilzone family; PRMSSoilzone adds the
    dprst ET/seepage accounting.

    Storage and fluxes are in inches (PRMS convention); *_vol are
    cubic feet via hru_in_to_cf.
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
    cov_type = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Cover type (0=bare, 1=grasses, 2=shrubs, 3=trees, "
        "4=coniferous)",
    )
    fastcoef_lin = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Linear preferential-flow routing coefficient [1/day]",
    )
    fastcoef_sq = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Non-linear preferential-flow routing coefficient [-]",
    )
    hru_percent_imperv = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of HRU area that is impervious [-]",
    )
    pref_flow_den = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Preferential-flow pore density [-]",
    )
    pref_flow_infil_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of infiltration to preferential flow [-]",
    )
    sat_threshold = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Soil saturation threshold above field capacity [inches]",
    )
    slowcoef_lin = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Linear gravity-flow routing coefficient [1/day]",
    )
    slowcoef_sq = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Non-linear gravity-flow routing coefficient [-]",
    )
    soil2gw_max = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum capillary excess routed to GWR [inches/day]",
    )
    soil_moist_init_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Initial fraction of capillary storage [-]",
    )
    soil_moist_max = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Maximum capillary-reservoir water capacity [inches]",
    )
    soil_rechr_init_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Initial fraction of recharge-zone storage [-]",
    )
    soil_rechr_max_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Recharge-zone maximum as fraction of soil_moist_max [-]",
    )
    soil_type = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Soil type (SAND=1, LOAM=2, CLAY=3)",
    )
    ssr2gw_exp = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Gravity-drainage exponent to GWR [-]",
    )
    ssr2gw_rate = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Gravity-drainage rate coefficient to GWR [fraction/day]",
    )
    ssstor_init_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Initial fraction of gravity+pref storage [-]",
    )

    # -- derived parameters (initialize(); frozen after) --
    hru_frac_perv = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Pervious fraction of HRU area [-] (variant hook: "
        "_set_hru_frac_perv)",
    )
    soil_rechr_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Recharge-zone maximum storage [inches] (clamped)",
    )
    soil_lower_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Lower-zone maximum storage [inches]",
    )
    _sat_threshold = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="sat_threshold zeroed for INACTIVE|LAKE HRUs "
        "(upstream edited-copy parameter)",
    )
    _pref_flow_den = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="pref_flow_den zeroed for non-LAND HRUs "
        "(upstream edited-copy parameter)",
    )
    pref_flow_thrsh = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Gravity storage above which flow goes preferential "
        "[inches] (upstream 'variable', never written by its kernel)",
    )
    pref_flow_max = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Maximum preferential-flow storage [inches] "
        "(upstream 'variable', never written by its kernel)",
    )
    _pref_flow_flag = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.bool_,
        description="LAND HRU with pref_flow_den > 0",
    )
    _soil2gw_flag = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.bool_,
        description="soil2gw_max > 0",
    )

    # -- inputs --
    hru_impervevap = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Impervious area evaporation [inches over the HRU]",
    )
    hru_intcpevap = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="HRU area-weighted canopy evaporation [inches]",
    )
    infil_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Infiltration [inches over the HRU]",
    )
    potet = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Potential evapotranspiration [inches]",
    )
    transp_on = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Transpiration occurring (0/1 flag)",
    )
    snow_evap = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Evaporation and sublimation from the snowpack [inches]",
    )
    snowcov_area = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snow-covered area fraction [-]",
    )

    # -- MUTABLE inputs (runoff's variables, edited in place here) --
    sroff = DataArrayMeta(
        kind="mutable_input",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff [inches] -- dunnian flow is ADDED "
        "in place (upstream behavior, its own WARNING comment)",
    )
    sroff_vol = DataArrayMeta(
        kind="mutable_input",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff volume [cf] -- recomputed from the "
        "dunnian-updated sroff",
    )

    # -- variables --
    cap_infil_tot = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Capillary infiltration [inches over the HRU]",
    )
    cap_waterin = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Capillary reservoir water in [inches]",
    )
    dunnian_flow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Dunnian surface runoff [inches]",
    )
    hru_actet = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Actual evapotranspiration [inches over the HRU]",
    )
    perv_actet = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pervious actual ET [inches over pervious area]",
    )
    perv_actet_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Pervious actual ET [inches over the HRU]",
    )
    potet_lower = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Potential ET from the lower zone [inches]",
    )
    potet_rechr = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Potential ET from the recharge zone [inches]",
    )
    pref_flow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Preferential interflow [inches]",
    )
    pref_flow_in = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Preferential reservoir inflow [inches]",
    )
    pref_flow_infil = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Infiltration to the preferential reservoir [inches]",
    )
    pref_flow_stor = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Preferential-flow reservoir storage [inches]",
    )
    pref_flow_stor_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Preferential storage change [inches]",
    )
    pref_flow_stor_prev = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Preferential storage, previous timestep",
    )
    recharge = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Recharge to groundwater [inches]",
    )
    slow_flow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Slow (gravity) interflow [inches]",
    )
    slow_stor = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Gravity reservoir storage [inches]",
    )
    slow_stor_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Gravity storage change [inches]",
    )
    slow_stor_prev = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Gravity storage, previous timestep",
    )
    soil_lower = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Lower capillary-zone storage [inches]",
    )
    soil_lower_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Lower-zone storage change [inches]",
    )
    soil_lower_change_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Lower-zone storage change [inches over the HRU]",
    )
    soil_lower_prev = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Lower-zone storage, previous timestep (PRMSRunoff input)",
    )
    soil_lower_ratio = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Lower-zone storage fraction of maximum [-]",
    )
    soil_moist = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Capillary reservoir storage [inches]",
    )
    soil_moist_tot = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Total soil-zone storage [inches over the HRU]",
    )
    soil_rechr = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Recharge-zone storage [inches]",
    )
    soil_rechr_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Recharge-zone storage change [inches]",
    )
    soil_rechr_change_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Recharge-zone storage change [inches over the HRU]",
    )
    soil_rechr_prev = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Recharge-zone storage, previous timestep "
        "(PRMSRunoff input)",
    )
    soil_to_gw = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Capillary excess to the GWR [inches] "
        "(PRMSGroundwater input)",
    )
    soil_to_ssr = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Capillary excess to the gravity reservoir [inches]",
    )
    ssr_to_gw = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Gravity drainage to the GWR [inches] "
        "(PRMSGroundwater input)",
    )
    ssres_flow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Interflow to the stream network [inches]",
    )
    ssres_flow_vol = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Interflow volume [cubic feet]",
    )
    ssres_in = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Inflow to gravity+preferential reservoirs [inches]",
    )
    ssres_stor = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Gravity + preferential storage [inches]",
    )
    swale_actet = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Swale ponded-water actual ET [inches]",
    )
    unused_potet = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Unsatisfied potential ET [inches]",
    )

    # ------------------------------------------------------------------
    # Initialization (_initialize_soilzone_data, no-restart path)
    # ------------------------------------------------------------------

    def _set_hru_frac_perv(self) -> None:
        """hru_frac_perv WITHOUT depression storage: impervious area
        removed only (upstream shared init under dprst_flag == False;
        seed with 1 - imperv, then recompute from areas for active
        HRUs -- upstream order)."""
        obj = self._obj
        hru_type = obj["hru_type"].values
        hru_area = obj["hru_area"].values
        hru_area_imperv = obj["hru_percent_imperv"].values * hru_area
        hru_area_perv = hru_area - hru_area_imperv
        wh_active = np.where(hru_type != _INACTIVE)
        obj["hru_frac_perv"].values[:] = 1.0 - obj["hru_percent_imperv"].values
        obj["hru_frac_perv"].values[wh_active] = (
            hru_area_perv[wh_active] / hru_area[wh_active]
        )

    def initialize(self) -> None:
        """Upstream ``_initialize_soilzone_data`` in its exact order
        (init-time numpy staging is fine here). Derived geometry ->
        parameter_internal (frozen after); initial state -> variables.
        The ONLY variant-dependent piece is the ``_set_hru_frac_perv``
        hook."""
        obj = self._obj

        # -- zero-init all variables (nan-initialized ones upstream
        # are fully set below or first written by advance()) --
        for name in (
            "cap_infil_tot",
            "cap_waterin",
            "dunnian_flow",
            "hru_actet",
            "perv_actet",
            "perv_actet_hru",
            "potet_lower",
            "potet_rechr",
            "pref_flow",
            "pref_flow_in",
            "pref_flow_infil",
            "pref_flow_stor",
            "pref_flow_stor_change",
            "recharge",
            "slow_flow",
            "slow_stor",
            "slow_stor_change",
            "soil_lower_change",
            "soil_lower_change_hru",
            "soil_lower_ratio",
            "soil_moist_tot",
            "soil_rechr_change",
            "soil_rechr_change_hru",
            "soil_to_gw",
            "soil_to_ssr",
            "ssr_to_gw",
            "ssres_flow",
            "ssres_flow_vol",
            "ssres_in",
            "swale_actet",
            "unused_potet",
        ):
            obj[name].values[:] = 0.0

        hru_type = obj["hru_type"].values

        # -- pervious fraction (variant hook; the dprst variant removes
        # the depression area) --
        self._set_hru_frac_perv()

        # soil_rechr_max PRE-clamp (initial soil_rechr seeds from this)
        soil_rechr_max = obj["soil_rechr_max"].values
        soil_rechr_max[:] = (
            obj["soil_rechr_max_frac"].values * obj["soil_moist_max"].values
        )

        # -- edited-copy parameters --
        wh_inactive_or_lake = np.where(
            (hru_type == _INACTIVE) | (hru_type == _LAKE)
        )
        sat_threshold = obj["_sat_threshold"].values
        sat_threshold[:] = obj["sat_threshold"].values
        sat_threshold[wh_inactive_or_lake] = 0.0
        pref_flow_den = obj["_pref_flow_den"].values
        pref_flow_den[:] = obj["pref_flow_den"].values
        pref_flow_den[np.where(hru_type != _LAND)] = 0.0

        pfif = obj["pref_flow_infil_frac"].values
        if (pfif.min() < 0.0) or (pfif.max() > 1.0):
            raise ValueError(
                "Values of pref_flow_infil_frac outside of [0,1]. "
                "If values are all -1, you must set pref_flow_infil_frac "
                "to pref_flow_den in the parameter file yourself."
            )

        # -- initial capillary state (sm_climateflow, no-restart) --
        soil_moist = obj["soil_moist"].values
        soil_rechr = obj["soil_rechr"].values
        soil_moist[:] = (
            obj["soil_moist_init_frac"].values * obj["soil_moist_max"].values
        )
        soil_rechr[:] = obj["soil_rechr_init_frac"].values * soil_rechr_max

        # -- initial gravity+preferential storage --
        ssres_stor = obj["ssres_stor"].values
        ssres_stor[:] = obj["ssstor_init_frac"].values * sat_threshold
        ssres_stor[wh_inactive_or_lake] = 0.0

        # -- upstream "adjust_parameters" block, exact order. The ONE
        # true-parameter edit (soil_moist_max < 1e-5) cannot happen
        # against frozen parameters -- guard; everything else edits
        # derived params / initial values and is applied (silently) --
        if (obj["soil_moist_max"].values < 1.0e-5).any():
            raise NotImplementedError(
                "PRMSSoilzone: soil_moist_max < 1e-5 requires upstream's "
                "parameter adjustment, not ported (frozen parameters)"
            )
        soil_rechr_max[:] = np.where(
            soil_rechr_max < 1.0e-5, 1.0e-5, soil_rechr_max
        )
        soil_rechr_max[:] = np.where(
            soil_rechr_max > obj["soil_moist_max"].values,
            obj["soil_moist_max"].values,
            soil_rechr_max,
        )
        soil_rechr[:] = np.where(
            soil_rechr > soil_rechr_max, soil_rechr_max, soil_rechr
        )
        soil_moist[:] = np.where(
            soil_moist > obj["soil_moist_max"].values,
            obj["soil_moist_max"].values,
            soil_moist,
        )
        soil_rechr[:] = np.where(
            soil_rechr > soil_moist, soil_moist, soil_rechr
        )
        ssres_stor[:] = np.where(
            ssres_stor > sat_threshold, sat_threshold, ssres_stor
        )

        # (_swale_limit: computed upstream, never read by its kernel --
        # not ported)

        # -- preferential-flow thresholds by hru_type --
        pref_flow_thrsh = obj["pref_flow_thrsh"].values
        pref_flow_max = obj["pref_flow_max"].values
        pref_flow_thrsh[:] = 0.0
        pref_flow_max[:] = 0.0
        wh_swale = np.where(hru_type == _SWALE)
        wh_land = np.where(hru_type == _LAND)
        pref_flow_thrsh[wh_swale] = sat_threshold[wh_swale]
        pref_flow_thrsh[wh_land] = sat_threshold[wh_land] * (
            1.0 - pref_flow_den[wh_land]
        )
        pref_flow_max[wh_land] = (
            sat_threshold[wh_land] - pref_flow_thrsh[wh_land]
        )
        obj["_pref_flow_flag"].values[:] = (hru_type == _LAND) & (
            pref_flow_den > 0.0
        )

        # -- split initial ssres_stor into slow / preferential --
        slow_stor = obj["slow_stor"].values
        pref_flow_stor = obj["pref_flow_stor"].values
        wh_land_or_swale = np.where((hru_type == _LAND) | (hru_type == _SWALE))
        slow_stor[wh_land_or_swale] = np.minimum(
            ssres_stor[wh_land_or_swale], pref_flow_thrsh[wh_land_or_swale]
        )
        pref_flow_stor[wh_land_or_swale] = (
            ssres_stor[wh_land_or_swale] - slow_stor[wh_land_or_swale]
        )

        obj["_soil2gw_flag"].values[:] = obj["soil2gw_max"].values > 0.0

        # (soil_zone_max: computed upstream, never read -- not ported)
        obj["soil_moist_tot"].values[:] = (
            ssres_stor + soil_moist * obj["hru_frac_perv"].values
        )

        obj["soil_lower"].values[:] = soil_moist - soil_rechr
        soil_lower_max = obj["soil_lower_max"].values
        soil_lower_max[:] = obj["soil_moist_max"].values - soil_rechr_max

        wh_soil_lower_stor = np.where(soil_lower_max > 0.0)
        obj["soil_lower_ratio"].values[wh_soil_lower_stor] = (
            obj["soil_lower"].values[wh_soil_lower_stor]
            / soil_lower_max[wh_soil_lower_stor]
        )

        # -- *_prev: upstream nan-inits these; advance() runs before
        # the first calculate, so seeding with the current state is
        # equivalent --
        obj["pref_flow_stor_prev"].values[:] = pref_flow_stor
        obj["soil_rechr_prev"].values[:] = soil_rechr
        obj["soil_lower_prev"].values[:] = obj["soil_lower"].values
        obj["slow_stor_prev"].values[:] = slow_stor

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        obj = self._obj
        obj["pref_flow_stor_prev"].values[:] = obj["pref_flow_stor"].values
        obj["soil_rechr_prev"].values[:] = obj["soil_rechr"].values
        obj["soil_lower_prev"].values[:] = obj["soil_lower"].values
        obj["slow_stor_prev"].values[:] = obj["slow_stor"].values

    @staticmethod
    @numba.njit
    def _calculate(
        # in/out state + outputs (written in place)
        cap_infil_tot: np.ndarray,
        cap_waterin: np.ndarray,
        dunnian_flow: np.ndarray,
        hru_actet: np.ndarray,
        perv_actet: np.ndarray,
        perv_actet_hru: np.ndarray,
        potet_lower: np.ndarray,
        potet_rechr: np.ndarray,
        pref_flow: np.ndarray,
        pref_flow_in: np.ndarray,
        pref_flow_infil: np.ndarray,
        pref_flow_stor: np.ndarray,
        pref_flow_stor_change: np.ndarray,
        recharge: np.ndarray,
        slow_flow: np.ndarray,
        slow_stor: np.ndarray,
        slow_stor_change: np.ndarray,
        soil_lower: np.ndarray,
        soil_lower_change: np.ndarray,
        soil_lower_change_hru: np.ndarray,
        soil_lower_ratio: np.ndarray,
        soil_moist: np.ndarray,
        soil_moist_tot: np.ndarray,
        soil_rechr: np.ndarray,
        soil_rechr_change: np.ndarray,
        soil_rechr_change_hru: np.ndarray,
        soil_to_gw: np.ndarray,
        soil_to_ssr: np.ndarray,
        ssr_to_gw: np.ndarray,
        ssres_flow: np.ndarray,
        ssres_flow_vol: np.ndarray,
        ssres_in: np.ndarray,
        ssres_stor: np.ndarray,
        swale_actet: np.ndarray,
        unused_potet: np.ndarray,
        # mutable inputs (runoff's variables, edited in place)
        sroff: np.ndarray,
        sroff_vol: np.ndarray,
        # prior state (read-only here; advance() maintains)
        pref_flow_stor_prev: np.ndarray,
        soil_lower_prev: np.ndarray,
        soil_rechr_prev: np.ndarray,
        slow_stor_prev: np.ndarray,
        # inputs
        hru_impervevap: np.ndarray,
        hru_intcpevap: np.ndarray,
        infil_hru: np.ndarray,
        potet: np.ndarray,
        transp_on: np.ndarray,
        snow_evap: np.ndarray,
        snowcov_area: np.ndarray,
        # parameters + derived
        hru_type: np.ndarray,
        hru_in_to_cf: np.ndarray,
        cov_type: np.ndarray,
        fastcoef_lin: np.ndarray,
        fastcoef_sq: np.ndarray,
        pref_flow_infil_frac: np.ndarray,
        slowcoef_lin: np.ndarray,
        slowcoef_sq: np.ndarray,
        soil2gw_max: np.ndarray,
        soil_moist_max: np.ndarray,
        soil_type: np.ndarray,
        ssr2gw_exp: np.ndarray,
        ssr2gw_rate: np.ndarray,
        hru_frac_perv: np.ndarray,
        soil_rechr_max: np.ndarray,
        soil_lower_max: np.ndarray,
        sat_threshold: np.ndarray,
        pref_flow_den: np.ndarray,
        pref_flow_thrsh: np.ndarray,
        pref_flow_max: np.ndarray,
        pref_flow_flag: np.ndarray,
        soil2gw_flag: np.ndarray,
    ) -> None:
        # the PRMSSoilzone kernel with the two dprst terms deleted
        # (upstream shared kernel under dprst_flag == False: no
        # dprst_evap_hru in the hru_actet seed, no dprst_seep_hru in
        # recharge); everything else identical, same njit helpers
        nhru = soil_moist.shape[0]
        gwin = 0.0  # GSFLOW upslope inflow, hardwired zero upstream

        for hh in range(nhru):
            # upstream pre-loop array lines, per element: diagnostic
            # resets, snow_free, and the hru_actet seed
            soil_to_gw[hh] = 0.0
            soil_to_ssr[hh] = 0.0
            ssr_to_gw[hh] = 0.0
            slow_flow[hh] = 0.0
            ssres_flow[hh] = 0.0
            potet_rechr[hh] = 0.0
            potet_lower[hh] = 0.0
            snow_free = 1.0 - snowcov_area[hh]
            hru_actet[hh] = (
                hru_impervevap[hh] + hru_intcpevap[hh] + snow_evap[hh]
            )

            dunnianflw = 0.0
            dunnianflw_pfr = 0.0
            dunnianflw_gvr = 0.0
            prefflow = 0.0

            avail_potet = np.maximum(0.0, potet[hh] - hru_actet[hh])

            # capillary maxin is on the pervious area
            capwater_maxin = infil_hru[hh] / hru_frac_perv[hh]

            # Compute preferential flow and storage, and any dunnian
            if pref_flow_infil_frac[hh] != 0.0:
                pref_flow_maxin = 0.0
                pref_flow_infil[hh] = 0.0

                if capwater_maxin > 0.0:
                    # PRMSIV Step 1 (eqn 1-121): partition infil
                    pref_flow_maxin = capwater_maxin * pref_flow_infil_frac[hh]
                    # PRMSIV Step 3 (eqn 1-124)
                    capwater_maxin = capwater_maxin - pref_flow_maxin
                    # renormalize to whole HRU
                    pref_flow_maxin = pref_flow_maxin * hru_frac_perv[hh]

                    # PRMSIV Step 2 (eqns 1-122, 1-123): PFR storage,
                    # excess to Dunnian
                    pref_flow_stor[hh] = pref_flow_stor[hh] + pref_flow_maxin
                    dunnianflw_pfr = np.maximum(
                        0.0, pref_flow_stor[hh] - pref_flow_max[hh]
                    )
                    if dunnianflw_pfr > 0.0:
                        pref_flow_stor[hh] = pref_flow_max[hh]
                    pref_flow_infil[hh] = pref_flow_maxin - dunnianflw_pfr

            # whole HRU
            cap_infil_tot[hh] = capwater_maxin * hru_frac_perv[hh]

            # ****** Add infiltration to soil and compute excess
            cap_waterin[hh] = capwater_maxin

            # PRMSIV Steps 4, 5, 6 (see compute_soilmoist)
            if (capwater_maxin + soil_moist[hh]) > 0.0:
                (
                    cap_waterin[hh],
                    soil_moist[hh],
                    soil_rechr[hh],
                    soil_to_gw[hh],
                    soil_to_ssr[hh],
                ) = compute_soilmoist(
                    soil2gw_flag[hh],
                    hru_frac_perv[hh],
                    soil_moist_max[hh],
                    soil_rechr_max[hh],
                    soil2gw_max[hh],
                    cap_waterin[hh],
                    soil_moist[hh],
                    soil_rechr[hh],
                    soil_to_gw[hh],
                    soil_to_ssr[hh],
                )
                cap_waterin[hh] = cap_waterin[hh] * hru_frac_perv[hh]

            topfr = 0.0
            # soil_to_ssr also known as gvr_maxin
            availh2o = slow_stor[hh] + soil_to_ssr[hh]

            if hru_type[hh] == _LAND:
                # PRMSIV Step 7 (eqn 1-133): gvr excess to preferential
                topfr = max(0.0, availh2o - pref_flow_thrsh[hh])
                # PRMSIV eqn 1-134
                ssresin = soil_to_ssr[hh] - topfr
                slow_stor[hh] = max(0.0, availh2o - topfr)

                # PRMSIV Step 9: slow contribution to interflow
                if slow_stor[hh] > 0.0:
                    (
                        slow_stor[hh],
                        slow_flow[hh],
                    ) = compute_interflow(
                        slowcoef_lin[hh],
                        slowcoef_sq[hh],
                        ssresin,
                        slow_stor[hh],
                        slow_flow[hh],
                    )

            elif hru_type[hh] == _SWALE:
                slow_stor[hh] = availh2o

            if (slow_stor[hh] > 0.0) and (ssr2gw_rate[hh] > 0.0):
                (
                    ssr_to_gw[hh],
                    slow_stor[hh],
                ) = compute_gwflow(
                    ssr2gw_rate[hh],
                    ssr2gw_exp[hh],
                    slow_stor[hh],
                )

            # Compute contribution to Dunnian flow from PFR, if any
            if pref_flow_den[hh] > 0.0:
                # PRMSIV eqn 1-135
                availh2o = pref_flow_stor[hh] + topfr
                dunnianflw_gvr = max(0.0, availh2o - pref_flow_max[hh])
                if dunnianflw_gvr > 0.0:
                    # PRMSIV eqn 1-136
                    topfr = max(0.0, topfr - dunnianflw_gvr)

                pref_flow_in[hh] = pref_flow_infil[hh] + topfr
                pref_flow_stor[hh] = pref_flow_stor[hh] + topfr
                if pref_flow_stor[hh] > 0.0:
                    (
                        pref_flow_stor[hh],
                        prefflow,
                    ) = compute_interflow(
                        fastcoef_lin[hh],
                        fastcoef_sq[hh],
                        pref_flow_in[hh],
                        pref_flow_stor[hh],
                        prefflow,
                    )
            elif hru_type[hh] == _LAND:
                dunnianflw_gvr = topfr

            perv_actet[hh] = 0.0

            # Compute actual evapotranspiration
            if soil_moist[hh] > 0.0:
                (
                    soil_moist[hh],
                    soil_rechr[hh],
                    avail_potet,
                    potet_rechr[hh],
                    potet_lower[hh],
                    perv_actet[hh],
                ) = compute_szactet(
                    transp_on[hh],
                    cov_type[hh],
                    soil_type[hh],
                    soil_moist_max[hh],
                    soil_rechr_max[hh],
                    snow_free,
                    soil_moist[hh],
                    soil_rechr[hh],
                    avail_potet,
                    potet_rechr[hh],
                    potet_lower[hh],
                )

            hru_actet[hh] = hru_actet[hh] + perv_actet[hh] * hru_frac_perv[hh]
            avail_potet = potet[hh] - hru_actet[hh]  # upstream (unused)
            soil_lower[hh] = soil_moist[hh] - soil_rechr[hh]

            if hru_type[hh] == _LAND:
                dunnianflw = dunnianflw_gvr + dunnianflw_pfr
                dunnian_flow[hh] = dunnianflw

                # Treat pref_flow as interflow
                ssres_flow[hh] = slow_flow[hh]
                if pref_flow_den[hh] > 0.0:
                    pref_flow[hh] = prefflow
                    ssres_flow[hh] = ssres_flow[hh] + prefflow

                # Treat dunnianflw as surface runoff to streams --
                # upstream's own WARNING: modifies srunoff's sroff
                sroff[hh] = sroff[hh] + dunnian_flow[hh]
                ssres_stor[hh] = slow_stor[hh] + pref_flow_stor[hh]

            else:
                # For swales
                availh2o = slow_stor[hh] - sat_threshold[hh]
                swale_actet[hh] = 0.0
                if availh2o > 0.0:
                    # ponding: storage > sat_threshold
                    unsatisfied_et = potet[hh] - hru_actet[hh]
                    if unsatisfied_et > 0.0:
                        availh2o = min(availh2o, unsatisfied_et)
                        swale_actet[hh] = availh2o
                        hru_actet[hh] = hru_actet[hh] + swale_actet[hh]
                        slow_stor[hh] = slow_stor[hh] - swale_actet[hh]
                ssres_stor[hh] = slow_stor[hh]

            ssres_in[hh] = soil_to_ssr[hh] + pref_flow_infil[hh] + gwin
            unused_potet[hh] = potet[hh] - hru_actet[hh]

            # upstream post-loop array lines, folded per element
            if soil_lower_max[hh] > 0.0:
                soil_lower_ratio[hh] = soil_lower[hh] / soil_lower_max[hh]
            soil_moist_tot[hh] = (
                ssres_stor[hh] + soil_moist[hh] * hru_frac_perv[hh]
            )
            recharge[hh] = soil_to_gw[hh] + ssr_to_gw[hh]
            pref_flow_stor_change[hh] = (
                pref_flow_stor[hh] - pref_flow_stor_prev[hh]
            )
            soil_lower_change[hh] = soil_lower[hh] - soil_lower_prev[hh]
            soil_rechr_change[hh] = soil_rechr[hh] - soil_rechr_prev[hh]
            slow_stor_change[hh] = slow_stor[hh] - slow_stor_prev[hh]
            soil_lower_change_hru[hh] = (
                soil_lower_change[hh] * hru_frac_perv[hh]
            )
            soil_rechr_change_hru[hh] = (
                soil_rechr_change[hh] * hru_frac_perv[hh]
            )
            perv_actet_hru[hh] = perv_actet[hh] * hru_frac_perv[hh]
            ssres_flow_vol[hh] = ssres_flow[hh] * hru_in_to_cf[hh]
            # upstream _calculate wrapper: sroff_vol from updated sroff
            sroff_vol[hh] = sroff[hh] * hru_in_to_cf[hh]

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate(
            obj["cap_infil_tot"].values,
            obj["cap_waterin"].values,
            obj["dunnian_flow"].values,
            obj["hru_actet"].values,
            obj["perv_actet"].values,
            obj["perv_actet_hru"].values,
            obj["potet_lower"].values,
            obj["potet_rechr"].values,
            obj["pref_flow"].values,
            obj["pref_flow_in"].values,
            obj["pref_flow_infil"].values,
            obj["pref_flow_stor"].values,
            obj["pref_flow_stor_change"].values,
            obj["recharge"].values,
            obj["slow_flow"].values,
            obj["slow_stor"].values,
            obj["slow_stor_change"].values,
            obj["soil_lower"].values,
            obj["soil_lower_change"].values,
            obj["soil_lower_change_hru"].values,
            obj["soil_lower_ratio"].values,
            obj["soil_moist"].values,
            obj["soil_moist_tot"].values,
            obj["soil_rechr"].values,
            obj["soil_rechr_change"].values,
            obj["soil_rechr_change_hru"].values,
            obj["soil_to_gw"].values,
            obj["soil_to_ssr"].values,
            obj["ssr_to_gw"].values,
            obj["ssres_flow"].values,
            obj["ssres_flow_vol"].values,
            obj["ssres_in"].values,
            obj["ssres_stor"].values,
            obj["swale_actet"].values,
            obj["unused_potet"].values,
            obj["sroff"].values,
            obj["sroff_vol"].values,
            obj["pref_flow_stor_prev"].values,
            obj["soil_lower_prev"].values,
            obj["soil_rechr_prev"].values,
            obj["slow_stor_prev"].values,
            obj["hru_impervevap"].values,
            obj["hru_intcpevap"].values,
            obj["infil_hru"].values,
            obj["potet"].values,
            obj["transp_on"].values,
            obj["snow_evap"].values,
            obj["snowcov_area"].values,
            obj["hru_type"].values,
            obj["hru_in_to_cf"].values,
            obj["cov_type"].values,
            obj["fastcoef_lin"].values,
            obj["fastcoef_sq"].values,
            obj["pref_flow_infil_frac"].values,
            obj["slowcoef_lin"].values,
            obj["slowcoef_sq"].values,
            obj["soil2gw_max"].values,
            obj["soil_moist_max"].values,
            obj["soil_type"].values,
            obj["ssr2gw_exp"].values,
            obj["ssr2gw_rate"].values,
            obj["hru_frac_perv"].values,
            obj["soil_rechr_max"].values,
            obj["soil_lower_max"].values,
            obj["_sat_threshold"].values,
            obj["_pref_flow_den"].values,
            obj["pref_flow_thrsh"].values,
            obj["pref_flow_max"].values,
            obj["_pref_flow_flag"].values,
            obj["_soil2gw_flag"].values,
        )


class PRMSSoilzone(PRMSSoilzoneNoDprst):
    """PRMS soil zone: the NoDprst core PLUS depression-storage ET and
    seepage accounting per HRU.

    Storage and fluxes are in inches (PRMS convention); *_vol are
    cubic feet via hru_in_to_cf.
    """

    # ------------------------------------------------------------------
    # Field declarations ADDED to the NoDprst core (names verbatim)
    # ------------------------------------------------------------------

    # -- process parameters --
    dprst_frac = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of HRU area that has surface depressions [-]",
    )

    # -- inputs --
    dprst_evap_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Depression evaporation [inches over the HRU]",
    )
    dprst_seep_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Depression seepage to groundwater [inches over the HRU]",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _set_hru_frac_perv(self) -> None:
        """hru_frac_perv with the dprst area ALSO removed (upstream
        order: imperv first, then dprst for active HRUs, then
        recompute) -- same values as PRMSRunoff's."""
        obj = self._obj
        hru_type = obj["hru_type"].values
        hru_area = obj["hru_area"].values
        hru_area_imperv = obj["hru_percent_imperv"].values * hru_area
        hru_area_perv = hru_area - hru_area_imperv
        wh_active = np.where(hru_type != _INACTIVE)
        dprst_area_max = obj["dprst_frac"].values * hru_area
        hru_area_perv[wh_active] = (
            hru_area_perv[wh_active] - dprst_area_max[wh_active]
        )
        obj["hru_frac_perv"].values[:] = 1.0 - obj["hru_percent_imperv"].values
        obj["hru_frac_perv"].values[wh_active] = (
            hru_area_perv[wh_active] / hru_area[wh_active]
        )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @staticmethod
    @numba.njit
    def _calculate(
        # in/out state + outputs (written in place)
        cap_infil_tot: np.ndarray,
        cap_waterin: np.ndarray,
        dunnian_flow: np.ndarray,
        hru_actet: np.ndarray,
        perv_actet: np.ndarray,
        perv_actet_hru: np.ndarray,
        potet_lower: np.ndarray,
        potet_rechr: np.ndarray,
        pref_flow: np.ndarray,
        pref_flow_in: np.ndarray,
        pref_flow_infil: np.ndarray,
        pref_flow_stor: np.ndarray,
        pref_flow_stor_change: np.ndarray,
        recharge: np.ndarray,
        slow_flow: np.ndarray,
        slow_stor: np.ndarray,
        slow_stor_change: np.ndarray,
        soil_lower: np.ndarray,
        soil_lower_change: np.ndarray,
        soil_lower_change_hru: np.ndarray,
        soil_lower_ratio: np.ndarray,
        soil_moist: np.ndarray,
        soil_moist_tot: np.ndarray,
        soil_rechr: np.ndarray,
        soil_rechr_change: np.ndarray,
        soil_rechr_change_hru: np.ndarray,
        soil_to_gw: np.ndarray,
        soil_to_ssr: np.ndarray,
        ssr_to_gw: np.ndarray,
        ssres_flow: np.ndarray,
        ssres_flow_vol: np.ndarray,
        ssres_in: np.ndarray,
        ssres_stor: np.ndarray,
        swale_actet: np.ndarray,
        unused_potet: np.ndarray,
        # mutable inputs (runoff's variables, edited in place)
        sroff: np.ndarray,
        sroff_vol: np.ndarray,
        # prior state (read-only here; advance() maintains)
        pref_flow_stor_prev: np.ndarray,
        soil_lower_prev: np.ndarray,
        soil_rechr_prev: np.ndarray,
        slow_stor_prev: np.ndarray,
        # inputs
        dprst_evap_hru: np.ndarray,
        dprst_seep_hru: np.ndarray,
        hru_impervevap: np.ndarray,
        hru_intcpevap: np.ndarray,
        infil_hru: np.ndarray,
        potet: np.ndarray,
        transp_on: np.ndarray,
        snow_evap: np.ndarray,
        snowcov_area: np.ndarray,
        # parameters + derived
        hru_type: np.ndarray,
        hru_in_to_cf: np.ndarray,
        cov_type: np.ndarray,
        fastcoef_lin: np.ndarray,
        fastcoef_sq: np.ndarray,
        pref_flow_infil_frac: np.ndarray,
        slowcoef_lin: np.ndarray,
        slowcoef_sq: np.ndarray,
        soil2gw_max: np.ndarray,
        soil_moist_max: np.ndarray,
        soil_type: np.ndarray,
        ssr2gw_exp: np.ndarray,
        ssr2gw_rate: np.ndarray,
        hru_frac_perv: np.ndarray,
        soil_rechr_max: np.ndarray,
        soil_lower_max: np.ndarray,
        sat_threshold: np.ndarray,
        pref_flow_den: np.ndarray,
        pref_flow_thrsh: np.ndarray,
        pref_flow_max: np.ndarray,
        pref_flow_flag: np.ndarray,
        soil2gw_flag: np.ndarray,
    ) -> None:
        nhru = soil_moist.shape[0]
        gwin = 0.0  # GSFLOW upslope inflow, hardwired zero upstream

        for hh in range(nhru):
            # upstream pre-loop array lines, per element: diagnostic
            # resets, snow_free, and the hru_actet seed (dprst ACTIVE)
            soil_to_gw[hh] = 0.0
            soil_to_ssr[hh] = 0.0
            ssr_to_gw[hh] = 0.0
            slow_flow[hh] = 0.0
            ssres_flow[hh] = 0.0
            potet_rechr[hh] = 0.0
            potet_lower[hh] = 0.0
            snow_free = 1.0 - snowcov_area[hh]
            hru_actet[hh] = (
                hru_impervevap[hh]
                + hru_intcpevap[hh]
                + snow_evap[hh]
                + dprst_evap_hru[hh]
            )

            dunnianflw = 0.0
            dunnianflw_pfr = 0.0
            dunnianflw_gvr = 0.0
            prefflow = 0.0

            avail_potet = np.maximum(0.0, potet[hh] - hru_actet[hh])

            # capillary maxin is on the pervious area
            capwater_maxin = infil_hru[hh] / hru_frac_perv[hh]

            # Compute preferential flow and storage, and any dunnian
            if pref_flow_infil_frac[hh] != 0.0:
                pref_flow_maxin = 0.0
                pref_flow_infil[hh] = 0.0

                if capwater_maxin > 0.0:
                    # PRMSIV Step 1 (eqn 1-121): partition infil
                    pref_flow_maxin = capwater_maxin * pref_flow_infil_frac[hh]
                    # PRMSIV Step 3 (eqn 1-124)
                    capwater_maxin = capwater_maxin - pref_flow_maxin
                    # renormalize to whole HRU
                    pref_flow_maxin = pref_flow_maxin * hru_frac_perv[hh]

                    # PRMSIV Step 2 (eqns 1-122, 1-123): PFR storage,
                    # excess to Dunnian
                    pref_flow_stor[hh] = pref_flow_stor[hh] + pref_flow_maxin
                    dunnianflw_pfr = np.maximum(
                        0.0, pref_flow_stor[hh] - pref_flow_max[hh]
                    )
                    if dunnianflw_pfr > 0.0:
                        pref_flow_stor[hh] = pref_flow_max[hh]
                    pref_flow_infil[hh] = pref_flow_maxin - dunnianflw_pfr

            # whole HRU
            cap_infil_tot[hh] = capwater_maxin * hru_frac_perv[hh]

            # ****** Add infiltration to soil and compute excess
            cap_waterin[hh] = capwater_maxin

            # PRMSIV Steps 4, 5, 6 (see compute_soilmoist)
            if (capwater_maxin + soil_moist[hh]) > 0.0:
                (
                    cap_waterin[hh],
                    soil_moist[hh],
                    soil_rechr[hh],
                    soil_to_gw[hh],
                    soil_to_ssr[hh],
                ) = compute_soilmoist(
                    soil2gw_flag[hh],
                    hru_frac_perv[hh],
                    soil_moist_max[hh],
                    soil_rechr_max[hh],
                    soil2gw_max[hh],
                    cap_waterin[hh],
                    soil_moist[hh],
                    soil_rechr[hh],
                    soil_to_gw[hh],
                    soil_to_ssr[hh],
                )
                cap_waterin[hh] = cap_waterin[hh] * hru_frac_perv[hh]

            topfr = 0.0
            # soil_to_ssr also known as gvr_maxin
            availh2o = slow_stor[hh] + soil_to_ssr[hh]

            if hru_type[hh] == _LAND:
                # PRMSIV Step 7 (eqn 1-133): gvr excess to preferential
                topfr = max(0.0, availh2o - pref_flow_thrsh[hh])
                # PRMSIV eqn 1-134
                ssresin = soil_to_ssr[hh] - topfr
                slow_stor[hh] = max(0.0, availh2o - topfr)

                # PRMSIV Step 9: slow contribution to interflow
                if slow_stor[hh] > 0.0:
                    (
                        slow_stor[hh],
                        slow_flow[hh],
                    ) = compute_interflow(
                        slowcoef_lin[hh],
                        slowcoef_sq[hh],
                        ssresin,
                        slow_stor[hh],
                        slow_flow[hh],
                    )

            elif hru_type[hh] == _SWALE:
                slow_stor[hh] = availh2o

            if (slow_stor[hh] > 0.0) and (ssr2gw_rate[hh] > 0.0):
                (
                    ssr_to_gw[hh],
                    slow_stor[hh],
                ) = compute_gwflow(
                    ssr2gw_rate[hh],
                    ssr2gw_exp[hh],
                    slow_stor[hh],
                )

            # Compute contribution to Dunnian flow from PFR, if any
            if pref_flow_den[hh] > 0.0:
                # PRMSIV eqn 1-135
                availh2o = pref_flow_stor[hh] + topfr
                dunnianflw_gvr = max(0.0, availh2o - pref_flow_max[hh])
                if dunnianflw_gvr > 0.0:
                    # PRMSIV eqn 1-136
                    topfr = max(0.0, topfr - dunnianflw_gvr)

                pref_flow_in[hh] = pref_flow_infil[hh] + topfr
                pref_flow_stor[hh] = pref_flow_stor[hh] + topfr
                if pref_flow_stor[hh] > 0.0:
                    (
                        pref_flow_stor[hh],
                        prefflow,
                    ) = compute_interflow(
                        fastcoef_lin[hh],
                        fastcoef_sq[hh],
                        pref_flow_in[hh],
                        pref_flow_stor[hh],
                        prefflow,
                    )
            elif hru_type[hh] == _LAND:
                dunnianflw_gvr = topfr

            perv_actet[hh] = 0.0

            # Compute actual evapotranspiration
            if soil_moist[hh] > 0.0:
                (
                    soil_moist[hh],
                    soil_rechr[hh],
                    avail_potet,
                    potet_rechr[hh],
                    potet_lower[hh],
                    perv_actet[hh],
                ) = compute_szactet(
                    transp_on[hh],
                    cov_type[hh],
                    soil_type[hh],
                    soil_moist_max[hh],
                    soil_rechr_max[hh],
                    snow_free,
                    soil_moist[hh],
                    soil_rechr[hh],
                    avail_potet,
                    potet_rechr[hh],
                    potet_lower[hh],
                )

            hru_actet[hh] = hru_actet[hh] + perv_actet[hh] * hru_frac_perv[hh]
            avail_potet = potet[hh] - hru_actet[hh]  # upstream (unused)
            soil_lower[hh] = soil_moist[hh] - soil_rechr[hh]

            if hru_type[hh] == _LAND:
                dunnianflw = dunnianflw_gvr + dunnianflw_pfr
                dunnian_flow[hh] = dunnianflw

                # Treat pref_flow as interflow
                ssres_flow[hh] = slow_flow[hh]
                if pref_flow_den[hh] > 0.0:
                    pref_flow[hh] = prefflow
                    ssres_flow[hh] = ssres_flow[hh] + prefflow

                # Treat dunnianflw as surface runoff to streams --
                # upstream's own WARNING: modifies srunoff's sroff
                sroff[hh] = sroff[hh] + dunnian_flow[hh]
                ssres_stor[hh] = slow_stor[hh] + pref_flow_stor[hh]

            else:
                # For swales
                availh2o = slow_stor[hh] - sat_threshold[hh]
                swale_actet[hh] = 0.0
                if availh2o > 0.0:
                    # ponding: storage > sat_threshold
                    unsatisfied_et = potet[hh] - hru_actet[hh]
                    if unsatisfied_et > 0.0:
                        availh2o = min(availh2o, unsatisfied_et)
                        swale_actet[hh] = availh2o
                        hru_actet[hh] = hru_actet[hh] + swale_actet[hh]
                        slow_stor[hh] = slow_stor[hh] - swale_actet[hh]
                ssres_stor[hh] = slow_stor[hh]

            ssres_in[hh] = soil_to_ssr[hh] + pref_flow_infil[hh] + gwin
            unused_potet[hh] = potet[hh] - hru_actet[hh]

            # upstream post-loop array lines, folded per element
            if soil_lower_max[hh] > 0.0:
                soil_lower_ratio[hh] = soil_lower[hh] / soil_lower_max[hh]
            soil_moist_tot[hh] = (
                ssres_stor[hh] + soil_moist[hh] * hru_frac_perv[hh]
            )
            recharge[hh] = soil_to_gw[hh] + ssr_to_gw[hh] + dprst_seep_hru[hh]
            pref_flow_stor_change[hh] = (
                pref_flow_stor[hh] - pref_flow_stor_prev[hh]
            )
            soil_lower_change[hh] = soil_lower[hh] - soil_lower_prev[hh]
            soil_rechr_change[hh] = soil_rechr[hh] - soil_rechr_prev[hh]
            slow_stor_change[hh] = slow_stor[hh] - slow_stor_prev[hh]
            soil_lower_change_hru[hh] = (
                soil_lower_change[hh] * hru_frac_perv[hh]
            )
            soil_rechr_change_hru[hh] = (
                soil_rechr_change[hh] * hru_frac_perv[hh]
            )
            perv_actet_hru[hh] = perv_actet[hh] * hru_frac_perv[hh]
            ssres_flow_vol[hh] = ssres_flow[hh] * hru_in_to_cf[hh]
            # upstream _calculate wrapper: sroff_vol from updated sroff
            sroff_vol[hh] = sroff[hh] * hru_in_to_cf[hh]

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate(
            obj["cap_infil_tot"].values,
            obj["cap_waterin"].values,
            obj["dunnian_flow"].values,
            obj["hru_actet"].values,
            obj["perv_actet"].values,
            obj["perv_actet_hru"].values,
            obj["potet_lower"].values,
            obj["potet_rechr"].values,
            obj["pref_flow"].values,
            obj["pref_flow_in"].values,
            obj["pref_flow_infil"].values,
            obj["pref_flow_stor"].values,
            obj["pref_flow_stor_change"].values,
            obj["recharge"].values,
            obj["slow_flow"].values,
            obj["slow_stor"].values,
            obj["slow_stor_change"].values,
            obj["soil_lower"].values,
            obj["soil_lower_change"].values,
            obj["soil_lower_change_hru"].values,
            obj["soil_lower_ratio"].values,
            obj["soil_moist"].values,
            obj["soil_moist_tot"].values,
            obj["soil_rechr"].values,
            obj["soil_rechr_change"].values,
            obj["soil_rechr_change_hru"].values,
            obj["soil_to_gw"].values,
            obj["soil_to_ssr"].values,
            obj["ssr_to_gw"].values,
            obj["ssres_flow"].values,
            obj["ssres_flow_vol"].values,
            obj["ssres_in"].values,
            obj["ssres_stor"].values,
            obj["swale_actet"].values,
            obj["unused_potet"].values,
            obj["sroff"].values,
            obj["sroff_vol"].values,
            obj["pref_flow_stor_prev"].values,
            obj["soil_lower_prev"].values,
            obj["soil_rechr_prev"].values,
            obj["slow_stor_prev"].values,
            obj["dprst_evap_hru"].values,
            obj["dprst_seep_hru"].values,
            obj["hru_impervevap"].values,
            obj["hru_intcpevap"].values,
            obj["infil_hru"].values,
            obj["potet"].values,
            obj["transp_on"].values,
            obj["snow_evap"].values,
            obj["snowcov_area"].values,
            obj["hru_type"].values,
            obj["hru_in_to_cf"].values,
            obj["cov_type"].values,
            obj["fastcoef_lin"].values,
            obj["fastcoef_sq"].values,
            obj["pref_flow_infil_frac"].values,
            obj["slowcoef_lin"].values,
            obj["slowcoef_sq"].values,
            obj["soil2gw_max"].values,
            obj["soil_moist_max"].values,
            obj["soil_type"].values,
            obj["ssr2gw_exp"].values,
            obj["ssr2gw_rate"].values,
            obj["hru_frac_perv"].values,
            obj["soil_rechr_max"].values,
            obj["soil_lower_max"].values,
            obj["_sat_threshold"].values,
            obj["_pref_flow_den"].values,
            obj["pref_flow_thrsh"].values,
            obj["pref_flow_max"].values,
            obj["_pref_flow_flag"].values,
            obj["_soil2gw_flag"].values,
        )
