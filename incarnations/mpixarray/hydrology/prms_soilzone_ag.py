"""
hydrology/prms_soilzone_ag.py
=============================
PRMSSoilzoneAg: the PRMS/GSFLOW agricultural (dual-area) soil zone,
ported from pywatershed
(pywatershed/hydrology/prms_soilzone_ag_obs_et.py and
prms_soilzone_ag.py; PRMS 5.2.1 / GSFLOW agricultural physics; the
kernel is upstream's ``_calculate_numpy``, Fortran szrun_ag()).

The obs-ET iteration variant (PRMSSoilzoneAgObsET) EXTENDS this core
(stage 4); validation = fgr_ag_2yr output_spinup / output_analysis at
upstream's 1e-5 + per-variable exceptions (GSFLOW Fortran answers).

Why PRMSSoilzoneAg(Process) and NOT PRMSSoilzoneAg(PRMSSoilzone)
----------------------------------------------------------------
This looks like a violation of the additive-variant stance (PORTS.md
"How variants are done here": variants EXTEND a minimal core) until
you check the interfaces. The rule actually applied across the ag
ports is: **extend when the interface is a genuine superset; sibling
when it isn't.**

- **The blocker: extending PRMSSoilzone requires one true
  SUBTRACTION.** Field-by-field, the ag class's parameters and
  variables are strict supersets of PRMSSoilzone's -- fine. Its
  inputs are NOT: plain soilzone takes ``infil_hru`` (whole-HRU
  depth), the ag soilzone takes ``infil`` (pervious-area depth) plus
  ``infil_ag`` -- dual-area accounting needs per-area depths, so
  upstream changed the interface. Inheriting from PRMSSoilzone would
  leave an ``infil_hru`` declaration that assembly demands be fed;
  removing an inherited field is the thing the framework deliberately
  makes nearly unwritable. The escapes are a tombstone mechanism
  (legitimizing subtractive subclassing -- the exact pattern these
  ports invert) or accepting ``infil_hru`` and dividing back out by a
  now-time-varying ``hru_frac_perv`` (interface dishonesty plus float
  noise against parity). Both rejected.

- **Inheritance would buy only the declaration blocks.** Every method
  is overridden anyway: ``initialize`` (dual-area, ag storages),
  ``advance`` (ag prevs), the kernel (dual-area logic is threaded
  through the entire loop, not appended to it), ``calculate``. The
  real computational sharing is at the module njit-helper level
  (``compute_soilmoist`` / ``compute_interflow`` / ``compute_gwflow``
  / ``compute_szactet`` -- upstream's ag kernel calls the same four
  from prms_soilzone.py), and module-level sharing works across
  sibling classes for free: the STARFIT precedent, sibling leaves
  over shared kernels.

- **Contrast with runoff, where extension IS used.** PRMSRunoffAg's
  upstream interface is a strict superset of PRMSRunoff's
  (parameters, inputs -- just adds ``ag_frac`` --, variables, no
  removals), so ``PRMSRunoffAg(PRMSRunoff)`` is honest; its only
  wrinkle is ``hru_perv``/``hru_frac_perv`` going frozen
  parameter_derived -> per-step variable (dynamic ``ag_frac``),
  handled by declaration override.

- **Upstream agrees with the sibling reading**: its
  PRMSSoilzoneAgObsET derives from ConservativeProcess directly, not
  from PRMSSoilzone. The dual-area soil zone is a parallel
  formulation, not soilzone-plus.

- **The cost is duplicating ~50 DataArrayMeta blocks** -- pure
  declarations: cheap, greppable, pinned by parity tests. If a real
  shared core ever emerges, both families can be refactored onto it
  later (stance clause (c)); an artificial ancestor is not
  manufactured now to save declaration lines.

Within the ag family the hierarchy IS clean additive:
``PRMSSoilzoneAg`` is the minimal core (non-iterating dual-area soil
zone; ``ag_irrigation_add`` declared here as a normal, zero variable
feeding the kernel -- irrigation water is ag-core physics), and
``PRMSSoilzoneAgObsET`` EXTENDS it by adding ``aet_observed``, the
convergence parameters, and the iteration diagnostics, overriding
``calculate()`` with the It0 save/restore loop that SETS
``ag_irrigation_add`` and re-runs the same kernel. The core never
knows iteration exists.

Port notes (upstream quirks preserved verbatim)
-----------------------------------------------
- The AG ``compute_soilmoist`` call uses ``soil2gw_max``, NOT
  ``ag_soil2gw_max`` (upstream preserves this Fortran bug and says
  so) -- hence ``ag_soil2gw_max`` is unused and not declared. Also
  unused/not ported: ``ag_covden_sum``/``ag_covden_win``,
  ``soil_zone_max`` (computed upstream, never read), the Budget
  machinery, restart, calc_method, verbose, ``adjust_parameters``
  warn machinery (true-parameter edits guard with
  NotImplementedError as in prms_soilzone; derived/state clamps
  applied silently).
- ``ag_actet`` adds canopy interception BACK after the ag szactet
  (GSFLOW 2.4.1); the ag AET target subtracts ONLY hru_intcpevap.
- ``soil_lower_ratio`` carries upstream's deliberate python-side
  1e-4 rounding cap (caps ratio AND soil_lower).
- LAKE HRUs early-out after the hru_actet seed; INACTIVE HRUs are
  skipped entirely; the storage-change lines run for ALL HRUs
  post-loop (with the dynamic-ag_frac redistribution subtraction).
- ``iter_aet_flag`` is a runtime kernel option: this core passes
  False and its (zero) ``ag_irrigation_add``; the aet_external
  argument is never read under False (the core passes
  ag_irrigation_add again -- benign aliasing, neither is written).
- ``hru_area_perv``/``hru_frac_perv``/``ag_area``/``soil_moist_tot``
  are per-step VARIABLES (dynamic ag_frac): the upstream
  itime_step==0 area block runs at istep0 in calculate() (ag_frac is
  an input, not fed at initialize); ``_update_areas`` (upstream
  _update_ag_areas, np.isclose semantics hand-coded) runs each step
  BEFORE the kernel and redistributes storages when ag_frac changes.
"""

import numba
import numpy as np

from globals import Time
from hydrology.prms_soilzone import (
    _INACTIVE,
    _LAKE,
    _SWALE,
    compute_gwflow,
    compute_interflow,
    compute_soilmoist,
    compute_szactet,
)
from process import DataArrayMeta, Process


def _meta(kind, description, dtype=np.float64, restart=False):
    return DataArrayMeta(
        kind=kind,
        dims=("space",),
        dtype=dtype,
        description=description,
        restart=restart,
    )


class PRMSSoilzoneAg(Process):
    """PRMS/GSFLOW agricultural soil zone: pervious AND agricultural
    capillary reservoirs per HRU plus the shared gravity/preferential
    reservoirs; produces interflow, recharge (soil_to_gw/ssr_to_gw),
    dunnian runoff (added IN PLACE to sroff), and per-area actual ET.

    The minimal core of the ag-soilzone family (no obs-ET iteration;
    ag_irrigation_add stays zero). Storage/fluxes in inches; *_vol in
    cubic feet via hru_in_to_cf.
    """

    # -- dis_hru variables (grid-owned; dis-first sourcing) --
    hru_type = _meta(
        "parameter",
        "HRU type (INACTIVE=0, LAND=1, LAKE=2, SWALE=3)",
        np.int64,
    )
    hru_area = _meta("parameter", "HRU area [acres]")
    hru_in_to_cf = _meta(
        "parameter", "Conversion of inches over the HRU to cubic feet"
    )

    # -- process parameters --
    cov_type = _meta("parameter", "Cover type (0=bare ... )", np.int64)
    dprst_frac = _meta(
        "parameter", "Fraction of HRU area with surface depressions [-]"
    )
    fastcoef_lin = _meta(
        "parameter", "Linear preferential-flow routing coefficient [1/day]"
    )
    fastcoef_sq = _meta(
        "parameter", "Non-linear preferential-flow routing coefficient [-]"
    )
    hru_percent_imperv = _meta(
        "parameter", "Fraction of HRU area that is impervious [-]"
    )
    pref_flow_den = _meta("parameter", "Preferential-flow pore density [-]")
    pref_flow_infil_frac = _meta(
        "parameter", "Fraction of infiltration to preferential flow [-]"
    )
    sat_threshold = _meta(
        "parameter", "Soil saturation threshold above field capacity [in]"
    )
    slowcoef_lin = _meta(
        "parameter", "Linear gravity-flow routing coefficient [1/day]"
    )
    slowcoef_sq = _meta(
        "parameter", "Non-linear gravity-flow routing coefficient [-]"
    )
    soil_moist_max = _meta(
        "parameter", "Maximum capillary-reservoir water capacity [inches]"
    )
    soil_moist_init_frac = _meta(
        "parameter", "Initial fraction of capillary storage [-]"
    )
    soil_rechr_init_frac = _meta(
        "parameter", "Initial fraction of recharge-zone storage [-]"
    )
    soil_rechr_max_frac = _meta(
        "parameter", "Recharge-zone maximum as fraction of soil_moist_max [-]"
    )
    soil_type = _meta("parameter", "Soil type (SAND=1 ... )", np.int64)
    soil2gw_max = _meta(
        "parameter", "Maximum capillary excess routed to GWR [inches/day]"
    )
    ssr2gw_exp = _meta("parameter", "Gravity-drainage exponent to GWR [-]")
    ssr2gw_rate = _meta(
        "parameter", "Gravity-drainage rate coefficient [fraction/day]"
    )
    ssstor_init_frac = _meta(
        "parameter", "Initial fraction of gravity+pref storage [-]"
    )
    ag_soil_type = _meta("parameter", "Ag soil type", np.int64)
    ag_cov_type = _meta("parameter", "Ag cover type", np.int64)
    ag_soil_moist_max = _meta(
        "parameter", "Maximum ag capillary-reservoir capacity [inches]"
    )
    ag_soil_moist_init_frac = _meta(
        "parameter", "Initial fraction of ag capillary storage [-]"
    )
    ag_soil_rechr_max_frac = _meta(
        "parameter", "Ag recharge-zone max as fraction of ag_soil_moist_max"
    )
    ag_soil_rechr_init_frac = _meta(
        "parameter", "Initial fraction of ag recharge-zone storage [-]"
    )

    # -- derived parameters (initialize(); STATIC, no ag_frac dep) --
    soil_rechr_max = _meta(
        "parameter_derived", "Recharge-zone maximum storage [in] (clamped)"
    )
    ag_soil_rechr_max = _meta(
        "parameter_derived", "Ag recharge-zone maximum storage [in] (clamped)"
    )
    _sat_threshold = _meta(
        "parameter_derived", "sat_threshold zeroed for INACTIVE|LAKE"
    )
    _pref_flow_den = _meta(
        "parameter_derived", "pref_flow_den zeroed for non-LAND"
    )
    pref_flow_thrsh = _meta(
        "parameter_derived", "Gravity storage above which flow goes pref [in]"
    )
    pref_flow_max = _meta(
        "parameter_derived", "Maximum preferential-flow storage [inches]"
    )
    soil_lower_max = _meta(
        "parameter_derived", "Lower-zone maximum storage [inches]"
    )
    ag_soil_lower_stor_max = _meta(
        "parameter_derived",
        "Ag lower-zone maximum [in] (PRE-clamp ag_soil_rechr_max, upstream "
        "order)",
    )
    hru_area_imperv = _meta("parameter_derived", "Impervious HRU area [acres]")

    # -- inputs --
    dprst_evap_hru = _meta("input", "Depression evaporation [in over HRU]")
    dprst_seep_hru = _meta("input", "Depression seepage to GW [in over HRU]")
    hru_impervevap = _meta("input", "Impervious evaporation [in over HRU]")
    hru_intcpevap = _meta("input", "Canopy evaporation [in over HRU]")
    infil = _meta("input", "Pervious infiltration [inches over PERVIOUS area]")
    infil_ag = _meta("input", "Ag infiltration [inches over AG area]")
    potet = _meta("input", "Potential evapotranspiration [inches]")
    transp_on = _meta("input", "Transpiration occurring (0/1 flag)")
    snow_evap = _meta("input", "Snow evaporation/sublimation [inches]")
    snowcov_area = _meta("input", "Snow-covered area fraction [-]")
    ag_frac = _meta(
        "input", "Agricultural fraction of HRU area [-] (TIME-VARYING)"
    )

    # -- MUTABLE inputs (runoff's variables, edited in place here) --
    sroff = _meta(
        "mutable_input", "Surface runoff [in] -- dunnian ADDED in place"
    )
    sroff_vol = _meta(
        "mutable_input", "Surface runoff volume [cf] -- recomputed"
    )

    # -- variables: areas (per-step under dynamic ag_frac) --
    # per-step areas: _update_areas reads the PREVIOUS step's areas
    # (old_ag_frac = ag_area/harea), and _istep0_areas only runs at
    # time zero -- so a restart must restore them (restart=True is a
    # prognostic-state marker here, not a storage)
    hru_area_perv = _meta(
        "variable", "Pervious HRU area [acres] (per-step)", restart=True
    )
    hru_frac_perv = _meta(
        "variable", "Pervious fraction of HRU area [-] (per-step)"
    )
    ag_area = _meta(
        "variable",
        "Agricultural HRU area [acres] (per-step)",
        restart=True,
    )

    # -- variables: pervious / shared --
    cap_infil_tot = _meta("variable", "Capillary infiltration [in over HRU]")
    cap_waterin = _meta("variable", "Capillary reservoir water in [inches]")
    dunnian_flow = _meta("variable", "Dunnian surface runoff [inches]")
    hru_actet = _meta("variable", "Actual ET [inches over the HRU]")
    perv_actet = _meta("variable", "Pervious actual ET [in over perv area]")
    perv_actet_hru = _meta("variable", "Pervious actual ET [in over HRU]")
    potet_lower = _meta("variable", "Potential ET from lower zone [inches]")
    potet_rechr = _meta("variable", "Potential ET from recharge zone [in]")
    pref_flow = _meta("variable", "Preferential interflow [inches]")
    pref_flow_in = _meta("variable", "Preferential reservoir inflow [in]")
    pref_flow_infil = _meta("variable", "Infiltration to pref reservoir [in]")
    pref_flow_stor = _meta(
        "variable", "Preferential reservoir storage [in]", restart=True
    )
    pref_flow_stor_change = _meta("variable", "Pref storage change [inches]")
    pref_flow_stor_prev = _meta("variable", "Pref storage, previous step")
    recharge = _meta("variable", "Recharge to groundwater [inches]")
    slow_flow = _meta("variable", "Slow (gravity) interflow [inches]")
    slow_stor = _meta(
        "variable", "Gravity reservoir storage [inches]", restart=True
    )
    slow_stor_change = _meta("variable", "Gravity storage change [inches]")
    slow_stor_prev = _meta("variable", "Gravity storage, previous step")
    soil_lower = _meta(
        "variable", "Lower capillary-zone storage [inches]", restart=True
    )
    soil_lower_change = _meta("variable", "Lower-zone storage change [in]")
    soil_lower_change_hru = _meta(
        "variable", "Lower-zone storage change [in over HRU]"
    )
    soil_lower_prev = _meta(
        "variable", "Lower-zone storage, previous step (PRMSRunoffAg input)"
    )
    soil_lower_ratio = _meta("variable", "Lower-zone fraction of maximum [-]")
    soil_moist = _meta(
        "variable", "Capillary reservoir storage [inches]", restart=True
    )
    soil_moist_tot = _meta(
        "variable", "Total soil-zone storage [in over HRU] (per-step areas)"
    )
    soil_rechr = _meta(
        "variable", "Recharge-zone storage [inches]", restart=True
    )
    soil_rechr_change = _meta("variable", "Recharge-zone change [inches]")
    soil_rechr_change_hru = _meta(
        "variable", "Recharge-zone change [in over HRU]"
    )
    soil_rechr_prev = _meta(
        "variable", "Recharge-zone storage, previous step (RunoffAg input)"
    )
    soil_saturated = _meta("variable", "Pervious soil saturated flag (0/1)")
    soil_to_gw = _meta("variable", "Capillary excess to GWR [inches]")
    soil_to_ssr = _meta("variable", "Capillary excess to gravity [inches]")
    perv_soil_to_gw = _meta("variable", "Pervious part of soil_to_gw [in]")
    perv_soil_to_gvr = _meta("variable", "Pervious part of soil_to_ssr [in]")
    ssr_to_gw = _meta("variable", "Gravity drainage to GWR [inches]")
    ssres_flow = _meta("variable", "Interflow to stream network [inches]")
    ssres_flow_vol = _meta("variable", "Interflow volume [cubic feet]")
    ssres_in = _meta("variable", "Inflow to gravity+pref reservoirs [in]")
    ssres_stor = _meta("variable", "Gravity + preferential storage [inches]")
    swale_actet = _meta("variable", "Swale ponded-water actual ET [inches]")
    unused_potet = _meta("variable", "Unsatisfied potential ET [inches]")
    perv_infil_hru = _meta(
        "variable", "Pervious infiltration [inches over HRU]"
    )

    # -- variables: agricultural --
    ag_cap_infil_tot = _meta("variable", "Ag infiltration [in over HRU]")
    ag_soil_moist = _meta(
        "variable", "Ag capillary storage [inches]", restart=True
    )
    ag_soil_moist_prev = _meta(
        "variable", "Ag capillary storage, previous step (RunoffAg input)"
    )
    ag_soil_moist_change = _meta("variable", "Ag capillary change [inches]")
    ag_soil_moist_change_hru = _meta(
        "variable", "Ag capillary change [in over HRU]"
    )
    ag_soil_rechr = _meta(
        "variable", "Ag recharge-zone storage [inches]", restart=True
    )
    ag_soil_rechr_prev = _meta(
        "variable", "Ag recharge-zone storage, previous step (RunoffAg input)"
    )
    ag_soil_rechr_change = _meta("variable", "Ag recharge-zone change [in]")
    ag_soil_rechr_change_hru = _meta(
        "variable", "Ag recharge-zone change [in over HRU]"
    )
    ag_soil_lower = _meta(
        "variable", "Ag lower-zone storage [inches]", restart=True
    )
    ag_soil_lower_change = _meta("variable", "Ag lower-zone change [inches]")
    ag_soil_lower_change_hru = _meta(
        "variable", "Ag lower-zone change [in over HRU]"
    )
    ag_actet = _meta(
        "variable", "Ag actual ET [in] (canopy interception added back)"
    )
    hru_ag_actet = _meta("variable", "Ag actual ET [inches over the HRU]")
    ag_potet_rechr = _meta("variable", "Ag potential ET, recharge zone [in]")
    ag_potet_lower = _meta("variable", "Ag potential ET, lower zone [in]")
    ag_soil_to_gw = _meta("variable", "Ag part of soil_to_gw [inches]")
    ag_soil_to_gvr = _meta("variable", "Ag part of soil_to_ssr [inches]")
    ag_hortonian = _meta(
        "variable", "Ag hortonian runoff [in] (zeroed; never written upstream)"
    )
    ag_soil_saturated = _meta("variable", "Ag soil saturated flag (0/1)")
    unused_ag_et = _meta(
        "variable", "Unsatisfied ag ET target [in] (= per-HRU unsatisfied)"
    )
    ag_irrigation_add = _meta(
        "variable",
        "Irrigation water added to ag area [in] (ZERO in this core; SET by "
        "the ObsET iteration)",
    )
    ag_infil_hru = _meta("variable", "Ag infiltration [inches over HRU]")

    # -- variables: dynamic-ag_frac redistribution tracking --
    ag_soil_moist_redistribution = _meta(
        "variable", "Ag capillary redistribution on ag_frac change [in]"
    )
    ag_soil_rechr_redistribution = _meta(
        "variable", "Ag recharge redistribution on ag_frac change [in]"
    )
    soil_rechr_redistribution = _meta(
        "variable", "Pervious recharge redistribution on ag_frac change [in]"
    )
    soil_lower_redistribution = _meta(
        "variable", "Pervious lower redistribution on ag_frac change [in]"
    )
    slow_stor_redistribution = _meta(
        "variable", "Gravity redistribution on ag_frac change [in]"
    )

    _ZERO_VARS = (
        "hru_area_perv",
        "hru_frac_perv",
        "ag_area",
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
        "pref_flow_stor_prev",
        "recharge",
        "slow_flow",
        "slow_stor",
        "slow_stor_change",
        "slow_stor_prev",
        "soil_lower",
        "soil_lower_change",
        "soil_lower_change_hru",
        "soil_lower_prev",
        "soil_lower_ratio",
        "soil_moist",
        "soil_moist_tot",
        "soil_rechr",
        "soil_rechr_change",
        "soil_rechr_change_hru",
        "soil_rechr_prev",
        "soil_saturated",
        "soil_to_gw",
        "soil_to_ssr",
        "perv_soil_to_gw",
        "perv_soil_to_gvr",
        "ssr_to_gw",
        "ssres_flow",
        "ssres_flow_vol",
        "ssres_in",
        "ssres_stor",
        "swale_actet",
        "unused_potet",
        "perv_infil_hru",
        "ag_cap_infil_tot",
        "ag_soil_moist",
        "ag_soil_moist_prev",
        "ag_soil_moist_change",
        "ag_soil_moist_change_hru",
        "ag_soil_rechr",
        "ag_soil_rechr_prev",
        "ag_soil_rechr_change",
        "ag_soil_rechr_change_hru",
        "ag_soil_lower",
        "ag_soil_lower_change",
        "ag_soil_lower_change_hru",
        "ag_actet",
        "hru_ag_actet",
        "ag_potet_rechr",
        "ag_potet_lower",
        "ag_soil_to_gw",
        "ag_soil_to_gvr",
        "ag_hortonian",
        "ag_soil_saturated",
        "unused_ag_et",
        "ag_irrigation_add",
        "ag_infil_hru",
        "ag_soil_moist_redistribution",
        "ag_soil_rechr_redistribution",
        "soil_rechr_redistribution",
        "soil_lower_redistribution",
        "slow_stor_redistribution",
    )

    # ------------------------------------------------------------------
    # Initialization (_initialize_soilzone_ag_data, the parts that do
    # NOT depend on ag_frac; area block runs at istep0 in calculate)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        obj = self._obj
        for name in self._ZERO_VARS:
            obj[name].values[:] = 0.0

        hru_type = obj["hru_type"].values
        hru_area = obj["hru_area"].values

        obj["hru_area_imperv"].values[:] = (
            obj["hru_percent_imperv"].values * hru_area
        )

        soil_rechr_max = obj["soil_rechr_max"].values
        soil_rechr_max[:] = (
            obj["soil_rechr_max_frac"].values * obj["soil_moist_max"].values
        )
        ag_soil_rechr_max = obj["ag_soil_rechr_max"].values
        ag_soil_rechr_max[:] = (
            obj["ag_soil_rechr_max_frac"].values
            * obj["ag_soil_moist_max"].values
        )

        wh_inactive_or_lake = np.where(
            (hru_type == _INACTIVE) | (hru_type == _LAKE)
        )
        sat_threshold = obj["_sat_threshold"].values
        sat_threshold[:] = obj["sat_threshold"].values
        sat_threshold[wh_inactive_or_lake] = 0.0
        pref_flow_den = obj["_pref_flow_den"].values
        pref_flow_den[:] = obj["pref_flow_den"].values
        pref_flow_den[np.where(hru_type != 1)] = 0.0  # non-LAND

        pfif = obj["pref_flow_infil_frac"].values
        if (pfif.min() < 0.0) or (pfif.max() > 1.0):
            raise ValueError(
                "Values of pref_flow_infil_frac outside of [0,1]."
            )

        # -- initial states (no-restart path) --
        soil_moist = obj["soil_moist"].values
        soil_rechr = obj["soil_rechr"].values
        ag_soil_moist = obj["ag_soil_moist"].values
        ag_soil_rechr = obj["ag_soil_rechr"].values
        soil_moist[:] = (
            obj["soil_moist_init_frac"].values * obj["soil_moist_max"].values
        )
        soil_rechr[:] = obj["soil_rechr_init_frac"].values * soil_rechr_max
        ag_soil_moist[:] = (
            obj["ag_soil_moist_init_frac"].values
            * obj["ag_soil_moist_max"].values
        )
        ag_soil_rechr[:] = (
            obj["ag_soil_rechr_init_frac"].values * ag_soil_rechr_max
        )

        ssres_stor = obj["ssres_stor"].values
        ssres_stor[:] = obj["ssstor_init_frac"].values * sat_threshold
        ssres_stor[wh_inactive_or_lake] = 0.0

        # no ag storage on LAKE|INACTIVE (ag_frac==0 asserted at istep0)
        ag_soil_moist[wh_inactive_or_lake] = 0.0
        ag_soil_rechr[wh_inactive_or_lake] = 0.0

        # upstream order: ag_soil_lower_stor_max from the PRE-clamp
        # ag_soil_rechr_max
        obj["ag_soil_lower_stor_max"].values[:] = (
            obj["ag_soil_moist_max"].values - ag_soil_rechr_max
        )

        # -- upstream "adjust_parameters" block, exact order. TRUE
        # parameter edits cannot happen against frozen parameters --
        if (obj["soil_moist_max"].values < 1.0e-5).any():
            raise NotImplementedError(
                "PRMSSoilzoneAg: soil_moist_max < 1e-5 requires upstream's "
                "parameter adjustment, not ported (frozen parameters)"
            )
        if (obj["ag_soil_moist_max"].values < 1.0e-5).any():
            raise NotImplementedError(
                "PRMSSoilzoneAg: ag_soil_moist_max < 1e-5 requires "
                "upstream's parameter adjustment, not ported"
            )
        soil_rechr_max[:] = np.where(
            soil_rechr_max < 1.0e-5, 1.0e-5, soil_rechr_max
        )
        ag_soil_rechr_max[:] = np.where(
            ag_soil_rechr_max < 1.0e-5, 1.0e-5, ag_soil_rechr_max
        )
        soil_rechr_max[:] = np.where(
            soil_rechr_max > obj["soil_moist_max"].values,
            obj["soil_moist_max"].values,
            soil_rechr_max,
        )
        ag_soil_rechr_max[:] = np.where(
            ag_soil_rechr_max > obj["ag_soil_moist_max"].values,
            obj["ag_soil_moist_max"].values,
            ag_soil_rechr_max,
        )
        soil_rechr[:] = np.where(
            soil_rechr > soil_rechr_max, soil_rechr_max, soil_rechr
        )
        ag_soil_rechr[:] = np.where(
            ag_soil_rechr > ag_soil_rechr_max,
            ag_soil_rechr_max,
            ag_soil_rechr,
        )
        soil_moist[:] = np.where(
            soil_moist > obj["soil_moist_max"].values,
            obj["soil_moist_max"].values,
            soil_moist,
        )
        ag_soil_moist[:] = np.where(
            ag_soil_moist > obj["ag_soil_moist_max"].values,
            obj["ag_soil_moist_max"].values,
            ag_soil_moist,
        )
        soil_rechr[:] = np.where(
            soil_rechr > soil_moist, soil_moist, soil_rechr
        )
        ag_soil_rechr[:] = np.where(
            ag_soil_rechr > ag_soil_moist, ag_soil_moist, ag_soil_rechr
        )
        ssres_stor[:] = np.where(
            ssres_stor > sat_threshold, sat_threshold, ssres_stor
        )

        # -- preferential-flow thresholds by hru_type --
        pref_flow_thrsh = obj["pref_flow_thrsh"].values
        pref_flow_max = obj["pref_flow_max"].values
        pref_flow_thrsh[:] = 0.0
        pref_flow_max[:] = 0.0
        wh_swale = np.where(hru_type == _SWALE)
        wh_land = np.where(hru_type == 1)
        pref_flow_thrsh[wh_swale] = sat_threshold[wh_swale]
        pref_flow_thrsh[wh_land] = sat_threshold[wh_land] * (
            1.0 - pref_flow_den[wh_land]
        )
        pref_flow_max[wh_land] = (
            sat_threshold[wh_land] - pref_flow_thrsh[wh_land]
        )

        # -- split initial ssres_stor into slow / preferential --
        slow_stor = obj["slow_stor"].values
        pref_flow_stor = obj["pref_flow_stor"].values
        wh_land_or_swale = np.where((hru_type == 1) | (hru_type == _SWALE))
        slow_stor[wh_land_or_swale] = np.minimum(
            ssres_stor[wh_land_or_swale], pref_flow_thrsh[wh_land_or_swale]
        )
        pref_flow_stor[wh_land_or_swale] = (
            ssres_stor[wh_land_or_swale] - slow_stor[wh_land_or_swale]
        )

        # SCALAR flag (upstream; unlike plain soilzone's per-HRU
        # array). NOTE MPI: this is any() over the LOCAL rows;
        # upstream's is global-any. Identical whenever pref_flow_den
        # is uniformly zero/nonzero across ranks (fgr_ag_2yr: all
        # zero); a mixed domain would need a global reduction, but
        # initialize() is LOCAL by contract -- revisit if such a
        # domain arrives (the flag gates the topfr carve-out even
        # where den == 0, so local!=global WOULD change answers).
        self._pref_flow_flag = bool((pref_flow_den > 0.0).any())

        obj["soil_lower"].values[:] = soil_moist - soil_rechr
        soil_lower_max = obj["soil_lower_max"].values
        soil_lower_max[:] = obj["soil_moist_max"].values - soil_rechr_max
        wh_soil_lower_stor = np.where(soil_lower_max > 0.0)
        obj["soil_lower_ratio"].values[wh_soil_lower_stor] = (
            obj["soil_lower"].values[wh_soil_lower_stor]
            / soil_lower_max[wh_soil_lower_stor]
        )

        # -- *_prev seeds --
        obj["soil_rechr_prev"].values[:] = soil_rechr
        obj["soil_lower_prev"].values[:] = obj["soil_lower"].values
        obj["ag_soil_moist_prev"].values[:] = ag_soil_moist
        obj["ag_soil_rechr_prev"].values[:] = ag_soil_rechr
        obj["pref_flow_stor_prev"].values[:] = pref_flow_stor
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
        obj["ag_soil_moist_prev"].values[:] = obj["ag_soil_moist"].values
        obj["ag_soil_rechr_prev"].values[:] = obj["ag_soil_rechr"].values

    @staticmethod
    @numba.njit
    def _update_areas(
        # updated in place
        ag_area: np.ndarray,
        hru_area_perv: np.ndarray,
        hru_frac_perv: np.ndarray,
        ag_soil_moist: np.ndarray,
        ag_soil_rechr: np.ndarray,
        slow_stor: np.ndarray,
        soil_moist: np.ndarray,
        soil_rechr: np.ndarray,
        # read-only
        ag_frac: np.ndarray,
        hru_area: np.ndarray,
        hru_area_imperv: np.ndarray,
        dprst_frac: np.ndarray,
        hru_type: np.ndarray,
    ) -> None:
        """upstream _update_ag_areas per element (dprst ACTIVE port);
        np.isclose semantics hand-coded: |a-b| <= 1e-8 + 1e-5*|b|."""
        for hh in range(hru_area.shape[0]):
            harea = hru_area[hh]
            old_ag_frac = ag_area[hh] / harea
            old_perv = hru_area_perv[hh]

            new_ag_area = ag_frac[hh] * harea
            new_perv = harea - hru_area_imperv[hh]
            if hru_type[hh] != _INACTIVE:
                new_perv = new_perv - dprst_frac[hh] * harea - new_ag_area

            # ag storage redistribution when ag_frac changed
            # (isclose b = old_ag_frac)
            if not (
                abs(ag_frac[hh] - old_ag_frac)
                <= 1.0e-8 + 1.0e-5 * abs(old_ag_frac)
            ):
                if ag_soil_moist[hh] > 0.0:
                    if ag_frac[hh] > 0.0:
                        if ag_frac[hh] < old_ag_frac:
                            # DECREASE: depths unchanged, excess volume
                            # to slow_stor
                            slow_stor[hh] = slow_stor[hh] + ag_soil_moist[
                                hh
                            ] * (old_ag_frac - ag_frac[hh])
                        elif old_ag_frac < ag_frac[hh]:
                            # INCREASE: scale down to conserve volume
                            scale = old_ag_frac / ag_frac[hh]
                            ag_soil_moist[hh] = ag_soil_moist[hh] * scale
                            ag_soil_rechr[hh] = ag_soil_rechr[hh] * scale
                    else:
                        # ag_frac went to ZERO: all water to slow_stor
                        slow_stor[hh] = (
                            slow_stor[hh] + ag_soil_moist[hh] * old_ag_frac
                        )
                        ag_soil_moist[hh] = 0.0
                        ag_soil_rechr[hh] = 0.0

            # pervious scaling when pervious area changed
            # (isclose b = new_perv)
            perv_changed = not (
                abs(old_perv - new_perv) <= 1.0e-8 + 1.0e-5 * abs(new_perv)
            )
            if perv_changed and (new_perv > 0.0):
                scale_perv = old_perv / new_perv
                soil_moist[hh] = soil_moist[hh] * scale_perv
                soil_rechr[hh] = soil_rechr[hh] * scale_perv
            if (old_perv > 0.0) and (abs(new_perv) <= 1.0e-8):
                soil_moist[hh] = 0.0
                soil_rechr[hh] = 0.0

            ag_area[hh] = new_ag_area
            hru_area_perv[hh] = new_perv
            if hru_type[hh] != _INACTIVE:
                hru_frac_perv[hh] = new_perv / harea

    @staticmethod
    @numba.njit
    def _pre(
        soil_lower: np.ndarray,
        ag_soil_moist_redistribution: np.ndarray,
        ag_soil_rechr_redistribution: np.ndarray,
        soil_rechr_redistribution: np.ndarray,
        soil_lower_redistribution: np.ndarray,
        slow_stor_redistribution: np.ndarray,
        soil_moist: np.ndarray,
        soil_rechr: np.ndarray,
        ag_soil_moist: np.ndarray,
        ag_soil_rechr: np.ndarray,
        slow_stor: np.ndarray,
        soil_rechr_prev: np.ndarray,
        soil_lower_prev: np.ndarray,
        slow_stor_prev: np.ndarray,
        ag_soil_moist_prev: np.ndarray,
        ag_soil_rechr_prev: np.ndarray,
    ) -> None:
        """Pre-kernel: soil_lower from the (possibly rescaled) state;
        redistribution = current - prev (nonzero only when ag_frac
        changed this step)."""
        for hh in range(soil_moist.shape[0]):
            soil_lower[hh] = soil_moist[hh] - soil_rechr[hh]
            ag_soil_moist_redistribution[hh] = (
                ag_soil_moist[hh] - ag_soil_moist_prev[hh]
            )
            ag_soil_rechr_redistribution[hh] = (
                ag_soil_rechr[hh] - ag_soil_rechr_prev[hh]
            )
            soil_rechr_redistribution[hh] = (
                soil_rechr[hh] - soil_rechr_prev[hh]
            )
            soil_lower_redistribution[hh] = (
                soil_lower[hh] - soil_lower_prev[hh]
            )
            slow_stor_redistribution[hh] = slow_stor[hh] - slow_stor_prev[hh]

    @staticmethod
    @numba.njit
    def _calculate(
        # options
        iter_aet_flag: bool,
        pref_flow_flag: bool,
        # state (in/out)
        soil_moist: np.ndarray,
        soil_rechr: np.ndarray,
        ag_soil_moist: np.ndarray,
        ag_soil_rechr: np.ndarray,
        pref_flow_stor: np.ndarray,
        slow_stor: np.ndarray,
        ssres_stor: np.ndarray,
        # outputs (written in place)
        soil_to_gw: np.ndarray,
        soil_to_ssr: np.ndarray,
        perv_soil_to_gw: np.ndarray,
        perv_soil_to_gvr: np.ndarray,
        ag_soil_to_gw: np.ndarray,
        ag_soil_to_gvr: np.ndarray,
        ssr_to_gw: np.ndarray,
        slow_flow: np.ndarray,
        ssres_flow: np.ndarray,
        potet_rechr: np.ndarray,
        potet_lower: np.ndarray,
        ag_potet_rechr: np.ndarray,
        ag_potet_lower: np.ndarray,
        cap_waterin: np.ndarray,
        cap_infil_tot: np.ndarray,
        ag_cap_infil_tot: np.ndarray,
        pref_flow_in: np.ndarray,
        pref_flow_infil: np.ndarray,
        pref_flow: np.ndarray,
        perv_actet: np.ndarray,
        perv_actet_hru: np.ndarray,
        ag_actet: np.ndarray,
        hru_ag_actet: np.ndarray,
        hru_actet: np.ndarray,
        soil_lower: np.ndarray,
        ag_soil_lower: np.ndarray,
        dunnian_flow: np.ndarray,
        soil_moist_tot: np.ndarray,
        ssres_in: np.ndarray,
        recharge: np.ndarray,
        unused_potet: np.ndarray,
        unused_ag_et: np.ndarray,
        ag_hortonian: np.ndarray,
        ag_soil_saturated: np.ndarray,
        swale_actet: np.ndarray,
        soil_saturated: np.ndarray,
        soil_lower_ratio: np.ndarray,
        # mutable input
        sroff: np.ndarray,
        # inputs
        dprst_evap_hru: np.ndarray,
        dprst_seep_hru: np.ndarray,
        hru_impervevap: np.ndarray,
        hru_intcpevap: np.ndarray,
        infil: np.ndarray,
        infil_ag: np.ndarray,
        potet: np.ndarray,
        transp_on: np.ndarray,
        snow_evap: np.ndarray,
        snowcov_area: np.ndarray,
        aet_external: np.ndarray,
        ag_irrigation_add: np.ndarray,
        # parameters + derived + per-step areas
        hru_type: np.ndarray,
        cov_type: np.ndarray,
        ag_cov_type: np.ndarray,
        fastcoef_lin: np.ndarray,
        fastcoef_sq: np.ndarray,
        hru_frac_perv: np.ndarray,
        ag_frac: np.ndarray,
        ag_area: np.ndarray,
        hru_area_perv: np.ndarray,
        pref_flow_den: np.ndarray,
        pref_flow_infil_frac: np.ndarray,
        sat_threshold: np.ndarray,
        slowcoef_lin: np.ndarray,
        slowcoef_sq: np.ndarray,
        soil_moist_max: np.ndarray,
        soil_rechr_max: np.ndarray,
        soil_type: np.ndarray,
        ag_soil_type: np.ndarray,
        ag_soil_moist_max: np.ndarray,
        ag_soil_rechr_max: np.ndarray,
        soil2gw_max: np.ndarray,
        ssr2gw_exp: np.ndarray,
        ssr2gw_rate: np.ndarray,
        pref_flow_max: np.ndarray,
        pref_flow_thrsh: np.ndarray,
        soil_lower_max: np.ndarray,
    ) -> None:
        # upstream _calculate_numpy (szrun_ag) verbatim per element,
        # MINUS the irrigation-adjustment block (ObsET's
        # _adjust_irrigation) and the storage-change lines (_post)
        nhru = hru_type.shape[0]

        soil_to_gw[:] = 0.0
        soil_to_ssr[:] = 0.0
        perv_soil_to_gw[:] = 0.0
        perv_soil_to_gvr[:] = 0.0
        ag_soil_to_gw[:] = 0.0
        ag_soil_to_gvr[:] = 0.0
        ssr_to_gw[:] = 0.0
        slow_flow[:] = 0.0
        ssres_flow[:] = 0.0
        potet_rechr[:] = 0.0
        potet_lower[:] = 0.0
        ag_potet_rechr[:] = 0.0
        ag_potet_lower[:] = 0.0
        ag_actet[:] = 0.0
        hru_ag_actet[:] = 0.0
        ag_soil_saturated[:] = 0.0
        ag_hortonian[:] = 0.0
        unused_ag_et[:] = 0.0
        ag_cap_infil_tot[:] = 0.0
        ag_soil_lower[:] = 0.0
        swale_actet[:] = 0.0
        soil_saturated[:] = 0.0
        dunnian_flow[:] = 0.0
        pref_flow_infil[:] = 0.0
        pref_flow_in[:] = 0.0
        pref_flow[:] = 0.0

        for hh in range(nhru):
            if hru_type[hh] == _INACTIVE:
                continue

            snow_free = 1.0 - snowcov_area[hh]

            # Initial AET from impervious, interception, snow, dprst
            hruactet = hru_impervevap[hh] + hru_intcpevap[hh] + snow_evap[hh]
            hruactet = hruactet + dprst_evap_hru[hh]

            if hru_type[hh] == _LAKE:
                unused_potet[hh] = potet[hh] - hruactet
                hru_actet[hh] = hruactet
                continue

            perv_frac = hru_frac_perv[hh]
            agfrac = ag_frac[hh]
            perv_area = hru_area_perv[hh]
            agarea = ag_area[hh]
            perv_on_flag = perv_area > 0.0
            ag_on_flag = agarea > 0.0

            avail_potet = potet[hh] - hruactet
            if avail_potet < 0.0:
                avail_potet = 0.0
                hruactet = potet[hh]

            dunnianflw = 0.0
            dunnianflw_pfr = 0.0

            capwater_maxin = infil[hh]
            ag_water_maxin = infil_ag[hh]
            if iter_aet_flag:
                ag_water_maxin = ag_water_maxin + ag_irrigation_add[hh]

            prefflow = 0.0
            if pref_flow_flag:
                if pref_flow_infil_frac[hh] > 0.0:
                    ag_pref_flow_maxin = 0.0
                    cap_pref_flow_maxin = 0.0

                    if ag_water_maxin > 0.0:
                        ag_pref_flow_maxin = (
                            ag_water_maxin * pref_flow_infil_frac[hh]
                        )
                        ag_water_maxin = ag_water_maxin - ag_pref_flow_maxin
                        ag_pref_flow_maxin = ag_pref_flow_maxin * agfrac

                    if capwater_maxin > 0.0:
                        cap_pref_flow_maxin = (
                            capwater_maxin * pref_flow_infil_frac[hh]
                        )
                        capwater_maxin = capwater_maxin - cap_pref_flow_maxin
                        cap_pref_flow_maxin = cap_pref_flow_maxin * perv_frac

                    pref_flow_maxin = cap_pref_flow_maxin + ag_pref_flow_maxin
                    pref_flow_stor[hh] = pref_flow_stor[hh] + pref_flow_maxin
                    dunnianflw_pfr = max(
                        0.0, pref_flow_stor[hh] - pref_flow_max[hh]
                    )
                    if dunnianflw_pfr > 0.0:
                        pref_flow_stor[hh] = pref_flow_max[hh]
                    pref_flow_infil[hh] = pref_flow_maxin - dunnianflw_pfr

            if perv_on_flag:
                cap_infil_tot[hh] = capwater_maxin * perv_frac
            if ag_on_flag:
                ag_cap_infil_tot[hh] = ag_water_maxin * agfrac

            # soil moisture, pervious then agricultural (the AG call
            # uses soil2gw_max NOT ag_soil2gw_max: upstream-preserved
            # Fortran bug)
            perv_soil_to_gvr[hh] = 0.0
            if perv_on_flag:
                if capwater_maxin + soil_moist[hh] > 0.0:
                    (
                        capwater_maxin,
                        soil_moist[hh],
                        soil_rechr[hh],
                        perv_soil_to_gw[hh],
                        perv_soil_to_gvr[hh],
                    ) = compute_soilmoist(
                        soil2gw_max[hh] > 0.0,
                        perv_frac,
                        soil_moist_max[hh],
                        soil_rechr_max[hh],
                        soil2gw_max[hh],
                        capwater_maxin,
                        soil_moist[hh],
                        soil_rechr[hh],
                        perv_soil_to_gw[hh],
                        perv_soil_to_gvr[hh],
                    )

            if ag_on_flag:
                if ag_water_maxin + ag_soil_moist[hh] > 0.0:
                    (
                        ag_water_maxin,
                        ag_soil_moist[hh],
                        ag_soil_rechr[hh],
                        ag_soil_to_gw[hh],
                        ag_soil_to_gvr[hh],
                    ) = compute_soilmoist(
                        soil2gw_max[hh] > 0.0,
                        agfrac,
                        ag_soil_moist_max[hh],
                        ag_soil_rechr_max[hh],
                        soil2gw_max[hh],
                        ag_water_maxin,
                        ag_soil_moist[hh],
                        ag_soil_rechr[hh],
                        ag_soil_to_gw[hh],
                        ag_soil_to_gvr[hh],
                    )

            soil_to_gw[hh] = perv_soil_to_gw[hh] + ag_soil_to_gw[hh]
            soil_to_ssr[hh] = perv_soil_to_gvr[hh] + ag_soil_to_gvr[hh]
            cap_waterin[hh] = capwater_maxin * perv_frac

            # slow interflow and gravity drainage
            availh2o = slow_stor[hh] + soil_to_ssr[hh]
            topfr = 0.0
            if hru_type[hh] != _SWALE:
                if pref_flow_flag:
                    topfr = max(0.0, availh2o - pref_flow_thrsh[hh])
                ssresin = soil_to_ssr[hh] - topfr
                slow_stor[hh] = availh2o - topfr
                if slow_stor[hh] > 0.0:
                    slow_stor[hh], slow_flow[hh] = compute_interflow(
                        slowcoef_lin[hh],
                        slowcoef_sq[hh],
                        ssresin,
                        slow_stor[hh],
                        slow_flow[hh],
                    )
            else:
                slow_stor[hh] = availh2o

            if slow_stor[hh] > 0.0 and ssr2gw_rate[hh] > 0.0:
                ssr_to_gw[hh], slow_stor[hh] = compute_gwflow(
                    ssr2gw_rate[hh],
                    ssr2gw_exp[hh],
                    slow_stor[hh],
                )

            # dunnian from PFR (upstream's exact branch structure)
            dunnianflw_gvr = 0.0
            if pref_flow_flag:
                if pref_flow_max[hh] > 0.0:
                    availh2o = pref_flow_stor[hh] + topfr
                    dunnianflw_gvr = max(0.0, availh2o - pref_flow_max[hh])
                    if dunnianflw_gvr > 0.0:
                        topfr = topfr - dunnianflw_gvr
                        if topfr < 0.0:
                            topfr = 0.0
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
            elif not (pref_flow_max[hh] > 0.0):
                if hru_type[hh] != _SWALE:
                    dunnianflw_gvr = topfr

            # actual evapotranspiration: ag first, then pervious
            pervactet = 0.0
            agactet = 0.0
            ag_hruactet = 0.0
            unsatisfied_ag_et = 0.0

            if ag_on_flag:
                if iter_aet_flag:
                    ag_AETtarget = aet_external[hh]
                else:
                    ag_AETtarget = potet[hh]

                ag_avail_targetAET = ag_AETtarget - hru_intcpevap[hh]
                if ag_avail_targetAET < 0.0:
                    ag_avail_targetAET = 0.0

                if ag_avail_targetAET > 0.0:
                    ag_et_type_gt_1 = (
                        (transp_on[hh] != 0.0) and (ag_cov_type[hh] > 0)
                    ) or (snow_free >= 0.01)
                    if ag_et_type_gt_1:
                        ag_pcts = ag_soil_moist[hh] / ag_soil_moist_max[hh]
                        if ag_pcts > 0.9999:
                            ag_soil_saturated[hh] = 1.0

                    (
                        ag_soil_moist[hh],
                        ag_soil_rechr[hh],
                        _discard,
                        ag_potet_rechr[hh],
                        ag_potet_lower[hh],
                        agactet,
                    ) = compute_szactet(
                        transp_on[hh],
                        ag_cov_type[hh],
                        ag_soil_type[hh],
                        ag_soil_moist_max[hh],
                        ag_soil_rechr_max[hh],
                        snow_free,
                        ag_soil_moist[hh],
                        ag_soil_rechr[hh],
                        ag_avail_targetAET,
                        ag_potet_rechr[hh],
                        ag_potet_lower[hh],
                    )
                    ag_hruactet = agactet * agfrac

                unsatisfied_ag_et = ag_avail_targetAET - agactet
                unused_ag_et[hh] = unsatisfied_ag_et
                # add canopy interception back (GSFLOW 2.4.1)
                ag_actet[hh] = agactet + hru_intcpevap[hh]

            avail_potet = potet[hh] - hruactet - ag_hruactet
            if avail_potet < 0.0:
                avail_potet = 0.0

            if soil_moist[hh] > 0.0 and avail_potet > 0.0:
                et_type_gt_1 = (
                    (transp_on[hh] != 0.0) and (cov_type[hh] > 0)
                ) or (snow_free >= 0.01)
                if et_type_gt_1:
                    pcts = soil_moist[hh] / soil_moist_max[hh]
                    if pcts > 0.9999:
                        soil_saturated[hh] = 1.0

                (
                    soil_moist[hh],
                    soil_rechr[hh],
                    _discard,
                    potet_rechr[hh],
                    potet_lower[hh],
                    pervactet,
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

            perv_actet_hru[hh] = pervactet * perv_frac
            hru_ag_actet[hh] = ag_hruactet
            hru_actet[hh] = hruactet + perv_actet_hru[hh] + ag_hruactet
            perv_actet[hh] = pervactet

            soil_lower[hh] = soil_moist[hh] - soil_rechr[hh]

            if hru_type[hh] != _SWALE:
                dunnianflw = dunnianflw_gvr + dunnianflw_pfr
                dunnian_flow[hh] = dunnianflw

                ssres_flow[hh] = slow_flow[hh]
                if pref_flow_flag:
                    if pref_flow_max[hh] > 0.0:
                        pref_flow[hh] = prefflow
                        ssres_flow[hh] = ssres_flow[hh] + prefflow

                # dunnian to surface runoff (upstream in-place edit)
                sroff[hh] = sroff[hh] + dunnian_flow[hh]
                ssres_stor[hh] = slow_stor[hh]
                if pref_flow_flag:
                    ssres_stor[hh] = ssres_stor[hh] + pref_flow_stor[hh]
            else:
                availh2o = slow_stor[hh] - sat_threshold[hh]
                swale_actet[hh] = 0.0
                if availh2o > 0.0:
                    unsatisfied_et = potet[hh] - hru_actet[hh]
                    if unsatisfied_et > 0.0:
                        availh2o = min(availh2o, unsatisfied_et)
                        swale_actet[hh] = availh2o
                        hru_actet[hh] = hru_actet[hh] + swale_actet[hh]
                        slow_stor[hh] = slow_stor[hh] - swale_actet[hh]
                ssres_stor[hh] = slow_stor[hh]

            if soil_lower_max[hh] > 0.0:
                soil_lower_ratio[hh] = soil_lower[hh] / soil_lower_max[hh]
                # upstream's deliberate python-side 1e-4 rounding cap
                if soil_lower_ratio[hh] > 1.0:
                    excess = soil_lower_ratio[hh] - 1.0
                    if excess < 1e-4:
                        soil_lower_ratio[hh] = 1.0
                        soil_lower[hh] = soil_lower_max[hh]

            ssres_in[hh] = soil_to_ssr[hh]
            if pref_flow_flag:
                ssres_in[hh] = ssres_in[hh] + pref_flow_infil[hh]

            unused_potet[hh] = potet[hh] - hru_actet[hh]

            soil_moist_tot[hh] = (
                ssres_stor[hh]
                + soil_moist[hh] * perv_frac
                + ag_soil_moist[hh] * agfrac
            )

            recharge[hh] = soil_to_gw[hh] + ssr_to_gw[hh]
            recharge[hh] = recharge[hh] + dprst_seep_hru[hh]

            if ag_on_flag:
                ag_soil_lower[hh] = ag_soil_moist[hh] - ag_soil_rechr[hh]

    @staticmethod
    @numba.njit
    def _post(
        # written in place
        pref_flow_stor_change: np.ndarray,
        soil_lower_change: np.ndarray,
        soil_rechr_change: np.ndarray,
        slow_stor_change: np.ndarray,
        soil_lower_change_hru: np.ndarray,
        soil_rechr_change_hru: np.ndarray,
        ag_soil_moist_change: np.ndarray,
        ag_soil_rechr_change: np.ndarray,
        ag_soil_lower_change: np.ndarray,
        ag_soil_moist_change_hru: np.ndarray,
        ag_soil_rechr_change_hru: np.ndarray,
        ag_soil_lower_change_hru: np.ndarray,
        sroff_vol: np.ndarray,
        ssres_flow_vol: np.ndarray,
        perv_infil_hru: np.ndarray,
        ag_infil_hru: np.ndarray,
        # read-only
        pref_flow_stor: np.ndarray,
        pref_flow_stor_prev: np.ndarray,
        soil_lower: np.ndarray,
        soil_lower_prev: np.ndarray,
        soil_lower_redistribution: np.ndarray,
        soil_rechr: np.ndarray,
        soil_rechr_prev: np.ndarray,
        soil_rechr_redistribution: np.ndarray,
        slow_stor: np.ndarray,
        slow_stor_prev: np.ndarray,
        slow_stor_redistribution: np.ndarray,
        hru_frac_perv: np.ndarray,
        ag_soil_moist: np.ndarray,
        ag_soil_moist_prev: np.ndarray,
        ag_soil_moist_redistribution: np.ndarray,
        ag_soil_rechr: np.ndarray,
        ag_soil_rechr_prev: np.ndarray,
        ag_soil_rechr_redistribution: np.ndarray,
        ag_frac: np.ndarray,
        sroff: np.ndarray,
        ssres_flow: np.ndarray,
        hru_in_to_cf: np.ndarray,
        infil: np.ndarray,
        infil_ag: np.ndarray,
    ) -> None:
        """upstream post-kernel lines folded per element: storage
        changes (redistribution-corrected, ALL hrus) + volumes +
        infiltration budget terms."""
        for hh in range(sroff.shape[0]):
            pref_flow_stor_change[hh] = (
                pref_flow_stor[hh] - pref_flow_stor_prev[hh]
            )
            soil_lower_change[hh] = (
                soil_lower[hh]
                - soil_lower_prev[hh]
                - soil_lower_redistribution[hh]
            )
            soil_rechr_change[hh] = (
                soil_rechr[hh]
                - soil_rechr_prev[hh]
                - soil_rechr_redistribution[hh]
            )
            slow_stor_change[hh] = (
                slow_stor[hh]
                - slow_stor_prev[hh]
                - slow_stor_redistribution[hh]
            )
            soil_lower_change_hru[hh] = (
                soil_lower_change[hh] * hru_frac_perv[hh]
            )
            soil_rechr_change_hru[hh] = (
                soil_rechr_change[hh] * hru_frac_perv[hh]
            )
            ag_soil_moist_change[hh] = (
                ag_soil_moist[hh]
                - ag_soil_moist_prev[hh]
                - ag_soil_moist_redistribution[hh]
            )
            ag_soil_rechr_change[hh] = (
                ag_soil_rechr[hh]
                - ag_soil_rechr_prev[hh]
                - ag_soil_rechr_redistribution[hh]
            )
            ag_soil_lower_change[hh] = (
                ag_soil_moist_change[hh] - ag_soil_rechr_change[hh]
            )
            ag_soil_moist_change_hru[hh] = (
                ag_soil_moist_change[hh] * ag_frac[hh]
            )
            ag_soil_rechr_change_hru[hh] = (
                ag_soil_rechr_change[hh] * ag_frac[hh]
            )
            ag_soil_lower_change_hru[hh] = (
                ag_soil_lower_change[hh] * ag_frac[hh]
            )
            sroff_vol[hh] = sroff[hh] * hru_in_to_cf[hh]
            ssres_flow_vol[hh] = ssres_flow[hh] * hru_in_to_cf[hh]
            perv_infil_hru[hh] = infil[hh] * hru_frac_perv[hh]
            ag_infil_hru[hh] = infil_ag[hh] * ag_frac[hh]

    def _calculate_family(self, iter_aet_flag: bool) -> None:
        """One physics pass over the shared family kernel (the ObsET
        subclass calls this inside its iteration loop)."""
        obj = self._obj
        if iter_aet_flag:
            aet_external = obj["AET_external"].values
        else:
            # never read under iter_aet_flag=False; benign aliasing
            aet_external = obj["ag_irrigation_add"].values
        self._calculate(
            iter_aet_flag,
            self._pref_flow_flag,
            obj["soil_moist"].values,
            obj["soil_rechr"].values,
            obj["ag_soil_moist"].values,
            obj["ag_soil_rechr"].values,
            obj["pref_flow_stor"].values,
            obj["slow_stor"].values,
            obj["ssres_stor"].values,
            obj["soil_to_gw"].values,
            obj["soil_to_ssr"].values,
            obj["perv_soil_to_gw"].values,
            obj["perv_soil_to_gvr"].values,
            obj["ag_soil_to_gw"].values,
            obj["ag_soil_to_gvr"].values,
            obj["ssr_to_gw"].values,
            obj["slow_flow"].values,
            obj["ssres_flow"].values,
            obj["potet_rechr"].values,
            obj["potet_lower"].values,
            obj["ag_potet_rechr"].values,
            obj["ag_potet_lower"].values,
            obj["cap_waterin"].values,
            obj["cap_infil_tot"].values,
            obj["ag_cap_infil_tot"].values,
            obj["pref_flow_in"].values,
            obj["pref_flow_infil"].values,
            obj["pref_flow"].values,
            obj["perv_actet"].values,
            obj["perv_actet_hru"].values,
            obj["ag_actet"].values,
            obj["hru_ag_actet"].values,
            obj["hru_actet"].values,
            obj["soil_lower"].values,
            obj["ag_soil_lower"].values,
            obj["dunnian_flow"].values,
            obj["soil_moist_tot"].values,
            obj["ssres_in"].values,
            obj["recharge"].values,
            obj["unused_potet"].values,
            obj["unused_ag_et"].values,
            obj["ag_hortonian"].values,
            obj["ag_soil_saturated"].values,
            obj["swale_actet"].values,
            obj["soil_saturated"].values,
            obj["soil_lower_ratio"].values,
            obj["sroff"].values,
            obj["dprst_evap_hru"].values,
            obj["dprst_seep_hru"].values,
            obj["hru_impervevap"].values,
            obj["hru_intcpevap"].values,
            obj["infil"].values,
            obj["infil_ag"].values,
            obj["potet"].values,
            obj["transp_on"].values,
            obj["snow_evap"].values,
            obj["snowcov_area"].values,
            aet_external,
            obj["ag_irrigation_add"].values,
            obj["hru_type"].values,
            obj["cov_type"].values,
            obj["ag_cov_type"].values,
            obj["fastcoef_lin"].values,
            obj["fastcoef_sq"].values,
            obj["hru_frac_perv"].values,
            obj["ag_frac"].values,
            obj["ag_area"].values,
            obj["hru_area_perv"].values,
            obj["_pref_flow_den"].values,
            obj["pref_flow_infil_frac"].values,
            obj["_sat_threshold"].values,
            obj["slowcoef_lin"].values,
            obj["slowcoef_sq"].values,
            obj["soil_moist_max"].values,
            obj["soil_rechr_max"].values,
            obj["soil_type"].values,
            obj["ag_soil_type"].values,
            obj["ag_soil_moist_max"].values,
            obj["ag_soil_rechr_max"].values,
            obj["soil2gw_max"].values,
            obj["ssr2gw_exp"].values,
            obj["ssr2gw_rate"].values,
            obj["pref_flow_max"].values,
            obj["pref_flow_thrsh"].values,
            obj["soil_lower_max"].values,
        )

    def _pre_kernel(self) -> None:
        """istep0 areas (once) + per-step area update + pre lines."""
        obj = self._obj
        self._update_areas(
            obj["ag_area"].values,
            obj["hru_area_perv"].values,
            obj["hru_frac_perv"].values,
            obj["ag_soil_moist"].values,
            obj["ag_soil_rechr"].values,
            obj["slow_stor"].values,
            obj["soil_moist"].values,
            obj["soil_rechr"].values,
            obj["ag_frac"].values,
            obj["hru_area"].values,
            obj["hru_area_imperv"].values,
            obj["dprst_frac"].values,
            obj["hru_type"].values,
        )
        self._pre(
            obj["soil_lower"].values,
            obj["ag_soil_moist_redistribution"].values,
            obj["ag_soil_rechr_redistribution"].values,
            obj["soil_rechr_redistribution"].values,
            obj["soil_lower_redistribution"].values,
            obj["slow_stor_redistribution"].values,
            obj["soil_moist"].values,
            obj["soil_rechr"].values,
            obj["ag_soil_moist"].values,
            obj["ag_soil_rechr"].values,
            obj["slow_stor"].values,
            obj["soil_rechr_prev"].values,
            obj["soil_lower_prev"].values,
            obj["slow_stor_prev"].values,
            obj["ag_soil_moist_prev"].values,
            obj["ag_soil_rechr_prev"].values,
        )

    def _istep0_areas(self) -> None:
        """upstream _initialize_soilzone_ag_data area block: runs at
        time zero in calculate() (ag_frac now fed); once-only numpy
        staging."""
        obj = self._obj
        hru_type = obj["hru_type"].values
        hru_area = obj["hru_area"].values

        wh_no_ag = np.where((hru_type == _LAKE) | (hru_type == _INACTIVE))
        # GSFLOW zeroes ag_frac on this mask; upstream (and this port)
        # will NOT edit inputs -- assert instead
        assert (obj["ag_frac"].values[wh_no_ag] == 0.0).all()

        np.multiply(obj["ag_frac"].values, hru_area, out=obj["ag_area"].values)
        hru_area_perv = obj["hru_area_perv"].values
        hru_area_perv[:] = hru_area - obj["hru_area_imperv"].values
        wh_active = np.where(hru_type != _INACTIVE)
        dprst_area_max = obj["dprst_frac"].values * hru_area
        hru_area_perv[wh_active] = (
            hru_area_perv[wh_active]
            - dprst_area_max[wh_active]
            - obj["ag_area"].values[wh_active]
        )
        obj["hru_frac_perv"].values[:] = 0.0
        obj["hru_frac_perv"].values[wh_active] = (
            hru_area_perv[wh_active] / hru_area[wh_active]
        )
        obj["soil_moist_tot"].values[:] = (
            obj["ssres_stor"].values
            + obj["soil_moist"].values * obj["hru_frac_perv"].values
            + obj["ag_soil_moist"].values * obj["ag_frac"].values
        )

    def _post_kernel(self) -> None:
        obj = self._obj
        self._post(
            obj["pref_flow_stor_change"].values,
            obj["soil_lower_change"].values,
            obj["soil_rechr_change"].values,
            obj["slow_stor_change"].values,
            obj["soil_lower_change_hru"].values,
            obj["soil_rechr_change_hru"].values,
            obj["ag_soil_moist_change"].values,
            obj["ag_soil_rechr_change"].values,
            obj["ag_soil_lower_change"].values,
            obj["ag_soil_moist_change_hru"].values,
            obj["ag_soil_rechr_change_hru"].values,
            obj["ag_soil_lower_change_hru"].values,
            obj["sroff_vol"].values,
            obj["ssres_flow_vol"].values,
            obj["perv_infil_hru"].values,
            obj["ag_infil_hru"].values,
            obj["pref_flow_stor"].values,
            obj["pref_flow_stor_prev"].values,
            obj["soil_lower"].values,
            obj["soil_lower_prev"].values,
            obj["soil_lower_redistribution"].values,
            obj["soil_rechr"].values,
            obj["soil_rechr_prev"].values,
            obj["soil_rechr_redistribution"].values,
            obj["slow_stor"].values,
            obj["slow_stor_prev"].values,
            obj["slow_stor_redistribution"].values,
            obj["hru_frac_perv"].values,
            obj["ag_soil_moist"].values,
            obj["ag_soil_moist_prev"].values,
            obj["ag_soil_moist_redistribution"].values,
            obj["ag_soil_rechr"].values,
            obj["ag_soil_rechr_prev"].values,
            obj["ag_soil_rechr_redistribution"].values,
            obj["ag_frac"].values,
            obj["sroff"].values,
            obj["ssres_flow"].values,
            obj["hru_in_to_cf"].values,
            obj["infil"].values,
            obj["infil_ag"].values,
        )

    def calculate(self, dt: np.float64, time: Time) -> None:
        if time.current_index == 0:
            self._istep0_areas()
        self._pre_kernel()
        self._calculate_family(iter_aet_flag=False)
        self._post_kernel()


class PRMSSoilzoneAgObsET(PRMSSoilzoneAg):
    """PRMSSoilzoneAg PLUS observed-AET iteration: per timestep,
    repeatedly re-run the family kernel from saved initial state (It0),
    ADDING irrigation water (``ag_irrigation_add``) until the ag actual
    ET matches the observed target within ``soilzone_aet_converge`` (or
    the soil deficit / iteration caps bind). The core kernel is reused
    unchanged; the irrigation adjustment (upstream's in-kernel block)
    and the convergence diagnostics live only here.
    """

    # -- process parameters (ADDED; iteration control) --
    max_soilzone_ag_iter = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.int64,
        description="Maximum obs-AET iterations per timestep",
    )
    soilzone_aet_converge = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.float64,
        description="Convergence tolerance on unsatisfied ag AET [inches]",
    )
    ag_soilwater_deficit_min = _meta(
        "parameter", "Minimum ag soil-water deficit to allow irrigation [-]"
    )

    # -- inputs (ADDED) --
    aet_observed = _meta(
        "input", "Observed actual ET [inches] (-1.0 = missing)"
    )

    # -- variables (ADDED; obs-ET machinery + diagnostics) --
    AET_external = _meta(
        "variable", "Validated observed-AET target [inches] (upstream name)"
    )
    ag_soilwater_deficit = _meta(
        "variable", "Ag soil-water deficit fraction [-] (iteration diag)"
    )
    ag_irrigation_add_vol = _meta(
        "variable", "Irrigation volume added [acre-inches]"
    )
    ag_aet_external_vol = _meta(
        "variable", "Ag actual-ET volume [acre-inches]"
    )
    ag_irrigation_hru_source = _meta(
        "variable", "Irrigation water added [inches over the HRU]"
    )
    iter_count = _meta(
        "variable", "Iterations performed this timestep", np.int64
    )
    iter_end_status = _meta(
        "variable",
        "Convergence code (-1 no-ag, 0 converged, 1 deficit-limited, "
        "2 max-iters, 3 no-transp, -9 never)",
        np.int64,
    )

    _OBS_ZERO_VARS = (
        "AET_external",
        "ag_soilwater_deficit",
        "ag_irrigation_add_vol",
        "ag_aet_external_vol",
        "ag_irrigation_hru_source",
        "iter_count",
        "iter_end_status",
    )

    def initialize(self) -> None:
        super().initialize()
        obj = self._obj
        for name in self._OBS_ZERO_VARS:
            obj[name].values[:] = 0
        # It0 iteration scratch (persistent; prime directive: memory)
        nhru = obj["hru_area"].values.shape[0]
        self._it0 = {
            name: np.zeros(nhru, dtype=np.float64)
            for name in (
                "soil_moist",
                "soil_rechr",
                "ssres_stor",
                "slow_stor",
                "ag_soil_moist",
                "ag_soil_rechr",
                "pref_flow_stor",
            )
        }

    @staticmethod
    @numba.njit
    def _prep_aet(
        AET_external: np.ndarray,
        ag_irrigation_add: np.ndarray,
        ag_irrigation_add_vol: np.ndarray,
        ag_aet_external_vol: np.ndarray,
        aet_observed: np.ndarray,
        ag_frac: np.ndarray,
        transp_on: np.ndarray,
    ) -> int:
        """Upstream per-step ObsET prep (climate_hru_debug.f90 lines
        113-127): copy observed to the working target, zero negative
        non-missing values on ag HRUs. Returns the transp_on count
        (the loop's transp_on.any())."""
        n_transp = 0
        for hh in range(AET_external.shape[0]):
            ag_irrigation_add[hh] = 0.0
            ag_irrigation_add_vol[hh] = 0.0
            ag_aet_external_vol[hh] = 0.0
            AET_external[hh] = aet_observed[hh]
            if (
                (AET_external[hh] < 0.0)
                and (AET_external[hh] != -1.0)
                and (ag_frac[hh] > 0.0)
            ):
                AET_external[hh] = 0.0
            if transp_on[hh] != 0.0:
                n_transp += 1
        return n_transp

    @staticmethod
    @numba.njit
    def _adjust_irrigation(
        ag_soilwater_deficit: np.ndarray,
        ag_irrigation_add: np.ndarray,
        unused_ag_et: np.ndarray,
        ag_soil_moist: np.ndarray,
        ag_soil_moist_max: np.ndarray,
        transp_on: np.ndarray,
        ag_area: np.ndarray,
        soil_iter: int,
        max_soilzone_ag_iter: int,
        soilzone_aet_converge: float,
        ag_soilwater_deficit_min: np.ndarray,
    ):
        """The irrigation-adjustment block extracted from upstream's
        kernel (its lines 2181-2210 + the post-loop reduction): decide
        which HRUs need more irrigation and bump ag_irrigation_add.
        unused_ag_et carries the kernel's per-HRU unsatisfied ag ET."""
        add_estimated_irrigation = False
        num_hrus_ag_iter = 0
        unsatisfied_big = 0.0
        ag_soilwater_deficit[:] = 0.0
        for hh in range(ag_area.shape[0]):
            if (ag_area[hh] > 0.0) and (transp_on[hh] != 0.0):
                unsatisfied_ag_et = unused_ag_et[hh]
                if unsatisfied_ag_et > soilzone_aet_converge:
                    ag_soilwater_deficit[hh] = (
                        ag_soil_moist_max[hh] - ag_soil_moist[hh]
                    ) / ag_soil_moist_max[hh]
                    if ag_soilwater_deficit[hh] > ag_soilwater_deficit_min[hh]:
                        # only add irrigation if we'll iterate again
                        if soil_iter < max_soilzone_ag_iter:
                            unsatisfied_max = unsatisfied_ag_et
                            if unsatisfied_ag_et > ag_soil_moist_max[hh]:
                                unsatisfied_max = ag_soil_moist_max[hh]
                            else:
                                # speed convergence after 20 iters
                                if soil_iter > 20:
                                    unsatisfied_max = (
                                        unsatisfied_max + unsatisfied_ag_et
                                    )
                                # upstream marks needs-irrigation ONLY
                                # in this (uncapped) branch
                                add_estimated_irrigation = True
                                num_hrus_ag_iter += 1
                            ag_irrigation_add[hh] = (
                                ag_irrigation_add[hh] + unsatisfied_max
                            )
                            if unsatisfied_max > unsatisfied_big:
                                unsatisfied_big = unsatisfied_max
        return add_estimated_irrigation, num_hrus_ag_iter, unsatisfied_big

    @staticmethod
    @numba.njit
    def _iter_diag(
        iter_count: np.ndarray,
        iter_end_status: np.ndarray,
        ag_irrigation_add_vol: np.ndarray,
        ag_aet_external_vol: np.ndarray,
        ag_irrigation_hru_source: np.ndarray,
        ag_irrigation_add: np.ndarray,
        ag_actet: np.ndarray,
        ag_area: np.ndarray,
        ag_frac: np.ndarray,
        transp_on: np.ndarray,
        unused_ag_et: np.ndarray,
        ag_soilwater_deficit: np.ndarray,
        ag_soilwater_deficit_min: np.ndarray,
        soil_iter: int,
        max_soilzone_ag_iter: int,
        soilzone_aet_converge: float,
    ) -> None:
        """Post-iteration diagnostics + the iter-only volume/source
        outputs (upstream _calculate tail)."""
        for hh in range(ag_frac.shape[0]):
            if ag_frac[hh] <= 0.0:
                iter_end_status[hh] = -1
            elif transp_on[hh] == 0.0:
                iter_end_status[hh] = 3
            elif unused_ag_et[hh] <= soilzone_aet_converge:
                iter_end_status[hh] = 0
            elif ag_soilwater_deficit[hh] <= ag_soilwater_deficit_min[hh]:
                iter_end_status[hh] = 1
            elif soil_iter >= max_soilzone_ag_iter:
                iter_end_status[hh] = 2
            else:
                iter_end_status[hh] = -9  # upstream: "should never"
            iter_count[hh] = soil_iter
            ag_irrigation_add_vol[hh] = ag_irrigation_add[hh] * ag_area[hh]
            ag_aet_external_vol[hh] = ag_actet[hh] * ag_area[hh]
            ag_irrigation_hru_source[hh] = ag_irrigation_add[hh] * ag_frac[hh]

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        if time.current_index == 0:
            self._istep0_areas()
        self._pre_kernel()

        # store initial values for iteration (Fortran It0 variables)
        for name, buf in self._it0.items():
            buf[:] = obj[name].values

        n_transp = self._prep_aet(
            obj["AET_external"].values,
            obj["ag_irrigation_add"].values,
            obj["ag_irrigation_add_vol"].values,
            obj["ag_aet_external_vol"].values,
            obj["aet_observed"].values,
            obj["ag_frac"].values,
            obj["transp_on"].values,
        )

        max_iter = int(obj["max_soilzone_ag_iter"].values[0])
        converge = float(obj["soilzone_aet_converge"].values[0])

        keep_iterating = True
        soil_iter = 1
        while keep_iterating:
            if soil_iter > 1:
                for name, buf in self._it0.items():
                    if name == "pref_flow_stor" and not (self._pref_flow_flag):
                        continue
                    obj[name].values[:] = buf

            self._calculate_family(iter_aet_flag=True)

            (
                add_estimated_irrigation,
                _num_hrus_ag_iter,
                _unsatisfied_big,
            ) = self._adjust_irrigation(
                obj["ag_soilwater_deficit"].values,
                obj["ag_irrigation_add"].values,
                obj["unused_ag_et"].values,
                obj["ag_soil_moist"].values,
                obj["ag_soil_moist_max"].values,
                obj["transp_on"].values,
                obj["ag_area"].values,
                soil_iter,
                max_iter,
                converge,
                obj["ag_soilwater_deficit_min"].values,
            )

            if not add_estimated_irrigation:
                keep_iterating = False
            soil_iter += 1
            if n_transp == 0:
                keep_iterating = False
            if soil_iter > max_iter:
                keep_iterating = False

        soil_iter -= 1

        self._iter_diag(
            obj["iter_count"].values,
            obj["iter_end_status"].values,
            obj["ag_irrigation_add_vol"].values,
            obj["ag_aet_external_vol"].values,
            obj["ag_irrigation_hru_source"].values,
            obj["ag_irrigation_add"].values,
            obj["ag_actet"].values,
            obj["ag_area"].values,
            obj["ag_frac"].values,
            obj["transp_on"].values,
            obj["unused_ag_et"].values,
            obj["ag_soilwater_deficit"].values,
            obj["ag_soilwater_deficit_min"].values,
            soil_iter,
            max_iter,
            converge,
        )

        self._post_kernel()
