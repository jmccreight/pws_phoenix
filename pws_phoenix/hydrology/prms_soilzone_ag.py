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
  is overridden anyway: ``init_from_params`` / ``init_state``
  (dual-area, ag storages),
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
  parameter_internal -> per-step variable (dynamic ``ag_frac``),
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
  an input, not fed when the init hooks run), unless the areas were
  supplied as initial state (``_istep0_areas``); ``_update_areas``
  (upstream _update_ag_areas, np.isclose semantics hand-coded) runs
  each step BEFORE the kernel and redistributes storages when ag_frac
  changes.
"""

import numba
import numpy as np

from pws_phoenix import NUMBA_PARALLEL

from pws_phoenix.clock import Time
from pws_phoenix.hydrology.prms_soilzone import (
    _INACTIVE,
    _LAKE,
    _SWALE,
    _ZERO_INCHES,
    compute_gwflow,
    compute_interflow,
    compute_soilmoist,
    compute_szactet,
)
from pws_phoenix.process import DataArrayMeta, Process

# _update_areas's np.isclose on the pervious AREA: its absolute
# tolerance is in acres (the fraction isclose beside it stays bare).
# Named so the units test (tests/test_kernel_units.py) can patch it to
# a Quantity; numba freezes it, compiled code unchanged.
_ISCLOSE_ATOL_ACRES = 1.0e-8

# ObsET's _prep_aet: the observed-AET missing-value sentinel is in
# inches (compared with the inch input), and its volume outputs are
# acre-inches; named for the same reason as above
_AET_MISSING_INCHES = -1.0
_ZERO_ACRE_INCHES = 0.0

# the state the obs-ET iteration rewinds between passes (Fortran It0
# variables), in the order the It0 buffers are held
_IT0_NAMES = (
    "soil_moist",
    "soil_rechr",
    "ssres_stor",
    "slow_stor",
    "ag_soil_moist",
    "ag_soil_rechr",
    "pref_flow_stor",
)


def _meta(
    kind,
    description,
    dtype=np.float64,
    *,
    units,
    restart=False,
    derivation=None,
):
    """A ("space",) declaration. ``units`` is REQUIRED (None only for
    an int field): tests/test_units.py's source scanner sees only
    literal ``DataArrayMeta(`` calls, so a default here once hid ~100
    unit-less float fields from it (Nov 2026)."""
    return DataArrayMeta(
        kind=kind,
        dims=("space",),
        dtype=dtype,
        description=description,
        restart=restart,
        derivation=derivation,
        units=units,
    )


def _term(description):
    """A mass-budget term: a variable in inches (units are required on
    budget terms; all of this family's terms are plain variables)."""
    return _meta("variable", description, units="inches")


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
        units=None,
    )
    hru_area = _meta("parameter", "HRU area [acres]", units="acres")
    hru_in_to_cf = _meta(
        "parameter",
        "Conversion of inches over the HRU to cubic feet",
        derivation="hru_area * 43560.0 / 12.0",
        units="cubic feet/inch",
    )

    # -- process parameters --
    cov_type = _meta(
        "parameter", "Cover type (0=bare ... )", np.int64, units=None
    )
    dprst_frac = _meta(
        "parameter",
        "Fraction of HRU area with surface depressions [-]",
        units="fraction",
    )
    fastcoef_lin = _meta(
        "parameter",
        "Linear preferential-flow routing coefficient [1/day]",
        units="1/day",
    )
    fastcoef_sq = _meta(
        "parameter",
        "Non-linear preferential-flow routing coefficient [-]",
        units="1/(inch day)",
    )
    hru_percent_imperv = _meta(
        "parameter",
        "Fraction of HRU area that is impervious [-]",
        units="fraction",
    )
    pref_flow_den = _meta(
        "parameter", "Preferential-flow pore density [-]", units="fraction"
    )
    pref_flow_infil_frac = _meta(
        "parameter",
        "Fraction of infiltration to preferential flow [-]",
        units="fraction",
    )
    sat_threshold = _meta(
        "parameter",
        "Soil saturation threshold above field capacity [in]",
        units="inches",
    )
    slowcoef_lin = _meta(
        "parameter",
        "Linear gravity-flow routing coefficient [1/day]",
        units="1/day",
    )
    slowcoef_sq = _meta(
        "parameter",
        "Non-linear gravity-flow routing coefficient [-]",
        units="1/(inch day)",
    )
    soil_moist_max = _meta(
        "parameter",
        "Maximum capillary-reservoir water capacity [inches]",
        units="inches",
    )
    soil_moist_init_frac = _meta(
        "parameter",
        "Initial fraction of capillary storage [-]",
        units="fraction",
    )
    soil_rechr_init_frac = _meta(
        "parameter",
        "Initial fraction of recharge-zone storage [-]",
        units="fraction",
    )
    soil_rechr_max_frac = _meta(
        "parameter",
        "Recharge-zone maximum as fraction of soil_moist_max [-]",
        units="fraction",
    )
    soil_type = _meta(
        "parameter", "Soil type (SAND=1 ... )", np.int64, units=None
    )
    soil2gw_max = _meta(
        "parameter",
        "Maximum capillary excess routed to GWR [inches/day]",
        units="inches/day",
    )
    ssr2gw_exp = _meta(
        "parameter", "Gravity-drainage exponent to GWR [-]", units="fraction"
    )
    ssr2gw_rate = _meta(
        "parameter",
        "Gravity-drainage rate coefficient [fraction/day]",
        units="1/day",
    )
    ssstor_init_frac = _meta(
        "parameter",
        "Initial fraction of gravity+pref storage [-]",
        units="fraction",
    )
    ag_soil_type = _meta("parameter", "Ag soil type", np.int64, units=None)
    ag_cov_type = _meta("parameter", "Ag cover type", np.int64, units=None)
    ag_soil_moist_max = _meta(
        "parameter",
        "Maximum ag capillary-reservoir capacity [inches]",
        units="inches",
    )
    ag_soil_moist_init_frac = _meta(
        "parameter",
        "Initial fraction of ag capillary storage [-]",
        units="fraction",
    )
    ag_soil_rechr_max_frac = _meta(
        "parameter",
        "Ag recharge-zone max as fraction of ag_soil_moist_max",
        units="fraction",
    )
    ag_soil_rechr_init_frac = _meta(
        "parameter",
        "Initial fraction of ag recharge-zone storage [-]",
        units="fraction",
    )

    # -- derived parameters (init_from_params(); STATIC, no ag_frac dep) --
    soil_rechr_max = _meta(
        "parameter_internal",
        "Recharge-zone maximum storage [in] (clamped)",
        units="inches",
    )
    ag_soil_rechr_max = _meta(
        "parameter_internal",
        "Ag recharge-zone maximum storage [in] (clamped)",
        units="inches",
    )
    _sat_threshold = _meta(
        "parameter_internal",
        "sat_threshold zeroed for INACTIVE|LAKE",
        units="inches",
    )
    _pref_flow_den = _meta(
        "parameter_internal",
        "pref_flow_den zeroed for non-LAND",
        units="fraction",
    )
    pref_flow_thrsh = _meta(
        "parameter_internal",
        "Gravity storage above which flow goes pref [in]",
        units="inches",
    )
    pref_flow_max = _meta(
        "parameter_internal",
        "Maximum preferential-flow storage [inches]",
        units="inches",
    )
    soil_lower_max = _meta(
        "parameter_internal",
        "Lower-zone maximum storage [inches]",
        units="inches",
    )
    ag_soil_lower_stor_max = _meta(
        "parameter_internal",
        "Ag lower-zone maximum [in] (PRE-clamp ag_soil_rechr_max, upstream "
        "order)",
        units="inches",
    )
    hru_area_imperv = _meta(
        "parameter_internal", "Impervious HRU area [acres]", units="acres"
    )

    # -- inputs --
    dprst_evap_hru = _meta(
        "input", "Depression evaporation [in over HRU]", units="inches"
    )
    dprst_seep_hru = _meta(
        "input", "Depression seepage to GW [in over HRU]", units="inches"
    )
    hru_impervevap = _meta(
        "input", "Impervious evaporation [in over HRU]", units="inches"
    )
    hru_intcpevap = _meta(
        "input", "Canopy evaporation [in over HRU]", units="inches"
    )
    infil = _meta(
        "input",
        "Pervious infiltration [inches over PERVIOUS area]",
        units="inches",
    )
    infil_ag = _meta(
        "input", "Ag infiltration [inches over AG area]", units="inches"
    )
    potet = _meta(
        "input", "Potential evapotranspiration [inches]", units="inches"
    )
    transp_on = _meta(
        "input", "Transpiration occurring (0/1 flag)", units="none"
    )
    snow_evap = _meta(
        "input", "Snow evaporation/sublimation [inches]", units="inches"
    )
    snowcov_area = _meta(
        "input", "Snow-covered area fraction [-]", units="fraction"
    )
    ag_frac = _meta(
        "input",
        "Agricultural fraction of HRU area [-] (TIME-VARYING)",
        units="fraction",
    )

    # -- MUTABLE inputs (runoff's variables, edited in place here) --
    sroff = _meta(
        "mutable_input",
        "Surface runoff [in] -- dunnian ADDED in place",
        units="inches",
    )
    sroff_vol = _meta(
        "mutable_input",
        "Surface runoff volume [cf] -- recomputed",
        units="cubic feet",
    )

    # -- variables: areas (per-step under dynamic ag_frac) --
    # per-step areas: _update_areas reads the PREVIOUS step's areas
    # (old_ag_frac = ag_area/harea), and _istep0_areas only runs at
    # time zero -- so a restart must restore them (restart=True is a
    # prognostic-state marker here, not a storage)
    hru_area_perv = _meta(
        "variable",
        "Pervious HRU area [acres] (per-step)",
        restart=True,
        units="acres",
    )
    hru_frac_perv = _meta(
        "variable",
        "Pervious fraction of HRU area [-] (per-step)",
        units="fraction",
    )
    ag_area = _meta(
        "variable",
        "Agricultural HRU area [acres] (per-step)",
        restart=True,
        units="acres",
    )

    # -- variables: pervious / shared --
    cap_infil_tot = _meta(
        "variable", "Capillary infiltration [in over HRU]", units="inches"
    )
    cap_waterin = _meta(
        "variable", "Capillary reservoir water in [inches]", units="inches"
    )
    dunnian_flow = _term("Dunnian surface runoff [inches]")
    hru_actet = _meta(
        "variable", "Actual ET [inches over the HRU]", units="inches"
    )
    perv_actet = _meta(
        "variable", "Pervious actual ET [in over perv area]", units="inches"
    )
    perv_actet_hru = _term("Pervious actual ET [in over HRU]")
    potet_lower = _meta(
        "variable", "Potential ET from lower zone [inches]", units="inches"
    )
    potet_rechr = _meta(
        "variable", "Potential ET from recharge zone [in]", units="inches"
    )
    pref_flow = _term("Preferential interflow [inches]")
    pref_flow_in = _meta(
        "variable", "Preferential reservoir inflow [in]", units="inches"
    )
    pref_flow_infil = _meta(
        "variable", "Infiltration to pref reservoir [in]", units="inches"
    )
    pref_flow_stor = _meta(
        "variable",
        "Preferential reservoir storage [in]",
        restart=True,
        units="inches",
    )
    pref_flow_stor_change = _term("Pref storage change [inches]")
    pref_flow_stor_prev = _meta(
        "variable", "Pref storage, previous step", units="inches"
    )
    recharge = _meta(
        "variable", "Recharge to groundwater [inches]", units="inches"
    )
    slow_flow = _term("Slow (gravity) interflow [inches]")
    slow_stor = _meta(
        "variable",
        "Gravity reservoir storage [inches]",
        restart=True,
        units="inches",
    )
    slow_stor_change = _term("Gravity storage change [inches]")
    slow_stor_prev = _meta(
        "variable", "Gravity storage, previous step", units="inches"
    )
    soil_lower = _meta(
        "variable",
        "Lower capillary-zone storage [inches]",
        restart=True,
        units="inches",
    )
    soil_lower_change = _meta(
        "variable", "Lower-zone storage change [in]", units="inches"
    )
    soil_lower_change_hru = _term("Lower-zone storage change [in over HRU]")
    soil_lower_prev = _meta(
        "variable",
        "Lower-zone storage, previous step (PRMSRunoffAg input)",
        units="inches",
    )
    soil_lower_ratio = _meta(
        "variable", "Lower-zone fraction of maximum [-]", units="fraction"
    )
    soil_moist = _meta(
        "variable",
        "Capillary reservoir storage [inches]",
        restart=True,
        units="inches",
    )
    soil_moist_tot = _meta(
        "variable",
        "Total soil-zone storage [in over HRU] (per-step areas)",
        units="inches",
    )
    soil_rechr = _meta(
        "variable",
        "Recharge-zone storage [inches]",
        restart=True,
        units="inches",
    )
    soil_rechr_change = _meta(
        "variable", "Recharge-zone change [inches]", units="inches"
    )
    soil_rechr_change_hru = _term("Recharge-zone change [in over HRU]")
    soil_rechr_prev = _meta(
        "variable",
        "Recharge-zone storage, previous step (RunoffAg input)",
        units="inches",
    )
    soil_saturated = _meta(
        "variable", "Pervious soil saturated flag (0/1)", units="none"
    )
    soil_to_gw = _meta(
        "variable", "Capillary excess to GWR [inches]", units="inches"
    )
    soil_to_ssr = _meta(
        "variable", "Capillary excess to gravity [inches]", units="inches"
    )
    perv_soil_to_gw = _term("Pervious part of soil_to_gw [in]")
    perv_soil_to_gvr = _meta(
        "variable", "Pervious part of soil_to_ssr [in]", units="inches"
    )
    ssr_to_gw = _term("Gravity drainage to GWR [inches]")
    ssres_flow = _meta(
        "variable", "Interflow to stream network [inches]", units="inches"
    )
    ssres_flow_vol = _meta(
        "variable", "Interflow volume [cubic feet]", units="cubic feet"
    )
    ssres_in = _meta(
        "variable", "Inflow to gravity+pref reservoirs [in]", units="inches"
    )
    ssres_stor = _meta(
        "variable", "Gravity + preferential storage [inches]", units="inches"
    )
    swale_actet = _term("Swale ponded-water actual ET [inches]")
    unused_potet = _meta(
        "variable", "Unsatisfied potential ET [inches]", units="inches"
    )
    perv_infil_hru = _term("Pervious infiltration [inches over HRU]")

    # -- variables: agricultural --
    ag_cap_infil_tot = _meta(
        "variable", "Ag infiltration [in over HRU]", units="inches"
    )
    ag_soil_moist = _meta(
        "variable",
        "Ag capillary storage [inches]",
        restart=True,
        units="inches",
    )
    ag_soil_moist_prev = _meta(
        "variable",
        "Ag capillary storage, previous step (RunoffAg input)",
        units="inches",
    )
    ag_soil_moist_change = _meta(
        "variable", "Ag capillary change [inches]", units="inches"
    )
    ag_soil_moist_change_hru = _meta(
        "variable", "Ag capillary change [in over HRU]", units="inches"
    )
    ag_soil_rechr = _meta(
        "variable",
        "Ag recharge-zone storage [inches]",
        restart=True,
        units="inches",
    )
    ag_soil_rechr_prev = _meta(
        "variable",
        "Ag recharge-zone storage, previous step (RunoffAg input)",
        units="inches",
    )
    ag_soil_rechr_change = _meta(
        "variable", "Ag recharge-zone change [in]", units="inches"
    )
    ag_soil_rechr_change_hru = _term("Ag recharge-zone change [in over HRU]")
    ag_soil_lower = _meta(
        "variable",
        "Ag lower-zone storage [inches]",
        restart=True,
        units="inches",
    )
    ag_soil_lower_change = _meta(
        "variable", "Ag lower-zone change [inches]", units="inches"
    )
    ag_soil_lower_change_hru = _term("Ag lower-zone change [in over HRU]")
    ag_actet = _meta(
        "variable",
        "Ag actual ET [in] (canopy interception added back)",
        units="inches",
    )
    hru_ag_actet = _term("Ag actual ET [inches over the HRU]")
    ag_potet_rechr = _meta(
        "variable", "Ag potential ET, recharge zone [in]", units="inches"
    )
    ag_potet_lower = _meta(
        "variable", "Ag potential ET, lower zone [in]", units="inches"
    )
    ag_soil_to_gw = _term("Ag part of soil_to_gw [inches]")
    ag_soil_to_gvr = _meta(
        "variable", "Ag part of soil_to_ssr [inches]", units="inches"
    )
    ag_hortonian = _meta(
        "variable",
        "Ag hortonian runoff [in] (zeroed; never written upstream)",
        units="inches",
    )
    ag_soil_saturated = _meta(
        "variable", "Ag soil saturated flag (0/1)", units="none"
    )
    unused_ag_et = _meta(
        "variable",
        "Unsatisfied ag ET target [in] (= per-HRU unsatisfied)",
        units="inches",
    )
    ag_irrigation_add = _meta(
        "variable",
        "Irrigation water added to ag area [in] (ZERO in this core; SET by "
        "the ObsET iteration)",
        units="inches",
    )
    ag_infil_hru = _term("Ag infiltration [inches over HRU]")

    # -- variables: dynamic-ag_frac redistribution tracking --
    ag_soil_moist_redistribution = _meta(
        "variable",
        "Ag capillary redistribution on ag_frac change [in]",
        units="inches",
    )
    ag_soil_rechr_redistribution = _meta(
        "variable",
        "Ag recharge redistribution on ag_frac change [in]",
        units="inches",
    )
    soil_rechr_redistribution = _meta(
        "variable",
        "Pervious recharge redistribution on ag_frac change [in]",
        units="inches",
    )
    soil_lower_redistribution = _meta(
        "variable",
        "Pervious lower redistribution on ag_frac change [in]",
        units="inches",
    )
    slow_stor_redistribution = _meta(
        "variable",
        "Gravity redistribution on ag_frac change [in]",
        units="inches",
    )

    # -- budgets --
    # pywatershed's PRMSSoilzoneAg.get_mass_budget_terms plus
    # swale_actet as an output (the swale branch removes it from
    # slow_stor; see prms_soilzone.py). ag_irrigation_add is zero in
    # this core; ObsET adds it (as ag_irrigation_hru_source) to inputs.
    _MASS_BUDGET = {
        "inputs": ("perv_infil_hru", "ag_infil_hru"),
        "outputs": (
            "perv_actet_hru",
            "hru_ag_actet",
            "perv_soil_to_gw",
            "ag_soil_to_gw",
            "ssr_to_gw",
            "slow_flow",
            "dunnian_flow",
            "pref_flow",
            "swale_actet",
        ),
        "storage_changes": (
            "soil_rechr_change_hru",
            "soil_lower_change_hru",
            "slow_stor_change",
            "pref_flow_stor_change",
            "ag_soil_rechr_change_hru",
            "ag_soil_lower_change_hru",
        ),
    }
    budgets = {"mass": _MASS_BUDGET}

    # init_state() computes soil_lower = soil_moist - soil_rechr and
    # clamps soil_rechr <= soil_moist from the DEFAULTS (see
    # Process.state_groups). The two areas are the previous step's
    # values that _update_areas reads; the time-zero area block is
    # skipped only when they were supplied (see _istep0_areas), so
    # one without the other is an error.
    state_groups = (
        ("soil_moist", "soil_rechr", "soil_lower"),
        ("ag_area", "hru_area_perv"),
    )

    # zeroed by init_state(); hru_area_perv is NOT here: init_state()
    # leaves it at nan (see _istep0_areas)
    _ZERO_VARS = (
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

    def allocate(self) -> None:
        """The kernel argument tuples that hold field references only,
        assembled ONCE (the refs never change: DEVELOPER.md "Per-step
        data access"). The family kernel's tuple also carries
        ``_pref_flow_flag``, which is derived from a parameter, so
        init_from_params() builds that one."""
        self._update_areas_args = self._hold(
            "ag_area",
            "hru_area_perv",
            "hru_frac_perv",
            "ag_soil_moist",
            "ag_soil_rechr",
            "slow_stor",
            "soil_moist",
            "soil_rechr",
            "ag_frac",
            "hru_area",
            "hru_area_imperv",
            "dprst_frac",
            "hru_type",
        )
        self._pre_args = self._hold(
            "soil_lower",
            "ag_soil_moist_redistribution",
            "ag_soil_rechr_redistribution",
            "soil_rechr_redistribution",
            "soil_lower_redistribution",
            "slow_stor_redistribution",
            "soil_moist",
            "soil_rechr",
            "ag_soil_moist",
            "ag_soil_rechr",
            "slow_stor",
            "soil_rechr_prev",
            "soil_lower_prev",
            "slow_stor_prev",
            "ag_soil_moist_prev",
            "ag_soil_rechr_prev",
        )
        self._post_args = self._hold(
            "pref_flow_stor_change",
            "soil_lower_change",
            "soil_rechr_change",
            "slow_stor_change",
            "soil_lower_change_hru",
            "soil_rechr_change_hru",
            "ag_soil_moist_change",
            "ag_soil_rechr_change",
            "ag_soil_lower_change",
            "ag_soil_moist_change_hru",
            "ag_soil_rechr_change_hru",
            "ag_soil_lower_change_hru",
            "sroff_vol",
            "ssres_flow_vol",
            "perv_infil_hru",
            "ag_infil_hru",
            "pref_flow_stor",
            "pref_flow_stor_prev",
            "soil_lower",
            "soil_lower_prev",
            "soil_lower_redistribution",
            "soil_rechr",
            "soil_rechr_prev",
            "soil_rechr_redistribution",
            "slow_stor",
            "slow_stor_prev",
            "slow_stor_redistribution",
            "hru_frac_perv",
            "ag_soil_moist",
            "ag_soil_moist_prev",
            "ag_soil_moist_redistribution",
            "ag_soil_rechr",
            "ag_soil_rechr_prev",
            "ag_soil_rechr_redistribution",
            "ag_frac",
            "sroff",
            "ssres_flow",
            "hru_in_to_cf",
            "infil",
            "infil_ag",
        )

    def init_from_params(self) -> None:
        """The parameter half of upstream
        ``_initialize_soilzone_ag_data``: derived maxima, the
        edited-copy parameters, the parameter checks, the
        preferential-flow thresholds and flag."""
        refs = self._refs
        hru_type = refs["hru_type"]
        hru_area = refs["hru_area"]

        refs["hru_area_imperv"][:] = refs["hru_percent_imperv"] * hru_area

        soil_rechr_max = refs["soil_rechr_max"]
        soil_rechr_max[:] = (
            refs["soil_rechr_max_frac"] * refs["soil_moist_max"]
        )
        ag_soil_rechr_max = refs["ag_soil_rechr_max"]
        ag_soil_rechr_max[:] = (
            refs["ag_soil_rechr_max_frac"] * refs["ag_soil_moist_max"]
        )

        wh_inactive_or_lake = np.where(
            (hru_type == _INACTIVE) | (hru_type == _LAKE)
        )
        sat_threshold = refs["_sat_threshold"]
        sat_threshold[:] = refs["sat_threshold"]
        sat_threshold[wh_inactive_or_lake] = 0.0
        pref_flow_den = refs["_pref_flow_den"]
        pref_flow_den[:] = refs["pref_flow_den"]
        pref_flow_den[np.where(hru_type != 1)] = 0.0  # non-LAND

        pfif = refs["pref_flow_infil_frac"]
        if (pfif.min() < 0.0) or (pfif.max() > 1.0):
            raise ValueError(
                "Values of pref_flow_infil_frac outside of [0,1]."
            )

        # upstream order: ag_soil_lower_stor_max from the PRE-clamp
        # ag_soil_rechr_max
        refs["ag_soil_lower_stor_max"][:] = (
            refs["ag_soil_moist_max"] - ag_soil_rechr_max
        )

        # -- upstream "adjust_parameters" block, the parameter clamps
        # in its order (the state clamps it interleaves are in
        # init_state()). TRUE parameter edits cannot happen against
        # frozen parameters --
        if (refs["soil_moist_max"] < 1.0e-5).any():
            raise NotImplementedError(
                "PRMSSoilzoneAg: soil_moist_max < 1e-5 requires upstream's "
                "parameter adjustment, not ported (frozen parameters)"
            )
        if (refs["ag_soil_moist_max"] < 1.0e-5).any():
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
            soil_rechr_max > refs["soil_moist_max"],
            refs["soil_moist_max"],
            soil_rechr_max,
        )
        ag_soil_rechr_max[:] = np.where(
            ag_soil_rechr_max > refs["ag_soil_moist_max"],
            refs["ag_soil_moist_max"],
            ag_soil_rechr_max,
        )

        # -- preferential-flow thresholds by hru_type --
        pref_flow_thrsh = refs["pref_flow_thrsh"]
        pref_flow_max = refs["pref_flow_max"]
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

        # SCALAR flag (upstream; unlike plain soilzone's per-HRU
        # array). NOTE MPI: this is any() over the LOCAL rows;
        # upstream's is global-any. Identical whenever pref_flow_den
        # is uniformly zero/nonzero across ranks (fgr_ag_2yr: all
        # zero); a mixed domain would need a global reduction, but
        # init_from_params() is LOCAL by contract -- revisit if such a
        # domain arrives (the flag gates the topfr carve-out even
        # where den == 0, so local!=global WOULD change answers).
        self._pref_flow_flag = bool((pref_flow_den > 0.0).any())

        refs["soil_lower_max"][:] = refs["soil_moist_max"] - soil_rechr_max

        # -- the family kernel's argument tuples: rebuilt here because
        # they carry _pref_flow_flag (a tuple of references, no array
        # is allocated); the iterating form needs AET_external (an
        # ObsET-leaf field) --
        flags = (False, True) if "AET_external" in refs else (False,)
        self._family_args = {
            flag: self._build_family_args(flag) for flag in flags
        }

    def init_state(self, time: Time) -> None:
        """The state half of upstream ``_initialize_soilzone_ag_data``
        (the no-restart path), in its order: starting storages from
        the ``*_init_frac`` parameters and the maxima, upstream's
        value clamps, and what follows from them. The area block needs
        the input ``ag_frac`` and stays at time zero in calculate()
        (see _istep0_areas)."""
        refs = self._refs
        for name in self._ZERO_VARS:
            refs[name][:] = 0.0
        # not yet derived from ag_frac (see _istep0_areas)
        refs["hru_area_perv"][:] = np.nan

        hru_type = refs["hru_type"]
        wh_inactive_or_lake = np.where(
            (hru_type == _INACTIVE) | (hru_type == _LAKE)
        )
        sat_threshold = refs["_sat_threshold"]
        soil_rechr_max = refs["soil_rechr_max"]
        ag_soil_rechr_max = refs["ag_soil_rechr_max"]

        # -- initial states (no-restart path). Upstream seeds the two
        # recharge storages from their maxima BEFORE it clamps the
        # maxima; the frozen soil_rechr_max / ag_soil_rechr_max hold
        # the clamped values, so the pre-clamp products are recomputed
        # here, in place --
        soil_moist = refs["soil_moist"]
        soil_rechr = refs["soil_rechr"]
        ag_soil_moist = refs["ag_soil_moist"]
        ag_soil_rechr = refs["ag_soil_rechr"]
        soil_moist[:] = refs["soil_moist_init_frac"] * refs["soil_moist_max"]
        soil_rechr[:] = refs["soil_rechr_max_frac"] * refs["soil_moist_max"]
        soil_rechr *= refs["soil_rechr_init_frac"]
        ag_soil_moist[:] = (
            refs["ag_soil_moist_init_frac"] * refs["ag_soil_moist_max"]
        )
        ag_soil_rechr[:] = (
            refs["ag_soil_rechr_max_frac"] * refs["ag_soil_moist_max"]
        )
        ag_soil_rechr *= refs["ag_soil_rechr_init_frac"]

        ssres_stor = refs["ssres_stor"]
        ssres_stor[:] = refs["ssstor_init_frac"] * sat_threshold
        ssres_stor[wh_inactive_or_lake] = 0.0

        # no ag storage on LAKE|INACTIVE (ag_frac==0 asserted at istep0)
        ag_soil_moist[wh_inactive_or_lake] = 0.0
        ag_soil_rechr[wh_inactive_or_lake] = 0.0

        # -- the state clamps of upstream's "adjust_parameters" block,
        # in its order, against the clamped maxima --
        soil_rechr[:] = np.where(
            soil_rechr > soil_rechr_max, soil_rechr_max, soil_rechr
        )
        ag_soil_rechr[:] = np.where(
            ag_soil_rechr > ag_soil_rechr_max,
            ag_soil_rechr_max,
            ag_soil_rechr,
        )
        soil_moist[:] = np.where(
            soil_moist > refs["soil_moist_max"],
            refs["soil_moist_max"],
            soil_moist,
        )
        ag_soil_moist[:] = np.where(
            ag_soil_moist > refs["ag_soil_moist_max"],
            refs["ag_soil_moist_max"],
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

        # -- split initial ssres_stor into slow / preferential --
        slow_stor = refs["slow_stor"]
        pref_flow_stor = refs["pref_flow_stor"]
        pref_flow_thrsh = refs["pref_flow_thrsh"]
        wh_land_or_swale = np.where((hru_type == 1) | (hru_type == _SWALE))
        slow_stor[wh_land_or_swale] = np.minimum(
            ssres_stor[wh_land_or_swale], pref_flow_thrsh[wh_land_or_swale]
        )
        pref_flow_stor[wh_land_or_swale] = (
            ssres_stor[wh_land_or_swale] - slow_stor[wh_land_or_swale]
        )

        refs["soil_lower"][:] = soil_moist - soil_rechr
        soil_lower_max = refs["soil_lower_max"]
        wh_soil_lower_stor = np.where(soil_lower_max > 0.0)
        refs["soil_lower_ratio"][wh_soil_lower_stor] = (
            refs["soil_lower"][wh_soil_lower_stor]
            / soil_lower_max[wh_soil_lower_stor]
        )

        # -- *_prev seeds --
        refs["soil_rechr_prev"][:] = soil_rechr
        refs["soil_lower_prev"][:] = refs["soil_lower"]
        refs["ag_soil_moist_prev"][:] = ag_soil_moist
        refs["ag_soil_rechr_prev"][:] = ag_soil_rechr
        refs["pref_flow_stor_prev"][:] = pref_flow_stor
        refs["slow_stor_prev"][:] = slow_stor

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        refs = self._refs
        refs["pref_flow_stor_prev"][:] = refs["pref_flow_stor"]
        refs["soil_rechr_prev"][:] = refs["soil_rechr"]
        refs["soil_lower_prev"][:] = refs["soil_lower"]
        refs["slow_stor_prev"][:] = refs["slow_stor"]
        refs["ag_soil_moist_prev"][:] = refs["ag_soil_moist"]
        refs["ag_soil_rechr_prev"][:] = refs["ag_soil_rechr"]

    # NOT cache=True: the kernels call prms_soilzone.py's jitted helpers;
    # a cached caller never notices an edited callee (DEVELOPER.md
    # "Numba cache").
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
                        ag_soil_moist[hh] = _ZERO_INCHES
                        ag_soil_rechr[hh] = _ZERO_INCHES

            # pervious scaling when pervious area changed
            # (isclose b = new_perv)
            perv_changed = not (
                abs(old_perv - new_perv)
                <= _ISCLOSE_ATOL_ACRES + 1.0e-5 * abs(new_perv)
            )
            if perv_changed and (new_perv > 0.0):
                scale_perv = old_perv / new_perv
                soil_moist[hh] = soil_moist[hh] * scale_perv
                soil_rechr[hh] = soil_rechr[hh] * scale_perv
            if (old_perv > 0.0) and (abs(new_perv) <= _ISCLOSE_ATOL_ACRES):
                soil_moist[hh] = _ZERO_INCHES
                soil_rechr[hh] = _ZERO_INCHES

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
    @numba.njit(parallel=NUMBA_PARALLEL)
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

        soil_to_gw[:] = _ZERO_INCHES
        soil_to_ssr[:] = _ZERO_INCHES
        perv_soil_to_gw[:] = _ZERO_INCHES
        perv_soil_to_gvr[:] = _ZERO_INCHES
        ag_soil_to_gw[:] = _ZERO_INCHES
        ag_soil_to_gvr[:] = _ZERO_INCHES
        ssr_to_gw[:] = _ZERO_INCHES
        slow_flow[:] = _ZERO_INCHES
        ssres_flow[:] = _ZERO_INCHES
        potet_rechr[:] = _ZERO_INCHES
        potet_lower[:] = _ZERO_INCHES
        ag_potet_rechr[:] = _ZERO_INCHES
        ag_potet_lower[:] = _ZERO_INCHES
        ag_actet[:] = _ZERO_INCHES
        hru_ag_actet[:] = _ZERO_INCHES
        ag_soil_saturated[:] = 0.0
        ag_hortonian[:] = _ZERO_INCHES
        unused_ag_et[:] = _ZERO_INCHES
        ag_cap_infil_tot[:] = _ZERO_INCHES
        ag_soil_lower[:] = _ZERO_INCHES
        swale_actet[:] = _ZERO_INCHES
        soil_saturated[:] = 0.0
        dunnian_flow[:] = _ZERO_INCHES
        pref_flow_infil[:] = _ZERO_INCHES
        pref_flow_in[:] = _ZERO_INCHES
        pref_flow[:] = _ZERO_INCHES

        for hh in numba.prange(nhru):
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
                avail_potet = _ZERO_INCHES
                hruactet = potet[hh]

            dunnianflw = _ZERO_INCHES
            dunnianflw_pfr = _ZERO_INCHES

            capwater_maxin = infil[hh]
            ag_water_maxin = infil_ag[hh]
            if iter_aet_flag:
                ag_water_maxin = ag_water_maxin + ag_irrigation_add[hh]

            prefflow = _ZERO_INCHES
            if pref_flow_flag:
                if pref_flow_infil_frac[hh] > 0.0:
                    ag_pref_flow_maxin = _ZERO_INCHES
                    cap_pref_flow_maxin = _ZERO_INCHES

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
                        _ZERO_INCHES, pref_flow_stor[hh] - pref_flow_max[hh]
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
            perv_soil_to_gvr[hh] = _ZERO_INCHES
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
            topfr = _ZERO_INCHES
            if hru_type[hh] != _SWALE:
                if pref_flow_flag:
                    topfr = max(_ZERO_INCHES, availh2o - pref_flow_thrsh[hh])
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
            dunnianflw_gvr = _ZERO_INCHES
            if pref_flow_flag:
                if pref_flow_max[hh] > 0.0:
                    availh2o = pref_flow_stor[hh] + topfr
                    dunnianflw_gvr = max(
                        _ZERO_INCHES, availh2o - pref_flow_max[hh]
                    )
                    if dunnianflw_gvr > 0.0:
                        topfr = topfr - dunnianflw_gvr
                        if topfr < 0.0:
                            topfr = _ZERO_INCHES
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
            pervactet = _ZERO_INCHES
            agactet = _ZERO_INCHES
            ag_hruactet = _ZERO_INCHES
            unsatisfied_ag_et = _ZERO_INCHES

            if ag_on_flag:
                if iter_aet_flag:
                    ag_AETtarget = aet_external[hh]
                else:
                    ag_AETtarget = potet[hh]

                ag_avail_targetAET = ag_AETtarget - hru_intcpevap[hh]
                if ag_avail_targetAET < 0.0:
                    ag_avail_targetAET = _ZERO_INCHES

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
                avail_potet = _ZERO_INCHES

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
                swale_actet[hh] = _ZERO_INCHES
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

    def _hold(self, *names: str) -> tuple:
        """The named field ndarrays as a tuple, for a kernel call made
        every step: ``self._kernel(*self._kernel_args)``."""
        return tuple(self._refs[nn] for nn in names)

    def _calculate_family(self, iter_aet_flag: bool) -> None:
        """One physics pass over the shared family kernel (the ObsET
        subclass calls this inside its iteration loop)."""
        self._calculate(*self._family_args[iter_aet_flag])

    def _build_family_args(self, iter_aet_flag: bool) -> tuple:
        """The family kernel's ~80 arguments, assembled ONCE per flag
        value at the end of init_from_params() (see the _family_args
        note there)."""
        refs = self._refs
        if iter_aet_flag:
            aet_external = refs["AET_external"]
        else:
            # never read under iter_aet_flag=False; benign aliasing
            aet_external = refs["ag_irrigation_add"]
        return (
            iter_aet_flag,
            self._pref_flow_flag,
            refs["soil_moist"],
            refs["soil_rechr"],
            refs["ag_soil_moist"],
            refs["ag_soil_rechr"],
            refs["pref_flow_stor"],
            refs["slow_stor"],
            refs["ssres_stor"],
            refs["soil_to_gw"],
            refs["soil_to_ssr"],
            refs["perv_soil_to_gw"],
            refs["perv_soil_to_gvr"],
            refs["ag_soil_to_gw"],
            refs["ag_soil_to_gvr"],
            refs["ssr_to_gw"],
            refs["slow_flow"],
            refs["ssres_flow"],
            refs["potet_rechr"],
            refs["potet_lower"],
            refs["ag_potet_rechr"],
            refs["ag_potet_lower"],
            refs["cap_waterin"],
            refs["cap_infil_tot"],
            refs["ag_cap_infil_tot"],
            refs["pref_flow_in"],
            refs["pref_flow_infil"],
            refs["pref_flow"],
            refs["perv_actet"],
            refs["perv_actet_hru"],
            refs["ag_actet"],
            refs["hru_ag_actet"],
            refs["hru_actet"],
            refs["soil_lower"],
            refs["ag_soil_lower"],
            refs["dunnian_flow"],
            refs["soil_moist_tot"],
            refs["ssres_in"],
            refs["recharge"],
            refs["unused_potet"],
            refs["unused_ag_et"],
            refs["ag_hortonian"],
            refs["ag_soil_saturated"],
            refs["swale_actet"],
            refs["soil_saturated"],
            refs["soil_lower_ratio"],
            refs["sroff"],
            refs["dprst_evap_hru"],
            refs["dprst_seep_hru"],
            refs["hru_impervevap"],
            refs["hru_intcpevap"],
            refs["infil"],
            refs["infil_ag"],
            refs["potet"],
            refs["transp_on"],
            refs["snow_evap"],
            refs["snowcov_area"],
            aet_external,
            refs["ag_irrigation_add"],
            refs["hru_type"],
            refs["cov_type"],
            refs["ag_cov_type"],
            refs["fastcoef_lin"],
            refs["fastcoef_sq"],
            refs["hru_frac_perv"],
            refs["ag_frac"],
            refs["ag_area"],
            refs["hru_area_perv"],
            refs["_pref_flow_den"],
            refs["pref_flow_infil_frac"],
            refs["_sat_threshold"],
            refs["slowcoef_lin"],
            refs["slowcoef_sq"],
            refs["soil_moist_max"],
            refs["soil_rechr_max"],
            refs["soil_type"],
            refs["ag_soil_type"],
            refs["ag_soil_moist_max"],
            refs["ag_soil_rechr_max"],
            refs["soil2gw_max"],
            refs["ssr2gw_exp"],
            refs["ssr2gw_rate"],
            refs["pref_flow_max"],
            refs["pref_flow_thrsh"],
            refs["soil_lower_max"],
        )

    def _pre_kernel(self) -> None:
        """istep0 areas (once) + per-step area update + pre lines."""
        self._update_areas(*self._update_areas_args)
        self._pre(*self._pre_args)

    def _istep0_areas(self) -> None:
        """upstream _initialize_soilzone_ag_data area block: runs at
        time zero in calculate() (ag_frac now fed); once-only numpy
        staging.

        Skipped when the areas were SUPPLIED as initial state.
        init_state() leaves hru_area_perv at nan ("not yet derived
        from ag_frac"); supplied state replaces the nan. The supplied
        areas are the previous step's, which _update_areas must read
        to see a change of ag_frac on the first step and move the
        storages accordingly; this block would overwrite them with the
        current step's."""
        refs = self._refs
        if not np.isnan(refs["hru_area_perv"]).all():
            return
        hru_type = refs["hru_type"]
        hru_area = refs["hru_area"]

        wh_no_ag = np.where((hru_type == _LAKE) | (hru_type == _INACTIVE))
        # GSFLOW zeroes ag_frac on this mask; upstream (and this port)
        # will NOT edit inputs -- assert instead
        assert (refs["ag_frac"][wh_no_ag] == 0.0).all()

        np.multiply(refs["ag_frac"], hru_area, out=refs["ag_area"])
        hru_area_perv = refs["hru_area_perv"]
        hru_area_perv[:] = hru_area - refs["hru_area_imperv"]
        wh_active = np.where(hru_type != _INACTIVE)
        dprst_area_max = refs["dprst_frac"] * hru_area
        hru_area_perv[wh_active] = (
            hru_area_perv[wh_active]
            - dprst_area_max[wh_active]
            - refs["ag_area"][wh_active]
        )
        refs["hru_frac_perv"][:] = 0.0
        refs["hru_frac_perv"][wh_active] = (
            hru_area_perv[wh_active] / hru_area[wh_active]
        )
        refs["soil_moist_tot"][:] = (
            refs["ssres_stor"]
            + refs["soil_moist"] * refs["hru_frac_perv"]
            + refs["ag_soil_moist"] * refs["ag_frac"]
        )

    def _post_kernel(self) -> None:
        self._post(*self._post_args)

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
        units="inches",
        description="Convergence tolerance on unsatisfied ag AET [inches]",
    )
    ag_soilwater_deficit_min = _meta(
        "parameter",
        "Minimum ag soil-water deficit to allow irrigation [-]",
        units="fraction",
    )

    # -- inputs (ADDED) --
    aet_observed = _meta(
        "input", "Observed actual ET [inches] (-1.0 = missing)", units="inches"
    )

    # -- variables (ADDED; obs-ET machinery + diagnostics) --
    AET_external = _meta(
        "variable",
        "Validated observed-AET target [inches] (upstream name)",
        units="inches",
    )
    ag_soilwater_deficit = _meta(
        "variable",
        "Ag soil-water deficit fraction [-] (iteration diag)",
        units="fraction",
    )
    ag_irrigation_add_vol = _meta(
        "variable",
        "Irrigation volume added [acre-inches]",
        units="acre-inches",
    )
    ag_aet_external_vol = _meta(
        "variable", "Ag actual-ET volume [acre-inches]", units="acre-inches"
    )
    ag_irrigation_hru_source = _term(
        "Irrigation water added [inches over the HRU]"
    )
    iter_count = _meta(
        "variable", "Iterations performed this timestep", np.int64, units=None
    )
    iter_end_status = _meta(
        "variable",
        "Convergence code (-1 no-ag, 0 converged, 1 deficit-limited, "
        "2 max-iters, 3 no-transp, -9 never)",
        np.int64,
        units=None,
    )

    # -- budgets (REDECLARED: the iteration's irrigation water enters
    # the ag capillary reservoir, `ag_water_maxin += ag_irrigation_add`;
    # its whole-HRU-basis copy is the input term, as upstream) --
    budgets = {
        "mass": {
            **PRMSSoilzoneAg._MASS_BUDGET,
            "inputs": (
                *PRMSSoilzoneAg._MASS_BUDGET["inputs"],
                "ag_irrigation_hru_source",
            ),
        },
    }

    _OBS_ZERO_VARS = (
        "AET_external",
        "ag_soilwater_deficit",
        "ag_irrigation_add_vol",
        "ag_aet_external_vol",
        "ag_irrigation_hru_source",
        "iter_count",
        "iter_end_status",
    )

    def allocate(self) -> None:
        super().allocate()
        refs = self._refs
        # It0 iteration scratch (persistent; prime directive: memory),
        # one buffer per _IT0_NAMES entry, same order. No starting
        # value matters: calculate() overwrites every buffer from the
        # live state before reading it.
        nhru = refs["hru_area"].shape[0]
        self._it0 = tuple(np.zeros(nhru, dtype=np.float64) for _ in _IT0_NAMES)
        self._it0_state = self._hold(*_IT0_NAMES)
        self._prep_aet_args = self._hold(
            "AET_external",
            "ag_irrigation_add",
            "ag_irrigation_add_vol",
            "ag_aet_external_vol",
            "aet_observed",
            "ag_frac",
            "transp_on",
        )

    def init_from_params(self) -> None:
        super().init_from_params()
        refs = self._refs
        # the iteration caps, read once per parameter set
        self._max_iter = int(refs["max_soilzone_ag_iter"][0])
        self._converge = float(refs["soilzone_aet_converge"][0])

    def init_state(self, time: Time) -> None:
        super().init_state(time)
        refs = self._refs
        for name in self._OBS_ZERO_VARS:
            refs[name][:] = 0

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
            ag_irrigation_add[hh] = _ZERO_INCHES
            ag_irrigation_add_vol[hh] = _ZERO_ACRE_INCHES
            ag_aet_external_vol[hh] = _ZERO_ACRE_INCHES
            AET_external[hh] = aet_observed[hh]
            if (
                (AET_external[hh] < 0.0)
                and (AET_external[hh] != _AET_MISSING_INCHES)
                and (ag_frac[hh] > 0.0)
            ):
                AET_external[hh] = _ZERO_INCHES
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
        if time.current_index == 0:
            self._istep0_areas()
        self._pre_kernel()
        refs = self._refs

        # store initial values for iteration (Fortran It0 variables)
        for live, buf in zip(self._it0_state, self._it0, strict=True):
            buf[:] = live

        n_transp = self._prep_aet(*self._prep_aet_args)

        max_iter = self._max_iter
        converge = self._converge

        keep_iterating = True
        soil_iter = 1
        while keep_iterating:
            if soil_iter > 1:
                for name, live, buf in zip(
                    _IT0_NAMES, self._it0_state, self._it0, strict=True
                ):
                    if name == "pref_flow_stor" and not (self._pref_flow_flag):
                        continue
                    live[:] = buf

            self._calculate_family(iter_aet_flag=True)

            (
                add_estimated_irrigation,
                _num_hrus_ag_iter,
                _unsatisfied_big,
            ) = self._adjust_irrigation(
                refs["ag_soilwater_deficit"],
                refs["ag_irrigation_add"],
                refs["unused_ag_et"],
                refs["ag_soil_moist"],
                refs["ag_soil_moist_max"],
                refs["transp_on"],
                refs["ag_area"],
                soil_iter,
                max_iter,
                converge,
                refs["ag_soilwater_deficit_min"],
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
            refs["iter_count"],
            refs["iter_end_status"],
            refs["ag_irrigation_add_vol"],
            refs["ag_aet_external_vol"],
            refs["ag_irrigation_hru_source"],
            refs["ag_irrigation_add"],
            refs["ag_actet"],
            refs["ag_area"],
            refs["ag_frac"],
            refs["transp_on"],
            refs["unused_ag_et"],
            refs["ag_soilwater_deficit"],
            refs["ag_soilwater_deficit_min"],
            soil_iter,
            max_iter,
            converge,
        )

        self._post_kernel()
