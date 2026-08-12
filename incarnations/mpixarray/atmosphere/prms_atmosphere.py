"""
atmosphere/prms_atmosphere.py
=============================
PRMSAtmosphere: the PRMS atmospheric boundary layer, ported from
pywatershed (pywatershed/atmosphere/prms_atmosphere.py; PRMS 5.2.1:
climate_hru temperature/precip adjustment, ddsolrad degree-day
shortwave, Jensen-Haise (1963) potential ET, transp_tindex
transpiration).

Eighth REAL port (July 2026) -- the LAST forcing producer: with this
live, the whole chain runs from the CBH inputs (prcp/tmax/tmin) + the
solar tables (atmosphere/prms_solar_geometry.py factory) + parameters.

THE structural departure: upstream computes ALL variables for ALL
times at initialization ("effectively a complete preprocessing of the
input CBH files" -- its own words, which also note that full-time
initialization "may not be tractable for large domains"). This port
IS the per-step streaming version upstream anticipates: every
calculation except transpiration is per-day elementwise and folds
straight into the in-place kernel; the transp_tindex season logic is
genuinely sequential and becomes per-step STATE (transp_on, tmax_sum,
and the private check flag), with upstream's "time zero calculations"
run inside the kernel on the first step (istep0). Per-element
operation order matches upstream's vectorized expressions, so results
are unchanged.

Declarations/decisions:

- pptmix is a VARIABLE here; downstream PRMSCanopy declares it
  mutable_input and zeroes it in place AFTER this process each step
  (schedule order) -- the same shared field.
- The solar tables are (ndoy, space) parameters indexed by
  current_doy (same seam as PRMSSnow's soltab).
- monthly parameters are (nmonth, space), indexed by current_month-1
  in-kernel; the "1-month" (annual) cbh-adjustment parameter variant
  upstream supports is NOT ported (nhm domains are monthly) -- the
  declared dims reject it loudly.
- temp_units is a ('scalar',) parameter consumed in initialize() (the
  transp_tmax F conversion -- upstream's "candidate for worst code
  lines").
- hru_cossl and transp_tmax_f are parameter_internal (space,).
- Southern-hemisphere domains raise NotImplementedError in
  initialize() (upstream raises inside its is_summer logic).

Family structure (ADDITIVE; PORTS.md "How variants are done here"):
upstream's PRMSAtmosphereTranspFrost REPLACES the tindex parameters
(not a superset), so the shared physics lives in
``PRMSAtmosphereBase`` (everything but transpiration; declares
``transp_on`` since both leaves produce it) and each leaf ADDS its
transpiration: ``PRMSAtmosphere`` = transp_tindex state machine;
``PRMSAtmosphereTranspFrost`` = the stateless
spring_frost <= jsol <= fall_frost window (jsol = solar day of year,
Time.jsol); ``PRMSAtmosphereTranspFrostDyn`` = the same window with
the frost dates re-declared as time-varying inputs (PRMS dynamic
parameters). The transpiration section was the final, self-contained
piece of the kernel loop; splitting it into a second per-leaf kernel
is arithmetic-identical.

Deliberately NOT ported: Budget; adapters; restart (upstream's own
comment: "The restart capability in PRMS does not work"); netcdf
plumbing; unused declared params (doy, hru_area, hru_aspect);
verbose.

Parameter provenance: hru_slope/hru_lat are DIS_HRU; the 22 process
parameters live in parameters_PRMSAtmosphere.nc; tmax_allsnow /
tmax_allrain_offset are shared with PRMSSnow's file (identical NHM
values).
"""

import numba
import numpy as np

from atmosphere.solar_constants import solf as _SOLF
from globals import Time
from process import DataArrayMeta, Process

# pywatershed constants
_NEARZERO = 1.0e-6  # nearzero
_INCH2CM = 2.54


class PRMSAtmosphereBase(Process):
    """PRMS atmospheric boundary layer WITHOUT transpiration: CBH
    temperature/precipitation adjustment and rain/snow partitioning,
    degree-day shortwave, Jensen-Haise potential ET -- per timestep.

    Abstract family core: the leaves add their transpiration model
    (PRMSAtmosphere = transp_tindex; PRMSAtmosphereTranspFrost =
    frost window) and own calculate(). ``transp_on`` is declared HERE
    (every leaf produces it).

    Temperatures in degF (tmaxf/tminf) and degC (t*c); precip in
    inches; radiation in cal/cm^2 (Langleys); potet in inches.
    """

    # ------------------------------------------------------------------
    # Field declarations (names verbatim from pywatershed)
    # ------------------------------------------------------------------

    # -- dis_hru variables (grid-owned; dis-first sourcing) --
    hru_slope = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="HRU slope [rise/run]",
    )
    hru_lat = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="HRU latitude [degrees N]",
    )

    # -- process parameters (monthly x HRU) --
    tmax_cbh_adj = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Monthly adjustment to CBH maximum temperature [degF]",
    )
    tmin_cbh_adj = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Monthly adjustment to CBH minimum temperature [degF]",
    )
    tmax_allsnow = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Maximum temperature for all-snow precipitation [degF]",
    )
    tmax_allrain_offset = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Offset above tmax_allsnow for all-rain [degF]",
    )
    snow_cbh_adj = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Monthly CBH precipitation adjustment for snow [-]",
    )
    rain_cbh_adj = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Monthly CBH precipitation adjustment for rain [-]",
    )
    adjmix_rain = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Monthly rain-fraction adjustment for mixed events [-]",
    )
    dday_slope = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Degree-day slope coefficient [dday/degF]",
    )
    dday_intcp = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Degree-day intercept coefficient [dday]",
    )
    radmax = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Maximum fraction of potential solar radiation [-]",
    )
    ppt_rad_adj = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Precip threshold above which radiation is adjusted "
        "[inches]",
    )
    radadj_intcp = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Radiation-adjustment intercept on precip days [-]",
    )
    radadj_slope = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Radiation-adjustment slope on precip days [1/degF]",
    )
    tmax_index = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Index temperature for radiation adjustment [degF]",
    )
    jh_coef = DataArrayMeta(
        kind="parameter",
        dims=("nmonth", "space"),
        dtype=np.float64,
        description="Monthly Jensen-Haise air-temperature coefficient "
        "[1/degF]",
    )

    # -- process parameters (per HRU) --
    jh_coef_hru = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Jensen-Haise per-HRU air-temperature coefficient [degF]",
    )
    radj_sppt = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Radiation adjustment for summer precip days [-]",
    )
    radj_wppt = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Radiation adjustment for winter precip days [-]",
    )
    # -- static solar tables (prms_solar_geometry factory product) --
    soltab_potsw = DataArrayMeta(
        kind="parameter",
        dims=("ndoy", "space"),
        dtype=np.float64,
        description="Potential shortwave on the sloped surface per Julian "
        "day [cal/cm^2] -- static table indexed by current_doy",
        derivation="compute_soltabs(hru_slope, hru_aspect, hru_lat)",
    )
    soltab_horad_potsw = DataArrayMeta(
        kind="parameter",
        dims=("ndoy", "space"),
        dtype=np.float64,
        description="Potential shortwave on a horizontal plane per Julian "
        "day [cal/cm^2] -- static table indexed by current_doy",
        derivation="compute_soltabs(hru_slope, hru_aspect, hru_lat)",
    )

    # -- derived parameters (initialize(); frozen after) --
    hru_cossl = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="cos(arctan(hru_slope))",
    )

    # -- inputs (the CBH files) --
    prcp = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH precipitation [inches]",
    )
    tmax = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH maximum air temperature [degF]",
    )
    tmin = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH minimum air temperature [degF]",
    )

    # -- variables --
    tmaxf = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Adjusted maximum air temperature [degF]",
    )
    tminf = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Adjusted minimum air temperature [degF]",
    )
    tmaxc = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Adjusted maximum air temperature [degC]",
    )
    tminc = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Adjusted minimum air temperature [degC]",
    )
    tavgc = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Adjusted average air temperature [degC]",
    )
    prmx = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of rain in a mixed event [-]",
    )
    hru_ppt = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Adjusted precipitation on the HRU [inches]",
    )
    hru_rain = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Rain on the HRU [inches]",
    )
    hru_snow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Snow on the HRU [inches]",
    )
    pptmix = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Rain/snow mix flag (0/1) -- PRMSCanopy edits the "
        "shared field in place downstream",
    )
    swrad = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Shortwave radiation on the HRU [cal/cm^2]",
    )
    orad_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Shortwave on a horizontal plane at the HRU [cal/cm^2]",
    )
    ccov_hru = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Cloud cover fraction on the HRU [-] (from swrad "
        "vs potential; relocated here from upstream stream_temp's "
        "in-aggregation computation)",
    )
    potet = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Potential evapotranspiration (Jensen-Haise) [inches]",
    )
    transp_on = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Transpiration occurring (0/1 flag; leaf-computed)",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        obj = self._obj
        for name in (
            "tmaxf",
            "tminf",
            "tmaxc",
            "tminc",
            "tavgc",
            "prmx",
            "hru_ppt",
            "hru_rain",
            "hru_snow",
            "pptmix",
            "swrad",
            "orad_hru",
            "ccov_hru",
            "potet",
            "transp_on",
        ):
            obj[name].values[:] = 0.0

        # upstream's is_summer logic assumes the northern hemisphere
        if not (obj["hru_lat"].values > 0.0).any():
            raise NotImplementedError(
                "PRMSAtmosphere: southern-hemisphere domains are not "
                "implemented (upstream raises the same)"
            )

        obj["hru_cossl"].values[:] = np.cos(np.arctan(obj["hru_slope"].values))

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        pass  # transp_on/tmax_sum/transp_check persist; no *_prev vars

    @staticmethod
    @numba.njit
    def _calculate(
        # outputs + state (written in place)
        tmaxf: np.ndarray,
        tminf: np.ndarray,
        tmaxc: np.ndarray,
        tminc: np.ndarray,
        tavgc: np.ndarray,
        prmx: np.ndarray,
        hru_ppt: np.ndarray,
        hru_rain: np.ndarray,
        hru_snow: np.ndarray,
        pptmix: np.ndarray,
        swrad: np.ndarray,
        orad_hru: np.ndarray,
        potet: np.ndarray,
        # inputs
        prcp: np.ndarray,
        tmax: np.ndarray,
        tmin: np.ndarray,
        # parameters + derived
        tmax_cbh_adj: np.ndarray,
        tmin_cbh_adj: np.ndarray,
        tmax_allsnow: np.ndarray,
        tmax_allrain_offset: np.ndarray,
        snow_cbh_adj: np.ndarray,
        rain_cbh_adj: np.ndarray,
        adjmix_rain: np.ndarray,
        dday_slope: np.ndarray,
        dday_intcp: np.ndarray,
        radmax: np.ndarray,
        ppt_rad_adj: np.ndarray,
        radadj_intcp: np.ndarray,
        radadj_slope: np.ndarray,
        tmax_index: np.ndarray,
        jh_coef: np.ndarray,
        jh_coef_hru: np.ndarray,
        radj_sppt: np.ndarray,
        radj_wppt: np.ndarray,
        soltab_potsw: np.ndarray,
        soltab_horad_potsw: np.ndarray,
        hru_cossl: np.ndarray,
        # time context
        current_month: np.int64,
        current_doy: np.int64,
    ) -> None:
        nhru = tmaxf.shape[0]
        mm = current_month - 1
        # is_summer (northern hemisphere; sm_prms_time.f90)
        is_summer = (current_doy >= 79) and (current_doy <= 265)

        for ii in range(nhru):
            # ---- adjust_temperature ----
            tmaxf[ii] = tmax[ii] + tmax_cbh_adj[mm, ii]
            tminf[ii] = tmin[ii] + tmin_cbh_adj[mm, ii]
            tminc[ii] = (tminf[ii] - 32.0) * (5 / 9)
            tmaxc[ii] = (tmaxf[ii] - 32.0) * (5 / 9)
            tavgc[ii] = (tmaxc[ii] + tminc[ii]) / 2.0

            # ---- adjust_precip (order MATTERS; PRMS logic is
            # if(all_snow) elif(all_rain) else(mixed), masked in
            # reverse upstream) ----
            tmax_allrain = tmax_allsnow[mm, ii] + tmax_allrain_offset[mm, ii]
            tdiff = tmaxf[ii] - tminf[ii]
            if tdiff < _NEARZERO:
                tdiff = 1.0e-4
            prmx[ii] = (
                (tmaxf[ii] - tmax_allsnow[mm, ii]) / tdiff
            ) * adjmix_rain[mm, ii]
            if prmx[ii] < 0.0:
                prmx[ii] = 0.0
            if prmx[ii] > 1.0:
                prmx[ii] = 1.0
            if (tminf[ii] > tmax_allsnow[mm, ii]) or (
                tmaxf[ii] >= tmax_allrain
            ):
                prmx[ii] = 1.0
            if tmaxf[ii] <= tmax_allsnow[mm, ii]:
                prmx[ii] = 0.0
            # climate_hru's condition for calling climateflow
            if prcp[ii] <= 0.0:
                prmx[ii] = 0.0

            # amounts from prmx (mixed default, all-snow/all-rain
            # overrides -- upstream mask order collapses to this)
            if prmx[ii] <= 0.0:
                hru_ppt[ii] = prcp[ii] * snow_cbh_adj[mm, ii]
                hru_snow[ii] = hru_ppt[ii]
                hru_rain[ii] = 0.0
            elif prmx[ii] >= 1.0:
                hru_ppt[ii] = prcp[ii] * rain_cbh_adj[mm, ii]
                hru_rain[ii] = hru_ppt[ii]
                hru_snow[ii] = 0.0
            else:
                hru_ppt[ii] = prcp[ii] * snow_cbh_adj[mm, ii]
                hru_rain[ii] = prmx[ii] * hru_ppt[ii]
                hru_snow[ii] = hru_ppt[ii] - hru_rain[ii]

            if (
                (hru_ppt[ii] > 0.0)
                and (tmaxf[ii] > tmax_allsnow[mm, ii])
                and (
                    (tminf[ii] <= tmax_allsnow[mm, ii])
                    and (tmaxf[ii] < tmax_allrain)
                )
                and (prmx[ii] < 1.0)
            ):
                pptmix[ii] = 1.0
            else:
                pptmix[ii] = 0.0

            # ---- degree-day shortwave (ddsolrad) ----
            dday = dday_slope[mm, ii] * tmaxf[ii] + dday_intcp[mm, ii] + 1.0
            if dday < 1.0:
                dday = 1.0

            if dday < 26.0:
                kp = int(dday)
                radadj = _SOLF[kp - 1] + (
                    (_SOLF[kp] - _SOLF[kp - 1]) * (dday - kp)
                )
                if radadj > radmax[mm, ii]:
                    radadj = radmax[mm, ii]
            else:
                radadj = radmax[mm, ii]

            pptadj = 1.0
            if hru_ppt[ii] > ppt_rad_adj[mm, ii]:
                pptadj = radadj_intcp[mm, ii] + radadj_slope[mm, ii] * (
                    tmaxf[ii] - tmax_index[mm, ii]
                )
                if pptadj > 1.0:
                    pptadj = 1.0
                if tmaxf[ii] < tmax_index[mm, ii]:
                    pptadj = radj_sppt[ii]
                    if tmaxf[ii] < tmax_allrain:
                        pptadj = radj_wppt[ii]
                    if (tmaxf[ii] >= tmax_allrain) and (not is_summer):
                        pptadj = radj_wppt[ii]

            radadj = radadj * pptadj
            if radadj < 0.2:
                radadj = 0.2
            swrad[ii] = (
                soltab_potsw[current_doy - 1, ii] * radadj / hru_cossl[ii]
            )
            orad_hru[ii] = radadj * soltab_horad_potsw[current_doy - 1, ii]

            # ---- potential ET (Jensen-Haise; mixes degC and degF,
            # as upstream notes) ----
            tavgf = (tavgc[ii] * 9 / 5) + 32
            elh = (597.3 - (0.5653 * tavgc[ii])) * _INCH2CM
            potet[ii] = (
                jh_coef[mm, ii] * (tavgf - jh_coef_hru[ii]) * swrad[ii] / elh
            )
            if potet[ii] < 0.0:
                potet[ii] = 0.0

    def _calculate_base(self, time: Time) -> None:
        """One pass of the shared (transpiration-free) kernel; each
        leaf's calculate() runs this then its transpiration kernel."""
        obj = self._obj
        self._calculate(
            obj["tmaxf"].values,
            obj["tminf"].values,
            obj["tmaxc"].values,
            obj["tminc"].values,
            obj["tavgc"].values,
            obj["prmx"].values,
            obj["hru_ppt"].values,
            obj["hru_rain"].values,
            obj["hru_snow"].values,
            obj["pptmix"].values,
            obj["swrad"].values,
            obj["orad_hru"].values,
            obj["potet"].values,
            obj["prcp"].values,
            obj["tmax"].values,
            obj["tmin"].values,
            obj["tmax_cbh_adj"].values,
            obj["tmin_cbh_adj"].values,
            obj["tmax_allsnow"].values,
            obj["tmax_allrain_offset"].values,
            obj["snow_cbh_adj"].values,
            obj["rain_cbh_adj"].values,
            obj["adjmix_rain"].values,
            obj["dday_slope"].values,
            obj["dday_intcp"].values,
            obj["radmax"].values,
            obj["ppt_rad_adj"].values,
            obj["radadj_intcp"].values,
            obj["radadj_slope"].values,
            obj["tmax_index"].values,
            obj["jh_coef"].values,
            obj["jh_coef_hru"].values,
            obj["radj_sppt"].values,
            obj["radj_wppt"].values,
            obj["soltab_potsw"].values,
            obj["soltab_horad_potsw"].values,
            obj["hru_cossl"].values,
            np.int64(time.month),
            np.int64(time.doy),
        )
        self._compute_ccov_hru(
            obj["ccov_hru"].values,
            obj["swrad"].values,
            obj["soltab_potsw"].values,
            obj["hru_cossl"].values,
            np.int64(time.doy),
        )

    @staticmethod
    @numba.njit
    def _compute_ccov_hru(
        # output (written in place)
        ccov_hru: np.ndarray,
        # input + parameters
        swrad: np.ndarray,
        soltab_potsw: np.ndarray,
        hru_cossl: np.ndarray,
        current_doy: np.int64,
    ) -> None:
        """Cloud cover fraction from swrad vs potential shortwave --
        VERBATIM the per-HRU block of upstream stream_temp's
        aggregation (PRMS 5.2.1.1 stream_temp.f90 lines 760-778;
        pywatershed _compute_segment_aggregates_numba). Relocated to
        atmosphere per the chain-stage decision that Maps never
        ORIGINATE variables: ccov_hru is an HRU met quantity, computed
        where swrad lives; the hru->segment aggregation Maps then
        carry it to seg_ccov like any other met variable."""
        for ii in range(ccov_hru.shape[0]):
            potsw = soltab_potsw[current_doy - 1, ii]
            if potsw <= 10.0:
                ccov = 1.0 - (swrad[ii] / 10.0 * hru_cossl[ii])
            else:
                ccov = 1.0 - (swrad[ii] / potsw * hru_cossl[ii])
            if ccov < _NEARZERO:
                ccov = 0.0
            elif ccov > 1.0:
                ccov = 1.0
            ccov_hru[ii] = ccov


class PRMSAtmosphere(PRMSAtmosphereBase):
    """PRMSAtmosphereBase PLUS the transp_tindex transpiration model:
    the temperature-index season state machine (sum tmaxf above
    freezing from transp_beg until transp_tmax is exceeded; off at
    transp_end), with upstream's "time zero calculations" on the
    first step."""

    # -- process parameters (ADDED) --
    transp_beg = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Month to begin summing tmaxf for transpiration",
    )
    transp_end = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Month transpiration ends",
    )
    transp_tmax = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Cumulative tmax to trigger transpiration [temp_units]",
    )
    temp_units = DataArrayMeta(
        kind="parameter",
        dims=("scalar",),
        dtype=np.int64,
        description="Temperature units of transp_tmax (0=degF, 1=degC)",
    )

    # -- derived parameters (ADDED) --
    transp_tmax_f = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="transp_tmax in degF (temp_units conversion)",
    )

    # -- variables (ADDED; sequential season state) --
    tmax_sum = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Accumulated tmaxf toward the transpiration trigger "
        "[degF]",
    )
    transp_check = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.int64,
        description="In the transpiration checking period (0/1; upstream "
        "private state)",
    )

    def initialize(self) -> None:
        super().initialize()
        obj = self._obj
        obj["tmax_sum"].values[:] = 0.0
        obj["transp_check"].values[:] = 0

        # transp_tmax units ("candidate for worst code lines" upstream)
        if int(obj["temp_units"].values[0]) == 0:
            obj["transp_tmax_f"].values[:] = obj["transp_tmax"].values
        else:
            obj["transp_tmax_f"].values[:] = (
                obj["transp_tmax"].values * (9.0 / 5.0)
            ) + 32.0

    @staticmethod
    @numba.njit
    def _transp_tindex(
        transp_on: np.ndarray,
        tmax_sum: np.ndarray,
        transp_check: np.ndarray,
        tmaxf: np.ndarray,
        transp_beg: np.ndarray,
        transp_end: np.ndarray,
        transp_tmax_f: np.ndarray,
        current_month: np.int64,
        current_dom: np.int64,
        istep0: np.int64,
    ) -> None:
        # transp_tindex sequential state, extracted verbatim from the
        # (previously monolithic) atmosphere kernel -- it was the
        # final, self-contained section of the loop
        for ii in range(transp_on.shape[0]):
            if istep0 == 1:
                # upstream "time zero calculations" (start = this step)
                motmp = current_month + 12
                if current_month == transp_beg[ii]:
                    if current_dom > 10:
                        transp_on[ii] = 1.0
                    else:
                        transp_check[ii] = 1
                elif transp_end[ii] > transp_beg[ii]:
                    if (current_month > transp_beg[ii]) and (
                        current_month < transp_end[ii]
                    ):
                        transp_on[ii] = 1.0
                else:
                    if (current_month > transp_beg[ii]) or (
                        motmp < transp_end[ii] + 12
                    ):
                        transp_on[ii] = 1.0

            # (state carries between steps via the persistent arrays;
            # upstream's tt-1 copy is implicit)
            if current_dom == 1:
                # check for end of period
                if current_month == transp_end[ii]:
                    transp_on[ii] = 0.0
                    transp_check[ii] = 0
                    tmax_sum[ii] = 0.0
                # check for month to turn the check switch on
                if current_month == transp_beg[ii]:
                    transp_check[ii] = 1
                    tmax_sum[ii] = 0.0

            # in the checking period: sum tmaxf above freezing until
            # the index parameter is exceeded, then transpiration is on
            if transp_check[ii] == 1:
                if tmaxf[ii] > 32.0:
                    tmax_sum[ii] = tmax_sum[ii] + tmaxf[ii]
                if tmax_sum[ii] > transp_tmax_f[ii]:
                    transp_on[ii] = 1.0
                    transp_check[ii] = 0
                    tmax_sum[ii] = 0.0

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate_base(time)
        self._transp_tindex(
            obj["transp_on"].values,
            obj["tmax_sum"].values,
            obj["transp_check"].values,
            obj["tmaxf"].values,
            obj["transp_beg"].values,
            obj["transp_end"].values,
            obj["transp_tmax_f"].values,
            np.int64(time.month),
            np.int64(time.day_of_month),
            np.int64(1 if time.current_index == 0 else 0),
        )


class PRMSAtmosphereTranspFrost(PRMSAtmosphereBase):
    """PRMSAtmosphereBase PLUS the frost-window transpiration model
    (upstream PRMSAtmosphereTranspFrost / PRMS transp_frost.f90):
    transpiration is on between the (last) spring frost and the
    (first, killing) fall frost, both given as SOLAR-year days
    (Time.jsol: 1-based days since the most recent Dec 22).
    Stateless -- no time-zero block, no accumulation."""

    # -- process parameters (ADDED) --
    spring_frost = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Last spring frost [solar day of year]",
    )
    fall_frost = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="First killing fall frost [solar day of year]",
    )

    @staticmethod
    @numba.njit
    def _transp_frost(
        transp_on: np.ndarray,
        spring_frost: np.ndarray,
        fall_frost: np.ndarray,
        jsol: np.int64,
    ) -> None:
        for ii in range(transp_on.shape[0]):
            if (jsol >= spring_frost[ii]) and (jsol <= fall_frost[ii]):
                transp_on[ii] = 1.0
            else:
                transp_on[ii] = 0.0

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate_base(time)
        self._transp_frost(
            obj["transp_on"].values,
            obj["spring_frost"].values,
            obj["fall_frost"].values,
            np.int64(time.jsol),
        )


class PRMSAtmosphereTranspFrostDyn(PRMSAtmosphereTranspFrost):
    """PRMSAtmosphereTranspFrost with DYNAMIC frost dates (the PRMS
    dyn_springfrost_flag / dyn_fallfrost_flag configuration):
    spring_frost and fall_frost become time-varying INPUTS (served per
    step -- forward-filled from the control's PRMS dynamic-parameter
    files by the supplier) instead of static parameters. A pure
    declaration override: the frost-window kernel and calculate() are
    inherited unchanged, reading the same-named buffers that are now
    refilled each step. Validated exactly (transp_on is 0/1) against
    the fgr_ag_2yr analysis GSFLOW answers."""

    # -- process inputs (OVERRIDE: parameter -> time-varying input) --
    spring_frost = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Last spring frost [solar day of year] "
        "(TIME-VARYING: PRMS dynamic parameter)",
    )
    fall_frost = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="First killing fall frost [solar day of year] "
        "(TIME-VARYING: PRMS dynamic parameter)",
    )
