"""
hydrology/prms_canopy.py
========================
PRMSCanopy: canopy interception, ported from pywatershed
(pywatershed/hydrology/prms_canopy.py; PRMS 5.2.1 physics, PRMS-IV
documentation: Markstrom et al. 2015, USGS TM 6-B7).

Fifth REAL process port (July 2026) -- produces the canopy throughfall
products consumed by runoff and soilzone (``net_rain``/``net_snow``/
``net_ppt``/``hru_intcpevap``/``intcp_changeover``). Ported: field
declarations (names verbatim) and the numerics of ``_calculate_numpy``
+ ``intercept`` (leading underscore dropped), rewritten to the
in-place, out-first, zero-per-step-allocation convention.

**Mutable input**: ``pptmix`` (the atmosphere's rain/snow-mix flag) --
canopy ZEROES it where interception converts a trace snowfall to rain.
Canopy never READS pptmix, so feeding the (post-edit) generated answer
file as input is exact.

Quirks preserved / decisions:

- **hru_type is hardwired all-LAND**: upstream sets
  ``self._hru_type = LAND`` for every HRU (NHM supports nothing else)
  and IGNORES the discretization's hru_type -- we keep the LAKE
  branches verbatim against a module constant (numba folds them).
- ``intcp_form`` is recomputed from scratch each step (upstream
  allocates a fresh -9999 array; here a declared int64 variable,
  unconditionally written per element).
- ``intcp_transp_on`` state initializes to 0 = OFF (upstream
  get_init_values), flipping on the first summer transition.
- ``epan_coef`` hardwired 1.0 (pan ET not ported upstream either).
- upstream njits with fastmath=True; we keep strict IEEE (validated at
  upstream's own 1e-12 canopy standard).

Deliberately NOT ported: Budget/ConservativeProcess; adapters;
restart; calc_method; verbose; ``imbalance_behavior``;
``update_net_precip`` (dead code upstream); the unused ``time_length``
kernel argument.

Parameter provenance: the 7 process parameters live in
parameters_PRMSCanopy.nc (``cov_type`` is shared with PRMSSoilzone's
file -- identical NHM values); no dis variables are needed (hru_type
deliberately unused, see above).
"""

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process

# pywatershed constants
_NEARZERO = 1.0e-6  # nearzero
# dnearzero = epsilon64: pywatershed HARDCODES 2.23e-16 (slightly
# above np.finfo(float64).eps) -- verbatim, threshold branches differ
_DNEARZERO = 2.23e-16
_BARESOIL = 0  # CovType.BARESOIL
_GRASSES = 1  # CovType.GRASSES
_LAND = 1  # HruType.LAND
_LAKE = 2  # HruType.LAKE
_RAIN = 0
_SNOW = 1
_OFF = 0
_ACTIVE = 1
# upstream: self._hru_type = LAND for every HRU (NHM); the dis
# hru_type is deliberately ignored. numba folds the dead LAKE branch.
_HRU_TYPE = _LAND


@numba.njit
def intercept(precip, stor_max, cov, intcp_stor, net_precip):
    net_precip = precip * (1.0 - cov)
    intcp_stor = intcp_stor + precip
    if intcp_stor > stor_max:
        net_precip = net_precip + (intcp_stor - stor_max) * cov
        intcp_stor = stor_max
    return intcp_stor, net_precip


class PRMSCanopy(Process):
    """PRMS canopy interception: rain/snow interception by seasonal
    cover density, evaporation/sublimation of the intercepted store,
    and the winter<->summer cover-density changeover.

    Storage and fluxes are in inches (canopy-cover-relative for
    intcp_*, HRU-relative for hru_*)."""

    # ------------------------------------------------------------------
    # Field declarations (names verbatim from pywatershed)
    # ------------------------------------------------------------------

    # -- process parameters --
    cov_type = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.int64,
        description="Cover type (0=bare, 1=grasses, 2=shrubs, 3=trees, "
        "4=coniferous)",
    )
    covden_sum = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Summer vegetation cover density [-]",
    )
    covden_win = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Winter vegetation cover density [-]",
    )
    srain_intcp = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Summer rain interception storage capacity [inches]",
    )
    wrain_intcp = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Winter rain interception storage capacity [inches]",
    )
    snow_intcp = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Snow interception storage capacity [inches]",
    )
    potet_sublim = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Fraction of PET sublimated from intercepted snow [-]",
    )

    # -- inputs --
    pk_ice_prev = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snowpack ice, previous timestep [inches]",
    )
    freeh2o_prev = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snowpack free water, previous timestep [inches]",
    )
    transp_on = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Transpiration occurring (0/1 flag)",
    )
    hru_ppt = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Precipitation on the HRU [inches]",
    )
    hru_rain = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Rain on the HRU [inches]",
    )
    hru_snow = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Snow on the HRU [inches]",
    )
    potet = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Potential evapotranspiration [inches]",
    )

    # -- MUTABLE input (atmosphere's flag, zeroed in place here) --
    pptmix = DataArrayMeta(
        kind="mutable_input",
        dims=("space",),
        dtype=np.float64,
        description="Rain/snow mix flag (0/1) -- ZEROED in place where "
        "intercepted trace snow becomes rain (never read here)",
    )

    # -- variables --
    net_ppt = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Precipitation through the canopy [inches]",
    )
    net_rain = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Rain through the canopy [inches]",
    )
    net_snow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Snow through the canopy [inches]",
    )
    intcp_changeover = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Canopy throughfall from cover-density changeover "
        "+ bare-soil excess [inches]",
    )
    intcp_evap = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Canopy evaporation/sublimation [inches over cover]",
    )
    intcp_form = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.int64,
        description="Interception form (RAIN=0, SNOW=1)",
    )
    intcp_stor = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Interception storage [inches over cover]",
    )
    intcp_transp_on = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.int64,
        description="Transpiration-season state of the canopy (0/1)",
    )
    hru_intcpevap = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Canopy evaporation [inches over the HRU]",
    )
    hru_intcpstor = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Interception storage [inches over the HRU]",
    )
    hru_intcpstor_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Interception storage change [inches over the HRU]",
    )
    hru_intcpstor_old = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Interception storage, previous timestep",
    )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        obj = self._obj
        for name in (
            "net_ppt",
            "net_rain",
            "net_snow",
            "intcp_changeover",
            "intcp_evap",
            "intcp_stor",
            "hru_intcpevap",
            "hru_intcpstor",
            "hru_intcpstor_change",
            "hru_intcpstor_old",
        ):
            obj[name].values[:] = 0.0
        obj["intcp_form"].values[:] = _RAIN  # upstream init 0
        obj["intcp_transp_on"].values[:] = _OFF  # upstream init 0

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        obj = self._obj
        obj["hru_intcpstor_old"].values[:] = obj["hru_intcpstor"].values

    @staticmethod
    @numba.njit
    def _calculate(
        # in/out state + outputs (written in place)
        net_ppt: np.ndarray,
        net_rain: np.ndarray,
        net_snow: np.ndarray,
        intcp_changeover: np.ndarray,
        intcp_evap: np.ndarray,
        intcp_form: np.ndarray,
        intcp_stor: np.ndarray,
        intcp_transp_on: np.ndarray,
        hru_intcpevap: np.ndarray,
        hru_intcpstor: np.ndarray,
        hru_intcpstor_change: np.ndarray,
        # mutable input (atmosphere's flag, zeroed in place)
        pptmix: np.ndarray,
        # prior state (read-only here; advance() maintains)
        hru_intcpstor_old: np.ndarray,
        # inputs
        pk_ice_prev: np.ndarray,
        freeh2o_prev: np.ndarray,
        transp_on: np.ndarray,
        hru_ppt: np.ndarray,
        hru_rain: np.ndarray,
        hru_snow: np.ndarray,
        potet: np.ndarray,
        # parameters
        cov_type: np.ndarray,
        covden_sum: np.ndarray,
        covden_win: np.ndarray,
        srain_intcp: np.ndarray,
        wrain_intcp: np.ndarray,
        snow_intcp: np.ndarray,
        potet_sublim: np.ndarray,
    ) -> None:
        nhru = net_ppt.shape[0]
        for ii in range(nhru):
            netrain = hru_rain[ii]
            netsnow = hru_snow[ii]

            if transp_on[ii] == _ACTIVE:
                cov = covden_sum[ii]
                stor_max_rain = srain_intcp[ii]
            else:
                cov = covden_win[ii]
                stor_max_rain = wrain_intcp[ii]

            intcp_form[ii] = _RAIN
            if hru_snow[ii] > 0.0:
                intcp_form[ii] = _SNOW

            intcpstor = intcp_stor[ii]
            intcpevap = 0.0
            changeover = 0.0
            extra_water = 0.0

            # lake or bare ground hrus (hru_type hardwired LAND -- the
            # LAKE half is dead, kept verbatim)
            if _HRU_TYPE == _LAKE or cov_type[ii] == _BARESOIL:
                if cov_type[ii] == _BARESOIL and intcpstor > 0.0:
                    extra_water = intcp_stor[ii]
                intcpstor = 0.0

            # ***** go from summer to winter cover density
            if transp_on[ii] == _OFF and intcp_transp_on[ii] == _ACTIVE:
                intcp_transp_on[ii] = _OFF
                if intcpstor > 0.0:
                    diff = covden_sum[ii] - cov
                    changeover = intcpstor * diff
                    if cov > 0.0:
                        if changeover < 0.0:
                            intcpstor = intcpstor * covden_sum[ii] / cov
                            changeover = 0.0
                    else:
                        intcpstor = 0.0

            # **** go from winter to summer cover density
            elif transp_on[ii] == _ACTIVE and intcp_transp_on[ii] == _OFF:
                intcp_transp_on[ii] = _ACTIVE
                if intcpstor > 0.0:
                    diff = covden_win[ii] - cov
                    changeover = intcpstor * diff
                    if cov > 0.0:
                        if changeover < 0.0:
                            intcpstor = intcpstor * covden_win[ii] / cov
                            changeover = 0.0
                    else:
                        intcpstor = 0.0

            # ***** determine the amount of interception from rain
            if _HRU_TYPE != _LAKE and cov_type[ii] != _BARESOIL:
                if hru_rain[ii] > 0.0:
                    if cov > 0.0:
                        if cov_type[ii] > _GRASSES:
                            intcpstor, netrain = intercept(
                                hru_rain[ii],
                                stor_max_rain,
                                cov,
                                intcpstor,
                                netrain,
                            )
                        elif cov_type[ii] == _GRASSES:
                            # no snowpack and no snowfall: grasses can
                            # intercept rain
                            if (
                                pk_ice_prev[ii] + freeh2o_prev[ii]
                            ) < _DNEARZERO and netsnow < _NEARZERO:
                                intcpstor, netrain = intercept(
                                    hru_rain[ii],
                                    stor_max_rain,
                                    cov,
                                    intcpstor,
                                    netrain,
                                )

            # ***** determine amount of interception from snow
            if hru_snow[ii] > 0.0:
                if cov > 0.0:
                    if cov_type[ii] > _GRASSES:
                        intcpstor, netsnow = intercept(
                            hru_snow[ii],
                            snow_intcp[ii],
                            cov,
                            intcpstor,
                            netsnow,
                        )
                        if netsnow < _NEARZERO:
                            netrain = netrain + netsnow
                            netsnow = 0.0
                            pptmix[ii] = 0

            # ***** evaporation or sublimation of interception
            # (if precipitation, assume neither)
            if intcpstor > 0.0:
                if hru_ppt[ii] < _NEARZERO:
                    epan_coef = 1.0  # pan ET not ported (upstream todo)
                    evrn = potet[ii] / epan_coef
                    evsn = potet[ii] * potet_sublim[ii]

                    if intcp_form[ii] == _SNOW:
                        z = intcpstor - evsn
                        if z > 0:
                            intcpstor = z
                            intcpevap = evsn
                        else:
                            intcpevap = intcpstor
                            intcpstor = 0.0
                    else:
                        d = intcpstor - evrn
                        if d > 0.0:
                            intcpstor = d
                            intcpevap = evrn
                        else:
                            intcpevap = intcpstor
                            intcpstor = 0.0

            if intcpevap * cov > potet[ii]:
                last = intcpevap
                if cov > 0.0:
                    intcpevap = potet[ii] / cov
                else:
                    intcpevap = 0.0
                intcpstor = intcpstor + last - intcpevap

            # store calculated values in output variables
            intcp_evap[ii] = intcpevap
            intcp_stor[ii] = intcpstor
            net_rain[ii] = netrain
            net_snow[ii] = netsnow
            net_ppt[ii] = netrain + netsnow
            hru_intcpstor[ii] = intcpstor * cov
            hru_intcpevap[ii] = intcpevap * cov
            intcp_changeover[ii] = changeover + extra_water

            # upstream post-kernel array line, folded per element
            hru_intcpstor_change[ii] = (
                hru_intcpstor[ii] - hru_intcpstor_old[ii]
            )

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate(
            obj["net_ppt"].values,
            obj["net_rain"].values,
            obj["net_snow"].values,
            obj["intcp_changeover"].values,
            obj["intcp_evap"].values,
            obj["intcp_form"].values,
            obj["intcp_stor"].values,
            obj["intcp_transp_on"].values,
            obj["hru_intcpevap"].values,
            obj["hru_intcpstor"].values,
            obj["hru_intcpstor_change"].values,
            obj["pptmix"].values,
            obj["hru_intcpstor_old"].values,
            obj["pk_ice_prev"].values,
            obj["freeh2o_prev"].values,
            obj["transp_on"].values,
            obj["hru_ppt"].values,
            obj["hru_rain"].values,
            obj["hru_snow"].values,
            obj["potet"].values,
            obj["cov_type"].values,
            obj["covden_sum"].values,
            obj["covden_win"].values,
            obj["srain_intcp"].values,
            obj["wrain_intcp"].values,
            obj["snow_intcp"].values,
            obj["potet_sublim"].values,
        )
