"""
hydrology/prms_groundwater.py
=============================
PRMSGroundwaterNoDprst + PRMSGroundwater: the PRMS groundwater
reservoir, ported from pywatershed
(pywatershed/hydrology/prms_groundwater.py; PRMS 5.2.1 physics, PRMS-IV
documentation: Markstrom et al. 2015, USGS TM 6-B7).

First REAL process port (July 2026). Ported: the field declarations
(names verbatim -- the goal-4 pathway) and the numerics of
_calculate_numpy, rewritten to this framework's in-place, out-first,
zero-per-step-allocation kernel convention. The kernel is an explicit
element loop with SCALAR temporaries: pywatershed's staged array
expressions would materialize named intermediate arrays every step,
even under numba (expression fusion does not eliminate named arrays).
Per element the operation order is identical, so results match
pywatershed's to within its own autotest tolerance (1e-13).

**Variant structure (ADDITIVE -- see PORTS.md "How variants are done
here")**: pywatershed derives PRMSGroundwaterNoDprst FROM
PRMSGroundwater by subtraction (re-declared interface, a per-step zero
array fed to the shared kernel). Here the hierarchy points the right
way: ``PRMSGroundwaterNoDprst`` is the minimal core (soil_to_gw +
ssr_to_gw inflows) and ``PRMSGroundwater`` EXTENDS it by adding the
``dprst_seep_hru`` input and the kernel term that consumes it. Each
class owns its kernel; the core never touches dprst.

Deliberately NOT ported:
- Budget / ConservativeProcess machinery -- FLAGGED for a later design
  pass; when it comes, scrutinize its design (e.g. the separation /
  combination of mass and energy budgets).
- adapters (adaptable / adapter_factory) -- the Model wires inputs.
- restart read/write; verbose; the calc_method switch (numba is THE
  compute path here); the dprst_flag=False zero-input hack (the
  additive NoDprst base above replaces it).
- gwstor_min: declared by pywatershed's get_parameters() but unused by
  its kernel.

Parameter provenance (pywatershed utils/separate_nhm_params.py):
hru_area and hru_in_to_cf are DIS_HRU variables -- they belong to the
hru discretization (parameters_dis_hru.nc), not to this process
(parameters_PRMSGroundwater.nc). Both are declared as plain "parameter"
fields (the declaration states the NEED); the Model sources declared
parameters DIS-FIRST, so they arrive via
``Discretization(..., parameters=parameters_dis_hru.nc)`` and land on
the grid's shared dataset (structurally shared with any other process
that declares them). Under MPI the distributed grid's dis variables
ride in the combined input file instead.
"""

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process


class PRMSGroundwaterNoDprst(Process):
    """PRMS groundwater reservoir without depression storage: one
    linear reservoir per HRU, fed by soil_to_gw + ssr_to_gw.

    The minimal core of the groundwater family; PRMSGroundwater adds
    the depression-storage seepage inflow.

    Storage and fluxes are in inches over the HRU (PRMS convention);
    the kernel works in volume (acre-inches) internally and
    gwres_flow_vol is in cubic feet (via hru_in_to_cf).
    """

    # ------------------------------------------------------------------
    # Field declarations (names verbatim from pywatershed)
    # ------------------------------------------------------------------

    # -- dis_hru variables (grid-owned; see module docstring) --
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
    )

    # -- process parameters --
    gwflow_coef = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater routing coefficient [1/day]",
    )
    gwsink_coef = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater sink coefficient [1/day]",
    )

    # -- inputs --
    soil_to_gw = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description=(
            "Portion of excess capillary flow that drains to the GWR"
        ),
    )
    ssr_to_gw = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Drainage from the gravity reservoir to the GWR",
    )

    # -- variables --
    gwres_stor = DataArrayMeta(
        kind="variable",
        restart=True,
        dims=("space",),
        dtype=np.float64,
        description="Groundwater reservoir storage [inches]",
        initial="gwstor_init",
    )
    gwres_stor_old = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater reservoir storage, previous timestep",
        initial="gwstor_init",
    )
    gwres_flow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater discharge to the stream network [inches]",
    )
    gwres_sink = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater loss to the sink (leaves the model) [inches]",
    )
    gwres_stor_change = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Change in groundwater storage over the timestep [inches]",
    )
    gwres_flow_vol = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Groundwater discharge volume [cubic feet]",
    )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        self._obj["gwres_stor_old"].values[:] = self._obj["gwres_stor"].values

    @staticmethod
    @numba.njit
    def _calculate(
        gwres_stor: np.ndarray,
        gwres_flow: np.ndarray,
        gwres_sink: np.ndarray,
        gwres_stor_change: np.ndarray,
        gwres_flow_vol: np.ndarray,
        gwres_stor_old: np.ndarray,
        soil_to_gw: np.ndarray,
        ssr_to_gw: np.ndarray,
        hru_area: np.ndarray,
        hru_in_to_cf: np.ndarray,
        gwflow_coef: np.ndarray,
        gwsink_coef: np.ndarray,
    ) -> None:
        for ii in range(gwres_stor.shape[0]):
            area = hru_area[ii]
            # to volume (acre-inches); the dprst kernel below with a
            # zero dprst_seep_hru reduces to exactly this sum
            stor = gwres_stor[ii] * area
            stor = stor + (soil_to_gw[ii] * area + ssr_to_gw[ii] * area)
            flow = stor * gwflow_coef[ii]
            stor = stor - flow
            sink = stor * gwsink_coef[ii]
            if sink > stor:
                sink = stor
            stor = stor - sink
            # back to inches over the HRU
            gwres_stor[ii] = stor / area
            gwres_flow[ii] = flow / area
            gwres_sink[ii] = sink / area
            gwres_stor_change[ii] = gwres_stor[ii] - gwres_stor_old[ii]
            gwres_flow_vol[ii] = gwres_flow[ii] * hru_in_to_cf[ii]

    def calculate(self, dt: np.float64, time: Time) -> None:
        self._calculate(
            self._obj["gwres_stor"].values,
            self._obj["gwres_flow"].values,
            self._obj["gwres_sink"].values,
            self._obj["gwres_stor_change"].values,
            self._obj["gwres_flow_vol"].values,
            self._obj["gwres_stor_old"].values,
            self._obj["soil_to_gw"].values,
            self._obj["ssr_to_gw"].values,
            self._obj["hru_area"].values,
            self._obj["hru_in_to_cf"].values,
            self._obj["gwflow_coef"].values,
            self._obj["gwsink_coef"].values,
        )


class PRMSGroundwater(PRMSGroundwaterNoDprst):
    """PRMS groundwater reservoir with depression storage: adds the
    dprst_seep_hru inflow to the NoDprst core.

    Storage and fluxes are in inches over the HRU (PRMS convention);
    the kernel works in volume (acre-inches) internally and
    gwres_flow_vol is in cubic feet (via hru_in_to_cf).
    """

    # -- inputs (ADDED to the NoDprst core) --
    dprst_seep_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description=("Seepage from surface-depression storage to the GWR"),
    )

    @staticmethod
    @numba.njit
    def _calculate(
        gwres_stor: np.ndarray,
        gwres_flow: np.ndarray,
        gwres_sink: np.ndarray,
        gwres_stor_change: np.ndarray,
        gwres_flow_vol: np.ndarray,
        gwres_stor_old: np.ndarray,
        soil_to_gw: np.ndarray,
        ssr_to_gw: np.ndarray,
        dprst_seep_hru: np.ndarray,
        hru_area: np.ndarray,
        hru_in_to_cf: np.ndarray,
        gwflow_coef: np.ndarray,
        gwsink_coef: np.ndarray,
    ) -> None:
        for ii in range(gwres_stor.shape[0]):
            area = hru_area[ii]
            # to volume (acre-inches); element-wise operation order
            # matches pywatershed's _calculate_numpy exactly
            stor = gwres_stor[ii] * area
            stor = stor + (
                soil_to_gw[ii] * area
                + ssr_to_gw[ii] * area
                + dprst_seep_hru[ii] * area
            )
            flow = stor * gwflow_coef[ii]
            stor = stor - flow
            sink = stor * gwsink_coef[ii]
            if sink > stor:
                sink = stor
            stor = stor - sink
            # back to inches over the HRU
            gwres_stor[ii] = stor / area
            gwres_flow[ii] = flow / area
            gwres_sink[ii] = sink / area
            gwres_stor_change[ii] = gwres_stor[ii] - gwres_stor_old[ii]
            gwres_flow_vol[ii] = gwres_flow[ii] * hru_in_to_cf[ii]

    def calculate(self, dt: np.float64, time: Time) -> None:
        self._calculate(
            self._obj["gwres_stor"].values,
            self._obj["gwres_flow"].values,
            self._obj["gwres_sink"].values,
            self._obj["gwres_stor_change"].values,
            self._obj["gwres_flow_vol"].values,
            self._obj["gwres_stor_old"].values,
            self._obj["soil_to_gw"].values,
            self._obj["ssr_to_gw"].values,
            self._obj["dprst_seep_hru"].values,
            self._obj["hru_area"].values,
            self._obj["hru_in_to_cf"].values,
            self._obj["gwflow_coef"].values,
            self._obj["gwsink_coef"].values,
        )
