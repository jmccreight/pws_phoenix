"""
hydrology/prms_hydraulic_geometry.py
====================================
PRMSHydraulicGeometryWidthOnly + PRMSHydraulicGeometryFull:
flow-dependent hydraulic geometry for stream segments, ported from
pywatershed (pywatershed/hydrology/prms_hydraulic_geometry.py; PRMS
5.2.1 strmflow_character.f90). Stage 1 of the stream-temperature arc:
produces seg_flow_width consumed by PRMSStreamTemp.

Power laws on segment outflow (converted to cms):
width = width_alpha * flow^width_m; depth = depth_alpha * flow^depth_m;
area = width * depth; velocity = flow / area;
residence time = area * length / flow. All zero where flow is zero.

Family structure (ADDITIVE; PORTS.md "How variants are done here"):
upstream inverts again -- its WidthOnly(Full) SUBTRACTS the depth
parameters and overwrites them with the PRMS defaults (its own
comment: "I dont love the design"). Here the direction is fixed with
the declaration-override seam: ``PRMSHydraulicGeometryWidthOnly`` is
the core, declaring ``depth_alpha``/``depth_m`` as
``parameter_internal`` filled with the strmflow_character defaults
(0.27 / 0.39) at initialize; ``PRMSHydraulicGeometryFull`` EXTENDS it
by overriding those two declarations to kind="parameter" (supplied)
and no-op'ing the default fill. One shared kernel. NHM parameter
files (drb included) carry only the width parameters, so WidthOnly is
the class the nhm_stream_temp configuration exercises.

Validation naming: PRMS writes these as seg_width / seg_depth /
seg_area / seg_velocity / seg_res_time; pywatershed renamed the
variables seg_flow_* (res_time unchanged) -- the parity test maps
names explicitly (upstream's own test silently skips all but
seg_res_time via skip_missing_ans).

Parameter provenance: seg_length is a DIS_SEG variable (dis-first
sourcing via parameters_dis_seg.nc); width_alpha/width_m live in
parameters_PRMSHydraulicGeometry*.nc.
"""

import numba
import numpy as np

from globals import Time
from process import DataArrayMeta, Process

# pywatershed constants
_NEARZERO = 1.0e-6
_CFS_TO_CMS = 0.028316847
# strmflow_character.f90 depth defaults (range 0.12-0.63 m / 0.38-0.40)
_DEPTH_ALPHA_DEFAULT = 0.27
_DEPTH_M_DEFAULT = 0.39


class PRMSHydraulicGeometryWidthOnly(Process):
    """Hydraulic geometry with the PRMS default depth relationship:
    the family core (NHM parameter files carry width only).

    Flow in cfs (seg_outflow) converted to cms internally; width/depth
    in meters, area m^2, velocity m/s, residence time seconds.
    """

    # -- dis_seg variables (grid-owned; dis-first sourcing) --
    seg_length = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Segment length [m]",
    )

    # -- process parameters --
    width_alpha = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Width power-law coefficient [m per (cms)^width_m]",
    )
    width_m = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Width power-law exponent [-]",
    )

    # -- depth relationship: DERIVED defaults in this core; the Full
    # variant overrides these declarations to supplied parameters --
    depth_alpha = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Depth power-law coefficient [m per (cms)^depth_m] "
        "(strmflow_character default 0.27 in this core)",
    )
    depth_m = DataArrayMeta(
        kind="parameter_internal",
        dims=("space",),
        dtype=np.float64,
        description="Depth power-law exponent [-] "
        "(strmflow_character default 0.39 in this core)",
    )

    # -- inputs --
    seg_outflow = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Streamflow leaving each segment [cfs]",
    )

    # -- variables --
    seg_flow_width = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Flow width (PRMS seg_width) [m]",
    )
    seg_flow_depth = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Flow depth (PRMS seg_depth) [m]",
    )
    seg_flow_area = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Flow cross-sectional area (PRMS seg_area) [m^2]",
    )
    seg_flow_velocity = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Flow velocity (PRMS seg_velocity) [m/s]",
    )
    seg_res_time = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Segment residence time [seconds]",
    )

    def initialize(self) -> None:
        obj = self._obj
        for name in (
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_area",
            "seg_flow_velocity",
            "seg_res_time",
        ):
            obj[name].values[:] = 0.0
        self._set_depth_params()

    def _set_depth_params(self) -> None:
        """The strmflow_character defaults (Full overrides to no-op:
        its depth_alpha/depth_m are supplied parameters)."""
        self._obj["depth_alpha"].values[:] = _DEPTH_ALPHA_DEFAULT
        self._obj["depth_m"].values[:] = _DEPTH_M_DEFAULT

    def advance(self) -> None:
        pass  # no *_prev state

    @staticmethod
    @numba.njit
    def _calculate(
        # outputs (written in place)
        seg_flow_width: np.ndarray,
        seg_flow_depth: np.ndarray,
        seg_flow_area: np.ndarray,
        seg_flow_velocity: np.ndarray,
        seg_res_time: np.ndarray,
        # input
        seg_outflow: np.ndarray,
        # parameters + derived
        width_alpha: np.ndarray,
        width_m: np.ndarray,
        depth_alpha: np.ndarray,
        depth_m: np.ndarray,
        seg_length: np.ndarray,
    ) -> None:
        for ss in range(seg_outflow.shape[0]):
            flow_cms = seg_outflow[ss] * _CFS_TO_CMS
            seg_flow_width[ss] = 0.0
            seg_flow_depth[ss] = 0.0
            seg_flow_area[ss] = 0.0
            seg_flow_velocity[ss] = 0.0
            seg_res_time[ss] = 0.0
            if flow_cms > 0.0:
                seg_flow_width[ss] = width_alpha[ss] * flow_cms ** width_m[ss]
                seg_flow_depth[ss] = depth_alpha[ss] * flow_cms ** depth_m[ss]
                seg_flow_area[ss] = seg_flow_width[ss] * seg_flow_depth[ss]
                if seg_flow_area[ss] > _NEARZERO:
                    seg_flow_velocity[ss] = flow_cms / seg_flow_area[ss]
                seg_res_time[ss] = (
                    seg_flow_area[ss] * seg_length[ss]
                ) / flow_cms

    def calculate(self, dt: np.float64, time: Time) -> None:
        obj = self._obj
        self._calculate(
            obj["seg_flow_width"].values,
            obj["seg_flow_depth"].values,
            obj["seg_flow_area"].values,
            obj["seg_flow_velocity"].values,
            obj["seg_res_time"].values,
            obj["seg_outflow"].values,
            obj["width_alpha"].values,
            obj["width_m"].values,
            obj["depth_alpha"].values,
            obj["depth_m"].values,
            obj["seg_length"].values,
        )


class PRMSHydraulicGeometryFull(PRMSHydraulicGeometryWidthOnly):
    """Hydraulic geometry with SUPPLIED depth parameters: the core
    plus depth_alpha/depth_m as real parameters (declaration
    override, parameter_internal -> parameter)."""

    depth_alpha = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Depth power-law coefficient [m per (cms)^depth_m]",
    )
    depth_m = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Depth power-law exponent [-]",
    )

    def _set_depth_params(self) -> None:
        pass  # supplied parameters; nothing to fill
