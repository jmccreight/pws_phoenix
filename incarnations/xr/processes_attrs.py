"""
processes_attrs.py
==================
Upper and Lower processes reimplemented using the @process decorator and
declarative field markers from base_attrs.py.

Compare with processes.py side-by-side:

  processes.py                       processes_attrs.py
  ------------                       ------------------
  class Upper(Process):              @process
      def __init__(self, ...):       class Upper:
          super().__init__(...)
      @staticmethod                  param_up_0 = parameter(...)
      def get_parameters():          forcing_0  = input_var(...)
          return ("param_up_0", ...) flow       = variable(...)
      @staticmethod
      def get_variables():           @staticmethod
          return {"flow": {...}}     def advance(ds): ...
      def advance(self): ...
      def calculate(self, dt): ...   @staticmethod
                                     def calculate(ds, dt): ...

Key differences:
  - advance/calculate take ds explicitly (no self); the process IS the Dataset
  - declarations are co-located with the class, not split across static methods
  - @process handles __new__ so Upper(...) returns an xr.Dataset
"""

import numpy as np
import xarray as xr
from base_attrs import DataArrayMeta, process


@process
class Upper:
    # ------------------------------------------------------------------
    # Field declarations (replace get_parameters / get_inputs /
    # get_variables static methods from processes.py)
    # ------------------------------------------------------------------
    param_up_0 = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Upper zone parameter 0",
    )
    param_up_1 = DataArrayMeta(
        kind="parameter",
        dims=("time", "space"),
        dtype=np.float64,
        description="Upper zone parameter 1 (time-varying)",
    )
    param_common = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Parameter shared with Lower",
    )
    forcing_0 = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Primary forcing (read-only)",
    )
    forcing_common = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Common forcing shared with Lower (read-only)",
    )
    flow = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Flow (public -- shared downstream with Lower)",
        initial="flow_initial",
    )
    flow_previous = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Flow at previous time step",
    )

    # ------------------------------------------------------------------
    # Computation -- ds is the process Dataset, no self
    # ------------------------------------------------------------------

    @staticmethod
    def advance(ds: xr.Dataset) -> None:
        ds["flow_previous"].values[:] = ds["flow"].values

    @staticmethod
    def calculate(ds: xr.Dataset, dt: np.float64) -> None:
        for loc in ds["space"]:
            ds["flow"][loc] = (
                ds["flow_previous"][loc] * np.float64(0.95)
                + ds["forcing_0"][loc]
            )


@process
class Lower:
    # ------------------------------------------------------------------
    # Field declarations
    # ------------------------------------------------------------------
    param_low_0 = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Lower zone parameter 0",
    )
    param_common = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="Parameter shared with Upper",
    )
    flow = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Flow from Upper (read-only input)",
    )
    forcing_common = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Common forcing shared with Upper (read-only)",
    )
    storage = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Storage",
        initial="storage_initial",
    )
    storage_previous = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="Storage at previous time step",
    )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    @staticmethod
    def advance(ds: xr.Dataset) -> None:
        ds["storage_previous"].values[:] = ds["storage"].values

    @staticmethod
    def calculate(ds: xr.Dataset, dt: np.float64) -> None:
        for loc in ds["space"]:
            ds["storage"][loc] = ds["storage_previous"][loc] * np.float64(
                0.95
            ) + ds["flow"][loc] * np.float64(0.12)
