"""
processes_attrs2.py
===================
Upper and Lower as explicit Process subclasses using base_attrs2.py.

Compare with processes_attrs.py:

  processes_attrs.py                 processes_attrs2.py
  ------------------                 -------------------
  @process                           class Upper(Process):  # no decorator
  class Upper:                           ...
      ...                                def advance(self): ...
      @staticmethod
      def advance(ds): ...               def calculate(self, dt):
      @staticmethod                          self._obj[...] = self._calculate(...)
      def calculate(ds, dt): ...
                                         @staticmethod
                                         def _calculate(...):  # numba target
                                             ...

  Upper(parameters=..., **kwargs)    Upper.new(parameters=..., **kwargs)

Key differences from processes_attrs.py:
  - No @process decorator -- Upper/Lower are plain Process subclasses
  - Construction via Upper.new(...) / Lower.new(...) -- explicit classmethod
    on the ABC, returns xr.Dataset; no __new__ tricks
  - advance / calculate are instance methods; self._obj is the Dataset
  - _calculate is a @staticmethod taking raw numpy arrays -- the natural
    target for @numba.jit when performance work begins
  - PWS._registry populated explicitly at the bottom of this file
"""

import numpy as np
import xarray as xr
from base_attrs2 import PWS, DataArrayMeta, Process


class Upper(Process):
    # ------------------------------------------------------------------
    # Field declarations
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
    # Computation
    # ------------------------------------------------------------------

    def advance(self, ds: xr.Dataset) -> None:
        ds["flow_previous"].values[:] = ds["flow"].values

    @staticmethod
    def _calculate(
        flow_previous: np.ndarray,
        forcing_0: np.ndarray,
        dt: np.float64,
    ) -> np.ndarray:
        # Pure numpy -- decorate with @numba.jit when ready
        return flow_previous * np.float64(0.95) + forcing_0

    def calculate(self, ds: xr.Dataset, dt: np.float64) -> None:
        ds["flow"].values[:] = self._calculate(
            ds["flow_previous"].values,
            ds["forcing_0"].values,
            dt,
        )


class Lower(Process):
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

    def advance(self, ds: xr.Dataset) -> None:
        ds["storage_previous"].values[:] = ds["storage"].values

    @staticmethod
    def _calculate(
        storage_previous: np.ndarray,
        flow: np.ndarray,
        dt: np.float64,
    ) -> np.ndarray:
        # Pure numpy -- decorate with @numba.jit when ready
        return storage_previous * np.float64(0.95) + flow * np.float64(0.12)

    def calculate(self, ds: xr.Dataset, dt: np.float64) -> None:
        ds["storage"].values[:] = self._calculate(
            ds["storage_previous"].values,
            ds["flow"].values,
            dt,
        )


# ---------------------------------------------------------------------------
# Register in the PWS accessor registry.
# Explicit assignment mirrors PWS_Dataset_jlm.py's pattern of class
# attributes -- no decorator magic required.
# ---------------------------------------------------------------------------

PWS._registry["Upper"] = Upper
PWS._registry["Lower"] = Lower
