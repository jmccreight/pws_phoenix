"""
processes_concrete.py
=====================
Upper and Lower: concrete Process subclasses for the toy Upper/Lower model.

  - No decorator -- Upper/Lower subclass Process directly.
  - Auto-registered in Process._registry via __init_subclass__.
  - Construction via Upper.new(...) -- classmethod on the ABC.
  - advance(self) / calculate(self, dt) are instance methods using self._obj.
  - _calculate is a @staticmethod taking raw numpy arrays -- the natural
    target for @numba.jit when performance work begins.
  - Upper/Lower added as class attrs on PWS for construction syntax:
      Upper.new(...) or xr.Dataset.pws.Upper.new(...)
"""

import numpy as np
from globals import Time
from process import PWS, DataArrayMeta, Process


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
        dims=("month", "space"),
        dtype=np.float64,
        description="Upper zone parameter 1 (time-varying: cyclic monthly)",
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

    def advance(self) -> None:
        self._obj["flow_previous"].values[:] = self._obj["flow"].values

    @staticmethod
    def _calculate(
        flow_previous: np.ndarray,
        forcing_0: np.ndarray,
        param_up_1: np.ndarray,
        dt: np.float64,
    ) -> np.ndarray:
        # Pure numpy -- decorate with @numba.jit when ready
        return flow_previous * np.float64(0.95) + forcing_0 * param_up_1

    def calculate(self, dt: np.float64, time: Time) -> None:
        # param_up_1 is cyclic-monthly; index by current month
        param_up_1_now = self._obj["param_up_1"].values[time.month - 1]
        self._obj["flow"].values[:] = self._calculate(
            self._obj["flow_previous"].values,
            self._obj["forcing_0"].values,
            param_up_1_now,
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
    forcing_low = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Lower zone's own forcing (read-only)",
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

    def advance(self) -> None:
        self._obj["storage_previous"].values[:] = self._obj["storage"].values

    @staticmethod
    def _calculate(
        storage_previous: np.ndarray,
        flow: np.ndarray,
        forcing_low: np.ndarray,
        dt: np.float64,
    ) -> np.ndarray:
        # Pure numpy -- decorate with @numba.jit when ready
        return (
            storage_previous * np.float64(0.95)
            + flow * np.float64(0.12)
            + forcing_low * np.float64(0.10)
        )

    def calculate(self, dt: np.float64, time: Time) -> None:
        self._obj["storage"].values[:] = self._calculate(
            self._obj["storage_previous"].values,
            self._obj["flow"].values,
            self._obj["forcing_low"].values,
            dt,
        )


# ---------------------------------------------------------------------------
# Attach Process subclasses to PWS for convenient construction syntax:
#     Upper.new(...)  or  xr.Dataset.pws.Upper.new(...)
# Dispatch is handled automatically via Process._registry (__init_subclass__).
# ---------------------------------------------------------------------------

PWS.Upper = Upper  # type: ignore[attr-defined]
PWS.Lower = Lower  # type: ignore[attr-defined]
