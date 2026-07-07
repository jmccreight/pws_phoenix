"""
globals.py
==========
Global runtime objects shared across a model run. Deliberately split (see
pws_phoenix/CLAUDE.md):

  - Time:    the model clock -- runtime state, stepped every timestep, and
             passed all the way down to processes (and transparent for
             debugging).
  - Options: (future) construction-time run configuration, consumed at build.
             Today the Model `control` dict plays this role.

Time is the source of truth for "what step are we on": in the serial path the
Model drives the loop and sets it; in the MPI path mpixarray owns the time
*loop*, so Time is synced from the streaming step. Either way a Process reads it
the same way -- e.g. `time.month` for a time-varying parameter lookup.
"""

import numpy as np
import xarray as xr


class Time:
    """The model clock: a fixed, daily time axis known at initialization.

    Built from the model's `time` dim-coordinate. Exposes the current date and
    derived calendar fields -- `month`, `year`, `day_of_month`, `doy`
    (day-of-year), `dowy` (day-of-water-year, Oct 1) -- used by time-varying
    parameters and for debugging. All fields are 1-based (calendar-natural,
    matching pywatershed `utils/time_utils.py`); positional tv-param lookup
    therefore indexes with `field - 1` (see Upper.calculate). `set_index()`
    points it at a given timestep -- called each step by the serial Model (from
    its loop counter) and by the MPI path (from mpixarray's streaming step).
    """

    def __init__(self, times: xr.DataArray | np.ndarray) -> None:
        arr = times.values if isinstance(times, xr.DataArray) else times
        self._times: np.ndarray = np.asarray(arr)
        self.n_time: int = len(self._times)
        self.current_index: int = 0

    @property
    def current(self) -> np.datetime64:
        """The current model time as a numpy datetime64 scalar."""
        return self._times[self.current_index]

    @property
    def month(self) -> int:
        """Calendar month (1-12) of the current model time."""
        return int(self.current.astype("datetime64[M]").astype(int) % 12 + 1)

    @property
    def year(self) -> int:
        """Calendar year of the current model time."""
        return int(self.current.astype("datetime64[Y]").astype(int) + 1970)

    @property
    def day_of_month(self) -> int:
        """Day of month (1-based) of the current model time."""
        day = self.current.astype("datetime64[D]")
        mstart = self.current.astype("datetime64[M]").astype("datetime64[D]")
        return int((day - mstart).astype(int) + 1)

    @property
    def doy(self) -> int:
        """Day of year (1-based) of the current model time."""
        year_start = self.current.astype("datetime64[Y]")
        delta = (self.current - year_start).astype("timedelta64[D]")
        return int(delta.astype(int) + 1)

    @property
    def dowy(self) -> int:
        """Day of water year (1-based, Oct 1 start) of the current time."""
        year = self.year - (1 if self.month < 10 else 0)
        wy_start = np.datetime64(f"{year}-10-01")
        delta = (self.current - wy_start).astype("timedelta64[D]")
        return int(delta.astype(int) + 1)

    @property
    def current_epiweek(self) -> int:
        """CDC epiweek (1-53) of the current model time (matches
        pywatershed utils/time_utils.datetime_epiweek). Lazy-imports
        `epiweeks` -- only seasonal params (STARFIT) need it, so a run
        without them never requires the package."""
        import datetime as _datetime

        import epiweeks

        val = self.current.astype("datetime64[s]").astype(_datetime.datetime)
        return int(epiweeks.Week.fromdate(val).week)

    def set_index(self, index: int) -> None:
        """Point the clock at a specific timestep index."""
        self.current_index = index

    def __repr__(self) -> str:
        return (
            f"Time(current={self.current}, "
            f"index={self.current_index}/{self.n_time})"
        )
