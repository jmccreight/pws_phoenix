"""
mre_accessor_design.py
======================
Minimal, self-contained example illustrating the two key design ideas in
base_attrs.py:

  1. PWSAccessor  -- an xr.Dataset accessor whose methods are resolved
                     at *runtime* from callables stored in ds.attrs.
  2. process_factory -- a function that builds an xr.Dataset "process"
                        whose behaviour (advance / calculate) varies per
                        process *spec* class.

No NumPy arrays, no file I/O, no real hydrology -- just plain Python/xarray.

Run with:
    python mre_accessor_design.py
"""

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# 1.  The accessor
#     ds.attrs["advance"]   -> a callable(ds)
#     ds.attrs["calculate"] -> a callable(ds, dt)
#
#     The accessor is registered once, globally; the *behaviour* it dispatches
#     to is decided per-dataset at factory time.
# ---------------------------------------------------------------------------


@xr.register_dataset_accessor("pws")
class PWSAccessor:
    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def advance(self) -> None:
        """Copy current state to *_prev so calculate() has a snapshot."""
        self._ds.attrs["advance"](self._ds)

    def calculate(self, dt: float) -> None:
        """Update state for one time step."""
        self._ds.attrs["calculate"](self._ds, dt)

    # ---------- introspection (stored as plain tuples in attrs) ----------
    def params(self) -> tuple:
        return self._ds.attrs.get("params", ())

    def variables(self) -> tuple:
        return self._ds.attrs.get("variables", ())


# ---------------------------------------------------------------------------
# 2.  The factory
#     Given a *spec class* (which carries field declarations + static methods)
#     it returns an xr.Dataset whose .pws accessor dispatches to that class.
# ---------------------------------------------------------------------------


def process_factory(spec_cls, **init_values) -> xr.Dataset:
    """Build a process xr.Dataset from a spec class.

    Parameters
    ----------
    spec_cls:
        A class that declares:
          - class-level ``params``    tuple of str
          - class-level ``variables`` tuple of str
          - static method advance(ds)
          - static method calculate(ds, dt)
    **init_values:
        Scalar initial values for every name in params + variables.

    Returns
    -------
    xr.Dataset with all fields as 0-d DataArrays and behaviour baked into
    ds.attrs.
    """
    all_names = tuple(spec_cls.params) + tuple(spec_cls.variables)
    data_vars = {
        name: xr.DataArray(float(init_values.get(name, 0.0)))
        for name in all_names
    }
    ds = xr.Dataset(data_vars)
    ds.attrs["params"] = spec_cls.params
    ds.attrs["variables"] = spec_cls.variables
    ds.attrs["advance"] = spec_cls.advance  # <-- behaviour varies here
    ds.attrs["calculate"] = spec_cls.calculate  # <-- and here
    return ds


# ---------------------------------------------------------------------------
# 3.  Two concrete spec classes -- same accessor, different behaviour
# ---------------------------------------------------------------------------


class Decay:
    """State decays exponentially each step."""

    params = ("rate",)
    variables = ("value", "value_prev")

    @staticmethod
    def advance(ds: xr.Dataset) -> None:
        ds["value_prev"].values[()] = ds["value"].values[()]

    @staticmethod
    def calculate(ds: xr.Dataset, dt: float) -> None:
        ds["value"].values[()] = ds["value_prev"].values[()] * np.exp(
            -float(ds["rate"]) * dt
        )


class Growth:
    """State grows linearly each step."""

    params = ("rate",)
    variables = ("value", "value_prev")

    @staticmethod
    def advance(ds: xr.Dataset) -> None:
        ds["value_prev"].values[()] = ds["value"].values[()]

    @staticmethod
    def calculate(ds: xr.Dataset, dt: float) -> None:
        ds["value"].values[()] = (
            ds["value_prev"].values[()] + float(ds["rate"]) * dt
        )


# ---------------------------------------------------------------------------
# 4.  Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dt = 1.0
    steps = 4

    decay = process_factory(Decay, rate=0.5, value=100.0, value_prev=100.0)
    growth = process_factory(Growth, rate=3.0, value=0.0, value_prev=0.0)

    print(f"{'step':>4}  {'decay.value':>12}  {'growth.value':>12}")
    print("-" * 34)
    for step in range(steps):
        decay.pws.advance()
        growth.pws.advance()
        decay.pws.calculate(dt)
        growth.pws.calculate(dt)
        print(
            f"{step + 1:>4}"
            f"  {float(decay['value']):>12.4f}"
            f"  {float(growth['value']):>12.4f}"
        )

    print()
    print(
        "decay  params:",
        decay.pws.params(),
        "variables:",
        decay.pws.variables(),
    )
    print(
        "growth params:",
        growth.pws.params(),
        "variables:",
        growth.pws.variables(),
    )
    print()
    print("Both datasets share the same PWSAccessor class.")
    print("Their *behaviour* differs only because ds.attrs['advance'] and")
    print(
        "ds.attrs['calculate'] point to different spec-class static methods."
    )
