# PWS_Dataset_jlm2.py
# ====================
# Evolution of PWS_Dataset_jlm.py. Key changes:
#
#   jlm.py                             jlm2.py
#   ------                             -------
#   Behaviour.__init__ stores self._obj  No __init__ -- Behaviour is stateless
#   advance(self)                        advance(self, ds)
#   calculate(self, dt)                  calculate(self, ds, dt)
#   PWSAccessor stores self._behaviour   PWS stores self._ds + self._behaviour
#   self._behaviour.advance()            self._behaviour.advance(self._ds)
#
# What is a Behaviour?
# ---------------------
# A Behaviour subclass has no __init__ and stores no state -- it is a
# named set of methods. But it is not arbitrary: there is a contract
# between a Behaviour and the dataset it operates on. The dataset is
# built by that same Behaviour's from_dict() (or Process.new() in the
# full implementation), which guarantees the dataset has exactly the
# variables and parameters the methods expect. The accessor enforces
# the pairing at construction time via ds.attrs["behaviour_name"].
#
# So a Behaviour is best understood as a typed interface to a specific
# dataset shape. All state lives in the dataset; the Behaviour supplies
# the operations defined over that state. This separation also makes
# the methods natural targets for numba -- the inner _calculate
# staticmethod takes raw numpy arrays with no xarray overhead and
# is decorated with @numba.jit(nopython=True).

from abc import ABC, abstractmethod

import numba  # type: ignore[import-not-found]
import numpy as np
import xarray as xr


class Behaviour(ABC):
    """Pure strategy ABC -- no __init__, no stored state."""

    @staticmethod
    def required_parameters() -> tuple:
        return ("rate",)

    @property
    @abstractmethod
    def variables(self) -> tuple:
        pass

    @abstractmethod
    def advance(self, ds: xr.Dataset) -> None:
        pass

    @abstractmethod
    def calculate(self, ds: xr.Dataset, dt: float) -> None:
        pass

    @classmethod
    def from_dict(cls, **kwargs) -> xr.Dataset:
        data_vars = {
            name: xr.DataArray(float(kwargs.get(name, 0.0))) for name in kwargs
        }
        ds = xr.Dataset(data_vars)
        ds.attrs["behaviour_name"] = cls.__name__
        return ds


class Decay(Behaviour):
    @property
    def variables(self) -> tuple:
        return ("value", "value_prev")

    def advance(self, ds: xr.Dataset) -> None:
        ds["value_prev"].values[()] = ds["value"].values[()]

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate(value: np.ndarray, rate: float, dt: float) -> None:
        value[()] *= np.exp(-rate * dt)

    def calculate(self, ds: xr.Dataset, dt: float) -> None:
        self._calculate(ds["value"].values, float(ds["rate"]), dt)


class Growth(Behaviour):
    @property
    def variables(self) -> tuple:
        return ("value", "value_prev")

    def advance(self, ds: xr.Dataset) -> None:
        ds["value_prev"].values[()] = ds["value"].values[()]

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate(value: np.ndarray, rate: float, dt: float) -> None:
        value[()] += rate * dt

    def calculate(self, ds: xr.Dataset, dt: float) -> None:
        self._calculate(ds["value"].values, float(ds["rate"]), dt)


@xr.register_dataset_accessor("pws")
class PWS:
    # Behaviour subclasses available for construction and registry lookup
    Decay = Decay
    Growth = Growth

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds
        # the behavoir is "stateless" -- no ds stored there.
        self._behaviour = getattr(PWS, ds.attrs["behaviour_name"])()

    def advance(self) -> None:
        self._behaviour.advance(self._ds)

    def calculate(self, dt: float) -> None:
        self._behaviour.calculate(self._ds, dt)

    @property
    def params(self) -> tuple:
        return self._behaviour.required_parameters()

    @property
    def variables(self) -> tuple:
        return self._behaviour.variables


if __name__ == "__main__":
    dt = 1.0
    steps = 4

    decay = PWS.Decay.from_dict(rate=0.5, value=100.0, value_prev=100.0)
    growth = PWS.Growth.from_dict(rate=3.0, value=0.0, value_prev=0.0)

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
    print(f"decay behaviour:  {decay.attrs['behaviour_name']}")
    print(f"growth behaviour: {growth.attrs['behaviour_name']}")
    print(f"{decay.pws.variables=}")
    print(f"{growth.pws.variables=}")
