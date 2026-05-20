# PWS_Dataset_jlm2.py
# ====================
# Evolution of PWS_Dataset_jlm.py. Key changes:
#
#   jlm.py                               jlm2.py
#   ------                               -------
#   PWSAccessor named PWS                PWS (shorter)
#   dispatch via getattr(PWS, name)      dispatch via Behaviour._registry
#   Behaviour subclasses as PWS attrs    __init_subclass__ auto-registers
#     (manual maintenance)                 (automatic, scales to ~40 processes)
#   PWS attrs also for construction      PWS attrs kept for construction syntax
#   construction: PWS.Decay.from_dict()  also: xr.Dataset.pws.Decay.from_dict()
#
# Motivation: Polymorphism with the xarray accessor
# --------------------------------------------------
# xr.Dataset is a general-purpose container. In a hydrological model we
# have ~40 process types (Upper, Lower, Snowpack, ...), each with its own
# variables, parameters, and computation. The challenge is: how do we
# attach process-specific behaviour (advance, calculate) to a plain
# xr.Dataset without subclassing it (which xarray discourages)?
#
# The accessor pattern solves this. The order of events is:
#
#   1. @xr.register_dataset_accessor("pws") registers PWS once at import
#      time -- before any datasets exist.
#   2. Behaviour subclasses (Decay, Growth, ...) must be imported before
#      any dataset's .pws is accessed. Each import triggers
#      __init_subclass__, which populates Behaviour._registry.
#   3. Accessor instantiation is lazy: PWS(ds) is only called the first
#      time .pws is accessed on a specific dataset instance.
#   4. At that moment, ds.attrs["behaviour_name"] identifies the exact
#      Behaviour subclass in the registry. That subclass is instantiated
#      with ds and its advance() and calculate() methods are attached to
#      ds.pws. Every dataset self-configures its own accessor.
#
# What is a Behaviour?
# ---------------------
# A Behaviour is a stateful accessor-style object -- it stores self._obj
# (the dataset) and exposes advance() and calculate(dt) as instance methods.
# This mirrors the xarray accessor pattern and keeps call signatures clean.
#
# There is a contract between a Behaviour and the dataset it operates on:
# the dataset is built by that same Behaviour's from_dict() which guarantees
# the dataset has exactly the variables and parameters the methods expect.
# The accessor enforces the pairing at construction time via
# ds.attrs["behaviour_name"].
#
# Heavy computation is delegated to a @staticmethod _calculate(...) that
# takes raw numpy arrays -- no xarray overhead -- making it a natural
# target for @numba.jit(nopython=True). The _calculate mutates in-place.

from abc import ABC, abstractmethod

import numba  # type: ignore[import-not-found]
import numpy as np
import xarray as xr


class Behaviour(ABC):
    """Accessor-style ABC: stores self._obj and dispatches advance/calculate.
    Subclasses auto-register in _registry via __init_subclass__.
    """

    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        Behaviour._registry[cls.__name__] = cls

    def __init__(
        self,
        xarray_obj: xr.Dataset | xr.DataArray,  # | xr.DataTree
    ) -> None:
        self._obj = xarray_obj

    @staticmethod
    def required_parameters() -> tuple:
        return ("rate",)

    @property
    @abstractmethod
    def variables(self) -> tuple:
        pass

    @abstractmethod
    def advance(self) -> None:
        pass

    @abstractmethod
    def calculate(self, dt: float) -> None:
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

    def advance(self) -> None:
        self._obj["value_prev"].values[()] = self._obj["value"].values[()]  # type: ignore

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate(value: np.ndarray, rate: float, dt: float) -> None:
        value[()] *= np.exp(-rate * dt)  # type: ignore

    def calculate(self, dt: float) -> None:
        self._calculate(
            self._obj["value"].values, float(self._obj["rate"]), dt
        )


class Growth(Behaviour):
    @property
    def variables(self) -> tuple:
        return ("value", "value_prev")

    def advance(self) -> None:
        self._obj["value_prev"].values[()] = self._obj["value"].values[()]  # type: ignore

    @staticmethod
    @numba.jit(nopython=True)
    def _calculate(value: np.ndarray, rate: float, dt: float) -> None:
        value[()] += rate * dt  # type: ignore

    def calculate(self, dt: float) -> None:
        self._calculate(
            self._obj["value"].values, float(self._obj["rate"]), dt
        )


@xr.register_dataset_accessor("pws")
class PWS:
    # Construction syntax:
    #     Decay.from_dict(...)
    # or
    #     xr.Dataset.pws.Decay.from_dict(...)  # clear type of xr.Dataset
    # Dispatch is handled independently via Behaviour._registry (auto-populated
    # by __init_subclass__). These attrs are purely for initialization
    # syntax/convenience.
    Decay = Decay
    Growth = Growth

    def __init__(
        self,
        xarray_obj: xr.Dataset | xr.DataArray,  # | xr.DataTree
    ) -> None:
        self._obj = xarray_obj
        cls = Behaviour._registry[self._obj.attrs["behaviour_name"]]
        self._behaviour = cls(self._obj)

    def advance(self) -> None:
        self._behaviour.advance()

    def calculate(self, dt: float) -> None:
        self._behaviour.calculate(dt)

    @property
    def params(self) -> tuple:
        return self._behaviour.required_parameters()

    @property
    def variables(self) -> tuple:
        return self._behaviour.variables


if __name__ == "__main__":
    dt = 1.0
    steps = 4

    # I'd make these classes avail from a package level import of pywatershed
    # typically as pws, so these would similarly be pws.Decay or Decay.
    decay = Decay.from_dict(rate=0.5, value=100.0, value_prev=100.0)
    growth = Growth.from_dict(rate=3.0, value=0.0, value_prev=0.0)
    # decay = xr.Dataset.pws.Decay.from_dict(
    #     rate=0.5, value=100.0, value_prev=100.0
    # )
    # growth = xr.Dataset.pws.Growth.from_dict(
    #     rate=3.0, value=0.0, value_prev=0.0
    # )

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
