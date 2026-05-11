# We can define sub-modules for the pws accessor too.
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


class Behaviour(ABC):
    """Abstract base class to handle required methods"""

    def __init__(self, obj):
        self._obj = obj

    @staticmethod
    def required_parameters() -> tuple:
        return ("rate",)

    @property
    @abstractmethod
    def variables(self) -> tuple:
        pass

    @abstractmethod
    def advance(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass

    @classmethod
    def from_dict(cls, **kwargs):
        data_vars = {
            name: xr.DataArray(float(kwargs.get(name, 0.0))) for name in kwargs
        }
        ds = xr.Dataset(data_vars)
        ds.attrs["behaviour_name"] = (
            cls.__name__
        )  # <-- the string registry key
        return ds


class Decay(Behaviour):
    """Define the decay type"""

    @property
    def variables(self):
        return ("value", "value_prev")

    def advance(self) -> None:
        self._obj["value_prev"].values[()] = self._obj["value"].values[()]

    def calculate(self, dt: float) -> None:
        self._obj["value"].values[()] *= np.exp(-float(self._obj["rate"]) * dt)


class Growth(Behaviour):
    """Define the growth type"""

    @property
    def variables(self):
        return ("value", "value_prev")

    def advance(self) -> None:
        self._obj["value_prev"].values[()] = self._obj["value"].values[()]

    def calculate(self, dt: float) -> None:
        self._obj["value"].values[()] += float(self._obj["rate"]) * dt


@xr.register_dataset_accessor("pws")
class PWSAccessor:
    # Behaviour subclasses available for construction and registry lookup
    Decay = Decay
    Growth = Growth

    def __init__(self, obj: xr.Dataset) -> None:
        self._obj = obj
        # Resolve the behaviour instance once, at accessor-creation time
        cls = getattr(PWSAccessor, obj.attrs["behaviour_name"])
        self._behaviour = cls(obj)

    def advance(self) -> None:
        self._behaviour.advance()

    def calculate(self, dt: float) -> None:
        self._behaviour.calculate(dt)

    # ---------- introspection ----------
    @property
    def params(self) -> tuple:
        return self._behaviour.required_parameters() if self._behaviour else ()

    @property
    def variables(self) -> tuple:
        return self._behaviour.variables if self._behaviour else ()


if __name__ == "__main__":
    dt = 1.0
    steps = 4

    decay = PWSAccessor.Decay.from_dict(
        rate=0.5, value=100.0, value_prev=100.0
    )
    growth = PWSAccessor.Growth.from_dict(rate=3.0, value=0.0, value_prev=0.0)

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
