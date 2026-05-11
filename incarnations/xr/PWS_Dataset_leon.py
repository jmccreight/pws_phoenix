# mypy: ignore-errors
# We can define sub-modules for the pws accessor too.
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


class Behaviour(ABC):
    """Abstract base class to handle required methods"""

    def __init__(self, obj):
        self._obj = obj

    @staticmethod
    def required_parameters(self):
        return ("rate",)

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def advance(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass


class Decay(Behaviour):
    """Define the decay type"""

    @property
    def variables(self):
        return ("value", "value_prev")

    def advance(self) -> None:
        self._obj["value_prev"].values[()] = self._obj["value"].values[()]

    def calculate(self, dt: float) -> None:
        # Do you need 2 states here? k and k+1?
        self._obj["value"].values[()] *= np.exp(-float(self._obj["rate"]) * dt)


class Growth(Behaviour):
    """Define the growth type"""

    @property
    def variables(self):
        return ("value", "value_prev")

    def advance(self) -> None:
        self._obj["value_prev"].values[()] = self._obj["value"].values[()]

    def calculate(self, dt: float) -> None:
        # Do you need 2 states here? k and k+1?
        self._obj["value"].values[()] += float(self._obj["rate"]) * dt


@xr.register_dataset_accessor("pws")
class PWSAccessor:
    def __init__(self, obj: xr.Dataset) -> None:
        self._obj = obj
        self.decay = Decay(self._obj)
        self.grow = Growth(self._obj)

        # ---------- introspection (stored as plain tuples in attrs) ----------

    def params(self) -> tuple:
        return self._ds.attrs.get("params", ())

    def variables(self) -> tuple:
        return self._ds.attrs.get("variables", ())

    @classmethod
    def from_dict(cls, **kwargs):
        data_vars = {
            name: xr.DataArray(float(kwargs.get(name, 0.0))) for name in kwargs
        }
        out = cls(xr.Dataset(data_vars))
        return out._obj


if __name__ == "__main__":
    dt = 1.0
    steps = 4

    # these should be decay and growth and have unique advance and calculate
    # methods.
    thing_1 = xr.Dataset.pws.from_dict(rate=0.5, value=100.0, value_prev=100.0)
    thing_2 = xr.Dataset.pws.from_dict(rate=3.0, value=0.0, value_prev=0.0)

    print(f"{'step':>4}  {'decay.value':>12}  {'growth.value':>12}")
    print("-" * 34)
    for step in range(steps):
        # decay or grow can be called on either thing_1 or thing_2
        thing_1.pws.decay.advance()
        thing_2.pws.grow.advance()
        thing_1.pws.decay.calculate(dt)
        thing_2.pws.grow.calculate(dt)
        print(
            f"{step + 1:>4}"
            f"  {float(thing_1['value']):>12.4f}"
            f"  {float(thing_2['value']):>12.4f}"
        )

    print()

    print(f"{thing_1=}")
    print(f"{thing_2=}")
