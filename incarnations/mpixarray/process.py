"""
process.py
==========
The Process framework for incarnations/mpixarray: the DataArrayMeta field
descriptor and the Process ABC. Concrete processes (Upper, Lower, ...)
subclass Process in processes_concrete.py; the orchestrators (Model,
ModelMPI) live in model.py.

What is a Process?
------------------
A Process is a stateful, accessor-style view of its grid's ONE shared
dataset: it stores self._obj (the dataset) and exposes advance() and
calculate(dt, time) as instance methods. It does not own or build its
dataset -- the Model assembles one shared dataset per grid
(discretization) from its processes' field declarations and binds each
process to it directly (serial: Model._add_process_fields; MPI:
ModelMPI._build, where _obj is rebound to each streaming step).
Same-named variables are added to the grid dataset once, so
cross-process buffer sharing (param_shared_name, Upper.flow ->
Lower.flow) is structural -- the same named variable in the one dataset.

Fields are declared as class attributes with DataArrayMeta (kind:
parameter / input / mutable_input / variable); the introspection
classmethods (get_parameters, get_inputs, ...) read those declarations
and are what the Model assembles from.

Heavy computation is delegated to a @staticmethod _calculate(...) that
takes raw numpy arrays -- no xarray overhead -- compiled with
@numba.njit (nopython: uncompilable code raises, no silent object-mode
fallback). Convention: the output buffer(s) come first and are written
IN PLACE; _calculate returns nothing and allocates nothing per step
(numba fuses the array expressions, eliminating the temporaries).

Run tests with: pytest tests/ -v
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import xarray as xr

from globals import Time

# ---------------------------------------------------------------------------
# DataArrayMeta -- field descriptor for Process subclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataArrayMeta:
    """Metadata for a DataArray field declared on a Process subclass.

    Example:
        class Upper(Process):
            param_up_0 = DataArrayMeta(kind="parameter", dims=("space",),
                                       dtype=np.float64)
            forcing_up  = DataArrayMeta(kind="input",     dims=("space",),
                                       dtype=np.float64)
            flow       = DataArrayMeta(kind="variable",  dims=("space",),
                                       dtype=np.float64, initial="flow_initial")
    """

    kind: Literal[
        "parameter",
        "parameter_derived",
        "input",
        "mutable_input",
        "variable",
    ]
    dims: tuple[str, ...]
    dtype: type
    description: str = ""
    initial: str | None = None  # kwarg name supplying initial values


# ---------------------------------------------------------------------------
# Spec class introspection helpers
# ---------------------------------------------------------------------------


def _proc_subclass_mro(cls: type) -> tuple[type, ...]:
    """Return the MRO of cls excluding Process, ABC, and object.

    Reversed so base class fields are yielded before subclass fields,
    giving consistent ordering when walking the hierarchy.
    """
    _exclude = {"Process", "ABC", "object"}
    return tuple(
        cc for cc in reversed(cls.__mro__) if cc.__name__ not in _exclude
    )


def _keys_of_kind(cls: type, kind: str) -> tuple[str, ...]:
    """Return field names declared with a given kind on a Process subclass.

    Walks the full MRO so fields declared on intermediate base classes
    are included.
    """
    return tuple(
        name
        for cc in _proc_subclass_mro(cls)
        for name, val in vars(cc).items()
        if isinstance(val, DataArrayMeta) and val.kind == kind
    )


def _dict_of_kind(cls: type, kind: str) -> dict[str, DataArrayMeta]:
    """Return {name: DataArrayMeta} for fields of a given kind on a Process subclass.

    Walks the full MRO so fields declared on intermediate base classes
    are included.
    """
    return {
        name: val
        for cc in _proc_subclass_mro(cls)
        for name, val in vars(cc).items()
        if isinstance(val, DataArrayMeta) and val.kind == kind
    }


# ---------------------------------------------------------------------------
# Process ABC
# ---------------------------------------------------------------------------

_FILL_VALUE: dict[type, object] = {
    np.float64: np.nan,
    # ints have no nan; use a glaringly-invalid sentinel
    np.int64: np.iinfo(np.int64).min,
}


class Process(ABC):
    """Accessor-style ABC: a view of its grid's shared dataset.

    Stores self._obj (the dataset, bound by the Model) and dispatches
    advance()/calculate(). Subclasses declare fields as DataArrayMeta
    class attributes and auto-register in Process._registry via
    __init_subclass__.

    Numba:
        Heavy inner computation should be delegated to a @staticmethod
        _calculate(...) receiving raw numpy arrays, decorated with
        @numba.njit. The output buffer(s) come first and are written in
        place (no return, no per-step allocation):

        class Upper(Process):
            @staticmethod
            @numba.njit
            def _calculate(flow, flow_previous, forcing):
                flow[:] = flow_previous * 0.95 + forcing

            def calculate(self, dt: np.float64, time: Time) -> None:
                self._calculate(
                    self._obj["flow"].values,
                    self._obj["flow_previous"].values,
                    self._obj["forcing_up"].values,
                )
    """

    # Registry of concrete subclasses by class name, populated by
    # __init_subclass__. Used by config.load_model_yaml to resolve
    # process classes named as strings in a yaml configuration (import
    # the defining module first -- importing registers). The other
    # anticipated consumer is restart/checkpoint rehydration
    # (serialized state can only record a process *name*).
    _registry: dict[str, type] = {}

    # Home grid (co-registration). None = the model's single/default grid; a
    # subclass may set a default, a process_dict entry can override it.
    discretization: str | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        Process._registry[cls.__name__] = cls

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def __getitem__(self, key: str) -> xr.DataArray:
        """Subscript delegates to the bound dataset (a process is a view of its
        grid's shared dataset)."""
        return self._obj[key]

    def initialize(self) -> None:
        """Per-process init hook (default: no-op). Called ONCE by the Model
        after binding, IC loading, and input validation, before the run
        loop. Compute `kind="parameter_derived"` fields in place here
        (e.g. Muskingum coefficients from mann_n + dis variables); the
        Model freezes them (read-only) after all hooks run. Contract:
        LOCAL (no collectives), reads params/dis vars off self._obj, no
        Time (construction, not runtime)."""

    @abstractmethod
    def advance(self) -> None:
        """Copy current state to *_previous variables for the next timestep."""

    @abstractmethod
    def calculate(self, dt: np.float64, time: Time) -> None:
        """Update state variables for one timestep of length dt."""

    # ------------------------------------------------------------------
    # Introspection -- reads field-kind metadata from the class definition
    # ------------------------------------------------------------------

    @classmethod
    def get_parameters(cls) -> tuple[str, ...]:
        return _keys_of_kind(cls, "parameter")

    @classmethod
    def get_parameters_derived(cls) -> dict[str, DataArrayMeta]:
        """Parameters COMPUTED by initialize() rather than supplied
        (read-only after; metas returned for allocation)."""
        return _dict_of_kind(cls, "parameter_derived")

    @classmethod
    def get_inputs(cls) -> tuple[str, ...]:
        return _keys_of_kind(cls, "input")

    @classmethod
    def get_mutable_inputs(cls) -> tuple[str, ...]:
        return _keys_of_kind(cls, "mutable_input")

    @classmethod
    def get_variables(cls) -> dict[str, DataArrayMeta]:
        return _dict_of_kind(cls, "variable")

    @classmethod
    def get_var_names(cls) -> tuple[str, ...]:
        return _keys_of_kind(cls, "variable")
