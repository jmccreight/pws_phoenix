"""
base_attrs2.py
==============
Revised Process/accessor design. Compared with base_attrs.py:

  base_attrs.py                      base_attrs2.py
  -------------                      --------------
  PWSAccessor dispatches via         PWS accessor dispatches via
    ds.attrs["advance"](ds)            self._process.advance()
    ds.attrs["calculate"](ds, dt)      self._process.calculate(dt)
  advance/calculate are staticmethods  advance/calculate are instance methods
  callables stored in ds.attrs       only ds.attrs["process_name"] (str) stored
  standalone _make_process() fn      Process.new() classmethod
  @process decorator                 Process.__init_subclass__ auto-registers
  PWS._registry manual               Process._registry automatic

Motivation: Polymorphism with the xarray accessor
--------------------------------------------------
xr.Dataset is a general-purpose container. In pws_phoenix we have ~40
process types (Upper, Lower, Snowpack, ...), each with its own variables,
parameters, and computation. The challenge: how do we attach
process-specific behaviour (advance, calculate) to a plain xr.Dataset
without subclassing it (which xarray discourages)?

The accessor pattern can solve this. The order of events is:

  1. @xr.register_dataset_accessor("pws") registers PWS once at import
     time -- before any datasets exist.
  2. Process subclasses must be imported before any dataset's .pws is
     accessed. Each import triggers __init_subclass__, which populates
     Process._registry automatically.
  3. Accessor instantiation is lazy: PWS(ds) is only called the first
     time .pws is accessed on a specific dataset instance.
  4. At that moment, ds.attrs["process_name"] identifies the exact
     Process subclass in the registry. That subclass is instantiated
     with ds and its advance() and calculate() methods are attached to
     ds.pws. Every dataset self-configures its own accessor.

What is a Process?
------------------
A Process is a stateful accessor-style object -- it stores self._obj
(the dataset) and exposes advance() and calculate(dt) as instance methods.
This mirrors the xarray accessor pattern and keeps call signatures clean.

There is a contract between a Process and the dataset it operates on:
the dataset is built by that same Process's new() classmethod, which
guarantees the dataset has exactly the variables and parameters the
methods expect. The accessor enforces the pairing at construction time
via ds.attrs["process_name"].

Heavy computation is delegated to a @staticmethod _calculate(...) that
takes raw numpy arrays -- no xarray overhead -- making it a natural
target for @numba.jit(nopython=True).

Key design notes:
  - The PWS accessor is registered on xr.Dataset only. DataArray support
    is deferred for a future revision.
  - No Python callables are stored in ds.attrs. Only the string
    ds.attrs["process_name"] is stored.

Run tests with: pytest tests/ -v
"""

import dataclasses
import pathlib as pl
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import xarray as xr
from base import Input, Model, Output, open_xr  # noqa: F401

# ---------------------------------------------------------------------------
# DataArrayMeta -- unchanged from base_attrs.py
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataArrayMeta:
    """Metadata for a DataArray field declared on a Process subclass.

    Example:
        class Upper(Process):
            param_up_0 = DataArrayMeta(kind="parameter", dims=("space",),
                                       dtype=np.float64)
            forcing_0  = DataArrayMeta(kind="input",     dims=("space",),
                                       dtype=np.float64)
            flow       = DataArrayMeta(kind="variable",  dims=("space",),
                                       dtype=np.float64, initial="flow_initial")
    """

    kind: Literal["parameter", "input", "mutable_input", "variable"]
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

_FILL_VALUE: dict[type, object] = {np.float64: np.nan}
DataArrayMetaDict = dict[str, DataArrayMeta]


class Process(ABC):
    """Accessor-style ABC: stores self._obj and dispatches advance/calculate.
    Subclasses auto-register in Process._registry via __init_subclass__.

    Construction:
        Call the classmethod new() on the concrete subclass to build the
        xr.Dataset, then access .pws to get the configured accessor:

        ds = Upper.new(parameters=..., forcing_0=..., flow_initial=...)
        ds.pws.advance()
        ds.pws.calculate(dt)

    Numba:
        Heavy inner computation should be delegated to a @staticmethod
        _calculate(...) receiving raw numpy arrays, decorated with
        @numba.jit(nopython=True):

        class Upper(Process):
            @staticmethod
            @numba.jit(nopython=True)
            def _calculate(flow_prev, forcing):
                flow_prev[:] *= 0.95
                flow_prev[:] += forcing

            def calculate(self, dt: np.float64) -> None:
                self._calculate(
                    self._obj["flow_previous"].values,
                    self._obj["forcing_0"].values,
                )
    """

    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        Process._registry[cls.__name__] = cls

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    @classmethod
    def new(
        cls,
        parameters: xr.Dataset | pl.Path,
        **kwargs: xr.DataArray | Input | pl.Path,
    ) -> xr.Dataset:
        """Build the xr.Dataset for this process.

        Parameters arrive as a shared Dataset (may be file-backed/lazy);
        inputs and ICs arrive as individual DataArrays or Input objects.
        See base_attrs.py _make_process docstring for the full design note
        on why parameters and inputs are supplied differently.

        Args:
            parameters: Shared parameter Dataset (may be file-backed/lazy).
            **kwargs: Input objects, DataArrays, or IC DataArrays keyed by
                      field name.

        Returns:
            xr.Dataset with all process fields as data variables and
            ds.attrs["process_name"] set to cls.__name__.
        """
        if isinstance(parameters, pl.Path):
            parameters = xr.open_dataset(parameters)
        resolved: dict[str, xr.DataArray | xr.Dataset | Input] = {
            kk: (open_xr(vv) if isinstance(vv, pl.Path) else vv)
            for kk, vv in kwargs.items()
        }

        param_names = _keys_of_kind(cls, "parameter")
        input_names = _keys_of_kind(cls, "input")
        mutable_input_names = _keys_of_kind(cls, "mutable_input")
        variable_meta_dict: DataArrayMetaDict = _dict_of_kind(cls, "variable")

        # Load only needed parameters in-place on the shared parent Dataset.
        # Preserves buffer identity when the Dataset is file-backed (lazy).
        for pp in param_names:
            parameters[pp].load()
        ds = parameters[list(param_names)]
        for pp in param_names:
            parameters[pp].values.flags.writeable = False

        # Wire inputs.
        for ii in input_names:
            inp = resolved[ii]
            if isinstance(inp, Input):
                ds[ii] = inp.current_values
                assert ds[ii].values is inp.current_values.values
            else:
                ds[ii] = inp

        for oo in mutable_input_names:
            inp_mut = resolved[oo]
            if isinstance(inp_mut, Input):
                ds[oo] = inp_mut.current_values

        # Initialize state variables.
        sizes = ds.sizes
        for name, meta in variable_meta_dict.items():
            shape = tuple(sizes[d] for d in meta.dims)
            da = xr.DataArray(
                data=np.full(shape, _FILL_VALUE[meta.dtype], dtype=meta.dtype),
                dims=meta.dims,
                attrs={"description": meta.description},
            )
            if meta.initial is not None and meta.initial in resolved:
                da[:] = resolved[meta.initial]
            ds[name] = da

        # Store field-kind metadata as plain strings/tuples (no callables).
        ds.attrs["process_name"] = cls.__name__
        ds.attrs["get_parameters"] = param_names
        ds.attrs["get_inputs"] = input_names
        ds.attrs["get_mutable_inputs"] = mutable_input_names
        ds.attrs["get_var_names"] = tuple(variable_meta_dict.keys())
        # Note: get_variables (Dict[str, DataArrayMeta]) is not stored in
        # attrs because dicts with non-scalar values don't survive NetCDF
        # round-trips. Use ds.pws.get_variables() instead.

        return ds

    @abstractmethod
    def advance(self) -> None:
        """Copy current state to *_previous variables for the next timestep."""

    @abstractmethod
    def calculate(self, dt: np.float64) -> None:
        """Update state variables for one timestep of length dt."""

    # ------------------------------------------------------------------
    # Introspection -- reads field-kind metadata from the class definition
    # ------------------------------------------------------------------

    @classmethod
    def get_parameters(cls) -> tuple[str, ...]:
        return _keys_of_kind(cls, "parameter")

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


# ---------------------------------------------------------------------------
# PWS accessor
# ---------------------------------------------------------------------------
# NOTE: Registered on xr.Dataset only. xr.DataArray support is deferred --
# the per-variable accessor use-case needs further design thought.


@xr.register_dataset_accessor("pws")
class PWS:
    """Accessor providing process methods on a process xr.Dataset.

    Dispatch is resolved at accessor-creation time by reading
    ds.attrs["process_name"] and looking up the corresponding Process
    subclass in Process._registry. No callables are stored in ds.attrs.

    Process subclasses auto-register via __init_subclass__ when imported.
    Class attributes on PWS (Upper, Lower, ...) are provided for
    convenient construction syntax:

        Upper.new(parameters=..., **kwargs)
        xr.Dataset.pws.Upper.new(parameters=..., **kwargs)

    Usage:
        ds = Upper.new(parameters=..., **kwargs)
        ds.pws.advance()
        ds.pws.calculate(dt)
        ds.pws.get_parameters()     # -> tuple[str, ...]
        ds.pws.get_inputs()         # -> tuple[str, ...]
        ds.pws.get_mutable_inputs() # -> tuple[str, ...]
        ds.pws.get_variables()      # -> dict[str, DataArrayMeta]
        ds.pws.get_var_names()      # -> tuple[str, ...]
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj
        self._process = Process._registry[self._obj.attrs["process_name"]](
            self._obj
        )

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        """Advance process state to the next timestep."""
        self._process.advance()

    def calculate(self, dt: np.float64) -> None:
        """Perform calculations for the current timestep."""
        self._process.calculate(dt)

    # ------------------------------------------------------------------
    # Introspection -- delegates to the Process subclass classmethods
    # ------------------------------------------------------------------

    def get_parameters(self) -> tuple[str, ...]:
        return self._process.get_parameters()

    def get_inputs(self) -> tuple[str, ...]:
        return self._process.get_inputs()

    def get_mutable_inputs(self) -> tuple[str, ...]:
        return self._process.get_mutable_inputs()

    def get_variables(self) -> dict[str, DataArrayMeta]:
        return self._process.get_variables()

    def get_var_names(self) -> tuple[str, ...]:
        return self._process.get_var_names()


# ---------------------------------------------------------------------------
# ModelAttrs
# ---------------------------------------------------------------------------


class ModelAttrs(Model):
    """Model for use with Process subclasses from base_attrs2.py.

    Overrides _initialize_inputs_and_proceses to call cls.new(**init_dict)
    when the process class has a .new() classmethod (i.e. is a Process
    subclass from base_attrs2.py), falling back to cls(**init_dict) for
    old-style Process subclasses from base.py.

    Usage:
        with ModelAttrs(process_dict, control) as model:
            model.run(dt, n_steps)
    """

    def _initialize_inputs_and_proceses(self) -> None:  # noqa: spelling
        """Like Model._initialize_inputs_and_proceses but dispatches via
        cls.new() for Process subclasses that define it."""
        for kk, vv in self._process_dict.items():
            init_dict = {kkk: vvv for kkk, vvv in vv.items() if kkk != "class"}

            inputs_req = vv["class"].get_inputs()
            input_outputs_req = vv["class"].get_mutable_inputs()
            all_inputs = inputs_req + input_outputs_req

            for ii in all_inputs:
                if ii in init_dict.keys():
                    data_or_file = init_dict[ii]
                    if ii in inputs_req:
                        read_only = True
                    else:
                        raise ValueError("This should not happen from file.")
                    if ii not in self.inputs_dict.keys():
                        init_dict[ii] = Input(
                            data_or_file,
                            read_only=read_only,
                            load=self._load_all,
                        )
                        self.inputs_dict[ii] = init_dict[ii]
                        assert init_dict[ii].data is self.inputs_dict[ii].data
                        del data_or_file
                    else:
                        init_dict[ii] = self.inputs_dict[ii]
                else:
                    for pp in self.get_preceeding_processes(kk):
                        proc = self.model_dict[pp]
                        if isinstance(proc, xr.Dataset):
                            var_names = proc.pws.get_var_names()  # type: ignore[attr-defined]
                        else:
                            var_names = proc.get_variables()
                        if ii in var_names:
                            init_dict[ii] = self.model_dict[pp][ii]

            cls = vv["class"]
            self.model_dict[kk] = cls.new(**init_dict)

        return
