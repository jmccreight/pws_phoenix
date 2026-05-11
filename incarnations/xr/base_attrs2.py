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
  @process decorator                 PWS._registry populated explicitly in
                                       processes_attrs2.py

Key design notes:
  - The PWS accessor is registered on xr.Dataset only. DataArray support
    is deferred for a future revision.
  - Process subclasses are strategy objects: they hold no data of their own.
    self._obj is the xr.Dataset they operate on, set at accessor-creation time
    by a plain Process.__init__ call -- no __new__ tricks needed.
  - Construction of the dataset is via Process.new(parameters, **kwargs),
    a classmethod on the ABC. cls is the concrete subclass so DataArrayMeta
    introspection finds the right fields automatically.
  - advance() and calculate() are instance methods on the Process ABC.
    Subclasses should delegate heavy computation to a @staticmethod
    _calculate(...) that takes raw numpy arrays, making it a natural
    target for @numba.jit without xarray overhead.
  - No Python callables are stored in ds.attrs. Only the string
    ds.attrs["process_name"] is stored.
  - PWS._registry is populated explicitly at the bottom of
    processes_attrs2.py after the subclasses are defined.

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
# Spec class introspection helpers -- unchanged from base_attrs.py
# ---------------------------------------------------------------------------


def _keys_of_kind(cls: type, kind: str) -> tuple[str, ...]:
    """Return field names declared with a given kind on a Process subclass."""
    return tuple(
        name
        for name, val in vars(cls).items()
        if isinstance(val, DataArrayMeta) and val.kind == kind
    )


def _dict_of_kind(cls: type, kind: str) -> dict[str, DataArrayMeta]:
    """Return {name: DataArrayMeta} for fields of a given kind on a Process subclass."""
    return {
        name: val
        for name, val in vars(cls).items()
        if isinstance(val, DataArrayMeta) and val.kind == kind
    }


# ---------------------------------------------------------------------------
# Process ABC
# ---------------------------------------------------------------------------

_FILL_VALUE: dict[type, object] = {np.float64: np.nan}
DataArrayMetaDict = dict[str, DataArrayMeta]


class Process(ABC):
    """Abstract base class for all pws_phoenix process implementations.

    A Process instance is a pure strategy object -- it holds no data.
    The dataset is passed explicitly to advance() and calculate() each
    call, making the data flow visible and the object trivially stateless.

    Construction:
        Call the classmethod Process.new() (on the concrete subclass) to
        build the xr.Dataset.

        ds = Upper.new(parameters=..., forcing_0=..., flow_initial=...)
        ds.pws.advance()
        ds.pws.calculate(dt)

    Numba:
        Heavy inner computation should be delegated to a @staticmethod
        _calculate(...) receiving raw numpy arrays, which can then be
        decorated with @numba.jit:

        class Upper(Process):
            @staticmethod
            def _calculate(flow_prev, forcing, dt):
                # @numba.jit goes here -- pure numpy, no xarray
                return flow_prev * 0.95 + forcing

            def calculate(self, ds: xr.Dataset, dt: np.float64) -> None:
                ds["flow"].values[:] = self._calculate(
                    ds["flow_previous"].values,
                    ds["forcing_0"].values,
                    dt,
                )
    """

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
    def advance(self, ds: xr.Dataset) -> None:
        """Copy current state to *_previous variables for the next timestep."""

    @abstractmethod
    def calculate(self, ds: xr.Dataset, dt: np.float64) -> None:
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
    subclass in PWS._registry. No callables are stored in ds.attrs.

    PWS._registry is populated explicitly in processes_attrs2.py after
    the Process subclasses are defined:

        PWS._registry["Upper"] = Upper
        PWS._registry["Lower"] = Lower

    Usage:
        ds = Upper.new(parameters=..., **kwargs)
        ds.pws.advance()
        ds.pws.calculate(dt)
        ds.pws.get_parameters()     # -> Tuple[str, ...]
        ds.pws.get_inputs()         # -> Tuple[str, ...]
        ds.pws.get_mutable_inputs() # -> Tuple[str, ...]
        ds.pws.get_variables()      # -> Dict[str, DataArrayMeta]
        ds.pws.get_var_names()      # -> Tuple[str, ...]
    """

    _registry: dict[str, type] = {}  # process_name -> Process subclass

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds
        self._process = PWS._registry[ds.attrs["process_name"]]()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def advance(self) -> None:
        """Advance process state to the next timestep."""
        self._process.advance(self._ds)

    def calculate(self, dt: np.float64) -> None:
        """Perform calculations for the current timestep."""
        self._process.calculate(self._ds, dt)

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
# ModelAttrs -- unchanged pass-through from base_attrs.py
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
            if hasattr(cls, "new") and callable(cls.new):
                self.model_dict[kk] = cls.new(**init_dict)
            else:
                self.model_dict[kk] = cls(**init_dict)

        return
