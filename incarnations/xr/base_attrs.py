"""
base_attrs.py
=============
Alternative Process implementation where a process IS an xr.Dataset,
with a .pws accessor providing advance(), calculate(), and introspection.

Design principles (compare with base.py):

  base.py                          base_attrs.py
  -------                          -------------
  Process has self.data (Dataset)  Process IS an xr.Dataset
  process.advance()                process.pws.advance()
  process.calculate(dt)            process.pws.calculate(dt)
  get_parameters() static method   param = parameter(...) class-level marker
  get_inputs() static method       inp   = input_var(...) class-level marker
  get_variables() static method    var   = variable(...)  class-level marker
  process["flow"]                  process["flow"]        (unchanged -- native)

Input, Output, Model are imported unchanged from base.py.

Run tests with: pytest tests/ -v
"""

import dataclasses
import pathlib as pl
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from base import Input, Model, Output, open_xr  # noqa: F401

# ---------------------------------------------------------------------------
# Field markers -- class-level declarations that replace static method dicts
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FieldMeta:
    """Metadata attached to a declared process field."""

    kind: Literal["parameter", "input", "mutable_input", "variable"]
    dims: Tuple[str, ...]  # JLM: what is the elipsis here?
    dtype: type
    description: str = ""
    initial: Optional[str] = None  # kwarg name supplying initial values


def parameter(
    dims: Tuple[str, ...],
    dtype: type,
    description: str = "",
) -> _FieldMeta:
    """Declare a static (non-time-varying) parameter field."""
    return _FieldMeta(
        kind="parameter", dims=dims, dtype=dtype, description=description
    )


def input_var(
    dims: Tuple[str, ...],
    dtype: type,
    description: str = "",
) -> _FieldMeta:
    """Declare a read-only time-varying input field."""
    return _FieldMeta(
        kind="input", dims=dims, dtype=dtype, description=description
    )


def mutable_input(
    dims: Tuple[str, ...],
    dtype: type,
    description: str = "",
) -> _FieldMeta:
    """Declare a read-write time-varying input field."""
    return _FieldMeta(
        kind="mutable_input", dims=dims, dtype=dtype, description=description
    )


def variable(
    dims: Tuple[str, ...],
    dtype: type,
    description: str = "",
    initial: Optional[str] = None,
) -> _FieldMeta:
    """Declare a public state variable (shareable downstream)."""
    return _FieldMeta(
        kind="variable",
        dims=dims,
        dtype=dtype,
        description=description,
        initial=initial,
    )


# ---------------------------------------------------------------------------
# Spec class introspection helpers
# ---------------------------------------------------------------------------


def _fields_of_kind(cls: type, kind: str) -> Tuple[str, ...]:
    """Return field names declared with a given kind on a spec class."""
    return tuple(
        name
        for name, val in vars(cls).items()
        if isinstance(val, _FieldMeta) and val.kind == kind
    )


def _fields_meta_of_kind(cls: type, kind: str) -> Dict[str, _FieldMeta]:
    """Return {name: _FieldMeta} for fields of a given kind on a spec class."""
    return {
        name: val
        for name, val in vars(cls).items()
        if isinstance(val, _FieldMeta) and val.kind == kind
    }


# ---------------------------------------------------------------------------
# .pws xarray accessor
# ---------------------------------------------------------------------------


@xr.register_dataset_accessor("pws")
class PWSAccessor:
    """Accessor providing process methods on a process xr.Dataset.

    Usage:
        ds.pws.advance()
        ds.pws.calculate(dt)
        ds.pws.get_parameters()        # -> Tuple[str, ...]
        ds.pws.get_inputs()            # -> Tuple[str, ...]
        ds.pws.get_mutable_inputs()    # -> Tuple[str, ...]
        ds.pws.get_variables()         # -> Dict[str, _FieldMeta]
        ds.pws.get_var_names()         # -> Tuple[str, ...]

    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def advance(self) -> None:
        """Advance process state to the next time step."""
        self._ds.attrs["_advance"](self._ds)

    def calculate(self, dt: np.float64) -> None:
        """Perform calculations for the current time step."""
        self._ds.attrs["_calculate"](self._ds, dt)

    def get_parameters(self) -> Tuple[str, ...]:
        return self._ds.attrs.get("_parameters", ())

    def get_inputs(self) -> Tuple[str, ...]:
        return self._ds.attrs.get("_inputs", ())

    def get_mutable_inputs(self) -> Tuple[str, ...]:
        return self._ds.attrs.get("_mutable_inputs", ())

    def get_variables(self) -> Dict[str, _FieldMeta]:
        return self._ds.attrs.get("_variables", {})

    def get_var_names(self) -> Tuple[str, ...]:
        return tuple(self._ds.attrs.get("_variables", {}).keys())


# ---------------------------------------------------------------------------
# Process factory
# ---------------------------------------------------------------------------

_FILL_VALUE: Dict[type, Any] = {np.float64: np.nan}


def _make_process(
    spec_cls: Any,
    parameters: Union[xr.Dataset, pl.Path],
    **kwargs: Union[xr.DataArray, xr.Dataset, Input, pl.Path],
) -> xr.Dataset:
    """Build an xr.Dataset process from a spec class and runtime data.

    Mirrors the logic of Process.__init__ in base.py:
      - Selectively loads needed parameters in-place on the shared Dataset
        so buffer identity is preserved across variable selection.
      - Wires inputs and mutable inputs from Input objects or DataArrays.
      - Initialises state variables from metadata, optionally from ICs.
      - Stores advance/calculate callables and field metadata in ds.attrs
        so the .pws accessor can dispatch without knowing the spec class.

    Args:
        spec_cls: A process spec class decorated with @process.
        parameters: Shared parameter Dataset (may be file-backed/lazy).
        **kwargs: Input objects, DataArrays, or IC DataArrays keyed by
                  field name.

    Returns:
        xr.Dataset with all process fields as data variables and process
        metadata/callables stored in ds.attrs.
    """
    # Resolve any file paths to xarray objects.  When _make_process is called
    # directly (without Model._load_paths_to_data), callers may pass pl.Path
    # values for parameters and initial-condition kwargs.
    if isinstance(parameters, pl.Path):
        parameters = xr.open_dataset(parameters)
    resolved: Dict[str, Union[xr.DataArray, xr.Dataset, Input]] = {
        kk: (
            open_xr(vv)  # type: ignore[arg-type]
            if isinstance(vv, pl.Path)
            else vv
        )
        for kk, vv in kwargs.items()
    }

    param_names = _fields_of_kind(spec_cls, "parameter")
    input_names = _fields_of_kind(spec_cls, "input")
    mutable_input_names = _fields_of_kind(spec_cls, "mutable_input")
    variable_metas = _fields_meta_of_kind(spec_cls, "variable")

    # Load only needed parameters in-place on the shared parent Dataset.
    # This preserves buffer identity when the Dataset is file-backed (lazy).
    # See mre_buffer_share_testing.py for the full explanation.
    for pp in param_names:
        parameters[pp].load()
    ds = parameters[list(param_names)]
    for pp in param_names:
        parameters[pp].values.flags.writeable = False

    # Wire inputs -- same logic as Process.__init__ in base.py.
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

    # Initialise state variables.
    sizes = ds.sizes
    for name, meta in variable_metas.items():
        shape = tuple(sizes[d] for d in meta.dims)
        da = xr.DataArray(
            data=np.full(shape, _FILL_VALUE[meta.dtype], dtype=meta.dtype),
            dims=meta.dims,
            attrs={"description": meta.description},
        )
        if meta.initial is not None and meta.initial in resolved:
            da[:] = resolved[meta.initial]
        ds[name] = da

    # Store metadata and callables for the .pws accessor.
    ds.attrs["_parameters"] = param_names
    ds.attrs["_inputs"] = input_names
    ds.attrs["_mutable_inputs"] = mutable_input_names
    ds.attrs["_variables"] = variable_metas
    ds.attrs["_advance"] = spec_cls.advance
    ds.attrs["_calculate"] = spec_cls.calculate

    return ds


# ---------------------------------------------------------------------------
# @process decorator
# ---------------------------------------------------------------------------


def process(cls: Any) -> Any:
    """Class decorator that turns a spec class into a Dataset factory.

    After decoration, calling UpperSpec(parameters=..., **kwargs) returns an
    xr.Dataset rather than an UpperSpec instance.  Class-level introspection
    methods (get_parameters, get_inputs, get_variables) are also attached so
    that Model from base.py can inspect the spec before instantiation.

    The decorated class is otherwise unchanged -- advance() and calculate()
    remain as static methods on the class and are stored in ds.attrs at
    factory time.

    Example:
        @process
        class Upper:
            param_up_0 = parameter(dims=("space",), dtype=np.float64)
            forcing_0  = input_var(dims=("space",),  dtype=np.float64)
            flow       = variable(dims=("space",),   dtype=np.float64,
                                  initial="flow_initial")

            @staticmethod
            def advance(ds: xr.Dataset) -> None:
                ds["flow_previous"].values[:] = ds["flow"].values

            @staticmethod
            def calculate(ds: xr.Dataset, dt: np.float64) -> None:
                ds["flow"].values[:] = (
                    ds["flow_previous"].values * 0.95
                    + ds["forcing_0"].values
                )
    """

    # Attach class-level introspection methods compatible with Model's
    # expectations (Model calls get_inputs(), get_variables(), etc. on the
    # class before instantiation).
    cls.get_parameters = classmethod(
        lambda c: _fields_of_kind(cls, "parameter")
    )
    cls.get_inputs = classmethod(lambda c: _fields_of_kind(cls, "input"))
    cls.get_mutable_inputs = classmethod(
        lambda c: _fields_of_kind(cls, "mutable_input")
    )
    cls.get_variables = classmethod(
        lambda c: _fields_meta_of_kind(cls, "variable")
    )
    cls.get_var_names = classmethod(
        lambda c: tuple(_fields_meta_of_kind(cls, "variable").keys())
    )

    # Override __new__ so that calling the class returns an xr.Dataset.
    # Python allows __new__ to return any object; if it is not an instance
    # of cls the returned object is used directly without calling __init__.
    def __new__(
        factory_cls: type,
        parameters: xr.Dataset,
        **kwargs: Union[xr.DataArray, Input],
    ) -> xr.Dataset:
        return _make_process(cls, parameters, **kwargs)

    cls.__new__ = __new__  # type: ignore[method-assign, assignment]

    return cls
