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
import warnings
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import xarray as xr
from base import Input, Model, Output, open_xr  # noqa: F401

# Optional MPI support -- present when mpixarray is installed.
try:
    from mpixarray import open_dataset as mpi_open_dataset

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

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
    return tuple(cc for cc in reversed(cls.__mro__) if cc.__name__ not in _exclude)


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
        """Build the xr.Dataset for this process (serial path).

        Parameters arrive as a shared Dataset (may be file-backed/lazy);
        inputs and ICs arrive as individual DataArrays or Input objects.
        Buffer sharing across processes is preserved by loading parameters
        in-place on the shared parent and wiring inputs by reference.

        The MPI path does NOT use new(): ModelMPI builds one space-decomposed
        streaming dataset and binds processes to it directly (cls(ds_mpi)),
        so buffer sharing there is structural rather than emulated.

        Args:
            parameters: Shared parameter Dataset (may be file-backed/lazy).
            **kwargs: Input objects, DataArrays, or IC DataArrays keyed by
                      field name.

        Returns:
            xr.Dataset with all process fields and ds.attrs["process_name"].
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

        # Wire inputs (by reference -- preserves cross-process buffer sharing).
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
        self._process = Process._registry[self._obj.attrs["process_name"]](self._obj)

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
# Model (serial) and ModelMPI
# ---------------------------------------------------------------------------


class ModelAttrs(Model):  # noqa: keep for backward compat
    """Serial model for Process subclasses from base_attrs2.py.

    Calls cls.new(**init_dict) for each process. MPI is not used.

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


class ModelMPI(ModelAttrs):
    """MPI streaming model: one space-decomposed dataset, time-streamed.

    Catches up to the mpixarray streaming API:

        open_dataset -> parallelize(dims=["space"]) -> set_streaming("time")
        -> open_writer -> declare buffers -> create -> iter_time -> write

    A *single* decomposed dataset (``ds_mpi``) carries every process's state,
    parameters, per-step input buffers, and streaming outputs. Because there
    is one dataset, cross-process buffer sharing (param_common, forcing_common,
    Upper.flow -> Lower.flow) is *structural* -- the same named variable -- not
    emulated by hand and asserted with ``a.values is b.values``.

    The single ``parallelize()`` call is the spatial decomposition: the seam
    where a Discretization will live. For now there is exactly one
    discretization ("space"); multiple discretizations are future work.

    control dict:
        input_file        single combined input source (forcings + time-varying
                          params as (time, space); static params + ICs as (space,))
        output_file       streamed output file
        output_var_names  state vars to stream to disk (see note below)

    process_dict carries only the classes and their order (order defines the
    Upper -> Lower dependency direction):
        {"upper": {"class": Upper}, "lower": {"class": Lower}}

    NOTE (mpixarray limitation): declaring a 2nd ``to_netcdf=True`` variable
    currently trips the "cannot pickle 'module' object" deepcopy (the first
    such declaration stamps the writer handle onto the shared coord attrs, and
    the next ``from_numpy`` deepcopies them). So Phase 1 streams ONE output var
    to disk; validate any others from final in-memory state. Revisit when that
    mpixarray deepcopy bug is fixed.

    Usage:
        with ModelMPI(process_dict, control) as model:
            model.run(dt)
    """

    def __init__(self, process_dict: dict, control: dict) -> None:
        self._process_dict = process_dict
        self._control = control
        self._finalized = False
        self.model_dict: dict = {}     # proc_name -> bound Process instance
        self.inputs_dict: dict = {}    # unused: inputs stream from step.mpi.src
        self._build()

    # -- introspection helpers ------------------------------------------

    def _proc_classes(self) -> dict:
        return {kk: vv["class"] for kk, vv in self._process_dict.items()}

    def _all_variable_names(self) -> set:
        names: set = set()
        for cls in self._proc_classes().values():
            names |= set(cls.get_var_names())
        return names

    # -- construction ---------------------------------------------------

    def _build(self) -> None:
        f64 = np.float64
        control = self._control
        input_file = str(control["input_file"])
        output_file = str(control["output_file"])
        out_var_names = list(control["output_var_names"])

        # ---- Discretization: open + decompose once over space ----
        ds_input, comm = mpi_open_dataset(input_file)
        self._ds_input = ds_input
        n_time = int(ds_input.sizes["time"])
        self._ntime = n_time

        ds_input_mpi, comm = ds_input.mpi.parallelize(
            dims=["space"], scheme="single", comm=comm
        )
        self._comm = comm

        # ---- Stream over time. set_streaming drops the time-dimensioned
        #      input vars (served per step via step.mpi.src) and keeps the
        #      space-only static params + ICs as local buffers on ds_mpi. ----
        ds_mpi = ds_input_mpi.mpi.set_streaming(
            "time", window=1, out_times=list(range(n_time))
        )

        # Realize the static (space-only) survivors -- params + ICs -- in
        # memory. Otherwise they stay lazy file-backed and each `.values`
        # re-reads, so buffer sharing (param_common is ...) would not hold.
        for name in list(ds_mpi.data_vars):
            ds_mpi[name] = ds_mpi[name].load()

        proc_classes = self._proc_classes()

        # Time-varying parameters don't fit the streaming dichotomy
        # (set_streaming drops time-dim vars; static params must be space-only).
        # Phase 1 drops them -- warn explicitly. See pws_phoenix/CLAUDE.md.
        for proc_name, cls in proc_classes.items():
            for pname, meta in _dict_of_kind(cls, "parameter").items():
                if "time" in meta.dims:
                    warnings.warn(
                        f"Time-varying parameter {pname!r} {meta.dims} on "
                        f"process {proc_name!r} is unsupported in the MPI "
                        f"streaming path (Phase 1) and is dropped.",
                        UserWarning,
                        stacklevel=2,
                    )

        # File-backed inputs that were dropped: refilled from src each step.
        # An input is file-backed (vs. produced upstream) if no process owns
        # a variable of that name.
        var_names = self._all_variable_names()
        file_input_names: list = []
        for cls in proc_classes.values():
            for ii in tuple(cls.get_inputs()) + tuple(cls.get_mutable_inputs()):
                if ii not in var_names and ii not in file_input_names:
                    file_input_names.append(ii)
        self._file_input_names = file_input_names

        # ---- Open writer ----
        ds_mpi.mpi.open_writer(output_file, comm=comm)

        # ---- Declare buffers: to_netcdf=False FIRST, to_netcdf=True LAST,
        #      then create() (mpixarray declaration-ordering gotcha). ----
        def _declare(name, dims, fill, to_netcdf):
            ds_mpi.mpi[name] = {
                "dims": dims,
                "dtype": f64,
                "fill_value": f64(fill),
                "to_netcdf": to_netcdf,
                "comm": comm,
            }

        declared: set = set()
        # (a) state variables -- one buffer per name across processes
        for cls in proc_classes.values():
            for name in cls.get_var_names():
                if name not in declared:
                    _declare(name, ("space",), np.nan, False)
                    declared.add(name)
        # (b) per-step input buffers (refilled from src)
        for name in file_input_names:
            if name not in declared:
                _declare(name, ("space",), np.nan, False)
                declared.add(name)
        # (c) streaming output buffers (to_netcdf=True) -- declared LAST
        output_map: dict = {}  # state var name -> on-disk output buffer name
        for name in out_var_names:
            buf = f"{name}_out"
            _declare(buf, ("time", "space"), 0.0, True)
            output_map[name] = buf
        self._output_map = output_map

        ds_mpi.mpi.create(coords=None, data_vars=None)

        # ---- Load initial conditions into state buffers. ICs are the
        #      space-only vars that survived set_streaming (from the file). ----
        for cls in proc_classes.values():
            for name, meta in cls.get_variables().items():
                if meta.initial is not None and meta.initial in ds_mpi.data_vars:
                    ds_mpi[name].values[:] = ds_mpi[meta.initial].values

        self._ds_mpi = ds_mpi

        # ---- Bind processes to the shared dataset (no .pws: one dataset
        #      hosts many processes, so dispatch on instances directly). ----
        for proc_name, cls in proc_classes.items():
            self.model_dict[proc_name] = cls(ds_mpi)

    # -- run ------------------------------------------------------------

    def run(self, dt: np.float64, n_steps: np.int32 | None = None) -> None:
        """Stream the time loop via iter_time(); n_steps is ignored (the
        streaming source defines the count)."""
        if self._finalized:
            raise RuntimeError("Cannot run a finalized Model.")
        ds_mpi = self._ds_mpi
        procs = list(self.model_dict.values())
        for step in ds_mpi.mpi.iter_time():
            src = step.mpi.src
            # Refill this step's input buffers from the source slab.
            for name in self._file_input_names:
                step[name].values[:] = src[name].values[0, :]
            # Bind processes to the current step (shares the persistent
            # state buffers, so the Markov chain carries across steps).
            for proc in procs:
                proc._obj = step
            for proc in procs:
                proc.advance()
            for proc in procs:
                proc.calculate(dt)
            # Record state into the streaming output slab(s) and write.
            for src_name, buf in self._output_map.items():
                step[buf].values[0, :] = step[src_name].values[:]
                if step.mpi.is_output_step:
                    step[buf].mpi.write()

    # -- finalize -------------------------------------------------------

    def finalize(self) -> None:
        if self._finalized:
            return
        self._ds_mpi.mpi.finalize()
        self._ds_input.mpi.finalize()
        self._finalized = True
