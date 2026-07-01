"""
model.py
========
Orchestrators for the mpixarray Process framework:

  - Model:    serial orchestrator -- loads data, builds each process via its
              Process.new() factory, wires dependencies (inputs by reference ->
              cross-process buffer sharing), runs the serial time loop, and
              writes output via the Output collector (one zarr store).
  - ModelMPI: MPI streaming orchestrator -- one space-decomposed dataset
              streamed over time (mpixarray). Subclasses Model but overrides
              __init__/run/finalize; there, buffer sharing is structural (all
              processes view one decomposed dataset).

Model is process-aware by design (it builds processes via Process.new() and
dispatches through the `pws` accessor); there is no separate process-agnostic
orchestrator layer.
"""

import pathlib as pl
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Literal, Union

import numpy as np
import xarray as xr

from data_io import Input, Output, open_xr
from discretization import Discretization
from globals import Time
from process import _dict_of_kind  # also registers the `pws` xr accessor

# Optional MPI support -- present when mpixarray is installed. The serial Model
# never needs it; ModelMPI does, and fails at construction if it is absent.
try:
    from mpixarray import open_dataset as mpi_open_dataset

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


class Model:
    """Serial simulation orchestrator for Process subclasses.

    Loads data, builds each process via its Process.new() factory, wires
    dependencies (inputs by reference -> cross-process buffer sharing), runs
    the serial time loop, and writes output via the Output collector (one zarr
    store) when the control dict requests it.

    Usage:
        with Model(process_dict, control) as model:
            model.run(dt, n_steps)
    """

    def __init__(
        self,
        process_dict: Dict[str, Any],
        control: Dict[str, Any],
        load_all: Union[bool, None] = None,
    ) -> None:
        self._passed_process_dict = process_dict
        self._process_dict = deepcopy(process_dict)
        # process name -> home grid (co-registration: process_dict override,
        # else the class default, else the single default grid "space").
        self._proc_grid: Dict[str, str] = {
            kk: self._resolve_grid(vv) for kk, vv in self._process_dict.items()
        }
        self._opened_files: List[Union[xr.DataArray, xr.Dataset]] = []
        self._finalized = False

        if load_all is None:
            self._load_all = control.get("load_all", False)
        else:
            self._load_all = load_all

        self._load_paths_to_data()

        self.model_dict: Dict[str, Any] = {}
        self.inputs_dict: Dict[str, Input] = {}
        self._initialize_inputs_and_proceses()
        del self._process_dict

        self._set_time()
        # one Discretization per distinct home grid. The Process framework's
        # spatial dim is "space" for every grid (different grids = different
        # datasets/sizes on that dim); the grid *name* is the dict key.
        grids = dict.fromkeys(self._proc_grid.values())
        self.discretizations = {g: Discretization(["space"]) for g in grids}

        self.current_time_index = np.array([0], dtype=np.int32)
        self.current_time = (
            np.array([self.times[0].values], dtype="datetime64[D]")
            if self.times is not None
            else np.array([0], dtype=np.int32)
        )

        # Optional output collector (serial path). Created when the control
        # dict requests output. The MPI path streams output via mpixarray and
        # overrides __init__, so it never reaches this block.
        self.output = None
        if "output_var_names" in control or "output_dir" in control:
            if (
                "output_var_names" not in control
                or "output_dir" not in control
            ):
                raise ValueError(
                    "output_var_names and output_dir must both be specified "
                    "in the control dict."
                )
            if "time_chunk_size" in control:
                time_chunk_size = control["time_chunk_size"]
            else:
                time_chunk_size = 365
                warnings.warn(
                    "time_chunk_size not specified in control dict, using "
                    "default value of 365.",
                    UserWarning,
                )
            self.output = Output(
                time_chunk_size=time_chunk_size,
                variable_names=control["output_var_names"],
                output_store=pl.Path(control["output_dir"]) / "output.zarr",
                n_times=self.ntime,
                time_values=self.times.values,
            )
            self.output.initialize_variable_tracking(self.model_dict)

    def _load_paths_to_data(self) -> None:
        """Convert file paths in process_dict to xarray objects."""
        shared = self._load_shared_data_files()
        for proc in self._process_dict.values():
            for key in proc:
                val = proc[key]
                if isinstance(val, pl.Path):
                    if val in shared:
                        proc[key] = shared[val]
                    else:
                        opened = open_xr(val, load=self._load_all)
                        proc[key] = opened
                        self._opened_files.append(opened)

    def _load_shared_data_files(
        self,
    ) -> Dict[pl.Path, Union[xr.DataArray, xr.Dataset]]:
        """Open repeated file paths once, return {path: xarray_obj}."""
        from collections import Counter

        flat_paths = [
            v
            for outer in self._process_dict.values()
            for v in outer.values()
            if isinstance(v, pl.Path)
        ]
        shared = {}
        for path, count in Counter(flat_paths).items():
            if count > 1:
                opened = open_xr(path, load=self._load_all)
                shared[path] = opened
                self._opened_files.append(opened)
        return shared

    def _resolve_grid(self, entry: Dict[str, Any]) -> str:
        """A process entry's home grid: process_dict override, else the class
        default (`Process.discretization`), else the single grid "space"."""
        grid = entry.get("discretization") or entry["class"].discretization
        return grid if grid is not None else "space"

    def _initialize_inputs_and_proceses(self) -> None:  # noqa: spelling
        """Build each process via its Process.new() factory; wire inputs by
        reference (preserving cross-process buffer sharing) and the upstream
        variables produced by preceding processes."""
        for kk, vv in self._process_dict.items():
            init_dict = {
                kkk: vvv
                for kkk, vvv in vv.items()
                if kkk not in ("class", "discretization")
            }

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

    def _set_time(self) -> None:
        """Set time dimensions from the first input's `time` dim-coordinate."""
        kk0 = list(self.inputs_dict.keys())[0]
        data = self.inputs_dict[kk0].data
        self.ntime = data.sizes["time"]
        self.times = data["time"]
        self.time_index = data["time"]
        self.time = Time(self.times)

    def get_preceeding_processes(self, proc_name: str) -> List[str]:
        """Return process names defined before proc_name."""
        preceding = []
        for pp in self._process_dict:
            if proc_name != pp:
                preceding.append(pp)
            else:
                return preceding
        raise ValueError("Unreachable.")

    def advance(self) -> None:
        """Advance all inputs and processes."""
        for ii in self.inputs_dict.values():
            ii.advance()
        for pp in self.model_dict.values():
            if isinstance(pp, xr.Dataset):
                pp.pws.advance()
            else:
                pp.advance()

    def calculate(self, dt: np.float64) -> None:
        """Calculate all processes."""
        for vv in self.model_dict.values():
            if isinstance(vv, xr.Dataset):
                vv.pws.calculate(dt, self.time)
            else:
                vv.calculate(dt, self.time)

    def run(self, dt: np.float64, n_steps: np.int32) -> None:
        """Run simulation for n_steps time steps."""
        if self._finalized:
            raise RuntimeError("Cannot run a finalized Model.")
        for tt in range(n_steps):
            self.current_time_index[0] = tt
            self.current_time[0] = self.times[tt].values
            self.time.set_index(tt)
            self.advance()
            self.calculate(dt=dt)
            if self.output is not None:
                self.output.collect_current_timestep(tt)

    def finalize(self) -> None:
        """Close all file handles opened during initialization."""
        if self._finalized:
            return
        for inp in self.inputs_dict.values():
            inp.close()
        for f in self._opened_files:
            f.close()
        if self.output is not None:
            self.output.finalize()
        self._finalized = True

    def __enter__(self) -> "Model":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> Literal[False]:
        self.finalize()
        return False


class ModelMPI(Model):
    """MPI streaming model: one space-decomposed dataset, time-streamed.

    Catches up to the mpixarray streaming API:

        open_dataset -> parallelize(dims=["space"]) -> set_streaming("time")
        -> open_writer -> declare buffers -> create -> iter_time -> write

    A *single* decomposed dataset (``ds_mpi_stream``) carries every process's
    state, parameters, per-step input buffers, and streaming outputs. Because
    there is one dataset, cross-process buffer sharing (param_common,
    Upper.flow -> Lower.flow) is *structural* -- the same named variable -- not
    emulated by hand and asserted with ``a.values is b.values``.

    The spatial decomposition now lives in ``Discretization``
    (discretization.py; serial gets a degenerate one). For now there is exactly
    one discretization ("space"); multiple discretizations are future work.

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
        self.model_dict: dict = {}  # proc_name -> bound Process instance
        self.inputs_dict: dict = {}  # unused: inputs stream from step.mpi.src
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
        self.time = Time(ds_input["time"])

        self.discretizations = {"space": Discretization(["space"], comm=comm)}
        ds_mpi = self.discretizations["space"].decompose(ds_input)
        comm = self.discretizations["space"].comm  # parallelize may refine
        self._comm = comm

        # ---- Stream over time. set_streaming drops the time-dimensioned
        #      input vars (served per step via step.mpi.src) and keeps the
        #      space-only static params + ICs as buffers on ds_mpi_stream. ----
        ds_mpi_stream = ds_mpi.mpi.set_streaming(
            "time", window=1, out_times=list(range(n_time))
        )

        # Realize the static (space-only) survivors -- params + ICs -- in
        # memory. Otherwise they stay lazy file-backed and each `.values`
        # re-reads, so buffer sharing (param_common is ...) would not hold.
        for name in list(ds_mpi_stream.data_vars):
            ds_mpi_stream[name] = ds_mpi_stream[name].load()

        proc_classes = self._proc_classes()

        # Taxonomy guard: a parameter on the model `time` axis is really an
        # input. A true time-varying parameter lives on its own axis (e.g.
        # `month`) and stays resident. See pws_phoenix/CLAUDE.md.
        for proc_name, cls in proc_classes.items():
            for pname, meta in _dict_of_kind(cls, "parameter").items():
                if "time" in meta.dims:
                    raise ValueError(
                        f"Parameter {pname!r} on process {proc_name!r} has "
                        f"the model 'time' dim {meta.dims}: a variable on "
                        "model time is an input, not a parameter. Use a "
                        "non-'time' axis (e.g. 'month') for a time-varying "
                        "parameter."
                    )

        # File-backed inputs that were dropped: refilled from src each step.
        # An input is file-backed (vs. produced upstream) if no process owns
        # a variable of that name.
        var_names = self._all_variable_names()
        file_input_names: list = []
        for cls in proc_classes.values():
            for ii in tuple(cls.get_inputs()) + tuple(
                cls.get_mutable_inputs()
            ):
                if ii not in var_names and ii not in file_input_names:
                    file_input_names.append(ii)
        self._file_input_names = file_input_names

        # ---- Open writer ----
        ds_mpi_stream.mpi.open_writer(output_file, comm=comm)

        # ---- Declare buffers: to_netcdf=False FIRST, to_netcdf=True LAST,
        #      then create() (mpixarray declaration-ordering gotcha). ----
        def _declare(name, dims, fill, to_netcdf):
            ds_mpi_stream.mpi[name] = {
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

        ds_mpi_stream.mpi.create(coords=None, data_vars=None)

        # ---- Load initial conditions into state buffers. ICs are the
        #      space-only vars that survived set_streaming (from the file). ----
        for cls in proc_classes.values():
            for name, meta in cls.get_variables().items():
                if (
                    meta.initial is not None
                    and meta.initial in ds_mpi_stream.data_vars
                ):
                    ds_mpi_stream[name].values[:] = ds_mpi_stream[
                        meta.initial
                    ].values

        self._ds_mpi_stream = ds_mpi_stream

        # ---- Bind processes to the shared dataset (no .pws: one dataset
        #      hosts many processes, so dispatch on instances directly). ----
        for proc_name, cls in proc_classes.items():
            self.model_dict[proc_name] = cls(ds_mpi_stream)

    # -- run ------------------------------------------------------------

    def run(self, dt: np.float64, n_steps: np.int32 | None = None) -> None:
        """Stream the time loop via iter_time(); n_steps is ignored (the
        streaming source defines the count)."""
        if self._finalized:
            raise RuntimeError("Cannot run a finalized Model.")
        ds_mpi_stream = self._ds_mpi_stream
        procs = list(self.model_dict.values())
        for tt, step in enumerate(ds_mpi_stream.mpi.iter_time()):
            self.time.set_index(tt)
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
                proc.calculate(dt, self.time)
            # Record state into the streaming output slab(s) and write.
            for src_name, buf in self._output_map.items():
                step[buf].values[0, :] = step[src_name].values[:]
                if step.mpi.is_output_step:
                    step[buf].mpi.write()

    # -- finalize -------------------------------------------------------

    def finalize(self) -> None:
        if self._finalized:
            return
        self._ds_mpi_stream.mpi.finalize()
        self._ds_input.mpi.finalize()
        self._finalized = True
