"""
model.py
========
Orchestrators for the mpixarray Process framework:

  - Model:    serial orchestrator -- loads data, assembles one shared dataset
              per discretization (grid) and binds each process to it directly
              (cross-process buffer sharing is structural -- the same named
              variable), runs the serial time loop, and writes output via the
              Output collector (one zarr store).
  - ModelMPI: MPI streaming orchestrator -- one space-decomposed dataset
              streamed over time (mpixarray). Subclasses Model but overrides
              __init__/run/finalize; there, buffer sharing is structural (all
              processes view one decomposed dataset).

Model is process-aware by design (it assembles each grid's shared dataset from
its processes' field specs and binds the process instances to it); there is no
separate process-agnostic orchestrator layer.
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
from process import _FILL_VALUE, _dict_of_kind

# Optional MPI support -- present when mpixarray is installed. The serial Model
# never needs it; ModelMPI does, and fails at construction if it is absent.
try:
    from mpixarray import open_dataset as mpi_open_dataset

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


class Model:
    """Serial simulation orchestrator for Process subclasses.

    Loads data, assembles one shared dataset per discretization (grid) and
    binds each process to it directly (cross-process buffer sharing is
    structural -- the same named variable), runs the serial time loop, and
    writes output via the Output collector (one zarr store) when the control
    dict requests it.

    Usage:
        with Model(process_dict, control) as model:
            model.run(dt, n_steps)
    """

    def __init__(
        self,
        process_dict: Dict[str, Any],
        control: Dict[str, Any],
        load_all: Union[bool, None] = None,
        maps: Union[Dict[str, Any], None] = None,
    ) -> None:
        self._passed_process_dict = process_dict
        self._process_dict = deepcopy(process_dict)
        # process name -> home grid (co-registration: process_dict override,
        # else the class default, else the single default grid "space").
        self._proc_grid: Dict[str, str] = {
            kk: self._resolve_grid(vv) for kk, vv in self._process_dict.items()
        }
        self.maps: Dict[str, Any] = maps or {}
        # one Discretization per distinct home grid; the grid's dict key IS its
        # real spatial dim name (1-D grids: grid identity == dim). The dis owns
        # the grid's shared dataset, assembled + attached during _initialize.
        grids = dict.fromkeys(self._proc_grid.values())
        self.discretizations: Dict[str, Discretization] = {
            g: Discretization([g]) for g in grids
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
        """Assemble ONE shared dataset per grid from its processes' field specs,
        then bind each process to it. Same-named vars are added once, so
        cross-process sharing (param_shared_name, Upper.flow -> Lower.flow) is
        structural. Each grid's Discretization owns the resulting dataset."""
        grid_procs: Dict[str, List[str]] = {}
        for kk in self._process_dict:
            grid_procs.setdefault(self._proc_grid[kk], []).append(kk)

        for grid, proc_names in grid_procs.items():
            grid_ds = xr.Dataset()
            for kk in proc_names:
                self._add_process_fields(kk, grid, grid_ds)
            self.discretizations[grid].dataset = grid_ds

        # Bind in process_dict order (assembly above groups by grid, but the
        # author's order IS the execution schedule).
        for kk in self._process_dict:
            grid_ds = self.discretizations[self._proc_grid[kk]].dataset
            self.model_dict[kk] = self._process_dict[kk]["class"](grid_ds)

        self._validate_inputs_resolved()
        self._resolve_maps()

    def _validate_inputs_resolved(self) -> None:
        """Every declared process input must be present on its grid's shared
        dataset after assembly -- supplied in the process_dict entry, produced
        by a process on the same grid, or fed by a registered Map. Fail fast
        here; a missing input would otherwise surface as a KeyError
        mid-calculate (or silently read a stale variable)."""
        for proc_name, proc in self.model_dict.items():
            grid = self._proc_grid[proc_name]
            grid_ds = self.discretizations[grid].dataset
            needed = proc.get_inputs() + proc.get_mutable_inputs()
            missing = [ii for ii in needed if ii not in grid_ds]
            if missing:
                raise ValueError(
                    f"process '{proc_name}' on grid '{grid}' has "
                    f"unresolved input(s) {missing}: each input must be "
                    "supplied in the process_dict entry, produced by a "
                    "process on the same grid, or fed by a registered Map "
                    f"targeting grid '{grid}' and that variable name."
                )

    def _add_process_fields(
        self, proc_name: str, grid: str, grid_ds: xr.Dataset
    ) -> None:
        """Add a process's params / inputs / state vars to its grid's shared
        dataset, skipping names already present (structural sharing)."""
        vv = self._process_dict[proc_name]
        cls = vv["class"]
        init_dict = {
            kkk: vvv
            for kkk, vvv in vv.items()
            if kkk not in ("class", "discretization")
        }

        # -- parameters: a Dataset (or a Path opened here), loaded once.
        # Required when the process declares any parameter. --
        parameters: xr.Dataset | pl.Path | None = init_dict.get("parameters")
        if isinstance(parameters, pl.Path):
            parameters = xr.open_dataset(parameters)
        for pp in cls.get_parameters():
            if pp in grid_ds:
                continue
            if parameters is None:
                raise ValueError(
                    f"process '{proc_name}' declares parameter '{pp}' but "
                    "no 'parameters' were supplied"
                )
            parameters[pp].load()
            grid_ds[pp] = parameters[pp]
            grid_ds[pp].values.flags.writeable = False

        # -- inputs (read-only + mutable) --
        inputs_req = cls.get_inputs()
        for ii in inputs_req + cls.get_mutable_inputs():
            if ii in grid_ds:
                continue  # produced/added on this grid -> structural share
            if ii in init_dict:
                if ii not in self.inputs_dict:
                    self.inputs_dict[ii] = Input(
                        init_dict[ii],
                        read_only=(ii in inputs_req),
                        load=self._load_all,
                    )
                grid_ds[ii] = self.inputs_dict[ii].current_values
            else:
                # cross-grid input -> the feeding Map's target buffer
                for mm in self.maps.values():
                    if mm.target_grid == grid and mm.target_var == ii:
                        grid_ds[ii] = mm.target_values
                        break

        # -- state variables: initialised (fill + optional initial), once.
        # A process declares its spatial dim as the placeholder "space"; bind
        # it to this grid's real dim (the grid key). Params/inputs already
        # arrive on the real dim, so only state vars need resolving. --
        real_dim = self.discretizations[grid].dims[0]
        for name, meta in cls.get_variables().items():
            if name in grid_ds:
                continue
            dims = tuple(real_dim if dd == "space" else dd for dd in meta.dims)
            shape = tuple(grid_ds.sizes[dd] for dd in dims)
            da = xr.DataArray(
                np.full(shape, _FILL_VALUE[meta.dtype], dtype=meta.dtype),
                dims=dims,
                attrs={"description": meta.description},
            )
            if meta.initial is not None and meta.initial in init_dict:
                da[:] = init_dict[meta.initial]
            grid_ds[name] = da

    def _resolve_maps(self) -> None:
        """Assign each Map to its FIRST consumer in the execution order and
        validate the one-pass schedule. A mapped value is a per-step constant
        after its single apply -- guaranteed statically here (one variable
        owner; all declared source-grid writers precede the first consumer),
        not by runtime tracking. Later consumers re-read the same target
        buffer. Raises ValueError on: unknown grids, missing source variable,
        weights shape vs grid sizes, an unused map, an ambiguous (multi-owner)
        source variable, or a writer ordered at/after the first consumer."""
        position = {kk: ii for ii, kk in enumerate(self.model_dict)}
        self._proc_maps: Dict[str, List[Any]] = {
            kk: [] for kk in self.model_dict
        }
        for map_name, mm in self.maps.items():
            for gg in (mm.source_grid, mm.target_grid):
                if gg not in self.discretizations:
                    raise ValueError(
                        f"Map '{map_name}': grid '{gg}' is not a grid of "
                        f"any process (grids: {list(self.discretizations)})."
                    )
            source_ds = self.discretizations[mm.source_grid].dataset
            target_ds = self.discretizations[mm.target_grid].dataset
            if mm.source_var not in source_ds:
                raise ValueError(
                    f"Map '{map_name}': source variable '{mm.source_var}' "
                    f"not found on grid '{mm.source_grid}'."
                )
            n_target = target_ds.sizes[mm.target_grid]
            n_source = source_ds.sizes[mm.source_grid]
            if mm.weights.shape != (n_target, n_source):
                raise ValueError(
                    f"Map '{map_name}': weights shape {mm.weights.shape} != "
                    f"(n_target, n_source) = ({n_target}, {n_source}) for "
                    f"'{mm.source_grid}' -> '{mm.target_grid}'."
                )
            if (
                mm.target_var not in target_ds
                or target_ds[mm.target_var].values
                is not mm.target_values.values
            ):
                raise ValueError(
                    f"Map '{map_name}' is unused: '{mm.target_var}' on grid "
                    f"'{mm.target_grid}' is not fed by this map (it was "
                    "supplied directly or produced on that grid)."
                )

            # The source variable's declared write set: its (single) owner
            # plus any mutable_input declarers, all on the source grid.
            owners = [
                kk
                for kk, proc in self.model_dict.items()
                if self._proc_grid[kk] == mm.source_grid
                and mm.source_var in proc.get_var_names()
            ]
            if len(owners) > 1:
                raise ValueError(
                    f"Map '{map_name}': source variable '{mm.source_var}' "
                    f"has multiple owners on grid '{mm.source_grid}' "
                    f"{owners}; the once-per-step apply needs one."
                )
            writers = owners + [
                kk
                for kk, proc in self.model_dict.items()
                if self._proc_grid[kk] == mm.source_grid
                and mm.source_var in proc.get_mutable_inputs()
            ]
            consumers = [
                kk
                for kk, proc in self.model_dict.items()
                if self._proc_grid[kk] == mm.target_grid
                and mm.target_var
                in proc.get_inputs() + proc.get_mutable_inputs()
            ]
            first_consumer = min(consumers, key=lambda kk: position[kk])
            late = [
                ww
                for ww in writers
                if position[ww] >= position[first_consumer]
            ]
            if late:
                raise ValueError(
                    f"Map '{map_name}': source-grid writer(s) {late} of "
                    f"'{mm.source_var}' run at or after the map's first "
                    f"consumer '{first_consumer}' -- a one-pass order must "
                    "finish computing (and mutating) a mapped variable "
                    "before it crosses the grid boundary."
                )
            self._proc_maps[first_consumer].append(mm)

    def _set_time(self) -> None:
        """Set time dimensions from the first input's `time` dim-coordinate."""
        kk0 = list(self.inputs_dict.keys())[0]
        data = self.inputs_dict[kk0].data
        self.ntime = data.sizes["time"]
        self.times = data["time"]
        self.time_index = data["time"]
        self.time = Time(self.times)

    def advance(self) -> None:
        """Advance all inputs and processes."""
        for ii in self.inputs_dict.values():
            ii.advance()
        for pp in self.model_dict.values():
            pp.advance()

    def calculate(self, dt: np.float64) -> None:
        """Calculate each process in order. A Map is applied exactly once per
        step, immediately before its first consumer (see _resolve_maps);
        later consumers re-read the same target buffer."""
        for proc_name, vv in self.model_dict.items():
            for mm in self._proc_maps[proc_name]:
                mm.apply(self.discretizations[mm.source_grid].dataset)
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
    there is one dataset, cross-process buffer sharing (param_shared_name,
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
        # re-reads, so buffer sharing (param_shared_name is ...) would not
        # hold.
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

        # Fail fast if a file-backed input is absent from the input file --
        # it would otherwise KeyError mid-run on the first src refill.
        missing = [
            name for name in file_input_names if name not in ds_input.data_vars
        ]
        if missing:
            raise ValueError(
                f"input(s) {missing} are not produced by any process and "
                f"were not found in input_file {input_file!r}."
            )

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

        # ---- Bind processes to the shared dataset (one dataset hosts
        #      many processes; dispatch on the instances directly). ----
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
