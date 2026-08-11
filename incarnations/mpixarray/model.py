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

    Restart: ``model.write_restart(dir)`` after a run saves the
    prognostic state (the ``DataArrayMeta(restart=True)`` variables +
    any ``get_restart_state()`` hook state) to self-locating per-grid
    files; ``control["restart_read"] = dir`` warm-starts a new model
    from them, resuming at the step after the saved time. USER
    CONVENIENCE BY DESIGN: a warm start takes the SAME (superset)
    input files as the original run -- inputs need only COVER the
    restart time; the model indexes in (serial) or time-slices the
    stream (MPI), so no restart-specific input preparation is ever
    needed. Restart files are format-identical across serial and MPI:
    either can warm-start from the other's.

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
        discretizations: Union[Dict[str, Discretization], None] = None,
    ) -> None:
        # Structure-only copy (NO deepcopy): the model may replace values
        # (Path -> opened object) without mutating the caller's dict, but the
        # data are shared by reference -- the caller's in-memory arrays ARE
        # the working buffers, and the model's read-only flags (parameters,
        # read-only inputs) apply to them. See "Prime directive: memory" in
        # pws_phoenix/CLAUDE.md.
        self._process_dict = {kk: dict(vv) for kk, vv in process_dict.items()}
        # process name -> home grid (co-registration: process_dict override,
        # else the class default, else the single default grid "space").
        self._proc_grid: Dict[str, str] = {
            kk: self._resolve_grid(vv) for kk, vv in self._process_dict.items()
        }
        self.maps: Dict[str, Any] = maps or {}
        # one Discretization per distinct home grid; the grid's dict key IS its
        # real spatial dim name (1-D grids: grid identity == dim). The dis owns
        # the grid's shared dataset, assembled + attached during _initialize.
        # Caller-supplied Discretizations carry dis-owned parameters
        # (dis_hru/dis_seg style); grids not supplied get the degenerate
        # default.
        grids = dict.fromkeys(self._proc_grid.values())
        provided = discretizations or {}
        unknown_grids = [gg for gg in provided if gg not in grids]
        if unknown_grids:
            raise ValueError(
                f"discretizations {unknown_grids} match no process's home "
                f"grid (grids in use: {list(grids)})."
            )
        self.discretizations: Dict[str, Discretization] = {
            gg: provided.get(gg) or Discretization([gg]) for gg in grids
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

        # Warm start (construction-time option): restore the flagged
        # prognostic state written by write_restart() and resume AFTER
        # the file's state time. Cold start: _start_index = 0.
        self._start_index = 0
        if control.get("restart_read") is not None:
            self._read_restart(pl.Path(control["restart_read"]))

        self.current_time_index = np.array([self._start_index], dtype=np.int32)
        self.current_time = np.array(
            [self.times[self._start_index].values], dtype="datetime64[D]"
        )

        # Optional output collector (serial path). Created when the control
        # dict requests output. The MPI path streams output via mpixarray and
        # overrides __init__, so it never reaches this block.
        self.output = None
        if "output_var_names" in control or "output_serial_zarr" in control:
            if (
                "output_var_names" not in control
                or "output_serial_zarr" not in control
            ):
                raise ValueError(
                    "output_var_names and output_serial_zarr must both be "
                    "specified in the control dict."
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
                output_store=pl.Path(control["output_serial_zarr"]),
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

    @staticmethod
    def _resolve_grid(entry: Dict[str, Any]) -> str:
        """A process entry's home grid: process_dict override, else the class
        default (`Process.discretization`), else the single grid "space"."""
        grid = entry.get("discretization") or entry["class"].discretization
        return grid if grid is not None else "space"

    @classmethod
    def input_spec(
        cls,
        process_dict: Dict[str, Any],
        maps: Union[Dict[str, Any], None] = None,
        include_optional: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """The model's INPUT CONTRACT, from declarations alone -- a DRY
        RUN of assembly's resolution logic: no data are loaded, only
        each entry's "class" and "discretization" are read (all other
        entry keys -- parameters, forcings -- are ignored), and map
        objects contribute only their grid/variable wiring.

        The contract is about what must be SUPPLIED, so the top level
        is required-vs-optional, then grid, then category:

        spec["required"][grid]:
        - "external_inputs": {name: {"consumers": [...]}} -- the
          INPUTS proper: time-varying data served in model-time
          lockstep (forcings/boundary data). A shared name is
          supplied ONCE and shared structurally.
        - "parameters": {name: [declaring processes]} -- AUTHORED
          static supplied data (the modeler's data/calibration).
          Where each name is PACKAGED (a grid's dis dataset --
          consulted first at assembly -- vs a process parameters
          dataset) is not resolved here: that is a data-preparation
          decision, and assembly accepts either. Note PRMS
          initial-condition parameters (`*_init*` by its own naming
          convention, consumed by initialize()) are in this list --
          they are ordinary supplied parameters.
        - "derivable_parameters": {name: {"processes",
          "derivation"}} -- REQUIRED at assembly exactly like
          "parameters", but obtainable by the stated derivation (a
          factory/formula over other supplied quantities; by
          convention the string NAMES ITS INPUTS). PATENTLY DISTINCT
          from ``kind="parameter_internal"``: internal parameters are
          computed by initialize() and NEVER supplied (reported under
          optional "internal_parameters"); derivable parameters ARE
          supplied -- you may just generate them instead of authoring
          them. (Under future compute-if-absent semantics this
          category may become optional-with-default.)
        - "initial_values": {init_name: {"variable", "process"}} --
          the DataArrayMeta ``initial=`` seams: OPTIONAL direct
          supplies of a state variable's starting values via a
          process_dict entry key.
        - "initial_state": {process: [variables]} -- the RESTARTABLE
          warm-start surface: every ``restart=True`` variable is a
          settable initial condition, supplied all-at-once via
          ``control["restart_read"]`` (see ``write_restart``).
          Optional supply, but part of the contract's surface.

        spec["maps"] (present when maps are given) -- REQUIRED supply
        too, kept beside "required" because that dict's keys are
        grids: each Map in the configuration implies ONE weights
        matrix. Per map name: {"source_grid", "target_grid",
        "source_var", "target_var", "weights_shape" (a
        ("n_<target_grid>", "n_<source_grid>") tuple -- rows map INTO
        the target), "derivation"} -- the derivation string (from
        ``Map(derivation=...)``) says how the matrix is obtained when
        the map's author knows (e.g. the translation layer derives
        NHM aggregation weights); None means the modeler supplies it.

        spec["optional"][grid] (informational; included only when
        ``include_optional=True``):
        - "internal_inputs": {name: {"producer", "consumers"}} --
          inputs satisfied on-grid by structural sharing (another
          process's variable or derived parameter, including
          prior-step back-edges; or a same-named supplied parameter).
        - "internal_parameters": {name: [processes]} -- computed by
          initialize(); never supplied.
        - "map_fed_inputs": {name: {"map", "source_grid",
          "source_var", "consumers"}} -- satisfied by a Map's target
          buffer.

        This is the single source of truth for "what does this model
        consume" -- consumers (the examples/00_input_contract.py notebook, the
        PRMS translation layer) should call it rather than maintain
        lists by hand. Schedule/order validity (writers before
        consumers, map placement) is NOT checked here; real assembly
        does that.
        """
        maps = maps or {}
        proc_grid = {
            kk: cls._resolve_grid(vv) for kk, vv in process_dict.items()
        }
        grid_procs: Dict[str, List[str]] = {}
        for kk in process_dict:
            grid_procs.setdefault(proc_grid[kk], []).append(kk)

        spec: Dict[str, Dict[str, Any]] = {"required": {}}
        if maps:
            spec["maps"] = {
                map_name: {
                    "source_grid": mm.source_grid,
                    "target_grid": mm.target_grid,
                    "source_var": mm.source_var,
                    "target_var": mm.target_var,
                    "weights_shape": (
                        f"n_{mm.target_grid}",
                        f"n_{mm.source_grid}",
                    ),
                    "derivation": getattr(mm, "derivation", None),
                }
                for map_name, mm in maps.items()
            }
        if include_optional:
            spec["optional"] = {}
        for grid, proc_names in grid_procs.items():
            parameters: Dict[str, List[str]] = {}
            derivable: Dict[str, Dict[str, Any]] = {}
            internal: Dict[str, List[str]] = {}
            var_owner: Dict[str, str] = {}
            initial_values: Dict[str, Any] = {}
            initial_state: Dict[str, List[str]] = {}
            for kk in proc_names:
                pcls = process_dict[kk]["class"]
                for name, meta in _dict_of_kind(pcls, "parameter").items():
                    if meta.derivation is not None:
                        entry = derivable.setdefault(
                            name,
                            {"processes": [], "derivation": meta.derivation},
                        )
                        entry["processes"].append(kk)
                    else:
                        parameters.setdefault(name, []).append(kk)
                for name in pcls.get_parameters_internal():
                    internal.setdefault(name, []).append(kk)
                for name, meta in pcls.get_variables().items():
                    var_owner.setdefault(name, kk)
                    if meta.initial is not None:
                        initial_values[meta.initial] = {
                            "variable": name,
                            "process": kk,
                        }
                restartable = list(pcls.get_restart_variables())
                if restartable:
                    initial_state[kk] = restartable
            # a name declared derivable by ANY process is derivable
            # (another process may declare it plain -- the derivation
            # knowledge counts once)
            for name in list(parameters):
                if name in derivable:
                    derivable[name]["processes"] = sorted(
                        set(derivable[name]["processes"])
                        | set(parameters.pop(name))
                    )

            consumers: Dict[str, List[str]] = {}
            for kk in proc_names:
                pcls = process_dict[kk]["class"]
                for name in tuple(pcls.get_inputs()) + tuple(
                    pcls.get_mutable_inputs()
                ):
                    consumers.setdefault(name, []).append(kk)
            map_feed = {
                mm.target_var: {
                    "map": map_name,
                    "source_grid": mm.source_grid,
                    "source_var": mm.source_var,
                }
                for map_name, mm in maps.items()
                if mm.target_grid == grid
            }

            inputs_internal: Dict[str, Any] = {}
            inputs_map_fed: Dict[str, Any] = {}
            inputs_external: Dict[str, Any] = {}
            for name, cons in consumers.items():
                if name in var_owner:
                    inputs_internal[name] = {
                        "producer": var_owner[name],
                        "consumers": cons,
                    }
                elif name in internal:
                    inputs_internal[name] = {
                        "producer": internal[name][0],
                        "consumers": cons,
                    }
                elif name in map_feed:
                    inputs_map_fed[name] = {
                        **map_feed[name],
                        "consumers": cons,
                    }
                elif name in parameters or name in derivable:
                    # a same-named supplied parameter satisfies the input
                    # structurally (assembly adds it to the grid dataset)
                    declarer = (
                        parameters[name][0]
                        if name in parameters
                        else derivable[name]["processes"][0]
                    )
                    inputs_internal[name] = {
                        "producer": f"(parameter of {declarer})",
                        "consumers": cons,
                    }
                else:
                    inputs_external[name] = {"consumers": cons}

            spec["required"][grid] = {
                "external_inputs": inputs_external,
                "parameters": parameters,
                "derivable_parameters": derivable,
                "initial_values": initial_values,
                "initial_state": initial_state,
            }
            if include_optional:
                spec["optional"][grid] = {
                    "internal_inputs": inputs_internal,
                    "internal_parameters": internal,
                    "map_fed_inputs": inputs_map_fed,
                }
        return spec

    def _initialize_inputs_and_proceses(self) -> None:  # (sic)
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
        self._run_initialize_hooks()
        self._resolve_maps()

    def _run_initialize_hooks(self) -> None:
        """Call each process's initialize() (author order), then freeze
        derived parameters -- read-only once ALL hooks have run (a shared
        derived name is written by its owner before the freeze)."""
        for kk in self.model_dict:
            self.model_dict[kk].initialize()
        for kk, proc in self.model_dict.items():
            grid_ds = self.discretizations[self._proc_grid[kk]].dataset
            for name in proc.get_parameters_internal():
                grid_ds[name].values.flags.writeable = False

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

        # -- parameters: sourced DIS-FIRST (grid-owned variables live on
        # the Discretization's own dataset -- dis_hru/dis_seg style), then
        # the process 'parameters' Dataset (or a Path opened here).
        # Loaded once, read-only. A process still DECLARES the dis vars
        # it reads; the declaration states the need, the dis is just the
        # first-priority source. --
        parameters: xr.Dataset | pl.Path | None = init_dict.get("parameters")
        if isinstance(parameters, pl.Path):
            parameters = xr.open_dataset(parameters)
        dis_params = self.discretizations[grid].parameters
        for pp in cls.get_parameters():
            if pp in grid_ds:
                continue
            if dis_params is not None and pp in dis_params:
                source = dis_params
            elif parameters is not None and pp in parameters:
                source = parameters
            else:
                raise ValueError(
                    f"process '{proc_name}' declares parameter '{pp}' but "
                    f"it is in neither grid '{grid}'s discretization "
                    "parameters nor the supplied 'parameters'"
                )
            source[pp].load()
            grid_ds[pp] = source[pp]
            grid_ds[pp].values.flags.writeable = False

        # -- inputs (read-only + mutable) --
        real_dim = self.discretizations[grid].dims[0]
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
                current = self.inputs_dict[ii].current_values
                if current.dims != (real_dim,):
                    # A wrong-named spatial dim would silently ride along
                    # as a SECOND dim on the grid dataset and only "work"
                    # by size coincidence (e.g. pywatershed forcings on
                    # "nhm_id" vs params on "nhru").
                    raise ValueError(
                        f"process '{proc_name}': input '{ii}' arrives on "
                        f"dims {current.dims} but grid '{grid}' expects "
                        f"('{real_dim}',) -- rename the input's spatial "
                        "dim to the grid dim (xr .rename)."
                    )
                grid_ds[ii] = current
            else:
                # cross-grid input -> the feeding Map's target buffer
                for mm in self.maps.values():
                    if mm.target_grid == grid and mm.target_var == ii:
                        grid_ds[ii] = mm.target_values
                        break

        # -- state variables + derived parameters: allocated (fill +
        # optional initial), once. Derived parameters are computed by the
        # process's initialize() hook and frozen after all hooks run.
        # A process declares its spatial dim as the placeholder "space"; bind
        # it to this grid's real dim (the grid key; `real_dim` above).
        # Params/inputs already arrive on the real dim (inputs validated
        # above), so only these need resolving. --
        allocated = cls.get_variables() | cls.get_parameters_internal()
        for name, meta in allocated.items():
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

    def _resolve_maps(
        self, grid_sizes: Union[Dict[str, int], None] = None
    ) -> None:
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
            # grid_sizes overrides a grid's dataset size for the weights
            # shape check (ModelMPI passes the GLOBAL size of the
            # decomposed grid; its dataset only knows the rank-local
            # extent).
            sizes_override = grid_sizes or {}
            n_target = sizes_override.get(
                mm.target_grid, target_ds.sizes.get(mm.target_grid)
            )
            n_source = sizes_override.get(
                mm.source_grid, source_ds.sizes.get(mm.source_grid)
            )
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
        if not self.inputs_dict:
            raise ValueError(
                "The model has no time-dimensioned inputs to set the "
                "clock from (inputs_dict is empty)."
            )
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
        """Run simulation for n_steps time steps (from the restart
        resume point when warm-started; from time zero otherwise)."""
        if self._finalized:
            raise RuntimeError("Cannot run a finalized Model.")
        start = self._start_index
        for tt in range(start, start + int(n_steps)):
            self.current_time_index[0] = tt
            self.current_time[0] = self.times[tt].values
            self.time.set_index(tt)
            self.advance()
            self.calculate(dt=dt)
            if self.output is not None:
                self.output.collect_current_timestep(tt)

    # -- restart ---------------------------------------------------------

    def write_restart(self, directory: Union[str, pl.Path]) -> None:
        """Write per-grid restart files for the LAST COMPLETED step:
        every process's flagged prognostic variables
        (``DataArrayMeta(restart=True)`` -> ``get_restart_variables()``)
        plus any python-attr restart state (``get_restart_state()``,
        namespaced ``{process}__{name}``), stamped with the state's
        timestamp (attrs["state_time"]) -- the file is self-locating:
        a warm-started model resumes at the FOLLOWING step of its own
        time axis. Grids with no restart content write no file.
        Files: ``{date}_restart_{grid}.nc``."""
        dd = pl.Path(directory)
        dd.mkdir(parents=True, exist_ok=True)
        tt = int(self.current_time_index[0])
        state_time = self.times[tt].values
        date_str = np.datetime_as_string(state_time.astype("datetime64[D]"))
        for grid, disc in self.discretizations.items():
            assert disc.dataset is not None
            ds_out = xr.Dataset()
            for proc_name, proc in self.model_dict.items():
                if self._proc_grid[proc_name] != grid:
                    continue
                for name in type(proc).get_restart_variables():
                    if name not in ds_out:
                        ds_out[name] = disc.dataset[name]
                for key, arr in proc.get_restart_state().items():
                    full = f"{proc_name}__{key}"
                    ds_out[full] = xr.DataArray(
                        arr,
                        dims=[f"{full}_d{ii}" for ii in range(arr.ndim)],
                    )
            if not ds_out.data_vars:
                continue
            ds_out.attrs["state_time"] = str(state_time)
            ds_out.to_netcdf(dd / f"{date_str}_restart_{grid}.nc")

    def _read_restart(self, directory: pl.Path) -> None:
        """Restore state from write_restart() files and set the resume
        index. Validates: exactly one file per grid-with-restart-content
        (and none for grids without), the file's variable set == the
        current flag set (a flag change between write and read fails
        loudly), consistent state_time across grids, and that the state
        time exists on this model's time axis with steps remaining."""
        state_times = set()
        for grid in self.discretizations:
            expected: set = set()
            hook_procs = []
            for proc_name, proc in self.model_dict.items():
                if self._proc_grid[proc_name] != grid:
                    continue
                expected |= set(type(proc).get_restart_variables())
                hook_procs.append(proc_name)
            found = sorted(directory.glob(f"*_restart_{grid}.nc"))
            if not expected and not found:
                continue
            if len(found) != 1:
                raise ValueError(
                    f"restart_read: grid '{grid}' needs exactly one "
                    f"*_restart_{grid}.nc in {directory}; found "
                    f"{[ff.name for ff in found]}."
                )
            ds_in = xr.load_dataset(found[0])
            state_times.add(str(ds_in.attrs["state_time"]))
            all_names = [str(nn) for nn in ds_in.data_vars]
            file_vars = {nn for nn in all_names if "__" not in nn}
            if file_vars != expected:
                raise ValueError(
                    f"restart file {found[0].name}: variable set "
                    f"{sorted(file_vars)} != this model's flagged set "
                    f"{sorted(expected)} (flags changed since the file "
                    "was written?)."
                )
            for name in file_vars:
                self._restore_grid_var(grid, name, ds_in[name].values)
            for proc_name in hook_procs:
                prefix = f"{proc_name}__"
                state = {
                    nn[len(prefix) :]: ds_in[nn].values
                    for nn in all_names
                    if nn.startswith(prefix)
                }
                if state:
                    self.model_dict[proc_name].set_restart_state(state)
        if not state_times:
            raise ValueError(
                f"restart_read: no restart files found in {directory} "
                "and no process declares restart variables."
            )
        if len(state_times) != 1:
            raise ValueError(
                f"restart_read: inconsistent state times across grid "
                f"files: {sorted(state_times)}."
            )
        state_time = np.datetime64(state_times.pop())
        where = np.where(self.times.values == state_time)[0]
        if where.size != 1:
            raise ValueError(
                f"restart state time {state_time} not found on this "
                "model's time axis (inputs must cover the restart "
                "time)."
            )
        self._start_index = int(where[0]) + 1
        if self._start_index >= self.ntime:
            raise ValueError(
                f"restart state time {state_time} is this model's "
                "final step: nothing to run."
            )
        for inp in self.inputs_dict.values():
            inp.set_index(self._start_index)

    def _restore_grid_var(
        self, grid: str, name: str, values: np.ndarray
    ) -> None:
        """Restore one restart variable into its grid dataset (the
        subclass seam: restart files are FULL-extent, and ModelMPI
        restores only its rank's block of the distributed grid)."""
        grid_ds = self.discretizations[grid].dataset
        assert grid_ds is not None
        grid_ds[name].values[:] = values

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
    """MPI streaming model: a distributed grid time-streamed via mpixarray,
    plus optional serial grids replicated on every rank, coupled by Maps.

    The distributed grid (``control["mpi_grid"]``, default ``"space"``)
    keeps the Phase 1 pipeline:

        open_dataset -> parallelize(dims=[mpi_grid]) -> set_streaming("time")
        -> open_writer -> declare buffers -> create -> iter_time -> write

    ONE decomposed dataset (``ds_mpi_stream``) carries the distributed
    grid's state, parameters, per-step input buffers, and streaming outputs
    -- cross-process buffer sharing there is *structural* (the same named
    variable). Serial grids (Step B) are assembled by the SERIAL machinery
    (``_add_process_fields`` + ``Input``), identically on every rank -- that
    IS the replication (uniform SPMD run loop; no rank branches); their data
    never enter the mpixarray dataset. A Map whose source is the distributed
    grid must be a ``MapMPI`` (distributed mat-vec: local partial product +
    Allreduce fills the target buffer on every rank). Serial->distributed
    maps (scatter) are not implemented. This hand-rolled cross-grid comm is
    the INTERIM implementation; the mpixarray streaming-datatree work is
    expected to absorb it.

    control dict:
        input_file        single combined input source for the DISTRIBUTED
                          grid (forcings + time-varying params as
                          (time, mpi_grid); static params + ICs as
                          (mpi_grid,))
        output_parallel_netcdf
                          the parallel-NetCDF file streamed by the
                          mpixarray writer (distributed grid)
        output_var_names  state vars to write, routed by OWNING grid:
                          distributed-grid vars stream to
                          output_parallel_netcdf (see note below);
                          serial-grid vars are collected by a rank-0 zarr
                          Output (everyone computes, one writes)
        output_serial_zarr
                          the serial zarr store path itself (".zarr"
                          suffix, used verbatim); required iff serial-grid
                          output vars are requested
        time_chunk_size   serial-grid Output time chunking (default 365)
        mpi_grid          name of the distributed grid/dim (default "space")
        restart_read      directory of write_restart() files to warm-start
                          from: saved state restores over the cold init
                          and the stream is time-sliced to BEGIN at the
                          resume step. Takes the SAME (superset)
                          input_file as the original run -- it need only
                          COVER the restart time; restart files are
                          serial/MPI interchangeable (see the Model
                          docstring).

    process_dict is as in Model (order = schedule; serial-grid entries carry
    "discretization" + their data); distributed-grid entries need only
    {"class": ...} -- their data come from input_file.

    NOTE (mpixarray limitation): declaring a 2nd ``to_netcdf=True`` variable
    currently trips the "cannot pickle 'module' object" deepcopy (the first
    such declaration stamps the writer handle onto the shared coord attrs, and
    the next ``from_numpy`` deepcopies them). So Phase 1 streams ONE output var
    to disk; validate any others from final in-memory state. Revisit when that
    mpixarray deepcopy bug is fixed.

    Usage:
        with ModelMPI(process_dict, control, maps=maps) as model:
            model.run(dt)
    """

    def __init__(
        self,
        process_dict: dict,
        control: dict,
        maps: Union[Dict[str, Any], None] = None,
        discretizations: Union[Dict[str, Discretization], None] = None,
    ) -> None:
        # Structure-only copy, as in Model (serial-grid Path values are
        # replaced in place by _load_paths_to_data).
        self._process_dict = {kk: dict(vv) for kk, vv in process_dict.items()}
        # Caller-supplied Discretizations (dis-owned parameters): SERIAL
        # grids only -- the distributed grid's data ride in input_file.
        self._provided_discs: Dict[str, Discretization] = discretizations or {}
        self._control = control
        self.maps: Dict[str, Any] = maps or {}
        self._proc_grid: Dict[str, str] = {
            kk: self._resolve_grid(vv) for kk, vv in self._process_dict.items()
        }
        self._finalized = False
        self.model_dict: dict = {}  # proc_name -> bound Process instance
        # Serial-grid Inputs, advanced in lockstep with the stream (the
        # distributed grid's inputs stream from step.mpi.src instead).
        self.inputs_dict: Dict[str, Input] = {}
        self._opened_files: List[Union[xr.DataArray, xr.Dataset]] = []
        self._load_all = True  # serial-grid data are small; keep in memory
        self._build()

    # -- construction ---------------------------------------------------

    def _build(self) -> None:
        f64 = np.float64
        control = self._control
        input_file = str(control["input_file"])
        output_parallel_netcdf = str(control["output_parallel_netcdf"])
        out_var_names = list(control["output_var_names"])
        mpi_grid = control.get("mpi_grid", "space")
        self._mpi_grid = mpi_grid

        serial_grids = {
            gg for gg in self._proc_grid.values() if gg != mpi_grid
        }
        bad_discs = [
            gg for gg in self._provided_discs if gg not in serial_grids
        ]
        if bad_discs:
            raise ValueError(
                f"discretizations {bad_discs}: pass Discretizations for "
                f"SERIAL grids only ({sorted(serial_grids)}); the "
                f"distributed grid '{mpi_grid}' takes its data from "
                "input_file."
            )

        mpi_classes = {
            kk: vv["class"]
            for kk, vv in self._process_dict.items()
            if self._proc_grid[kk] == mpi_grid
        }

        # Taxonomy guard (all grids): a parameter on the model `time` axis
        # is really an input. A true time-varying parameter lives on its own
        # axis (e.g. `month`) and stays resident. See pws_phoenix/CLAUDE.md.
        for proc_name, vv in self._process_dict.items():
            for pname, meta in _dict_of_kind(vv["class"], "parameter").items():
                if "time" in meta.dims:
                    raise ValueError(
                        f"Parameter {pname!r} on process {proc_name!r} has "
                        f"the model 'time' dim {meta.dims}: a variable on "
                        "model time is an input, not a parameter. Use a "
                        "non-'time' axis (e.g. 'month') for a time-varying "
                        "parameter."
                    )

        # ---- Distributed grid: open + decompose once over mpi_grid ----
        ds_input, comm = mpi_open_dataset(input_file)
        self._ds_input = ds_input
        n_time = int(ds_input.sizes["time"])
        self._ntime = n_time
        self.time = Time(ds_input["time"])
        # restart machinery (shared with the serial base): the full
        # time axis + the last-computed-step tracker
        self.times = ds_input["time"]
        self.ntime = n_time
        self.current_time_index = np.zeros(1, dtype=np.int64)

        # ---- Restart (warm start), part 1 of 2: locate the resume
        #      index BEFORE the stream is built, and LAZILY time-slice
        #      the input dataset so the stream simply BEGINS at the
        #      resume step (the superset input file serves any restart
        #      -- no duplicate files, no fast-forward). Model time
        #      stays GLOBALLY indexed (run() offsets by the start
        #      index), so istep0-gated blocks cannot re-fire. Full
        #      validation + the state restore are part 2, at the end
        #      of _build once the buffers exist. ----
        self._start_index = 0
        restart_read = control.get("restart_read")
        if restart_read is not None:
            self._start_index = self._peek_restart_resume_index(
                pl.Path(restart_read)
            )
        if self._start_index:
            ds_source = ds_input.isel(time=slice(self._start_index, None))
        else:
            ds_source = ds_input
        n_stream = n_time - self._start_index

        self.discretizations = {
            mpi_grid: Discretization([mpi_grid], comm=comm)
        }
        ds_mpi = self.discretizations[mpi_grid].decompose(ds_source)
        comm = self.discretizations[mpi_grid].comm  # parallelize may refine
        self._comm = comm

        # ---- Stream over time. set_streaming drops the time-dimensioned
        #      input vars (served per step via step.mpi.src) and keeps the
        #      space-only static params + ICs as buffers on ds_mpi_stream. ----
        ds_mpi_stream = ds_mpi.mpi.set_streaming(
            "time", window=1, out_times=list(range(n_stream))
        )

        # Realize the static (space-only) survivors -- params + ICs -- in
        # memory. Otherwise they stay lazy file-backed and each `.values`
        # re-reads, so buffer sharing (param_shared_name is ...) would not
        # hold.
        for name in list(ds_mpi_stream.data_vars):
            ds_mpi_stream[name] = ds_mpi_stream[name].load()

        # File-backed inputs that were dropped: refilled from src each step.
        # Distributed grid only -- an input there is file-backed (vs.
        # produced upstream) if no distributed-grid process owns a variable
        # of that name. Serial-grid inputs go through Input objects instead.
        mpi_var_names: set = set()
        for cls in mpi_classes.values():
            mpi_var_names |= set(cls.get_var_names())
        file_input_names: list = []
        for cls in mpi_classes.values():
            for ii in tuple(cls.get_inputs()) + tuple(
                cls.get_mutable_inputs()
            ):
                if ii not in mpi_var_names and ii not in file_input_names:
                    file_input_names.append(ii)
        self._file_input_names = file_input_names

        # Fail fast if a file-backed input is absent from the input file --
        # it would otherwise KeyError mid-run on the first src refill.
        missing = [
            name for name in file_input_names if name not in ds_input.data_vars
        ]
        if missing:
            raise ValueError(
                f"input(s) {missing} are not produced by any process on "
                f"grid '{mpi_grid}' and were not found in input_file "
                f"{input_file!r}."
            )
        # ... and fail fast on wrong dims: decompose() splits mpi_grid
        # only, so an input on another spatial dim (e.g. pywatershed's
        # "nhm_id" vs "nhru") stays FULL extent and broadcast-errors on
        # the first src refill, mid-collective.
        bad_dims = {
            name: ds_input[name].dims
            for name in file_input_names
            if ds_input[name].dims != ("time", mpi_grid)
        }
        if bad_dims:
            raise ValueError(
                f"input_file streamed input(s) must have dims "
                f"('time', '{mpi_grid}'); got {bad_dims}. Rename the "
                "offending dim(s) to the distributed grid dim when "
                "assembling the input file."
            )

        # Split requested outputs by OWNING grid (the process declaring the
        # var as a state variable): distributed-grid vars stream via the
        # mpixarray writer; serial-grid vars are collected by a rank-0 zarr
        # Output (created at the end of _build).
        serial_var_grid: Dict[str, str] = {}
        for kk, vv in self._process_dict.items():
            if self._proc_grid[kk] == mpi_grid:
                continue
            for name in vv["class"].get_var_names():
                serial_var_grid[name] = self._proc_grid[kk]
        ambiguous = [
            nn
            for nn in out_var_names
            if nn in mpi_var_names and nn in serial_var_grid
        ]
        bad_out = [
            nn
            for nn in out_var_names
            if nn not in mpi_var_names and nn not in serial_var_grid
        ]
        if ambiguous or bad_out:
            raise ValueError(
                f"output_var_names: {ambiguous} are owned on multiple grids "
                f"and {bad_out} are owned on none -- each output var must "
                "be a state variable of exactly one grid's processes."
            )
        stream_out_names = [nn for nn in out_var_names if nn in mpi_var_names]
        serial_out_names = [
            nn for nn in out_var_names if nn in serial_var_grid
        ]

        # ---- Open writer ----
        ds_mpi_stream.mpi.open_writer(output_parallel_netcdf, comm=comm)

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
        for cls in mpi_classes.values():
            for name in cls.get_var_names():
                if name not in declared:
                    _declare(name, (mpi_grid,), np.nan, False)
                    declared.add(name)
        # (a2) derived parameters -- computed by initialize() pre-loop.
        # Dims resolve like the serial path: "space" -> the distributed
        # grid dim; other declared dims (e.g. PRMSSnow's "nmonth") pass
        # through -- they exist on ds_mpi_stream via the static params
        # that carry them and replicate whole (only mpi_grid is
        # decomposed).
        # NOTE: non-f64 dtypes (e.g. int64) are untested against the
        # mpixarray buffer declaration until a distributed process
        # declares one.
        for cls in mpi_classes.values():
            for name, meta in cls.get_parameters_internal().items():
                if name not in declared:
                    dims = tuple(
                        mpi_grid if dd == "space" else dd for dd in meta.dims
                    )
                    ds_mpi_stream.mpi[name] = {
                        "dims": dims,
                        "dtype": meta.dtype,
                        "fill_value": meta.dtype(_FILL_VALUE[meta.dtype]),
                        "to_netcdf": False,
                        "comm": comm,
                    }
                    declared.add(name)
        # (b) per-step input buffers (refilled from src)
        for name in file_input_names:
            if name not in declared:
                _declare(name, (mpi_grid,), np.nan, False)
                declared.add(name)
        # (c) streaming output buffers (to_netcdf=True) -- declared LAST
        output_map: dict = {}  # state var name -> on-disk output buffer name
        for name in stream_out_names:
            buf = f"{name}_out"
            _declare(buf, ("time", mpi_grid), 0.0, True)
            output_map[name] = buf
        self._output_map = output_map

        ds_mpi_stream.mpi.create(coords=None, data_vars=None)

        # ---- Load initial conditions into state buffers. ICs are the
        #      space-only vars that survived set_streaming (from the file). ----
        for cls in mpi_classes.values():
            for name, meta in cls.get_variables().items():
                if (
                    meta.initial is not None
                    and meta.initial in ds_mpi_stream.data_vars
                ):
                    ds_mpi_stream[name].values[:] = ds_mpi_stream[
                        meta.initial
                    ].values

        self._ds_mpi_stream = ds_mpi_stream
        self.discretizations[mpi_grid].dataset = ds_mpi_stream

        # ---- Serial grids (Step B): replicated on every rank. Assembled
        #      by the serial machinery, identically on each rank (from
        #      deterministic/shared inputs) -- that IS the replication.
        #      Their data never enter the mpixarray dataset. ----
        serial_grid_procs: Dict[str, List[str]] = {}
        for kk, grid in self._proc_grid.items():
            if grid != mpi_grid:
                serial_grid_procs.setdefault(grid, []).append(kk)
        for grid in serial_grid_procs:
            self.discretizations[grid] = self._provided_discs.get(
                grid
            ) or Discretization([grid])
        self._load_paths_to_data()
        for grid, proc_names in serial_grid_procs.items():
            grid_ds = xr.Dataset()
            for kk in proc_names:
                self._add_process_fields(kk, grid, grid_ds)
            self.discretizations[grid].dataset = grid_ds

        # ---- Bind in process_dict (author) order; distributed-grid
        #      processes are rebound to each streaming step in run(). ----
        for kk, vv in self._process_dict.items():
            self.model_dict[kk] = vv["class"](
                self.discretizations[self._proc_grid[kk]].dataset
            )

        self._validate_inputs_resolved()
        # Per-process init hooks are LOCAL (collective-free) by contract,
        # so running them inside SPMD assembly is safe.
        self._run_initialize_hooks()

        # ---- Maps: configure the parallel boundary, then resolve. The
        #      decomposed extent comes from an allgather of local sizes
        #      (scheme "single" = contiguous rank-ordered blocks). ----
        local_n = int(ds_mpi_stream.sizes[mpi_grid])
        sizes = comm.allgather(local_n)
        start = int(sum(sizes[: comm.rank]))
        n_global = int(sum(sizes))
        # this rank's block of the global extent (contiguous
        # rank-ordered) -- also the restart restore/gather arithmetic
        self._decomp_slice = (start, start + local_n)
        for map_name, mm in self.maps.items():
            if mm.target_grid == mpi_grid:
                raise ValueError(
                    f"Map '{map_name}' targets the distributed grid "
                    f"'{mpi_grid}': serial->distributed maps (scatter) "
                    "are not implemented (Step B is distributed->serial)."
                )
            if mm.source_grid == mpi_grid:
                if not hasattr(mm, "set_decomposition"):
                    raise ValueError(
                        f"Map '{map_name}' crosses the parallel boundary "
                        f"from '{mpi_grid}': use MapMPI, not Map."
                    )
                mm.set_decomposition(comm, start, start + local_n)
        self._resolve_maps(grid_sizes={mpi_grid: n_global})

        # ---- Serial-grid output: rank 0 collects to a zarr Output --
        #      replicated grids mean everyone computes, one writes. The
        #      rank branch is safe by construction: Output holds NO
        #      collectives (local buffering + disk IO only).
        self.output = None
        if serial_out_names:
            if "output_serial_zarr" not in control:
                raise ValueError(
                    f"output_var_names {serial_out_names} live on serial "
                    "grids: control['output_serial_zarr'] (a .zarr path) "
                    "is required for the rank-0 zarr Output."
                )
            if comm.rank == 0:
                self.output = Output(
                    time_chunk_size=control.get("time_chunk_size", 365),
                    variable_names=serial_out_names,
                    output_store=pl.Path(control["output_serial_zarr"]),
                    time_values=ds_input["time"].values,
                )
                # Track serial-grid processes only: their _obj is
                # persistent. (Distributed-grid procs are rebound to each
                # step, so refs captured here would go stale.)
                self.output.initialize_variable_tracking(
                    {
                        kk: proc
                        for kk, proc in self.model_dict.items()
                        if self._proc_grid[kk] != mpi_grid
                    }
                )

        # ---- Restart (warm start), part 2 of 2: full validation +
        #      the state restore, AFTER the init hooks so the saved
        #      state overwrites initialize()'s cold state. EVERY rank
        #      reads the full-extent files and restores its own block
        #      (SPMD-uniform, no collectives). Recomputes the same
        #      start index located by part 1. ----
        if restart_read is not None:
            self._read_restart(pl.Path(restart_read))

    # -- run ------------------------------------------------------------

    def run(self, dt: np.float64, n_steps: np.int32 | None = None) -> None:
        """Stream the time loop via iter_time(); n_steps is ignored (the
        streaming source defines the count). Serial-grid inputs advance in
        lockstep with the stream; a Map is applied exactly once per step,
        immediately before its first consumer (see Model._resolve_maps).
        The loop is SPMD-uniform: every rank executes the same statements
        every step (no rank branches -- serial grids are replicated)."""
        if self._finalized:
            raise RuntimeError("Cannot run a finalized Model.")
        ds_mpi_stream = self._ds_mpi_stream
        mpi_grid = self._mpi_grid
        mpi_procs = [
            proc
            for kk, proc in self.model_dict.items()
            if self._proc_grid[kk] == mpi_grid
        ]
        # warm start: the stream was time-sliced at build to BEGIN at
        # the resume index, so `tt` here is stream-local; model time
        # stays GLOBALLY indexed via the offset (istep0-gated blocks
        # must not re-fire at the resume step).
        start = self._start_index
        for tt, step in enumerate(ds_mpi_stream.mpi.iter_time()):
            gtt = tt + start
            self.current_time_index[0] = gtt
            self.time.set_index(gtt)
            src = step.mpi.src
            # Refill this step's input buffers from the source slab.
            for name in self._file_input_names:
                step[name].values[:] = src[name].values[0, :]
            # Serial-grid inputs advance in lockstep with the stream.
            for inp in self.inputs_dict.values():
                inp.advance()
            # Rebind distributed-grid processes (and their disc's dataset)
            # to the current step -- shares the persistent state buffers,
            # so the Markov chain carries across steps. Serial-grid
            # processes stay bound to their persistent grid dataset.
            self.discretizations[mpi_grid].dataset = step
            for proc in mpi_procs:
                proc._obj = step
            for proc in self.model_dict.values():
                proc.advance()
            for proc_name, proc in self.model_dict.items():
                for mm in self._proc_maps[proc_name]:
                    mm.apply(self.discretizations[mm.source_grid].dataset)
                proc.calculate(dt, self.time)
            # Record state into the streaming output slab(s) and write.
            for src_name, buf in self._output_map.items():
                step[buf].values[0, :] = step[src_name].values[:]
                if step.mpi.is_output_step:
                    step[buf].mpi.write()
            # Serial-grid output: rank 0 only (collective-free branch).
            if self.output is not None:
                self.output.collect_current_timestep(gtt)

    # -- restart ---------------------------------------------------------

    def _peek_restart_resume_index(self, directory: pl.Path) -> int:
        """Locate the resume index BEFORE the stream is built (part 1
        of the warm start; the input dataset is time-sliced to start
        there): read state_time from every restart file's attrs and
        find the following step on the FULL time axis. Full per-grid
        validation and the state restore are _read_restart's job (part
        2), once the buffers exist -- it recomputes this same index."""
        found = sorted(directory.glob("*_restart_*.nc"))
        if not found:
            raise ValueError(
                f"restart_read: no restart files found in {directory}."
            )
        state_times = set()
        for ff in found:
            with xr.open_dataset(ff) as ds_in:
                state_times.add(str(ds_in.attrs["state_time"]))
        if len(state_times) != 1:
            raise ValueError(
                f"restart_read: inconsistent state times across grid "
                f"files: {sorted(state_times)}."
            )
        state_time = np.datetime64(state_times.pop())
        where = np.where(self.times.values == state_time)[0]
        if where.size != 1:
            raise ValueError(
                f"restart state time {state_time} not found on this "
                "model's time axis (inputs must cover the restart "
                "time)."
            )
        start = int(where[0]) + 1
        if start >= self.ntime:
            raise ValueError(
                f"restart state time {state_time} is this model's "
                "final step: nothing to run."
            )
        return start

    def write_restart(self, directory: Union[str, pl.Path]) -> None:
        """Serial-FORMAT full-extent restart files, written by rank 0
        (gather-then-write, mirroring the rank-0 zarr Output stance):
        the distributed grid's flagged variables are allgathered
        (contiguous rank-ordered blocks -> concatenate = global order);
        serial (replicated) grids write from rank 0's identical copy.
        The files are format-identical to serial Model.write_restart,
        so serial and MPI runs can warm-start each other. The loop is
        collective-uniform (every rank joins every allgather; only the
        writes are rank-branched, and they hold no collectives) and
        ends on a Barrier so the files are complete before any rank
        proceeds -- e.g. straight into a warm-started model."""
        comm = self._comm
        dd = pl.Path(directory)
        if comm.rank == 0:
            dd.mkdir(parents=True, exist_ok=True)
        tt = int(self.current_time_index[0])
        state_time = self.times[tt].values
        date_str = np.datetime_as_string(state_time.astype("datetime64[D]"))
        for grid, disc in self.discretizations.items():
            distributed = grid == self._mpi_grid
            ds_out = xr.Dataset()
            for proc_name, proc in self.model_dict.items():
                if self._proc_grid[proc_name] != grid:
                    continue
                for name in type(proc).get_restart_variables():
                    if name in ds_out:
                        continue
                    if distributed:
                        parts = comm.allgather(
                            self._ds_mpi_stream[name].values.copy()
                        )
                        ds_out[name] = ((grid,), np.concatenate(parts))
                    else:
                        assert disc.dataset is not None
                        ds_out[name] = disc.dataset[name]
                state = proc.get_restart_state()
                if distributed and state:
                    raise NotImplementedError(
                        f"process '{proc_name}': python-attr restart "
                        "state (get_restart_state) on the DISTRIBUTED "
                        "grid is not implemented -- no distributed "
                        "process carries hook state today."
                    )
                for key, arr in state.items():
                    full = f"{proc_name}__{key}"
                    ds_out[full] = xr.DataArray(
                        arr,
                        dims=[f"{full}_d{ii}" for ii in range(arr.ndim)],
                    )
            if not ds_out.data_vars:
                continue
            ds_out.attrs["state_time"] = str(state_time)
            if comm.rank == 0:
                ds_out.to_netcdf(dd / f"{date_str}_restart_{grid}.nc")
        comm.Barrier()

    def _restore_grid_var(
        self, grid: str, name: str, values: np.ndarray
    ) -> None:
        """Distributed grid: restore this rank's contiguous block of
        the full-extent restart variable; serial grids restore whole
        (replicated), as in the base."""
        if grid == self._mpi_grid:
            aa, bb = self._decomp_slice
            self._ds_mpi_stream[name].values[:] = values[aa:bb]
        else:
            super()._restore_grid_var(grid, name, values)

    # -- finalize -------------------------------------------------------

    def finalize(self) -> None:
        if self._finalized:
            return
        for inp in self.inputs_dict.values():
            inp.close()
        for ff in self._opened_files:
            ff.close()
        if self.output is not None:
            self.output.finalize()
        self._ds_mpi_stream.mpi.finalize()
        self._ds_input.mpi.finalize()
        self._finalized = True
