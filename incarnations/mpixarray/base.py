"""
base.py (minimal)
=================
Minimal base classes for incarnations/mpixarray.

Contains only what is needed by base_attrs2.py:
  - open_xr: open a NetCDF file as DataArray or Dataset
  - Input:   time-varying input wrapper with advance() / current_values
  - Output:  serial buffered, time-chunked zarr writer (adapted from pywatershed)
  - Model:   orchestration base class (serial time loop + optional Output)

The serial Model writes output via the Output collector (one zarr store). The
MPI path (ModelMPI in base_attrs2.py) does NOT use Output -- it streams output
through mpixarray (set_streaming + iter_time + .mpi.write).
"""

import pathlib as pl
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Literal, Union

import numpy as np
import xarray as xr

# zarr is imported lazily inside Output (serial-only), keeping the module
# import surface minimal; the MPI path streams via mpixarray and never uses it.


def open_xr(path: pl.Path, load: bool = False) -> Union[xr.DataArray, xr.Dataset]:
    """Open a NetCDF file and return a DataArray (single var) or Dataset."""
    ds = xr.open_dataset(path)
    da_ds: Union[xr.DataArray, xr.Dataset]
    if len(ds.data_vars) == 1:
        da_ds = ds[list(ds.data_vars)[0]]
    else:
        da_ds = ds
    if load:
        da_ds = da_ds.load()
    return da_ds


class Input:
    """Time-varying input wrapper.

    Wraps a DataArray (from file or memory), advances through time steps,
    and exposes current_values for the current time step.
    """

    def __init__(
        self,
        data_or_file: Union[xr.DataArray, pl.Path],
        read_only: bool = False,
        load: bool = False,
    ) -> None:
        self._input_file: Union[pl.Path, None] = None
        if isinstance(data_or_file, pl.Path):
            self.data = xr.open_dataarray(data_or_file)
            self._input_file = data_or_file
        else:
            self.data = data_or_file
        if load:
            self.data = self.data.load()
        if read_only:
            self.data.values.flags.writeable = False
        self._current_index = np.int64(-1)
        self._current_values = np.nan * self.data[0, :]
        self._closed = False

    def advance(self) -> None:
        """Advance current_values to the next time step."""
        self._current_index += np.int64(1)
        self._current_values[:] = self.data[self._current_index, :]

    @property
    def current_values(self) -> xr.DataArray:
        """Values at the current time step."""
        return self._current_values

    def close(self) -> None:
        """Close file handle if this Input opened a file."""
        if not self._closed:
            if self._input_file is not None:
                self.data.close()
            self._closed = True


class Output:
    """Buffered, time-chunked output of model variables to a single zarr store.

    Buffers each tracked variable in memory for time_chunk_size steps, then
    region-writes the full chunk into a pre-sized zarr store; a partial tail is
    flushed at finalize. Adapted from pywatershed's chunked zarr writer.

    Serial only. The MPI path streams output through mpixarray
    (set_streaming + iter_time + .mpi.write) and does not use this class.

    This class:
    - Tracks references to specified variables from model processes
    - Buffers data for time_chunk_size time steps
    - Region-writes complete chunks into one zarr store (all vars together)
    - Handles a partial chunk at simulation end
    """

    def __init__(
        self,
        time_chunk_size: int,
        variable_names: List[str],
        output_store: pl.Path,
        n_times: int,
        time_values: np.ndarray,
    ) -> None:
        """Initialize Output manager for buffered, time-chunked zarr writing.

        Args:
            time_chunk_size: Number of time steps to buffer in memory before a
                chunk is region-written. Also the store's time chunk size.
            variable_names: Variables to write -- one data_var each in the
                single output store.
            output_store: Path to the (single) zarr store directory.
            n_times: Total number of time steps; sizes the store's time dim.
            time_values: 1-D datetime coordinate of length n_times.

        Note:
            Tailored to read Model.model_dict (key -> process), where each
            process's variables are identified via get_var_names()/
            get_variables() (Process) or the pws accessor (xr.Dataset).
        """
        self.time_chunk_size = time_chunk_size
        self.variable_names = variable_names
        self.output_store = pl.Path(output_store)
        self.output_store.parent.mkdir(parents=True, exist_ok=True)
        self.n_times = int(n_times)
        self.time_values = time_values

        # Track variable references and data
        self.variable_refs: Dict[str, xr.DataArray] = {}
        self.process_map: Dict[str, str] = {}  # var_name -> process_name
        self.data_buffers: Dict[str, np.ndarray] = {}
        self.current_time_step = 0

        # zarr store, opened lazily on the first chunk write
        self._zarr_store = None
        self._zarr_initialized = False

    def initialize_variable_tracking(self, model_dict: Dict[str, Any]) -> None:
        """Reference the requested variables on their processes and allocate
        buffers. The zarr store is created lazily on the first write. Must be
        called after model processes are initialized.

        Raises:
            ValueError: If any requested variable is not found in any process.
        """
        for var_name in self.variable_names:
            found = False
            for process_name, process_obj in model_dict.items():
                var_names = (
                    process_obj.pws.get_var_names()  # type: ignore[attr-defined]
                    if isinstance(process_obj, xr.Dataset)
                    else process_obj.get_variables()
                )
                if var_name in var_names:
                    self.variable_refs[var_name] = process_obj[var_name]
                    self.process_map[var_name] = process_name
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Variable '{var_name}' not found in any process"
                )

        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        """Initialize numpy arrays to store time chunk data for each variable."""
        for var_name, var_ref in self.variable_refs.items():
            # Buffer: (time_chunk_size, *spatial_dims)
            buffer_shape = (self.time_chunk_size,) + var_ref.shape
            self.data_buffers[var_name] = np.empty(
                buffer_shape, dtype=var_ref.dtype
            )

    def collect_current_timestep(self, time_index: int) -> None:
        """Buffer this step's values; flush a full chunk when the buffer fills.

        Args:
            time_index: Global time index.
        """
        buffer_index = self.current_time_step % self.time_chunk_size

        for var_name, var_ref in self.variable_refs.items():
            self.data_buffers[var_name][buffer_index] = var_ref.values

        self.current_time_step += 1

        if self.current_time_step % self.time_chunk_size == 0:
            self._write_buffer_chunk()

    def _initialize_zarr(self) -> None:
        """Create the pre-sized zarr store (all vars, full time extent) and
        reopen it for region writes. Done once, on the first flush."""
        if self._zarr_initialized:
            return
        import zarr

        data_vars: dict = {}
        coords: dict = {"time": self.time_values}
        encoding: dict = {}
        for var_name, var_ref in self.variable_refs.items():
            spatial_shape = var_ref.shape
            spatial_dim = str(var_ref.dims[0])
            # Placeholder zeros, filled incrementally by region writes.
            data_vars[var_name] = (
                ["time", spatial_dim],
                np.zeros((self.n_times,) + spatial_shape, dtype=var_ref.dtype),
            )
            # Carry the spatial coordinate (e.g. space_coord) if present.
            coord_name = f"{spatial_dim}_coord"
            if coord_name in var_ref.coords and coord_name not in coords:
                coords[coord_name] = (
                    spatial_dim,
                    var_ref.coords[coord_name].values,
                )
            # One chunk spans the full spatial extent; time chunked by buffer.
            encoding[var_name] = {
                "chunks": (self.time_chunk_size,) + spatial_shape
            }

        ds = xr.Dataset(data_vars, coords=coords)
        ds.to_zarr(
            self.output_store, mode="w", encoding=encoding, consolidated=False
        )
        self._zarr_store = zarr.open(str(self.output_store), mode="r+")
        self._zarr_initialized = True

    def _write_buffer_chunk(self) -> None:
        """Region-write the full in-memory buffer to the zarr store."""
        if not self._zarr_initialized:
            self._initialize_zarr()
        chunk_end = self.current_time_step
        chunk_start = chunk_end - self.time_chunk_size
        for var_name in self.variable_names:
            self._zarr_store[var_name][chunk_start:chunk_end] = (
                self.data_buffers[var_name]
            )

    def finalize(self) -> None:
        """Region-write any remaining partial buffer and release the store."""
        if not self._zarr_initialized:
            self._initialize_zarr()
        remaining = self.current_time_step % self.time_chunk_size
        if remaining > 0:
            chunk_end = self.current_time_step
            chunk_start = chunk_end - remaining
            for var_name in self.variable_names:
                self._zarr_store[var_name][chunk_start:chunk_end] = (
                    self.data_buffers[var_name][:remaining]
                )
        self._zarr_store = None


class Model:
    """Base simulation orchestrator.

    Loads data, initializes processes, wires dependencies, runs the time loop.
    Subclasses override _initialize_inputs_and_proceses() for MPI or other
    backends.
    """

    def __init__(
        self,
        process_dict: Dict[str, Any],
        control: Dict[str, Any],
        load_all: Union[bool, None] = None,
    ) -> None:
        self._passed_process_dict = process_dict
        self._process_dict = deepcopy(process_dict)
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
            if "output_var_names" not in control or "output_dir" not in control:
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

    def _initialize_inputs_and_proceses(self) -> None:  # noqa: spelling
        """Initialize Input and Process objects, wire dependencies."""
        for kk, vv in self._process_dict.items():
            init_dict = {k: v for k, v in vv.items() if k != "class"}

            inputs_req = vv["class"].get_inputs()
            input_outputs_req = vv["class"].get_mutable_inputs()
            all_inputs = inputs_req + input_outputs_req

            for ii in all_inputs:
                if ii in init_dict:
                    data_or_file = init_dict[ii]
                    read_only = ii in inputs_req
                    if not read_only:
                        raise ValueError("This should not happen from file.")
                    if ii not in self.inputs_dict:
                        init_dict[ii] = Input(
                            data_or_file,
                            read_only=read_only,
                            load=self._load_all,
                        )
                        self.inputs_dict[ii] = init_dict[ii]
                        del data_or_file
                    else:
                        init_dict[ii] = self.inputs_dict[ii]
                else:
                    for pp in self.get_preceeding_processes(kk):
                        proc = self.model_dict[pp]
                        var_names = (
                            proc.pws.get_var_names()
                            if isinstance(proc, xr.Dataset)
                            else proc.get_variables()
                        )
                        if ii in var_names:
                            init_dict[ii] = self.model_dict[pp][ii]

            self.model_dict[kk] = vv["class"](**init_dict)

    def _set_time(self) -> None:
        """Set time dimensions from first input."""
        kk0 = list(self.inputs_dict.keys())[0]
        self.ntime = self.inputs_dict[kk0].data.sizes["time"]
        self.time_index = self.inputs_dict[kk0].data.time
        self.times = self.inputs_dict[kk0].data.time_coord

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
                vv.pws.calculate(dt)
            else:
                vv.calculate(dt)

    def run(self, dt: np.float64, n_steps: np.int32) -> None:
        """Run simulation for n_steps time steps."""
        if self._finalized:
            raise RuntimeError("Cannot run a finalized Model.")
        for tt in range(n_steps):
            self.current_time_index[0] = tt
            self.current_time[0] = self.times[tt].values
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
