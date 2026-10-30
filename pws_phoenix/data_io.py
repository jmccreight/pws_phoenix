"""
data_io.py
==========
IO primitives for pws_phoenix:
  - open_xr: open a NetCDF file as DataArray or Dataset
  - Input:   time-varying input wrapper with advance() / current_values
  - Output:  serial buffered, time-chunked zarr writer (adapted from pywatershed)

These are process-agnostic and sit at the base of the import stack
(model.py builds on them; process.py no longer imports data_io).

Output is the serial writer (one zarr store). The MPI path (ModelMPI in
model.py) does NOT use Output -- it streams output through mpixarray
(as_iterable + iterate + .mpi.write).
"""

import pathlib as pl
from typing import Any, Dict, List, Union

import numpy as np
import xarray as xr
import zarr


def open_xr(
    path: pl.Path, load: bool = False
) -> Union[xr.DataArray, xr.Dataset]:
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
        # the ndarrays behind the per-step copy (DEVELOPER.md "Per-step
        # data access"): a DataArray index + setitem cost ~100 us a step.
        # A lazy (file-backed, load=False) input keeps reading one slice
        # per step through xarray -- its whole array is not in memory.
        self._in_memory: bool = self.data.variable._in_memory
        self._data_values: np.ndarray | None = (
            self.data.values if self._in_memory else None
        )
        self._current: np.ndarray = self._current_values.values

    def advance(self) -> None:
        """Advance current_values to the next time step."""
        self._current_index += np.int64(1)
        if self._in_memory:
            self._current[:] = self._data_values[self._current_index]
        else:
            self._current[:] = self.data[self._current_index, :].values

    def set_index(self, index: int) -> None:
        """Fast-forward (restart): the NEXT advance() serves `index`.
        current_values is left untouched -- the run loop advances
        before any calculate reads it."""
        self._current_index = np.int64(index) - np.int64(1)

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
    """Buffered, time-chunked serial writer to a single zarr store.

    The store is created ONCE at initialize_variable_tracking: xarray writes
    the coordinates (the full `time` axis, the spatial dim-coordinates) and
    each variable is then declared with zarr at its full (n_time, space)
    shape, time-chunked by time_chunk_size. Declaring is free -- a zarr
    chunk exists on disk only once written -- so peak memory = the
    per-variable chunk buffers; nothing is materialized (see "Prime
    directive: memory" in pws_phoenix/CLAUDE.md). Each full buffer is
    written as a region `arr[start:end] = buffer` (no xarray in the loop;
    the earlier `to_zarr(append_dim="time")` cost ~2 s of store
    re-validation per append, independent of data size); a partial tail is
    written at finalize. Adapted from pywatershed's chunked zarr writer.

    Serial only. The MPI path streams output through mpixarray
    (as_iterable + iterate + .mpi.write) and does not use this class.
    (Pre-sized store + region writes is also the shape concurrent writers
    would need.)
    """

    def __init__(
        self,
        time_chunk_size: int,
        variable_names: List[str],
        output_store: pl.Path,
        time_values: np.ndarray,
    ) -> None:
        """Initialize Output manager for buffered, time-chunked zarr writing.

        Args:
            time_chunk_size: Number of time steps to buffer in memory before
                a chunk is written. Also the store's time chunk size.
            variable_names: Variables to write -- one data_var each in the
                single output store.
            output_store: Path to the (single) zarr store directory; must
                carry the ".zarr" suffix and is used verbatim.
            time_values: 1-D datetime coordinate for the full run (sizes
                the store's time axis).

        Note:
            Tailored to read Model.model_dict (key -> Process instance),
            where each process's variables are identified via
            get_var_names().
        """
        self.time_chunk_size = time_chunk_size
        self.variable_names = variable_names
        self.output_store = pl.Path(output_store)
        if self.output_store.suffix != ".zarr":
            raise ValueError(
                f"output_store {str(self.output_store)!r} must carry the "
                "'.zarr' suffix (a zarr store is a directory)."
            )
        self.output_store.parent.mkdir(parents=True, exist_ok=True)
        self.time_values = time_values

        # Track variable references and data
        self.variable_refs: Dict[str, xr.DataArray] = {}
        self.process_map: Dict[str, str] = {}  # var_name -> process_name
        self.data_buffers: Dict[str, np.ndarray] = {}
        self._arrays: Dict[str, zarr.Array] = {}  # var_name -> store array
        self.current_time_step = 0

    def initialize_variable_tracking(self, model_dict: Dict[str, Any]) -> None:
        """Reference the requested variables on their processes, allocate
        buffers and create the (pre-sized, unmaterialized) zarr store. Must
        be called after model processes are initialized.

        Raises:
            ValueError: If any requested variable is not found in any process.
        """
        for var_name in self.variable_names:
            found = False
            for process_name, process_obj in model_dict.items():
                if var_name in process_obj.get_var_names():
                    self.variable_refs[var_name] = process_obj[var_name]
                    self.process_map[var_name] = process_name
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Variable '{var_name}' not found in any process"
                )

        self._initialize_buffers()
        self._create_store()

    def _create_store(self) -> None:
        """Create the store: coordinates via xarray (so `time` round-trips
        as datetime64), then each variable declared at full shape with
        zarr. No data chunk is written here."""
        coords: dict = {"time": self.time_values}
        for var_ref in self.variable_refs.values():
            spatial_dim = str(var_ref.dims[0])
            # Carry the spatial dim-coordinate (e.g. "space") if present.
            if spatial_dim in var_ref.coords and spatial_dim not in coords:
                coords[spatial_dim] = (
                    spatial_dim,
                    var_ref.coords[spatial_dim].values,
                )
        xr.Dataset(coords=coords).to_zarr(
            self.output_store, mode="w", consolidated=False
        )
        group = zarr.open_group(self.output_store, mode="r+")
        n_time = len(self.time_values)
        for var_name, var_ref in self.variable_refs.items():
            # One chunk spans the full spatial extent; time chunked by buffer.
            self._arrays[var_name] = group.create_array(
                var_name,
                shape=(n_time,) + var_ref.shape,
                chunks=(self.time_chunk_size,) + var_ref.shape,
                dtype=var_ref.dtype,
                dimension_names=("time", str(var_ref.dims[0])),
            )

    def _initialize_buffers(self) -> None:
        """Initialize numpy arrays to store time chunk data for each variable."""
        for var_name, var_ref in self.variable_refs.items():
            # Buffer: (time_chunk_size, *spatial_dims)
            buffer_shape = (self.time_chunk_size,) + var_ref.shape
            self.data_buffers[var_name] = np.empty(
                buffer_shape, dtype=var_ref.dtype
            )

    def collect_current_timestep(self, time_index: int) -> None:
        """Buffer this step's values; write a full chunk when the buffer
        fills.

        Args:
            time_index: Global time index.
        """
        buffer_index = self.current_time_step % self.time_chunk_size

        for var_name, var_ref in self.variable_refs.items():
            self.data_buffers[var_name][buffer_index] = var_ref.values

        self.current_time_step += 1

        if self.current_time_step % self.time_chunk_size == 0:
            self._write_chunk(self.time_chunk_size)

    def _write_chunk(self, n_valid: int) -> None:
        """Write the first n_valid buffered steps to their time region."""
        chunk_end = self.current_time_step
        chunk_start = chunk_end - n_valid
        for var_name, buffer in self.data_buffers.items():
            self._arrays[var_name][chunk_start:chunk_end] = buffer[:n_valid]

    def finalize(self) -> None:
        """Write any remaining partial buffer, then shrink the store's time
        axis to the steps actually written (a run may stop short of
        time_values; resize is metadata-only, dropping any chunk beyond)."""
        remaining = self.current_time_step % self.time_chunk_size
        if remaining > 0:
            self._write_chunk(remaining)
        n_written = self.current_time_step
        if n_written < len(self.time_values):
            group = zarr.open_group(self.output_store, mode="r+")
            group["time"].resize((n_written,))
            for arr in self._arrays.values():
                arr.resize((n_written,) + arr.shape[1:])
