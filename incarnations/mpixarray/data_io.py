"""
data_io.py
==========
IO primitives for incarnations/mpixarray:
  - open_xr: open a NetCDF file as DataArray or Dataset
  - Input:   time-varying input wrapper with advance() / current_values
  - Output:  serial buffered, time-chunked zarr writer (adapted from pywatershed)

These are process-agnostic and sit at the base of the import stack
(model.py builds on them; process.py no longer imports data_io).

Output is the serial writer (one zarr store). The MPI path (ModelMPI in
model.py) does NOT use Output -- it streams output through mpixarray
(set_streaming + iter_time + .mpi.write).
"""

import pathlib as pl
from typing import Any, Dict, List, Union

import numpy as np
import xarray as xr

# Output needs no zarr import of its own -- xarray's zarr backend does the IO.


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
    """Buffered, time-chunked serial writer to a single zarr store.

    Buffers each tracked variable in memory for time_chunk_size steps, then
    APPENDS the chunk to the store along `time` (the first append creates
    the store, with time chunks of time_chunk_size); a partial tail is
    appended at finalize. Peak memory = the per-variable chunk buffers; the
    store is never pre-sized or materialized in memory (see "Prime
    directive: memory" in pws_phoenix/CLAUDE.md). Adapted from pywatershed's
    chunked zarr writer.

    Serial only. The MPI path streams output through mpixarray
    (set_streaming + iter_time + .mpi.write) and does not use this class.
    (Append-along-time is sufficient here BECAUSE the writer is serial;
    concurrent writers would need a pre-sized store + region writes.)
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
                a chunk is appended. Also the store's time chunk size.
            variable_names: Variables to write -- one data_var each in the
                single output store.
            output_store: Path to the (single) zarr store directory; must
                carry the ".zarr" suffix and is used verbatim.
            time_values: 1-D datetime coordinate for the full run (sliced
                per appended chunk).

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
        self.current_time_step = 0
        self._store_created = False

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

    def _initialize_buffers(self) -> None:
        """Initialize numpy arrays to store time chunk data for each variable."""
        for var_name, var_ref in self.variable_refs.items():
            # Buffer: (time_chunk_size, *spatial_dims)
            buffer_shape = (self.time_chunk_size,) + var_ref.shape
            self.data_buffers[var_name] = np.empty(
                buffer_shape, dtype=var_ref.dtype
            )

    def collect_current_timestep(self, time_index: int) -> None:
        """Buffer this step's values; append a full chunk when the buffer
        fills.

        Args:
            time_index: Global time index.
        """
        buffer_index = self.current_time_step % self.time_chunk_size

        for var_name, var_ref in self.variable_refs.items():
            self.data_buffers[var_name][buffer_index] = var_ref.values

        self.current_time_step += 1

        if self.current_time_step % self.time_chunk_size == 0:
            self._append_chunk(self.time_chunk_size)

    def _append_chunk(self, n_valid: int) -> None:
        """Append the first n_valid buffered steps to the store along `time`.
        The first append creates the store (chunk encoding set there); no
        data beyond the chunk itself is ever materialized."""
        chunk_end = self.current_time_step
        chunk_start = chunk_end - n_valid
        data_vars: dict = {}
        encoding: dict = {}
        coords: dict = {"time": self.time_values[chunk_start:chunk_end]}
        for var_name, var_ref in self.variable_refs.items():
            spatial_dim = str(var_ref.dims[0])
            data_vars[var_name] = (
                ["time", spatial_dim],
                self.data_buffers[var_name][:n_valid],
            )
            # Carry the spatial dim-coordinate (e.g. "space") if present.
            if spatial_dim in var_ref.coords and spatial_dim not in coords:
                coords[spatial_dim] = (
                    spatial_dim,
                    var_ref.coords[spatial_dim].values,
                )
            # One chunk spans the full spatial extent; time chunked by buffer.
            encoding[var_name] = {
                "chunks": (self.time_chunk_size,) + var_ref.shape
            }

        ds = xr.Dataset(data_vars, coords=coords)
        if not self._store_created:
            ds.to_zarr(
                self.output_store,
                mode="w",
                encoding=encoding,
                consolidated=False,
            )
            self._store_created = True
        else:
            ds.to_zarr(
                self.output_store,
                append_dim="time",
                consolidated=False,
            )

    def finalize(self) -> None:
        """Append any remaining partial buffer."""
        remaining = self.current_time_step % self.time_chunk_size
        if remaining > 0:
            self._append_chunk(remaining)
