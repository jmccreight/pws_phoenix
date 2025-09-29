import pathlib as pl
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import netCDF4 as nc
import numpy as np
import xarray as xr

parent_dir = (pl.Path("./") / __file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils import timer  # noqa


# TODO is there a fast index order in python, does it matter?
# TODO: pass through an option to load on open?
# TODO: have get_variables and get_variable_names so a list of var names is
#       easily accessible?
# TODO: why does the time coord need passed around?


def open_xr(path: pl.Path) -> Union[xr.DataArray, xr.Dataset]:
    """Return appropriate DataArray or Dataset given a file.

    Args:
        path: The path of the file to open.
    """
    ds = xr.open_dataset(path)
    if len(ds.data_vars) == 1:
        return ds[list(ds.data_vars)[0]]
    else:
        return ds


class Input:
    # TODO: needs a control to find the start time index as the first index
    def __init__(
        self,
        data_or_file: Union[xr.DataArray, pl.Path],
        read_only: bool = False,
    ) -> None:
        if isinstance(data_or_file, pl.Path):
            self.data = xr.open_dataarray(data_or_file)
            self._input_file = data_or_file
        else:
            self.data = data_or_file
            self._input_file = None
        # <
        if read_only:
            self.data.values.flags.writeable = True
            # self.data.values.flags.writeable = False  # restore
        # <
        self._current_index = np.int64(-1)
        self._current_values = np.nan * self.data[:, 0]
        return

    def advance(self) -> None:
        self._current_index += np.int64(1)
        self._current_values[:] = self.data[:, self._current_index]
        return

    @property
    def current_values(self) -> xr.DataArray:
        return self._current_values

    @staticmethod
    def from_file(
        file_path: pl.Path, var_name: Optional[str] = None
    ) -> "Input":
        if var_name is None:
            return Input(xr.open_dataarray(file_path))
        else:
            return Input(xr.open_dataset(file_path)[var_name])


class Output:
    def __init__(
        self,
        time_chunk_size: int,
        variable_names: List[str],
        output_dir: pl.Path = pl.Path("output"),
    ) -> None:
        """
        Output class for writing model variables to NetCDF files in time chunks.

        Args:
            time_chunk_size: Number of time steps to collect before writing to file
            variable_names: List of variable names to track and output
            output_dir: Directory to write output files (default: "output")
        """
        self.time_chunk_size = time_chunk_size
        self.variable_names = variable_names
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Track variable references and data
        self.variable_refs: Dict[str, xr.DataArray] = {}
        self.process_map: Dict[str, str] = {}  # var_name -> process_name
        self.data_buffers: Dict[str, np.ndarray] = {}
        self.current_time_step = 0
        self.chunk_start_time = 0

        # File handles for appending
        self.file_handles: Dict[str, nc.Dataset] = {}
        self.files_initialized = False

    def setup_variable_tracking(self, model_dict: Dict[str, Process]) -> None:
        """
        Find and store references to requested variables from model processes.

        Args:
            model_dict: Dictionary of process_name -> process_object
        """
        for var_name in self.variable_names:
            found = False
            for process_name, process_obj in model_dict.items():
                if var_name in process_obj.get_variables():
                    self.variable_refs[var_name] = process_obj[var_name]
                    self.process_map[var_name] = process_name
                    found = True
                    break

            if not found:
                raise ValueError(
                    f"Variable '{var_name}' not found in any process"
                )

        # Initialize data buffers
        self._initialize_buffers()

        # Initialize NetCDF files upfront
        for var_name in self.variable_names:
            file_path = self.output_dir / f"{var_name}.nc"
            self._initialize_netcdf_file(var_name, file_path)

        self.files_initialized = True

    def _initialize_buffers(self) -> None:
        """Initialize numpy arrays to store time chunk data for each variable."""
        for var_name, var_ref in self.variable_refs.items():
            # Create buffer: (time_chunk_size, *spatial_dims)
            spatial_shape = var_ref.shape
            buffer_shape = (self.time_chunk_size,) + spatial_shape
            self.data_buffers[var_name] = np.empty(
                buffer_shape, dtype=var_ref.dtype
            )

    def collect_timestep(
        self, time_index: int, time_coord: np.datetime64
    ) -> None:
        """
        Collect current variable values for this time step.

        Args:
            time_index: Global time index
            time_coord: Time coordinate value
        """
        buffer_index = self.current_time_step % self.time_chunk_size

        # Collect data from all tracked variables
        for var_name, var_ref in self.variable_refs.items():
            self.data_buffers[var_name][buffer_index] = var_ref.values

        self.current_time_step += 1

        # Write chunk when buffer is full
        if buffer_index == self.time_chunk_size - 1:
            self._write_chunk(time_coord)

    def _write_chunk(self, time_coord: np.datetime64) -> None:
        """Write current data buffer to NetCDF files."""
        chunk_end_time = self.chunk_start_time + self.time_chunk_size

        for var_name in self.variable_names:
            file_path = self.output_dir / f"{var_name}.nc"

            # Write data to file
            with nc.Dataset(file_path, "a") as ncfile:
                # Get current time dimension size
                time_dim_size = ncfile.dimensions["time"].size

                # Write time chunk data
                ncfile.variables[var_name][
                    time_dim_size : time_dim_size + self.time_chunk_size
                ] = self.data_buffers[var_name]

                # Write time coordinates if available
                if (
                    hasattr(time_coord, "__len__")
                    and len(time_coord) >= chunk_end_time
                ):
                    ncfile.variables["time"][
                        time_dim_size : time_dim_size + self.time_chunk_size
                    ] = time_coord[self.chunk_start_time : chunk_end_time]

        self.chunk_start_time = chunk_end_time

    def _initialize_netcdf_file(
        self, var_name: str, file_path: pl.Path
    ) -> None:
        """Initialize NetCDF file structure for a variable."""
        var_ref = self.variable_refs[var_name]

        with nc.Dataset(file_path, "w") as ncfile:
            # Create dimensions
            ncfile.createDimension("time", None)  # Unlimited dimension

            for dim_name, dim_size in zip(var_ref.dims, var_ref.shape):
                if dim_name != "time":  # Skip time dimension
                    ncfile.createDimension(dim_name, dim_size)

            # Create time variable
            time_var = ncfile.createVariable("time", "f8", ("time",))
            time_var.units = "days"
            time_var.calendar = "standard"

            # Create coordinate variables
            for dim_name in var_ref.dims:
                if (
                    dim_name != "time"
                    and f"{dim_name}_coord" in var_ref.coords
                ):
                    coord_var = ncfile.createVariable(
                        f"{dim_name}_coord", "f8", (dim_name,)
                    )
                    coord_var[:] = var_ref.coords[f"{dim_name}_coord"].values

            # Create main variable
            dims_with_time = ("time",) + var_ref.dims
            var_nc = ncfile.createVariable(
                var_name, var_ref.dtype, dims_with_time
            )

            # Copy attributes
            for attr_name, attr_val in var_ref.attrs.items():
                setattr(var_nc, attr_name, attr_val)

    def finalize(self, time_coord: np.datetime64 = None) -> None:
        """Write any remaining data in buffers and close files."""
        remaining_steps = self.current_time_step % self.time_chunk_size
        if remaining_steps > 0:
            # Write partial chunk
            for var_name in self.variable_names:
                file_path = self.output_dir / f"{var_name}.nc"

                with nc.Dataset(file_path, "a") as ncfile:
                    time_dim_size = ncfile.dimensions["time"].size

                    # Write only the used portion of buffer
                    ncfile.variables[var_name][
                        time_dim_size : time_dim_size + remaining_steps
                    ] = self.data_buffers[var_name][:remaining_steps]

                    if hasattr(time_coord, "__len__"):
                        end_idx = self.chunk_start_time + remaining_steps
                        if len(time_coord) >= end_idx:
                            ncfile.variables["time"][
                                time_dim_size : time_dim_size + remaining_steps
                            ] = time_coord[self.chunk_start_time : end_idx]


class Process:
    def __init__(
        self, parameters: xr.Dataset, **kwargs: Union[xr.DataArray, xr.Dataset]
    ) -> None:
        import itertools

        # add parameters, inputs, input_outputs, and public variables
        # Note: It is somewhat strange that inputs of parameters are treated
        #       coming from a dataset while inputs and input_outputs are
        #       treated as coming from dataarrays. Conceptually, it makes some
        #       sense for the following reasons:
        #       1. parameters are not time varying
        #       2. parameters generally have somewhat less impact on output
        #       3. parameters are "endogenous" or internal, while inputs
        #          are external and might be coming from various other places/
        dim_names = [vv["dims"] for vv in self.get_variables().values()]
        dim_names = set(itertools.chain.from_iterable(dim_names))
        # Strange naming assumption of coords. Could use a mapping from
        # dims to coords somewhere
        coords = {}
        for dd in dim_names:
            coords[f"{dd}_coord"] = (dd, parameters[dd].values)
        # <
        self.data = xr.Dataset(coords=coords)
        # parameters
        for pp in self.get_parameters():
            # in this case we want read-only copies??
            self[pp] = parameters[pp]
            self[pp].values.flags.writeable = False
        for ii in self.get_inputs():
            # Input object set on self?4
            if isinstance(kwargs[ii], Input):
                self[ii] = kwargs[ii].current_values
                assert id(self[ii].values) == id(
                    kwargs[ii].current_values.values
                )
            else:
                self[ii] = kwargs[ii]
                assert id(self[ii].values) == id(kwargs[ii].values)

            # TODO write a test for the above references
        for oo in self.get_input_outputs():
            self[oo] = kwargs[oo].current_values
            assert id(self[oo].values) == id(kwargs[oo].current_values.values)
        for kk, vv in self.get_variables().items():
            self[kk] = self._var_from_metadata(vv, **kwargs)
        for kk, vv in self._get_private_variables().items():
            self[kk] = self._var_from_metadata(vv)
        # <
        return

    def _var_from_metadata(
        self,
        var_meta: Dict[str, Any],
        **kwargs: Union[xr.DataArray, xr.Dataset],
    ) -> xr.DataArray:
        # TODO move this map to constants somewhere
        fill_value_map = {np.float64: np.nan}
        sizes = self.data.sizes
        dims = tuple([sizes[dd] for dd in var_meta["dims"]])
        da = xr.DataArray(
            data=np.full(
                dims,
                fill_value_map[var_meta["dtype"]],
                dtype=var_meta["dtype"],
            ),
            dims=var_meta["dims"],
            attrs=var_meta["metadata"],
        )
        if (
            "initial" in var_meta.keys()
            and var_meta["initial"] in kwargs.keys()
        ):
            da[:] = kwargs[var_meta["initial"]]
        # <
        return da

    def __getitem__(self, name: str) -> xr.DataArray:
        # TODO: may want to use getattr to get other properties?
        return self.data[name]

    def __setitem__(self, name: str, value: xr.DataArray) -> None:
        self.data[name] = value  # [name]
        return

    def __delitem__(self, name: str) -> None:
        del self.data[name]
        return

    @staticmethod
    def get_parameters() -> Tuple[str, ...]:
        raise NotImplementedError()

    @staticmethod
    def get_inputs() -> Tuple[str, ...]:
        raise NotImplementedError()

    @staticmethod
    def get_input_outputs() -> Tuple[str, ...]:
        raise NotImplementedError()

    @staticmethod
    def get_variables() -> Dict[str, Dict[str, Any]]:
        # TODO: improve the inner Dict[str, Any] typehint once the definition
        # is a bit clearer.
        raise NotImplementedError()

    @staticmethod
    def _get_private_variables() -> Dict[str, Dict[str, Any]]:
        # TODO: improve the inner Dict[str, Any] typehint once the definition
        # is a bit clearer.
        raise NotImplementedError()


class Model:
    """
    Principle: the model should transform all paths in to datasets for
    parameters and dataarrays for all other fields. No process should take
    paths.

    # TODO: improve typehinting for control
    """

    def __init__(
        self,
        process_dict: Dict[
            str, Dict[str, Union[Process, pl.Path, xr.DataArray, xr.Dataset]]
        ],
        control: Dict[str, Any],
    ) -> None:
        from copy import deepcopy

        self._passed_process_dict = process_dict
        self._process_dict = deepcopy(process_dict)

        self._paths_to_data_proc_dict()

        # wire up the model
        self.model_dict = {}
        self.inputs_dict = {}
        self._set_inputs_and_model_dicts()

        self._set_time()

        # TODO: make the following a method
        # Setup output tracking if specified in control
        self.output = None
        if "output_var_names" in control:
            variable_names = control["output_var_names"]

            # Check for time_chunk_size, use default if missing
            if "time_chunk_size" in control:
                time_chunk_size = control["time_chunk_size"]
            else:
                time_chunk_size = 365
                warnings.warn(
                    "The time_chunk_size not specified in control dict, using default value of 365.",
                    UserWarning,
                )

            # Create Output object and setup variable tracking
            self.output = Output(
                time_chunk_size=time_chunk_size, variable_names=variable_names
            )
            self.output.setup_variable_tracking(self.model_dict)

        return

    def _paths_to_data_proc_dict(self) -> None:
        """All input paths to Dataset or DataArray without opening files twice.

        If inputs come as memory, we dont need to do anything (though that's
        potentially a source of user error)

        """
        repeated_paths = self._get_repeated_paths()
        for proc_name in self._process_dict.keys():
            proc = self._process_dict[proc_name]
            for input_key in proc.keys():
                input_val = proc[input_key]
                if isinstance(input_val, pl.Path):
                    if input_val in repeated_paths.keys():
                        proc[input_key] = repeated_paths[input_val]
                    else:
                        proc[input_key] = open_xr(input_val)

        # TODO: write test that repeated path has same memory id
        # assert id(self._process_dict["upper"]["parameters"]) == id(
        #     self._process_dict["lower"]["parameters"]
        # )
        return

    def _get_repeated_paths(
        self,
    ) -> Dict[pl.Path, Union[xr.DataArray, xr.Dataset]]:
        """Open repeated paths in the process_dict once key against path.

        This is intended to work with parameter files or forcing files specified
        across multiple processes.
        """
        from collections import Counter

        flat_proc_dict_paths = []
        for outer_val in self._process_dict.values():
            for inner_val in outer_val.values():
                if isinstance(inner_val, pl.Path):
                    flat_proc_dict_paths.append(inner_val)

        repeated_paths_list = [
            kk for kk, vv in Counter(flat_proc_dict_paths).items() if vv > 1
        ]
        repeated_paths_data = {kk: open_xr(kk) for kk in repeated_paths_list}
        return repeated_paths_data

    def _set_inputs_and_model_dicts(self) -> None:
        """Initialize inputs and processes to processes above.

        The inputs are in self.inputs_dict and the wired model is in
        self.model_dict.
        """
        for kk, vv in self._process_dict.items():
            # get the process dict with the classes removed
            init_dict = {kkk: vvv for kkk, vvv in vv.items() if kkk != "class"}

            inputs_req = vv["class"].get_inputs()
            input_outputs_req = vv["class"].get_input_outputs()
            all_inputs = inputs_req + input_outputs_req
            for ii in all_inputs:
                if ii in init_dict.keys():
                    # combine all inputs/input_outputs into a dict that will be
                    # advanced
                    data_or_file = init_dict[ii]
                    if ii in inputs_req:
                        read_only = True
                    else:
                        raise ValueError("This should not happen from file.")
                    # <
                    init_dict[ii] = Input(data_or_file, read_only=read_only)
                    self.inputs_dict[ii] = init_dict[ii]
                    assert init_dict[ii].data is self.inputs_dict[ii].data
                    del data_or_file

                else:
                    # inputs not in init_dict need to be found elsewhere in
                    # the processes above
                    for pp in self.procs_above(kk):
                        if ii in self.model_dict[pp].get_variables():
                            init_dict[ii] = self.model_dict[pp][ii]

                # <<<
                self.model_dict[kk] = vv["class"](**init_dict)
                # TODO: test refs across processes

        return

    def _set_time(self) -> None:
        # TODO: would take start and end times and check for these
        # TODO: would check consistency of this across all inputs
        kk0 = list(self.inputs_dict.keys())[0]
        self.ntime = self.inputs_dict[kk0].data.sizes["time"]
        self.time_index = self.inputs_dict[kk0].data.time
        self.times = self.inputs_dict[kk0].data.time_coord
        return None

    def procs_above(self, proc_name: str) -> List[str]:
        procs_above = []
        for pp in self._process_dict:
            if proc_name != pp:
                procs_above.append(pp)
            else:
                return procs_above

        raise ValueError("This should be unreachable.")
        return

    def advance(self) -> None:
        for vv in self.inputs_dict.values():
            vv.advance()
        for vv in self.model_dict.values():
            vv.advance()

    def calculate(self, dt: np.float64) -> None:
        for vv in self.model_dict.values():
            vv.calculate(dt)

    def run(
        self, dt: np.float64, n_steps: np.int32, verbose: bool = False
    ) -> None:
        for tt in range(n_steps):
            self.advance()
            self.calculate(dt=dt)

            # Collect output data if output object is provided
            if self.output is not None:
                self.output.collect_timestep(tt, self.times[tt].values)

            if verbose:
                print(f"{tt=}")
                print(f"{self.inputs_dict['forcing_0'].current_values=}")
                print(f"{self.model_dict['upper']['forcing_0']=}")
                print(f"{self.model_dict['upper']['flow']=}")
                print(f"{self.model_dict['lower']['flow']=}")
                print(f"{self.model_dict['lower']['storage']=}")

        # Finalize output if provided
        # TODO: potentially separate to a finalize method.
        if self.output is not None:
            self.output.finalize(
                self.times if hasattr(self, "times") else None
            )
        return
