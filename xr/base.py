import pathlib as pl
import sys
import warnings
from typing import Any, Dict, List, Tuple, Union

import netCDF4 as nc
import numpy as np
import xarray as xr

parent_dir = (pl.Path("./") / __file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils import timer  # noqa


# TODO: pass through an option to load on open?
# TODO: have get_variables and get_variable_names so a list of var names is
#       easily accessible?
# TODO: why does the time coord need passed around?
# TODO: read the output class in-depth
# TODO: review all TODOs scattered in the code.
# TODO: Output: can we collect the memory over all inputs and processes to
#       estimate total memory usage (will not include local/temp vars), add
#       to this the memory usage of the Output memory buffers. Finally, take
#       a user requested max memory, then solve:
#       time_chunk_size = (max - (process + init)) / buffer_per_time
#       and then round down slightly.
# TODO: what are the keys on the model dict, just from the passed model dict?


def open_xr(
    path: pl.Path, load: bool = False
) -> Union[xr.DataArray, xr.Dataset]:
    """Open a NetCDF file and return the most appropriate xarray object.

    Automatically determines whether to return a Dataset or DataArray based on
    the file contents.
    Args:
        path: Path to the NetCDF file to open.
        load_all: Load all variables in the returned object.

    Returns:
        DataArray if file contains exactly one data variable, otherwise
        Dataset.
    """
    ds = xr.open_dataset(path)
    if len(ds.data_vars) == 1:
        da_ds = ds[list(ds.data_vars)[0]]
    else:
        da_ds = ds

    if load:
        da_ds = da_ds.load()

    return da_ds


class Input:
    """Handles time-varying input data for model processes.

    Manages input data that varies over time, providing functionality to advance
    through time steps and access current values. Can load data from files or
    accept pre-loaded DataArrays. Maintains an internal time index and provides
    the current time slice.

    Attributes:
        data: The complete time-varying input DataArray.
        current_values: The values at the current time step.

    Example:
        >>> from pathlib import Path
        >>> # From file
        >>> forcing = Input(Path("forcing_data.nc"))
        >>> forcing.advance()  # Move to first time step
        >>> current = forcing.current_values  # Get current time slice

        >>> # From DataArray
        >>> data = xr.DataArray(
        ...     np.random.rand(100, 365), dims=['space', 'time']
        ... )
        >>> input_obj = Input(data)
    """

    # TODO: needs a control to find the start time index as the first index
    def __init__(
        self,
        data_or_file: Union[xr.DataArray, pl.Path],
        read_only: bool = False,
        load: bool = False,
    ) -> None:
        """Initialize Input object with data from file or memory.

        Args:
            data_or_file: Either a path to a NetCDF file containing a DataArray,
                or a pre-loaded xarray DataArray. The array should have time
                as one of its dimensions.
            read_only: If True, marks the underlying data as read-only to
                prevent accidental modifications. Default is False.
            load: If True, loads all data into memory. Default is False.

        Note:
            The time index starts at -1, so call advance() to move to the first
            time step before accessing current_values.
        """
        self._input_file: Union[pl.Path, None] = None
        if isinstance(data_or_file, pl.Path):
            self.data = xr.open_dataarray(data_or_file)
            self._input_file = data_or_file
        else:
            self.data = data_or_file

        # <
        if load:
            self.data = self.data.load()
        if read_only:
            self.data.values.flags.writeable = False

        # <
        self._current_index = np.int64(-1)
        self._current_values = np.nan * self.data[0, :]
        return

    def advance(self) -> None:
        """Advance current_values to the next time step."""
        self._current_index += np.int64(1)
        self._current_values[:] = self.data[self._current_index, :]
        return

    @property
    def current_values(self) -> xr.DataArray:
        """Get the data values for the current time step.

        Returns:
            DataArray data at the current time index with shape of the spatial
            dimensions only.

        Note:
            Returns NaN values if advance() has not been called yet.
        """
        return self._current_values


class Process:
    """Base class for modeled processes.

    Abstract base class that provides the framework for implementing model processes.
    Each process manages its own parameters, inputs, and variables within an xarray
    Dataset structure. Subclasses must implement abstract methods to define their
    specific parameters, inputs, and variables.

    A process represents a computational component that:
    - Takes parameters (static configuration) from a Dataset
    - Receives inputs (time-varying or from other processes) from DataArrays
    - Maintains internal variables and state
    - Provides advance() and calculate() methods for time stepping

    Attributes:
        data: xarray Dataset containing all process variables and coordinates.
    """

    def __init__(self, parameters: xr.Dataset, **kwargs: xr.DataArray) -> None:
        """Initialize the process with parameters and input data.

        Sets up the process's internal xarray Dataset with proper coordinates,
        parameters, inputs, and variables. Handles the initialization
        of variable references.

        Args:
            parameters: Dataset containing static parameters for this process.
            **kwargs: To handle input data and initial values. Keys should
            match names returned by get_inputs(), get_input_outputs(), and
            variable initial conditions indicated by its metadata key
            "intitial". Values are DataArrays.
        """
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
        dim_names = list(set(itertools.chain.from_iterable(dim_names)))
        # Strange naming assumption of coords. Could use a mapping from
        # dims to coords somewhere
        coords = {}
        for dd in dim_names:
            coords[f"{dd}_coord"] = (dd, parameters[dd].values)
        # <
        self.data = xr.Dataset(coords=coords)
        # parameters
        for pp in self.get_parameters():
            # in this case we want read-only refs.
            self[pp] = parameters[pp]
            # This seems successful but in
            # test_base.py::TestModel::test_get_repeated_paths, these are
            # copied if shared??
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
        """Create a variable DataArray from metadata specification.

        Initialized DataArrays for process variables based on their metadata
        definitions.

        Args:
            var_meta: Dictionary containing variable metadata with keys:
                - 'dims': tuple of dimension names
                - 'dtype': numpy data type
                - 'metadata': dictionary of attributes
                - 'initial': (optional) name of initial value in kwargs
            **kwargs: Keyword arguments that may contain initial values.

        Returns:
            DataArray of indicated shape and dtype, either filled with fill
            value for the dtype or optionally filled with initial values.
        """
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
        """Access DataArrays from the internal Dataset by variable name."""
        return self.data[name]

    def __setitem__(self, name: str, value: xr.DataArray) -> None:
        """Add or update DataArray in the internal Dataset."""
        self.data[name] = value
        return

    def __delitem__(self, name: str) -> None:
        """Remove DataArray from the internal Dataset."""
        del self.data[name]
        return

    @staticmethod
    def get_parameters() -> Tuple[str, ...]:
        """Return names of parameters required by this process.

        Abstract method that must be implemented by subclasses to specify
        which parameters from the parameter dataset are needed.

        Returns:
            Tuple of parameter names that will be extracted from the
            parameters dataset during initialization.
        """
        raise NotImplementedError()

    @staticmethod
    def get_inputs() -> Tuple[str, ...]:
        """Return names of time-varying inputs required by this process.

        Abstract method that must be implemented by subclasses to specify
        which external time-varying inputs are needed (e.g., forcing data).

        Returns:
            Tuple of input names that must be provided as Input objects
            at initialization or found in variable list of upstream Processes.
        """
        raise NotImplementedError()

    @staticmethod
    def get_input_outputs() -> Tuple[str, ...]:
        """Return names of modifiable time-varying inputs required by this process.

        Abstract method that must be implemented by subclasses to specify
        which external modifiable time-varying inputs are needed.

        Returns:
            Tuple of input names that must be provided as Input objects
            at initialization or found in variable list of upstream Processes.
        """
        raise NotImplementedError()

    @staticmethod
    def get_variables() -> Dict[str, Dict[str, Any]]:
        """Return metadata for public variables of this process.

        Abstract method that must be implemented by subclasses to define
        the variables that this process creates and maintains.

        Returns:
            Dictionary mapping variable names to their metadata dictionaries.
            Each metadata dict should contain:
                - 'dims': tuple of dimension names
                - 'dtype': numpy data type
                - 'metadata': dict of attributes
                - 'initial': (optional) initial value parameter name
        """
        # TODO: improve the inner Dict[str, Any] typehint once the definition
        # is a bit clearer.
        raise NotImplementedError()

    @staticmethod
    def get_var_names() -> Tuple[str, ...]:
        """Return names for public variables of this process.

        Returns:
            Tuple of variable names.
        """
        return tuple(Process.get_variables().keys())

    @staticmethod
    def _get_private_variables() -> Dict[str, Dict[str, Any]]:
        """Return metadata for private/internal variables.

        Abstract method for defining variables used internally by the process
        but not exposed to other processes. Often returns empty dict.

        Returns:
            Dictionary with same structure as get_variables() but for
            internal-use variables.
        """
        # TODO: improve the inner Dict[str, Any] typehint once the definition
        # is a bit clearer.
        raise NotImplementedError()

    def advance(self) -> None:
        raise NotImplementedError()

    def calculate(self, dt: np.float64) -> None:
        raise NotImplementedError()


class Output:
    """Manages time-chunked output of model variables to NetCDF files.

    Appends time chunks of model variables from memory buffers into separate
    NetCDF files The choice of time_chunk_size can minimizes I/O overhead and
    balance memory limitation. Each tracked variable gets its own NetCDF file
    with proper dimensions, coordinates, and metadata.

    This class:
    - Tracks references to specified variables from model processes
    - Buffers data for the specified number of time steps
    - Writes complete chunks to NetCDF files
    - Handles partial chunks at simulation end

    Attributes:
        time_chunk_size: Number of time steps collected before writing.
        variable_names: List of variable names being tracked.
        output_dir: Directory where NetCDF files are written.

    Example:
        >>> output = Output(
        ...     time_chunk_size=100,
        ...     variable_names=['temperature', 'pressure'],
        ...     output_dir=Path('results')
        ... )
        >>> # Used internally by Model class during run()
    """

    # This definition needs to come after Process
    def __init__(
        self,
        time_chunk_size: int,
        variable_names: List[str],
        output_dir: pl.Path,
        current_time_ref: np.ndarray,
        time_datum: np.datetime64,
    ) -> None:
        """Initialize Output manager for time-chunked NetCDF writing.

        Args:
            time_chunk_size: Number of time steps to collect in memory before
                writing to disk. Larger values use more memory but may be more
                efficient for I/O.
            variable_names: List of variable names to track and output. Each
                variable will get its own NetCDF file named {var_name}.nc.
            output_dir: Directory where output files will be created. Will be
                created if it doesn't exist.
            current_time_ref: Reference to array containing current time index.
            time_datum: Reference time for CF convention "days since datum".

        Note:
            This class is somewhat taylored to read Model.model_dict, which
            is simply has key: Process where the available variables in
            each Process are identified by Process.get_variables().
        """
        self.time_chunk_size = time_chunk_size
        self.variable_names = variable_names
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.current_time_ref = current_time_ref
        self.time_datum = time_datum

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
        """Set up tracking of specified variables from model processes.

        Searches through all processes to find the requested variables and
        makes references to them. Initializes memory buffers and NetCDF files.
        (This method must be called after model processes are initialized.)

        Args:
            model_dict: Dictionary of keyed process objects. Each process
                must implement the get_variables() method per it's base class.

        Raises:
            ValueError: If any requested variable is not found in any process.
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

    def collect_timestep(self, time_index: int) -> None:
        """
        Collect current variable values for this time step.

        Args:
            time_index: Global time index
        """
        buffer_index = self.current_time_step % self.time_chunk_size

        # Collect data from all tracked variables
        for var_name, var_ref in self.variable_refs.items():
            self.data_buffers[var_name][buffer_index] = var_ref.values

        self.current_time_step += 1

        # Write chunk when buffer is full
        if buffer_index == self.time_chunk_size - 1:
            self._write_chunk()

    def _write_chunk(self) -> None:
        """Write current data buffer to NetCDF files."""
        chunk_end_time = self.chunk_start_time + self.time_chunk_size

        # Calculate time values as days since datum
        time_values = []
        for i in range(self.time_chunk_size):
            # Calculate days since datum from actual time coordinates
            # Note: This assumes we're writing a complete chunk, so we calculate
            # the time for each step based on the current position in the chunk
            time_values.append(float(self.chunk_start_time + i))

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

                # Write time coordinates as days since datum
                ncfile.variables["time"][
                    time_dim_size : time_dim_size + self.time_chunk_size
                ] = time_values

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
                    ncfile.createDimension(str(dim_name), dim_size)

            # Create time variable
            time_var = ncfile.createVariable("time", "f8", ("time",))
            time_var.units = f"days since {str(self.time_datum)[:10]} 00:00:00"

            # Create coordinate variables and track their names
            coord_names = ["time"]
            for dim_name in var_ref.dims:
                if (
                    dim_name != "time"
                    and f"{dim_name}_coord" in var_ref.coords
                ):
                    coord_var_name = f"{dim_name}_coord"
                    coord_var = ncfile.createVariable(
                        coord_var_name, "f8", (str(dim_name),)
                    )
                    coord_var[:] = var_ref.coords[coord_var_name].values
                    coord_names.append(coord_var_name)

            # Create main variable
            dims_with_time = ("time",) + tuple(str(d) for d in var_ref.dims)
            # Convert numpy dtype to NetCDF4 compatible string
            dtype_str = var_ref.dtype.str
            var_nc = ncfile.createVariable(var_name, dtype_str, dims_with_time)

            # Copy attributes
            for attr_name, attr_val in var_ref.attrs.items():
                setattr(var_nc, attr_name, attr_val)

            # Add coordinates attribute to help xarray identify coordinate variables
            var_nc.coordinates = " ".join(coord_names)

    def finalize(self) -> None:
        """Write any remaining data in buffers and close files."""
        remaining_steps = self.current_time_step % self.time_chunk_size
        if remaining_steps > 0:
            # Calculate time values for remaining steps as days since datum
            time_values = []
            for i in range(remaining_steps):
                current_step = self.chunk_start_time + i
                days_since_datum = float(
                    current_step
                )  # Simple day counting from start
                time_values.append(days_since_datum)

            # Write partial chunk
            for var_name in self.variable_names:
                file_path = self.output_dir / f"{var_name}.nc"

                with nc.Dataset(file_path, "a") as ncfile:
                    time_dim_size = ncfile.dimensions["time"].size

                    # Write only the used portion of buffer
                    ncfile.variables[var_name][
                        time_dim_size : time_dim_size + remaining_steps
                    ] = self.data_buffers[var_name][:remaining_steps]

                    # Write time coordinates as days since datum
                    ncfile.variables["time"][
                        time_dim_size : time_dim_size + remaining_steps
                    ] = time_values


class Model:
    """Orchestrates and executes process-based simulations.

    The Model class is the main simulation engine that coordinates multiple
    processes, manages data flow between them, handles time stepping, and
    optionally manages output. It transforms file paths to in-memory data
    structures and wires processes together based on their input/output
    dependencies.

    Key responsibilities:
    - Load and manage parameter/forcing files
    - Initialize and wire together Process instances
    - Coordinate time stepping across all processes
    - Manage optional output collection and writing

    Design principle: All file I/O happens during initialization. Processes
    operate on in-memory xarray objects during simulation for performance.

    Attributes:
        inputs_dict: Dictionary of Input objects for time-varying input data
        model_dict: Dictionary of initialized process instances
        output: Optional Output manager for writing results


    """

    # TODO: improve typehinting for process_dict and control
    # str, Dict[str, Union[Process, pl.Path, xr.DataArray, xr.Dataset]]

    def __init__(
        self,
        process_dict: Dict[str, Any],
        control: Dict[str, Any],
        load_all: Union[bool, None] = None,
    ) -> None:
        """Initialize Model with process definitions and control settings.

        Constructs the a model simulation by loading data files,
        initializing process instances, wiring their dependencies, and
        optionally setting up output management.
        Args:
            process_dict: Dictionary defining processes and their configuration.
                Keys are user defined, values are dictionaries containing:
                - 'class': Process class to instantiate
                - 'parameters': Path to a NetCDF parameter file or a Dataset.
                - Keys matching process input requirements which are
                  external to the model chain. Values supplied are either a
                  Path to a netCDF file or a DataArray in memory.
                - Keys matching process initial values requirements, with
                  values supplied are either a Path to a netCDF file or a
                  DataArray in memory.
            control: Dictionary containing simulation control settings:
                - 'output_var_names': (optional) List of variables to output
                - 'output_dir': (optional) Directory for output files
                - 'time_chunk_size': (optional) Output chunking size (default: 365)
                - 'load_all': (optional) Whether to load all data into memory
                - to be continued
            load_all: Whether to load all xarray objects into memory. If None,
                checks control dict for 'load_all' key, defaults to False.

        Raises:
            ValueError: If only one of output_var_names or output_dir is
                specified without the other.

        Note:
            When output_var_names and output_dir are provided, an Output
            manager is created to handle time-chunked writing of simulation
            results. If time_chunk_size is not specified, defaults to 365
            with a warning.
        """
        from copy import deepcopy

        self._passed_process_dict = process_dict
        self._process_dict = deepcopy(process_dict)

        # Set load_all option
        if load_all is None:
            self._load_all = control.get("load_all", False)
        else:
            self._load_all = load_all

        self._paths_to_data_proc_dict()

        # wire up the model
        self.model_dict: Dict[str, Process] = {}
        self.inputs_dict: Dict[str, Input] = {}
        self._set_inputs_and_model_dicts()
        del self._process_dict

        self._set_time()

        # Initialize time tracking arrays
        self.current_time_index = np.array([0], dtype=np.int32)
        self.current_time = np.array(
            [self.times[0].values], dtype="datetime64[D]"
        )

        # TODO: make the following a method
        # Setup output tracking if specified in control
        self.output = None
        if "output_var_names" in control or "output_dir" in control:
            if (
                "output_var_names" not in control
                or "output_dir" not in control
            ):
                raise ValueError(
                    "output_var_names and output_dir must noth be specified "
                    "in the control."
                )

            # Check for time_chunk_size, use default if missing
            if "time_chunk_size" in control:
                time_chunk_size = control["time_chunk_size"]
            else:
                time_chunk_size = 365
                warnings.warn(
                    "The time_chunk_size not specified in control dict, using "
                    "default value of 365.",
                    UserWarning,
                )

            # Create Output object with time reference and datum
            self.output = Output(
                time_chunk_size=time_chunk_size,
                variable_names=control["output_var_names"],
                output_dir=control["output_dir"],
                current_time_ref=self.current_time_index,
                time_datum=self.times[0].values,
            )
            self.output.setup_variable_tracking(self.model_dict)

        return

    def _paths_to_data_proc_dict(self, load_all: bool = False) -> None:
        """Convert file paths in process_dict to loaded xarray objects.

        All input paths to Dataset or DataArray without opening files twice,
        repeats are managed by references.

        If inputs come as memory, we dont need to do anything (though that's
        potentially a source of user error).

        Args:
            load_all: load data of all xarray objects instantiated.

        """
        repeated_paths = self._get_repeated_paths(load_all=load_all)
        for proc_name in self._process_dict.keys():
            proc = self._process_dict[proc_name]
            for input_key in proc.keys():
                input_val = proc[input_key]
                if isinstance(input_val, pl.Path):
                    if input_val in repeated_paths.keys():
                        proc[input_key] = repeated_paths[input_val]
                    else:
                        proc[input_key] = open_xr(input_val, load=load_all)

                if input_key == "parameters":
                    for vv in proc[input_key].keys():
                        proc[input_key][vv].values.flags.writeable = False

        # TODO: write test that repeated path has same memory id
        # assert id(self._process_dict["upper"]["parameters"]) == id(
        #     self._process_dict["lower"]["parameters"]
        # )
        return

    def _get_repeated_paths(
        self, load_all: bool = False
    ) -> Dict[pl.Path, Union[xr.DataArray, xr.Dataset]]:
        """Open repeated paths in the process_dict once against the key path.

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
        repeated_paths_data = {
            kk: open_xr(kk, load=load_all) for kk in repeated_paths_list
        }
        return repeated_paths_data

    def _set_inputs_and_model_dicts(self) -> None:
        """Initialize Input and Process objects, wire dependencies.

        Initialize inputs and processes wiring processes to processes above.

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
                    init_dict[ii] = Input(
                        data_or_file,
                        read_only=read_only,
                        load=self._load_all,
                    )
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
        """Set time dimensions from first input for simulation."""
        # TODO: would take start and end times and check for these
        # TODO: would check consistency of this across all inputs
        kk0 = list(self.inputs_dict.keys())[0]
        self.ntime = self.inputs_dict[kk0].data.sizes["time"]
        self.time_index = self.inputs_dict[kk0].data.time
        self.times = self.inputs_dict[kk0].data.time_coord

        # Enforce datetime64[D] resolution for consistent NetCDF output
        if (
            hasattr(self.times, "dtype")
            and self.times.dtype != "datetime64[D]"
        ):
            import warnings

            warnings.warn(
                f"Time coordinate has dtype {self.times.dtype}, expected datetime64[D]. "
                "This may cause issues with NetCDF timedelta decoding.",
                UserWarning,
            )

        return None

    def procs_above(self, proc_name: str) -> List[str]:
        """Return list of processes defined above/before given process name."""
        procs_above = []
        for pp in self._process_dict:
            if proc_name != pp:
                procs_above.append(pp)
            else:
                return procs_above

        raise ValueError("This should be unreachable.")
        return

    def advance(self) -> None:
        """Advance all inputs and processes to next time step."""
        for ii in self.inputs_dict.values():
            ii.advance()
        for pp in self.model_dict.values():
            pp.advance()
        # <
        return

    def calculate(self, dt: np.float64) -> None:
        """Execute calculations for all processes at current time step."""
        for vv in self.model_dict.values():
            vv.calculate(dt)
        # <
        return

    def run(
        self, dt: np.float64, n_steps: np.int32, verbose: bool = False
    ) -> None:
        """Execute simulation for specified time steps and finalize."""
        for tt in range(n_steps):
            # Update both time tracking arrays
            self.current_time_index[0] = tt
            self.current_time[0] = self.times[tt].values

            self.advance()
            self.calculate(dt=dt)

            # Collect output data if output object is provided
            if self.output is not None:
                self.output.collect_timestep(tt)

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
            self.output.finalize()
        return
