import pathlib as pl
import sys
from typing import Any, Dict
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

# Import from the base module
sys.path.append(str(pl.Path(__file__).parent.parent))
from base import Input, Model, Output, Process, open_xr


class TestOpenXr:
    """Test the open_xr utility function."""

    def test_open_xr_single_variable_returns_dataarray(self, tmp_path):
        """Test that open_xr returns DataArray for single-variable files."""
        # Create a NetCDF file with a single data variable
        file_path = tmp_path / "single_var.nc"

        data = xr.DataArray(
            [[1, 2, 3], [4, 5, 6]],
            dims=["time", "space"],
            coords={"time": [0, 1], "space": [0, 1, 2]},
        )
        data.to_netcdf(file_path)

        result = open_xr(file_path)

        assert isinstance(result, xr.DataArray)
        xr.testing.assert_equal(result, data)

    def test_open_xr_multiple_variables_returns_dataset(self, tmp_path):
        """Test that open_xr returns Dataset for multi-variable files."""
        # Create a NetCDF file with multiple data variables
        file_path = tmp_path / "multi_var.nc"

        ds = xr.Dataset(
            {
                "temp": (["time", "space"], [[1, 2, 3], [4, 5, 6]]),
                "humidity": (
                    ["time", "space"],
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                ),
            },
            coords={"time": [0, 1], "space": [0, 1, 2]},
        )
        ds.to_netcdf(file_path)

        result = open_xr(file_path)

        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"temp", "humidity"}


class TestInput:
    """Test the Input class."""

    @pytest.fixture
    def sample_dataarray(self):
        """Create a sample DataArray for testing."""
        return xr.DataArray(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # (time, space) format
            dims=["time", "space"],
            coords={"time": [0, 1, 2], "space": [0, 1, 2]},
        )

    @pytest.fixture
    def sample_file(self, sample_dataarray, tmp_path):
        """Create a sample NetCDF file for testing."""
        file_path = tmp_path / "input.nc"
        sample_dataarray.to_netcdf(file_path)
        return file_path

    def test_init_with_dataarray(self, sample_dataarray):
        """Test Input initialization with DataArray."""
        input_obj = Input(sample_dataarray)

        assert input_obj.data is sample_dataarray
        assert input_obj._input_file is None
        assert input_obj._current_index == -1

        # Check initial current_values is NaN with correct shape
        assert input_obj.current_values.shape == (3,)  # space dimension
        assert np.all(np.isnan(input_obj.current_values.values))

    def test_init_with_file(self, sample_file):
        """Test Input initialization with file path."""
        input_obj = Input(sample_file)

        assert isinstance(input_obj.data, xr.DataArray)
        assert input_obj._input_file == sample_file
        assert input_obj._current_index == -1

    def test_init_with_read_only(self, sample_dataarray):
        """Test Input initialization with read_only flag."""
        input_obj = Input(sample_dataarray, read_only=True)

        assert input_obj.data.values.flags.writeable is False

    def test_advance(self, sample_dataarray):
        """Test the advance method."""
        input_obj = Input(sample_dataarray)

        # Initial state
        assert input_obj._current_index == -1

        # First advance
        input_obj.advance()
        assert input_obj._current_index == 0
        np.testing.assert_array_equal(
            input_obj.current_values.values,
            sample_dataarray[input_obj._current_index, :],
        )

        # Second advance
        input_obj.advance()
        assert input_obj._current_index == 1
        np.testing.assert_array_equal(
            input_obj.current_values.values,
            sample_dataarray[input_obj._current_index, :],
        )

        # Third advance
        input_obj.advance()
        assert input_obj._current_index == 2
        np.testing.assert_array_equal(
            input_obj.current_values.values,
            sample_dataarray[input_obj._current_index, :],
        )

    def test_current_values_property(self, sample_dataarray):
        """Test the current_values property."""
        input_obj = Input(sample_dataarray)

        # Before advance - should be NaN
        current = input_obj.current_values
        assert isinstance(current, xr.DataArray)
        assert current.shape == (3,)  # space dimension
        assert np.all(np.isnan(current.values))

        # After advance
        input_obj.advance()
        current = input_obj.current_values
        np.testing.assert_array_equal(current.values, [1, 2, 3])


class MockProcess(Process):
    """Mock Process class for testing."""

    @staticmethod
    def get_parameters():
        return ("param1", "param2")

    @staticmethod
    def get_inputs():
        return ("input1",)

    @staticmethod
    def get_input_outputs():
        return ()

    @staticmethod
    def get_variables():
        return {
            "var1": {
                "dims": ("space",),
                "dtype": np.float64,
                "metadata": {"description": "Test variable 1"},
                "initial": "var1_initial",
            },
            "var2": {
                "dims": ("space",),
                "dtype": np.float64,
                "metadata": {"description": "Test variable 2"},
            },
        }

    @staticmethod
    def _get_private_variables():
        return {
            "private_var": {
                "dims": ("space",),
                "dtype": np.float64,
                "metadata": {"description": "Private variable"},
            }
        }

    def advance(self):
        pass

    def calculate(self, dt):
        pass


class TestProcess:
    """Test the Process base class."""

    @pytest.fixture
    def sample_parameters(self):
        """Create sample parameters Dataset."""
        return xr.Dataset(
            {
                "param1": (["space"], [1.0, 2.0, 3.0]),
                "param2": (["space"], [0.5, 0.6, 0.7]),
                "space": [0, 1, 2],
            }
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input DataArray."""
        return xr.DataArray(
            [[10, 20, 30], [40, 50, 60]],  # (time, space)
            dims=["time", "space"],
            coords={"time": [0, 1], "space": [0, 1, 2]},
        )

    def test_init(self, sample_parameters, sample_input):
        """Test Process initialization."""
        var1_initial = xr.DataArray(
            [100.0, 200.0, 300.0], dims=["space"], coords={"space": [0, 1, 2]}
        )

        input_obj = Input(sample_input)

        process = MockProcess(
            parameters=sample_parameters,
            input1=input_obj.current_values,
            var1_initial=var1_initial,
        )

        # Check that parameters are set and read-only
        assert "param1" in process.data
        assert "param2" in process.data
        np.testing.assert_array_equal(
            process["param1"].values, sample_parameters["param1"].values
        )
        assert process["param1"].values.flags.writeable is False

        # Check that inputs are set
        assert "input1" in process.data
        np.all(np.isnan(process["input1"].values))

        # Check that variables are created
        assert "var1" in process.data
        assert "var2" in process.data
        np.testing.assert_array_equal(
            process["var1"].values, var1_initial.values
        )

        # Check that private variables are created
        assert "private_var" in process.data

    def test_var_from_metadata(self, sample_parameters):
        """Test the _var_from_metadata method."""
        input_obj = Input(xr.DataArray([[1, 2, 3]], dims=["time", "space"]))

        process = MockProcess(
            parameters=sample_parameters,
            input1=input_obj.current_values,
            var1_initial=xr.DataArray([1, 2, 3], dims=["space"]),
        )

        # Test variable with initial value
        var_metadata = {
            "dims": ("space",),
            "dtype": np.float64,
            "metadata": {"description": "Test variable"},
            "initial": "test_initial",
        }
        test_initial = xr.DataArray([10.0, 20.0, 30.0], dims=["space"])

        result = process._var_from_metadata(
            var_metadata, test_initial=test_initial
        )

        assert isinstance(result, xr.DataArray)
        np.testing.assert_equal(result.values, test_initial.values)
        assert result.attrs["description"] == "Test variable"

    def test_getitem_setitem_delitem(self, sample_parameters, sample_input):
        """Test dictionary-like access methods."""
        input_obj = Input(sample_input)

        process = MockProcess(
            parameters=sample_parameters,
            input1=input_obj.current_values,
            var1_initial=xr.DataArray([1, 2, 3], dims=["space"]),
        )

        # Test __getitem__
        param1 = process["param1"]
        assert isinstance(param1, xr.DataArray)
        np.testing.assert_equal(
            param1.values, sample_parameters["param1"].values
        )

        # Test __setitem__
        new_var = xr.DataArray([100, 200, 300], dims=["space"])
        process["new_var"] = new_var
        assert "new_var" in process.data
        np.testing.assert_equal(process["new_var"].values, new_var.values)

        # Test __delitem__
        del process["new_var"]
        assert "new_var" not in process.data

    def test_base_process_abstract_methods(self):
        """Test that Process base class abstract methods raise NotImplementedError."""
        # Test that calling static methods on the base Process class raises NotImplementedError
        with pytest.raises(NotImplementedError):
            Process.get_parameters()

        with pytest.raises(NotImplementedError):
            Process.get_inputs()

        with pytest.raises(NotImplementedError):
            Process.get_input_outputs()

        with pytest.raises(NotImplementedError):
            Process.get_variables()

        with pytest.raises(NotImplementedError):
            Process._get_private_variables()


class TestOutput:
    """Test the Output class."""

    @pytest.fixture
    def sample_time_ref(self):
        """Create a sample time reference array."""
        return np.array([0], dtype=np.int64)

    @pytest.fixture
    def sample_time_datum(self):
        """Create a sample time datum."""
        return np.datetime64("2000-01-01")

    def test_init(self, tmp_path, sample_time_ref, sample_time_datum):
        """Test Output initialization."""
        output_dir = tmp_path / "output"
        variable_names = ["var1", "var2"]

        output = Output(
            time_chunk_size=10,
            variable_names=variable_names,
            output_dir=output_dir,
            current_time_ref=sample_time_ref,
            time_datum=sample_time_datum,
        )

        assert output.time_chunk_size == 10
        assert output.variable_names == variable_names
        assert output.output_dir == output_dir
        assert output_dir.exists()  # Should be created
        assert output.current_time_ref is sample_time_ref
        assert output.time_datum == sample_time_datum
        assert output.current_time_step == 0
        assert output.chunk_start_time == 0

    def test_setup_variable_tracking(
        self, tmp_path, sample_time_ref, sample_time_datum
    ):
        """Test variable tracking setup."""
        output_dir = tmp_path / "output"

        # Create mock model dict
        mock_var1 = xr.DataArray([1.0, 2.0, 3.0], dims=["space"])
        mock_var2 = xr.DataArray([10.0, 20.0, 30.0], dims=["space"])

        mock_process = Mock()
        mock_process.get_variables.return_value = {"var1": {}, "var2": {}}
        mock_process.__getitem__ = (
            lambda self, key: mock_var1 if key == "var1" else mock_var2
        )

        model_dict = {"process1": mock_process}

        output = Output(
            time_chunk_size=5,
            variable_names=["var1", "var2"],
            output_dir=output_dir,
            current_time_ref=sample_time_ref,
            time_datum=sample_time_datum,
        )

        output.setup_variable_tracking(model_dict)  # type: ignore

        # Check that variables are tracked
        assert "var1" in output.variable_refs
        assert "var2" in output.variable_refs
        assert output.process_map["var1"] == "process1"
        assert output.process_map["var2"] == "process1"

        # Check that buffers are initialized
        assert "var1" in output.data_buffers
        assert "var2" in output.data_buffers
        # (time_chunk_size, space)
        assert output.data_buffers["var1"].shape == (5, 3)

        # Check that NetCDF files are created
        assert (output_dir / "var1.nc").exists()
        assert (output_dir / "var2.nc").exists()
        assert output.files_initialized is True

    def test_setup_variable_tracking_missing_variable(
        self, tmp_path, sample_time_ref, sample_time_datum
    ):
        """Test that setup raises error for missing variables."""
        output_dir = tmp_path / "output"

        mock_process = Mock()
        mock_process.get_variables.return_value = {"var1": {}}

        def mock_getitem(self, key):
            if key == "var1":
                return xr.DataArray([1, 2, 3])
            else:
                raise KeyError(f"Variable '{key}' not found")

        mock_process.__getitem__ = mock_getitem

        model_dict = {"process1": mock_process}

        output = Output(
            time_chunk_size=5,
            variable_names=["var1", "missing_var"],
            output_dir=output_dir,
            current_time_ref=sample_time_ref,
            time_datum=sample_time_datum,
        )

        with pytest.raises(
            ValueError, match="Variable 'missing_var' not found"
        ):
            output.setup_variable_tracking(model_dict)  # type: ignore

    def test_collect_timestep(
        self, tmp_path, sample_time_ref, sample_time_datum
    ):
        """Test timestep data collection."""
        output_dir = tmp_path / "output"

        nspace = 3
        ntime = 7
        tv_input_array = xr.DataArray(
            np.arange(ntime * nspace).reshape(ntime, nspace),
            dims=["time", "space"],
            coords={
                "time_coord": ("time", np.arange(ntime)),
                "space_coord": ("space", np.arange(nspace)),
            },
            attrs={},
        )
        tv_data = Input(tv_input_array)

        mock_process = Mock()
        mock_process.get_variables.return_value = {"var1": {}}
        mock_process.__getitem__ = lambda obj, key: tv_data.current_values

        model_dict = {"process1": mock_process}

        time_chunk_size = 4
        output = Output(
            time_chunk_size=time_chunk_size,
            variable_names=["var1"],
            output_dir=output_dir,
            current_time_ref=sample_time_ref,
            time_datum=sample_time_datum,
        )

        output.setup_variable_tracking(model_dict)  # type: ignore

        for tt in range(time_chunk_size):
            tv_data.advance()
            output.collect_timestep(tt)

        assert np.all(
            output.data_buffers["var1"].ravel()
            == np.arange(nspace * (time_chunk_size))
        )

        for tt in range(time_chunk_size, ntime):
            tv_data.advance()
            output.collect_timestep(tt)

        assert np.all(
            output.data_buffers["var1"][0 : ntime - time_chunk_size, :].ravel()
            == (
                np.arange(nspace * (ntime - time_chunk_size))
                + time_chunk_size * nspace
            )
        )

        assert output.data_buffers["var1"].shape == (time_chunk_size, nspace)
        assert output.current_time_step == ntime

    def test_initialize_netcdf_file(
        self, tmp_path, sample_time_ref, sample_time_datum
    ):
        """Test NetCDF file initialization."""
        output_dir = tmp_path / "output"

        # Create a sample variable
        sample_var = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["space"],
            coords={"space_coord": ("space", [0, 1, 2])},
            attrs={"description": "Test variable", "units": "meters"},
        )

        output = Output(
            time_chunk_size=5,
            variable_names=["test_var"],
            output_dir=output_dir,
            current_time_ref=sample_time_ref,
            time_datum=sample_time_datum,
        )

        output.variable_refs["test_var"] = sample_var

        file_path = output_dir / "test_var.nc"
        output._initialize_netcdf_file("test_var", file_path)

        # Check file was created and has correct structure
        assert file_path.exists()

        da = xr.open_dataarray(file_path)

        # Check dimensions
        assert "time" in da.dims
        assert "space" in da.dims
        assert da.sizes["space"] == 3

        # Check variables
        assert "time" in da.coords
        assert "space_coord" in da.coords
        assert "test_var" == da.name

        # Check variable attributes
        assert da.attrs["description"] == "Test variable"
        assert da.attrs["units"] == "meters"

        # Check some encodings / variable attributes
        assert da["time"].encoding["units"] == "days since 2000-01-01 00:00:00"
        assert da.encoding["coordinates"] == "time space_coord"
        da.close()

    def test_finalize_with_remaining_data(
        self, tmp_path, sample_time_ref, sample_time_datum
    ):
        """Test finalize method with remaining data in buffers."""
        output_dir = tmp_path / "output"

        # Create mock variable with matching coordinate size
        mock_var = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["space"],
            coords={"space_coord": ("space", [0, 1, 2])},
            attrs={},
        )

        mock_process = Mock()
        mock_process.get_variables.return_value = {"var1": {}}
        mock_process.__getitem__ = lambda obj, key: mock_var

        model_dict = {"process1": mock_process}

        output = Output(
            time_chunk_size=5,
            variable_names=["var1"],
            output_dir=output_dir,
            current_time_ref=sample_time_ref,
            time_datum=sample_time_datum,
        )

        output.setup_variable_tracking(model_dict)  # type: ignore[arg-type]

        # Manually set some data in buffer (simulating partial chunk)
        expected_data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        output.data_buffers["var1"][0] = expected_data[0, :]
        output.data_buffers["var1"][1] = expected_data[1, :]
        output.current_time_step = 2  # Only 2 steps collected

        # Finalize should write the partial chunk
        output.finalize()

        # Check that data was written to file
        var_file = output_dir / "var1.nc"
        da = xr.open_dataarray(var_file)
        assert da.sizes["time"] == 2  # Only 2 time steps
        np.testing.assert_array_equal(da.values, expected_data)
        da.close()


class TestModel:
    """Test the Model class."""

    @pytest.fixture
    def sample_process_dict(self):
        """Create a sample process dictionary."""
        forcing = xr.DataArray(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
            ],  # (time, space)
            dims=["time", "space"],
            coords={
                "time_coord": ("time", [0, 1, 2, 3, 4]),
                "space_coord": ("space", [0, 1, 2]),
            },
        )

        flow_initial = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["space"],
            coords={"space_coord": ("space", [0, 1, 2])},
        )

        parameters = xr.Dataset(
            {
                "param1": (["space"], [1.0, 2.0, 3.0]),
                "param2": (["space"], [0.5, 0.6, 0.7]),
                "space": [0, 1, 2],
            }
        )

        return {
            "mock_process": {
                "class": MockProcess,
                "parameters": parameters,
                "input1": forcing,
                "var1_initial": flow_initial,
            }
        }

    @pytest.fixture
    def sample_control_config(self, tmp_path):
        """Create a sample control configuration."""
        return {
            "output_var_names": ["var1"],
            "output_dir": tmp_path / "output",
            "time_chunk_size": 10,
        }

    def test_init(self, sample_process_dict, sample_control_config):
        """Test Model initialization."""
        model = Model(sample_process_dict, sample_control_config)

        # Check that model dict is populated
        assert "mock_process" in model.model_dict
        assert isinstance(model.model_dict["mock_process"], MockProcess)

        # Check that output is set up if output_var_names is provided
        assert isinstance(model.output, Output)

        # Check time setup
        assert hasattr(model, "current_time")
        assert hasattr(model, "times")

    def test_init_without_output(self, sample_process_dict, tmp_path):
        """Test Model initialization without output configuration."""
        # No output_var_names or output_dir
        control_config: Dict[str, Any] = {}

        model = Model(sample_process_dict, control_config)

        assert model.output is None

    def test_paths_to_data_proc_dict_and_load_all(self, tmp_path):
        """Test _paths_to_data_proc_dict method and load_all."""
        # Create test files with actual NetCDF data
        forcing_file = tmp_path / "forcing.nc"
        forcing_data = xr.DataArray(
            [[1, 2, 3], [4, 5, 6]],
            dims=["time", "space"],
            coords={
                "time_coord": ("time", [0, 1]),
                "space_coord": ("space", [0, 1, 2]),
            },
        )
        forcing_data.to_netcdf(forcing_file)

        param_file = tmp_path / "params.nc"
        param_data = xr.Dataset(
            {
                "param1": (["space"], [1.0, 2.0, 3.0]),
                "param2": (["space"], [0.5, 0.6, 0.7]),
                "space": [0, 1, 2],
            }
        )
        param_data.to_netcdf(param_file)

        proc_dict = {
            "process1": {
                "class": MockProcess,
                "parameters": param_file,
                "input1": forcing_file,
                "var1_initial": xr.DataArray(
                    [10, 20, 30],
                    dims=["space"],
                    coords={"space_coord": ("space", [0, 1, 2])},
                ),
            }
        }

        # Test basic functionality (load_all=False by default)
        model = Model(proc_dict, {})

        # Check that paths were converted to data
        assert isinstance(model.model_dict["process1"]["param1"], xr.DataArray)
        assert isinstance(model.inputs_dict["input1"], Input)

        # Test with load_all=True via control dict
        control_with_load_all = {"load_all": True}
        model_load_control = Model(proc_dict, control_with_load_all)

        # Check that data was loaded (chunks should be None)
        for var_name in ["param1", "param2"]:
            var_data = model_load_control.model_dict["process1"][var_name]
            assert isinstance(var_data, xr.DataArray)
            assert var_data.chunks is None, (
                f"{var_name} should be loaded (chunks=None)"
            )

        # Test with load_all=True via parameter
        model_load_param = Model(proc_dict, {}, load_all=True)

        # Check that data was loaded (chunks should be None)
        for var_name in ["param1", "param2"]:
            var_data = model_load_param.model_dict["process1"][var_name]
            assert isinstance(var_data, xr.DataArray)
            assert var_data.chunks is None, (
                f"{var_name} should be loaded (chunks=None)"
            )

        # Test that parameter overrides control dict
        control_false = {"load_all": False}
        model_override = Model(proc_dict, control_false, load_all=True)

        # Check that data was loaded (parameter should override control)
        for var_name in ["param1", "param2"]:
            var_data = model_override.model_dict["process1"][var_name]
            assert isinstance(var_data, xr.DataArray)
            assert var_data.chunks is None, (
                f"{var_name} should be loaded (parameter overrides control)"
            )

    def test_run(self, sample_process_dict, sample_control_config):
        """Test the run method with process execution and output collection."""
        model = Model(sample_process_dict, sample_control_config)
        dt = np.float64(1.0)
        n_time_steps = np.int32(3)  # Match the available time steps

        # Mock both process and output methods
        with (
            patch.object(
                model.model_dict["mock_process"], "advance"
            ) as mock_advance,
            patch.object(
                model.model_dict["mock_process"], "calculate"
            ) as mock_calculate,
            patch.object(model.output, "collect_timestep") as mock_collect,
            patch.object(model.output, "finalize") as mock_finalize,
        ):
            model.run(dt, n_time_steps)

            # Check that advance and calculate were called for each time step
            assert mock_advance.call_count == 3
            assert mock_calculate.call_count == 3

            # Check that calculate was called with correct dt
            for call in mock_calculate.call_args_list:
                assert call[0][0] == dt

            # Check that output was collected for each time step
            assert mock_collect.call_count == 3
            mock_finalize.assert_called_once()

            # Check that collect_timestep was called with correct time indices
            expected_calls = [((0,),), ((1,),), ((2,),)]
            assert mock_collect.call_args_list == expected_calls

    def test_get_repeated_paths(self, tmp_path):
        """Test the _get_repeated_paths method with actual repeated paths."""
        # Create test file with actual NetCDF data
        file1 = tmp_path / "file1.nc"

        # Create minimal NetCDF file
        data1 = xr.Dataset(
            {
                "param1": (["space"], [1.0, 2.0, 3.0]),
                "param2": (["space"], [0.5, 0.6, 0.7]),
                "space": [0, 1, 2],
            }
        )
        data1.to_netcdf(file1)

        proc_dict = {
            "process1": {
                "class": MockProcess,
                "parameters": file1,  # Same file used by both processes
                "input1": xr.DataArray(
                    [[1, 2, 3]],
                    dims=["time", "space"],
                    coords={
                        "time_coord": ("time", [0]),
                        "space_coord": ("space", [0, 1, 2]),
                    },
                ),
                "var1_initial": xr.DataArray([10, 20, 30], dims=["space"]),
            },
            "process2": {
                "class": MockProcess,
                "parameters": file1,  # Same file as process1 (repeated)
                "input1": xr.DataArray(
                    [[4, 5, 6]],
                    dims=["time", "space"],
                    coords={
                        "time_coord": ("time", [0]),
                        "space_coord": ("space", [0, 1, 2]),
                    },
                ),
                "var1_initial": xr.DataArray([40, 50, 60], dims=["space"]),
            },
        }

        model = Model(proc_dict, {})

        # Test that repeated paths were detected by checking shared parameter DataArrays
        # Get the intersection of parameters from both processes
        process1_params = set(MockProcess.get_parameters())
        process2_params = set(MockProcess.get_parameters())
        shared_params = process1_params & process2_params

        # For each shared parameter, verify the DataArrays are the same object
        for param_name in shared_params:

            assert (
                model.model_dict["process1"][param_name]
                is model.model_dict["process2"][param_name]
            ), (
                f"Parameter '{param_name}' should be the same DataArray object "
                f"for repeated paths"
            )

    def test_procs_above(self, sample_process_dict, sample_control_config):
        """Test the procs_above method."""
        # For a simple test, just create two independent processes
        # The procs_above method mainly checks process ordering
        model = Model(sample_process_dict, sample_control_config)

        # Test procs_above with a simple case
        procs = model.procs_above("mock_process")

        # Since there's only one process, it should have no dependencies
        assert len(procs) == 0


if __name__ == "__main__":
    pytest.main([__file__])
