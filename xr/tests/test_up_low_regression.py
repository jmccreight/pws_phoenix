import pathlib as pl
import sys
from typing import Any, Dict

import numpy as np
import pytest
import xarray as xr

# Import from the toy model modules
sys.path.append(str(pl.Path(__file__).parent.parent))
from base import Model
from processes import Lower, Upper


class TestRegression:
    """Regression tests with parameterized fixtures for memory vs file initialization."""

    # ============ DIMENSION FIXTURES ============
    @pytest.fixture
    def dimensions(self):
        """Create dimension data for tests."""

        n_years = 1
        n_space = 20
        start_year = 2000
        end_year = start_year + n_years
        start_time = np.datetime64(f"{start_year}-01-01")
        end_time = np.datetime64(f"{end_year}-01-01") - np.timedelta64(1, "D")
        time = np.arange(start_time, end_time, dtype="datetime64[D]")
        n_time = len(time)
        space = np.arange(n_space)

        return {
            "n_years": n_years,
            "n_space": n_space,
            "n_time": n_time,
            "time": time,
            "space": space,
        }

    # ============ PARAMETER FIXTURES ============
    @pytest.fixture
    def parameters_memory(self, dimensions):
        """Create parameters Dataset in memory."""
        return xr.Dataset(
            data_vars=dict(
                param_up_0=(
                    ["space"],
                    np.random.uniform(
                        low=0.1, high=1, size=dimensions["n_space"]
                    ),
                ),
                param_up_1=(
                    ["space", "time"],
                    np.random.uniform(
                        low=0.1,
                        high=1,
                        size=(dimensions["n_space"], dimensions["n_time"]),
                    ),
                ),
                param_low_0=(
                    ["space"],
                    np.random.uniform(
                        low=0.17, high=0.23, size=dimensions["n_space"]
                    ),
                ),
            ),
            coords=dict(
                space_coord=("space", dimensions["space"]),
                time_coord=("time", dimensions["time"]),
            ),
            attrs=dict(description="Flow parameters."),
        )

    @pytest.fixture
    def parameters_file(self, parameters_memory, tmp_path):
        """Write parameters to file and return path."""
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        parameter_file = data_dir / "parameters.nc"
        parameters_memory.to_netcdf(parameter_file)
        return parameter_file

    # ============ FORCING FIXTURES ============
    @pytest.fixture
    def forcing_memory(self, dimensions):
        """Create forcing DataArray in memory."""
        sin_data = np.sin(
            np.arange(
                0,
                2 * np.pi * dimensions["n_years"],
                2 * np.pi * dimensions["n_years"] / dimensions["n_time"],
            )
        )
        shifts = np.random.uniform(
            low=10, high=100, size=dimensions["n_space"]
        )
        forcing_0_data = (
            np.broadcast_to(
                sin_data.transpose(),
                (dimensions["n_space"], dimensions["n_time"]),
            )
            + np.broadcast_to(
                shifts, (dimensions["n_time"], dimensions["n_space"])
            ).transpose()
        )

        return xr.DataArray(
            data=forcing_0_data,
            dims=["space", "time"],
            coords=dict(
                space_coord=("space", dimensions["space"]),
                time_coord=("time", dimensions["time"]),
            ),
            attrs=dict(
                description="Primal forcing.",
                units="parsecs",
            ),
        )

    @pytest.fixture
    def forcing_file(self, forcing_memory, tmp_path):
        """Write forcing to file and return path."""
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        forcing_file = data_dir / "forcing_0.nc"
        forcing_memory.to_netcdf(forcing_file)
        return forcing_file

    # ============ INITIAL CONDITION FIXTURES ============
    @pytest.fixture
    def flow_ic_memory(self, dimensions):
        """Create flow initial conditions DataArray in memory."""
        flow_ic_data = np.random.uniform(
            low=100, high=1000, size=dimensions["n_space"]
        )
        return xr.DataArray(
            data=flow_ic_data,
            dims=["space"],
            coords=dict(space_coord=("space", dimensions["space"])),
            attrs=dict(
                description="Initial flow.",
                units="cumecs",
                time=str(dimensions["time"][0]),
            ),
        )

    @pytest.fixture
    def flow_ic_file(self, flow_ic_memory, tmp_path):
        """Write flow initial conditions to file and return path."""
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        flow_ic_file = data_dir / "flow_ic.nc"
        flow_ic_memory.to_netcdf(flow_ic_file)
        return flow_ic_file

    @pytest.fixture
    def storage_ic_memory(self, dimensions):
        """Create storage initial conditions DataArray in memory."""
        storage_ic_data = np.random.uniform(
            low=100, high=500, size=dimensions["n_space"]
        )
        return xr.DataArray(
            data=storage_ic_data,
            dims=["space"],
            coords=dict(space_coord=("space", dimensions["space"])),
            attrs=dict(
                description="Initial storage.",
                units="quibits",
                time=str(dimensions["time"][0]),
            ),
        )

    @pytest.fixture
    def storage_ic_file(self, storage_ic_memory, tmp_path):
        """Write storage initial conditions to file and return path."""
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        storage_ic_file = data_dir / "storage_ic.nc"
        storage_ic_memory.to_netcdf(storage_ic_file)
        return storage_ic_file

    # ============ PARAMETERIZED DATA FIXTURES ============
    @pytest.fixture(params=["memory", "file"])
    def parameters_data(self, request, parameters_memory, parameters_file):
        """Parameterized fixture returning either memory or file parameters."""
        if request.param == "memory":
            return parameters_memory
        else:
            return parameters_file

    @pytest.fixture(params=["memory", "file"])
    def forcing_data(self, request, forcing_memory, forcing_file):
        """Parameterized fixture returning either memory or file forcing."""
        if request.param == "memory":
            return forcing_memory
        else:
            return forcing_file

    @pytest.fixture(params=["memory", "file"])
    def flow_ic_data(self, request, flow_ic_memory, flow_ic_file):
        """Parameterized fixture returning either memory or file flow IC."""
        if request.param == "memory":
            return flow_ic_memory
        else:
            return flow_ic_file

    @pytest.fixture(params=["memory", "file"])
    def storage_ic_data(self, request, storage_ic_memory, storage_ic_file):
        """Parameterized fixture returning either memory or file storage IC."""
        if request.param == "memory":
            return storage_ic_memory
        else:
            return storage_ic_file

    # ============ CONTROL FIXTURES ============
    @pytest.fixture
    def control_config(self, tmp_path):
        """Create control configuration."""
        output_dir = tmp_path / "output"
        return {
            "output_var_names": ["flow", "storage_previous"],
            "output_dir": output_dir,
            "time_chunk_size": 10,
        }

    # ============ ANSWERS FIXTURE ============
    @pytest.fixture
    def answers(
        self, dimensions, forcing_memory, flow_ic_memory, storage_ic_memory
    ):
        """Compute vectorized expected results once for all parameterized tests."""

        def vectorized_upper_calculation(
            forcing_0_data, flow_initial_data, n_time
        ):
            """Vectorized version of Upper process calculations."""
            # Extract numpy arrays from xarray objects
            if hasattr(forcing_0_data, "values"):
                forcing_0 = forcing_0_data.values
            else:
                forcing_0 = forcing_0_data

            if hasattr(flow_initial_data, "values"):
                flow_initial = flow_initial_data.values
            else:
                flow_initial = flow_initial_data

            flow = np.zeros((len(flow_initial), n_time))
            flow_previous = np.zeros((len(flow_initial), n_time))

            # Time loop matches model.run(): advance() then calculate()
            for t in range(n_time):
                # advance: flow_previous = flow (from previous time step)
                if t == 0:
                    flow_previous[:, t] = (
                        flow_initial  # First advance uses initial
                    )
                else:
                    flow_previous[:, t] = flow[:, t - 1]

                # calculate: flow = flow_previous * 0.95 + forcing_0
                flow[:, t] = flow_previous[:, t] * 0.95 + forcing_0[:, t]

            return flow, flow_previous

        def vectorized_lower_calculation(flow, storage_initial_data, n_time):
            """Vectorized version of Lower process calculations."""
            # Extract numpy arrays from xarray objects
            if hasattr(storage_initial_data, "values"):
                storage_initial = storage_initial_data.values
            else:
                storage_initial = storage_initial_data

            storage = np.zeros((len(storage_initial), n_time))
            storage_previous = np.zeros((len(storage_initial), n_time))

            # Time loop matches model.run(): advance() then calculate()
            for t in range(n_time):
                # advance: storage_previous = storage (from previous time step)
                if t == 0:
                    storage_previous[:, t] = (
                        storage_initial  # First advance uses initial
                    )
                else:
                    storage_previous[:, t] = storage[:, t - 1]

                # calculate: storage = storage_previous * 0.95 + flow * 0.12
                storage[:, t] = (
                    storage_previous[:, t] * 0.95 + flow[:, t] * 0.12
                )

            return storage, storage_previous

        # Compute all expected results once
        expected_flow, expected_flow_prev = vectorized_upper_calculation(
            forcing_memory, flow_ic_memory, dimensions["n_time"]
        )

        expected_storage, expected_storage_prev = vectorized_lower_calculation(
            expected_flow, storage_ic_memory, dimensions["n_time"]
        )

        # Return all results in a dictionary
        return {
            "expected_flow": expected_flow,
            "expected_flow_prev": expected_flow_prev,
            "expected_storage": expected_storage,
            "expected_storage_prev": expected_storage_prev,
        }

    # ============ CONSOLIDATED TEST ============
    def test_model_regression(
        self,
        dimensions,
        parameters_data,
        forcing_data,
        flow_ic_data,
        storage_ic_data,
        control_config,
        answers,
    ):
        """Comprehensive regression test with parameterized memory/file inputs."""
        # Setup process dictionary with parameterized data
        process_dict: Dict[str, Dict[str, Any]] = {
            "upper": {
                "class": Upper,
                "forcing_0": forcing_data,
                "flow_initial": flow_ic_data,
                "parameters": parameters_data,
            },
            "lower": {
                "class": Lower,
                "storage_initial": storage_ic_data,
                "parameters": parameters_data,
            },
        }

        # Create model - this should work with both memory and file inputs
        model = Model(process_dict, control_config)
        dt = np.float64(1.0)

        # Store initial reference IDs for testing
        initial_refs = {
            "upper_flow_id": id(model.model_dict["upper"]["flow"].values),
            "lower_storage_id": id(
                model.model_dict["lower"]["storage"].values
            ),
        }

        # Run model
        model.run(dt, np.int32(dimensions["n_time"]))

        # Get expected results from answers fixture (computed only once)
        expected_flow = answers["expected_flow"]
        expected_flow_prev = answers["expected_flow_prev"]
        expected_storage = answers["expected_storage"]
        expected_storage_prev = answers["expected_storage_prev"]

        # Test in-memory values against vectorized calculations
        np.testing.assert_allclose(
            model.model_dict["upper"]["flow"].values,
            expected_flow[:, -1],
            rtol=1e-12,
            err_msg="Upper flow values don't match vectorized calculation",
        )

        np.testing.assert_allclose(
            model.model_dict["upper"]["flow_previous"].values,
            expected_flow_prev[:, -1],
            rtol=1e-12,
            err_msg="Upper flow_previous values don't match vectorized calculation",
        )

        np.testing.assert_allclose(
            model.model_dict["lower"]["storage"].values,
            expected_storage[:, -1],
            rtol=1e-12,
            err_msg="Lower storage values don't match vectorized calculation",
        )

        np.testing.assert_allclose(
            model.model_dict["lower"]["storage_previous"].values,
            expected_storage_prev[:, -1],
            rtol=1e-12,
            err_msg="Lower storage_previous values don't match vectorized calculation",
        )

        # Test output NetCDF files
        flow_da = xr.open_dataarray(control_config["output_dir"] / "flow.nc")
        storage_prev_da = xr.open_dataarray(
            control_config["output_dir"] / "storage_previous.nc"
        )

        # Check that output files contain expected time series
        # NetCDF files have (time, space) shape, so transpose expected arrays
        np.testing.assert_allclose(
            flow_da.values,
            expected_flow.T,  # Transpose from (space, time) to (time, space)
            rtol=1e-12,
            err_msg="Output flow NetCDF doesn't match vectorized calculation",
        )

        np.testing.assert_allclose(
            storage_prev_da.values,
            expected_storage_prev.T,  # Transpose from (space, time) to (time, space)
            rtol=1e-12,
            err_msg="Output storage_previous NetCDF doesn't match vectorized calculation",
        )

        # Test that references are still valid after run
        assert (
            id(model.model_dict["upper"]["flow"].values)
            == initial_refs["upper_flow_id"]
        ), "Upper flow reference changed during run"

        assert (
            id(model.model_dict["lower"]["storage"].values)
            == initial_refs["lower_storage_id"]
        ), "Lower storage reference changed during run"

        # Test cross-process references (Lower should reference Upper's flow)
        assert id(model.model_dict["lower"]["flow"].values) == id(
            model.model_dict["upper"]["flow"].values
        ), "Lower process doesn't reference Upper's flow values"

        flow_da.close()
        storage_prev_da.close()
