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
    """Regression tests based on main.py functionality."""

    @pytest.fixture
    def setup_data(self, tmp_path):
        """Set up test data similar to main.py."""
        np.random.seed(42)  # For reproducible results

        # Dimensions
        n_years = 4
        n_space = 20

        start_year = 2000
        end_year = start_year + n_years
        start_time = np.datetime64(f"{start_year}-01-01")
        end_time = np.datetime64(f"{end_year}-01-01") - np.timedelta64(1, "D")
        time = np.arange(start_time, end_time, dtype="datetime64[D]")
        n_time = len(time)

        space = np.arange(n_space)

        # Create test data directory
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir()

        # Parameters
        parameter_file = data_dir / "parameters.nc"
        param_ds = xr.Dataset(
            data_vars=dict(
                param_up_0=(
                    ["space"],
                    np.random.uniform(low=0.1, high=1, size=n_space),
                ),
                param_up_1=(
                    ["space", "time"],
                    np.random.uniform(low=0.1, high=1, size=(n_space, n_time)),
                ),
                param_low_0=(
                    ["space"],
                    np.random.uniform(low=0.17, high=0.23, size=(n_space)),
                ),
            ),
            coords=dict(
                space_coord=("space", space),
                time_coord=("time", time),
            ),
            attrs=dict(description="Flow parameters."),
        )
        param_ds.to_netcdf(parameter_file)

        # Forcing
        forcing_0_file = data_dir / "forcing_0.nc"
        sin_data = np.sin(
            np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)
        )
        shifts = np.random.uniform(low=10, high=100, size=n_space)
        forcing_0_data = (
            np.broadcast_to(sin_data.transpose(), (n_space, n_time))
            + np.broadcast_to(shifts, (n_time, n_space)).transpose()
        )
        forcing_0 = xr.DataArray(
            data=forcing_0_data,
            dims=["space", "time"],
            coords=dict(
                space_coord=("space", space),
                time_coord=("time", time),
            ),
            attrs=dict(
                description="Primal forcing.",
                units="parsecs",
            ),
        )
        forcing_0.to_netcdf(forcing_0_file)

        # Initial conditions
        flow_ic_file = data_dir / "flow_ic.nc"
        flow_ic_data = np.random.uniform(low=100, high=1000, size=n_space)
        flow_ic = xr.DataArray(
            data=flow_ic_data,
            dims=["space"],
            coords=dict(space_coord=("space", space)),
            attrs=dict(
                description="Initial flow.",
                units="cumecs",
                time=str(time[0]),
            ),
        )
        flow_ic.to_netcdf(flow_ic_file)

        storage_ic_file = data_dir / "storage_ic.nc"
        storage_ic_data = np.random.uniform(low=100, high=500, size=n_space)
        storage_ic = xr.DataArray(
            data=storage_ic_data,
            dims=["space"],
            coords=dict(space_coord=("space", space)),
            attrs=dict(
                description="Initial storage.",
                units="quibits",
                time=str(time[0]),
            ),
        )
        storage_ic.to_netcdf(storage_ic_file)

        # Setup process dictionary
        process_dict: Dict[str, Dict[str, Any]] = {
            "upper": {
                "class": Upper,
                "forcing_0": forcing_0_file,
                "flow_initial": flow_ic_file,
                "parameters": parameter_file,
            },
            "lower": {
                "class": Lower,
                "storage_initial": storage_ic_file,
                "parameters": parameter_file,
            },
        }

        output_dir = tmp_path / "output"
        control = {
            "output_var_names": ["flow", "storage_previous"],
            "output_dir": output_dir,
            "time_chunk_size": 10,
        }

        return {
            "process_dict": process_dict,
            "control": control,
            "n_time": n_time,
            "n_space": n_space,
            "time": time,
            "space": space,
            "forcing_0_data": forcing_0_data,
            "flow_ic_data": flow_ic_data,
            "storage_ic_data": storage_ic_data,
            "param_ds": param_ds,
            "output_dir": output_dir,
        }

    def vectorized_upper_calculation(self, forcing_0, flow_initial, n_time):
        """Vectorized version of Upper process calculations."""
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

    def vectorized_lower_calculation(self, flow, storage_initial, n_time):
        """Vectorized version of Lower process calculations."""
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
            storage[:, t] = storage_previous[:, t] * 0.95 + flow[:, t] * 0.12

        return storage, storage_previous

    def test_model_regression(self, setup_data):
        """Test complete model run against vectorized calculations."""
        data = setup_data
        dt = np.float64(1.0)

        # Create and run model
        model = Model(data["process_dict"], data["control"])

        # Store initial reference IDs for testing
        initial_refs = {
            "upper_flow_id": id(model.model_dict["upper"]["flow"].values),
            "lower_storage_id": id(
                model.model_dict["lower"]["storage"].values
            ),
        }

        # Run model
        model.run(dt, np.int32(data["n_time"]))

        # Test vectorized calculations
        expected_flow, expected_flow_prev = self.vectorized_upper_calculation(
            data["forcing_0_data"], data["flow_ic_data"], data["n_time"]
        )

        expected_storage, expected_storage_prev = (
            self.vectorized_lower_calculation(
                expected_flow, data["storage_ic_data"], data["n_time"]
            )
        )

        # Test in-memory values against vectorized calculations
        # Note: Model runs for n_time steps, but final values are at last time step
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
        flow_ds = xr.open_dataset(
            data["output_dir"] / "flow.nc", decode_timedelta=False
        )
        storage_prev_ds = xr.open_dataset(
            data["output_dir"] / "storage_previous.nc", decode_timedelta=False
        )

        # Check that output files contain expected time series
        # NetCDF files have (time, space) shape, so transpose expected arrays
        np.testing.assert_allclose(
            flow_ds["flow"].values,
            expected_flow.T,  # Transpose from (space, time) to (time, space)
            rtol=1e-12,
            err_msg="Output flow NetCDF doesn't match vectorized calculation",
        )

        np.testing.assert_allclose(
            storage_prev_ds["storage_previous"].values,
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

        flow_ds.close()
        storage_prev_ds.close()
