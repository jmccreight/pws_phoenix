"""Regression tests for base_attrs2.py -- Upper/Lower toy model via ModelAttrs.

Mirrors test_up_low_regression_attrs.py exactly, with two changes:
  - imports from base_attrs2 / processes_attrs2
  - Upper/Lower are now explicit Process subclasses; advance/calculate are
    instance methods dispatched via the PWS accessor through PWS._registry
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from base_attrs2 import ModelAttrs
from processes_attrs2 import Lower, Upper


class TestRegressionAttrs2:
    """Regression tests for base_attrs2.py process design.

    Identical data fixtures and answer computation as TestRegressionAttrs.
    The test method drives the model loop via ModelAttrs, confirming that the
    PWS._registry dispatch produces numerically identical results to the
    callable-in-attrs approach in base_attrs.py.

        upper = Upper(parameters=..., **kwargs)  # returns xr.Dataset
        upper.pws.advance()                       # via PWS accessor -> registry
        upper.pws.calculate(dt)                   # via PWS accessor -> registry
        upper["flow"]                             # native Dataset access
    """

    # ============ DIMENSION FIXTURES ============

    @pytest.fixture
    def dimensions(self):
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
        return xr.Dataset(
            data_vars=dict(
                param_up_0=(
                    ["space"],
                    np.random.uniform(
                        low=0.1, high=1, size=dimensions["n_space"]
                    ),
                ),
                param_up_1=(
                    ["time", "space"],
                    np.random.uniform(
                        low=0.1,
                        high=1,
                        size=(dimensions["n_time"], dimensions["n_space"]),
                    ),
                ),
                param_low_0=(
                    ["space"],
                    np.random.uniform(
                        low=0.17, high=0.23, size=dimensions["n_space"]
                    ),
                ),
                param_common=(
                    ["space"],
                    np.random.uniform(
                        low=0, high=0, size=dimensions["n_space"]
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
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        parameter_file = data_dir / "parameters.nc"
        parameters_memory.to_netcdf(parameter_file)
        return parameter_file

    # ============ FORCING FIXTURES ============

    @pytest.fixture
    def forcing_memory(self, dimensions):
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
        forcing_0_data = sin_data[:, np.newaxis] + shifts[np.newaxis, :]
        return xr.DataArray(
            data=forcing_0_data,
            dims=["time", "space"],
            coords=dict(
                space_coord=("space", dimensions["space"]),
                time_coord=("time", dimensions["time"]),
            ),
            attrs=dict(description="Primal forcing.", units="parsecs"),
            name="forcing_0",
        )

    @pytest.fixture
    def forcing_file(self, forcing_memory, tmp_path):
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        forcing_file = data_dir / "forcing_0.nc"
        forcing_memory.to_netcdf(forcing_file)
        return forcing_file

    @pytest.fixture
    def forcing_common_memory(self, dimensions):
        forcing_common_data = np.ones(
            (dimensions["n_time"], dimensions["n_space"])
        )
        return xr.DataArray(
            data=forcing_common_data,
            dims=["time", "space"],
            coords=dict(
                space_coord=("space", dimensions["space"]),
                time_coord=("time", dimensions["time"]),
            ),
            attrs=dict(description="Common forcing.", units="parsecs"),
            name="forcing_common",
        )

    @pytest.fixture
    def forcing_common_file(self, forcing_common_memory, tmp_path):
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        forcing_file = data_dir / "forcing_common.nc"
        forcing_common_memory.to_netcdf(forcing_file)
        return forcing_file

    # ============ INITIAL CONDITION FIXTURES ============

    @pytest.fixture
    def flow_ic_memory(self, dimensions):
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
            name="flow_ic",
        )

    @pytest.fixture
    def flow_ic_file(self, flow_ic_memory, tmp_path):
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        flow_ic_file = data_dir / "flow_ic.nc"
        flow_ic_memory.to_netcdf(flow_ic_file)
        return flow_ic_file

    @pytest.fixture
    def storage_ic_memory(self, dimensions):
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
            name="storage_ic",
        )

    @pytest.fixture
    def storage_ic_file(self, storage_ic_memory, tmp_path):
        data_dir = tmp_path / "toy_model_1_data"
        data_dir.mkdir(exist_ok=True)
        storage_ic_file = data_dir / "storage_ic.nc"
        storage_ic_memory.to_netcdf(storage_ic_file)
        return storage_ic_file

    # ============ PARAMETERIZED DATA FIXTURE ============

    @pytest.fixture(params=["memory", "file"])
    def data_as_memory_or_file(
        self,
        request,
        parameters_memory,
        parameters_file,
        forcing_memory,
        forcing_file,
        forcing_common_memory,
        forcing_common_file,
        flow_ic_memory,
        flow_ic_file,
        storage_ic_memory,
        storage_ic_file,
    ):
        if request.param == "memory":
            return {
                "parameters": parameters_memory,
                "forcing": forcing_memory,
                "forcing_common": forcing_common_memory,
                "flow_ic": flow_ic_memory,
                "storage_ic": storage_ic_memory,
            }
        else:
            return {
                "parameters": parameters_file,
                "forcing": forcing_file,
                "forcing_common": forcing_common_file,
                "flow_ic": flow_ic_file,
                "storage_ic": storage_ic_file,
            }

    # ============ ANSWERS FIXTURE ============

    @pytest.fixture
    def answers(
        self, dimensions, forcing_memory, flow_ic_memory, storage_ic_memory
    ):
        """Vectorized ground-truth -- identical to the answers fixture in TestRegressionAttrs."""

        def vectorized_upper_calculation(
            forcing_0_data, flow_initial_data, n_time
        ):
            forcing_0 = (
                forcing_0_data.values
                if hasattr(forcing_0_data, "values")
                else forcing_0_data
            )
            flow_initial = (
                flow_initial_data.values
                if hasattr(flow_initial_data, "values")
                else flow_initial_data
            )
            flow = np.zeros((n_time, len(flow_initial)))
            flow_previous = np.zeros((n_time, len(flow_initial)))
            for t in range(n_time):
                flow_previous[t, :] = (
                    flow_initial if t == 0 else flow[t - 1, :]
                )
                flow[t, :] = flow_previous[t, :] * 0.95 + forcing_0[t, :]
            return flow, flow_previous

        def vectorized_lower_calculation(flow, storage_initial_data, n_time):
            storage_initial = (
                storage_initial_data.values
                if hasattr(storage_initial_data, "values")
                else storage_initial_data
            )
            storage = np.zeros((n_time, len(storage_initial)))
            storage_previous = np.zeros((n_time, len(storage_initial)))
            for t in range(n_time):
                storage_previous[t, :] = (
                    storage_initial if t == 0 else storage[t - 1, :]
                )
                storage[t, :] = (
                    storage_previous[t, :] * 0.95 + flow[t, :] * 0.12
                )
            return storage, storage_previous

        expected_flow, expected_flow_prev = vectorized_upper_calculation(
            forcing_memory, flow_ic_memory, dimensions["n_time"]
        )
        expected_storage, expected_storage_prev = vectorized_lower_calculation(
            expected_flow, storage_ic_memory, dimensions["n_time"]
        )
        return {
            "expected_flow": expected_flow,
            "expected_flow_prev": expected_flow_prev,
            "expected_storage": expected_storage,
            "expected_storage_prev": expected_storage_prev,
        }

    # ============ CONTROL FIXTURE ============

    @pytest.fixture
    def control_config(self, tmp_path):
        output_dir = tmp_path / "output"
        return {
            "output_var_names": ["flow", "storage_previous"],
            "output_dir": output_dir,
            "time_chunk_size": 10,
        }

    # ============ TESTS ============

    def test_registry_populated(self):
        """PWS._registry must contain Upper and Lower after import.
        Registration happens via explicit assignment at the bottom of
        processes_attrs2.py -- no decorator required.
        """
        from base_attrs2 import PWS

        assert "Upper" in PWS._registry, "Upper missing from PWS._registry"
        assert "Lower" in PWS._registry, "Lower missing from PWS._registry"
        assert PWS._registry["Upper"] is Upper
        assert PWS._registry["Lower"] is Lower

    def test_process_name_in_attrs(
        self,
        dimensions,
        parameters_memory,
        forcing_memory,
        forcing_common_memory,
        flow_ic_memory,
        storage_ic_memory,
    ):
        """ds.attrs['process_name'] must be set and no callables in attrs."""
        upper = Upper.new(
            parameters=parameters_memory,
            forcing_0=forcing_memory[0],
            forcing_common=forcing_common_memory[0],
            flow_initial=flow_ic_memory,
        )
        assert upper.attrs["process_name"] == "Upper"
        for val in upper.attrs.values():
            assert not callable(val), f"callable found in attrs: {val}"

    def test_model_regression(
        self,
        dimensions,
        data_as_memory_or_file,
        control_config,
        answers,
    ):
        """Full regression test using ModelAttrs with process_dict."""
        parameters_data = data_as_memory_or_file["parameters"]
        forcing_data = data_as_memory_or_file["forcing"]
        forcing_common_data = data_as_memory_or_file["forcing_common"]
        flow_ic_data = data_as_memory_or_file["flow_ic"]
        storage_ic_data = data_as_memory_or_file["storage_ic"]

        process_dict = {
            "upper": {
                "class": Upper,
                "forcing_0": forcing_data,
                "forcing_common": forcing_common_data,
                "flow_initial": flow_ic_data,
                "parameters": parameters_data,
            },
            "lower": {
                "class": Lower,
                "forcing_common": forcing_common_data,
                "storage_initial": storage_ic_data,
                "parameters": parameters_data,
            },
        }

        dt = np.float64(1.0)
        with ModelAttrs(process_dict, control_config) as model:
            model.run(dt, np.int32(dimensions["n_time"]))

        assert model.model_dict["upper"]["param_common"].values is (
            model.model_dict["lower"]["param_common"].values
        ), "Shared parameter references broken"

        assert model.model_dict["upper"]["forcing_common"].values is (
            model.model_dict["lower"]["forcing_common"].values
        ), "Shared forcing data references broken"

        assert model.model_dict["upper"]["flow"].values is (
            model.model_dict["lower"]["flow"].values
        ), "Shared inter-process variable references broken"

        expected_flow = answers["expected_flow"]
        expected_flow_prev = answers["expected_flow_prev"]
        expected_storage = answers["expected_storage"]
        expected_storage_prev = answers["expected_storage_prev"]

        np.testing.assert_allclose(
            model.model_dict["upper"]["flow"].values,
            expected_flow[-1, :],
            rtol=1e-12,
            err_msg="Upper flow does not match vectorized calculation",
        )
        np.testing.assert_allclose(
            model.model_dict["upper"]["flow_previous"].values,
            expected_flow_prev[-1, :],
            rtol=1e-12,
            err_msg="Upper flow_previous does not match vectorized calculation",
        )
        np.testing.assert_allclose(
            model.model_dict["lower"]["storage"].values,
            expected_storage[-1, :],
            rtol=1e-12,
            err_msg="Lower storage does not match vectorized calculation",
        )
        np.testing.assert_allclose(
            model.model_dict["lower"]["storage_previous"].values,
            expected_storage_prev[-1, :],
            rtol=1e-12,
            err_msg="Lower storage_previous does not match vectorized calculation",
        )

        flow_da = xr.load_dataarray(control_config["output_dir"] / "flow.nc")
        storage_prev_da = xr.load_dataarray(
            control_config["output_dir"] / "storage_previous.nc"
        )
        np.testing.assert_allclose(
            flow_da.values,
            expected_flow,
            rtol=1e-12,
            err_msg="Output flow NetCDF does not match vectorized calculation",
        )
        np.testing.assert_allclose(
            storage_prev_da.values,
            expected_storage_prev,
            rtol=1e-12,
            err_msg=(
                "Output storage_previous NetCDF does not match "
                "vectorized calculation"
            ),
        )
