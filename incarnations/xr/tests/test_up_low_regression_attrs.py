import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from base import Input
from processes_attrs import Lower, Upper


class TestRegressionAttrs:
    """Regression tests for base_attrs.py process design.

    Data fixtures are identical to TestRegression in test_up_low_regression.py.
    The test method drives the model loop manually, demonstrating the new API:

        upper = Upper(parameters=..., **kwargs)  # returns xr.Dataset
        upper.pws.advance()                       # via accessor
        upper.pws.calculate(dt)                   # via accessor
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
        """Vectorized ground-truth matching the answers fixture in test_up_low_regression."""

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

    # ============ TEST ============

    def test_process_regression(
        self,
        dimensions,
        data_as_memory_or_file,
        answers,
    ):
        """Regression test driving the model loop manually via the .pws accessor.

        Demonstrates the base_attrs.py design:
          - Upper(...) and Lower(...) return xr.Datasets (not Process instances)
          - process["var"] is native Dataset access -- unchanged from base.py
          - process.pws.advance() / process.pws.calculate(dt) via accessor
          - Buffer sharing between processes is verified with `is` assertions
        """
        parameters_data = data_as_memory_or_file["parameters"]
        forcing_data = data_as_memory_or_file["forcing"]
        forcing_common_data = data_as_memory_or_file["forcing_common"]
        flow_ic_data = data_as_memory_or_file["flow_ic"]
        storage_ic_data = data_as_memory_or_file["storage_ic"]

        # Create Input objects for time-varying external forcing.
        input_forcing_0 = Input(forcing_data, read_only=True)
        input_forcing_common = Input(forcing_common_data, read_only=True)

        # Instantiate processes -- each returns an xr.Dataset.
        upper: xr.Dataset = Upper(  # type: ignore[call-arg, assignment]
            parameters=parameters_data,
            forcing_0=input_forcing_0,
            forcing_common=input_forcing_common,
            flow_initial=flow_ic_data,
        )
        lower: xr.Dataset = Lower(  # type: ignore[call-arg, assignment]
            parameters=parameters_data,
            flow=upper["flow"],  # share Upper's flow buffer directly
            forcing_common=input_forcing_common,
            storage_initial=storage_ic_data,
        )

        # ---- time loop using the .pws accessor ----
        dt = np.float64(1.0)
        for _t in range(dimensions["n_time"]):
            # advance all inputs first (in-place [:] update propagates to
            # all process Datasets sharing the same buffer)
            input_forcing_0.advance()
            input_forcing_common.advance()
            # advance then calculate -- same ordering as Model.run()
            upper.pws.advance()
            lower.pws.advance()
            upper.pws.calculate(dt)
            lower.pws.calculate(dt)

        # ---- numerical correctness: final time step ----
        np.testing.assert_allclose(
            upper["flow"].values,
            answers["expected_flow"][-1, :],
            rtol=1e-12,
            err_msg="Upper flow does not match vectorized calculation",
        )
        np.testing.assert_allclose(
            upper["flow_previous"].values,
            answers["expected_flow_prev"][-1, :],
            rtol=1e-12,
            err_msg="Upper flow_previous does not match vectorized calculation",
        )
        np.testing.assert_allclose(
            lower["storage"].values,
            answers["expected_storage"][-1, :],
            rtol=1e-12,
            err_msg="Lower storage does not match vectorized calculation",
        )
        np.testing.assert_allclose(
            lower["storage_previous"].values,
            answers["expected_storage_prev"][-1, :],
            rtol=1e-12,
            err_msg="Lower storage_previous does not match vectorized calculation",
        )

        # Note: buffer-sharing (is) assertions are tested in
        # test_up_low_regression.py via Model, which handles shared-path
        # deduplication.  When processes are driven manually with file inputs
        # each call to Upper/Lower opens its own Dataset, so sharing is not
        # guaranteed without that deduplication layer.
