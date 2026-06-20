"""Serial regression for the Upper/Lower toy model via Model (model.py).

Shares `dimensions`, `make_toy_input`, and `compute_answers` with the MPI
regression (see conftest.py). The one toy dataset is either kept in memory or
round-tripped through per-input NetCDF files (the `memory`/`file`
parameterization) and fed to Model; output is a single zarr store.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from model import Model
from process import Process
from processes_concrete import Lower, Upper

# Upper's and Lower's parameter sets (union passed to both; each selects its own)
PARAM_NAMES = ["param_up_0", "param_up_1", "param_low_0", "param_common"]


class TestRegression:
    """Regression tests for the serial Process design (model.py / process.py)."""

    # ============ FIXTURES ============

    @pytest.fixture
    def toy_ds(self, dimensions, make_toy_input):
        """A fresh in-memory copy of the unified toy input per test."""
        return make_toy_input(dimensions)

    @pytest.fixture(params=["memory", "file"])
    def model_inputs(self, request, toy_ds, tmp_path):
        """Process-dict inputs sliced from the one toy dataset.

        `memory` passes in-memory DataArrays/Dataset; `file` writes one NetCDF
        per input (so the serial Model's path-opening + shared-file dedup are
        exercised) and passes the paths.
        """
        if request.param == "memory":
            return {
                "parameters": toy_ds[PARAM_NAMES],
                "forcing_0": toy_ds["forcing_0"],
                "forcing_common": toy_ds["forcing_common"],
                "flow_initial": toy_ds["flow_initial"],
                "storage_initial": toy_ds["storage_initial"],
            }

        data_dir = tmp_path / "toy_model_data"
        data_dir.mkdir(exist_ok=True)
        paths = {
            "parameters": data_dir / "parameters.nc",
            "forcing_0": data_dir / "forcing_0.nc",
            "forcing_common": data_dir / "forcing_common.nc",
            "flow_initial": data_dir / "flow_initial.nc",
            "storage_initial": data_dir / "storage_initial.nc",
        }
        toy_ds[PARAM_NAMES].to_netcdf(paths["parameters"])
        for name in (
            "forcing_0",
            "forcing_common",
            "flow_initial",
            "storage_initial",
        ):
            toy_ds[name].to_netcdf(paths[name])
        return paths

    @pytest.fixture
    def answers(self, toy_ds, dimensions, compute_answers):
        return compute_answers(
            toy_ds["forcing_0"].values,
            toy_ds["flow_initial"].values,
            toy_ds["storage_initial"].values,
            dimensions["n_time"],
            toy_ds["param_up_1"].values,
            dimensions["time"],
        )

    @pytest.fixture
    def control_config(self, tmp_path):
        return {
            "output_var_names": ["flow", "storage_previous"],
            "output_dir": tmp_path / "output",
            "time_chunk_size": 10,
        }

    # ============ TESTS ============

    def test_registry_populated(self):
        """Upper/Lower auto-register in Process._registry on import."""
        assert Process._registry["Upper"] is Upper
        assert Process._registry["Lower"] is Lower

    def test_process_name_in_attrs(self, toy_ds):
        """ds.attrs['process_name'] is set and no callables are stored in attrs."""
        upper = Upper.new(
            parameters=toy_ds[["param_up_0", "param_up_1", "param_common"]],
            forcing_0=toy_ds["forcing_0"][0],
            forcing_common=toy_ds["forcing_common"][0],
            flow_initial=toy_ds["flow_initial"],
        )
        assert upper.attrs["process_name"] == "Upper"
        # attrs must stay callable-free: process.py resolves behavior via
        # Process._registry (keyed by the process_name string), not by stashing
        # advance/calculate in attrs. Callables would also break NetCDF/zarr
        # serialization and pickle/deepcopy of the dataset.
        for val in upper.attrs.values():
            assert not callable(val), f"callable found in attrs: {val}"

    def test_model_regression(
        self, dimensions, model_inputs, control_config, answers
    ):
        """Full regression: run Model, check buffer sharing, numerics, output."""
        process_dict = {
            "upper": {
                "class": Upper,
                "forcing_0": model_inputs["forcing_0"],
                "forcing_common": model_inputs["forcing_common"],
                "flow_initial": model_inputs["flow_initial"],
                "parameters": model_inputs["parameters"],
            },
            "lower": {
                "class": Lower,
                "forcing_common": model_inputs["forcing_common"],
                "storage_initial": model_inputs["storage_initial"],
                "parameters": model_inputs["parameters"],
            },
        }

        dt = np.float64(1.0)
        with Model(process_dict, control_config) as model:
            model.run(dt, np.int32(dimensions["n_time"]))

        # -- buffer sharing (by-reference wiring) --
        # Serial case: internal data are not deleted/closed by finalize.
        assert (
            model.model_dict["upper"]["param_common"].values
            is model.model_dict["lower"]["param_common"].values
        ), "Shared parameter references broken"
        assert (
            model.model_dict["upper"]["forcing_common"].values
            is model.model_dict["lower"]["forcing_common"].values
        ), "Shared forcing references broken"
        assert (
            model.model_dict["upper"]["flow"].values
            is model.model_dict["lower"]["flow"].values
        ), "Shared inter-process variable references broken"

        # -- final in-memory state --
        np.testing.assert_allclose(
            model.model_dict["upper"]["flow"].values,
            answers["expected_flow"][-1, :],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            model.model_dict["upper"]["flow_previous"].values,
            answers["expected_flow_prev"][-1, :],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            model.model_dict["lower"]["storage"].values,
            answers["expected_storage"][-1, :],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            model.model_dict["lower"]["storage_previous"].values,
            answers["expected_storage_prev"][-1, :],
            rtol=1e-12,
        )

        # -- streamed zarr output (full time series) --
        output_ds = xr.open_zarr(
            control_config["output_dir"] / "output.zarr", consolidated=False
        )
        np.testing.assert_allclose(
            output_ds["flow"].values, answers["expected_flow"], rtol=1e-12
        )
        np.testing.assert_allclose(
            output_ds["storage_previous"].values,
            answers["expected_storage_prev"],
            rtol=1e-12,
        )
