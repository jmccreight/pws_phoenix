"""Unit tests for base_attrs.py -- DataArrayMeta, @process decorator, PWSAccessor, and ModelAttrs."""

import pathlib as pl
import sys
from typing import Any, Dict

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from base import Input, Output
from base_attrs import (
    DataArrayMeta,
    ModelAttrs,
    _dict_of_kind,
    _keys_of_kind,
    _make_process,
    process,
)

# ---------------------------------------------------------------------------
# Shared mock spec
# ---------------------------------------------------------------------------


@process
class MockSpec:
    param1 = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="param1",
    )
    param2 = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="param2",
    )
    input1 = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="input1",
    )
    var1 = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="var1",
        initial="var1_initial",
    )
    var2 = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="var2",
    )

    @staticmethod
    def advance(ds: xr.Dataset) -> None:
        pass

    @staticmethod
    def calculate(ds: xr.Dataset, dt: np.float64) -> None:
        pass


# ---------------------------------------------------------------------------
# Shared fixtures (module-level so all classes can use them)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_parameters():
    return xr.Dataset(
        {
            "param1": (["space"], [1.0, 2.0, 3.0]),
            "param2": (["space"], [0.5, 0.6, 0.7]),
            "space": [0, 1, 2],
        }
    )


@pytest.fixture
def sample_input():
    return xr.DataArray(
        [[10, 20, 30], [40, 50, 60]],
        dims=["time", "space"],
        coords={
            "time_coord": ("time", [0, 1]),
            "space_coord": ("space", [0, 1, 2]),
        },
    )


@pytest.fixture
def var1_initial():
    return xr.DataArray(
        [100.0, 200.0, 300.0],
        dims=["space"],
        coords={"space_coord": ("space", [0, 1, 2])},
    )


@pytest.fixture
def process_ds(sample_parameters, sample_input, var1_initial):
    """A fully constructed MockSpec process Dataset."""
    input_obj = Input(sample_input)
    return _make_process(
        MockSpec,
        sample_parameters,
        input1=input_obj,
        var1_initial=var1_initial,
    )


# ---------------------------------------------------------------------------
# TestDataArrayMeta
# ---------------------------------------------------------------------------


class TestDataArrayMeta:
    def test_required_fields(self):
        meta = DataArrayMeta(
            kind="parameter", dims=("space",), dtype=np.float64
        )
        assert meta.kind == "parameter"
        assert meta.dims == ("space",)
        assert meta.dtype is np.float64
        assert meta.description == ""
        assert meta.initial is None

    def test_optional_fields(self):
        meta = DataArrayMeta(
            kind="variable",
            dims=("time", "space"),
            dtype=np.float64,
            description="my var",
            initial="my_ic",
        )
        assert meta.description == "my var"
        assert meta.initial == "my_ic"

    def test_all_valid_kinds(self):
        for kind in ("parameter", "input", "mutable_input", "variable"):
            meta = DataArrayMeta(kind=kind, dims=("space",), dtype=np.float64)  # type: ignore[arg-type]
            assert meta.kind == kind

    def test_multi_dim(self):
        meta = DataArrayMeta(
            kind="parameter", dims=("time", "space"), dtype=np.float64
        )
        assert meta.dims == ("time", "space")


# ---------------------------------------------------------------------------
# TestIntrospectionHelpers
# ---------------------------------------------------------------------------


class TestIntrospectionHelpers:
    def test_keys_of_kind_parameter(self):
        assert _keys_of_kind(MockSpec, "parameter") == ("param1", "param2")

    def test_keys_of_kind_input(self):
        assert _keys_of_kind(MockSpec, "input") == ("input1",)

    def test_keys_of_kind_mutable_input(self):
        assert _keys_of_kind(MockSpec, "mutable_input") == ()

    def test_keys_of_kind_variable(self):
        assert _keys_of_kind(MockSpec, "variable") == ("var1", "var2")

    def test_dict_of_kind_parameter(self):
        result = _dict_of_kind(MockSpec, "parameter")
        assert set(result.keys()) == {"param1", "param2"}
        assert all(isinstance(v, DataArrayMeta) for v in result.values())

    def test_dict_of_kind_variable(self):
        result = _dict_of_kind(MockSpec, "variable")
        assert set(result.keys()) == {"var1", "var2"}
        assert result["var1"].initial == "var1_initial"
        assert result["var2"].initial is None


# ---------------------------------------------------------------------------
# TestProcessDecorator
# ---------------------------------------------------------------------------


class TestProcessDecorator:
    def test_get_parameters_classmethod(self):
        assert MockSpec.get_parameters() == ("param1", "param2")  # type: ignore[attr-defined]

    def test_get_inputs_classmethod(self):
        assert MockSpec.get_inputs() == ("input1",)  # type: ignore[attr-defined]

    def test_get_mutable_inputs_classmethod(self):
        assert MockSpec.get_mutable_inputs() == ()  # type: ignore[attr-defined]

    def test_get_var_names_classmethod(self):
        assert MockSpec.get_var_names() == ("var1", "var2")  # type: ignore[attr-defined]

    def test_get_variables_classmethod(self):
        result = MockSpec.get_variables()  # type: ignore[attr-defined]
        assert isinstance(result, dict)
        assert set(result.keys()) == {"var1", "var2"}
        assert all(isinstance(v, DataArrayMeta) for v in result.values())

    def test_instantiation_returns_dataset(
        self, sample_parameters, sample_input, var1_initial
    ):
        """Calling a @process class returns xr.Dataset, not a MockSpec."""
        input_obj = Input(sample_input)
        ds = _make_process(
            MockSpec,
            sample_parameters,
            input1=input_obj,
            var1_initial=var1_initial,
        )
        assert isinstance(ds, xr.Dataset)


# ---------------------------------------------------------------------------
# TestMakeProcess
# ---------------------------------------------------------------------------


class TestMakeProcess:
    def test_parameters_present(
        self, sample_parameters, sample_input, var1_initial
    ):
        input_obj = Input(sample_input)
        ds = _make_process(
            MockSpec,
            sample_parameters,
            input1=input_obj,
            var1_initial=var1_initial,
        )
        assert "param1" in ds
        assert "param2" in ds

    def test_parameters_read_only(
        self, sample_parameters, sample_input, var1_initial
    ):
        input_obj = Input(sample_input)
        ds = _make_process(
            MockSpec,
            sample_parameters,
            input1=input_obj,
            var1_initial=var1_initial,
        )
        assert ds["param1"].values.flags.writeable is False
        assert ds["param2"].values.flags.writeable is False

    def test_parameter_buffer_identity(
        self, sample_parameters, sample_input, var1_initial
    ):
        input_obj = Input(sample_input)
        ds = _make_process(
            MockSpec,
            sample_parameters,
            input1=input_obj,
            var1_initial=var1_initial,
        )
        assert ds["param1"].values is sample_parameters["param1"].values
        assert ds["param2"].values is sample_parameters["param2"].values

    def test_input_buffer_identity(
        self, sample_parameters, sample_input, var1_initial
    ):
        input_obj = Input(sample_input)
        ds = _make_process(
            MockSpec,
            sample_parameters,
            input1=input_obj,
            var1_initial=var1_initial,
        )
        assert ds["input1"].values is input_obj.current_values.values

    def test_variable_initial_value(
        self, sample_parameters, sample_input, var1_initial
    ):
        input_obj = Input(sample_input)
        ds = _make_process(
            MockSpec,
            sample_parameters,
            input1=input_obj,
            var1_initial=var1_initial,
        )
        np.testing.assert_array_equal(ds["var1"].values, var1_initial.values)

    def test_variable_default_nan(
        self, sample_parameters, sample_input, var1_initial
    ):
        input_obj = Input(sample_input)
        ds = _make_process(
            MockSpec,
            sample_parameters,
            input1=input_obj,
            var1_initial=var1_initial,
        )
        assert np.all(np.isnan(ds["var2"].values))

    def test_attrs_keys_present(self, process_ds):
        for key in (
            "get_parameters",
            "get_inputs",
            "get_mutable_inputs",
            "get_variables",
            "get_var_names",
            "advance",
            "calculate",
        ):
            assert key in process_ds.attrs, f"Missing attrs key: {key}"

    def test_attrs_callables(self, process_ds):
        assert callable(process_ds.attrs["advance"])
        assert callable(process_ds.attrs["calculate"])


# ---------------------------------------------------------------------------
# TestPWSAccessor
# ---------------------------------------------------------------------------


class TestPWSAccessor:
    def test_get_parameters(self, process_ds):
        assert process_ds.pws.get_parameters() == ("param1", "param2")

    def test_get_inputs(self, process_ds):
        assert process_ds.pws.get_inputs() == ("input1",)

    def test_get_mutable_inputs(self, process_ds):
        assert process_ds.pws.get_mutable_inputs() == ()

    def test_get_var_names(self, process_ds):
        assert process_ds.pws.get_var_names() == ("var1", "var2")

    def test_get_variables(self, process_ds):
        result = process_ds.pws.get_variables()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"var1", "var2"}

    def test_advance_dispatches(self, process_ds):
        called = []

        def mock_advance(ds):
            called.append(True)

        process_ds.attrs["advance"] = mock_advance
        process_ds.pws.advance()
        assert len(called) == 1

    def test_calculate_dispatches(self, process_ds):
        received_dt = []

        def mock_calculate(ds, dt):
            received_dt.append(dt)

        process_ds.attrs["calculate"] = mock_calculate
        process_ds.pws.calculate(np.float64(2.0))
        assert received_dt == [np.float64(2.0)]

    def test_pws_on_plain_dataset_raises_on_advance(self):
        """Plain Dataset without process attrs raises KeyError on .pws.advance()."""
        ds = xr.Dataset({"x": xr.DataArray([1.0, 2.0], dims=["space"])})
        with pytest.raises(KeyError):
            ds.pws.advance()


# ---------------------------------------------------------------------------
# TestModelAttrs
# ---------------------------------------------------------------------------


class TestModelAttrs:
    @pytest.fixture
    def sample_process_dict(
        self, sample_parameters, sample_input, var1_initial
    ):
        return {
            "mock": {
                "class": MockSpec,
                "parameters": sample_parameters,
                "input1": sample_input,
                "var1_initial": var1_initial,
            }
        }

    @pytest.fixture
    def sample_control_config(self, tmp_path):
        return {
            "output_var_names": ["var1"],
            "output_dir": tmp_path / "output",
            "time_chunk_size": 10,
        }

    def test_init(self, sample_process_dict, sample_control_config):
        model = ModelAttrs(sample_process_dict, sample_control_config)
        assert "mock" in model.model_dict
        assert isinstance(model.model_dict["mock"], xr.Dataset)
        assert isinstance(model.output, Output)
        assert hasattr(model, "current_time")
        assert hasattr(model, "times")

    def test_init_without_output(self, sample_process_dict):
        model = ModelAttrs(sample_process_dict, {})
        assert model.output is None

    def test_model_dict_is_dataset(self, sample_process_dict):
        model = ModelAttrs(sample_process_dict, {})
        ds = model.model_dict["mock"]
        assert isinstance(ds, xr.Dataset)
        assert "param1" in ds
        assert "var1" in ds
        assert "input1" in ds

    def test_run(self, sample_process_dict, sample_control_config):
        model = ModelAttrs(sample_process_dict, sample_control_config)
        dt = np.float64(1.0)
        n_steps = np.int32(2)
        model.run(dt, n_steps)
        model.finalize()
        assert model._finalized is True

    def test_finalize_and_context_manager(
        self, sample_process_dict, sample_control_config
    ):
        with ModelAttrs(sample_process_dict, sample_control_config) as model:
            model.run(dt=np.float64(1.0), n_steps=np.int32(2))
        assert model._finalized is True
        for input_obj in model.inputs_dict.values():
            assert input_obj._closed is True

    def test_cannot_run_after_finalize(
        self, sample_process_dict, sample_control_config
    ):
        model = ModelAttrs(sample_process_dict, sample_control_config)
        model.run(np.float64(1.0), np.int32(2))
        model.finalize()
        with pytest.raises(RuntimeError, match="Cannot run a finalized Model"):
            model.run(np.float64(1.0), np.int32(1))

    def test_get_repeated_paths(self, tmp_path, sample_input, var1_initial):
        """Parameters from a shared file path are loaded once -- buffers shared."""
        file1 = tmp_path / "params.nc"
        xr.Dataset(
            {
                "param1": (["space"], [1.0, 2.0, 3.0]),
                "param2": (["space"], [0.5, 0.6, 0.7]),
                "space": [0, 1, 2],
            }
        ).to_netcdf(file1)

        proc_dict: Dict[str, Any] = {
            "proc1": {
                "class": MockSpec,
                "parameters": file1,
                "input1": sample_input,
                "var1_initial": var1_initial,
            },
            "proc2": {
                "class": MockSpec,
                "parameters": file1,
                "input1": xr.DataArray(
                    [[7, 8, 9], [10, 11, 12]],
                    dims=["time", "space"],
                    coords={
                        "time_coord": ("time", [0, 1]),
                        "space_coord": ("space", [0, 1, 2]),
                    },
                ),
                "var1_initial": xr.DataArray(
                    [40.0, 50.0, 60.0],
                    dims=["space"],
                    coords={"space_coord": ("space", [0, 1, 2])},
                ),
            },
        }

        model = ModelAttrs(proc_dict, {})

        for param_name in MockSpec.get_parameters():  # type: ignore[attr-defined]
            assert (
                model.model_dict["proc1"][param_name].values
                is model.model_dict["proc2"][param_name].values
            ), f"Buffer not shared for parameter '{param_name}'"
