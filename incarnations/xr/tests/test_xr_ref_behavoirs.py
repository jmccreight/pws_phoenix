"""
mre_buffer_share_testing.py
===========================
MREs for numpy buffer-sharing through xarray DataArrays and Datasets.

Purpose:
  1. Document xarray behaviour that base.py relies on for zero-copy
     inter-process communication.
  2. Candidate tests for the xarray test suite.

Run with: pytest mre_buffer_share_testing.py -v

General finding (current xarray, numpy-backed DataArrays)
----------------------------------------------------------
Dataset.__setitem__ preserves buffer identity regardless of whether the key
is new or existing, whether coordinates are present, or whether dtypes differ
(xarray replaces the variable wholesale rather than casting into the existing
slot).

Dataset variable selection (ds[["v1","v2"]]) preserves buffer identity when
the Dataset is backed by in-memory numpy arrays, with or without coordinates.
However, when the Dataset is opened lazily from a file (xr.open_dataset),
variables are backed by NetCDF readers, not numpy arrays. In that case,
variable selection produces a new lazy Dataset and each .values access reads
from disk, producing a fresh numpy array -- buffer identity is NOT preserved.

This is the operation in Process.__init__:

    self.data = parameters[list(self.get_parameters())]

In practice, parameters come from files opened via xr.open_dataset. The copy
DOES happen there, and the .values= trick on line 216 of base.py is
load-bearing, not redundant. The "copy above" comment is correct.
The .values= trick forces a read on the shared parent Dataset (caching the
numpy array there), then assigns that cached array into self.data, so all
processes sharing the same file end up pointing at the same buffer.
"""

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arr():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def arr2():
    return np.array([4.0, 5.0, 6.0])


@pytest.fixture
def new_data():
    return [7.0, 8.0, 9.0]


@pytest.fixture
def nc_params_file(tmp_path, arr, arr2):
    """NetCDF file containing v1=arr, v2=arr2, v3=zeros, all on dim 'space'."""
    nc_file = tmp_path / "params.nc"
    xr.Dataset(
        {
            "v1": xr.DataArray(arr, dims=["space"]),
            "v2": xr.DataArray(arr2, dims=["space"]),
            "v3": xr.DataArray(np.zeros(3), dims=["space"]),
        }
    ).to_netcdf(nc_file)
    return nc_file


# ---------------------------------------------------------------------------
# Section 1: Dataset.__setitem__ always preserves buffer identity
# ---------------------------------------------------------------------------


class TestSetitemPreservesBuffer:
    """Dataset.__setitem__ preserves the numpy buffer in all tested cases."""

    def test_new_key(self, arr):
        ds = xr.Dataset()
        ds["x"] = xr.DataArray(arr, dims=["space"])
        assert ds["x"].values is arr

    def test_existing_key(self, arr):
        ds = xr.Dataset({"x": xr.DataArray(np.zeros(3), dims=["space"])})
        ds["x"] = xr.DataArray(arr, dims=["space"])
        assert ds["x"].values is arr

    def test_with_coordinates(self, arr):
        # necessary? how many possible permutation are there of this on kwargs
        da = xr.DataArray(
            arr,
            dims=["space"],
            coords={"space_coord": ("space", [10, 20, 30])},
        )
        ds = xr.Dataset()
        ds["x"] = da
        assert ds["x"].values is arr

    def test_dtype_mismatch_replaces_wholesale(self, arr):
        # initial variable is float32 -- xarray replaces, not casts
        ds = xr.Dataset(
            {"x": xr.DataArray(np.zeros(3, dtype=np.float32), dims=["space"])}
        )
        ds["x"] = xr.DataArray(arr, dims=["space"])
        assert ds["x"].values is arr
        assert ds["x"].dtype == arr.dtype

    def test_variable_selection_without_coordinates(self, arr, arr2):
        ds = xr.Dataset(
            {
                "v1": xr.DataArray(arr, dims=["space"]),
                "v2": xr.DataArray(arr2, dims=["space"]),
                "v3": xr.DataArray(np.zeros(3), dims=["space"]),
            }
        )
        subset = ds[["v1", "v2"]]
        assert ds["v1"].values is subset["v1"].values is arr
        assert ds["v2"].values is subset["v2"].values is arr2

    def test_variable_selection_file_backed_copies_buffer(
        self, nc_params_file
    ):
        # Lazy file-backed Dataset: variable selection produces a new lazy
        # Dataset. Each .values access reads from disk -- fresh numpy array
        # each time, so buffer identity is NOT preserved across the selection.
        # This is why the .values= trick in Process.__init__ is load-bearing.
        ds = xr.open_dataset(nc_params_file)
        subset = ds[["v1", "v2"]]
        assert subset["v1"].values is not ds["v1"].values
        assert subset["v2"].values is not ds["v2"].values
        ds.close()

    def test_variable_selection_file_backed_loaded_preserves_buffer(
        self, nc_params_file
    ):
        # After .load(), all variables are in-memory numpy arrays. Variable
        # selection then behaves like the in-memory case: buffer identity IS
        # preserved. Contrast with the lazy case above.
        ds = xr.open_dataset(nc_params_file).load()
        subset = ds[["v1", "v2"]]
        assert subset["v1"].values is ds["v1"].values
        assert subset["v2"].values is ds["v2"].values
        ds.close()

    def test_selective_variable_load_inplace_preserves_buffer(
        self, nc_params_file
    ):
        # Selective per-variable in-place load before variable selection.
        # ds["v1"].load() loads that variable's data into the parent Dataset
        # in-place, leaving other variables (v3) still lazy.
        # Subsequent variable selection on the partially-loaded Dataset
        # preserves buffer identity for the loaded variables only.
        # This is more memory-efficient than ds.load() when only a subset
        # of variables is needed -- the Process.__init__ use case.
        var_list = ["v1", "v2"]
        ds = xr.open_dataset(nc_params_file)
        for vv in var_list:
            ds[vv].load()
        subset = ds[var_list]
        assert subset["v1"].values is ds["v1"].values
        assert subset["v2"].values is ds["v2"].values
        # v3 was not loaded -- still lazy, still on disk
        assert not ds["v3"].variable._in_memory
        ds.close()


# ---------------------------------------------------------------------------
# Section 2: .values= set establishes deliberate buffer sharing
# ---------------------------------------------------------------------------


class TestValuesSet:
    """.values= stores by reference; enables sharing across multiple Datasets."""

    def test_standalone(self, arr):
        da = xr.DataArray(np.zeros(3), dims=["space"])
        da.values = arr
        assert da.values is arr

    def test_dataset_member(self, arr):
        ds = xr.Dataset({"x": xr.DataArray(np.zeros(3), dims=["space"])})
        ds["x"].values = arr
        assert ds["x"].values is arr

    def test_shared_across_two_datasets(self, arr, new_data):
        ds_a = xr.Dataset({"a": xr.DataArray(np.zeros(3), dims=["space"])})
        ds_b = xr.Dataset({"b": xr.DataArray(np.zeros(3), dims=["space"])})
        ds_a["a"].values = arr
        ds_b["b"].values = arr
        assert ds_a["a"].values is ds_b["b"].values is arr

        ds_a["a"].values[:] = new_data
        assert ds_a["a"].values is ds_b["b"].values is arr
        np.testing.assert_array_equal(ds_b["b"].values, new_data)
        np.testing.assert_array_equal(arr, new_data)


# ---------------------------------------------------------------------------
# Section 3: In-place [:] updates propagate to all buffer holders
# ---------------------------------------------------------------------------


class TestInplaceUpdatePropagation:
    """[:] updates propagate to every object sharing the same numpy buffer."""

    def test_python_alias(self, arr, arr2):
        da = xr.DataArray(arr, dims=["space"])
        da_ref = da
        da[:] = arr2
        assert da_ref.values is da.values is arr is not arr2
        np.testing.assert_array_equal(da_ref.values, arr2)
        np.testing.assert_array_equal(arr, arr2)

    def test_through_dataset_setitem(self, arr, arr2):
        """__setitem__ preserves buffer, so [:] on da propagates to ds["x"]."""
        da = xr.DataArray(arr, dims=["space"])
        ds = xr.Dataset()
        ds["x"] = da
        da[:] = arr2
        assert ds["x"].values is da.values is arr is not arr2
        np.testing.assert_array_equal(arr, arr2)

    def test_through_dataset_values_set(self, arr, arr2):
        """After .values= establishes sharing, [:] on arr propagates to ds["x"]."""
        ds = xr.Dataset({"x": xr.DataArray(np.zeros(3), dims=["space"])})
        ds["x"].values = arr
        arr[:] = arr2
        np.testing.assert_array_equal(ds["x"].values, arr2)

    def test_input_advance_pattern(self):
        """Input.advance() [:] update propagates to all Process Datasets sharing the buffer."""
        n_time, n_space = 3, 4
        forcing_data = np.arange(float(n_time * n_space)).reshape(
            n_time, n_space
        )
        current_values = xr.DataArray(np.full(n_space, np.nan), dims=["space"])
        ds_upper = xr.Dataset(
            {"forcing": xr.DataArray(np.zeros(n_space), dims=["space"])}
        )
        ds_lower = xr.Dataset(
            {"forcing": xr.DataArray(np.zeros(n_space), dims=["space"])}
        )
        ds_upper["forcing"].values = current_values.values
        ds_lower["forcing"].values = current_values.values
        assert (
            ds_upper["forcing"].values
            is ds_lower["forcing"].values
            is current_values.values
        )

        for t in range(n_time):
            current_values[:] = forcing_data[t, :]
            np.testing.assert_array_equal(
                ds_upper["forcing"].values, forcing_data[t, :]
            )
            np.testing.assert_array_equal(
                ds_lower["forcing"].values, forcing_data[t, :]
            )
            assert (
                ds_upper["forcing"].values
                is ds_lower["forcing"].values
                is current_values.values
            )
