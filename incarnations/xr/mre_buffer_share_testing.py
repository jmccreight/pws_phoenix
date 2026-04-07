"""
mre_buffer_sharing.py
=====================
Minimal Reproducible Examples (MREs) for numpy buffer-sharing through xarray
DataArrays and Datasets.

Two purposes:
  1. Document the observed xarray behaviour that the process-based modeling
     framework in base.py relies on for zero-copy inter-process communication.
  2. Serve as a candidate test contribution to the xarray test suite so that
     any future xarray release that silently changes these semantics is caught.

Run with:
    pytest mre_buffer_sharing.py -v

Each test is self-contained and annotated.  Tests are grouped into three
sections corresponding to the three sharing patterns used in base.py.

General finding (current xarray, numpy-backed DataArrays)
----------------------------------------------------------
Dataset.__setitem__ preserves buffer identity unconditionally for numpy-backed
DataArrays.  This holds regardless of:
  - whether the key is new or already exists in the Dataset
  - whether coordinates are present on the DataArray
  - whether the incoming dtype differs from the existing variable's dtype
    (xarray replaces the variable wholesale rather than casting into the
    existing slot, so no copy occurs and the Dataset's variable dtype changes
    to match the incoming array)

Consequence for base.py:
  The .values= trick in Process.__init__ (used to share parameter buffers
  across processes) appears to be redundant in current xarray -- plain
  Dataset.__setitem__ already preserves the buffer.  The trick was likely
  necessary against an older xarray version.  It is retained as defensive
  programming, but the "copy above" comment in Process.__init__ should be
  revisited against the version history to confirm when the behaviour changed.
"""

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Section 1 – Establishing the baseline: does Dataset.__setitem__ copy or ref?
#
# base.py relies on buffer sharing but the naive assignment path (ds["k"] = da)
# turns out to behave differently depending on whether the Dataset already
# contains the key.  Both cases are probed here.
# ---------------------------------------------------------------------------


def test_setitem_into_empty_dataset_preserves_buffer():
    """
    Assigning a DataArray into a fresh Dataset key preserves buffer identity.

    We initially assumed xarray would copy here -- it does not.  The buffer
    of the original numpy array is shared directly with the Dataset variable.
    """
    arr = np.array([1.0, 2.0, 3.0])
    da = xr.DataArray(arr, dims=["space"])

    ds = xr.Dataset()
    ds["x"] = da

    assert ds["x"].values is arr


def test_setitem_overwrite_existing_dataset_key_preserves_buffer():
    """
    Overwriting an existing Dataset key with a new DataArray also preserves
    buffer identity -- xarray replaces the variable wholesale.

    We initially assumed xarray would copy here -- it does not.
    """
    arr = np.array([1.0, 2.0, 3.0])
    da_original = xr.DataArray(np.zeros(3), dims=["space"])
    da_new = xr.DataArray(arr, dims=["space"])

    ds = xr.Dataset({"x": da_original})
    ds["x"] = da_new

    assert ds["x"].values is arr


# ---------------------------------------------------------------------------
# Section 2 – The .values = trick: guaranteed reference, no copy.
#
# Setting DataArray.values = arr calls Variable._data = as_compatible_data(arr)
# which stores the array object directly.  This is how base.py works around
# the copy behaviour in Section 1.
# ---------------------------------------------------------------------------


def test_values_assignment_preserves_buffer_identity():
    """
    Assigning to DataArray.values stores the array by reference.
    After `da.values = arr`, `da.values is arr` is True.
    """
    arr = np.array([1.0, 2.0, 3.0])
    da = xr.DataArray(np.zeros(3), dims=["space"])

    da.values = arr

    assert da.values is arr


def test_values_assignment_on_dataset_member_preserves_identity():
    """
    The same .values = trick works on a DataArray that already lives inside a
    Dataset.  This is the exact pattern used in Process.__init__ for parameters:

        self.data = parameters[param_names]   # creates Dataset with copied buffers
        self.data["p"].values = parameters["p"].values  # restores shared buffer
    """
    arr = np.array([1.0, 2.0, 3.0])
    ds = xr.Dataset({"x": xr.DataArray(np.zeros(3), dims=["space"])})

    ds["x"].values = arr

    assert ds["x"].values is arr


def test_values_assignment_shared_across_two_datasets():
    """
    If two Dataset members are given the same backing array via .values =,
    they share the buffer.  Mutation through one is immediately visible through
    the other -- no copy, no re-assignment.

    This is how param_common is shared between Upper and Lower in the toy model.
    """
    arr = np.array([1.0, 2.0, 3.0])
    ds_a = xr.Dataset({"x": xr.DataArray(np.zeros(3), dims=["space"])})
    ds_b = xr.Dataset({"x": xr.DataArray(np.zeros(3), dims=["space"])})

    ds_a["x"].values = arr
    ds_b["x"].values = arr

    assert ds_a["x"].values is ds_b["x"].values is arr

    # In-place update through one Dataset propagates to the other.
    ds_a["x"].values[:] = [7.0, 8.0, 9.0]
    np.testing.assert_array_equal(ds_b["x"].values, [7.0, 8.0, 9.0])


# ---------------------------------------------------------------------------
# Section 3 – In-place [:] update propagation (the Input.advance() pattern).
#
# Input._current_values is a DataArray allocated once at init.  Each call to
# Input.advance() does current_values[:] = data[t, :] -- an in-place slice
# assignment that overwrites the buffer without creating a new array object.
# Any Process that received this DataArray (via __setitem__ into its Dataset)
# sees the updated values only if the buffer is shared.
#
# But Section 1 showed that __setitem__ copies.  So how does this work?
# We probe whether the Input.advance() pattern actually relies on Dataset
# sharing at all, or whether it works through the DataArray object directly.
# ---------------------------------------------------------------------------


def test_inplace_slice_update_visible_through_original_dataarray():
    """
    In-place [:] update on a DataArray is visible through any other reference
    to the same DataArray object (not Dataset).

    This is the simplest case: two Python names bound to the same DataArray.
    """
    arr = np.array([1.0, 2.0, 3.0])
    da = xr.DataArray(arr, dims=["space"])
    da_ref = da  # same object, different name

    da[:] = [4.0, 5.0, 6.0]

    np.testing.assert_array_equal(da_ref.values, [4.0, 5.0, 6.0])
    # The original numpy array is also updated in-place.
    np.testing.assert_array_equal(arr, [4.0, 5.0, 6.0])


def test_inplace_slice_update_NOT_visible_through_dataset_copy():
    """
    Because Dataset.__setitem__ copies the buffer (Section 1), an in-place
    update to the original DataArray is NOT visible through the Dataset copy.

    This test documents the limitation: the Input.advance() pattern cannot
    rely on Dataset membership for propagation -- it relies on direct DataArray
    reference sharing (same Python object or same .values buffer via the trick).
    """
    arr = np.array([1.0, 2.0, 3.0])
    da = xr.DataArray(arr, dims=["space"])
    ds = xr.Dataset()
    ds["x"] = da  # copies buffer

    da[:] = [4.0, 5.0, 6.0]

    # ds["x"] still holds the old values -- the update did NOT propagate.
    assert not np.all(ds["x"].values == 4.0)


def test_inplace_slice_update_visible_after_values_trick():
    """
    After the .values = trick is applied, in-place [:] updates DO propagate
    through the Dataset.

    This is the combined pattern: use .values = to establish the shared buffer,
    then rely on [:] updates propagating.  This is how Process.__init__ wires
    parameters and how Input._current_values propagation is guaranteed.
    """
    arr = np.array([1.0, 2.0, 3.0])
    ds = xr.Dataset({"x": xr.DataArray(np.zeros(3), dims=["space"])})

    ds["x"].values = arr  # establish shared buffer

    arr[:] = [4.0, 5.0, 6.0]  # in-place update via original array

    np.testing.assert_array_equal(ds["x"].values, [4.0, 5.0, 6.0])


def test_input_advance_pattern_end_to_end():
    """
    Full end-to-end simulation of the Input.advance() + Process sharing pattern.

    Two 'processes' (represented as Datasets) share the same current_values
    buffer via the .values = trick.  Calling advance() (in-place [:] update)
    on the shared DataArray propagates to both process Datasets simultaneously.

    This mirrors how forcing_common is shared between Upper and Lower in the
    toy model without any per-timestep copy or re-assignment.
    """
    # Simulate forcing data: 3 timesteps, 4 spatial points.
    forcing_data = np.arange(12.0).reshape(3, 4)

    # current_values is allocated once (mimics Input.__init__).
    current_values = xr.DataArray(np.full(4, np.nan), dims=["space"])

    # Two processes store the same current_values buffer via .values = trick.
    ds_upper = xr.Dataset(
        {"forcing": xr.DataArray(np.zeros(4), dims=["space"])}
    )
    ds_lower = xr.Dataset(
        {"forcing": xr.DataArray(np.zeros(4), dims=["space"])}
    )

    ds_upper["forcing"].values = current_values.values
    ds_lower["forcing"].values = current_values.values

    assert (
        ds_upper["forcing"].values
        is ds_lower["forcing"].values
        is current_values.values
    )

    # Simulate three advance() calls.
    for t in range(3):
        current_values[:] = forcing_data[t, :]  # Input.advance()

        np.testing.assert_array_equal(
            ds_upper["forcing"].values, forcing_data[t, :]
        )
        np.testing.assert_array_equal(
            ds_lower["forcing"].values, forcing_data[t, :]
        )


# ---------------------------------------------------------------------------
# Section 4 – Probing the boundaries of __setitem__ buffer preservation.
#
# The two xfail tests above showed that simple assignment preserves buffers.
# Here we probe three conditions that might cause xarray to copy instead:
#   A. Dataset variable selection (ds[["v1","v2"]]) -- the exact operation
#      used in Process.__init__ that the base.py comment says "copies above".
#   B. DataArray with coordinates -- alignment logic may trigger a copy.
#   C. dtype mismatch -- type promotion almost certainly forces a copy.
# ---------------------------------------------------------------------------


def test_dataset_variable_selection_buffer_identity():
    """
    Probe A: does selecting variables from a Dataset (`ds[["v1","v2"]]`)
    preserve buffer identity in the returned Dataset?

    This is the specific operation in Process.__init__:
        self.data = parameters[list(self.get_parameters())]
    The base.py comment says "Apparently a copy above happens" -- we check.
    """
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([4.0, 5.0, 6.0])
    ds = xr.Dataset(
        {
            "v1": xr.DataArray(arr1, dims=["space"]),
            "v2": xr.DataArray(arr2, dims=["space"]),
            "v3": xr.DataArray(np.zeros(3), dims=["space"]),
        }
    )

    subset = ds[["v1", "v2"]]

    # Does the subset share buffers with the original?
    assert subset["v1"].values is arr1, (
        "Dataset variable selection copied the buffer for v1"
    )
    assert subset["v2"].values is arr2, (
        "Dataset variable selection copied the buffer for v2"
    )


def test_setitem_with_coordinates_preserves_buffer_identity():
    """
    Probe B: does assigning a DataArray that carries coordinates into a
    Dataset still preserve the data buffer?

    Coordinate alignment is a common trigger for xarray copies.  base.py
    DataArrays generally carry space/time coordinates, so this matters.
    """
    arr = np.array([1.0, 2.0, 3.0])
    da = xr.DataArray(
        arr,
        dims=["space"],
        coords={"space_coord": ("space", [10, 20, 30])},
    )

    ds = xr.Dataset()
    ds["x"] = da

    assert ds["x"].values is arr, (
        "Buffer identity lost when DataArray with coordinates is assigned "
        "into a Dataset"
    )


def test_setitem_dtype_mismatch_preserves_buffer_and_changes_dtype():
    """
    Probe C: assigning a float64 DataArray into a Dataset that previously held
    a float32 variable preserves buffer identity and changes the variable dtype.

    xarray does not cast into the existing slot -- it replaces the variable
    wholesale with the incoming one.  No copy occurs, and the Dataset variable
    takes on the new dtype.  This is consistent with the general finding that
    Dataset.__setitem__ always preserves buffer identity for numpy-backed arrays.
    """
    arr_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    da_float32 = xr.DataArray(
        np.array([1.0, 2.0, 3.0], dtype=np.float32), dims=["space"]
    )

    ds = xr.Dataset({"x": da_float32})
    ds["x"] = xr.DataArray(arr_float64, dims=["space"])

    # Buffer is preserved -- xarray replaced the variable, did not cast.
    assert ds["x"].values is arr_float64
    # The dtype of the Dataset variable is now float64, not float32.
    assert ds["x"].dtype == np.float64
