"""
maps.SparseWeights: the nonzero entries of a Map's weight matrix (row
index, column index, value) -- the form a Map holds, whether it was
given that form or a dense matrix -- and the two forms a weights file
of the model yaml may hold. Data-free.
"""

import numpy as np
import pytest
import xarray as xr
import yaml
from pws_phoenix.config import load_model_yaml
from pws_phoenix.maps import Map, SparseWeights

# 3 targets x 5 sources: row 2 and column 4 are empty; row 0 has three
# entries (the case where the order of the sum matters)
DENSE = np.array(
    [
        [0.1, 0.0, 0.7, 0.2, 0.0],
        [0.0, 0.3, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)
# the same entries, out of order
ROWS = np.array([1, 0, 0, 0])
COLS = np.array([1, 3, 0, 2])
VALS = np.array([0.3, 0.2, 0.1, 0.7])


def _assert_by_row_equal(found: tuple, expected: tuple) -> None:
    for found_array, expected_array in zip(found, expected, strict=True):
        np.testing.assert_array_equal(found_array, expected_array)
        assert found_array.dtype == expected_array.dtype


def test_unsorted_entries_match_dense():
    """Entries given in any order are the dense matrix's, in its
    order."""
    sparse = SparseWeights(ROWS, COLS, VALS, DENSE.shape)
    from_dense = SparseWeights.from_dense(DENSE)
    assert sparse.shape == from_dense.shape == (3, 5)
    np.testing.assert_array_equal(sparse.dense(), DENSE)
    _assert_by_row_equal(sparse.by_row(), from_dense.by_row())
    indptr, cols, vals = sparse.by_row()
    np.testing.assert_array_equal(indptr, [0, 3, 4, 4])
    np.testing.assert_array_equal(cols, [0, 2, 3, 1])
    np.testing.assert_array_equal(vals, [0.1, 0.7, 0.2, 0.3])


@pytest.mark.parametrize("start, stop", [(0, 2), (2, 5), (1, 4), (4, 5)])
def test_column_block(start, stop):
    """One rank's block of the source grid: the entries of
    ``dense[:, start:stop]``, columns counted from ``start``."""
    sparse = SparseWeights(ROWS, COLS, VALS, DENSE.shape)
    block = SparseWeights.from_dense(DENSE[:, start:stop])
    _assert_by_row_equal(
        sparse.by_row(col_start=start, col_stop=stop), block.by_row()
    )


def test_zero_value_dropped():
    sparse = SparseWeights([0, 1], [0, 1], [0.0, 2.0], (2, 2))
    np.testing.assert_array_equal(sparse.rows, [1])
    np.testing.assert_array_equal(sparse.cols, [1])
    np.testing.assert_array_equal(sparse.vals, [2.0])


def _apply(weights, source: np.ndarray) -> np.ndarray:
    mm = Map(
        weights=weights,
        grid={"hru": "segment"},
        variable={"flow": "seg_flow"},
    )
    mm.bind(xr.Dataset({"flow": ("hru", source)}))
    mm.apply()
    return mm.target_values.values


def test_map_apply_same_from_either_form():
    source = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    from_dense = _apply(DENSE, source)
    from_sparse = _apply(SparseWeights(ROWS, COLS, VALS, DENSE.shape), source)
    np.testing.assert_array_equal(from_sparse, from_dense)
    np.testing.assert_allclose(from_dense, DENSE @ source, rtol=1e-14)


def test_zero_weight_keeps_nan_out():
    """A zero weight is no entry, in either form: a nan source under it
    does not reach the target."""
    source = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    with_zero = SparseWeights(
        np.append(ROWS, 2), np.append(COLS, 4), np.append(VALS, 0.0), (3, 5)
    )
    for weights in (DENSE, with_zero):
        assert np.isfinite(_apply(weights, source)).all()


def test_bad_entries_raise():
    with pytest.raises(ValueError, match="more than once"):
        SparseWeights([0, 0], [1, 1], [1.0, 2.0], (2, 2))
    with pytest.raises(ValueError, match="rows must be in"):
        SparseWeights([2], [0], [1.0], (2, 2))
    with pytest.raises(ValueError, match="cols must be in"):
        SparseWeights([0], [-1], [1.0], (2, 2))
    with pytest.raises(ValueError, match="must be integers"):
        SparseWeights([0.0], [0], [1.0], (2, 2))
    with pytest.raises(ValueError, match="one length"):
        SparseWeights([0, 1], [0], [1.0], (2, 2))


def test_file_round_trip(tmp_path):
    """to_dataset -> NetCDF -> from_dataset gives the same weights."""
    sparse = SparseWeights(ROWS, COLS, VALS, DENSE.shape)
    path = tmp_path / "weights.nc"
    sparse.to_dataset("weights_vol").to_netcdf(path)
    ds = xr.load_dataset(path)
    assert set(ds.data_vars) == {
        "weights_vol_rows",
        "weights_vol_cols",
        "weights_vol_vals",
    }
    found = SparseWeights.from_dataset(ds, "weights_vol")
    assert found.shape == sparse.shape
    for part in ("rows", "cols", "vals"):
        found_array = getattr(found, part)
        np.testing.assert_array_equal(found_array, getattr(sparse, part))
        assert found_array.dtype == getattr(sparse, part).dtype


def test_dense_matrix_in_file_raises():
    """A file from before the sparse form (one dense variable of the
    matrix's name) says to write it again."""
    ds = xr.Dataset({"weights_vol": (("nsegment", "nhru"), DENSE)})
    with pytest.raises(ValueError, match="written again"):
        SparseWeights.from_dataset(ds, "weights_vol")


def _write_model_yaml(path, weights_by_map: dict) -> None:
    """A model yaml with no processes: one Map per entry, its
    ``weights:`` as given (an inline matrix or a file name)."""
    config = {
        "control": {},
        "processes": {},
        "maps": {
            name: {
                "class": "Map",
                "weights": weights,
                "grid": {"hru": "segment"},
                "variable": {"flow": "seg_flow"},
            }
            for name, weights in weights_by_map.items()
        },
    }
    with open(path, "w") as ff:
        yaml.safe_dump(config, ff, sort_keys=False)


def test_model_yaml_reads_either_form(tmp_path):
    """A map's ``weights:`` in the model yaml -- an inline matrix, a
    file that holds the full matrix, a file that holds the nonzero
    entries -- gives the same Map."""
    xr.DataArray(DENSE, dims=["segment", "hru"], name="weights").to_netcdf(
        tmp_path / "dense.nc"
    )
    sparse = SparseWeights(ROWS, COLS, VALS, DENSE.shape)
    sparse.to_dataset("weights").to_netcdf(tmp_path / "sparse.nc")
    yaml_file = tmp_path / "model.yaml"
    _write_model_yaml(
        yaml_file,
        {"inline": DENSE.tolist(), "dense": "dense.nc", "sparse": "sparse.nc"},
    )
    _, _, maps, _ = load_model_yaml(yaml_file)
    assert list(maps) == ["inline", "dense", "sparse"]
    for mm in maps.values():
        assert mm.weights.shape == DENSE.shape
        np.testing.assert_array_equal(mm.weights.dense(), DENSE)


def test_model_yaml_weights_file_of_neither_form_raises(tmp_path):
    two_variables = xr.Dataset(
        {"aa": ("hru", np.zeros(5)), "bb": ("hru", np.zeros(5))}
    )
    two_variables.to_netcdf(tmp_path / "neither.nc")
    yaml_file = tmp_path / "model.yaml"
    _write_model_yaml(yaml_file, {"neither": "neither.nc"})
    with pytest.raises(ValueError, match="holds one variable"):
        load_model_yaml(yaml_file)


def test_dense_weights_must_be_2d():
    with pytest.raises(ValueError, match="2-D"):
        Map(
            weights=np.zeros(3),
            grid={"hru": "segment"},
            variable={"flow": "seg_flow"},
        )
