"""data_io.Input: the per-step buffer of a time-varying input. Also
data_io.open_xr, which opens the files Model reads from paths."""

import numpy as np
import pytest
import xarray as xr
from xarray.backends.file_manager import FILE_CACHE

from pws_phoenix.data_io import Input, open_xr

# an Input takes either order of its two dims and serves the same rows
ORDERS = [("time", "nhru"), ("nhru", "time")]


def _in_order(by_day: np.ndarray, dims: tuple[str, str]) -> np.ndarray:
    """The (time, nhru) array `by_day` as a C-ordered array on `dims`:
    itself, or a transposed copy."""
    if dims[0] == "time":
        return by_day
    return np.ascontiguousarray(by_day.T)


@pytest.mark.parametrize("dims", ORDERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_current_values_float64(dtype, dims):
    """The per-step buffer is float64 whatever the data hold, and holds
    the data's values exactly."""
    by_day = np.arange(12, dtype=dtype).reshape(4, 3)
    values = _in_order(by_day, dims)
    data = xr.DataArray(values, dims=dims)
    inp = Input(data)
    assert inp.current_values.dims == ("nhru",)
    assert inp.current_values.dtype == np.float64
    assert np.isnan(inp.current_values.values).all()
    # the data are not widened: only the one-step buffer is
    assert inp.data.values is values
    for tt in range(by_day.shape[0]):
        inp.advance()
        assert inp.current_values.dtype == np.float64
        np.testing.assert_array_equal(inp.current_values.values, by_day[tt])


@pytest.mark.parametrize("dims", ORDERS)
@pytest.mark.parametrize("read_only", [False, True])
def test_file_backed_input_stays_on_disk(tmp_path, read_only, dims):
    """A file-backed Input reads one day per advance(): the whole
    array is never in memory, read-only or not."""
    by_day = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = tmp_path / "prcp.nc"
    xr.DataArray(_in_order(by_day, dims), dims=dims, name="prcp").to_netcdf(
        path
    )
    # opened and left on disk on purpose: that is the case under test
    with xr.open_dataset(path) as ds:
        assert ds["prcp"].dims == dims
        inp = Input(ds["prcp"], read_only=read_only)
        assert not inp.data.variable._in_memory
        for tt in range(by_day.shape[0]):
            inp.advance()
            np.testing.assert_array_equal(
                inp.current_values.values, by_day[tt]
            )
        assert not inp.data.variable._in_memory


@pytest.mark.parametrize("dims", ORDERS)
def test_load_reads_the_whole_array(tmp_path, dims):
    """load=True (control["load_all"]): the array is read once, in the
    file's order, and the same days are served."""
    by_day = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = tmp_path / "prcp.nc"
    xr.DataArray(_in_order(by_day, dims), dims=dims, name="prcp").to_netcdf(
        path
    )
    with xr.open_dataset(path) as ds:
        inp = Input(ds["prcp"], load=True)
        assert inp.data.variable._in_memory
        assert inp.data.dims == dims
        for tt in range(by_day.shape[0]):
            inp.advance()
            np.testing.assert_array_equal(
                inp.current_values.values, by_day[tt]
            )


@pytest.mark.parametrize("dims", ORDERS)
def test_set_index_and_reset(dims):
    """set_index() and reset() count days in either order."""
    by_day = np.arange(12, dtype=np.float64).reshape(4, 3)
    inp = Input(xr.DataArray(_in_order(by_day, dims), dims=dims))
    inp.set_index(2)
    inp.advance()
    np.testing.assert_array_equal(inp.current_values.values, by_day[2])
    inp.reset()
    assert np.isnan(inp.current_values.values).all()
    inp.advance()
    np.testing.assert_array_equal(inp.current_values.values, by_day[0])


@pytest.mark.parametrize(
    "dims", [("day", "nhru"), ("time",), ("time", "nhru", "layer")]
)
def test_dims_of_neither_order_raise(tmp_path, dims):
    """Anything but ("time", <grid>) or (<grid>, "time") raises, and an
    Input that opened the file itself closes it first."""
    data = xr.DataArray(np.zeros((2,) * len(dims)), dims=dims, name="prcp")
    with pytest.raises(ValueError, match=r"Input 'prcp' is on dims"):
        Input(data)
    path = tmp_path / "prcp.nc"
    data.to_netcdf(path)
    n_open = len(FILE_CACHE)
    with pytest.raises(ValueError, match=r"Input 'prcp' is on dims"):
        Input(path)
    assert len(FILE_CACHE) == n_open


@pytest.mark.parametrize("dims", ORDERS)
@pytest.mark.parametrize("file_backed", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("scale", "offset"), [(12.0, 0.0), (1.8, 32.0)])
def test_conversion_in_the_one_step_buffer(
    tmp_path, file_backed, dtype, scale, offset, dims
):
    """set_conversion(): every served row is scale * data + offset,
    computed in float64 after the row is widened, and the data
    themselves are never edited."""
    by_day = (np.arange(12).reshape(4, 3) / 7.0).astype(dtype)
    values = _in_order(by_day, dims)
    kept = values.copy()
    expected = by_day.astype(np.float64) * scale + offset
    data = xr.DataArray(values, dims=dims, name="tmax")

    def check(inp: Input) -> None:
        inp.set_conversion(scale, offset)
        for tt in range(by_day.shape[0]):
            inp.advance()
            np.testing.assert_array_equal(
                inp.current_values.values, expected[tt]
            )

    if file_backed:
        path = tmp_path / "tmax.nc"
        data.to_netcdf(path)
        with xr.open_dataset(path) as ds:
            inp = Input(ds["tmax"], read_only=True)
            check(inp)
            assert not inp.data.variable._in_memory
            assert inp.data.dtype == dtype
    else:
        inp = Input(data, read_only=True)
        check(inp)
        assert inp.data.values is values
    np.testing.assert_array_equal(values, kept)
    assert values.dtype == dtype


@pytest.mark.parametrize("n_vars", [1, 2])
def test_open_xr_close_closes_the_file(tmp_path, n_vars):
    """close() on what open_xr returns closes the file, for one
    variable (a DataArray) and for several (a Dataset). xarray holds
    every open file in FILE_CACHE."""
    path = tmp_path / "data.nc"
    xr.Dataset(
        {f"var_{ii}": ("x", np.arange(3.0)) for ii in range(n_vars)}
    ).to_netcdf(path)
    n_open = len(FILE_CACHE)
    opened = open_xr(path)
    assert isinstance(opened, xr.DataArray if n_vars == 1 else xr.Dataset)
    assert len(FILE_CACHE) == n_open + 1
    opened.close()
    assert len(FILE_CACHE) == n_open
