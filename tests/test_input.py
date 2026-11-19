"""data_io.Input: the per-step buffer of a time-varying input."""

import numpy as np
import pytest
import xarray as xr

from pws_phoenix.data_io import Input


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_current_values_float64(dtype):
    """The per-step buffer is float64 whatever the data hold, and holds
    the data's values exactly."""
    values = np.arange(12, dtype=dtype).reshape(4, 3)
    data = xr.DataArray(values, dims=("time", "nhru"))
    inp = Input(data)
    assert inp.current_values.dtype == np.float64
    assert np.isnan(inp.current_values.values).all()
    # the data are not widened: only the one-step buffer is
    assert inp.data.values is values
    for tt in range(values.shape[0]):
        inp.advance()
        assert inp.current_values.dtype == np.float64
        np.testing.assert_array_equal(inp.current_values.values, values[tt])


@pytest.mark.parametrize("read_only", [False, True])
def test_file_backed_input_stays_on_disk(tmp_path, read_only):
    """A file-backed Input reads one time row per advance(): the whole
    array is never in memory, read-only or not."""
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = tmp_path / "prcp.nc"
    xr.DataArray(values, dims=("time", "nhru"), name="prcp").to_netcdf(path)
    # opened and left on disk on purpose: that is the case under test
    with xr.open_dataset(path) as ds:
        inp = Input(ds["prcp"], read_only=read_only)
        assert not inp.data.variable._in_memory
        for tt in range(values.shape[0]):
            inp.advance()
            np.testing.assert_array_equal(
                inp.current_values.values, values[tt]
            )
        assert not inp.data.variable._in_memory


@pytest.mark.parametrize("file_backed", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("scale", "offset"), [(12.0, 0.0), (1.8, 32.0)])
def test_conversion_in_the_one_step_buffer(
    tmp_path, file_backed, dtype, scale, offset
):
    """set_conversion(): every served row is scale * data + offset,
    computed in float64 after the row is widened, and the data
    themselves are never edited."""
    values = (np.arange(12).reshape(4, 3) / 7.0).astype(dtype)
    kept = values.copy()
    expected = values.astype(np.float64) * scale + offset
    data = xr.DataArray(values, dims=("time", "nhru"), name="tmax")

    def check(inp: Input) -> None:
        inp.set_conversion(scale, offset)
        for tt in range(values.shape[0]):
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
