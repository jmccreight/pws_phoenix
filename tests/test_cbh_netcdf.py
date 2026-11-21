"""prms_translate.write_cbh_netcdf: CBH text -> one combined forcing
netCDF (the first step of translating a PRMS domain once, ahead of
any run)."""

import numpy as np
import pytest
import xarray as xr

from conftest import PWS_TEST_DATA
from pws_phoenix.prms_translate import cbh, load_cbh, write_cbh_netcdf

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
DRB_CBH = {"prcp": DRB_DIR / "prcp.cbh", "tmax": DRB_DIR / "tmax.cbh"}


def _fake_cbh(arrays: dict[str, xr.DataArray], monkeypatch) -> dict[str, str]:
    """Make load_cbh return `arrays` by file name; return cbh_paths."""
    monkeypatch.setattr(
        cbh, "load_cbh", lambda source: arrays[source.rsplit("/", 1)[1]]
    )
    return {name: f"/nowhere/{name}" for name in arrays}


def _array(name: str, values, start: str = "1979-01-01") -> xr.DataArray:
    values = np.asarray(values, dtype=np.float64)
    time = np.datetime64(start) + np.arange(values.shape[0])
    return xr.DataArray(
        values, dims=("time", "nhru"), coords={"time": time}, name=name
    )


def test_write_cbh_netcdf_contents(tmp_path):
    out = write_cbh_netcdf(DRB_CBH, tmp_path / "cbh.nc")
    assert out == tmp_path / "cbh.nc"
    assert not (tmp_path / "cbh.nc.partial").exists()
    ds = xr.load_dataset(out)
    assert list(ds.data_vars) == ["prcp", "tmax"]
    assert set(ds.coords) == {"time", "nhru"}
    # positions along the grid, the coordinate ModelMPI needs
    np.testing.assert_array_equal(
        ds["nhru"].values, np.arange(ds.sizes["nhru"])
    )
    for name, source in DRB_CBH.items():
        ours = load_cbh(source)
        assert ds[name].dims == ("time", "nhru")
        # drb's CBH values are exact in float32
        assert ds[name].dtype == np.float32
        np.testing.assert_array_equal(
            ds[name].values.astype(np.float64), ours.values
        )
        np.testing.assert_array_equal(ds["time"].values, ours["time"].values)
        assert ds[name].attrs["source_file"] == str(source.resolve())
        assert ds[name].attrs["units"] == ours.attrs["units"]


def test_write_cbh_netcdf_reuse_and_force(tmp_path, monkeypatch):
    out = write_cbh_netcdf(DRB_CBH, tmp_path / "cbh.nc")

    # same CBH files: reused, and no CBH file is parsed
    def no_parse(source):
        raise AssertionError(f"parsed {source}")

    with monkeypatch.context() as patch:
        patch.setattr(cbh, "load_cbh", no_parse)
        assert write_cbh_netcdf(DRB_CBH, out) == out

    # other CBH files (here: a different set of variables): raises
    one = {"prcp": DRB_CBH["prcp"]}
    with pytest.raises(FileExistsError, match="force=True"):
        write_cbh_netcdf(one, out)
    assert list(xr.load_dataset(out).data_vars) == ["prcp", "tmax"]

    # force rewrites
    write_cbh_netcdf(one, out, force=True)
    assert list(xr.load_dataset(out).data_vars) == ["prcp"]


def test_write_cbh_netcdf_float64_when_float32_is_inexact(
    tmp_path, monkeypatch
):
    arrays = {
        "exact": _array("exact", [[0.5, 1.0], [np.nan, 2.0]]),
        "inexact": _array("inexact", [[0.1, 1.0], [0.0, 2.0]]),
    }
    out = write_cbh_netcdf(_fake_cbh(arrays, monkeypatch), tmp_path / "cbh.nc")
    ds = xr.load_dataset(out)
    assert ds["exact"].dtype == np.float32
    assert ds["inexact"].dtype == np.float64
    for name, da in arrays.items():
        np.testing.assert_array_equal(
            ds[name].values.astype(np.float64), da.values
        )


def test_write_cbh_netcdf_raises(tmp_path, monkeypatch):
    out = tmp_path / "cbh.nc"
    with pytest.raises(ValueError, match="empty"):
        write_cbh_netcdf({}, out)

    # the time axes differ
    arrays = {
        "aa": _array("aa", np.zeros((3, 2))),
        "bb": _array("bb", np.zeros((3, 2)), start="1979-01-02"),
    }
    with pytest.raises(ValueError, match="time axis"):
        write_cbh_netcdf(_fake_cbh(arrays, monkeypatch), out)
    # a failed write leaves no file at the product's path
    assert not out.exists()

    # the file holds another variable than the one it was given for
    arrays = {"aa": _array("zz", np.zeros((3, 2)))}
    with pytest.raises(ValueError, match="holds 'zz'"):
        write_cbh_netcdf(_fake_cbh(arrays, monkeypatch), out)
