"""utils.transpose_netcdf: a copy of a netCDF with the two dims of its
2-D time variables swapped."""

import netCDF4
import numpy as np
import pytest
import xarray as xr

from pws_phoenix.utils import transpose_netcdf as module
from pws_phoenix.utils.transpose_netcdf import main, transpose_netcdf

N_TIME = 11
N_HRU = 7


def _write_source(path, sources=("prcp.cbh", "tmax.cbh")):
    """A forcing file as cbh.write_cbh_netcdf writes one, plus an
    (nhru,) variable and a global attribute."""
    rng = np.random.default_rng(0)
    time = np.datetime64("1979-01-01", "ns") + np.arange(
        N_TIME
    ) * np.timedelta64(1, "D")
    ds = xr.Dataset(
        {
            "prcp": (
                ("time", "nhru"),
                rng.random((N_TIME, N_HRU), dtype=np.float32),
                {"units": "inches", "source_file": sources[0]},
            ),
            "tmax": (
                ("time", "nhru"),
                rng.random((N_TIME, N_HRU)),
                {"units": "degF", "source_file": sources[1]},
            ),
            "hru_area": (("nhru",), np.arange(N_HRU, dtype=np.float64)),
        },
        coords={"time": time},
        attrs={"title": "a toy forcing file"},
    )
    ds.to_netcdf(path)
    return ds


def test_contents(tmp_path):
    source = tmp_path / "cbh.nc"
    path = tmp_path / "cbh_transpose.nc"
    _write_source(source)
    assert transpose_netcdf(source, path) == path
    assert not list(tmp_path.glob("*.partial"))
    with xr.open_dataset(source) as ds, xr.open_dataset(path) as copy:
        assert copy.attrs == ds.attrs
        assert set(copy.variables) == set(ds.variables)
        for name in ("prcp", "tmax"):
            assert ds[name].dims == ("time", "nhru")
            assert copy[name].dims == ("nhru", "time")
            assert copy[name].dtype == ds[name].dtype
            assert copy[name].attrs == ds[name].attrs
            np.testing.assert_array_equal(copy[name].values, ds[name].values.T)
            # stored without chunks
            assert copy[name].encoding["chunksizes"] is None
        np.testing.assert_array_equal(copy["time"].values, ds["time"].values)
        assert copy["time"].encoding["units"] == ds["time"].encoding["units"]
        xr.testing.assert_identical(copy["hru_area"], ds["hru_area"])
    with netCDF4.Dataset(path) as nc:
        assert nc["prcp"].chunking() == "contiguous"
        assert nc["tmax"].chunking() == "contiguous"


def test_transposed_twice_is_the_source(tmp_path):
    source = tmp_path / "cbh.nc"
    _write_source(source)
    once = transpose_netcdf(source, tmp_path / "once.nc")
    twice = transpose_netcdf(once, tmp_path / "twice.nc")
    with xr.open_dataset(source) as ds, xr.open_dataset(twice) as back:
        xr.testing.assert_identical(back, ds)
        for name in ("prcp", "tmax"):
            assert back[name].dtype == ds[name].dtype


@pytest.mark.parametrize("rows_per_slab", [1, 3, N_HRU, 10 * N_HRU])
def test_written_in_slabs(tmp_path, monkeypatch, rows_per_slab):
    """The copy does not depend on the slab: one row of the copy at a
    time, a slab that does not divide the rows, exactly all of them,
    more than all of them."""
    source = tmp_path / "cbh.nc"
    path = tmp_path / "cbh_transpose.nc"
    ds = _write_source(source)
    # one row of the float64 variable's copy is N_TIME * 8 bytes
    monkeypatch.setattr(module, "SLAB_BYTES", rows_per_slab * N_TIME * 8)
    transpose_netcdf(source, path)
    with xr.open_dataset(path) as copy:
        for name in ("prcp", "tmax"):
            np.testing.assert_array_equal(copy[name].values, ds[name].values.T)


def test_reuse_and_force(tmp_path):
    source = tmp_path / "cbh.nc"
    path = tmp_path / "cbh_transpose.nc"
    _write_source(source)
    transpose_netcdf(source, path)
    # a rewrite renames a new file into place: a new inode
    inode = path.stat().st_ino
    assert transpose_netcdf(source, path) == path
    assert path.stat().st_ino == inode
    transpose_netcdf(source, path, force=True)
    assert path.stat().st_ino != inode

    # the same variables written from other CBH files: not this copy
    other = tmp_path / "other.nc"
    _write_source(other, sources=("elsewhere/prcp.cbh", "tmax.cbh"))
    with pytest.raises(FileExistsError, match="not the transposed copy"):
        transpose_netcdf(other, path)
    transpose_netcdf(other, path, force=True)
    with xr.open_dataset(path) as copy:
        assert copy["prcp"].attrs["source_file"] == "elsewhere/prcp.cbh"

    # a file in the source's own order is not its transposed copy
    with pytest.raises(FileExistsError, match="not the transposed copy"):
        transpose_netcdf(source, other)


def test_raises(tmp_path):
    source = tmp_path / "cbh.nc"
    _write_source(source)
    with pytest.raises(ValueError, match="source and path are both"):
        transpose_netcdf(source, source)
    static = tmp_path / "static.nc"
    xr.Dataset({"hru_area": (("nhru",), np.arange(3.0))}).to_netcdf(static)
    with pytest.raises(ValueError, match="nothing to transpose"):
        transpose_netcdf(static, tmp_path / "static_transpose.nc")
    assert not (tmp_path / "static_transpose.nc").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_main(tmp_path, capsys):
    source = tmp_path / "cbh.nc"
    path = tmp_path / "cbh_transpose.nc"
    _write_source(source)
    main([str(source), str(path)])
    assert capsys.readouterr().out.strip() == str(path)
    inode = path.stat().st_ino
    main([str(source), str(path), "--force"])
    assert path.stat().st_ino != inode
