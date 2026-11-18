"""CBH ASCII -> pws_phoenix external-input DataArrays (via pyPRMS).

pyPRMS's Cbh parses the ASCII straight to an xr.Dataset with a real
time axis and PRMS-internal variable names (hru_ppt, tmax_hru, ...,
humidity_hru -- note pyPRMS itself performs the rhavg -> humidity_hru
mapping). This module owns the remaining pws_phoenix semantics:

- rename to the pywatershed-verbatim input names (prcp/tmax/tmin;
  humidity_hru stays -- it feeds the hru->segment Map by that name);
- float32 -> float64 widening (CBH files store float32, the model
  computes float64; exact, and done at input preparation -- a
  load-time allocation, not a model buffer);
- slice to the control's simulation window.

``write_cbh_netcdf`` is the one-time translation of the CBH text to
ONE combined forcing netCDF (a large domain's text parse costs minutes
and tens of GB per file; see its docstring).
"""

import pathlib as pl
from collections.abc import Mapping
import warnings

import numpy as np
import xarray as xr
from pyPRMS import Cbh

from pws_phoenix.prms_translate import prms_units
from pws_phoenix.prms_translate.readers import prms_metadata

# pyPRMS/PRMS-internal -> pws_phoenix (pywatershed-verbatim) names
_CBH_RENAMES = {
    "hru_ppt": "prcp",
    "tmax_hru": "tmax",
    "tmin_hru": "tmin",
    # the OpenET observed-AET CBH (the control's AET_cbh_file) -> the
    # PRMSSoilzoneAgObsET input; missing stays -1.0 (the kernel's own
    # missing-value convention)
    "actet": "aet_observed",
}


def load_cbh(
    path: str | pl.Path,
    start_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
) -> xr.DataArray:
    """One CBH file -> one float64 (time, nhru) DataArray at its
    pws_phoenix input name, optionally sliced to [start, end]."""
    with warnings.catch_warnings():
        # suppress pyPRMS/pandas conversion FutureWarnings; note the
        # printed "No control object provided" NOTICE is a plain
        # print inside pyPRMS and passes through -- harmless (the
        # data are unaffected), just noisy
        warnings.simplefilter("ignore")
        ds = Cbh(str(path), metadata=prms_metadata()).data
    names = [str(nn) for nn in ds.data_vars]
    if len(names) != 1:
        raise ValueError(
            f"CBH file {path}: expected exactly one variable, got {names}."
        )
    da = ds[names[0]].rename(_CBH_RENAMES.get(names[0], names[0]))
    # pyPRMS stamps a 1-based OBJECT-dtype nhru index coordinate;
    # dropped (object coords break zarr encoding downstream, and the
    # framework's space dims are coordinate-free)
    da = da.drop_vars([cc for cc in da.coords if cc != "time"])
    if start_time is not None or end_time is not None:
        da = da.sel(time=slice(start_time, end_time))
    da = da.astype(np.float64)
    # pyPRMS's CBH units ("in", "degree_fahrenheit") -> vocabulary key
    key = prms_units.resolve(str(da.name), da.attrs.get("units"))
    if key is not None:
        da.attrs["units"] = key
    return da


def cbh_netcdf_sources(path: str | pl.Path) -> dict[str, str | None]:
    """The CBH file each variable of a write_cbh_netcdf product was
    written from: variable name -> its ``source_file`` attribute."""
    with xr.open_dataset(path) as ds:
        return {
            str(name): ds[name].attrs.get("source_file")
            for name in ds.data_vars
        }


def write_cbh_netcdf(
    cbh_paths: Mapping[str, str | pl.Path],
    path: str | pl.Path,
    force: bool = False,
) -> pl.Path:
    """CBH text files -> ONE combined forcing netCDF: a (time, nhru)
    variable per pws_phoenix input name (`cbh_paths` is keyed as
    ``PrmsRunConfig.cbh_paths`` is). Returns `path`.

    The WHOLE of each file is written, with no simulation window: the
    product belongs to the CBH files, not to one control file.

    Each variable is float32 when that holds its values exactly, else
    float64, and carries a ``source_file`` attribute (its CBH file's
    resolved path). One variable is in memory at a time: each is
    parsed (load_cbh) and appended to the file before the next. All
    variables must share one time axis. No `nhru` coordinate, no
    compression, no chunking.

    An existing `path` is REUSED (nothing is parsed) when its
    variables' ``source_file`` attributes equal `cbh_paths`; when they
    differ, it raises. ``force=True`` rewrites it in either case.

    The file is written under ``<path>.partial`` and renamed at the
    end, so a file at `path` is always a complete one."""
    path = pl.Path(path)
    sources = {
        str(name): str(pl.Path(source).resolve())
        for name, source in cbh_paths.items()
    }
    if not sources:
        raise ValueError("write_cbh_netcdf: cbh_paths is empty.")
    if path.exists() and not force:
        found = cbh_netcdf_sources(path)
        if found == sources:
            return path
        raise FileExistsError(
            f"{path} exists and was not written from the requested CBH "
            f"files. Its source_file attributes: {found}. Requested: "
            f"{sources}. Pass force=True to rewrite it."
        )

    partial = path.with_name(path.name + ".partial")
    time: np.ndarray | None = None
    for name, source in sources.items():
        da = load_cbh(source)
        if da.name != name:
            raise ValueError(
                f"CBH file {source} holds {da.name!r}, but it was "
                f"given for {name!r}."
            )
        # a second, half-size copy of the variable; a translation-time
        # allocation, released with the float64 array before the next
        # file is parsed
        narrow = da.astype(np.float32)
        if np.array_equal(narrow.values, da.values, equal_nan=True):
            da = narrow
        da.attrs["source_file"] = source
        if time is None:
            time = da["time"].values
            da.to_dataset().to_netcdf(partial, mode="w")
        else:
            if not np.array_equal(da["time"].values, time):
                raise ValueError(
                    f"CBH file {source} ({name}): its time axis "
                    f"({da.sizes['time']} days from "
                    f"{da['time'].values[0]}) differs from that of "
                    f"{next(iter(sources.values()))} ({time.size} days "
                    f"from {time[0]})."
                )
            # the time coordinate is in the file already
            da.drop_vars("time").to_dataset().to_netcdf(partial, mode="a")
    partial.replace(path)
    return path
