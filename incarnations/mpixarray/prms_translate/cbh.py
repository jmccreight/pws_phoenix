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
"""

import pathlib as pl
import warnings

import numpy as np
import xarray as xr
from pyPRMS import Cbh

from prms_translate.readers import prms_metadata

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
    return da.astype(np.float64)
