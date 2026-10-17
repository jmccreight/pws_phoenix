"""PRMS dynamic-parameter file reader (upstream ask #2 for pyPRMS).

pyPRMS (0.9.10) has NO reader for PRMS dynamic-parameter files
(dyn_ag_frac.param and kin); pywatershed's PrmsDynamicParameter is
the only existing one, and this package must not depend on
pywatershed (the migrate-to-pyPRMS rule) -- so the READ semantics are
reimplemented here, minimal and pyPRMS-shaped (xarray out). Pinned
against pywatershed's reader in tests/test_prms_translate.py.

Format: free header lines (typically naming the space ids), an
optional ``####`` end-of-header marker, then whitespace-separated
rows ``year month day v_0 v_1 ... v_{n-1}`` at IRREGULAR dates --
each row holds from its date until the next row's date (PRMS
forward-fill semantics, applied by ``forward_fill``).
"""

import pathlib as pl

import numpy as np
import xarray as xr


def load_dynamic_parameter(
    path: str | pl.Path, dim: str = "nhru"
) -> xr.DataArray:
    """Read a dynamic-parameter file -> float64 (time, `dim`)
    DataArray on the file's own (irregular) dates."""
    dates = []
    rows = []
    n_vals = None
    with open(path) as ff:
        in_header = True
        for line in ff:
            parts = line.split()
            if not parts:
                continue
            if in_header:
                if parts[0] == "####":
                    in_header = False
                    continue
                # a row starting with a 4-digit year ends the header
                if not (parts[0].isdigit() and len(parts[0]) == 4):
                    continue
                in_header = False
            yy, mm, dd = (int(pp) for pp in parts[:3])
            vals = np.array(parts[3:], dtype=np.float64)
            if n_vals is None:
                n_vals = vals.size
            elif vals.size != n_vals:
                raise ValueError(
                    f"dynamic parameter file {path}: row for "
                    f"{yy}-{mm}-{dd} has {vals.size} values, "
                    f"expected {n_vals}."
                )
            dates.append(np.datetime64(f"{yy:04d}-{mm:02d}-{dd:02d}"))
            rows.append(vals)
    if not rows:
        raise ValueError(f"dynamic parameter file {path}: no data rows.")
    return xr.DataArray(
        np.stack(rows),
        dims=("time", dim),
        coords={"time": np.array(dates)},
        name=pl.Path(path).stem,
    )


def forward_fill(
    dyn: xr.DataArray, times: xr.DataArray | np.ndarray
) -> xr.DataArray:
    """PRMS semantics: each file date's values hold until the next
    file date -> a (time, space) DataArray on the model time axis
    `times`. Times before the first file date get the first row (the
    PRMS 'active from the start' convention for spin-up windows)."""
    times = np.asarray(times).astype("datetime64[D]")
    file_dates = dyn["time"].values.astype("datetime64[D]")
    idx = np.searchsorted(file_dates, times, side="right") - 1
    idx = np.clip(idx, 0, file_dates.size - 1)
    space_dim = dyn.dims[1]
    return xr.DataArray(
        dyn.values[idx, :],
        dims=("time", space_dim),
        coords={"time": times},
        name=dyn.name,
    )
