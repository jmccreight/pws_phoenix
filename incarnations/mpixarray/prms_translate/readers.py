"""pyPRMS decoding shim: control + parameter files at FULL precision.

pyPRMS deficiency worked around here (upstream ask #1): ParameterFile
parses PRMS type-F parameters as float32 AT THE TEXT LINE
(ParameterFile.py `np.array(..., dtype=PTYPE_TO_DTYPE[param_dtype])`,
with the type code from the parameter file itself); the metadata
datatype only recasts AFTERWARD, so the ~1e-9 truncation is baked in.
That is fatal for pws_phoenix's bitwise parity standards (pywatershed
parses the same text at float64). The fix is two-part and VERIFIED
exact against pywatershed's float64 parse on drb_2yr:

1. patch ``pyPRMS.constants.PTYPE_TO_DTYPE[2] = np.float64`` (the
   parse-time map; a module-global -- it makes every in-process pyPRMS
   float parse MORE precise, never less);
2. widen the metadata ``datatype`` declarations float32 -> float64 (or
   the Parameter data setter re-truncates).

Dim conventions normalized here (not a pyPRMS bug): 2-D parameters
arrive ``(nhru, nmonths)``; pws_phoenix/pywatershed store
``(nmonths, nhru)`` -- transposed on load. Integer parameters arrive
int32; widened to int64 (the framework's declared integer dtype).
These load-time conversions are input PREPARATION allocations, made
once, outside the model (memory prime directive: not model buffers).
"""

import pathlib as pl

import numpy as np
import pyPRMS.constants
import xarray as xr
from pyPRMS import ControlFile, MetaData, ParameterFile

# the NHM files in use are PRMS 5.2.1.1
PRMS_VERSION = "5.2.1.1"


def _ensure_f64_parse() -> None:
    """Patch pyPRMS to parse PRMS type-F (float) parameter text as
    float64 (see the module docstring; upstream ask #1)."""
    pyPRMS.constants.PTYPE_TO_DTYPE[2] = np.float64


def prms_metadata(version: str = PRMS_VERSION) -> dict:
    """The pyPRMS metadata dict for `version`, with every parameter's
    declared datatype widened float32 -> float64 (part 2 of the
    full-precision parse; see the module docstring)."""
    md = MetaData(version=version).metadata
    for meta in md["parameters"].values():
        if meta["datatype"] == "float32":
            meta["datatype"] = "float64"
    return md


def load_control(path: str | pl.Path) -> ControlFile:
    """Read a PRMS control file (pyPRMS ControlFile: ``.get(name)``,
    ``.modules``, ``.cbh_files``, ``.dynamic_parameters``). NOTE:
    avoid pyPRMS's ``.additional_modules`` -- it assumes output-flag
    variables (basinOutON_OFF, ...) that NHM control files omit, and
    raises."""
    return ControlFile(
        str(path), metadata=prms_metadata(), verbose=False
    )


def load_parameters(path: str | pl.Path) -> xr.Dataset:
    """Read a PRMS parameter file (myparam.param) into ONE flat
    xr.Dataset at pws_phoenix conventions: float64 (exact vs a direct
    float64 parse of the text), int64, and 2-D parameters transposed
    to ``(nmonths, nhru)``. Scalar parameters keep their ``(one,)``
    dim (packaging decides how to serve them)."""
    _ensure_f64_parse()
    pf = ParameterFile(str(path), metadata=prms_metadata(), verbose=False)
    data_vars = {}
    for name, par in pf.parameters.items():
        dims = tuple(par.dimensions.keys())
        # pyPRMS returns PYTHON SCALARS for ('one',)-dim parameters;
        # keep them (1,)-shaped arrays like everything else
        vals = np.atleast_1d(np.asarray(par.data))
        if vals.ndim == 2:
            dims = dims[::-1]
            vals = vals.T
        if vals.dtype == np.int32:
            vals = vals.astype(np.int64)
        data_vars[name] = (dims, vals)
    return xr.Dataset(data_vars)
