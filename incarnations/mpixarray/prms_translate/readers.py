"""pyPRMS decoding shim: control + parameter files at FULL precision.

pyPRMS deficiencies worked around here (the upstream asks):

1. (ask #1) ParameterFile parses PRMS type-F parameters as float32 AT
   THE TEXT LINE (ParameterFile.py `np.array(..., dtype=
   PTYPE_TO_DTYPE[param_dtype])`, with the type code from the
   parameter file itself); the metadata datatype only recasts
   AFTERWARD, so the ~1e-9 truncation is baked in. That is fatal for
   pws_phoenix's bitwise parity standards (pywatershed parses the same
   text at float64). The fix is two-part and VERIFIED exact against
   pywatershed's float64 parse on drb_2yr: patch
   ``pyPRMS.constants.PTYPE_TO_DTYPE[2] = np.float64`` (the parse-time
   map; a module-global -- it makes every in-process pyPRMS float
   parse MORE precise, never less) AND widen the metadata ``datatype``
   declarations float32 -> float64 (or the Parameter data setter
   re-truncates).
2. (ask #3) pyPRMS's master metadata predates the GSFLOW agricultural
   extensions: ParameterFile SILENTLY SKIPS any parameter its
   metadata does not know (fgr_ag_2yr's myparam.param loses all 13 ag
   parameters), ControlFile RAISES on unknown control entries
   (analysis.control cannot load), and Cbh raises KeyError on the
   OpenET ``actet`` variable. All injected in ``prms_metadata``
   below.
3. (ask #4) a multi-valued ``param_file`` control entry (the PRMS
   multiple-parameter-file feature, e.g. ``myparam.param`` +
   ``transp_frost.param``) breaks ControlFile: the entry's metadata
   context is ``scalar``, so the reader allocates
   ``np.zeros(n, dtype=np.str_)`` (dtype '<U1') and truncates every
   path to its first character before the values setter rejects the
   array. Fixed by flipping the entry's context to ``array``
   (``from_control`` normalizes single/multi to a list either way).
4. (ask #5) ParameterFile cannot read a PARTIAL parameter file (the
   secondary files of the multi-file feature, e.g.
   ``transp_frost.param``: parameter blocks only, no header/dimension
   sections) -- its header scan consumes the whole file looking for
   the ``** Dimensions **`` marker. ``load_parameters`` synthesizes
   the missing sections from the files already read and hands pyPRMS
   the result, so the actual PARSING stays 100% pyPRMS.
5. (ask #6) every ``*_dynamic`` control entry (the dynamic-parameter
   file paths) ships with ``force_default: True`` in the control
   metadata, so the values GETTER returns the default path and the
   one written in the control file is unreachable. Stripped in
   ``prms_metadata``.

Issue drafts for all of these -- each demonstrated (and two FIXED) by
a feature branch in the mpix/pyPRMS clone -- live in
``prms_translate/UPSTREAM_ASKS.md``, together with the migration map
of what dies here when each ask lands upstream.

Dim conventions normalized here (not a pyPRMS bug): 2-D parameters
arrive ``(nhru, nmonths)``; pws_phoenix/pywatershed store
``(nmonths, nhru)`` -- transposed on load. Integer parameters arrive
int32; widened to int64 (the framework's declared integer dtype).
These load-time conversions are input PREPARATION allocations, made
once, outside the model (memory prime directive: not model buffers).
"""

import pathlib as pl
import tempfile
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pyPRMS.constants
import xarray as xr
from pyPRMS import ControlFile, MetaData, ParameterFile

# the NHM files in use are PRMS 5.2.1.1
PRMS_VERSION = "5.2.1.1"

# GSFLOW agricultural parameters absent from pyPRMS's master metadata
# (ask #3 above): name -> (datatype, dimension, description);
# dims/dtypes verified against the fgr_ag_2yr parameter file blocks
_AG_PARAMETER_METADATA = {
    "ag_cov_type": (
        "int32",
        "nhru",
        "Vegetation cover type of the agricultural area",
    ),
    "ag_covden_sum": (
        "float64",
        "nhru",
        "Summer vegetation cover density of the agricultural area",
    ),
    "ag_covden_win": (
        "float64",
        "nhru",
        "Winter vegetation cover density of the agricultural area",
    ),
    "ag_frac": (
        "float64",
        "nhru",
        "Fraction of each HRU that is agricultural",
    ),
    "ag_soil2gw_max": (
        "float64",
        "nhru",
        "Maximum agricultural soil-water routed directly to "
        "groundwater [inches]",
    ),
    "ag_soil_moist_init_frac": (
        "float64",
        "nhru",
        "Initial fraction of agricultural soil-moisture capacity",
    ),
    "ag_soil_moist_max": (
        "float64",
        "nhru",
        "Maximum agricultural soil-moisture capacity [inches]",
    ),
    "ag_soil_rechr_init_frac": (
        "float64",
        "nhru",
        "Initial fraction of agricultural recharge-zone capacity",
    ),
    "ag_soil_rechr_max_frac": (
        "float64",
        "nhru",
        "Maximum agricultural recharge zone as fraction of ag_soil_moist_max",
    ),
    "ag_soil_type": (
        "int32",
        "nhru",
        "Soil type of the agricultural area",
    ),
    "ag_soilwater_deficit_min": (
        "float64",
        "nhru",
        "Minimum agricultural soil-water deficit to begin irrigation",
    ),
    "max_soilzone_ag_iter": (
        "int32",
        "one",
        "Maximum soilzone iterations to match the AET target",
    ),
    "soilzone_aet_converge": (
        "float64",
        "one",
        "Convergence criterion for the iterative AET match",
    ),
}

# GSFLOW agricultural CONTROL entries absent from pyPRMS's control
# metadata (part of ask #3; ControlFile.add RAISES on unknown entries,
# so fgr_ag_2yr's analysis.control cannot load at all without these):
# name -> (datatype, default, description)
_AG_CONTROL_METADATA = {
    "iter_aet_flag": (
        "int32",
        np.int32(0),
        "Flag for iterative soilzone AET matching",
    ),
    "forcing_check_flag": (
        "int32",
        np.int32(0),
        "Flag to check forcing data",
    ),
    "dyn_ag_frac_flag": (
        "int32",
        np.int32(0),
        "Flag for dynamic ag_frac",
    ),
    "AET_cbh_file": (
        "string",
        np.str_("AET.cbh"),
        "Pathname of the CBH file of observed actual ET",
    ),
    "PET_cbh_file": (
        "string",
        np.str_("PET.cbh"),
        "Pathname of the CBH file of observed potential ET",
    ),
    "ag_frac_dynamic": (
        "string",
        np.str_("dyn_ag_frac.param"),
        "Pathname of the dynamic ag_frac parameter file",
    ),
    "AET_module": (
        "string",
        np.str_("climate_hru"),
        "Module for observed actual ET",
    ),
    "PET_ag_module": (
        "string",
        np.str_("climate_hru"),
        "Module for agricultural potential ET",
    ),
}


def _ensure_f64_parse() -> None:
    """Patch pyPRMS to parse PRMS type-F (float) parameter text as
    float64 (see the module docstring; upstream ask #1)."""
    pyPRMS.constants.PTYPE_TO_DTYPE[2] = np.float64


def prms_metadata(version: str = PRMS_VERSION) -> dict:
    """The pyPRMS metadata dict for `version`, adjusted per the module
    docstring: every parameter's declared datatype widened float32 ->
    float64 (part 2 of the full-precision parse), the GSFLOW
    agricultural parameters injected, the OpenET ``actet`` CBH
    variable added, and ``param_file`` flipped to array context (the
    PRMS multiple-parameter-file feature)."""
    md = MetaData(version=version).metadata
    for meta in md["parameters"].values():
        if meta["datatype"] == "float32":
            meta["datatype"] = "float64"
    # master entries are defaultdict(list) -- pyPRMS leans on missing
    # keys resolving to [] -- so injected entries must be too
    for name, (dtype, dim, desc) in _AG_PARAMETER_METADATA.items():
        md["parameters"][name] = defaultdict(
            list,
            {
                "datatype": dtype,
                "description": desc,
                "units": "none",
                "dimensions": [dim],
                "modules": ["soilzone_ag"],
            },
        )
    # float64: pyPRMS parses CBH text at the metadata dtype, and
    # pywatershed's aet converter (the oracle) parses this one at
    # float64 -- unlike its float32-parsed forcings, whose entries
    # therefore stay float32 here
    md["cbh"]["actet"] = defaultdict(
        list,
        {
            "datatype": "float64",
            "description": "Observed actual ET distributed to each "
            "HRU (OpenET)",
            "units": "inches",
            "dimensions": ["nhru"],
            "modules": ["climate_hru"],
        },
    )
    md["control"]["param_file"]["context"] = "array"
    for name, (dtype, default, desc) in _AG_CONTROL_METADATA.items():
        md["control"][name] = {
            "datatype": dtype,
            "description": desc,
            "context": "scalar",
            "default": default,
        }
    # ask #6: force_default on the *_dynamic path entries shadows the
    # control file's actual values with the metadata defaults
    for entry in md["control"].values():
        entry.pop("force_default", None)
    return md


def load_control(path: str | pl.Path) -> ControlFile:
    """Read a PRMS control file (pyPRMS ControlFile: ``.get(name)``,
    ``.modules``, ``.cbh_files``, ``.dynamic_parameters``). NOTE:
    avoid pyPRMS's ``.additional_modules`` -- it assumes output-flag
    variables (basinOutON_OFF, ...) that NHM control files omit, and
    raises."""
    return ControlFile(str(path), metadata=prms_metadata(), verbose=False)


def _is_full_parameter_file(path: pl.Path) -> bool:
    """A FULL parameter file carries the ``** Dimensions **`` section
    marker; the secondary files of the PRMS multiple-parameter-file
    feature are PARTIAL (parameter blocks only)."""
    for line in path.read_text().splitlines():
        if line.strip("* ") == "Dimensions":
            return True
    return False


def _parse_one(path: pl.Path, dims: dict[str, int]) -> xr.Dataset:
    """One parameter file -> a flat Dataset, via pyPRMS. A partial
    file (see ``_is_full_parameter_file``) gets the missing header +
    dimension sections synthesized from `dims` (the dimensions of the
    files already read) so the actual parsing stays pyPRMS (upstream
    ask #5)."""
    if not _is_full_parameter_file(path):
        synthesized = "".join(
            f"{name}\n{size}\n" for name, size in dims.items()
        )
        text = (
            f"synthesized header for partial parameter file {path.name}\n"
            "** Dimensions **\n####\n"
            + synthesized
            + "** Parameters **\n"
            + path.read_text()
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".param", delete=False
        ) as ff:
            ff.write(text)
            tmp = pl.Path(ff.name)
        try:
            return _parse_one(tmp, dims)
        finally:
            tmp.unlink()
    pf = ParameterFile(str(path), metadata=prms_metadata(), verbose=False)
    data_vars = {}
    for name, par in pf.parameters.items():
        par_dims = tuple(par.dimensions.keys())
        # pyPRMS returns PYTHON SCALARS for ('one',)-dim parameters;
        # keep them (1,)-shaped arrays like everything else
        vals = np.atleast_1d(np.asarray(par.data))
        if vals.ndim == 2:
            par_dims = par_dims[::-1]
            vals = vals.T
        if vals.dtype == np.int32:
            vals = vals.astype(np.int64)
        data_vars[name] = (par_dims, vals)
    return xr.Dataset(data_vars)


def load_parameters(
    path: str | pl.Path | Sequence[str | pl.Path],
) -> xr.Dataset:
    """Read PRMS parameter file(s) into ONE flat xr.Dataset at
    pws_phoenix conventions: float64 (exact vs a direct float64 parse
    of the text), int64, and 2-D parameters transposed to
    ``(nmonths, nhru)``. Scalar parameters keep their ``(one,)`` dim
    (packaging decides how to serve them).

    A sequence is the PRMS multiple-parameter-file feature (the
    control's ``param_file`` can list several): the first file must be
    FULL; later files may be partial and OVERRIDE earlier definitions
    (PRMS/pyPRMS "updated with new values" semantics)."""
    _ensure_f64_parse()
    paths = (
        [pl.Path(path)]
        if isinstance(path, (str, pl.Path))
        else [pl.Path(pp) for pp in path]
    )
    if not _is_full_parameter_file(paths[0]):
        raise ValueError(
            f"parameter file {paths[0]}: the first (or only) file "
            "must be a FULL parameter file (with a Dimensions "
            "section); partial files can only follow one."
        )
    ds = _parse_one(paths[0], {})
    for extra_path in paths[1:]:
        sizes = {str(kk): int(vv) for kk, vv in ds.sizes.items()}
        extra = _parse_one(extra_path, sizes)
        for name in extra.data_vars:
            ds[name] = extra[name]
    return ds
