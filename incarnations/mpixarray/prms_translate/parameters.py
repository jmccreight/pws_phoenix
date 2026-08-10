"""PRMS flat parameter namespace -> per-process datasets.

The pws_phoenix-SPECIFIC half of parameter translation (does not
migrate to pyPRMS): the split of myparam.param's flat namespace is
driven BY THE CLASS DECLARATIONS (each process's declared
``kind="parameter"`` fields), not by hand-maintained per-process name
lists -- the contract does the deciding (pywatershed's
separate_nhm_params role, contract-driven).

- Derived here: ``hru_in_to_cf`` = hru_area * 43560 / 12 (pywatershed
  prms_parameters.py verbatim; pinned exact in the tests).
- Computed elsewhere, never file-sourced: the soltabs (the
  ``compute_soltabs`` factory; pass them in via ``extra``) and
  ``segment_order`` (the Discretization's ``topo_order``).
- Dim conventions fall out of ONE rule: source dims are renamed
  POSITIONALLY onto the declaration's dims, with ``"space"`` resolved
  to the process grid (PRMS ``one -> scalar`` and
  ``nmonths -> nmonth`` need no special cases).
"""

import numpy as np
import xarray as xr

from process import _dict_of_kind

# pywatershed constants.py verbatim
FT2_PER_ACRE = 43560.0
INCHES_PER_FOOT = 12.0

# supplied via `extra` (the compute_soltabs factory), never from the
# parameter file
COMPUTED_PARAMETERS = ("soltab_potsw", "soltab_horad_potsw")
# owned by the Discretization (topo_order derivation), never packaged
DIS_OWNED_PARAMETERS = ("segment_order",)

# process_dict slot -> grid, for the slots control.resolve_classes
# emits (the NHM two-grid layout)
SLOT_GRIDS = {
    "prms_atmosphere": "nhru",
    "prms_canopy": "nhru",
    "prms_snow": "nhru",
    "prms_runoff": "nhru",
    "prms_soilzone": "nhru",
    "prms_groundwater": "nhru",
    "prms_channel": "nsegment",
    "prms_hydraulic_geometry": "nsegment",
    "prms_stream_temp": "nsegment",
}


def derive_hru_in_to_cf(params: xr.Dataset) -> xr.DataArray:
    """Inches over an HRU -> cubic feet (pywatershed verbatim)."""
    return xr.DataArray(
        params["hru_area"].values * FT2_PER_ACRE / INCHES_PER_FOOT,
        dims=("nhru",),
        name="hru_in_to_cf",
    )


def package_parameters(
    params: xr.Dataset,
    classes: dict[str, type],
    grids: dict[str, str] | None = None,
    extra: dict[str, xr.DataArray] | None = None,
) -> dict[str, xr.Dataset]:
    """Split the flat parameter Dataset (``readers.load_parameters``)
    into one Dataset per process slot, containing exactly what each
    class declares. `extra` supplies the computed parameters (the
    soltabs) and overrides by name; ``hru_in_to_cf`` is derived if
    absent. A declared parameter found nowhere raises, naming the
    parameter and the process."""
    grids = dict(SLOT_GRIDS) if grids is None else grids
    extra = dict(extra or {})
    if "hru_in_to_cf" not in params and "hru_in_to_cf" not in extra:
        extra["hru_in_to_cf"] = derive_hru_in_to_cf(params)

    out: dict[str, xr.Dataset] = {}
    for slot, cls in classes.items():
        if slot not in grids:
            raise KeyError(
                f"process slot {slot!r}: no grid given (grids= or "
                "parameters.SLOT_GRIDS)."
            )
        grid_dim = grids[slot]
        das: dict[str, xr.DataArray] = {}
        for name, meta in _dict_of_kind(cls, "parameter").items():
            if name in DIS_OWNED_PARAMETERS:
                continue
            if name in extra:
                src = extra[name]
            elif name in params:
                src = params[name]
            else:
                raise KeyError(
                    f"parameter {name!r} (process {slot!r}) is not in "
                    "the PRMS parameter file and was not supplied via "
                    "`extra`."
                )
            declared = tuple(
                grid_dim if dd == "space" else dd for dd in meta.dims
            )
            if declared == ("scalar",) and src.values.size > 1:
                # PRMS dimensions some parameters spatially (e.g. the
                # snow densities den_init/den_max/settle_const on
                # nhru) that the ports declare SCALAR: collapse iff
                # the field is uniform, else the port cannot honor it
                vals = src.values
                if not (vals == vals.flat[0]).all():
                    raise ValueError(
                        f"parameter {name!r} (process {slot!r}): "
                        "declared scalar but spatially VARYING in the "
                        "parameter file -- the port assumes a "
                        "constant."
                    )
                das[name] = xr.DataArray(
                    vals.flat[:1].copy(), dims=("scalar",)
                )
                continue
            if len(src.dims) != len(declared):
                raise ValueError(
                    f"parameter {name!r} (process {slot!r}): source "
                    f"dims {src.dims} do not match the declared rank "
                    f"{declared}."
                )
            if tuple(src.dims) != declared:
                src = src.rename(dict(zip(src.dims, declared)))
            das[name] = src
        if das:
            out[slot] = xr.Dataset(das)
    return out


def volume_map_weights(params: xr.Dataset) -> np.ndarray:
    """The 0/1 hru->segment weights for the three lateral-volume Maps
    (from ``hru_segment``; hru_segment == 0 -> no segment)."""
    hru_segment = params["hru_segment"].values
    n_seg = params.sizes["nsegment"]
    weights = np.zeros((n_seg, hru_segment.shape[0]))
    for ii in range(hru_segment.shape[0]):
        if hru_segment[ii] > 0:
            weights[hru_segment[ii] - 1, ii] = 1.0
    return weights
