"""The pre-processing suite: execute derivations, stamp, verify.

The contract lists DERIVABLE parameters and derivable map weights --
required at assembly, obtainable by their stated derivations. By
default the assembler executes those derivations IN-CHAIN (in memory,
from the very arrays being supplied), so alignment is guaranteed by
construction. This module is the OPT-IN alternative for workflows
where the derivation inputs are fixed across many runs (forcing
ensembles, repeated experiments): compute the artifacts ONCE, save
them WITH PROVENANCE STAMPS, and verify the stamps at every use.

The stamp design (JLM, Aug 2026): each artifact variable carries

- ``derivation``: the contract's derivation string (which, by
  convention, NAMES ITS INPUTS);
- ``derived_from_<input>``: a sha256 digest (dtype + shape + bytes)
  of each named input AS THE DERIVATION SAW IT;
- ``digest``: the artifact's own digest (catches file tampering or
  corruption independent of the inputs).

Verification recomputes the input digests from the arrays actually
being supplied to the current run and compares -- a mismatch raises
loudly, naming the artifact and the drifted input. Digests match only
on BIT-identical arrays: in this codebase that is the standard, not a
fragility (parameters are parsed to exact float64 for this reason) --
a perturbed ``hru_slope`` (calibration, ensembles over static
parameters) flags every artifact derived from it as stale. Those
workflows should not use cached artifacts at all: the in-chain
default exists for them.
"""

import hashlib
import pathlib as pl

import numpy as np
import xarray as xr

from pws_phoenix.atmosphere.prms_solar_geometry import compute_soltabs
from pws_phoenix.discretization import Discretization
from pws_phoenix.hydrology.prms_stream_temp import derive_aggregation_weights
from pws_phoenix.maps import SparseWeights
from pws_phoenix.prms_translate.control import from_control
from pws_phoenix.prms_translate.parameters import (
    derive_hru_in_to_cf,
    volume_map_weights,
)
from pws_phoenix.prms_translate.readers import load_control, load_parameters

# artifact name -> the weights key of derive_aggregation_weights
_AGG_WEIGHTS = ("flow", "swrad", "met", "humid")


def digest_array(arr: np.ndarray) -> str:
    """sha256 over dtype + shape + bytes: matches only bit-identical
    arrays (deliberately -- see the module docstring)."""
    aa = np.ascontiguousarray(arr)
    hh = hashlib.sha256()
    hh.update(str(aa.dtype).encode())
    hh.update(str(aa.shape).encode())
    hh.update(aa.tobytes())
    return "sha256:" + hh.hexdigest()


def _stamp(
    da: xr.DataArray,
    derivation: str,
    inputs: dict[str, np.ndarray],
) -> xr.DataArray:
    da = da.copy()
    da.attrs["derivation"] = derivation
    for name, arr in inputs.items():
        da.attrs[f"derived_from_{name}"] = digest_array(arr)
    da.attrs["digest"] = digest_array(da.values)
    return da


def _stamp_weights(
    out: xr.Dataset,
    name: str,
    weights: SparseWeights,
    derivation: str,
    inputs: dict[str, np.ndarray],
) -> None:
    """Add a map-weights matrix to `out` in its file form
    (``SparseWeights.to_dataset``: ``<name>_rows``, ``<name>_cols``,
    ``<name>_vals``), each of the three variables stamped."""
    for var_name, da in weights.to_dataset(name).data_vars.items():
        out[var_name] = _stamp(da, derivation, inputs)


def derive_preprocessed(params: xr.Dataset, stream_temp: bool) -> xr.Dataset:
    """Execute the derivations on `params` and return the stamped
    artifacts (no file is written): the derivable parameters (solar
    tables, segment_order, hru_in_to_cf) and the map-weights matrices
    (the lateral-volume 0/1 matrix; with `stream_temp`, the
    stream-temperature aggregation matrices), each matrix as its
    nonzero entries (``SparseWeights.to_dataset``).

    The stamps hold digests of `params`' arrays, so derive from the
    parameters a run will be given: ``verify_preprocessed`` compares
    against those."""
    out = xr.Dataset()

    soltabs = compute_soltabs(
        params[["hru_slope", "hru_aspect", "hru_lat"]], hru_dim="nhru"
    )
    soltab_inputs = {
        nn: params[nn].values for nn in ("hru_slope", "hru_aspect", "hru_lat")
    }
    for name in ("soltab_potsw", "soltab_horad_potsw"):
        out[name] = _stamp(
            soltabs[name],
            "compute_soltabs(hru_slope, hru_aspect, hru_lat)",
            soltab_inputs,
        )

    out["hru_in_to_cf"] = _stamp(
        derive_hru_in_to_cf(params),
        "hru_area * 43560.0 / 12.0",
        {"hru_area": params["hru_area"].values},
    )

    seg_dis = Discretization(
        ["nsegment"],
        parameters=xr.Dataset({"tosegment": params["tosegment"]}),
        topo_order={"segment_order": "tosegment"},
    )
    assert seg_dis.parameters is not None
    segment_order = seg_dis.parameters["segment_order"].values.astype(np.int64)
    out["segment_order"] = _stamp(
        xr.DataArray(segment_order, dims=("nsegment",)),
        "Discretization(topo_order={'segment_order': 'tosegment'})",
        {"tosegment": params["tosegment"].values},
    )

    _stamp_weights(
        out,
        "weights_vol",
        volume_map_weights(params),
        "prms_translate.volume_map_weights(params): 0/1 assignment "
        "matrix from hru_segment",
        {"hru_segment": params["hru_segment"].values},
    )

    if stream_temp:
        agg_inputs = {
            "hru_segment": params["hru_segment"].values,
            "hru_area": params["hru_area"].values,
            "tosegment": params["tosegment"].values,
            "segment_order": segment_order,
            "seg_close": params["seg_close"].values,
        }
        agg = derive_aggregation_weights(
            params["hru_segment"].values,
            params["hru_area"].values,
            params["tosegment"].values,
            segment_order,
            params["seg_close"].values,
        )
        for key in _AGG_WEIGHTS:
            _stamp_weights(
                out,
                f"weights_{key}",
                SparseWeights.from_dense(agg[key]),
                "derive_aggregation_weights(hru_segment, hru_area, "
                "tosegment, segment_order, seg_close)",
                agg_inputs,
            )

    out.attrs["note"] = (
        "prms_translate preprocessed artifacts; stamps verified at "
        "assembly (see prms_translate/preprocess.py)"
    )
    return out


def write_preprocessed(
    control_file: str | pl.Path, path: str | pl.Path
) -> pl.Path:
    """Execute the derivations of the configuration `control_file`
    names (``derive_preprocessed`` on its parameter file(s)) and save
    the artifacts to ONE stamped NetCDF. Returns `path`. Consume with
    ``assemble_from_control(..., preprocessed=path)`` -- stamps are
    verified there. A domain translated ahead of time gets this file
    from ``translate_domain`` instead."""
    control_file = pl.Path(control_file)
    ctl = load_control(control_file)
    cfg = from_control(ctl, control_file)
    params = load_parameters(cfg.param_files)
    out = derive_preprocessed(params, "prms_stream_temp" in cfg.classes)
    out.attrs["control_file"] = str(control_file)
    path = pl.Path(path)
    out.to_netcdf(path)
    return path


def verify_preprocessed(artifacts: xr.Dataset, params: xr.Dataset) -> None:
    """Verify every artifact's stamps against the arrays being
    supplied NOW: its own digest (tamper/corruption) and one digest
    per named derivation input. Inputs resolve from `params` first,
    then from the artifacts themselves (segment_order feeds the
    aggregation weights and is itself a verified artifact -- a chain
    of trust back to tosegment). Raises ValueError naming the
    artifact and the drifted input."""
    for name in artifacts.data_vars:
        da = artifacts[name]
        own = da.attrs.get("digest")
        if own is not None and digest_array(da.values) != own:
            raise ValueError(
                f"preprocessed artifact {name!r} does not match its "
                "own digest (file tampered or corrupted): regenerate "
                "with write_preprocessed()."
            )
        for key, stamped in da.attrs.items():
            if not key.startswith("derived_from_"):
                continue
            input_name = key[len("derived_from_") :]
            if input_name in params:
                current = params[input_name].values
            elif input_name in artifacts:
                current = artifacts[input_name].values
            else:
                raise ValueError(
                    f"preprocessed artifact {name!r}: derivation "
                    f"input {input_name!r} not found in the current "
                    "supply to verify against."
                )
            if digest_array(np.asarray(current)) != stamped:
                raise ValueError(
                    f"preprocessed artifact {name!r} is STALE: its "
                    f"derivation input {input_name!r} differs from "
                    "the one it was derived from. Regenerate with "
                    "write_preprocessed() (for a translated domain: "
                    "translate_domain --only preprocessed --force), or "
                    "use the in-chain default (no preprocessed= "
                    "argument)."
                )


def cut_preprocessed(
    artifacts: xr.Dataset, params: xr.Dataset, hru_positions: np.ndarray
) -> xr.Dataset:
    """The artifacts of a domain cut to some of its hrus
    (``cut_translation``): each artifact on ``nhru`` is CUT to
    `hru_positions`, so it holds the source's values exactly, and is
    stamped again against `params`, the cut's parameters (the inputs
    named by its own ``derived_from_*`` attributes). Every other
    artifact is dropped: the segment order, and the map weights, whose
    entries index hrus by position. Raises ValueError for an artifact
    on ``nhru`` with a derivation input that `params` lacks."""
    out = xr.Dataset()
    # one read of the span from the first position to the last, then
    # the hrus out of it in memory
    first = int(hru_positions[0])
    span = slice(first, int(hru_positions[-1]) + 1)
    for name in artifacts.data_vars:
        da = artifacts[name]
        if "nhru" not in da.dims:
            continue
        input_names = [
            str(key)[len("derived_from_") :]
            for key in da.attrs
            if str(key).startswith("derived_from_")
        ]
        missing = [nn for nn in input_names if nn not in params]
        if missing:
            raise ValueError(
                f"preprocessed artifact {name!r}: its derivation "
                f"inputs {missing} are not in the cut's parameters."
            )
        out[name] = _stamp(
            da.isel(nhru=span).load().isel(nhru=hru_positions - first),
            da.attrs["derivation"],
            {nn: params[nn].values for nn in input_names},
        )
    out.attrs = dict(artifacts.attrs)
    return out
