"""Legacy PRMS files -> a PARALLEL (ModelMPI) run.

The serial ModelKit lives entirely in memory; a parallel run cannot:
mpixarray STREAMS the distributed grid from ONE combined NetCDF on
disk, while the serial (segment) grid is rebuilt identically on every
rank. So the parallel path has two halves:

- ``write_mpi_input_file(kit, path, n_days=None)`` -- the PREP half,
  run serially (a notebook cell, a script): merge the kit's
  distributed-grid content into the combined input file -- static
  parameters + initial values as ``(nhru, ...)``, forcings as
  ``(time, nhru)``, real dim-coordinates. ``n_days`` truncates the
  time axis (the stream defines the run length).
- ``mpi_model_from_control(control_file, control)`` -- the parallel
  one-liner, callable ONLY under mpirun (SPMD: every rank calls it):
  rebuilds the kit from the legacy files (deterministic, so the
  replicated serial grid is identical across ranks), strips the
  distributed grid's in-memory data (the input file serves it),
  converts every Map to MapMPI, and constructs ModelMPI.

Required control keys beyond the serial dict: ``input_file`` (from
the prep half) and ``output_parallel_netcdf`` (the parallel stream);
``mpi_grid`` defaults to ``"nhru"``. Output: with no
``output_var_names`` of your own, the control file's ``*OutVar_names``
are injected PER DESTINATION -- distributed-grid names stream to
``output_parallel_netcdf`` (any number; the one-variable mpixarray
limit fell Oct 28, 2026), serial-grid names go to the rank-0 zarr only
if ``output_serial_zarr`` was supplied (else they are left out). An
explicit list wins verbatim. Dropped names are not reported here (no
ModelKit is returned; see assemble_from_control).
"""

import pathlib as pl
from typing import Any

import numpy as np
import xarray as xr

from pws_phoenix.maps import MapMPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.prms_translate.assemble import (
    ModelKit,
    assemble_from_control,
    requested_output_names,
)


def write_mpi_input_file(
    kit: ModelKit,
    path: str | pl.Path,
    n_days: int | None = None,
    mpi_grid: str = "nhru",
) -> pl.Path:
    """Write the combined distributed-grid input file from a (serial)
    ModelKit: every mpi-grid process entry's parameter dataset plus
    its loose DataArray entries (forcings, initial-value seams),
    merged into one dataset with real dim-coordinates. Returns the
    path."""
    merged = xr.Dataset()
    for entry in kit.process_dict.values():
        if entry["discretization"] != mpi_grid:
            continue
        if "parameters" in entry:
            # same-named parameters across processes carry identical
            # values (packaged from one flat source): first wins
            merged = merged.merge(entry["parameters"], compat="override")
        for key, val in entry.items():
            if key in ("class", "discretization", "parameters"):
                continue
            if isinstance(val, xr.DataArray) and key not in merged:
                merged[key] = val
    if n_days is not None:
        merged = merged.isel(time=slice(0, int(n_days)))
    # real dim-coordinates for parallelize/as_iterable; ns times
    merged = merged.assign_coords(
        {mpi_grid: np.arange(merged.sizes[mpi_grid])}
    )
    if "time" in merged.coords:
        merged = merged.assign_coords(
            time=merged["time"].values.astype("datetime64[ns]")
        )
    path = pl.Path(path)
    merged.to_netcdf(path)
    return path


def mpi_model_from_control(
    control_file: str | pl.Path,
    control: dict[str, Any],
) -> ModelMPI:
    """The parallel one-liner (call it on EVERY rank, under mpirun).
    See the module docstring for the required control keys; the
    distributed grid's data come from ``control["input_file"]``
    (written by write_mpi_input_file), everything else is rebuilt
    identically per rank from the legacy files."""
    control = dict(control)
    mpi_grid = control.setdefault("mpi_grid", "nhru")
    missing = [
        kk
        for kk in ("input_file", "output_parallel_netcdf")
        if kk not in control
    ]
    if missing:
        raise ValueError(
            f"mpi_model_from_control: control needs {missing} (see "
            "prms_translate.assemble_mpi; write_mpi_input_file "
            "produces input_file)."
        )

    kit = assemble_from_control(control_file)

    process_dict: dict[str, dict[str, Any]] = {}
    mpi_vars: set[str] = set()
    for slot, entry in kit.process_dict.items():
        if entry["discretization"] == mpi_grid:
            # data ride in input_file; keep class + grid only
            process_dict[slot] = {
                "class": entry["class"],
                "discretization": mpi_grid,
            }
            mpi_vars |= set(entry["class"].get_var_names())
        else:
            process_dict[slot] = entry

    # The control file's output request, per destination (the mirror of
    # the serial gate "injected only when a store was supplied"): the
    # parallel NetCDF is mandatory, the serial zarr is the caller's.
    if "output_var_names" not in control and kit.config.output_var_names:
        kept, _dropped = requested_output_names(kit.config, kit.process_dict)
        control["output_var_names"] = [
            nn
            for nn in kept
            if nn in mpi_vars or "output_serial_zarr" in control
        ]

    maps = {
        name: MapMPI(
            weights=mm.weights,
            grid={mm.source_grid: mm.target_grid},
            variable={mm.source_var: mm.target_var},
            derivation=mm.derivation,
        )
        for name, mm in kit.maps.items()
    }
    discretizations = {
        gg: dd for gg, dd in kit.discretizations.items() if gg != mpi_grid
    }
    return ModelMPI(
        process_dict, control, maps=maps, discretizations=discretizations
    )
