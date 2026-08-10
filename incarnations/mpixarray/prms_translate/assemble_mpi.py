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
``mpi_grid`` defaults to ``"nhru"``. NOTE the known mpixarray limit:
at most ONE distributed-grid variable may be streamed to disk
(``output_var_names``); serial-grid output goes to the rank-0 zarr
(``output_serial_zarr``) as usual. PRMS-control output names are NOT
auto-injected on the MPI path -- give ``output_var_names``
explicitly.
"""

import pathlib as pl
from typing import Any

import numpy as np
import xarray as xr

from map import MapMPI
from model import ModelMPI
from prms_translate.assemble import ModelKit, assemble_from_control


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
    # real dim-coordinates for parallelize/set_streaming; ns times
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

    streamed = [
        nn for nn in control.get("output_var_names", ()) if nn in mpi_vars
    ]
    if len(streamed) > 1:
        raise NotImplementedError(
            "mpixarray streams at most ONE distributed-grid output "
            f"variable today; requested {streamed}. Serial-grid "
            "variables (the rank-0 zarr) are not limited."
        )

    maps = {
        name: MapMPI(
            weights=mm.weights,
            grid={mm.source_grid: mm.target_grid},
            variable={mm.source_var: mm.target_var},
        )
        for name, mm in kit.maps.items()
    }
    discretizations = {
        gg: dd
        for gg, dd in kit.discretizations.items()
        if gg != mpi_grid
    }
    return ModelMPI(
        process_dict, control, maps=maps, discretizations=discretizations
    )
