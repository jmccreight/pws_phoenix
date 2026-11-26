"""Legacy PRMS files -> a PARALLEL (ModelMPI) run.

The serial ModelKit lives entirely in memory; a parallel run cannot:
mpixarray STREAMS the distributed grid from ONE combined NetCDF on
disk, while the serial (segment) grid is rebuilt identically on every
rank. So the parallel path has two halves:

- ``write_mpi_input_file(kit, path, n_days=None, transpose=False)``
  -- the PREP half,
  run serially (a notebook cell, a script): merge the kit's
  distributed-grid content into the combined input file -- static
  parameters as ``(nhru, ...)``, forcings as ``(time, nhru)``, real
  dim-coordinates. Initial state is NOT in the file: it is supplied
  in memory, in process_dict, as in serial. ``n_days`` truncates the
  time axis (the stream defines the run length). ``transpose=True``
  writes the forcings as ``(nhru, time)``: ModelMPI reads either
  order, and that one is the faster read at a deep
  ``control["mpi_window_steps"]``.
- ``mpi_model_from_control(control_file, control)`` -- the parallel
  one-liner, callable ONLY under mpirun (SPMD: every rank calls it):
  rebuilds the kit from the legacy files (deterministic, so the
  replicated serial grid is identical across ranks), strips the
  distributed grid's in-memory data (the input file serves it) except
  its supplied initial state,
  converts every Map to MapMPI, and constructs ModelMPI.

- ``mpi_model_from_translation(translation_yaml, control,
  transposed=False)`` -- the parallel one-liner for a domain
  translated ahead of time (translate_domain), under mpirun like the
  one above. There is NO prep half: the input file is the
  translation's forcing file (``cbh.nc``, or ``cbh_transpose.nc``
  with ``transposed=True``), and the distributed grid's parameters
  stay in process_dict, where ModelMPI takes each rank's block.

Required control key beyond the serial dict, for
mpi_model_from_control: ``input_file`` (from the prep half);
mpi_model_from_translation sets it and raises when it is given.
``mpi_grid`` defaults to ``"nhru"``. Output: with no
``output_var_names`` of your own, the control file's ``*OutVar_names``
are injected PER DESTINATION -- distributed-grid names stream to
``output_parallel_netcdf`` (any number; the one-variable mpixarray
limit fell Oct 28, 2026) and serial-grid names go to the rank-0 zarr,
each only if its store (``output_parallel_netcdf``,
``output_serial_zarr``) was supplied (else they are left out). With
neither store the run writes nothing. An explicit list wins verbatim.
Dropped names are not reported here (no ModelKit is returned; see
assemble_from_control). Both one-liners treat output this way.

``replace_classes={slot: class}`` (both one-liners) puts another class
in a ``process_dict`` slot after the kit is built: the caller's way
to choose a class the control file cannot name, such as
``{"prms_channel": PRMSChannelMPI}`` (the channel divided over the
ranks, ``hydrology/prms_channel_mpi.py``). The entry's data are
unchanged, so the class must take the same parameters and state.
"""

import pathlib as pl
from typing import Any

import numpy as np
import xarray as xr
import yaml

from pws_phoenix.maps import MapMPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.prms_translate.assemble import (
    ModelKit,
    assemble_from_control,
    assemble_from_translation,
    requested_output_names,
)


def write_mpi_input_file(
    kit: ModelKit,
    path: str | pl.Path,
    n_days: int | None = None,
    mpi_grid: str = "nhru",
    transpose: bool = False,
) -> pl.Path:
    """Write the combined distributed-grid input file from a (serial)
    ModelKit: every mpi-grid process entry's parameter dataset plus
    its loose DataArray entries (forcings), merged into one dataset
    with real dim-coordinates. Supplied initial state (restart
    variables under their own names) is skipped: ModelMPI takes it
    from process_dict. Returns the path.

    ``transpose=True`` writes every ``(time, mpi_grid)`` variable as
    ``(mpi_grid, time)``. ModelMPI takes either order and gives the
    same answers; which one reads faster depends on
    ``control["mpi_window_steps"]`` (machine_probes/README.md "The
    order of the dims decides the read"). The transposed copy of each
    variable is made by the write, one variable at a time."""
    merged = xr.Dataset()
    for entry in kit.process_dict.values():
        if entry["discretization"] != mpi_grid:
            continue
        if "parameters" in entry:
            # same-named parameters across processes carry identical
            # values (packaged from one flat source): first wins
            merged = merged.merge(entry["parameters"], compat="override")
        state_names = set(entry["class"].get_restart_variables())
        for key, val in entry.items():
            if key in ("class", "discretization", "parameters"):
                continue
            if key in state_names:
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
    if transpose:
        for name in list(merged.data_vars):
            if merged[name].dims == ("time", mpi_grid):
                # a view; to_netcdf writes it in the new order
                merged[name] = merged[name].transpose(mpi_grid, "time")
    path = pl.Path(path)
    merged.to_netcdf(path)
    return path


def _mpi_model_from_kit(
    kit: ModelKit,
    control: dict[str, Any],
    keep_parameters: bool,
    replace_classes: dict[str, type] | None = None,
) -> ModelMPI:
    """A (serial) ModelKit -> ModelMPI, on every rank. `control` is
    complete (``input_file``, ``mpi_grid``) and is edited here: the
    control file's output names are added to it.

    A distributed-grid entry keeps its class, its grid and its
    supplied initial state (full extent; each rank takes its block).
    `keep_parameters`: it also keeps its ``"parameters"`` dataset
    (ModelMPI copies each rank's block); otherwise they are dropped
    with everything else it holds, because ``input_file`` has them.
    `replace_classes`: see the module docstring."""
    mpi_grid = control["mpi_grid"]
    replace_classes = replace_classes or {}
    unknown = sorted(set(replace_classes) - set(kit.process_dict))
    if unknown:
        # the same arguments on every rank: all ranks raise
        raise ValueError(
            f"replace_classes names {unknown}, which are not process "
            f"slots of this model ({list(kit.process_dict)})."
        )
    process_dict: dict[str, dict[str, Any]] = {}
    mpi_vars: set[str] = set()
    for slot, entry in kit.process_dict.items():
        if entry["discretization"] == mpi_grid:
            kept_names = set(entry["class"].get_restart_variables())
            if keep_parameters:
                kept_names.add("parameters")
            process_dict[slot] = {
                "class": entry["class"],
                "discretization": mpi_grid,
            } | {name: entry[name] for name in entry if name in kept_names}
            mpi_vars |= set(entry["class"].get_var_names(include_budget=True))
        else:
            process_dict[slot] = entry
    for slot, cls in replace_classes.items():
        # a new entry: the kit's is not edited
        process_dict[slot] = process_dict[slot] | {"class": cls}

    # The control file's output request, per destination (the mirror of
    # the serial gate "injected only when a store was supplied"): a
    # name is kept when the caller gave the store of its grid. With
    # neither store the list is empty and the run writes nothing.
    if "output_var_names" not in control and kit.config.output_var_names:
        kept, _dropped = requested_output_names(kit.config, kit.process_dict)
        control["output_var_names"] = [
            nn
            for nn in kept
            if (
                "output_parallel_netcdf" in control
                if nn in mpi_vars
                else "output_serial_zarr" in control
            )
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


def mpi_model_from_control(
    control_file: str | pl.Path,
    control: dict[str, Any],
    replace_classes: dict[str, type] | None = None,
) -> ModelMPI:
    """The parallel one-liner (call it on EVERY rank, under mpirun).
    See the module docstring for the required control keys; the
    distributed grid's data come from ``control["input_file"]``
    (written by write_mpi_input_file), everything else is rebuilt
    identically per rank from the legacy files."""
    control = dict(control)
    control.setdefault("mpi_grid", "nhru")
    if "input_file" not in control:
        raise ValueError(
            "mpi_model_from_control: control needs 'input_file' (see "
            "prms_translate.assemble_mpi; write_mpi_input_file "
            "produces it)."
        )
    kit = assemble_from_control(control_file)
    return _mpi_model_from_kit(
        kit, control, keep_parameters=False, replace_classes=replace_classes
    )


def mpi_model_from_translation(
    translation_yaml: str | pl.Path,
    control: dict[str, Any] | None = None,
    transposed: bool = False,
    replace_classes: dict[str, type] | None = None,
) -> ModelMPI:
    """The parallel one-liner for a domain translated ahead of time
    (call it on EVERY rank, under mpirun). `translation_yaml` is the
    ``<stem>_translation.yaml`` that ``translate_domain`` wrote; see
    assemble_from_translation for what is taken from it.

    - The streamed input file is the yaml's ``forcing_file``, on
      (time, nhru), or with ``transposed=True`` its
      ``forcing_file_transpose``, on (nhru, time)
      (``translate_domain(cbh_transpose=True)`` writes it). The
      answers are the same; which reads faster depends on
      ``control["mpi_window_steps"]``, and the build warns on rank 0
      when the file chosen is the slower one by 100 times or more
      (data_io.slow_order_note). ``control["input_file"]`` raises:
      the yaml names the file.
    - The hru parameters stay in process_dict at full extent on every
      rank during the build; ModelMPI keeps a copy of each rank's
      block.
    - The run covers the control file's ``start_time`` to
      ``end_time``; ones given in `control` take precedence.
    - Output is as in mpi_model_from_control (module docstring): with
      no store in `control` the run writes nothing.
    - ``replace_classes``: see the module docstring.

    Every raise here comes from the same files on every rank, so all
    ranks raise."""
    control = dict(control or {})
    if "input_file" in control:
        raise ValueError(
            "mpi_model_from_translation: control gives 'input_file' "
            f"({control['input_file']}), but the translation yaml "
            "names the forcing file. Drop it; transposed=True chooses "
            "the (nhru, time) copy."
        )
    yaml_path = pl.Path(translation_yaml)
    with open(yaml_path) as ff:
        info = yaml.safe_load(ff)
    kit = assemble_from_translation(yaml_path)
    key = "forcing_file_transpose" if transposed else "forcing_file"
    if info.get(key) is None:
        # only the transposed copy can be missing here:
        # assemble_from_translation raised on no forcing_file
        raise FileNotFoundError(
            f"{yaml_path} names no {key}: run translate_domain with "
            "cbh_transpose=True (--cbh-transpose)."
        )
    # the kit's control holds the control file's start_time and
    # end_time
    control = kit.control | control
    control["input_file"] = yaml_path.parent / info[key]
    mpi_grid = control.setdefault("mpi_grid", "nhru")

    # What is left in a distributed-grid entry must be streamed from
    # the forcing file (its path), kept (parameters, initial state) or
    # structure. An input held in memory has no file to stream from.
    for slot, entry in kit.process_dict.items():
        if entry["discretization"] != mpi_grid:
            continue
        state_names = set(entry["class"].get_restart_variables())
        in_memory = [
            name
            for name, value in entry.items()
            if isinstance(value, xr.DataArray) and name not in state_names
        ]
        if in_memory:
            raise NotImplementedError(
                f"process '{slot}': input(s) {in_memory} are held in "
                "memory (a static parameter served as a time-constant "
                "input), and ModelMPI streams the inputs of grid "
                f"'{mpi_grid}' from the forcing file only."
            )
    return _mpi_model_from_kit(
        kit, control, keep_parameters=True, replace_classes=replace_classes
    )
