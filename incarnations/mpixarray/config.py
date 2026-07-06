"""
config.py
=========
Minimal yaml model configuration: ``load_model_yaml(path)`` returns the
``(process_dict, control, maps)`` triple that feeds ``Model`` or
``ModelMPI`` equally.

A deliberate PROBE for the future ``Options`` design (see "Global state"
in pws_phoenix/CLAUDE.md) -- the schema is minimal and may change. yaml
cannot hold classes or arrays, so:

  - a process entry's ``class:`` is a string resolved via
    ``Process._registry`` -- import the modules defining your concrete
    process classes BEFORE calling ``load_model_yaml`` (importing
    registers them);
  - a map entry's ``class:`` is ``"Map"`` or ``"MapMPI"``; its
    ``weights:`` is either an inline (nested-list) matrix or a path to a
    single-variable NetCDF (opened, loaded, closed);
  - every path string is resolved RELATIVE TO THE YAML FILE's directory
    (absolute paths pass through);
  - the ``processes:`` mapping ORDER IS THE EXECUTION SCHEDULE (yaml
    mappings load in document order). If you WRITE a config
    programmatically, pass ``sort_keys=False`` to ``yaml.safe_dump`` --
    its default alphabetization silently reorders the schedule (the
    one-pass order validation will catch it at build, but save yourself
    the trip).

Schema (maps optional):

    control:
      input_file: hru_input.nc
      output_parallel_netcdf: hru_output.nc
      output_serial_zarr: segment_output.zarr
      output_var_names: [flow, storage]
      time_chunk_size: 10
      mpi_grid: hru
    processes:
      upper:
        class: Upper
        discretization: hru
      lower:
        class: Lower
        discretization: segment
        parameters: low_params.nc
        forcing_low: forcing_low.nc
        storage_initial: storage_initial.nc
    maps:
      hru_to_seg:
        class: MapMPI
        weights: weights.nc
        grid: {hru: segment}
        variable: {flow: flow}
"""

import pathlib as pl
from typing import Any

import numpy as np
import xarray as xr
import yaml

from map import Map, MapMPI
from process import Process

_MAP_CLASSES: dict[str, type] = {"Map": Map, "MapMPI": MapMPI}
_CONTROL_PATH_KEYS = (
    "input_file",
    "output_parallel_netcdf",
    "output_serial_zarr",
)


def _resolve_path(value: str, base_dir: pl.Path) -> pl.Path:
    """A path string from the yaml, resolved relative to the yaml's dir."""
    path = pl.Path(value)
    return path if path.is_absolute() else base_dir / path


def _load_weights(spec: Any, base_dir: pl.Path) -> np.ndarray:
    """Map weights: an inline nested list, or a path to a single-variable
    NetCDF."""
    if isinstance(spec, str):
        with xr.open_dataarray(_resolve_path(spec, base_dir)) as da:
            return da.load().values
    return np.asarray(spec)


def load_model_yaml(
    yaml_file: pl.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a fully-serialized model configuration.

    Returns ``(process_dict, control, maps)`` -- model-agnostic (feeds
    ``Model`` or ``ModelMPI``). See the module docstring for the schema
    and resolution rules.
    """
    yaml_file = pl.Path(yaml_file)
    base_dir = yaml_file.parent
    with open(yaml_file) as ff:
        cfg = yaml.safe_load(ff)

    process_dict: dict[str, Any] = {}
    for proc_name, entry in cfg["processes"].items():
        resolved: dict[str, Any] = {}
        for key, val in entry.items():
            if key == "class":
                if val not in Process._registry:
                    raise ValueError(
                        f"process '{proc_name}': class {val!r} is not in "
                        "Process._registry (import the module defining it "
                        "before load_model_yaml); registered: "
                        f"{sorted(Process._registry)}"
                    )
                resolved[key] = Process._registry[val]
            elif key == "discretization":
                resolved[key] = val
            else:  # data entries arrive as path strings
                resolved[key] = _resolve_path(val, base_dir)
        process_dict[proc_name] = resolved

    control: dict[str, Any] = dict(cfg["control"])
    for key in _CONTROL_PATH_KEYS:
        if key in control:
            control[key] = _resolve_path(control[key], base_dir)

    maps: dict[str, Any] = {}
    for map_name, entry in cfg.get("maps", {}).items():
        if entry["class"] not in _MAP_CLASSES:
            raise ValueError(
                f"map '{map_name}': class {entry['class']!r} must be one "
                f"of {sorted(_MAP_CLASSES)}."
            )
        maps[map_name] = _MAP_CLASSES[entry["class"]](
            weights=_load_weights(entry["weights"], base_dir),
            grid=dict(entry["grid"]),
            variable=dict(entry["variable"]),
        )

    return process_dict, control, maps
