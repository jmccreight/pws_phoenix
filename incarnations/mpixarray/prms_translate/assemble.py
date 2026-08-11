"""One-call assembly: legacy PRMS files -> a pws_phoenix Model.

Two entry points, layered:

- ``assemble_from_control(control_file, control=None) -> ModelKit`` --
  the ASSEMBLY KIT: everything ``Model()`` takes
  (process_dict/control/maps/discretizations), open for inspection
  and modification BEFORE construction. This is the seam for cases
  the one-liner does not cover: edit the kit, then ``kit.model()``.
- ``model_from_control(control_file, control=None) -> Model`` -- the
  one-liner (builds the kit, constructs the Model; construction runs
  the full assembly validation).

These live HERE and not on Model because the core never imports
prms_translate -- the dependency rule decides the API home. The
control file names the parameter file (``param_file``) and the CBH
files (``*_day``), so the control path is the ONLY required input.

Supported ("many cases"): the NHM mainline configurations -- dprst
on/off, stream temperature off or on in its dynamic-shade +
CBH-humidity form (the nhm.control / nhm_stream_temp.control
shapes). The constant-shade and per-segment-humidity stream-temp
leaves RESOLVE (control.py) but their assembly is not wired yet and
raises here; dynamic parameters and the ag family already raise in
resolution. Everything unsupported fails LOUDLY, never silently.
"""

import pathlib as pl
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from atmosphere.prms_solar_geometry import compute_soltabs
from discretization import Discretization
from hydrology.prms_stream_temp import (
    AGGREGATION_MAP_SPEC,
    PRMSStreamTemp,
    derive_aggregation_weights,
)
from map import Map
from model import Model
from process import DataArrayMeta, Process
from prms_translate.cbh import load_cbh
from prms_translate.control import PrmsRunConfig, from_control
from prms_translate.parameters import (
    SLOT_GRIDS,
    package_parameters,
    volume_map_weights,
)
from prms_translate.readers import load_control, load_parameters


class HumidityCarrier(Process):
    """The humidity CBH forcing on the hru grid: an hru process must
    OWN the external data so the hru->segment humidity Map can source
    it (a Map never originates a quantity)."""

    humidity_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH relative humidity [percent]",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt, time) -> None:
        pass


@dataclass
class ModelKit:
    """Everything ``Model()`` takes, plus the run configuration for
    provenance. Modify any piece, then ``kit.model()``."""

    process_dict: dict[str, dict[str, Any]]
    control: dict[str, Any]
    maps: dict[str, Map]
    discretizations: dict[str, Discretization]
    config: PrmsRunConfig = field(repr=False)
    # PRMS-requested output names this model has no variable for
    # (PRMS-only diagnostics etc.); the kept ones are in
    # control["output_var_names"] when a store was supplied
    dropped_output_var_names: list[str] = field(default_factory=list)

    def model(self) -> Model:
        """Construct the Model (construction IS the validation)."""
        return Model(
            self.process_dict,
            self.control,
            maps=self.maps,
            discretizations=self.discretizations,
        )


def assemble_from_control(
    control_file: str | pl.Path,
    control: dict[str, Any] | None = None,
) -> ModelKit:
    """Legacy PRMS control file -> a ModelKit (see module docstring).
    `control` is the pws_phoenix control dict passed through to the
    Model (output options, restart_read, ...)."""
    control_file = pl.Path(control_file)
    ctl = load_control(control_file)
    cfg = from_control(ctl, control_file)
    param_file = (
        control_file.parent / str(ctl.get("param_file").values)
    ).resolve()
    params = load_parameters(param_file)

    stream_temp = "prms_stream_temp" in cfg.classes
    if stream_temp:
        leaf = cfg.classes["prms_stream_temp"]
        if leaf is not PRMSStreamTemp:
            raise NotImplementedError(
                f"assembly for stream-temp leaf {leaf.__name__} is "
                "not wired (only the dynamic-shade + CBH-humidity "
                "configuration is): build the kit by hand -- see "
                "examples/01_prms_legacy_translation.py for the recipe."
            )
        if "humidity_hru" not in cfg.cbh_paths:
            raise ValueError(
                "stream temperature needs the humidity CBH "
                "(humidity_day) in the control file."
            )

    # -- discretizations + the computed parameters --
    discretizations = {
        "nhru": Discretization(["nhru"]),
        "nsegment": Discretization(
            ["nsegment"],
            parameters=xr.Dataset({"tosegment": params["tosegment"]}),
            topo_order={"segment_order": "tosegment"},
        ),
    }
    soltabs = compute_soltabs(
        params[["hru_slope", "hru_aspect", "hru_lat"]], hru_dim="nhru"
    )

    # -- process_dict in NHM (resolution) order; the humidity carrier
    # slots in just before the first segment-grid process --
    process_dict: dict[str, dict[str, Any]] = {}
    carrier_inserted = False
    for slot, cls in cfg.classes.items():
        grid = SLOT_GRIDS[slot]
        if stream_temp and not carrier_inserted and grid == "nsegment":
            process_dict["humidity_carrier"] = {
                "class": HumidityCarrier,
                "discretization": "nhru",
            }
            carrier_inserted = True
        process_dict[slot] = {"class": cls, "discretization": grid}

    # -- Maps: the three lateral volumes always; the ten stream-temp
    # aggregations (basis-probed weights) when stream temp is on --
    def _map(source: str, target: str, ww: np.ndarray, derivation: str) -> Map:
        return Map(
            weights=ww,
            grid={"nhru": "nsegment"},
            variable={source: target},
            derivation=derivation,
        )

    vol_derivation = (
        "prms_translate.volume_map_weights(params): 0/1 assignment "
        "matrix from hru_segment"
    )
    vol_weights = volume_map_weights(params)
    maps = {
        "sroff_vol": _map(
            "sroff_vol", "seg_sroff_vol", vol_weights, vol_derivation
        ),
        "ssres_vol": _map(
            "ssres_flow_vol",
            "seg_ssres_flow_vol",
            vol_weights,
            vol_derivation,
        ),
        "gw_vol": _map(
            "gwres_flow_vol",
            "seg_gwres_flow_vol",
            vol_weights,
            vol_derivation,
        ),
    }
    if stream_temp:
        seg_dis = discretizations["nsegment"].parameters
        assert seg_dis is not None
        agg_weights = derive_aggregation_weights(
            params["hru_segment"].values,
            params["hru_area"].values,
            params["tosegment"].values,
            seg_dis["segment_order"].values.astype(np.int64),
            params["seg_close"].values,
        )
        agg_derivation = (
            "derive_aggregation_weights(hru_segment, hru_area, "
            "tosegment, segment_order, seg_close)"
        )
        maps |= {
            target: _map(source, target, agg_weights[wkey], agg_derivation)
            for target, (source, wkey) in AGGREGATION_MAP_SPEC.items()
        }

    # -- the contract drives the supply --
    spec = Model.input_spec(process_dict, maps=maps)
    packaged = package_parameters(
        params,
        cfg.classes,
        extra={str(nn): soltabs[nn] for nn in soltabs.data_vars},
    )
    for slot, ds in packaged.items():
        process_dict[slot]["parameters"] = ds
    for grid, gg in spec["required"].items():
        for init_name, info in gg["initial_values"].items():
            process_dict[info["process"]][init_name] = params[init_name]
        for name, info in gg["external_inputs"].items():
            if name not in cfg.cbh_paths:
                raise KeyError(
                    f"external input {name!r} (consumer "
                    f"{info['consumers'][0]!r}) has no CBH file in "
                    f"the control (*_day entries found: "
                    f"{sorted(cfg.cbh_paths)})."
                )
            process_dict[info["consumers"][0]][name] = load_cbh(
                cfg.cbh_paths[name], cfg.start_time, cfg.end_time
            )

    # -- PRMS output requests: NAMES translate, filtered to variables
    # this model actually has; where to write is ALWAYS the caller's
    # decision (the control file's output paths are never translated
    # -- see PrmsRunConfig.output_var_names). Injected only when the
    # caller supplied a store and no explicit name list of their own.
    kit_control = dict(control or {})
    dropped: list[str] = []
    if (
        cfg.output_var_names
        and "output_serial_zarr" in kit_control
        and "output_var_names" not in kit_control
    ):
        available: set[str] = set()
        for entry in process_dict.values():
            available |= set(entry["class"].get_var_names())
        kit_control["output_var_names"] = [
            nn for nn in cfg.output_var_names if nn in available
        ]
        dropped = [nn for nn in cfg.output_var_names if nn not in available]

    return ModelKit(
        process_dict=process_dict,
        control=kit_control,
        maps=maps,
        discretizations=discretizations,
        config=cfg,
        dropped_output_var_names=dropped,
    )


def model_from_control(
    control_file: str | pl.Path,
    control: dict[str, Any] | None = None,
) -> Model:
    """The one-liner: legacy PRMS control file -> a constructed (and
    thereby validated) pws_phoenix Model. See assemble_from_control
    for the kit underneath."""
    return assemble_from_control(control_file, control).model()
