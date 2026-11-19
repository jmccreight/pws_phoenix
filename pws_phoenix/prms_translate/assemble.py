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

A domain translated ahead of time (``translate_domain``: the CBH text
and the parameter file written once as netCDF) has the same two
layers, ``assemble_from_translation`` and ``model_from_translation``.
They take the path of the domain's ``<stem>_translation.yaml``.

These live HERE and not on Model because the core never imports
prms_translate -- the dependency rule decides the API home. The
control file names the parameter file (``param_file``) and the CBH
files (``*_day``), so the control path is the ONLY required input.

Supported ("many cases"): the NHM mainline configurations -- dprst
on/off, stream temperature off or on in its dynamic-shade +
CBH-humidity form or its per-segment-humidity form (the
nhm.control / nhm_stream_temp.control / nhm_stream_temp_seg_humid_*
shapes), transp_frost (static or dynamic frost dates, incl. the
multi-param_file controls that carry the frost parameters), and the
agricultural family (soilzone_ag: spinup and iter_aet/ObsET analysis
shapes, static or dynamic ag_frac). The constant-shade stream-temp
leaf RESOLVES (control.py) but its assembly is not wired yet and
raises here. Everything
unsupported fails LOUDLY, never silently.

Dynamic parameters follow PRMS semantics: a translated dyn_*_flag
serves the named input from its dynamic-parameter file
(forward-filled onto model time); without the flag, the static
parameter of the same name is served time-constant (tiled). PRMS's
PET_cbh_file is deliberately not consumed (see control.py:
CBH_ENTRY_TO_INPUT).
"""

import hashlib
import pathlib as pl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr
import yaml

from pws_phoenix.atmosphere.prms_solar_geometry import compute_soltabs
from pws_phoenix.discretization import Discretization
from pws_phoenix.flow_graph_build import NewNode, splice_flow_graph
from pws_phoenix.hydrology.prms_stream_temp import (
    AGGREGATION_MAP_SPEC,
    PRMSStreamTemp,
    PRMSStreamTempSegHumidity,
    derive_aggregation_weights,
)
from pws_phoenix.maps import Map
from pws_phoenix.model import Model
from pws_phoenix.process import DataArrayMeta, Process
from pws_phoenix.prms_translate.cbh import load_cbh
from pws_phoenix.prms_translate.control import PrmsRunConfig, from_control
from pws_phoenix.prms_translate.dyn_param import (
    forward_fill,
    load_dynamic_parameter,
)
from pws_phoenix.prms_translate.parameters import (
    SLOT_GRIDS,
    package_parameters,
    volume_map_weights,
)
from pws_phoenix.prms_translate.readers import load_control, load_parameters


class HumidityCarrier(Process):
    """The humidity CBH forcing on the hru grid: an hru process must
    OWN the external data so the hru->segment humidity Map can source
    it (a Map never originates a quantity)."""

    humidity_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        units="percent",
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


def requested_output_names(
    cfg: PrmsRunConfig, process_dict: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Split the control file's output REQUEST into the names this model
    has a variable for (kept, request order) and the rest (dropped:
    PRMS-only diagnostics etc.)."""
    available: set[str] = set()
    for entry in process_dict.values():
        available |= set(entry["class"].get_var_names(include_budget=True))
    kept = [nn for nn in cfg.output_var_names if nn in available]
    dropped = [nn for nn in cfg.output_var_names if nn not in available]
    return kept, dropped


def seg_inflow_from_segment_flow_init(
    segment_flow_init: xr.DataArray, tosegment: xr.DataArray
) -> xr.DataArray:
    """PRMS's segment_flow_init -> the starting seg_inflow: each
    segment's initial flow becomes its downstream segment's inflow.
    pywatershed's channel initialization VERBATIM, including the
    ASSIGNMENT at confluences (the last upstream segment in index
    order wins, not the sum). `tosegment` is 1-based, 0 = outlet."""
    tosegment0 = tosegment.values - 1
    seg_inflow = np.zeros(segment_flow_init.shape, dtype=np.float64)
    for iseg in range(tosegment0.shape[0]):
        jseg = tosegment0[iseg]
        if jseg < 0:
            continue
        seg_inflow[jseg] = segment_flow_init.values[iseg]
    # a flow in the same units as segment_flow_init
    attrs = {
        kk: vv for kk, vv in segment_flow_init.attrs.items() if kk == "units"
    }
    return xr.DataArray(seg_inflow, dims=segment_flow_init.dims, attrs=attrs)


def prms_initial_state(params: xr.Dataset) -> dict[str, xr.DataArray]:
    """Translate PRMS's initial-condition PARAMETERS into pws_phoenix
    supplied initial state: restart variables under their OWN names.
    PRMS consumes state from parameters; pws_phoenix does not -- this
    is the one place that convention lives. Only names present in
    `params` are translated (an absent one leaves the state to the
    process's init_from_params() default).

    - gwres_stor <- gwstor_init (a rename).
    - seg_inflow <- segment_flow_init, propagated one segment
      downstream (seg_inflow_from_segment_flow_init)."""
    state: dict[str, xr.DataArray] = {}
    if "gwstor_init" in params:
        state["gwres_stor"] = params["gwstor_init"]
    if "segment_flow_init" in params:
        state["seg_inflow"] = seg_inflow_from_segment_flow_init(
            params["segment_flow_init"], params["tosegment"]
        )
    return state


def assemble_from_control(
    control_file: str | pl.Path,
    control: dict[str, Any] | None = None,
    preprocessed: str | pl.Path | None = None,
    cbh_paths: Mapping[str, str | pl.Path] | None = None,
) -> ModelKit:
    """Legacy PRMS control file -> a ModelKit (see module docstring).
    `control` is the pws_phoenix control dict passed through to the
    Model (output options, restart_read, ...). `preprocessed` is an
    OPTIONAL artifacts file from
    ``prms_translate.write_preprocessed`` -- its stamped derivables
    (solar tables, segment_order, hru_in_to_cf) and map weights are
    VERIFIED against this run's parameters and used instead of
    deriving in-chain (see prms_translate/preprocess.py for the
    provenance/staleness design). `cbh_paths` OVERRIDES the control
    file's forcing-file entries by pws_phoenix input name (``prcp``,
    ``tmax``, ``tmin``, ``humidity_hru``, ...) -- for control files
    whose ``*_day`` entries do not name the files actually present
    (PRMS's template control names ``precip.day``; a domain may ship
    ``prcp.cbh``)."""
    control_file = pl.Path(control_file)
    ctl = load_control(control_file)
    cfg = from_control(ctl, control_file)
    if cbh_paths:
        cfg.cbh_paths = dict(cfg.cbh_paths) | {
            name: pl.Path(path) for name, path in cbh_paths.items()
        }
    params = load_parameters(cfg.param_files)
    return _assemble(
        cfg,
        params,
        lambda name: load_cbh(
            cfg.cbh_paths[name], cfg.start_time, cfg.end_time
        ),
        control,
        preprocessed,
    )


def _assemble(
    cfg: PrmsRunConfig,
    params: xr.Dataset,
    forcing: Callable[[str], xr.DataArray | pl.Path],
    control: dict[str, Any] | None,
    preprocessed: str | pl.Path | None,
) -> ModelKit:
    """A run configuration and its parameters -> a ModelKit: the part
    of assembly that does not depend on where the parameters and the
    forcings were read from. `forcing(name)` returns what the Model is
    given for the CBH input `name` (a key of ``cfg.cbh_paths``): a
    (time, nhru) DataArray, or the path of a file that holds it.
    `control` and `preprocessed` are as in assemble_from_control."""
    artifacts: xr.Dataset | None = None
    if preprocessed is not None:
        from pws_phoenix.prms_translate.preprocess import verify_preprocessed

        artifacts = xr.load_dataset(preprocessed)
        verify_preprocessed(artifacts, params)

    stream_temp = "prms_stream_temp" in cfg.classes
    # the humidity CBH reaches the segments through the hru->segment
    # humidity Map (PRMSStreamTemp); PRMSStreamTempSegHumidity computes
    # seg_humid from its seg_humidity parameter instead
    cbh_humidity = False
    if stream_temp:
        leaf = cfg.classes["prms_stream_temp"]
        if leaf not in (PRMSStreamTemp, PRMSStreamTempSegHumidity):
            raise NotImplementedError(
                f"assembly for stream-temp leaf {leaf.__name__} is "
                "not wired (only the dynamic-shade configurations "
                "are): build the kit by hand -- see "
                "examples/01_prms_legacy_translation.py for the recipe."
            )
        cbh_humidity = "seg_humid" in leaf.get_inputs()
        if cbh_humidity and "humidity_hru" not in cfg.cbh_paths:
            raise ValueError(
                "stream temperature needs the humidity CBH "
                "(humidity_day) in the control file."
            )

    # -- discretizations + the derivable parameters (from the
    # verified artifacts when preprocessed, else derived in-chain) --
    # the segment dis carries the network ids (nhm_seg) beside the
    # topology: no process declares them, but insert_flow_graph needs
    # them to splice by nhm_seg and to label the graph's nodes
    seg_dis_vars = {"tosegment": params["tosegment"]}
    if "nhm_seg" in params:
        seg_dis_vars["nhm_seg"] = params["nhm_seg"]
    if artifacts is not None:
        seg_dis_vars["segment_order"] = artifacts["segment_order"]
        discretizations = {
            "nhru": Discretization(["nhru"]),
            "nsegment": Discretization(
                ["nsegment"], parameters=xr.Dataset(seg_dis_vars)
            ),
        }
        soltabs = artifacts[["soltab_potsw", "soltab_horad_potsw"]]
    else:
        discretizations = {
            "nhru": Discretization(["nhru"]),
            "nsegment": Discretization(
                ["nsegment"],
                parameters=xr.Dataset(seg_dis_vars),
                topo_order={"segment_order": "tosegment"},
            ),
        }
        soltabs = compute_soltabs(
            params[["hru_slope", "hru_aspect", "hru_lat"]],
            hru_dim="nhru",
        )

    # -- process_dict in NHM (resolution) order; the humidity carrier
    # (CBH humidity only) slots in just before the first segment-grid
    # process --
    process_dict: dict[str, dict[str, Any]] = {}
    carrier_inserted = False
    for slot, cls in cfg.classes.items():
        grid = SLOT_GRIDS[slot]
        if cbh_humidity and not carrier_inserted and grid == "nsegment":
            process_dict["humidity_carrier"] = {
                "class": HumidityCarrier,
                "discretization": "nhru",
            }
            carrier_inserted = True
        process_dict[slot] = {"class": cls, "discretization": grid}

    # -- Maps: the three lateral volumes always; the ten stream-temp
    # aggregations (basis-probed weights) when stream temp is on, nine
    # without CBH humidity (no seg_humid Map) --
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
    vol_weights = (
        artifacts["weights_vol"].values
        if artifacts is not None
        else volume_map_weights(params)
    )
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
        if artifacts is not None:
            agg_weights = {
                kk: artifacts[f"weights_{kk}"].values
                for kk in ("flow", "swrad", "met", "humid")
            }
        else:
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
            if cbh_humidity or target != "seg_humid"
        }

    # -- the contract drives the supply --
    spec = Model.input_spec(process_dict, maps=maps)
    extra = {str(nn): soltabs[nn] for nn in soltabs.data_vars}
    if artifacts is not None:
        extra["hru_in_to_cf"] = artifacts["hru_in_to_cf"]
    packaged = package_parameters(params, cfg.classes, extra=extra)
    for slot, ds in packaged.items():
        process_dict[slot]["parameters"] = ds

    # dynamic parameters -> time-varying inputs on the model window
    # (forward-fill = PRMS semantics; load-time prep allocations)
    times = np.arange(
        cfg.start_time,
        cfg.end_time + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    dyn_inputs = {
        name: forward_fill(load_dynamic_parameter(path), times)
        for name, path in cfg.dyn_param_paths.items()
    }
    for name, da in dyn_inputs.items():
        # a dynamic parameter has the static parameter's units
        if name in params and "units" in params[name].attrs:
            da.attrs["units"] = params[name].attrs["units"]

    # PRMS initial-condition parameters -> supplied state, given to the
    # process that owns each restart variable
    initial_state = prms_initial_state(params)
    for entry in process_dict.values():
        for name in entry["class"].get_restart_variables():
            if name in initial_state:
                entry[name] = initial_state[name]

    for grid, gg in spec["required"].items():
        for name, info in gg["external_inputs"].items():
            consumer = info["consumers"][0]
            if name in dyn_inputs:
                process_dict[consumer][name] = dyn_inputs[name]
            elif name in cfg.cbh_paths:
                process_dict[consumer][name] = forcing(name)
            elif name in params and params[name].dims == ("nhru",):
                # a time-varying INPUT declaration backed by a STATIC
                # parameter (PRMS dynamic-parameter default: no dyn
                # flag -> the parameter holds; e.g. ag_frac in the
                # spinup shape): serve it time-constant
                process_dict[consumer][name] = xr.DataArray(
                    np.tile(params[name].values, (times.size, 1)),
                    dims=("time", "nhru"),
                    coords={"time": times},
                    attrs=params[name].attrs,
                )
            else:
                raise KeyError(
                    f"external input {name!r} (consumer {consumer!r}) "
                    "has no CBH file, dynamic-parameter file, or "
                    "same-named static parameter in the control "
                    f"(*_day/CBH entries found: {sorted(cfg.cbh_paths)}"
                    f"; dynamic: {sorted(dyn_inputs)})."
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
        kit_control["output_var_names"], dropped = requested_output_names(
            cfg, process_dict
        )

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
    preprocessed: str | pl.Path | None = None,
    cbh_paths: Mapping[str, str | pl.Path] | None = None,
) -> Model:
    """The one-liner: legacy PRMS control file -> a constructed (and
    thereby validated) pws_phoenix Model. See assemble_from_control
    for the kit underneath (and for `preprocessed` / `cbh_paths`)."""
    return assemble_from_control(
        control_file, control, preprocessed=preprocessed, cbh_paths=cbh_paths
    ).model()


def assemble_from_translation(
    translation_yaml: str | pl.Path,
    control: dict[str, Any] | None = None,
) -> ModelKit:
    """A domain translated ahead of time -> a ModelKit.
    `translation_yaml` is the ``<stem>_translation.yaml`` that
    ``translate_domain`` wrote; no CBH or parameter text is read.

    - Options (the classes, the simulation window, the requested
      output names) come from the control copy beside the yaml.
    - Parameters come from the yaml's ``parameter_file``, and the
      derived parameters and map weights from its
      ``preprocessed_file`` when it names one (else they are derived
      here, as in assemble_from_control).
    - Each forcing entry of the kit's process_dict is the PATH of the
      yaml's ``forcing_file``: the Model opens it once and reads one
      time row per step. That file holds the whole record, so the
      kit's control carries the control file's ``start_time`` and
      ``end_time``; ones given in `control` take precedence.

    The kit is otherwise that of assemble_from_control on the original
    control file. `control` is as there.

    Raises when the control copy is not the control file that was
    translated (the yaml's ``control_sha256``), and when the yaml
    names no forcing file or no parameter file (a step that was not
    run)."""
    yaml_path = pl.Path(translation_yaml)
    with open(yaml_path) as ff:
        info = yaml.safe_load(ff)
    domain_dir = yaml_path.parent

    control_copy = domain_dir / info["control_copy"]
    sha256 = hashlib.sha256(control_copy.read_bytes()).hexdigest()
    if sha256 != info["control_sha256"]:
        raise ValueError(
            f"{control_copy} is not the control file that {yaml_path} "
            f"was written from (sha256 {sha256}, the yaml records "
            f"{info['control_sha256']}): translate the domain again."
        )
    not_run = [
        key for key in ("forcing_file", "parameter_file") if info[key] is None
    ]
    if not_run:
        raise FileNotFoundError(
            f"{yaml_path} names no {' and no '.join(not_run)}: run "
            "translate_domain with those steps."
        )

    cfg = from_control(load_control(control_copy), control_copy)
    if cfg.dyn_param_paths:
        raise NotImplementedError(
            f"{control_copy} names dynamic-parameter files "
            f"({sorted(cfg.dyn_param_paths)}): a translated domain "
            "does not hold them yet."
        )
    # from_control resolved the copy's file entries against the copy's
    # directory, but they are relative to the ORIGINAL's: the files
    # that were read are the ones the yaml records
    cfg.param_files = [pl.Path(pp) for pp in info["parameter_sources"]]
    cfg.cbh_paths = {
        name: pl.Path(pp) for name, pp in info["forcing_sources"].items()
    }

    params = xr.load_dataset(domain_dir / info["parameter_file"])
    forcing_file = domain_dir / info["forcing_file"]
    preprocessed = info.get("preprocessed_file")
    kit_control = {
        "start_time": cfg.start_time,
        "end_time": cfg.end_time,
    } | (control or {})
    return _assemble(
        cfg,
        params,
        lambda name: forcing_file,
        kit_control,
        None if preprocessed is None else domain_dir / preprocessed,
    )


def model_from_translation(
    translation_yaml: str | pl.Path,
    control: dict[str, Any] | None = None,
) -> Model:
    """The one-liner: a domain translated ahead of time -> a
    constructed (and thereby validated) pws_phoenix Model. See
    assemble_from_translation for the kit underneath."""
    return assemble_from_translation(translation_yaml, control).model()


def insert_flow_graph(
    kit: ModelKit,
    new_nodes: list[NewNode],
    *,
    n_substeps: int = 24,
    io_in_cfs: bool = True,
    class_name: str = "FlowGraph",
) -> ModelKit:
    """Replace the kit's PRMSChannel with a FlowGraph that has
    ``new_nodes`` spliced into the channel network (the pws_phoenix
    form of pywatershed's ``prms_channel_flow_graph_to_model_dict``).

    The ``prms_channel`` slot becomes ``flow_graph`` (same position in
    the schedule) on a new ``nnodes`` grid; the three lateral-volume
    Maps are padded with zero rows for the new nodes and re-targeted to
    the graph's ``node_*_vol`` inputs; the ``nsegment`` grid goes away.
    Initial state supplied in the channel entry (``seg_inflow``) is
    carried onto the graph's channel rows, zero on the new nodes; the
    new nodes' own initial state (``NewNode.initial_state``) is added.
    ``control["output_var_names"]`` is re-filtered to the graph's
    variables (``seg_inflow``/``seg_outflow`` survive -- the channel
    node declares them as its rows' state, now on ``nnodes``; anything
    the graph lacks is recorded in ``dropped_output_var_names``). The
    graph-level names (``node_outflows``, ``lake_release``, ...) are
    never auto-added -- request them yourself.

    Not wired: a kit with other segment-grid processes (stream
    temperature, hydraulic geometry) -- they consume ``seg_outflow`` on
    ``nsegment``; feeding them from the graph is a later step.
    """
    if "prms_channel" not in kit.process_dict:
        raise ValueError("the kit has no 'prms_channel' slot to replace")
    others = [
        slot
        for slot, entry in kit.process_dict.items()
        if entry["discretization"] == "nsegment" and slot != "prms_channel"
    ]
    if others:
        raise NotImplementedError(
            f"segment-grid processes {others} beside the channel are not "
            "wired to a FlowGraph yet (they consume seg_outflow on "
            "nsegment)."
        )
    seg_dis = kit.discretizations["nsegment"].parameters
    assert seg_dis is not None
    if "nhm_seg" not in seg_dis:
        raise ValueError(
            "the segment dis carries no 'nhm_seg' (the parameter file "
            "lacks it); splicing by nhm_seg is impossible"
        )
    fg = splice_flow_graph(
        seg_dis,
        kit.process_dict["prms_channel"]["parameters"],
        new_nodes,
        n_substeps=n_substeps,
        io_in_cfs=io_in_cfs,
        class_name=class_name,
    )

    # -- process_dict: the channel slot becomes the graph, in place --
    process_dict: dict[str, dict[str, Any]] = {}
    for slot, entry in kit.process_dict.items():
        if slot == "prms_channel":
            graph_entry = {
                "class": fg.graph_class,
                "discretization": "nnodes",
                "parameters": fg.parameters,
            }
            # supplied channel state rides onto the graph: zeros on
            # the new nodes' rows (what init_from_params_type writes there)
            for name in fg.graph_class.get_restart_variables():
                if name in entry:
                    padded = np.zeros(fg.n_nodes, dtype=np.float64)
                    padded[: fg.n_seg] = entry[name].values
                    # the channel state's own units ride along
                    graph_entry[name] = xr.DataArray(
                        padded,
                        dims="nnodes",
                        attrs={
                            kk: vv
                            for kk, vv in entry[name].attrs.items()
                            if kk == "units"
                        },
                    )
            # the new nodes' initial state (NewNode.initial_state); on
            # a name the channel also supplied, a new node's value
            # wins on its own row
            for name, values in fg.initial_state.items():
                if name in graph_entry:
                    values = values.fillna(graph_entry[name])
                graph_entry[name] = values
            process_dict["flow_graph"] = graph_entry
        else:
            process_dict[slot] = entry

    # -- Maps: pad + re-target the ones that fed the channel --
    maps: dict[str, Map] = {}
    for name, mm in kit.maps.items():
        if mm.target_grid != "nsegment":
            maps[name] = mm
            continue
        maps[name] = Map(
            weights=fg.pad_weights(mm.weights),
            grid={mm.source_grid: "nnodes"},
            variable={mm.source_var: f"node_{mm.source_var}"},
            derivation=(
                f"{mm.derivation}; zero rows appended for the "
                f"{fg.n_nodes - fg.n_seg} spliced flow-graph nodes"
            ),
        )

    discretizations = {
        grid: disc
        for grid, disc in kit.discretizations.items()
        if grid != "nsegment"
    }
    discretizations["nnodes"] = fg.discretization

    # -- output names: the channel's variables no longer exist --
    control = dict(kit.control)
    dropped = list(kit.dropped_output_var_names)
    if "output_var_names" in control:
        available: set[str] = set()
        for entry in process_dict.values():
            available |= set(entry["class"].get_var_names(include_budget=True))
        kept = [nn for nn in control["output_var_names"] if nn in available]
        dropped += [
            nn for nn in control["output_var_names"] if nn not in available
        ]
        control["output_var_names"] = kept

    return ModelKit(
        process_dict=process_dict,
        control=control,
        maps=maps,
        discretizations=discretizations,
        config=kit.config,
        dropped_output_var_names=dropped,
    )
