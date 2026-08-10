"""PRMS control semantics -> pws_phoenix classes + run configuration.

The pws_phoenix-SPECIFIC half of the translation layer (this part
does NOT migrate to pyPRMS): the control file's module names and
flags select concrete Process classes, in NHM schedule order. The
resolution core (`resolve_classes`) is a pure function of primitives
(two dicts), so it is data-free testable; `from_control` adapts a
pyPRMS ControlFile onto it.

Anything the ported classes cannot honor raises NotImplementedError
NAMING the offending module/flag -- silence would mean silently
running different physics than the control asked for.
"""

import pathlib as pl
from dataclasses import dataclass, field

import numpy as np

from atmosphere.prms_atmosphere import PRMSAtmosphere
from hydrology.prms_canopy import PRMSCanopy
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import (
    PRMSGroundwater,
    PRMSGroundwaterNoDprst,
)
from hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometryWidthOnly,
)
from hydrology.prms_runoff import PRMSRunoff, PRMSRunoffNoDprst
from hydrology.prms_snow import PRMSSnow
from hydrology.prms_soilzone import PRMSSoilzone, PRMSSoilzoneNoDprst
from hydrology.prms_stream_temp import (
    PRMSStreamTemp,
    PRMSStreamTempConstantShade,
    PRMSStreamTempSegHumidity,
)

# ---------------------------------------------------------------------
# module tables
# ---------------------------------------------------------------------

# the five control slots PRMSAtmosphere covers as ONE process (CBH
# forcings + jh potet + ddsolrad + tindex transpiration); pyPRMS
# normalizes climate_hru to temperature_hru/precipitation_hru
_ATMOSPHERE_SLOTS = {
    "temp_module": {"temperature_hru", "climate_hru"},
    "precip_module": {"precipitation_hru", "climate_hru"},
    "et_module": {"potet_jh"},
    "solrad_module": {"ddsolrad"},
    "transp_module": {"transp_tindex"},
}

# control module slots with no pws_phoenix counterpart needed (basin =
# area bookkeeping the dis owns; obs = gage data passthrough)
_IGNORED_SLOTS = ("basin_module", "obs_module")

# control entry -> pws_phoenix external-input name (pywatershed
# verbatim; humidity keeps the pyPRMS/PRMS-internal name because it
# feeds the hru->segment humidity Map under exactly that name)
CBH_ENTRY_TO_INPUT = {
    "precip_day": "prcp",
    "tmax_day": "tmax",
    "tmin_day": "tmin",
    "humidity_day": "humidity_hru",
}

_FLAG_NAMES = (
    "dprst_flag",
    "stream_temp_flag",
    "stream_temp_shade_flag",
    "strmtemp_humidity_flag",
    "init_vars_from_file",
)


@dataclass
class PrmsRunConfig:
    """Everything the assembly step needs from a control file."""

    control_path: pl.Path
    start_time: np.datetime64
    end_time: np.datetime64
    # process_dict slot -> pws_phoenix class, in NHM schedule order
    classes: dict[str, type]
    # pws_phoenix input name -> CBH file path
    cbh_paths: dict[str, pl.Path]
    dynamic_parameters: list[str] = field(default_factory=list)
    # the PRMS output REQUEST: nhruOutVar_names + nsegmentOutVar_names
    # combined (each gated by its *OutON_OFF flag). NAMES only -- the
    # control's output PATHS (nhruOutBaseFileName etc.) are
    # deliberately never translated: they are run-machine-relative
    # (drb's even point into the reference-answers directory) and two
    # of them map onto one destination; where to write is always the
    # caller's pws control-dict decision (output_serial_zarr).
    output_var_names: list[str] = field(default_factory=list)


def _require(modules: dict[str, str], slot: str, allowed: set[str]) -> None:
    got = modules.get(slot)
    if got not in allowed:
        raise NotImplementedError(
            f"control {slot} = {got!r}: no pws_phoenix port (ported: "
            f"{sorted(allowed)})."
        )


def resolve_classes(
    modules: dict[str, str], flags: dict[str, int]
) -> dict[str, type]:
    """Module names + flags -> {process_dict slot: class} in NHM
    schedule order. Pure function of primitives (data-free testable);
    raises NotImplementedError naming anything unported."""
    dprst = bool(flags.get("dprst_flag", 0))

    for slot, allowed in _ATMOSPHERE_SLOTS.items():
        _require(modules, slot, allowed)
    _require(modules, "intcp_module", {"intcp"})
    _require(modules, "snow_module", {"snowcomp"})
    _require(modules, "srunoff_module", {"srunoff_smidx"})
    # soilzone_ag IS ported (the PRMSSoilzoneAg family) but its
    # control-driven assembly (dynamic ag_frac + AET files) is not
    # wired into this layer yet
    _require(modules, "soilzone_module", {"soilzone"})
    _require(modules, "gw_module", {"gwflow"})
    _require(modules, "strmflow_module", {"muskingum_mann"})

    classes: dict[str, type] = {
        "prms_atmosphere": PRMSAtmosphere,
        "prms_canopy": PRMSCanopy,
        "prms_snow": PRMSSnow,
        "prms_runoff": PRMSRunoff if dprst else PRMSRunoffNoDprst,
        "prms_soilzone": PRMSSoilzone if dprst else PRMSSoilzoneNoDprst,
        "prms_groundwater": (
            PRMSGroundwater if dprst else PRMSGroundwaterNoDprst
        ),
        "prms_channel": PRMSChannel,
    }

    if flags.get("stream_temp_flag", 0):
        shade = bool(flags.get("stream_temp_shade_flag", 0))
        humid = bool(flags.get("strmtemp_humidity_flag", 0))
        if shade and humid:
            raise NotImplementedError(
                "stream_temp_shade_flag=1 AND strmtemp_humidity_flag=1:"
                " no combined constant-shade + per-segment-humidity "
                "leaf is ported."
            )
        if shade:
            leaf: type = PRMSStreamTempConstantShade
        elif humid:
            leaf = PRMSStreamTempSegHumidity
        else:
            # dynamic shade + CBH humidity (fed through the
            # hru->segment humidity Map)
            leaf = PRMSStreamTemp
        classes["prms_hydraulic_geometry"] = PRMSHydraulicGeometryWidthOnly
        classes["prms_stream_temp"] = leaf

    return classes


def from_control(ctl, control_path: str | pl.Path) -> PrmsRunConfig:
    """Adapt a pyPRMS ControlFile into a PrmsRunConfig (window, class
    resolution, CBH paths resolved relative to the control file)."""
    control_path = pl.Path(control_path)
    flags = {
        nn: int(ctl.get(nn).values)
        for nn in _FLAG_NAMES
        if ctl.exists(nn)
    }
    if flags.get("init_vars_from_file", 0):
        raise NotImplementedError(
            "init_vars_from_file=1: PRMS binary restart files are not "
            "translated (pws_phoenix has its own restart; see "
            "Model.write_restart)."
        )
    classes = resolve_classes(dict(ctl.modules), flags)

    cbh_paths = {}
    for entry, input_name in CBH_ENTRY_TO_INPUT.items():
        if ctl.exists(entry):
            cbh_paths[input_name] = (
                control_path.parent / str(ctl.get(entry).values)
            ).resolve()

    dyn = (
        list(ctl.dynamic_parameters)
        if ctl.has_dynamic_parameters
        else []
    )

    # output-variable NAMES (see the PrmsRunConfig field note: paths
    # are deliberately not translated), each list honored only when
    # its ON_OFF flag is set
    output_var_names: list[str] = []
    for flag_name, list_name in (
        ("nhruOutON_OFF", "nhruOutVar_names"),
        ("nsegmentOutON_OFF", "nsegmentOutVar_names"),
    ):
        on = (
            int(ctl.get(flag_name).values) if ctl.exists(flag_name) else 0
        )
        if on and ctl.exists(list_name):
            output_var_names.extend(
                str(vv)
                for vv in np.atleast_1d(ctl.get(list_name).values)
            )

    return PrmsRunConfig(
        control_path=control_path,
        start_time=np.datetime64(ctl.get("start_time").values, "D"),
        end_time=np.datetime64(ctl.get("end_time").values, "D"),
        classes=classes,
        cbh_paths=cbh_paths,
        dynamic_parameters=dyn,
        output_var_names=output_var_names,
    )
