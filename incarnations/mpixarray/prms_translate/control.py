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

from atmosphere.prms_atmosphere import (
    PRMSAtmosphere,
    PRMSAtmosphereTranspFrost,
    PRMSAtmosphereTranspFrostDyn,
)
from hydrology.prms_canopy import PRMSCanopy
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import (
    PRMSGroundwater,
    PRMSGroundwaterNoDprst,
)
from hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometryWidthOnly,
)
from hydrology.prms_runoff import (
    PRMSRunoff,
    PRMSRunoffAg,
    PRMSRunoffNoDprst,
)
from hydrology.prms_snow import PRMSSnow
from hydrology.prms_soilzone import PRMSSoilzone, PRMSSoilzoneNoDprst
from hydrology.prms_soilzone_ag import (
    PRMSSoilzoneAg,
    PRMSSoilzoneAgObsET,
)
from hydrology.prms_stream_temp import (
    PRMSStreamTemp,
    PRMSStreamTempConstantShade,
    PRMSStreamTempSegHumidity,
)

# ---------------------------------------------------------------------
# module tables
# ---------------------------------------------------------------------

# the four control slots the PRMSAtmosphere family covers as ONE
# process, plus transpiration (CBH forcings + jh potet + ddsolrad);
# pyPRMS normalizes climate_hru to temperature_hru/precipitation_hru.
# transp_module selects the family LEAF (tindex vs frost window), so
# it is handled separately in resolve_classes.
_ATMOSPHERE_SLOTS = {
    "temp_module": {"temperature_hru", "climate_hru"},
    "precip_module": {"precipitation_hru", "climate_hru"},
    "et_module": {"potet_jh"},
    "solrad_module": {"ddsolrad"},
}
_TRANSP_MODULES = {"transp_tindex", "transp_frost"}

# control module slots with no pws_phoenix counterpart needed (basin =
# area bookkeeping the dis owns; obs = gage data passthrough)
_IGNORED_SLOTS = ("basin_module", "obs_module")

# control entry -> pws_phoenix external-input name (pywatershed
# verbatim; humidity keeps the pyPRMS/PRMS-internal name because it
# feeds the hru->segment humidity Map under exactly that name; the
# OpenET AET file feeds PRMSSoilzoneAgObsET's iteration target).
# PET_cbh_file is deliberately NOT here: the fgr answers prove potet
# stays Jensen-Haise even under iter_aet (analysis potet ==
# spinup potet bit-for-bit), so the PET CBH is PRMS-side accounting
# our port (like pywatershed) never consumes.
CBH_ENTRY_TO_INPUT = {
    "precip_day": "prcp",
    "tmax_day": "tmax",
    "tmin_day": "tmin",
    "humidity_day": "humidity_hru",
    "AET_cbh_file": "aet_observed",
}

_FLAG_NAMES = (
    "dprst_flag",
    "stream_temp_flag",
    "stream_temp_shade_flag",
    "strmtemp_humidity_flag",
    "init_vars_from_file",
    "iter_aet_flag",
    "dyn_ag_frac_flag",
    "dyn_fallfrost_flag",
    "dyn_springfrost_flag",
)

# the PRMS dynamic parameters this layer translates: flag -> (the
# control entry holding the file path, the pws_phoenix INPUT it
# feeds). Any OTHER dyn_*_flag set in a control raises in
# resolve_classes -- silence would drop a requested time variation.
DYNAMIC_PARAM_ENTRIES = {
    "dyn_ag_frac_flag": ("ag_frac_dynamic", "ag_frac"),
    "dyn_springfrost_flag": ("springfrost_dynamic", "spring_frost"),
    "dyn_fallfrost_flag": ("fallfrost_dynamic", "fall_frost"),
}


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
    # the control's param_file entries, resolved (the PRMS
    # multiple-parameter-file feature: the first is full, later ones
    # partial overrides -- readers.load_parameters takes the list)
    param_files: list[pl.Path]
    # pws_phoenix input name -> PRMS dynamic-parameter file path
    # (forward-fill onto model time; see DYNAMIC_PARAM_ENTRIES)
    dyn_param_paths: dict[str, pl.Path] = field(default_factory=dict)
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

    # any requested dynamic parameter this layer does not translate
    # is a loud stop (PRMS has ~20 dyn_*_flag families; three are
    # wired -- see DYNAMIC_PARAM_ENTRIES)
    for name, val in flags.items():
        if (
            name.startswith("dyn_")
            and name.endswith("_flag")
            and val
            and name not in DYNAMIC_PARAM_ENTRIES
        ):
            raise NotImplementedError(
                f"control {name} = {val}: this dynamic parameter is "
                "not translated (wired: "
                f"{sorted(DYNAMIC_PARAM_ENTRIES)})."
            )

    for slot, allowed in _ATMOSPHERE_SLOTS.items():
        _require(modules, slot, allowed)
    _require(modules, "transp_module", _TRANSP_MODULES)
    _require(modules, "intcp_module", {"intcp"})
    _require(modules, "snow_module", {"snowcomp"})
    _require(modules, "srunoff_module", {"srunoff_smidx"})
    _require(modules, "soilzone_module", {"soilzone", "soilzone_ag"})
    _require(modules, "gw_module", {"gwflow"})
    _require(modules, "strmflow_module", {"muskingum_mann"})

    # transp_module selects the atmosphere LEAF; with the frost
    # window, the dyn frost flags select static parameters vs
    # time-varying inputs (both-or-neither: PRMS reads both files in
    # transp_frost.f90, and no mixed leaf is ported)
    dyn_spring = bool(flags.get("dyn_springfrost_flag", 0))
    dyn_fall = bool(flags.get("dyn_fallfrost_flag", 0))
    if modules.get("transp_module") == "transp_frost":
        if dyn_spring != dyn_fall:
            raise NotImplementedError(
                "dyn_springfrost_flag and dyn_fallfrost_flag must "
                "match: no mixed static/dynamic frost leaf is ported."
            )
        atmosphere: type = (
            PRMSAtmosphereTranspFrostDyn
            if dyn_spring
            else PRMSAtmosphereTranspFrost
        )
    else:
        atmosphere = PRMSAtmosphere

    # soilzone_ag selects the agricultural family (runoff AND
    # soilzone move together; iter_aet_flag selects the observed-AET
    # iteration leaf)
    ag = modules.get("soilzone_module") == "soilzone_ag"
    iter_aet = bool(flags.get("iter_aet_flag", 0))
    if ag and not dprst:
        raise NotImplementedError(
            "soilzone_ag with dprst_flag=0: the ag family extends the "
            "dprst-active classes only (no NoDprst ag variants are "
            "ported)."
        )
    if (iter_aet or flags.get("dyn_ag_frac_flag", 0)) and not ag:
        raise NotImplementedError(
            "iter_aet_flag/dyn_ag_frac_flag without "
            "soilzone_module=soilzone_ag: agricultural flags on a "
            "non-ag soilzone."
        )
    if ag:
        runoff: type = PRMSRunoffAg
        soilzone: type = PRMSSoilzoneAgObsET if iter_aet else PRMSSoilzoneAg
    else:
        runoff = PRMSRunoff if dprst else PRMSRunoffNoDprst
        soilzone = PRMSSoilzone if dprst else PRMSSoilzoneNoDprst

    classes: dict[str, type] = {
        "prms_atmosphere": atmosphere,
        "prms_canopy": PRMSCanopy,
        "prms_snow": PRMSSnow,
        "prms_runoff": runoff,
        "prms_soilzone": soilzone,
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
    resolution, paths resolved relative to the control file)."""
    control_path = pl.Path(control_path)
    # every dyn_*_flag PRESENT in the control joins the named flags,
    # so resolve_classes can reject untranslated dynamic parameters
    flag_names = set(_FLAG_NAMES) | {
        str(nn)
        for nn in ctl.control_variables
        if str(nn).startswith("dyn_") and str(nn).endswith("_flag")
    }
    flags = {
        nn: int(ctl.get(nn).values)
        for nn in sorted(flag_names)
        if ctl.exists(nn)
    }
    if flags.get("init_vars_from_file", 0):
        raise NotImplementedError(
            "init_vars_from_file=1: PRMS binary restart files are not "
            "translated (pws_phoenix has its own restart; see "
            "Model.write_restart)."
        )
    classes = resolve_classes(dict(ctl.modules), flags)

    # the PRMS multiple-parameter-file feature: param_file may list
    # several (first full, later partial overrides)
    param_files = [
        (control_path.parent / str(pp)).resolve()
        for pp in np.atleast_1d(ctl.get("param_file").values)
    ]

    cbh_paths = {}
    for entry, input_name in CBH_ENTRY_TO_INPUT.items():
        if ctl.exists(entry):
            cbh_paths[input_name] = (
                control_path.parent / str(ctl.get(entry).values)
            ).resolve()

    # dynamic-parameter files, keyed by the input they feed (NOTE:
    # pyPRMS's .dynamic_parameters listing is unreliable -- it
    # misreads its own valid_values table -- so this is flag-driven)
    dyn_param_paths = {}
    for flag_name, (entry, input_name) in DYNAMIC_PARAM_ENTRIES.items():
        if flags.get(flag_name, 0):
            dyn_param_paths[input_name] = (
                control_path.parent / str(ctl.get(entry).values)
            ).resolve()

    # output-variable NAMES (see the PrmsRunConfig field note: paths
    # are deliberately not translated), each list honored only when
    # its ON_OFF flag is set
    output_var_names: list[str] = []
    for flag_name, list_name in (
        ("nhruOutON_OFF", "nhruOutVar_names"),
        ("nsegmentOutON_OFF", "nsegmentOutVar_names"),
    ):
        on = int(ctl.get(flag_name).values) if ctl.exists(flag_name) else 0
        if on and ctl.exists(list_name):
            output_var_names.extend(
                str(vv) for vv in np.atleast_1d(ctl.get(list_name).values)
            )

    return PrmsRunConfig(
        control_path=control_path,
        start_time=np.datetime64(ctl.get("start_time").values, "D"),
        end_time=np.datetime64(ctl.get("end_time").values, "D"),
        classes=classes,
        cbh_paths=cbh_paths,
        param_files=param_files,
        dyn_param_paths=dyn_param_paths,
        dynamic_parameters=sorted(dyn_param_paths),
        output_var_names=output_var_names,
    )
