"""Shared pieces for the ASV benchmarks (no benchmarks in here).

Design: benchmarks/README.md. The one mechanism worth restating:
ASV re-runs a benchmark's ``setup`` hook before EVERY timed call
(asv_runner 0.3.2, README section 3), so the model is NOT built in a
``setup``. ``get_model()`` builds it the first time a benchmark in the
subprocess asks and keeps it in a module variable; the first call
(build + numba compile) falls in ASV's warm-up period and is
discarded.
"""

import os
import pathlib as pl

import numpy as np
import xarray as xr

from pws_phoenix.discretization import Discretization
from pws_phoenix.hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometryFull,
)
from pws_phoenix.hydrology.prms_stream_temp import (
    PRMSStreamTempConstantShade,
    PRMSStreamTempSegHumidity,
)
from pws_phoenix.model import Model
from pws_phoenix.prms_translate import (
    assemble_from_control,
    from_control,
    load_control,
    model_from_control,
)

# the same data-root rule as tests/conftest.py: the sibling
# pywatershed clone's test_data (populated from the pws-prms-oracle),
# PWS_TEST_DATA overriding
MPIX_ROOT = pl.Path(__file__).resolve().parents[2]
PWS_TEST_DATA = pl.Path(
    os.environ.get("PWS_TEST_DATA", MPIX_ROOT / "pywatershed" / "test_data")
)
DRB = PWS_TEST_DATA / "drb_2yr"
FGR = PWS_TEST_DATA / "fgr_ag_2yr"
# the complete NHM through stream temperature: 9 processes, 13 Maps
CONTROL = DRB / "nhm_stream_temp.control"

# label -> control file. The first is the whole-model subject
# (bench_model.py); the others exist for the process classes they ADD
# (README section 4 ledger; VARIANT_ROWS below)
CONTROLS = {
    "drb_stream_temp": CONTROL,
    "drb_no_dprst": DRB / "nhm_no_dprst.control",
    "fgr_spinup": FGR / "spinup.control",
    "fgr_analysis": FGR / "analysis.control",
}

# hand-built segment-grid models (README section 4 ledger): the
# classes model_from_control cannot select, fed the drb nhm_stream_temp
# ANSWERS as disk forcings -- the standalone tests' recipe
# (tests/test_prms_hydraulic_geometry.py, test_prms_stream_temp_
# const_shade.py, test_prms_stream_temp_seg_humidity.py). label ->
# {process_dict key: class}; _build_hand_built assembles them.
HAND_BUILT: dict[str, dict[str, type]] = {
    # HydraulicGeometryFull's seg_flow_width feeds ConstantShade live
    "drb_seg_hg_full": {
        "prms_hydraulic_geometry": PRMSHydraulicGeometryFull,
        "prms_stream_temp": PRMSStreamTempConstantShade,
    },
    "drb_seg_humid": {"prms_stream_temp": PRMSStreamTempSegHumidity},
}
STREAM_TEMP_ANSWERS = DRB / "output_stream_temp"

DT = np.float64(86400.0)  # seconds (the channel's s_per_time)

# ---- coverage ledger (README section 4), checked by
# tests/test_benchmark_coverage.py: every Process subclass and every
# FlowGraph node type (_flow_graph.GRAPHS) is benchmarked, or is on
# that test's explicit not-yet-covered list.
BENCHMARKED_CONTROLS = list(CONTROLS.values())
BENCHMARKED_DATA = BENCHMARKED_CONTROLS + [
    DRB / "parameters_PRMSStreamTemp.nc",
    DRB / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
    DRB / "parameters_dis_seg.nc",
    DRB / "parameters_dis_hru.nc",
    STREAM_TEMP_ANSWERS / "seg_outflow.nc",
]

# process_dict keys of CONTROL's model, NHM schedule order
# (pws_phoenix/prms_translate/control.py resolve_classes)
PROCESS_NAMES = [
    "prms_atmosphere",
    "prms_canopy",
    "prms_snow",
    "prms_runoff",
    "prms_soilzone",
    "prms_groundwater",
    "prms_channel",
    "prms_hydraulic_geometry",
    "prms_stream_temp",
]

# "label/process_dict key" rows of bench_process.ProcessVariants: in
# each control, ONLY the slots whose class no earlier row benchmarks
# (the no-repeat rule); tests/test_benchmark_coverage.py checks both
# directions against resolve_classes
VARIANT_ROWS = [
    "drb_no_dprst/prms_runoff",  # PRMSRunoffNoDprst
    "drb_no_dprst/prms_soilzone",  # PRMSSoilzoneNoDprst
    "drb_no_dprst/prms_groundwater",  # PRMSGroundwaterNoDprst
    "fgr_spinup/prms_atmosphere",  # PRMSAtmosphereTranspFrost
    "fgr_spinup/prms_runoff",  # PRMSRunoffAg
    "fgr_spinup/prms_soilzone",  # PRMSSoilzoneAg
    "fgr_analysis/prms_atmosphere",  # PRMSAtmosphereTranspFrostDyn
    "fgr_analysis/prms_soilzone",  # PRMSSoilzoneAgObsET
    "drb_seg_hg_full/prms_hydraulic_geometry",  # PRMSHydraulicGeometryFull
    "drb_seg_hg_full/prms_stream_temp",  # PRMSStreamTempConstantShade
    "drb_seg_humid/prms_stream_temp",  # PRMSStreamTempSegHumidity
]


def split_row(row: str) -> tuple[str, str]:
    """A "label/process" row -> (CONTROLS key, process_dict key)."""
    label, name = row.split("/")
    return label, name


def benchmarked_classes() -> dict[str, str]:
    """row -> class name for every benchmarked (control, slot): the
    nine of CONTROL plus VARIANT_ROWS (class resolution only; no data
    loaded)."""
    resolved: dict[str, dict[str, type]] = {
        label: from_control(load_control(path), path).classes
        for label, path in CONTROLS.items()
    } | HAND_BUILT
    rows = [f"drb_stream_temp/{nn}" for nn in PROCESS_NAMES] + VARIANT_ROWS
    out = {}
    for row in rows:
        label, name = split_row(row)
        out[row] = resolved[label][name].__name__
    return out


def covered_process_classes() -> set[str]:
    return set(benchmarked_classes().values())


# ASV timing attributes shared by the suite (README section 5): an
# explicit `number` keeps ASV's auto-calibration from settling on 1
# after a multi-second first call; `warmup_time` must hold the model
# build + numba compile; the 60 s default `timeout` cannot.
TIMING_ATTRS = {
    "number": 10,
    "repeat": 10,
    "rounds": 1,
    "warmup_time": 2.0,
    "timeout": 900,
}


def build_model(
    label: str = "drb_stream_temp", control: dict | None = None
) -> Model:
    """A fresh, un-run model of CONTROLS[label] (the whole-model build
    cost) or of HAND_BUILT[label]."""
    if label in HAND_BUILT:
        return _build_hand_built(label)
    return model_from_control(CONTROLS[label], control)


# the stream-temp family's segment-grid inputs, all in the answers dir
_SEG_FORCINGS = (
    "seg_outflow",
    "seg_lateral_inflow",
    "seg_flow_width",
    "seg_tave_air",
    "seg_humid",
    "seg_ccov",
    "seg_melt",
    "seg_rain",
    "seg_potet",
    "seginc_swrad",
    "seginc_sroff",
    "seginc_ssflow",
    "seginc_gwflow",
)


def _seg_forcings(names) -> dict[str, xr.DataArray]:
    # the answer files are on pywatershed's nhm_seg dim
    return {
        nn: xr.load_dataarray(STREAM_TEMP_ANSWERS / f"{nn}.nc").rename(
            {"nhm_seg": "nsegment"}
        )
        for nn in names
    }


def _build_hand_built(label: str) -> Model:
    classes = HAND_BUILT[label]
    st_params = xr.merge(
        [
            xr.load_dataset(DRB / "parameters_PRMSStreamTemp.nc"),
            xr.load_dataset(DRB / "parameters_PRMSStreamShadeDynamic.nc"),
            xr.load_dataset(DRB / "parameters_dis_seg.nc")[["lat_temp_adj"]],
            xr.load_dataset(DRB / "parameters_dis_hru.nc")[["hru_area"]],
        ],
        compat="no_conflicts",
    )
    nseg = st_params.sizes["nsegment"]
    process_dict: dict[str, dict] = {}
    if label == "drb_seg_hg_full":
        hg_params = xr.load_dataset(
            DRB / "parameters_PRMSHydraulicGeometryWidthOnly.nc"
        )
        # WidthOnly's built-in depth defaults, made explicit
        hg_params["depth_alpha"] = (("nsegment",), np.full(nseg, 0.27))
        hg_params["depth_m"] = (("nsegment",), np.full(nseg, 0.39))
        # synthetic seasonal shade fractions (the test's values)
        st_params["segshade_sum"] = (
            ("nsegment",),
            np.linspace(0.05, 0.45, nseg),
        )
        st_params["segshade_win"] = (
            ("nsegment",),
            np.linspace(0.02, 0.20, nseg),
        )
        forcings = _seg_forcings(_SEG_FORCINGS)
        process_dict["prms_hydraulic_geometry"] = {
            "class": classes["prms_hydraulic_geometry"],
            "discretization": "nsegment",
            "parameters": hg_params,
            "seg_outflow": forcings.pop("seg_outflow"),
        }
        # seg_flow_width comes live from hydraulic geometry; seg_outflow
        # is shared on the grid (fed once, above)
        del forcings["seg_flow_width"]
    else:
        # seg_humid is this leaf's computed variable (from the
        # seg_humidity parameter)
        forcings = _seg_forcings(
            [nn for nn in _SEG_FORCINGS if nn != "seg_humid"]
        )
    process_dict["prms_stream_temp"] = {
        "class": classes["prms_stream_temp"],
        "discretization": "nsegment",
        "parameters": st_params,
        **forcings,
    }
    discretizations = {
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DRB / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }
    return Model(process_dict, {}, discretizations=discretizations)


_MODELS: dict[str, Model] = {}


def get_model(label: str = "drb_stream_temp") -> Model:
    """The subprocess's one shared model per control: built, run one
    step (every kernel compiled, every input and Map buffer filled),
    derived parameters made writable again -- then reused by every
    later call."""
    if label not in _MODELS:
        model = build_model(label)
        model.run(DT, np.int32(1))
        _unfreeze_parameters_internal(model)
        _MODELS[label] = model
    return _MODELS[label]


_OUTPUT_VAR_NAMES: list[str] | None = None


def get_output_var_names() -> list[str]:
    """CONTROL's PRMS output request (nhruOutVar_names +
    nsegmentOutVar_names) filtered to the variables the model has --
    what assemble_from_control injects as control["output_var_names"]
    when handed a store path (the path is never used here). Resolved
    once per subprocess."""
    global _OUTPUT_VAR_NAMES
    if _OUTPUT_VAR_NAMES is None:
        kit = assemble_from_control(
            CONTROL, {"output_serial_zarr": "unused.zarr"}
        )
        _OUTPUT_VAR_NAMES = list(kit.control["output_var_names"])
    return _OUTPUT_VAR_NAMES


def _unfreeze_parameters_internal(model: Model) -> None:
    """Undo the Model's post-initialize freeze (model.py
    _run_initialize_hooks sets writeable = False on every
    parameter_internal array) so `initialize()` can be re-timed.
    Benchmark-only: no kernel reads the flag."""
    for proc in model.model_dict.values():
        for name in proc.get_parameters_internal():
            proc[name].values.flags.writeable = True
