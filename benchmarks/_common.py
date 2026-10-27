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
# the complete NHM through stream temperature: 9 processes, 13 Maps
CONTROL = DRB / "nhm_stream_temp.control"

DT = np.float64(86400.0)  # seconds (the channel's s_per_time)

# ---- coverage ledger (README section 4), checked by
# tests/test_benchmark_coverage.py: every Process subclass and every
# FlowGraph node type is benchmarked, or is on that test's explicit
# not-yet-covered list. A control file here = its model's classes
# count as covered; a node type here = a graph benchmark exists.
BENCHMARKED_CONTROLS = [CONTROL]
BENCHMARKED_NODE_TYPES: set[str] = set()


def covered_process_classes() -> set[str]:
    """Class names of every process in the benchmarked controls'
    models (class resolution only; no data loaded)."""
    names: set[str] = set()
    for path in BENCHMARKED_CONTROLS:
        cfg = from_control(load_control(path), path)
        names |= {cls.__name__ for cls in cfg.classes.values()}
    return names

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


def build_model(control: dict | None = None) -> Model:
    """A fresh, un-run model of CONTROL (the whole-model build cost)."""
    return model_from_control(CONTROL, control)


_MODEL: Model | None = None


def get_model() -> Model:
    """The subprocess's one shared model: built, run one step (every
    kernel compiled, every input and Map buffer filled), derived
    parameters made writable again -- then reused by every later
    call."""
    global _MODEL
    if _MODEL is None:
        model = build_model()
        model.run(DT, np.int32(1))
        _unfreeze_parameters_internal(model)
        _MODEL = model
    return _MODEL


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
