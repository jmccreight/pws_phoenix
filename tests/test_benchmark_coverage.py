"""Every Process subclass and every FlowGraph node type is covered by
the ASV benchmarks -- or is on the explicit not-yet-covered lists
below (benchmarks/README.md section 4, "coverage rule").

The lists can only SHRINK honestly: the asserts are equalities, so a
class that becomes covered must be removed here, and a new class that
is not benchmarked must be added here (a visible decision, not a
silent gap). Classes are gathered by MODULE (pws_phoenix.hydrology.*,
pws_phoenix.atmosphere.*), so the toy Upper/Lower, the translation
layer's HumidityCarrier and factory-made FlowGraph classes -- all
Process subclasses in `Process._registry` -- are out of scope.
"""

import importlib
import pathlib as pl
import pkgutil
import sys

import pytest

import pws_phoenix.atmosphere
import pws_phoenix.hydrology
from pws_phoenix.process import Process

# the benchmarks package is not installed; it lives at the repo root
sys.path.insert(0, str(pl.Path(__file__).resolve().parents[1]))
from benchmarks._common import (  # noqa: E402
    BENCHMARKED_CONTROLS,
    BENCHMARKED_NODE_TYPES,
    covered_process_classes,
)

_missing = [str(ff) for ff in BENCHMARKED_CONTROLS if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason="benchmarked control files missing: " + ", ".join(_missing),
)

PROCESS_PACKAGES = (pws_phoenix.hydrology, pws_phoenix.atmosphere)

# abstract by convention (no @abstractmethod, so inspect.isabstract
# cannot tell): never instantiated, never benchmarked
ABSTRACT_BASES = {"PRMSAtmosphereBase", "PRMSStreamTempBase"}

# the shrinking lists -- the README ledger's remaining rows
PROCESSES_NOT_YET_COVERED = {
    # drb nhm_no_dprst.control
    "PRMSGroundwaterNoDprst",
    "PRMSRunoffNoDprst",
    "PRMSSoilzoneNoDprst",
    # fgr_ag_2yr ag controls
    "PRMSRunoffAg",
    "PRMSSoilzoneAg",
    "PRMSSoilzoneAgObsET",
    # drb nhm_transp_frost.control (+ dyn frost flags)
    "PRMSAtmosphereTranspFrost",
    "PRMSAtmosphereTranspFrostDyn",
    # drb stream-temp variants
    "PRMSStreamTempSegHumidity",
    "PRMSStreamTempConstantShade",
    "PRMSHydraulicGeometryFull",
}
NODE_TYPES_NOT_YET_COVERED = {
    "PRMSChannelFlowNode",
    "PassThroughFlowNode",
    "StarfitFlowNode",
    "StarfitDailyFlowNode",
    "ObsInFlowNode",
    "SourceSinkFlowNode",
    "StarfitSourceSinkFlowNode",
}


def _import_all(package) -> list[str]:
    """Import every module of a package (registers its Process
    subclasses); return the module names."""
    names = []
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        importlib.import_module(info.name)
        names.append(info.name)
    return names


def test_process_classes_covered():
    modules = set()
    for pkg in PROCESS_PACKAGES:
        modules |= set(_import_all(pkg))
    in_scope = {
        name
        for name, cls in Process._registry.items()
        if cls.__module__ in modules
    } - ABSTRACT_BASES
    covered = covered_process_classes()
    assert covered <= in_scope  # the benchmarked controls resolve to real classes
    assert in_scope - covered == PROCESSES_NOT_YET_COVERED


def test_flow_node_types_covered():
    node_types = set()
    for name in _import_all(pws_phoenix.hydrology):
        if not name.endswith("_flow_node"):
            continue
        module = importlib.import_module(name)
        node_types |= {
            attr
            for attr, obj in vars(module).items()
            if isinstance(obj, type)
            and attr.endswith("FlowNode")
            and obj.__module__ == name
        }
    assert node_types - BENCHMARKED_NODE_TYPES == NODE_TYPES_NOT_YET_COVERED
    assert BENCHMARKED_NODE_TYPES <= node_types
