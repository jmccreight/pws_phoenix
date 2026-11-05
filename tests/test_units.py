"""The units vocabulary (stage 1 of units mode): every declared
``units`` string is a vocabulary key, every vocabulary expression parses
in pint, every float field declares units, and FlowGraph composition
resolves the io-units placeholders. No runtime behavior is tested here
(stage 2); pint is a test dependency only."""

import ast
import pathlib

import pytest
from pws_phoenix import units as units_mod
from pws_phoenix.flow_graph import make_flow_graph
from pws_phoenix.hydrology.obsin_flow_node import ObsInFlowNode
from pws_phoenix.hydrology.pass_through_flow_node import PassThroughFlowNode
from pws_phoenix.hydrology.prms_channel_flow_node import PRMSChannelFlowNode
from pws_phoenix.hydrology.source_sink_flow_node import SourceSinkFlowNode
from pws_phoenix.hydrology.starfit_daily_flow_node import StarfitDailyFlowNode
from pws_phoenix.hydrology.starfit_flow_node import StarfitFlowNode
from pws_phoenix.hydrology.starfit_source_sink_flow_node import (
    StarfitSourceSinkFlowNode,
)
from pws_phoenix.process import _resolved_fields
from pws_phoenix.units import IO_PLACEHOLDERS, VOCABULARY, quantity

PACKAGE = pathlib.Path(units_mod.__file__).parent

ALL_NODE_TYPES = (
    PRMSChannelFlowNode,
    PassThroughFlowNode,
    ObsInFlowNode,
    SourceSinkFlowNode,
    StarfitFlowNode,
    StarfitDailyFlowNode,
    StarfitSourceSinkFlowNode,
)


def _declarations():
    """(file, line, name, dtype source, units literal or None) for every
    ``DataArrayMeta(...)`` in the package, from the source text -- so an
    unimported module or an unregistered class cannot hide one."""
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            calls = []
            if isinstance(node, ast.Assign) and isinstance(
                node.value, ast.Call
            ):
                calls = [
                    (tt.id, node.value)
                    for tt in node.targets
                    if isinstance(tt, ast.Name)
                ]
            elif isinstance(node, ast.Dict):
                calls = [
                    (kk.value, vv)
                    for kk, vv in zip(node.keys, node.values)
                    if isinstance(kk, ast.Constant)
                    and isinstance(vv, ast.Call)
                ]
            for name, call in calls:
                if getattr(call.func, "id", None) != "DataArrayMeta":
                    continue
                kw = {k.arg: k.value for k in call.keywords}
                if "dtype" not in kw:
                    continue  # a factory (budget.py) -- not a declaration
                units = kw.get("units")
                units = (
                    units.value
                    if isinstance(units, ast.Constant)
                    else ("<expr>" if units is not None else None)
                )
                yield (
                    str(path.relative_to(PACKAGE)),
                    call.lineno,
                    name,
                    ast.unparse(kw["dtype"]),
                    units,
                )


DECLARATIONS = list(_declarations())


def test_declarations_found():
    assert len(DECLARATIONS) > 400


@pytest.mark.parametrize(
    "key", [kk for kk, vv in VOCABULARY.items() if vv is not None]
)
def test_vocabulary_parses(key):
    qq = quantity(key)
    assert qq.magnitude > 0


def test_vocabulary_unexpressible_raise():
    with pytest.raises(ValueError):
        quantity("io flow units")
    with pytest.raises(KeyError):
        quantity("inch")  # not a key: the plural is the vocabulary


def test_placeholders_are_vocabulary():
    for key, (cfs_units, cms_units) in IO_PLACEHOLDERS.items():
        assert VOCABULARY[key] is None
        assert VOCABULARY[cfs_units] is not None
        assert VOCABULARY[cms_units] is not None


def test_declared_units_in_vocabulary():
    bad = [
        row
        for row in DECLARATIONS
        if row[4] is not None
        and row[4] != "<expr>"
        and row[4] not in VOCABULARY
    ]
    assert bad == []


def test_float_fields_declare_units():
    missing = [
        row for row in DECLARATIONS if "float" in row[3] and row[4] is None
    ]
    assert missing == []


@pytest.mark.parametrize("io_in_cfs", [True, False])
def test_flow_graph_composition_resolves_placeholders(io_in_cfs):
    cls = make_flow_graph(
        ALL_NODE_TYPES,
        class_name=f"UnitsTestGraph{int(io_in_cfs)}",
        io_in_cfs=io_in_cfs,
    )
    fields = _resolved_fields(cls)
    leftover = {
        nn: mm.units
        for nn, mm in fields.items()
        if mm.units in IO_PLACEHOLDERS
    }
    assert leftover == {}
    which = 0 if io_in_cfs else 1
    assert (
        fields["node_outflows"].units
        == IO_PLACEHOLDERS["io flow units"][which]
    )
    assert (
        fields["lake_storage"].units
        == IO_PLACEHOLDERS["io storage units"][which]
    )
    assert (
        fields["node_sroff_vol"].units
        == (IO_PLACEHOLDERS["io flow units x s"][which])
    )
    # internal STARFIT state stays cms/MCM regardless of graph units
    assert fields["lake_storage_sub"].units == "MCM"
    # the module declarations are untouched (shared objects)
    assert StarfitFlowNode.fields["lake_storage"].units == "io storage units"
