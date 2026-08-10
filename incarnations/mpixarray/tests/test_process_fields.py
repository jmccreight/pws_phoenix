"""Unit pins for DataArrayMeta field resolution (process.py).

The MRO walk resolves ONE final declaration per field name: a subclass
redeclaration OVERRIDES the base's declaration entirely, including its
kind, and a redeclared name keeps the ordering position of its first
declaration. This is the seam the additive variants stand on (e.g.
PRMSRunoffAg turning the frozen ``parameter_derived`` pervious
geometry into a per-step ``variable`` under dynamic ``ag_frac``) --
without resolution, a redeclared name would be listed twice (or under
two kinds) and assembly would break.

No data, no Model -- pure class introspection.
"""

import pathlib as pl
import sys

import numpy as np

sys.path.append(str(pl.Path(__file__).parent.parent))
from globals import Time
from process import DataArrayMeta, Process


class _CoreProc(Process):
    geom = DataArrayMeta(
        kind="parameter_derived",
        dims=("space",),
        dtype=np.float64,
        description="frozen geometry in the core",
    )
    par = DataArrayMeta(
        kind="parameter",
        dims=("space",),
        dtype=np.float64,
        description="a supplied parameter",
    )
    forc = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="a forcing",
    )
    state = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="a state variable",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt: np.float64, time: Time) -> None:
        pass


class _VariantProc(_CoreProc):
    # kind OVERRIDE: the frozen core geometry becomes per-step state
    geom = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="geometry, recomputed per step in the variant",
    )
    # same-kind override: refined metadata
    state = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="a state variable (variant-refined)",
    )
    extra = DataArrayMeta(
        kind="variable",
        dims=("space",),
        dtype=np.float64,
        description="added by the variant",
    )


class TestFieldResolution:
    def test_base_unaffected(self):
        assert _CoreProc.get_parameters() == ("par",)
        assert tuple(_CoreProc.get_parameters_derived()) == ("geom",)
        assert _CoreProc.get_inputs() == ("forc",)
        assert _CoreProc.get_var_names() == ("state",)

    def test_kind_override_moves_field(self):
        """geom is a variable in the variant -- and ONLY a variable."""
        assert "geom" in _VariantProc.get_var_names()
        assert "geom" not in _VariantProc.get_parameters_derived()

    def test_no_duplicates(self):
        names = _VariantProc.get_var_names()
        assert len(names) == len(set(names))
        all_names = (
            _VariantProc.get_parameters()
            + tuple(_VariantProc.get_parameters_derived())
            + _VariantProc.get_inputs()
            + _VariantProc.get_mutable_inputs()
            + _VariantProc.get_var_names()
        )
        assert len(all_names) == len(set(all_names))

    def test_ordering_first_declaration_wins(self):
        """A redeclared name keeps its base position; new fields follow."""
        assert _VariantProc.get_var_names() == ("geom", "state", "extra")

    def test_same_kind_override_takes_subclass_meta(self):
        meta = _VariantProc.get_variables()["state"]
        assert meta.description == "a state variable (variant-refined)"

    def test_inherited_untouched_fields(self):
        assert _VariantProc.get_parameters() == ("par",)
        assert _VariantProc.get_inputs() == ("forc",)
