"""
hydrology/pass_through_flow_node.py
===================================
PassThroughFlowNode: a node TYPE for the FlowGraph -- gives what it
takes, stores nothing. Ported from pywatershed
hydrology/pass_through_flow_node.py, re-expressed as DATA (see
flow_graph.py): no per-node objects, just this type's field
declarations + init; its substep physics is (Stage 1) inlined in the
flow_graph kernel switch.
"""

import numpy as np

from process import DataArrayMeta


class PassThroughFlowNode:
    """Node type: outflow_substep = upstream + lateral; daily outflow =
    accumulated inflow / n_substeps; no storage, no sink/source."""

    type_name = "pass_through"

    fields = {
        "accum_inflow": DataArrayMeta(
            kind="variable",
            dims=("space",),
            dtype=np.float64,
            description="Pass-through inflow accumulator [cfs] "
            "(kernel work buffer)",
        ),
    }

    @staticmethod
    def initialize_type(dataset) -> None:
        dataset["accum_inflow"].values[:] = 0.0

    @staticmethod
    def advance_type(dataset) -> None:
        pass
