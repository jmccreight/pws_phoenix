"""FlowGraph lifecycle timings -- the same four as bench_process, on
the `flow_graph` process of each graph in `_flow_graph.GRAPHS`:

    drb_graph_channel    456 channel nodes; calculate vs
                         ProcessLifecycle.time_calculate(prms_channel)
                         = the per-node dispatch cost of the graph
                         kernel over the array muskingum
    ucb_graph_all_types  1942 segments + 11 inserted nodes, all seven
                         node types in one 24-substep cfs graph

time_initialize rebuilds the graph-state namedtuple and recomputes the
Muskingum coefficients and the reservoirs' initial state (the hook is
idempotent; derived parameters are made writable by get_graph_model).
"""

from benchmarks._common import DT, TIMING_ATTRS
from benchmarks._flow_graph import GRAPHS, get_graph_model


class FlowGraphLifecycle:
    params = list(GRAPHS)
    param_names = ["graph"]

    number = TIMING_ATTRS["number"]
    repeat = TIMING_ATTRS["repeat"]
    rounds = TIMING_ATTRS["rounds"]
    warmup_time = TIMING_ATTRS["warmup_time"]
    timeout = TIMING_ATTRS["timeout"]

    def time_init(self, label):
        proc = get_graph_model(label).model_dict["flow_graph"]
        type(proc)(proc._obj)

    def time_initialize(self, label):
        get_graph_model(label).model_dict["flow_graph"].initialize()

    def time_advance(self, label):
        get_graph_model(label).model_dict["flow_graph"].advance()

    def time_calculate(self, label):
        model = get_graph_model(label)
        model.model_dict["flow_graph"].calculate(DT, model.time)
