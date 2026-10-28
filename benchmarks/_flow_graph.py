"""FlowGraph models for the benchmarks (bench_flow_graph.py; no
benchmarks in here). Four graphs (README section 4 ledger):

    drb_graph_channel    the drb segment network as a pure channel
                         graph (456 nodes, tests/test_flow_graph.py
                         "pure_channel") -- its calculate against
                         PRMSChannel's is the scalar-node dispatch cost
    ucb_graph_channel    control: the ucb network (1942 segments) as
                         the same one-type channel graph -- network
                         SIZE alone against drb_graph_channel
    drb_graph_all_types_class
                         control: the SEVEN-type class on the plain
                         drb network, every node a channel (the
                         reservoir/obsin/source_sink columns declared
                         but unused) -- class COMPOSITION alone, the
                         literal_unroll chain at the same 456 nodes
    ucb_graph_all_types  the ucb network (1942 segments) with EVERY
                         node type spliced in: the mixed test's Big
                         Sandy chain (PT -> starfit -> PT above nhm_seg
                         44426), a daily-STARFIT chain above 44409, a
                         STARFIT-source-sink chain, a neutral
                         source_sink splice and an obsin headwater on
                         three further segments -- one 24-substep cfs
                         graph composing all seven types (the
                         literal_unroll dispatch at its widest). Big
                         Sandy's parameters sit on every reservoir row
                         (timing only; the physics are not a validated
                         scenario).

Graph-building arithmetic follows tests/test_flow_graph.py and
tests/test_mixed_channel_starfit.py (the pywatershed helpers'
INTERCEPTION semantics: a target's upstreams are redirected into the
inserted chain). The hru->node aggregation is PRE-COMPUTED (volumes @
weights.T, as the tests do); no Maps.
"""

import datetime

import numpy as np
import xarray as xr

from benchmarks._common import (
    DT,
    PWS_TEST_DATA,
    _unfreeze_parameters_internal,
)
from pws_phoenix.discretization import Discretization
from pws_phoenix.flow_graph import make_flow_graph
from pws_phoenix.hydrology.obsin_flow_node import ObsInFlowNode
from pws_phoenix.hydrology.pass_through_flow_node import PassThroughFlowNode
from pws_phoenix.hydrology.prms_channel_flow_node import PRMSChannelFlowNode
from pws_phoenix.hydrology.source_sink_flow_node import SourceSinkFlowNode
from pws_phoenix.hydrology.starfit_daily_flow_node import StarfitDailyFlowNode
from pws_phoenix.hydrology.starfit_flow_node import (
    _OMEGA,
    StarfitFlowNode,
    cms_to_cfs,
)
from pws_phoenix.hydrology.starfit_source_sink_flow_node import (
    StarfitSourceSinkFlowNode,
)
from pws_phoenix.model import Model

VOL_NAMES = ("sroff_vol", "ssres_flow_vol", "gwres_flow_vol")
DIS_FLOAT_VARS = ("seg_length", "seg_slope", "seg_depth")
GRAND_PARAM_FILE = PWS_TEST_DATA / "starfit" / "istarf_conus_grand.nc"
BIG_SANDY_GRAND_ID = 419
NHM_SEG_STARFIT_ABOVE = 44426
NHM_SEG_DAILY_ABOVE = 44409
N_SUBSTEPS = 24
# pywatershed's own combined-node autotest request: a tiny constant sink
SOURCE_SINK_REQUEST = -28.0e-17

ALL_TYPES = (
    PRMSChannelFlowNode,
    PassThroughFlowNode,
    StarfitFlowNode,
    StarfitDailyFlowNode,
    StarfitSourceSinkFlowNode,
    ObsInFlowNode,
    SourceSinkFlowNode,
)
# label -> node types composed; checked by tests/test_benchmark_coverage.py
GRAPHS: dict[str, tuple] = {
    "drb_graph_channel": (PRMSChannelFlowNode,),
    "ucb_graph_channel": (PRMSChannelFlowNode,),
    "drb_graph_all_types_class": ALL_TYPES,
    "ucb_graph_all_types": ALL_TYPES,
}
BENCHMARKED_NODE_TYPES = {
    cls.__name__ for types in GRAPHS.values() for cls in types
}


def graph_data_files() -> list:
    """What the two graphs read (for the coverage test's skipif)."""
    files = [GRAND_PARAM_FILE]
    for domain in ("drb_2yr", "ucb_2yr"):
        ddir = PWS_TEST_DATA / domain
        files += [
            ddir / "parameters_PRMSChannel.nc",
            ddir / "parameters_dis_seg.nc",
            *[ddir / "output" / f"{nn}.nc" for nn in VOL_NAMES],
        ]
    return files


def _domain(domain: str):
    """channel params, dis_seg, and a node-volume-input builder (the
    conftest pyws_domain factory, minus pytest)."""
    ddir = PWS_TEST_DATA / domain
    channel = xr.load_dataset(ddir / "parameters_PRMSChannel.nc")
    dis_seg = xr.load_dataset(ddir / "parameters_dis_seg.nc")
    hru_segment = channel["hru_segment"].values
    n_seg = channel.sizes["nsegment"]
    weights = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            weights[hru_segment[ihru] - 1, ihru] = 1.0

    def node_vol_input(name: str, n_extra: int) -> xr.DataArray:
        hru_da = xr.load_dataarray(ddir / "output" / f"{name}.nc")
        node_vals = hru_da.values @ weights.T  # (time, n_seg)
        if n_extra:
            zeros = np.zeros((node_vals.shape[0], n_extra))
            node_vals = np.concatenate([node_vals, zeros], axis=1)
        return xr.DataArray(
            node_vals,
            dims=("time", "nnodes"),
            coords={"time": hru_da["time"].values},
            name=f"node_{name}",
        )

    return channel, dis_seg, node_vol_input


def _graph_model(
    dis_seg,
    channel,
    node_vol_input,
    graph_class,
    to_graph_index,
    node_type,
    n_extra,
    extra_params=None,
    extra_inputs=None,
) -> Model:
    """Assemble the nnodes Discretization + the one-process Model from
    the composed topology (the tests' shared tail)."""
    n_nodes = to_graph_index.shape[0]

    def pad(vals, fill):
        if not n_extra:
            return vals
        tail = np.full(n_extra, fill, dtype=vals.dtype)
        return np.concatenate([vals, tail])

    graph_dis_vars = {
        "to_graph_index": ("nnodes", to_graph_index),
        "node_type": ("nnodes", node_type),
        "segment_type": ("nnodes", pad(dis_seg["segment_type"].values, 0)),
    }
    for vv in DIS_FLOAT_VARS:
        graph_dis_vars[vv] = ("nnodes", pad(dis_seg[vv].values, np.nan))
    discretizations = {
        "nnodes": Discretization(
            ["nnodes"],
            parameters=xr.Dataset(graph_dis_vars),
            topo_order={"node_order": "to_graph_index"},
            topo_one_based=False,
        ),
    }
    graph_params = xr.Dataset(
        {
            vv: ("nnodes", pad(channel[vv].values, np.nan))
            for vv in ("mann_n", "x_coef")
        }
    )
    for name, vals in (extra_params or {}).items():
        graph_params[name] = ("nnodes", vals)
    node_inputs = {
        f"node_{nn}": node_vol_input(nn, n_extra) for nn in VOL_NAMES
    }
    process_dict = {
        "flow_graph": {
            "class": graph_class,
            "discretization": "nnodes",
            "parameters": graph_params,
            **node_inputs,
            **(extra_inputs or {}),
        },
    }
    assert n_nodes == node_inputs["node_sroff_vol"].shape[1]
    return Model(process_dict, {}, discretizations=discretizations)


def _channel_graph(domain: str) -> Model:
    channel, dis_seg, node_vol_input = _domain(domain)
    n_seg = dis_seg.sizes["nsegment"]
    graph_class = make_flow_graph(
        (PRMSChannelFlowNode,), class_name="BenchChannelFlowGraph"
    )
    to_graph_index = dis_seg["tosegment"].values.astype(np.int64) - 1
    node_type = np.full(
        n_seg, graph_class.node_type_code("prms_channel"), dtype=np.int64
    )
    return _graph_model(
        dis_seg,
        channel,
        node_vol_input,
        graph_class,
        to_graph_index,
        node_type,
        0,
    )


def _nor_midpoint_initial_storage(bs, init_date) -> float:
    """pywatershed's nan-initial_storage fallback (the mixed test's
    copy): the NOR midpoint fraction of capacity at init_date's
    epiweek, in MCM."""
    import epiweeks

    epiweek = epiweeks.Week.fromdate(init_date).week
    sin_t = np.sin(2.0 * np.pi * _OMEGA * epiweek)
    cos_t = np.cos(2.0 * np.pi * _OMEGA * epiweek)

    def nor(prefix):
        return min(
            float(bs[f"{prefix}_max"]),
            max(
                float(bs[f"{prefix}_min"]),
                float(bs[f"{prefix}_mu"])
                + float(bs[f"{prefix}_alpha"]) * sin_t
                + float(bs[f"{prefix}_beta"]) * cos_t,
            ),
        )

    pct_res_cap = (nor("NORlo") + nor("NORhi")) / 2 / 100
    return float(bs["GRanD_CAP_MCM"]) * pct_res_cap


def _all_types_class():
    return make_flow_graph(
        ALL_TYPES,
        class_name="BenchAllTypesFlowGraph",
        n_substeps=N_SUBSTEPS,
        io_in_cfs=True,
    )


def _all_types_extras(n_nodes: int, times) -> tuple[dict, dict]:
    """What the seven-type class DECLARES beyond the channel's fields,
    with no member rows: every STARFIT parameter nan, flow_min and
    source_sink_storage_min zero, obs_flow -1 (pass through) and a
    zero source_sink request on every node. The type hooks only read
    their own rows, so a graph with no reservoir/obsin/source_sink
    nodes runs on these as-is."""
    extra_params = {
        nn: np.full(n_nodes, np.nan)
        for nn, mm in StarfitFlowNode.fields.items()
        if mm.kind == "parameter"
    }
    extra_params["flow_min"] = np.zeros(n_nodes)
    extra_params["source_sink_storage_min"] = np.zeros(n_nodes)

    def flat_input(name, fill):
        return xr.DataArray(
            np.full((times.shape[0], n_nodes), fill),
            dims=("time", "nnodes"),
            coords={"time": times},
            name=name,
        )

    extra_inputs = {
        "node_obs_flow": flat_input("node_obs_flow", -1.0),
        "node_source_sink": flat_input("node_source_sink", 0.0),
    }
    return extra_params, extra_inputs


def _all_types_class_channel_graph(domain: str) -> Model:
    """The seven-type class on the plain network, every node a channel."""
    channel, dis_seg, node_vol_input = _domain(domain)
    n_seg = dis_seg.sizes["nsegment"]
    graph_class = _all_types_class()
    to_graph_index = dis_seg["tosegment"].values.astype(np.int64) - 1
    node_type = np.full(
        n_seg, graph_class.node_type_code("prms_channel"), dtype=np.int64
    )
    times = node_vol_input("sroff_vol", 0)["time"].values
    extra_params, extra_inputs = _all_types_extras(n_seg, times)
    return _graph_model(
        dis_seg,
        channel,
        node_vol_input,
        graph_class,
        to_graph_index,
        node_type,
        0,
        extra_params,
        extra_inputs,
    )


def _all_types_graph(domain: str) -> Model:
    channel, dis_seg, node_vol_input = _domain(domain)
    n_seg = dis_seg.sizes["nsegment"]
    nhm_seg = dis_seg["nhm_seg"].values
    graph_class = _all_types_class()

    # appended nodes: three PT -> reservoir -> PT chains, a neutral
    # source_sink splice, an obsin headwater
    new = {}
    for ii, name in enumerate(
        (
            "pt_sf_above",
            "starfit",
            "pt_sf_below",
            "pt_sfd_above",
            "starfit_daily",
            "pt_sfd_below",
            "pt_sfss_above",
            "starfit_source_sink",
            "pt_sfss_below",
            "source_sink",
            "obsin",
        )
    ):
        new[name] = n_seg + ii
    n_extra = len(new)
    n_nodes = n_seg + n_extra

    to_graph_index = np.full(n_nodes, -1, dtype=np.int64)
    to_graph_index[:n_seg] = dis_seg["tosegment"].values.astype(np.int64) - 1

    def splice_chain(wh_target, chain):
        """Redirect the target's upstreams into chain[0]; wire the
        chain -> target (the pywatershed helper semantics)."""
        wh_ups = np.where(to_graph_index[:n_seg] == wh_target)[0]
        to_graph_index[wh_ups] = chain[0]
        for from_node, to_node in zip(chain[:-1], chain[1:], strict=True):
            to_graph_index[from_node] = to_node
        to_graph_index[chain[-1]] = wh_target

    wh_sf = int(np.where(nhm_seg == NHM_SEG_STARFIT_ABOVE)[0][0])
    wh_sfd = int(np.where(nhm_seg == NHM_SEG_DAILY_ABOVE)[0][0])
    # three more DISTINCT targets with upstreams, lowest indices first
    has_ups = np.unique(to_graph_index[:n_seg][to_graph_index[:n_seg] >= 0])
    others = [int(ww) for ww in has_ups if ww not in (wh_sf, wh_sfd)][:3]
    wh_sfss, wh_ss, wh_obsin = others

    splice_chain(
        wh_sf, [new["pt_sf_above"], new["starfit"], new["pt_sf_below"]]
    )
    splice_chain(
        wh_sfd,
        [new["pt_sfd_above"], new["starfit_daily"], new["pt_sfd_below"]],
    )
    splice_chain(
        wh_sfss,
        [
            new["pt_sfss_above"],
            new["starfit_source_sink"],
            new["pt_sfss_below"],
        ],
    )
    splice_chain(wh_ss, [new["source_sink"]])
    to_graph_index[new["obsin"]] = wh_obsin  # headwater: intercepts nothing

    node_type = np.full(
        n_nodes, graph_class.node_type_code("prms_channel"), dtype=np.int64
    )
    for name, inode in new.items():
        type_name = "pass_through" if name.startswith("pt_") else name
        node_type[inode] = graph_class.node_type_code(type_name)

    # -- parameters: Big Sandy on every reservoir row --
    with xr.load_dataset(GRAND_PARAM_FILE) as ds:
        big_sandy = ds.where(
            ds.grand_id == BIG_SANDY_GRAND_ID, drop=True
        ).isel(nreservoirs=0)
    reservoir_rows = [
        new["starfit"],
        new["starfit_daily"],
        new["starfit_source_sink"],
    ]
    times = node_vol_input("sroff_vol", n_extra)["time"].values
    extra_params, extra_inputs = _all_types_extras(n_nodes, times)
    for nn in extra_params:
        if nn in StarfitFlowNode.fields and nn != "initial_storage":
            extra_params[nn][reservoir_rows] = float(big_sandy[nn])

    init_date = (
        (times[0] - np.timedelta64(1, "D"))
        .astype("datetime64[s]")
        .astype(datetime.datetime)
    )
    init_stor_mcm = _nor_midpoint_initial_storage(big_sandy, init_date)
    extra_params["initial_storage"][reservoir_rows] = (
        init_stor_mcm
        * cms_to_cfs  # cfs graph: MCM -> Mcf (cm_to_cf == cms_to_cfs)
    )

    # -- per-step inputs: the source_sink splice stays neutral (0.0),
    # the obsin headwater passes through (-1.0); the combined node sinks
    extra_inputs["node_source_sink"].values[:, new["starfit_source_sink"]] = (
        SOURCE_SINK_REQUEST
    )
    return _graph_model(
        dis_seg,
        channel,
        node_vol_input,
        graph_class,
        to_graph_index,
        node_type,
        n_extra,
        extra_params,
        extra_inputs,
    )


def build_graph_model(label: str) -> Model:
    if label == "drb_graph_channel":
        return _channel_graph("drb_2yr")
    if label == "ucb_graph_channel":
        return _channel_graph("ucb_2yr")
    if label == "drb_graph_all_types_class":
        return _all_types_class_channel_graph("drb_2yr")
    if label == "ucb_graph_all_types":
        return _all_types_graph("ucb_2yr")
    raise KeyError(label)


_MODELS: dict[str, Model] = {}


def get_graph_model(label: str) -> Model:
    """One warmed graph model per label per subprocess (the
    _common.get_model rule)."""
    if label not in _MODELS:
        model = build_graph_model(label)
        model.run(DT, np.int32(1))
        _unfreeze_parameters_internal(model)
        _MODELS[label] = model
    return _MODELS[label]
