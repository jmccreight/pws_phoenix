"""
prms_translate.assemble.insert_flow_graph: a FlowGraph spliced into a
kit built from a PRMS control file (the pws_phoenix form of
pywatershed's prms_channel_flow_graph_to_model_dict).

Domain = ucb_2yr (Big Sandy, as in tests/test_mixed_channel_starfit.py
and pywatershed's test_starfit_flow_graph.py), the full NHM from CBH.
The pin is SELF-CONSISTENT -- no pywatershed answers are involved: the
same kit with and without the graph, every segment not downstream of
the reservoir splice must agree at pywatershed's own scalar-node
standard (1e-10), the bracketing pass-throughs are transparent, and
outflow == release + spill at the reservoir. Big Sandy's nan
initial_storage is filled with half capacity here (the NOR-midpoint
rule is caller data-prep; the structural checks hold for any valid
value).
"""

import numpy as np
import pytest
import xarray as xr
from conftest import PWS_TEST_DATA
from pws_phoenix.flow_graph_build import NewNode
from pws_phoenix.hydrology.pass_through_flow_node import PassThroughFlowNode
from pws_phoenix.hydrology.starfit_flow_node import (
    StarfitFlowNode,
    cms_to_cfs,
)
from pws_phoenix.prms_translate.assemble import (
    assemble_from_control,
    insert_flow_graph,
)

UCB_DIR = PWS_TEST_DATA / "ucb_2yr"
UCB_CONTROL = UCB_DIR / "nhm.control"
DRB_STREAM_TEMP_CONTROL = PWS_TEST_DATA / "drb_2yr" / "nhm_stream_temp.control"
GRAND_PARAM_FILE = PWS_TEST_DATA / "starfit" / "istarf_conus_grand.nc"
BIG_SANDY_GRAND_ID = 419
NHM_SEG_RESERVOIR_ABOVE = 44426  # PT2 -> STARFIT -> PT3 -> this seg
NHM_SEG_PASS_THROUGH_ABOVE = 44409  # PT1 -> this seg
RESERVOIR_ID = 999
N_DAYS = 30
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)
RTOL = ATOL = 1.0e-10  # pywatershed's scalar-node-vs-array standard
STRUCT_RTOL = 1.0e-12
GRAPH_OUTPUTS = [
    "node_outflows",
    "node_upstream_inflows",
    "lake_release",
    "lake_spill",
]

_ucb_needed = [
    UCB_CONTROL,
    UCB_DIR / "myparam.param",
    UCB_DIR / "prcp.cbh",
    GRAND_PARAM_FILE,
]
_ucb_missing = [str(ff) for ff in _ucb_needed if not ff.exists()]
ucb_skipif = pytest.mark.skipif(
    bool(_ucb_missing),
    reason="ucb_2yr legacy files missing: " + ", ".join(_ucb_missing[:3]),
)
drb_skipif = pytest.mark.skipif(
    not DRB_STREAM_TEMP_CONTROL.exists(),
    reason=f"missing {DRB_STREAM_TEMP_CONTROL}",
)


def big_sandy_nodes() -> tuple[list[NewNode], NewNode]:
    """The Big Sandy chain (PT2 -> STARFIT -> PT3 above 44426, plus a
    lone PT1 above 44409) as NewNodes; returns (collated list, the
    reservoir node)."""
    with xr.load_dataset(GRAND_PARAM_FILE) as ds:
        row = ds.where(ds.grand_id == BIG_SANDY_GRAND_ID, drop=True).isel(
            nreservoirs=0
        )
    params = {
        name: float(row[name])
        for name, meta in StarfitFlowNode.fields.items()
        if meta.kind == "parameter" and name != "initial_storage"
    }
    # cfs graph: storages in Mcf (cm_to_cf == cms_to_cfs)
    params["initial_storage"] = 0.5 * params["GRanD_CAP_MCM"] * cms_to_cfs
    pt3 = NewNode(PassThroughFlowNode, NHM_SEG_RESERVOIR_ABOVE, 9003)
    starfit = NewNode(StarfitFlowNode, pt3, RESERVOIR_ID, params)
    pt1 = NewNode(PassThroughFlowNode, NHM_SEG_PASS_THROUGH_ABOVE, 9001)
    pt2 = NewNode(PassThroughFlowNode, starfit, 9002)
    return [starfit, pt1, pt2, pt3], starfit


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The plain kit and the graph kit, both run N_DAYS; returns the
    two output stores plus the graph bookkeeping."""
    out_dir = tmp_path_factory.mktemp("fg_kit")
    plain = assemble_from_control(
        UCB_CONTROL,
        control={
            "budget": "error",
            "output_serial_zarr": out_dir / "plain.zarr",
            "output_var_names": ["seg_outflow"],
            "time_chunk_size": N_DAYS,
        },
    )
    seg_dis = plain.discretizations["nsegment"].parameters
    tosegment0 = seg_dis["tosegment"].values.astype(np.int64) - 1
    nhm_seg = seg_dis["nhm_seg"].values

    nodes, _ = big_sandy_nodes()
    graph = insert_flow_graph(
        assemble_from_control(
            UCB_CONTROL,
            control={
                "budget": "error",
                "output_serial_zarr": out_dir / "graph.zarr",
                "output_var_names": GRAPH_OUTPUTS,
                "time_chunk_size": N_DAYS,
            },
        ),
        nodes,
    )
    for kit in (plain, graph):
        with kit.model() as model:
            model.run(S_PER_TIME, np.int32(N_DAYS))

    n_seg = tosegment0.shape[0]
    rows = {nn.node_id: n_seg + ii for ii, nn in enumerate(nodes)}
    return {
        "plain": plain,
        "graph": graph,
        "plain_ds": xr.open_zarr(out_dir / "plain.zarr", consolidated=False),
        "graph_ds": xr.open_zarr(out_dir / "graph.zarr", consolidated=False),
        "n_seg": n_seg,
        "tosegment0": tosegment0,
        "nhm_seg": nhm_seg,
        "rows": rows,
    }


@ucb_skipif
class TestInsertFlowGraph:
    def test_kit_structure(self, runs):
        plain, graph = runs["plain"], runs["graph"]
        # the channel slot is replaced IN PLACE in the schedule
        plain_slots = list(plain.process_dict)
        graph_slots = list(graph.process_dict)
        assert "prms_channel" not in graph_slots
        assert graph_slots.index("flow_graph") == plain_slots.index(
            "prms_channel"
        )
        assert graph_slots[: len(graph_slots) - 1] == plain_slots[:-1]
        assert set(graph.discretizations) == {"nhru", "nnodes"}
        # the three volume maps: padded, re-targeted
        n_hru = plain.maps["sroff_vol"].weights.shape[1]
        for name in ("sroff_vol", "ssres_vol", "gw_vol"):
            mm = graph.maps[name]
            assert mm.target_grid == "nnodes"
            assert mm.target_var == f"node_{mm.source_var}"
            assert mm.weights.shape == (runs["n_seg"] + 4, n_hru)
            np.testing.assert_array_equal(
                mm.weights[: runs["n_seg"]], plain.maps[name].weights
            )
            assert (mm.weights[runs["n_seg"] :] == 0.0).all()
            assert "spliced" in mm.derivation
        # node ids label the grid
        np.testing.assert_array_equal(
            graph.discretizations["nnodes"].parameters["nnodes"].values,
            [*runs["nhm_seg"], RESERVOIR_ID, 9001, 9002, 9003],
        )

    def test_unaffected_segments_match_plain_channel(self, runs):
        """Every segment not downstream of the reservoir splice agrees
        with the plain PRMSChannel run (pywatershed's wh_ignore walk
        from 44426 downstream, inclusive; the transparent 44409 chain
        is NOT excluded)."""
        n_seg, tosegment0 = runs["n_seg"], runs["tosegment0"]
        unaffected = np.full(n_seg, True)
        ind = int(np.where(runs["nhm_seg"] == NHM_SEG_RESERVOIR_ABOVE)[0][0])
        while ind >= 0:
            unaffected[ind] = False
            ind = tosegment0[ind]
        assert 0 < unaffected.sum() < n_seg
        np.testing.assert_allclose(
            runs["graph_ds"]["node_outflows"].values[:, :n_seg][:, unaffected],
            runs["plain_ds"]["seg_outflow"].values[:, unaffected],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_reservoir_structure(self, runs):
        ds, rows = runs["graph_ds"], runs["rows"]
        outflows = ds["node_outflows"].values
        upstream = ds["node_upstream_inflows"].values
        i_sf, i_pt2, i_pt3 = rows[RESERVOIR_ID], rows[9002], rows[9003]
        np.testing.assert_allclose(
            outflows[:, i_pt2], upstream[:, i_sf], rtol=STRUCT_RTOL
        )
        np.testing.assert_allclose(
            outflows[:, i_pt3], outflows[:, i_sf], rtol=STRUCT_RTOL
        )
        np.testing.assert_allclose(
            outflows[:, i_sf],
            ds["lake_release"].values[:, i_sf]
            + ds["lake_spill"].values[:, i_sf],
            rtol=STRUCT_RTOL,
        )
        assert (outflows[:, i_sf] > 0.0).any()

    def test_output_selectable_by_node_id(self, runs):
        """The nnodes coordinate survives to the zarr store."""
        ds = runs["graph_ds"]
        by_id = ds["node_outflows"].sel(nnodes=RESERVOIR_ID).values
        np.testing.assert_array_equal(
            by_id, ds["node_outflows"].values[:, runs["rows"][RESERVOIR_ID]]
        )


@ucb_skipif
def test_output_names_refiltered(tmp_path):
    """The PRMS request is re-filtered to the GRAPH's variables (the
    channel node declares seg_inflow/seg_outflow as its rows' state,
    so those survive); nothing is auto-added for the graph; anything
    that drops is recorded."""
    kit = assemble_from_control(
        UCB_CONTROL, control={"output_serial_zarr": tmp_path / "x.zarr"}
    )
    requested = kit.control["output_var_names"]
    assert "seg_outflow" in requested
    nodes, _ = big_sandy_nodes()
    graph = insert_flow_graph(kit, nodes)
    available = set()
    for entry in graph.process_dict.values():
        available |= set(entry["class"].get_var_names())
    kept = graph.control["output_var_names"]
    assert kept == [nn for nn in requested if nn in available]
    assert "seg_outflow" in kept
    assert "node_outflows" not in kept
    assert graph.dropped_output_var_names == kit.dropped_output_var_names + [
        nn for nn in requested if nn not in available
    ]


@ucb_skipif
def test_supplied_channel_state_carried_to_graph():
    """Initial state supplied in the channel entry lands on the
    graph's channel rows; the new nodes' rows are zero."""
    kit = assemble_from_control(UCB_CONTROL)
    channel = kit.process_dict["prms_channel"]
    n_seg = kit.discretizations["nsegment"].parameters.sizes["nsegment"]
    supplied = np.arange(1.0, n_seg + 1.0)
    channel["seg_inflow"] = xr.DataArray(supplied, dims="nsegment")
    nodes, _ = big_sandy_nodes()
    graph_entry = insert_flow_graph(kit, nodes).process_dict["flow_graph"]
    carried = graph_entry["seg_inflow"]
    assert carried.dims == ("nnodes",)
    np.testing.assert_array_equal(carried.values[:n_seg], supplied)
    np.testing.assert_array_equal(carried.values[n_seg:], np.zeros(len(nodes)))


@ucb_skipif
def test_no_channel_slot_raises():
    kit = assemble_from_control(UCB_CONTROL)
    del kit.process_dict["prms_channel"]
    with pytest.raises(ValueError, match="prms_channel"):
        insert_flow_graph(kit, [])


@drb_skipif
def test_stream_temp_kit_not_wired():
    kit = assemble_from_control(DRB_STREAM_TEMP_CONTROL)
    with pytest.raises(NotImplementedError, match="seg_outflow"):
        insert_flow_graph(kit, [])
