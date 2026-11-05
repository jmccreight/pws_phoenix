"""
flow_graph_build.splice_flow_graph: the library form of the graph-
building arithmetic that tests/test_flow_graph.py and
tests/test_mixed_channel_starfit.py built by hand.

- synthetic (data-free, CI-able): a 5-segment network; interception,
  in-series chains in both collated orders, padding, the coordinate,
  pad_weights, and every error branch;
- pins against the two hand-built scenarios (the pywatershed doctest
  pass-through splice on drb_2yr; the Big Sandy chain on ucb_2yr) so
  the existing tests can switch to the helper without changing what
  they assemble.
"""

import numpy as np
import pytest
import xarray as xr
from conftest import pyws_domain_files
from pws_phoenix.flow_graph_build import NewNode, splice_flow_graph
from pws_phoenix.hydrology.pass_through_flow_node import PassThroughFlowNode
from pws_phoenix.hydrology.prms_channel_flow_node import PRMSChannelFlowNode
from pws_phoenix.hydrology.source_sink_flow_node import SourceSinkFlowNode
from pws_phoenix.hydrology.starfit_flow_node import StarfitFlowNode

# -- a toy network: 10 -> 30, 20 -> 30, 30 -> 50, 40 -> 50, 50 = outlet
NHM_SEG = np.array([10, 20, 30, 40, 50], dtype=np.int64)
TOSEGMENT = np.array([3, 3, 5, 5, 0], dtype=np.int64)  # 1-based, 0 outlet
N_SEG = 5


@pytest.fixture
def toy_dis_seg():
    return xr.Dataset(
        {
            "nhm_seg": ("nsegment", NHM_SEG),
            "tosegment": ("nsegment", TOSEGMENT),
            "seg_length": ("nsegment", np.arange(1.0, 6.0)),
            "seg_slope": ("nsegment", np.full(N_SEG, 0.01)),
            "seg_depth": ("nsegment", np.full(N_SEG, 2.0)),
            "segment_type": ("nsegment", np.zeros(N_SEG, dtype=np.int64)),
            "seg_cum_area": ("nsegment", np.arange(10.0, 15.0)),
        }
    )


@pytest.fixture
def toy_channel_params():
    return xr.Dataset(
        {
            "mann_n": ("nsegment", np.full(N_SEG, 0.04)),
            "x_coef": ("nsegment", np.full(N_SEG, 0.2)),
            # on another dim: must be ignored by the padding
            "hru_segment": ("nhru", np.array([1, 2, 3, 4, 5, 5])),
        }
    )


class TestSynthetic:
    def test_pure_channel(self, toy_dis_seg, toy_channel_params):
        kit = splice_flow_graph(toy_dis_seg, toy_channel_params, [])
        dis = kit.discretization.parameters
        assert kit.n_seg == kit.n_nodes == N_SEG
        np.testing.assert_array_equal(
            dis["to_graph_index"].values, TOSEGMENT - 1
        )
        assert (dis["node_type"].values == 0).all()
        np.testing.assert_array_equal(dis["nnodes"].values, NHM_SEG)
        assert "node_order" in dis
        assert set(kit.parameters.data_vars) == {"mann_n", "x_coef"}
        assert kit.graph_class.node_type_names == {0: "prms_channel"}

    def test_intercept(self, toy_dis_seg, toy_channel_params):
        kit = splice_flow_graph(
            toy_dis_seg,
            toy_channel_params,
            [NewNode(PassThroughFlowNode, flows_to=30, node_id=999)],
        )
        dis = kit.discretization.parameters
        # 10, 20 now flow into the new node (row 5), which flows into 30
        np.testing.assert_array_equal(
            dis["to_graph_index"].values, [5, 5, 4, 4, -1, 2]
        )
        np.testing.assert_array_equal(dis["node_type"].values, [0] * 5 + [1])
        assert kit.graph_class.node_type_names == {
            0: "prms_channel",
            1: "pass_through",
        }
        np.testing.assert_array_equal(dis["nnodes"].values, [*NHM_SEG, 999])
        assert kit.new_node_rows == [5]
        # padding: floats nan, ints 0; topology vars rebuilt, not padded
        assert np.isnan(dis["seg_length"].values[5])
        assert dis["segment_type"].values[5] == 0
        assert "tosegment" not in dis
        assert "nhm_seg" not in dis.data_vars
        assert np.isnan(kit.parameters["mann_n"].values[5])
        assert "hru_segment" not in kit.parameters

    @pytest.mark.parametrize("reverse", [False, True])
    def test_series_chain_any_collated_order(
        self, toy_dis_seg, toy_channel_params, reverse
    ):
        # PT_above -> STARFIT -> PT_below -> 30 (the Big Sandy shape)
        pt_below = NewNode(PassThroughFlowNode, flows_to=30, node_id=903)
        sf = NewNode(
            StarfitFlowNode,
            flows_to=pt_below,
            node_id=901,
            parameters={"GRanD_CAP_MCM": 100.0, "initial_storage": 40.0},
        )
        pt_above = NewNode(PassThroughFlowNode, flows_to=sf, node_id=902)
        nodes = [pt_above, sf, pt_below]  # worst case: deps come later
        expected = [5, 5, 4, 4, -1, 6, 7, 2]
        if reverse:
            nodes = nodes[::-1]
            expected = [7, 7, 4, 4, -1, 2, 5, 6]
        kit = splice_flow_graph(toy_dis_seg, toy_channel_params, nodes)
        dis = kit.discretization.parameters
        np.testing.assert_array_equal(dis["to_graph_index"].values, expected)
        np.testing.assert_array_equal(
            dis["nnodes"].values[N_SEG:], [nn.node_id for nn in nodes]
        )
        # the STARFIT row carries its parameters, every other row nan
        row = N_SEG + nodes.index(sf)
        cap = kit.parameters["GRanD_CAP_MCM"].values
        assert cap[row] == 100.0
        assert np.isnan(np.delete(cap, row)).all()
        assert kit.parameters["initial_storage"].values[row] == 40.0
        # a declared STARFIT parameter not supplied stays nan
        assert np.isnan(kit.parameters["NORhi_mu"].values).all()
        # types numbered by first appearance in the collated list
        first = nodes[0].node_type.type_name
        assert kit.graph_class.node_type_names[1] == first

    def test_pad_weights(self, toy_dis_seg, toy_channel_params):
        kit = splice_flow_graph(
            toy_dis_seg,
            toy_channel_params,
            [NewNode(PassThroughFlowNode, flows_to=50, node_id=1)],
        )
        weights = np.arange(N_SEG * 6, dtype=float).reshape(N_SEG, 6)
        padded = kit.pad_weights(weights)
        assert padded.shape == (N_SEG + 1, 6)
        np.testing.assert_array_equal(padded[:N_SEG], weights)
        assert (padded[N_SEG] == 0.0).all()
        with pytest.raises(ValueError, match="rows"):
            kit.pad_weights(weights[:-1])

    def test_errors(self, toy_dis_seg, toy_channel_params):
        with pytest.raises(ValueError, match="intercept nhm_seg 30"):
            splice_flow_graph(
                toy_dis_seg,
                toy_channel_params,
                [
                    NewNode(PassThroughFlowNode, flows_to=30, node_id=1),
                    NewNode(PassThroughFlowNode, flows_to=30, node_id=2),
                ],
            )
        with pytest.raises(ValueError, match="not found exactly once"):
            splice_flow_graph(
                toy_dis_seg,
                toy_channel_params,
                [NewNode(PassThroughFlowNode, flows_to=31, node_id=1)],
            )
        orphan = NewNode(PassThroughFlowNode, flows_to=30, node_id=1)
        with pytest.raises(ValueError, match="in-series"):
            splice_flow_graph(
                toy_dis_seg,
                toy_channel_params,
                [NewNode(PassThroughFlowNode, flows_to=orphan, node_id=2)],
            )
        with pytest.raises(ValueError, match="unique"):
            splice_flow_graph(
                toy_dis_seg,
                toy_channel_params,
                [NewNode(PassThroughFlowNode, flows_to=30, node_id=10)],
            )
        with pytest.raises(ValueError, match="unknown parameters"):
            splice_flow_graph(
                toy_dis_seg,
                toy_channel_params,
                [
                    NewNode(
                        SourceSinkFlowNode,
                        flows_to=30,
                        node_id=1,
                        parameters={"flow_min": 0.0, "flow_max": 1.0},
                    )
                ],
            )
        with pytest.raises(ValueError, match="PRMSChannelFlowNode"):
            splice_flow_graph(
                toy_dis_seg,
                toy_channel_params,
                [NewNode(PRMSChannelFlowNode, flows_to=30, node_id=1)],
            )


# -- pins against the hand-built scenarios ------------------------------


def _hand_built_doctest_splice(dis_seg_ds, nhm_seg_above):
    """tests/test_flow_graph.py build_graph_run, pass-through case."""
    n_seg = dis_seg_ds.sizes["nsegment"]
    to_graph_index = np.zeros(n_seg + 1, dtype=np.int64)
    to_graph_index[:n_seg] = (
        dis_seg_ds["tosegment"].values.astype(np.int64) - 1
    )
    wh_above = int(
        np.where(dis_seg_ds["nhm_seg"].values == nhm_seg_above)[0][0]
    )
    wh_below = np.where(to_graph_index[:n_seg] == wh_above)[0]
    to_graph_index[-1] = wh_above
    to_graph_index[wh_below] = n_seg
    return to_graph_index


def _hand_built_big_sandy(dis_seg_ds):
    """tests/test_mixed_channel_starfit.py graph_run: rows n_seg..n_seg+3
    = starfit, pt1 (-> 44409), pt2 (-> starfit), pt3 (starfit -> pt3 ->
    44426)."""
    n_seg = dis_seg_ds.sizes["nsegment"]
    i_starfit, i_pt1, i_pt2, i_pt3 = n_seg, n_seg + 1, n_seg + 2, n_seg + 3
    to_graph_index = np.full(n_seg + 4, -1, dtype=np.int64)
    to_graph_index[:n_seg] = (
        dis_seg_ds["tosegment"].values.astype(np.int64) - 1
    )
    nhm_seg = dis_seg_ds["nhm_seg"].values

    def splice_chain(nhm_target, chain):
        wh_target = int(np.where(nhm_seg == nhm_target)[0][0])
        wh_ups = np.where(to_graph_index[:n_seg] == wh_target)[0]
        to_graph_index[wh_ups] = chain[0]
        for from_node, to_node in zip(chain[:-1], chain[1:], strict=True):
            to_graph_index[from_node] = to_node
        to_graph_index[chain[-1]] = wh_target

    splice_chain(44426, [i_pt2, i_starfit, i_pt3])
    splice_chain(44409, [i_pt1])
    return to_graph_index


@pytest.mark.skipif(
    any(not ff.exists() for ff in pyws_domain_files("drb_2yr")),
    reason="pywatershed drb_2yr test data not generated",
)
def test_pin_doctest_splice_drb(pyws_domain):
    drb = pyws_domain("drb_2yr")
    kit = splice_flow_graph(
        drb["dis_seg_ds"],
        drb["channel_params_ds"],
        [NewNode(PassThroughFlowNode, flows_to=1829, node_id=-1)],
    )
    np.testing.assert_array_equal(
        kit.discretization.parameters["to_graph_index"].values,
        _hand_built_doctest_splice(drb["dis_seg_ds"], 1829),
    )
    for vv in ("seg_length", "seg_slope", "seg_depth"):
        hand = np.concatenate([drb["dis_seg_ds"][vv].values, [np.nan]])
        np.testing.assert_array_equal(
            kit.discretization.parameters[vv].values, hand
        )
    for vv in ("mann_n", "x_coef"):
        hand = np.concatenate([drb["channel_params_ds"][vv].values, [np.nan]])
        np.testing.assert_array_equal(kit.parameters[vv].values, hand)


@pytest.mark.skipif(
    any(not ff.exists() for ff in pyws_domain_files("ucb_2yr")),
    reason="pywatershed ucb_2yr test data not generated",
)
def test_pin_big_sandy_chain_ucb(pyws_domain):
    ucb = pyws_domain("ucb_2yr")
    # collated in the hand-built row order: starfit, pt1, pt2, pt3
    pt3 = NewNode(PassThroughFlowNode, flows_to=44426, node_id=9003)
    starfit = NewNode(StarfitFlowNode, flows_to=pt3, node_id=999)
    pt1 = NewNode(PassThroughFlowNode, flows_to=44409, node_id=9001)
    pt2 = NewNode(PassThroughFlowNode, flows_to=starfit, node_id=9002)
    kit = splice_flow_graph(
        ucb["dis_seg_ds"],
        ucb["channel_params_ds"],
        [starfit, pt1, pt2, pt3],
    )
    dis = kit.discretization.parameters
    np.testing.assert_array_equal(
        dis["to_graph_index"].values, _hand_built_big_sandy(ucb["dis_seg_ds"])
    )
    n_seg = kit.n_seg
    code = kit.graph_class.node_type_code
    expected_type = np.full(n_seg + 4, code("prms_channel"))
    expected_type[n_seg] = code("starfit")
    expected_type[n_seg + 1 :] = code("pass_through")
    np.testing.assert_array_equal(dis["node_type"].values, expected_type)
    assert int(dis["nnodes"].sel(nnodes=999)) == 999
    assert dis["segment_type"].values[n_seg:].tolist() == [0, 0, 0, 0]
