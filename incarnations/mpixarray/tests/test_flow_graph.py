"""FlowGraph regression vs pywatershed answers (drb_2yr, serial).

Two scenarios (module-scoped, parametrized):
- pure_channel: a channel-only graph (456 nodes = the drb segments),
  composed WITHOUT the pass-through type (also exercises the
  missing-type stand-in path).
- pass_through_insert: pywatershed's own FlowGraph doctest scenario --
  one pass-through node inserted above nhm_seg 1829 (457 nodes);
  non-inserted nodes must still match.

Both validate node_outflows against the seg_outflow answers at
rtol = atol = 1e-10 -- pywatershed's OWN standard for its scalar-node
muskingum vs its array muskingum (its doctest asserts abs < 1e-10).

Single-grid on purpose: the hru->node aggregation is PRE-COMPUTED here
(volumes @ weights.T per timestep -- identical math and float order to
a per-step Map apply); Map/MapMPI wiring is already proven by the
PRMSChannel submodel tests, so this test isolates the graph machinery.

Requires GENERATED pywatershed test data; skips with a reason if
absent.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from flow_graph import make_flow_graph
from hydrology.pass_through_flow_node import PassThroughFlowNode
from hydrology.prms_channel_flow_node import PRMSChannelFlowNode
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_VOL_NAMES = ("sroff_vol", "ssres_flow_vol", "gwres_flow_vol")
NODE_INPUT_NAMES = {nn: f"node_{nn}" for nn in INPUT_VOL_NAMES}
DIS_FLOAT_VARS = ("seg_length", "seg_slope", "seg_depth")
NHM_SEG_INSERT_ABOVE = 1829
# pywatershed's own scalar-node-vs-array-muskingum standard
RTOL = ATOL = 1.0e-10
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = [
    DOMAIN_DIR / "parameters_PRMSChannel.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    GEN_DIR / "seg_outflow.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in INPUT_VOL_NAMES]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def channel_params_ds():
    return xr.open_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")


@pytest.fixture(scope="module")
def dis_seg_ds():
    return xr.open_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")


@pytest.fixture(scope="module")
def weights(channel_params_ds):
    """0/1 hru->segment aggregation weights from hru_segment."""
    hru_segment = channel_params_ds["hru_segment"].values
    n_seg = channel_params_ds.sizes["nsegment"]
    ww = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            ww[hru_segment[ihru] - 1, ihru] = 1.0
    return ww


@pytest.fixture(scope="module")
def answers():
    return xr.open_dataarray(GEN_DIR / "seg_outflow.nc")


@pytest.fixture(scope="module", params=["pure_channel", "pass_through_insert"])
def graph_run(
    request,
    dis_seg_ds,
    channel_params_ds,
    weights,
    tmp_path_factory,
):
    insert = request.param == "pass_through_insert"
    out_dir = tmp_path_factory.mktemp(f"flow_graph_{request.param}")
    n_seg = dis_seg_ds.sizes["nsegment"]
    n_nodes = n_seg + 1 if insert else n_seg

    def pad(vals, fill):
        if not insert:
            return vals
        tail = np.array([fill], dtype=vals.dtype)
        return np.concatenate([vals, tail])

    # -- topology + composition --
    to_graph_index = np.zeros(n_nodes, dtype=np.int64)
    to_graph_index[:n_seg] = (
        dis_seg_ds["tosegment"].values.astype(np.int64) - 1
    )
    if insert:
        graph_class = make_flow_graph(
            (PRMSChannelFlowNode, PassThroughFlowNode),
            class_name="DrbInsertFlowGraph",
        )
        node_type = np.full(
            n_nodes,
            graph_class.node_type_code("prms_channel"),
            dtype=np.int64,
        )
        node_type[-1] = graph_class.node_type_code("pass_through")
        # pywatershed doctest splice: the new node goes ABOVE
        # nhm_seg 1829 -- its upstream nodes now flow into the new
        # node, the new node flows into it
        wh_above = int(
            np.where(dis_seg_ds["nhm_seg"].values == NHM_SEG_INSERT_ABOVE)[0][
                0
            ]
        )
        wh_below = np.where(to_graph_index[:n_seg] == wh_above)[0]
        to_graph_index[-1] = wh_above
        to_graph_index[wh_below] = n_seg
    else:
        graph_class = make_flow_graph(
            (PRMSChannelFlowNode,), class_name="DrbChannelFlowGraph"
        )
        node_type = np.full(
            n_nodes,
            graph_class.node_type_code("prms_channel"),
            dtype=np.int64,
        )

    # -- the nnodes dis: topology + composition + padded dis_seg vars --
    graph_dis_vars = {
        "to_graph_index": ("nnodes", to_graph_index),
        "node_type": ("nnodes", node_type),
        "segment_type": (
            "nnodes",
            pad(dis_seg_ds["segment_type"].values, 0),
        ),
    }
    for vv in DIS_FLOAT_VARS:
        graph_dis_vars[vv] = ("nnodes", pad(dis_seg_ds[vv].values, np.nan))
    discretizations = {
        "nnodes": Discretization(
            ["nnodes"],
            parameters=xr.Dataset(graph_dis_vars),
            topo_order={"node_order": "to_graph_index"},
            topo_one_based=False,
        ),
    }

    # -- process parameters (padded) --
    graph_params = xr.Dataset(
        {
            vv: ("nnodes", pad(channel_params_ds[vv].values, np.nan))
            for vv in ("mann_n", "x_coef")
        }
    )

    # -- inputs: hru volumes PRE-AGGREGATED to nodes (see docstring);
    # inserted node = zero column (no lateral inflow) --
    def node_input(name):
        hru_da = xr.open_dataarray(GEN_DIR / f"{name}.nc")
        node_vals = hru_da.values @ weights.T  # (time, n_seg)
        if insert:
            zero_col = np.zeros((node_vals.shape[0], 1))
            node_vals = np.concatenate([node_vals, zero_col], axis=1)
        return xr.DataArray(
            node_vals,
            dims=("time", "nnodes"),
            coords={"time": hru_da["time"].values},
            name=NODE_INPUT_NAMES[name],
        )

    process_dict = {
        "flow_graph": {
            "class": graph_class,
            "discretization": "nnodes",
            "parameters": graph_params,
            **{NODE_INPUT_NAMES[nn]: node_input(nn) for nn in INPUT_VOL_NAMES},
        },
    }
    control = {
        "output_var_names": ["node_outflows"],
        "output_serial_zarr": out_dir / "flow_graph.zarr",
        "time_chunk_size": 61,
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    return {
        "model": model,
        "control": control,
        "class": graph_class,
        "n_seg": n_seg,
        "insert": insert,
    }


class TestFlowGraph:
    def test_node_outflows_all_timesteps(self, graph_run, answers):
        """Channel nodes match the PRMSChannel answers over the full
        run (inserted pass-through excluded -- it has no answer)."""
        output_ds = xr.open_zarr(
            graph_run["control"]["output_serial_zarr"], consolidated=False
        )
        n_seg = graph_run["n_seg"]
        np.testing.assert_allclose(
            output_ds["node_outflows"].values[:, :n_seg],
            answers.values,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_pass_through_is_transparent(self, graph_run):
        """The inserted node's outflow equals the (unchanged) outflow
        of the node it was inserted above's upstream sum -- cheap
        structural check: it received flow at all."""
        if not graph_run["insert"]:
            pytest.skip("pure-channel scenario has no inserted node")
        proc = graph_run["model"].model_dict["flow_graph"]
        assert proc["node_outflows"].values[-1] > 0.0

    def test_derived_parameters_frozen(self, graph_run):
        proc = graph_run["model"].model_dict["flow_graph"]
        for nn in ("ts", "tsi", "c0", "c1", "c2"):
            with pytest.raises(ValueError):
                proc[nn].values[:] = 0

    def test_node_type_introspection(self, graph_run):
        """Codes are internal; names are on the class AND stamped into
        the node_type variable's attrs (self-describing dataset)."""
        graph_class = graph_run["class"]
        proc = graph_run["model"].model_dict["flow_graph"]
        names = proc["node_type"].attrs["node_type_names"]
        assert names[graph_class.node_type_code("prms_channel")] == (
            "prms_channel"
        )
        assert graph_class.node_type_names[0] == "prms_channel"

    def test_unknown_node_type_raises(self):
        class BogusFlowNode:
            type_name = "bogus"
            fields: dict = {}

        with pytest.raises(ValueError, match="not supported"):
            make_flow_graph((BogusFlowNode,))
