"""Mixed channel + STARFIT FlowGraph (Big Sandy on ucb_2yr), both modes.

Mirrors pywatershed autotest/test_starfit_flow_graph.py: the Big Sandy
reservoir (istarf_conus_grand.nc, grand_id 419 -- its REAL location,
the Upper Colorado Basin domain) and three pass-through nodes spliced
into the ucb channel graph, parametrized over the reservoir mode --
the FIRST composition of three node types, and of the channel with a
reservoir, through the registry dispatch:

- hourly: StarfitFlowNode (nhrs_substep = 1 in the 24-substep graph)
- daily:  StarfitDailyFlowNode (day-constant outflow computed at the
  previous day's end -- the one-day-lag mode; see that module)

Geometry (pywatershed's own scenario; its prms_channel_flow_graph
helpers INTERCEPT -- the target's upstreams are redirected into the
inserted chain):

    [44426's original upstreams] -> PT2 -> STARFIT -> PT3 -> seg 44426
    [44409's original upstreams] -> PT1 -> seg 44409

(The pywatershed test's synthetic disconnected node is not replicated:
the allow_disconnected_nodes knob is deliberately not ported.)

Validation -- pywatershed's own rigor for this test (it pastes the
graph's outputs in as the expected values for the new nodes, so there
is no reference for the reservoir):
- segments NOT downstream of 44426 match the PRMS seg_outflow answers
  at 1e-10 over the full run. 44409's chain has only a TRANSPARENT
  pass-through, so 44409 and its downstream must still match -- the
  in-network interception exercise rides free.
- structure at the reservoir, full run: PT2's outflow == STARFIT's
  upstream inflow; PT3's outflow == STARFIT's outflow; outflow ==
  release + spill.

Data-prep: Big Sandy initial_storage is nan and start_time NaT.
pywatershed's node seeds storage at the NOR midpoint,
(min_nor + max_nor)/2/100 * GRanD_CAP_MCM, at the epiweek of the
control INIT time (one day before the first timestep) when start_time
is NaT. That in-node fallback is deliberately not ported (our
initialize_type raises on nan), so it is replicated HERE, test-side,
and supplied in io units (the graph is cfs -> millions of cubic feet).

Requires GENERATED pywatershed ucb_2yr test data (the drb tests use
drb_2yr; ucb generation is the same pywatershed workflow) + the
istarf_conus_grand.nc parameters; skips with a reason if absent.
"""

import datetime
from typing import Any

import numpy as np
import pytest
import xarray as xr
from conftest import (
    PYWS_INPUT_VOL_NAMES,
    PWS_TEST_DATA,
    pyws_domain_files,
)
from pws_phoenix.flow_graph_build import NewNode, splice_flow_graph
from pws_phoenix.hydrology.pass_through_flow_node import PassThroughFlowNode
from pws_phoenix.hydrology.starfit_daily_flow_node import StarfitDailyFlowNode
from pws_phoenix.hydrology.starfit_flow_node import (
    _OMEGA,
    StarfitFlowNode,
    cms_to_cfs,
)
from pws_phoenix.model import Model

DOMAIN = "ucb_2yr"
GRAND_PARAM_FILE = PWS_TEST_DATA / "starfit" / "istarf_conus_grand.nc"
BIG_SANDY_GRAND_ID = 419
NHM_SEG_RESERVOIR_ABOVE = 44426  # PT2 -> STARFIT -> PT3 -> this seg
NHM_SEG_PASS_THROUGH_ABOVE = 44409  # PT1 -> this seg

NODE_INPUT_NAMES = {nn: f"node_{nn}" for nn in PYWS_INPUT_VOL_NAMES}
N_SUBSTEPS = 24
# unaffected segments: pywatershed's own scalar-node standard;
# structural identities at the reservoir: tighter
RTOL = ATOL = 1.0e-10
STRUCT_RTOL = 1.0e-12
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)
OUTPUT_VARS = [
    "node_outflows",
    "node_upstream_inflows",
    "lake_release",
    "lake_spill",
]
# per-node mass budget (cfs terms, 24 substeps): error mode at
# the default 1e-10; the written residual is pinned at roundoff
BUDGET_TOL = 1.0e-8

_needed = [GRAND_PARAM_FILE, *pyws_domain_files(DOMAIN)]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        f"pywatershed {DOMAIN}/starfit test data not found; missing: "
        + ", ".join(_missing[:3])
    ),
)

# values are duck-typed node-type classes (see the make_flow_graph
# contract), hence Any
RESERVOIR_MODES: dict[str, Any] = {
    "hourly": StarfitFlowNode,
    "daily": StarfitDailyFlowNode,
}


def nor_midpoint_initial_storage(bs, init_date):
    """pywatershed's nan-initial_storage fallback, test-side: the NOR
    midpoint fraction of capacity at `init_date`'s epiweek (raw, no
    53->52 fold -- init uses the raw week), in MCM."""
    import epiweeks

    epiweek = epiweeks.Week.fromdate(init_date).week
    sin_t = np.sin(2.0 * np.pi * _OMEGA * epiweek)
    cos_t = np.cos(2.0 * np.pi * _OMEGA * epiweek)
    max_nor = min(
        float(bs["NORhi_max"]),
        max(
            float(bs["NORhi_min"]),
            float(bs["NORhi_mu"])
            + float(bs["NORhi_alpha"]) * sin_t
            + float(bs["NORhi_beta"]) * cos_t,
        ),
    )
    min_nor = min(
        float(bs["NORlo_max"]),
        max(
            float(bs["NORlo_min"]),
            float(bs["NORlo_mu"])
            + float(bs["NORlo_alpha"]) * sin_t
            + float(bs["NORlo_beta"]) * cos_t,
        ),
    )
    pct_res_cap = (min_nor + max_nor) / 2 / 100
    return float(bs["GRanD_CAP_MCM"]) * pct_res_cap


@pytest.fixture(scope="module")
def big_sandy():
    with xr.load_dataset(GRAND_PARAM_FILE) as ds:
        return ds.where(ds.grand_id == BIG_SANDY_GRAND_ID, drop=True).isel(
            nreservoirs=0
        )


@pytest.fixture(scope="module", params=list(RESERVOIR_MODES.keys()))
def graph_run(request, pyws_domain, big_sandy, tmp_path_factory):
    mode = request.param
    starfit_class = RESERVOIR_MODES[mode]
    ucb = pyws_domain(DOMAIN)
    dis_seg_ds = ucb["dis_seg_ds"]
    channel_params_ds = ucb["channel_params_ds"]
    out_dir = tmp_path_factory.mktemp(f"mixed_starfit_{mode}")
    n_seg = dis_seg_ds.sizes["nsegment"]

    # inputs (zero columns for the 4 new nodes)
    node_inputs = {
        NODE_INPUT_NAMES[nn]: ucb["node_vol_input"](
            nn, NODE_INPUT_NAMES[nn], 4
        )
        for nn in PYWS_INPUT_VOL_NAMES
    }

    # -- Big Sandy's row: the GRAND parameters + the nan initial_storage
    # -> NOR midpoint at the epiweek of one day BEFORE the first
    # timestep (pywatershed's control init time), in io units (cfs
    # graph: MCM -> Mcf) --
    starfit_params = {
        nn: float(big_sandy[nn])
        for nn, mm in StarfitFlowNode.fields.items()
        if mm.kind == "parameter" and nn != "initial_storage"
    }
    time0 = node_inputs["node_sroff_vol"]["time"].values[0]
    init_date = (
        (time0 - np.timedelta64(1, "D"))
        .astype("datetime64[s]")
        .astype(datetime.datetime)
    )
    init_stor_mcm = nor_midpoint_initial_storage(big_sandy, init_date)
    starfit_params["initial_storage"] = (
        init_stor_mcm * cms_to_cfs  # cm_to_cf == cms_to_cfs
    )

    # -- the two INTERCEPTING insertions, collated in pywatershed's
    # scenario order (rows n_seg..n_seg+3): starfit, pt1, pt2, pt3 --
    pt3 = NewNode(PassThroughFlowNode, NHM_SEG_RESERVOIR_ABOVE, 9003)
    starfit = NewNode(starfit_class, pt3, 999, starfit_params)
    pt1 = NewNode(PassThroughFlowNode, NHM_SEG_PASS_THROUGH_ABOVE, 9001)
    pt2 = NewNode(PassThroughFlowNode, starfit, 9002)
    graph = splice_flow_graph(
        dis_seg_ds,
        channel_params_ds,
        [starfit, pt1, pt2, pt3],
        n_substeps=N_SUBSTEPS,
        io_in_cfs=True,
        class_name=f"UcbBigSandy_{mode}_Graph",
    )
    graph_class = graph.graph_class
    i_starfit, _, i_pt2, i_pt3 = graph.new_node_rows
    wh_44426 = int(
        np.where(dis_seg_ds["nhm_seg"].values == NHM_SEG_RESERVOIR_ABOVE)[0][0]
    )
    discretizations = {"nnodes": graph.discretization}

    process_dict = {
        "flow_graph": {
            "class": graph_class,
            "discretization": "nnodes",
            "parameters": graph.parameters,
            **node_inputs,
        },
    }
    budget_prefix = f"{graph_class.__name__}_mass"
    control = {
        "output_var_names": [
            *OUTPUT_VARS,
            f"{budget_prefix}_residual",
            f"{budget_prefix}_storage_changes_sum",
        ],
        "output_serial_zarr": out_dir / "mixed_graph.zarr",
        "time_chunk_size": 61,
        "budget": "error",
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))

    return {
        "model": model,
        "control": control,
        "mode": mode,
        "n_seg": n_seg,
        "wh_44426": wh_44426,
        "i_starfit": i_starfit,
        "i_pt2": i_pt2,
        "i_pt3": i_pt3,
        "tosegment0": dis_seg_ds["tosegment"].values.astype(np.int64) - 1,
        "answers": ucb["seg_outflow"],
        "budget_prefix": budget_prefix,
    }


@pytest.fixture(scope="module")
def output_ds(graph_run):
    return xr.open_zarr(
        graph_run["control"]["output_serial_zarr"], consolidated=False
    )


class TestMixedChannelStarfit:
    def test_mass_budget(self, graph_run, output_ds):
        """Per-node residual is roundoff every day on all three node
        types (the run already passed error mode); the reservoir row
        carries a nonzero storage term (the hourly node's state delta;
        the daily node's flux-defined change) -- the budget sees the
        reservoir, the channel/pass-through rows are identities."""
        prefix = graph_run["budget_prefix"]
        residual = output_ds[f"{prefix}_residual"].values
        assert np.all(np.abs(residual) < BUDGET_TOL)
        dstor = output_ds[f"{prefix}_storage_changes_sum"].values
        assert (dstor[:, graph_run["i_starfit"]] != 0.0).any()
        assert np.all(dstor[:, graph_run["i_pt2"]] == 0.0)

    def test_unaffected_segments_match(self, graph_run, output_ds):
        """Every segment NOT downstream of the reservoir splice matches
        the PRMSChannel answers over the full run (pywatershed's
        wh_ignore walk, from 44426 downstream inclusive). 44409's
        chain is a transparent pass-through, so it is NOT ignored."""
        n_seg = graph_run["n_seg"]
        tosegment0 = graph_run["tosegment0"]
        unaffected = np.full(n_seg, True)
        ind = graph_run["wh_44426"]
        while ind >= 0:
            unaffected[ind] = False
            ind = tosegment0[ind]
        assert unaffected.sum() < n_seg  # the walk found something
        np.testing.assert_allclose(
            output_ds["node_outflows"].values[:, :n_seg][:, unaffected],
            graph_run["answers"].values[:, unaffected],
            rtol=RTOL,
            atol=ATOL,
            err_msg=graph_run["mode"],
        )

    def test_reservoir_structure(self, graph_run, output_ds):
        """Full-run identities at the reservoir (the pywatershed
        checks): the bracketing pass-throughs are transparent and
        outflow decomposes into release + spill."""
        outflows = output_ds["node_outflows"].values
        upstream = output_ds["node_upstream_inflows"].values
        ii_sf = graph_run["i_starfit"]
        # PT2 (above) delivers its outflow as STARFIT's upstream inflow
        np.testing.assert_allclose(
            outflows[:, graph_run["i_pt2"]],
            upstream[:, ii_sf],
            rtol=STRUCT_RTOL,
            err_msg="pt2 -> starfit",
        )
        # PT3 (below) passes STARFIT's outflow through
        np.testing.assert_allclose(
            outflows[:, graph_run["i_pt3"]],
            outflows[:, ii_sf],
            rtol=STRUCT_RTOL,
            err_msg="starfit -> pt3",
        )
        np.testing.assert_allclose(
            outflows[:, ii_sf],
            output_ds["lake_release"].values[:, ii_sf]
            + output_ds["lake_spill"].values[:, ii_sf],
            rtol=STRUCT_RTOL,
            err_msg="outflow = release + spill",
        )
        # the reservoir did something (received + released real flow)
        assert (outflows[:, ii_sf] > 0.0).any()
