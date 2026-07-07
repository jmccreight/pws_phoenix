"""STARFIT-family node regression vs pywatershed reference data.

Mirrors pywatershed autotest/test_starfit_flow_node.py AND
test_starfit_source_sink_flow_node.py (cms cases): the 115 reservoirs
of `STARFIT_INDS_TEST` (start_time == 1995-01-01 and end_time >=
2001-12-31) run as ISOLATED OUTLET nodes (to_graph_index = -1) in a
one-type graph fed by lake_inflow.nc, 1995-01-01..2001-12-31 (2557
days). The reference values are TIME-MEANS from offline STARFIT runs
(starfit_mean_output_1995-2001.nc), so the check compares time-means
of lake_storage / lake_release / lake_spill at rtol = atol = 1e-7 --
pywatershed's own tolerance for this comparison.

THE "FAKE DAILY" TRICK (n_substeps=1): the reference is a DAILY
formulation -- each day's release from that day's inflow and current
storage, CONCURRENTLY. These are the HOURLY node types, but run with
ONE substep of 24 hours (pywatershed nhrs_substep=24, its autotests'
`for ss in range(1)`): with one substep per day, "this substep's
inflow" IS the day's inflow and the release is computed from current
storage -- exactly the concurrent daily formulation, so the hourly
physics reproduce the daily reference. This is NOT the same as
StarfitDailyFlowNode (compute_daily): that type lives in sub-daily
graphs and applies a constant outflow computed at the PREVIOUS day's
end (a one-day lag), so it can never match this reference tightly --
see hydrology/starfit_daily_flow_node.py and
tests/test_starfit_daily_parity.py.

Four scenarios (module-scoped, parametrized = node class x units,
exactly the pywatershed autotest matrix):
- starfit: StarfitFlowNode as above.
- starfit_source_sink: StarfitSourceSinkFlowNode with pywatershed's
  own autotest data for it -- a TINY constant sink (-28e-17) and
  source_sink_storage_min = 0 -- validated against the SAME reference
  means at 1e-7, plus a check that the applied diversion equals the
  request (limited only where a reservoir drains to the minimum).
- each in cms (io_in_cfs=False; the reference's native units) AND cfs
  (io_in_cfs=True; inflows/initial_storage/answers converted with the
  pywatershed constants, as its autotest does -- checks the io-unit
  boundary layer through the same numerics).

Data-prep here (pywatershed does these inside the node; parameters
freeze at assembly in this framework):
- nan Obs_MEANFLOW_CUMECS <- inflow_mean (6 of the 115),
- inflow [m^3/s] -> a node volume input [m^3] (x 86400; the graph
  kernel divides by s_per_time).

Requires pywatershed test_data/starfit files; skips with a reason if
absent.
"""

import pathlib as pl
import sys
from typing import Any

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from conftest import STARFIT_INDS_TEST
from discretization import Discretization
from flow_graph import make_flow_graph
from hydrology.starfit_flow_node import (
    StarfitFlowNode,
    cms_to_cfs,
)
from hydrology.starfit_source_sink_flow_node import (
    StarfitSourceSinkFlowNode,
)
from model import Model

MPIX_ROOT = pl.Path(__file__).parents[4]
STARFIT_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "starfit"
PARAM_FILE = STARFIT_DIR / "starfit_original_parameters.nc"
INFLOW_FILE = STARFIT_DIR / "lake_inflow.nc"
ANSWER_FILE = STARFIT_DIR / "starfit_mean_output_1995-2001.nc"

OUTPUT_VARS = ["lake_storage", "lake_release", "lake_spill"]
# pywatershed's own autotest tolerance for this comparison
RTOL = ATOL = 1.0e-7
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = [PARAM_FILE, INFLOW_FILE, ANSWER_FILE]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed starfit test data not found; missing: "
        + ", ".join(_missing)
    ),
)


@pytest.fixture(scope="module")
def graph_params():
    with xr.open_dataset(PARAM_FILE) as ds:
        sub = ds.isel(nreservoirs=STARFIT_INDS_TEST).load()
    # pywatershed node data-prep (see module docstring)
    obs = sub["Obs_MEANFLOW_CUMECS"].values
    wh_nan = np.isnan(obs)
    obs[wh_nan] = sub["inflow_mean"].values[wh_nan]
    param_names = [
        nn
        for nn, mm in StarfitFlowNode.fields.items()
        if mm.kind == "parameter"
    ]
    return xr.Dataset({nn: ("nnodes", sub[nn].values) for nn in param_names})


@pytest.fixture(scope="module")
def inflow_da():
    with xr.open_dataset(INFLOW_FILE) as ds:
        return ds["lake_inflow"].isel(grand_id=STARFIT_INDS_TEST).load()


@pytest.fixture(scope="module")
def answers():
    with xr.open_dataset(ANSWER_FILE) as ds:
        return ds.isel(grand_id=STARFIT_INDS_TEST).load()


# pywatershed's own autotest request for the combined node: a tiny
# constant sink, fully applied every substep (storage >> |request|)
SOURCE_SINK_REQUEST = -28.0e-17
# values are duck-typed node-type classes (see the make_flow_graph
# contract), hence Any
NODE_CLASSES: dict[str, Any] = {
    "starfit": StarfitFlowNode,
    "starfit_source_sink": StarfitSourceSinkFlowNode,
}
SCENARIOS = [f"{nn}-{uu}" for nn in NODE_CLASSES for uu in ("cms", "cfs")]


@pytest.fixture(scope="module", params=SCENARIOS)
def graph_run(request, graph_params, inflow_da, tmp_path_factory):
    class_key, units = request.param.rsplit("-", 1)
    node_class = NODE_CLASSES[class_key]
    io_in_cfs = units == "cfs"
    # the graph's flow-unit factor for io-side data (1.0 for cms)
    io_factor = cms_to_cfs if io_in_cfs else 1.0
    out_dir = tmp_path_factory.mktemp(f"{request.param}_flow_node")
    n_nodes = len(STARFIT_INDS_TEST)

    graph_class = make_flow_graph(
        (node_class,),
        class_name=f"{node_class.__name__}{units}Graph",
        n_substeps=1,
        io_in_cfs=io_in_cfs,
    )

    # every reservoir is an isolated outlet
    discretizations = {
        "nnodes": Discretization(
            ["nnodes"],
            parameters=xr.Dataset(
                {
                    "to_graph_index": (
                        "nnodes",
                        np.full(n_nodes, -1, dtype=np.int64),
                    ),
                    "node_type": (
                        "nnodes",
                        np.full(
                            n_nodes,
                            graph_class.node_type_code(node_class.type_name),
                            dtype=np.int64,
                        ),
                    ),
                }
            ),
            topo_order={"node_order": "to_graph_index"},
            topo_one_based=False,
        ),
    }

    # inflow [m^3/s] as a node VOLUME input (see module docstring);
    # the other two lateral volumes are zero
    time_vals = inflow_da["time"].values

    def vol_input(name, vals):
        return xr.DataArray(
            vals,
            dims=("time", "nnodes"),
            coords={"time": time_vals},
            name=name,
        )

    # io-side data arrive in the graph's flow units (the pywatershed
    # autotest conversions): initial_storage x cm_to_cf and inflows x
    # cms_to_cfs when cfs (io_factor = 1.0 when cms)
    params = graph_params.copy()  # structure only; arrays shared
    params["initial_storage"] = (
        "nnodes",
        graph_params["initial_storage"].values * io_factor,
    )
    extra_inputs = {}
    if node_class is StarfitSourceSinkFlowNode:
        # the pywatershed autotest setup for the combined node (its
        # request values are the same numbers in either unit system)
        params["source_sink_storage_min"] = (
            "nnodes",
            np.zeros(n_nodes),
        )
        extra_inputs["node_source_sink"] = vol_input(
            "node_source_sink",
            np.full(inflow_da.shape, SOURCE_SINK_REQUEST),
        )

    process_dict = {
        "flow_graph": {
            "class": graph_class,
            "discretization": "nnodes",
            "parameters": params,
            "node_sroff_vol": vol_input(
                "node_sroff_vol",
                inflow_da.values * io_factor * float(S_PER_TIME),
            ),
            "node_ssres_flow_vol": vol_input(
                "node_ssres_flow_vol",
                np.zeros(inflow_da.shape),
            ),
            "node_gwres_flow_vol": vol_input(
                "node_gwres_flow_vol",
                np.zeros(inflow_da.shape),
            ),
            **extra_inputs,
        },
    }
    control = {
        "output_var_names": OUTPUT_VARS,
        "output_serial_zarr": out_dir / "starfit_graph.zarr",
        "time_chunk_size": 61,
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    return {
        "model": model,
        "control": control,
        "class": node_class,
        "io_factor": io_factor,
    }


class TestStarfitFlowNode:
    def test_time_means_match_starfit(self, graph_run, answers):
        """Time-means of storage/release/spill vs the offline STARFIT
        reference means (the pywatershed autotest check); the
        reference is cms-native, so expected values scale by the
        graph's io_factor (x cms_to_cfs when cfs, "same for storage"
        -- the autotest's own conversion)."""
        output_ds = xr.open_zarr(
            graph_run["control"]["output_serial_zarr"],
            consolidated=False,
        )
        for var in OUTPUT_VARS:
            np.testing.assert_allclose(
                output_ds[var].values.mean(axis=0),
                answers[f"{var}_mean"].values * graph_run["io_factor"],
                rtol=RTOL,
                atol=ATOL,
                err_msg=var,
            )

    def test_outflow_is_release_plus_spill(self, graph_run):
        """Structural: the graph outflow harvest equals release +
        spill (final in-memory state)."""
        proc = graph_run["model"].model_dict["flow_graph"]
        np.testing.assert_allclose(
            proc["node_outflows"].values,
            proc["lake_release"].values + proc["lake_spill"].values,
            rtol=1e-15,
        )

    def test_applied_diversion(self, graph_run):
        """The applied sink is the request wherever storage exceeds
        the minimum, and LIMITED (to zero here) where a reservoir sits
        at the minimum -- one reservoir drains to empty storage on the
        final day, exercising the min-storage branch. Also: the
        harvest into the graph's node_sink_source (final in-memory
        state)."""
        if graph_run["class"] is not StarfitSourceSinkFlowNode:
            pytest.skip("plain starfit has no diversion")
        proc = graph_run["model"].model_dict["flow_graph"]
        applied = proc["lake_sink_source"].values
        # never positive, never more sink than requested (ulp slack)
        assert (applied <= 0.0).all()
        assert (applied >= SOURCE_SINK_REQUEST * (1.0 + 1.0e-9)).all()
        # limited exactly where the reservoir sat at the minimum
        # storage (= 0 here) when the diversion was decided (the
        # start-of-substep storage); fully applied everywhere else
        at_min = proc["lake_storage_old_sub"].values == 0.0
        np.testing.assert_array_equal(applied[at_min], 0.0)
        np.testing.assert_allclose(
            applied[~at_min], SOURCE_SINK_REQUEST, rtol=1e-12
        )
        assert at_min.sum() < 5  # the drained case is the exception
        np.testing.assert_array_equal(
            proc["node_sink_source"].values,
            proc["lake_sink_source"].values,
        )
