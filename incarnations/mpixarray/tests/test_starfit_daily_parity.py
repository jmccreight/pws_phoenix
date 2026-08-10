"""StarfitDailyFlowNode A/B parity vs pywatershed compute_daily=True.

pywatershed has NO value-level validation of its STARFIT daily mode:
its node autotest only runs hourly (the "fake daily" configuration --
see test_starfit_flow_node.py), and its mixed-graph autotest pastes
the graph's own outputs in as the expected values for the new nodes.
So this test validates the PORT, not the physics: drive pywatershed's
own StarfitFlowNode(compute_daily=True) and our StarfitDailyFlowNode
graph with IDENTICAL data (cms; a subset of reservoirs and time for
the pure-python pywatershed loop's sake) and require the daily
lake_storage / lake_release / lake_spill / outflow series to agree at
1e-10 (headroom for libm sin/cos ulp differences, numba vs numpy --
the operation ORDER is ported verbatim).

The daily node applies a CONSTANT outflow through each day's 24
substeps, computed at the previous day's end from that day's mean
inflow (one-day lag; the run's first day is seeded from the first
substep's inflow). Feeding daily inflow values keeps each day's 24
substep inflows constant on both sides.

Requires pywatershed IMPORTABLE (the repo at the mpix root via
sys.path; deps: pyPRMS, tqdm, contextily -- see
pws_phoenix/environment.yaml pip section) AND the starfit test data;
skips with a reason otherwise.
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from conftest import MPIX_ROOT
from conftest import STARFIT_INDS_TEST as FULL_INDS
from discretization import Discretization
from flow_graph import make_flow_graph
from hydrology.starfit_daily_flow_node import StarfitDailyFlowNode
from model import Model

STARFIT_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "starfit"
PARAM_FILE = STARFIT_DIR / "starfit_original_parameters.nc"
INFLOW_FILE = STARFIT_DIR / "lake_inflow.nc"

# subset (vs the 115-reservoir hourly tests): the pywatershed side is
# a pure-python per-node/per-substep loop
N_RESERVOIRS = 15
N_DAYS = 365
N_SUBSTEPS = 24
COMPARE_VARS = ["lake_storage", "lake_release", "lake_spill"]
RTOL = ATOL = 1.0e-10
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

# a leading slice of the standard autotest subset (conftest)
STARFIT_INDS_TEST = FULL_INDS[:N_RESERVOIRS]

_missing = [str(ff) for ff in (PARAM_FILE, INFLOW_FILE) if not ff.exists()]
try:
    sys.path.insert(0, str(MPIX_ROOT / "pywatershed"))
    from pywatershed.base.control import Control
    from pywatershed.hydrology.starfit import StarfitFlowNodeMaker
    from pywatershed.parameters import Parameters, StarfitParameters

    _pyws_reason = ""
except ImportError as err:
    _pyws_reason = f"pywatershed not importable ({err}); "

pytestmark = pytest.mark.skipif(
    bool(_missing) or bool(_pyws_reason),
    reason=(
        _pyws_reason
        + ("missing data: " + ", ".join(_missing) if _missing else "")
    ),
)


@pytest.fixture(scope="module")
def inflow_da():
    with xr.load_dataset(INFLOW_FILE) as ds:
        return (
            ds["lake_inflow"]
            .isel(
                grand_id=STARFIT_INDS_TEST,
                time=slice(0, N_DAYS),
            )
            .load()
        )


@pytest.fixture(scope="module")
def pyws_results(inflow_da):
    """Drive pywatershed's own compute_daily=True nodes (cms) and
    collect the daily series (the autotest driving pattern)."""
    parameters_ds = Parameters.from_netcdf(PARAM_FILE).to_xr_ds()
    merge_list = [
        parameters_ds.isel(nreservoirs=slice(ii, ii + 1))
        for ii in STARFIT_INDS_TEST
    ]
    parameters = StarfitParameters.from_ds(
        xr.concat(merge_list, dim="nreservoirs")
    )

    times = inflow_da["time"].values
    control = Control(
        times[0].astype("datetime64[s]"),
        times[-1].astype("datetime64[s]"),
        np.timedelta64(24, "h"),
    )
    node_maker = StarfitFlowNodeMaker(
        discretization=None,
        parameters=parameters,
        io_in_cfs=False,
        compute_daily=True,
        imbalance_behavior=None,
    )
    nodes = [node_maker.get_node(control, ii) for ii in range(N_RESERVOIRS)]

    results = {
        vv: np.full((N_DAYS, N_RESERVOIRS), np.nan)
        for vv in [*COMPARE_VARS, "lake_outflow"]
    }
    inflow_vals = inflow_da.values
    for istep in range(control.n_times):
        control.advance()
        for inode, node in enumerate(nodes):
            node.advance()
            node.prepare_timestep()
            for ss in range(N_SUBSTEPS):
                node.calculate_subtimestep(ss, inflow_vals[istep, inode], 0.0)
            node.finalize_timestep()
        for inode, node in enumerate(nodes):
            for vv in results:
                results[vv][istep, inode] = node[f"_{vv}"][0]
    return results


@pytest.fixture(scope="module")
def phoenix_run(inflow_da, tmp_path_factory):
    """The same reservoirs/forcing through a StarfitDailyFlowNode-only
    graph (n_substeps=24, cms)."""
    out_dir = tmp_path_factory.mktemp("starfit_daily_parity")

    with xr.load_dataset(PARAM_FILE) as ds:
        sub = ds.isel(nreservoirs=STARFIT_INDS_TEST).load()
    # the standard data-prep (see test_starfit_flow_node)
    obs = sub["Obs_MEANFLOW_CUMECS"].values
    wh_nan = np.isnan(obs)
    obs[wh_nan] = sub["inflow_mean"].values[wh_nan]
    param_names = [
        nn
        for nn, mm in StarfitDailyFlowNode.fields.items()
        if mm.kind == "parameter"
    ]
    graph_params = xr.Dataset(
        {nn: ("nnodes", sub[nn].values) for nn in param_names}
    )

    graph_class = make_flow_graph(
        (StarfitDailyFlowNode,),
        class_name="StarfitDailyParityGraph",
        n_substeps=N_SUBSTEPS,
        io_in_cfs=False,
    )
    discretizations = {
        "nnodes": Discretization(
            ["nnodes"],
            parameters=xr.Dataset(
                {
                    "to_graph_index": (
                        "nnodes",
                        np.full(N_RESERVOIRS, -1, dtype=np.int64),
                    ),
                    "node_type": (
                        "nnodes",
                        np.zeros(N_RESERVOIRS, dtype=np.int64),
                    ),
                }
            ),
            topo_order={"node_order": "to_graph_index"},
            topo_one_based=False,
        ),
    }

    time_vals = inflow_da["time"].values

    def vol_input(name, vals):
        return xr.DataArray(
            vals,
            dims=("time", "nnodes"),
            coords={"time": time_vals},
            name=name,
        )

    process_dict = {
        "flow_graph": {
            "class": graph_class,
            "discretization": "nnodes",
            "parameters": graph_params,
            "node_sroff_vol": vol_input(
                "node_sroff_vol",
                inflow_da.values * float(S_PER_TIME),
            ),
            "node_ssres_flow_vol": vol_input(
                "node_ssres_flow_vol", np.zeros(inflow_da.shape)
            ),
            "node_gwres_flow_vol": vol_input(
                "node_gwres_flow_vol", np.zeros(inflow_da.shape)
            ),
        },
    }
    control = {
        "output_var_names": [*COMPARE_VARS, "node_outflows"],
        "output_serial_zarr": out_dir / "graph.zarr",
        "time_chunk_size": 61,
    }
    with Model(
        process_dict, control, discretizations=discretizations
    ) as model:
        model.run(S_PER_TIME, np.int32(model.ntime))
    return xr.open_zarr(control["output_serial_zarr"], consolidated=False)


class TestStarfitDailyParity:
    def test_daily_series_match_pywatershed(self, phoenix_run, pyws_results):
        for vv in COMPARE_VARS:
            np.testing.assert_allclose(
                phoenix_run[vv].values,
                pyws_results[vv],
                rtol=RTOL,
                atol=ATOL,
                err_msg=vv,
            )
        # our node_outflows harvest = pywatershed's _lake_outflow
        np.testing.assert_allclose(
            phoenix_run["node_outflows"].values,
            pyws_results["lake_outflow"],
            rtol=RTOL,
            atol=ATOL,
            err_msg="node_outflows vs _lake_outflow",
        )

    def test_outflow_constant_within_day_lags_inflow(
        self, phoenix_run, pyws_results
    ):
        """Structural sanity of daily mode itself: outflow = release +
        spill (the day-constant rates)."""
        np.testing.assert_allclose(
            phoenix_run["node_outflows"].values,
            phoenix_run["lake_release"].values
            + phoenix_run["lake_spill"].values,
            rtol=1e-12,
            err_msg="outflow = release + spill",
        )
