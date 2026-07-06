"""MPI submodel regression: PRMSGroundwater -> MapMPI x3 -> PRMSChannel
vs pywatershed answers (drb_2yr).

THE Stage-2 target shape with real physics: the hru grid is distributed
(mpixarray streaming; groundwater + the flux carrier), the segment grid
is REPLICATED on every rank (muskingum is sequential in segment order),
and the three lateral-inflow volumes cross the parallel boundary via
MapMPI (local weight-column partial matmul + Allreduce). Output is
routed by owning grid: gwres_flow streams to parallel NetCDF; the six
channel variables are collected by the rank-0 zarr Output.

Every rank builds the segment side identically (same files, same
topo_order) -- that IS the replication; the final-state test asserting
on EVERY rank doubles as a replication-consistency check.

All collective MPI ops live in module-scoped fixtures; test methods are
pure asserts (see test_up_low_regression_mpi.py for the pattern).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_channel_mpi.py -v

Tolerances mirror test_prms_channel.py: pywatershed's 1e-13, except the
cancellation-amplified seg_stor_change (see PER_VAR_TOL there; the
small carrier class and tolerance dict are duplicated here to keep the
MPI test self-contained).
"""

import pathlib as pl
import shutil
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI

sys.path.insert(0, str(pl.Path(__file__).parent.parent))
from discretization import Discretization
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import PRMSGroundwater
from map import MapMPI
from model import ModelMPI
from process import DataArrayMeta, Process

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

GW_INPUT_NAMES = ("soil_to_gw", "ssr_to_gw", "dprst_seep_hru")
CARRIER_INPUT_NAMES = ("sroff_vol", "ssres_flow_vol")
STREAMED_NAME = "gwres_flow"  # the one to_netcdf var (mpixarray limit)
ANSWER_NAMES = (
    "seg_lateral_inflow",
    "seg_upstream_inflow",
    "seg_inflow",
    "seg_outflow",
    "seg_stor_change",
    "channel_outflow_vol",
)
RTOL = ATOL = 1.0e-13
# duplicated from test_prms_channel.py (cancellation-amplified diff)
PER_VAR_TOL = {"seg_stor_change": (1.0e-7, 1.0e-4)}  # (rtol, atol)
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = [
    DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
    DOMAIN_DIR / "parameters_PRMSChannel.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in GW_INPUT_NAMES
    + CARRIER_INPUT_NAMES
    + ANSWER_NAMES
    + (STREAMED_NAME,)
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


class HruChannelFluxes(Process):
    """Carrier for not-yet-ported producers (PRMSRunoff/PRMSSoilzone) --
    duplicated from test_prms_channel.py. Distributed here: its inputs
    stream from the combined input file like any file-backed input."""

    sroff_vol = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Surface runoff volume [cf] (from disk)",
    )
    ssres_flow_vol = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="Interflow volume [cf] (from disk)",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt: np.float64, time) -> None:
        pass


@pytest.fixture(scope="module")
def mpi_paths():
    """Rank 0 assembles the ONE combined hru input file (gw params +
    dis_hru vars + IC + all five forcings, renamed nhm_id -> nhru) and
    broadcasts the temp dir; rank 0 cleans up (no barrier in teardown)."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "prms_channel_mpi_data"
    paths = {
        "input_file": data_dir / "model_input.nc",
        "output_file": data_dir / "model_output.nc",
        "output_zarr": data_dir / "channel_output.zarr",
    }
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        gw_params = xr.open_dataset(
            DOMAIN_DIR / "parameters_PRMSGroundwater.nc"
        )
        dis_hru = xr.open_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        forcings = [
            xr.open_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
            for nn in GW_INPUT_NAMES + CARRIER_INPUT_NAMES
        ]
        combined = xr.merge(
            [
                gw_params[["gwflow_coef", "gwsink_coef", "gwstor_init"]],
                dis_hru[["hru_area", "hru_in_to_cf"]],
                *forcings,
            ],
            compat="no_conflicts",
        )
        combined = combined.assign_coords(
            nhru=np.arange(combined.sizes["nhru"])
        )
        combined.to_netcdf(paths["input_file"])
    comm.Barrier()
    yield paths

    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def weights():
    """0/1 aggregation weights from hru_segment -- built identically on
    every rank (deterministic file read, no collectives)."""
    channel_params = xr.open_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")
    hru_segment = channel_params["hru_segment"].values
    n_seg = channel_params.sizes["nsegment"]
    ww = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            ww[hru_segment[ihru] - 1, ihru] = 1.0
    return ww


@pytest.fixture(scope="module")
def answers():
    names = ANSWER_NAMES + (STREAMED_NAME,)
    return {nn: xr.open_dataarray(GEN_DIR / f"{nn}.nc") for nn in names}


@pytest.fixture(scope="module")
def mpi_run(mpi_paths, weights):
    """Build + run + finalize ModelMPI ONCE; every collective lives here."""
    comm = MPI.COMM_WORLD
    channel_params = xr.open_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")
    process_dict = {
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
        },
        "hru_channel_fluxes": {
            "class": HruChannelFluxes,
            "discretization": "nhru",
        },
        "prms_channel": {
            "class": PRMSChannel,
            "discretization": "nsegment",
            "parameters": channel_params,
            "segment_flow_init": channel_params["segment_flow_init"],
        },
    }
    maps = {
        "sroff": MapMPI(
            weights=weights,
            grid={"nhru": "nsegment"},
            variable={"sroff_vol": "seg_sroff_vol"},
        ),
        "ssres": MapMPI(
            weights=weights,
            grid={"nhru": "nsegment"},
            variable={"ssres_flow_vol": "seg_ssres_flow_vol"},
        ),
        "gw": MapMPI(
            weights=weights,
            grid={"nhru": "nsegment"},
            variable={"gwres_flow_vol": "seg_gwres_flow_vol"},
        ),
    }
    discretizations = {
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }
    control = {
        "input_file": mpi_paths["input_file"],
        "output_parallel_netcdf": mpi_paths["output_file"],
        "output_var_names": [STREAMED_NAME] + list(ANSWER_NAMES),
        "output_serial_zarr": mpi_paths["output_zarr"],
        "time_chunk_size": 61,
        "mpi_grid": "nhru",
    }
    model = ModelMPI(
        process_dict,
        control,
        maps=maps,
        discretizations=discretizations,
    )
    model.run(S_PER_TIME)

    channel = model.model_dict["prms_channel"]
    local_n = int(model._ds_mpi_stream.sizes["nhru"])
    result = {
        "output_file": mpi_paths["output_file"],
        "output_zarr": mpi_paths["output_zarr"],
        # replicated serial grid: capture final state directly (no
        # gather); asserting on EVERY rank checks replication too
        "final": {
            nn: channel[nn].values.copy()
            for nn in ("seg_outflow", "seg_inflow")
        },
        # each MapMPI holds exactly this rank's weight columns
        "map_local_cols": [mm._weights_local.shape[1] for mm in maps.values()],
        "local_n": local_n,
        "derived_frozen": not channel["c0"].values.flags.writeable,
    }
    model.finalize()
    comm.Barrier()  # output files fully flushed before reads
    return result


@pytest.mark.mpi(min_size=2)
class TestPRMSChannelSubmodelMPI:
    # -- channel outputs over ALL timesteps (rank-0 zarr Output) --
    def test_channel_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        output_ds = xr.open_zarr(mpi_run["output_zarr"], consolidated=False)
        for nn in ANSWER_NAMES:
            rtol_var, atol_var = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            np.testing.assert_allclose(
                output_ds[nn].values,
                answers[nn].values,
                rtol=rtol_var,
                atol=atol_var,
                err_msg=f"variable '{nn}' differs from pywatershed",
            )

    # -- streamed distributed var (parallel NetCDF, rank 0) --
    def test_streamed_gwres_flow(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.open_dataset(mpi_run["output_file"]) as ds_out:
            flow_out = ds_out[f"{STREAMED_NAME}_out"].values
        np.testing.assert_allclose(
            flow_out, answers[STREAMED_NAME].values, rtol=RTOL, atol=ATOL
        )

    # -- replicated final state, asserted on EVERY rank --
    def test_final_state_every_rank(self, mpi_run, answers):
        for nn, vals in mpi_run["final"].items():
            np.testing.assert_allclose(
                vals,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' final state differs",
            )

    # -- structural: decomposed maps + frozen derived params --
    def test_map_decomposition_and_derived(self, mpi_run):
        assert mpi_run["derived_frozen"]
        for n_cols in mpi_run["map_local_cols"]:
            assert n_cols == mpi_run["local_n"]
