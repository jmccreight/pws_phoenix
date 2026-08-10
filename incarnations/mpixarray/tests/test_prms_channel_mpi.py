"""MPI submodel regression: the FULL ported model chain
(PRMSAtmosphere + PRMSCanopy + PRMSSnow + PRMSRunoff + PRMSSoilzone +
PRMSGroundwater -> MapMPI x3 -> PRMSChannel) vs pywatershed answers
(drb_2yr).

THE target shape, the whole model from the raw CBH files: the hru
grid is distributed (mpixarray streaming; atmosphere -> canopy ->
snow -> runoff -> soilzone -> groundwater in NHM order, ALL
inter-process fluxes by structural sharing on each rank's local
block -- including canopy's in-place edit of atmosphere's pptmix,
soilzone's dunnian edit of runoff's sroff/sroff_vol MUTABLE inputs,
and the prior-step back-edges: snow's pack state to canopy,
soilzone's soil state to runoff), the segment grid is REPLICATED on
every rank (muskingum is sequential in segment order), and the three
lateral-inflow volumes cross the parallel boundary via MapMPI (local
weight-column partial matmul + Allreduce). The solar tables come from
the LIVE compute_soltabs factory (rank 0, into the combined input
file). The ONLY time-varying inputs in the file are prcp/tmax/tmin.
Output is routed by owning grid: gwres_flow streams to parallel
NetCDF; the six channel variables are collected by the rank-0 zarr
Output.

Every rank builds the segment side identically (same files, same
topo_order) -- that IS the replication; the final-state test asserting
on EVERY rank doubles as a replication-consistency check.

All collective MPI ops live in module-scoped fixtures; test methods are
pure asserts (see test_up_low_regression_mpi.py for the pattern).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_channel_mpi.py -v

Criterion: the snow_live mode of test_prms_channel.py -- (1e-2, 1e-2)
with an outlier fraction, the fastmath-answers ceiling for the full
chain (see there; the strict plumbing canary is that test's snow_disk
mode, and per-process precision lives in the standalone tests).
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
from atmosphere.prms_atmosphere import PRMSAtmosphere
from atmosphere.prms_solar_geometry import compute_soltabs
from discretization import Discretization
from hydrology.prms_canopy import PRMSCanopy
from hydrology.prms_channel import PRMSChannel
from hydrology.prms_groundwater import PRMSGroundwater
from hydrology.prms_runoff import PRMSRunoff
from hydrology.prms_snow import PRMSSnow
from hydrology.prms_soilzone import PRMSSoilzone
from map import MapMPI
from model import ModelMPI

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

# the ONLY time-varying inputs: the raw CBH files
CBH_NAMES = ("prcp", "tmax", "tmin")
STREAMED_NAME = "gwres_flow"  # the one to_netcdf var (mpixarray limit)
ANSWER_NAMES = (
    "seg_lateral_inflow",
    "seg_upstream_inflow",
    "seg_inflow",
    "seg_outflow",
    "seg_stor_change",
    "channel_outflow_vol",
)
# the snow_live criterion (see test_prms_channel.py): the
# fastmath-answers ceiling for the full chain
RTOL = ATOL = 1.0e-2
OUTLIER_FRACTION = 1.0e-3
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = (
    [
        DOMAIN_DIR / "parameters_PRMSAtmosphere.nc",
        DOMAIN_DIR / "parameters_PRMSCanopy.nc",
        DOMAIN_DIR / "parameters_PRMSSnow.nc",
        DOMAIN_DIR / "parameters_PRMSRunoff.nc",
        DOMAIN_DIR / "parameters_PRMSSoilzone.nc",
        DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
        DOMAIN_DIR / "parameters_PRMSChannel.nc",
        DOMAIN_DIR / "parameters_dis_hru.nc",
        DOMAIN_DIR / "parameters_dis_seg.nc",
    ]
    + [DOMAIN_DIR / f"{nn}.nc" for nn in CBH_NAMES]
    + [GEN_DIR / f"{nn}.nc" for nn in ANSWER_NAMES + (STREAMED_NAME,)]
)
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def mpi_paths():
    """Rank 0 assembles the ONE combined hru input file (runoff +
    soilzone params + gw params + dis_hru vars + IC + the 13 disk
    forcings, renamed nhm_id -> nhru) and broadcasts the temp dir;
    rank 0 cleans up (no barrier in teardown)."""
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
        gw_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSGroundwater.nc")
        runoff_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSRunoff.nc")
        soilzone_params = xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSSoilzone.nc"
        )
        canopy_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSCanopy.nc")
        snow_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSnow.nc")
        atmos_params = xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSAtmosphere.nc"
        )
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        # the LIVE solar-geometry factory supplies the soltab tables
        soltabs = compute_soltabs(dis_hru)
        forcings = [
            xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
            .rename({"nhm_id": "nhru"})
            .astype(np.float64)
            for nn in CBH_NAMES
        ]
        combined = xr.merge(
            [
                # the declared process params; overlaps across files
                # (dprst_frac, hru_percent_imperv, soil_moist_max,
                # cov_type, covden_*, potet_sublim, tmax_allsnow,
                # tmax_allrain_offset) merge -- identical NHM values
                atmos_params,
                soltabs[["soltab_potsw", "soltab_horad_potsw"]],
                canopy_params,
                snow_params,
                runoff_params,
                soilzone_params,
                gw_params[["gwflow_coef", "gwsink_coef", "gwstor_init"]],
                dis_hru[
                    [
                        "hru_type",
                        "hru_area",
                        "hru_in_to_cf",
                        "hru_slope",
                        "hru_lat",
                    ]
                ],
                *forcings,
            ],
            compat="no_conflicts",
        )
        combined = combined.assign_coords(nhru=np.arange(combined.sizes["nhru"]))
        combined.to_netcdf(paths["input_file"])
    comm.Barrier()
    yield paths

    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def weights():
    """0/1 aggregation weights from hru_segment -- built identically on
    every rank (deterministic file read, no collectives)."""
    channel_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")
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
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in names}


@pytest.fixture(scope="module")
def mpi_run(mpi_paths, weights):
    """Build + run + finalize ModelMPI ONCE; every collective lives here."""
    comm = MPI.COMM_WORLD
    channel_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")
    # dict order = schedule: NHM order (atmosphere -> canopy -> snow
    # -> runoff -> soilzone -> gw); see the serial test for
    # shared-buffer semantics
    process_dict = {
        "prms_atmosphere": {
            "class": PRMSAtmosphere,
            "discretization": "nhru",
        },
        "prms_canopy": {
            "class": PRMSCanopy,
            "discretization": "nhru",
        },
        "prms_snow": {
            "class": PRMSSnow,
            "discretization": "nhru",
        },
        "prms_runoff": {
            "class": PRMSRunoff,
            "discretization": "nhru",
        },
        "prms_soilzone": {
            "class": PRMSSoilzone,
            "discretization": "nhru",
        },
        "prms_groundwater": {
            "class": PRMSGroundwater,
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
        # TODO: JLM: would be nice to combine these maps if the variable could
        # be a dict[dict] ( do the individual maps even need keys)?
        # TODO: JLM: how to encode maps on file? just weights?
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
    # TODO: JLM: it's a bit odd how the other discretization lives
    # in the input file but the processes are both defined in the
    # process_dict.
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
            nn: channel[nn].values.copy() for nn in ("seg_outflow", "seg_inflow")
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
            bad = ~np.isclose(
                output_ds[nn].values,
                answers[nn].values,
                rtol=RTOL,
                atol=ATOL,
            )
            frac = bad.mean()
            assert frac <= OUTLIER_FRACTION, (
                f"variable '{nn}': {frac:.3%} of segment-days outside "
                f"tolerance (allowed {OUTLIER_FRACTION:.3%})"
            )

    # -- streamed distributed var (parallel NetCDF, rank 0) --
    def test_streamed_gwres_flow(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_file"]) as ds_out:
            flow_out = ds_out[f"{STREAMED_NAME}_out"].values
        bad = ~np.isclose(
            flow_out, answers[STREAMED_NAME].values, rtol=RTOL, atol=ATOL
        )
        assert bad.mean() <= OUTLIER_FRACTION

    # -- replicated final state, asserted on EVERY rank --
    def test_final_state_every_rank(self, mpi_run, answers):
        for nn, vals in mpi_run["final"].items():
            bad = ~np.isclose(
                vals, answers[nn].values[-1, :], rtol=RTOL, atol=ATOL
            )
            assert bad.mean() <= 0.01, f"'{nn}' final state differs"

    # -- structural: decomposed maps + frozen derived params --
    def test_map_decomposition_and_derived(self, mpi_run):
        assert mpi_run["derived_frozen"]
        for n_cols in mpi_run["map_local_cols"]:
            assert n_cols == mpi_run["local_n"]
