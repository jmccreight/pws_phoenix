"""MPI regression: the FULL chain through stream temperature.

The stage-4 serial model (test_prms_stream_temp_full_chain.py) in THE
target MPI shape (test_prms_channel_mpi.py + stream temp): the hru
grid distributed (atmosphere -> canopy -> snow -> runoff -> soilzone
-> groundwater, plus the humidity_hru carrier fed from the input
file), the segment grid REPLICATED on every rank running
PRMSChannel -> PRMSHydraulicGeometryWidthOnly -> PRMSStreamTemp (all
segment-side couplings by structural sharing), and THIRTEEN MapMPI
crossing the parallel boundary: the three lateral-inflow volumes +
the ten aggregation Maps (kernel-probed weights, derived identically
on every rank -- deterministic file reads, no collectives).

Criterion: the stage-4 serial standard -- stream-temp family 5e-3
with the measured outlier allowance (the two pywatershed answer
generations differ from each other via snow knife-edge flips; see the
serial test's module docstring). MapMPI's partial-sum + Allreduce
reorders float accumulation vs the serial matmul at ~1e-13 --
irrelevant at 5e-3. Replication is checked directly: every rank's
final segment state must be BIT-IDENTICAL to rank 0's.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_stream_temp_full_chain_mpi.py -v
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
from hydrology.prms_hydraulic_geometry import PRMSHydraulicGeometryWidthOnly
from hydrology.prms_runoff import PRMSRunoff
from hydrology.prms_snow import PRMSSnow
from hydrology.prms_soilzone import PRMSSoilzone
from hydrology.prms_stream_temp import (
    AGGREGATION_MAP_SPEC,
    PRMSStreamTemp,
    derive_aggregation_weights,
)
from map import MapMPI
from model import ModelMPI
from process import DataArrayMeta, Process

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR_ST = DOMAIN_DIR / "output_stream_temp"

CBH_NAMES = ("prcp", "tmax", "tmin")
STREAMED_NAME = "gwres_flow"  # the one to_netcdf var (mpixarray limit)
ST_ANSWER_NAMES = (
    "seg_tave_water",
    "seg_tave_upstream",
    "seg_tave_gw",
    "seg_tave_ss",
    "seg_tave_lat",
    "seg_shade",
)
RTOL = ATOL = 5.0e-3
MAX_OUTLIER_FRAC = 5.0e-3  # the stage-4 serial standard
S_PER_TIME = np.float64(60.0 * 60.0 * 24.0)

_needed = (
    [
        DOMAIN_DIR / f"parameters_PRMS{nn}.nc"
        for nn in (
            "Atmosphere",
            "Canopy",
            "Snow",
            "Runoff",
            "Soilzone",
            "Groundwater",
            "Channel",
            "HydraulicGeometryWidthOnly",
            "StreamTemp",
            "StreamShadeDynamic",
        )
    ]
    + [
        DOMAIN_DIR / "parameters_dis_hru.nc",
        DOMAIN_DIR / "parameters_dis_seg.nc",
    ]
    + [DOMAIN_DIR / f"{nn}.nc" for nn in CBH_NAMES]
    + [GEN_DIR_ST / f"{nn}.nc" for nn in ST_ANSWER_NAMES]
    + [GEN_DIR_ST / "humidity_hru.nc"]
)
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr (incl. nhm_stream_temp) data not "
        "generated; missing: " + ", ".join(_missing[:3])
    ),
)


class HumidityCarrier(Process):
    """The humidity CBH forcing on the hru grid (see the serial
    stage-4 test); under MPI it rides the combined input file."""

    humidity_hru = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        description="CBH relative humidity [percent]",
    )

    def advance(self) -> None:
        pass

    def calculate(self, dt: np.float64, time) -> None:
        pass


@pytest.fixture(scope="module")
def mpi_paths():
    """Rank 0 assembles the ONE combined hru input file (the chain
    test's contents + humidity_hru) and broadcasts the temp dir."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "prms_stream_temp_full_chain_mpi_data"
    paths = {
        "input_file": data_dir / "model_input.nc",
        "output_file": data_dir / "model_output.nc",
        "output_zarr": data_dir / "segment_output.zarr",
    }
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)

        def _params(name):
            return xr.load_dataset(DOMAIN_DIR / f"parameters_PRMS{name}.nc")

        gw_params = _params("Groundwater")
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        soltabs = compute_soltabs(dis_hru)
        forcings = [
            xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
            .rename({"nhm_id": "nhru"})
            .astype(np.float64)
            for nn in CBH_NAMES
        ] + [
            xr.load_dataarray(GEN_DIR_ST / "humidity_hru.nc").rename(
                {"nhm_id": "nhru"}
            )
        ]
        combined = xr.merge(
            [
                _params("Atmosphere"),
                soltabs[["soltab_potsw", "soltab_horad_potsw"]],
                _params("Canopy"),
                _params("Snow"),
                _params("Runoff"),
                _params("Soilzone"),
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
        combined = combined.assign_coords(
            nhru=np.arange(combined.sizes["nhru"])
        )
        combined.to_netcdf(paths["input_file"])
    comm.Barrier()
    yield paths

    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def answers():
    return {
        nn: xr.load_dataarray(GEN_DIR_ST / f"{nn}.nc")
        for nn in ST_ANSWER_NAMES
    }


@pytest.fixture(scope="module")
def mpi_run(mpi_paths):
    """Build + run + finalize ModelMPI ONCE; every collective lives
    here."""
    comm = MPI.COMM_WORLD
    channel_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSChannel.nc")
    dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    st = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSStreamTemp.nc")
    st_params = xr.merge(
        [
            st,
            xr.load_dataset(
                DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
            ),
            dis_seg[["lat_temp_adj"]],
            dis_hru[["hru_area"]],
        ],
        compat="no_conflicts",
    )

    discretizations = {
        "nsegment": Discretization(
            ["nsegment"],
            parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
            topo_order={"segment_order": "tosegment"},
        ),
    }

    # weight matrices, built identically on every rank (deterministic
    # file reads + the derive probes; no collectives)
    hru_segment = channel_params["hru_segment"].values
    n_seg = channel_params.sizes["nsegment"]
    vol_weights = np.zeros((n_seg, hru_segment.shape[0]))
    for ihru in range(hru_segment.shape[0]):
        if hru_segment[ihru] > 0:
            vol_weights[hru_segment[ihru] - 1, ihru] = 1.0
    dis_topo = Discretization(
        ["nsegment"],
        parameters=DOMAIN_DIR / "parameters_dis_seg.nc",
        topo_order={"segment_order": "tosegment"},
    ).parameters
    assert dis_topo is not None
    agg_weights = derive_aggregation_weights(
        st["hru_segment"].values.astype(np.int64),
        dis_hru["hru_area"].values,
        dis_seg["tosegment"].values.astype(np.int64),
        dis_topo["segment_order"].values.astype(np.int64),
        st["seg_close"].values,
    )
    maps = {
        "sroff_vol": MapMPI(
            weights=vol_weights,
            grid={"nhru": "nsegment"},
            variable={"sroff_vol": "seg_sroff_vol"},
        ),
        "ssres_vol": MapMPI(
            weights=vol_weights,
            grid={"nhru": "nsegment"},
            variable={"ssres_flow_vol": "seg_ssres_flow_vol"},
        ),
        "gw_vol": MapMPI(
            weights=vol_weights,
            grid={"nhru": "nsegment"},
            variable={"gwres_flow_vol": "seg_gwres_flow_vol"},
        ),
        **{
            target: MapMPI(
                weights=agg_weights[wkey],
                grid={"nhru": "nsegment"},
                variable={source: target},
            )
            for target, (source, wkey) in AGGREGATION_MAP_SPEC.items()
        },
    }

    process_dict = {
        "prms_atmosphere": {
            "class": PRMSAtmosphere,
            "discretization": "nhru",
        },
        "prms_canopy": {"class": PRMSCanopy, "discretization": "nhru"},
        "prms_snow": {"class": PRMSSnow, "discretization": "nhru"},
        "prms_runoff": {"class": PRMSRunoff, "discretization": "nhru"},
        "prms_soilzone": {
            "class": PRMSSoilzone,
            "discretization": "nhru",
        },
        "prms_groundwater": {
            "class": PRMSGroundwater,
            "discretization": "nhru",
        },
        "humidity_carrier": {
            "class": HumidityCarrier,
            "discretization": "nhru",
        },
        "prms_channel": {
            "class": PRMSChannel,
            "discretization": "nsegment",
            "parameters": channel_params,
            "segment_flow_init": channel_params["segment_flow_init"],
        },
        "prms_hydraulic_geometry": {
            "class": PRMSHydraulicGeometryWidthOnly,
            "discretization": "nsegment",
            "parameters": DOMAIN_DIR
            / "parameters_PRMSHydraulicGeometryWidthOnly.nc",
        },
        "prms_stream_temp": {
            "class": PRMSStreamTemp,
            "discretization": "nsegment",
            "parameters": st_params,
        },
    }
    control = {
        "input_file": mpi_paths["input_file"],
        "output_parallel_netcdf": mpi_paths["output_file"],
        "output_var_names": [STREAMED_NAME] + list(ST_ANSWER_NAMES),
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

    stream_temp = model.model_dict["prms_stream_temp"]
    final = {
        nn: stream_temp[nn].values.copy()
        for nn in ("seg_tave_water", "seg_tave_gw", "seg_shade")
    }
    # replication check data: rank 0's final state, on every rank
    final_root = comm.bcast(final, root=0)
    model.finalize()
    comm.Barrier()  # output files fully flushed before reads
    return {
        "output_zarr": mpi_paths["output_zarr"],
        "final": final,
        "final_root": final_root,
    }


@pytest.mark.mpi(min_size=2)
class TestPRMSStreamTempFullChainMPI:
    # -- stream-temp outputs over ALL timesteps (rank-0 zarr Output) --
    def test_stream_temp_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        output_ds = xr.open_zarr(mpi_run["output_zarr"], consolidated=False)
        for nn in ST_ANSWER_NAMES:
            actual = output_ds[nn].values
            desired = answers[nn].values
            finite = np.isfinite(actual)
            if nn == "seg_tave_water":
                nan_cols = np.isnan(actual).all(axis=0)
                assert (np.isnan(actual) == nan_cols[None, :]).all()
            bad = ~np.isclose(
                actual[finite], desired[finite], rtol=RTOL, atol=ATOL
            )
            frac = bad.mean()
            assert frac <= MAX_OUTLIER_FRAC, (
                f"variable '{nn}': {frac:.3%} of finite seg-days "
                f"outside {RTOL} (allowed {MAX_OUTLIER_FRAC:.3%})"
            )

    # -- replication: every rank's final segment state == rank 0's --
    def test_replication_bit_identical(self, mpi_run):
        for nn, vals in mpi_run["final"].items():
            assert np.array_equal(
                vals, mpi_run["final_root"][nn], equal_nan=True
            ), f"'{nn}' final state differs from rank 0 (replication)"
