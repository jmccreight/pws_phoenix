"""MPI regression: the ported PRMSAtmosphere via ModelMPI vs
pywatershed.

Same drb_2yr domain and answers as the serial test
(test_prms_atmosphere.py), distributed over the "nhru" grid with the
mpixarray streaming pipeline. Rank 0 assembles the ONE combined input
file (params + the (ndoy, nhru) soltab tables + dis vars + the CBH
forcings widened to f64); `swrad` streams to parallel NetCDF and is
validated globally over all timesteps; a representative set of the
remaining variables is validated from final state, gathered
rank-ordered. The transp_tindex state machine runs per rank on its
local block (purely per-element).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_atmosphere_mpi.py -v

rtol = atol = 1e-5 matches pywatershed's own atmosphere autotest
standard (see the serial test).
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
from model import ModelMPI

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

CBH_NAMES = ("prcp", "tmax", "tmin")
DIS_NAMES = ("hru_slope", "hru_lat")
STREAMED_NAME = "swrad"  # the one to_netcdf var (mpixarray limit)
FINAL_STATE_NAMES = (
    "tmaxf",
    "tavgc",
    "prmx",
    "hru_ppt",
    "hru_rain",
    "hru_snow",
    "orad_hru",
    "potet",
    "transp_on",
)
# pywatershed's own atmosphere autotest comparison standard
RTOL = ATOL = 1.0e-5

_needed = (
    [
        DOMAIN_DIR / "parameters_PRMSAtmosphere.nc",
        DOMAIN_DIR / "parameters_dis_hru.nc",
        GEN_DIR / "soltab_potsw.nc",
        GEN_DIR / "soltab_horad_potsw.nc",
    ]
    + [DOMAIN_DIR / f"{nn}.nc" for nn in CBH_NAMES]
    + [GEN_DIR / f"{nn}.nc" for nn in FINAL_STATE_NAMES + (STREAMED_NAME,)]
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
    """Rank 0 assembles the ONE combined input file and broadcasts the
    temp dir; rank 0 cleans up afterward (no barrier in teardown)."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "prms_atmosphere_mpi_data"
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        proc_params = xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSAtmosphere.nc"
        )
        soltabs = [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc")
            .rename({"doy": "ndoy", "nhm_id": "nhru"})
            .to_dataset(name=nn)
            for nn in ("soltab_potsw", "soltab_horad_potsw")
        ]
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        forcings = [
            xr.load_dataarray(DOMAIN_DIR / f"{nn}.nc")
            .rename({"nhm_id": "nhru"})
            .astype(np.float64)
            for nn in CBH_NAMES
        ]
        combined = xr.merge(
            [
                proc_params,
                *soltabs,
                dis_hru[list(DIS_NAMES)],
                *forcings,
            ],
            compat="no_conflicts",
        )
        combined = combined.assign_coords(
            nhru=np.arange(combined.sizes["nhru"])
        )
        combined.to_netcdf(input_file)
    comm.Barrier()
    yield {"input_file": input_file, "output_file": output_file}

    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def answers():
    """Ground truth read straight from pywatershed's answer files --
    collective-free, every rank reads identically."""
    names = FINAL_STATE_NAMES + (STREAMED_NAME,)
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in names}


@pytest.fixture(scope="module")
def mpi_run(mpi_paths):
    """Build + run + finalize ModelMPI ONCE; every collective lives here."""
    comm = MPI.COMM_WORLD
    process_dict = {
        "prms_atmosphere": {
            "class": PRMSAtmosphere,
            "discretization": "nhru",
        },
    }
    control = {
        "input_file": mpi_paths["input_file"],
        "output_parallel_netcdf": mpi_paths["output_file"],
        "output_var_names": [STREAMED_NAME],
        "mpi_grid": "nhru",
    }
    model = ModelMPI(process_dict, control)
    model.run(np.float64(1.0))

    final = {
        nn: np.concatenate(
            comm.allgather(model._ds_mpi_stream[nn].values.copy())
        )
        for nn in FINAL_STATE_NAMES
    }
    model.finalize()
    comm.Barrier()  # output file fully flushed before reads
    return {"output_file": mpi_paths["output_file"], "final": final}


@pytest.mark.mpi(min_size=2)
class TestPRMSAtmosphereMPI:
    # -- streamed swrad over all timesteps (global, rank 0) --
    def test_streamed_swrad_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.open_dataset(mpi_run["output_file"]) as ds_out:
            out = ds_out[f"{STREAMED_NAME}_out"].values
        np.testing.assert_allclose(
            out,
            answers[STREAMED_NAME].values,
            rtol=RTOL,
            atol=ATOL,
        )

    # -- final state of everything else, gathered globally --
    def test_final_state_gathered(self, mpi_run, answers):
        for nn, vals in mpi_run["final"].items():
            np.testing.assert_allclose(
                vals,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' final state differs",
            )
