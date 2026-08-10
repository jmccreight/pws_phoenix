"""MPI regression: the ported PRMSSnow via ModelMPI vs pywatershed.

Same drb_2yr domain and answers as the serial test (test_prms_snow.py),
distributed over the "nhru" grid with the mpixarray streaming pipeline.
This is also the first MPI test with NON-space parameter dims riding
the combined input file: the (nmonth, nhru) monthly params and the
(ndoy, nhru) soltab table get their nhru axis decomposed per rank
(mpixarray isels only the parallelized dim), while (ndeplval,) and
('scalar',) params replicate whole.

`pkwater_equiv` streams to parallel NetCDF and is validated globally
over all timesteps at pywatershed's own 1e-3 snow standard; iso and
snow_evap are validated from final state, gathered rank-ordered (the
knife-edge-amplified tcal/through_rain are covered by the serial
tests; see test_prms_snow.py).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_snow_mpi.py -v
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
from hydrology.prms_snow import PRMSSnow
from model import ModelMPI

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

INPUT_NAMES = (
    "hru_ppt",
    "hru_intcpevap",
    "net_ppt",
    "net_rain",
    "net_snow",
    "orad_hru",
    "potet",
    "pptmix",
    "prmx",
    "swrad",
    "tavgc",
    "tmaxc",
    "tminc",
    "transp_on",
)
STREAMED_NAME = "pkwater_equiv"  # the one to_netcdf var
FINAL_STATE_NAMES = ("iso", "snow_evap")
# pywatershed's own snow autotest comparison standard
RTOL = ATOL = 1.0e-3

_needed = [
    DOMAIN_DIR / "parameters_PRMSSnow.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    GEN_DIR / "soltab_horad_potsw.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in INPUT_NAMES + FINAL_STATE_NAMES + (STREAMED_NAME,)
]
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
    """Rank 0 assembles the ONE combined input file (params incl. the
    2-D monthly / soltab tables + dis hru_type + forcings) and
    broadcasts the temp dir; rank 0 cleans up (no barrier in
    teardown)."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "prms_snow_mpi_data"
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        proc_params = xr.load_dataset(DOMAIN_DIR / "parameters_PRMSSnow.nc")
        soltab = xr.load_dataarray(
            GEN_DIR / "soltab_horad_potsw.nc"
        ).rename({"doy": "ndoy", "nhm_id": "nhru"})
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        forcings = [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
            for nn in INPUT_NAMES
        ]
        combined = xr.merge(
            [
                proc_params,
                soltab.to_dataset(name="soltab_horad_potsw"),
                dis_hru[["hru_type"]],
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
        "prms_snow": {
            "class": PRMSSnow,
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
            comm.allgather(
                np.asarray(
                    model._ds_mpi_stream[nn].values, dtype=np.float64
                ).copy()
            )
        )
        for nn in FINAL_STATE_NAMES
    }
    model.finalize()
    comm.Barrier()  # output file fully flushed before reads
    return {"output_file": mpi_paths["output_file"], "final": final}


@pytest.mark.mpi(min_size=2)
class TestPRMSSnowMPI:
    # -- streamed pkwater_equiv over all timesteps (global, rank 0) --
    def test_streamed_pkwater_equiv_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_file"]) as ds_out:
            out = ds_out[f"{STREAMED_NAME}_out"].values
        np.testing.assert_allclose(
            out,
            answers[STREAMED_NAME].values,
            rtol=RTOL,
            atol=ATOL,
        )

    # -- final state, gathered globally --
    def test_final_state_gathered(self, mpi_run, answers):
        for nn, vals in mpi_run["final"].items():
            np.testing.assert_allclose(
                vals,
                answers[nn].values[-1, :],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"variable '{nn}' final state differs",
            )
