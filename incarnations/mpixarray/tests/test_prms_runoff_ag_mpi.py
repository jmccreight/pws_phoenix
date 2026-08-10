"""MPI regression: the ported PRMSRunoffAg via ModelMPI.

Same fgr_ag_2yr spinup configuration and answers as the serial test
(test_prms_runoff_ag.py), distributed over the "nhru" grid (612 HRUs).
Rank 0 assembles the ONE combined input file (params + dis vars + the
16 disk forcings + the tiled static ag_frac); `sroff` streams to
parallel NetCDF and is validated globally over all timesteps; a
representative final-state set is gathered rank-ordered. The istep0
ag-area carve-out and the post-kernel area update are purely
per-element (each rank derives its own rows).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_runoff_ag_mpi.py -v

Tolerances: upstream's ag standard (1e-5; dprst_vol_open (3e-4,3e-4);
GSFLOW Fortran answers).
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
from hydrology.prms_runoff import PRMSRunoffAg
from model import ModelMPI

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "fgr_ag_2yr"
GEN_DIR = DOMAIN_DIR / "output_spinup"

DISK_INPUT_NAMES = (
    "soil_lower_prev",
    "soil_rechr_prev",
    "ag_soil_moist_prev",
    "ag_soil_rechr_prev",
    "net_rain",
    "net_ppt",
    "net_snow",
    "potet",
    "snowmelt",
    "snow_evap",
    "pkwater_equiv",
    "pptmix_nopack",
    "snowcov_area",
    "through_rain",
    "hru_intcpevap",
    "intcp_changeover",
)
DIS_NAMES = ("hru_type", "hru_area", "hru_in_to_cf")
STREAMED_NAME = "sroff"  # the one to_netcdf var (mpixarray limit)
FINAL_STATE_NAMES = (
    "infil",
    "infil_ag",
    "imperv_stor",
    "hru_impervstor",
    "dprst_vol_open",
    "dprst_stor_hru",
    "dprst_seep_hru",
    "hru_sroffp",
)
RTOL = ATOL = 1.0e-5
PER_VAR_TOL = {  # (rtol, atol)
    "dprst_vol_open": (3.0e-4, 3.0e-4),
}

_needed = [
    DOMAIN_DIR / "parameters_PRMSRunoffAg.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
    DOMAIN_DIR / "ag_frac_static.nc",
] + [
    GEN_DIR / f"{nn}.nc"
    for nn in DISK_INPUT_NAMES + FINAL_STATE_NAMES + (STREAMED_NAME,)
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed fgr_ag_2yr test data not present/generated; missing: "
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
    data_dir = pl.Path(tmp) / "prms_runoff_ag_mpi_data"
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        proc_params = xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSRunoffAg.nc"
        )
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        forcings = [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
            for nn in DISK_INPUT_NAMES
        ]
        template = forcings[7]  # potet
        ag_static = xr.load_dataarray(DOMAIN_DIR / "ag_frac_static.nc")
        ag_frac = xr.DataArray(
            np.tile(ag_static.values, (template.sizes["time"], 1)),
            dims=("time", "nhru"),
            coords={"time": template["time"], "nhru": template["nhru"]},
            name="ag_frac",
        )
        combined = xr.merge(
            [proc_params, dis_hru[list(DIS_NAMES)], ag_frac, *forcings],
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
    names = FINAL_STATE_NAMES + (STREAMED_NAME,)
    return {nn: xr.load_dataarray(GEN_DIR / f"{nn}.nc") for nn in names}


@pytest.fixture(scope="module")
def mpi_run(mpi_paths):
    """Build + run + finalize ModelMPI ONCE; every collective lives here."""
    comm = MPI.COMM_WORLD
    process_dict = {
        "prms_runoff_ag": {
            "class": PRMSRunoffAg,
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
class TestPRMSRunoffAgMPI:
    def test_streamed_sroff_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_file"]) as ds_out:
            flow_out = ds_out[f"{STREAMED_NAME}_out"].values
        np.testing.assert_allclose(
            flow_out,
            answers[STREAMED_NAME].values,
            rtol=RTOL,
            atol=ATOL,
        )

    def test_final_state_gathered(self, mpi_run, answers):
        for nn, vals in mpi_run["final"].items():
            rtol, atol = PER_VAR_TOL.get(nn, (RTOL, ATOL))
            np.testing.assert_allclose(
                vals,
                answers[nn].values[-1, :],
                rtol=rtol,
                atol=atol,
                err_msg=f"variable '{nn}' final state differs",
            )
