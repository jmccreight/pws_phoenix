"""R3: MPI perfect-restart tests (ModelMPI warm start).

The machinery under test (the flags themselves are validated by the
serial R2 suites): rank-0 gather-then-write of the distributed grid's
flagged state (contiguous rank-ordered blocks -> a full-extent,
SERIAL-FORMAT restart file), every-rank read + own-block restore, and
the warm start's LAZY TIME-SLICE of the input dataset -- the stream
begins at the resume step (fed to parallelize/set_streaming as an
isel view; the superset input file serves any restart) while model
time stays globally indexed (istep0 gates must not re-fire).

Two configurations:

- the two-grid TOY (Upper distributed on "hru" -> MapMPI -> Lower
  replicated on "segment"): distributed AND serial-grid restore, the
  Map across the restart boundary, plus the INTEROP leg -- a SERIAL
  model warm-starts from the MPI-written restart files (the files are
  format-identical by design).
- PRMSGroundwater on drb_2yr (765 HRUs over the ranks = UNEVEN
  decomposition): the block slice arithmetic on real data.

The recipe per configuration: m_ac streams the full window; m_ab
streams a TRUNCATED input file to the restart point and writes
restart; m_bc streams the full window warm-started from it. Final
state must be BIT-identical, per rank, in every variable. Truncated
input files (not an early loop exit) end the a->b run: the stream
defines the step count, and mpixarray has no mid-stream stop.

All collective MPI ops live in the module-scoped fixtures; test
methods are pure (collective-free) asserts -- a failing rank can never
interrupt a collective and hang the others.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_restart_mpi.py -v
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
from hydrology.prms_groundwater import PRMSGroundwater
from map import Map, MapMPI
from model import Model, ModelMPI
from processes_concrete import Lower, Upper

MPIX_ROOT = pl.Path(__file__).parents[4]
DOMAIN_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output"

# a -> c full window; b = the restart point (a -> b truncated file)
TOY_N_AC = 60
TOY_N_AB = 30
GW_N_AC = 90
GW_N_AB = 45

GW_INPUT_NAMES = ("soil_to_gw", "ssr_to_gw", "dprst_seep_hru")

_gw_needed = [
    DOMAIN_DIR / "parameters_PRMSGroundwater.nc",
    DOMAIN_DIR / "parameters_dis_hru.nc",
] + [GEN_DIR / f"{nn}.nc" for nn in GW_INPUT_NAMES]
_gw_missing = [str(ff) for ff in _gw_needed if not ff.exists()]
gw_skipif = pytest.mark.skipif(
    bool(_gw_missing),
    reason=(
        "pywatershed drb_2yr test data not generated; missing: "
        + ", ".join(_gw_missing[:3])
    ),
)


@pytest.fixture(scope="module")
def tmp_dir():
    """One shared temp dir (rank 0 creates + broadcasts + cleans up;
    no teardown barrier on purpose -- see test_up_low_regression_mpi.py
    for the reasoning)."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    yield pl.Path(tmp)
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# two-grid toy: distributed Upper -> MapMPI -> replicated Lower
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_paths(tmp_dir, dimensions, make_two_grid_toy):
    """Rank 0 writes the a->c and a->b hru input files (the kitchen-sink
    combined file; only forcing_up carries time and is truncated)."""
    comm = MPI.COMM_WORLD
    toy = make_two_grid_toy(dimensions)
    data_dir = tmp_dir / "restart_toy"
    paths = {
        "toy": toy,
        "data_dir": data_dir,
        "input_ac": data_dir / "hru_input_ac.nc",
        "input_ab": data_dir / "hru_input_ab.nc",
        "restart_dir": data_dir / "restart",
    }
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        for key, nn in (("input_ac", TOY_N_AC), ("input_ab", TOY_N_AB)):
            xr.Dataset(
                data_vars=dict(
                    forcing_up=toy["forcing_up"].isel(time=slice(0, nn)),
                    param_up_0=toy["up_params"]["param_up_0"],
                    param_up_1=toy["up_params"]["param_up_1"],
                    param_shared_name=toy["up_params"]["param_shared_name"],
                    flow_initial=toy["up_flow_initial"],
                ),
            ).to_netcdf(paths[key])
    comm.Barrier()
    return paths


@pytest.fixture(scope="module")
def toy_run(toy_paths, two_grid_weights):
    """The recipe on the toy; every collective lives here."""
    comm = MPI.COMM_WORLD
    toy = toy_paths["toy"]
    restart_dir = toy_paths["restart_dir"]

    def build_mpi(input_key, out_name, extra_control):
        process_dict = {
            "upper": {"class": Upper, "discretization": "hru"},
            "lower": {
                "class": Lower,
                "discretization": "segment",
                "parameters": toy["low_params"],
                "forcing_low": toy["forcing_low"],
                "storage_initial": toy["low_storage_initial"],
            },
        }
        maps = {
            "hru_to_seg": MapMPI(
                weights=two_grid_weights,
                grid={"hru": "segment"},
                variable={"flow": "flow"},
            )
        }
        control = {
            "input_file": toy_paths[input_key],
            "output_parallel_netcdf": toy_paths["data_dir"] / out_name,
            "output_var_names": ["flow"],
            "mpi_grid": "hru",
            **extra_control,
        }
        return ModelMPI(process_dict, control, maps=maps)

    def collect(model):
        upper_local = {
            nn: model._ds_mpi_stream[nn].values.copy()
            for nn in Upper.get_var_names()
        }
        upper_global = {
            nn: np.concatenate(comm.allgather(vv))
            for nn, vv in upper_local.items()
        }
        seg_ds = model.discretizations["segment"].dataset
        lower = {
            nn: seg_ds[nn].values.copy() for nn in Lower.get_var_names()
        }
        return upper_local, upper_global, lower

    dt = np.float64(1.0)
    m_ac = build_mpi("input_ac", "out_ac.nc", {})
    m_ac.run(dt)
    upper_ac, upper_ac_global, lower_ac = collect(m_ac)
    m_ac.finalize()

    m_ab = build_mpi("input_ab", "out_ab.nc", {})
    m_ab.run(dt)
    m_ab.write_restart(restart_dir)  # ends on a Barrier
    m_ab.finalize()

    m_bc = build_mpi("input_ac", "out_bc.nc", {"restart_read": restart_dir})
    start_ok = m_bc._start_index == TOY_N_AB
    m_bc.run(dt)
    upper_bc, _, lower_bc = collect(m_bc)
    m_bc.finalize()
    comm.Barrier()  # out_bc.nc fully flushed before the rank-0 read
    # the input dataset is time-SLICED at build, so the warm run's
    # streamed output covers exactly the computed window
    n_out_bc = None
    if comm.rank == 0:
        with xr.load_dataset(toy_paths["data_dir"] / "out_bc.nc") as ds:
            n_out_bc = int(ds.sizes["time"])

    # -- interop: a SERIAL model warm-starts from the MPI-written
    # restart files (rank 0 only; the serial model holds no
    # collectives, so the rank branch cannot hang) --
    serial_upper = serial_lower = None
    if comm.rank == 0:
        process_dict = {
            "upper": {
                "class": Upper,
                "discretization": "hru",
                "parameters": toy["up_params"],
                "forcing_up": toy["forcing_up"],
                "flow_initial": toy["up_flow_initial"],
            },
            "lower": {
                "class": Lower,
                "discretization": "segment",
                "parameters": toy["low_params"],
                "forcing_low": toy["forcing_low"],
                "storage_initial": toy["low_storage_initial"],
            },
        }
        maps = {
            "hru_to_seg": Map(
                weights=two_grid_weights,
                grid={"hru": "segment"},
                variable={"flow": "flow"},
            )
        }
        with Model(
            process_dict, {"restart_read": restart_dir}, maps=maps
        ) as m_serial:
            assert m_serial._start_index == TOY_N_AB
            m_serial.run(dt, np.int32(TOY_N_AC - TOY_N_AB))
            hru_ds = m_serial.discretizations["hru"].dataset
            seg_ds = m_serial.discretizations["segment"].dataset
            serial_upper = {
                nn: hru_ds[nn].values.copy() for nn in Upper.get_var_names()
            }
            serial_lower = {
                nn: seg_ds[nn].values.copy() for nn in Lower.get_var_names()
            }
    comm.Barrier()

    return {
        "start_ok": start_ok,
        "n_out_bc": n_out_bc,
        "upper_ac": upper_ac,
        "upper_bc": upper_bc,
        "upper_ac_global": upper_ac_global,
        "lower_ac": lower_ac,
        "lower_bc": lower_bc,
        "serial_upper": serial_upper,
        "serial_lower": serial_lower,
    }


@pytest.mark.mpi(min_size=2)
class TestRestartTwoGridToy:
    def test_resume_index(self, toy_run):
        assert toy_run["start_ok"]

    def test_stream_starts_at_resume(self, toy_run):
        """The sliced stream: the warm run's output file covers only
        the computed window (no fill-value prefix)."""
        if MPI.COMM_WORLD.rank != 0:
            return
        assert toy_run["n_out_bc"] == TOY_N_AC - TOY_N_AB

    def test_distributed_state_bit_identical(self, toy_run):
        for nn, vv in toy_run["upper_ac"].items():
            np.testing.assert_array_equal(
                toy_run["upper_bc"][nn],
                vv,
                err_msg=f"distributed 'upper' variable '{nn}' not "
                "bit-identical after MPI restart",
            )

    def test_replicated_state_bit_identical(self, toy_run):
        for nn, vv in toy_run["lower_ac"].items():
            np.testing.assert_array_equal(
                toy_run["lower_bc"][nn],
                vv,
                err_msg=f"replicated 'lower' variable '{nn}' not "
                "bit-identical after MPI restart",
            )

    def test_serial_warm_start_from_mpi_files(self, toy_run):
        """The interop claim: serial resumes from MPI restart files."""
        if MPI.COMM_WORLD.rank != 0:
            return
        for nn, vv in toy_run["upper_ac_global"].items():
            np.testing.assert_array_equal(
                toy_run["serial_upper"][nn],
                vv,
                err_msg=f"serial-from-MPI 'upper' variable '{nn}' not "
                "bit-identical to the continuous MPI run",
            )
        for nn, vv in toy_run["lower_ac"].items():
            np.testing.assert_array_equal(
                toy_run["serial_lower"][nn],
                vv,
                err_msg=f"serial-from-MPI 'lower' variable '{nn}' not "
                "bit-identical to the continuous MPI run",
            )


# ----------------------------------------------------------------------
# PRMSGroundwater on drb_2yr: uneven decomposition (765 HRUs)
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def gw_paths(tmp_dir):
    """Rank 0 assembles the a->c and a->b combined input files (as in
    test_prms_groundwater_mpi.py, forcings truncated in time)."""
    comm = MPI.COMM_WORLD
    data_dir = tmp_dir / "restart_gw"
    paths = {
        "data_dir": data_dir,
        "input_ac": data_dir / "gw_input_ac.nc",
        "input_ab": data_dir / "gw_input_ab.nc",
        "restart_dir": data_dir / "restart",
    }
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        proc_params = xr.load_dataset(
            DOMAIN_DIR / "parameters_PRMSGroundwater.nc"
        )
        dis_hru = xr.load_dataset(DOMAIN_DIR / "parameters_dis_hru.nc")
        forcings = [
            xr.load_dataarray(GEN_DIR / f"{nn}.nc").rename({"nhm_id": "nhru"})
            for nn in GW_INPUT_NAMES
        ]
        combined = xr.merge(
            [
                proc_params[["gwflow_coef", "gwsink_coef", "gwstor_init"]],
                dis_hru[["hru_area", "hru_in_to_cf"]],
                *forcings,
            ],
            compat="no_conflicts",
        )
        combined = combined.assign_coords(
            nhru=np.arange(combined.sizes["nhru"])
        )
        for key, nn in (("input_ac", GW_N_AC), ("input_ab", GW_N_AB)):
            combined.isel(time=slice(0, nn)).to_netcdf(paths[key])
    comm.Barrier()
    return paths


@pytest.fixture(scope="module")
def gw_run(gw_paths):
    """The recipe on drb groundwater; every collective lives here."""
    restart_dir = gw_paths["restart_dir"]

    def build(input_key, out_name, extra_control):
        process_dict = {
            "prms_groundwater": {
                "class": PRMSGroundwater,
                "discretization": "nhru",
            },
        }
        control = {
            "input_file": gw_paths[input_key],
            "output_parallel_netcdf": gw_paths["data_dir"] / out_name,
            "output_var_names": ["gwres_flow"],
            "mpi_grid": "nhru",
            **extra_control,
        }
        return ModelMPI(process_dict, control)

    def collect(model):
        return {
            nn: model._ds_mpi_stream[nn].values.copy()
            for nn in PRMSGroundwater.get_var_names()
        }

    dt = np.float64(1.0)
    m_ac = build("input_ac", "out_ac.nc", {})
    m_ac.run(dt)
    state_ac = collect(m_ac)
    m_ac.finalize()

    m_ab = build("input_ab", "out_ab.nc", {})
    m_ab.run(dt)
    m_ab.write_restart(restart_dir)  # ends on a Barrier
    m_ab.finalize()

    m_bc = build("input_ac", "out_bc.nc", {"restart_read": restart_dir})
    start_ok = m_bc._start_index == GW_N_AB
    m_bc.run(dt)
    state_bc = collect(m_bc)
    m_bc.finalize()

    return {"start_ok": start_ok, "ac": state_ac, "bc": state_bc}


@gw_skipif
@pytest.mark.mpi(min_size=2)
class TestRestartGroundwaterMPI:
    def test_resume_index(self, gw_run):
        assert gw_run["start_ok"]

    def test_state_bit_identical(self, gw_run):
        """Per-rank compare: a wrong block slice diverges some rank."""
        for nn, vv in gw_run["ac"].items():
            np.testing.assert_array_equal(
                gw_run["bc"][nn],
                vv,
                err_msg=f"groundwater variable '{nn}' not bit-identical "
                "after MPI restart (this rank's block)",
            )
