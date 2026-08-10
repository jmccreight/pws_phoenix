"""
test_two_grid_mpi.py
====================
pytest-mpi Step B regression: Upper on a DISTRIBUTED "hru" grid (mpixarray
parallelize + streaming), Lower on a serial "segment" grid REPLICATED on
every rank, coupled by MapMPI (local partial product + Allreduce). Validates
against the same ground truth as the serial two-grid test (the conftest
two-grid factories; deterministic per seed, so every rank rebuilds identical
segment data -- that determinism IS what replication relies on).

The mpi_run fixture is parametrized over three INPUT STYLES:

  - "mixed": segment data passed in memory, hru data from the one input
    file -- a demonstration that both work side by side;
  - "files": fully file-based data -- segment inputs and the Map weights
    are NetCDF Paths (each rank opens the small files);
  - "yaml":  fully serialized -- process classes by name, all paths, and
    the map spec come from model.yaml via config.load_model_yaml.

As in test_up_low_regression_mpi.py, ALL collective MPI ops live in the
module-scoped fixtures; the test_* methods are pure (collective-free)
asserts, so a failing rank can never interrupt a collective and hang the
others. Every rank sees fixture params in the same order, keeping
collectives aligned across the parametrized runs.

`flow` (hru) is streamed to disk (parallel NetCDF) and validated over all
timesteps on rank 0; `storage` (segment) is collected by the rank-0 zarr
Output and likewise validated over all timesteps. The replicated segment
grid's final state is additionally validated on EVERY rank directly (no
gather needed) plus an allgather cross-rank identity check.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_two_grid_mpi.py -v

Prerequisites: pytest-mpi installed; run under mpirun with >= 2 ranks.
"""

import pathlib as pl
import shutil
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr
import yaml
from mpi4py import MPI

sys.path.insert(0, str(pl.Path(__file__).parent.parent))
from config import load_model_yaml
from map import MapMPI
from model import ModelMPI
from processes_concrete import Lower, Upper

INPUT_STYLES = ("mixed", "files", "yaml")


@pytest.fixture(scope="module")
def two_grid_toy(dimensions, make_two_grid_toy):
    """Module-scoped toy: deterministic -> identical on every rank."""
    return make_two_grid_toy(dimensions)


@pytest.fixture(scope="module")
def answers(dimensions, two_grid_toy, two_grid_weights, compute_two_grid_answers):
    """Ground truth, recomputed identically on every rank."""
    return compute_two_grid_answers(two_grid_toy, two_grid_weights, dimensions)


@pytest.fixture(scope="module")
def mpi_paths(two_grid_toy, two_grid_weights):
    """Rank 0 writes ALL on-disk inputs -- the hru input file, the segment
    input files, the Map weights, and model.yaml (with RELATIVE path
    strings, exercising config's resolve-relative-to-the-yaml rule) --
    then broadcasts the temp dir. Rank 0 cleans up at module teardown (no
    barrier there on purpose -- see test_up_low_regression_mpi.py for the
    reasoning)."""
    comm = MPI.COMM_WORLD
    toy = two_grid_toy
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp) / "two_grid_mpi_data"
    paths = {
        "data_dir": data_dir,
        "input_file": data_dir / "hru_input.nc",
        "low_params": data_dir / "low_params.nc",
        "forcing_low": data_dir / "forcing_low.nc",
        "storage_initial": data_dir / "storage_initial.nc",
        "weights": data_dir / "weights.nc",
        "yaml_file": data_dir / "model.yaml",
    }
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        # The distributed grid's kitchen-sink input file: forcing +
        # time-varying param as (time|month, hru); static params + IC as
        # (hru,). Segment data never enter it.
        xr.Dataset(
            data_vars=dict(
                forcing_up=toy["forcing_up"],
                param_up_0=toy["up_params"]["param_up_0"],
                param_up_1=toy["up_params"]["param_up_1"],
                param_shared_name=toy["up_params"]["param_shared_name"],
                flow_initial=toy["up_flow_initial"],
            ),
        ).to_netcdf(paths["input_file"])
        # Segment inputs + weights, one file each (the bare DataArrays
        # need a name for NetCDF).
        toy["low_params"].to_netcdf(paths["low_params"])
        toy["forcing_low"].rename("forcing_low").to_netcdf(paths["forcing_low"])
        toy["low_storage_initial"].rename("storage_initial").to_netcdf(
            paths["storage_initial"]
        )
        xr.DataArray(
            two_grid_weights, dims=["segment", "hru"], name="weights"
        ).to_netcdf(paths["weights"])
        # The fully-serialized configuration (relative path strings).
        cfg = {
            "control": {
                "input_file": "hru_input.nc",
                "output_parallel_netcdf": "hru_output_yaml.nc",
                "output_serial_zarr": "segment_output_yaml.zarr",
                "output_var_names": ["flow", "storage"],
                "time_chunk_size": 10,
                "mpi_grid": "hru",
            },
            "processes": {
                "upper": {"class": "Upper", "discretization": "hru"},
                "lower": {
                    "class": "Lower",
                    "discretization": "segment",
                    "parameters": "low_params.nc",
                    "forcing_low": "forcing_low.nc",
                    "storage_initial": "storage_initial.nc",
                },
            },
            "maps": {
                "hru_to_seg": {
                    "class": "MapMPI",
                    "weights": "weights.nc",
                    "grid": {"hru": "segment"},
                    "variable": {"flow": "flow"},
                }
            },
        }
        # sort_keys=False is LOAD-BEARING: the processes mapping order IS
        # the execution schedule, and safe_dump alphabetizes by default
        # (which would put lower before upper -> the one-pass order
        # validation rightly rejects it).
        with open(paths["yaml_file"], "w") as ff:
            yaml.safe_dump(cfg, ff, sort_keys=False)
    comm.Barrier()
    yield paths
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module", params=INPUT_STYLES)
def mpi_run(request, mpi_paths, two_grid_toy, two_grid_weights):
    """Build + run + finalize ModelMPI ONCE per input style; every
    collective lives here. Output paths are per-style so the runs don't
    collide."""
    style = request.param
    comm = MPI.COMM_WORLD
    toy = two_grid_toy
    data_dir = mpi_paths["data_dir"]

    if style == "yaml":
        process_dict, control, maps = load_model_yaml(mpi_paths["yaml_file"])
    else:
        if style == "mixed":  # segment data in memory
            lower_data = {
                "parameters": toy["low_params"],
                "forcing_low": toy["forcing_low"],
                "storage_initial": toy["low_storage_initial"],
            }
            weights = two_grid_weights
        else:  # "files": segment inputs and the weights all from disk
            # TODO: JLM: is this branch worthwhile? it requires files to
            # feed memory but that seems not like a thing of concern for us.
            lower_data = {
                "parameters": mpi_paths["low_params"],
                "forcing_low": mpi_paths["forcing_low"],
                "storage_initial": mpi_paths["storage_initial"],
            }
            with xr.load_dataarray(mpi_paths["weights"]) as da:
                weights = da.load().values

        process_dict = {
            "upper": {"class": Upper, "discretization": "hru"},
            "lower": {
                "class": Lower,
                "discretization": "segment",
                # NOTE: "flow" is NOT provided -- the MapMPI feeds it.
                **lower_data,
            },
        }
        maps = {
            "hru_to_seg": MapMPI(
                weights=weights,
                grid={"hru": "segment"},
                variable={"flow": "flow"},
            )
        }
        control = {
            # TODO: JLM: this "input_file mechanism seems insufficient for
            # multiple parallel discretizations. or is there ever only one?
            # to look at is what is loaded from yaml.
            "input_file": mpi_paths["input_file"],
            # Routed by owning grid: flow (hru) streams via mpixarray;
            # storage (segment) goes to the rank-0 zarr Output.
            "output_parallel_netcdf": data_dir / f"hru_output_{style}.nc",
            "output_serial_zarr": data_dir / f"segment_output_{style}.zarr",
            "output_var_names": ["flow", "storage"],
            "time_chunk_size": 10,  # full chunks + a partial tail
            "mpi_grid": "hru",
        }

    model = ModelMPI(process_dict, control, maps=maps)
    model.run(np.float64(1.0))

    seg_ds = model.discretizations["segment"].dataset
    storage_final = seg_ds["storage"].values.copy()
    lower_flow_final = seg_ds["flow"].values.copy()
    # The map's target buffer must BE Lower's cross-grid input (zero-copy).
    map_wired = seg_ds["flow"].values is maps["hru_to_seg"].target_values.values
    # Replicated segment grid: every rank must hold the identical answer.
    gathered = comm.allgather(storage_final)
    replicated_identical = all(np.array_equal(gathered[0], gg) for gg in gathered)
    model.finalize()
    comm.Barrier()  # ensure output files are fully flushed before reads
    return {
        "style": style,
        "output_parallel_netcdf": control["output_parallel_netcdf"],
        "serial_store": control["output_serial_zarr"],
        "storage_final": storage_final,
        "lower_flow_final": lower_flow_final,
        "map_wired": map_wired,
        "replicated_identical": replicated_identical,
    }


@pytest.mark.mpi(min_size=2)
class TestTwoGridMPI:
    """Step B: distributed hru grid -> MapMPI -> replicated segment grid,
    per input style (mixed / files / yaml)."""

    def test_map_buffer_wired(self, mpi_run):
        assert mpi_run["map_wired"]

    def test_replicated_identical_across_ranks(self, mpi_run):
        assert mpi_run["replicated_identical"]

    # -- streamed hru flow over all timesteps (global); validated rank 0 --
    def test_streamed_flow_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_parallel_netcdf"]) as ds_out:
            flow_out = ds_out["flow_out"].values  # (n_time, n_hru) global
        np.testing.assert_allclose(flow_out, answers["flow"], rtol=1e-12)

    # -- replicated segment state: full answer on EVERY rank, no gather --
    def test_segment_storage_final(self, mpi_run, answers):
        np.testing.assert_allclose(
            mpi_run["storage_final"], answers["storage"][-1], rtol=1e-12
        )

    def test_mapped_flow_final(self, mpi_run, answers, two_grid_weights):
        np.testing.assert_allclose(
            mpi_run["lower_flow_final"],
            two_grid_weights @ answers["flow"][-1],
            rtol=1e-12,
        )

    # -- serial-grid storage over ALL timesteps: rank-0 zarr Output --
    def test_serial_grid_storage_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.open_zarr(mpi_run["serial_store"], consolidated=False) as ds_out:
            storage_out = ds_out["storage"].values  # (n_time, n_segment)
        np.testing.assert_allclose(storage_out, answers["storage"], rtol=1e-12)
