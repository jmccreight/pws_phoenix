"""
test_two_grid_mpi.py
====================
pytest-mpi Step B regression: Upper on a DISTRIBUTED "hru" grid (mpixarray
parallelize + streaming), Lower on a serial "segment" grid held by
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
Output and likewise validated over all timesteps.

Each input style is run TWICE: as the default, where the segment grid's
process runs on rank 0 only, and with control["serial_grids_all_ranks"],
where it runs on every rank. The default's final segment state is
validated on rank 0 and is the build's on the other ranks; the
all-ranks run's is validated on EVERY rank directly (no gather needed)
plus an allgather cross-rank identity check; rank 0's state and both
output stores are the same in the two runs, bit for bit.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_two_grid_mpi.py -v

Prerequisites: pytest-mpi installed; run under mpirun with >= 2 ranks.
"""

import pathlib as pl
import shutil
import tempfile

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI

from conftest import two_grid_mpi_paths, write_two_grid_mpi_files
from pws_phoenix.config import load_model_yaml
from pws_phoenix.maps import MapMPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.processes_concrete import Lower, Upper

INPUT_STYLES = ("mixed", "files", "yaml")


@pytest.fixture(scope="module")
def two_grid_toy(dimensions, make_two_grid_toy):
    """Module-scoped toy: deterministic -> identical on every rank."""
    return make_two_grid_toy(dimensions)


@pytest.fixture(scope="module")
def answers(
    dimensions, two_grid_toy, two_grid_weights, compute_two_grid_answers
):
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
    paths = two_grid_mpi_paths(data_dir)
    if comm.rank == 0:
        write_two_grid_mpi_files(data_dir, toy, two_grid_weights)
    comm.Barrier()
    yield paths
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module", params=INPUT_STYLES)
def mpi_run(request, mpi_paths, two_grid_toy, two_grid_weights):
    """Build + run + finalize ModelMPI TWICE per input style (the
    default, then control["serial_grids_all_ranks"]); every collective
    lives here. Output paths are per-style and per-run so the runs
    don't collide. The keys without a suffix are the default run's."""
    style = request.param
    comm = MPI.COMM_WORLD
    default = _run(
        style, False, comm, mpi_paths, two_grid_toy, two_grid_weights
    )
    all_ranks = _run(
        style, True, comm, mpi_paths, two_grid_toy, two_grid_weights
    )
    return default | {
        "all_ranks": all_ranks,
        "rank0_storage_final": comm.bcast(default["storage_final"], root=0),
    }


def _run(
    style, serial_grids_all_ranks, comm, mpi_paths, toy, two_grid_weights
):
    """One build + run + finalize; new process_dict, maps and control
    each time (a MapMPI is bound to its model)."""
    data_dir = mpi_paths["data_dir"]
    tag = f"{style}_all_ranks" if serial_grids_all_ranks else style

    if style == "yaml":
        process_dict, control, maps, _ = load_model_yaml(
            mpi_paths["yaml_file"]
        )
    else:
        if style == "mixed":  # segment data in memory
            lower_data = {
                "parameters": toy["low_params"],
                "forcing_low": toy["forcing_low"],
                "storage": toy["low_storage_initial"],
            }
            upper_data = {"flow": toy["up_flow_initial"]}
            weights = two_grid_weights
        else:  # "files": segment inputs and the weights all from disk
            # TODO: JLM: is this branch worthwhile? it requires files to
            # feed memory but that seems not like a thing of concern for us.
            lower_data = {
                "parameters": mpi_paths["low_params"],
                "forcing_low": mpi_paths["forcing_low"],
                "storage": mpi_paths["storage_initial"],
            }
            upper_data = {"flow": mpi_paths["flow_initial"]}
            with xr.load_dataarray(mpi_paths["weights"]) as da:
                weights = da.load().values

        process_dict = {
            "upper": {
                "class": Upper,
                "discretization": "hru",
                **upper_data,
            },
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
            "output_var_names": ["flow", "storage"],
            "time_chunk_size": 10,  # full chunks + a partial tail
            "mpi_grid": "hru",
        }
    # the yaml names its own stores: both styles get per-run ones here
    control = control | {
        "output_parallel_netcdf": data_dir / f"hru_output_{tag}.nc",
        "output_serial_zarr": data_dir / f"segment_output_{tag}.zarr",
        "serial_grids_all_ranks": serial_grids_all_ranks,
    }

    model = ModelMPI(process_dict, control, maps=maps)
    seg_ds = model.discretizations["segment"].dataset
    storage_built = seg_ds["storage"].values.copy()
    model.run(np.float64(1.0))

    storage_final = seg_ds["storage"].values.copy()
    lower_flow_final = seg_ds["flow"].values.copy()
    # The map's target buffer must BE Lower's cross-grid input (zero-copy).
    map_wired = (
        seg_ds["flow"].values is maps["hru_to_seg"].target_values.values
    )
    # Compared across ranks: identical when the segment grid runs on
    # every rank.
    gathered = comm.allgather(storage_final)
    replicated_identical = all(
        np.array_equal(gathered[0], gg) for gg in gathered
    )
    model.finalize()
    comm.Barrier()  # ensure output files are fully flushed before reads
    return {
        "style": style,
        "output_parallel_netcdf": control["output_parallel_netcdf"],
        "serial_store": control["output_serial_zarr"],
        "storage_built": storage_built,
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
        """With serial_grids_all_ranks every rank holds the same
        segment state; by default only rank 0 ran it."""
        assert mpi_run["all_ranks"]["replicated_identical"]
        assert not mpi_run["replicated_identical"]

    def test_segment_grid_runs_on_rank0_only(self, mpi_run):
        """The default: rank 0 holds the all-ranks run's segment state
        bit for bit, and the other ranks the values of the build."""
        if MPI.COMM_WORLD.rank == 0:
            expected = mpi_run["all_ranks"]["storage_final"]
        else:
            expected = mpi_run["storage_built"]
            assert not np.array_equal(expected, mpi_run["rank0_storage_final"])
        np.testing.assert_array_equal(mpi_run["storage_final"], expected)

    def test_stores_same_in_both_runs(self, mpi_run):
        """Both output stores do not depend on where the segment grid
        runs, bit for bit."""
        if MPI.COMM_WORLD.rank != 0:
            return
        other = mpi_run["all_ranks"]
        with (
            xr.load_dataset(mpi_run["output_parallel_netcdf"]) as ds_nc,
            xr.load_dataset(other["output_parallel_netcdf"]) as ds_nc_all,
        ):
            np.testing.assert_array_equal(
                ds_nc["flow_out"].values, ds_nc_all["flow_out"].values
            )
        with (
            xr.open_zarr(mpi_run["serial_store"], consolidated=False) as ds_z,
            xr.open_zarr(
                other["serial_store"], consolidated=False
            ) as ds_z_all,
        ):
            np.testing.assert_array_equal(
                ds_z["storage"].values, ds_z_all["storage"].values
            )

    # -- streamed hru flow over all timesteps (global); validated rank 0 --
    def test_streamed_flow_all_timesteps(self, mpi_run, answers):
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(mpi_run["output_parallel_netcdf"]) as ds_out:
            flow_out = ds_out["flow_out"].values  # (n_time, n_hru) global
        np.testing.assert_allclose(flow_out, answers["flow"], rtol=1e-12)

    # -- segment state: the full answer on EVERY rank (no gather) when
    # the segment grid runs on all of them, on rank 0 by default --
    def test_segment_storage_final(self, mpi_run, answers):
        np.testing.assert_allclose(
            mpi_run["all_ranks"]["storage_final"],
            answers["storage"][-1],
            rtol=1e-12,
        )
        if MPI.COMM_WORLD.rank == 0:
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
        with xr.open_zarr(
            mpi_run["serial_store"], consolidated=False
        ) as ds_out:
            storage_out = ds_out["storage"].values  # (n_time, n_segment)
        np.testing.assert_allclose(storage_out, answers["storage"], rtol=1e-12)


# -- buffer identity across streaming rebinds ---------------------------------
#
# What is tested here, and why (DEVELOPER.md "Per-step data access"):
# a process reads its fields through `self._obj[name].values`, where
# `self._obj` is the grid dataset. Under MPI the Model rebinds every
# distributed-grid process to each streaming window (`proc._obj =
# step`, model.py run loop), so `self._obj` is a DIFFERENT Dataset
# object each step. The question is whether the ndarray BEHIND each
# name is the same one all run -- the precondition for a process to
# fetch its buffers once and keep the references (a Dataset lookup
# costs ~10-20 us; the kernels cost less than all the lookups). It
# should hold because mpixarray's `iterate` re-points only
# time-dimensioned file variables (which ModelMPI shadows with
# (hru,) buffers refilled IN PLACE from `step.mpi.src`) and leaves
# RAM-only buffers alone ("keeps its state across steps",
# mpixarray/iterable.py), and the per-step `assign_coords` is a
# shallow copy. This pins that reading of the two codebases: the
# probe records the memory address of every declared field it reads,
# every step, through the normal path; each name must map to ONE
# address. A failure means a buffer was reallocated under a name --
# a reference cache would then compute on stale data.


class UpperProbe(Upper):
    """Upper that records the address of each field buffer it reads,
    every step (class-level: the record outlives the rebinding)."""

    addresses: dict[str, set[int]] = {}
    n_steps: int = 0

    def calculate(self, dt, time):
        names = (
            self.get_parameters()
            + self.get_inputs()
            + tuple(self.get_variables())
        )
        for name in names:
            self.addresses.setdefault(name, set()).add(
                self._obj[name].values.ctypes.data
            )
        UpperProbe.n_steps += 1
        super().calculate(dt, time)


@pytest.fixture(scope="module")
def buffer_probe_run(mpi_paths, two_grid_toy, two_grid_weights):
    """Build + run + finalize ONCE with UpperProbe on the distributed
    grid ("mixed" style; probe-suffixed outputs). Every collective
    lives here."""
    toy = two_grid_toy
    data_dir = mpi_paths["data_dir"]
    process_dict = {
        "upper": {
            "class": UpperProbe,
            "discretization": "hru",
            "flow": toy["up_flow_initial"],
        },
        "lower": {
            "class": Lower,
            "discretization": "segment",
            "parameters": toy["low_params"],
            "forcing_low": toy["forcing_low"],
            "storage": toy["low_storage_initial"],
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
        "input_file": mpi_paths["input_file"],
        "output_parallel_netcdf": data_dir / "hru_output_probe.nc",
        "output_serial_zarr": data_dir / "segment_output_probe.zarr",
        "output_var_names": ["flow", "storage"],
        "time_chunk_size": 10,
        "mpi_grid": "hru",
    }
    UpperProbe.addresses.clear()
    UpperProbe.n_steps = 0
    model = ModelMPI(process_dict, control, maps=maps)
    model.run(np.float64(1.0))
    model.finalize()
    MPI.COMM_WORLD.Barrier()
    return {
        "addresses": {kk: set(vv) for kk, vv in UpperProbe.addresses.items()},
        "n_steps": UpperProbe.n_steps,
    }


@pytest.mark.mpi(min_size=2)
def test_step_buffers_persist_across_rebinds(buffer_probe_run):
    """Every buffer a distributed-grid process reads is ONE ndarray for
    the whole run: inputs (refilled in place), parameters, state."""
    assert buffer_probe_run["n_steps"] > 1
    assert set(buffer_probe_run["addresses"]) >= {
        "param_up_0",
        "param_up_1",
        "forcing_up",
        "flow",
        "flow_previous",
    }
    moved = {
        name: len(addrs)
        for name, addrs in buffer_probe_run["addresses"].items()
        if len(addrs) != 1
    }
    assert not moved, f"buffers reallocated across steps: {moved}"
