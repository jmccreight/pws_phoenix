"""Budget error mode under MPI: the collective raise.

``ModelMPI._raise_if_imbalanced`` agrees across ranks with one
``allreduce(MAX)`` of the local imbalanced count before raising, so
that a rank whose own block balances still raises with the others (a
lone raising rank would hang the rest at the next collective). The
serial suite (test_budget.py) cannot exercise that; here a bucket
leaks on ONE rank only (the last rank; N_SPACE = 5 over the ranks)
and every rank must raise -- with the leaky rank's block size as the
count in its message. "warn" stays local: only the leaky rank
warns, nobody raises.

All collectives live in the module fixture; the test methods are pure
asserts over allgathered results. finalize() runs on every rank after
the collective raise (all raise together, so the writer close cannot
hang).

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_budget_mpi.py -v
"""

import pathlib as pl
import shutil
import tempfile
import warnings

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI
from pws_phoenix.model import ModelMPI
from test_budget import N_SPACE, N_TIME, _Bucket, _inputs

LEAK_RANK = MPI.COMM_WORLD.size - 1


class _RankLeakyBucket(_Bucket):
    """Leaks 0.1/step on LEAK_RANK only; balanced everywhere else."""

    def _loss(self) -> np.ndarray:
        return 0.1 if MPI.COMM_WORLD.rank == LEAK_RANK else 0.0


@pytest.fixture(scope="module")
def input_file():
    """Rank 0 writes the kitchen-sink input file (inflow over time,
    leak_coef + storage_init static) on the default "space" grid."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    path = pl.Path(tmp) / "bucket_input.nc"
    if comm.rank == 0:
        inp = _inputs(np.random.default_rng(0))
        xr.Dataset(
            {
                "inflow": inp["inflow"],
                "leak_coef": inp["parameters"]["leak_coef"],
                "storage_init": inp["storage_init"],
            }
        ).to_netcdf(path)
    comm.Barrier()
    yield path
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def runs(input_file):
    """Run the rank-leaky bucket in "error" and "warn" mode; gather
    per-rank (raised message or None, number of budget warnings)."""
    comm = MPI.COMM_WORLD
    out = {}
    for mode in ("error", "warn"):
        model = ModelMPI(
            {"bucket": {"class": _RankLeakyBucket}},
            {
                "input_file": input_file,
                "output_parallel_netcdf": input_file.parent / f"{mode}.nc",
                "output_var_names": ["storage"],
                "budget": mode,
            },
        )
        aa, bb = model._decomp_slice
        out["leak_count"] = comm.bcast(bb - aa, root=LEAK_RANK)
        raised = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model.run(np.float64(1.0))
            except ValueError as err:
                raised = str(err)
        model.finalize()  # every rank, together (see module docstring)
        n_warned = sum(1 for ww in caught if "budget" in str(ww.message))
        out[mode] = comm.allgather((raised, n_warned))
    return out


@pytest.mark.mpi(min_size=2)
class TestBudgetCollectiveRaise:
    def test_setup_is_a_subset_rank_leak(self):
        """The premise: more than one rank, and the leaky rank owns at
        least one location (N_SPACE over the ranks)."""
        assert MPI.COMM_WORLD.size >= 2
        assert N_SPACE >= MPI.COMM_WORLD.size
        assert N_TIME >= 1

    def test_error_mode_every_rank_raises(self, runs):
        """All ranks raise, each carrying the allreduce(MAX) count =
        the leaky rank's block size (1 when N_SPACE = 5 over 4 ranks);
        ranks whose own block balances raise with that same count."""
        expected = f"{runs['leak_count']} location(s)"
        for rank, (raised, _) in enumerate(runs["error"]):
            assert raised is not None, f"rank {rank} did not raise"
            assert "_RankLeakyBucket budget imbalanced" in raised
            assert expected in raised, (rank, raised)

    def test_error_mode_only_leaky_rank_warns(self, runs):
        """The per-rank warning is local: only the leaky rank has a
        nonzero count to report."""
        for rank, (_, n_warned) in enumerate(runs["error"]):
            assert n_warned == (1 if rank == LEAK_RANK else 0), rank

    def test_warn_mode_is_local(self, runs):
        """Mode "warn" raises nowhere; the leaky rank warns once per step."""
        for rank, (raised, n_warned) in enumerate(runs["warn"]):
            assert raised is None, (rank, raised)
            assert n_warned == (N_TIME if rank == LEAK_RANK else 0), rank
