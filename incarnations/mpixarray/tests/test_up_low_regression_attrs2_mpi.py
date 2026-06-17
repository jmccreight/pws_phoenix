"""
test_up_low_regression_attrs2_mpi.py
=====================================
pytest-mpi version of the MPI streaming regression for the Upper/Lower toy
model via ModelMPI. Same checks as the former script, restructured as a
pytest class (cf. mpixarray ref_behaviors/..._pytest.py).

Pattern (deadlock-safe): the whole MPI pipeline -- write the combined input on
rank 0, build ModelMPI, run the streaming loop, finalize -- runs ONCE in
setup_class, where ALL collective MPI ops live. Per-rank results, buffer-sharing
facts, and the streamed output file are captured there; the test_* methods are
pure (collective-free) asserts, so a failing rank can never interrupt a
collective and hang the others.

Phase 1 scope: one space-decomposed dataset streamed over time
(set_streaming + iter_time); `flow` is streamed to disk and validated over all
timesteps (global), `storage_previous` from final in-memory state (per rank),
pending the mpixarray multi-output-var (deepcopy) fix. param_up_1 (time-varying
parameter) is omitted -- a Phase 2 design question.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_up_low_regression_attrs2_mpi.py -v

Prerequisites: pytest-mpi installed; run under mpirun with >= 2 ranks.
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
from base_attrs2 import ModelMPI
from processes_attrs2 import Lower, Upper


@pytest.mark.mpi(min_size=2)
class TestRegressionAttrs2MPI:
    """MPI streaming regression for the Upper/Lower toy model via ModelMPI.

    setup_class drives the full collective pipeline once and stashes results on
    the class; the test_* methods only assert on those captured values.
    """

    @classmethod
    def setup_class(cls):
        comm = MPI.COMM_WORLD
        cls.comm = comm
        rank = comm.rank

        # ---- dimensions ----
        n_years = 1
        n_space = 20
        start_year = 2000
        start_time = np.datetime64(f"{start_year}-01-01")
        end_time = (
            np.datetime64(f"{start_year + n_years}-01-01")
            - np.timedelta64(1, "D")
        )
        time = np.arange(start_time, end_time, dtype="datetime64[D]")
        n_time = len(time)
        space = np.arange(n_space)
        cls.n_time = n_time

        # ---- combined input file: rank 0 writes, dir broadcast to all ----
        # time/space are dim coordinates (set_streaming needs the streaming dim
        # to be a coord). param_up_1 (time-varying parameter) is omitted.
        np.random.seed(42)
        tmp_dir = tempfile.mkdtemp() if rank == 0 else None
        tmp_dir = comm.bcast(tmp_dir, root=0)
        cls._tmp_dir = tmp_dir
        data_dir = pl.Path(tmp_dir) / "toy_model_mpi_data"
        input_file = data_dir / "model_input.nc"
        cls.output_file = data_dir / "model_output.nc"

        if rank == 0:
            data_dir.mkdir(exist_ok=True)
            sin_data = np.sin(
                np.arange(0, 2 * np.pi * n_years, 2 * np.pi * n_years / n_time)
            )
            shifts = np.random.uniform(10, 100, n_space)
            forcing_0 = sin_data[:, np.newaxis] + shifts[np.newaxis, :]
            xr.Dataset(
                data_vars=dict(
                    forcing_0=(["time", "space"], forcing_0),
                    forcing_common=(
                        ["time", "space"],
                        np.ones((n_time, n_space)),
                    ),
                    param_up_0=(["space"], np.random.uniform(0.1, 1, n_space)),
                    param_low_0=(
                        ["space"],
                        np.random.uniform(0.17, 0.23, n_space),
                    ),
                    param_common=(["space"], np.zeros(n_space)),
                    flow_initial=(
                        ["space"],
                        np.random.uniform(100, 1000, n_space),
                    ),
                    storage_initial=(
                        ["space"],
                        np.random.uniform(100, 500, n_space),
                    ),
                ),
                coords=dict(time=("time", time), space=("space", space)),
            ).to_netcdf(input_file)
        comm.Barrier()

        # ---- expected answers (read the actual file; identical on all ranks) ----
        with xr.open_dataset(input_file) as ds_in:
            forcing_0_vals = ds_in["forcing_0"].values
            flow_ic_vals = ds_in["flow_initial"].values
            storage_ic_vals = ds_in["storage_initial"].values

        expected_flow = np.zeros((n_time, n_space))
        expected_flow_prev = np.zeros((n_time, n_space))
        for tt in range(n_time):
            expected_flow_prev[tt, :] = (
                flow_ic_vals if tt == 0 else expected_flow[tt - 1, :]
            )
            expected_flow[tt, :] = (
                expected_flow_prev[tt, :] * 0.95 + forcing_0_vals[tt, :]
            )
        expected_storage = np.zeros((n_time, n_space))
        expected_storage_prev = np.zeros((n_time, n_space))
        for tt in range(n_time):
            expected_storage_prev[tt, :] = (
                storage_ic_vals if tt == 0 else expected_storage[tt - 1, :]
            )
            expected_storage[tt, :] = (
                expected_storage_prev[tt, :] * 0.95 + expected_flow[tt, :] * 0.12
            )
        cls.expected_flow = expected_flow
        cls.expected_storage_prev = expected_storage_prev

        # ---- build + run the streaming model (all collectives happen here) ----
        process_dict = {"upper": {"class": Upper}, "lower": {"class": Lower}}
        control = {
            "input_file": input_file,
            "output_file": cls.output_file,
            "output_var_names": ["flow"],  # one streamed output (ModelMPI note)
        }
        model = ModelMPI(process_dict, control)
        model.run(np.float64(1.0))

        # local space slice (single scheme -> contiguous blocks in rank order)
        local_n = int(model._ds_mpi.sizes["space"])
        local_ns = comm.allgather(local_n)
        offset = sum(local_ns[:rank])
        cls.sl = slice(offset, offset + local_n)

        # structural buffer-sharing facts (one ds_mpi -> shared by reference)
        upper = model.model_dict["upper"]
        lower = model.model_dict["lower"]
        cls.shared_param_common = (
            upper._obj["param_common"].values
            is lower._obj["param_common"].values
        )
        cls.shared_forcing_common = (
            upper._obj["forcing_common"].values
            is lower._obj["forcing_common"].values
        )
        cls.shared_flow = (
            upper._obj["flow"].values is lower._obj["flow"].values
        )

        # final in-memory state (copy before finalize closes the store)
        cls.local_flow_final = model._ds_mpi["flow"].values.copy()
        cls.local_storage_prev_final = (
            model._ds_mpi["storage_previous"].values.copy()
        )

        model.finalize()  # streams/closes the output store
        comm.Barrier()  # ensure the output file is fully flushed before reads

    @classmethod
    def teardown_class(cls):
        # Only rank 0 touched the temp dir after setup, so no barrier is needed
        # (and omitting it avoids a teardown hang if a method failed on a rank).
        if cls.comm.rank == 0 and cls._tmp_dir is not None:
            shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    # ---- structural buffer sharing (single ds_mpi) ----
    def test_shared_param_common(self):
        assert self.shared_param_common

    def test_shared_forcing_common(self):
        assert self.shared_forcing_common

    def test_shared_flow_upper_lower(self):
        assert self.shared_flow

    # ---- final in-memory state, per-rank local slice ----
    def test_upper_flow_final(self):
        np.testing.assert_allclose(
            self.local_flow_final, self.expected_flow[-1, self.sl], rtol=1e-12
        )

    def test_lower_storage_previous_final(self):
        np.testing.assert_allclose(
            self.local_storage_prev_final,
            self.expected_storage_prev[-1, self.sl],
            rtol=1e-12,
        )

    # ---- streamed output file (global), validated on rank 0 only ----
    def test_streamed_flow_all_timesteps(self):
        if self.comm.rank != 0:
            return
        with xr.open_dataset(self.output_file) as ds_out:
            flow_out = ds_out["flow_out"].values  # (n_time, n_space) global
        np.testing.assert_allclose(flow_out, self.expected_flow, rtol=1e-12)
