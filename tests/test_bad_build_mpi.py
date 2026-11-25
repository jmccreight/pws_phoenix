"""A ModelMPI build that raises releases what it opened.

A constructor that raises returns no object, so neither finalize()
nor a `with` block can close its files: ModelMPI.__init__ does it
(ModelMPI._release). Two raises are provoked on the Upper/Lower toy
model, on every rank:

- early: a bad control["mpi_window_steps"], raised when only the
  input file is open;
- late: a Map that targets the distributed grid, raised after the
  NetCDF output writer is open.

After the late one a second model is built on the SAME output path
and run: its streamed flow must be the toy answers.

The model object of a build that raises is reached through a subclass
that records `self` before building.

All collective MPI ops live in the module-scoped fixture; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_bad_build_mpi.py -v
"""

import pathlib as pl
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI
from pws_phoenix.maps import Map
from pws_phoenix.model import ModelMPI
from pws_phoenix.processes_concrete import Lower, Upper


class _RecordingModelMPI(ModelMPI):
    """Keeps the object of a build that raises."""

    built: list[ModelMPI] = []

    def _build(self) -> None:
        _RecordingModelMPI.built.append(self)
        super()._build()


@pytest.fixture(scope="module")
def answers(dimensions, make_toy_input, compute_answers):
    """Vectorized ground truth, recomputed identically on every rank."""
    ds = make_toy_input(dimensions)
    return compute_answers(
        ds["forcing_up"].values,
        ds["flow_initial"].values,
        ds["storage_initial"].values,
        dimensions["n_time"],
        ds["param_up_1"].values,
        dimensions["time"],
        ds["forcing_low"].values,
    )


@pytest.fixture(scope="module")
def bad_builds(dimensions, make_toy_input):
    """Two builds that raise, then a build and a run on the same
    output path; every collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp)
    input_file = data_dir / "model_input.nc"
    output_file = data_dir / "model_output.nc"
    if comm.rank == 0:
        # initial state stays OUT of the combined file
        make_toy_input(dimensions).drop_vars(
            ["flow_initial", "storage_initial"]
        ).to_netcdf(input_file)
    comm.Barrier()

    # deterministic toy: every rank builds the same full-extent state
    ds = make_toy_input(dimensions)

    def process_dict() -> dict[str, dict[str, Any]]:
        return {
            "upper": {"class": Upper, "flow": ds["flow_initial"]},
            "lower": {"class": Lower, "storage": ds["storage_initial"]},
        }

    control = {
        "input_file": input_file,
        "output_parallel_netcdf": output_file,
        "output_var_names": ["flow"],
    }
    result: dict[str, Any] = {"output_file": output_file}

    # early: raised when only the input file is open
    try:
        _RecordingModelMPI(process_dict(), control | {"mpi_window_steps": 0})
    except ValueError as err:
        result["early_error"] = str(err)
    early = _RecordingModelMPI.built[-1]
    result["early_finalized"] = early._finalized
    result["early_has_stream"] = hasattr(early, "_ds_mpi_stream")

    # late: raised after the output writer is open
    n_space = dimensions["n_space"]
    bad_map = Map(
        weights=np.eye(n_space),
        grid={"elsewhere": "space"},
        variable={"flow": "flow"},
    )
    try:
        _RecordingModelMPI(process_dict(), control, maps={"bad": bad_map})
    except ValueError as err:
        result["late_error"] = str(err)
    late = _RecordingModelMPI.built[-1]
    result["late_is_second_build"] = late is not early
    result["late_finalized"] = late._finalized
    result["late_writer_open"] = late._ds_mpi_stream.mpi.has_writer

    # the same output path, built again and run
    model = ModelMPI(process_dict(), control)
    # what the probe above reads on a model whose writer IS open
    result["built_writer_open"] = model._ds_mpi_stream.mpi.has_writer
    model.run(np.float64(1.0))
    model.finalize()
    comm.Barrier()  # the output file is complete before any read

    yield result
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestBadBuild:
    def test_early_raise_released(self, bad_builds):
        assert "mpi_window_steps" in bad_builds.get("early_error", "")
        assert bad_builds["early_finalized"]
        # raised before the stream existed: only the input was open
        assert not bad_builds["early_has_stream"]

    def test_late_raise_closed_writer(self, bad_builds):
        assert "targets the distributed grid" in bad_builds.get(
            "late_error", ""
        )
        assert bad_builds["late_is_second_build"]
        assert bad_builds["late_finalized"]
        # the probe reads True on a built model, whose writer is open
        assert bad_builds["built_writer_open"]
        # and False on the model whose build raised
        assert not bad_builds["late_writer_open"]

    def test_same_output_path_reused(self, bad_builds, answers):
        """The file the bad build created is written again, whole."""
        if MPI.COMM_WORLD.rank != 0:
            return
        with xr.load_dataset(bad_builds["output_file"]) as ds_out:
            flow_out = ds_out["flow_out"].values
        np.testing.assert_allclose(
            flow_out, answers["expected_flow"], rtol=1e-12
        )
