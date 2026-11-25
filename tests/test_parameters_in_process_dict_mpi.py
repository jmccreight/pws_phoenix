"""ModelMPI takes the distributed grid's parameters from process_dict.

A distributed-grid entry of process_dict may carry "parameters", an
xr.Dataset at FULL extent as in the serial Model; each rank keeps a
copy of its block, and control["input_file"] then holds only the
streamed inputs. The Upper/Lower toy model is run twice on every rank:

- combined: the parameters and the forcings in one input file;
- separate: the forcings alone in the input file, the parameters in
  process_dict. That file has no `month` dim, so the (month, space)
  parameter brings it.

The two runs must write the same output, bit for bit.

Three builds that raise, on every rank: a parameter given in both
places, a parameter that is not at full extent, and "parameters" given
as a path.

All collective MPI ops live in the module-scoped fixture; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_parameters_in_process_dict_mpi.py
"""

import pathlib as pl
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import xarray as xr
from mpi4py import MPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.processes_concrete import Lower, Upper

UPPER_PARAMETERS = ["param_up_0", "param_up_1", "param_shared_name"]
LOWER_PARAMETERS = ["param_low_0", "param_shared_name"]
OUTPUT_NAMES = ["flow", "storage"]


def _raised(
    process_dict: dict[str, dict[str, Any]],
    control: dict[str, Any],
    kind: type[Exception],
) -> str:
    """The message of the `kind` this build raises; "" when it builds
    (the model is then closed)."""
    try:
        model = ModelMPI(process_dict, control)
    except kind as err:
        return str(err)
    model.finalize()
    return ""


@pytest.fixture(scope="module")
def runs(dimensions, make_toy_input):
    """Both runs and the three builds that raise; every collective
    lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp)
    combined_file = data_dir / "combined_input.nc"
    forcing_file = data_dir / "forcing_input.nc"
    if comm.rank == 0:
        toy = make_toy_input(dimensions)
        # initial state stays OUT of the input file
        toy.drop_vars(["flow_initial", "storage_initial"]).to_netcdf(
            combined_file
        )
        toy[["forcing_up", "forcing_low"]].to_netcdf(forcing_file)
    comm.Barrier()

    # deterministic toy: every rank builds the same full-extent data
    ds = make_toy_input(dimensions)

    def process_dict(with_parameters: bool) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {
            "upper": {"class": Upper, "flow": ds["flow_initial"]},
            "lower": {"class": Lower, "storage": ds["storage_initial"]},
        }
        if with_parameters:
            entries["upper"]["parameters"] = ds[UPPER_PARAMETERS]
            entries["lower"]["parameters"] = ds[LOWER_PARAMETERS]
        return entries

    def control(input_file: pl.Path, output_name: str) -> dict[str, Any]:
        return {
            "input_file": input_file,
            "output_parallel_netcdf": data_dir / output_name,
            "output_var_names": OUTPUT_NAMES,
        }

    result: dict[str, Any] = {
        "combined_output": data_dir / "combined_output.nc",
        "separate_output": data_dir / "separate_output.nc",
    }

    model = ModelMPI(
        process_dict(False), control(combined_file, "combined_output.nc")
    )
    model.run(np.float64(1.0))
    model.finalize()
    comm.Barrier()

    supplied = process_dict(True)
    model = ModelMPI(supplied, control(forcing_file, "separate_output.nc"))
    stream = model._ds_mpi_stream
    block0, block1 = model._decomp_slice
    result["block_values"] = all(
        np.array_equal(
            stream[name].values, ds[name].values[..., block0:block1]
        )
        for name in set(UPPER_PARAMETERS + LOWER_PARAMETERS)
    )
    result["block_is_a_copy"] = not any(
        np.shares_memory(stream[name].values, ds[name].values)
        for name in set(UPPER_PARAMETERS + LOWER_PARAMETERS)
    )
    result["shared_parameter_is_one_array"] = np.shares_memory(
        model.model_dict["upper"]._refs["param_shared_name"],
        model.model_dict["lower"]._refs["param_shared_name"],
    )
    result["model_dropped_the_datasets"] = not any(
        "parameters" in entry for entry in model._process_dict.values()
    )
    result["caller_keeps_the_datasets"] = all(
        "parameters" in entry for entry in supplied.values()
    )
    model.run(np.float64(1.0))
    model.finalize()
    comm.Barrier()  # the output files are complete before any read

    # the builds that raise: each raises before the writer is opened
    result["given_twice"] = _raised(
        process_dict(True),
        control(combined_file, "raises_output.nc"),
        ValueError,
    )
    short = process_dict(True)
    short["upper"]["parameters"] = ds[UPPER_PARAMETERS].isel(
        space=slice(0, dimensions["n_space"] - 1)
    )
    result["not_full_extent"] = _raised(
        short, control(forcing_file, "raises_output.nc"), ValueError
    )
    as_path = process_dict(True)
    as_path["upper"]["parameters"] = combined_file
    result["as_path"] = _raised(
        as_path, control(forcing_file, "raises_output.nc"), TypeError
    )

    yield result
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestParametersInProcessDict:
    def test_same_output_as_the_combined_file(self, runs):
        if MPI.COMM_WORLD.rank != 0:
            return
        combined = xr.load_dataset(runs["combined_output"])
        separate = xr.load_dataset(runs["separate_output"])
        for name in OUTPUT_NAMES:
            values = separate[f"{name}_out"].values
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(
                values, combined[f"{name}_out"].values
            )

    def test_each_rank_holds_a_copy_of_its_block(self, runs):
        assert runs["block_values"]
        assert runs["block_is_a_copy"]

    def test_shared_parameter_is_one_array(self, runs):
        assert runs["shared_parameter_is_one_array"]

    def test_only_the_caller_keeps_the_full_extent(self, runs):
        assert runs["model_dropped_the_datasets"]
        assert runs["caller_keeps_the_datasets"]

    def test_given_twice_raises(self, runs):
        assert "give it once" in runs["given_twice"]

    def test_not_full_extent_raises(self, runs):
        assert "full extent" in runs["not_full_extent"]

    def test_path_raises(self, runs):
        assert "must be an xr.Dataset" in runs["as_path"]
