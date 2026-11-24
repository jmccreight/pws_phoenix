"""A raise from what rank 0 runs ALONE ends the MPI job.

By default a serial grid's processes and inputs run on rank 0 only
(ModelMPI.run). When one of them raises, no other rank does, and they
would wait forever at the next collective (a MapMPI's Allreduce).
ModelMPI._call_alone therefore prints the traceback and ends the job
(comm.Abort).

An abort ends every rank, pytest included, so this is NOT an _mpi
file: ONE pytest process writes the two-grid toy's files and launches
the ranks itself (the launcher of ${MPI_CMD}, default ``mpirun -n
4``), then judges how the job ended. A job that does not end within
JOB_TIMEOUT_S is the hang this guards against.

In an environment without MPI (the serial one) there is nothing to
launch; the test then asserts that mpi4py is indeed not installed.
"""

import importlib.util
import os
import shlex
import shutil
import signal
import subprocess
import sys

import pytest

from conftest import write_two_grid_mpi_files

JOB_TIMEOUT_S = 180
MESSAGE = "raised by the segment grid's process"

# What every rank runs: the two-grid toy from its yaml, with a Lower
# (the segment grid's process) that raises at its third calculate.
SCRIPT = f'''
import sys

import numpy as np

from pws_phoenix.config import load_model_yaml
from pws_phoenix.model import ModelMPI
from pws_phoenix.processes_concrete import Lower


class LowerRaises(Lower):
    n_calls = 0

    def calculate(self, dt, time):
        LowerRaises.n_calls += 1
        if LowerRaises.n_calls == 3:
            raise RuntimeError("{MESSAGE}")
        super().calculate(dt, time)


process_dict, control, maps, _ = load_model_yaml(sys.argv[1])
process_dict["lower"]["class"] = LowerRaises
control["serial_grids_all_ranks"] = sys.argv[2] == "all_ranks"
model = ModelMPI(process_dict, control, maps=maps)
model.run(np.float64(1.0))
print("the run returned", flush=True)
'''


@pytest.mark.parametrize("where", ["rank0", "all_ranks"])
def test_serial_grid_raise_ends_the_job(
    tmp_path, dimensions, make_two_grid_toy, two_grid_weights, where
):
    """ "rank0": the default, the guard ends the job. "all_ranks":
    control["serial_grids_all_ranks"], where every rank raises the
    same error itself and the guard is not used."""
    launcher = shlex.split(os.environ.get("MPI_CMD", "mpirun -n 4"))
    if shutil.which(launcher[0]) is None:
        # the serial environment: no launcher, and no MPI to launch
        assert importlib.util.find_spec("mpi4py") is None
        return

    toy = make_two_grid_toy(dimensions)
    paths = write_two_grid_mpi_files(tmp_path / "data", toy, two_grid_weights)
    script = tmp_path / "raise_on_segment_grid.py"
    script.write_text(SCRIPT)

    # its own process group, so that a job that hangs is ended whole
    job = subprocess.Popen(
        [
            *launcher,
            sys.executable,
            str(script),
            str(paths["yaml_file"]),
            where,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = job.communicate(timeout=JOB_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(job.pid, signal.SIGKILL)
        job.communicate()
        raise AssertionError(
            f"the MPI job did not end within {JOB_TIMEOUT_S} s: the "
            "other ranks are waiting on rank 0"
        ) from None

    assert job.returncode != 0
    assert "the run returned" not in stdout
    assert MESSAGE in stderr
    assert ("raised on rank 0 alone" in stderr) == (where == "rank0")
