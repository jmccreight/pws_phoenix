"""PRMSChannelMPI (hydrology/prms_channel_mpi.py): the channel with
its segments divided over the ranks gives PRMSChannel's answers bit
for bit.

drb_2yr from its translation (mpi_model_from_translation), 765 hrus
over the ranks, 456 segments, for two control files:

- "nhm" (nhm.control): the channel is alone on the segment grid;
- "stream_temp" (nhm_stream_temp.control): hydraulic geometry and
  stream temperature follow it on the segment grid. They run on
  rank 0 only and read the channel's seg_outflow there, which is NOT
  among this case's output names: rank 0 is given it at every step
  because another process reads it.

Each run is made twice: with PRMSChannel, which rank 0 runs alone on
the whole network (the reference), and with PRMSChannelMPI in its
slot (replace_classes={"prms_channel": PRMSChannelMPI}).

Two periods, at window depth 7:

- "first": Mar 1 to Mar 30, 1979, a cold start; a restart is written
  at its end;
- "second": to Apr 30, warm-started from that restart
  (control["restart_read"]). mpi_model_from_translation cannot make
  this build: it always supplies the translation's initial state, and
  supplied state with a restart raises. So the second period is built
  here from the kit with the initial state taken out
  (_warm_model_from_translation).

Compared between the two classes, on rank 0, byte for byte:

- every step of the segment variables in the rank-0 zarr store;
- every variable of every segment-grid process after run() (for the
  channel: what gather_all() gives rank 0);
- the segment grid's restart file of the first period.

The second period also tests that a rank's blocks start from the
state of the restart, not from the cold start.

One build must raise on every rank: PRMSChannelMPI with
control["serial_grids_all_ranks"]. The other raises of
ModelMPI._decompose_processes are tested without MPI, in
tests/test_prms_channel_blocks.py.

All collective MPI ops live in the module-scoped fixture; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi \
        tests/test_prms_channel_divided_mpi.py -v
"""

import pathlib as pl
import shutil
import tempfile
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from mpi4py import MPI

from conftest import PWS_TEST_DATA
from pws_phoenix.hydrology.prms_channel import PRMSChannel
from pws_phoenix.hydrology.prms_channel_mpi import PRMSChannelMPI
from pws_phoenix.model import ModelMPI
from pws_phoenix.prms_translate import (
    assemble_from_translation,
    mpi_model_from_translation,
)
from pws_phoenix.prms_translate.assemble_mpi import _mpi_model_from_kit
from pws_phoenix.prms_translate.translate_domain import translate_domain

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
DT = np.float64(86400.0)

FIRST_DAY = "1979-03-01"
RESTART_DAY = "1979-03-30"
LAST_DAY = "1979-04-30"
N_FIRST = 30
N_SECOND = 31

# case -> (control file, the segment variables written)
CASES: dict[str, tuple[str, tuple[str, ...]]] = {
    "nhm": ("nhm.control", ("seg_outflow", "seg_stor_change")),
    "stream_temp": (
        "nhm_stream_temp.control",
        ("seg_stor_change", "seg_tave_water"),
    ),
}
CLASSES = {"reference": PRMSChannel, "divided": PRMSChannelMPI}
PERIODS = ("first", "second")
TESTED = [(case, period) for case in CASES for period in PERIODS]
STREAMED_NAME = "gwres_flow"
CHANNEL_NAMES = PRMSChannel.get_var_names()

_needed = [DRB_DIR / control_file for control_file, _names in CASES.values()]
_needed += [
    DRB_DIR / "myparam.param",
    DRB_DIR / "prcp.cbh",
    DRB_DIR / "tmax.cbh",
    DRB_DIR / "tmin.cbh",
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason="drb_2yr legacy files missing: " + ", ".join(_missing[:3]),
)


def _same_bytes(aa: np.ndarray, bb: np.ndarray) -> bool:
    """The same dtype, shape and bytes: nan equals nan, -0.0 is not
    0.0."""
    return (
        aa.dtype == bb.dtype
        and aa.shape == bb.shape
        and aa.tobytes() == bb.tobytes()
    )


def _warm_model_from_translation(
    yaml_path: pl.Path, control: dict[str, Any], cls: type
) -> ModelMPI:
    """mpi_model_from_translation without the translation's initial
    state, for a build with control["restart_read"]."""
    kit = assemble_from_translation(yaml_path)
    for entry in kit.process_dict.values():
        for name in entry["class"].get_restart_variables():
            entry.pop(name, None)
    with open(yaml_path) as ff:
        info = yaml.safe_load(ff)
    control = (
        kit.control
        | control
        | {
            "input_file": yaml_path.parent / info["forcing_file"],
            "mpi_grid": "nhru",
        }
    )
    return _mpi_model_from_kit(
        kit,
        control,
        keep_parameters=True,
        replace_classes={"prms_channel": cls},
    )


def _run(model: ModelMPI, restart_dir: pl.Path | None) -> dict[str, Any]:
    """Run, write the restart, record (collective)."""
    comm = MPI.COMM_WORLD
    channel = model.model_dict["prms_channel"]
    segment_procs = {
        slot: proc
        for slot, proc in model.model_dict.items()
        if model._proc_grid[slot] == "nsegment"
    }
    result: dict[str, Any] = {
        "class": type(channel).__name__,
        "segment_slots": list(segment_procs),
        # the channel's variables that another segment-grid process
        # reads
        "read_by_others": sorted(
            {
                name
                for slot, proc in segment_procs.items()
                if slot != "prms_channel"
                for name in proc.get_inputs()
                if name in CHANNEL_NAMES
            }
        ),
    }
    if isinstance(channel, PRMSChannelMPI):
        # the segments each rank routes, and rank 0's main stem
        result["block_sizes"] = comm.allgather(channel._block.n_own)
        result["main_stem_size"] = comm.bcast(
            None if channel._main is None else channel._main.n_own, root=0
        )
        result["names_each_step"] = channel._each_step.names
    model.run(DT)
    # this rank's values of every segment-grid variable after run()
    result["state"] = {
        name: proc[name].values.copy()
        for proc in segment_procs.values()
        for name in type(proc).get_var_names(include_budget=True)
    }
    if restart_dir is not None:
        model.write_restart(restart_dir)
    model.finalize()
    comm.Barrier()
    return result


@pytest.fixture(scope="module")
def divided_runs():
    """Per case: the translation (rank 0), then the two periods for
    each class. Then the build that must raise (all ranks). Every
    collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None

    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    yaml_paths: dict[str, pl.Path] = {}
    for case, (control_file, serial_names) in CASES.items():
        data_dir = pl.Path(tmp) / case
        yaml_path = None
        if comm.rank == 0:
            data_dir.mkdir()
            yaml_path = translate_domain(
                DRB_DIR / control_file, data_dir / "domain"
            )
        yaml_path = comm.bcast(yaml_path, root=0)
        assert yaml_path is not None
        yaml_paths[case] = yaml_path
        for tag, cls in CLASSES.items():
            restart_dir = data_dir / f"restart_{tag}"
            for period in PERIODS:
                first = period == "first"
                control: dict[str, Any] = {
                    "start_time": FIRST_DAY,
                    "end_time": RESTART_DAY if first else LAST_DAY,
                    "mpi_window_steps": 7,
                    "output_parallel_netcdf": data_dir / f"{tag}_{period}.nc",
                    "output_serial_zarr": data_dir / f"{tag}_{period}.zarr",
                    "output_var_names": [STREAMED_NAME, *serial_names],
                }
                if first:
                    model = mpi_model_from_translation(
                        yaml_path,
                        control,
                        replace_classes={"prms_channel": cls},
                    )
                else:
                    control["restart_read"] = restart_dir
                    model = _warm_model_from_translation(
                        yaml_path, control, cls
                    )
                runs[(case, tag, period)] = _run(
                    model, restart_dir if first else None
                ) | {
                    "zarr": control["output_serial_zarr"],
                    "restart_dir": restart_dir,
                }

    # the build that must raise, on every rank
    error = None
    try:
        mpi_model_from_translation(
            yaml_paths["nhm"],
            {"serial_grids_all_ranks": True},
            replace_classes={"prms_channel": PRMSChannelMPI},
        )
    except NotImplementedError as err:
        error = str(err)
    comm.Barrier()

    yield {"runs": runs, "all_ranks_error": error}
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestPRMSChannelDivided:
    @pytest.mark.parametrize(("case", "period"), TESTED)
    def test_the_classes_asked_for_ran(self, divided_runs, case, period):
        for tag, cls in CLASSES.items():
            run = divided_runs["runs"][(case, tag, period)]
            assert run["class"] == cls.__name__

    def test_the_cases_differ_in_the_segment_grid(self, divided_runs):
        """Alone in one, followed by two processes in the other, one
        of which reads a channel variable that is not written."""
        runs = divided_runs["runs"]
        nhm = runs[("nhm", "divided", "first")]
        assert nhm["segment_slots"] == ["prms_channel"]
        assert nhm["read_by_others"] == []
        stream_temp = runs[("stream_temp", "divided", "first")]
        assert stream_temp["segment_slots"] == [
            "prms_channel",
            "prms_hydraulic_geometry",
            "prms_stream_temp",
        ]
        assert "seg_outflow" in stream_temp["read_by_others"]
        assert "seg_outflow" not in CASES["stream_temp"][1]

    @pytest.mark.parametrize(("case", "period"), TESTED)
    def test_the_segments_are_divided(self, divided_runs, case, period):
        """Every segment is routed by one rank's block or by the main
        stem, and more than one rank routes segments."""
        run = divided_runs["runs"][(case, "divided", period)]
        n_seg = run["state"]["seg_outflow"].shape[0]
        sizes = run["block_sizes"]
        assert len(sizes) == MPI.COMM_WORLD.size
        assert sum(sizes) + run["main_stem_size"] == n_seg
        assert sum(size > 0 for size in sizes) >= 2
        assert run["main_stem_size"] > 0

    @pytest.mark.parametrize(("case", "period"), TESTED)
    def test_rank_0_is_given_what_it_reads(self, divided_runs, case, period):
        """At each step: the channel's variables that are written or
        that another segment-grid process reads, in the order of the
        class's declarations."""
        run = divided_runs["runs"][(case, "divided", period)]
        read = set(CASES[case][1]) | set(run["read_by_others"])
        assert run["names_each_step"] == tuple(
            name for name in CHANNEL_NAMES if name in read
        )
        assert 0 < len(run["names_each_step"]) < len(CHANNEL_NAMES)

    @pytest.mark.parametrize(("case", "period"), TESTED)
    def test_every_step_of_the_output(self, divided_runs, case, period):
        if MPI.COMM_WORLD.rank != 0:
            return
        stores = {
            tag: xr.open_zarr(
                divided_runs["runs"][(case, tag, period)]["zarr"],
                consolidated=False,
            )
            for tag in CLASSES
        }
        # a warm start's store holds the steps it ran
        n_days = N_FIRST if period == "first" else N_SECOND
        for name in CASES[case][1]:
            reference = stores["reference"][name].values
            assert reference.shape[0] == n_days
            # the run wrote something that varies
            assert np.unique(reference[-1]).shape[0] > 10
            assert _same_bytes(stores["divided"][name].values, reference), name

    @pytest.mark.parametrize(("case", "period"), TESTED)
    def test_every_variable_after_run(self, divided_runs, case, period):
        """On rank 0, after run(): all the channel's variables, not
        only those it was given at each step, and every variable of
        the processes that follow the channel."""
        if MPI.COMM_WORLD.rank != 0:
            return
        reference = divided_runs["runs"][(case, "reference", period)]["state"]
        divided = divided_runs["runs"][(case, "divided", period)]["state"]
        assert set(CHANNEL_NAMES) <= set(reference)
        assert sorted(divided) == sorted(reference)
        for name in reference:
            assert _same_bytes(divided[name], reference[name]), name

    @pytest.mark.parametrize("case", list(CASES))
    def test_restart_file(self, divided_runs, case):
        if MPI.COMM_WORLD.rank != 0:
            return
        found = {}
        for tag in CLASSES:
            run = divided_runs["runs"][(case, tag, "first")]
            (path,) = run["restart_dir"].glob("*_restart_nsegment.nc")
            found[tag] = xr.load_dataset(path)
        names = sorted(map(str, found["reference"].data_vars))
        assert set(PRMSChannel.get_restart_variables()) <= set(names)
        assert sorted(map(str, found["divided"].data_vars)) == names
        for name in names:
            assert _same_bytes(
                found["divided"][name].values, found["reference"][name].values
            ), name
        assert found["divided"].attrs == found["reference"].attrs

    def test_all_ranks_option_raises(self, divided_runs):
        """On every rank (a rank that did not raise would hang)."""
        error = divided_runs["all_ranks_error"]
        assert error is not None
        assert "serial_grids_all_ranks" in error
        assert "PRMSChannelMPI" in error
