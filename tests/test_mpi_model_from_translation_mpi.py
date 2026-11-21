"""prms_translate.mpi_model_from_translation: a domain translated
ahead of time (translate_domain) run under MPI.

There is no combined input file. The stream is the translation's
forcing file, cbh.nc on (time, nhru), or with transposed=True its
copy cbh_transpose.nc on (nhru, time); the hru parameters stay in
process_dict and each rank keeps its block.

The NHM on drb_2yr (nhm.control), 765 hrus over the ranks. The
reference of every run is mpi_model_from_control on the combined
file of write_mpi_input_file at depth 1, compared bit for bit: the
final values of every variable on every rank (both grids), every
step of two streamed variables and the nhru coordinate in the NetCDF
file, and every step of one segment variable in the rank-0 zarr
store.

Each forcing file is run twice:

- over two months given in control, at window depth 7;
- over the control file's own dates, which are the file's whole
  record, with mpi_window_steps=None (one window = the whole run).

The build warns once, on rank 0, when the file chosen is the slower
one for the reads (data_io.slow_order_note): only the whole record
from cbh.nc is.

With nothing but the yaml the model builds over the control file's
dates and opens no output.

Builds that must raise, on every rank: transposed=True when the yaml
names no transposed copy, control["input_file"], and a forcing file
with no nhru coordinate.

All collective MPI ops live in the module-scoped fixture; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi \
        tests/test_mpi_model_from_translation_mpi.py -v
"""

import pathlib as pl
import shutil
import tempfile
import warnings
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from mpi4py import MPI

from conftest import PWS_TEST_DATA
from pws_phoenix.data_io import READ_PIECES_WARN_RATIO, read_pieces
from pws_phoenix.model import ModelMPI
from pws_phoenix.prms_translate import (
    assemble_from_control,
    mpi_model_from_control,
    mpi_model_from_translation,
    write_mpi_input_file,
)
from pws_phoenix.prms_translate.translate_domain import translate_domain

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
CONTROL_FILE = DRB_DIR / "nhm.control"
DT = np.float64(86400.0)

# the control file's own dates: the whole record of the CBH files
RECORD_FIRST = "1979-01-01"
RECORD_LAST = "1980-12-31"
RECORD_N_DAYS = 731
# the two months given in control
MONTHS_FIRST = "1979-03-01"
MONTHS_LAST = "1979-04-30"
MONTHS_N_DAYS = 61

# period -> (dates given in control, first day, last day, days, the
#            window depth of the runs from the translation)
PERIODS: dict[str, tuple[dict[str, str], str, str, int, int | None]] = {
    "two_months": (
        {"start_time": MONTHS_FIRST, "end_time": MONTHS_LAST},
        MONTHS_FIRST,
        MONTHS_LAST,
        MONTHS_N_DAYS,
        7,
    ),
    # None = one window = the whole run
    "whole_record": ({}, RECORD_FIRST, RECORD_LAST, RECORD_N_DAYS, None),
}
# file -> the transposed= argument
FILES = {"cbh.nc": False, "cbh_transpose.nc": True}
TESTED = [(period, name) for period in PERIODS for name in FILES]

STREAMED_NAMES = ["sroff", "gwres_flow"]
SERIAL_NAME = "seg_outflow"
BAD_BUILD_TAGS = ["no_transposed_copy", "input_file_given", "no_coordinate"]

_needed = [
    CONTROL_FILE,
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


def _output_control(data_dir: pl.Path, tag: str) -> dict[str, Any]:
    return {
        "output_parallel_netcdf": data_dir / f"output_{tag}.nc",
        "output_serial_zarr": data_dir / f"output_{tag}.zarr",
        "output_var_names": STREAMED_NAMES + [SERIAL_NAME],
    }


def _built(model: ModelMPI, caught: list) -> dict[str, Any]:
    """What a build showed (collective: rank 0's block, the size of
    the grid)."""
    comm = MPI.COMM_WORLD
    start, stop = model._decomp_slice
    return {
        "input_file": pl.Path(model._control["input_file"]),
        "ntime": int(model.ntime),
        "first": model.times.values[0],
        "last": model.times.values[-1],
        "window_steps": model._window_steps,
        "stream_time_first": dict(model._stream_time_first),
        "n_slow_order_warnings": sum(
            "slower order" in str(ww.message) for ww in caught
        ),
        "n_block_rank_0": comm.bcast(stop - start, root=0),
        "n_grid": comm.allreduce(stop - start, op=MPI.SUM),
        "writer_open": model._ds_mpi_stream.mpi.has_writer,
        "has_zarr_output": model.output is not None,
    }


def _final_state(model: ModelMPI) -> dict[str, np.ndarray]:
    """This rank's values of every variable of every grid."""
    state: dict[str, np.ndarray] = {}
    for proc_name, proc in model.model_dict.items():
        grid = model._proc_grid[proc_name]
        dataset = model.discretizations[grid].dataset
        for name in type(proc).get_var_names(include_budget=True):
            state[f"{grid}/{name}"] = dataset[name].values.copy()
    return state


def _run(model: ModelMPI, built: dict[str, Any]) -> dict[str, Any]:
    model.run(DT)
    result = built | {"state": _final_state(model)}
    model.finalize()
    MPI.COMM_WORLD.Barrier()
    return result


@pytest.fixture(scope="module")
def translation_runs():
    """Prep (rank 0), the reference and the two files for each
    period, a build from the yaml alone, then the builds that must
    raise (all ranks); every collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp)
    domain_dir = data_dir / "domain"
    combined_file = data_dir / "mpi_input.nc"
    # two more yamls in the domain directory, each a copy with one
    # entry changed (the control copy they are checked against is
    # not changed)
    no_copy_yaml = domain_dir / "no_transposed_copy_translation.yaml"
    no_coordinate_yaml = domain_dir / "no_coordinate_translation.yaml"

    yaml_path = None
    if comm.rank == 0:
        yaml_path = translate_domain(
            CONTROL_FILE, domain_dir, cbh_transpose=True
        )
        write_mpi_input_file(
            assemble_from_control(CONTROL_FILE), combined_file
        )
        with open(yaml_path) as ff:
            info = yaml.safe_load(ff)
        with open(no_copy_yaml, "w") as ff:
            yaml.safe_dump(info | {"forcing_file_transpose": None}, ff)
        xr.load_dataset(domain_dir / info["forcing_file"]).drop_vars(
            "nhru"
        ).to_netcdf(domain_dir / "cbh_no_coordinate.nc")
        with open(no_coordinate_yaml, "w") as ff:
            yaml.safe_dump(info | {"forcing_file": "cbh_no_coordinate.nc"}, ff)
    yaml_path = comm.bcast(yaml_path, root=0)
    assert yaml_path is not None

    runs: dict[Any, dict[str, Any]] = {}
    for period, (dates, _first, _last, _n_days, depth) in PERIODS.items():
        # the reference: the combined file, depth 1
        control = {
            "input_file": combined_file,
            "mpi_grid": "nhru",
            **dates,
            **_output_control(data_dir, f"{period}_reference"),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = mpi_model_from_control(CONTROL_FILE, control=control)
        runs[(period, "reference")] = _run(model, _built(model, caught)) | {
            "control": control
        }
        for name, transposed in FILES.items():
            control = {
                "mpi_window_steps": depth,
                **dates,
                **_output_control(data_dir, f"{period}_{name}"),
            }
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = mpi_model_from_translation(
                    yaml_path, control, transposed=transposed
                )
            runs[(period, name)] = _run(model, _built(model, caught)) | {
                "control": control
            }

    # nothing but the yaml: built, not run
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = mpi_model_from_translation(yaml_path)
    yaml_alone = _built(model, caught)
    model.finalize()
    comm.Barrier()

    # Builds that must raise. The first two raise before anything is
    # opened; the third in ModelMPI.__init__, which closes the file
    # it opened (tests/test_bad_build_mpi.py).
    bad_builds: dict[str, tuple[type[Exception], pl.Path, dict, bool]] = {
        "no_transposed_copy": (FileNotFoundError, no_copy_yaml, {}, True),
        "input_file_given": (
            ValueError,
            yaml_path,
            {"input_file": combined_file},
            False,
        ),
        "no_coordinate": (ValueError, no_coordinate_yaml, {}, False),
    }
    errors: dict[str, str] = {}
    for tag, (error, bad_yaml, control, transposed) in bad_builds.items():
        try:
            mpi_model_from_translation(
                bad_yaml, control, transposed=transposed
            )
        except error as err:
            errors[tag] = str(err)
    comm.Barrier()

    if comm.rank == 0:
        left = sorted(pp.name for pp in data_dir.iterdir())
    else:
        left = None
    yield {
        "runs": runs,
        "yaml_alone": yaml_alone,
        "errors": errors,
        "domain_dir": domain_dir,
        "left": comm.bcast(left, root=0),
    }
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestMpiModelFromTranslation:
    @pytest.mark.parametrize(("period", "name"), TESTED)
    def test_the_file_asked_for_is_streamed(
        self, translation_runs, period, name
    ):
        run = translation_runs["runs"][(period, name)]
        assert run["input_file"] == translation_runs["domain_dir"] / name
        found = run["stream_time_first"]
        assert sorted(found) == ["prcp", "tmax", "tmin"]
        # cbh.nc is on (time, nhru), its copy on (nhru, time)
        assert set(found.values()) == {not FILES[name]}

    @pytest.mark.parametrize(("period", "name"), TESTED)
    def test_time_axis(self, translation_runs, period, name):
        """The dates given in control, else the control file's."""
        _dates, first, last, n_days, depth = PERIODS[period]
        for tag in (name, "reference"):
            run = translation_runs["runs"][(period, tag)]
            assert run["ntime"] == n_days
            assert run["first"] == np.datetime64(first)
            assert run["last"] == np.datetime64(last)
        run = translation_runs["runs"][(period, name)]
        assert run["window_steps"] == (n_days if depth is None else depth)
        assert (
            translation_runs["runs"][(period, "reference")]["window_steps"]
            == 1
        )
        # depth 7 must leave a shallower last window
        assert MONTHS_N_DAYS % 7 != 0

    @pytest.mark.parametrize(("period", "name"), TESTED)
    def test_slow_order_warning(self, translation_runs, period, name):
        """One warning on rank 0, none elsewhere, and only when the
        file chosen takes READ_PIECES_WARN_RATIO times the pieces of
        the other for rank 0's block and this depth. The file's time
        axis is the whole record in every run."""
        run = translation_runs["runs"][(period, name)]
        pieces = {
            time_first: read_pieces(
                run["n_block_rank_0"],
                run["n_grid"],
                run["window_steps"],
                RECORD_N_DAYS,
                time_first,
            )
            for time_first in (True, False)
        }
        time_first = not FILES[name]
        slow = (
            pieces[time_first]
            >= READ_PIECES_WARN_RATIO * pieces[not time_first]
        )
        # of the four runs only the whole record from cbh.nc is slow:
        # one piece per day against one piece
        assert slow == ((period, name) == ("whole_record", "cbh.nc"))
        expected = 1 if (slow and MPI.COMM_WORLD.rank == 0) else 0
        assert run["n_slow_order_warnings"] == expected

    @pytest.mark.parametrize(("period", "name"), TESTED)
    def test_final_state_bit_identical(self, translation_runs, period, name):
        """Every variable of both grids, this rank's values."""
        runs = translation_runs["runs"]
        reference = runs[(period, "reference")]["state"]
        state = runs[(period, name)]["state"]
        assert state.keys() == reference.keys()
        for var_name, values in reference.items():
            np.testing.assert_array_equal(
                state[var_name], values, err_msg=var_name
            )

    @pytest.mark.parametrize(("period", "name"), TESTED)
    def test_streamed_output_bit_identical(
        self, translation_runs, period, name
    ):
        """Every step in the NetCDF file, its time coordinate and its
        nhru coordinate (the one the forcing file holds)."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = translation_runs["runs"]
        ds_ref = xr.load_dataset(
            runs[(period, "reference")]["control"]["output_parallel_netcdf"]
        )
        ds_out = xr.load_dataset(
            runs[(period, name)]["control"]["output_parallel_netcdf"]
        )
        assert ds_out.sizes["time"] == PERIODS[period][3]
        for coordinate in ("time", "nhru"):
            np.testing.assert_array_equal(
                ds_out[coordinate].values, ds_ref[coordinate].values
            )
        np.testing.assert_array_equal(
            ds_out["nhru"].values, np.arange(ds_out.sizes["nhru"])
        )
        for var_name in STREAMED_NAMES:
            buf = f"{var_name}_out"
            np.testing.assert_array_equal(
                ds_out[buf].values, ds_ref[buf].values, err_msg=var_name
            )

    @pytest.mark.parametrize(("period", "name"), TESTED)
    def test_serial_zarr_bit_identical(self, translation_runs, period, name):
        """Every step of a segment variable in the rank-0 zarr store."""
        if MPI.COMM_WORLD.rank != 0:
            return
        runs = translation_runs["runs"]
        with (
            xr.open_zarr(
                runs[(period, "reference")]["control"]["output_serial_zarr"],
                consolidated=False,
            ) as ds_ref,
            xr.open_zarr(
                runs[(period, name)]["control"]["output_serial_zarr"],
                consolidated=False,
            ) as ds_out,
        ):
            assert ds_out.sizes["time"] == PERIODS[period][3]
            np.testing.assert_array_equal(
                ds_out["time"].values, ds_ref["time"].values
            )
            np.testing.assert_array_equal(
                ds_out[SERIAL_NAME].values, ds_ref[SERIAL_NAME].values
            )

    def test_yaml_alone(self, translation_runs):
        """No control: cbh.nc, the control file's dates, depth 1 and
        no output of either kind."""
        built = translation_runs["yaml_alone"]
        assert built["input_file"] == translation_runs["domain_dir"] / "cbh.nc"
        assert built["ntime"] == RECORD_N_DAYS
        assert built["first"] == np.datetime64(RECORD_FIRST)
        assert built["last"] == np.datetime64(RECORD_LAST)
        assert built["window_steps"] == 1
        assert built["n_slow_order_warnings"] == 0
        assert not built["writer_open"]
        assert not built["has_zarr_output"]
        # nothing outside the domain directory but the combined file
        # and the outputs of the six runs
        expected = ["domain", "mpi_input.nc"]
        for period in PERIODS:
            for tag in ("reference", *FILES):
                expected += [
                    f"output_{period}_{tag}.nc",
                    f"output_{period}_{tag}.zarr",
                ]
        assert translation_runs["left"] == sorted(expected)

    @pytest.mark.parametrize("tag", BAD_BUILD_TAGS)
    def test_bad_build_raises(self, translation_runs, tag):
        message = translation_runs["errors"].get(tag, "")
        expected = {
            "no_transposed_copy": "names no forcing_file_transpose",
            "input_file_given": "control gives 'input_file'",
            "no_coordinate": "has no coordinate for the distributed grid",
        }
        assert expected[tag] in message
