"""Legacy PRMS ObsET ag config -> PARALLEL run (fgr analysis.control).

The full agricultural analysis shape (PRMSSoilzoneAgObsET iteration +
dynamic ag_frac + dynamic frost dates + observed AET) through the
translation parallel path: write_mpi_input_file carries the ag inputs
automatically (they are ordinary loose (time, nhru) kit entries, like
the CBH forcings) and mpi_model_from_control streams them per step.

The claim under test is BIT-identity of the hru-local ag physics per
rank block vs the serial one-liner. The It0 AET iteration is entirely
per-HRU and collective-free inside calculate(), so rank-LOCAL loop
exit is value-equivalent to the serial loop (a converged HRU
recomputes identically from the restored It0 state on every extra
pass) -- the "local-exit-equivalence" argument recorded when
ObsET-MPI was deferred in the ag arc, now tested. The window is
chosen to reach the growing season so the iteration is genuinely
live (asserted).

All collective MPI ops live in the module-scoped fixture; tests are
pure asserts.

Run with:
    mpirun -n 4 pytest --with-mpi tests/test_prms_translate_ag_mpi.py -v
"""

import pathlib as pl
import shutil
import sys
import tempfile

import numpy as np
import pytest
from mpi4py import MPI

sys.path.insert(0, str(pl.Path(__file__).parent.parent))
pytest.importorskip("pyPRMS", reason="pyPRMS not installed")

from prms_translate import (  # noqa: E402
    assemble_from_control,
    model_from_control,
    mpi_model_from_control,
    write_mpi_input_file,
)

MPIX_ROOT = pl.Path(__file__).parents[4]
FGR_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "fgr_ag_2yr"
CONTROL_FILE = FGR_DIR / "analysis.control"

# Jan 1 -> mid-July 2000: well into the growing season, so the AET
# iteration and irrigation are active (asserted in the fixture)
N_DAYS = 200
DT = np.float64(86400.0)

# hru-local physics + diagnostics compared bit-for-bit per rank
# block; transp_on covers the dynamic frost INPUTS under streaming
COMPARE_VARS = (
    "hru_actet",
    "ag_soil_moist",
    "ag_irrigation_add",
    "iter_count",
    "transp_on",
)

_needed = [
    CONTROL_FILE,
    FGR_DIR / "myparam.param",
    FGR_DIR / "prcp.cbh",
    FGR_DIR / "actet_openet.cbh",
    FGR_DIR / "dyn_ag_frac.param",
    FGR_DIR / "spring_frost.dyn",
    FGR_DIR / "fall_frost.dyn",
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason="fgr_ag_2yr legacy files missing: " + ", ".join(_missing[:3]),
)


@pytest.fixture(scope="module")
def mpi_run():
    """Prep (rank 0) + serial reference (rank 0, broadcast) + the
    parallel run (all ranks); every collective lives here."""
    comm = MPI.COMM_WORLD
    tmp = tempfile.mkdtemp() if comm.rank == 0 else None
    tmp = comm.bcast(tmp, root=0)
    assert tmp is not None
    data_dir = pl.Path(tmp)
    input_file = data_dir / "mpi_input.nc"

    serial = None
    if comm.rank == 0:
        kit = assemble_from_control(CONTROL_FILE)
        write_mpi_input_file(kit, input_file, n_days=N_DAYS)
        with model_from_control(CONTROL_FILE) as serial_model:
            serial_model.run(DT, np.int32(N_DAYS))
            soilzone = serial_model.model_dict["prms_soilzone"]
            atmosphere = serial_model.model_dict["prms_atmosphere"]
            serial = {
                nn: (
                    atmosphere[nn] if nn == "transp_on" else soilzone[nn]
                ).values.copy()
                for nn in COMPARE_VARS
            }
    serial = comm.bcast(serial, root=0)
    comm.Barrier()

    model = mpi_model_from_control(
        CONTROL_FILE,
        control={
            "input_file": input_file,
            "output_parallel_netcdf": data_dir / "mpi_output.nc",
            "output_var_names": ["sroff"],
            "mpi_grid": "nhru",
        },
    )
    model.run(DT)
    aa, bb = model._decomp_slice
    local = {nn: model._ds_mpi_stream[nn].values.copy() for nn in COMPARE_VARS}
    model.finalize()
    comm.Barrier()

    yield {"serial": serial, "local": local, "block": (aa, bb)}
    if comm.rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
class TestObsEtAgParallel:
    def test_iteration_is_live(self, mpi_run):
        """The window must actually exercise the AET iteration and
        irrigation, or the bit-identity claim is vacuous."""
        assert mpi_run["serial"]["iter_count"].max() >= 2
        assert (mpi_run["serial"]["ag_irrigation_add"] > 0.0).any()

    @pytest.mark.parametrize("name", COMPARE_VARS)
    def test_hru_local_bit_identical(self, mpi_run, name):
        """No cross-hru comm touches the ag physics: exact per-rank
        block vs the serial one-liner, iteration diagnostics
        included."""
        aa, bb = mpi_run["block"]
        np.testing.assert_array_equal(
            mpi_run["local"][name], mpi_run["serial"][name][aa:bb]
        )
