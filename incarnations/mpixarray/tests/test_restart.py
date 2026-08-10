"""Restart framework: flags, write/read, perfect restart (toy model).

The R1 mechanism tests, on the Upper/Lower toys (no external data):

- ``DataArrayMeta(restart=True)`` -> ``get_restart_variables()`` (the
  declaration-derived prognostic set; flags live on CURRENT state
  variables only -- advance() regenerates the ``*_previous`` copies).
- ``Model.write_restart(dir)`` writes self-locating per-grid files
  (state timestamp in attrs); ``control["restart_read"]`` restores
  the state, fast-forwards the inputs, and resumes at the FOLLOWING
  step of the model's own time axis.
- THE PERFECT-RESTART recipe (pywatershed's standard): a continuous
  run a->c must be BIT-IDENTICAL to run a->b + write + a fresh model
  warm-started b->c. This is what polices flag completeness for every
  process (a forgotten prognostic flag diverges here).
- Loud failures: a flag set that changed between write and read; an
  empty restart directory.

Per-process perfect-restart tests (real processes, drb data) ride the
same recipe -- see the restart arc (R2).
"""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
from model import Model
from process import Process
from processes_concrete import Lower, Upper

PARAM_NAMES = [
    "param_up_0",
    "param_up_1",
    "param_low_0",
    "param_shared_name",
]
STATE_NAMES = ("flow", "flow_previous", "storage", "storage_previous")
DT = np.float64(1.0)


def _make_model(toy_ds, control):
    process_dict = {
        "upper": {
            "class": Upper,
            "forcing_up": toy_ds["forcing_up"],
            "flow_initial": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "lower": {
            "class": Lower,
            "forcing_low": toy_ds["forcing_low"],
            "storage_initial": toy_ds["storage_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
    }
    return Model(process_dict, control)


@pytest.fixture(scope="module")
def toy_ds(dimensions, make_toy_input):
    return make_toy_input(dimensions)


def test_restart_variables_from_flags():
    """The prognostic set derives from the declarations."""
    assert Upper.get_restart_variables() == ("flow",)
    assert Lower.get_restart_variables() == ("storage",)
    # default hooks: no python-attr state; unexpected state raises
    proc = Upper.__new__(Upper)
    assert proc.get_restart_state() == {}
    with pytest.raises(ValueError, match="no set_restart_state"):
        proc.set_restart_state({"bogus": np.zeros(3)})


def test_perfect_restart(toy_ds, dimensions, tmp_path):
    """Continuous a->c == (a->b + write) then fresh model b->c,
    bit-identical in every state variable."""
    ntime = dimensions["n_time"]
    idx_b = ntime // 2
    restart_dir = tmp_path / "restarts"

    with _make_model(toy_ds, {}) as model_ac:
        model_ac.run(DT, np.int32(ntime))

    with _make_model(toy_ds, {}) as model_ab:
        model_ab.run(DT, np.int32(idx_b))
        model_ab.write_restart(restart_dir)

    # the file is self-locating: one per grid with restart content,
    # stamped with the state time of the last completed step (idx_b-1)
    files = sorted(restart_dir.glob("*_restart_space.nc"))
    assert len(files) == 1
    with xr.load_dataset(files[0]) as ds_rst:
        assert set(ds_rst.data_vars) == {"flow", "storage"}
        assert (
            np.datetime64(ds_rst.attrs["state_time"])
            == (toy_ds["time"].values[idx_b - 1])
        )

    with _make_model(toy_ds, {"restart_read": restart_dir}) as model_bc:
        assert model_bc._start_index == idx_b
        model_bc.run(DT, np.int32(ntime - idx_b))

    for nn in STATE_NAMES:
        proc = "upper" if nn.startswith("flow") else "lower"
        np.testing.assert_array_equal(
            model_bc.model_dict[proc][nn].values,
            model_ac.model_dict[proc][nn].values,
            err_msg=f"'{nn}' not bit-identical after restart",
        )


def test_restart_flag_change_fails_loudly(toy_ds, dimensions, tmp_path):
    """A restart file whose variable set differs from the model's
    current flags is rejected."""
    restart_dir = tmp_path / "restarts"
    with _make_model(toy_ds, {}) as model_ab:
        model_ab.run(DT, np.int32(2))
        model_ab.write_restart(restart_dir)
    ff = next(restart_dir.glob("*_restart_space.nc"))
    doctored = xr.load_dataset(ff).drop_vars("flow")
    ff.unlink()
    doctored.to_netcdf(ff)
    with pytest.raises(ValueError, match="variable set"):
        _make_model(toy_ds, {"restart_read": restart_dir})


def test_restart_empty_dir_fails_loudly(toy_ds, tmp_path):
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    with pytest.raises(ValueError, match="needs exactly one"):
        _make_model(toy_ds, {"restart_read": empty})


def test_flags_are_variables_only():
    """Guard: restart flags only make sense on kind='variable' --
    every flagged field in the registry is a variable."""
    for name, cls in Process._registry.items():
        for vv in cls.get_restart_variables():
            assert vv in cls.get_var_names(), f"{name}.{vv}"
