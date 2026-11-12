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

import numpy as np
import pytest
import xarray as xr
from conftest import cold_start_state
from pws_phoenix.model import Model
from pws_phoenix.process import DataArrayMeta, Process
from pws_phoenix.processes_concrete import Lower, Upper

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
            **cold_start_state(control, flow=toy_ds["flow_initial"]),
            "parameters": toy_ds[PARAM_NAMES],
        },
        "lower": {
            "class": Lower,
            "forcing_low": toy_ds["forcing_low"],
            **cold_start_state(control, storage=toy_ds["storage_initial"]),
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


def test_supplied_state_with_restart_raises(toy_ds, tmp_path):
    """One source of starting state: supplied state + restart_read
    raises (before the restart is read)."""
    process_dict = {
        "upper": {
            "class": Upper,
            "forcing_up": toy_ds["forcing_up"],
            "flow": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
    }
    with pytest.raises(ValueError, match="use one source"):
        Model(process_dict, {"restart_read": tmp_path})


class _DefaultsToZero(Process):
    """A restart variable whose initialize() default is zero."""

    level = DataArrayMeta(
        kind="variable", dims=("space",), dtype=np.float64, restart=True
    )

    def initialize(self) -> None:
        self._refs["level"][:] = 0.0

    def advance(self) -> None:
        pass

    def calculate(self, dt, time) -> None:
        pass


def test_supplied_state_overrides_initialize(toy_ds):
    """Supplied state is written AFTER initialize(): it wins over the
    default, as a restart does."""
    n_space = toy_ds.sizes["space"]
    supplied = xr.DataArray(np.arange(n_space, dtype=float), dims="space")
    process_dict = {
        # FIRST: its forcing and parameters give the grid its time axis
        # and "space" size before _DefaultsToZero allocates `level`
        "upper": {
            "class": Upper,
            "forcing_up": toy_ds["forcing_up"],
            "flow": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "proc": {"class": _DefaultsToZero, "level": supplied},
    }
    with Model(process_dict, {}) as model:
        level = model.model_dict["proc"]["level"].values
        np.testing.assert_array_equal(level, supplied.values)
        assert level is not supplied.values  # copied in (state), not shared


class _TwoLevels(Process):
    """initialize() computes `level_b` from `level_a`'s default, so
    the two are one state group."""

    level_a = DataArrayMeta(
        kind="variable", dims=("space",), dtype=np.float64, restart=True
    )
    level_b = DataArrayMeta(
        kind="variable", dims=("space",), dtype=np.float64, restart=True
    )
    state_groups = (("level_a", "level_b"),)

    def initialize(self) -> None:
        self._refs["level_a"][:] = 1.0
        self._refs["level_b"][:] = 2.0 * self._refs["level_a"]

    def advance(self) -> None:
        pass

    def calculate(self, dt, time) -> None:
        pass


def _two_levels_model(toy_ds, **state):
    process_dict = {
        # FIRST: gives the grid its time axis and "space" size
        "upper": {
            "class": Upper,
            "forcing_up": toy_ds["forcing_up"],
            "flow": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "proc": {"class": _TwoLevels, **state},
    }
    return Model(process_dict, {})


def test_partial_state_group_raises(toy_ds):
    """Part of a state group raises, naming what is missing; the whole
    group, or none of it, is accepted."""
    n_space = toy_ds.sizes["space"]
    level_a = xr.DataArray(np.full(n_space, 5.0), dims="space")
    level_b = xr.DataArray(np.full(n_space, 7.0), dims="space")

    with pytest.raises(ValueError, match=r"without \['level_b'\]"):
        _two_levels_model(toy_ds, level_a=level_a)

    with _two_levels_model(toy_ds) as model:
        np.testing.assert_array_equal(
            model.model_dict["proc"]["level_b"].values, 2.0
        )
    with _two_levels_model(toy_ds, level_a=level_a, level_b=level_b) as model:
        np.testing.assert_array_equal(
            model.model_dict["proc"]["level_a"].values, 5.0
        )
        np.testing.assert_array_equal(
            model.model_dict["proc"]["level_b"].values, 7.0
        )


class _NoDefault(Process):
    """`level` has no default (required state); a negative starting
    level is reported by check_initial_state()."""

    level = DataArrayMeta(
        kind="variable", dims=("space",), dtype=np.float64, restart=True
    )
    required_state = ("level",)

    def check_initial_state(self) -> list[str]:
        n_bad = int((self._refs["level"] < 0.0).sum())
        return [f"{n_bad} negative level(s)"] if n_bad else []

    def advance(self) -> None:
        pass

    def calculate(self, dt, time) -> None:
        pass


def _no_default_process_dict(toy_ds, control=None, **state):
    return {
        # FIRST: gives the grid its time axis and "space" size
        "upper": {
            "class": Upper,
            "forcing_up": toy_ds["forcing_up"],
            **cold_start_state(control or {}, flow=toy_ds["flow_initial"]),
            "parameters": toy_ds[PARAM_NAMES],
        },
        "proc": {"class": _NoDefault, **state},
    }


def test_required_state_missing_raises(toy_ds):
    """Required state not supplied raises at assembly; supplied, it is
    accepted and is what the model starts from."""
    with pytest.raises(ValueError, match=r"\['level'\] has no default"):
        Model(_no_default_process_dict(toy_ds), {})

    n_space = toy_ds.sizes["space"]
    level = xr.DataArray(np.full(n_space, 3.0), dims="space")
    with Model(_no_default_process_dict(toy_ds, level=level), {}) as model:
        np.testing.assert_array_equal(
            model.model_dict["proc"]["level"].values, 3.0
        )


def test_required_state_from_restart(toy_ds, tmp_path):
    """A restart satisfies required state: nothing is supplied on the
    warm start."""
    n_space = toy_ds.sizes["space"]
    level = xr.DataArray(np.full(n_space, 3.0), dims="space")
    restart_dir = tmp_path / "restarts"
    with Model(_no_default_process_dict(toy_ds, level=level), {}) as model:
        model.run(DT, np.int32(2))
        model.write_restart(restart_dir)
    control = {"restart_read": restart_dir}
    process_dict = _no_default_process_dict(toy_ds, control=control)
    with Model(process_dict, control) as model:
        np.testing.assert_array_equal(
            model.model_dict["proc"]["level"].values, 3.0
        )


def test_check_initial_state_raises(toy_ds):
    """A problem a process reports about the starting state in place
    (here: supplied) makes the Model raise, naming the process."""
    n_space = toy_ds.sizes["space"]
    level = xr.DataArray(np.full(n_space, -1.0), dims="space")
    with pytest.raises(
        ValueError, match=rf"process 'proc': {n_space} negative level"
    ):
        Model(_no_default_process_dict(toy_ds, level=level), {})


def test_state_groups_are_restart_variables():
    """Guard: every name in a declared state group, and every required
    state name, is one of that class's restart variables."""
    # importing registers the classes that declare groups
    from pws_phoenix.hydrology import (  # noqa: F401
        prms_runoff,
        prms_soilzone,
        prms_soilzone_ag,
    )

    declaring = [cc for cc in Process._registry.values() if cc.state_groups]
    assert {"PRMSSoilzone", "PRMSSoilzoneAg", "PRMSRunoffAg"} <= {
        cc.__name__ for cc in declaring
    }
    for name, cls in Process._registry.items():
        restart = set(cls.get_restart_variables())
        for group in cls.state_groups:
            assert set(group) <= restart, f"{name}: {group}"
        assert set(cls.required_state) <= restart, f"{name}: required"


def test_flags_are_variables_only():
    """Guard: restart flags only make sense on variables (declared or
    generated budget terms) -- every flagged field in the registry is
    one."""
    for name, cls in Process._registry.items():
        for vv in cls.get_restart_variables():
            assert vv in cls.get_var_names(include_budget=True), f"{name}.{vv}"
