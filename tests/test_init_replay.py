"""Acceptance test of the init hook split: both replays.

The three hooks (Process docstrings) exist so that a built model can
be run again without being rebuilt:

- STATE REPLAY (what a later ``Model.reset()`` will do): after any
  number of steps, ``init_state(time)`` + ``initialize_budgets()`` +
  the supplied initial state must put every variable, and every array
  of ``get_restart_state()``, back to its value right after the
  build -- bit for bit. Before the replay every variable is
  overwritten with its allocation fill and the private state is
  poisoned, so a default that only the build put in place is missed.
- PARAMETER REPLAY (what a later ``Model.set_parameters()`` will do):
  ``init_from_params()`` run a second time must reproduce every
  ``parameter_internal`` bit for bit, replace no private array (only
  ``allocate()`` may allocate) and change nothing else (it writes no
  variable).

Both replays are helpers here, not Model methods: ``reset()`` and
``set_parameters()`` arrive with the calibration reuse loop.

This module: the toy model (no external data) plus a toy that uses
each hook for its purpose, and three toys that break one rule each --
the replays must catch all three.
"""

import numpy as np
import pytest
import xarray as xr
from pws_phoenix.clock import Time
from pws_phoenix.model import Model
from pws_phoenix.process import _FILL_VALUE, DataArrayMeta, Process
from pws_phoenix.processes_concrete import Lower, Upper

DT = np.float64(1.0)
N_STEPS_BEFORE_REPLAY = np.int32(3)
PARAM_NAMES = [
    "param_up_0",
    "param_up_1",
    "param_low_0",
    "param_shared_name",
]


# ----------------------------------------------------------------------
# the replays
# ----------------------------------------------------------------------


def _variables(model: Model) -> dict[tuple[str, str], np.ndarray]:
    """Every variable of the model, budget variables included, keyed
    (grid, name): a name shared by two processes of a grid is ONE
    array."""
    out = {}
    for key, proc in model.model_dict.items():
        grid = model._proc_grid[key]
        for name in proc.get_variables(include_budget=True):
            out[(grid, name)] = proc._refs[name]
    return out


def _variable_fills(model: Model) -> dict[tuple[str, str], object]:
    out = {}
    for key, proc in model.model_dict.items():
        grid = model._proc_grid[key]
        for name, meta in proc.get_variables(include_budget=True).items():
            out[(grid, name)] = _FILL_VALUE[meta.dtype]
    return out


def _private_state(model: Model) -> dict[tuple[str, str], np.ndarray]:
    """Every array of every process's get_restart_state(), keyed
    (process key, name)."""
    return {
        (key, name): array
        for key, proc in model.model_dict.items()
        for name, array in proc.get_restart_state().items()
    }


def _state(model: Model) -> dict[tuple[str, str], np.ndarray]:
    return _variables(model) | _private_state(model)


def snapshot_state(model: Model) -> dict[tuple[str, str], np.ndarray]:
    return {key: array.copy() for key, array in _state(model).items()}


def _poison(array: np.ndarray) -> None:
    if np.issubdtype(array.dtype, np.floating):
        array[...] = np.nan
    elif np.issubdtype(array.dtype, np.integer):
        array[...] = np.iinfo(array.dtype).min
    else:
        array[...] = ~array


def replay_state(model: Model) -> None:
    """Fill every variable with its allocation fill and poison the
    private state, then do what the build does after the freeze:
    init_state(time) and initialize_budgets() per process with the
    clock at index 0, then the supplied initial state."""
    fills = _variable_fills(model)
    for key, array in _variables(model).items():
        array[...] = fills[key]
    for array in _private_state(model).values():
        _poison(array)

    model.time.set_index(0)
    for proc in model.model_dict.values():
        proc.init_state(model.time)
        proc.initialize_budgets()
    model._apply_supplied_state()


def assert_state_equal(
    model: Model, snapshot: dict[tuple[str, str], np.ndarray]
) -> None:
    state = _state(model)
    assert state.keys() == snapshot.keys()
    for key, expected in snapshot.items():
        np.testing.assert_array_equal(
            state[key],
            expected,
            err_msg=f"{key} differs from its value after the build",
        )


def _parameters_internal(model: Model) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for key, proc in model.model_dict.items():
        grid = model._proc_grid[key]
        for name in proc.get_parameters_internal():
            out[(grid, name)] = proc._refs[name]
    return out


def _parameter_internal_fills(model: Model) -> dict[tuple[str, str], object]:
    out = {}
    for key, proc in model.model_dict.items():
        grid = model._proc_grid[key]
        for name, meta in proc.get_parameters_internal().items():
            out[(grid, name)] = _FILL_VALUE[meta.dtype]
    return out


def _collect(value: object, path: str, arrays: dict, scalars: dict) -> None:
    """Arrays and plain numbers reachable from one attribute, through
    tuples, lists and dicts (a tuple of array references, _refs, a
    graph-state namedtuple)."""
    if isinstance(value, np.ndarray):
        arrays[path] = value
    elif isinstance(value, bool | int | float | np.generic):
        scalars[path] = value
    elif isinstance(value, tuple | list):
        for ii, item in enumerate(value):
            _collect(item, f"{path}[{ii}]", arrays, scalars)
    elif isinstance(value, dict):
        for name, item in value.items():
            _collect(item, f"{path}[{name!r}]", arrays, scalars)


def _private(model: Model) -> tuple[dict, dict]:
    """Every array and plain number held as a python attribute of a
    process: private arrays, tuples of references, and through _refs
    every field of the process."""
    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, object] = {}
    for key, proc in model.model_dict.items():
        for attr, value in vars(proc).items():
            _collect(value, f"{key}.{attr}", arrays, scalars)
    return arrays, scalars


def replay_parameters(model: Model) -> None:
    """Overwrite every parameter_internal with its allocation fill,
    run init_from_params() over all processes again (author order, as
    the build does) and refreeze. Asserts: parameter_internal bit for
    bit; no attribute array replaced; no other array or number
    changed."""
    internal = _parameters_internal(model)
    fills = _parameter_internal_fills(model)
    arrays_before, scalars_before = _private(model)
    values_before = {
        path: array.copy() for path, array in arrays_before.items()
    }

    for key, array in internal.items():
        array.flags.writeable = True
        array[...] = fills[key]
    for proc in model.model_dict.values():
        proc.init_from_params()
    for array in internal.values():
        array.flags.writeable = False

    arrays_after, scalars_after = _private(model)
    assert arrays_after.keys() == arrays_before.keys()
    for path, array in arrays_after.items():
        assert array is arrays_before[path], (
            f"{path} was reallocated by init_from_params()"
        )
        np.testing.assert_array_equal(
            array,
            values_before[path],
            err_msg=f"{path} changed in the second init_from_params()",
        )
    assert scalars_after.keys() == scalars_before.keys()
    for path, value in scalars_after.items():
        np.testing.assert_array_equal(
            value,
            scalars_before[path],
            err_msg=f"{path} changed in the second init_from_params()",
        )


# ----------------------------------------------------------------------
# a toy that uses each hook for its purpose
# ----------------------------------------------------------------------


def _meta(kind, description, units="inches", **kwargs):
    return DataArrayMeta(
        kind=kind,
        dims=("space",),
        dtype=np.float64,
        description=description,
        units=units,
        **kwargs,
    )


class ReplayToy(Process):
    """A leaking bucket. The starting storage is a RULE relative to a
    parameter (half of capacity), as the PRMS ``*_init_frac`` are.
    Its variables carry a ``bucket_`` prefix: Lower owns ``storage``
    on this grid, and one name on one grid is one array."""

    capacity = _meta("parameter", "storage scale")
    inflow = _meta("input", "inflow")
    leak_coef = _meta(
        "parameter_internal",
        "fraction leaving per step: 1 / (1 + capacity)",
        units="fraction",
    )
    bucket_storage = _meta("variable", "storage", restart=True)
    # written by advance() before any calculate(): no default
    bucket_storage_prev = _meta("variable", "storage, previous step")
    bucket_outflow = _meta("variable", "outflow")
    bucket_storage_change = _meta("variable", "storage - storage_prev")

    budgets = {
        "mass": {
            "inputs": ("inflow",),
            "outputs": ("bucket_outflow",),
            "storage_changes": ("bucket_storage_change",),
        },
    }

    def allocate(self) -> None:
        refs = self._refs
        n_space = refs["capacity"].shape[0]
        # parameter-derived table, filled by init_from_params
        self._weights = np.empty(n_space, dtype=np.float64)
        # private state, filled by init_state: steps above the mean
        self._n_high_steps = np.zeros(n_space, dtype=np.int64)
        self._change_args = (
            refs["bucket_storage_change"],
            refs["bucket_storage"],
            refs["bucket_storage_prev"],
        )

    def init_from_params(self) -> None:
        refs = self._refs
        refs["leak_coef"][:] = 1.0 / (1.0 + refs["capacity"])
        self._weights[:] = refs["capacity"] / refs["capacity"].sum()
        self._mean_capacity = float(refs["capacity"].mean())

    def init_state(self, time: Time) -> None:
        refs = self._refs
        refs["bucket_storage"][:] = 0.5 * refs["capacity"]
        refs["bucket_outflow"][:] = 0.0
        refs["bucket_storage_change"][:] = 0.0
        self._n_high_steps[:] = 0

    def advance(self) -> None:
        self._refs["bucket_storage_prev"][:] = self._refs["bucket_storage"]

    def calculate(self, dt: np.float64, time: Time) -> None:
        refs = self._refs
        storage_change, storage, storage_prev = self._change_args
        refs["bucket_outflow"][:] = refs["leak_coef"] * storage_prev
        storage[:] = storage_prev + refs["inflow"] - refs["bucket_outflow"]
        storage_change[:] = storage - storage_prev
        self._n_high_steps[storage > self._mean_capacity] += 1

    def get_restart_state(self) -> dict[str, np.ndarray]:
        return {"n_high_steps": self._n_high_steps}

    def set_restart_state(self, state: dict[str, np.ndarray]) -> None:
        self._n_high_steps[:] = state["n_high_steps"]


# ----------------------------------------------------------------------
# three toys that break one rule each
# ----------------------------------------------------------------------


class VariableDefaultInParamsToy(ReplayToy):
    """A variable's default set in init_from_params(), not in
    init_state()."""

    def init_from_params(self) -> None:
        super().init_from_params()
        self._refs["bucket_outflow"][:] = 0.0

    def init_state(self, time: Time) -> None:
        refs = self._refs
        refs["bucket_storage"][:] = 0.5 * refs["capacity"]
        refs["bucket_storage_change"][:] = 0.0
        self._n_high_steps[:] = 0


class PrivateStateNotResetToy(ReplayToy):
    """Private state whose only default is its allocation."""

    def init_state(self, time: Time) -> None:
        refs = self._refs
        refs["bucket_storage"][:] = 0.5 * refs["capacity"]
        refs["bucket_outflow"][:] = 0.0
        refs["bucket_storage_change"][:] = 0.0


class AllocatesInParamsToy(ReplayToy):
    """A private array created by init_from_params()."""

    def init_from_params(self) -> None:
        super().init_from_params()
        refs = self._refs
        self._weights = refs["capacity"] / refs["capacity"].sum()


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_ds(dimensions, make_toy_input):
    return make_toy_input(dimensions)


def _make_model(toy_ds, toy_class):
    """Upper -> Lower with SUPPLIED starting state, and a bucket that
    starts from its own default, on one grid; budgets judged."""
    n_space = toy_ds.sizes["space"]
    bucket_parameters = xr.Dataset(
        {"capacity": ("space", np.linspace(2.0, 12.0, n_space))},
        coords={"space": toy_ds["space"]},
    )
    process_dict = {
        "upper": {
            "class": Upper,
            "forcing_up": toy_ds["forcing_up"],
            "flow": toy_ds["flow_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "lower": {
            "class": Lower,
            "forcing_low": toy_ds["forcing_low"],
            "storage": toy_ds["storage_initial"],
            "parameters": toy_ds[PARAM_NAMES],
        },
        "bucket": {
            "class": toy_class,
            "inflow": toy_ds["forcing_low"].rename("inflow"),
            "parameters": bucket_parameters,
        },
    }
    return Model(process_dict, {"budget": "error"})


def test_state_replay(toy_ds):
    with _make_model(toy_ds, ReplayToy) as model:
        snapshot = snapshot_state(model)
        model.run(DT, N_STEPS_BEFORE_REPLAY)

        # the steps moved the state the replay must put back
        bucket = model.model_dict["bucket"]
        assert (bucket._n_high_steps > 0).any()
        accum = bucket._refs["ReplayToy_mass_residual_accum"]
        assert np.isfinite(accum).all()
        assert not np.array_equal(
            model.model_dict["upper"]._refs["flow"], toy_ds["flow_initial"]
        )

        replay_state(model)
        assert_state_equal(model, snapshot)

        # supplied state and a parameter-relative default, by value
        np.testing.assert_array_equal(
            model.model_dict["upper"]._refs["flow"], toy_ds["flow_initial"]
        )
        np.testing.assert_array_equal(
            bucket._refs["bucket_storage"], 0.5 * bucket._refs["capacity"]
        )


def test_parameter_replay(toy_ds):
    with _make_model(toy_ds, ReplayToy) as model:
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        replay_parameters(model)
        leak_coef = model.model_dict["bucket"]._refs["leak_coef"]
        assert not leak_coef.flags.writeable
        assert np.isfinite(leak_coef).all()


@pytest.mark.parametrize(
    "toy_class, message",
    [
        (VariableDefaultInParamsToy, "'bucket_outflow'"),
        (PrivateStateNotResetToy, "'n_high_steps'"),
    ],
)
def test_state_replay_catches(toy_ds, toy_class, message):
    with _make_model(toy_ds, toy_class) as model:
        snapshot = snapshot_state(model)
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        replay_state(model)
        with pytest.raises(AssertionError, match=message):
            assert_state_equal(model, snapshot)


@pytest.mark.parametrize(
    "toy_class, message",
    [
        (AllocatesInParamsToy, r"bucket\._weights was reallocated"),
        (
            VariableDefaultInParamsToy,
            r"bucket\._refs\['bucket_outflow'\] changed",
        ),
    ],
)
def test_parameter_replay_catches(toy_ds, toy_class, message):
    with _make_model(toy_ds, toy_class) as model:
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        with pytest.raises(AssertionError, match=message):
            replay_parameters(model)
