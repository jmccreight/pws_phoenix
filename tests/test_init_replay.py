"""Acceptance test of the init hook split: both replays.

The three hooks (Process docstrings) exist so that a built model can
be run again without being rebuilt:

- STATE REPLAY (``Model.reset()``): after any number of steps,
  ``init_state(time)`` + ``initialize_budgets()`` + the supplied
  initial state must put every array of every grid dataset, and
  every array of ``get_restart_state()``, back to its value right
  after the build -- bit for bit. reset() first overwrites every
  variable with its allocation fill, and the test poisons the private
  state before it, so a default that only the build put in place is
  missed. The same steps run again after the reset must then
  reproduce the first run bit for bit.
- PARAMETER REPLAY (``Model.set_parameters()`` with no new values):
  ``init_from_params()`` run a second time must reproduce every
  ``parameter_internal`` bit for bit, replace no private array (only
  ``allocate()`` may allocate) and change nothing else (it writes no
  variable).
- NEW PARAMETERS: ``set_parameters(values)`` -> ``reset()`` -> run
  must equal, in every array and bit for bit, a fresh build whose
  parameter datasets hold the new values.
- RESULTS IN MEMORY (``control["collect_var_names"]``, the
  ``ResultCollector`` held as ``model.results``): row ``tt`` is the
  variable after step ``tt``, the zarr Output holds the same numbers,
  and a run repeated after a reset fills the same buffer with the
  same values.

This module: the toy model (no external data) plus a toy that uses
each hook for its purpose, and three toys that break one rule each --
the replays must catch all three; then the complete NHM with stream
temperature, built from the drb PRMS control (every ported NHM
process, two grids, supplied starting state from the translation);
then the ucb model with a FlowGraph spliced in (muskingum channel,
pass-through and STARFIT node types), where the reservoir's starting
storage has no default and comes only from supplied state.
"""

import numpy as np
import pytest
import xarray as xr
from conftest import PWS_TEST_DATA, cold_start_state
from pws_phoenix.clock import Time
from pws_phoenix.data_io import ResultCollector
from pws_phoenix.hydrology.starfit_daily_flow_node import StarfitDailyFlowNode
from pws_phoenix.hydrology.starfit_flow_node import StarfitFlowNode
from pws_phoenix.model import Model
from pws_phoenix.process import _FILL_VALUE, DataArrayMeta, Process
from pws_phoenix.processes_concrete import Lower, Upper
from pws_phoenix.prms_translate import model_from_control
from pws_phoenix.prms_translate.assemble import (
    assemble_from_control,
    insert_flow_graph,
)
from test_prms_translate_flow_graph import (
    UCB_CONTROL,
    big_sandy_nodes,
    ucb_skipif,
)

DT = np.float64(1.0)
DT_DAY = np.float64(86400.0)
DRB_DIR = PWS_TEST_DATA / "drb_2yr"
_drb_needed = [
    DRB_DIR / "nhm_stream_temp.control",
    DRB_DIR / "myparam.param",
    DRB_DIR / "prcp.cbh",
    DRB_DIR / "rhavg.cbh",
]
_drb_missing = [str(ff) for ff in _drb_needed if not ff.exists()]
drb_skipif = pytest.mark.skipif(
    bool(_drb_missing),
    reason="drb_2yr legacy files missing: " + ", ".join(_drb_missing[:3]),
)
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


def _dataset_arrays(model: Model) -> dict[tuple[str, str], np.ndarray]:
    """Every array of every grid dataset, keyed (grid, name):
    parameters, parameter_internal, the inputs' current values, Map
    target buffers and variables (budget variables included). A name
    shared by two processes of a grid is ONE array."""
    return {
        (grid, str(name)): disc.dataset[name].values
        for grid, disc in model.discretizations.items()
        for name in disc.dataset.data_vars
    }


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
    return _dataset_arrays(model) | _private_state(model)


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
    """Poison the private state, then Model.reset(), which refills
    every variable with its allocation fill before it runs
    init_state(time)."""
    for array in _private_state(model).values():
        _poison(array)
    model.reset()
    assert model._next_index == 0
    assert model.current_time_index[0] == 0
    assert model.current_time[0] == model.times.values[0]


def assert_state_equal(
    model: Model,
    snapshot: dict[tuple[str, str], np.ndarray],
    when: str = "after the build",
) -> None:
    state = _state(model)
    assert state.keys() == snapshot.keys()
    for key, expected in snapshot.items():
        np.testing.assert_array_equal(
            state[key],
            expected,
            err_msg=f"{key} differs from its value {when}",
        )


def assert_rerun_equal(
    model: Model,
    dt: np.float64,
    stepped: dict[tuple[str, str], np.ndarray],
) -> None:
    """On a model just reset: the same steps run again reproduce the
    first run (`stepped`, its snapshot) bit for bit."""
    model.run(dt, N_STEPS_BEFORE_REPLAY)
    assert model._next_index == N_STEPS_BEFORE_REPLAY
    assert_state_equal(model, stepped, when="after the first run")


def _parameters_internal(model: Model) -> dict[tuple[str, str], np.ndarray]:
    out = {}
    for key, proc in model.model_dict.items():
        grid = model._proc_grid[key]
        for name in proc.get_parameters_internal():
            out[(grid, name)] = proc._refs[name]
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
    """Model.set_parameters() with no new values: it overwrites every
    parameter_internal with its allocation fill, runs
    init_from_params() over all processes again (author order, as
    the build does) and refreezes. Asserts: parameter_internal bit
    for bit; no attribute array replaced; no other array or number
    changed. The model then needs a reset() before it steps."""
    internal = _parameters_internal(model)
    arrays_before, scalars_before = _private(model)
    values_before = {
        path: array.copy() for path, array in arrays_before.items()
    }

    model.set_parameters({})
    for array in internal.values():
        assert not array.flags.writeable

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


class CheckedToy(ReplayToy):
    """Not a rule-breaker: a parameter check that raises in
    init_from_params(), before anything is computed."""

    def init_from_params(self) -> None:
        if (self._refs["capacity"] <= 0.0).any():
            raise ValueError("capacity must be positive")
        super().init_from_params()


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_ds(dimensions, make_toy_input):
    return make_toy_input(dimensions)


def _make_model(
    toy_ds, toy_class, control=None, capacity=None, calibration_mode=True
):
    """Upper -> Lower with SUPPLIED starting state (none when
    `control` restarts), and a bucket that starts from its own
    default, on one grid; budgets judged. `capacity` is the bucket's
    parameter (the model freezes the array it is given).
    `calibration_mode=False` leaves the key out of the control."""
    control = {"budget": "error"} | (control or {})
    if calibration_mode:
        control["calibration_mode"] = True
    n_space = toy_ds.sizes["space"]
    if capacity is None:
        capacity = np.linspace(2.0, 12.0, n_space)
    bucket_parameters = xr.Dataset(
        {"capacity": ("space", capacity)},
        coords={"space": toy_ds["space"]},
    )
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
        "bucket": {
            "class": toy_class,
            "inflow": toy_ds["forcing_low"].rename("inflow"),
            "parameters": bucket_parameters,
        },
    }
    return Model(process_dict, control)


def test_state_replay(toy_ds):
    with _make_model(toy_ds, ReplayToy) as model:
        snapshot = snapshot_state(model)
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        stepped = snapshot_state(model)

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
        # an input's current values are nan again, as at the build
        assert np.isnan(bucket._refs["inflow"]).all()

        assert_rerun_equal(model, DT, stepped)


def test_reset_raises(toy_ds, tmp_path):
    """reset() and set_parameters() refuse a model with a zarr
    Output, a warm-started model and a finalized model."""
    control = {
        "output_var_names": ["flow"],
        "output_serial_zarr": tmp_path / "out.zarr",
        "time_chunk_size": 2,
    }
    with _make_model(toy_ds, ReplayToy, control) as model:
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        with pytest.raises(NotImplementedError, match="zarr Output"):
            model.reset()
        with pytest.raises(NotImplementedError, match="zarr Output"):
            model.set_parameters({})

    restart_dir = tmp_path / "restarts"
    with _make_model(toy_ds, ReplayToy) as model:
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        model.write_restart(restart_dir)
    with _make_model(
        toy_ds, ReplayToy, {"restart_read": restart_dir}
    ) as model:
        with pytest.raises(NotImplementedError, match="restart_read"):
            model.reset()
        with pytest.raises(NotImplementedError, match="restart_read"):
            model.set_parameters({})

    with pytest.raises(RuntimeError, match="finalized"):
        model.reset()
    with pytest.raises(RuntimeError, match="finalized"):
        model.set_parameters({})


def test_set_parameters(toy_ds):
    """set_parameters -> reset -> run equals a fresh build with the
    new values, in every array. The bucket's starting storage is a
    rule relative to the parameter, so the reset must recompute it."""
    new_capacity = np.linspace(3.0, 30.0, toy_ds.sizes["space"])
    with _make_model(toy_ds, ReplayToy) as model:
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        model.set_parameters({"capacity": new_capacity})

        # written into the model's own array, which is frozen again
        capacity = model.model_dict["bucket"]._refs["capacity"]
        assert capacity is not new_capacity
        np.testing.assert_array_equal(capacity, new_capacity)
        assert not capacity.flags.writeable
        assert new_capacity.flags.writeable

        # the state in place belongs to the old parameters
        with pytest.raises(RuntimeError, match=r"call reset\(\)"):
            model.step(DT)
        model.reset()
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        stepped = snapshot_state(model)

    with _make_model(toy_ds, ReplayToy, capacity=new_capacity.copy()) as fresh:
        fresh.run(DT, N_STEPS_BEFORE_REPLAY)
        assert_state_equal(
            fresh, stepped, when="in the model whose parameters were set"
        )


@pytest.mark.parametrize(
    "control", [{}, {"calibration_mode": False}], ids=["absent", "false"]
)
def test_set_parameters_needs_calibration_mode(toy_ds, control):
    """Without control["calibration_mode"] = True set_parameters()
    raises before it writes anything, and the model runs on as
    built. reset() is not guarded."""
    with _make_model(
        toy_ds, ReplayToy, control, calibration_mode=False
    ) as model:
        capacity = model.model_dict["bucket"]._refs["capacity"]
        capacity_built = capacity.copy()
        for values in ({}, {"capacity": 99.0}):
            with pytest.raises(RuntimeError, match="calibration_mode"):
                model.set_parameters(values)
        np.testing.assert_array_equal(capacity, capacity_built)
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        model.reset()


def test_set_parameters_raises(toy_ds):
    n_space = toy_ds.sizes["space"]
    with _make_model(toy_ds, CheckedToy) as model:
        bucket = model.model_dict["bucket"]
        capacity = bucket._refs["capacity"]
        capacity_built = capacity.copy()

        # every name and value is checked before anything is written
        for values, grid, error, message in (
            ({"no_such_name": 1.0}, None, ValueError, "not a parameter"),
            ({"capacity": 1.0}, "nowhere", ValueError, "grid 'nowhere'"),
            ({"leak_coef": 0.5}, None, ValueError, "parameter_internal"),
            ({"capacity": np.ones(n_space + 1)}, None, ValueError, "shape"),
            (
                {"capacity": np.ones(n_space, dtype=complex)},
                None,
                TypeError,
                "dtype",
            ),
        ):
            with pytest.raises(error, match=message):
                model.set_parameters({"capacity": 99.0} | values, grid=grid)
        np.testing.assert_array_equal(capacity, capacity_built)
        model.step(DT)

        # a parameter check raises in init_from_params(): the values
        # stay written (a scalar, broadcast), the internals are
        # frozen again, and the model refuses to go on
        with pytest.raises(ValueError, match="capacity must be positive"):
            model.set_parameters({"capacity": -1.0})
        assert (capacity == -1.0).all()
        assert not bucket._refs["leak_coef"].flags.writeable
        with pytest.raises(RuntimeError, match="set_parameters"):
            model.reset()
        with pytest.raises(RuntimeError, match=r"call reset\(\)"):
            model.step(DT)

        # until a set_parameters() succeeds
        model.set_parameters({"capacity": capacity_built})
        model.reset()
        model.run(DT, N_STEPS_BEFORE_REPLAY)
        assert np.isfinite(bucket._refs["bucket_storage"]).all()


COLLECT_NAMES = ["flow", "bucket_storage"]


def _step_and_check_rows(model, dt, n_steps):
    """Step the model; after each step the collector's row of that
    step equals the live variable."""
    refs = {
        name: proc._refs[name]
        for name in model.results.values
        for proc in model.model_dict.values()
        if name in proc.get_var_names(include_budget=True)
    }
    assert refs.keys() == model.results.values.keys()
    for _ in range(int(n_steps)):
        tt = model._next_index
        model.step(dt)
        assert model.results.n_collected == tt + 1
        for name, ref in refs.items():
            np.testing.assert_array_equal(
                model.results.values[name][tt], ref, err_msg=name
            )


def test_result_collector(toy_ds, tmp_path):
    n_time = toy_ds.sizes["time"]
    n_space = toy_ds.sizes["space"]

    # one run that both collects and writes zarr: the same numbers
    store = tmp_path / "out.zarr"
    control = {
        "collect_var_names": COLLECT_NAMES,
        "output_var_names": COLLECT_NAMES,
        "output_serial_zarr": store,
        "time_chunk_size": 2,
    }
    with _make_model(toy_ds, ReplayToy, control) as model:
        assert model.results.n_collected == 0
        for name in COLLECT_NAMES:
            assert model.results.values[name].shape == (n_time, n_space)
            assert np.isnan(model.results.values[name]).all()
        model.run(DT, np.int32(n_time))
        assert model.results.n_collected == n_time
        collected = model.results.values
    with xr.open_zarr(store, consolidated=False) as written:
        for name in COLLECT_NAMES:
            assert np.isfinite(collected[name]).all()
            np.testing.assert_array_equal(
                collected[name], written[name].values, err_msg=name
            )

    # reuse: the same buffers, the same values, the count from zero
    control = {"collect_var_names": COLLECT_NAMES}
    with _make_model(toy_ds, ReplayToy, control) as model:
        buffers = dict(model.results.values)
        _step_and_check_rows(model, DT, n_time)
        first = {name: array.copy() for name, array in buffers.items()}
        model.reset()
        assert model.results.n_collected == 0
        # a shorter run: its rows are counted, the rest are left
        _step_and_check_rows(model, DT, N_STEPS_BEFORE_REPLAY)
        assert model.results.n_collected == N_STEPS_BEFORE_REPLAY
        model.run(DT, np.int32(n_time - N_STEPS_BEFORE_REPLAY))
        for name, array in model.results.values.items():
            assert array is buffers[name]
            np.testing.assert_array_equal(array, first[name], err_msg=name)
            np.testing.assert_array_equal(array, collected[name], err_msg=name)


def test_result_collector_raises(toy_ds):
    control = {"collect_var_names": ["no_such_name"]}
    with pytest.raises(ValueError, match="'no_such_name' not found"):
        _make_model(toy_ds, ReplayToy, control)


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


# ----------------------------------------------------------------------
# the complete NHM with stream temperature, from the drb PRMS control
# ----------------------------------------------------------------------


@drb_skipif
def test_drb_stream_temp_kit_replays():
    """Both replays on every ported NHM process at once: the hru grid
    (atmosphere through groundwater), the segment grid (channel,
    hydraulic geometry, stream temperature with its silos held in
    attributes) and the starting state the translation supplies
    (gwres_stor, seg_inflow)."""
    with model_from_control(
        DRB_DIR / "nhm_stream_temp.control",
        control={"budget": "error", "calibration_mode": True},
    ) as model:
        assert model._supplied_state
        snapshot = snapshot_state(model)
        # most variables hold a default the build put in place, so the
        # replay cannot pass by leaving the fill (measured Nov 2026:
        # 155 of 184 variables; 31 parameter_internal; 4 state arrays
        # held in attributes)
        n_default = sum(
            not np.array_equal(
                snapshot[key],
                np.full_like(snapshot[key], fill),
                equal_nan=True,
            )
            for key, fill in _variable_fills(model).items()
        )
        assert n_default > 100
        assert len(_private_state(model)) == 4
        model.run(DT_DAY, N_STEPS_BEFORE_REPLAY)
        stepped = snapshot_state(model)

        # the steps moved state of each kind the replay must put back
        stream_temp = model.model_dict["prms_stream_temp"]
        assert (stream_temp._gw_index > 0).any()
        assert (stream_temp._gw_silo > -98.0).any()
        n_moved = sum(
            not np.array_equal(array, snapshot[key], equal_nan=True)
            for key, array in _state(model).items()
        )
        assert n_moved > 100

        # parameters first, on the stepped model: a variable written
        # by init_from_params() would show as a changed array
        assert _parameters_internal(model)
        replay_parameters(model)
        replay_state(model)
        assert_state_equal(model, snapshot)
        # both grids, with the Maps between them, run again
        assert model.maps
        assert_rerun_equal(model, DT_DAY, stepped)


# one parameter per kind of consequence: starting storages that are a
# rule relative to it (soil_moist_max), parameter_internal values
# (x_coef -> the Muskingum coefficients), a (month, space) parameter
# (dday_slope), and two read only by a kernel
DRB_PARAMETER_FACTORS = {
    "soil_moist_max": 0.9,
    "x_coef": 0.9,
    "dday_slope": 1.05,
    "gwflow_coef": 1.1,
    "carea_max": 0.9,
}


@drb_skipif
def test_drb_set_parameters():
    """set_parameters -> reset -> run on the complete NHM equals a
    fresh build whose parameter datasets hold the new values: every
    array of both grids, bit for bit."""
    control_file = DRB_DIR / "nhm_stream_temp.control"
    control = {"budget": "error", "calibration_mode": True}
    with model_from_control(control_file, control=control) as model:
        new_values = {}
        for name, factor in DRB_PARAMETER_FACTORS.items():
            ((_, array),) = [
                item
                for item in model._parameter_arrays.items()
                if item[0][1] == name
            ]
            new_values[name] = factor * array
        with pytest.raises(ValueError, match="owned by the Discretization"):
            model.set_parameters({"tosegment": 1})

        model.run(DT_DAY, N_STEPS_BEFORE_REPLAY)
        stepped_built = snapshot_state(model)
        model.set_parameters(new_values)
        model.reset()
        model.run(DT_DAY, N_STEPS_BEFORE_REPLAY)
        stepped = snapshot_state(model)
        # the new values moved the run: more arrays than the five set
        n_moved = sum(
            not np.array_equal(array, stepped_built[key], equal_nan=True)
            for key, array in stepped.items()
        )
        assert n_moved > 3 * len(new_values)

    kit = assemble_from_control(control_file, control=control)
    n_written = 0
    for entry in kit.process_dict.values():
        parameters = entry.get("parameters")
        for name, values in new_values.items():
            if parameters is not None and name in parameters:
                parameters[name].values[...] = values
                n_written += 1
    assert n_written >= len(new_values)
    with kit.model() as fresh:
        fresh.run(DT_DAY, N_STEPS_BEFORE_REPLAY)
        assert_state_equal(
            fresh, stepped, when="in the model whose parameters were set"
        )


@drb_skipif
def test_drb_result_collector():
    """Variables of both grids collected through a run, a reset and
    the same run again."""
    names = ["seg_outflow", "soil_moist", "gwres_flow"]
    with model_from_control(
        DRB_DIR / "nhm_stream_temp.control",
        control={"budget": "error", "collect_var_names": names},
    ) as model:
        _step_and_check_rows(model, DT_DAY, N_STEPS_BEFORE_REPLAY)
        assert model.results is not None
        first = {
            name: array.copy() for name, array in model.results.values.items()
        }
        for name in names:
            assert np.isfinite(first[name][:N_STEPS_BEFORE_REPLAY]).all()
            assert np.isnan(first[name][N_STEPS_BEFORE_REPLAY:]).all()
        assert {array.shape[1] for array in first.values()} == {
            model.discretizations[grid].dataset.sizes[grid]
            for grid in ("nhru", "nsegment")
        }

        model.reset()
        _step_and_check_rows(model, DT_DAY, N_STEPS_BEFORE_REPLAY)
        for name, array in model.results.values.items():
            np.testing.assert_array_equal(array, first[name], err_msg=name)

        # an integer or bool variable has no nan to fill with
        not_float = next(
            name
            for proc in model.model_dict.values()
            for name in proc.get_var_names()
            if not np.issubdtype(proc._refs[name].dtype, np.floating)
        )
        with pytest.raises(ValueError, match="only float variables"):
            ResultCollector([not_float], model.model_dict, model.ntime)


# ----------------------------------------------------------------------
# a FlowGraph with a STARFIT reservoir, spliced into the ucb model
# ----------------------------------------------------------------------


@ucb_skipif
@pytest.mark.parametrize(
    "starfit_class",
    [StarfitFlowNode, StarfitDailyFlowNode],
    ids=["starfit", "starfit_daily"],
)
def test_ucb_starfit_graph_replays(starfit_class):
    """Both replays through a composed FlowGraph, whose hooks forward
    to three node types. The reservoir storage lake_storage_sub is
    required initial state: init_state leaves it at the allocation
    nan and only the supplied state fills the reservoir's row."""
    nodes, _ = big_sandy_nodes(starfit_class)
    kit = insert_flow_graph(
        assemble_from_control(
            UCB_CONTROL,
            control={"budget": "error", "calibration_mode": True},
        ),
        nodes,
    )
    with kit.model() as model:
        key = ("nnodes", "lake_storage_sub")
        assert key in model._supplied_state
        snapshot = snapshot_state(model)
        # one reservoir row; every other node's row is nan
        wh_reservoir = np.isfinite(snapshot[key])
        assert wh_reservoir.sum() == 1

        model.run(DT_DAY, N_STEPS_BEFORE_REPLAY)
        stepped = snapshot_state(model)
        # the steps moved the reservoir's storage: the hourly node
        # marches lake_storage_sub, the daily node lake_storage
        refs = model.model_dict["flow_graph"]._refs
        assert any(
            not np.array_equal(
                refs[name][wh_reservoir],
                snapshot[("nnodes", name)][wh_reservoir],
                equal_nan=True,
            )
            for name in ("lake_storage_sub", "lake_storage")
        )

        replay_parameters(model)
        replay_state(model)
        assert_state_equal(model, snapshot)
        # StarfitDailyFlowNode's first-substep seed is gated on step
        # 0: the rerun needs the step cursor back at the start
        assert_rerun_equal(model, DT_DAY, stepped)
