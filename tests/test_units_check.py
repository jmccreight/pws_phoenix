"""The Model boundary of units mode (stage 2-M): control["units"] =
None | "check" | "convert" and control["units_patch"], on the two-grid
toy (Upper on hru -> Map -> Lower on segment). Synthetic, CI-able."""

import pathlib as pl

import numpy as np
import pytest
import xarray as xr
from pws_phoenix.maps import Map
from pws_phoenix.model import Model
from pws_phoenix.process import DataArrayMeta
from pws_phoenix.processes_concrete import Lower, Upper
from pws_phoenix.units import affine, reconcile, same

# the toy's declared units (processes_concrete.py)
UNITS = {
    "param_up_0": "fraction",
    "param_up_1": "fraction",
    "param_shared_name": "fraction",
    "param_low_0": "fraction",
    "forcing_up": "inches",
    "forcing_low": "inches",
    # supplied initial state (restart variables)
    "flow": "inches",
    "storage": "inches",
}
# toy key -> the state variable it supplies
STATE_KEYS = {"up_flow_initial": "flow", "low_storage_initial": "storage"}


def _stamp(toy: dict) -> dict:
    """The toy with attrs["units"] = the declared units everywhere."""
    out = dict(toy)
    for key in ("up_params", "low_params"):
        ds = toy[key].copy()  # shallow: arrays shared
        for name in ds.data_vars:
            ds[name].attrs["units"] = UNITS[str(name)]
        out[key] = ds
    for key in ("forcing_up", "forcing_low"):
        da = toy[key].copy(deep=False)
        da.attrs["units"] = UNITS[key]
        out[key] = da
    for key, name in STATE_KEYS.items():
        da = toy[key].copy(deep=False)
        da.attrs["units"] = UNITS[name]
        out[key] = da
    return out


def _process_dict(toy: dict, *, state: bool = True) -> dict:
    """`state=False` omits the supplied initial state (a warm start
    takes it from the restart files)."""
    process_dict = {
        "upper": {
            "class": Upper,
            "discretization": "hru",
            "parameters": toy["up_params"],
            "forcing_up": toy["forcing_up"],
        },
        "lower": {
            "class": Lower,
            "discretization": "segment",
            "parameters": toy["low_params"],
            "forcing_low": toy["forcing_low"],
        },
    }
    if state:
        process_dict["upper"]["flow"] = toy["up_flow_initial"]
        process_dict["lower"]["storage"] = toy["low_storage_initial"]
    return process_dict


def _maps(weights) -> dict:
    return {
        "hru_to_seg": Map(
            weights=weights,
            grid={"hru": "segment"},
            variable={"flow": "flow"},
        )
    }


def _run(toy, weights, control, n_time) -> dict:
    with Model(_process_dict(toy), control, maps=_maps(weights)) as model:
        model.run(np.float64(1.0), np.int32(n_time))
        return {
            "flow": model.model_dict["upper"]["flow"].values.copy(),
            "storage": model.model_dict["lower"]["storage"].values.copy(),
        }


@pytest.fixture
def toy(dimensions, make_two_grid_toy):
    return make_two_grid_toy(dimensions)


@pytest.fixture
def n_time(dimensions):
    return dimensions["n_time"]


# ----------------------------------------------------------------------
# the pure functions
# ----------------------------------------------------------------------


def test_affine_and_same():
    assert affine("feet", "inches") == (12.0, 0.0)
    assert affine("percent", "fraction") == (0.01, 0.0)
    scale, offset = affine("degF", "degC")
    assert scale == 5.0 / 9.0
    np.testing.assert_allclose(offset, -160.0 / 9.0, rtol=1e-14)
    assert same("fraction", "none")
    assert not same("inches", "feet")
    assert same("dday", "dday")  # no pint expression: string equality
    assert not same("dday", "fraction")
    with pytest.raises(ValueError, match="cannot be converted"):
        affine("inches", "degF")


def test_reconcile_converts_in_place_and_restores_flag():
    da = xr.DataArray(
        np.array([32.0, 212.0]), dims=["x"], attrs={"units": "degF"}
    )
    da.values.flags.writeable = False
    reconcile("t", da, "degC", "convert", {}, convertible=True, context="test")
    np.testing.assert_allclose(da.values, [0.0, 100.0], atol=1e-12)
    assert da.attrs["units"] == "degC"
    assert not da.values.flags.writeable


def test_reconcile_messages():
    da = xr.DataArray(np.ones(2), dims=["x"])
    with pytest.raises(ValueError, match="has no units"):
        reconcile(
            "v", da, "inches", "check", {}, convertible=True, context="c"
        )
    da.attrs["units"] = "inch"
    with pytest.raises(ValueError, match="not in the pws_phoenix vocabulary"):
        reconcile(
            "v", da, "inches", "check", {}, convertible=True, context="c"
        )
    da.attrs["units"] = "feet"
    with pytest.raises(ValueError, match="'convert' converts"):
        reconcile(
            "v", da, "inches", "check", {}, convertible=True, context="c"
        )
    with pytest.raises(ValueError, match="streamed/lazy"):
        reconcile(
            "v", da, "inches", "convert", {}, convertible=False, context="c"
        )
    # the patch wins over the attribute
    reconcile(
        "v",
        da,
        "inches",
        "check",
        {"v": "inches"},
        convertible=True,
        context="c",
    )
    # mode None: nothing looked at
    reconcile("v", da, "inches", None, {}, convertible=False, context="c")
    # unit-less declaration: nothing looked at
    reconcile("v", da, None, "check", {}, convertible=False, context="c")


# ----------------------------------------------------------------------
# the Model boundary
# ----------------------------------------------------------------------


def test_default_mode_ignores_units(toy, two_grid_weights, n_time):
    # the toy carries no units attrs; the default never looks
    _run(toy, two_grid_weights, {}, n_time)


def test_bad_mode_raises(toy, two_grid_weights):
    with pytest.raises(ValueError, match=r"control\['units'\]"):
        Model(
            _process_dict(toy),
            {"units": "verify"},
            maps=_maps(two_grid_weights),
        )


def test_check_requires_units_attrs(toy, two_grid_weights):
    with pytest.raises(ValueError, match="parameter.*has no units"):
        Model(
            _process_dict(toy),
            {"units": "check"},
            maps=_maps(two_grid_weights),
        )


def test_check_passes_when_stamped(toy, two_grid_weights, n_time):
    answers = _run(toy, two_grid_weights, {}, n_time)
    got = _run(_stamp(toy), two_grid_weights, {"units": "check"}, n_time)
    np.testing.assert_array_equal(got["flow"], answers["flow"])
    np.testing.assert_array_equal(got["storage"], answers["storage"])


def test_patch_supplies_missing_units(toy, two_grid_weights, n_time):
    patch = dict(UNITS)
    _run(
        toy, two_grid_weights, {"units": "check", "units_patch": patch}, n_time
    )


def test_check_rejects_mismatch(toy, two_grid_weights):
    stamped = _stamp(toy)
    stamped["forcing_up"].attrs["units"] = "feet"
    with pytest.raises(ValueError, match="input.*'forcing_up'.*'feet'"):
        Model(
            _process_dict(stamped),
            {"units": "check"},
            maps=_maps(two_grid_weights),
        )


def test_convert_reproduces_declared_units_run(toy, two_grid_weights, n_time):
    """forcing_up in feet (/12) and param_up_0 in percent (x100) under
    'convert' give the same answers as the declared-units run."""
    answers = _run(_stamp(toy), two_grid_weights, {"units": "check"}, n_time)
    other = _stamp(toy)
    other["forcing_up"] = (other["forcing_up"] / 12.0).assign_attrs(
        units="feet"
    )
    up = other["up_params"].copy()
    up["param_up_0"] = (up["param_up_0"] * 100.0).assign_attrs(units="percent")
    other["up_params"] = up
    got = _run(other, two_grid_weights, {"units": "convert"}, n_time)
    np.testing.assert_allclose(got["flow"], answers["flow"], rtol=1e-12)
    np.testing.assert_allclose(got["storage"], answers["storage"], rtol=1e-12)
    # the conversion happened IN PLACE on the caller's arrays (zero-copy
    # rule) and the attrs now say so
    assert other["forcing_up"].attrs["units"] == "inches"
    np.testing.assert_allclose(
        other["forcing_up"].values, toy["forcing_up"].values
    )
    assert up["param_up_0"].attrs["units"] == "fraction"


def test_convert_file_input(toy, two_grid_weights, n_time, tmp_path: pl.Path):
    """A file-backed read-only input is converted too: Input.__init__
    sets the read-only flag through `.values`, which materializes the
    lazily opened array (pre-existing behavior), so it is resident at
    the time of the check. (The streamed, never-resident case is the
    MPI input_file -- test_units_check_mpi.py.)"""
    answers = _run(_stamp(toy), two_grid_weights, {"units": "check"}, n_time)
    stamped = _stamp(toy)
    path = tmp_path / "forcing_up_feet.nc"
    (stamped["forcing_up"] / 12.0).assign_attrs(units="feet").rename(
        "forcing_up"
    ).to_netcdf(path)
    stamped["forcing_up"] = path
    with pytest.raises(ValueError, match="input.*'forcing_up'.*'feet'"):
        Model(
            _process_dict(stamped),
            {"units": "check"},
            maps=_maps(two_grid_weights),
        )
    got = _run(stamped, two_grid_weights, {"units": "convert"}, n_time)
    np.testing.assert_allclose(got["flow"], answers["flow"], rtol=1e-12)


# ----------------------------------------------------------------------
# supplied initial state + restart files: judged like parameters, but
# COPIED into the model's buffer, so "convert" converts the model's copy
# ----------------------------------------------------------------------


def test_check_requires_units_on_supplied_state(toy, two_grid_weights):
    stamped = _stamp(toy)
    stamped["up_flow_initial"] = toy["up_flow_initial"]  # no attrs
    with pytest.raises(ValueError, match="initial state: 'flow'.*no units"):
        Model(
            _process_dict(stamped),
            {"units": "check"},
            maps=_maps(two_grid_weights),
        )
    # a bare ndarray carries no attrs at all: the same error ...
    stamped["up_flow_initial"] = toy["up_flow_initial"].values
    with pytest.raises(ValueError, match="initial state: 'flow'.*no units"):
        Model(
            _process_dict(stamped),
            {"units": "check"},
            maps=_maps(two_grid_weights),
        )
    # ... and the patch is the way to say what it is
    Model(
        _process_dict(stamped),
        {"units": "check", "units_patch": {"flow": "inches"}},
        maps=_maps(two_grid_weights),
    ).finalize()


def test_check_rejects_supplied_state_mismatch(toy, two_grid_weights):
    stamped = _stamp(toy)
    stamped["low_storage_initial"].attrs["units"] = "feet"
    with pytest.raises(ValueError, match="initial state: 'storage'.*'feet'"):
        Model(
            _process_dict(stamped),
            {"units": "check"},
            maps=_maps(two_grid_weights),
        )


def test_convert_supplied_state_leaves_callers_array(
    toy, two_grid_weights, n_time
):
    """flow and storage supplied in feet (/12): 'convert' converts the
    MODEL's copy -- the declared-units answers -- and the caller's
    arrays stay as supplied (they are not the working buffers)."""
    answers = _run(_stamp(toy), two_grid_weights, {"units": "check"}, n_time)
    other = _stamp(toy)
    for key in STATE_KEYS:
        other[key] = (other[key] / 12.0).assign_attrs(units="feet")
    supplied = {key: other[key].values.copy() for key in STATE_KEYS}
    got = _run(other, two_grid_weights, {"units": "convert"}, n_time)
    np.testing.assert_allclose(got["flow"], answers["flow"], rtol=1e-12)
    np.testing.assert_allclose(got["storage"], answers["storage"], rtol=1e-12)
    for key in STATE_KEYS:
        assert other[key].attrs["units"] == "feet"
        np.testing.assert_array_equal(other[key].values, supplied[key])


def test_restart_file_units(toy, two_grid_weights, n_time, tmp_path: pl.Path):
    """A restart file is a netCDF file anyone can write: its declared
    variables are judged like supplied state."""
    stamped = _stamp(toy)
    maps = _maps(two_grid_weights)
    n_first = n_time // 2
    dt = np.float64(1.0)
    answers = _run(stamped, two_grid_weights, {"units": "check"}, n_time)
    restart_dir = tmp_path / "restarts"
    with Model(_process_dict(stamped), {"units": "check"}, maps=maps) as mm:
        mm.run(dt, np.int32(n_first))
        mm.write_restart(restart_dir)

    def warm(directory: pl.Path, control: dict) -> dict:
        control = control | {"restart_read": directory}
        with Model(
            _process_dict(stamped, state=False), control, maps=maps
        ) as model:
            model.run(dt, np.int32(n_time - n_first))
            return {
                "flow": model.model_dict["upper"]["flow"].values.copy(),
                "storage": model.model_dict["lower"]["storage"].values.copy(),
            }

    # the model's own files carry the declared units
    got = warm(restart_dir, {"units": "check"})
    np.testing.assert_array_equal(got["flow"], answers["flow"])
    np.testing.assert_array_equal(got["storage"], answers["storage"])

    def rewrite(tag: str, units: str | None, scale: float) -> pl.Path:
        """The restart files with segment `storage` rescaled/restamped."""
        directory = tmp_path / tag
        directory.mkdir()
        for path in restart_dir.glob("*.nc"):
            ds = xr.load_dataset(path)
            if "storage" in ds:
                # arithmetic keeps attrs: set or remove units outright
                ds["storage"] = ds["storage"] * scale
                ds["storage"].attrs.pop("units", None)
                if units is not None:
                    ds["storage"].attrs["units"] = units
            ds.to_netcdf(directory / path.name)
        return directory

    with pytest.raises(ValueError, match="restart file.*'storage'.*no units"):
        warm(rewrite("no_units", None, 1.0), {"units": "check"})
    feet_dir = rewrite("feet", "feet", 1.0 / 12.0)
    with pytest.raises(ValueError, match="restart file.*'storage'.*'feet'"):
        warm(feet_dir, {"units": "check"})
    got = warm(feet_dir, {"units": "convert"})
    np.testing.assert_allclose(got["flow"], answers["flow"], rtol=1e-12)
    np.testing.assert_allclose(got["storage"], answers["storage"], rtol=1e-12)
    # the default mode never looks
    warm(rewrite("default", None, 1.0), {})


class LowerFeet(Lower):
    """Lower whose mapped input is declared in feet: the Map would have
    to change units, which a Map never does."""

    flow = DataArrayMeta(
        kind="input",
        dims=("space",),
        dtype=np.float64,
        units="feet",
        description="flow in feet (test)",
    )


def test_map_units_must_agree(toy, two_grid_weights):
    stamped = _stamp(toy)
    pdict = _process_dict(stamped)
    pdict["lower"]["class"] = LowerFeet
    with pytest.raises(ValueError, match="a Map carries one quantity"):
        Model(pdict, {"units": "check"}, maps=_maps(two_grid_weights))
    # unchecked by default
    Model(pdict, {}, maps=_maps(two_grid_weights)).finalize()
