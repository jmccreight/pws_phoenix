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
}


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
    return out


def _process_dict(toy: dict) -> dict:
    return {
        "upper": {
            "class": Upper,
            "discretization": "hru",
            "parameters": toy["up_params"],
            "forcing_up": toy["forcing_up"],
            "flow": toy["up_flow_initial"],
        },
        "lower": {
            "class": Lower,
            "discretization": "segment",
            "parameters": toy["low_params"],
            "forcing_low": toy["forcing_low"],
            "storage": toy["low_storage_initial"],
        },
    }


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
