"""prms_translate.assemble_from_translation / model_from_translation:
a domain translated ahead of time (translate_domain) gives the kit and
the answers of the original control file (assemble_from_control), on
drb_2yr's nhm.control."""

import pathlib as pl
import shutil

import numpy as np
import pytest
import xarray as xr
import yaml

from conftest import PWS_TEST_DATA
from pws_phoenix.prms_translate import (
    assemble,
    assemble_from_control,
    assemble_from_translation,
    model_from_control,
    model_from_translation,
)
from pws_phoenix.prms_translate.translate_domain import translate_domain

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
NHM = DRB_DIR / "nhm.control"
FORCING_NAMES = ["prcp", "tmax", "tmin"]
START = np.datetime64("1979-01-01")
END = np.datetime64("1980-12-31")
N_TIME = 731
DT = np.float64(86400.0)
# one variable of each process on the way from the forcings to the
# outlet, kept in memory for the whole run
COLLECT_NAMES = [
    "potet",
    "pkwater_equiv",
    "sroff_vol",
    "ssres_flow_vol",
    "gwres_stor",
    "gwres_flow",
    "seg_outflow",
]
# process_dict entry keys that are not data given by the translation
STRUCTURE_KEYS = ("class", "discretization", "parameters")


@pytest.fixture(scope="module")
def translation_yaml(tmp_path_factory):
    """nhm.control translated once."""
    return translate_domain(NHM, tmp_path_factory.mktemp("domain"))


@pytest.fixture
def moved_yaml(translation_yaml, tmp_path):
    """The translated domain copied to another directory: a copy a
    test may edit, and the case of a domain directory that was
    moved."""
    moved = tmp_path / "moved"
    shutil.copytree(translation_yaml.parent, moved)
    return moved / translation_yaml.name


def _assert_same_variables(found: xr.Dataset, expected: xr.Dataset):
    assert sorted(map(str, found.variables)) == sorted(
        map(str, expected.variables)
    )
    for name in expected.variables:
        xr.testing.assert_identical(found[name], expected[name])


def test_kit_equals_control_kit(translation_yaml):
    domain_dir = translation_yaml.parent
    # the translated domain holds a preprocessed file, so its derived
    # parameters carry that file's stamps: the control kit to compare
    # with is the one given the same file
    info = yaml.safe_load(translation_yaml.read_text())
    assert info["preprocessed_file"] == "nhm_preprocessed.nc"
    expected = assemble_from_control(
        NHM, preprocessed=domain_dir / "nhm_preprocessed.nc"
    )
    kit = assemble_from_translation(translation_yaml)

    # -- the run configuration: options from the control copy, the
    # files that were read from the yaml (NOT resolved from the copy,
    # whose directory holds neither) --
    assert kit.config.control_path == domain_dir / "nhm.control"
    assert kit.config.classes == expected.config.classes
    assert kit.config.start_time == expected.config.start_time == START
    assert kit.config.end_time == expected.config.end_time == END
    assert kit.config.param_files == expected.config.param_files
    assert kit.config.cbh_paths == expected.config.cbh_paths
    assert kit.config.output_var_names == expected.config.output_var_names

    # -- control: the window of the control file, nothing else --
    assert expected.control == {}
    assert kit.control == {"start_time": START, "end_time": END}

    # -- process_dict --
    assert list(kit.process_dict) == list(expected.process_dict)
    forcing_file = domain_dir / "cbh.nc"
    forcing = xr.load_dataset(forcing_file)
    found_forcings = []
    for slot, entry in expected.process_dict.items():
        found = kit.process_dict[slot]
        assert sorted(found) == sorted(entry)
        assert found["class"] is entry["class"]
        assert found["discretization"] == entry["discretization"]
        _assert_same_variables(found["parameters"], entry["parameters"])
        for name in entry:
            if name in STRUCTURE_KEYS:
                continue
            if name in FORCING_NAMES:
                # the path of the combined file; its variable holds the
                # control kit's values exactly (float32 -> float64)
                assert found[name] == forcing_file
                assert forcing[name].dtype == np.float32
                window = forcing[name].sel(time=slice(START, END))
                np.testing.assert_array_equal(
                    window.values, entry[name].values
                )
                np.testing.assert_array_equal(
                    window["time"].values, entry[name]["time"].values
                )
                assert window.attrs["units"] == entry[name].attrs["units"]
                found_forcings.append(name)
            else:
                # supplied initial state
                xr.testing.assert_identical(found[name], entry[name])
    assert sorted(found_forcings) == FORCING_NAMES

    # -- maps --
    assert list(kit.maps) == list(expected.maps)
    for name, mm in expected.maps.items():
        found_map = kit.maps[name]
        np.testing.assert_array_equal(
            found_map.weights.dense(), mm.weights.dense()
        )
        for attr in ("source_grid", "target_grid", "source_var", "target_var"):
            assert getattr(found_map, attr) == getattr(mm, attr)

    # -- discretizations --
    assert list(kit.discretizations) == list(expected.discretizations)
    for grid, disc in expected.discretizations.items():
        found_disc = kit.discretizations[grid]
        assert found_disc.dims == disc.dims
        if disc.parameters is None:
            assert found_disc.parameters is None
        else:
            assert found_disc.parameters is not None
            _assert_same_variables(found_disc.parameters, disc.parameters)


def test_run_equals_control_run(translation_yaml):
    """The whole two years, every collected value identical. The
    translated domain's forcings stay on disk as float32 and reach the
    kernels as float64."""
    control = {"budget": "error", "collect_var_names": COLLECT_NAMES}
    with model_from_control(NHM, control=control) as model:
        assert model.ntime == N_TIME
        model.run(DT, np.int32(N_TIME))
        assert model.results is not None
        expected = model.results.values

    with model_from_translation(translation_yaml, control=control) as model:
        assert model.ntime == N_TIME
        assert model.times.values[0] == START
        assert model.times.values[-1] == END
        # the three forcings are one opened file
        assert len(model._opened_files) == 1
        model.run(DT, np.int32(N_TIME))
        for name in FORCING_NAMES:
            inp = model.inputs_dict[name]
            assert inp.data.dtype == np.float32
            assert not inp.data.variable._in_memory
            assert inp.current_values.dtype == np.float64
        assert model.results is not None
        results = model.results.values

    for name in COLLECT_NAMES:
        assert np.isfinite(expected[name]).any()
        np.testing.assert_array_equal(results[name], expected[name])


def test_moved_domain_and_callers_window(moved_yaml):
    """A domain directory that was moved still builds (the yaml's
    products are relative to it), and a window in the caller's control
    takes precedence over the control file's."""
    start = np.datetime64("1979-03-01")
    end = np.datetime64("1979-03-10")
    kit = assemble_from_translation(
        moved_yaml, control={"start_time": start, "end_time": end}
    )
    assert kit.control == {"start_time": start, "end_time": end}
    assert kit.process_dict["prms_atmosphere"]["prcp"] == (
        moved_yaml.parent / "cbh.nc"
    )
    with kit.model() as model:
        assert model.ntime == 10
        assert model.times.values[0] == start
        assert model.times.values[-1] == end


def test_without_preprocessed_file_derives(moved_yaml):
    """A domain translated without the preprocessed step: the derived
    parameters are computed at assembly, as assemble_from_control
    computes them."""
    info = yaml.safe_load(moved_yaml.read_text())
    info["preprocessed_file"] = None
    moved_yaml.write_text(yaml.safe_dump(info, sort_keys=False))
    kit = assemble_from_translation(moved_yaml)
    expected = assemble_from_control(NHM)
    for slot, entry in expected.process_dict.items():
        _assert_same_variables(
            kit.process_dict[slot]["parameters"], entry["parameters"]
        )
    segment_dis = kit.discretizations["nsegment"].parameters
    expected_dis = expected.discretizations["nsegment"].parameters
    assert segment_dis is not None and expected_dis is not None
    _assert_same_variables(segment_dis, expected_dis)


def test_stale_preprocessed_file_raises(moved_yaml):
    """The parameter file rewritten without its preprocessed file: the
    run that reads the pair raises."""
    parameters = moved_yaml.parent / "nhm_parameters.nc"
    params = xr.load_dataset(parameters)
    params["hru_slope"] = params["hru_slope"] * 1.0001
    params.to_netcdf(parameters)
    with pytest.raises(ValueError, match="STALE.*hru_slope"):
        assemble_from_translation(moved_yaml)


def test_edited_control_copy_raises(moved_yaml):
    control_copy = moved_yaml.parent / "nhm.control"
    control_copy.write_text(control_copy.read_text() + "\n")
    with pytest.raises(ValueError, match="is not the control file that"):
        assemble_from_translation(moved_yaml)


@pytest.mark.parametrize("key", ["forcing_file", "parameter_file"])
def test_step_not_run_raises(moved_yaml, key):
    info = yaml.safe_load(moved_yaml.read_text())
    info[key] = None
    moved_yaml.write_text(yaml.safe_dump(info, sort_keys=False))
    with pytest.raises(FileNotFoundError, match=f"names no {key}"):
        assemble_from_translation(moved_yaml)


def test_dynamic_parameters_raise(moved_yaml, monkeypatch):
    """translate_domain refuses such a control; this guards a yaml
    made by hand."""
    from_control = assemble.from_control

    def with_dynamic_parameter(ctl, control_path):
        cfg = from_control(ctl, control_path)
        cfg.dyn_param_paths = {"ag_frac": pl.Path("ag_frac.dyn")}
        return cfg

    monkeypatch.setattr(assemble, "from_control", with_dynamic_parameter)
    with pytest.raises(NotImplementedError, match="dynamic-parameter"):
        assemble_from_translation(moved_yaml)
