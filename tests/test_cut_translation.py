"""prms_translate.cut_translation: a translated domain cut to some of
its hrus, on drb_2yr's nhm.control (765 hrus).

- A model of the cut gives those hrus' columns of the whole domain's
  model, identical, from either forcing file, and so does a block of
  the cut.
- The cut's files hold the source's values at those hrus and nothing
  on another grid.
- verify_translation passes on the cut.
- The hrus are named by position or by nhm_id, and both give one cut.
- A build of the cut that keeps the channel raises; so do hrus that
  are not in the source, and a second cut into the same directory.
"""

import shutil

import numpy as np
import pytest
import xarray as xr
import yaml

from conftest import PWS_TEST_DATA
from pws_phoenix.prms_translate import (
    assemble_from_translation,
    model_from_translation,
    verify_translation,
)
from pws_phoenix.prms_translate import cut_translation as cut_module
from pws_phoenix.prms_translate.cut_translation import cut_translation
from pws_phoenix.prms_translate.parameters import SLOT_GRIDS
from pws_phoenix.prms_translate.translate_domain import translate_domain

NHM = PWS_TEST_DATA / "drb_2yr" / "nhm.control"
NHRU = 765
# hrus scattered over the domain, its first and its last among them
POSITIONS = np.unique(np.concatenate([np.arange(3, NHRU, 7), [0, 1, 764]]))
N_CUT = POSITIONS.size
# a block of the cut, as a calibration process builds it
CUT_BLOCK = (10, 50)
START = np.datetime64("1979-01-01")
END = np.datetime64("1979-12-31")
N_TIME = 365
DT = np.float64(86400.0)
HRU_SLOTS = [slot for slot, grid in SLOT_GRIDS.items() if grid == "nhru"]
# one variable of each process on the hru grid
COLLECT_NAMES = [
    "potet",
    "pkwater_equiv",
    "sroff_vol",
    "ssres_flow_vol",
    "gwres_stor",
    "gwres_flow",
]
CONTROL = {
    "start_time": START,
    "end_time": END,
    "budget": "error",
    "collect_var_names": COLLECT_NAMES,
}
# the forcing file -> the transposed= argument
FILES = {"cbh.nc": False, "cbh_transpose.nc": True}
HRU_ARTIFACT_NAMES = ["soltab_potsw", "soltab_horad_potsw", "hru_in_to_cf"]


@pytest.fixture(scope="module")
def translation_yaml(tmp_path_factory):
    """nhm.control translated once, with the transposed copy of the
    forcing file."""
    return translate_domain(
        NHM, tmp_path_factory.mktemp("domain"), cbh_transpose=True
    )


@pytest.fixture(scope="module")
def cut_yaml(translation_yaml, tmp_path_factory):
    """The domain cut to POSITIONS."""
    return cut_translation(
        translation_yaml,
        tmp_path_factory.mktemp("cut") / "domain",
        hru_positions=POSITIONS,
    )


def _run(model) -> dict[str, np.ndarray]:
    """Run a built model over its window; a copy of what it
    collected."""
    with model:
        assert model.ntime == N_TIME
        model.run(DT, np.int32(N_TIME))
        assert model.results is not None
        return {
            name: model.results.values[name].copy() for name in COLLECT_NAMES
        }


@pytest.fixture(scope="module")
def whole_run(translation_yaml):
    """The six hru processes on the whole domain."""
    results = _run(
        model_from_translation(
            translation_yaml, control=CONTROL, processes=HRU_SLOTS
        )
    )
    for name in COLLECT_NAMES:
        assert results[name].shape == (N_TIME, NHRU)
        assert np.isfinite(results[name]).all()
    return results


def _files(yaml_path) -> dict[str, xr.Dataset]:
    """The four data files of a translated domain, loaded and closed,
    by their yaml key."""
    info = yaml.safe_load(yaml_path.read_text())
    return {
        key: xr.load_dataset(yaml_path.parent / info[key])
        for key in (
            "parameter_file",
            "preprocessed_file",
            "forcing_file",
            "forcing_file_transpose",
        )
    }


@pytest.mark.parametrize("transposed", FILES.values(), ids=FILES.keys())
def test_cut_equals_whole_columns(cut_yaml, whole_run, transposed):
    kit = assemble_from_translation(
        cut_yaml, control=CONTROL, processes=HRU_SLOTS, transposed=transposed
    )
    if transposed:
        # one day per step from (nhru, time): the slow order
        with pytest.warns(UserWarning, match="slower order"):
            model = kit.model()
    else:
        model = kit.model()
    results = _run(model)
    for name in COLLECT_NAMES:
        np.testing.assert_array_equal(
            results[name], whole_run[name][:, POSITIONS], err_msg=name
        )


def test_block_of_the_cut_equals_whole_columns(cut_yaml, whole_run):
    start, stop = CUT_BLOCK
    results = _run(
        model_from_translation(
            cut_yaml,
            control=CONTROL,
            processes=HRU_SLOTS,
            hru_block=CUT_BLOCK,
            transposed=True,
        )
    )
    for name in COLLECT_NAMES:
        np.testing.assert_array_equal(
            results[name],
            whole_run[name][:, POSITIONS[start:stop]],
            err_msg=name,
        )


def test_cut_parameter_file(translation_yaml, cut_yaml):
    source = _files(translation_yaml)["parameter_file"]
    cut = _files(cut_yaml)["parameter_file"]
    assert cut.sizes["nhru"] == N_CUT
    assert cut.attrs["cut_from"] == str(translation_yaml.resolve())
    assert cut.attrs["source_files"] == source.attrs["source_files"]
    # nothing on another grid, and its dims are recorded
    assert set(cut.sizes) <= {"nhru", *cut_module.WHOLE_DIMS}
    dropped = {
        str(name)
        for name in source.data_vars
        if not set(source[name].dims) <= {"nhru", *cut_module.WHOLE_DIMS}
    }
    assert {"tosegment", "poi_gage_id"} <= dropped
    assert set(cut.attrs["dropped_dims"].split()) == {"nsegment", "npoigages"}
    assert set(cut.data_vars) == set(source.data_vars) - dropped
    # the source's values, dtypes and attributes at those hrus
    for name in cut.data_vars:
        wanted = source[name]
        if "nhru" in wanted.dims:
            wanted = wanted.isel(nhru=POSITIONS)
        assert cut[name].dims == wanted.dims, name
        assert cut[name].dtype == wanted.dtype, name
        assert cut[name].attrs == wanted.attrs, name
        np.testing.assert_array_equal(
            cut[name].values, wanted.values, err_msg=name
        )
    np.testing.assert_array_equal(
        cut["nhm_id"].values, source["nhm_id"].values[POSITIONS]
    )


def test_cut_preprocessed_file(translation_yaml, cut_yaml):
    """The derived arrays on nhru are the source's at those hrus; the
    segment order and the map weights are gone."""
    source = _files(translation_yaml)["preprocessed_file"]
    cut = _files(cut_yaml)["preprocessed_file"]
    assert "segment_order" in source and "weights_vol_rows" in source
    assert set(cut.data_vars) == set(HRU_ARTIFACT_NAMES)
    for name in HRU_ARTIFACT_NAMES:
        np.testing.assert_array_equal(
            cut[name].values,
            source[name].isel(nhru=POSITIONS).values,
            err_msg=name,
        )
        assert (
            cut[name].attrs["derivation"] == (source[name].attrs["derivation"])
        )
        # stamped again: the stamps are of the cut arrays
        assert cut[name].attrs["digest"] != source[name].attrs["digest"]


@pytest.mark.parametrize("key", ["forcing_file", "forcing_file_transpose"])
def test_cut_forcing_files(translation_yaml, cut_yaml, key):
    source = _files(translation_yaml)[key]
    cut = _files(cut_yaml)[key]
    assert set(cut.data_vars) == set(source.data_vars)
    np.testing.assert_array_equal(cut["time"].values, source["time"].values)
    np.testing.assert_array_equal(cut["nhru"].values, np.arange(N_CUT))
    for name in source.data_vars:
        wanted = source[name].isel(nhru=POSITIONS)
        assert cut[name].dims == wanted.dims, name
        assert cut[name].dtype == wanted.dtype, name
        assert cut[name].attrs == wanted.attrs, name
        np.testing.assert_array_equal(
            cut[name].values, wanted.values, err_msg=name
        )


def test_cut_yaml(translation_yaml, cut_yaml):
    source = yaml.safe_load(translation_yaml.read_text())
    cut = yaml.safe_load(cut_yaml.read_text())
    assert cut == source | {"cut_from": str(translation_yaml.resolve())}
    assert (cut_yaml.parent / cut["control_copy"]).read_bytes() == (
        translation_yaml.parent / source["control_copy"]
    ).read_bytes()


def test_verify_translation_passes_on_the_cut(cut_yaml):
    verify_translation(cut_yaml)


def test_nhm_ids_from_the_command_line(
    translation_yaml, cut_yaml, tmp_path, capsys
):
    """The same hrus named by nhm_id, in another order, through the
    command line: the same cut."""
    with xr.open_dataset(
        translation_yaml.parent / "nhm_parameters.nc"
    ) as params:
        nhm_ids = params["nhm_id"].values[POSITIONS][::-1]
    ids_file = tmp_path / "nhm_ids.txt"
    np.savetxt(ids_file, nhm_ids, fmt="%d")
    out_dir = tmp_path / "by_id"
    cut_module.main(
        [
            str(translation_yaml),
            "--out",
            str(out_dir),
            "--nhm-ids-file",
            str(ids_file),
        ]
    )
    by_id_yaml = out_dir / cut_yaml.name
    assert capsys.readouterr().out.strip() == str(by_id_yaml)
    wanted = _files(cut_yaml)
    found = _files(by_id_yaml)
    for key, ds in wanted.items():
        xr.testing.assert_identical(found[key], ds)


def test_positions_from_the_command_line(translation_yaml, cut_yaml, tmp_path):
    positions_file = tmp_path / "positions.txt"
    np.savetxt(positions_file, POSITIONS, fmt="%d")
    out_dir = tmp_path / "by_position"
    cut_module.main(
        [
            str(translation_yaml),
            "--out",
            str(out_dir),
            "--positions-file",
            str(positions_file),
        ]
    )
    xr.testing.assert_identical(
        _files(out_dir / cut_yaml.name)["parameter_file"],
        _files(cut_yaml)["parameter_file"],
    )


def test_source_without_optional_files(translation_yaml, tmp_path):
    """A source with no transposed copy and no preprocessed file: the
    forcing is read from the (time, nhru) file, and the cut has
    neither file."""
    moved = tmp_path / "moved"
    shutil.copytree(translation_yaml.parent, moved)
    moved_yaml = moved / translation_yaml.name
    info = yaml.safe_load(moved_yaml.read_text())
    info["forcing_file_transpose"] = None
    info["preprocessed_file"] = None
    moved_yaml.write_text(yaml.safe_dump(info, sort_keys=False))

    made = cut_translation(
        moved_yaml, tmp_path / "cut", hru_positions=POSITIONS
    )
    cut_info = yaml.safe_load(made.read_text())
    assert cut_info["forcing_file_transpose"] is None
    assert cut_info["preprocessed_file"] is None
    assert sorted(pp.name for pp in made.parent.iterdir()) == sorted(
        [
            "cbh.nc",
            "nhm.control",
            "nhm_parameters.nc",
            "nhm_translation.yaml",
        ]
    )
    source = xr.load_dataset(moved / "cbh.nc")
    cut = xr.load_dataset(made.parent / "cbh.nc")
    for name in source.data_vars:
        np.testing.assert_array_equal(
            cut[name].values, source[name].values[:, POSITIONS], err_msg=name
        )
    verify_translation(made)


def test_cut_with_the_channel_raises(cut_yaml):
    """Without `processes` the control file's channel is kept, and a
    cut holds nothing on the segment grid."""
    with pytest.raises(ValueError, match="cut of the hrus.*prms_channel"):
        assemble_from_translation(cut_yaml)


@pytest.mark.parametrize(
    "hrus, message",
    [
        ({}, "one of them"),
        ({"hru_positions": [3, 5], "nhm_ids": [5310]}, "one of them"),
        ({"hru_positions": [5, 3]}, "ascending"),
        ({"hru_positions": [3, 3]}, "ascending"),
        ({"hru_positions": [-1, 3]}, "ascending"),
        ({"hru_positions": [3, NHRU]}, f"from 0 to {NHRU - 1}"),
        ({"hru_positions": []}, "ascending"),
        ({"nhm_ids": []}, "empty"),
        ({"nhm_ids": [5308, 5308]}, "repeats"),
        ({"nhm_ids": [5308, 1]}, "1 of the 2 nhm_ids are not in"),
    ],
)
def test_hrus_not_of_the_source_raise(
    translation_yaml, tmp_path, hrus, message
):
    out_dir = tmp_path / "cut"
    with pytest.raises(ValueError, match=message):
        cut_translation(translation_yaml, out_dir, **hrus)
    # nothing was written
    assert not out_dir.exists()


def test_second_cut_into_a_directory_raises(translation_yaml, cut_yaml):
    with pytest.raises(FileExistsError, match="A cut is written once"):
        cut_translation(
            translation_yaml, cut_yaml.parent, hru_positions=[0, 1]
        )
    # and so does a cut into the source's own directory
    with pytest.raises(FileExistsError, match="A cut is written once"):
        cut_translation(
            translation_yaml, translation_yaml.parent, hru_positions=[0, 1]
        )
