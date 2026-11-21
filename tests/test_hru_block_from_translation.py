"""prms_translate.assemble_from_translation(hru_block=(start, stop)):
the kit of a block of hrus read alone from a translated domain, on
drb_2yr's nhm.control (765 hrus) with the six hru processes.

- The blocks of the domain, run as separate models, give the hru
  variables of the whole model, identical, from the forcing file on
  (time, nhru) and from its transposed copy.
- A block's kit holds nothing at the domain's extent: every array on
  nhru has the block's length and owns its memory, and each forcing
  is in memory in the file's dtype and dim order.
- The calibration reuse loop (set_parameters -> reset -> run) on a
  block equals a fresh build of the block with those values.
- PRMS's rule for pref_flow_infil_frac is judged on the whole domain.
- `transposed=True` without a block hands the Model the path of the
  transposed copy.
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
)
from pws_phoenix.prms_translate.parameters import SLOT_GRIDS
from pws_phoenix.prms_translate.translate_domain import translate_domain

NHM = PWS_TEST_DATA / "drb_2yr" / "nhm.control"
NHRU = 765
FORCING_NAMES = ["prcp", "tmax", "tmin"]
# the caller's window: the first of the record's two years
START = np.datetime64("1979-01-01")
END = np.datetime64("1979-12-31")
N_TIME = 365
DT = np.float64(86400.0)
# the process_dict slots on the hru grid (nhm.control selects all six)
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
# uneven blocks that cover the domain
BLOCKS = [(0, 200), (200, 400), (400, 600), (600, NHRU)]
BLOCK = BLOCKS[1]
# the forcing file -> the transposed= argument
FILES = {"cbh.nc": False, "cbh_transpose.nc": True}
# process_dict entry keys that are not arrays
STRUCTURE_KEYS = ("class", "discretization", "parameters")


@pytest.fixture(scope="module")
def translation_yaml(tmp_path_factory):
    """nhm.control translated once, with the transposed copy of the
    forcing file."""
    return translate_domain(
        NHM, tmp_path_factory.mktemp("domain"), cbh_transpose=True
    )


@pytest.fixture
def moved_yaml(translation_yaml, tmp_path):
    """A copy of the translated domain that a test may edit."""
    moved = tmp_path / "moved"
    shutil.copytree(translation_yaml.parent, moved)
    return moved / translation_yaml.name


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
    """The six hru processes on the whole domain, forcing read from
    the file one day per step."""
    results = _run(
        model_from_translation(
            translation_yaml, control=CONTROL, processes=HRU_SLOTS
        )
    )
    for name in COLLECT_NAMES:
        assert results[name].shape == (N_TIME, NHRU)
        assert np.isfinite(results[name]).all()
    return results


def _memory_size(array: np.ndarray) -> int:
    """The number of elements of the array that owns `array`'s
    memory."""
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    return owner.size


@pytest.mark.parametrize("transposed", FILES.values(), ids=FILES.keys())
def test_blocks_equal_whole(translation_yaml, whole_run, transposed):
    """Each block run alone gives its columns of the whole run,
    identical, over the year."""
    for start, stop in BLOCKS:
        results = _run(
            model_from_translation(
                translation_yaml,
                control=CONTROL,
                processes=HRU_SLOTS,
                hru_block=(start, stop),
                transposed=transposed,
            )
        )
        for name in COLLECT_NAMES:
            np.testing.assert_array_equal(
                results[name],
                whole_run[name][:, start:stop],
                err_msg=f"{name}, hrus [{start}, {stop})",
            )


@pytest.mark.parametrize("file_name", FILES.keys())
def test_block_holds_only_its_hrus(translation_yaml, file_name):
    """The kit of a block: the whole kit's values cut to the block,
    in arrays of the block's length that own their memory; each
    forcing in memory as its file stores it."""
    start, stop = BLOCK
    n_block = stop - start
    block = {"nhru": slice(start, stop)}
    transposed = FILES[file_name]
    expected = assemble_from_translation(translation_yaml, processes=HRU_SLOTS)
    kit = assemble_from_translation(
        translation_yaml,
        control=CONTROL,
        processes=HRU_SLOTS,
        hru_block=BLOCK,
        transposed=transposed,
    )
    assert kit.control == CONTROL
    assert sorted(kit.process_dict) == sorted(HRU_SLOTS)
    assert list(kit.discretizations) == ["nhru"]
    assert kit.maps == {}

    forcing = xr.load_dataset(translation_yaml.parent / file_name)
    dims = ("nhru", "time") if transposed else ("time", "nhru")
    found_forcings = []
    n_on_nhru = 0
    for slot, entry in expected.process_dict.items():
        found = kit.process_dict[slot]
        assert sorted(found) == sorted(entry)
        xr.testing.assert_identical(
            found["parameters"], entry["parameters"].isel(block)
        )
        for name, da in found["parameters"].data_vars.items():
            if "nhru" in da.dims:
                assert da.sizes["nhru"] == n_block
                assert _memory_size(da.values) == da.size, name
                n_on_nhru += 1
        for name in entry:
            if name in STRUCTURE_KEYS:
                continue
            da = found[name]
            assert da.sizes["nhru"] == n_block
            assert _memory_size(da.values) == da.size, name
            if name in FORCING_NAMES:
                assert da.variable._in_memory
                assert da.dtype == np.float32
                assert da.dims == forcing[name].dims == dims
                assert list(da.coords) == ["time"]
                assert da.sizes["time"] == N_TIME
                xr.testing.assert_identical(
                    da,
                    forcing[name]
                    .sel(time=slice(START, END))
                    .isel(block)
                    .drop_vars("nhru"),
                )
                found_forcings.append(name)
            else:
                # supplied initial state
                xr.testing.assert_identical(da, entry[name].isel(block))
    assert sorted(found_forcings) == FORCING_NAMES
    assert n_on_nhru > 0


def test_reuse_loop_on_a_block(translation_yaml):
    """set_parameters -> reset -> run on a built block equals a fresh
    build of the block with those values: a (nhru,) parameter and a
    (nmonth, nhru) one."""

    def block_kit():
        return assemble_from_translation(
            translation_yaml,
            control=CONTROL | {"calibration_mode": True},
            processes=HRU_SLOTS,
            hru_block=BLOCK,
            transposed=True,
        )

    factors = {
        ("prms_groundwater", "gwflow_coef"): 0.5,
        ("prms_atmosphere", "jh_coef"): 1.1,
    }

    kit = block_kit()
    new_values = {
        name: kit.process_dict[slot]["parameters"][name].values * factor
        for (slot, name), factor in factors.items()
    }
    assert new_values["gwflow_coef"].shape == (BLOCK[1] - BLOCK[0],)
    assert new_values["jh_coef"].shape == (12, BLOCK[1] - BLOCK[0])
    with kit.model() as model:
        model.run(DT, np.int32(N_TIME))
        assert model.results is not None
        first = {
            name: model.results.values[name].copy() for name in COLLECT_NAMES
        }
        model.set_parameters(new_values)
        model.reset()
        model.run(DT, np.int32(N_TIME))
        reused = {
            name: model.results.values[name].copy() for name in COLLECT_NAMES
        }

    fresh_kit = block_kit()
    for (slot, name), factor in factors.items():
        fresh_kit.process_dict[slot]["parameters"][name].values[...] *= factor
    fresh = _run(fresh_kit.model())

    for name in COLLECT_NAMES:
        np.testing.assert_array_equal(reused[name], fresh[name])
    # the new values changed the answer
    assert not np.array_equal(first["potet"], reused["potet"])
    assert not np.array_equal(first["gwres_flow"], reused["gwres_flow"])


def test_pref_flow_rule_is_judged_on_the_domain(moved_yaml):
    """pref_flow_infil_frac negative on the hrus of one block only:
    alone, that block would take PRMS's "not specified" rule and the
    others would not. Every block raises instead, as the whole domain
    does."""
    parameters = moved_yaml.parent / "nhm_parameters.nc"
    params = xr.load_dataset(parameters)
    values = params["pref_flow_den"].values.copy()
    values[slice(*BLOCKS[0])] = -1.0
    params["pref_flow_infil_frac"] = (
        params["pref_flow_den"].dims,
        values,
        params["pref_flow_den"].attrs,
    )
    params.to_netcdf(parameters)
    match = "pref_flow_infil_frac is negative on 200 of 765 hrus"
    for hru_block in BLOCKS:
        with pytest.raises(ValueError, match=match):
            assemble_from_translation(
                moved_yaml, processes=HRU_SLOTS, hru_block=hru_block
            )


def test_transposed_without_a_block(translation_yaml):
    """Without a block, `transposed` chooses the file whose path the
    Model is given; the answers are those of the other file."""
    control = CONTROL | {"end_time": np.datetime64("1979-01-10")}
    values = {}
    for file_name, transposed in FILES.items():
        kit = assemble_from_translation(
            translation_yaml,
            control=control,
            processes=HRU_SLOTS,
            transposed=transposed,
        )
        for name in FORCING_NAMES:
            assert kit.process_dict["prms_atmosphere"][name] == (
                translation_yaml.parent / file_name
            )
        if transposed:
            # one day per step from (nhru, time): the slow order
            with pytest.warns(UserWarning, match="slower order"):
                model = kit.model()
        else:
            model = kit.model()
        with model:
            assert model.ntime == 10
            model.run(DT, np.int32(10))
            assert model.results is not None
            values[file_name] = {
                name: model.results.values[name][:10].copy()
                for name in COLLECT_NAMES
            }
    for name in COLLECT_NAMES:
        np.testing.assert_array_equal(
            values["cbh_transpose.nc"][name], values["cbh.nc"][name]
        )


@pytest.mark.parametrize(
    "hru_block", [(-1, 10), (10, 10), (20, 10), (0, NHRU + 1)]
)
def test_block_outside_domain_raises(translation_yaml, hru_block):
    with pytest.raises(ValueError, match=f"stop <= {NHRU}"):
        assemble_from_translation(
            translation_yaml, processes=HRU_SLOTS, hru_block=hru_block
        )


def test_block_with_segment_process_raises(translation_yaml):
    """`hru_block` without `processes`: the control file's channel is
    kept, and it is not on the nhru grid."""
    with pytest.raises(ValueError, match="hru_block.*prms_channel"):
        assemble_from_translation(translation_yaml, hru_block=BLOCK)


@pytest.mark.parametrize("hru_block", [None, BLOCK], ids=["whole", "block"])
def test_no_transposed_copy_raises(moved_yaml, hru_block):
    info = yaml.safe_load(moved_yaml.read_text())
    info["forcing_file_transpose"] = None
    moved_yaml.write_text(yaml.safe_dump(info, sort_keys=False))
    with pytest.raises(
        FileNotFoundError, match="names no forcing_file_transpose"
    ):
        assemble_from_translation(
            moved_yaml,
            processes=HRU_SLOTS,
            hru_block=hru_block,
            transposed=True,
        )
