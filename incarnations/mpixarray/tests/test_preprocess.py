"""The pre-processing suite: execute-stamp-verify (drb legacy data).

The guarantees under test: saved artifacts are BIT-identical to the
in-chain derivations (so a preprocessed assembly is the same model);
verification passes on aligned inputs and raises loudly on a drifted
input (staleness) or a tampered artifact (self-digest)."""

import pathlib as pl
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.append(str(pl.Path(__file__).parent.parent))
pytest.importorskip("pyPRMS", reason="pyPRMS not installed")

from prms_translate import (  # noqa: E402
    assemble_from_control,
    load_parameters,
    verify_preprocessed,
    write_preprocessed,
)

MPIX_ROOT = pl.Path(__file__).parents[4]
DRB_DIR = MPIX_ROOT / "pywatershed" / "test_data" / "drb_2yr"
CONTROL_FILE = DRB_DIR / "nhm_stream_temp.control"

_needed = [
    CONTROL_FILE,
    DRB_DIR / "myparam.param",
    DRB_DIR / "prcp.cbh",
    DRB_DIR / "rhavg.cbh",
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason="drb_2yr legacy files missing: " + ", ".join(_missing[:3]),
)


@pytest.fixture(scope="module")
def artifacts_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("preprocess") / "preprocessed.nc"
    return write_preprocessed(CONTROL_FILE, path)


def test_write_verify_and_bit_identity(artifacts_path):
    """Roundtrip verifies clean, and the artifacts equal the in-chain
    derivations bit-for-bit."""
    artifacts = xr.load_dataset(artifacts_path)
    params = load_parameters(DRB_DIR / "myparam.param")
    verify_preprocessed(artifacts, params)  # aligned: no raise

    expected = {
        "soltab_potsw",
        "soltab_horad_potsw",
        "hru_in_to_cf",
        "segment_order",
        "weights_vol",
        "weights_flow",
        "weights_swrad",
        "weights_met",
        "weights_humid",
    }
    assert set(artifacts.data_vars) == expected
    # a stamp: the derivation + one digest per named input + own digest
    attrs = artifacts["soltab_potsw"].attrs
    assert "compute_soltabs" in attrs["derivation"]
    for nn in ("hru_slope", "hru_aspect", "hru_lat"):
        assert attrs[f"derived_from_{nn}"].startswith("sha256:")
    assert attrs["digest"].startswith("sha256:")

    # bit-identity vs the in-chain kit
    kit = assemble_from_control(CONTROL_FILE)
    np.testing.assert_array_equal(
        artifacts["soltab_potsw"].values,
        kit.process_dict["prms_atmosphere"]["parameters"][
            "soltab_potsw"
        ].values,
    )
    np.testing.assert_array_equal(
        artifacts["weights_vol"].values, kit.maps["sroff_vol"].weights
    )
    np.testing.assert_array_equal(
        artifacts["weights_met"].values,
        kit.maps["seg_tave_air"].weights,
    )


def test_assemble_with_preprocessed_is_the_same_model(artifacts_path):
    """A preprocessed assembly supplies bit-identical pieces -- same
    arrays, same model."""
    pre = assemble_from_control(CONTROL_FILE, preprocessed=artifacts_path)
    plain = assemble_from_control(CONTROL_FILE)
    for slot in ("prms_atmosphere", "prms_snow", "prms_runoff"):
        pre_ds = pre.process_dict[slot]["parameters"]
        plain_ds = plain.process_dict[slot]["parameters"]
        assert set(pre_ds.data_vars) == set(plain_ds.data_vars)
        for name in pre_ds.data_vars:
            np.testing.assert_array_equal(
                pre_ds[name].values, plain_ds[name].values
            )
    for name in pre.maps:
        np.testing.assert_array_equal(
            pre.maps[name].weights, plain.maps[name].weights
        )
    pre_dis = pre.discretizations["nsegment"].parameters
    plain_dis = plain.discretizations["nsegment"].parameters
    assert pre_dis is not None and plain_dis is not None
    np.testing.assert_array_equal(
        pre_dis["segment_order"].values,
        plain_dis["segment_order"].values,
    )


def test_stale_input_raises(artifacts_path):
    """A drifted derivation input (calibration, ensembles over static
    parameters) flags every artifact derived from it."""
    artifacts = xr.load_dataset(artifacts_path)
    params = load_parameters(DRB_DIR / "myparam.param")
    perturbed = params.copy()
    perturbed["hru_slope"] = perturbed["hru_slope"] * 1.0001
    with pytest.raises(ValueError, match="STALE.*hru_slope"):
        verify_preprocessed(artifacts, perturbed)


def test_tampered_artifact_raises(artifacts_path):
    artifacts = xr.load_dataset(artifacts_path)
    params = load_parameters(DRB_DIR / "myparam.param")
    artifacts["hru_in_to_cf"].values[0] += 1.0
    with pytest.raises(ValueError, match="own digest"):
        verify_preprocessed(artifacts, params)
