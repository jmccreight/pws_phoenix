"""prms_translate.translate_domain: a PRMS control file -> the flat
domain directory (forcing netCDF, parameters netCDF, control copy,
translation yaml)."""

import hashlib
import shutil

import pytest
import xarray as xr
import yaml

from conftest import PWS_TEST_DATA
from pws_phoenix.prms_translate import cbh, load_parameters
from pws_phoenix.prms_translate import translate_domain as module
from pws_phoenix.prms_translate.translate_domain import main, translate_domain

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
FGR_DIR = PWS_TEST_DATA / "fgr_ag_2yr"
NHM = DRB_DIR / "nhm.control"
NHM_CBH = {
    "prcp": str((DRB_DIR / "prcp.cbh").resolve()),
    "tmax": str((DRB_DIR / "tmax.cbh").resolve()),
    "tmin": str((DRB_DIR / "tmin.cbh").resolve()),
}


def _no_parse(*args, **kwargs):
    raise AssertionError(f"parsed {args}")


@pytest.fixture(scope="module")
def domain(tmp_path_factory):
    """nhm.control translated once, every step."""
    out_dir = tmp_path_factory.mktemp("domain")
    yaml_path = translate_domain(NHM, out_dir)
    assert yaml_path == out_dir / "nhm_translation.yaml"
    return out_dir


def test_products(domain):
    names = {pp.name for pp in domain.iterdir()}
    assert names >= {
        "cbh.nc",
        "nhm.control",
        "nhm_parameters.nc",
        "nhm_translation.yaml",
    }
    assert not [nn for nn in names if nn.endswith(".partial")]
    assert yaml.safe_load((domain / "nhm_translation.yaml").read_text()) == {
        "control_file": str(NHM.resolve()),
        "control_sha256": hashlib.sha256(NHM.read_bytes()).hexdigest(),
        "control_copy": "nhm.control",
        "forcing_file": "cbh.nc",
        "forcing_sources": NHM_CBH,
        "parameter_file": "nhm_parameters.nc",
        "parameter_sources": [str((DRB_DIR / "myparam.param").resolve())],
        "preprocessed_file": None,
    }
    assert (domain / "nhm.control").read_bytes() == NHM.read_bytes()
    assert cbh.cbh_netcdf_sources(domain / "cbh.nc") == NHM_CBH

    written = xr.load_dataset(domain / "nhm_parameters.nc")
    assert written.attrs.pop("source_files") == str(
        (DRB_DIR / "myparam.param").resolve()
    )
    xr.testing.assert_identical(
        written, load_parameters(DRB_DIR / "myparam.param")
    )


def test_rerun_parses_nothing(domain, monkeypatch):
    before = (domain / "nhm_translation.yaml").read_text()
    monkeypatch.setattr(cbh, "load_cbh", _no_parse)
    monkeypatch.setattr(module, "load_parameters", _no_parse)
    yaml_path = translate_domain(NHM, domain)
    assert yaml_path.read_text() == before


def test_control_with_other_cbh_files(domain):
    """nhm_stream_temp.control adds the humidity CBH: cbh.nc is not
    its forcing file, whether or not the cbh step is selected."""
    control = DRB_DIR / "nhm_stream_temp.control"
    for only in (None, ["parameters"]):
        with pytest.raises(FileExistsError, match="--cbh-name"):
            translate_domain(control, domain, only=only)
    assert not (domain / "nhm_stream_temp_translation.yaml").exists()

    yaml_path = translate_domain(control, domain, cbh_name="humid")
    info = yaml.safe_load(yaml_path.read_text())
    assert info["forcing_file"] == "cbh_humid.nc"
    assert info["parameter_file"] == "nhm_stream_temp_parameters.nc"
    assert sorted(cbh.cbh_netcdf_sources(domain / "cbh_humid.nc")) == [
        "humidity_hru",
        "prcp",
        "tmax",
        "tmin",
    ]
    # the first control's forcing file is untouched
    assert cbh.cbh_netcdf_sources(domain / "cbh.nc") == NHM_CBH


def test_only_parameters_control_guard_and_force(tmp_path, monkeypatch):
    """A control file that changes after translation: the copy guards
    it; force replaces the copy and rewrites the selected step."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    control = source_dir / "nhm.control"
    shutil.copy(NHM, control)
    shutil.copy(DRB_DIR / "myparam.param", source_dir)
    out_dir = tmp_path / "domain"

    yaml_path = translate_domain(control, out_dir, only=["parameters"])
    info = yaml.safe_load(yaml_path.read_text())
    assert info["forcing_file"] is None
    assert info["forcing_sources"]["prcp"] == str(
        (source_dir / "prcp.cbh").resolve()
    )
    assert info["parameter_file"] == "nhm_parameters.nc"
    assert not (out_dir / "cbh.nc").exists()

    # the control file changes (its first line is a free-text header)
    lines = control.read_text().splitlines(keepends=True)
    control.write_text("".join(["changed header\n"] + lines[1:]))
    with pytest.raises(FileExistsError, match="control file changed"):
        translate_domain(control, out_dir, only=["parameters"])
    assert (out_dir / "nhm.control").read_bytes() == NHM.read_bytes()

    parsed = []
    real = module.load_parameters

    def counting(paths):
        parsed.append(paths)
        return real(paths)

    monkeypatch.setattr(module, "load_parameters", counting)
    yaml_path = translate_domain(
        control, out_dir, only=["parameters"], force=True
    )
    assert len(parsed) == 1
    assert (out_dir / "nhm.control").read_bytes() == control.read_bytes()
    info = yaml.safe_load(yaml_path.read_text())
    assert (
        info["control_sha256"]
        == hashlib.sha256(control.read_bytes()).hexdigest()
    )


def test_raises_before_writing(tmp_path):
    out_dir = tmp_path / "domain"
    with pytest.raises(ValueError, match="unknown steps"):
        translate_domain(NHM, out_dir, only=["weights"])
    # fgr_ag_2yr's analysis control sets three dyn_*_flag entries
    with pytest.raises(NotImplementedError, match="dynamic-parameter"):
        translate_domain(FGR_DIR / "analysis.control", out_dir)
    assert not out_dir.exists()


def test_main(tmp_path, capsys):
    out_dir = tmp_path / "domain"
    main([str(NHM), "--out", str(out_dir), "--only", "parameters"])
    yaml_path = out_dir / "nhm_translation.yaml"
    assert capsys.readouterr().out.splitlines()[-1] == str(yaml_path)
    assert sorted(pp.name for pp in out_dir.iterdir()) == [
        "nhm.control",
        "nhm_parameters.nc",
        "nhm_translation.yaml",
    ]
