"""PRMS units -> pws_phoenix vocabulary (prms_translate.prms_units):
the resolve/agree rules on synthetic strings (CI-able) and, on the drb
kit, that every translated parameter and forcing a process declares
carries a vocabulary key the declaration agrees with."""

import pytest
import xarray as xr

pytest.importorskip("pyPRMS", reason="pyPRMS not installed")

from conftest import PWS_TEST_DATA  # noqa: E402
from pws_phoenix.prms_translate import (  # noqa: E402
    assemble_from_control,
    load_parameters,
)
from pws_phoenix.prms_translate import prms_units  # noqa: E402
from pws_phoenix.process import _dict_of_kind  # noqa: E402
from pws_phoenix.units import VOCABULARY  # noqa: E402

DRB_DIR = PWS_TEST_DATA / "drb_2yr"
_drb_needed = [
    DRB_DIR / "nhm.control",
    DRB_DIR / "myparam.param",
    DRB_DIR / "prcp.cbh",
]
drb_skipif = pytest.mark.skipif(
    not all(pp.exists() for pp in _drb_needed),
    reason=f"drb_2yr translation inputs missing: {_drb_needed}",
)

FLAGS = {"temp_units": 0, "precip_units": 0, "elev_units": 1}


# ----------------------------------------------------------------------
# resolve / agree, synthetic
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, prms, expected",
    [
        ("seg_length", "meters", "m"),
        ("tmax_hru", "degree_fahrenheit", "degF"),  # pyPRMS CBH
        ("hru_ppt", "in", "inches"),
        ("gwflow_coef", "fraction/day", "1/day"),
        ("snow_cbh_adj", " decimal fraction", "fraction"),  # PRMS typo
        ("tmax_allsnow", "temp_units", "degF"),
        ("hru_elev", "elev_units", "m"),
        ("dday_slope", "dday/temp_units", "dday/delta_degF"),
        ("hru_in_to_cf", "cubic feet", "cubic feet/inch"),  # by name
        ("lat_temp_adj", "degrees Celsius", "delta_degC"),  # by name
        ("den_init", "gm/cm3", "fraction"),  # by name
        ("glacrva_coef", "m**(3-2*glacrva_exp)", None),  # unmapped
        ("anything", None, None),
    ],
)
def test_resolve(name, prms, expected):
    assert prms_units.resolve(name, prms, FLAGS) == expected


def test_resolve_celsius_flag():
    assert (
        prms_units.resolve("tmax_allsnow", "temp_units", {"temp_units": 1})
        == "degC"
    )


def test_resolve_flag_needs_value():
    with pytest.raises(ValueError, match="temp_units"):
        prms_units.resolve("tmax_allsnow", "temp_units", None)
    with pytest.raises(ValueError, match="temp_units"):
        prms_units.resolve("tmax_allsnow", "temp_units", {"elev_units": 0})


def test_resolve_metric_precip_not_implemented():
    with pytest.raises(NotImplementedError, match="precip_units = 1"):
        prms_units.resolve("rain_cbh_adj", "precip_units", {"precip_units": 1})


def test_tables_are_vocabulary():
    for table in (
        prms_units.PRMS_UNITS_BY_NAME,
        prms_units.PRMS_UNITS_BY_STRING,
    ):
        assert set(table.values()) <= set(VOCABULARY)


@pytest.mark.parametrize(
    "declared, stamped, expected",
    [
        ("inches", "inches", True),
        ("inches", "feet", False),
        ("inches", None, False),
        ("fraction", "none", True),  # our split, not PRMS's
        ("none", "fraction", True),
        ("fraction", "percent", False),
        (None, "month", True),  # unit-less (int) declaration: not judged
    ],
)
def test_agree(declared, stamped, expected):
    assert prms_units.agree(declared, stamped) is expected


# ----------------------------------------------------------------------
# the drb kit: every declared parameter/forcing is stamped and agrees
# ----------------------------------------------------------------------


@drb_skipif
def test_drb_parameters_translated():
    params = load_parameters(DRB_DIR / "myparam.param")
    # PRMS's words are gone from every parameter we can map ...
    assert params["seg_length"].attrs["units"] == "m"
    assert params["tmax_allsnow"].attrs["units"] == "degF"  # temp_units=0
    assert params["hru_elev"].attrs["units"] == "m"  # elev_units=1 (NHM)
    assert params["gwflow_coef"].attrs["units"] == "1/day"
    # ... and an unmapped one keeps them (judged only if declared)
    raw = [
        str(nn)
        for nn in params.data_vars
        if params[nn].attrs.get("units") not in VOCABULARY
    ]
    assert "seg_length" not in raw


@drb_skipif
def test_drb_kit_inputs_carry_vocabulary_units():
    """Every supplied parameter, forcing and dynamic input in the drb
    kit has attrs["units"] in the vocabulary, agreeing with the
    consumer's declaration -- what Model's control["units"]="check"
    (stage 2) will read."""
    kit = assemble_from_control(DRB_DIR / "nhm.control")
    judged = 0
    for slot, entry in kit.process_dict.items():
        cls = entry["class"]
        declared = (
            _dict_of_kind(cls, "parameter")
            | _dict_of_kind(cls, "input")
            | _dict_of_kind(cls, "mutable_input")
        )
        for name, value in entry.items():
            if name in ("class", "discretization"):
                continue
            if isinstance(value, xr.Dataset):
                items = [(str(nn), value[nn]) for nn in value.data_vars]
            elif isinstance(value, xr.DataArray):
                items = [(name, value)]
            else:
                continue  # initial values (plain arrays), paths
            for vname, da in items:
                if vname not in declared or declared[vname].units is None:
                    continue  # int/bool declarations are not judged
                stamped = da.attrs.get("units")
                assert stamped in VOCABULARY, (slot, vname, stamped)
                assert prms_units.agree(declared[vname].units, stamped), (
                    slot,
                    vname,
                    declared[vname].units,
                    stamped,
                )
                judged += 1
    assert judged > 100


@drb_skipif
def test_drb_kit_builds_under_check():
    """The Model boundary on real translated inputs: the drb NHM kit
    builds with control["units"] = "check" and no patches (stage 2-M
    reading what stage 2-T stamped)."""
    kit = assemble_from_control(
        DRB_DIR / "nhm.control", control={"units": "check"}
    )
    model = kit.model()
    assert model._units_mode == "check"
    model.finalize()
