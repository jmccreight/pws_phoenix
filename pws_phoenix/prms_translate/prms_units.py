"""PRMS unit strings -> the pws_phoenix units vocabulary.

PRMS fixes a parameter's units by its FORMAT (``seg_length`` IS
meters; ``tmax_allsnow`` IS in ``temp_units``), described by pyPRMS's
bundled metadata in PRMS's own words (``"decimal fraction"``,
``"fraction/day"``, ``"temp_units"``). This module is the one-way
translation of those words to ``pws_phoenix.units.VOCABULARY`` keys,
in three layers:

1. :data:`PRMS_UNITS_BY_NAME` -- parameters whose PRMS metadata string
   is wrong or vague (``hru_in_to_cf: "cubic feet"`` is cubic feet PER
   INCH; ``den_init: "gm/cm3"`` is a specific gravity the kernels use
   dimensionless; ``width_alpha: "unknown"``). Name wins over string.
2. the PRMS flag placeholders ``temp_units`` / ``precip_units`` /
   ``elev_units`` (and the compound ``dday/temp_units``), resolved from
   the values of the SAME-NAMED parameters in the file.
3. :data:`PRMS_UNITS_BY_STRING` -- spelling (``"meters"`` -> ``"m"``,
   ``"degree_fahrenheit"`` -> ``"degF"``, pyPRMS's CBH strings).

:func:`resolve` returns ``None`` for a string it cannot map, so a
parameter no process declares (glacier, lakes, ...) keeps its raw PRMS
string; the judgment -- stamped key vs the process declaration -- is
``parameters.package_parameters``'s, for DECLARED parameters only.
This is the PRMS boundary: a disagreement there is a bug in this table
or in a declaration, never something to patch (the Model-boundary
``control["units_patch"]`` serves pws_phoenix-native inputs).
"""

from __future__ import annotations

from collections.abc import Mapping

from pws_phoenix.units import VOCABULARY

# PRMS flag values (pyPRMS.constants has no names for these)
_TEMP_UNITS = {0: "degF", 1: "degC"}
_ELEV_UNITS = {0: "feet", 1: "m"}
_PRECIP_UNITS = {0: "inches"}  # 1 = mm: the kernels are imperial
FLAG_PARAMETERS = ("temp_units", "precip_units", "elev_units")

# the two dimensionless keys are OUR semantic split, not PRMS's
DIMENSIONLESS = frozenset({"fraction", "none"})

# -- 1. by name: PRMS's metadata string is wrong or vague here --------
PRMS_UNITS_BY_NAME: dict[str, str] = {
    "hru_in_to_cf": "cubic feet/inch",  # PRMS: "cubic feet"
    "den_init": "fraction",  # PRMS: "gm/cm3" (specific gravity)
    "den_max": "fraction",  # PRMS: "gm/cm3"
    "settle_const": "1/day",  # PRMS: "decimal fraction"
    "mann_n": "seconds/m^(1/3)",  # PRMS: "seconds/meters"
    "width_alpha": "m per cms^exp",  # PRMS: "unknown"
    "width_m": "fraction",  # PRMS: "unknown"
    "depth_alpha": "m per cms^exp",  # PRMS: "meters"
    "depth_m": "fraction",  # PRMS: "none"
    "dprst_vol_thres_open": "acre-inches",  # PRMS: "unknown"
    "soilzone_aet_converge": "inches",  # PRMS: "unknown"
    "hru_slope": "fraction",  # PRMS: "decimal fraction" (rise/run)
    "hru_lat": "degrees",  # PRMS: "degrees North"
    "seg_lat": "degrees",  # PRMS: "degrees north"
    "hru_aspect": "degrees",  # PRMS: "angular degrees"
    "radadj_intcp": "dday",  # PRMS: "dday"
    "radadj_slope": "dday/delta_degF",  # PRMS: "dday/temp_units"
    "cecn_coef": "cal/(cm^2 delta_degC)",
    "jh_coef": "1/delta_degF",  # PRMS: "per degrees Fahrenheit"
    "soil2gw_max": "inches/day",  # PRMS: "inches"
    "fastcoef_sq": "1/(inch day)",  # PRMS: dimensionless
    "slowcoef_sq": "1/(inch day)",  # PRMS: dimensionless
    "tsi": "hours",
    "lso": "days",
}

# -- 3. by string: spelling --------------------------------------------
PRMS_UNITS_BY_STRING: dict[str, str] = {
    "inches": "inches",
    "in": "inches",  # pywatershed CBH netcdf
    "inch": "inches",  # pyPRMS Cbh (its own strings, not the metadata)
    "inches/day": "inches/day",
    "1.0/inch": "1/inch",
    "feet": "feet",
    "acres": "acres",
    "acre-inches": "acre-inches",
    "cubic feet": "cubic feet",
    "cubicfeet": "cubic feet",
    "cfs": "cfs",
    "meters": "m",
    "meter": "m",
    "m s-1": "m/s",
    "cubic meters per second": "cms",
    "decimal fraction": "fraction",
    "decimal_fraction": "fraction",
    "dimensionless": "fraction",
    "none": "none",
    "percent": "percent",
    "percentage": "percent",
    "fraction/day": "1/day",
    "days": "days",
    "hours": "hours",
    "seconds": "seconds",
    "degrees Fahrenheit": "degF",
    "degree_fahrenheit": "degF",  # pyPRMS CBH
    "degrees Celsius": "degC",
    "degC": "degC",
    "per degrees Fahrenheit": "1/delta_degF",
    "Langleys": "cal/cm^2",
    "dday": "dday",
    "Solar date": "day of year",
}


def resolve(
    name: str,
    prms_units: str | None,
    flags: Mapping[str, int] | None = None,
) -> str | None:
    """The vocabulary key for parameter/forcing `name` whose PRMS
    metadata says `prms_units`; None when unmapped (the caller keeps
    the raw string). `flags` = ``{"temp_units": 0, ...}`` values from
    the parameter file (needed only for flag-placeholder strings;
    their absence raises when one is met)."""
    if name in PRMS_UNITS_BY_NAME:
        return PRMS_UNITS_BY_NAME[name]
    if prms_units is None:
        return None
    text = prms_units.strip()
    if text in ("temp_units", "precip_units", "elev_units"):
        return _resolve_flag(text, flags)
    if text == "dday/temp_units":
        return "dday/delta_" + _resolve_flag("temp_units", flags)
    return PRMS_UNITS_BY_STRING.get(text)


def _resolve_flag(flag: str, flags: Mapping[str, int] | None) -> str:
    if flags is None or flag not in flags:
        raise ValueError(
            f"PRMS units placeholder {flag!r} needs the value of the "
            f"{flag!r} parameter (not in the parameter file?)."
        )
    value = int(flags[flag])
    table = {
        "temp_units": _TEMP_UNITS,
        "precip_units": _PRECIP_UNITS,
        "elev_units": _ELEV_UNITS,
    }[flag]
    if value not in table:
        raise NotImplementedError(
            f"PRMS {flag} = {value}: only {sorted(table)} are supported "
            "(the pws_phoenix kernels work in PRMS's imperial units)."
        )
    return table[value]


def agree(declared: str | None, stamped: str | None) -> bool:
    """Do a declaration's units and a translated file's units name the
    same quantity? Exact key match, or both dimensionless. A
    declaration without units (int/bool fields: codes, flags, indices)
    is not judged."""
    if declared is None or declared == stamped:
        return True
    return declared in DIMENSIONLESS and stamped in DIMENSIONLESS


def _check_table() -> None:
    for table in (PRMS_UNITS_BY_NAME, PRMS_UNITS_BY_STRING):
        bad = sorted(set(table.values()) - set(VOCABULARY))
        if bad:
            raise AssertionError(f"not vocabulary keys: {bad}")


_check_table()
