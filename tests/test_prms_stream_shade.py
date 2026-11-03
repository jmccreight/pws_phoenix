"""Standalone regression: the shday shade machinery vs pywatershed.

Drives the verbatim-extracted _shday_vectorized (dynamic shade from
topography + vegetation; drb runs stream_temp_shade_flag = 0) over
the full drb_2yr nhm_stream_temp period -- per day: the solar
declination and summer flag exactly as pywatershed's stream-temp
process derives them, and the day's seg_flow_width from the generated
answers -- and compares the shade fraction against seg_shade at
pywatershed's OWN stream-temp family standard: rtol = atol = 5e-3.
Its test comment: "small numerical differences in the iteration loop
and in the trig results for seg_shade drive discrepencies just above
32-bit precision ... errors dont grow with time". Observed here: max
|diff| ~2.4e-3, ~93% of points within 1e-5 -- the same
Fortran-vs-python trig/iteration noise upstream describes (the
answers are Fortran shday). This pins the shade physics in isolation
before PRMSStreamTemp composes it (stage 3).

test_tabulated_shade_is_exact pins the Oct 2026 split of shday
(prms_stream_shade module docstring): PRMSStreamTemp's per-step path
-- the (day of year, segment) tables of _shday_statics +
_shade_all -- must equal _shday_vectorized bit for bit, since the two
run the same operations in the same order.

Requires drb_2yr with GENERATED nhm_stream_temp answers; skips with a
clear reason if absent.
"""

import numpy as np
import pytest
import xarray as xr
from conftest import PWS_TEST_DATA
from pws_phoenix.hydrology.prms_stream_shade import _shday_vectorized
from pws_phoenix.hydrology.prms_stream_temp import (
    _shade_all,
    _shade_statics_table,
)

DOMAIN_DIR = PWS_TEST_DATA / "drb_2yr"
GEN_DIR = DOMAIN_DIR / "output_stream_temp"

# pywatershed's own standard for the stream-temp family (see module
# docstring; its comment pins seg_shade trig noise vs Fortran)
RTOL = ATOL = 5.0e-3
# upstream stream-temp constants for the declination precompute
_DAYS_YR = 365.25

_needed = [
    DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc",
    DOMAIN_DIR / "parameters_dis_seg.nc",
    GEN_DIR / "seg_flow_width.nc",
    GEN_DIR / "seg_shade.nc",
]
_missing = [str(ff) for ff in _needed if not ff.exists()]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=(
        "pywatershed drb_2yr nhm_stream_temp data not generated; "
        "missing: " + ", ".join(_missing[:3])
    ),
)


def test_shday_all_days_all_segments():
    params = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
    )
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    seg_flow_width = xr.load_dataarray(GEN_DIR / "seg_flow_width.nc")
    seg_shade = xr.load_dataarray(GEN_DIR / "seg_shade.nc")

    seg_lat_rad = dis_seg["seg_lat"].values * (np.pi / 180.0)
    maxiter = int(params["maxiter_sntemp"].values[0])
    pp = {
        nn: params[nn].values
        for nn in (
            "azrh",
            "alte",
            "altw",
            "vce",
            "voe",
            "vhe",
            "vdemx",
            "vdemn",
            "vcw",
            "vow",
            "vhw",
            "vdwmx",
            "vdwmn",
        )
    }

    times = seg_flow_width["time"].values.astype("datetime64[D]")
    year_starts = times.astype("datetime64[Y]").astype("datetime64[D]")
    doys = (times - year_starts).astype(int) + 1  # 1-based

    shades = np.zeros_like(seg_shade.values)
    for tt in range(times.shape[0]):
        doy = int(doys[tt])
        # upstream _precompute_solar_geometry, indexed by doy
        declination = 0.40928 * np.cos(
            ((2.0 * np.pi) / _DAYS_YR) * (172.0 - doy)
        )
        summer_flag = 1 if 121 <= doy <= 273 else 0
        shades[tt, :], _svis = _shday_vectorized(
            seg_lat_rad,
            declination,
            seg_flow_width.values[tt, :],
            pp["azrh"],
            pp["alte"],
            pp["altw"],
            pp["vce"],
            pp["voe"],
            pp["vhe"],
            pp["vdemx"],
            pp["vdemn"],
            summer_flag,
            pp["vcw"],
            pp["vow"],
            pp["vhw"],
            pp["vdwmx"],
            pp["vdwmn"],
            maxiter,
        )

    np.testing.assert_allclose(
        shades,
        seg_shade.values,
        rtol=RTOL,
        atol=ATOL,
        err_msg="seg_shade differs from pywatershed",
    )


def test_tabulated_shade_is_exact():
    """The tabulated per-step path == the verbatim composition, exactly,
    on four days of year spanning the year (both summer-flag states,
    both solstice sides), each with some row of the answers' flow
    widths (any row serves: the check is path-vs-path, not vs answers)."""
    params = xr.load_dataset(
        DOMAIN_DIR / "parameters_PRMSStreamShadeDynamic.nc"
    )
    dis_seg = xr.load_dataset(DOMAIN_DIR / "parameters_dis_seg.nc")
    seg_flow_width = xr.load_dataarray(GEN_DIR / "seg_flow_width.nc")

    seg_lat_rad = dis_seg["seg_lat"].values * (np.pi / 180.0)
    maxiter = int(params["maxiter_sntemp"].values[0])
    names = (
        "azrh",
        "alte",
        "altw",
        "vce",
        "voe",
        "vhe",
        "vdemx",
        "vdemn",
        "vcw",
        "vow",
        "vhw",
        "vdwmx",
        "vdwmn",
    )
    pp = {nn: params[nn].values for nn in names}
    nseg = seg_lat_rad.shape[0]

    # the PRMSStreamTemp.initialize recipe: the (366,) declination
    # table, then the five (366, nseg) tables
    kk = np.arange(366) + 1.0
    declination = 0.40928 * np.cos(((2.0 * np.pi) / _DAYS_YR) * (172.0 - kk))
    tables = [np.zeros((366, nseg)) for _ in range(5)]
    _shade_statics_table(
        *tables,
        declination,
        seg_lat_rad,
        pp["azrh"],
        pp["alte"],
        pp["altw"],
        maxiter,
    )

    seg_shade = np.zeros(nseg)
    seg_svi = np.zeros(nseg)
    for tt, doy in ((0, 274), (91, 1), (182, 92), (273, 183)):
        summer_flag = 1 if 121 <= doy <= 273 else 0
        width = seg_flow_width.values[tt, :]
        veg = tuple(pp[nn] for nn in names)
        _shade_all(
            seg_shade,
            seg_svi,
            doy - 1,
            float(declination[doy - 1]),
            summer_flag,
            *tables,
            seg_lat_rad,
            width,
            pp["azrh"],
            *veg[3:],
        )
        shades, svis = _shday_vectorized(
            seg_lat_rad,
            float(declination[doy - 1]),
            width,
            *veg[:8],
            summer_flag,
            *veg[8:],
            maxiter,
        )
        np.testing.assert_array_equal(seg_shade, shades)
        np.testing.assert_array_equal(seg_svi, svis)
